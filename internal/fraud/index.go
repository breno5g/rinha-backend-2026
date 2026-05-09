package fraud

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"time"
)

const (
	labelLegit uint8 = 0
	labelFraud uint8 = 1

	// quantizationScale maps a normalized float in [0, 1] to int16 in [0, 8192].
	// Mirrors the .NET implementation: preserves enough precision that distance
	// ties between similar refs are rare, while still fitting in a single
	// VPMADDWD lane without overflow (8192² × 14 dims ≈ 939M, well below int32 max).
	quantizationScale = 8192

	// sentinelInt16 represents "no last_transaction" in quantized space. The
	// negative full-scale value ensures the sentinel sits far outside [0, 8192],
	// preserving the "outsider" property of the float -1 used in Vectorize.
	sentinelInt16 int16 = -8192
)

// Index holds the reference dataset, quantized to int16 for high precision
// while keeping memory and cache pressure reasonable. Layout is flat with a
// 16-element (32-byte) stride per ref:
//
//	vectors[i*physicalStride : i*physicalStride+VectorDim] is the i-th reference.
//
// The trailing 2 lanes of each ref are always zero — they exist so AVX2 can
// load all 16 int16 with a single VMOVDQU without overrunning into the next
// reference. At 32 bytes per row, 3M references occupy ~96 MB (vs ~168 MB for
// float32) and AVX2 reads all 14 dims in one instruction.
//
// `tree` is an optional VP-tree that prunes exact search to O(log N) on average.
// `ivf` is an optional Inverted-File index for approximate search.
// `topK` dispatches to whichever is set, with IVF preferred when both are present.
// When neither is set, Score falls back to brute force.
type Index struct {
	Constants *Constants
	vectors   []int16
	labels    []uint8
	tree      []vpNode
	ivf       *ivfIndex
}

type referenceEntry struct {
	Vector []float32 `json:"vector"`
	Label  string    `json:"label"`
}

// quantize maps a float32 value to its int16 representation. Values in [0, 1]
// are scaled to [0, 8192]. Anything strictly negative is treated as the
// "no-data" sentinel (only -1 appears in practice). Out-of-range positive
// values are clamped to 8192.
func quantize(f float32) int16 {
	if f < 0 {
		return sentinelInt16
	}
	if f >= 1 {
		return quantizationScale
	}
	return int16(math.Round(float64(f) * quantizationScale))
}

// quantizeVector applies quantize() element-wise and returns a 16-element
// buffer (14 logical dims + 2 zero pad). The pad lanes are required so AVX2
// can load all 16 int16 in a single VMOVDQU; their zero value contributes
// nothing to the squared L2 distance.
func quantizeVector(v [VectorDim]float32) [physicalStride]int16 {
	var out [physicalStride]int16
	for i, value := range v {
		out[i] = quantize(value)
	}
	return out
}

// IndexKind selects which secondary structure (if any) to build at startup.
type IndexKind string

const (
	// KindBrute uses linear scan over all references.
	KindBrute IndexKind = "brute"
	// KindVP builds a vantage-point tree (exact, ~46MB for 3M refs).
	KindVP IndexKind = "vp"
	// KindIVF builds an Inverted File index (approximate, ~12MB for 3M refs).
	KindIVF IndexKind = "ivf"
)

// LoadIndex reads references.json.gz, quantizes each vector to int16, builds
// the requested secondary structure, and returns the ready-to-query Index.
// It also loads normalization constants and the MCC risk table.
func LoadIndex(kind IndexKind, referencesGzPath, normalizationPath, mccRiskPath string) (*Index, error) {
	constants, err := LoadConstants(normalizationPath, mccRiskPath)
	if err != nil {
		return nil, err
	}

	loadStart := time.Now()

	gzipFile, err := os.Open(referencesGzPath)
	if err != nil {
		return nil, fmt.Errorf("open references: %w", err)
	}
	defer gzipFile.Close()

	gzipReader, err := gzip.NewReader(gzipFile)
	if err != nil {
		return nil, fmt.Errorf("gzip reader: %w", err)
	}
	defer gzipReader.Close()

	decoder := json.NewDecoder(gzipReader)
	if openingToken, err := decoder.Token(); err != nil || openingToken != json.Delim('[') {
		return nil, fmt.Errorf("expected JSON array start, got %v (%w)", openingToken, err)
	}

	const expectedReferenceCount = 3_000_000
	vectors := make([]int16, 0, expectedReferenceCount*physicalStride)
	labels := make([]uint8, 0, expectedReferenceCount)

	var entry referenceEntry
	for decoder.More() {
		entry.Vector = entry.Vector[:0]
		if err := decoder.Decode(&entry); err != nil {
			return nil, fmt.Errorf("decode entry %d: %w", len(labels), err)
		}
		if len(entry.Vector) != VectorDim {
			return nil, fmt.Errorf("entry %d: expected %d dims, got %d", len(labels), VectorDim, len(entry.Vector))
		}
		for _, dimensionValue := range entry.Vector {
			vectors = append(vectors, quantize(dimensionValue))
		}
		// Zero pad to physicalStride so AVX2 loads can read 32 bytes safely.
		for pad := VectorDim; pad < physicalStride; pad++ {
			vectors = append(vectors, 0)
		}

		switch entry.Label {
		case "fraud":
			labels = append(labels, labelFraud)
		case "legit":
			labels = append(labels, labelLegit)
		default:
			return nil, fmt.Errorf("entry %d: unknown label %q", len(labels), entry.Label)
		}
	}

	if _, err := decoder.Token(); err != nil && err != io.EOF {
		return nil, fmt.Errorf("expected JSON array end: %w", err)
	}

	log.Printf("loaded %d references in %s (%.1f MB of int8 vectors)",
		len(labels), time.Since(loadStart).Round(time.Millisecond),
		float64(len(vectors))/1024/1024)

	idx := &Index{Constants: constants, vectors: vectors, labels: labels}

	switch kind {
	case KindBrute:
		log.Printf("using brute force search (no secondary structure)")
	case KindVP:
		treeStart := time.Now()
		log.Printf("building VP-tree over %d references ...", len(labels))
		idx.tree = buildVPTree(vectors, len(labels))
		log.Printf("VP-tree ready in %s (%d nodes, %.1f MB)",
			time.Since(treeStart).Round(time.Millisecond),
			len(idx.tree),
			float64(len(idx.tree)*16)/1024/1024)
	case KindIVF:
		ivfStart := time.Now()
		log.Printf("building IVF index (%d centroids, sample=%d, iters=%d, nprobe=%d) ...",
			defaultNumCentroids, defaultKMeansSampleSize, defaultKMeansIterations, defaultNprobe)
		idx.ivf = buildIVF(vectors, len(labels),
			defaultNumCentroids, defaultKMeansIterations, defaultKMeansSampleSize, defaultNprobe)
		log.Printf("IVF ready in %s (%d centroids, %.1f MB)",
			time.Since(ivfStart).Round(time.Millisecond),
			idx.ivf.numCentroids,
			float64(len(idx.ivf.refOrder)*4+len(idx.ivf.centroids))/1024/1024)
	default:
		return nil, fmt.Errorf("unknown index kind: %q", kind)
	}

	return idx, nil
}

// Score vectorizes the payload, quantizes it, runs k-NN, and returns
// `approved` + `fraud_score` per the rules.
func (idx *Index) Score(payload *Payload) (approved bool, score float32, err error) {
	count, err := idx.FraudCount(payload)
	if err != nil {
		return false, 0, err
	}
	score = float32(count) / float32(K)
	approved = score < Threshold
	return approved, score, nil
}

// FraudCount returns the number of "fraud"-labeled neighbors among the K
// nearest references for the given payload. Hot-path callers should use this
// directly and pick a pre-serialized response — it skips the float division
// and bool comparison Score wraps around.
func (idx *Index) FraudCount(payload *Payload) (int, error) {
	queryFloat, err := Vectorize(payload, idx.Constants)
	if err != nil {
		return 0, err
	}
	queryVector := quantizeVector(queryFloat)
	top := idx.topK(queryVector)
	count := 0
	for _, n := range top {
		if n.label == labelFraud {
			count++
		}
	}
	return count, nil
}

// topK dispatches to the most efficient available search structure:
// IVF (approximate) → VP-tree (exact) → brute force (exact, fallback).
func (idx *Index) topK(query [physicalStride]int16) [K]neighbor {
	if idx.ivf != nil {
		return idx.ivfTopK(query)
	}
	if len(idx.tree) > 0 {
		return idx.vpTreeTopK(query)
	}
	return idx.bruteForceTopK(query)
}

// neighbor is a single reference ranked by its squared L2 distance to the query.
// The distance is stored as int32: max per-dim diff is 255 (between -128 and
// +127), so per-dim diff² ≤ 65025, and the 14-dim sum stays comfortably in int32.
type neighbor struct {
	squaredDistance int32
	label           uint8
}

// bruteForceTopK scans every reference and returns the K nearest, sorted
// ascending by squared L2 distance. Used as a correctness oracle for the
// VP-tree search and as the fallback when no tree is built.
//
// The inner distance is computed via l2SquaredDistanceInt16, which dispatches
// to AVX2 when available. The SIMD kernel processes all 14 dims (16 lanes
// including pad) in a single VMOVDQU + VPSUBW + VPMADDWD chain, so the per-
// ref `bound` check is the only early-exit we need.
func (idx *Index) bruteForceTopK(query [physicalStride]int16) [K]neighbor {
	// nearestNeighbors stays sorted ascending by distance:
	//   [0] = closest, [K-1] = K-th closest.
	var nearestNeighbors [K]neighbor
	for slot := range nearestNeighbors {
		nearestNeighbors[slot].squaredDistance = math.MaxInt32
	}
	// kthBestDistance is the "admission threshold" into the top-K. Any reference
	// whose distance is ≥ this can be skipped without further work.
	kthBestDistance := nearestNeighbors[K-1].squaredDistance

	querySlice := query[:]
	referenceCount := len(idx.labels)
	for referenceIndex := 0; referenceIndex < referenceCount; referenceIndex++ {
		vectorOffset := referenceIndex * physicalStride
		refSlice := idx.vectors[vectorOffset : vectorOffset+physicalStride]

		squaredDistance := l2SquaredDistanceInt16(querySlice, refSlice)
		if squaredDistance >= kthBestDistance {
			continue
		}

		// Insertion-sort into the top-K, then refresh the threshold.
		referenceLabel := idx.labels[referenceIndex]
		insertionPos := K - 1
		for insertionPos > 0 && nearestNeighbors[insertionPos-1].squaredDistance > squaredDistance {
			nearestNeighbors[insertionPos] = nearestNeighbors[insertionPos-1]
			insertionPos--
		}
		nearestNeighbors[insertionPos] = neighbor{squaredDistance: squaredDistance, label: referenceLabel}
		kthBestDistance = nearestNeighbors[K-1].squaredDistance
	}
	return nearestNeighbors
}

// knnFraudCount is a thin wrapper around bruteForceTopK that returns the
// number of "fraud"-labeled references among the K nearest. Kept for tests
// and benchmarks that want to exercise the brute-force path.
func (idx *Index) knnFraudCount(query [physicalStride]int16) int {
	top := idx.bruteForceTopK(query)
	count := 0
	for _, n := range top {
		if n.label == labelFraud {
			count++
		}
	}
	return count
}

// Size returns the number of references loaded into the index.
func (idx *Index) Size() int { return len(idx.labels) }
