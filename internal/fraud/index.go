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

	quantizationScale = 8192

	sentinelInt16 int16 = -8192
)

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

func quantize(f float32) int16 {
	if f < 0 {
		return sentinelInt16
	}
	if f >= 1 {
		return quantizationScale
	}
	return int16(math.Round(float64(f) * quantizationScale))
}

func quantizeVector(v [VectorDim]float32) [physicalStride]int16 {
	var out [physicalStride]int16
	for i, value := range v {
		out[i] = quantize(value)
	}
	return out
}

type IndexKind string

const (
	KindBrute IndexKind = "brute"

	KindVP IndexKind = "vp"

	KindIVF IndexKind = "ivf"
)

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

func (idx *Index) Score(payload *Payload) (approved bool, score float32, err error) {
	count, err := idx.FraudCount(payload)
	if err != nil {
		return false, 0, err
	}
	score = float32(count) / float32(K)
	approved = score < Threshold
	return approved, score, nil
}

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

func (idx *Index) topK(query [physicalStride]int16) [K]neighbor {
	if idx.ivf != nil {
		return idx.ivfTopK(query)
	}
	if len(idx.tree) > 0 {
		return idx.vpTreeTopK(query)
	}
	return idx.bruteForceTopK(query)
}

type neighbor struct {
	squaredDistance int32
	label           uint8
}

func (idx *Index) bruteForceTopK(query [physicalStride]int16) [K]neighbor {

	var nearestNeighbors [K]neighbor
	for slot := range nearestNeighbors {
		nearestNeighbors[slot].squaredDistance = math.MaxInt32
	}

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

func (idx *Index) Size() int { return len(idx.labels) }
