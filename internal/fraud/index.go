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

	// quantizationScale maps a normalized float in [0, 1] to int8 in [0, 127].
	// We keep the negative half of int8 reserved for the sentinel below.
	quantizationScale = 127

	// sentinelInt8 represents "no last_transaction" in quantized space. Sitting
	// at -128 places the sentinel far outside [0, 127], which preserves the
	// "outsider" property of the float -1: distances from sentinel to any real
	// value stay large, exactly like in the float reference implementation.
	sentinelInt8 int8 = -128
)

// Index holds the reference dataset, quantized to int8 for memory and cache wins.
// Layout is flat: vectors[i*VectorDim : (i+1)*VectorDim] is the i-th reference.
//
// At 1 byte per dimension, 3M references occupy ~42 MB (vs ~168 MB for float32).
// The 4× reduction means more vectors fit in L2/L3 cache during scanning,
// which is where most of the speedup comes from — not the cheaper arithmetic.
type Index struct {
	Constants *Constants
	vectors   []int8
	labels    []uint8
}

type referenceEntry struct {
	Vector []float32 `json:"vector"`
	Label  string    `json:"label"`
}

// quantize maps a float32 value to its int8 representation. Values in [0, 1]
// are scaled to [0, 127]. Anything strictly negative is treated as the
// "no-data" sentinel (only -1 appears in practice). Out-of-range positive
// values are clamped to 127.
func quantize(f float32) int8 {
	if f < 0 {
		return sentinelInt8
	}
	if f >= 1 {
		return quantizationScale
	}
	return int8(math.Round(float64(f) * quantizationScale))
}

// quantizeVector applies quantize() element-wise.
func quantizeVector(v [VectorDim]float32) [VectorDim]int8 {
	var out [VectorDim]int8
	for i, value := range v {
		out[i] = quantize(value)
	}
	return out
}

// LoadIndex reads references.json.gz, quantizes each vector to int8 in place,
// and packs everything into the flat layout. It also loads normalization
// constants and the MCC risk table.
func LoadIndex(referencesGzPath, normalizationPath, mccRiskPath string) (*Index, error) {
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
	vectors := make([]int8, 0, expectedReferenceCount*VectorDim)
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

	return &Index{Constants: constants, vectors: vectors, labels: labels}, nil
}

// Score vectorizes the payload, quantizes it, and runs KNN brute force on int8.
func (idx *Index) Score(payload *Payload) (approved bool, score float32, err error) {
	queryFloat, err := Vectorize(payload, idx.Constants)
	if err != nil {
		return false, 0, err
	}
	queryVector := quantizeVector(queryFloat)
	fraudCount := idx.knnFraudCount(queryVector)
	score = float32(fraudCount) / float32(K)
	approved = score < Threshold
	return approved, score, nil
}

// neighbor is a single reference ranked by its squared L2 distance to the query.
// The distance is stored as int32: max per-dim diff is 255 (between -128 and
// +127), so per-dim diff² ≤ 65025, and the 14-dim sum stays comfortably in int32.
type neighbor struct {
	squaredDistance int32
	label           uint8
}

// knnFraudCount finds the K nearest references by squared L2 distance and
// returns how many are labeled "fraud". Uses int8 arithmetic (cheap) plus an
// insertion-sorted top-K with an "admission threshold" for early termination.
func (idx *Index) knnFraudCount(query [VectorDim]int8) int {
	// nearestNeighbors stays sorted ascending by distance:
	//   [0] = closest, [K-1] = K-th closest.
	var nearestNeighbors [K]neighbor
	for slot := range nearestNeighbors {
		nearestNeighbors[slot].squaredDistance = math.MaxInt32
	}
	// kthBestDistance is the "admission threshold" into the top-K. Any reference
	// whose partial distance reaches this value can be skipped without further work.
	kthBestDistance := nearestNeighbors[K-1].squaredDistance

	referenceCount := len(idx.labels)
	for referenceIndex := 0; referenceIndex < referenceCount; referenceIndex++ {
		vectorOffset := referenceIndex * VectorDim

		// Accumulate squared L2 dimension-by-dimension. Bail out early if the
		// partial sum already crosses the admission threshold — saves us the
		// remaining multiplications/additions on this reference.
		var squaredDistance int32
		for dim := 0; dim < VectorDim; dim++ {
			diff := int32(query[dim]) - int32(idx.vectors[vectorOffset+dim])
			squaredDistance += diff * diff
			if squaredDistance >= kthBestDistance {
				break
			}
		}
		if squaredDistance >= kthBestDistance {
			continue
		}

		// Insertion-sort into the top-K, then refresh the threshold so the
		// next iteration prunes more aggressively.
		referenceLabel := idx.labels[referenceIndex]
		insertionPos := K - 1
		for insertionPos > 0 && nearestNeighbors[insertionPos-1].squaredDistance > squaredDistance {
			nearestNeighbors[insertionPos] = nearestNeighbors[insertionPos-1]
			insertionPos--
		}
		nearestNeighbors[insertionPos] = neighbor{squaredDistance: squaredDistance, label: referenceLabel}
		kthBestDistance = nearestNeighbors[K-1].squaredDistance
	}

	fraudCount := 0
	for _, topNeighbor := range nearestNeighbors {
		if topNeighbor.label == labelFraud {
			fraudCount++
		}
	}
	return fraudCount
}

// Size returns the number of references loaded into the index.
func (idx *Index) Size() int { return len(idx.labels) }
