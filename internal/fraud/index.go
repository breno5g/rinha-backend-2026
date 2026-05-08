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
)

// Index holds the reference dataset in a flat layout for cache-friendly scanning.
// `vectors` is a single contiguous slice of length len(labels) * VectorDim:
// the i-th reference vector lives at vectors[i*VectorDim : (i+1)*VectorDim].
type Index struct {
	Constants *Constants
	vectors   []float32
	labels    []uint8
}

type referenceEntry struct {
	Vector []float32 `json:"vector"`
	Label  string    `json:"label"`
}

// LoadIndex reads and decompresses references.json.gz, parses each entry,
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

	// Pre-allocate assuming ~3M entries; the slices grow if the dataset is larger.
	const expectedReferenceCount = 3_000_000
	vectors := make([]float32, 0, expectedReferenceCount*VectorDim)
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
		vectors = append(vectors, entry.Vector...)

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

	log.Printf("loaded %d references in %s (%.1f MB of vectors)",
		len(labels), time.Since(loadStart).Round(time.Millisecond),
		float64(len(vectors)*4)/1024/1024)

	return &Index{Constants: constants, vectors: vectors, labels: labels}, nil
}

// Score vectorizes the payload, finds the K nearest references via brute force
// on squared L2 distance, and returns `approved` + `fraud_score` per the rules.
func (idx *Index) Score(payload *Payload) (approved bool, score float32, err error) {
	queryVector, err := Vectorize(payload, idx.Constants)
	if err != nil {
		return false, 0, err
	}
	fraudCount := idx.knnFraudCount(queryVector)
	score = float32(fraudCount) / float32(K)
	approved = score < Threshold
	return approved, score, nil
}

// neighbor is a single reference ranked by its squared L2 distance to the query.
type neighbor struct {
	squaredDistance float32
	label           uint8
}

// knnFraudCount returns the number of "fraud"-labeled references among the K
// nearest to the query vector. Uses an insertion-sorted top-K array — at K=5
// this beats a heap because of constant-factor wins on tiny K.
func (idx *Index) knnFraudCount(query [VectorDim]float32) int {
	// nearestNeighbors stays sorted ascending by distance:
	//   [0] = closest, [K-1] = K-th closest.
	var nearestNeighbors [K]neighbor
	for slot := range nearestNeighbors {
		nearestNeighbors[slot].squaredDistance = math.MaxFloat32
	}
	// kthBestDistance is the "admission threshold" into the top-K. Any reference
	// whose distance reaches this value can be skipped without further work.
	kthBestDistance := nearestNeighbors[K-1].squaredDistance

	referenceCount := len(idx.labels)
	for referenceIndex := 0; referenceIndex < referenceCount; referenceIndex++ {
		vectorOffset := referenceIndex * VectorDim

		// Accumulate squared L2 dimension-by-dimension. Bail out early if the
		// partial sum already crosses the admission threshold — saves us the
		// remaining multiplications/additions on this reference.
		var squaredDistance float32
		for dim := 0; dim < VectorDim; dim++ {
			diff := query[dim] - idx.vectors[vectorOffset+dim]
			squaredDistance += diff * diff
			if squaredDistance >= kthBestDistance {
				break
			}
		}
		if squaredDistance >= kthBestDistance {
			continue
		}

		// Insertion-sort this reference into the top-K, then refresh the
		// admission threshold so the next iteration prunes more aggressively.
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
