package fraud

import (
	"math/rand"
	"testing"
)

// makeBenchIndex builds a synthetic Index of `numReferences` rows that mimics
// the real dataset shape: most dims in the [0,127] quantized range, with the
// `last_transaction`-related dims occasionally taking the sentinel value.
//
// The numbers don't need to be statistically realistic — we're benchmarking
// the SCAN, not the classifier. We just need entries that exercise the early-
// termination branch the same way real data would.
func makeBenchIndex(numReferences int, seed int64) *Index {
	rng := rand.New(rand.NewSource(seed))
	vectors := make([]int16, 0, numReferences*physicalStride)
	labels := make([]uint8, 0, numReferences)

	for i := 0; i < numReferences; i++ {
		for dim := 0; dim < VectorDim; dim++ {
			isSentinelDim := dim == 5 || dim == 6
			if isSentinelDim && rng.Float32() < 0.30 {
				vectors = append(vectors, sentinelInt16)
				continue
			}
			vectors = append(vectors, int16(rng.Intn(128)))
		}
		// Zero pad to physicalStride.
		for pad := VectorDim; pad < physicalStride; pad++ {
			vectors = append(vectors, 0)
		}
		if rng.Float32() < 0.30 {
			labels = append(labels, labelFraud)
		} else {
			labels = append(labels, labelLegit)
		}
	}
	return &Index{vectors: vectors, labels: labels}
}

func makeBenchQuery(seed int64) [physicalStride]int16 {
	rng := rand.New(rand.NewSource(seed))
	var q [physicalStride]int16
	for dim := 0; dim < VectorDim; dim++ {
		q[dim] = int16(rng.Intn(128))
	}
	// q[VectorDim:physicalStride] stays zero.
	return q
}

// BenchmarkKNN_3M is the production-scale scan: 3M vectors, 14 dims each.
// Run with: go test -bench=KNN_3M -benchtime=10x ./internal/fraud
func BenchmarkKNN_3M(b *testing.B) {
	idx := makeBenchIndex(3_000_000, 42)
	query := makeBenchQuery(7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.knnFraudCount(query)
	}
}

// BenchmarkKNN_100k is a fast feedback loop while iterating on the inner loop.
func BenchmarkKNN_100k(b *testing.B) {
	idx := makeBenchIndex(100_000, 42)
	query := makeBenchQuery(7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.knnFraudCount(query)
	}
}

// BenchmarkKNN_1k is for testing the dispatch / setup overhead, not the scan.
func BenchmarkKNN_1k(b *testing.B) {
	idx := makeBenchIndex(1_000, 42)
	query := makeBenchQuery(7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.knnFraudCount(query)
	}
}
