package fraud

import (
	"math/rand"
	"testing"
)

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

	return q
}

func BenchmarkKNN_3M(b *testing.B) {
	idx := makeBenchIndex(3_000_000, 42)
	query := makeBenchQuery(7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.knnFraudCount(query)
	}
}

func BenchmarkKNN_100k(b *testing.B) {
	idx := makeBenchIndex(100_000, 42)
	query := makeBenchQuery(7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.knnFraudCount(query)
	}
}

func BenchmarkKNN_1k(b *testing.B) {
	idx := makeBenchIndex(1_000, 42)
	query := makeBenchQuery(7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.knnFraudCount(query)
	}
}
