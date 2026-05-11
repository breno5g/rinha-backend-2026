package fraud

import (
	"testing"
)

func recallAt5(bruteForce, viaIVF [K]neighbor) float64 {
	bfDistances := map[int32]struct{}{}
	for _, n := range bruteForce {
		bfDistances[n.squaredDistance] = struct{}{}
	}
	hits := 0
	for _, n := range viaIVF {
		if _, ok := bfDistances[n.squaredDistance]; ok {
			hits++
		}
	}
	return float64(hits) / float64(K)
}

func TestIVF_RecallAcrossNprobe(t *testing.T) {
	const numRefs = 50_000
	const numCentroids = 64
	const numQueries = 50

	idx := makeBenchIndex(numRefs, 1)
	idx.ivf = buildIVF(idx.vectors, numRefs, numCentroids, 5, 5_000, 1)

	queries := make([][physicalStride]int16, numQueries)
	for q := range queries {
		queries[q] = makeBenchQuery(int64(q + 1000))
	}

	bruteForceResults := make([][K]neighbor, numQueries)
	for q, query := range queries {
		bruteForceResults[q] = idx.bruteForceTopK(query)
	}

	probes := []int{1, 4, 8, 16, 32, numCentroids}
	t.Logf("recall@5 across nprobe values (%d centroids, %d queries):", numCentroids, numQueries)
	prevRecall := 0.0
	for _, p := range probes {
		idx.ivf.nprobe = p
		var totalRecall float64
		for q, query := range queries {
			ivfTop := idx.ivfTopK(query)
			totalRecall += recallAt5(bruteForceResults[q], ivfTop)
		}
		avgRecall := totalRecall / float64(numQueries)
		t.Logf("  nprobe=%-3d  recall@5=%.3f", p, avgRecall)
		if avgRecall < prevRecall-0.01 {
			t.Errorf("recall regressed at nprobe=%d: prev=%.3f, now=%.3f", p, prevRecall, avgRecall)
		}
		prevRecall = avgRecall
	}

	idx.ivf.nprobe = numCentroids
	for q, query := range queries {
		ivfTop := idx.ivfTopK(query)
		if r := recallAt5(bruteForceResults[q], ivfTop); r < 1.0 {
			t.Errorf("query %d: nprobe=numCentroids should give recall=1.0, got %.3f", q, r)
		}
	}
}

func BenchmarkKNN_IVF_3M(b *testing.B) {
	idx := makeBenchIndex(3_000_000, 42)
	idx.ivf = buildIVF(idx.vectors, idx.Size(),
		defaultNumCentroids, defaultKMeansIterations, defaultKMeansSampleSize, defaultNprobe)
	query := makeBenchQuery(7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.ivfTopK(query)
	}
}

func BenchmarkKNN_IVF_3M_nprobe1(b *testing.B) {
	idx := makeBenchIndex(3_000_000, 42)
	idx.ivf = buildIVF(idx.vectors, idx.Size(),
		defaultNumCentroids, defaultKMeansIterations, defaultKMeansSampleSize, 1)
	query := makeBenchQuery(7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.ivfTopK(query)
	}
}

func BenchmarkKNN_IVF_3M_nprobe64(b *testing.B) {
	idx := makeBenchIndex(3_000_000, 42)
	idx.ivf = buildIVF(idx.vectors, idx.Size(),
		defaultNumCentroids, defaultKMeansIterations, defaultKMeansSampleSize, 64)
	query := makeBenchQuery(7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.ivfTopK(query)
	}
}
