package fraud

import (
	"testing"
)

func TestVPTreeMatchesBruteForce(t *testing.T) {
	const numRefs = 1_000
	idx := makeBenchIndex(numRefs, 1)
	idx.tree = buildVPTree(idx.vectors, idx.Size())

	if len(idx.tree) != numRefs {
		t.Fatalf("expected %d nodes (one per reference), got %d", numRefs, len(idx.tree))
	}

	const numQueries = 30
	for q := 0; q < numQueries; q++ {
		query := makeBenchQuery(int64(q + 100))
		bruteForce := idx.bruteForceTopK(query)
		viaTree := idx.vpTreeTopK(query)

		for i := 0; i < K; i++ {
			if bruteForce[i].squaredDistance != viaTree[i].squaredDistance {
				t.Errorf("query %d: top-%d distance mismatch — brute force=%d, VP-tree=%d\n  brute force=%v\n  VP-tree    =%v",
					q, i, bruteForce[i].squaredDistance, viaTree[i].squaredDistance, bruteForce, viaTree)
				break
			}
		}
	}
}

func TestVPTreeBuildEmpty(t *testing.T) {
	if got := buildVPTree(nil, 0); got != nil {
		t.Errorf("buildVPTree(nil, 0) = %v, want nil", got)
	}
}

func BenchmarkKNN_VP_3M(b *testing.B) {
	idx := makeBenchIndex(3_000_000, 42)
	idx.tree = buildVPTree(idx.vectors, idx.Size())
	query := makeBenchQuery(7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.vpTreeTopK(query)
	}
}

func BenchmarkKNN_VP_100k(b *testing.B) {
	idx := makeBenchIndex(100_000, 42)
	idx.tree = buildVPTree(idx.vectors, idx.Size())
	query := makeBenchQuery(7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.vpTreeTopK(query)
	}
}
