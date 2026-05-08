package fraud

import "testing"

func TestQuantize(t *testing.T) {
	cases := []struct {
		in   float32
		want int8
	}{
		{-1, sentinelInt8},
		{0, 0},
		{0.25, 32},  // round(0.25 * 127) = round(31.75) = 32
		{0.5, 64},   // round(0.5 * 127)  = round(63.5)  = 64 (Go rounds half away from zero)
		{0.7826, 99}, // matches the legit example dim 3 (0.7826 → ~99)
		{1, 127},
		{1.5, 127}, // clamped at the upper bound
	}
	for _, c := range cases {
		if got := quantize(c.in); got != c.want {
			t.Errorf("quantize(%v) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestQuantizeVector_LegitExample(t *testing.T) {
	floatVector := [VectorDim]float32{0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006}
	got := quantizeVector(floatVector)

	if got[5] != sentinelInt8 || got[6] != sentinelInt8 {
		t.Errorf("expected sentinel at dims 5,6; got %d,%d", got[5], got[6])
	}
	if got[10] != 127 {
		t.Errorf("dim 10 (card_present=1) should quantize to 127; got %d", got[10])
	}
	if got[9] != 0 || got[11] != 0 {
		t.Errorf("dims 9,11 (booleans = 0) should quantize to 0; got %d,%d", got[9], got[11])
	}
}

// TestKNN_SyntheticAllFraud sanity-checks the KNN: in a tiny dataset where
// every reference is fraud, a query must return K fraud votes.
func TestKNN_SyntheticAllFraud(t *testing.T) {
	idx := &Index{
		vectors: make([]int8, 100*VectorDim), // all zeros
		labels:  make([]uint8, 100),
	}
	for i := range idx.labels {
		idx.labels[i] = labelFraud
	}
	var query [VectorDim]int8
	if got := idx.knnFraudCount(query); got != K {
		t.Errorf("all-fraud dataset should give K=%d fraud votes, got %d", K, got)
	}
}

// TestKNN_PicksClosestNotJustFirst verifies the algorithm doesn't simply pick
// the first K references. We place K legit refs as exact matches first, then
// K fraud refs further away — the legit refs must win.
func TestKNN_PicksClosestNotJustFirst(t *testing.T) {
	const numRefs = 20
	idx := &Index{
		vectors: make([]int8, numRefs*VectorDim),
		labels:  make([]uint8, numRefs),
	}
	// First K refs: exact match (zeros), labeled legit.
	for i := 0; i < K; i++ {
		idx.labels[i] = labelLegit
	}
	// Next K refs: far away (all 100s), labeled fraud.
	for i := K; i < 2*K; i++ {
		for dim := 0; dim < VectorDim; dim++ {
			idx.vectors[i*VectorDim+dim] = 100
		}
		idx.labels[i] = labelFraud
	}
	// Remaining: don't matter.

	var query [VectorDim]int8 // zeros — exact match for the first K refs
	if got := idx.knnFraudCount(query); got != 0 {
		t.Errorf("query should match the K legit refs, got %d fraud votes", got)
	}
}
