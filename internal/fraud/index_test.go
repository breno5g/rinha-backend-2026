package fraud

import "testing"

func TestQuantize(t *testing.T) {
	cases := []struct {
		in   float32
		want int16
	}{
		{-1, sentinelInt16},
		{0, 0},
		{0.25, 2048},
		{0.5, 4096},
		{0.7826, 6411},
		{1, 8192},
		{1.5, 8192},
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

	if got[5] != sentinelInt16 || got[6] != sentinelInt16 {
		t.Errorf("expected sentinel at dims 5,6; got %d,%d", got[5], got[6])
	}
	if got[10] != quantizationScale {
		t.Errorf("dim 10 (card_present=1) should quantize to %d; got %d", int16(quantizationScale), got[10])
	}
	if got[9] != 0 || got[11] != 0 {
		t.Errorf("dims 9,11 (booleans = 0) should quantize to 0; got %d,%d", got[9], got[11])
	}
}

func TestKNN_SyntheticAllFraud(t *testing.T) {
	idx := &Index{
		vectors: make([]int16, 100*physicalStride),
		labels:  make([]uint8, 100),
	}
	for i := range idx.labels {
		idx.labels[i] = labelFraud
	}
	var query [physicalStride]int16
	if got := idx.knnFraudCount(query); got != K {
		t.Errorf("all-fraud dataset should give K=%d fraud votes, got %d", K, got)
	}
}

func TestKNN_PicksClosestNotJustFirst(t *testing.T) {
	const numRefs = 20
	idx := &Index{
		vectors: make([]int16, numRefs*physicalStride),
		labels:  make([]uint8, numRefs),
	}

	for i := 0; i < K; i++ {
		idx.labels[i] = labelLegit
	}

	for i := K; i < 2*K; i++ {
		for dim := 0; dim < VectorDim; dim++ {
			idx.vectors[i*physicalStride+dim] = 100
		}
		idx.labels[i] = labelFraud
	}

	var query [physicalStride]int16
	if got := idx.knnFraudCount(query); got != 0 {
		t.Errorf("query should match the K legit refs, got %d fraud votes", got)
	}
}
