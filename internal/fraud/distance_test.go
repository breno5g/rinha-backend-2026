package fraud

import (
	"math/rand"
	"testing"
)

func TestL2SquaredDistance_KernelMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 1000; trial++ {
		query := make([]int16, physicalStride)
		ref := make([]int16, physicalStride)
		for d := 0; d < VectorDim; d++ {
			query[d] = int16(rng.Intn(quantizationScale + 1))
			ref[d] = int16(rng.Intn(quantizationScale + 1))
		}

		got := l2SquaredDistanceInt16(query, ref)
		want := l2SquaredDistanceInt16Scalar(query, ref)
		if got != want {
			t.Fatalf("trial %d: kernel=%d scalar=%d query=%v ref=%v",
				trial, got, want, query[:VectorDim], ref[:VectorDim])
		}
	}
}

func TestL2SquaredDistance_PadIsIgnored(t *testing.T) {
	query := make([]int16, physicalStride)
	ref := make([]int16, physicalStride)
	for d := 0; d < VectorDim; d++ {
		query[d] = int16(d * 100)
		ref[d] = int16(d * 200)
	}
	wantBaseline := l2SquaredDistanceInt16Scalar(query, ref)
	if got := l2SquaredDistanceInt16(query, ref); got != wantBaseline {
		t.Fatalf("zero-pad: kernel=%d scalar=%d", got, wantBaseline)
	}
}

func TestL2SquaredDistance_SentinelMagnitude(t *testing.T) {
	query := make([]int16, physicalStride)
	ref := make([]int16, physicalStride)
	query[5] = sentinelInt16
	ref[5] = 100
	got := l2SquaredDistanceInt16(query, ref)
	want := l2SquaredDistanceInt16Scalar(query, ref)
	if got != want {
		t.Fatalf("sentinel: kernel=%d scalar=%d", got, want)
	}

	if got < 60_000_000 {
		t.Fatalf("sentinel produced unexpectedly small distance: %d", got)
	}
}
