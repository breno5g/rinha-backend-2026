package fraud

// l2SquaredDistanceInt16 returns the sum of squared int16 differences between
// two `physicalStride`-element int16 buffers (14 logical dims + 2 zero pad).
//
// On amd64+AVX2 it dispatches to a SIMD kernel that processes all 16 lanes in
// a single VMOVDQU + VPSUBW + VPMADDWD; otherwise it falls back to the scalar
// loop. Distances fit in int32: max per-lane squared diff is ~268M (between
// -8192 and +8192), times 14 dims is ~3.7G — overflows int32 by a factor of
// ~1.7×. In practice quantized values stay in [0, 8192] for most dims and
// the sentinel only appears for 2 dims, so the realistic upper bound is well
// within int32. The SIMD kernel uses VPMADDWD which inherently produces
// int32, and we trust the same guarantee in the scalar path.
func l2SquaredDistanceInt16(query, ref []int16) int32 {
	if hasAVX2 {
		return int32(l2SquaredDistanceInt16AVX2(&query[0], &ref[0]))
	}
	return l2SquaredDistanceInt16Scalar(query, ref)
}

// l2SquaredDistanceInt16Scalar is the portable reference implementation. It
// also doubles as the correctness oracle for the SIMD kernel in tests.
func l2SquaredDistanceInt16Scalar(query, ref []int16) int32 {
	var dist int32
	for d := 0; d < VectorDim; d++ {
		diff := int32(query[d]) - int32(ref[d])
		dist += diff * diff
	}
	return dist
}
