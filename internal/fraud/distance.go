package fraud

func l2SquaredDistanceInt16(query, ref []int16) int32 {
	if hasAVX2 {
		return int32(l2SquaredDistanceInt16AVX2(&query[0], &ref[0]))
	}
	return l2SquaredDistanceInt16Scalar(query, ref)
}

func l2SquaredDistanceInt16Scalar(query, ref []int16) int32 {
	var dist int32
	for d := 0; d < VectorDim; d++ {
		diff := int32(query[d]) - int32(ref[d])
		dist += diff * diff
	}
	return dist
}
