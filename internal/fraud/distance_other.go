//go:build !amd64

package fraud

var hasAVX2 = false

func l2SquaredDistanceInt16AVX2(query, ref *int16) uint32 {
	panic("l2SquaredDistanceInt16AVX2 called on non-amd64 platform")
}
