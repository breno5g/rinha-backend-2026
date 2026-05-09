//go:build !amd64

package fraud

// hasAVX2 is permanently false on non-amd64 platforms; the scalar fallback in
// distance.go is the only available path.
var hasAVX2 = false

// l2SquaredDistanceInt16AVX2 is never called on non-amd64 (hasAVX2 is false),
// but the symbol must exist so distance.go compiles unconditionally.
func l2SquaredDistanceInt16AVX2(query, ref *int16) uint32 {
	panic("l2SquaredDistanceInt16AVX2 called on non-amd64 platform")
}
