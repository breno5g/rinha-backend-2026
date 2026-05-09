//go:build amd64

package fraud

import "golang.org/x/sys/cpu"

// l2SquaredDistanceInt16AVX2 is implemented in distance_amd64.s.
// query and ref must point to 32 readable bytes (16 int16: 14 dims + 2 zero pad).
//
//go:noescape
func l2SquaredDistanceInt16AVX2(query, ref *int16) uint32

// hasAVX2 is set at startup when the CPU supports AVX2. The hot-path dispatch
// in distance.go reads this once per call to choose between the SIMD kernel
// and the scalar fallback.
var hasAVX2 = cpu.X86.HasAVX2
