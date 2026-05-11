//go:build amd64

package fraud

import "golang.org/x/sys/cpu"

func l2SquaredDistanceInt16AVX2(query, ref *int16) uint32

var hasAVX2 = cpu.X86.HasAVX2
