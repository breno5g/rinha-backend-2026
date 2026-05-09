//go:build amd64

#include "textflag.h"

// l2SquaredDistanceInt16AVX2 computes the squared L2 distance between two
// 16-lane int16 buffers (32 bytes each) using AVX2.
//
//   func l2SquaredDistanceInt16AVX2(query, ref *int16) uint32
//
// Both pointers must reference at least 32 readable bytes. Logical
// VectorDim=14 dims live at lanes 0..13; lanes 14..15 are zero pad so the
// full 16-lane load contributes nothing extra to the sum.
//
// Strategy:
//   VMOVDQU    : load 16 int16 (one ymm)
//   VPSUBW     : per-lane int16 subtract (diff)
//   VPMADDWD   : (a*a) + (b*b) per pair → 8 int32 partial sums
//   horizontal-reduce 8 int32 → single int32
TEXT ·l2SquaredDistanceInt16AVX2(SB), NOSPLIT, $0-20
	MOVQ query+0(FP), AX
	MOVQ ref+8(FP), BX

	VMOVDQU  (AX), Y0
	VMOVDQU  (BX), Y1
	VPSUBW   Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3

	// Reduce 8 int32 in Y3 to 1 int32 in low lane of X3.
	VEXTRACTI128 $1, Y3, X4    // X4 = upper 128 bits of Y3 (4 int32)
	VPADDD       X4, X3, X3    // X3 = 4 int32 partial sums
	VPSHUFD      $0x4E, X3, X4 // swap pairs: [c,d,a,b]
	VPADDD       X4, X3, X3    // X3 = 4 × (a+c, b+d, ...)
	VPSHUFD      $0xB1, X3, X4 // swap doublewords: [b+d, a+c, ...]
	VPADDD       X4, X3, X3    // X3 lane 0 = a+b+c+d

	VMOVD X3, AX
	VZEROUPPER
	MOVL  AX, ret+16(FP)
	RET
