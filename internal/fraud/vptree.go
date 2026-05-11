package fraud

import (
	"math"
	"sort"
)

type vpNode struct {
	vantageRefIdx int32
	radius        float32
	insideIdx     int32
	outsideIdx    int32
}

func buildVPTree(vectors []int16, numRefs int) []vpNode {
	if numRefs == 0 {
		return nil
	}

	allIndices := make([]int32, numRefs)
	for i := range allIndices {
		allIndices[i] = int32(i)
	}

	nodes := make([]vpNode, 0, numRefs)

	var build func(refsToPartition []int32) int32
	build = func(refsToPartition []int32) int32 {
		if len(refsToPartition) == 0 {
			return -1
		}

		vantage := refsToPartition[0]
		nodeIdx := int32(len(nodes))
		nodes = append(nodes, vpNode{
			vantageRefIdx: vantage,
			insideIdx:     -1,
			outsideIdx:    -1,
		})

		if len(refsToPartition) == 1 {
			return nodeIdx
		}

		remaining := refsToPartition[1:]
		type refDistance struct {
			distSq int32
			refIdx int32
		}
		distances := make([]refDistance, len(remaining))
		for i, refIdx := range remaining {
			distances[i] = refDistance{
				distSq: squaredDistanceBetweenRefs(vectors, vantage, refIdx),
				refIdx: refIdx,
			}
		}

		sort.Slice(distances, func(i, j int) bool {
			return distances[i].distSq < distances[j].distSq
		})
		medianPos := len(distances) / 2
		nodes[nodeIdx].radius = float32(math.Sqrt(float64(distances[medianPos].distSq)))

		insideRefs := make([]int32, medianPos)
		outsideRefs := make([]int32, len(distances)-medianPos)
		for i, d := range distances {
			if i < medianPos {
				insideRefs[i] = d.refIdx
			} else {
				outsideRefs[i-medianPos] = d.refIdx
			}
		}

		nodes[nodeIdx].insideIdx = build(insideRefs)
		nodes[nodeIdx].outsideIdx = build(outsideRefs)
		return nodeIdx
	}
	build(allIndices)
	return nodes
}

func squaredDistanceBetweenRefs(vectors []int16, a, b int32) int32 {
	baseA := int(a) * physicalStride
	baseB := int(b) * physicalStride
	return l2SquaredDistanceInt16(
		vectors[baseA:baseA+physicalStride],
		vectors[baseB:baseB+physicalStride],
	)
}

func squaredDistanceQueryRef(vectors []int16, query [physicalStride]int16, b int32) int32 {
	baseB := int(b) * physicalStride
	return l2SquaredDistanceInt16(query[:], vectors[baseB:baseB+physicalStride])
}

func (idx *Index) vpTreeTopK(query [physicalStride]int16) [K]neighbor {
	var top [K]neighbor
	for slot := range top {
		top[slot].squaredDistance = math.MaxInt32
	}
	if len(idx.tree) == 0 {
		return top
	}

	considerCandidate := func(refIdx int32, distSq int32) int32 {
		kthBest := top[K-1].squaredDistance
		if distSq >= kthBest {
			return kthBest
		}
		insertionPos := K - 1
		for insertionPos > 0 && top[insertionPos-1].squaredDistance > distSq {
			top[insertionPos] = top[insertionPos-1]
			insertionPos--
		}
		top[insertionPos] = neighbor{squaredDistance: distSq, label: idx.labels[refIdx]}
		return top[K-1].squaredDistance
	}

	tauFromKthBest := func(kthBest int32) float32 {
		if kthBest == math.MaxInt32 {
			return math.MaxFloat32
		}
		return float32(math.Sqrt(float64(kthBest)))
	}

	var visit func(nodeIdx int32)
	visit = func(nodeIdx int32) {
		if nodeIdx < 0 {
			return
		}
		node := idx.tree[nodeIdx]

		distSqToVantage := squaredDistanceQueryRef(idx.vectors, query, node.vantageRefIdx)
		kthBest := considerCandidate(node.vantageRefIdx, distSqToVantage)
		distToVantage := float32(math.Sqrt(float64(distSqToVantage)))
		tau := tauFromKthBest(kthBest)

		if distToVantage <= node.radius {

			visit(node.insideIdx)

			tau = tauFromKthBest(top[K-1].squaredDistance)
			if distToVantage+tau >= node.radius {
				visit(node.outsideIdx)
			}
		} else {

			visit(node.outsideIdx)
			tau = tauFromKthBest(top[K-1].squaredDistance)
			if distToVantage-tau <= node.radius {
				visit(node.insideIdx)
			}
		}
	}
	visit(0)
	return top
}
