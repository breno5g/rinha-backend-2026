package fraud

import (
	"math"
	"sort"
)

// vpNode is one node in a vantage-point tree. The tree stores only int32
// indices into the parent Index's flat vectors/labels arrays — it doesn't
// duplicate any vector data, just the partitioning structure.
//
// At 16 bytes per node, a 3M-reference tree fits in ~48 MB.
type vpNode struct {
	vantageRefIdx int32   // index into Index.vectors / Index.labels
	radius        float32 // L2 distance threshold (median to children); pre-sqrt'd at build
	insideIdx     int32   // child holding refs with distance ≤ radius (-1 if none)
	outsideIdx    int32   // child holding refs with distance >  radius (-1 if none)
}

// buildVPTree partitions every reference into a vantage-point tree, returning
// a flat node array (root at index 0). Each ref appears as the vantage of
// exactly one node, so the tree has exactly numRefs nodes.
//
// Build cost is O(N log N): at each level, every still-unassigned ref is
// distance-checked against the local vantage, then sorted by that distance
// to find the median radius.
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

		// Pick the first ref as the vantage. Random selection would balance
		// pathological cases better, but the dataset is shuffled enough that
		// sequential picks build well-balanced trees in practice.
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

		// For every remaining ref, compute the squared L2 distance to the vantage.
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

		// Sort to find the median; the median distance becomes this node's radius.
		sort.Slice(distances, func(i, j int) bool {
			return distances[i].distSq < distances[j].distSq
		})
		medianPos := len(distances) / 2
		nodes[nodeIdx].radius = float32(math.Sqrt(float64(distances[medianPos].distSq)))

		// Refs with distance ≤ median go inside; the rest go outside.
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

// squaredDistanceBetweenRefs computes the squared L2 distance between two
// reference vectors stored in the flat int16 array.
func squaredDistanceBetweenRefs(vectors []int16, a, b int32) int32 {
	baseA := int(a) * physicalStride
	baseB := int(b) * physicalStride
	return l2SquaredDistanceInt16(
		vectors[baseA:baseA+physicalStride],
		vectors[baseB:baseB+physicalStride],
	)
}

// squaredDistanceQueryRef computes the squared L2 distance from a query vector
// (passed by value) to the b-th reference in the flat int16 array.
func squaredDistanceQueryRef(vectors []int16, query [physicalStride]int16, b int32) int32 {
	baseB := int(b) * physicalStride
	return l2SquaredDistanceInt16(query[:], vectors[baseB:baseB+physicalStride])
}

// vpTreeTopK finds the K nearest references to `query` by traversing the
// VP-tree and pruning subtrees that the triangle inequality proves can't
// contain a closer point.
//
// The pruning rule, given the current k-th-best distance `tau` and a node
// with vantage v, radius r, and query distance d = dist(query, v):
//
//   - "inside" (refs with dist ≤ r from v) lies between |d-r| and (d+r) from query.
//     Visit it iff d - tau ≤ r  (i.e., search ball might reach inside).
//   - "outside" (refs with dist > r from v) lies at least (r - d) from query
//     (when d < r), otherwise no useful lower bound.
//     Visit it iff d + tau ≥ r  (i.e., search ball might reach outside).
//
// We descend into the side containing the query first because that side is
// likely to tighten `tau`, which then prunes the other side more aggressively.
func (idx *Index) vpTreeTopK(query [physicalStride]int16) [K]neighbor {
	var top [K]neighbor
	for slot := range top {
		top[slot].squaredDistance = math.MaxInt32
	}
	if len(idx.tree) == 0 {
		return top
	}

	// considerCandidate inserts a ref into the sorted top-K if it's closer than
	// the current k-th-best. Returns the (possibly updated) k-th-best distance.
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

	// tauFromKthBest converts the squared distance back to L2 for the
	// triangle-inequality checks. Sentinel kthBest == MaxInt32 means "no
	// candidate yet" — return MaxFloat32 so every subtree is visited.
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
			// Query lies inside the vantage's sphere — descend inside first.
			visit(node.insideIdx)
			// The first recursion may have tightened tau; refresh before deciding outside.
			tau = tauFromKthBest(top[K-1].squaredDistance)
			if distToVantage+tau >= node.radius {
				visit(node.outsideIdx)
			}
		} else {
			// Query lies outside — descend outside first.
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
