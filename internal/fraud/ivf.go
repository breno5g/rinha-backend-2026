package fraud

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
)

// SetIVFNprobe overrides the configured nprobe at runtime. Returns an error
// when the index has no IVF structure or when the value is out of range.
func (idx *Index) SetIVFNprobe(value string) error {
	if idx.ivf == nil {
		return fmt.Errorf("IVF index is not built; cannot tune nprobe")
	}
	n, err := strconv.Atoi(value)
	if err != nil || n < 1 || n > idx.ivf.numCentroids {
		return fmt.Errorf("nprobe must be an integer in [1, %d], got %q", idx.ivf.numCentroids, value)
	}
	idx.ivf.nprobe = n
	return nil
}

// SetIVFFullNprobe configures the second-pass probe count for two-stage search.
// Set to 0 (or any value ≤ nprobe) to disable two-stage and always use the
// fast pass.
func (idx *Index) SetIVFFullNprobe(value string) error {
	if idx.ivf == nil {
		return fmt.Errorf("IVF index is not built; cannot tune fullNprobe")
	}
	n, err := strconv.Atoi(value)
	if err != nil || n < 0 || n > idx.ivf.numCentroids {
		return fmt.Errorf("fullNprobe must be an integer in [0, %d], got %q", idx.ivf.numCentroids, value)
	}
	idx.ivf.fullNprobe = n
	return nil
}

// IVF (Inverted File) index parameters.
//
// numCentroids — how finely we partition the space. Standard rule of thumb is
// √N ≈ 1700 for 3M points; we use a smaller value here because k-means cost
// scales linearly with this number and we want a tractable startup time.
//
// nprobe — how many of the nearest centroids we actually scan at query time.
// Tunes the recall × latency trade-off:
//
//   nprobe = 1   →  fastest, lowest recall
//   nprobe = 32  →  slower, near-perfect recall
//
// kMeansSampleSize / kMeansIterations control build cost. K-means runs on a
// random sample (cheap), then we assign every ref to its nearest centroid.
const (
	defaultNumCentroids     = 2048
	defaultNprobe           = 8
	defaultKMeansSampleSize = 100_000
	defaultKMeansIterations = 12
)

// ivfIndex stores everything needed to do an IVF lookup.
//
// Memory profile for 3M refs and 256 centroids:
//   centroids       :   256 × 14 = 3.5 KB
//   refOrder        :   3M  × 4  = 12 MB
//   clusterOffsets  :   257 × 4  = 1 KB
//
// All ~12 MB total — much lighter than the VP-tree's 46 MB.
type ivfIndex struct {
	centroids      []int8  // flat: numCentroids × VectorDim
	numCentroids   int
	nprobe         int     // fast/single-stage probe count
	fullNprobe     int     // expanded probe count for borderline queries (0 disables two-stage)
	refOrder       []int32 // refs reordered by cluster
	clusterOffsets []int32 // length = numCentroids+1; cluster c spans refOrder[clusterOffsets[c]:clusterOffsets[c+1]]
}

// buildIVF runs k-means on a sample of `vectors`, assigns every ref to its
// nearest centroid, and packs the results into an inverted file structure.
func buildIVF(vectors []int8, numRefs, numCentroids, kMeansIters, sampleSize, nprobe int) *ivfIndex {
	centroids := initializeCentroids(vectors, numRefs, numCentroids, sampleSize)
	runKMeansOnSample(vectors, numRefs, centroids, numCentroids, sampleSize, kMeansIters)

	allAssignments := assignAllRefs(vectors, numRefs, centroids, numCentroids)
	refOrder, clusterOffsets := buildInvertedLists(allAssignments, numCentroids)

	return &ivfIndex{
		centroids:      centroids,
		numCentroids:   numCentroids,
		nprobe:         nprobe,
		refOrder:       refOrder,
		clusterOffsets: clusterOffsets,
	}
}

// initializeCentroids picks numCentroids distinct refs at random as the seed
// centroids. K-means iterations refine these.
func initializeCentroids(vectors []int8, numRefs, numCentroids, sampleSize int) []int8 {
	if sampleSize > numRefs {
		sampleSize = numRefs
	}
	rng := rand.New(rand.NewSource(42))
	pickedRefs := make(map[int32]bool, numCentroids)
	centroids := make([]int8, numCentroids*VectorDim)
	for c := 0; c < numCentroids; c++ {
		var pickedRef int32
		for {
			pickedRef = int32(rng.Intn(numRefs))
			if !pickedRefs[pickedRef] {
				pickedRefs[pickedRef] = true
				break
			}
		}
		copy(centroids[c*VectorDim:(c+1)*VectorDim],
			vectors[int(pickedRef)*VectorDim:(int(pickedRef)+1)*VectorDim])
	}
	return centroids
}

// runKMeansOnSample iterates the standard k-means loop on a random sample
// of the dataset. Iterating on a sample (not the full 3M) is the only reason
// build stays fast: the sample is large enough to converge representative
// centroids, but small enough that O(samples × centroids × dim) is cheap.
func runKMeansOnSample(vectors []int8, numRefs int, centroids []int8, numCentroids, sampleSize, iters int) {
	if sampleSize > numRefs {
		sampleSize = numRefs
	}
	rng := rand.New(rand.NewSource(7))

	sampleIndices := make([]int32, sampleSize)
	for i := range sampleIndices {
		sampleIndices[i] = int32(rng.Intn(numRefs))
	}

	sampleAssignments := make([]int32, sampleSize)
	dimSums := make([]int64, numCentroids*VectorDim)
	clusterCounts := make([]int32, numCentroids)

	for iter := 0; iter < iters; iter++ {
		// Assign step: every sample ref → nearest centroid.
		for i, refIdx := range sampleIndices {
			sampleAssignments[i] = nearestCentroidBruteForce(vectors, refIdx, centroids, numCentroids)
		}

		// Update step: each centroid becomes the mean of its assigned refs.
		for i := range dimSums {
			dimSums[i] = 0
		}
		for i := range clusterCounts {
			clusterCounts[i] = 0
		}
		for i, refIdx := range sampleIndices {
			cluster := sampleAssignments[i]
			clusterCounts[cluster]++
			refOffset := int(refIdx) * VectorDim
			centroidOffset := int(cluster) * VectorDim
			for d := 0; d < VectorDim; d++ {
				dimSums[centroidOffset+d] += int64(vectors[refOffset+d])
			}
		}
		for c := 0; c < numCentroids; c++ {
			if clusterCounts[c] == 0 {
				// Empty cluster — re-seed from a random sample ref so we don't
				// lose this slot for the next iteration.
				resampledRef := sampleIndices[rng.Intn(len(sampleIndices))]
				copy(centroids[c*VectorDim:(c+1)*VectorDim],
					vectors[int(resampledRef)*VectorDim:(int(resampledRef)+1)*VectorDim])
				continue
			}
			centroidOffset := c * VectorDim
			count := int64(clusterCounts[c])
			for d := 0; d < VectorDim; d++ {
				centroids[centroidOffset+d] = int8(dimSums[centroidOffset+d] / count)
			}
		}
	}
}

// assignAllRefs maps every ref to its nearest centroid. This is the most
// expensive step of the build (numRefs × numCentroids × dim operations) but
// it's a one-time cost at startup and embarrassingly parallelizable if needed.
func assignAllRefs(vectors []int8, numRefs int, centroids []int8, numCentroids int) []int32 {
	assignments := make([]int32, numRefs)
	for refIdx := 0; refIdx < numRefs; refIdx++ {
		assignments[refIdx] = nearestCentroidBruteForce(vectors, int32(refIdx), centroids, numCentroids)
	}
	return assignments
}

// nearestCentroidBruteForce returns the index of the centroid closest to the
// given ref. Used both during k-means iterations (on sample) and final
// assignment (on the whole dataset).
func nearestCentroidBruteForce(vectors []int8, refIdx int32, centroids []int8, numCentroids int) int32 {
	var bestCentroid int32 = 0
	var bestDistance int32 = math.MaxInt32
	refOffset := int(refIdx) * VectorDim

	for c := 0; c < numCentroids; c++ {
		centroidOffset := c * VectorDim
		var distance int32
		for d := 0; d < VectorDim; d++ {
			diff := int32(vectors[refOffset+d]) - int32(centroids[centroidOffset+d])
			distance += diff * diff
			if distance >= bestDistance {
				break
			}
		}
		if distance < bestDistance {
			bestDistance = distance
			bestCentroid = int32(c)
		}
	}
	return bestCentroid
}

// buildInvertedLists rearranges refs so that all refs assigned to cluster c
// occupy a contiguous range refOrder[clusterOffsets[c]:clusterOffsets[c+1]].
// This lets the search loop scan a cluster as a tight, cache-friendly range.
func buildInvertedLists(assignments []int32, numCentroids int) (refOrder []int32, clusterOffsets []int32) {
	clusterCounts := make([]int32, numCentroids)
	for _, c := range assignments {
		clusterCounts[c]++
	}

	clusterOffsets = make([]int32, numCentroids+1)
	var runningOffset int32
	for c := 0; c < numCentroids; c++ {
		clusterOffsets[c] = runningOffset
		runningOffset += clusterCounts[c]
	}
	clusterOffsets[numCentroids] = runningOffset

	refOrder = make([]int32, len(assignments))
	cursors := make([]int32, numCentroids)
	copy(cursors, clusterOffsets[:numCentroids])
	for refIdx, cluster := range assignments {
		refOrder[cursors[cluster]] = int32(refIdx)
		cursors[cluster]++
	}
	return refOrder, clusterOffsets
}

// ivfTopK is the public k-NN entry point. If `fullNprobe` is configured and
// the cheap pass returns a borderline result (2 or 3 fraud votes among top-K),
// it re-runs with the larger probe budget to confirm. Otherwise it returns
// the fast result directly.
//
// Why two-stage: most queries are clearly fraud (5/0 votes) or clearly legit
// (0/5). Only the ~20-30% borderline queries — where the fast pass returns
// 2 or 3 fraud votes — actually benefit from a wider probe radius. Spending
// the extra work only when needed cuts the average per-query CPU significantly
// while keeping the recall high on the cases that matter most.
func (idx *Index) ivfTopK(query [VectorDim]int8) [K]neighbor {
	var top [K]neighbor
	for slot := range top {
		top[slot].squaredDistance = math.MaxInt32
	}
	if idx.ivf == nil {
		return top
	}

	fast := idx.ivfTopKWithNprobe(query, idx.ivf.nprobe)
	if idx.ivf.fullNprobe <= idx.ivf.nprobe {
		return fast
	}
	fraudVotes := 0
	for _, n := range fast {
		if n.label == labelFraud {
			fraudVotes++
		}
	}
	// 0/1 → clearly legit; 4/5 → clearly fraud. 2/3 is the gray zone.
	if fraudVotes <= 1 || fraudVotes >= 4 {
		return fast
	}
	return idx.ivfTopKWithNprobe(query, idx.ivf.fullNprobe)
}

// ivfTopKWithNprobe runs a single-pass IVF search with the given probe count.
// Computes distance to every centroid, picks the top-nprobe nearest, then
// brute-forces inside those clusters maintaining a top-K with an admission
// threshold and early termination per ref.
func (idx *Index) ivfTopKWithNprobe(query [VectorDim]int8, nprobe int) [K]neighbor {
	var top [K]neighbor
	for slot := range top {
		top[slot].squaredDistance = math.MaxInt32
	}
	if idx.ivf == nil {
		return top
	}
	ivf := idx.ivf

	// Step 1: query → all centroids.
	type centroidDistance struct {
		clusterIdx int32
		distance   int32
	}
	centroidDistances := make([]centroidDistance, ivf.numCentroids)
	for c := 0; c < ivf.numCentroids; c++ {
		centroidOffset := c * VectorDim
		var distance int32
		for d := 0; d < VectorDim; d++ {
			diff := int32(query[d]) - int32(ivf.centroids[centroidOffset+d])
			distance += diff * diff
		}
		centroidDistances[c] = centroidDistance{int32(c), distance}
	}

	// Step 2: pick top-nprobe nearest centroids. Full sort is fine since
	// numCentroids is small (≤4096).
	sort.Slice(centroidDistances, func(i, j int) bool {
		return centroidDistances[i].distance < centroidDistances[j].distance
	})

	probesToScan := nprobe
	if probesToScan > ivf.numCentroids {
		probesToScan = ivf.numCentroids
	}

	// Step 3: scan refs inside the chosen clusters.
	kthBest := top[K-1].squaredDistance
	for probe := 0; probe < probesToScan; probe++ {
		cluster := centroidDistances[probe].clusterIdx
		clusterStart := ivf.clusterOffsets[cluster]
		clusterEnd := ivf.clusterOffsets[cluster+1]

		for cursor := clusterStart; cursor < clusterEnd; cursor++ {
			refIdx := ivf.refOrder[cursor]
			refOffset := int(refIdx) * VectorDim

			var distance int32
			for d := 0; d < VectorDim; d++ {
				diff := int32(query[d]) - int32(idx.vectors[refOffset+d])
				distance += diff * diff
				if distance >= kthBest {
					break
				}
			}
			if distance >= kthBest {
				continue
			}

			refLabel := idx.labels[refIdx]
			insertionPos := K - 1
			for insertionPos > 0 && top[insertionPos-1].squaredDistance > distance {
				top[insertionPos] = top[insertionPos-1]
				insertionPos--
			}
			top[insertionPos] = neighbor{squaredDistance: distance, label: refLabel}
			kthBest = top[K-1].squaredDistance
		}
	}
	return top
}
