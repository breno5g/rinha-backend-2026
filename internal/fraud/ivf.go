package fraud

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"strconv"
	"sync"
)

type centroidDistance struct {
	clusterIdx int32
	distance   int32
}

var centroidScratchPool = sync.Pool{
	New: func() any {
		s := make([]centroidDistance, 0, defaultNumCentroids)
		return &s
	},
}

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

const (
	defaultNumCentroids     = 2048
	defaultNprobe           = 8
	defaultKMeansSampleSize = 100_000
	defaultKMeansIterations = 12
)

type ivfIndex struct {
	centroids      []int16
	numCentroids   int
	nprobe         int
	fullNprobe     int
	refOrder       []int32
	clusterOffsets []int32
}

func buildIVF(vectors []int16, numRefs, numCentroids, kMeansIters, sampleSize, nprobe int) *ivfIndex {
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

func initializeCentroids(vectors []int16, numRefs, numCentroids, sampleSize int) []int16 {
	if sampleSize > numRefs {
		sampleSize = numRefs
	}
	rng := rand.New(rand.NewSource(42))
	pickedRefs := make(map[int32]bool, numCentroids)
	centroids := make([]int16, numCentroids*physicalStride)
	for c := 0; c < numCentroids; c++ {
		var pickedRef int32
		for {
			pickedRef = int32(rng.Intn(numRefs))
			if !pickedRefs[pickedRef] {
				pickedRefs[pickedRef] = true
				break
			}
		}
		copy(centroids[c*physicalStride:(c+1)*physicalStride],
			vectors[int(pickedRef)*physicalStride:(int(pickedRef)+1)*physicalStride])
	}
	return centroids
}

func runKMeansOnSample(vectors []int16, numRefs int, centroids []int16, numCentroids, sampleSize, iters int) {
	if sampleSize > numRefs {
		sampleSize = numRefs
	}
	rng := rand.New(rand.NewSource(7))

	sampleIndices := make([]int32, sampleSize)
	for i := range sampleIndices {
		sampleIndices[i] = int32(rng.Intn(numRefs))
	}

	sampleAssignments := make([]int32, sampleSize)

	dimSums := make([]int64, numCentroids*physicalStride)
	clusterCounts := make([]int32, numCentroids)

	for iter := 0; iter < iters; iter++ {

		for i, refIdx := range sampleIndices {
			sampleAssignments[i] = nearestCentroidBruteForce(vectors, refIdx, centroids, numCentroids)
		}

		for i := range dimSums {
			dimSums[i] = 0
		}
		for i := range clusterCounts {
			clusterCounts[i] = 0
		}
		for i, refIdx := range sampleIndices {
			cluster := sampleAssignments[i]
			clusterCounts[cluster]++
			refOffset := int(refIdx) * physicalStride
			centroidOffset := int(cluster) * physicalStride
			for d := 0; d < VectorDim; d++ {
				dimSums[centroidOffset+d] += int64(vectors[refOffset+d])
			}
		}
		for c := 0; c < numCentroids; c++ {
			if clusterCounts[c] == 0 {

				resampledRef := sampleIndices[rng.Intn(len(sampleIndices))]
				copy(centroids[c*physicalStride:(c+1)*physicalStride],
					vectors[int(resampledRef)*physicalStride:(int(resampledRef)+1)*physicalStride])
				continue
			}
			centroidOffset := c * physicalStride
			count := int64(clusterCounts[c])
			for d := 0; d < VectorDim; d++ {
				centroids[centroidOffset+d] = int16(dimSums[centroidOffset+d] / count)
			}

		}
	}
}

func assignAllRefs(vectors []int16, numRefs int, centroids []int16, numCentroids int) []int32 {
	assignments := make([]int32, numRefs)
	for refIdx := 0; refIdx < numRefs; refIdx++ {
		assignments[refIdx] = nearestCentroidBruteForce(vectors, int32(refIdx), centroids, numCentroids)
	}
	return assignments
}

func nearestCentroidBruteForce(vectors []int16, refIdx int32, centroids []int16, numCentroids int) int32 {
	var bestCentroid int32 = 0
	var bestDistance int32 = math.MaxInt32
	refOffset := int(refIdx) * physicalStride
	refSlice := vectors[refOffset : refOffset+physicalStride]

	for c := 0; c < numCentroids; c++ {
		centroidOffset := c * physicalStride
		distance := l2SquaredDistanceInt16(refSlice, centroids[centroidOffset:centroidOffset+physicalStride])
		if distance < bestDistance {
			bestDistance = distance
			bestCentroid = int32(c)
		}
	}
	return bestCentroid
}

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

func (idx *Index) ivfTopK(query [physicalStride]int16) [K]neighbor {
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

	if fraudVotes <= 1 || fraudVotes >= 4 {
		return fast
	}
	return idx.ivfTopKWithNprobe(query, idx.ivf.fullNprobe)
}

func (idx *Index) ivfTopKWithNprobe(query [physicalStride]int16, nprobe int) [K]neighbor {
	var top [K]neighbor
	for slot := range top {
		top[slot].squaredDistance = math.MaxInt32
	}
	if idx.ivf == nil {
		return top
	}
	ivf := idx.ivf
	querySlice := query[:]

	scratchPtr := centroidScratchPool.Get().(*[]centroidDistance)
	centroidDistances := *scratchPtr
	if cap(centroidDistances) < ivf.numCentroids {
		centroidDistances = make([]centroidDistance, ivf.numCentroids)
	} else {
		centroidDistances = centroidDistances[:ivf.numCentroids]
	}
	defer func() {
		*scratchPtr = centroidDistances[:0]
		centroidScratchPool.Put(scratchPtr)
	}()

	for c := 0; c < ivf.numCentroids; c++ {
		centroidOffset := c * physicalStride
		distance := l2SquaredDistanceInt16(querySlice, ivf.centroids[centroidOffset:centroidOffset+physicalStride])
		centroidDistances[c] = centroidDistance{int32(c), distance}
	}

	slices.SortFunc(centroidDistances, func(a, b centroidDistance) int {
		return int(a.distance - b.distance)
	})

	probesToScan := nprobe
	if probesToScan > ivf.numCentroids {
		probesToScan = ivf.numCentroids
	}

	kthBest := top[K-1].squaredDistance
	for probe := 0; probe < probesToScan; probe++ {
		cluster := centroidDistances[probe].clusterIdx
		clusterStart := ivf.clusterOffsets[cluster]
		clusterEnd := ivf.clusterOffsets[cluster+1]

		for cursor := clusterStart; cursor < clusterEnd; cursor++ {
			refIdx := ivf.refOrder[cursor]
			refOffset := int(refIdx) * physicalStride

			distance := l2SquaredDistanceInt16(querySlice, idx.vectors[refOffset:refOffset+physicalStride])
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
