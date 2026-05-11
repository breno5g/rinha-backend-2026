package fraud

import (
	"path/filepath"
	"testing"
)

func TestBinaryRoundtrip(t *testing.T) {
	const numRefs = 5_000
	const numCentroids = 32

	original := makeBenchIndex(numRefs, 1)
	original.ivf = buildIVF(original.vectors, numRefs, numCentroids, 5, 2_000, 4)

	tempDir := t.TempDir()
	binaryPath := filepath.Join(tempDir, "index.bin")
	if err := original.SaveBinary(binaryPath); err != nil {
		t.Fatalf("SaveBinary: %v", err)
	}

	loaded, err := LoadBinary(binaryPath, nil)
	if err != nil {
		t.Fatalf("LoadBinary: %v", err)
	}

	if got, want := len(loaded.vectors), len(original.vectors); got != want {
		t.Fatalf("vectors len: got %d, want %d", got, want)
	}
	for i := range original.vectors {
		if loaded.vectors[i] != original.vectors[i] {
			t.Fatalf("vectors[%d]: got %d, want %d", i, loaded.vectors[i], original.vectors[i])
		}
	}
	for i := range original.labels {
		if loaded.labels[i] != original.labels[i] {
			t.Fatalf("labels[%d]: got %d, want %d", i, loaded.labels[i], original.labels[i])
		}
	}
	if loaded.ivf.numCentroids != original.ivf.numCentroids {
		t.Fatalf("numCentroids: got %d, want %d", loaded.ivf.numCentroids, original.ivf.numCentroids)
	}
	if loaded.ivf.nprobe != original.ivf.nprobe {
		t.Fatalf("nprobe: got %d, want %d", loaded.ivf.nprobe, original.ivf.nprobe)
	}
	for i := range original.ivf.centroids {
		if loaded.ivf.centroids[i] != original.ivf.centroids[i] {
			t.Fatalf("centroids[%d]: got %d, want %d", i, loaded.ivf.centroids[i], original.ivf.centroids[i])
		}
	}
	for i := range original.ivf.refOrder {
		if loaded.ivf.refOrder[i] != original.ivf.refOrder[i] {
			t.Fatalf("refOrder[%d]: got %d, want %d", i, loaded.ivf.refOrder[i], original.ivf.refOrder[i])
		}
	}
	for i := range original.ivf.clusterOffsets {
		if loaded.ivf.clusterOffsets[i] != original.ivf.clusterOffsets[i] {
			t.Fatalf("clusterOffsets[%d]: got %d, want %d", i, loaded.ivf.clusterOffsets[i], original.ivf.clusterOffsets[i])
		}
	}

	for q := 0; q < 10; q++ {
		query := makeBenchQuery(int64(q + 50))
		originalTop := original.ivfTopK(query)
		loadedTop := loaded.ivfTopK(query)
		for i := 0; i < K; i++ {
			if originalTop[i].squaredDistance != loadedTop[i].squaredDistance {
				t.Errorf("query %d, top-%d: got %d, want %d",
					q, i, loadedTop[i].squaredDistance, originalTop[i].squaredDistance)
			}
		}
	}
}
