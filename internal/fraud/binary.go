package fraud

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"unsafe"
)

// Custom binary format for a pre-built IVF index. Designed to be fast to load
// (a few large reads, no parsing) and to support memory-mapped access so two
// API instances on the same host share the same physical pages via the kernel
// page cache.
//
// Field order is chosen so that int32 fields land on 4-byte aligned offsets,
// which lets LoadBinaryMmap reinterpret bytes as []int32 with unsafe.Slice
// without copying. Header is 32 bytes (already 4-aligned), and all int32
// fields immediately follow it.
//
// Layout (little-endian):
//
//   [binaryHeader]                                   32 bytes
//   [int32 refOrder       : NumRefs]                 4 * NumRefs
//   [int32 clusterOffsets : NumCentroids + 1]        4 * (NumCentroids + 1)
//   [int8  vectors        : NumRefs * VectorDim]     14 * NumRefs
//   [uint8 labels         : NumRefs]                 NumRefs
//   [int8  centroids      : NumCentroids * VectorDim] 14 * NumCentroids
//
// For 3M refs and 256 centroids, the file is ~57 MB.

const (
	binaryMagic   uint32 = 0x52494E48 // "RINH"
	binaryVersion uint32 = 1
	headerSize           = 32
)

type binaryHeader struct {
	Magic         uint32
	Version       uint32
	NumRefs       uint32
	NumCentroids  uint32
	DefaultNprobe uint32
	Reserved      [3]uint32
}

// SaveBinary serializes the IVF index (and its underlying vectors/labels)
// to a single file. Used by the preprocess CLI at build time.
func (idx *Index) SaveBinary(path string) error {
	if idx.ivf == nil {
		return fmt.Errorf("only IVF indexes can be saved")
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	header := binaryHeader{
		Magic:         binaryMagic,
		Version:       binaryVersion,
		NumRefs:       uint32(len(idx.labels)),
		NumCentroids:  uint32(idx.ivf.numCentroids),
		DefaultNprobe: uint32(idx.ivf.nprobe),
	}
	if err := binary.Write(file, binary.LittleEndian, header); err != nil {
		return fmt.Errorf("write header: %w", err)
	}
	// int32 fields first — keeps them 4-aligned for zero-copy mmap views.
	if err := binary.Write(file, binary.LittleEndian, idx.ivf.refOrder); err != nil {
		return fmt.Errorf("write refOrder: %w", err)
	}
	if err := binary.Write(file, binary.LittleEndian, idx.ivf.clusterOffsets); err != nil {
		return fmt.Errorf("write clusterOffsets: %w", err)
	}
	if _, err := file.Write(int8AsBytes(idx.vectors)); err != nil {
		return fmt.Errorf("write vectors: %w", err)
	}
	if _, err := file.Write(idx.labels); err != nil {
		return fmt.Errorf("write labels: %w", err)
	}
	if _, err := file.Write(int8AsBytes(idx.ivf.centroids)); err != nil {
		return fmt.Errorf("write centroids: %w", err)
	}
	return nil
}

// LoadBinary reads a previously-saved IVF index file into Go-allocated slices
// (heap, copies). Use this when you don't need or can't use mmap.
func LoadBinary(path string, constants *Constants) (*Index, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	header, err := readHeader(file)
	if err != nil {
		return nil, err
	}

	numRefs := int(header.NumRefs)
	numCentroids := int(header.NumCentroids)

	refOrder := make([]int32, numRefs)
	if err := binary.Read(file, binary.LittleEndian, refOrder); err != nil {
		return nil, fmt.Errorf("read refOrder: %w", err)
	}
	clusterOffsets := make([]int32, numCentroids+1)
	if err := binary.Read(file, binary.LittleEndian, clusterOffsets); err != nil {
		return nil, fmt.Errorf("read clusterOffsets: %w", err)
	}
	vectors := make([]int8, numRefs*VectorDim)
	if _, err := io.ReadFull(file, int8AsBytes(vectors)); err != nil {
		return nil, fmt.Errorf("read vectors: %w", err)
	}
	labels := make([]uint8, numRefs)
	if _, err := io.ReadFull(file, labels); err != nil {
		return nil, fmt.Errorf("read labels: %w", err)
	}
	centroids := make([]int8, numCentroids*VectorDim)
	if _, err := io.ReadFull(file, int8AsBytes(centroids)); err != nil {
		return nil, fmt.Errorf("read centroids: %w", err)
	}

	return assembleIndex(constants, header, vectors, labels, centroids, refOrder, clusterOffsets), nil
}

// readHeader parses and validates the 32-byte header from any io.Reader.
func readHeader(r io.Reader) (*binaryHeader, error) {
	var header binaryHeader
	if err := binary.Read(r, binary.LittleEndian, &header); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	if header.Magic != binaryMagic {
		return nil, fmt.Errorf("bad magic 0x%X (not a Rinha IVF binary?)", header.Magic)
	}
	if header.Version != binaryVersion {
		return nil, fmt.Errorf("unsupported binary version %d (expected %d)", header.Version, binaryVersion)
	}
	return &header, nil
}

func assembleIndex(
	constants *Constants,
	header *binaryHeader,
	vectors []int8,
	labels []uint8,
	centroids []int8,
	refOrder []int32,
	clusterOffsets []int32,
) *Index {
	return &Index{
		Constants: constants,
		vectors:   vectors,
		labels:    labels,
		ivf: &ivfIndex{
			centroids:      centroids,
			numCentroids:   int(header.NumCentroids),
			nprobe:         int(header.DefaultNprobe),
			refOrder:       refOrder,
			clusterOffsets: clusterOffsets,
		},
	}
}

// int8AsBytes reinterprets a slice of int8 as a slice of byte without copying.
// Safe because int8 and byte have identical size and alignment.
func int8AsBytes(s []int8) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s))
}
