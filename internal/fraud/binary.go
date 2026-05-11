package fraud

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"unsafe"
)

const (
	binaryMagic uint32 = 0x52494E48

	binaryVersion uint32 = 3
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

	if err := binary.Write(file, binary.LittleEndian, idx.ivf.refOrder); err != nil {
		return fmt.Errorf("write refOrder: %w", err)
	}
	if err := binary.Write(file, binary.LittleEndian, idx.ivf.clusterOffsets); err != nil {
		return fmt.Errorf("write clusterOffsets: %w", err)
	}
	if _, err := file.Write(int16AsBytes(idx.vectors)); err != nil {
		return fmt.Errorf("write vectors: %w", err)
	}
	if _, err := file.Write(idx.labels); err != nil {
		return fmt.Errorf("write labels: %w", err)
	}
	if _, err := file.Write(int16AsBytes(idx.ivf.centroids)); err != nil {
		return fmt.Errorf("write centroids: %w", err)
	}
	return nil
}

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
	vectors := make([]int16, numRefs*physicalStride)
	if _, err := io.ReadFull(file, int16AsBytes(vectors)); err != nil {
		return nil, fmt.Errorf("read vectors: %w", err)
	}
	labels := make([]uint8, numRefs)
	if _, err := io.ReadFull(file, labels); err != nil {
		return nil, fmt.Errorf("read labels: %w", err)
	}
	centroids := make([]int16, numCentroids*physicalStride)
	if _, err := io.ReadFull(file, int16AsBytes(centroids)); err != nil {
		return nil, fmt.Errorf("read centroids: %w", err)
	}

	return assembleIndex(constants, header, vectors, labels, centroids, refOrder, clusterOffsets), nil
}

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
	vectors []int16,
	labels []uint8,
	centroids []int16,
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

func int16AsBytes(s []int16) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*2)
}
