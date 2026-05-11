//go:build linux

package fraud

import (
	"encoding/binary"
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

func LoadBinaryMmap(path string, constants *Constants) (*Index, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	defer file.Close()

	stat, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}
	size := int(stat.Size())
	if size < headerSize {
		return nil, fmt.Errorf("file too small: %d bytes", size)
	}

	data, err := unix.Mmap(int(file.Fd()), 0, size, unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}

	header := &binaryHeader{
		Magic:         binary.LittleEndian.Uint32(data[0:4]),
		Version:       binary.LittleEndian.Uint32(data[4:8]),
		NumRefs:       binary.LittleEndian.Uint32(data[8:12]),
		NumCentroids:  binary.LittleEndian.Uint32(data[12:16]),
		DefaultNprobe: binary.LittleEndian.Uint32(data[16:20]),
	}
	if header.Magic != binaryMagic {
		return nil, fmt.Errorf("bad magic 0x%X", header.Magic)
	}
	if header.Version != binaryVersion {
		return nil, fmt.Errorf("unsupported binary version %d", header.Version)
	}

	numRefs := int(header.NumRefs)
	numCentroids := int(header.NumCentroids)

	offset := headerSize

	refOrder := unsafe.Slice((*int32)(unsafe.Pointer(&data[offset])), numRefs)
	offset += numRefs * 4

	clusterOffsets := unsafe.Slice((*int32)(unsafe.Pointer(&data[offset])), numCentroids+1)
	offset += (numCentroids + 1) * 4

	vectors := unsafe.Slice((*int16)(unsafe.Pointer(&data[offset])), numRefs*physicalStride)
	offset += numRefs * physicalStride * 2

	labels := data[offset : offset+numRefs]
	offset += numRefs

	centroids := unsafe.Slice((*int16)(unsafe.Pointer(&data[offset])), numCentroids*physicalStride)
	offset += numCentroids * physicalStride * 2

	if offset != size {
		return nil, fmt.Errorf("file size mismatch: parsed %d, file %d", offset, size)
	}

	return assembleIndex(constants, header, vectors, labels, centroids, refOrder, clusterOffsets), nil
}
