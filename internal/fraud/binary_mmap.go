//go:build linux

package fraud

import (
	"encoding/binary"
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

// LoadBinaryMmap maps the index file into the process address space and
// builds an Index whose slices point directly into the mapped region.
// No copies are made for the bulk arrays.
//
// Two API instances on the same host both calling LoadBinaryMmap on the
// same file share the same kernel page cache: the physical memory holding
// the index is allocated once, regardless of how many readers there are.
// That's the entire point of using mmap here.
//
// The returned Index borrows from the mmap region; tearing it down is
// out of scope (process lifetime).
func LoadBinaryMmap(path string, constants *Constants) (*Index, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	// We can close the fd after mmap; the kernel keeps the mapping alive.
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

	// The header is little-endian fields packed as written by encoding/binary.
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

	vectors := unsafe.Slice((*int8)(unsafe.Pointer(&data[offset])), numRefs*VectorDim)
	offset += numRefs * VectorDim

	labels := data[offset : offset+numRefs]
	offset += numRefs

	centroids := unsafe.Slice((*int8)(unsafe.Pointer(&data[offset])), numCentroids*VectorDim)
	offset += numCentroids * VectorDim

	if offset != size {
		return nil, fmt.Errorf("file size mismatch: parsed %d, file %d", offset, size)
	}

	return assembleIndex(constants, header, vectors, labels, centroids, refOrder, clusterOffsets), nil
}
