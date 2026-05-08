// preprocess builds the IVF index from the raw references.json.gz at build
// time and serializes it into a single binary file (~57 MB). The runtime
// container then loads that binary in <1s instead of paying the ~35s of
// gzip + JSON parse + k-means + assignment we'd otherwise face on every cold start.
//
// Usage:
//   preprocess --in /resources --out /build/index.bin
//
// All paths are required; the tool is meant to run in a Dockerfile build stage,
// not interactively.
package main

import (
	"flag"
	"log"
	"path/filepath"
	"time"

	"github.com/breno5g/rinha-2026/internal/fraud"
)

func main() {
	resourcesDir := flag.String("in", "", "directory containing references.json.gz, normalization.json, mcc_risk.json")
	outputPath := flag.String("out", "", "output path for the .bin file")
	flag.Parse()

	if *resourcesDir == "" || *outputPath == "" {
		log.Fatal("--in and --out are required")
	}

	start := time.Now()
	idx, err := fraud.LoadIndex(
		fraud.KindIVF,
		filepath.Join(*resourcesDir, "references.json.gz"),
		filepath.Join(*resourcesDir, "normalization.json"),
		filepath.Join(*resourcesDir, "mcc_risk.json"),
	)
	if err != nil {
		log.Fatalf("LoadIndex: %v", err)
	}
	log.Printf("index built in %s", time.Since(start).Round(time.Millisecond))

	saveStart := time.Now()
	if err := idx.SaveBinary(*outputPath); err != nil {
		log.Fatalf("SaveBinary: %v", err)
	}
	log.Printf("binary saved to %s in %s", *outputPath, time.Since(saveStart).Round(time.Millisecond))
}
