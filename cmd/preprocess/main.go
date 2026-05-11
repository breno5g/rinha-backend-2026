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
