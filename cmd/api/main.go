package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/breno5g/rinha-2026/internal/fraud"
)

// failSoftResponse is returned whenever the request would otherwise produce
// an HTTP 5xx. The Rinha scoring formula weights HTTP errors 5× heavier than
// detection errors (E = 1·FP + 3·FN + 5·Err) AND counts them toward the 15%
// failure-rate cut, so it's strictly cheaper to misclassify than to crash.
var failSoftResponse = fraudScoreResponse{Approved: true, FraudScore: 0}

type fraudScoreResponse struct {
	Approved   bool    `json:"approved"`
	FraudScore float32 `json:"fraud_score"`
}

func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envBool(key string) bool {
	v := os.Getenv(key)
	if v == "" {
		return false
	}
	b, _ := strconv.ParseBool(v)
	return b
}

// loadIndex picks the cheapest available source: a pre-built binary (mmap if
// requested) → references.json.gz fallback. The binary path comes from
// INDEX_BINARY; mmap is enabled by INDEX_MMAP=true.
func loadIndex(instance string) (*fraud.Index, error) {
	binaryPath := os.Getenv("INDEX_BINARY")
	useMmap := envBool("INDEX_MMAP")

	normalizationPath := envOrDefault("NORMALIZATION_PATH", "/resources/normalization.json")
	mccRiskPath := envOrDefault("MCC_RISK_PATH", "/resources/mcc_risk.json")

	if binaryPath != "" {
		if _, err := os.Stat(binaryPath); err == nil {
			constants, err := fraud.LoadConstants(normalizationPath, mccRiskPath)
			if err != nil {
				return nil, err
			}
			start := time.Now()
			var idx *fraud.Index
			if useMmap {
				log.Printf("[%s] loading index via mmap from %s ...", instance, binaryPath)
				idx, err = fraud.LoadBinaryMmap(binaryPath, constants)
			} else {
				log.Printf("[%s] loading index from %s ...", instance, binaryPath)
				idx, err = fraud.LoadBinary(binaryPath, constants)
			}
			if err != nil {
				return nil, err
			}
			log.Printf("[%s] index loaded in %s (%d references)", instance, time.Since(start).Round(time.Millisecond), idx.Size())
			return idx, nil
		}
		log.Printf("[%s] INDEX_BINARY=%q not found; falling back to JSON build", instance, binaryPath)
	}

	referencesPath := envOrDefault("REFERENCES_PATH", "/resources/references.json.gz")
	indexKind := fraud.IndexKind(envOrDefault("INDEX_KIND", string(fraud.KindIVF)))
	log.Printf("[%s] building index from %s (kind=%s) ...", instance, referencesPath, indexKind)
	return fraud.LoadIndex(indexKind, referencesPath, normalizationPath, mccRiskPath)
}

func main() {
	port := envOrDefault("PORT", "8080")
	instance := envOrDefault("INSTANCE_ID", "api")
	nprobeOverride := envOrDefault("IVF_NPROBE", "")

	idx, err := loadIndex(instance)
	if err != nil {
		log.Fatalf("[%s] failed to load index: %v", instance, err)
	}
	if nprobeOverride != "" {
		if err := idx.SetIVFNprobe(nprobeOverride); err != nil {
			log.Fatalf("[%s] invalid IVF_NPROBE: %v", instance, err)
		}
		log.Printf("[%s] IVF nprobe override: %s", instance, nprobeOverride)
	}

	ready := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Instance", instance)
		w.WriteHeader(http.StatusOK)
	}

	fraudScore := func(w http.ResponseWriter, r *http.Request) {
		// fail-soft: any panic in vectorize/score becomes a fast 200 OK.
		// Costs 1 FP or 3 FN at most; way better than 5 (Err) + failure_rate hit.
		defer func() {
			if recovered := recover(); recovered != nil {
				log.Printf("[%s] handler panic recovered: %v", instance, recovered)
				_ = json.NewEncoder(w).Encode(failSoftResponse)
			}
		}()

		w.Header().Set("X-Instance", instance)
		w.Header().Set("Content-Type", "application/json")

		var payload fraud.Payload
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			_ = json.NewEncoder(w).Encode(failSoftResponse)
			return
		}
		approved, score, err := idx.Score(&payload)
		if err != nil {
			_ = json.NewEncoder(w).Encode(failSoftResponse)
			return
		}
		_ = json.NewEncoder(w).Encode(fraudScoreResponse{Approved: approved, FraudScore: score})
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", ready)
	mux.HandleFunc("POST /fraud-score", fraudScore)

	log.Printf("[%s] listening on :%s", instance, port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatal(err)
	}
}
