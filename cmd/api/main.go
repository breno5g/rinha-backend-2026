package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/breno5g/rinha-2026/internal/fraud"
)

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

func main() {
	port := envOrDefault("PORT", "8080")
	instance := envOrDefault("INSTANCE_ID", "api")
	referencesPath := envOrDefault("REFERENCES_PATH", "/resources/references.json.gz")
	normalizationPath := envOrDefault("NORMALIZATION_PATH", "/resources/normalization.json")
	mccRiskPath := envOrDefault("MCC_RISK_PATH", "/resources/mcc_risk.json")

	log.Printf("[%s] loading references from %s ...", instance, referencesPath)
	idx, err := fraud.LoadIndex(referencesPath, normalizationPath, mccRiskPath)
	if err != nil {
		log.Fatalf("[%s] failed to load index: %v", instance, err)
	}
	log.Printf("[%s] index ready (%d references)", instance, idx.Size())

	ready := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Instance", instance)
		w.WriteHeader(http.StatusOK)
	}

	fraudScore := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Instance", instance)
		w.Header().Set("Content-Type", "application/json")

		var p fraud.Payload
		if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
			http.Error(w, "invalid payload", http.StatusBadRequest)
			return
		}
		approved, score, err := idx.Score(&p)
		if err != nil {
			// fail-soft: never return 5xx (Err pesa 5x na fórmula)
			_ = json.NewEncoder(w).Encode(fraudScoreResponse{Approved: true, FraudScore: 0})
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
