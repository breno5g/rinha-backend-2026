package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
)

type fraudScoreResponse struct {
	Approved   bool    `json:"approved"`
	FraudScore float64 `json:"fraud_score"`
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	instance := os.Getenv("INSTANCE_ID")

	ready := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Instance", instance)
		w.WriteHeader(http.StatusOK)
	}

	fraudScore := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Instance", instance)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(fraudScoreResponse{
			Approved:   true,
			FraudScore: 0,
		})
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", ready)
	mux.HandleFunc("POST /fraud-score", fraudScore)

	log.Printf("[%s] listening on :%s", instance, port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatal(err)
	}
}
