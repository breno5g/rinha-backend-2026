package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"runtime/debug"
	"strconv"
	"sync"
	"time"

	"github.com/breno5g/rinha-2026/internal/fraud"
)

var precomputedResponses [fraud.K + 1][]byte

var failSoftResponse []byte

var payloadPool = sync.Pool{
	New: func() any { return new(fraud.Payload) },
}

func resetPayload(p *fraud.Payload) {
	knownMerchants := p.Customer.KnownMerchants[:0]
	*p = fraud.Payload{}
	p.Customer.KnownMerchants = knownMerchants
}

func buildResponses() {
	type body struct {
		Approved   bool    `json:"approved"`
		FraudScore float32 `json:"fraud_score"`
	}
	for count := 0; count <= fraud.K; count++ {
		score := float32(count) / float32(fraud.K)
		buf, err := json.Marshal(body{Approved: score < fraud.Threshold, FraudScore: score})
		if err != nil {
			log.Fatalf("precompute response[%d]: %v", count, err)
		}
		precomputedResponses[count] = buf
	}

	failSoftResponse = precomputedResponses[0]
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

func listen(instance string) (net.Listener, string, error) {
	if socket := os.Getenv("SOCKET_PATH"); socket != "" {

		_ = os.Remove(socket)
		ln, err := net.Listen("unix", socket)
		if err != nil {
			return nil, "", fmt.Errorf("listen unix %s: %w", socket, err)
		}

		if err := os.Chmod(socket, 0o666); err != nil {
			ln.Close()
			return nil, "", fmt.Errorf("chmod %s: %w", socket, err)
		}
		log.Printf("[%s] listening on unix:%s", instance, socket)
		return ln, "unix:" + socket, nil
	}
	port := envOrDefault("PORT", "8080")
	ln, err := net.Listen("tcp", ":"+port)
	if err != nil {
		return nil, "", fmt.Errorf("listen tcp :%s: %w", port, err)
	}
	log.Printf("[%s] listening on :%s", instance, port)
	return ln, ":" + port, nil
}

func tuneRuntime(instance string) {
	debug.FreeOSMemory()
	switch os.Getenv("GC_MODE") {
	case "off":
		debug.SetGCPercent(-1)
		log.Printf("[%s] GC disabled after init", instance)
	case "high":
		debug.SetGCPercent(1000)
		log.Printf("[%s] GC set to high threshold (1000%%)", instance)
	}
}

func main() {
	instance := envOrDefault("INSTANCE_ID", "api")
	nprobeOverride := envOrDefault("IVF_NPROBE", "")
	fullNprobeOverride := envOrDefault("IVF_FULL_NPROBE", "")

	buildResponses()

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
	if fullNprobeOverride != "" {
		if err := idx.SetIVFFullNprobe(fullNprobeOverride); err != nil {
			log.Fatalf("[%s] invalid IVF_FULL_NPROBE: %v", instance, err)
		}
		log.Printf("[%s] IVF fullNprobe (two-stage) override: %s", instance, fullNprobeOverride)
	}

	tuneRuntime(instance)

	ready := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Instance", instance)
		w.WriteHeader(http.StatusOK)
	}

	fraudScore := func(w http.ResponseWriter, r *http.Request) {

		defer func() {
			if recovered := recover(); recovered != nil {
				log.Printf("[%s] handler panic recovered: %v", instance, recovered)
				_, _ = w.Write(failSoftResponse)
			}
		}()

		w.Header().Set("Content-Type", "application/json")

		payload := payloadPool.Get().(*fraud.Payload)
		defer func() {
			resetPayload(payload)
			payloadPool.Put(payload)
		}()

		if err := json.NewDecoder(r.Body).Decode(payload); err != nil {
			_, _ = w.Write(failSoftResponse)
			return
		}
		count, err := idx.FraudCount(payload)
		if err != nil {
			_, _ = w.Write(failSoftResponse)
			return
		}
		_, _ = w.Write(precomputedResponses[count])
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", ready)
	mux.HandleFunc("POST /fraud-score", fraudScore)

	listener, addr, err := listen(instance)
	if err != nil {
		log.Fatalf("[%s] %v", instance, err)
	}
	server := &http.Server{Handler: mux}
	log.Printf("[%s] serving on %s", instance, addr)
	if err := server.Serve(listener); err != nil {
		log.Fatal(err)
	}
}
