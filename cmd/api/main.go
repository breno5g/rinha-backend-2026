package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/breno5g/rinha-2026/internal/fraud"
)

// precomputedResponses holds the 6 possible HTTP response bodies (one per
// fraud-vote count 0..5). Each is the JSON {"approved":bool,"fraud_score":float}
// already serialized into a byte slice. The hot path picks slot[count] and
// writes it directly — zero allocation, zero serialization.
var precomputedResponses [fraud.K + 1][]byte

// failSoftResponse is returned whenever the request would otherwise produce
// an HTTP 5xx. The Rinha scoring formula weights HTTP errors 5× heavier than
// detection errors (E = 1·FP + 3·FN + 5·Err) AND counts them toward the 15%
// failure-rate cut, so it's strictly cheaper to misclassify than to crash.
var failSoftResponse []byte

// payloadPool reuses fraud.Payload structs across requests. Decode resets all
// scalar fields, but slices and the LastTransaction pointer must be cleared
// explicitly before returning to the pool so stale data can't leak between
// requests.
var payloadPool = sync.Pool{
	New: func() any { return new(fraud.Payload) },
}

func resetPayload(p *fraud.Payload) {
	*p = fraud.Payload{}
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
	// Fail-soft is "approved=true, fraud_score=0" — same as count=0.
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

// listen binds the HTTP server: Unix domain socket if SOCKET_PATH is set
// (matches the .NET reference layout — nginx upstream uses unix:/run/sock/api1.sock),
// otherwise TCP on PORT. Unix sockets remove the loopback TCP overhead,
// which dominates p99 once the search itself drops below ~400µs.
func listen(instance string) (net.Listener, string, error) {
	if socket := os.Getenv("SOCKET_PATH"); socket != "" {
		// Remove any stale socket from a previous crash so Listen succeeds.
		_ = os.Remove(socket)
		ln, err := net.Listen("unix", socket)
		if err != nil {
			return nil, "", fmt.Errorf("listen unix %s: %w", socket, err)
		}
		// nginx (running in a sibling container) needs to reach this socket
		// through the shared /run/sock volume — it must be world-readable.
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

	ready := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Instance", instance)
		w.WriteHeader(http.StatusOK)
	}

	fraudScore := func(w http.ResponseWriter, r *http.Request) {
		// fail-soft: any panic in vectorize/score becomes a fast 200 OK with
		// the precomputed "approved" body. Costs at most 1 FP / 3 FN; way
		// cheaper than a 5xx (5 in the error formula + 15% cut).
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
