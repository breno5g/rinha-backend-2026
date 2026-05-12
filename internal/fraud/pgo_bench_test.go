package fraud

import (
	"testing"
)

func makeBenchIVFIndex(numReferences, numCentroids int, seed int64) *Index {
	idx := makeBenchIndex(numReferences, seed)
	idx.Constants = testConstants()
	idx.ivf = buildIVF(idx.vectors, len(idx.labels),
		numCentroids, 6, 5000, defaultNprobe)
	return idx
}

func benchPayload() *Payload {
	p := &Payload{}
	p.ID = "tx-pgo-bench"
	p.Transaction.Amount = 384.88
	p.Transaction.Installments = 3
	p.Transaction.RequestedAt = "2026-03-11T20:23:35Z"
	p.Customer.AvgAmount = 769.76
	p.Customer.TxCount24h = 3
	p.Customer.KnownMerchants = []string{"MERC-009", "MERC-001", "MERC-002"}
	p.Merchant.ID = "MERC-001"
	p.Merchant.MCC = "5912"
	p.Merchant.AvgAmount = 298.95
	p.Terminal.IsOnline = false
	p.Terminal.CardPresent = true
	p.Terminal.KmFromHome = 13.7
	p.LastTransaction = &struct {
		Timestamp     string  `json:"timestamp"`
		KmFromCurrent float64 `json:"km_from_current"`
	}{Timestamp: "2026-03-11T14:58:35Z", KmFromCurrent: 18.86}
	return p
}

func BenchmarkFraudHotPath(b *testing.B) {
	idx := makeBenchIVFIndex(50_000, 128, 42)
	payload := benchPayload()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = idx.FraudCount(payload)
	}
}
