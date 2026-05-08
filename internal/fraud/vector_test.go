package fraud

import (
	"math"
	"testing"
)

func testConstants() *Constants {
	return &Constants{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
		MccRisk: map[string]float64{
			"5411": 0.15,
			"5812": 0.30,
			"5912": 0.20,
			"5944": 0.45,
			"7801": 0.80,
			"7802": 0.75,
			"7995": 0.85,
			"4511": 0.35,
			"5311": 0.25,
			"5999": 0.50,
		},
	}
}

func vectorsClose(t *testing.T, got [VectorDim]float32, want [VectorDim]float32, tol float32) {
	t.Helper()
	for i := 0; i < VectorDim; i++ {
		if math.Abs(float64(got[i]-want[i])) > float64(tol) {
			t.Errorf("dim %d: got %.6f, want %.6f (tol %g)", i, got[i], want[i], tol)
		}
	}
}

func TestVectorize_LegitExample(t *testing.T) {
	p := &Payload{}
	p.ID = "tx-1329056812"
	p.Transaction.Amount = 41.12
	p.Transaction.Installments = 2
	p.Transaction.RequestedAt = "2026-03-11T18:45:53Z"
	p.Customer.AvgAmount = 82.24
	p.Customer.TxCount24h = 3
	p.Customer.KnownMerchants = []string{"MERC-003", "MERC-016"}
	p.Merchant.ID = "MERC-016"
	p.Merchant.MCC = "5411"
	p.Merchant.AvgAmount = 60.25
	p.Terminal.IsOnline = false
	p.Terminal.CardPresent = true
	p.Terminal.KmFromHome = 29.23
	p.LastTransaction = nil

	want := [VectorDim]float32{0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006}
	got, err := Vectorize(p, testConstants())
	if err != nil {
		t.Fatalf("Vectorize: %v", err)
	}
	vectorsClose(t, got, want, 1e-3)
}

func TestVectorize_FraudExample(t *testing.T) {
	p := &Payload{}
	p.ID = "tx-3330991687"
	p.Transaction.Amount = 9505.97
	p.Transaction.Installments = 10
	p.Transaction.RequestedAt = "2026-03-14T05:15:12Z"
	p.Customer.AvgAmount = 81.28
	p.Customer.TxCount24h = 20
	p.Customer.KnownMerchants = []string{"MERC-008", "MERC-007", "MERC-005"}
	p.Merchant.ID = "MERC-068"
	p.Merchant.MCC = "7802"
	p.Merchant.AvgAmount = 54.86
	p.Terminal.IsOnline = false
	p.Terminal.CardPresent = true
	p.Terminal.KmFromHome = 952.27
	p.LastTransaction = nil

	want := [VectorDim]float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}
	got, err := Vectorize(p, testConstants())
	if err != nil {
		t.Fatalf("Vectorize: %v", err)
	}
	vectorsClose(t, got, want, 1e-3)
}

func TestVectorize_UnknownMCC(t *testing.T) {
	p := &Payload{}
	p.Transaction.RequestedAt = "2026-03-11T18:45:53Z"
	p.Merchant.MCC = "9999"
	p.Customer.AvgAmount = 100
	got, err := Vectorize(p, testConstants())
	if err != nil {
		t.Fatalf("Vectorize: %v", err)
	}
	if got[12] != 0.5 {
		t.Errorf("unknown MCC should map to 0.5, got %v", got[12])
	}
}

func TestClamp01(t *testing.T) {
	cases := []struct {
		in, want float64
	}{
		{-0.5, 0},
		{0, 0},
		{0.5, 0.5},
		{1, 1},
		{1.25, 1},
	}
	for _, c := range cases {
		if got := clamp01(c.in); got != c.want {
			t.Errorf("clamp01(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}