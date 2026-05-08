package fraud

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

const (
	VectorDim = 14
	K         = 5
	Threshold = 0.6
)

type Payload struct {
	ID          string `json:"id"`
	Transaction struct {
		Amount       float64 `json:"amount"`
		Installments int     `json:"installments"`
		RequestedAt  string  `json:"requested_at"`
	} `json:"transaction"`
	Customer struct {
		AvgAmount      float64  `json:"avg_amount"`
		TxCount24h     int      `json:"tx_count_24h"`
		KnownMerchants []string `json:"known_merchants"`
	} `json:"customer"`
	Merchant struct {
		ID        string  `json:"id"`
		MCC       string  `json:"mcc"`
		AvgAmount float64 `json:"avg_amount"`
	} `json:"merchant"`
	Terminal struct {
		IsOnline    bool    `json:"is_online"`
		CardPresent bool    `json:"card_present"`
		KmFromHome  float64 `json:"km_from_home"`
	} `json:"terminal"`
	LastTransaction *struct {
		Timestamp     string  `json:"timestamp"`
		KmFromCurrent float64 `json:"km_from_current"`
	} `json:"last_transaction"`
}

type Constants struct {
	MaxAmount            float64 `json:"max_amount"`
	MaxInstallments      float64 `json:"max_installments"`
	AmountVsAvgRatio     float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float64 `json:"max_minutes"`
	MaxKm                float64 `json:"max_km"`
	MaxTxCount24h        float64 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float64 `json:"max_merchant_avg_amount"`

	MccRisk map[string]float64
}

func LoadConstants(normalizationPath, mccRiskPath string) (*Constants, error) {
	c := &Constants{}
	data, err := os.ReadFile(normalizationPath)
	if err != nil {
		return nil, fmt.Errorf("read normalization: %w", err)
	}
	if err := json.Unmarshal(data, c); err != nil {
		return nil, fmt.Errorf("parse normalization: %w", err)
	}

	data, err = os.ReadFile(mccRiskPath)
	if err != nil {
		return nil, fmt.Errorf("read mcc_risk: %w", err)
	}
	if err := json.Unmarshal(data, &c.MccRisk); err != nil {
		return nil, fmt.Errorf("parse mcc_risk: %w", err)
	}
	return c, nil
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func weekdayMonSun(t time.Time) int {
	return (int(t.Weekday()) + 6) % 7
}

func Vectorize(p *Payload, c *Constants) ([VectorDim]float32, error) {
	var v [VectorDim]float32

	requestedAt, err := time.Parse(time.RFC3339, p.Transaction.RequestedAt)
	if err != nil {
		return v, fmt.Errorf("parse requested_at: %w", err)
	}
	requestedAt = requestedAt.UTC()

	v[0] = float32(clamp01(p.Transaction.Amount / c.MaxAmount))
	v[1] = float32(clamp01(float64(p.Transaction.Installments) / c.MaxInstallments))

	var amountVsAvg float64
	if p.Customer.AvgAmount > 0 {
		amountVsAvg = (p.Transaction.Amount / p.Customer.AvgAmount) / c.AmountVsAvgRatio
	}
	v[2] = float32(clamp01(amountVsAvg))

	v[3] = float32(float64(requestedAt.Hour()) / 23.0)
	v[4] = float32(float64(weekdayMonSun(requestedAt)) / 6.0)

	if p.LastTransaction == nil {
		v[5] = -1
		v[6] = -1
	} else {
		lastAt, err := time.Parse(time.RFC3339, p.LastTransaction.Timestamp)
		if err != nil {
			return v, fmt.Errorf("parse last_transaction.timestamp: %w", err)
		}
		minutes := requestedAt.Sub(lastAt.UTC()).Minutes()
		v[5] = float32(clamp01(minutes / c.MaxMinutes))
		v[6] = float32(clamp01(p.LastTransaction.KmFromCurrent / c.MaxKm))
	}

	v[7] = float32(clamp01(p.Terminal.KmFromHome / c.MaxKm))
	v[8] = float32(clamp01(float64(p.Customer.TxCount24h) / c.MaxTxCount24h))

	if p.Terminal.IsOnline {
		v[9] = 1
	}
	if p.Terminal.CardPresent {
		v[10] = 1
	}

	known := false
	for _, m := range p.Customer.KnownMerchants {
		if m == p.Merchant.ID {
			known = true
			break
		}
	}
	if !known {
		v[11] = 1
	}

	if risk, ok := c.MccRisk[p.Merchant.MCC]; ok {
		v[12] = float32(risk)
	} else {
		v[12] = 0.5
	}

	v[13] = float32(clamp01(p.Merchant.AvgAmount / c.MaxMerchantAvgAmount))

	return v, nil
}