# Rinha de Backend 2026

Submission for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — a card transaction fraud detection challenge using vector search.

> 🇧🇷 Versão em português: [README.md](./README.md)

## The challenge

Build an API that receives card transactions, turns each payload into a 14-dimension vector, searches the reference dataset for the 5 nearest vectors and decides to approve or deny based on the fraud ratio among the neighbors.

## Endpoints

The API listens on port `9999`:

- `GET /ready` — health check, must respond `2xx` when ready.
- `POST /fraud-score` — receives the transaction and returns `{ "approved": boolean, "fraud_score": number }`.

## Infrastructure constraints

- At least one load balancer and two API instances (round-robin).
- Total limit: **1 CPU and 350 MB of memory** across all services.
- Delivery via `docker-compose.yml` with public `linux-amd64` images.
- `bridge` network mode (`host` and `privileged` modes are not allowed).

## Scoring

The final score sums latency (p99) and detection quality, each ranging from -3000 to +3000.

## Repository layout

- `main` — full source code.
- `submission` — only what's needed to run the test, with `docker-compose.yml` at the root.

## Official documentation

Full specification at [rinha-de-backend-2026/docs](./rinha-de-backend-2026/docs/en/README.md).

## License

[MIT](./LICENSE)
