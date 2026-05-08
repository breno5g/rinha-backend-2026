# Rinha de Backend 2026

Submissão para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — um desafio de detecção de fraude em transações de cartão usando busca vetorial.

> 🇬🇧 English version: [README.en.md](./README.en.md)

## O desafio

Construir uma API que recebe transações de cartão, transforma cada payload em um vetor de 14 dimensões, busca no dataset de referência os 5 vetores mais próximos e decide aprovar ou negar com base na proporção de fraudes entre os vizinhos.

## Endpoints

A API responde na porta `9999`:

- `GET /ready` — health check, deve responder `2xx` quando pronta.
- `POST /fraud-score` — recebe a transação e retorna `{ "approved": boolean, "fraud_score": number }`.

## Restrições de infraestrutura

- Pelo menos um load balancer e duas instâncias da API (round-robin).
- Limite total: **1 CPU e 350 MB de memória** somando todos os serviços.
- Entrega via `docker-compose.yml` com imagens públicas `linux-amd64`.
- Rede em modo `bridge` (modo `host` e `privileged` não são permitidos).

## Pontuação

A nota final soma latência (p99) e qualidade de detecção, cada uma variando de -3000 a +3000.

## Estrutura do repositório

- `main` — código-fonte completo.
- `submission` — apenas o necessário para rodar o teste, com `docker-compose.yml` na raiz.

## Documentação oficial

A especificação completa está em [rinha-de-backend-2026/docs](./rinha-de-backend-2026/docs/br/README.md).

## Licença

[MIT](./LICENSE)
