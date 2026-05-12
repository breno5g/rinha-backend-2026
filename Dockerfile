# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 golang:1.25-alpine AS build
WORKDIR /src

# Copy module + sources first so layers can be cached.
COPY go.mod go.sum ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal

# Build both binaries.
# GOAMD64=v3 enables AVX2/BMI2 in code Go itself emits (the asm kernel always
# uses AVX2; this flag covers everything else).
# -pgo=auto picks up cmd/api/default.pgo for profile-guided optimization.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOAMD64=v3 \
    go build -pgo=auto -trimpath -ldflags="-s -w" -o /out/api ./cmd/api && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOAMD64=v3 \
    go build -trimpath -ldflags="-s -w" -o /out/preprocess ./cmd/preprocess

# ---- Preprocess stage: turn the raw references into a ready-to-mmap binary.
# Doing this in the image build means the runtime container can serve traffic
# in <1s instead of paying the ~35s of gzip + JSON parse + k-means + assignment.
FROM --platform=linux/amd64 golang:1.25-alpine AS preprocess
WORKDIR /work
COPY resources /work/resources
COPY --from=build /out/preprocess /usr/local/bin/preprocess
RUN mkdir -p /work/build && \
    /usr/local/bin/preprocess --in /work/resources --out /work/build/index.bin

# ---- Runtime image: only the API binary, the pre-built index, and the
# small constants files. References.json.gz is intentionally NOT copied.
FROM --platform=linux/amd64 gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/api /api
COPY --from=preprocess /work/build/index.bin /resources/index.bin
COPY resources/normalization.json /resources/normalization.json
COPY resources/mcc_risk.json /resources/mcc_risk.json
ENV INDEX_BINARY=/resources/index.bin
ENV INDEX_MMAP=true
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/api"]
