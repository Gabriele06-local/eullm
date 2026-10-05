# EuLLM Hub — multi-stage Rust build.
#
# Built from the repository root, like the engine, so that it compiles with
# the workspace's Cargo.lock — the versions CI tested:
#
#   docker build -f hub/Dockerfile -t eullm/hub .
#   docker compose build hub

# ── Stage 1: Build ──────────────────────────────────────────────
# rust:1, not a pinned minor: Cargo.toml declares edition 2024, which needs
# cargo 1.85 or newer, and 1.83 could not parse the manifest at all. Pinning
# the minor is what let that rot unnoticed.
FROM rust:1-bookworm AS builder

WORKDIR /build
# The workspace's manifest and lockfile, and its members: cargo reads every
# member's manifest, the engine's included, to build the hub alone.
COPY Cargo.toml Cargo.lock ./
COPY engine/ engine/
COPY hub/ hub/

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --locked -p eullm-hub \
    && cp target/release/eullm-hub /usr/local/bin/eullm-hub

# ── Stage 2: Runtime ────────────────────────────────────────────
FROM debian:bookworm-slim

# curl: the health check. The hub makes no outgoing connections and links
# no TLS library.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# /models is created here, owned by the user the hub runs as: a volume Docker
# creates itself is root's, and the hub could not store a model in it. The
# user is 10001, as in the engine's image, whose `models` volume it shares.
RUN groupadd -r -g 10001 eullm \
    && useradd -r -u 10001 -g eullm -s /usr/sbin/nologin eullm \
    && mkdir -p /models \
    && chown eullm:eullm /models
USER eullm

COPY --from=builder /usr/local/bin/eullm-hub /usr/local/bin/eullm-hub

VOLUME ["/models"]
ENV EULLM_HUB_STORAGE=/models

EXPOSE 3000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:3000/health || exit 1

ENTRYPOINT ["eullm-hub"]
