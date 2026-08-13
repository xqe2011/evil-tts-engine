#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="${HOME}/.cargo/bin:${PATH}"
rustup target add wasm32-unknown-unknown >/dev/null
cargo build --release --target wasm32-unknown-unknown
cp cargo-target/wasm32-unknown-unknown/release/tts_engine.wasm engine.wasm
ls -lh engine.wasm
