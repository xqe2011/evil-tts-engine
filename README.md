# evil-tts-engine

Standalone Bert-VITS2 V220-style TTS frontend (Rust → WASM) + Bun/ORT inference for the **evil** voice.

## Layout

```
engine/          Rust WASM frontend (normalize, EN G2P, SPM, bert pack)
models/          ONNX + config (LFS)
  evil_v220.onnx
  deberta_v3_large_hs.int8.onnx   # preferred; FP32 Deberta not shipped (~1.5GB)
  config.json
infer.ts         Bun CLI: engine.wasm + onnxruntime-web/wasm
examples/        sample WAVs
```

## Requirements

- [Bun](https://bun.sh)
- [Rust](https://rustup.rs) + `wasm32-unknown-unknown` (only to rebuild engine)
- Mac CPU / ORT WASM (no CUDA)

## Run (EN)

```bash
bun install
bun infer.ts --text "Hello, I am evil, an assistant by apple banana." --out examples/en.wav
```

## Rebuild frontend WASM

```bash
cd engine && ./build.sh
```

## Language support

| Lang | G2P | BERT features | Status |
|------|-----|---------------|--------|
| EN   | CMUdict + SPM (Deberta-v3) | int8 Deberta ONNX | working |
| ZH   | (planned / partial) | chinese-roberta ONNX not shipped | see later commits |
| JP   | OpenJTalk not in WASM | not shipped | blocked |

Deberta **FP32** (`deberta_v3_large_hs.onnx`) can replace the int8 file via `--deberta` if you have it locally.
