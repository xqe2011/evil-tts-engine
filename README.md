# evil-tts-engine

Standalone Bert-VITS2 V220-style TTS frontend (Rust → WASM) + Bun/ORT inference for the **evil** voice.

## Layout

```
engine/          Rust WASM frontend (normalize, EN/ZH G2P, SPM, bert pack)
models/
  evil_v220.onnx
  deberta_v3_large_hs.int8.onnx   # EN BERT (preferred). FP32 Deberta not shipped (~1.5GB).
  config.json
  zh-bert/                        # chinese-roberta tokenizer only (no ONNX weights)
infer.ts         Bun CLI: engine.wasm + onnxruntime-web/wasm
examples/        sample WAVs
```

## Requirements

- [Bun](https://bun.sh)
- [Rust](https://rustup.rs) + `wasm32-unknown-unknown` (only to rebuild engine)
- Mac CPU / ORT WASM (no CUDA)

## Run

```bash
bun install

# English (Deberta int8)
bun infer.ts --lang EN --text "Hello, I am evil, an assistant by apple banana." --out examples/en.wav

# Chinese G2P works; ZH BERT features are zeros unless you supply --zh-bert ONNX
bun infer.ts --lang ZH --text "你好，我是助手。" --out examples/zh.wav
```

## Rebuild frontend WASM

```bash
cd engine && ./build.sh
```

## Language support

| Lang | G2P | BERT features | Status |
|------|-----|---------------|--------|
| EN   | CMUdict + SPM | `deberta_v3_large_hs.int8.onnx` | working |
| ZH   | opencpop + `pinyin` (no jieba ToneSandhi) | **no ONNX** — zeros; tokenizer under `models/zh-bert/` | G2P + acoustics work; quality limited without ZH BERT |
| JP   | OpenJTalk | n/a | **blocked** in WASM — `infer.ts --lang JP` errors on purpose |

### ZH BERT note

Upstream `chinese-roberta-wwm-ext-large` weights were **not** present under `_work/Bert-VITS2/bert/` (tokenizer only). Exporting FP32/int8 ONNX on Mac CPU is heavy (~1GB+ download + convert). To plug in later:

1. Obtain `pytorch_model.bin` / safetensors for `hfl/chinese-roberta-wwm-ext-large`
2. Export hidden-states ONNX (dim 1024)
3. Pass `--zh-bert path/to.onnx` (and wire real WordPiece `input_ids` in the engine if needed)

### JP note

V220 Japanese G2P depends on **pyopenjtalk**. Porting OpenJTalk dictionaries + frontend into WASM was not done; no partial JP commit.
