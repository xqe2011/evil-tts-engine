# evil-tts-engine

Standalone Bert-VITS2 V220-style TTS frontend (Rust → WASM) + Bun/ORT inference for the **evil** voice.

## Layout

```
engine/          Rust WASM frontend (normalize, EN/ZH G2P, SPM, ZH vocab, bert pack)
models/
  evil_v220.onnx
  deberta_v3_large_hs.int8.onnx   # EN BERT. FP32 Deberta not shipped (~1.5GB).
  zh-bert/
    chinese_roberta_wwm_ext_large_hs.int8.onnx  # ZH BERT (hfl/chinese-roberta-wwm-ext-large)
    vocab.txt + tokenizer assets
  config.json
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

# English (Deberta int8 → bert_2)
bun infer.ts --lang EN --text "Hello, I am evil, an assistant by apple banana." --out examples/en.wav

# Chinese (chinese-roberta int8 → bert_0)
bun infer.ts --lang ZH --text "你好，我是助手。" --out examples/zh.wav
```

## Rebuild frontend WASM

```bash
cd engine && ./build.sh
```

## Language support

| Lang | G2P | BERT features | Status |
|------|-----|---------------|--------|
| EN   | CMUdict + SPM | `deberta_v3_large_hs.int8.onnx` → `bert_2` | working |
| ZH   | opencpop + `pinyin` (no jieba ToneSandhi) | `chinese_roberta_wwm_ext_large_hs.int8.onnx` → `bert_0` | working |
| JP   | OpenJTalk | n/a | **blocked** in WASM — `infer.ts --lang JP` errors on purpose |

### ZH notes

- BERT checkpoint matches Bert-VITS2: `hfl/chinese-roberta-wwm-ext-large`, ONNX output is `hidden_states[-3]` (1024-d).
- ZH requires that ONNX the same way EN requires Deberta. Missing the file is an error; there is no zero-BERT fallback.
- Engine emits real WordPiece `input_ids` via embedded `zh_vocab.txt` (char-level; aligns with `word2ph`).
- G2P skips jieba POS / ToneSandhi from upstream Python — pronunciation can differ slightly on sandhi cases.

### JP note

V220 Japanese G2P depends on **pyopenjtalk**. Porting OpenJTalk dictionaries + frontend into WASM was not done; no partial JP commit.
