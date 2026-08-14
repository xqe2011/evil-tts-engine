# evil-tts-engine

Standalone Bert-VITS2 V220-style TTS frontend (Rust → WASM) + Bun/ORT inference for the **evil** voice.

## Layout

```
engine/          Rust WASM frontend (normalize, EN/ZH/JP G2P, SPM, vocab, bert pack)
models/
  evil_v220.onnx
  deberta_v3_large_hs.int8.onnx / .fp16.onnx               # EN BERT → bert_2
  chinese_roberta_wwm_ext_large_hs.int8.onnx / .fp16.onnx  # ZH BERT → bert_0
  deberta_v2_large_japanese_char_wwm_hs.int8.onnx / .fp16.onnx  # JP BERT → bert_1
infer.ts         Bun CLI: engine.wasm + onnxruntime-web/wasm
web/             React UI: same pipeline, onnxruntime-web WebGPU only
scripts/         ONNX export helpers (JP BERT, FP16 conversion)
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

# Japanese (deberta-v2-japanese-char int8 → bert_1)
bun infer.ts --lang JP --text "こんにちは、世界！" --out examples/jp.wav

# Restrict BERTs: only load ZH+EN; JP-tagged text is skipped
bun infer.ts --lang zh,en --text "[ZH]你好[EN]hello[JP]こんにちは" --out examples/zh-en.wav

# FP16 BERT (acoustic is always evil_v220.onnx)
bun infer.ts --fp16 --lang EN --text "Hello, I am evil." --out examples/en-fp16.wav
```

## Web (React + WebGPU)

Browser demo of the same pipeline as `infer.ts`. ONNX Runtime uses the **WebGPU** execution provider only (no WASM EP fallback). Chrome / Edge 113+ with WebGPU.

```bash
cd web
bun install
bun run dev
```

From the repo root: `bun run web`

Open the printed localhost URL. The Vite server streams `models/` and `engine/engine.wasm` from the repo.

- Switch **int8** / **fp16** (reloads BERT; acoustic is always `evil_v220.onnx`)
- Switch **zh** / **jp** / **en** (loads that language’s BERT)
- Text in → synthesize → waveform, playback, WAV download, per-stage latency
- Fetch + WebGPU compile progress per file (ORT wasm, engine, acoustic, BERT)

First load is large (engine ~94MB + acoustic ~197MB + one BERT ~287–781MB).

## Rebuild frontend WASM

```bash
cd engine && ./build.sh
```

## Language support

| Lang | G2P | BERT features | Status |
|------|-----|---------------|--------|
| EN   | CMUdict + SPM | `deberta_v3_large_hs.int8.onnx` → `bert_2` | working |
| ZH   | jieba POS + ToneSandhi + cn2an + opencpop | `chinese_roberta_wwm_ext_large_hs.int8.onnx` → `bert_0` | working |
| JP   | jpreprocess + NAIST-JDIC (OpenJTalk-compatible labels) | `deberta_v2_large_japanese_char_wwm_hs.int8.onnx` → `bert_1` | working |

### ZH notes

- BERT checkpoint matches Bert-VITS2: `hfl/chinese-roberta-wwm-ext-large`, ONNX output is `hidden_states[-3]` (1024-d).
- ZH requires that ONNX the same way EN requires Deberta. Missing the file is an error; there is no zero-BERT fallback.
- Engine emits real WordPiece `input_ids` via embedded `zh_vocab.txt` (char-level; aligns with `word2ph`).
- G2P mirrors upstream V220 `chinese.py`: jieba POS segmentation, ToneSandhi, cn2an-style number reading (`chinese-number`), embedded pypinyin initials/finals map, opencpop phone lookup.
- Emotion (`emo`) stays zero-filled; no CLAP.

### JP notes

- BERT checkpoint matches Bert-VITS2 V220: `ku-nlp/deberta-v2-large-japanese-char-wwm`, ONNX output is `hidden_states[-3]` (1024-d) → acoustic `bert_1`.
- JP requires that ONNX the same way EN/ZH require their BERT files. Missing the file is an error; there is no zero-BERT fallback.
- G2P uses [`jpreprocess`](https://github.com/jpreprocess/jpreprocess) with bundled NAIST-JDIC (wasm32-compatible). `engine.wasm` is ~85MB because of the embedded dictionary.
- Normalization mirrors upstream V220 `japanese.py`: NFKC, `num2words` (via `num2words2-core`), comma-stripped thousands, currency expansion (`$`→ドル, etc.), then punctuation cleanup before G2P.
- Alpha/symbol reading map exists but is **not** applied in `text_normalize` (same as upstream — those calls are commented out in Bert-VITS2 V220).

### JP note (historical)

Earlier builds blocked `--lang JP` because pyopenjtalk could not compile to WASM. The jpreprocess path replaces that dependency.
