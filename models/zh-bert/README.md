# chinese-roberta-wwm-ext-large (ZH BERT)

Source: [`hfl/chinese-roberta-wwm-ext-large`](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) (Bert-VITS2 `bert_models.json` / V220 `chinese_bert.py`).

Shipped ONNX exports **hidden_states[-3]** (dim 1024), same slice as upstream Python:

| File | Notes |
|------|--------|
| `chinese_roberta_wwm_ext_large_hs.int8.onnx` | Dynamic int8 (~287MB). Prefer this for ORT wasm. |
| `vocab.txt` + tokenizer JSON | WordPiece; engine embeds `zh_vocab.txt` for char-level ids. |

```bash
bun infer.ts --lang ZH --text "你好，我是助手。" --out examples/zh.wav
# or override:
bun infer.ts --lang ZH --zh-bert models/zh-bert/chinese_roberta_wwm_ext_large_hs.int8.onnx ...
```

FP32 ONNX (~1.2GB) is not shipped (same policy as EN Deberta).
