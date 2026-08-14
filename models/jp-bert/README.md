# deberta-v2-large-japanese-char-wwm (JP BERT)

Source: [`ku-nlp/deberta-v2-large-japanese-char-wwm`](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm) (Bert-VITS2 V220 `japanese_bert.py` / `bert_models.json`).

Shipped ONNX exports **hidden_states[-3]** (dim 1024), same slice as upstream Python:

| File | Notes |
|------|--------|
| `deberta_v2_large_japanese_char_wwm_hs.int8.onnx` | Dynamic int8 (~425MB). Prefer this for ORT wasm. |
| `vocab.txt` | Char-level Deberta vocab; engine embeds `jp_vocab.txt` for `input_ids`. |

Regenerate (requires PyTorch + transformers in a local venv):

```bash
python scripts/export_jp_bert_onnx.py
# downloads weights into models/jp-bert/ if missing, writes FP32 + INT8 ONNX
```

```bash
bun infer.ts --lang JP --text "こんにちは、世界！" --out examples/jp.wav
# or override:
bun infer.ts --lang JP --jp-bert models/jp-bert/deberta_v2_large_japanese_char_wwm_hs.int8.onnx ...
```

FP32 ONNX (~1.1GB external weights) and `pytorch_model.bin` are not shipped (same policy as EN/ZH).
