# chinese-roberta-wwm-ext-large (tokenizer only)

Tokenizer / config copied for a future ZH BERT ONNX export.

**No `pytorch_model.bin` / ONNX shipped** — weights were not available in the upstream bert checkout used to build this repo.

When you have an ONNX with a `hidden` (or `last_hidden_state`) output of shape `[1, seq, 1024]`, place it here as e.g. `chinese_roberta_wwm_ext_large.onnx` and run:

```bash
bun infer.ts --lang ZH --zh-bert models/zh-bert/chinese_roberta_wwm_ext_large.onnx --text "你好" --out examples/zh.wav
```

Note: current WASM ZH path emits placeholder `input_ids` (zeros) sized to `chars+2`. A full WordPiece encode step is still required before real ZH BERT features work.
