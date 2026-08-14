import type { LangCode, Precision } from "./types";

export const ENGINE_URL = "/engine/engine.wasm";
export const EVIL_URL = "/models/evil_v220.onnx";

export function bertUrl(precision: Precision, lang: LangCode): string {
  const tag = precision === "fp16" ? "fp16" : "int8";
  if (lang === "EN") return `/models/deberta_v3_large_hs.${tag}.onnx`;
  if (lang === "ZH") return `/models/chinese_roberta_wwm_ext_large_hs.${tag}.onnx`;
  return `/models/deberta_v2_large_japanese_char_wwm_hs.${tag}.onnx`;
}

export function bertLabel(lang: LangCode): string {
  if (lang === "EN") return "EN Deberta";
  if (lang === "ZH") return "ZH RoBERTa";
  return "JP Deberta";
}
