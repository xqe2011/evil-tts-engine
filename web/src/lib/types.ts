export type LangCode = "ZH" | "JP" | "EN";
export type Precision = "int8" | "fp16";

export const LANGS: LangCode[] = ["ZH", "JP", "EN"];
export const LANG_ID: Record<LangCode, number> = { ZH: 0, JP: 1, EN: 2 };
export const LANG_BY_ID: LangCode[] = ["ZH", "JP", "EN"];
export const SAMPLE_RATE = 44100;

export const PLACEHOLDER: Record<LangCode, string> = {
  EN: "Hello, I am evil, an assistant by apple banana.",
  ZH: "你好，我是助手。",
  JP: "こんにちは、世界。",
};

export type InferParams = {
  text: string;
  lang: LangCode;
  sid: number;
  seed: number;
  lengthScale: number;
  sdpRatio: number;
  noiseScale: number;
  sdpNoiseScale: number;
  bertDim: number;
  emoDim: number;
  skipStart?: boolean;
  skipEnd?: boolean;
  multilang?: boolean;
};

export const DEFAULT_PARAMS: Omit<InferParams, "text" | "lang"> = {
  sid: 0,
  seed: 42,
  lengthScale: 1.0,
  sdpRatio: 0.2,
  noiseScale: 0.6,
  sdpNoiseScale: 0.8,
  bertDim: 1024,
  emoDim: 512,
};

export type LoadPhase = "fetch" | "compile" | "ready" | "error";

export type LoadItem = {
  id: string;
  label: string;
  loaded: number;
  total: number;
  phase: LoadPhase;
  error?: string;
};

export type Latency = {
  prepareMs: number;
  bertMs: number;
  packMs: number;
  acousticMs: number;
  totalMs: number;
  audioSec: number;
  rtf: number;
  multilang: boolean;
};

export type InferStage = "prepare" | "bert" | "pack" | "acoustic";
