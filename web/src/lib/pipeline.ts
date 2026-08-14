import * as ort from "onnxruntime-web/webgpu";
import {
  concatF32,
  hasLangTags,
  packBert,
  packZinEmo,
  prepare,
  prepareMultilang,
  trimBertSegment,
  type EngineExports,
} from "./engine";
import { ensureBert, type TtsRuntime } from "./runtime";
import { asI64, floatFeed, peakAbs, tensorToF32 } from "./tensor";
import type { InferParams, InferStage, LangCode, Latency, LoadItem } from "./types";
import { LANG_BY_ID, SAMPLE_RATE } from "./types";

export type TaggedSeg = { lang: LangCode; text: string };

export function parseLangCode(raw: string): LangCode {
  const v = raw.trim().toUpperCase();
  if (v === "ZH" || v === "JP" || v === "EN") return v;
  throw new Error(`unknown lang ${raw} (want ZH|JP|EN)`);
}

export function parseLangTags(text: string): TaggedSeg[] | null {
  const re = /\[(ZH|EN|JP)\]/gi;
  const segs: TaggedSeg[] = [];
  let lastLang: LangCode | undefined;
  let lastEnd = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (lastLang) {
      const content = text.slice(lastEnd, m.index).trim();
      if (content) segs.push({ lang: lastLang, text: content });
    }
    lastLang = parseLangCode(m[1]!);
    lastEnd = m.index + m[0].length;
  }
  if (lastLang) {
    const content = text.slice(lastEnd).trim();
    if (content) segs.push({ lang: lastLang, text: content });
  }
  return segs.length ? segs : null;
}

function langsInText(text: string, fallback: LangCode): LangCode[] {
  const segs = parseLangTags(text);
  if (!segs) return [fallback];
  const out: LangCode[] = [];
  for (const s of segs) {
    if (!out.includes(s.lang)) out.push(s.lang);
  }
  return out.length ? out : [fallback];
}

async function runBertHidden(
  sess: ort.InferenceSession,
  inputIds: Int32Array,
  bertDim: number,
  lang: LangCode,
): Promise<Float32Array> {
  const ids = asI64(inputIds);
  const mask = new BigInt64Array(ids.length);
  mask.fill(1n);
  const out = await sess.run({
    input_ids: new ort.Tensor("int64", ids, [1, ids.length]),
    attention_mask: new ort.Tensor("int64", mask, [1, mask.length]),
  });
  const hiddenTensor = out["hidden"] ?? out["last_hidden_state"] ?? Object.values(out)[0];
  if (!hiddenTensor) throw new Error(`no ${lang} bert hidden output`);
  const hData = await tensorToF32(hiddenTensor);
  const bertPeak = peakAbs(hData);
  console.info("[bert]", {
    lang,
    type: hiddenTensor.type,
    location: hiddenTensor.location,
    peak: bertPeak,
    length: hData.length,
    sample0: hData[0],
  });
  if (!Number.isFinite(bertPeak) || bertPeak === 0) {
    throw new Error(`${lang} BERT hidden is empty/NaN (peak=${bertPeak})`);
  }
  const need = inputIds.length * bertDim;
  if (hData.length !== need && hData.length !== 1 * need) {
    throw new Error(
      `${lang} BERT hidden length ${hData.length} != seq*bertDim ${need} (seq=${inputIds.length} bertDim=${bertDim}).`,
    );
  }
  return hData.length === need ? hData : hData.slice(0, need);
}

async function inferMultilang(
  runtime: TtsRuntime,
  params: InferParams,
  text: string,
  onProgress: (items: Record<string, LoadItem>) => void,
  onStage: (stage: InferStage) => void,
): Promise<{
  phones: Int32Array;
  tones: Int32Array;
  language: Int32Array;
  packed: ReturnType<typeof packBert>;
  bertMs: number;
  packMs: number;
}> {
  const skipStart = params.skipStart ?? false;
  const skipEnd = params.skipEnd ?? false;
  onStage("prepare");
  const ml = prepareMultilang(runtime.engine, text, skipStart, skipEnd);
  const nSeg = ml.segments.length;
  const enChunks: Float32Array[] = [];
  const zhChunks: Float32Array[] = [];
  const jaChunks: Float32Array[] = [];
  let bertMs = 0;
  let packMs = 0;

  for (let idx = 0; idx < nSeg; idx++) {
    const seg = ml.segments[idx]!;
    const lang = LANG_BY_ID[seg.bertLang] ?? "EN";
    const sess = await ensureBert(runtime, lang, onProgress);
    const segSkipStart = idx !== 0 || (skipStart && idx === 0);
    const segSkipEnd = idx !== nSeg - 1 || (skipEnd && idx === nSeg - 1);
    onStage("bert");
    const t0 = performance.now();
    const hData = await runBertHidden(sess, seg.inputIds, params.bertDim, lang);
    bertMs += performance.now() - t0;
    onStage("pack");
    const t1 = performance.now();
    const packed = packBert(
      runtime.engine,
      hData,
      seg.inputIds.length,
      seg.word2ph,
      params.bertDim,
      params.emoDim,
      params.seed + idx,
      params.sdpNoiseScale,
      seg.bertLang,
    );
    const trimmed = trimBertSegment(packed, params.bertDim, segSkipStart, segSkipEnd);
    packMs += performance.now() - t1;
    enChunks.push(trimmed.en);
    zhChunks.push(trimmed.zh);
    jaChunks.push(trimmed.ja);
  }

  const nPhone = ml.phones.length;
  const t2 = performance.now();
  const { emo, zin } = packZinEmo(runtime.engine, nPhone, params.emoDim, params.seed, params.sdpNoiseScale);
  packMs += performance.now() - t2;
  return {
    phones: ml.phones,
    tones: ml.tones,
    language: ml.language,
    packed: {
      nPhone,
      bertDim: params.bertDim,
      emoDim: params.emoDim,
      en: concatF32(enChunks),
      zh: concatF32(zhChunks),
      ja: concatF32(jaChunks),
      emo,
      zin,
    },
    bertMs,
    packMs,
  };
}

export async function infer(
  runtime: TtsRuntime,
  params: InferParams,
  onProgress: (items: Record<string, LoadItem>) => void,
  onStage: (stage: InferStage) => void,
): Promise<{ samples: Float32Array; latency: Latency }> {
  const tAll = performance.now();
  let text = params.text.trim();
  if (!text) throw new Error("text is empty");

  const engine: EngineExports = runtime.engine;
  const tagged = hasLangTags(engine, text);
  const needed = langsInText(text, params.lang);
  for (const lang of needed) {
    await ensureBert(runtime, lang, onProgress);
  }

  const useMultilang = params.multilang || tagged;
  let packed: ReturnType<typeof packBert>;
  let phones: Int32Array;
  let tones: Int32Array;
  let language: Int32Array;
  let prepareMs = 0;
  let bertMs = 0;
  let packMs = 0;

  if (useMultilang) {
    const t0 = performance.now();
    const ml = await inferMultilang(runtime, params, text, onProgress, onStage);
    prepareMs = performance.now() - t0 - ml.bertMs - ml.packMs;
    phones = ml.phones;
    tones = ml.tones;
    language = ml.language;
    packed = ml.packed;
    bertMs = ml.bertMs;
    packMs = ml.packMs;
  } else {
    onStage("prepare");
    const t0 = performance.now();
    const prep = prepare(engine, text, params.lang);
    prepareMs = performance.now() - t0;
    const sess = await ensureBert(runtime, params.lang, onProgress);
    onStage("bert");
    const t1 = performance.now();
    const hData = await runBertHidden(sess, prep.inputIds, params.bertDim, params.lang);
    bertMs = performance.now() - t1;
    onStage("pack");
    const t2 = performance.now();
    packed = packBert(
      engine,
      hData,
      prep.inputIds.length,
      prep.word2ph,
      params.bertDim,
      params.emoDim,
      params.seed,
      params.sdpNoiseScale,
      prep.bertLang,
    );
    packMs = performance.now() - t2;
    phones = prep.phones;
    tones = prep.tones;
    language = prep.language;
  }

  onStage("acoustic");
  const T = packed.nPhone;
  const D = packed.bertDim;
  const feeds: Record<string, ort.Tensor> = {
    x: new ort.Tensor("int64", asI64(phones), [1, T]),
    t: new ort.Tensor("int64", asI64(tones), [1, T]),
    language: new ort.Tensor("int64", asI64(language), [1, T]),
    bert_0: floatFeed(runtime.evil, "bert_0", packed.zh, [T, D]),
    bert_1: floatFeed(runtime.evil, "bert_1", packed.ja, [T, D]),
    bert_2: floatFeed(runtime.evil, "bert_2", packed.en, [T, D]),
    emo: floatFeed(runtime.evil, "emo", packed.emo, [packed.emoDim, 1]),
    sid: new ort.Tensor("int64", BigInt64Array.from([BigInt(params.sid)]), [1]),
    zin: floatFeed(runtime.evil, "zin", packed.zin, [1, 2, T]),
    length_scale: floatFeed(runtime.evil, "length_scale", Float32Array.from([params.lengthScale]), []),
    sdp_ratio: floatFeed(runtime.evil, "sdp_ratio", Float32Array.from([params.sdpRatio]), []),
    noise_scale: floatFeed(runtime.evil, "noise_scale", Float32Array.from([params.noiseScale]), []),
  };

  const tAc = performance.now();
  const audioOut = await runtime.evil.run(feeds);
  const acousticMs = performance.now() - tAc;
  const o = audioOut["o"] ?? Object.values(audioOut)[0];
  if (!o) throw new Error("no acoustic output");
  const samples = await tensorToF32(o);
  const peak = peakAbs(samples);
  console.info("[acoustic]", {
    T,
    type: o.type,
    location: o.location,
    length: samples.length,
    peak,
    sample0: samples[0],
  });
  if (samples.length === 0 || !Number.isFinite(peak) || peak < 1e-7) {
    throw new Error("acoustic returned silence");
  }
  const totalMs = performance.now() - tAll;
  const audioSec = samples.length / SAMPLE_RATE;
  return {
    samples,
    latency: {
      prepareMs,
      bertMs,
      packMs,
      acousticMs,
      totalMs,
      audioSec,
      rtf: audioSec > 0 ? totalMs / 1000 / audioSec : 0,
      multilang: useMultilang,
    },
  };
}
