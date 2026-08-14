import type { LangCode } from "./types";

export type EngineExports = {
  memory: WebAssembly.Memory;
  engine_alloc(n: number): number;
  engine_alloc_free(ptr: number, n: number): void;
  engine_prepare(textPtr: number, textLen: number, outLenPtr: number): number;
  engine_prepare_lang?(
    textPtr: number,
    textLen: number,
    lang: number,
    outLenPtr: number,
  ): number;
  engine_free(ptr: number, len: number): void;
  engine_alloc_f32(n: number): number;
  engine_free_f32(ptr: number, n: number): void;
  engine_pack_bert(
    hiddenPtr: number,
    seq: number,
    word2phPtr: number,
    nW2p: number,
    bertDim: number,
    emoDim: number,
    seed: number,
    sdpNoise: number,
    outLenPtr: number,
  ): number;
  engine_pack_bert_lang?(
    hiddenPtr: number,
    seq: number,
    word2phPtr: number,
    nW2p: number,
    bertDim: number,
    emoDim: number,
    seed: number,
    sdpNoise: number,
    bertLang: number,
    outLenPtr: number,
  ): number;
  engine_prepare_multilang?(
    textPtr: number,
    textLen: number,
    skipStart: number,
    skipEnd: number,
    outLenPtr: number,
  ): number;
  engine_has_lang_tags?(textPtr: number, textLen: number): number;
  engine_pack_zin_emo?(
    nPhone: number,
    emoDim: number,
    seed: number,
    sdpNoise: number,
    outLenPtr: number,
  ): number;
};

export type Prepared = {
  inputIds: Int32Array;
  phones: Int32Array;
  tones: Int32Array;
  language: Int32Array;
  word2ph: Int32Array;
  bertLang: number;
};

export type MultilangPrepared = {
  phones: Int32Array;
  tones: Int32Array;
  language: Int32Array;
  segments: Array<{
    inputIds: Int32Array;
    word2ph: Int32Array;
    bertLang: number;
  }>;
};

export type PackedBert = {
  nPhone: number;
  bertDim: number;
  emoDim: number;
  en: Float32Array;
  zh: Float32Array;
  ja: Float32Array;
  emo: Float32Array;
  zin: Float32Array;
};

const LANG_ID: Record<LangCode, number> = { ZH: 0, JP: 1, EN: 2 };

function u32(view: DataView, o: number) {
  return view.getUint32(o, true);
}
function i32s(view: DataView, o: number, n: number) {
  const a = new Int32Array(n);
  for (let i = 0; i < n; i++) a[i] = view.getInt32(o + i * 4, true);
  return a;
}
function f32s(buf: ArrayBuffer, byteOffset: number, n: number) {
  return new Float32Array(buf.slice(byteOffset, byteOffset + n * 4));
}

export async function instantiateEngine(bytes: ArrayBuffer): Promise<EngineExports> {
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports as unknown as EngineExports;
}

function allocText(engine: EngineExports, text: string) {
  const enc = new TextEncoder();
  const utf8 = enc.encode(text);
  const tPtr = engine.engine_alloc(utf8.length);
  new Uint8Array(engine.memory.buffer, tPtr, utf8.length).set(utf8);
  return { tPtr, utf8Len: utf8.length };
}

export function prepare(engine: EngineExports, text: string, lang: LangCode): Prepared {
  const enc = new TextEncoder();
  const utf8 = enc.encode(text);
  const tPtr = engine.engine_alloc(utf8.length);
  new Uint8Array(engine.memory.buffer, tPtr, utf8.length).set(utf8);
  const lenPtr = engine.engine_alloc(4);
  const langId = LANG_ID[lang];
  const blobPtr =
    engine.engine_prepare_lang != null
      ? engine.engine_prepare_lang(tPtr, utf8.length, langId, lenPtr)
      : engine.engine_prepare(tPtr, utf8.length, lenPtr);
  const len = new DataView(engine.memory.buffer).getUint32(lenPtr, true);
  engine.engine_alloc_free(lenPtr, 4);
  engine.engine_alloc_free(tPtr, utf8.length);
  if (!blobPtr || !len) throw new Error("engine_prepare failed");

  const copy = engine.memory.buffer.slice(blobPtr, blobPtr + len);
  engine.engine_free(blobPtr, len);
  const view = new DataView(copy);
  if (u32(view, 0) !== 0x54505331) throw new Error("bad prepare magic");
  const nIds = u32(view, 4);
  const nPh = u32(view, 8);
  const nW2 = u32(view, 12);
  const bertLang = u32(view, 16);
  let o = 20;
  const inputIds = i32s(view, o, nIds);
  o += nIds * 4;
  const phones = i32s(view, o, nPh);
  o += nPh * 4;
  const tones = i32s(view, o, nPh);
  o += nPh * 4;
  const language = i32s(view, o, nPh);
  o += nPh * 4;
  const word2ph = i32s(view, o, nW2);
  if (phones.length === 0) throw new Error("engine returned empty phones");
  return { inputIds, phones, tones, language, word2ph, bertLang };
}

export function hasLangTags(engine: EngineExports, text: string): boolean {
  if (engine.engine_has_lang_tags) {
    const { tPtr, utf8Len } = allocText(engine, text);
    const v = engine.engine_has_lang_tags(tPtr, utf8Len);
    engine.engine_alloc_free(tPtr, utf8Len);
    return v !== 0;
  }
  return /\[(ZH|EN|JP)\]/i.test(text);
}

export function prepareMultilang(
  engine: EngineExports,
  text: string,
  skipStart: boolean,
  skipEnd: boolean,
): MultilangPrepared {
  if (!engine.engine_prepare_multilang) {
    throw new Error("engine.wasm lacks engine_prepare_multilang");
  }
  const { tPtr, utf8Len } = allocText(engine, text);
  const lenPtr = engine.engine_alloc(4);
  const blobPtr = engine.engine_prepare_multilang(
    tPtr,
    utf8Len,
    skipStart ? 1 : 0,
    skipEnd ? 1 : 0,
    lenPtr,
  );
  const len = new DataView(engine.memory.buffer).getUint32(lenPtr, true);
  engine.engine_alloc_free(lenPtr, 4);
  engine.engine_alloc_free(tPtr, utf8Len);
  if (!blobPtr || !len) throw new Error("engine_prepare_multilang failed — check [ZH]/[EN]/[JP] tags");

  const copy = engine.memory.buffer.slice(blobPtr, blobPtr + len);
  engine.engine_free(blobPtr, len);
  const view = new DataView(copy);
  if (u32(view, 0) !== 0x54505332) throw new Error("bad multilang prepare magic");
  const nPh = u32(view, 4);
  const nSeg = u32(view, 8);
  let o = 12;
  const phones = i32s(view, o, nPh);
  o += nPh * 4;
  const tones = i32s(view, o, nPh);
  o += nPh * 4;
  const language = i32s(view, o, nPh);
  o += nPh * 4;
  const segments: MultilangPrepared["segments"] = [];
  for (let i = 0; i < nSeg; i++) {
    const nIds = u32(view, o);
    o += 4;
    const nW2 = u32(view, o);
    o += 4;
    const bertLang = u32(view, o);
    o += 4;
    const inputIds = i32s(view, o, nIds);
    o += nIds * 4;
    const word2ph = i32s(view, o, nW2);
    o += nW2 * 4;
    segments.push({ inputIds, word2ph, bertLang });
  }
  if (phones.length === 0) throw new Error("multilang prepare returned empty phones");
  return { phones, tones, language, segments };
}

export function trimBertSegment(
  packed: { en: Float32Array; zh: Float32Array; ja: Float32Array; nPhone: number },
  bertDim: number,
  skipStart: boolean,
  skipEnd: boolean,
) {
  let { en, zh, ja, nPhone } = packed;
  if (skipStart && nPhone >= 3) {
    en = en.slice(3 * bertDim);
    zh = zh.slice(3 * bertDim);
    ja = ja.slice(3 * bertDim);
    nPhone -= 3;
  }
  if (skipEnd && nPhone >= 2) {
    const keep = (nPhone - 2) * bertDim;
    en = en.slice(0, keep);
    zh = zh.slice(0, keep);
    ja = ja.slice(0, keep);
    nPhone -= 2;
  }
  return { en, zh, ja, nPhone };
}

export function concatF32(chunks: Float32Array[]) {
  const n = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Float32Array(n);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

export function packZinEmo(
  engine: EngineExports,
  nPhone: number,
  emoDim: number,
  seed: number,
  sdpNoise: number,
) {
  if (!engine.engine_pack_zin_emo) {
    throw new Error("engine.wasm lacks engine_pack_zin_emo");
  }
  const lenPtr = engine.engine_alloc(4);
  const blobPtr = engine.engine_pack_zin_emo(nPhone, emoDim, seed, sdpNoise, lenPtr);
  const len = new DataView(engine.memory.buffer).getUint32(lenPtr, true);
  engine.engine_alloc_free(lenPtr, 4);
  if (!blobPtr || !len) throw new Error("engine_pack_zin_emo failed");
  const copy = engine.memory.buffer.slice(blobPtr, blobPtr + len);
  engine.engine_free(blobPtr, len);
  const view = new DataView(copy);
  if (u32(view, 0) !== 0x54454d31) throw new Error("bad zin/emo magic");
  const emoD = u32(view, 4);
  const nPh = u32(view, 8);
  let o = 12;
  const emo = f32s(copy, o, emoD);
  o += emoD * 4;
  const zin = f32s(copy, o, 2 * nPh);
  return { emo, zin };
}

export function packBert(
  engine: EngineExports,
  hidden: Float32Array,
  seq: number,
  word2ph: Int32Array,
  bertDim: number,
  emoDim: number,
  seed: number,
  sdpNoise: number,
  bertLang: number,
): PackedBert {
  const hPtr = engine.engine_alloc_f32(hidden.length);
  new Float32Array(engine.memory.buffer, hPtr, hidden.length).set(hidden);
  const wPtr = engine.engine_alloc(word2ph.length * 4);
  new Int32Array(engine.memory.buffer, wPtr, word2ph.length).set(word2ph);
  const lenPtr = engine.engine_alloc(4);
  const blobPtr =
    engine.engine_pack_bert_lang != null
      ? engine.engine_pack_bert_lang(
          hPtr,
          seq,
          wPtr,
          word2ph.length,
          bertDim,
          emoDim,
          seed,
          sdpNoise,
          bertLang,
          lenPtr,
        )
      : engine.engine_pack_bert(
          hPtr,
          seq,
          wPtr,
          word2ph.length,
          bertDim,
          emoDim,
          seed,
          sdpNoise,
          lenPtr,
        );
  const len = new DataView(engine.memory.buffer).getUint32(lenPtr, true);
  engine.engine_alloc_free(lenPtr, 4);
  engine.engine_free_f32(hPtr, hidden.length);
  engine.engine_alloc_free(wPtr, word2ph.length * 4);
  if (!blobPtr || !len) throw new Error("engine_pack_bert failed");

  const copy = engine.memory.buffer.slice(blobPtr, blobPtr + len);
  engine.engine_free(blobPtr, len);
  const view = new DataView(copy);
  if (u32(view, 0) !== 0x54424e31) throw new Error("bad pack magic");
  const nPhone = u32(view, 4);
  const dim = u32(view, 8);
  const emoD = u32(view, 12);
  if (dim !== bertDim) throw new Error(`bert_dim mismatch packed=${dim} want=${bertDim}`);
  let o = 16;
  const en = f32s(copy, o, nPhone * dim);
  o += nPhone * dim * 4;
  const zh = f32s(copy, o, nPhone * dim);
  o += nPhone * dim * 4;
  const ja = f32s(copy, o, nPhone * dim);
  o += nPhone * dim * 4;
  const emo = f32s(copy, o, emoD);
  o += emoD * 4;
  const zin = f32s(copy, o, 2 * nPhone);
  return { nPhone, bertDim: dim, emoDim: emoD, en, zh, ja, emo, zin };
}
