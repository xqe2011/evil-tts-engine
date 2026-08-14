import * as ort from "onnxruntime-web/wasm";

/**
 * Bun TTS: engine.wasm (text/G2P) + ORT wasm (Deberta + acoustic).
 *
 * Examples:
 *   bun infer.ts --text "Hello" --out examples/en.wav
 *   bun infer.ts --lang ZH --text "你好，我是助手。" --out examples/zh.wav
 *   bun infer.ts --deberta models/deberta_v3_large_hs.int8.onnx
 *   bun infer.ts --lang JP --text "こんにちは、世界。" --out examples/jp.wav
 *   bun infer.ts --text "[ZH]你好[EN]hello[JP]こんにちは" --out examples/multilang.wav
 *   bun infer.ts --lang zh,en --text "[ZH]你好[EN]hello[JP]こんにちは" --out /tmp/zh-en.wav
 *   bun infer.ts --fp16 --lang EN --text "Hello" --out examples/en-fp16.wav
 */

const ROOT = import.meta.dir;
const SAMPLE_RATE = 44100;

/** 0=ZH, 1=JP, 2=EN — matches Bert-VITS2 V220 language_id_map */
export type LangCode = "ZH" | "JP" | "EN";

export type InferOptions = {
  text: string;
  out: string;
  /** G2P language for untagged text (first of `langs`). */
  lang: LangCode;
  /** BERT allow-list. Tagged segments outside this list are skipped. */
  langs: LangCode[];
  evil: string;
  /** EN Deberta ONNX (used when lang=EN). */
  deberta: string;
  /** ZH chinese-roberta ONNX (used when lang=ZH). */
  zhBert: string;
  /** JP deberta-v2-japanese-char ONNX (used when lang=JP). */
  jpBert: string;
  engine: string;
  /** Use models/*.fp16.onnx for BERT. Acoustic is always evil_v220.onnx. */
  fp16?: boolean;
  bertDim: number;
  emoDim: number;
  sid: number;
  seed: number;
  lengthScale: number;
  sdpRatio: number;
  noiseScale: number;
  sdpNoiseScale: number;
  debertaOutput?: string;
  evilOutput?: string;
  /** Force multilang path even without `[ZH]`/`[EN]`/`[JP]` tags. */
  multilang?: boolean;
  /** Trim leading blanks on first segment (V220 infer_multilang). */
  skipStart?: boolean;
  /** Trim trailing blanks on last segment (V220 infer_multilang). */
  skipEnd?: boolean;
};

const DEFAULTS: InferOptions = {
  text: "Hello, I am evil, an assistant by apple banana.",
  out: `${ROOT}/examples/en.wav`,
  lang: "EN",
  langs: ["ZH", "JP", "EN"],
  evil: `${ROOT}/models/evil_v220.onnx`,
  deberta: `${ROOT}/models/deberta_v3_large_hs.int8.onnx`,
  zhBert: `${ROOT}/models/chinese_roberta_wwm_ext_large_hs.int8.onnx`,
  jpBert: `${ROOT}/models/deberta_v2_large_japanese_char_wwm_hs.int8.onnx`,
  engine: `${ROOT}/engine/engine.wasm`,
  bertDim: 1024,
  emoDim: 512,
  sid: 0,
  seed: 42,
  lengthScale: 1.0,
  sdpRatio: 0.2,
  noiseScale: 0.6,
  sdpNoiseScale: 0.8,
};

const LANG_ID: Record<LangCode, number> = { ZH: 0, JP: 1, EN: 2 };

type EngineExports = {
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
  engine_assets_bytes(): number;
};

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

function parseArgs(argv: string[]): InferOptions {
  const opt: InferOptions = { ...DEFAULTS };
  const take = (i: number) => {
    const v = argv[i + 1];
    if (v === undefined || v.startsWith("--")) throw new Error(`missing value for ${argv[i]}`);
    return v;
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    switch (a) {
      case "--text":
        opt.text = take(i++);
        break;
      case "--out":
        opt.out = take(i++);
        break;
      case "--lang": {
        const list = parseLangList(take(i++));
        opt.langs = list;
        opt.lang = list[0]!;
        break;
      }
      case "--evil":
        opt.evil = take(i++);
        break;
      case "--deberta":
        opt.deberta = take(i++);
        break;
      case "--zh-bert":
        opt.zhBert = take(i++);
        break;
      case "--jp-bert":
        opt.jpBert = take(i++);
        break;
      case "--engine":
        opt.engine = take(i++);
        break;
      case "--fp16":
        opt.fp16 = true;
        opt.deberta = `${ROOT}/models/deberta_v3_large_hs.fp16.onnx`;
        opt.zhBert = `${ROOT}/models/chinese_roberta_wwm_ext_large_hs.fp16.onnx`;
        opt.jpBert = `${ROOT}/models/deberta_v2_large_japanese_char_wwm_hs.fp16.onnx`;
        break;
      case "--bert-dim":
        opt.bertDim = Number(take(i++));
        break;
      case "--emo-dim":
        opt.emoDim = Number(take(i++));
        break;
      case "--sid":
        opt.sid = Number(take(i++));
        break;
      case "--seed":
        opt.seed = Number(take(i++));
        break;
      case "--length-scale":
        opt.lengthScale = Number(take(i++));
        break;
      case "--sdp-ratio":
        opt.sdpRatio = Number(take(i++));
        break;
      case "--noise-scale":
        opt.noiseScale = Number(take(i++));
        break;
      case "--sdp-noise-scale":
        opt.sdpNoiseScale = Number(take(i++));
        break;
      case "--deberta-output":
        opt.debertaOutput = take(i++);
        break;
      case "--evil-output":
        opt.evilOutput = take(i++);
        break;
      case "--multilang":
        opt.multilang = true;
        break;
      case "--skip-start":
        opt.skipStart = true;
        break;
      case "--skip-end":
        opt.skipEnd = true;
        break;
      case "--help":
      case "-h":
        console.log(`Usage: bun infer.ts [options]
  --text STR
  --out PATH
  --lang ZH|JP|EN       untagged G2P lang; also BERT allow-list
                        comma/[list] ok: zh,en  or  [zh,en,jp]
                        default allow-list ZH,JP,EN (untagged still EN)
  --multilang           force multilang path
  --skip-start          trim leading blanks on first segment (V220 infer_multilang)
  --skip-end            trim trailing blanks on last segment
  --evil PATH           acoustic ONNX
  --deberta PATH        EN Deberta ONNX
  --zh-bert PATH        ZH chinese-roberta hidden-states ONNX
  --jp-bert PATH        JP deberta-v2-japanese-char hidden-states ONNX
  --engine PATH         frontend wasm
  --fp16                use models/*.fp16.onnx for EN/ZH/JP BERT (acoustic stays FP32)
  --bert-dim N --emo-dim N
  --sid N --seed N
  --length-scale F --sdp-ratio F --noise-scale F --sdp-noise-scale F`);
        process.exit(0);
      default:
        if (a.startsWith("-")) throw new Error(`unknown flag ${a}`);
        if (opt.text === DEFAULTS.text) opt.text = a;
        else if (opt.seed === DEFAULTS.seed && !Number.isNaN(Number(a))) opt.seed = Number(a);
        else opt.out = a;
    }
  }
  return opt;
}

function parseLangCode(raw: string): LangCode {
  const v = raw.trim().toUpperCase();
  if (v === "ZH" || v === "JP" || v === "EN") return v;
  throw new Error(`unknown lang ${raw} (want ZH|JP|EN)`);
}

/** `EN` / `zh,en` / `[zh, en, jp]` */
function parseLangList(raw: string): LangCode[] {
  const parts = raw
    .replace(/[\[\]\s]/g, "")
    .split(/[,+|]+/)
    .filter(Boolean);
  if (parts.length === 0) throw new Error("--lang needs at least one of ZH|JP|EN");
  const out: LangCode[] = [];
  for (const p of parts) {
    const lang = parseLangCode(p);
    if (!out.includes(lang)) out.push(lang);
  }
  return out;
}

type TaggedSeg = { lang: LangCode; text: string };

function parseLangTags(text: string): TaggedSeg[] | null {
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

function filterTaggedText(
  text: string,
  enabled: ReadonlySet<LangCode>,
): { text: string; skipped: TaggedSeg[] } {
  const segs = parseLangTags(text);
  if (!segs) return { text, skipped: [] };
  const kept: TaggedSeg[] = [];
  const skipped: TaggedSeg[] = [];
  for (const s of segs) {
    if (enabled.has(s.lang)) kept.push(s);
    else skipped.push(s);
  }
  for (const s of skipped) {
    console.error(`skip unsupported lang ${s.lang} (not in --lang ${[...enabled].join(",")}): ${s.text}`);
  }
  return {
    text: kept.map((s) => `[${s.lang}]${s.text}`).join(""),
    skipped,
  };
}

function bertPath(opt: InferOptions, lang: LangCode): string {
  if (lang === "EN") return opt.deberta;
  if (lang === "ZH") return opt.zhBert;
  return opt.jpBert;
}

type BertLoader = {
  get(lang: LangCode): Promise<ort.InferenceSession>;
  release(): Promise<void>;
};

function createBertLoader(opt: InferOptions, enabled: ReadonlySet<LangCode>): BertLoader {
  const cache = new Map<LangCode, ort.InferenceSession>();
  return {
    async get(lang) {
      if (!enabled.has(lang)) {
        throw new Error(`lang ${lang} is not in --lang allow-list`);
      }
      const hit = cache.get(lang);
      if (hit) return hit;
      const path = bertPath(opt, lang);
      if (!(await Bun.file(path).exists())) {
        throw new Error(`${lang} BERT ONNX not found: ${path}`);
      }
      const sess = await ort.InferenceSession.create(await Bun.file(path).arrayBuffer(), {
        executionProviders: ["wasm"],
      });
      cache.set(lang, sess);
      return sess;
    },
    async release() {
      for (const sess of cache.values()) {
        await sess.release();
      }
      cache.clear();
    },
  };
}

async function loadEngine(path: string): Promise<EngineExports> {
  const bytes = await Bun.file(path).arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports as unknown as EngineExports;
}

const LANG_BY_ID: LangCode[] = ["ZH", "JP", "EN"];

type Prepared = {
  inputIds: Int32Array;
  phones: Int32Array;
  tones: Int32Array;
  language: Int32Array;
  word2ph: Int32Array;
  bertLang: number;
};

type MultilangPrepared = {
  phones: Int32Array;
  tones: Int32Array;
  language: Int32Array;
  segments: Array<{
    inputIds: Int32Array;
    word2ph: Int32Array;
    bertLang: number;
  }>;
};

function allocText(engine: EngineExports, text: string) {
  const enc = new TextEncoder();
  const utf8 = enc.encode(text);
  const tPtr = engine.engine_alloc(utf8.length);
  new Uint8Array(engine.memory.buffer, tPtr, utf8.length).set(utf8);
  return { tPtr, utf8Len: utf8.length };
}

function prepare(engine: EngineExports, text: string, lang: LangCode): Prepared {
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
  // TPS1 v2: bert_lang at offset 16, arrays at 20
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
  if (phones.length === 0) {
    throw new Error("engine returned empty phones");
  }
  return { inputIds, phones, tones, language, word2ph, bertLang };
}

function hasLangTags(engine: EngineExports, text: string): boolean {
  if (engine.engine_has_lang_tags) {
    const { tPtr, utf8Len } = allocText(engine, text);
    const v = engine.engine_has_lang_tags!(tPtr, utf8Len);
    engine.engine_alloc_free(tPtr, utf8Len);
    return v !== 0;
  }
  return /\[(ZH|EN|JP)\]/.test(text);
}

function prepareMultilang(
  engine: EngineExports,
  text: string,
  skipStart: boolean,
  skipEnd: boolean,
): MultilangPrepared {
  if (!engine.engine_prepare_multilang) {
    throw new Error("engine.wasm lacks engine_prepare_multilang — rebuild with ./engine/build.sh");
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

function trimBertSegment(
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

function concatF32(chunks: Float32Array[]) {
  const n = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Float32Array(n);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

function packZinEmo(
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
  const blobPtr = engine.engine_pack_zin_emo!(nPhone, emoDim, seed, sdpNoise, lenPtr);
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

function packBert(
  engine: EngineExports,
  hidden: Float32Array,
  seq: number,
  word2ph: Int32Array,
  bertDim: number,
  emoDim: number,
  seed: number,
  sdpNoise: number,
  bertLang: number,
) {
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

function writeWav(path: string, samples: Float32Array, sr: number) {
  const data = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const x = Math.max(-1, Math.min(1, samples[i]!));
    data[i] = (x * 32767) | 0;
  }
  const buf = new ArrayBuffer(44 + data.length * 2);
  const v = new DataView(buf);
  const w = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  w(0, "RIFF");
  v.setUint32(4, 36 + data.length * 2, true);
  w(8, "WAVE");
  w(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  w(36, "data");
  v.setUint32(40, data.length * 2, true);
  new Int16Array(buf, 44).set(data);
  return Bun.write(path, buf);
}

function asI64(xs: ArrayLike<number>) {
  return BigInt64Array.from(xs as ArrayLike<number>, (x) => BigInt(x));
}

async function runBertHidden(
  opt: InferOptions,
  prep: { inputIds: Int32Array },
  lang: LangCode,
  berts: BertLoader,
): Promise<Float32Array> {
  const sess = await berts.get(lang);
  const ids = asI64(prep.inputIds);
  const mask = new BigInt64Array(ids.length);
  mask.fill(1n);
  const out = await sess.run({
    input_ids: new ort.Tensor("int64", ids, [1, ids.length]),
    attention_mask: new ort.Tensor("int64", mask, [1, mask.length]),
  });
  const hiddenName = lang === "EN" ? (opt.debertaOutput ?? "hidden") : "hidden";
  const hiddenTensor = out[hiddenName] ?? out["last_hidden_state"] ?? Object.values(out)[0];
  if (!hiddenTensor) throw new Error(`no ${lang} bert hidden output`);
  const hData = hiddenTensor.data as Float32Array;
  const seq = prep.inputIds.length;
  const need = seq * opt.bertDim;
  if (hData.length !== need && hData.length !== 1 * need) {
    throw new Error(
      `${lang} BERT hidden length ${hData.length} != seq*bertDim ${need} (seq=${seq} bertDim=${opt.bertDim}).`,
    );
  }
  return hData.length === need ? hData : hData.slice(0, need);
}

async function inferMultilang(
  opt: InferOptions,
  engine: EngineExports,
  berts: BertLoader,
): Promise<{ phones: Int32Array; tones: Int32Array; language: Int32Array; packed: {
  nPhone: number; bertDim: number; emoDim: number; en: Float32Array; zh: Float32Array; ja: Float32Array; emo: Float32Array; zin: Float32Array;
} }> {
  const skipStart = opt.skipStart ?? false;
  const skipEnd = opt.skipEnd ?? false;
  const ml = prepareMultilang(engine, opt.text, skipStart, skipEnd);
  const nSeg = ml.segments.length;
  const enChunks: Float32Array[] = [];
  const zhChunks: Float32Array[] = [];
  const jaChunks: Float32Array[] = [];

  for (let idx = 0; idx < nSeg; idx++) {
    const seg = ml.segments[idx]!;
    const lang = LANG_BY_ID[seg.bertLang] ?? "EN";
    const segSkipStart = idx !== 0 || (skipStart && idx === 0);
    const segSkipEnd = idx !== nSeg - 1 || (skipEnd && idx === nSeg - 1);
    const hData = await runBertHidden(opt, seg, lang, berts);
    const packed = packBert(
      engine,
      hData,
      seg.inputIds.length,
      seg.word2ph,
      opt.bertDim,
      opt.emoDim,
      opt.seed + idx,
      opt.sdpNoiseScale,
      seg.bertLang,
    );
    const trimmed = trimBertSegment(packed, opt.bertDim, segSkipStart, segSkipEnd);
    enChunks.push(trimmed.en);
    zhChunks.push(trimmed.zh);
    jaChunks.push(trimmed.ja);
  }

  const nPhone = ml.phones.length;
  const { emo, zin } = packZinEmo(engine, nPhone, opt.emoDim, opt.seed, opt.sdpNoiseScale);
  return {
    phones: ml.phones,
    tones: ml.tones,
    language: ml.language,
    packed: {
      nPhone,
      bertDim: opt.bertDim,
      emoDim: opt.emoDim,
      en: concatF32(enChunks),
      zh: concatF32(zhChunks),
      ja: concatF32(jaChunks),
      emo,
      zin,
    },
  };
}

export async function infer(
  optIn: InferOptions,
): Promise<{ path: string; samples: number; sampleRate: number; multilang: boolean }> {
  const opt = { ...optIn };
  const enabled = new Set(opt.langs);

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;

  const engine = await loadEngine(opt.engine);
  const tagged = hasLangTags(engine, opt.text);
  if (tagged) {
    const filtered = filterTaggedText(opt.text, enabled);
    if (!filtered.text) {
      throw new Error(
        `no supported language segments (allow-list: ${opt.langs.join(",")})`,
      );
    }
    opt.text = filtered.text;
  } else if (!enabled.has(opt.lang)) {
    throw new Error(`untagged --lang ${opt.lang} is not in allow-list ${opt.langs.join(",")}`);
  }

  const useMultilang = opt.multilang || hasLangTags(engine, opt.text);
  const berts = createBertLoader(opt, enabled);

  let packed: ReturnType<typeof packBert>;
  let phones: Int32Array;
  let tones: Int32Array;
  let language: Int32Array;

  try {
    if (useMultilang) {
      const ml = await inferMultilang(opt, engine, berts);
      phones = ml.phones;
      tones = ml.tones;
      language = ml.language;
      packed = ml.packed;
    } else {
      const prep = prepare(engine, opt.text, opt.lang);
      const hData = await runBertHidden(opt, prep, opt.lang, berts);
      packed = packBert(
        engine,
        hData,
        prep.inputIds.length,
        prep.word2ph,
        opt.bertDim,
        opt.emoDim,
        opt.seed,
        opt.sdpNoiseScale,
        prep.bertLang,
      );
      phones = prep.phones;
      tones = prep.tones;
      language = prep.language;
    }

    const evil = await ort.InferenceSession.create(await Bun.file(opt.evil).arrayBuffer(), {
      executionProviders: ["wasm"],
    });
    try {
      const T = packed.nPhone;
      const D = packed.bertDim;
      const feeds: Record<string, ort.Tensor> = {
        x: new ort.Tensor("int64", asI64(phones), [1, T]),
        t: new ort.Tensor("int64", asI64(tones), [1, T]),
        language: new ort.Tensor("int64", asI64(language), [1, T]),
        bert_0: new ort.Tensor("float32", packed.zh, [T, D]),
        bert_1: new ort.Tensor("float32", packed.ja, [T, D]),
        bert_2: new ort.Tensor("float32", packed.en, [T, D]),
        emo: new ort.Tensor("float32", packed.emo, [packed.emoDim, 1]),
        sid: new ort.Tensor("int64", BigInt64Array.from([BigInt(opt.sid)]), [1]),
        zin: new ort.Tensor("float32", packed.zin, [1, 2, T]),
        length_scale: new ort.Tensor("float32", Float32Array.from([opt.lengthScale]), []),
        sdp_ratio: new ort.Tensor("float32", Float32Array.from([opt.sdpRatio]), []),
        noise_scale: new ort.Tensor("float32", Float32Array.from([opt.noiseScale]), []),
      };

      const audioOut = await evil.run(feeds);
      const outName = opt.evilOutput ?? "o";
      const o = audioOut[outName] ?? Object.values(audioOut)[0];
      if (!o) throw new Error("no acoustic output");
      const wav = o.data as Float32Array;
      await writeWav(opt.out, wav, SAMPLE_RATE);
      return { path: opt.out, samples: wav.length, sampleRate: SAMPLE_RATE, multilang: useMultilang };
    } finally {
      await evil.release();
    }
  } finally {
    await berts.release();
  }
}
async function main() {
  const opt = parseArgs(process.argv.slice(2));
  const r = await infer(opt);
  console.log(
    JSON.stringify(
      {
        out: r.path,
        samples: r.samples,
        sampleRate: r.sampleRate,
        lang: opt.lang,
        langs: opt.langs,
        multilang: r.multilang,
        evil: opt.evil,
        deberta: opt.deberta,
        zhBert: opt.zhBert,
        jpBert: opt.jpBert,
        bertDim: opt.bertDim,
        emoDim: opt.emoDim,
        sid: opt.sid,
        seed: opt.seed,
      },
      null,
      0,
    ),
  );
  console.log(`Wrote ${r.path} samples=${r.samples} sr=${r.sampleRate} multilang=${r.multilang}`);
}

if (import.meta.main) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
