import * as ort from "onnxruntime-web/wasm";

/**
 * Bun TTS: engine.wasm (text/G2P) + ORT wasm (Deberta + acoustic).
 * All model/runtime knobs are inputs — swap evil/Deberta/sr without code edits.
 *
 * Examples:
 *   bun infer.ts --text "Hello" --out out.wav
 *   bun infer.ts --config target/tts/config.json --evil path/to/new.onnx --deberta path/to/bert.onnx
 *   bun infer.ts --sample-rate 22050 --bert-dim 768 --emo-dim 512 --sid 0
 */

const ROOT = import.meta.dir;

export type InferOptions = {
  text: string;
  out: string;
  /** Acoustic / voice ONNX (merged evil graph). */
  evil: string;
  /** Deberta (or other EN BERT) ONNX. */
  deberta: string;
  /** Frontend WASM (SPM/cmudict/G2P). */
  engine: string;
  /** Optional Bert-VITS2-style config.json — fills sampleRate if unset. */
  config?: string;
  sampleRate: number;
  /** Hidden size of Deberta features (1024 large / 768 base). */
  bertDim: number;
  /** Emotion vector size expected by acoustic model. */
  emoDim: number;
  sid: number;
  seed: number;
  lengthScale: number;
  sdpRatio: number;
  /** Posterior / seq noise scale (noise_scale). */
  noiseScale: number;
  /** SDP zin scale (sdp_noise_scale). */
  sdpNoiseScale: number;
  /** Deberta output name if not "hidden". */
  debertaOutput?: string;
  /** Acoustic output name if not "o". */
  evilOutput?: string;
};

const DEFAULTS: InferOptions = {
  text: "Hello, I am evil, an assistant by apple banana.",
  out: `${ROOT}/examples/en.wav`,
  evil: `${ROOT}/models/evil_v220.onnx`,
  deberta: `${ROOT}/models/deberta_v3_large_hs.int8.onnx`,
  engine: `${ROOT}/engine/engine.wasm`,
  config: `${ROOT}/models/config.json`,
  sampleRate: 44100,
  bertDim: 1024,
  emoDim: 512,
  sid: 0,
  seed: 42,
  lengthScale: 1.0,
  sdpRatio: 0.2,
  noiseScale: 0.6,
  sdpNoiseScale: 0.8,
};

type EngineExports = {
  memory: WebAssembly.Memory;
  engine_alloc(n: number): number;
  engine_alloc_free(ptr: number, n: number): void;
  engine_prepare(textPtr: number, textLen: number, outLenPtr: number): number;
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
      case "--evil":
        opt.evil = take(i++);
        break;
      case "--deberta":
        opt.deberta = take(i++);
        break;
      case "--engine":
        opt.engine = take(i++);
        break;
      case "--config":
        opt.config = take(i++);
        break;
      case "--sample-rate":
      case "--sr":
        opt.sampleRate = Number(take(i++));
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
      case "--help":
      case "-h":
        console.log(`Usage: bun infer.ts [options]
  --text STR
  --out PATH
  --evil PATH           acoustic ONNX
  --deberta PATH        BERT ONNX
  --engine PATH         frontend wasm
  --config PATH         optional config.json (reads data.sampling_rate)
  --sample-rate N       wav sample rate (default 44100)
  --bert-dim N          Deberta hidden size (default 1024)
  --emo-dim N           emotion vector size (default 512)
  --sid N --seed N
  --length-scale F --sdp-ratio F --noise-scale F --sdp-noise-scale F
  --deberta-output NAME --evil-output NAME`);
        process.exit(0);
      default:
        if (a.startsWith("-")) throw new Error(`unknown flag ${a}`);
        // positional: text [seed] [out] (compat)
        if (opt.text === DEFAULTS.text) opt.text = a;
        else if (opt.seed === DEFAULTS.seed && !Number.isNaN(Number(a))) opt.seed = Number(a);
        else opt.out = a;
    }
  }
  return opt;
}

async function applyConfigFile(opt: InferOptions): Promise<InferOptions> {
  if (!opt.config) return opt;
  const raw = await Bun.file(opt.config).text();
  const cfg = JSON.parse(raw) as {
    data?: { sampling_rate?: number; spk2id?: Record<string, number> };
  };
  const next = { ...opt };
  // Only fill sampleRate from config when still at default *and* user didn't pass --sample-rate.
  // Simpler rule: config wins for sampleRate if present, unless CLI already changed from default
  // via explicit flag tracking — here: if config has sampling_rate, use it when --sample-rate
  // was not in argv.
  const argv = process.argv.slice(2);
  const srFlag = argv.includes("--sample-rate") || argv.includes("--sr");
  if (!srFlag && cfg.data?.sampling_rate) {
    next.sampleRate = cfg.data.sampling_rate;
  }
  return next;
}

async function loadEngine(path: string): Promise<EngineExports> {
  const bytes = await Bun.file(path).arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports as unknown as EngineExports;
}

function prepare(engine: EngineExports, text: string) {
  const enc = new TextEncoder();
  const utf8 = enc.encode(text);
  const tPtr = engine.engine_alloc(utf8.length);
  new Uint8Array(engine.memory.buffer, tPtr, utf8.length).set(utf8);
  const lenPtr = engine.engine_alloc(4);
  const blobPtr = engine.engine_prepare(tPtr, utf8.length, lenPtr);
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
  let o = 16;
  const inputIds = i32s(view, o, nIds);
  o += nIds * 4;
  const phones = i32s(view, o, nPh);
  o += nPh * 4;
  const tones = i32s(view, o, nPh);
  o += nPh * 4;
  const language = i32s(view, o, nPh);
  o += nPh * 4;
  const word2ph = i32s(view, o, nW2);
  return { inputIds, phones, tones, language, word2ph };
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
) {
  const hPtr = engine.engine_alloc_f32(hidden.length);
  new Float32Array(engine.memory.buffer, hPtr, hidden.length).set(hidden);
  const wPtr = engine.engine_alloc(word2ph.length * 4);
  new Int32Array(engine.memory.buffer, wPtr, word2ph.length).set(word2ph);
  const lenPtr = engine.engine_alloc(4);
  const blobPtr = engine.engine_pack_bert(
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

export async function infer(optIn: InferOptions): Promise<{ path: string; samples: number; sampleRate: number }> {
  const opt = await applyConfigFile(optIn);

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;

  const engine = await loadEngine(opt.engine);
  const prep = prepare(engine, opt.text);

  const deberta = await ort.InferenceSession.create(await Bun.file(opt.deberta).arrayBuffer(), {
    executionProviders: ["wasm"],
  });
  const ids = asI64(prep.inputIds);
  const mask = new BigInt64Array(ids.length);
  mask.fill(1n);
  const bertOut = await deberta.run({
    input_ids: new ort.Tensor("int64", ids, [1, ids.length]),
    attention_mask: new ort.Tensor("int64", mask, [1, mask.length]),
  });
  const hiddenName = opt.debertaOutput ?? "hidden";
  const hiddenTensor = bertOut[hiddenName] ?? Object.values(bertOut)[0];
  if (!hiddenTensor) throw new Error("no deberta hidden output");
  const hData = hiddenTensor.data as Float32Array;
  const seq = prep.inputIds.length;
  const need = seq * opt.bertDim;
  if (hData.length !== need && hData.length !== 1 * need) {
    throw new Error(
      `Deberta hidden length ${hData.length} != seq*bertDim ${need} (seq=${seq} bertDim=${opt.bertDim}). Pass --bert-dim.`,
    );
  }

  const packed = packBert(
    engine,
    hData,
    seq,
    prep.word2ph,
    opt.bertDim,
    opt.emoDim,
    opt.seed,
    opt.sdpNoiseScale,
  );

  const evil = await ort.InferenceSession.create(await Bun.file(opt.evil).arrayBuffer(), {
    executionProviders: ["wasm"],
  });
  const T = packed.nPhone;
  const D = packed.bertDim;
  const feeds: Record<string, ort.Tensor> = {
    x: new ort.Tensor("int64", asI64(prep.phones), [1, T]),
    t: new ort.Tensor("int64", asI64(prep.tones), [1, T]),
    language: new ort.Tensor("int64", asI64(prep.language), [1, T]),
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
  await writeWav(opt.out, wav, opt.sampleRate);
  return { path: opt.out, samples: wav.length, sampleRate: opt.sampleRate };
}

async function main() {
  const opt = parseArgs(process.argv.slice(2));
  const r = await infer(opt);
  let max = 0;
  // re-read not needed; log from file size
  console.log(
    JSON.stringify(
      {
        out: r.path,
        samples: r.samples,
        sampleRate: r.sampleRate,
        evil: opt.evil,
        deberta: opt.deberta,
        bertDim: opt.bertDim,
        emoDim: opt.emoDim,
        sid: opt.sid,
        seed: opt.seed,
      },
      null,
      0,
    ),
  );
  console.log(`Wrote ${r.path} samples=${r.samples} sr=${r.sampleRate}`);
}

if (import.meta.main) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
