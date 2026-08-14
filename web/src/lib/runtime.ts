import * as ort from "onnxruntime-web/webgpu";
import { instantiateEngine, type EngineExports } from "./engine";
import { fetchWithProgress } from "./fetch";
import { bertLabel, bertUrl, ENGINE_URL, EVIL_URL } from "./models";
import type { LangCode, LoadItem, Precision } from "./types";

const WEBGPU_EP = [{ name: "webgpu" as const }];
const ORT_WASM = `${import.meta.env.BASE_URL}ort/ort-wasm-simd-threaded.asyncify.wasm`;
const ORT_MJS = `${import.meta.env.BASE_URL}ort/ort-wasm-simd-threaded.asyncify.mjs`;

export type TtsRuntime = {
  precision: Precision;
  adapterName: string;
  engine: EngineExports;
  evil: ort.InferenceSession;
  berts: Map<LangCode, ort.InferenceSession>;
};

type GpuInfo = { adapterName: string; shaderF16: boolean };

let ortReady: Promise<void> | null = null;
let engineReady: Promise<EngineExports> | null = null;
let gpuReady: Promise<GpuInfo> | null = null;
let sessionCreate: Promise<void> = Promise.resolve();

function patchItem(
  items: Record<string, LoadItem>,
  onProgress: (items: Record<string, LoadItem>) => void,
  id: string,
  patch: Partial<LoadItem>,
) {
  const prev = items[id];
  items[id] = {
    id,
    label: prev?.label ?? id,
    loaded: prev?.loaded ?? 0,
    total: prev?.total ?? 0,
    phase: prev?.phase ?? "fetch",
    ...patch,
  };
  onProgress({ ...items });
}

async function ensureWebGpu(): Promise<GpuInfo> {
  if (!gpuReady) {
    gpuReady = (async () => {
      if (!navigator.gpu) {
        throw new Error("WebGPU is not available. Use Chrome or Edge 113+.");
      }
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("No WebGPU adapter. GPU access may be blocked.");
      const shaderF16 = adapter.features.has("shader-f16");
      const info = adapter.info;
      const name =
        info?.description?.trim() ||
        [info?.vendor, info?.architecture].filter((s) => s?.trim()).join(" ") ||
        "WebGPU";
      return { adapterName: shaderF16 ? `${name} · f16` : name, shaderF16 };
    })();
  }
  return gpuReady;
}

async function ensureOrt(items: Record<string, LoadItem>, onProgress: (items: Record<string, LoadItem>) => void) {
  if (!ortReady) {
    ortReady = (async () => {
      patchItem(items, onProgress, "ort", {
        id: "ort",
        label: "ONNX Runtime (WebGPU)",
        phase: "fetch",
        loaded: 0,
        total: 0,
      });
      const wasm = await fetchWithProgress(ORT_WASM, (loaded, total) => {
        patchItem(items, onProgress, "ort", { loaded, total, phase: "fetch" });
      });
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
      ort.env.wasm.proxy = false;
      ort.env.wasm.wasmPaths = { wasm: ORT_WASM, mjs: ORT_MJS };
      ort.env.wasm.wasmBinary = wasm;
      ort.env.webgpu.powerPreference = "high-performance";
      patchItem(items, onProgress, "ort", { phase: "ready", loaded: wasm.byteLength, total: wasm.byteLength });
    })().catch((err) => {
      ortReady = null;
      throw err;
    });
  } else {
    patchItem(items, onProgress, "ort", {
      id: "ort",
      label: "ONNX Runtime (WebGPU)",
      phase: "ready",
    });
  }
  await ortReady;
}

async function ensureEngine(items: Record<string, LoadItem>, onProgress: (items: Record<string, LoadItem>) => void) {
  if (!engineReady) {
    engineReady = (async () => {
      patchItem(items, onProgress, "engine", {
        id: "engine",
        label: "engine.wasm (G2P)",
        phase: "fetch",
        loaded: 0,
        total: 0,
      });
      const bytes = await fetchWithProgress(ENGINE_URL, (loaded, total) => {
        patchItem(items, onProgress, "engine", { loaded, total, phase: "fetch" });
      });
      patchItem(items, onProgress, "engine", { phase: "compile", loaded: bytes.byteLength, total: bytes.byteLength });
      const engine = await instantiateEngine(bytes);
      patchItem(items, onProgress, "engine", { phase: "ready" });
      return engine;
    })().catch((err) => {
      engineReady = null;
      throw err;
    });
  } else {
    patchItem(items, onProgress, "engine", {
      id: "engine",
      label: "engine.wasm (G2P)",
      phase: "ready",
    });
  }
  return engineReady;
}

async function createSession(buffer: ArrayBuffer, precision: Precision): Promise<ort.InferenceSession> {
  const run = sessionCreate.then(() =>
    ort.InferenceSession.create(buffer, {
      executionProviders: WEBGPU_EP,
      graphOptimizationLevel: precision === "fp16" ? "disabled" : "all",
    }),
  );
  sessionCreate = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

async function loadOnnxSession(
  items: Record<string, LoadItem>,
  onProgress: (items: Record<string, LoadItem>) => void,
  id: string,
  label: string,
  url: string,
  precision: Precision,
): Promise<ort.InferenceSession> {
  patchItem(items, onProgress, id, { id, label, phase: "fetch", loaded: 0, total: 0 });
  const buf = await fetchWithProgress(url, (loaded, total) => {
    patchItem(items, onProgress, id, { loaded, total, phase: "fetch" });
  });
  patchItem(items, onProgress, id, { phase: "compile", loaded: buf.byteLength, total: buf.byteLength });
  try {
    const sess = await createSession(buf, precision);
    patchItem(items, onProgress, id, { phase: "ready" });
    return sess;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    patchItem(items, onProgress, id, { phase: "error", error: msg });
    throw e;
  }
}

export async function loadRuntime(opts: {
  precision: Precision;
  langs: LangCode[];
  previous?: TtsRuntime | null;
  onProgress: (items: Record<string, LoadItem>) => void;
}): Promise<TtsRuntime> {
  const items: Record<string, LoadItem> = {};
  const onProgress = opts.onProgress;

  const gpu = await ensureWebGpu();
  if (opts.precision === "fp16" && !gpu.shaderF16) {
    throw new Error("FP16 models need WebGPU shader-f16, which this GPU/browser does not enable.");
  }
  const [engine] = await Promise.all([ensureEngine(items, onProgress), ensureOrt(items, onProgress)]);

  const prev = opts.previous;
  const reusePrecision = prev && prev.precision === opts.precision;

  let evil = prev?.evil ?? null;
  if (!evil) {
    // Same FP32 graph on every platform / precision; BERT is what fp16 switches.
    evil = await loadOnnxSession(items, onProgress, "evil", "acoustic", EVIL_URL, "int8");
  } else {
    patchItem(items, onProgress, "evil", {
      id: "evil",
      label: "acoustic",
      phase: "ready",
    });
  }

  const berts = new Map<LangCode, ort.InferenceSession>();
  if (reusePrecision) {
    for (const [lang, sess] of prev.berts) {
      if (opts.langs.includes(lang)) berts.set(lang, sess);
      else {
        try {
          await sess.release();
        } catch {
          /* ignore */
        }
      }
    }
  } else if (prev) {
    for (const sess of prev.berts.values()) {
      try {
        await sess.release();
      } catch {
        /* ignore */
      }
    }
  }

  for (const lang of opts.langs) {
    const hit = berts.get(lang);
    if (hit) {
      patchItem(items, onProgress, `bert-${lang}`, {
        id: `bert-${lang}`,
        label: `${bertLabel(lang)} (${opts.precision})`,
        phase: "ready",
      });
      continue;
    }
    const sess = await loadOnnxSession(
      items,
      onProgress,
      `bert-${lang}`,
      `${bertLabel(lang)} (${opts.precision})`,
      bertUrl(opts.precision, lang),
      opts.precision,
    );
    berts.set(lang, sess);
  }

  return { precision: opts.precision, adapterName: gpu.adapterName, engine, evil, berts };
}

export async function ensureBert(
  runtime: TtsRuntime,
  lang: LangCode,
  onProgress: (items: Record<string, LoadItem>) => void,
): Promise<ort.InferenceSession> {
  const hit = runtime.berts.get(lang);
  if (hit) return hit;
  const items: Record<string, LoadItem> = {};
  const sess = await loadOnnxSession(
    items,
    onProgress,
    `bert-${lang}`,
    `${bertLabel(lang)} (${runtime.precision})`,
    bertUrl(runtime.precision, lang),
    runtime.precision,
  );
  runtime.berts.set(lang, sess);
  return sess;
}

export async function probeAcoustic(precision: Precision) {
  const gpu = await ensureWebGpu();
  await ensureOrt({}, () => undefined);
  const buf = await fetchWithProgress(EVIL_URL);
  const sess = await createSession(buf, "int8");
  const T = 16;
  const D = 1024;
  const fill = (n: number, v: number) => {
    const a = new Float32Array(n);
    a.fill(v);
    return a;
  };
  const { asI64, peakAbs, tensorToF32 } = await import("./tensor");
  const feeds: Record<string, ort.Tensor> = {
    x: new ort.Tensor("int64", asI64(new Int32Array(T).fill(10)), [1, T]),
    t: new ort.Tensor("int64", asI64(new Int32Array(T).fill(1)), [1, T]),
    language: new ort.Tensor("int64", asI64(new Int32Array(T).fill(2)), [1, T]),
    bert_0: new ort.Tensor("float32", fill(T * D, 0.05), [T, D]),
    bert_1: new ort.Tensor("float32", fill(T * D, 0.05), [T, D]),
    bert_2: new ort.Tensor("float32", fill(T * D, 0.05), [T, D]),
    emo: new ort.Tensor("float32", new Float32Array(512), [512, 1]),
    sid: new ort.Tensor("int64", BigInt64Array.from([0n]), [1]),
    zin: new ort.Tensor("float32", fill(2 * T, 0.4), [1, 2, T]),
    length_scale: new ort.Tensor("float32", Float32Array.from([1]), []),
    sdp_ratio: new ort.Tensor("float32", Float32Array.from([0.2]), []),
    noise_scale: new ort.Tensor("float32", Float32Array.from([0.6]), []),
  };
  const t0 = performance.now();
  const out = await sess.run(feeds);
  const runMs = performance.now() - t0;
  const o = out["o"] ?? Object.values(out)[0];
  if (!o) throw new Error("no acoustic output");
  const samples = await tensorToF32(o);
  let nan = 0;
  let inf = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i]!;
    if (Number.isNaN(v)) nan++;
    else if (!Number.isFinite(v)) inf++;
  }
  const result = {
    precision,
    shaderF16: gpu.shaderF16,
    adapterName: gpu.adapterName,
    outType: o.type,
    outLocation: o.location,
    length: samples.length,
    peak: peakAbs(samples),
    nan,
    inf,
    sample0: samples[0] ?? null,
    sample1: samples[1] ?? null,
    runMs,
    inTypes: sess.inputMetadata.map((m) => (m.isTensor ? `${m.name}:${m.type}` : m.name)),
    outTypes: sess.outputMetadata.map((m) => (m.isTensor ? `${m.name}:${m.type}` : m.name)),
  };
  await sess.release();
  return result;
}
