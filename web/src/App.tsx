import { useEffect, useMemo, useRef, useState } from "react";
import { infer } from "./lib/pipeline";
import { loadRuntime, probeAcoustic, type TtsRuntime } from "./lib/runtime";
import {
  DEFAULT_PARAMS,
  LANGS,
  PLACEHOLDER,
  SAMPLE_RATE,
  type InferStage,
  type LangCode,
  type Latency,
  type LoadItem,
  type Precision,
} from "./lib/types";
import { formatBytes, formatMs } from "./lib/fetch";
import { encodeWav } from "./lib/wav";

const STAGE_LABEL: Record<InferStage, string> = {
  prepare: "G2P / frontend",
  bert: "BERT",
  pack: "pack features",
  acoustic: "acoustic",
};

function pct(item: LoadItem): number {
  if (item.phase === "ready") return 100;
  if (item.phase === "compile") return 100;
  if (item.total > 0) return Math.min(100, (item.loaded / item.total) * 100);
  return item.loaded > 0 ? 50 : 0;
}

function Waveform({
  samples,
  progress,
  onSeek,
}: {
  samples: Float32Array;
  progress: number;
  onSeek: (ratio: number) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    const width = parent?.clientWidth ?? 800;
    const height = 96;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#100e0c";
    ctx.fillRect(0, 0, width, height);
    const buckets = Math.max(64, Math.floor(width));
    const step = samples.length / buckets;
    const mid = height / 2;
    ctx.fillStyle = "#ff5a47";
    let amp = 0;
    for (let i = 0; i < samples.length; i++) {
      const v = Math.abs(samples[i]!);
      if (v > amp) amp = v;
    }
    const scale = amp > 1e-6 ? height * 0.42 / amp : 0;
    for (let i = 0; i < buckets; i++) {
      const start = Math.floor(i * step);
      const end = Math.min(samples.length, Math.floor((i + 1) * step));
      let min = 0;
      let max = 0;
      for (let j = start; j < end; j++) {
        const v = samples[j]!;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const x = i;
      ctx.fillRect(x, mid - max * scale, 1, Math.max(1, (max - min) * scale));
    }
    if (progress > 0) {
      ctx.fillStyle = "rgba(143, 217, 180, 0.18)";
      ctx.fillRect(0, 0, width * Math.min(1, progress), height);
      ctx.fillStyle = "#8fd9b4";
      ctx.fillRect(width * Math.min(1, progress), 0, 1, height);
    }
  }, [samples, progress]);

  return (
    <div
      className="wave-wrap"
      onClick={(e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        onSeek((e.clientX - rect.left) / rect.width);
      }}
    >
      <canvas ref={ref} />
    </div>
  );
}

export function App() {
  const params = new URLSearchParams(window.location.search);
  const probeMode = params.get("probe") as Precision | null;
  const [precision, setPrecision] = useState<Precision>(
    params.get("precision") === "fp16" ? "fp16" : "int8",
  );
  const [lang, setLang] = useState<LangCode>("EN");
  const [text, setText] = useState(PLACEHOLDER.EN);
  const [loadItems, setLoadItems] = useState<Record<string, LoadItem>>({});
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  const [adapterName, setAdapterName] = useState("WebGPU");
  const [running, setRunning] = useState(false);
  const [stage, setStage] = useState<InferStage | null>(null);
  const [samples, setSamples] = useState<Float32Array | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [latency, setLatency] = useState<Latency | null>(null);
  const [playProgress, setPlayProgress] = useState(0);
  const [probeJson, setProbeJson] = useState<string | null>(null);
  const runtimeRef = useRef<TtsRuntime | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const loadGate = useRef(Promise.resolve());

  useEffect(() => {
    let cancelled = false;
    if (probeMode === "fp16" || probeMode === "int8") {
      setStatus("loading");
      setError(null);
      probeAcoustic(probeMode)
        .then((r) => {
          if (cancelled) return;
          setProbeJson(JSON.stringify(r, null, 2));
          setAdapterName(r.adapterName);
          setStatus("ready");
        })
        .catch((e) => {
          if (cancelled) return;
          setError(e instanceof Error ? e.message : String(e));
          setStatus("error");
        });
      return () => {
        cancelled = true;
      };
    }
    const task = loadGate.current.then(async () => {
      setStatus("loading");
      setError(null);
      try {
        const rt = await loadRuntime({
          precision,
          langs: [lang],
          previous: runtimeRef.current,
          onProgress: (items) => {
            if (!cancelled) setLoadItems(items);
          },
        });
        runtimeRef.current = rt;
        if (!cancelled) {
          setAdapterName(rt.adapterName);
          setStatus("ready");
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        if (!cancelled) {
          setError(msg);
          setStatus("error");
        }
      }
    });
    loadGate.current = task.then(
      () => undefined,
      () => undefined,
    );
    return () => {
      cancelled = true;
    };
  }, [precision, lang]);

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const items = useMemo(() => Object.values(loadItems), [loadItems]);
  const ready = status === "ready" && !running;

  async function onGenerate() {
    const rt = runtimeRef.current;
    if (!rt || running) return;
    setRunning(true);
    setError(null);
    setStage("prepare");
    try {
      const result = await infer(
        rt,
        { ...DEFAULT_PARAMS, text, lang },
        (next) => setLoadItems((prev) => ({ ...prev, ...next })),
        setStage,
      );
      const wav = encodeWav(result.samples, SAMPLE_RATE);
      const url = URL.createObjectURL(wav);
      setAudioUrl((old) => {
        if (old) URL.revokeObjectURL(old);
        return url;
      });
      setSamples(result.samples);
      setLatency(result.latency);
      setPlayProgress(0);
      requestAnimationFrame(() => {
        const el = audioRef.current;
        if (!el) return;
        void el.play();
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
      setStage(null);
    }
  }

  function switchLang(next: LangCode) {
    setLang(next);
    setText((cur) => {
      if (!cur.trim() || (LANGS as string[]).some((k) => PLACEHOLDER[k as LangCode] === cur)) {
        return PLACEHOLDER[next];
      }
      return cur;
    });
  }

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>
            evil <span>TTS</span>
          </h1>
          <p className="sub">Bert-VITS2 V220 · onnxruntime-web WebGPU only</p>
        </div>
        <div className="badge">
          <i className={`dot ${status === "error" ? "err" : status === "ready" ? "ok" : ""}`} />
          {status === "ready" ? adapterName : status === "error" ? "WebGPU error" : "loading"}
        </div>
      </header>

      {probeJson ? (
        <section className="panel">
          <h2>Acoustic probe</h2>
          <pre className="error" style={{ color: "var(--text)", fontFamily: "var(--mono)", fontSize: 12 }}>
            {probeJson}
          </pre>
        </section>
      ) : null}

      <section className="panel">
        <h2>Setup</h2>
        <div className="row">
          <label className="field">
            <span>Precision</span>
            <div className="seg">
              {(["int8", "fp16"] as const).map((p) => (
                <button key={p} className={precision === p ? "on" : ""} onClick={() => setPrecision(p)} type="button">
                  {p}
                </button>
              ))}
            </div>
          </label>
          <label className="field">
            <span>Language</span>
            <div className="seg">
              {LANGS.map((l) => (
                <button key={l} className={lang === l ? "on" : ""} onClick={() => switchLang(l)} type="button">
                  {l.toLowerCase()}
                </button>
              ))}
            </div>
          </label>
        </div>
      </section>

      <section className="panel">
        <h2>Model loading</h2>
        {items.length === 0 ? <p className="hint">Waiting…</p> : null}
        {items.map((item) => (
          <div className="item" key={item.id}>
            <div className="name">{item.label}</div>
            <div className={`bar ${item.phase}`}>
              <i style={item.phase === "compile" ? undefined : { width: `${pct(item)}%` }} />
            </div>
            <div className="meta">
              {item.phase === "compile"
                ? "compile"
                : item.phase === "ready"
                  ? "ready"
                  : item.phase === "error"
                    ? "error"
                    : `${formatBytes(item.loaded)}${item.total ? ` / ${formatBytes(item.total)}` : ""}`}
            </div>
          </div>
        ))}
      </section>

      <section className="panel">
        <h2>Text</h2>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={PLACEHOLDER[lang]}
          spellCheck={false}
        />
        <div className="actions">
          <button className="primary" type="button" disabled={!ready || !text.trim()} onClick={() => void onGenerate()}>
            {running ? `Computing · ${stage ? STAGE_LABEL[stage] : "…"}` : "Synthesize"}
          </button>
          <span className="hint">Untagged text uses {lang}. Tags like [ZH]…[EN]… load extra BERTs.</span>
        </div>
        {error ? <p className="error">{error}</p> : null}
      </section>

      {latency || samples ? (
        <section className="panel">
          <h2>Audio</h2>
          {latency ? (
            <div className="metrics">
              <div className="metric">
                <b>{formatMs(latency.totalMs)}</b>
                <span>total latency</span>
              </div>
              <div className="metric">
                <b>{formatMs(latency.bertMs)}</b>
                <span>BERT</span>
              </div>
              <div className="metric">
                <b>{formatMs(latency.acousticMs)}</b>
                <span>acoustic</span>
              </div>
              <div className="metric">
                <b>{formatMs(latency.prepareMs)}</b>
                <span>G2P</span>
              </div>
              <div className="metric">
                <b>{latency.audioSec.toFixed(2)} s</b>
                <span>audio</span>
              </div>
              <div className="metric">
                <b>{latency.rtf.toFixed(2)}×</b>
                <span>RTF {latency.multilang ? "· multilang" : ""}</span>
              </div>
            </div>
          ) : null}
          {samples ? (
            <Waveform
              samples={samples}
              progress={playProgress}
              onSeek={(ratio) => {
                const el = audioRef.current;
                if (!el || !el.duration) return;
                el.currentTime = ratio * el.duration;
              }}
            />
          ) : null}
          <div className="player">
            {audioUrl ? (
              <>
                <audio
                  ref={audioRef}
                  src={audioUrl}
                  controls
                  onTimeUpdate={(e) => {
                    const el = e.currentTarget;
                    setPlayProgress(el.duration ? el.currentTime / el.duration : 0);
                  }}
                />
                <a className="link" href={audioUrl} download={`evil-${lang.toLowerCase()}.wav`}>
                  download wav
                </a>
              </>
            ) : (
              <span className="hint">No audio yet</span>
            )}
          </div>
        </section>
      ) : null}
    </div>
  );
}
