export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "—";
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"];
  let v = n;
  let i = -1;
  do {
    v /= 1024;
    i++;
  } while (v >= 1024 && i < units.length - 1);
  return `${v >= 100 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

export function formatMs(ms: number): string {
  if (!Number.isFinite(ms)) return "—";
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

type Listener = (loaded: number, total: number) => void;

const inflight = new Map<string, { promise: Promise<ArrayBuffer>; listeners: Set<Listener> }>();

export async function fetchWithProgress(url: string, onProgress?: Listener): Promise<ArrayBuffer> {
  const existing = inflight.get(url);
  if (existing) {
    if (onProgress) existing.listeners.add(onProgress);
    try {
      const buf = await existing.promise;
      onProgress?.(buf.byteLength, buf.byteLength);
      return buf;
    } finally {
      if (onProgress) existing.listeners.delete(onProgress);
    }
  }

  const listeners = new Set<Listener>();
  if (onProgress) listeners.add(onProgress);

  const promise = (async () => {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) throw new Error(`fetch ${url} failed: ${res.status} ${res.statusText}`);
    const total = Number(res.headers.get("content-length")) || 0;
    if (!res.body) {
      const buf = await res.arrayBuffer();
      for (const l of listeners) l(buf.byteLength, buf.byteLength);
      return buf;
    }
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      for (const l of listeners) l(loaded, total);
    }
    const out = new Uint8Array(loaded);
    let o = 0;
    for (const c of chunks) {
      out.set(c, o);
      o += c.byteLength;
    }
    for (const l of listeners) l(loaded, total || loaded);
    return out.buffer;
  })();

  inflight.set(url, { promise, listeners });
  try {
    return await promise;
  } finally {
    inflight.delete(url);
  }
}
