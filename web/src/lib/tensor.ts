import * as ort from "onnxruntime-web/webgpu";

function f16toF32(u16: number): number {
  const s = (u16 & 0x8000) >> 15;
  const e = (u16 & 0x7c00) >> 10;
  const f = u16 & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s ? -Infinity : Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

export function f32ToF16Bits(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  const dv = new DataView(new ArrayBuffer(4));
  for (let i = 0; i < src.length; i++) {
    dv.setFloat32(0, src[i]!);
    const x = dv.getUint32(0);
    const sign = (x >>> 16) & 0x8000;
    const exp = (x >>> 23) & 0xff;
    const frac = x & 0x7fffff;
    if (exp === 255) {
      out[i] = sign | 0x7c00 | (frac ? 0x200 : 0);
      continue;
    }
    const e = exp - 127 + 15;
    if (e <= 0) {
      if (e < -10) {
        out[i] = sign;
        continue;
      }
      const m = (frac | 0x800000) >> (1 - e);
      out[i] = sign | ((m + 0x1000) >> 13);
      continue;
    }
    if (e >= 31) {
      out[i] = sign | 0x7c00;
      continue;
    }
    out[i] = sign | (e << 10) | ((frac + 0x1000) >> 13);
  }
  return out;
}

function f16BitsToF32(data: Uint16Array): Float32Array {
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = f16toF32(data[i]!);
  return out;
}

function isFloat16Array(data: unknown): data is ArrayLike<number> {
  const ctor = (globalThis as { Float16Array?: new () => ArrayLike<number> }).Float16Array;
  return typeof ctor === "function" && data instanceof ctor;
}

export async function tensorToF32(t: ort.Tensor): Promise<Float32Array> {
  const raw =
    t.location === "gpu-buffer" || t.location === "ml-tensor" || t.location === "texture"
      ? await t.getData()
      : t.data;

  if (t.type === "float16") {
    if (isFloat16Array(raw)) return Float32Array.from(raw);
    if (raw instanceof Uint16Array) return f16BitsToF32(raw);
    throw new Error("float16 tensor data is not Uint16Array");
  }
  if (raw instanceof Float32Array) return raw;
  if (isFloat16Array(raw)) return Float32Array.from(raw);
  if (raw instanceof Uint16Array) return f16BitsToF32(raw);
  if (ArrayBuffer.isView(raw) && !(raw instanceof BigInt64Array) && !(raw instanceof BigUint64Array)) {
    return Float32Array.from(raw as ArrayLike<number>);
  }
  throw new Error(`cannot convert tensor type ${t.type} (${t.location}) to float32`);
}

export function floatFeed(
  sess: ort.InferenceSession,
  name: string,
  data: Float32Array,
  dims: number[],
): ort.Tensor {
  const meta = sess.inputMetadata.find((m) => m.name === name);
  const ty = meta && meta.isTensor ? meta.type : "float32";
  if (ty === "float16") return new ort.Tensor("float16", f32ToF16Bits(data), dims);
  return new ort.Tensor("float32", data, dims);
}

export function asI64(xs: ArrayLike<number>): BigInt64Array {
  return BigInt64Array.from(xs as ArrayLike<number>, (x) => BigInt(x));
}

export function peakAbs(xs: Float32Array): number {
  let p = 0;
  for (let i = 0; i < xs.length; i++) {
    const v = Math.abs(xs[i]!);
    if (v > p) p = v;
  }
  return p;
}
