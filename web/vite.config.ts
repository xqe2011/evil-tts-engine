import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import basicSsl from "@vitejs/plugin-basic-ssl";
import react from "@vitejs/plugin-react";
import type { Connect, Plugin, PreviewServer, ViteDevServer } from "vite";
import { defineConfig } from "vite";

const webRoot = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(webRoot, "..");

function safeFile(dir: string, rel: string): string | null {
  if (!rel || rel.includes("\0") || path.isAbsolute(rel)) return null;
  const resolved = path.resolve(dir, rel);
  const root = path.resolve(dir);
  if (resolved !== root && !resolved.startsWith(root + path.sep)) return null;
  return resolved;
}

function contentType(file: string): string {
  if (file.endsWith(".mjs") || file.endsWith(".js")) return "text/javascript";
  if (file.endsWith(".wasm")) return "application/wasm";
  return "application/octet-stream";
}

const ORT_DIST = path.join(webRoot, "node_modules/onnxruntime-web/dist");
const ORT_FILES = ["ort-wasm-simd-threaded.asyncify.wasm", "ort-wasm-simd-threaded.asyncify.mjs"];

function serveRepoAssets(): Plugin {
  const mounts: Array<{ prefix: string; dir: string }> = [
    { prefix: "/models/", dir: path.join(repoRoot, "models") },
    { prefix: "/engine/", dir: path.join(repoRoot, "engine") },
    { prefix: "/ort/", dir: ORT_DIST },
  ];

  const middleware: Connect.NextHandleFunction = (req, res, next) => {
    const raw = req.url?.split("?")[0] ?? "";
    const mount = mounts.find((m) => raw.startsWith(m.prefix));
    if (!mount) {
      next();
      return;
    }
    const rel = decodeURIComponent(raw.slice(mount.prefix.length));
    const file = safeFile(mount.dir, rel);
    if (!file || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
      res.statusCode = 404;
      res.end("not found");
      return;
    }
    const stat = fs.statSync(file);
    res.setHeader("Content-Type", contentType(file));
    res.setHeader("Content-Length", String(stat.size));
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("ETag", `"${stat.mtimeMs.toFixed(0)}-${stat.size}"`);
    res.setHeader("Cross-Origin-Resource-Policy", "same-origin");
    res.setHeader("Accept-Ranges", "none");
    if (req.method === "HEAD") {
      res.end();
      return;
    }
    const stream = fs.createReadStream(file);
    stream.on("error", () => {
      if (!res.headersSent) res.statusCode = 500;
      res.end();
    });
    stream.pipe(res);
  };

  const attach = (server: ViteDevServer | PreviewServer) => {
    server.middlewares.use(middleware);
  };

  return {
    name: "serve-repo-assets",
    configureServer: attach,
    configurePreviewServer: attach,
    async writeBundle(options) {
      const dest = path.join(options.dir ?? path.join(webRoot, "dist"), "ort");
      fs.mkdirSync(dest, { recursive: true });
      for (const name of ORT_FILES) {
        fs.copyFileSync(path.join(ORT_DIST, name), path.join(dest, name));
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), basicSsl(), serveRepoAssets()],
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  build: {
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 30_000,
  },
  server: {
    host: true,
    port: 5173,
    // Empty proxy forces HTTPS over HTTP/1.1 so Bun can serve the cert.
    proxy: {},
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
