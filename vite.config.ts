import path from "node:path";
import { sveltekit } from "@sveltejs/kit/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [sveltekit(), tailwindcss()],
  resolve: {
    alias: {
      // Resolve from jax-js source (tsdown bundle is broken)
      "@jax-js/jax": path.resolve(__dirname, "jax-js/src/index.ts"),
    },
  },
  optimizeDeps: {
    exclude: ["@rollup/browser", "onnxruntime-web"],
  },
  build: {
    chunkSizeWarningLimit: 4000,
  },
  server: {
    fs: {
      allow: [path.resolve(__dirname, "jax-js")],
    },
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
