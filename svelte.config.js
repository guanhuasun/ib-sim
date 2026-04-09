import adapter from "@sveltejs/adapter-static";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

/** @type {import("@sveltejs/kit").Config} */
const config = {
  preprocess: vitePreprocess(),

  kit: {
    adapter: adapter({ fallback: "200.html" }),
    alias: {
      "$app.css": "src/app.css",
    },
    paths: {
      base: process.env.GITHUB_PAGES === "true" ? "/ib-sim" : "",
    },
  },
};

export default config;
