import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import mdx from "@mdx-js/rollup";
import wasm from "vite-plugin-wasm";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    wasm(),
    { enforce: "pre", ...mdx() },
    react({ include: /\.(jsx|js|mdx|md|tsx|ts)$/ }),
  ],
  assetsInclude: ["src/assets/**"],
  worker: {
    plugins: () => [wasm()],
    format: "es",
  },
  build: {
    target: "esnext",
    rollupOptions: {
      input: {
        main: "index.html",
        about: "help/about/index.html",
      },
    },
  },
});
