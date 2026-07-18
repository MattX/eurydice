import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import mdx from "@mdx-js/rollup";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";
import { remarkAlert } from "remark-github-blockquote-alert";
import wasm from "vite-plugin-wasm";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    wasm(),
    {
      enforce: "pre",
      ...mdx({
        remarkPlugins: [remarkGfm, remarkAlert],
        rehypePlugins: [rehypeSlug],
      }),
    },
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
        spec: "help/spec/index.html",
      },
    },
  },
});
