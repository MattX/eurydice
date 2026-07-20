import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import mdx from "@mdx-js/rollup";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";
import { remarkAlert } from "remark-github-blockquote-alert";
import remarkMath from "remark-math";
import wasm from "vite-plugin-wasm";
import { VitePWA } from "vite-plugin-pwa";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    wasm(),
    {
      enforce: "pre",
      ...mdx({
        remarkPlugins: [remarkGfm, remarkAlert, remarkMath],
        rehypePlugins: [rehypeKatex, rehypeSlug],
      }),
    },
    react({ include: /\.(jsx|js|mdx|md|tsx|ts)$/ }),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["logo.png"],
      manifest: {
        name: "Eurydice Dice Probability Calculator",
        short_name: "Eurydice",
        description:
          "Calculate and visualize tabletop dice probabilities in your browser.",
        theme_color: "#f4f7fa",
        background_color: "#f4f7fa",
        display: "standalone",
        start_url: "/",
        scope: "/",
        icons: [
          {
            src: "/pwa-192x192.png",
            sizes: "192x192",
            type: "image/png",
          },
          {
            src: "/pwa-512x512.png",
            sizes: "512x512",
            type: "image/png",
          },
        ],
      },
      workbox: {
        globPatterns: ["**/*.{html,js,css,wasm,svg,ico}"],
        cleanupOutdatedCaches: true,
      },
    }),
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
