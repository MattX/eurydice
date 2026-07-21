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

interface HelpRenderer {
  routes: string[];
  render: (route: string) => string;
}

function prerenderHelp() {
  const renderedRoutes = new Set<string>();
  let renderer: Promise<HelpRenderer> | undefined;
  let projectRoot = "";

  const loadRenderer = () => {
    const rendererUrl = `file://${projectRoot}/dist-ssr/entry-server.js`;
    renderer ??= import(rendererUrl) as Promise<HelpRenderer>;
    return renderer;
  };

  return {
    name: "prerender-help",
    apply: "build" as const,
    configResolved(config: { root: string }) {
      projectRoot = config.root;
    },
    async transformIndexHtml(html: string, context: { path: string }) {
      if (!context.path.startsWith("/help/")) return;

      const route = context.path.replace(/index\.html$/, "");
      const { routes, render } = await loadRenderer();
      if (!routes.includes(route)) {
        throw new Error(`No pre-renderer registered for ${route}`);
      }

      const markup = render(route);
      if (!markup.trim()) {
        throw new Error(`Pre-rendering ${route} produced empty markup`);
      }

      const rootPattern = /(<div\b[^>]*\bid=["']root["'][^>]*>)([\s\S]*?)(<\/div>)/g;
      const roots = Array.from(html.matchAll(rootPattern));
      if (roots.length !== 1) {
        throw new Error(
          `Expected exactly one root placeholder for ${route}, found ${roots.length}`,
        );
      }
      if (roots[0][2].trim()) {
        throw new Error(`Root placeholder for ${route} is not empty`);
      }

      renderedRoutes.add(route);
      return html.replace(rootPattern, (_match, open, _contents, close) =>
        `${open}${markup}${close}`,
      );
    },
    async closeBundle() {
      const { routes } = await loadRenderer();
      const missingRoutes = routes.filter((route) => !renderedRoutes.has(route));
      if (missingRoutes.length > 0) {
        throw new Error(
          `No output HTML was generated for: ${missingRoutes.join(", ")}`,
        );
      }
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig(({ isSsrBuild }) => ({
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
    ...(!isSsrBuild
      ? [
          prerenderHelp(),
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
        ]
      : []),
  ],
  assetsInclude: ["src/assets/**"],
  worker: {
    plugins: () => [wasm()],
    format: "es",
  },
  build: {
    target: "esnext",
    ...(isSsrBuild
      ? {
          outDir: "dist-ssr",
          rollupOptions: {
            input: "help/entry-server.tsx",
          },
        }
      : {
          rollupOptions: {
            input: {
              main: "index.html",
              about: "help/about/index.html",
              spec: "help/spec/index.html",
            },
          },
        }),
  },
}));
