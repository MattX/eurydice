import { StrictMode } from "react";
import { createRoot, hydrateRoot } from "react-dom/client";
import "katex/dist/katex.min.css";
import "remark-github-blockquote-alert/alert.css";
import SpecPage from "../../src/SpecPage";
import "../index.css";

const root = document.getElementById("root")!;
const page = (
  <StrictMode>
    <SpecPage />
  </StrictMode>
);

if (root.hasChildNodes()) {
  hydrateRoot(root, page);
} else {
  createRoot(root).render(page);
}
