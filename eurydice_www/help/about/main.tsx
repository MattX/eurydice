import { StrictMode } from "react";
import { createRoot, hydrateRoot } from "react-dom/client";
import About from "../../src/About.mdx";
import "../index.css";

const root = document.getElementById("root")!;
const page = (
  <StrictMode>
    <About />
  </StrictMode>
);

if (root.hasChildNodes()) {
  hydrateRoot(root, page);
} else {
  createRoot(root).render(page);
}
