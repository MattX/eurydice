import React, { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "katex/dist/katex.min.css";
import "remark-github-blockquote-alert/alert.css";
import SpecPage from "../../src/SpecPage";
import "../index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <SpecPage />
  </StrictMode>,
);
