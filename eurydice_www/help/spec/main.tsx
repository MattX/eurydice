import React, { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "remark-github-blockquote-alert/alert.css";
import Spec from "../../../spec.md";
import Header from "../../src/components/Header";
import "../index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Header />
    <main className="document-content prose dark:prose-invert mx-4 max-w-none pb-8 md:mx-auto md:max-w-4xl">
      <Spec />
    </main>
  </StrictMode>,
);
