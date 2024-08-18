import React, { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import About from "../src/About.mdx";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <About />
  </StrictMode>,
);
