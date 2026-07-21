import { type ComponentType, StrictMode } from "react";
import { renderToString } from "react-dom/server";
import About from "../src/About.mdx";
import SpecPage from "../src/SpecPage";

const helpPages: Record<string, ComponentType> = {
  "/help/about/": About,
  "/help/spec/": SpecPage,
};

export const routes = Object.keys(helpPages);

export function render(route: string): string {
  const Page = helpPages[route];
  if (!Page) {
    throw new Error(`No help page registered for ${route}`);
  }

  return renderToString(
    <StrictMode>
      <Page />
    </StrictMode>,
  );
}
