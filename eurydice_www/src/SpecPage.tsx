import { useRef } from "react";
import Spec from "../../spec.md";
import Header from "./components/Header";
import Toc from "./components/Toc";

export default function SpecPage() {
  const contentRef = useRef<HTMLElement>(null);

  return (
    <>
      <Header />
      <div className="mx-auto grid max-w-7xl px-4 pb-8 lg:grid-cols-[16rem_minmax(0,56rem)] lg:gap-10">
        <Toc contentRef={contentRef} />
        <main ref={contentRef} className="document-content prose dark:prose-invert min-w-0 max-w-none">
          <Spec />
        </main>
      </div>
    </>
  );
}
