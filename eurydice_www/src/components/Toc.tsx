import { type RefObject, useEffect, useRef, useState } from "react";

interface TocEntry {
  id: string;
  level: 2 | 3;
  title: string;
}

interface TocSection {
  entry: TocEntry;
  children: TocEntry[];
}

interface TocProps {
  contentRef: RefObject<HTMLElement | null>;
}

function groupEntries(entries: TocEntry[]): TocSection[] {
  const sections: TocSection[] = [];

  for (const entry of entries) {
    if (entry.level === 2) {
      sections.push({ entry, children: [] });
    } else {
      sections[sections.length - 1]?.children.push(entry);
    }
  }

  return sections;
}

export default function Toc({ contentRef }: TocProps) {
  const [entries, setEntries] = useState<TocEntry[]>([]);
  const [activeId, setActiveId] = useState("");
  const desktopNavRef = useRef<HTMLElement>(null);
  const mobileDetailsRef = useRef<HTMLDetailsElement>(null);

  useEffect(() => {
    const content = contentRef.current;
    if (!content) return;

    const headings = Array.from(content.querySelectorAll<HTMLHeadingElement>("h2[id], h3[id]"));
    const nextEntries = headings.map((heading) => ({
      id: heading.id,
      level: heading.tagName === "H2" ? (2 as const) : (3 as const),
      title: heading.textContent?.trim() ?? heading.id,
    }));

    const findCurrentHeading = () => {
      const passedHeadings = headings.filter((heading) => heading.getBoundingClientRect().top <= 112);
      const current = passedHeadings[passedHeadings.length - 1];
      return current?.id ?? headings[0]?.id ?? "";
    };

    const frame = requestAnimationFrame(() => {
      setEntries(nextEntries);
      setActiveId(window.location.hash.slice(1) || findCurrentHeading());
    });

    const observer = new IntersectionObserver(
      () => setActiveId(findCurrentHeading()),
      { rootMargin: "-96px 0px -75% 0px" },
    );

    headings.forEach((heading) => observer.observe(heading));

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [contentRef]);

  useEffect(() => {
    const nav = desktopNavRef.current;
    const activeLink = Array.from(nav?.querySelectorAll<HTMLAnchorElement>("a") ?? []).find(
      (link) => link.hash === `#${activeId}`,
    );
    if (!nav || !activeLink) return;

    const navBounds = nav.getBoundingClientRect();
    const linkBounds = activeLink.getBoundingClientRect();

    if (linkBounds.top < navBounds.top) {
      nav.scrollBy({ top: linkBounds.top - navBounds.top, behavior: "smooth" });
    } else if (linkBounds.bottom > navBounds.bottom) {
      nav.scrollBy({ top: linkBounds.bottom - navBounds.bottom, behavior: "smooth" });
    }
  }, [activeId]);

  const sections = groupEntries(entries);

  const renderLink = (entry: TocEntry) => (
    <a
      className={`block border-l-2 py-1.5 pr-2 text-sm leading-5 transition-colors ${
        entry.level === 3 ? "pl-5" : "pl-3"
      } ${
        activeId === entry.id
          ? "border-sky-600 font-semibold text-sky-700 dark:border-sky-400 dark:text-sky-300"
          : "border-transparent text-slate-600 hover:border-slate-300 hover:text-slate-950 dark:text-slate-400 dark:hover:border-slate-600 dark:hover:text-slate-100"
      }`}
      href={`#${entry.id}`}
      aria-current={activeId === entry.id ? "location" : undefined}
      onClick={() => {
        setActiveId(entry.id);
        if (mobileDetailsRef.current) mobileDetailsRef.current.open = false;
      }}
    >
      {entry.title}
    </a>
  );

  const contents = (
    <ol>
      {sections.map(({ entry, children }) => (
        <li key={entry.id}>
          {renderLink(entry)}
          {children.length > 0 && (
            <ol>
              {children.map((child) => (
                <li key={child.id}>{renderLink(child)}</li>
              ))}
            </ol>
          )}
        </li>
      ))}
    </ol>
  );

  return (
    <>
      <details
        ref={mobileDetailsRef}
        className="mb-6 rounded-md border border-slate-300 bg-slate-50 px-4 py-3 dark:border-slate-600 dark:bg-slate-700/40 lg:hidden"
      >
        <summary className="cursor-pointer font-semibold">On this page</summary>
        <nav className="mt-3 max-h-80 overflow-y-auto" aria-label="Specification contents">
          {contents}
        </nav>
      </details>

      <aside className="sticky top-4 hidden max-h-[calc(100vh-2rem)] self-start lg:block">
        <nav
          ref={desktopNavRef}
          className="max-h-[calc(100vh-2rem)] overflow-y-auto pr-3"
          aria-label="Specification contents"
        >
          <p className="mb-2 text-sm font-semibold text-slate-950 dark:text-slate-100">On this page</p>
          {contents}
        </nav>
      </aside>
    </>
  );
}
