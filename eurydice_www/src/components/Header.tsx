import React from "react";
import { ExternalWebsite, Github, D20Logo } from "./Icons";

interface HeaderProps {
  showTutorial?: boolean;
  onTutorialClick?: () => void;
}

const linkClass =
  "block rounded px-2 py-1 text-sm font-medium text-[var(--text-muted)] transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--text)]";

export default function Header({
  showTutorial = false,
  onTutorialClick,
}: HeaderProps) {
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);
  const close = () => setIsMenuOpen(false);

  return (
    <header
      className="sticky top-0 z-30 border-b"
      style={{
        background: "color-mix(in srgb, var(--bg) 88%, transparent)",
        backdropFilter: "blur(8px)",
        borderColor: "var(--border)",
      }}
    >
      <nav className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-1 gap-y-2 px-4 py-2.5">
        <a
          href="/"
          className="flex items-center gap-2 pr-2 font-semibold tracking-tight text-[var(--text)]"
        >
          <span style={{ color: "var(--accent)" }}>
            <D20Logo className="size-7" />
          </span>
          Eurydice
        </a>

        <button
          className="ml-auto flex size-8 flex-col items-center justify-center gap-1 rounded-md hover:bg-[var(--surface-2)] md:hidden"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          aria-label="Toggle menu"
        >
          <span
            className={`block h-0.5 w-5 bg-current transition-transform ${isMenuOpen ? "translate-y-1.5 rotate-45" : ""}`}
          ></span>
          <span
            className={`block h-0.5 w-5 bg-current transition-opacity ${isMenuOpen ? "opacity-0" : ""}`}
          ></span>
          <span
            className={`block h-0.5 w-5 bg-current transition-transform ${isMenuOpen ? "-translate-y-1.5 -rotate-45" : ""}`}
          ></span>
        </button>

        <ul
          className={`${isMenuOpen ? "flex" : "hidden"} w-full flex-col gap-0.5 md:flex md:w-auto md:flex-row md:items-center`}
        >
          <li>
            <a className={linkClass} href="/help/about/" onClick={close}>
              About
            </a>
          </li>
          <li>
            <a className={linkClass} href="/help/spec/" onClick={close}>
              Specification
            </a>
          </li>
          {showTutorial && (
            <li>
              <a
                className={linkClass}
                href="#"
                onClick={() => {
                  close();
                  if (
                    onTutorialClick &&
                    confirm(
                      "Opening the tutorial will clear the current code. Continue?",
                    )
                  ) {
                    onTutorialClick();
                  }
                }}
              >
                Tutorial
              </a>
            </li>
          )}

          <li aria-hidden="true" className="hidden md:block">
            <span
              className="mx-1 block h-4 w-px"
              style={{ background: "var(--border-strong)" }}
            />
          </li>

          <li>
            <a
              className={`${linkClass} flex items-center gap-1`}
              href="https://anydice.com"
              onClick={close}
            >
              AnyDice <ExternalWebsite />
            </a>
          </li>
          <li>
            <a
              className={`${linkClass} flex items-center gap-1`}
              href="https://anydice.com/docs"
              onClick={close}
            >
              AnyDice Docs <ExternalWebsite />
            </a>
          </li>
          <li>
            <a
              className={`${linkClass} flex items-center gap-1.5`}
              href="https://github.com/MattX/eurydice"
              aria-label="View source on GitHub"
              onClick={close}
            >
              <Github className="size-4" />
              <span className="md:sr-only">GitHub</span>
            </a>
          </li>
        </ul>
      </nav>
    </header>
  );
}
