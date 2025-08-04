import React from "react";
import { ExternalWebsite, Octocat } from "./Icons";

interface HeaderProps {
  showTutorial?: boolean;
  onTutorialClick?: () => void;
}

export default function Header({ showTutorial = false, onTutorialClick }: HeaderProps) {
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);

  return (
    <>
      <Octocat />
      <nav className="w-[calc(100%-60px)] p-4">
      <div className="md:hidden flex justify-between items-center">
        <a className="hover:underline" href="#">
          Eurydice
        </a>
        <button
          className="flex flex-col justify-center items-center w-6 h-6 space-y-1"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          aria-label="Toggle menu"
        >
          <span className={`block w-5 h-0.5 bg-current transform transition ${isMenuOpen ? 'rotate-45 translate-y-1.5' : ''}`}></span>
          <span className={`block w-5 h-0.5 bg-current transition ${isMenuOpen ? 'opacity-0' : ''}`}></span>
          <span className={`block w-5 h-0.5 bg-current transform transition ${isMenuOpen ? '-rotate-45 -translate-y-1.5' : ''}`}></span>
        </button>
      </div>
      <ul className={`${isMenuOpen ? 'flex' : 'hidden'} md:flex flex-col md:flex-row flex-wrap mt-4 md:mt-0 *:border-l-0 md:*:border-l *:border-gray-500 *:px-0 md:*:px-4 *:py-2 md:*:py-0`}>
        <li className="border-none hidden md:block">
          <a className="hover:underline" href="/" onClick={() => setIsMenuOpen(false)}>
            Eurydice
          </a>
        </li>
        <li>
          <a className="hover:underline block" href="/help/about/" onClick={() => setIsMenuOpen(false)}>
            About
          </a>
        </li>
        {showTutorial && (
          <li>
            <a
              className="hover:underline block"
              href="#"
              onClick={() => {
                setIsMenuOpen(false);
                if (onTutorialClick && confirm("Opening the tutorial will clear the current code. Continue?")) {
                  onTutorialClick();
                }
              }}
            >
              Tutorial
            </a>
          </li>
        )}
        <li>
          <a className="hover:underline block" href="https://anydice.com" onClick={() => setIsMenuOpen(false)}>
            AnyDice <ExternalWebsite />
          </a>
        </li>
        <li>
          <a className="hover:underline block" href="https://anydice.com/docs" onClick={() => setIsMenuOpen(false)}>
            AnyDice Documentation <ExternalWebsite />
          </a>
        </li>
      </ul>
    </nav>
    </>
  );
}