import React from "react";
import { DarkModeContext } from "./DarkModeContext";

function prefersDarkMode() {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  );
}

/** Provides the active system color scheme to the application. */
export function DarkModeSwitcher({ children }: { children: React.ReactNode }) {
  const [darkMode, setDarkMode] = React.useState(prefersDarkMode);

  React.useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const updateDarkMode = (event: MediaQueryListEvent) =>
      setDarkMode(event.matches);

    mediaQuery.addEventListener("change", updateDarkMode);
    return () => mediaQuery.removeEventListener("change", updateDarkMode);
  }, []);

  return (
    <DarkModeContext.Provider value={darkMode}>
      {children}
    </DarkModeContext.Provider>
  );
}
