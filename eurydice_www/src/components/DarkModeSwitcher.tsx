import React from "react";

export const DarkModeContext = React.createContext(false);

const darkModeMediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

/** Provides DarkModeContext as a context, allowing children to switch to light or dark mode. */
export function DarkModeSwitcher({ children }: { children: React.ReactNode }) {
  const [darkMode, setDarkMode] = React.useState(darkModeMediaQuery.matches);
  React.useEffect(() => {
    darkModeMediaQuery.addEventListener("change", (e) => {
      setDarkMode(e.matches);
    });
  }, []);
  return (
    <DarkModeContext.Provider value={darkMode}>
      {children}
    </DarkModeContext.Provider>
  );
}
