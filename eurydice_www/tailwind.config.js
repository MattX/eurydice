/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./about/index.html",
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [require("@tailwindcss/typography")],
};
