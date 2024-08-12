import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import {lezer} from '@lezer/generator/rollup'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      plugins: [lezer()]
    }
  }
})
