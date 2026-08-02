import { resolve } from "path"
import { defineConfig } from "electron-vite"
import react from "@vitejs/plugin-react"

export default defineConfig({
  main: {
    build: {
      lib: {
        entry: resolve(__dirname, "frontend/electron/main.ts")
      }
    }
  },
  preload: {
    build: {
      lib: {
        entry: resolve(__dirname, "frontend/electron/preload.ts")
      }
    }
  },
  renderer: {
    root: resolve(__dirname, "frontend/src"),
    plugins: [react()],
    build: {
      rollupOptions: {
        input: resolve(__dirname, "frontend/src/index.html")
      }
    }
  }
})
