import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the Flask backend so the browser talks to a
// single origin during development. For production builds, set VITE_API_BASE.
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: { output: { manualChunks: { charts: ["recharts"], react: ["react", "react-dom"] } } },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://127.0.0.1:5001",
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: 4173,
  },
});
