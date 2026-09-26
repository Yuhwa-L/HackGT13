import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy API + image routes to the FastAPI backend on :8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
      "/images": "http://localhost:8000",
      "/shop-images": "http://localhost:8000",
    },
  },
});
