import { defineConfig } from "vite";

export default defineConfig({
  base: "/frontend/",
  build: {
    outDir: "../frontend/dist",
    emptyOutDir: true
  }
});
