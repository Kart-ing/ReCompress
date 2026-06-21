import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base must match the GitHub Pages path: https://<user>.github.io/ReCompress/
// If deploying to Vercel/Netlify (served at root) instead, set base to "/".
export default defineConfig({
  plugins: [react()],
  base: "/ReCompress/",
});
