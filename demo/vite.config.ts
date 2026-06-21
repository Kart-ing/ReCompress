import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vercel/Netlify serve at root, so base is "/".
// (For GitHub Pages at https://<user>.github.io/ReCompress/, set base to "/ReCompress/".)
export default defineConfig({
  plugins: [react()],
  base: "/",
});
