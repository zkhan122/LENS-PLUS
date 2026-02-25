import fs from "node:fs";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const useHttps = env.DEV_HTTPS === "true";

  let https: { key: Buffer; cert: Buffer } | undefined;
  if (useHttps) {
    const keyFile = env.DEV_HTTPS_KEY_FILE;
    const certFile = env.DEV_HTTPS_CERT_FILE;

    if (!keyFile || !certFile) {
      throw new Error(
        "DEV_HTTPS=true requires DEV_HTTPS_KEY_FILE and DEV_HTTPS_CERT_FILE"
      );
    }

    if (!fs.existsSync(keyFile) || !fs.existsSync(certFile)) {
      throw new Error("HTTPS cert or key file not found");
    }

    https = {
      key: fs.readFileSync(keyFile),
      cert: fs.readFileSync(certFile)
    };
  }

  return {
    server: {
      host: "0.0.0.0",
      port: 5173,
      https,
      proxy: {
        "/api": {
          target: env.VITE_API_PROXY_TARGET || "http://localhost:8000",
          changeOrigin: true,
          secure: false,
          rewrite: (path) => path.replace(/^\/api/, "")
        }
      }
    }
  };
});
