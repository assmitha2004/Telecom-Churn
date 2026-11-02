import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In development, requests to /api are forwarded to the Flask server on port 5000.
// In production (Vercel), /api is served by the Python function in /api/index.py.
const apiProxy = { '/api': 'http://127.0.0.1:5000' };

export default defineConfig({
  plugins: [react()],
  server: { port: 3000, proxy: apiProxy },
  preview: { port: 3000, proxy: apiProxy },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom', 'react-router-dom'],
          charts: ['recharts'],
        },
      },
    },
  },
});
