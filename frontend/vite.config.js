import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/predict': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/classes': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/regions': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/seasons': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
