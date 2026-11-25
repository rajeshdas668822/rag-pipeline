import react from '@vitejs/plugin-react'
import path from 'path'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@components': path.resolve(import.meta.dirname, 'src/components'),
      '@hooks': path.resolve(import.meta.dirname, 'src/hooks'),
      '@lib': path.resolve(import.meta.dirname, 'src/lib'),
      '@types': path.resolve(import.meta.dirname, 'src/types'),
      '@store': path.resolve(import.meta.dirname, 'src/store'),
      '@styles': path.resolve(import.meta.dirname, 'src/styles'),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:11434',
        changeOrigin: true,
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
  },
})
