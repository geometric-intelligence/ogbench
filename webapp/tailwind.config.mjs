/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        'bg-primary': '#f8fafc',
        'bg-secondary': '#ffffff',
        'bg-card': '#ffffff',
        'bg-hover': '#f1f5f9',
        'border': '#e2e8f0',
        'text-primary': '#0f172a',
        'text-secondary': '#475569',
        'text-muted': '#94a3b8',
        'accent-gold': '#d97706',
        'accent-blue': '#3b82f6',
        'accent-orange': '#f97316',
        'accent-green': '#22c55e',
        'gnn': '#8b5cf6',
        'neural': '#ec4899',
        'baseline': '#0ea5e9',
      },
      fontFamily: {
        sans: ['DM Sans', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.4s ease-out',
      },
      keyframes: {
        fadeIn: {
          from: { opacity: '0', transform: 'translateY(10px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
};
