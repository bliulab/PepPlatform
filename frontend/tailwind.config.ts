import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{vue,ts}'],
  theme: {
    extend: {
      // Mirrors the design tokens in src/styles.css. Only the semantic
      // modifier classes (text-brass / text-moss on dataset chips) still rely
      // on these keys, so keep them in sync with the CSS variables.
      colors: {
        ink: '#0f172a',
        accent: '#0f766e',
        brass: '#d97706',
        moss: '#15803d',
      },
    },
  },
  plugins: [],
} satisfies Config
