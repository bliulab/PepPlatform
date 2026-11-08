/** Shared light/dark ECharts palette for the web UI. */
export type ChartMode = 'light' | 'dark'

export type ChartPalette = {
  /** Categorical series colors, ordered from primary to tertiary. */
  series: string[]
  accent: string
  secondary: string
  positive: string
  negative: string
  /** Independent test split; the primary curve in ROC/PR and split comparisons. */
  splitTest: string
  /** Validation split; always the dimmed/context counterpart of `splitTest`. */
  splitValidation: string
  /** Color used for the currently selected point/bar. */
  selected: string
  axis: string
  axisLabel: string
  split: string
  tooltipBg: string
  tooltipText: string
  /** Card background behind data points (point borders, heatmap gaps). */
  surface: string
  foreground: string
  /** Empty/not-available heatmap cell. */
  mutedCell: string
  /** Low → mid → high ramp for heatmaps. */
  heatmapRange: [string, string, string]
  emphasisBorder: string
}

const palettes: Record<ChartMode, ChartPalette> = {
  light: {
    series: ['#0f766e', '#7c3aed', '#b45309', '#be123c', '#0369a1', '#65a30d'],
    accent: '#0f766e',
    secondary: '#0d9488',
    positive: '#15803d',
    negative: '#dc2626',
    splitTest: '#0f766e',
    splitValidation: '#b45309',
    selected: '#0f172a',
    axis: '#e2e8f0',
    axisLabel: '#64748b',
    split: '#eef2f7',
    tooltipBg: '#0f172a',
    tooltipText: '#f8fafc',
    surface: '#ffffff',
    foreground: '#334155',
    mutedCell: '#eef2f7',
    heatmapRange: ['#f0fdfa', '#14b8a6', '#115e59'],
    emphasisBorder: '#0f766e',
  },
  dark: {
    series: ['#2dd4bf', '#a78bfa', '#fbbf24', '#fb7185', '#38bdf8', '#a3e635'],
    accent: '#2dd4bf',
    secondary: '#22d3ee',
    positive: '#4ade80',
    negative: '#f87171',
    splitTest: '#2dd4bf',
    splitValidation: '#fbbf24',
    selected: '#e6ebf4',
    axis: '#283246',
    axisLabel: '#94a3b8',
    split: '#1f2937',
    tooltipBg: '#0b0f17',
    tooltipText: '#e6ebf4',
    surface: '#121826',
    foreground: '#dbe3ef',
    mutedCell: '#1b2333',
    heatmapRange: ['#0d2b28', '#0d9488', '#5eead4'],
    emphasisBorder: '#5eead4',
  },
}

export function chartPalette(mode: ChartMode): ChartPalette {
  return palettes[mode]
}

export function readChartMode(): ChartMode {
  if (typeof document === 'undefined') return 'light'
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'
}
