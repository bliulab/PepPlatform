import { describe, expect, it } from 'vitest'
import { chartPalette, type ChartMode } from './chartTheme'

const modes: ChartMode[] = ['light', 'dark']

function hue(hex: string) {
  const r = Number.parseInt(hex.slice(1, 3), 16) / 255
  const g = Number.parseInt(hex.slice(3, 5), 16) / 255
  const b = Number.parseInt(hex.slice(5, 7), 16) / 255
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const delta = max - min
  if (delta === 0) return 0
  let degrees: number
  if (max === r) degrees = 60 * (((g - b) / delta) % 6)
  else if (max === g) degrees = 60 * ((b - r) / delta + 2)
  else degrees = 60 * ((r - g) / delta + 4)
  return (degrees + 360) % 360
}

function hueDistance(left: string, right: string) {
  const diff = Math.abs(hue(left) - hue(right)) % 360
  return diff > 180 ? 360 - diff : diff
}

describe('chart palette split colors', () => {
  it('keeps validation and test on distinct, well separated hues', () => {
    for (const mode of modes) {
      const palette = chartPalette(mode)
      expect(palette.splitTest).toMatch(/^#[0-9a-f]{6}$/i)
      expect(palette.splitValidation).toMatch(/^#[0-9a-f]{6}$/i)
      expect(palette.splitTest).not.toBe(palette.splitValidation)
      expect(hueDistance(palette.splitTest, palette.splitValidation)).toBeGreaterThanOrEqual(90)
    }
  })

  it('no longer borrows the accent/positive pair for the two splits', () => {
    for (const mode of modes) {
      const palette = chartPalette(mode)
      expect([palette.splitTest, palette.splitValidation]).not.toEqual([
        palette.accent,
        palette.positive,
      ])
      expect(palette.splitTest).not.toBe(palette.positive)
      expect(hueDistance(palette.splitValidation, palette.positive)).toBeGreaterThan(60)
    }
  })

  it('returns stable palette objects', () => {
    for (const mode of modes) {
      expect(chartPalette(mode)).toBe(chartPalette(mode))
    }
    expect(chartPalette('light')).not.toBe(chartPalette('dark'))
  })
})
