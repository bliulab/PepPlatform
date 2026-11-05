import { describe, expect, it } from 'vitest'
import type { FeatureImportanceItem } from '../types'
import {
  DEFAULT_VISIBLE_IMPORTANCE_LIMIT,
  createImportanceFormatter,
  formatFullImportance,
  selectVisibleImportanceItems,
} from './featureImportanceChart'

function item(name: string, importance: number, rank: number): FeatureImportanceItem {
  return { name, importance, rank }
}

describe('feature importance chart helpers', () => {
  it('selects the top ranked finite importance items', () => {
    const shuffled = Array.from({ length: 18 }, (_, index) =>
      item(`C${index + 1}`, 1 - index * 0.01, index + 1),
    )
      .reverse()
      .concat(item('invalid', Number.NaN, 2))

    const selected = selectVisibleImportanceItems(shuffled)

    expect(selected).toHaveLength(DEFAULT_VISIBLE_IMPORTANCE_LIMIT)
    expect(selected.map((entry) => entry.name)).toEqual(
      Array.from({ length: 15 }, (_, index) => `C${index + 1}`),
    )
  })

  it('falls back to importance when ranks tie or are missing', () => {
    const selected = selectVisibleImportanceItems([
      item('ranked-low', 0.1, 1),
      item('missing-rank-high', 0.9, Number.NaN),
      item('missing-rank-low', 0.2, Number.NaN),
      item('ranked-high', 0.8, 1),
    ])

    expect(selected.map((entry) => entry.name)).toEqual([
      'ranked-high',
      'ranked-low',
      'missing-rank-high',
      'missing-rank-low',
    ])
  })

  it('keeps visible labels uniform while increasing precision for meaningful gaps', () => {
    const formatter = createImportanceFormatter([0.03, 0.0100008, 0.0100042])

    expect(formatter.mode).toBe('fixed')
    expect(formatter.decimals).toBeGreaterThan(5)
    const decimals = formatter.decimals as number

    const left = formatter.format(0.0100008)
    const right = formatter.format(0.0100042)
    expect(left).not.toBe(right)
    expect(left.split('.')[1]).toHaveLength(decimals)
    expect(right.split('.')[1]).toHaveLength(decimals)
  })

  it('uses scientific notation for extremely small importance values', () => {
    const formatter = createImportanceFormatter([1.2e-7, 1.25e-7])

    expect(formatter.mode).toBe('exponential')
    expect(formatter.format(1.2e-7)).toContain('e-')
  })

  it('formats tooltip values with higher precision and without padded zeros', () => {
    expect(formatFullImportance(0.030005)).toBe('0.030005')
    expect(formatFullImportance(1.2e-7)).toBe('1.2e-7')
  })
})
