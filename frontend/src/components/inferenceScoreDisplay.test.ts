import { describe, expect, it } from 'vitest'
import {
  formatIndependentScore,
  independentScoreWidth,
  rankIndependentLabelScores,
  scoresUseUnitInterval,
} from './inferenceScoreDisplay'

describe('inference score display helpers', () => {
  it('formats independent label scores as 0-1 decimals', () => {
    expect(formatIndependentScore(0.84)).toBe('0.840')
    expect(formatIndependentScore(0.147)).toBe('0.147')
    expect(formatIndependentScore(0.04)).toBe('0.040')
    expect(formatIndependentScore(Number.NaN)).toBe('-')
  })

  it('preserves each score width without sum normalization', () => {
    expect(independentScoreWidth(0.84)).toBe('84%')
    expect(independentScoreWidth(0.147)).toBe('14.7%')
    expect(independentScoreWidth(0.04)).toBe('4%')
  })

  it('distinguishes unit-interval scores from unbounded margins', () => {
    expect(scoresUseUnitInterval([0.84, 0.147, 0.04])).toBe(true)
    expect(scoresUseUnitInterval([0.84, -0.2, 1.4])).toBe(false)
    expect(scoresUseUnitInterval([])).toBe(false)
  })

  it('ranks scores while preserving backend threshold decisions', () => {
    const ranked = rankIndependentLabelScores(
      [0.851, 0.848, 0.863],
      ['AFP', 'ABP', 'AMP'],
      [0.858, 0.724, 0.486],
      ['ABP', 'AMP'],
    )

    expect(ranked).toEqual([
      { index: 2, label: 'AMP', score: 0.863, threshold: 0.486, isPredicted: true },
      { index: 0, label: 'AFP', score: 0.851, threshold: 0.858, isPredicted: false },
      { index: 1, label: 'ABP', score: 0.848, threshold: 0.724, isPredicted: true },
    ])
    expect(ranked.filter((item) => item.isPredicted).map((item) => item.label)).toEqual([
      'AMP',
      'ABP',
    ])
  })

  it('marks missing thresholds as unavailable without inventing a fallback decision', () => {
    expect(rankIndependentLabelScores([0.9], ['AMP'], [], ['AMP'])).toEqual([
      { index: 0, label: 'AMP', score: 0.9, threshold: undefined, isPredicted: true },
    ])
  })
})
