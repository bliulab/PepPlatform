import { describe, expect, it } from 'vitest'
import type { AnalysisOperatingPoint, ExperimentAnalysis } from '../types'
import {
  analysisOperatingPoint,
  isMultilabelOperatingPoint,
  operatingPointValues,
  thresholdEntries,
} from './scoreDiagnostics'

const point: AnalysisOperatingPoint = {
  values: [0.78, 0.41],
  thresholds: [0.78, 0.41],
  label_names: ['AVP', 'APP'],
  threshold_by_label: { APP: 0.41, AVP: 0.78 },
  threshold_source: 'validation_f1',
}

describe('score diagnostic operating-point helpers', () => {
  it('detects and preserves the declared order of per-label thresholds', () => {
    expect(isMultilabelOperatingPoint(point)).toBe(true)
    expect(thresholdEntries(point)).toEqual([
      { label: 'AVP', value: 0.78 },
      { label: 'APP', value: 0.41 },
    ])
  })

  it('keeps scalar operating points on the binary path', () => {
    const binary = { threshold: 0.67 }
    expect(operatingPointValues(binary)).toEqual([0.67])
    expect(isMultilabelOperatingPoint(binary)).toBe(false)
  })

  it('reads the operating point from analysis metrics', () => {
    const analysis = { metrics: { operating_point: point } } as ExperimentAnalysis
    expect(analysisOperatingPoint(analysis)).toBe(point)
  })
})
