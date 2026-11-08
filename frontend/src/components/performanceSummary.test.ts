import { describe, expect, it } from 'vitest'
import type { ExperimentAnalysis } from '../types'
import { buildPerformanceSummary } from './performanceSummary'

function analysisFixture(): ExperimentAnalysis {
  return {
    schema_version: 5,
    curves: {
      validation: {
        average: 'micro',
        roc: { available: true, auc: 0.91, points: [] },
        pr: { available: true, aupr: 0.59, points: [] },
      },
      test: {
        average: 'micro',
        roc: { available: true, auc: 0.92, points: [] },
        pr: { available: true, aupr: 0.58, points: [] },
      },
    },
    metrics: {
      operating_point: { thresholds: [0.7, 0.4] },
      validation: { averaging: 'macro', f1: 0.54, auc: 0.81, aupr: 0.42, aupr_micro: 0.59 },
      test: { averaging: 'macro', f1: 0.49, sacc: 0.53, auc: 0.88, aupr: 0.5, aupr_micro: 0.58 },
    },
    projection_2d: { available: false, points: [] },
    projection_3d: { available: false, points: [] },
    clusters: { available: false },
    feature_importance: { available: false, items: [] },
    warnings: [],
  }
}

describe('performance summary model', () => {
  it('uses the paper-oriented multi-label test metrics in the requested order', () => {
    const model = buildPerformanceSummary(
      { best_epoch: 14, best_val_f1: 0.5429 },
      analysisFixture(),
      'pep_function',
    )

    expect(model.selection).toEqual({
      labelKey: 'detail.bestValMacroF1',
      value: 0.5429,
      bestEpoch: 14,
    })
    expect(model.primary.map((metric) => metric.id)).toEqual([
      'macro_f1',
      'sacc',
      'macro_auc',
      'macro_aupr',
    ])
    expect(model.primary.map((metric) => metric.value)).toEqual([0.49, 0.53, 0.88, 0.5])
  })

  it('never substitutes the micro PR area for multi-label macro AUPR', () => {
    const fixture = analysisFixture()
    delete fixture.metrics?.test?.aupr
    delete fixture.metrics?.validation?.aupr
    const model = buildPerformanceSummary({}, fixture, 'pep_function')

    expect(model.primary.find((metric) => metric.id === 'macro_aupr')?.value).toBeUndefined()
    expect(
      model.validationSupplemental.find((metric) => metric.id === 'macro_aupr'),
    ).toBeUndefined()
  })

  it('uses binary ROC and PR values as legacy fallbacks', () => {
    const fixture = analysisFixture()
    fixture.metrics = { operating_point: { threshold: 0.5 }, validation: {}, test: {} }
    fixture.curves.validation.average = 'binary'
    fixture.curves.test.average = 'binary'
    const model = buildPerformanceSummary(
      { test_f1: 0.7, test_acc: 0.8 },
      fixture,
      'pep_protein_interaction',
    )

    expect(model.isMultilabel).toBe(false)
    expect(model.primary.map((metric) => metric.id)).toEqual(['auc', 'aupr', 'f1', 'acc'])
    expect(model.primary.map((metric) => metric.value)).toEqual([0.92, 0.58, 0.7, 0.8])
  })
})
