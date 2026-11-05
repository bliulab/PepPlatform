import { describe, expect, it } from 'vitest'

import {
  experimentDisplayName,
  formatLocalizedDate,
  legacyGeneratedName,
  modelDisplayName,
  nextRerunNameForDisplay,
} from './localization'
import type { ExperimentRecord, ModelRecord } from './types'

const translate = (key: string, params: Record<string, unknown> = {}) => {
  const name = String(params.name ?? '')
  if (key === 'generatedNames.embeddings') return `${name} embeddings`
  if (key === 'generatedNames.model') return `${params.task || ''} model`.trim()
  if (key === 'generatedNames.pairFeatures')
    return `Pep[${params.peptide}] + Partner[${params.partner}]`
  if (key === 'generatedNames.rerun') return `${name} · rerun`
  if (key === 'generatedNames.rerunCount') return `${name} · rerun ${params.count}`
  if (key === 'agent.lanes.conservative') return 'Conservative'
  if (key === 'agent.lanes.balanced') return 'Balanced'
  if (key === 'agent.lanes.exploratory') return 'Exploratory'
  return key
}

function experiment(
  name: string,
  generated_name?: ExperimentRecord['generated_name'],
): ExperimentRecord {
  return {
    id: 'experiment-1',
    dataset_id: 'dataset-1',
    name,
    generated_name,
    task_type: 'pep_function',
    config: {},
    status: 'succeeded',
    log_path: '',
    artifact_dir: '',
    metrics: { events: [], summary: {} },
  }
}

describe('localized dates', () => {
  it('formats the same timestamp with the selected locale', () => {
    const options: Intl.DateTimeFormatOptions = { month: 'long', day: 'numeric', timeZone: 'UTC' }
    expect(formatLocalizedDate('2026-07-16T12:00:00Z', 'en', options)).toBe('July 16')
    expect(formatLocalizedDate('2026-07-16T12:00:00Z', 'zh', options)).toBe('7月16日')
  })
})

describe('generated names', () => {
  it('parses legacy rerun suffixes in either language', () => {
    expect(legacyGeneratedName('MLP · 重跑 · rerun')).toEqual({
      kind: 'rerun',
      base_name: 'MLP',
      base_kind: 'literal',
      count: 2,
    })
  })

  it('localizes a config-derived name and its Agent qualifier', () => {
    const record = experiment('MLP + ESM2 正则化增强', {
      kind: 'config',
      qualifier: 'balanced',
    })
    record.config = { tokenizer: 'esm2', encoder: 'index', model: 'mlp', classifier: 'mlp' }

    expect(experimentDisplayName(record, translate)).toBe('ESM2-Embedding-MLP · Balanced')
  })

  it('localizes the config base before adding a rerun suffix', () => {
    const record = experiment('MLP + ESM2 正则化增强 · 重跑', {
      kind: 'rerun',
      base_name: 'MLP + ESM2 正则化增强',
      base_kind: 'config',
      qualifier: 'balanced',
      count: 1,
    })
    record.config = { tokenizer: 'esm2', encoder: 'index', model: 'mlp', classifier: 'mlp' }

    expect(experimentDisplayName(record, translate)).toBe('ESM2-Embedding-MLP · Balanced · rerun')
  })

  it('uses semantic metadata instead of the persisted translated suffix', () => {
    const record = experiment('MLP · 重跑', { kind: 'rerun', base_name: 'MLP', count: 1 })
    expect(experimentDisplayName(record, translate)).toBe('MLP · rerun')
    expect(nextRerunNameForDisplay(record.name, record.generated_name, translate)).toBe(
      'MLP · rerun 2',
    )
  })

  it('keeps explicitly renamed model names unchanged', () => {
    const source = experiment('MLP · 重跑', { kind: 'rerun', base_name: 'MLP', count: 1 })
    const model: ModelRecord = {
      id: 'model-1',
      experiment_id: source.id,
      name: 'Publication two_stream / Single-Stream / 双塔 baseline',
      name_is_custom: true,
      task_type: source.task_type,
      bundle_path: '',
      supports_inference: true,
      metadata: {},
    }
    expect(modelDisplayName(model, source, translate)).toBe(
      'Publication two_stream / Single-Stream / 双塔 baseline',
    )
  })
})
