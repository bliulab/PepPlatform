import { describe, expect, it } from 'vitest'
import type { InferenceRunRecord } from '../types'
import { buildInferenceHistoryTable, pageInferenceHistoryRows } from './inferenceHistoryResults'

function sequenceRun(): InferenceRunRecord {
  return {
    id: 'sequence-run',
    model_id: 'model-1',
    input_kind: 'sequences',
    input_count: 2,
    prediction_count: 2,
    request: { sequences: ['ACD', 'GG'] },
    response: {
      model_id: 'model-1',
      labels: ['active', 'toxic'],
      predictions: [{ sequence: 'ACD', scores: [0.84, 0.147] }, { scores: [0.04, 0.6] }],
    },
  }
}

it('builds complete sequence history columns and falls back to request inputs', () => {
  const table = buildInferenceHistoryTable(sequenceRun())

  expect(table.columns.map((column) => column.kind)).toEqual([
    'index',
    'sequence',
    'top_label',
    'top_score',
    'label_score',
    'label_score',
  ])
  expect(table.rows[0]).toMatchObject({ sequence: 'ACD', topLabel: 'active', topScore: 0.84 })
  expect(table.rows[1]).toMatchObject({ sequence: 'GG', topLabel: 'toxic', topScore: 0.6 })
})

it('builds complete pair history columns and preserves interaction fields', () => {
  const run: InferenceRunRecord = {
    id: 'pair-run',
    model_id: 'model-1',
    input_kind: 'pairs',
    input_count: 1,
    prediction_count: 1,
    request: { pairs: [{ peptide: 'AC', protein: 'GG' }] },
    response: {
      model_id: 'model-1',
      labels: ['interaction'],
      predictions: [
        {
          scores: [0.75],
          score: 0.75,
          score_kind: 'probability',
          threshold: 0.5,
          is_interaction: true,
          prediction_label: 'Interact',
        },
      ],
    },
  }

  const table = buildInferenceHistoryTable(run)

  expect(table.columns.map((column) => column.kind)).toEqual([
    'index',
    'peptide',
    'partner',
    'prediction_label',
    'interaction_score',
    'score_kind',
    'threshold',
    'is_interaction',
    'label_score',
  ])
  expect(table.rows[0]).toMatchObject({ peptide: 'AC', partner: 'GG' })
  expect(table.rows[0].prediction).toMatchObject({ score: 0.75, is_interaction: true })
})

it('exposes score kind for ensemble sequence history without changing legacy columns', () => {
  const run = sequenceRun()
  run.response.score_kind = 'rank'
  run.response.ensemble = {
    method: 'rank_mean',
    batch_dependent: true,
    batch_digest: 'full-batch',
    input_count: 2,
    numpy_version: 'test',
    contract_version: 1,
  }
  const table = buildInferenceHistoryTable(run)
  expect(table.columns.map((column) => column.kind)).toContain('score_kind')
  expect(table.rows).toHaveLength(2)
  expect(table.rows.map((row) => row.sequence)).toEqual(['ACD', 'GG'])
})

describe('pageInferenceHistoryRows', () => {
  it('paginates at twenty rows and clamps an invalid page', () => {
    const source = sequenceRun()
    source.request.sequences = Array.from({ length: 41 }, (_, index) => `SEQ${index + 1}`)
    source.response.predictions = source.request.sequences.map((sequence) => ({
      sequence,
      scores: [0.5],
    }))
    const rows = buildInferenceHistoryTable(source).rows

    expect(pageInferenceHistoryRows(rows, 2)).toMatchObject({
      page: 2,
      totalPages: 3,
      start: 21,
      end: 40,
    })
    expect(pageInferenceHistoryRows(rows, 99)).toMatchObject({
      page: 3,
      totalPages: 3,
      start: 41,
      end: 41,
    })
  })
})
