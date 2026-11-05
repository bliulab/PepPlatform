import type { InferenceRunRecord, PredictionRecord } from '../types'

export type InferenceHistoryColumnKind =
  | 'index'
  | 'sequence'
  | 'peptide'
  | 'partner'
  | 'prediction_label'
  | 'interaction_score'
  | 'score_kind'
  | 'threshold'
  | 'is_interaction'
  | 'top_label'
  | 'top_score'
  | 'label_score'

export interface InferenceHistoryColumn {
  key: string
  kind: InferenceHistoryColumnKind
  label?: string
  scoreIndex?: number
}

export interface InferenceHistoryRow {
  index: number
  prediction: PredictionRecord
  sequence: string
  peptide: string
  partner: string
  topLabel: string
  topScore?: number
}

export interface InferenceHistoryTable {
  columns: InferenceHistoryColumn[]
  rows: InferenceHistoryRow[]
}

function labelAt(labels: string[], index: number) {
  return labels[index] || ''
}

function topScoreDetails(prediction: PredictionRecord, labels: string[]) {
  const scores = Array.isArray(prediction.scores) ? prediction.scores : []
  const finiteScores = scores
    .map((score, index) => ({ score, index }))
    .filter((item) => Number.isFinite(item.score))
  if (!finiteScores.length) return { topLabel: '', topScore: undefined }
  const top = finiteScores.reduce((best, item) => (item.score > best.score ? item : best))
  return { topLabel: labelAt(labels, top.index), topScore: top.score }
}

export function buildInferenceHistoryTable(run: InferenceRunRecord): InferenceHistoryTable {
  const labels = Array.isArray(run.response.labels) ? run.response.labels : []
  const predictions = Array.isArray(run.response.predictions) ? run.response.predictions : []
  const pairRun = run.input_kind === 'pairs'
  const columns: InferenceHistoryColumn[] = pairRun
    ? [
        { key: 'index', kind: 'index' },
        { key: 'peptide', kind: 'peptide' },
        { key: 'partner', kind: 'partner' },
        { key: 'prediction_label', kind: 'prediction_label' },
        { key: 'interaction_score', kind: 'interaction_score' },
        { key: 'score_kind', kind: 'score_kind' },
        { key: 'threshold', kind: 'threshold' },
        { key: 'is_interaction', kind: 'is_interaction' },
      ]
    : [
        { key: 'index', kind: 'index' },
        { key: 'sequence', kind: 'sequence' },
        { key: 'top_label', kind: 'top_label' },
        { key: 'top_score', kind: 'top_score' },
      ]
  if (!pairRun && run.response.ensemble) {
    columns.push({ key: 'score_kind', kind: 'score_kind' })
  }
  const scoreCount = Math.max(
    labels.length,
    ...predictions.map((prediction) =>
      Array.isArray(prediction.scores) ? prediction.scores.length : 0,
    ),
  )
  for (let index = 0; index < scoreCount; index += 1) {
    columns.push({
      key: `label_score_${index}`,
      kind: 'label_score',
      label: labelAt(labels, index),
      scoreIndex: index,
    })
  }

  const rows = predictions.map((prediction, index) => {
    const requestSequence = run.request.sequences?.[index] || ''
    const requestPair = run.request.pairs?.[index]
    const top = topScoreDetails(prediction, labels)
    return {
      index: index + 1,
      prediction,
      sequence: prediction.sequence || requestSequence,
      peptide: prediction.peptide || requestPair?.peptide || '',
      partner: prediction.protein || requestPair?.protein || '',
      ...top,
    }
  })

  return { columns, rows }
}

export function pageInferenceHistoryRows(rows: InferenceHistoryRow[], page: number, pageSize = 20) {
  const safeSize = Math.max(1, Math.floor(pageSize))
  const totalPages = Math.max(1, Math.ceil(rows.length / safeSize))
  const safePage = Math.max(1, Math.min(Math.floor(page), totalPages))
  const start = (safePage - 1) * safeSize
  return {
    page: safePage,
    totalPages,
    start: rows.length ? start + 1 : 0,
    end: Math.min(start + safeSize, rows.length),
    rows: rows.slice(start, start + safeSize),
  }
}
