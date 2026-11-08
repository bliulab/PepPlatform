import { modelRouteSummary } from './modelDisplay'
import { experimentDisplayName, formatLocalizedDate, type Translate } from './localization'
import type {
  ComparisonMetricKey,
  ComparisonMetricRow,
  ComparisonReportPayload,
  ComparisonSplitKey,
  ConfigDiffRow,
  DatasetRecord,
  ExperimentAnalysis,
  ExperimentRecord,
} from './types'

export const COMPARISON_MAX_EXPERIMENTS = 6

export const COMPARISON_METRICS: ComparisonMetricKey[] = [
  'f1',
  'auc',
  'aupr',
  'acc',
  'sacc',
  'balanced_accuracy',
  'precision',
  'recall',
  'specificity',
  'mcc',
]

export const COMPARISON_HIGHLIGHT_KEYS = [
  'tokenizer',
  'encoder',
  'model',
  'classifier',
  'sequence_model',
  'pair_strategy',
  'stream_mode',
  'streams',
  'interaction_level',
  'interaction_model',
  'predictor_type',
  'graph_predictor',
  'fusion_mode',
  'training_mode',
  'head_hidden_dims',
  'head_activation',
  'head_dropout',
  'epochs',
  'batch_size',
  'lr',
]

function compactJson(value: unknown) {
  if (value === undefined || value === null || value === '') return '-'
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function metricNumber(value: unknown) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : undefined
}

function metricValue(
  experiment: ExperimentRecord,
  split: ComparisonSplitKey,
  metric: ComparisonMetricKey,
) {
  const summary = experiment.metrics?.summary ?? {}
  const prefix = split === 'validation' ? 'val' : 'test'
  const directKey = `${prefix}_${metric}`
  const directValue = metricNumber(summary[directKey])
  if (directValue !== undefined) return { value: directValue, source: directKey }
  if (split === 'validation' && metric === 'f1') {
    const bestValue = metricNumber(summary.best_val_f1)
    if (bestValue !== undefined) return { value: bestValue, source: 'best_val_f1' }
  }
  if (split === 'validation' && metric === 'auc') {
    const bestValue = metricNumber(summary.best_val_auc)
    if (bestValue !== undefined) return { value: bestValue, source: 'best_val_auc' }
  }
  return { value: undefined, source: '' }
}

export function normalizeComparisonIds(value: unknown, max = COMPARISON_MAX_EXPERIMENTS) {
  const chunks = Array.isArray(value) ? value : [value]
  const ids = chunks
    .flatMap((item) => String(item ?? '').split(','))
    .map((item) => item.trim())
    .filter(Boolean)
  return Array.from(new Set(ids)).slice(0, max)
}

export function selectedComparisonIds(currentIds: string[], nextId: string, selected: boolean) {
  const existing = currentIds.filter((id) => id !== nextId)
  if (!selected) return existing
  return [...existing, nextId].slice(0, COMPARISON_MAX_EXPERIMENTS)
}

export function buildMetricRows(experiments: ExperimentRecord[]): ComparisonMetricRow[] {
  return COMPARISON_METRICS.map((metric) => ({
    metric,
    validation: experiments.map((experiment) => ({
      experiment_id: experiment.id,
      ...metricValue(experiment, 'validation', metric),
    })),
    test: experiments.map((experiment) => ({
      experiment_id: experiment.id,
      ...metricValue(experiment, 'test', metric),
    })),
  }))
}

export function experimentScore(
  experiment: ExperimentRecord,
  split: ComparisonSplitKey,
  metric: ComparisonMetricKey,
) {
  return metricValue(experiment, split, metric)
}

export function buildConfigDiff(experiments: ExperimentRecord[]): ConfigDiffRow[] {
  const keys = Array.from(
    new Set(experiments.flatMap((experiment) => Object.keys(experiment.config ?? {}))),
  )
  const highlightSet = new Set(COMPARISON_HIGHLIGHT_KEYS)
  return keys
    .map((key) => {
      const values = Object.fromEntries(
        experiments.map((experiment) => [experiment.id, compactJson(experiment.config?.[key])]),
      )
      const distinct = new Set(Object.values(values))
      const sources = Object.fromEntries(
        experiments.map((experiment) => [
          experiment.id,
          experiment.config_parameter_sources?.[key] ?? '',
        ]),
      )
      return {
        key,
        changed: distinct.size > 1,
        highlighted: highlightSet.has(key),
        values,
        sources,
      }
    })
    .sort((left, right) => {
      if (left.changed !== right.changed) return left.changed ? -1 : 1
      if (left.highlighted !== right.highlighted) return left.highlighted ? -1 : 1
      return left.key.localeCompare(right.key)
    })
}

function formatMetric(value: unknown) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric.toFixed(4) : '-'
}

function datasetName(dataset?: DatasetRecord) {
  return dataset?.name ?? 'Unknown dataset'
}

export interface ComparisonReportLocalization {
  locale: unknown
  translate: Translate
}

function metricCell(row: ComparisonMetricRow, split: ComparisonSplitKey, experimentId: string) {
  return row[split].find((cell) => cell.experiment_id === experimentId)
}

function analysisSummary(
  analysis: ExperimentAnalysis | undefined,
  localization?: ComparisonReportLocalization,
) {
  const tr = localization?.translate
  if (!analysis) return tr ? tr('comparisonExport.notLoaded') : 'not loaded'
  const availability = ['validation', 'test']
    .map((split) => {
      const curves = analysis.curves?.[split]
      const roc = curves?.roc?.available ? 'ROC' : ''
      const pr = curves?.pr?.available ? 'PR' : ''
      const splitText = tr ? tr(`comparisonExport.splits.${split}`) : split
      const availabilityText =
        [roc, pr].filter(Boolean).join('+') || (tr ? tr('comparisonExport.none') : 'none')
      return [splitText, availabilityText].join(': ')
    })
    .join('; ')
  const warningCount = analysis.warnings?.length ?? 0
  if (!warningCount) return availability
  return tr
    ? `${availability}; ${tr('comparisonExport.warningCount', { count: warningCount })}`
    : `${availability}; warnings=${warningCount}`
}

export function buildMarkdownReport(
  payload: ComparisonReportPayload,
  localization?: ComparisonReportLocalization,
) {
  const tr = localization?.translate
  const label = (key: string, fallback: string, params?: Record<string, unknown>) =>
    tr ? tr(`comparisonExport.${key}`, params) : fallback
  const displayName = (experiment: ExperimentRecord) =>
    tr ? experimentDisplayName(experiment, tr) : experiment.name
  const date = (value?: string | null) => {
    if (!value) return '-'
    if (!localization) return value
    return formatLocalizedDate(
      value,
      localization.locale,
      { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' },
      '-',
    )
  }
  const lines: string[] = []
  lines.push(`# ${label('title', 'PepPlatform Experiment Comparison')}`)
  lines.push('')
  lines.push(`${label('generatedAt', 'Generated at')}: ${date(payload.generated_at)}`)
  lines.push('')
  lines.push(`## ${label('experiments', 'Experiments')}`)
  lines.push('')
  lines.push(
    `| ${[
      label('experiment', 'Experiment'),
      label('task', 'Task'),
      label('dataset', 'Dataset'),
      label('status', 'Status'),
      label('route', 'Route'),
      label('finished', 'Finished'),
    ].join(' | ')} |`,
  )
  lines.push('| --- | --- | --- | --- | --- | --- |')
  for (const experiment of payload.experiments) {
    const dataset = payload.datasets[experiment.dataset_id]
    lines.push(
      `| ${displayName(experiment)} | ${tr ? tr(`tasks.${experiment.task_type}.label`) : experiment.task_type} | ${
        dataset?.name ?? label('unknownDataset', datasetName(dataset))
      } | ${tr ? tr(`status.${experiment.status}`) : experiment.status} | ${
        modelRouteSummary(experiment.config) || '-'
      } | ${date(experiment.finished_at)} |`,
    )
  }
  lines.push('')
  lines.push(`## ${label('metrics', 'Metrics')}`)
  lines.push('')
  lines.push(
    [
      `| ${label('metric', 'Metric')} | ${label('split', 'Split')} |`,
      ...payload.experiments.map((experiment) => `${displayName(experiment)} |`),
    ].join(' '),
  )
  lines.push(['| --- | --- |', ...payload.experiments.map(() => '--- |')].join(' '))
  for (const row of payload.metric_rows) {
    for (const split of ['test', 'validation'] as const) {
      const cells = payload.experiments.map((experiment) =>
        formatMetric(metricCell(row, split, experiment.id)?.value),
      )
      lines.push(
        `| ${tr ? tr(`leaderboard.metrics.${row.metric}`) : row.metric} | ${
          tr ? tr(`comparisonExport.splits.${split}`) : split
        } | ${cells.join(' | ')} |`,
      )
    }
  }
  lines.push('')
  lines.push(`## ${label('configDifferences', 'Config Differences')}`)
  lines.push('')
  const changedRows = payload.config_diff.filter((row) => row.changed)
  if (!changedRows.length) {
    lines.push(label('noConfigDifferences', 'No config differences among selected experiments.'))
  } else {
    lines.push(
      [
        `| ${label('key', 'Key')} |`,
        ...payload.experiments.map((experiment) => `${displayName(experiment)} |`),
      ].join(' '),
    )
    lines.push(['| --- |', ...payload.experiments.map(() => '--- |')].join(' '))
    for (const row of changedRows) {
      const cells = payload.experiments.map((experiment) => row.values[experiment.id] ?? '-')
      lines.push(`| ${row.key} | ${cells.join(' | ')} |`)
    }
  }
  lines.push('')
  lines.push(`## ${label('analysisAvailability', 'Analysis Availability')}`)
  lines.push('')
  for (const experiment of payload.experiments) {
    lines.push(
      `- ${displayName(experiment)}: ${analysisSummary(payload.analyses[experiment.id], localization)}`,
    )
  }
  return lines.join('\n')
}

export function buildJsonReport(payload: ComparisonReportPayload) {
  return {
    generated_at: payload.generated_at,
    experiments: payload.experiments.map((experiment) => ({
      id: experiment.id,
      name: experiment.name,
      task_type: experiment.task_type,
      dataset: payload.datasets[experiment.dataset_id] ?? null,
      status: experiment.status,
      created_at: experiment.created_at,
      started_at: experiment.started_at,
      finished_at: experiment.finished_at,
      config: experiment.config,
      metrics: experiment.metrics,
      route_summary: modelRouteSummary(experiment.config),
      analysis: payload.analyses[experiment.id] ?? null,
    })),
    metric_rows: payload.metric_rows,
    config_diff: payload.config_diff,
  }
}

export function comparisonFilename(extension: 'md' | 'json', now = new Date()) {
  const pad = (value: number) => String(value).padStart(2, '0')
  const stamp = [
    now.getFullYear(),
    pad(now.getMonth() + 1),
    pad(now.getDate()),
    '-',
    pad(now.getHours()),
    pad(now.getMinutes()),
  ].join('')
  return `pepplatform-experiment-comparison-${stamp}.${extension}`
}

export function downloadTextFile(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}
