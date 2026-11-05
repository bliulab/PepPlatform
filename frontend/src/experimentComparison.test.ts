import { describe, expect, it } from 'vitest'

import { buildConfigDiff, buildMarkdownReport, buildMetricRows } from './experimentComparison'
import type { ExperimentRecord } from './types'

function experiment(id: string, epochs: number, source: string): ExperimentRecord {
  return {
    id,
    dataset_id: 'dataset',
    name: id,
    task_type: 'pep_function',
    config: { model: 'cnn', epochs },
    config_parameter_sources: { model: 'explicit', epochs: source },
    status: 'succeeded',
    log_path: '',
    artifact_dir: '',
    metrics: { events: [], summary: {} },
  }
}

describe('buildConfigDiff', () => {
  it('keeps effective-parameter sources for default badges', () => {
    const rows = buildConfigDiff([
      experiment('default-run', 30, 'default'),
      experiment('explicit-run', 12, 'explicit'),
    ])
    const epochs = rows.find((row) => row.key === 'epochs')

    expect(epochs?.changed).toBe(true)
    expect(epochs?.values).toEqual({ 'default-run': '30', 'explicit-run': '12' })
    expect(epochs?.sources).toEqual({ 'default-run': 'default', 'explicit-run': 'explicit' })
  })
})

describe('buildMetricRows', () => {
  it('shows the complete validation snapshot supplied by the backend summary', () => {
    const run = experiment('cnn-run', 30, 'default')
    run.metrics.summary = {
      best_epoch: 22,
      best_val_f1: 0.5514,
      val_f1: 0.5514,
      val_auc: 0.862,
      val_aupr: 0.6533,
      val_acc: 0.4081,
    }

    const rows = buildMetricRows([run])

    expect(rows.find((row) => row.metric === 'f1')?.validation[0].value).toBe(0.5514)
    expect(rows.find((row) => row.metric === 'auc')?.validation[0].value).toBe(0.862)
    expect(rows.find((row) => row.metric === 'aupr')?.validation[0].value).toBe(0.6533)
    expect(rows.find((row) => row.metric === 'acc')?.validation[0].value).toBe(0.4081)
  })
})

describe('buildMarkdownReport', () => {
  it('localizes deterministic report copy and generated experiment names', () => {
    const run = experiment('persisted · rerun', 30, 'default')
    run.generated_name = { kind: 'rerun', base_name: 'persisted', count: 1 }
    run.finished_at = '2026-07-20T00:00:00Z'
    const translations: Record<string, string> = {
      'comparisonExport.title': '实验对比',
      'comparisonExport.generatedAt': '生成时间',
      'comparisonExport.experiments': '实验',
      'comparisonExport.experiment': '实验',
      'comparisonExport.task': '任务',
      'comparisonExport.dataset': '数据集',
      'comparisonExport.status': '状态',
      'comparisonExport.route': '路线',
      'comparisonExport.finished': '完成时间',
      'comparisonExport.metrics': '指标',
      'comparisonExport.metric': '指标',
      'comparisonExport.split': '数据划分',
      'comparisonExport.configDifferences': '配置差异',
      'comparisonExport.noConfigDifferences': '没有配置差异。',
      'comparisonExport.key': '配置项',
      'comparisonExport.analysisAvailability': '分析可用性',
      'comparisonExport.unknownDataset': '未知数据集',
      'comparisonExport.notLoaded': '未加载',
      'comparisonExport.none': '无',
      'comparisonExport.warningCount': '警告={count}',
      'comparisonExport.splits.test': '测试集',
      'comparisonExport.splits.validation': '验证集',
      'generatedNames.rerun': '{name} · 重跑',
      'tasks.pep_function.label': '肽功能预测',
      'status.succeeded': '成功',
    }
    const translate = (key: string, params: Record<string, unknown> = {}) =>
      (translations[key] ?? key).replace(/\{(\w+)\}/g, (_match, name) => String(params[name] ?? ''))
    const report = buildMarkdownReport(
      {
        generated_at: '2026-07-20T00:00:00Z',
        experiments: [run],
        datasets: {},
        metric_rows: buildMetricRows([run]),
        config_diff: buildConfigDiff([run]),
        analyses: {},
      },
      { locale: 'zh', translate },
    )

    expect(report).toContain('# 实验对比')
    expect(report).toContain('persisted · 重跑')
    expect(report).toContain('| 肽功能预测 | 未知数据集 | 成功 |')
    expect(report).toContain('## 配置差异')
    expect(report).not.toContain('persisted · rerun')
  })
})
