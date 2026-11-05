export type HeatmapKind = 'stratified' | 'labels' | 'patterns'
export interface HeatmapParams {
  split: string
  kind: HeatmapKind
  metric?: string
  pattern?: string
  scale?: string
}
export interface HeatmapCapabilities {
  version: number
  reason: string
  task_type?: string
  label_names: string[]
  splits: {
    key: string
    threshold_available: boolean
    types: { kind: HeatmapKind; available: boolean; reason: string }[]
  }[]
}
export interface HeatmapJob {
  id: string
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'canceled' | 'interrupted'
  params: HeatmapParams
  error: string | null
}
export interface HeatmapCell {
  row: string
  column: string
  value: number | null
  count: number
  positive?: number
  negative?: number
  prevalence?: number
  intersection?: number
  union?: number
  row_support?: number
  column_support?: number
  reason: string
  low_support?: boolean
  imbalanced_support?: boolean
}
export interface HeatmapResult {
  params: HeatmapParams
  rows: string[]
  columns: string[]
  cells: HeatmapCell[]
  row_total: number
  column_total: number
  row_offset: number
  column_offset: number
  search: string
  sample_count: number
  color_max: number
  count_scale: boolean
  provenance: Record<string, unknown>
  operating_point: Record<string, unknown> | null
  definitions: Record<string, unknown>
}

export function visibleCellValue(
  cell: HeatmapCell,
  kind: HeatmapKind,
  showSmall: boolean,
): number | null {
  return kind === 'stratified' && cell.low_support && !showSmall ? null : cell.value
}
export function shortGroup(value: string) {
  return value.startsWith('protein:') ? value.slice(0, 19) + '…' : value
}
export function sameParams(a: HeatmapParams, b: HeatmapParams) {
  return (
    a.kind === b.kind &&
    a.split === b.split &&
    (a.kind !== 'stratified' || a.metric === b.metric) &&
    (a.kind !== 'patterns' || (a.pattern === b.pattern && a.scale === b.scale))
  )
}

export const heatmapMessages = {
  zh: {
    title: '可视化分析',
    intro: '从整体表现，深入到分组与标签模式。',
    entry: '按需查看分层性能、逐标签指标与标签模式。',
    open: '打开可视化分析',
    kind: '分析类型',
    split: '数据划分',
    chooseSplit: '请选择数据划分',
    validation: '验证集',
    test: '测试集',
    test_warm: 'Warm 测试集',
    test_cold: 'Cold 测试集',
    stratified: '分层性能',
    labels: '逐标签指标',
    patterns: '标签模式',
    metric: '指标',
    acc: 'ACC（逐标签）',
    f1: 'F1',
    auc: 'ROC-AUC',
    aupr: 'AUPR（AP）',
    count: '样本数',
    prevalence: '阳性比例',
    pattern: '事件类型',
    truth: '真实标签共现',
    prediction: '预测标签共现',
    false_negative: '共同漏报',
    false_positive: '共同误报',
    scale: '数值类型',
    jaccard: 'Jaccard',
    generate: '生成分析',
    cancel: '取消任务',
    queued: '排队中',
    running: '正在核验与计算',
    succeeded: '分析完成',
    failed: '分析失败',
    canceled: '已取消',
    interrupted: '任务已中断，请重新生成',
    loading: '正在加载',
    empty: '选择分析类型和数据划分后，点击“生成分析”。',
    unavailable: '此分析当前不可用',
    small: '显示不足 20 个样本的分组数值',
    smallHint: '灰色表示缺失、不可计算或样本较少；低样本提示不代表统计显著性。',
    stale: '参数已更改。下方仍显示上次结果，点击“生成分析”更新。',
    result: '当前结果',
    samples: '样本',
    filter: '筛选分组或标签',
    previous: '上一页',
    next: '下一页',
    rows: '行',
    columns: '列',
    table: '查看数据表',
    png: '导出 PNG ＋ 来源 JSON',
    csv: '导出完整 CSV',
    details: '单元格详情',
    selectCell: '点击图中单元格或数据表中的数值查看详情。',
    value: '数值',
    positive: '阳性数',
    negative: '阴性数',
    intersection: '共同发生数',
    union: '至少一个发生数',
    row_support: '行事件数',
    column_support: '列事件数',
    noData: '无样本',
    imbalanced: '正例或负例少于 5 个，谨慎解释。',
    low: '样本数不足 20。',
    patternHint: '共现不等于类别混淆。预测与错误事件使用原实验冻结的验证集阈值。',
    metricHint: 'ACC 为逐标签二分类准确率；AUPR 使用 Average Precision。各指标独立解释。',
    postHoc: '此图是冻结模型的测试集事后分析，不用于重新选择模型或阈值。',
    source: '数据与参数来源',
    dimensions: '分组 × 原始肽长',
    scrollHint: '横向滚动可查看完整矩阵。',
    errors: {
      evidence_missing: '缺少所需的预测产物。',
      evidence_invalid: '预测产物无法读取或格式不一致。',
      alignment_missing: '缺少可靠的样本对应证据，无法关联分组信息。',
      alignment_invalid: '样本身份或顺序不一致。',
      dataset_changed: '数据身份与原实验不一致。',
      input_changed: '输入或分析版本已变化，请重新生成。',
      group_missing: '数据缺少可信的 allele、HLA 配对或靶标字段。',
      task_unsupported: '此任务不支持该分析类型。',
      label_mismatch: '标签名称、顺序或数量不一致。',
      threshold_missing: '缺少冻结阈值，只能进行不依赖阈值的分析。',
      threshold_mismatch: '冻结阈值记录不一致。',
      protocol_required: '缺少核实评估过程所需的记录。',
      scores_invalid: '标签或预测分数无效。',
      split_unavailable: '没有此数据划分的预测产物。',
      source_forbidden: '此数据来源不能通过 Web 可视化读取。',
      experiment_not_complete: '仅已完成的实验可进行此分析。',
      queue_full: '分析队列已满，请稍后重试。',
      job_missing: '分析任务不存在或不可访问。',
      result_not_ready: '分析结果尚未生成。',
      parameters_invalid: '分析参数无效。',
      service_stopped: '分析服务已停止。',
      analysis_failed: '分析失败，请检查产物或重新生成。',
      single_class: '该分组只有单一真实类别，排名指标不可用。',
      empty_union: '两个事件均未发生，Jaccard 不可用。',
      dataset_missing: '实验关联的数据集不存在。',
      interrupted: '分析已中断，请重新生成。',
      canceled: '分析已取消。',
    },
  },
  en: {
    title: 'Visual analysis',
    intro: 'Explore performance across groups and label patterns.',
    entry: 'Explore stratified performance, per-label metrics and label patterns on demand.',
    open: 'Open visual analysis',
    kind: 'Analysis type',
    split: 'Data split',
    chooseSplit: 'Select a split',
    validation: 'Validation',
    test: 'Test',
    test_warm: 'Warm test',
    test_cold: 'Cold test',
    stratified: 'Stratified performance',
    labels: 'Per-label metrics',
    patterns: 'Label patterns',
    metric: 'Metric',
    acc: 'ACC (per label)',
    f1: 'F1',
    auc: 'ROC-AUC',
    aupr: 'AUPR (AP)',
    count: 'Sample count',
    prevalence: 'Positive fraction',
    pattern: 'Event type',
    truth: 'True label co-occurrence',
    prediction: 'Predicted co-occurrence',
    false_negative: 'Joint false negatives',
    false_positive: 'Joint false positives',
    scale: 'Value type',
    jaccard: 'Jaccard',
    generate: 'Generate analysis',
    cancel: 'Cancel task',
    queued: 'Queued',
    running: 'Verifying and computing',
    succeeded: 'Analysis complete',
    failed: 'Analysis failed',
    canceled: 'Canceled',
    interrupted: 'Interrupted; generate again to retry',
    loading: 'Loading',
    empty: 'Choose an analysis and split, then select Generate analysis.',
    unavailable: 'This analysis is unavailable',
    small: 'Show values for groups with fewer than 20 samples',
    smallHint:
      'Gray indicates missing, undefined or low-support cells. Support flags do not indicate statistical significance.',
    stale: 'Parameters changed. The previous result is still shown; generate again to update.',
    result: 'Current result',
    samples: 'samples',
    filter: 'Filter groups or labels',
    previous: 'Previous',
    next: 'Next',
    rows: 'Rows',
    columns: 'Columns',
    table: 'View data table',
    png: 'Export PNG + provenance JSON',
    csv: 'Export full CSV',
    details: 'Cell details',
    selectCell: 'Select a cell in the chart or a value in the table to inspect it.',
    value: 'Value',
    positive: 'Positive count',
    negative: 'Negative count',
    intersection: 'Joint events',
    union: 'Either event',
    row_support: 'Row events',
    column_support: 'Column events',
    noData: 'No samples',
    imbalanced: 'Fewer than 5 positive or negative samples; interpret cautiously.',
    low: 'Fewer than 20 samples.',
    patternHint:
      'Co-occurrence is not class confusion. Predicted and error events use the original frozen validation thresholds.',
    metricHint:
      'ACC is per-label binary accuracy. AUPR is Average Precision. Interpret metric columns separately.',
    postHoc:
      'Post-hoc analysis of a frozen model on Test; not for selecting models or thresholds again.',
    source: 'Data and parameter provenance',
    dimensions: 'Group × original peptide length',
    scrollHint: 'Scroll horizontally to see the full matrix.',
    errors: {
      evidence_missing: 'Required prediction artifacts are missing.',
      evidence_invalid: 'Prediction evidence is unreadable or inconsistent.',
      alignment_missing: 'Reliable sample alignment evidence is missing.',
      alignment_invalid: 'Sample identities or order do not match.',
      dataset_changed: 'Dataset identity differs from the original experiment.',
      input_changed: 'Inputs or analysis version changed. Generate again.',
      group_missing: 'Trusted allele, HLA pair or target fields are missing.',
      task_unsupported: 'This analysis is not supported for this task.',
      label_mismatch: 'Label names, order or dimensions disagree.',
      threshold_missing:
        'Frozen thresholds are missing. Only threshold-independent analyses are available.',
      threshold_mismatch: 'Frozen operating point records disagree.',
      protocol_required: 'Saved records of the evaluation process are required.',
      scores_invalid: 'Invalid labels or prediction scores.',
      split_unavailable: 'No predictions exist for this split.',
      source_forbidden: 'This source cannot be read by Web visualizations.',
      experiment_not_complete: 'Only completed experiments can be analyzed.',
      queue_full: 'The analysis queue is full. Please retry later.',
      job_missing: 'Analysis task not found or inaccessible.',
      result_not_ready: 'Results are not ready.',
      parameters_invalid: 'Invalid analysis parameters.',
      service_stopped: 'Analysis service stopped.',
      analysis_failed: 'Analysis failed. Check the artifacts or retry.',
      single_class: 'Only one true class; ranking metrics are unavailable.',
      empty_union: 'Neither event occurred; Jaccard is undefined.',
      dataset_missing: 'The registered dataset is missing.',
      interrupted: 'Analysis interrupted. Generate again to retry.',
      canceled: 'Analysis canceled.',
    },
  },
}
