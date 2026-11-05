import { normalizeTaskType } from './taskTerminology'
import type { Capabilities, ExperimentRecord, ModelRecord } from './types'

export type FusionMethod = 'probability_mean' | 'logit_mean' | 'rank_mean'
export type WeightSource = 'equal' | 'custom' | 'validation'
export interface EnsemblePayload {
  name: string
  task_type: string
  dataset_id: string
  member_ids: string[]
  method: FusionMethod
  weight_source: WeightSource
  weights?: number[]
  grid_step: number
  objective: string
  objective_params?: Record<string, number>
  threshold_objective?: string
  threshold_objective_params?: Record<string, number>
  evaluate_test: boolean
  seed: number
}
export interface EnsembleCapabilities {
  enabled: boolean
  tasks: string[]
  methods: FusionMethod[]
  weight_sources: WeightSource[]
  objectives?: Capabilities['selection_objectives']
}
export interface EnsemblePreflight {
  valid: boolean
  errors: string[]
  error_codes: string[]
  validation_count?: number
  label_names?: string[]
}
export function memberUnavailableReason(model: ModelRecord, task: string): string {
  if (normalizeTaskType(model.task_type) !== normalizeTaskType(task)) return 'taskMismatch'
  if (!model.supports_inference) return 'inferenceUnavailable'
  if (['ensemble', 'web_ensemble'].includes(String(model.metadata.bundle_type))) return 'nested'
  // Serialized evaluation format compatibility; this is independent of experiment versions.
  if (
    model.metadata.evaluation_protocol_version !== 3 ||
    model.metadata.feature_fit_scope !== 'train_only'
  )
    return 'protocolRequired'
  return ''
}
export function ensemblePayloadError(payload: EnsemblePayload): string {
  if (!payload.dataset_id) return 'selectDataset'
  for (const [id, params] of [
    [payload.objective, payload.objective_params],
    [payload.threshold_objective, payload.threshold_objective_params],
  ] as const) {
    if (
      id === 'dual_f1_sacc' &&
      ['reference_f1', 'reference_sacc'].some(
        (key) => !Number.isFinite(params?.[key]) || Number(params?.[key]) <= 0,
      )
    )
      return 'invalidReferences'
  }
  if (
    payload.member_ids.length < 2 ||
    new Set(payload.member_ids).size !== payload.member_ids.length
  )
    return 'selectMembers'
  if (payload.weight_source === 'custom') {
    const weights = payload.weights ?? []
    if (
      weights.length !== payload.member_ids.length ||
      weights.some((x) => !Number.isFinite(x) || x < 0) ||
      weights.reduce((a, b) => a + b, 0) <= 0
    )
      return 'invalidWeights'
  }
  return ''
}

export function cloneEnsemblePayload(experiment: ExperimentRecord): EnsemblePayload | undefined {
  const definition = experiment.config.web_ensemble as
    Omit<EnsemblePayload, 'name' | 'dataset_id'> | undefined
  if (!definition || !Array.isArray(definition.member_ids)) return undefined
  return {
    ...definition,
    name: experiment.name,
    dataset_id: experiment.dataset_id,
    member_ids: [...definition.member_ids],
    weights: definition.weights ? [...definition.weights] : undefined,
  }
}

export const ensembleMessages = {
  zh: {
    title: '创建模型集成',
    eyebrow: '模型组合',
    intro: '组合已训练模型，在验证集上确定权重和阈值，生成可独立使用的集成模型。',
    name: '集成名称',
    task: '预测任务',
    dataset: '共同数据集',
    selectDataset: '请选择共同数据集',
    members: '选择成员模型',
    membersHint: '选择顺序会被保存。预检将核实验证集来源、样本顺序和标签定义。',
    selectMembers: '请选择至少两个不同的模型',
    selected: '已选 {count} 个成员',
    empty: '此任务暂无已注册模型。',
    method: '融合方法',
    probability_mean: '概率平均',
    logit_mean: 'Logit 平均',
    rank_mean: '排名平均',
    weightSource: '权重来源',
    equal: '等权',
    custom: '自定义',
    validation: '验证集选权',
    weight: '权重',
    objective: '选权目标',
    auc: 'AUC 优先，AUPR 决胜',
    aupr: 'AUPR 优先，AUC 决胜',
    dual_f1_sacc: 'Macro-F1 / SAcc 双目标',
    grid: '搜索步长',
    evaluateTest: '冻结集成后评估独立测试集',
    testHint: '默认只评估验证集。独立测试集按名称分别报告。',
    rankHint:
      '排名平均需要至少两行完整批次。分数随同批样本及顺序变化，不表示概率；推理历史会保存完整批次。',
    rankScore: '排名分数',
    preflight: '检查成员与数据',
    checking: '正在核验…',
    create: '创建集成',
    creating: '正在提交…',
    passed: '预检通过 · {count} 个验证样本',
    disabled: '当前部署尚未启用新集成的创建功能。已有集成仍可使用。',
    taskMismatch: '任务不同',
    inferenceUnavailable: '不支持可复用推理',
    nested: '不能选择集成作为成员',
    protocolRequired: '需要当前评估信息和特征仅在训练集拟合的记录',
    invalidWeights: '权重必须是非负数，且至少一个为正数',
    evidenceHint:
      '缺少训练前记录的数据身份时，系统无法为旧模型补造来源证明。能证明数据来源但缺少样本分数身份时，将重新推理验证集。',
    summary: '集成定义',
    thresholds: '冻结阈值',
    testSplits: '独立测试划分',
    unavailable: '暂无可用结果',
    validationMetrics: '验证集诊断',
    validationOnly: '本次仅评估验证集；这些指标用于选权和定阈值，不代表独立测试结果。',
    validationCurvesHint:
      '此处仅展示验证集诊断，不代表独立测试表现。多标签 ROC / PR 使用 micro 平均；如已评估独立测试，请查看对应划分结果。',
    validationRocAria: '验证集 ROC 曲线；多标签使用 micro 平均，不代表独立测试结果。',
    validationPrAria: '验证集 PR 曲线；多标签使用 micro 平均，不代表独立测试结果。',
    thresholdsHint: '逐标签阈值按记录的验证目标确定，冻结后用于独立测试和推理。',
    macro_f1: 'Macro-F1',
    sacc: 'SAcc',
    f1: 'F1',
    thresholdObjective: '阈值选择目标',
    reference: '参考值',
    invalidReferences: '双目标的两个参考值必须为有限正数',
    tasks: {
      pep_function: '治疗性肽功能预测',
      pep_hlai: '肽–HLA-I 配体呈递预测',
      pep_hlaii: '肽–HLA-II 配体呈递预测',
      pep_protein_interaction: '肽–蛋白相互作用预测',
    },
  },
  en: {
    title: 'Create model ensemble',
    eyebrow: 'Model composition',
    intro:
      'Combine trained models, select weights and thresholds on validation, and retain an independent ensemble model.',
    name: 'Ensemble name',
    task: 'Prediction task',
    dataset: 'Shared dataset',
    selectDataset: 'Select a shared dataset',
    members: 'Choose member models',
    membersHint:
      'Selection order is retained. Preflight verifies validation provenance, row order, and label definitions.',
    selectMembers: 'Select at least two different models',
    selected: '{count} members selected',
    empty: 'No registered models for this task.',
    method: 'Fusion method',
    probability_mean: 'Probability mean',
    logit_mean: 'Logit mean',
    rank_mean: 'Rank mean',
    weightSource: 'Weight source',
    equal: 'Equal',
    custom: 'Custom',
    validation: 'Validation search',
    weight: 'Weight',
    objective: 'Selection objective',
    auc: 'AUC first, AUPR tie-break',
    aupr: 'AUPR first, AUC tie-break',
    dual_f1_sacc: 'Macro-F1 / SAcc dual objective',
    grid: 'Search step',
    evaluateTest: 'Evaluate independent tests after freezing the ensemble',
    testHint: 'Validation only by default. Independent tests are reported by name.',
    rankHint:
      'Rank mean needs a complete batch of at least two rows. Scores depend on the batch and its order and are not probabilities. History retains the complete batch.',
    rankScore: 'Rank score',
    preflight: 'Check members and data',
    checking: 'Checking…',
    create: 'Create ensemble',
    creating: 'Submitting…',
    passed: 'Preflight passed · {count} validation samples',
    disabled:
      'New ensemble creation is disabled in this deployment. Existing ensembles remain usable.',
    taskMismatch: 'Different task',
    inferenceUnavailable: 'Reusable inference unavailable',
    nested: 'Nested ensembles are unavailable',
    protocolRequired: 'Current evaluation metadata and records of training-only feature fitting required',
    invalidWeights: 'Weights must be non-negative, with at least one positive value',
    evidenceHint:
      'Historical training provenance cannot be reconstructed from current files. When data identity is proven but score row identity is missing, validation is inferred again.',
    summary: 'Ensemble definition',
    thresholds: 'Frozen thresholds',
    testSplits: 'Independent test splits',
    unavailable: 'No results available',
    validationMetrics: 'Validation diagnostics',
    validationOnly:
      'Validation only: these metrics select weights and thresholds; they are not independent test results.',
    validationCurvesHint:
      'These charts show validation diagnostics, not independent test performance. Multi-label ROC / PR use micro averaging. If independent tests were evaluated, see their named split results.',
    validationRocAria:
      'Validation ROC curve; micro-averaged for multi-label tasks, not an independent test result.',
    validationPrAria:
      'Validation PR curve; micro-averaged for multi-label tasks, not an independent test result.',
    thresholdsHint:
      'Per-label thresholds follow the recorded validation objective and are frozen for independent tests and inference.',
    macro_f1: 'Macro-F1',
    sacc: 'SAcc',
    f1: 'F1',
    thresholdObjective: 'Threshold objective',
    reference: 'reference',
    invalidReferences: 'Both dual-objective references must be finite positive numbers',
    tasks: {
      pep_function: 'Therapeutic peptide function prediction',
      pep_hlai: 'Peptide–HLA-I ligand presentation prediction',
      pep_hlaii: 'Peptide–HLA-II ligand presentation prediction',
      pep_protein_interaction: 'Peptide–protein interaction prediction',
    },
  },
}
