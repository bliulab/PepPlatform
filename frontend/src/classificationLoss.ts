import { configText, usesEpochTraining } from './experimentRoutes'

export const SEQUENCE_LOSS_FIELDS = [
  'multilabel_loss',
  'multilabel_pos_weight_cap',
  'asl_gamma_pos',
  'asl_gamma_neg',
  'asl_clip',
] as const
export const PAIR_LOSS_FIELDS = ['loss_type', 'pos_weight_max'] as const
export const CLASSIFICATION_LOSS_FIELDS = [...SEQUENCE_LOSS_FIELDS, ...PAIR_LOSS_FIELDS]

export function classificationLossApplies(config: Record<string, unknown>) {
  return usesEpochTraining(config) && !config.ensemble_members
}

export function defaultClassificationLoss(config: Record<string, unknown>, isPairTask: boolean) {
  if (isPairTask) return 'WBCE'
  const pooledPretrainedMlp =
    ['esm2', 'protbert', 'prott5', 'esmc', 'esm3'].includes(configText(config, 'encoder')) &&
    !configText(config, 'sequence_model') &&
    configText(config, 'model') === 'mlp'
  return pooledPretrainedMlp ? 'WBCE' : 'BCE'
}

export function classificationLossParametersValid(config: Record<string, unknown>) {
  const bounds: [string, number, number][] = [
    ['multilabel_pos_weight_cap', 1, Infinity],
    ['pos_weight_max', 1, Infinity],
    ['asl_gamma_pos', 0, Infinity],
    ['asl_gamma_neg', 0, Infinity],
    ['asl_clip', 0, 1],
  ]
  return bounds.every(([key, min, max]) => {
    if (!(key in config)) return true
    const raw = config[key]
    if (typeof raw !== 'number' && typeof raw !== 'string') return false
    const value = typeof raw === 'string' && !raw.trim() ? NaN : Number(raw)
    return Number.isFinite(value) && value >= min && value < max
  })
}
