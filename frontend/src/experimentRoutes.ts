import { normalizePairStrategy } from './pairTerminology'

export const CLASSICAL_CLASSIFIERS = new Set([
  'svm',
  'rf',
  'extratrees',
  'histgb',
  'xgb',
  'lgb',
  'logreg',
])

export const TRAINABLE_DL_MODELS = new Set([
  'cnn',
  'rnn',
  'gru',
  'lstm',
  'transformer',
  'cross_attn',
  'bi_cross_attn',
  'mlp',
  'gcn',
  'gat',
  'hgt',
  'sage',
  'graph_transformer',
])

export type ModelDropoutScope =
  'none' | 'cnn' | 'sequence' | 'transformer' | 'graph' | 'interaction' | 'backbone'

export function dropoutScopeForModel(model: string): ModelDropoutScope {
  const normalized = String(model ?? '')
    .trim()
    .toLowerCase()
  if (normalized === 'mlp') return 'none'
  if (normalized === 'cnn') return 'cnn'
  if (['rnn', 'gru', 'lstm'].includes(normalized)) return 'sequence'
  if (normalized === 'transformer') return 'transformer'
  if (['gcn', 'gat', 'hgt', 'sage', 'graph_transformer'].includes(normalized)) return 'graph'
  if (['cross_attn', 'bi_cross_attn'].includes(normalized)) return 'interaction'
  return 'backbone'
}

export function configText(config: Record<string, unknown>, key: string) {
  if (key === 'pair_strategy') return normalizePairStrategy(config[key])
  return String(config[key] ?? '')
    .trim()
    .toLowerCase()
}

export function effectiveClassifier(config: Record<string, unknown>) {
  const classifier = configText(config, 'classifier')
  if (classifier) return classifier === 'calibrated_svm' ? 'svm' : classifier
  const model = configText(config, 'model')
  if (model === 'calibrated_svm') return 'svm'
  return CLASSICAL_CLASSIFIERS.has(model) ? model : ''
}

export function resolveClassicFeatureRoute(
  tokenizer: string,
  encoder: string,
  classicTokenizers: string[],
  classicEncoderOptions: Record<string, string[]>,
) {
  const resolvedTokenizer = classicTokenizers.includes(tokenizer)
    ? tokenizer
    : (classicTokenizers[0] ?? tokenizer)
  const validEncoders = classicEncoderOptions[resolvedTokenizer] ?? []
  const resolvedEncoder = validEncoders.includes(encoder) ? encoder : (validEncoders[0] ?? 'index')
  return { tokenizer: resolvedTokenizer, encoder: resolvedEncoder }
}

export function isPureClassicalClassifierRoute(config: Record<string, unknown>) {
  const rawModel = configText(config, 'model')
  const model = rawModel === 'calibrated_svm' ? 'svm' : rawModel
  const classifier = effectiveClassifier(config)
  return (
    CLASSICAL_CLASSIFIERS.has(model) &&
    CLASSICAL_CLASSIFIERS.has(classifier) &&
    !configText(config, 'sequence_model') &&
    configText(config, 'training_mode') !== 'export_embeddings'
  )
}

export function usesEpochTraining(config: Record<string, unknown>) {
  if (isPureClassicalClassifierRoute(config)) return false
  if (configText(config, 'training_mode') === 'export_embeddings') return true
  if (configText(config, 'sequence_model')) return true
  return TRAINABLE_DL_MODELS.has(configText(config, 'model'))
}

export function classifierDisplayName(classifier: string) {
  const labels: Record<string, string> = {
    svm: 'SVM',
    rf: 'Random Forest',
    extratrees: 'ExtraTrees',
    histgb: 'HistGradientBoosting',
    xgb: 'XGBoost',
    lgb: 'LightGBM',
    logreg: 'Logistic Regression',
  }
  return labels[classifier] ?? classifier
}
