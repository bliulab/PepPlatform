import { normalizePairFields } from './pairTerminology'
import { normalizeTaskType } from './taskTerminology'
import {
  CLASSICAL_CLASSIFIERS,
  classifierDisplayName,
  configText,
  effectiveClassifier,
  usesEpochTraining,
} from './experimentRoutes'
import type { ExperimentRecord, GeneratedNameQualifier, ModelRecord } from './types'

type NameTranslate = (key: string, params?: Record<string, unknown>) => string

const PRETRAINED_TOKENIZERS = new Set(['esm2', 'protbert', 'prott5', 'esmc', 'esm3'])
const TOKENIZER_DISPLAY_NAMES: Record<string, string> = {
  esm2: 'ESM2',
  protbert: 'ProtBERT',
  prott5: 'ProtT5',
  esmc: 'ESMC',
  esm3: 'ESM3',
}

const DISPLAY_NAMES: Record<string, string> = {
  cnn: 'CNN',
  rnn: 'RNN',
  gru: 'GRU',
  lstm: 'LSTM',
  mlp: 'MLP',
  gcn: 'GCN',
  gat: 'GAT',
  hgt: 'HGT',
  sage: 'GraphSAGE',
  transformer: 'Transformer',
  graph_transformer: 'Graph Transformer',
  cross_attn: 'Cross Attention',
  bi_cross_attn: 'Bi-Cross Attention',
  calibrated_svm: 'Calibrated SVM',
}

const METHOD_TOKENIZER_NAMES: Record<string, string> = {
  esm2: 'ESM2',
  protbert: 'ProtBERT',
  probert: 'ProtBERT',
  prott5: 'ProtT5',
  esmc: 'ESM-C',
  esm3: 'ESM3',
  kmer: 'Kmer',
  dr: 'DR',
  'top-n-gram': 'TopNGram',
  dt: 'DT',
  bpe: 'BPE',
  wordpiece: 'WordPiece',
  unigram: 'Unigram',
}

const METHOD_COMPONENT_NAMES: Record<string, string> = {
  index: 'Index',
  onehot: 'OneHot',
  property: 'Property',
  blosum62: 'BLOSUM62',
  pssm: 'PSSM',
  pssm_blosum62_hybrid: 'PSSM–BLOSUM62 Hybrid',
  psfm: 'PSFM',
  psfm_blosum62_hybrid: 'PSFM–BLOSUM62 Hybrid',
  word2vec: 'Word2Vec',
  glove: 'GloVe',
  fasttext: 'FastText',
  bow: 'BOW',
  tfidf: 'TF-IDF',
  textrank: 'TextRank',
  ac: 'AC',
  cc: 'CC',
  acc: 'ACC',
  pdt: 'PDT',
  ac_pssm: 'AC-PSSM',
  cc_pssm: 'CC-PSSM',
  acc_pssm: 'ACC-PSSM',
  ac_pssm_blosum62_hybrid: 'AC-PSSM–BLOSUM62 Hybrid',
  cc_pssm_blosum62_hybrid: 'CC-PSSM–BLOSUM62 Hybrid',
  acc_pssm_blosum62_hybrid: 'ACC-PSSM–BLOSUM62 Hybrid',
  lsa: 'LSA',
  legacy_lsa: 'LegacyLSA',
  lda: 'LDA',
  legacy_lda: 'LegacyLDA',
  labelled_lda: 'LabelledLDA',
  plsa: 'PLSA',
  legacy_kl_nmf: 'LegacyKLNMF',
  cnn: 'CNN',
  rnn: 'RNN',
  gru: 'GRU',
  lstm: 'LSTM',
  mlp: 'MLP',
  gcn: 'GCN',
  gat: 'GAT',
  hgt: 'HGT',
  sage: 'GraphSAGE',
  transformer: 'Transformer',
  graph_transformer: 'GraphTransformer',
  cross_attn: 'CrossAttn',
  bi_cross_attn: 'BiCrossAttn',
  svm: 'SVM',
  calibrated_svm: 'CalibratedSVM',
  rf: 'RF',
  extratrees: 'ExtraTrees',
  histgb: 'HistGB',
  xgb: 'XGB',
  lgb: 'LightGBM',
  logreg: 'LogReg',
}

export interface ModelMetricSummary {
  split: 'test' | 'validation' | ''
  f1?: string
  auc?: string
}

function rawText(config: Record<string, unknown>, key: string) {
  return String(config[key] ?? '').trim()
}

function titleText(value: string) {
  const normalized = value.trim().toLowerCase()
  if (!normalized) return ''
  if (CLASSICAL_CLASSIFIERS.has(normalized)) return classifierDisplayName(normalized)
  if (DISPLAY_NAMES[normalized]) return DISPLAY_NAMES[normalized]
  return normalized
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function compactMethodText(value: string, names = METHOD_COMPONENT_NAMES) {
  const normalized = value.trim().toLowerCase()
  if (!normalized) return ''
  if (names[normalized]) return names[normalized]
  return normalized
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join('')
}

function normalizedTokenizer(value: string) {
  const normalized = value.trim().toLowerCase()
  return normalized === 'probert' ? 'protbert' : normalized
}

function methodTokenizer(value: string) {
  return compactMethodText(value, METHOD_TOKENIZER_NAMES)
}

function methodEncoderForConfig(config: Record<string, unknown>) {
  const encoder = rawText(config, 'encoder')
  if (encoder.toLowerCase() !== 'blosum62') {
    return compactMethodText(encoder)
  }
  const k = Number(config.k ?? 1)
  return Number.isFinite(k) && k > 1 ? 'BLOSUM62DerivedMean' : 'BLOSUM62'
}

function esmcSource(config: Record<string, unknown>) {
  return (rawText(config, 'model_source') || rawText(config, 'esmc_model')).toLowerCase()
}

function esmcDisplayName(config: Record<string, unknown>, compact = false) {
  const source = esmcSource(config)
  if (source.includes('synthyra/esmplusplus_small')) {
    return compact ? 'ESM++-300M' : 'ESM++ 300M'
  }
  if (source.includes('esmc-300m-2024-12')) {
    return compact ? 'ESM-C-300M' : 'ESM-C 300M'
  }
  return compact ? 'ESM-C-600M' : 'ESM-C 600M'
}

function methodTokenizerForConfig(config: Record<string, unknown>) {
  const tokenizer = rawText(config, 'tokenizer')
  if (normalizedTokenizer(tokenizer) !== 'esmc') return methodTokenizer(tokenizer)
  return esmcDisplayName(config, true)
}

function tokenizerDisplayName(config: Record<string, unknown>) {
  const tokenizer = rawText(config, 'tokenizer')
  const tokenizerKey = normalizedTokenizer(tokenizer)
  if (tokenizerKey !== 'esmc') return TOKENIZER_DISPLAY_NAMES[tokenizerKey] ?? tokenizer
  return esmcDisplayName(config)
}

function methodRepresentation(config: Record<string, unknown>, interactionLevel = '') {
  const tokenizer = rawText(config, 'tokenizer')
  const encoder = rawText(config, 'encoder')
  const tokenizerKey = normalizedTokenizer(tokenizer)
  if (PRETRAINED_TOKENIZERS.has(tokenizerKey)) {
    const sequenceModel = configText(config, 'sequence_model')
    if (sequenceModel) return 'TokenEmbedding'
    return interactionLevel === 'token' ? 'Token' : 'Embedding'
  }
  if (!encoder || encoder.toLowerCase() === 'ppi_fusion') return ''
  return methodEncoderForConfig(config)
}

function methodPipeline(config: Record<string, unknown>, interactionLevel = '') {
  return [methodTokenizerForConfig(config), methodRepresentation(config, interactionLevel)]
    .filter(Boolean)
    .join('-')
}

function streamConfig(config: Record<string, unknown>, role: 'peptide' | 'partner') {
  const streams = config.streams
  if (!streams || typeof streams !== 'object' || Array.isArray(streams)) return null
  const stream = (streams as Record<string, unknown>)[role]
  return stream && typeof stream === 'object' && !Array.isArray(stream)
    ? (stream as Record<string, unknown>)
    : null
}

function exportedRepresentationModel(config: Record<string, unknown>) {
  if (configText(config, 'training_mode') !== 'export_embeddings') return ''
  const interactionModel = configText(config, 'interaction_model')
  const sequenceModel = configText(config, 'sequence_model')
  if (interactionModel) return interactionModel
  if (sequenceModel) return sequenceModel
  const model = configText(config, 'model')
  const classifier = effectiveClassifier(config)
  return model && model !== classifier ? model : ''
}

function displayClassifier(config: Record<string, unknown>) {
  const classifier = effectiveClassifier(config)
  const persistedHead = config.prediction_head
  const persistedKind =
    persistedHead && typeof persistedHead === 'object' && !Array.isArray(persistedHead)
      ? configText(persistedHead as Record<string, unknown>, 'kind')
      : ''
  if (
    classifier === 'mlp' &&
    (persistedKind === 'linear' ||
      (Array.isArray(config.head_hidden_dims) && config.head_hidden_dims.length === 0))
  ) {
    return 'linear'
  }
  const legacyAlias = ['classifier', 'model'].some(
    (key) => configText(config, key) === 'calibrated_svm',
  )
  return classifier === 'svm' &&
    (legacyAlias || configText(config, 'svm_calibration') === 'sigmoid')
    ? 'calibrated_svm'
    : classifier
}

function isExplicitStrictPairTopicRoute(
  config: Record<string, unknown>,
  peptideStream: Record<string, unknown> | null,
  partnerStream: Record<string, unknown> | null,
) {
  return Boolean(
    peptideStream &&
    partnerStream &&
    configText(peptideStream, 'encoder') === 'bow' &&
    configText(partnerStream, 'encoder') === 'bow' &&
    configText(config, 'pair_strategy') === 'two_stream' &&
    configText(config, 'interaction_level') === 'vector' &&
    configText(config, 'predictor_type') === 'classifier' &&
    configText(config, 'fusion_mode') === 'concat' &&
    CLASSICAL_CLASSIFIERS.has(effectiveClassifier(config)),
  )
}

function methodFeatureSignature(config: Record<string, unknown>) {
  config = normalizePairFields(config)
  const interactionLevel = configText(config, 'interaction_level')
  const peptideStream = streamConfig(config, 'peptide')
  const partnerStream = streamConfig(config, 'partner')
  let feature = ''

  if (peptideStream || partnerStream) {
    const peptide = peptideStream ? methodPipeline(peptideStream, interactionLevel) : ''
    const partner = partnerStream ? methodPipeline(partnerStream, interactionLevel) : ''
    feature =
      peptide && peptide === partner ? peptide : `Pep[${peptide || '-'}]+Partner[${partner || '-'}]`
  } else {
    feature = methodPipeline(config, interactionLevel)
  }

  const exportedModel = exportedRepresentationModel(config)
  if (exportedModel) {
    const exportedEmbedding = `${compactMethodText(exportedModel)}Emb`
    const rawEncoder = configText(config, 'encoder')
    const canReplaceIndex = !peptideStream && !partnerStream && rawEncoder === 'index'
    feature = canReplaceIndex
      ? [methodTokenizerForConfig(config), exportedEmbedding].filter(Boolean).join('-')
      : [feature, exportedEmbedding].filter(Boolean).join('-')
  }

  const reduction = configText(config, 'reduction')
  if (reduction && reduction !== 'none') {
    const taskType = normalizeTaskType(configText(config, 'task_type'))
    const pairRoute =
      Boolean(peptideStream || partnerStream) ||
      ['pep_protein_interaction', 'pep_hlaii', 'pep_hlai'].includes(taskType)
    const topicContract = config[`${reduction}_contract`]
    const persistedPairTopic =
      pairRoute &&
      topicContract &&
      typeof topicContract === 'object' &&
      (topicContract as Record<string, unknown>).route === 'pair_two_stream_classic_bow_v2'
    const currentPairTopic = Boolean(
      persistedPairTopic ||
      (pairRoute && isExplicitStrictPairTopicRoute(config, peptideStream, partnerStream)),
    )
    const currentPlsa = pairRoute ? currentPairTopic : Number(config.plsa_contract_version) === 1
    const displayReduction =
      reduction === 'plsa' && !currentPlsa
        ? 'legacy_kl_nmf'
        : pairRoute && reduction === 'lsa'
          ? 'legacy_lsa'
          : pairRoute && reduction === 'lda'
            ? currentPairTopic
              ? reduction
              : 'legacy_lda'
            : reduction
    feature = [feature, compactMethodText(displayReduction)].filter(Boolean).join('-')
  }
  return feature
}

function methodPredictor(config: Record<string, unknown>) {
  const classifier = displayClassifier(config)
  if (configText(config, 'training_mode') === 'export_embeddings' && classifier) {
    return compactMethodText(classifier)
  }
  return compactMethodText(activePredictor(config) || configText(config, 'model'))
}

function activePredictor(config: Record<string, unknown>) {
  const interactionLevel = configText(config, 'interaction_level')
  const predictorType = configText(config, 'predictor_type')
  const interactionModel = configText(config, 'interaction_model')
  const graphPredictor = configText(config, 'graph_predictor')
  const sequenceModel = configText(config, 'sequence_model')
  const classifier = displayClassifier(config)
  const model = configText(config, 'model')

  if (interactionLevel === 'token' && interactionModel) return interactionModel
  if (predictorType === 'sample_graph' && graphPredictor) return graphPredictor
  if (sequenceModel) return sequenceModel
  if (classifier) return classifier
  return model
}

export function modelFeatureSignature(config: Record<string, unknown>, translate?: NameTranslate) {
  config = normalizePairFields(config)
  const streams =
    config.streams && typeof config.streams === 'object' && !Array.isArray(config.streams)
      ? (config.streams as Record<string, unknown>)
      : null
  if (streams) {
    const streamText = (role: string) => {
      const stream =
        streams[role] && typeof streams[role] === 'object'
          ? (streams[role] as Record<string, unknown>)
          : {}
      const encoder = rawText(stream, 'encoder')
      const tokenizerKey = normalizedTokenizer(rawText(stream, 'tokenizer'))
      const name = tokenizerDisplayName(stream)
      if (PRETRAINED_TOKENIZERS.has(tokenizerKey)) return `${name} embeddings`
      return [name, encoder && encoder !== 'ppi_fusion' ? encoder : ''].filter(Boolean).join('/')
    }
    const peptide = streamText('peptide')
    const partner = streamText('partner')
    if (peptide || partner) {
      return translate
        ? translate('generatedNames.pairFeatures', {
            peptide: peptide || '-',
            partner: partner || '-',
          })
        : `Pep[${peptide || '-'}] + Partner[${partner || '-'}]`
    }
  }
  const tokenizer = rawText(config, 'tokenizer')
  const encoder = rawText(config, 'encoder')
  const tokenizerKey = tokenizer.toLowerCase()

  if (!tokenizer && !encoder) return ''
  if (
    PRETRAINED_TOKENIZERS.has(tokenizerKey) &&
    (!encoder || ['index', 'ppi_fusion', tokenizerKey].includes(encoder.toLowerCase()))
  ) {
    const name = tokenizerDisplayName(config)
    if (configText(config, 'sequence_model')) {
      return translate
        ? translate('generatedNames.residueEmbeddings', { name })
        : `${name} residue embeddings`
    }
    return translate ? translate('generatedNames.embeddings', { name }) : `${name} embeddings`
  }
  if (tokenizer && encoder && encoder.toLowerCase() !== 'index') return `${tokenizer}/${encoder}`
  return tokenizer || encoder
}

export function buildModelDefaultName(
  config: Record<string, unknown>,
  taskType?: string,
  translate?: NameTranslate,
  qualifier?: GeneratedNameQualifier,
) {
  const predictor = methodPredictor(config)
  const feature = methodFeatureSignature(config)
  const fallback = translate
    ? translate('generatedNames.model', { task: taskType || '' }).trim()
    : taskType
      ? `${taskType} model`
      : 'Model'

  const base = predictor && feature ? `${feature}-${predictor}` : predictor || feature || fallback
  return qualifier
    ? `${base} · ${translate ? translate(`agent.lanes.${qualifier}`) : titleText(qualifier)}`
    : base
}

export function modelConfigForDisplay(model: ModelRecord, experiment?: ExperimentRecord) {
  if (experiment?.config) return experiment.config
  const args = model.metadata.args
  if (args && typeof args === 'object' && !Array.isArray(args)) {
    return args as Record<string, unknown>
  }
  return {}
}

export function modelRouteSummary(config: Record<string, unknown>, completedEpochs?: number) {
  config = normalizePairFields(config)
  const feature = modelFeatureSignature(config)
  const predictor = titleText(activePredictor(config))
  const pieces = [feature, predictor].filter(Boolean)

  const trainingMode = configText(config, 'training_mode')
  if (trainingMode && trainingMode !== 'end_to_end') pieces.push(trainingMode)

  const epochs = Number(config.epochs)
  if (Number.isFinite(epochs) && usesEpochTraining(config)) {
    if (Number.isFinite(completedEpochs) && Number(completedEpochs) > 0) {
      pieces.push(`${completedEpochs} epochs`)
    } else {
      pieces.push(Number(config.early_stop_patience) > 0 ? `up to ${epochs} epochs` : `${epochs} epochs`)
    }
  } else if (CLASSICAL_CLASSIFIERS.has(effectiveClassifier(config))) {
    pieces.push('single fit')
  }

  return pieces.join(' · ')
}

function formatMetric(value: unknown) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric.toFixed(4) : ''
}

export function modelMetricSummary(experiment?: ExperimentRecord): ModelMetricSummary {
  const summary = experiment?.metrics?.summary ?? {}
  const testF1 = formatMetric(summary.test_f1)
  const testAuc = formatMetric(summary.test_auc)
  if (testF1 || testAuc)
    return { split: 'test', f1: testF1 || undefined, auc: testAuc || undefined }

  const valF1 = formatMetric(summary.val_f1 ?? summary.best_val_f1)
  const valAuc = formatMetric(summary.val_auc ?? summary.best_val_auc)
  if (valF1 || valAuc)
    return { split: 'validation', f1: valF1 || undefined, auc: valAuc || undefined }

  return { split: '' }
}
