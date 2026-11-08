import { normalizeTaskType } from '../taskTerminology'
import { normalizePairStrategy, normalizePairFields } from '../pairTerminology'
import { defineStore } from 'pinia'
import { CLASSICAL_CLASSIFIERS, TRAINABLE_DL_MODELS, usesEpochTraining } from '../experimentRoutes'
import {
  CLASSIFICATION_LOSS_FIELDS,
  PAIR_LOSS_FIELDS,
  SEQUENCE_LOSS_FIELDS,
  classificationLossApplies,
} from '../classificationLoss'
import type {
  ExperimentConfigChange,
  ExperimentDraftHandoff,
  ExperimentRecord,
  GeneratedExperimentName,
  GeneratedNameQualifier,
  PreflightReport,
} from '../types'

interface AgentCandidateHandoffMetadata {
  rationale?: string
  configChanges?: ExperimentConfigChange[]
  planId?: string
  actionId?: string
  lane?: GeneratedNameQualifier
  preflight?: PreflightReport
}

const PAIR_TASKS = new Set(['pep_protein_interaction', 'pep_hlaii', 'pep_hlai'])
const FLATTENABLE_SEQUENCE_ENCODERS = new Set([
  'onehot',
  'blosum62',
  'pssm',
  'pssm_blosum62_hybrid',
  'psfm',
  'psfm_blosum62_hybrid',
  'word2vec',
  'glove',
  'fasttext',
])
const PRETRAINED_MODEL_DEFAULTS: Record<string, string> = {
  esm2: 'facebook/esm2_t30_150M_UR50D',
  protbert: 'Rostlab/prot_bert',
  prott5: 'Rostlab/prot_t5_xl_uniref50',
  esmc: 'EvolutionaryScale/esmc-600m-2024-12',
  esm3: 'EvolutionaryScale/esm3-sm-open-v1',
}
const TOKENIZER_ALIASES: Record<string, string> = {
  probert: 'protbert',
}
const PRETRAINED_TOKENIZERS = new Set(Object.keys(PRETRAINED_MODEL_DEFAULTS))
const SEQUENCE_ARCHITECTURE_KEYS = new Set(['cnn', 'rnn', 'gru', 'lstm', 'transformer'])
const CHAIN_GRAPH_ARCHITECTURE_KEYS = new Set(['gcn', 'gat'])
const MODELING_ARCHITECTURE_KEYS = new Set([
  ...SEQUENCE_ARCHITECTURE_KEYS,
  ...CHAIN_GRAPH_ARCHITECTURE_KEYS,
])
const PAIR_INTERACTION_MODELS = new Set(['cross_attn', 'bi_cross_attn'])
const GRAPH_PREDICTORS = new Set(['gcn', 'gat', 'hgt', 'sage', 'graph_transformer'])
const PAIR_TOPIC_TOKENIZERS = new Set(['kmer', 'dr', 'top-n-gram', 'dt'])
export const PAIR_TOPIC_REDUCTIONS = new Set(['lda', 'plsa'])

function serializedClassifierForTrainingMode(trainingMode: unknown, classifier: string) {
  if (
    String(trainingMode ?? '')
      .trim()
      .toLowerCase() === 'export_embeddings'
  ) {
    return CLASSICAL_CLASSIFIERS.has(classifier) ? classifier : 'svm'
  }
  return 'mlp'
}

export type PairTopicRouteField =
  | 'featureRoute'
  | 'tokenizer'
  | 'encoder'
  | 'streamsLinked'
  | 'partnerFeatureRoute'
  | 'partnerTokenizer'
  | 'partnerEncoder'
  | 'pairStrategy'
  | 'interactionLevel'
  | 'predictorType'
  | 'fusionMode'
  | 'classifier'

export interface PairTopicRouteState {
  featureRoute: string
  tokenizer: string
  encoder: string
  streamsLinked: boolean
  partnerFeatureRoute: string
  partnerTokenizer: string
  partnerEncoder: string
  pairStrategy: string
  interactionLevel: string
  predictorType: string
  fusionMode: string
  classifier: string
}

export interface PairTopicRouteIssue {
  field: PairTopicRouteField
  expected: string
}

export function isPairTopicReduction(value: unknown) {
  return PAIR_TOPIC_REDUCTIONS.has(
    String(value ?? '')
      .trim()
      .toLowerCase(),
  )
}

export function pairTopicRouteIssues(state: PairTopicRouteState): PairTopicRouteIssue[] {
  const issues: PairTopicRouteIssue[] = []
  const primaryIsBow =
    state.featureRoute === 'classic' &&
    PAIR_TOPIC_TOKENIZERS.has(state.tokenizer) &&
    state.encoder === 'bow'
  if (!primaryIsBow) {
    issues.push({
      field: state.encoder === 'bow' ? 'tokenizer' : 'encoder',
      expected: 'classic BoW',
    })
  }

  const partnerIsBow =
    state.streamsLinked ||
    (state.partnerFeatureRoute === 'classic' &&
      PAIR_TOPIC_TOKENIZERS.has(state.partnerTokenizer) &&
      state.partnerEncoder === 'bow')
  if (!partnerIsBow) {
    issues.push({ field: 'partnerEncoder', expected: 'classic BoW' })
  }
  if (state.pairStrategy !== 'two_stream') {
    issues.push({ field: 'pairStrategy', expected: 'two_stream' })
  }
  if (state.interactionLevel !== 'vector') {
    issues.push({ field: 'interactionLevel', expected: 'vector' })
  }
  if (state.predictorType !== 'classifier') {
    issues.push({ field: 'predictorType', expected: 'classifier' })
  }
  if (state.fusionMode !== 'concat') {
    issues.push({ field: 'fusionMode', expected: 'concat' })
  }
  if (!CLASSICAL_CLASSIFIERS.has(state.classifier)) {
    issues.push({ field: 'classifier', expected: 'classical classifier' })
  }
  return issues
}

export function datasetLengthRoleMax(
  metadata: Record<string, unknown> | undefined,
  role: 'sequence' | 'peptide' | 'partner',
) {
  const lengthStats = metadata?.length_stats
  if (!lengthStats || typeof lengthStats !== 'object' || Array.isArray(lengthStats)) return 0
  const overall = (lengthStats as Record<string, unknown>).overall
  if (!overall || typeof overall !== 'object' || Array.isArray(overall)) return 0
  const stats = (overall as Record<string, unknown>)[role]
  if (!stats || typeof stats !== 'object' || Array.isArray(stats)) return 0
  const maximum = Number((stats as Record<string, unknown>).max ?? 0)
  return Number.isFinite(maximum) && maximum > 0 ? Math.floor(maximum) : 0
}

export function datasetLengthRoleMin(
  metadata: Record<string, unknown> | undefined,
  role: 'sequence' | 'peptide' | 'partner',
) {
  const lengthStats = metadata?.length_stats
  if (!lengthStats || typeof lengthStats !== 'object' || Array.isArray(lengthStats)) return 0
  const overall = (lengthStats as Record<string, unknown>).overall
  if (!overall || typeof overall !== 'object' || Array.isArray(overall)) return 0
  const stats = (overall as Record<string, unknown>)[role]
  if (!stats || typeof stats !== 'object' || Array.isArray(stats)) return 0
  const minimum = Number((stats as Record<string, unknown>).min ?? 0)
  return Number.isFinite(minimum) && minimum > 0 ? Math.floor(minimum) : 0
}
const PROFILE_DEPENDENT_TOKENIZERS = new Set(['top-n-gram', 'dt'])
export const HYBRID_PSSM_COVARIANCE_ENCODERS = new Set([
  'ac_pssm_blosum62_hybrid',
  'cc_pssm_blosum62_hybrid',
  'acc_pssm_blosum62_hybrid',
])
const PROFILE_DEPENDENT_ENCODERS = new Set([
  'pssm',
  'pssm_blosum62_hybrid',
  'psfm',
  'psfm_blosum62_hybrid',
  'ac_pssm',
  'cc_pssm',
  'acc_pssm',
  ...HYBRID_PSSM_COVARIANCE_ENCODERS,
])
const PSSM_DEPENDENT_ENCODERS = new Set([
  'pssm',
  'pssm_blosum62_hybrid',
  'ac_pssm',
  'cc_pssm',
  'acc_pssm',
])
const HYBRID_PSSM_COVARIANCE_INDEX_KEYS: Record<string, string> = {
  ac_pssm_blosum62_hybrid: 'ac_pssm_blosum62_hybrid_index_path',
  cc_pssm_blosum62_hybrid: 'cc_pssm_blosum62_hybrid_index_path',
  acc_pssm_blosum62_hybrid: 'acc_pssm_blosum62_hybrid_index_path',
}
export const COVARIANCE_ENCODERS = new Set(['ac', 'cc', 'acc'])
export const PDT_ENCODERS = new Set(['pdt'])
const PDT_DEFAULT_LAG = 8
const PDT_DEFAULT_MAX_RAW_LENGTH = 100000
const GENERAL_HIDDEN_CONFIG_KEYS = new Set([
  'report_aupr',
  'cache_features',
  'checkpoint_threshold_objective',
  'checkpoint_threshold_objective_params',
  'checkpoint_threshold_search_profile',
])
const PROFILE_GENERATION_CONFIG_KEYS = new Set([
  'profile_iterations',
  'profile_evalue',
  'profile_inclusion_evalue',
  'profile_threads',
  'profile_on_missing',
])
const PROFILE_INDEX_CONFIG_KEYS = new Set([
  'pssm_index_path',
  'psfm_index_path',
  'frequency_profile_index_path',
  'ac_pssm_blosum62_hybrid_index_path',
  'cc_pssm_blosum62_hybrid_index_path',
  'acc_pssm_blosum62_hybrid_index_path',
])
const DATASET_SCOPED_CONFIG_KEYS = new Set(['feature_dir'])
const CLONABLE_HIDDEN_CONFIG_KEYS = new Set([
  ...GENERAL_HIDDEN_CONFIG_KEYS,
  ...PROFILE_GENERATION_CONFIG_KEYS,
  ...PROFILE_INDEX_CONFIG_KEYS,
  ...DATASET_SCOPED_CONFIG_KEYS,
])
const CONTROLLED_CONFIG_KEYS = new Set([
  ...CLASSIFICATION_LOSS_FIELDS,
  'task_type',
  'tokenizer',
  'encoder',
  'model',
  'sequence_model',
  'training_mode',
  'classifier',
  'head_hidden_dims',
  'head_activation',
  'head_dropout',
  'residue_adapter_dim',
  'pair_strategy',
  'interaction_level',
  'interaction_model',
  'predictor_type',
  'graph_predictor',
  'fusion_mode',
  'ppi_interaction_mode',
  'hlaii_interaction_mode',
  'k',
  'ac_lag',
  'cc_lag',
  'acc_lag',
  'covariance_max_raw_length',
  'pdt_lag',
  'pdt_max_raw_length',
  'ac_pssm_lag',
  'cc_pssm_lag',
  'acc_pssm_lag',
  'ac_pssm_blosum62_hybrid_lag',
  'cc_pssm_blosum62_hybrid_lag',
  'acc_pssm_blosum62_hybrid_lag',
  'dr_max_distance',
  'dr_max_raw_length',
  'frequency_max_raw_length',
  'dr_dense_memory_limit_gib',
  'top_ngram_n',
  'topngram_dense_memory_limit_gib',
  'dt_max_distance',
  'dt_dense_memory_limit_gib',
  'embedding_vector_size',
  'embedding_window',
  'embedding_min_count',
  'embedding_epochs',
  'embedding_workers',
  'fasttext_bucket',
  'static_embedding_memory_limit_gib',
  'glove_x_max',
  'glove_alpha',
  'glove_learning_rate',
  'glove_batch_size',
  'max_length',
  'batch_size',
  'lr',
  'epochs',
  'dropout',
  'weight_decay',
  'grad_clip_norm',
  'early_stop_patience',
  'early_stop_min_delta',
  'checkpoint_objective',
  'checkpoint_objective_params',
  'threshold_objective',
  'threshold_objective_params',
  'threshold_search_profile',
  'tuning_objective',
  'tuning_objective_params',
  'reduction',
  'lsa_dim',
  'lsa_dense_memory_limit_gib',
  'lda_topics',
  'lda_dense_memory_limit_gib',
  'plsa_topics',
  'plsa_memory_limit_gib',
  'plsa_contract_version',
  'ml_scaler',
  'dl_scaler',
  'ml_reduction',
  'ml_reduction_dim',
  'tsvd_dense_memory_limit_gib',
  'pca_dense_memory_limit_gib',
  'kpca_memory_limit_gib',
  'kpca_kernel',
  'kpca_gamma',
  'ml_auto_tune',
  'ml_tuning_budget',
  'ml_tuning_objective',
  'ml_tuning_mode',
  'ml_tuning_max_candidates',
  'ml_cv_folds',
  'svm_c',
  'svm_class_weight',
  'svm_max_iter',
  'svm_tol',
  'svm_kernel',
  'svm_gamma',
  'svm_convergence_policy',
  'svm_calibration',
  'svm_calibration_cv_folds',
  'logreg_c',
  'logreg_class_weight',
  'logreg_max_iter',
  'logreg_convergence_policy',
  'rf_estimators',
  'rf_max_depth',
  'rf_min_samples_leaf',
  'et_estimators',
  'et_max_depth',
  'et_min_samples_leaf',
  'histgb_max_iter',
  'histgb_max_leaf_nodes',
  'histgb_lr',
  'histgb_l2_regularization',
  'xgb_estimators',
  'xgb_max_depth',
  'xgb_lr',
  'xgb_subsample',
  'xgb_colsample',
  'xgb_reg_alpha',
  'xgb_reg_lambda',
  'lgb_estimators',
  'lgb_max_depth',
  'lgb_lr',
  'lgb_subsample',
  'lgb_colsample',
  'lgb_reg_alpha',
  'lgb_reg_lambda',
  'enable_semantic_similarity',
  'semantic_similarity_weight',
  'semantic_loss_type',
  'semantic_positive_partners',
  'semantic_zero_partners',
  'enable_interaction_semantic_similarity',
  'interaction_semantic_mode',
  'interaction_semantic_similarity_weight',
  'hla_context_semantic_margin',
  'hlai_semantic_smoothing_tau',
  'esm2_model',
  'protbert_model',
  'probert_model',
  'prott5_model',
  'esmc_model',
  'esm3_model',
])

function hasConfigValue(config: Record<string, unknown>, key: string) {
  return key in config && config[key] !== null && config[key] !== undefined && config[key] !== ''
}

function configText(config: Record<string, unknown>, key: string, fallback = '') {
  if (!hasConfigValue(config, key)) return fallback
  return String(config[key]).trim()
}

function normalizedConfigText(config: Record<string, unknown>, key: string, fallback = '') {
  return configText(config, key, fallback).toLowerCase()
}

function normalizedTokenizer(value: string) {
  const normalized = value.trim().toLowerCase()
  return TOKENIZER_ALIASES[normalized] ?? normalized
}

function canonicalPretrainedEncoder(tokenizer: string, encoder = '') {
  const normalized = normalizedTokenizer(tokenizer)
  const raw = encoder.trim().toLowerCase()
  if (PRETRAINED_TOKENIZERS.has(normalized) && (!raw || raw === 'index' || raw === 'ppi_fusion')) {
    return normalized
  }
  return raw || normalized
}

function configNumber(config: Record<string, unknown>, key: string, fallback: number) {
  if (!hasConfigValue(config, key)) return fallback
  const value = Number(config[key])
  return Number.isFinite(value) ? value : fallback
}

function configBoolean(config: Record<string, unknown>, key: string, fallback: boolean) {
  if (!hasConfigValue(config, key)) return fallback
  const value = config[key]
  if (typeof value === 'boolean') return value
  if (typeof value === 'number') return value !== 0
  const normalized = String(value).trim().toLowerCase()
  if (['1', 'true', 'yes', 'on'].includes(normalized)) return true
  if (['0', 'false', 'no', 'off'].includes(normalized)) return false
  return fallback
}

function configHeadHiddenLayerCount(config: Record<string, unknown>): 0 | 1 | 2 {
  const value = config.head_hidden_dims
  if (!Array.isArray(value)) return 1
  const parsed = value.map(Number).filter((item) => Number.isInteger(item) && item > 0)
  if (parsed.length !== value.length) return 1
  if (parsed.length === 0) return 0
  return parsed.length === 1 ? 1 : 2
}

export function predictionHeadHiddenDims(layerCount: 0 | 1 | 2) {
  if (layerCount === 0) return []
  return layerCount === 1 ? [128] : [256, 128]
}

export function preflightAllowsTraining(
  datasetId: string,
  preflight: Pick<PreflightReport, 'valid'> | null | undefined,
  busy: boolean,
  handoffReady = true,
) {
  return Boolean(datasetId && preflight?.valid && !busy && handoffReady)
}

export function predictionHeadApplies(config: Record<string, unknown>) {
  const classifier =
    normalizedConfigText(config, 'classifier') || normalizedConfigText(config, 'model')
  const predictorType = normalizedConfigText(config, 'predictor_type')
  return classifier === 'mlp' && predictorType !== 'sample_graph'
}

function clonedConfigExtras(config: Record<string, unknown>) {
  return Object.fromEntries(
    Object.entries(config).filter(
      ([key, value]) =>
        CLONABLE_HIDDEN_CONFIG_KEYS.has(key) &&
        !CONTROLLED_CONFIG_KEYS.has(key) &&
        value !== null &&
        value !== undefined &&
        value !== '',
    ),
  )
}

function defaultInteractionSemanticMode(_taskType: string) {
  return 'profile_jaccard'
}

function normalizedPairModel(model: string) {
  if (model === 'cross_attn+mlp') return 'cross_attn'
  if (model === 'bi_cross_attn+mlp') return 'bi_cross_attn'
  return model
}

function inferredPairStrategy(
  config: Record<string, unknown>,
  model: string,
  sequenceModel: string,
) {
  const explicitStrategy = normalizePairStrategy(config.pair_strategy)
  if (explicitStrategy) return explicitStrategy
  const ppiMode = normalizedConfigText(config, 'ppi_interaction_mode')
  if (ppiMode === 'single') return 'single_stream'
  if (ppiMode === 'dual') return 'two_stream'
  const hlaiiMode = normalizedConfigText(config, 'hlaii_interaction_mode')
  if (hlaiiMode === 'concat') return 'single_stream'
  if (hlaiiMode === 'contact') return 'two_stream'
  if (sequenceModel || SEQUENCE_ARCHITECTURE_KEYS.has(model)) return 'single_stream'
  return 'two_stream'
}

function usesProfileFeatures(tokenizer: string, encoder: string) {
  return PROFILE_DEPENDENT_TOKENIZERS.has(tokenizer) || PROFILE_DEPENDENT_ENCODERS.has(encoder)
}

function configUsesTokenizer(config: Record<string, unknown>, tokenizers: Set<string>) {
  if (tokenizers.has(normalizedConfigText(config, 'tokenizer'))) return true
  const streams = config.streams
  if (!streams || typeof streams !== 'object' || Array.isArray(streams)) return false
  return Object.values(streams).some(
    (stream) =>
      Boolean(stream) &&
      typeof stream === 'object' &&
      !Array.isArray(stream) &&
      tokenizers.has(normalizedConfigText(stream as Record<string, unknown>, 'tokenizer')),
  )
}

function configUsesEncoder(config: Record<string, unknown>, encoders: Set<string>) {
  if (encoders.has(normalizedConfigText(config, 'encoder'))) return true
  const streams = config.streams
  if (!streams || typeof streams !== 'object' || Array.isArray(streams)) return false
  return Object.values(streams).some(
    (stream) =>
      Boolean(stream) &&
      typeof stream === 'object' &&
      !Array.isArray(stream) &&
      encoders.has(normalizedConfigText(stream as Record<string, unknown>, 'encoder')),
  )
}

function needsPssmIndex(tokenizer: string, encoder: string) {
  return PSSM_DEPENDENT_ENCODERS.has(encoder)
}

function hiddenConfigExtrasForRoute(
  extras: Record<string, unknown>,
  route: {
    datasetId: string
    sourceDatasetId: string
    tokenizer: string
    encoder: string
    partnerTokenizer: string
    partnerEncoder: string
    isPairTask: boolean
  },
) {
  const filtered: Record<string, unknown> = {}
  const sameDataset = Boolean(route.sourceDatasetId) && route.datasetId === route.sourceDatasetId
  const profileRoute =
    usesProfileFeatures(route.tokenizer, route.encoder) ||
    (route.isPairTask && usesProfileFeatures(route.partnerTokenizer, route.partnerEncoder))
  const frequencyProfileRoute =
    PROFILE_DEPENDENT_TOKENIZERS.has(route.tokenizer) ||
    (route.isPairTask && PROFILE_DEPENDENT_TOKENIZERS.has(route.partnerTokenizer))

  for (const [key, value] of Object.entries(extras)) {
    if (value === null || value === undefined || value === '') continue
    if (GENERAL_HIDDEN_CONFIG_KEYS.has(key)) {
      filtered[key] = value
    } else if (PROFILE_GENERATION_CONFIG_KEYS.has(key) && profileRoute) {
      filtered[key] = value
    } else if (DATASET_SCOPED_CONFIG_KEYS.has(key) && sameDataset && route.isPairTask) {
      filtered[key] = value
    } else if (
      key === 'pssm_index_path' &&
      sameDataset &&
      needsPssmIndex(route.tokenizer, route.encoder)
    ) {
      filtered[key] = value
    } else if (
      key === 'psfm_index_path' &&
      sameDataset &&
      [route.encoder, route.partnerEncoder].some((encoder) =>
        ['psfm', 'psfm_blosum62_hybrid'].includes(encoder),
      )
    ) {
      filtered[key] = value
    } else if (sameDataset && HYBRID_PSSM_COVARIANCE_INDEX_KEYS[route.encoder] === key) {
      filtered[key] = value
    } else if (key === 'frequency_profile_index_path' && sameDataset && frequencyProfileRoute) {
      filtered[key] = value
    }
  }

  return filtered
}

export const useExperimentDraftStore = defineStore('experimentDraft', {
  state: () => ({
    mode: 'recommended' as 'recommended' | 'custom',
    taskType: 'pep_function',
    datasetId: '',
    cloneSourceExperimentId: '',
    cloneSourceExperimentName: '',
    cloneSourceGeneratedName: null as GeneratedExperimentName | null,
    cloneSourceDatasetId: '',
    handoff: null as ExperimentDraftHandoff | null,
    initialStep: 1,
    clonedConfigExtras: {} as Record<string, unknown>,
    featureRoute: 'classic' as 'classic' | 'pretrained',
    tokenizer: 'kmer',
    encoder: 'index',
    reduction: 'none',
    classicPredictorRoute: 'modeling_architecture' as
      'modeling_architecture' | 'flattened_classifier',
    pretrainedPredictorRoute: 'pooled_classifier' as 'pooled_classifier' | 'residue_modeling',
    modelingArchitecture: 'cnn',
    residueAdapterDim: 128,
    classifier: 'mlp',
    trainingMode: 'end_to_end',
    headHiddenLayerCount: 1 as 0 | 1 | 2,
    headActivation: 'relu' as 'relu' | 'gelu' | 'silu',
    headDropout: 0.0,
    pairStrategy: 'two_stream',
    interactionLevel: 'vector',
    interactionModel: 'cross_attn',
    predictorType: 'classifier',
    graphPredictor: 'gcn',
    fusionMode: 'concat_all',
    streamsLinked: true,
    streamModel: '',
    partnerFeatureRoute: 'pretrained' as 'classic' | 'pretrained',
    partnerTokenizer: 'prott5',
    partnerEncoder: 'prott5',
    partnerStreamModel: '',
    partnerModelSource: PRETRAINED_MODEL_DEFAULTS.prott5,
    partnerMaxLength: 512,
    partnerK: 2,
    partnerAcLag: 10,
    partnerCcLag: 10,
    partnerAccLag: 10,
    partnerPdtLag: PDT_DEFAULT_LAG,
    partnerPdtLagExplicit: false,
    partnerPdtMaxRawLength: PDT_DEFAULT_MAX_RAW_LENGTH,
    partnerAcPssmLag: 10,
    partnerCcPssmLag: 10,
    partnerAccPssmLag: 10,
    partnerAcPssmBlosum62HybridLag: 10,
    partnerCcPssmBlosum62HybridLag: 10,
    partnerAccPssmBlosum62HybridLag: 10,
    partnerCovarianceMaxRawLength: 100000,
    partnerDrMaxDistance: 5,
    partnerDrMaxRawLength: 50000,
    partnerTopNgramN: 2,
    partnerDtMaxDistance: 5,
    partnerDtDenseMemoryLimitGib: 4.0,
    streamAlignmentDim: 256,
    epochs: 10,
    batchSize: 32,
    maxLength: 60,
    esm2Model: PRETRAINED_MODEL_DEFAULTS.esm2,
    protbertModel: PRETRAINED_MODEL_DEFAULTS.protbert,
    prott5Model: PRETRAINED_MODEL_DEFAULTS.prott5,
    esmcModel: PRETRAINED_MODEL_DEFAULTS.esmc,
    esm3Model: PRETRAINED_MODEL_DEFAULTS.esm3,
    learningRate: 0.001,
    k: 2,
    acLag: 10,
    ccLag: 10,
    accLag: 10,
    pdtLag: PDT_DEFAULT_LAG,
    pdtLagExplicit: false,
    pdtMaxRawLength: PDT_DEFAULT_MAX_RAW_LENGTH,
    acPssmLag: 10,
    ccPssmLag: 10,
    accPssmLag: 10,
    acPssmBlosum62HybridLag: 10,
    ccPssmBlosum62HybridLag: 10,
    accPssmBlosum62HybridLag: 10,
    covarianceMaxRawLength: 100000,
    drMaxDistance: 5,
    drMaxRawLength: 50000,
    frequencyMaxRawLength: 50000,
    drDenseMemoryLimitGib: 4.0,
    topNgramN: 2,
    topNgramDenseMemoryLimitGib: 4.0,
    dtMaxDistance: 5,
    dtDenseMemoryLimitGib: 4.0,
    embeddingVectorSize: 128,
    embeddingWindow: 5,
    embeddingMinCount: 1,
    embeddingEpochs: 20,
    embeddingWorkers: 1,
    fasttextBucket: 2000000,
    staticEmbeddingMemoryLimitGib: 4.0,
    gloveXMax: 100.0,
    gloveAlpha: 0.75,
    gloveLearningRate: 0.05,
    gloveBatchSize: 4096,
    enableRegularization: true,
    dropout: 0.0,
    weightDecay: 0.01,
    gradClipNorm: 0.0,
    earlyStopPatience: 0,
    earlyStopMinDelta: 0.0,
    enableSemanticSimilarity: false,
    semanticSimilarityWeight: 0.1,
    semanticLossType: 'legacy',
    classificationLossOverrides: {} as Record<string, unknown>,
    semanticZeroWeight: 0.25,
    semanticPositivePartners: 1 as number | string,
    semanticZeroPartners: 8 as number | string,
    interactionSemanticMode: 'profile_jaccard',
    interactionSemanticSimilarityWeight: 0.1,
    hlaContextSemanticMargin: 0.2,
    hlaiSemanticSmoothingTau: 20.0,
    enableEnsemble: false,
    ensembleMembers: '',
    ensembleMethod: 'probability_mean',
    ensembleWeightSource: 'validation',
    ensembleGridStep: 0.05,
    dlScaler: 'none',
    mlScaler: 'standard',
    mlReduction: 'none',
    mlReductionDim: 128,
    tsvdDenseMemoryLimitGib: 4.0,
    pcaDenseMemoryLimitGib: 4.0,
    kpcaMemoryLimitGib: 4.0,
    kpcaKernel: 'rbf',
    kpcaGamma: 0.1,
    mlAutoTune: false,
    mlTuningBudget: 'quick',
    mlTuningObjective: 'f1',
    checkpointObjective: '',
    thresholdObjective: '',
    checkpointObjectiveParams: {} as Record<string, number>,
    thresholdObjectiveParams: {} as Record<string, number>,
    tuningObjectiveParams: {} as Record<string, number>,
    mlTuningMaxCandidates: 6,
    mlCvFolds: 2,
    lsaDim: 128,
    lsaDenseMemoryLimitGib: 4.0,
    ldaTopics: 50,
    ldaDenseMemoryLimitGib: 4.0,
    plsaTopics: 50,
    plsaMemoryLimitGib: 4.0,
    svmC: 1.0,
    svmClassWeight: 'balanced',
    svmMaxIter: 2000,
    svmTol: 0.001,
    svmKernel: 'linear',
    svmGamma: 'scale',
    svmConvergencePolicy: 'bounded_retry',
    svmCalibration: 'none' as 'none' | 'sigmoid',
    logregC: 1.0,
    logregClassWeight: 'balanced',
    logregMaxIter: 2000,
    logregConvergencePolicy: 'bounded_retry',
    rfEstimators: 300,
    rfMaxDepth: 0,
    rfMinSamplesLeaf: 1,
    etEstimators: 300,
    etMaxDepth: 0,
    etMinSamplesLeaf: 1,
    histgbMaxIter: 300,
    histgbMaxLeafNodes: 31,
    histgbLearningRate: 0.1,
    histgbL2Regularization: 0.0,
    xgbEstimators: 300,
    xgbMaxDepth: 6,
    xgbLearningRate: 0.1,
    xgbSubsample: 1.0,
    xgbColsample: 1.0,
    xgbRegAlpha: 0.0,
    xgbRegLambda: 1.0,
    lgbEstimators: 300,
    lgbMaxDepth: 6,
    lgbLearningRate: 0.1,
    lgbSubsample: 1.0,
    lgbColsample: 1.0,
    lgbRegAlpha: 0.0,
    lgbRegLambda: 0.0,
  }),
  actions: {
    applyPdtDatasetLagDefaults(
      metadata: Record<string, unknown> | undefined,
      isPairTask: boolean,
      defaultLag = PDT_DEFAULT_LAG,
    ) {
      const datasetDefault = (minimum: number) =>
        minimum > 1 ? Math.min(defaultLag, minimum - 1) : defaultLag
      const primaryUsesPdt = this.tokenizer === 'kmer' && PDT_ENCODERS.has(this.encoder)

      if (!isPairTask) {
        if (primaryUsesPdt && !this.pdtLagExplicit) {
          this.pdtLag = datasetDefault(datasetLengthRoleMin(metadata, 'sequence'))
        }
        return
      }

      if (this.streamsLinked && primaryUsesPdt) {
        const peptideMinimum = datasetLengthRoleMin(metadata, 'peptide')
        const partnerMinimum = datasetLengthRoleMin(metadata, 'partner')
        const minimum =
          peptideMinimum > 0 && partnerMinimum > 0 ? Math.min(peptideMinimum, partnerMinimum) : 0
        if (!this.pdtLagExplicit) this.pdtLag = datasetDefault(minimum)
        return
      }

      if (primaryUsesPdt && !this.pdtLagExplicit) {
        this.pdtLag = datasetDefault(datasetLengthRoleMin(metadata, 'peptide'))
      }
      const partnerUsesPdt =
        this.partnerTokenizer === 'kmer' && PDT_ENCODERS.has(this.partnerEncoder)
      if (partnerUsesPdt && !this.partnerPdtLagExplicit) {
        this.partnerPdtLag = datasetDefault(datasetLengthRoleMin(metadata, 'partner'))
      }
    },
    applyProfileTokenizerDatasetLengthFloor(
      metadata: Record<string, unknown> | undefined,
      isPairTask: boolean,
    ) {
      if (!isPairTask) {
        if (!PROFILE_DEPENDENT_TOKENIZERS.has(this.tokenizer)) return
        const required = datasetLengthRoleMax(metadata, 'sequence')
        if (required > this.maxLength) this.maxLength = required
        return
      }
      const peptideRequired = datasetLengthRoleMax(metadata, 'peptide')
      const partnerRequired = datasetLengthRoleMax(metadata, 'partner')
      if (this.streamsLinked && PROFILE_DEPENDENT_TOKENIZERS.has(this.tokenizer)) {
        const required = Math.max(peptideRequired, partnerRequired)
        if (required > this.maxLength) this.maxLength = required
        if (required > this.partnerMaxLength) this.partnerMaxLength = required
        return
      }
      if (PROFILE_DEPENDENT_TOKENIZERS.has(this.tokenizer) && peptideRequired > this.maxLength) {
        this.maxLength = peptideRequired
      }
      if (
        PROFILE_DEPENDENT_TOKENIZERS.has(this.partnerTokenizer) &&
        partnerRequired > this.partnerMaxLength
      ) {
        this.partnerMaxLength = partnerRequired
      }
    },
    resetForTask(taskType: string, options: { preserveClone?: boolean } = {}) {
      taskType = normalizeTaskType(taskType)
      const isPairTask = PAIR_TASKS.has(taskType)
      if (!options.preserveClone) {
        this.cloneSourceExperimentId = ''
        this.cloneSourceExperimentName = ''
        this.cloneSourceGeneratedName = null
        this.cloneSourceDatasetId = ''
        this.handoff = null
        this.initialStep = 1
        this.clonedConfigExtras = {}
      }
      this.taskType = taskType
      this.datasetId = ''
      this.featureRoute = 'classic'
      this.tokenizer = 'kmer'
      this.encoder = 'index'
      this.reduction = 'none'
      this.classicPredictorRoute = 'modeling_architecture'
      this.pretrainedPredictorRoute = 'pooled_classifier'
      this.modelingArchitecture = 'cnn'
      this.residueAdapterDim = 128
      this.classifier = 'mlp'
      this.trainingMode = 'end_to_end'
      this.headHiddenLayerCount = 1
      this.headActivation = 'relu'
      this.headDropout = isPairTask ? 0.2 : 0.0
      this.pairStrategy = 'two_stream'
      this.interactionLevel = 'vector'
      this.interactionModel = 'cross_attn'
      this.predictorType = 'classifier'
      this.graphPredictor = 'gcn'
      this.fusionMode = 'concat_all'
      this.streamsLinked = true
      this.streamModel = ''
      this.partnerFeatureRoute = 'pretrained'
      this.partnerTokenizer = 'prott5'
      this.partnerEncoder = 'prott5'
      this.partnerStreamModel = ''
      this.partnerModelSource = PRETRAINED_MODEL_DEFAULTS.prott5
      this.partnerMaxLength = 512
      this.partnerK = 2
      this.partnerAcLag = 10
      this.partnerCcLag = 10
      this.partnerAccLag = 10
      this.partnerPdtLag = PDT_DEFAULT_LAG
      this.partnerPdtLagExplicit = false
      this.partnerPdtMaxRawLength = PDT_DEFAULT_MAX_RAW_LENGTH
      this.partnerAcPssmLag = 10
      this.partnerCcPssmLag = 10
      this.partnerAccPssmLag = 10
      this.partnerAcPssmBlosum62HybridLag = 10
      this.partnerCcPssmBlosum62HybridLag = 10
      this.partnerAccPssmBlosum62HybridLag = 10
      this.partnerCovarianceMaxRawLength = 100000
      this.partnerDrMaxDistance = 5
      this.partnerDrMaxRawLength = 50000
      this.partnerTopNgramN = 2
      this.partnerDtMaxDistance = 5
      this.partnerDtDenseMemoryLimitGib = 4.0
      this.streamAlignmentDim = 256
      this.epochs = 10
      this.batchSize = 32
      this.maxLength = isPairTask ? 512 : 60
      this.esm2Model = PRETRAINED_MODEL_DEFAULTS.esm2
      this.protbertModel = PRETRAINED_MODEL_DEFAULTS.protbert
      this.prott5Model = PRETRAINED_MODEL_DEFAULTS.prott5
      this.esmcModel = PRETRAINED_MODEL_DEFAULTS.esmc
      this.esm3Model = PRETRAINED_MODEL_DEFAULTS.esm3
      this.learningRate = 0.001
      this.k = 2
      this.acLag = 10
      this.ccLag = 10
      this.accLag = 10
      this.pdtLag = PDT_DEFAULT_LAG
      this.pdtLagExplicit = false
      this.pdtMaxRawLength = PDT_DEFAULT_MAX_RAW_LENGTH
      this.acPssmLag = 10
      this.ccPssmLag = 10
      this.accPssmLag = 10
      this.acPssmBlosum62HybridLag = 10
      this.ccPssmBlosum62HybridLag = 10
      this.accPssmBlosum62HybridLag = 10
      this.covarianceMaxRawLength = 100000
      this.drMaxDistance = 5
      this.drMaxRawLength = 50000
      this.frequencyMaxRawLength = 50000
      this.drDenseMemoryLimitGib = 4.0
      this.topNgramN = 2
      this.topNgramDenseMemoryLimitGib = 4.0
      this.dtMaxDistance = 5
      this.dtDenseMemoryLimitGib = 4.0
      this.enableRegularization = true
      this.dropout = isPairTask ? 0.2 : 0.0
      this.weightDecay = 0.01
      this.gradClipNorm = 0.0
      this.earlyStopPatience = isPairTask ? 5 : 0
      this.earlyStopMinDelta = 0.0
      this.enableSemanticSimilarity = false
      this.semanticSimilarityWeight = 0.1
      this.semanticLossType = 'legacy'
      this.classificationLossOverrides = {}
      this.semanticZeroWeight = 0.25
      this.semanticPositivePartners = 1
      this.semanticZeroPartners = 8
      this.interactionSemanticMode = defaultInteractionSemanticMode(taskType)
      this.interactionSemanticSimilarityWeight = 0.1
      this.hlaContextSemanticMargin = 0.2
      this.hlaiSemanticSmoothingTau = 20.0
      this.dlScaler = isPairTask ? 'standard' : 'none'
      this.mlScaler = 'standard'
      this.mlReduction = 'none'
      this.mlReductionDim = isPairTask ? 256 : 128
      this.tsvdDenseMemoryLimitGib = 4.0
      this.pcaDenseMemoryLimitGib = 4.0
      this.kpcaMemoryLimitGib = 4.0
      this.kpcaKernel = 'rbf'
      this.kpcaGamma = 0.1
      this.mlAutoTune = false
      this.mlTuningBudget = 'quick'
      this.mlTuningObjective = isPairTask ? 'auc' : 'f1'
      this.checkpointObjective = isPairTask ? 'auc' : 'macro_f1'
      this.thresholdObjective = isPairTask ? 'f1' : 'macro_f1'
      this.checkpointObjectiveParams = {}
      this.thresholdObjectiveParams = {}
      this.tuningObjectiveParams = {}
      this.mlTuningMaxCandidates = 6
      this.mlCvFolds = 2
      this.lsaDim = isPairTask ? 256 : 128
      this.lsaDenseMemoryLimitGib = 4.0
      this.ldaTopics = 50
      this.ldaDenseMemoryLimitGib = 4.0
      this.plsaTopics = 50
      this.plsaMemoryLimitGib = 4.0
      this.svmC = 1.0
      this.svmClassWeight = 'balanced'
      this.svmMaxIter = isPairTask ? 3000 : 2000
      this.svmTol = 0.001
      this.svmKernel = 'linear'
      this.svmGamma = 'scale'
      this.svmConvergencePolicy = 'bounded_retry'
      this.svmCalibration = 'none'
      this.logregC = 1.0
      this.logregClassWeight = 'balanced'
      this.logregMaxIter = 2000
      this.logregConvergencePolicy = 'bounded_retry'
      this.rfEstimators = 300
      this.rfMaxDepth = 0
      this.rfMinSamplesLeaf = 1
      this.etEstimators = 300
      this.etMaxDepth = 0
      this.etMinSamplesLeaf = 1
      this.histgbMaxIter = 300
      this.histgbMaxLeafNodes = 31
      this.histgbLearningRate = 0.1
      this.histgbL2Regularization = 0.0
      this.xgbEstimators = 300
      this.xgbMaxDepth = 6
      this.xgbLearningRate = 0.1
      this.xgbSubsample = 1.0
      this.xgbColsample = 1.0
      this.xgbRegAlpha = 0.0
      this.xgbRegLambda = 1.0
      this.lgbEstimators = 300
      this.lgbMaxDepth = 6
      this.lgbLearningRate = 0.1
      this.lgbSubsample = 1.0
      this.lgbColsample = 1.0
      this.lgbRegAlpha = 0.0
      this.lgbRegLambda = 0.0
      if (taskType === 'pep_protein_interaction' || taskType === 'pep_hlai') {
        this.featureRoute = 'pretrained'
        this.tokenizer = 'esm2'
        this.encoder = 'esm2'
      }
    },
    applyAgentCandidate(
      taskType: string,
      datasetId: string,
      config: Record<string, unknown>,
      title = 'Agent design',
      metadata: AgentCandidateHandoffMetadata = {},
    ) {
      this.applyHandoff({
        origin: 'agent',
        task_type: taskType,
        dataset_id: datasetId,
        config,
        title,
        rationale: metadata.rationale || '',
        config_changes: metadata.configChanges || [],
        agent_plan_id: metadata.planId,
        agent_action_id: metadata.actionId,
        agent_lane: metadata.lane,
        agent_preflight: metadata.preflight,
      })
      this.cloneSourceExperimentId = ''
      this.cloneSourceExperimentName = ''
      this.cloneSourceGeneratedName = null
    },
    detachAgentHandoff() {
      if (this.handoff?.origin === 'agent') this.handoff = null
    },
    applyHandoff(handoff: ExperimentDraftHandoff) {
      const syntheticExperiment: ExperimentRecord = {
        id: handoff.source_experiment_id || '',
        dataset_id: handoff.dataset_id,
        name: handoff.source_experiment_name || handoff.title,
        task_type: handoff.task_type,
        config: handoff.config,
        status: 'succeeded',
        log_path: '',
        artifact_dir: '',
        metrics: { events: [], summary: {} },
        model_id: null,
        error: '',
      }
      this.cloneFromExperiment(syntheticExperiment)
      this.handoff = handoff
      this.initialStep = 4
    },
    cloneFromExperiment(experiment: ExperimentRecord) {
      const config = normalizePairFields(experiment.config ?? {})
      const taskType =
        normalizeTaskType(experiment.task_type || normalizedConfigText(config, 'task_type', 'pep_function'))
      const isPairTask = PAIR_TASKS.has(taskType)

      this.resetForTask(taskType, { preserveClone: true })
      this.handoff = null
      this.mode = 'custom'
      this.datasetId = experiment.dataset_id
      this.cloneSourceExperimentId = experiment.id
      this.cloneSourceExperimentName = experiment.name
      this.cloneSourceGeneratedName = experiment.generated_name ?? null
      this.cloneSourceDatasetId = experiment.dataset_id
      this.initialStep = 4
      this.clonedConfigExtras = clonedConfigExtras(config)
      this.classificationLossOverrides = Object.fromEntries(
        (isPairTask ? PAIR_LOSS_FIELDS : SEQUENCE_LOSS_FIELDS)
          .filter((key) => hasConfigValue(config, key))
          .map((key) => [key, config[key]]),
      )

      const rawStreams =
        config.streams && typeof config.streams === 'object' && !Array.isArray(config.streams)
          ? (config.streams as Record<string, unknown>)
          : null
      const peptideStream =
        rawStreams?.peptide && typeof rawStreams.peptide === 'object'
          ? (rawStreams.peptide as Record<string, unknown>)
          : null
      const partnerStream =
        rawStreams?.partner && typeof rawStreams.partner === 'object'
          ? (rawStreams.partner as Record<string, unknown>)
          : null
      const tokenizer = normalizedTokenizer(
        peptideStream
          ? configText(peptideStream, 'tokenizer', this.tokenizer)
          : configText(config, 'tokenizer', this.tokenizer),
      )
      this.featureRoute = PRETRAINED_TOKENIZERS.has(tokenizer) ? 'pretrained' : 'classic'
      this.tokenizer = tokenizer
      this.encoder = configText(
        peptideStream ?? config,
        'encoder',
        this.featureRoute === 'pretrained' ? this.tokenizer : 'index',
      ).toLowerCase()
      if (this.featureRoute === 'pretrained') {
        this.encoder = canonicalPretrainedEncoder(this.tokenizer, this.encoder)
      }
      this.reduction = normalizedConfigText(config, 'reduction', this.reduction)
      if (PAIR_TASKS.has(this.taskType) && this.reduction === 'lsa') {
        this.reduction = 'none'
      }

      this.esm2Model = configText(config, 'esm2_model', PRETRAINED_MODEL_DEFAULTS.esm2)
      this.protbertModel = configText(
        config,
        'protbert_model',
        configText(config, 'probert_model', PRETRAINED_MODEL_DEFAULTS.protbert),
      )
      this.prott5Model = configText(config, 'prott5_model', PRETRAINED_MODEL_DEFAULTS.prott5)
      this.esmcModel = configText(config, 'esmc_model', PRETRAINED_MODEL_DEFAULTS.esmc)
      this.esm3Model = configText(config, 'esm3_model', PRETRAINED_MODEL_DEFAULTS.esm3)
      if (peptideStream && this.featureRoute === 'pretrained') {
        const source = configText(
          peptideStream,
          'model_source',
          PRETRAINED_MODEL_DEFAULTS[this.tokenizer] ?? '',
        )
        if (this.tokenizer === 'esm2') this.esm2Model = source
        if (this.tokenizer === 'protbert') this.protbertModel = source
        if (this.tokenizer === 'prott5') this.prott5Model = source
        if (this.tokenizer === 'esmc') this.esmcModel = source
        if (this.tokenizer === 'esm3') this.esm3Model = source
      }
      if (peptideStream) this.streamModel = configText(peptideStream, 'stream_model')
      if (partnerStream) {
        this.partnerTokenizer = normalizedTokenizer(
          configText(partnerStream, 'tokenizer', this.partnerTokenizer),
        )
        this.partnerFeatureRoute = PRETRAINED_TOKENIZERS.has(this.partnerTokenizer)
          ? 'pretrained'
          : 'classic'
        this.partnerEncoder = configText(
          partnerStream,
          'encoder',
          this.partnerFeatureRoute === 'pretrained' ? this.partnerTokenizer : 'index',
        ).toLowerCase()
        if (this.partnerFeatureRoute === 'pretrained') {
          this.partnerEncoder = canonicalPretrainedEncoder(
            this.partnerTokenizer,
            this.partnerEncoder,
          )
        }
        this.partnerStreamModel = configText(partnerStream, 'stream_model')
        this.partnerModelSource = configText(
          partnerStream,
          'model_source',
          PRETRAINED_MODEL_DEFAULTS[this.partnerTokenizer] ?? '',
        )
        this.streamsLinked = JSON.stringify(peptideStream) === JSON.stringify(partnerStream)
      }

      const model = normalizedConfigText(config, 'model')
      const pairModel = normalizedPairModel(model)
      const sequenceModel = normalizedConfigText(config, 'sequence_model')
      const classifierFromConfig = normalizedConfigText(config, 'classifier')
      const legacyCalibratedSvm =
        model === 'calibrated_svm' || classifierFromConfig === 'calibrated_svm'
      const effectiveClassifier = legacyCalibratedSvm
        ? 'svm'
        : classifierFromConfig ||
          (CLASSICAL_CLASSIFIERS.has(model) || model === 'mlp' ? model : this.classifier)

      if (!isPairTask) {
        if (sequenceModel) {
          this.classicPredictorRoute = 'modeling_architecture'
          this.modelingArchitecture = sequenceModel
          this.trainingMode = normalizedConfigText(config, 'training_mode', 'end_to_end')
          this.classifier = effectiveClassifier || 'mlp'
          this.pretrainedPredictorRoute =
            this.featureRoute === 'pretrained' && MODELING_ARCHITECTURE_KEYS.has(sequenceModel)
              ? 'residue_modeling'
              : 'pooled_classifier'
        } else {
          this.classicPredictorRoute =
            this.featureRoute === 'classic' &&
            FLATTENABLE_SEQUENCE_ENCODERS.has(this.encoder) &&
            CLASSICAL_CLASSIFIERS.has(effectiveClassifier)
              ? 'flattened_classifier'
              : 'modeling_architecture'
          this.pretrainedPredictorRoute = 'pooled_classifier'
          this.classifier = effectiveClassifier || this.classifier
          if (MODELING_ARCHITECTURE_KEYS.has(model)) {
            this.modelingArchitecture = model
            if (this.featureRoute === 'pretrained' && MODELING_ARCHITECTURE_KEYS.has(model)) {
              this.pretrainedPredictorRoute = 'residue_modeling'
              this.trainingMode = normalizedConfigText(config, 'training_mode', 'end_to_end')
              this.classifier = effectiveClassifier || 'mlp'
            }
          }
        }
      } else {
        const inferredSequenceModel =
          sequenceModel || (SEQUENCE_ARCHITECTURE_KEYS.has(pairModel) ? pairModel : '')
        const inferredInteractionModel =
          normalizedConfigText(config, 'interaction_model') ||
          (PAIR_INTERACTION_MODELS.has(pairModel) ? pairModel : '')
        const inferredGraphPredictor =
          normalizedConfigText(config, 'graph_predictor') ||
          (GRAPH_PREDICTORS.has(pairModel) ? pairModel : '')
        this.pairStrategy = inferredPairStrategy(config, pairModel, inferredSequenceModel)
        if (this.pairStrategy === 'single_stream') {
          this.modelingArchitecture = inferredSequenceModel || this.modelingArchitecture
          this.trainingMode = normalizedConfigText(config, 'training_mode', 'end_to_end')
          this.classifier = effectiveClassifier || 'mlp'
        } else {
          this.interactionLevel = normalizedConfigText(
            config,
            'interaction_level',
            inferredInteractionModel ? 'token' : 'vector',
          )
          if (this.interactionLevel === 'token') {
            this.interactionModel = inferredInteractionModel || this.interactionModel
            this.trainingMode = normalizedConfigText(config, 'training_mode', 'end_to_end')
            this.classifier = effectiveClassifier || 'mlp'
          } else {
            this.predictorType = normalizedConfigText(
              config,
              'predictor_type',
              inferredGraphPredictor ? 'sample_graph' : 'classifier',
            )
            this.graphPredictor = inferredGraphPredictor || this.graphPredictor
            this.classifier = effectiveClassifier || 'mlp'
          }
        }
        this.fusionMode = normalizedConfigText(config, 'fusion_mode', this.fusionMode)
      }

      this.headHiddenLayerCount = configHeadHiddenLayerCount(config)
      const headActivation = normalizedConfigText(config, 'head_activation', 'relu')
      this.headActivation = ['relu', 'gelu', 'silu'].includes(headActivation)
        ? (headActivation as 'relu' | 'gelu' | 'silu')
        : 'relu'
      this.headDropout = configNumber(
        config,
        'head_dropout',
        configNumber(config, 'dropout', this.headDropout),
      )
      this.residueAdapterDim = configNumber(config, 'residue_adapter_dim', 128)

      this.epochs = configNumber(config, 'epochs', this.epochs)
      this.batchSize = configNumber(config, 'batch_size', this.batchSize)
      this.maxLength = configNumber(config, 'max_length', this.maxLength)
      this.learningRate = configNumber(config, 'lr', this.learningRate)
      this.k = configNumber(config, 'k', this.k)
      this.acLag = configNumber(config, 'ac_lag', this.acLag)
      this.ccLag = configNumber(config, 'cc_lag', this.ccLag)
      this.accLag = configNumber(config, 'acc_lag', this.accLag)
      this.pdtLag = configNumber(config, 'pdt_lag', this.pdtLag)
      this.pdtLagExplicit = hasConfigValue(config, 'pdt_lag')
      this.pdtMaxRawLength = configNumber(config, 'pdt_max_raw_length', this.pdtMaxRawLength)
      this.acPssmLag = configNumber(config, 'ac_pssm_lag', this.acPssmLag)
      this.ccPssmLag = configNumber(config, 'cc_pssm_lag', this.ccPssmLag)
      this.accPssmLag = configNumber(config, 'acc_pssm_lag', this.accPssmLag)
      this.acPssmBlosum62HybridLag = configNumber(
        config,
        'ac_pssm_blosum62_hybrid_lag',
        this.acPssmBlosum62HybridLag,
      )
      this.ccPssmBlosum62HybridLag = configNumber(
        config,
        'cc_pssm_blosum62_hybrid_lag',
        this.ccPssmBlosum62HybridLag,
      )
      this.accPssmBlosum62HybridLag = configNumber(
        config,
        'acc_pssm_blosum62_hybrid_lag',
        this.accPssmBlosum62HybridLag,
      )
      this.covarianceMaxRawLength = configNumber(
        config,
        'covariance_max_raw_length',
        this.covarianceMaxRawLength,
      )
      this.drMaxDistance = configNumber(config, 'dr_max_distance', this.drMaxDistance)
      this.drMaxRawLength = configNumber(config, 'dr_max_raw_length', this.drMaxRawLength)
      this.frequencyMaxRawLength = configNumber(
        config,
        'frequency_max_raw_length',
        this.frequencyMaxRawLength,
      )
      this.drDenseMemoryLimitGib = configNumber(
        config,
        'dr_dense_memory_limit_gib',
        this.drDenseMemoryLimitGib,
      )
      this.topNgramN = configNumber(config, 'top_ngram_n', this.topNgramN)
      this.topNgramDenseMemoryLimitGib = configNumber(
        config,
        'topngram_dense_memory_limit_gib',
        this.topNgramDenseMemoryLimitGib,
      )
      this.dtMaxDistance = configNumber(config, 'dt_max_distance', this.dtMaxDistance)
      this.dtDenseMemoryLimitGib = configNumber(
        config,
        'dt_dense_memory_limit_gib',
        this.dtDenseMemoryLimitGib,
      )
      this.embeddingVectorSize = configNumber(
        config,
        'embedding_vector_size',
        this.embeddingVectorSize,
      )
      this.embeddingWindow = configNumber(config, 'embedding_window', this.embeddingWindow)
      this.embeddingMinCount = configNumber(config, 'embedding_min_count', this.embeddingMinCount)
      this.embeddingEpochs = configNumber(config, 'embedding_epochs', this.embeddingEpochs)
      this.embeddingWorkers = configNumber(config, 'embedding_workers', this.embeddingWorkers)
      this.fasttextBucket = configNumber(config, 'fasttext_bucket', this.fasttextBucket)
      this.staticEmbeddingMemoryLimitGib = configNumber(
        config,
        'static_embedding_memory_limit_gib',
        this.staticEmbeddingMemoryLimitGib,
      )
      this.gloveXMax = configNumber(config, 'glove_x_max', this.gloveXMax)
      this.gloveAlpha = configNumber(config, 'glove_alpha', this.gloveAlpha)
      this.gloveLearningRate = configNumber(config, 'glove_learning_rate', this.gloveLearningRate)
      this.gloveBatchSize = configNumber(config, 'glove_batch_size', this.gloveBatchSize)
      if (peptideStream) {
        const params =
          peptideStream.params && typeof peptideStream.params === 'object'
            ? (peptideStream.params as Record<string, unknown>)
            : {}
        this.maxLength = configNumber(peptideStream, 'max_length', this.maxLength)
        this.k = configNumber(params, 'k', this.k)
        this.acLag = configNumber(params, 'ac_lag', this.acLag)
        this.ccLag = configNumber(params, 'cc_lag', this.ccLag)
        this.accLag = configNumber(params, 'acc_lag', this.accLag)
        this.pdtLag = configNumber(params, 'pdt_lag', this.pdtLag)
        this.pdtLagExplicit = hasConfigValue(params, 'pdt_lag')
        this.pdtMaxRawLength = configNumber(params, 'pdt_max_raw_length', this.pdtMaxRawLength)
        this.acPssmLag = configNumber(params, 'ac_pssm_lag', this.acPssmLag)
        this.ccPssmLag = configNumber(params, 'cc_pssm_lag', this.ccPssmLag)
        this.accPssmLag = configNumber(params, 'acc_pssm_lag', this.accPssmLag)
        this.acPssmBlosum62HybridLag = configNumber(
          params,
          'ac_pssm_blosum62_hybrid_lag',
          this.acPssmBlosum62HybridLag,
        )
        this.ccPssmBlosum62HybridLag = configNumber(
          params,
          'cc_pssm_blosum62_hybrid_lag',
          this.ccPssmBlosum62HybridLag,
        )
        this.accPssmBlosum62HybridLag = configNumber(
          params,
          'acc_pssm_blosum62_hybrid_lag',
          this.accPssmBlosum62HybridLag,
        )
        this.covarianceMaxRawLength = configNumber(
          params,
          'covariance_max_raw_length',
          this.covarianceMaxRawLength,
        )
        this.drMaxDistance = configNumber(params, 'dr_max_distance', this.drMaxDistance)
        this.drMaxRawLength = configNumber(params, 'dr_max_raw_length', this.drMaxRawLength)
        this.frequencyMaxRawLength = configNumber(
          params,
          'frequency_max_raw_length',
          this.frequencyMaxRawLength,
        )
        this.topNgramN = configNumber(params, 'top_ngram_n', this.topNgramN)
        this.topNgramDenseMemoryLimitGib = configNumber(
          params,
          'topngram_dense_memory_limit_gib',
          this.topNgramDenseMemoryLimitGib,
        )
        this.dtMaxDistance = configNumber(params, 'dt_max_distance', this.dtMaxDistance)
        this.dtDenseMemoryLimitGib = configNumber(
          params,
          'dt_dense_memory_limit_gib',
          this.dtDenseMemoryLimitGib,
        )
      }
      if (partnerStream) {
        const params =
          partnerStream.params && typeof partnerStream.params === 'object'
            ? (partnerStream.params as Record<string, unknown>)
            : {}
        this.partnerMaxLength = configNumber(partnerStream, 'max_length', this.partnerMaxLength)
        this.partnerK = configNumber(params, 'k', this.partnerK)
        this.partnerAcLag = configNumber(params, 'ac_lag', this.partnerAcLag)
        this.partnerCcLag = configNumber(params, 'cc_lag', this.partnerCcLag)
        this.partnerAccLag = configNumber(params, 'acc_lag', this.partnerAccLag)
        this.partnerPdtLag = configNumber(params, 'pdt_lag', this.partnerPdtLag)
        this.partnerPdtLagExplicit = hasConfigValue(params, 'pdt_lag')
        this.partnerPdtMaxRawLength = configNumber(
          params,
          'pdt_max_raw_length',
          this.partnerPdtMaxRawLength,
        )
        this.partnerAcPssmLag = configNumber(params, 'ac_pssm_lag', this.partnerAcPssmLag)
        this.partnerCcPssmLag = configNumber(params, 'cc_pssm_lag', this.partnerCcPssmLag)
        this.partnerAccPssmLag = configNumber(params, 'acc_pssm_lag', this.partnerAccPssmLag)
        this.partnerAcPssmBlosum62HybridLag = configNumber(
          params,
          'ac_pssm_blosum62_hybrid_lag',
          this.partnerAcPssmBlosum62HybridLag,
        )
        this.partnerCcPssmBlosum62HybridLag = configNumber(
          params,
          'cc_pssm_blosum62_hybrid_lag',
          this.partnerCcPssmBlosum62HybridLag,
        )
        this.partnerAccPssmBlosum62HybridLag = configNumber(
          params,
          'acc_pssm_blosum62_hybrid_lag',
          this.partnerAccPssmBlosum62HybridLag,
        )
        this.partnerCovarianceMaxRawLength = configNumber(
          params,
          'covariance_max_raw_length',
          this.partnerCovarianceMaxRawLength,
        )
        this.partnerDrMaxDistance = configNumber(
          params,
          'dr_max_distance',
          this.partnerDrMaxDistance,
        )
        this.partnerDrMaxRawLength = configNumber(
          params,
          'dr_max_raw_length',
          this.partnerDrMaxRawLength,
        )
        this.partnerTopNgramN = configNumber(params, 'top_ngram_n', this.partnerTopNgramN)
        this.topNgramDenseMemoryLimitGib = configNumber(
          params,
          'topngram_dense_memory_limit_gib',
          this.topNgramDenseMemoryLimitGib,
        )
        this.partnerDtMaxDistance = configNumber(
          params,
          'dt_max_distance',
          this.partnerDtMaxDistance,
        )
        this.partnerDtDenseMemoryLimitGib = configNumber(
          params,
          'dt_dense_memory_limit_gib',
          this.partnerDtDenseMemoryLimitGib,
        )
      }
      this.streamAlignmentDim = configNumber(config, 'stream_alignment_dim', this.streamAlignmentDim)
      this.dropout = configNumber(config, 'dropout', this.dropout)
      this.weightDecay = configNumber(config, 'weight_decay', this.weightDecay)
      this.gradClipNorm = configNumber(config, 'grad_clip_norm', this.gradClipNorm)
      this.earlyStopPatience = configNumber(config, 'early_stop_patience', this.earlyStopPatience)
      this.earlyStopMinDelta = configNumber(config, 'early_stop_min_delta', this.earlyStopMinDelta)
      this.dlScaler = normalizedConfigText(config, 'dl_scaler', this.dlScaler)
      const ensembleMembersRaw = config.ensemble_members
      this.enableEnsemble = Array.isArray(ensembleMembersRaw)
        ? ensembleMembersRaw.length >= 2
        : String(ensembleMembersRaw ?? '')
            .split(',')
            .filter((value) => value.trim()).length >= 2
      this.ensembleMembers = Array.isArray(ensembleMembersRaw)
        ? ensembleMembersRaw.map(String).join(', ')
        : String(ensembleMembersRaw ?? '')
      this.ensembleMethod = normalizedConfigText(config, 'ensemble_method', 'probability_mean')
      this.ensembleWeightSource = normalizedConfigText(
        config,
        'ensemble_weight_source',
        'validation',
      )
      this.ensembleGridStep = configNumber(config, 'ensemble_grid_step', 0.05)
      this.mlScaler = normalizedConfigText(config, 'ml_scaler', this.mlScaler)
      this.mlReduction = normalizedConfigText(config, 'ml_reduction', this.mlReduction)
      this.mlReductionDim = configNumber(config, 'ml_reduction_dim', this.mlReductionDim)
      this.tsvdDenseMemoryLimitGib = configNumber(
        config,
        'tsvd_dense_memory_limit_gib',
        this.tsvdDenseMemoryLimitGib,
      )
      this.pcaDenseMemoryLimitGib = configNumber(
        config,
        'pca_dense_memory_limit_gib',
        this.pcaDenseMemoryLimitGib,
      )
      this.kpcaMemoryLimitGib = configNumber(
        config,
        'kpca_memory_limit_gib',
        this.kpcaMemoryLimitGib,
      )
      this.kpcaKernel = normalizedConfigText(config, 'kpca_kernel', this.kpcaKernel)
      this.kpcaGamma = configNumber(config, 'kpca_gamma', this.kpcaGamma)
      this.mlAutoTune = configBoolean(config, 'ml_auto_tune', this.mlAutoTune)
      this.mlTuningBudget = normalizedConfigText(config, 'ml_tuning_budget', this.mlTuningBudget)
      this.mlTuningObjective = normalizedConfigText(
        config,
        'tuning_objective',
        normalizedConfigText(config, 'ml_tuning_objective', this.mlTuningObjective),
      )
      this.checkpointObjective = normalizedConfigText(
        config,
        'checkpoint_objective',
        this.checkpointObjective,
      )
      this.thresholdObjective = normalizedConfigText(
        config,
        'threshold_objective',
        this.thresholdObjective,
      )
      for (const [field, key] of [
        ['checkpointObjectiveParams', 'checkpoint_objective_params'],
        ['thresholdObjectiveParams', 'threshold_objective_params'],
        ['tuningObjectiveParams', 'tuning_objective_params'],
      ] as const) {
        const value = config[key]
        this[field] =
          value && typeof value === 'object' && !Array.isArray(value)
            ? { ...(value as Record<string, number>) }
            : {}
      }
      this.mlTuningMaxCandidates = configNumber(
        config,
        'ml_tuning_max_candidates',
        this.mlTuningMaxCandidates,
      )
      this.mlCvFolds = configNumber(config, 'ml_cv_folds', this.mlCvFolds)
      this.lsaDim = configNumber(config, 'lsa_dim', this.lsaDim)
      this.lsaDenseMemoryLimitGib = configNumber(
        config,
        'lsa_dense_memory_limit_gib',
        this.lsaDenseMemoryLimitGib,
      )
      this.ldaTopics = configNumber(config, 'lda_topics', this.ldaTopics)
      this.ldaDenseMemoryLimitGib = configNumber(
        config,
        'lda_dense_memory_limit_gib',
        this.ldaDenseMemoryLimitGib,
      )
      this.plsaTopics = configNumber(config, 'plsa_topics', this.plsaTopics)
      this.plsaMemoryLimitGib = configNumber(
        config,
        'plsa_memory_limit_gib',
        this.plsaMemoryLimitGib,
      )

      this.svmC = configNumber(config, 'svm_c', this.svmC)
      this.svmClassWeight = normalizedConfigText(config, 'svm_class_weight', this.svmClassWeight)
      this.svmMaxIter = configNumber(config, 'svm_max_iter', this.svmMaxIter)
      this.svmTol = configNumber(config, 'svm_tol', this.svmTol)
      this.svmKernel = normalizedConfigText(config, 'svm_kernel', this.svmKernel)
      this.svmGamma = configText(config, 'svm_gamma', this.svmGamma)
      this.svmConvergencePolicy = normalizedConfigText(
        config,
        'svm_convergence_policy',
        this.svmConvergencePolicy,
      )
      const svmCalibration = legacyCalibratedSvm
        ? 'sigmoid'
        : normalizedConfigText(config, 'svm_calibration', this.svmCalibration)
      this.svmCalibration = svmCalibration === 'sigmoid' ? 'sigmoid' : 'none'
      this.logregC = configNumber(config, 'logreg_c', this.logregC)
      this.logregClassWeight = normalizedConfigText(
        config,
        'logreg_class_weight',
        this.logregClassWeight,
      )
      this.logregMaxIter = configNumber(config, 'logreg_max_iter', this.logregMaxIter)
      this.logregConvergencePolicy = normalizedConfigText(
        config,
        'logreg_convergence_policy',
        this.logregConvergencePolicy,
      )
      this.rfEstimators = configNumber(config, 'rf_estimators', this.rfEstimators)
      this.rfMaxDepth = configNumber(config, 'rf_max_depth', this.rfMaxDepth)
      this.rfMinSamplesLeaf = configNumber(config, 'rf_min_samples_leaf', this.rfMinSamplesLeaf)
      this.etEstimators = configNumber(config, 'et_estimators', this.etEstimators)
      this.etMaxDepth = configNumber(config, 'et_max_depth', this.etMaxDepth)
      this.etMinSamplesLeaf = configNumber(config, 'et_min_samples_leaf', this.etMinSamplesLeaf)
      this.histgbMaxIter = configNumber(config, 'histgb_max_iter', this.histgbMaxIter)
      this.histgbMaxLeafNodes = configNumber(
        config,
        'histgb_max_leaf_nodes',
        this.histgbMaxLeafNodes,
      )
      this.histgbLearningRate = configNumber(config, 'histgb_lr', this.histgbLearningRate)
      this.histgbL2Regularization = configNumber(
        config,
        'histgb_l2_regularization',
        this.histgbL2Regularization,
      )
      this.xgbEstimators = configNumber(config, 'xgb_estimators', this.xgbEstimators)
      this.xgbMaxDepth = configNumber(config, 'xgb_max_depth', this.xgbMaxDepth)
      this.xgbLearningRate = configNumber(config, 'xgb_lr', this.xgbLearningRate)
      this.xgbSubsample = configNumber(config, 'xgb_subsample', this.xgbSubsample)
      this.xgbColsample = configNumber(config, 'xgb_colsample', this.xgbColsample)
      this.xgbRegAlpha = configNumber(config, 'xgb_reg_alpha', this.xgbRegAlpha)
      this.xgbRegLambda = configNumber(config, 'xgb_reg_lambda', this.xgbRegLambda)
      this.lgbEstimators = configNumber(config, 'lgb_estimators', this.lgbEstimators)
      this.lgbMaxDepth = configNumber(config, 'lgb_max_depth', this.lgbMaxDepth)
      this.lgbLearningRate = configNumber(config, 'lgb_lr', this.lgbLearningRate)
      this.lgbSubsample = configNumber(config, 'lgb_subsample', this.lgbSubsample)
      this.lgbColsample = configNumber(config, 'lgb_colsample', this.lgbColsample)
      this.lgbRegAlpha = configNumber(config, 'lgb_reg_alpha', this.lgbRegAlpha)
      this.lgbRegLambda = configNumber(config, 'lgb_reg_lambda', this.lgbRegLambda)

      this.enableSemanticSimilarity = configBoolean(
        config,
        'enable_semantic_similarity',
        configBoolean(
          config,
          'enable_interaction_semantic_similarity',
          this.enableSemanticSimilarity,
        ),
      )
      this.semanticSimilarityWeight = configNumber(
        config,
        'semantic_similarity_weight',
        configNumber(
          config,
          'interaction_semantic_similarity_weight',
          this.semanticSimilarityWeight,
        ),
      )
      this.semanticZeroWeight = configNumber(
        config,
        'semantic_zero_weight',
        this.semanticZeroWeight,
      )
      // A historical config must not inherit a previously selected new loss.
      this.semanticLossType = normalizedConfigText(config, 'semantic_loss_type', 'legacy')
      this.semanticPositivePartners = configNumber(config, 'semantic_positive_partners', 1)
      this.semanticZeroPartners = configNumber(config, 'semantic_zero_partners', 8)
      this.interactionSemanticMode = normalizedConfigText(
        config,
        'interaction_semantic_mode',
        defaultInteractionSemanticMode(taskType),
      )
      this.interactionSemanticSimilarityWeight = configNumber(
        config,
        'interaction_semantic_similarity_weight',
        this.interactionSemanticSimilarityWeight,
      )
      this.hlaContextSemanticMargin = configNumber(
        config,
        'hla_context_semantic_margin',
        this.hlaContextSemanticMargin,
      )
      this.hlaiSemanticSmoothingTau = configNumber(
        config,
        'hlai_semantic_smoothing_tau',
        this.hlaiSemanticSmoothingTau,
      )

      const dlRegFields = [
        'dropout',
        'head_dropout',
        'weight_decay',
        'grad_clip_norm',
        'early_stop_patience',
        'early_stop_min_delta',
      ]
      const hasDlRegFields = dlRegFields.some((key) => hasConfigValue(config, key))
      const dlRegDisabled =
        hasDlRegFields &&
        this.dropout === 0 &&
        this.headDropout === 0 &&
        this.weightDecay === 0 &&
        this.gradClipNorm === 0 &&
        this.earlyStopPatience === 0 &&
        this.earlyStopMinDelta === 0
      const classWeightDisabled =
        (effectiveClassifier === 'svm' && this.svmClassWeight === 'none') ||
        (effectiveClassifier === 'logreg' && this.logregClassWeight === 'none')
      const xgbRegDisabled =
        effectiveClassifier === 'xgb' &&
        this.xgbSubsample === 1 &&
        this.xgbColsample === 1 &&
        this.xgbRegAlpha === 0 &&
        this.xgbRegLambda === 1
      const lgbRegDisabled =
        effectiveClassifier === 'lgb' &&
        this.lgbSubsample === 1 &&
        this.lgbColsample === 1 &&
        this.lgbRegAlpha === 0 &&
        this.lgbRegLambda === 0
      const histgbRegDisabled =
        effectiveClassifier === 'histgb' && this.histgbL2Regularization === 0
      if (
        dlRegDisabled ||
        classWeightDisabled ||
        xgbRegDisabled ||
        lgbRegDisabled ||
        histgbRegDisabled
      ) {
        this.enableRegularization = false
      }
    },
    buildStreamSpec(role: 'peptide' | 'partner') {
      const partner = role === 'partner'
      const featureRoute = partner ? this.partnerFeatureRoute : this.featureRoute
      const tokenizer = partner ? this.partnerTokenizer : this.tokenizer
      const encoder =
        featureRoute === 'pretrained'
          ? canonicalPretrainedEncoder(tokenizer, tokenizer)
          : partner
            ? this.partnerEncoder
            : this.encoder
      const params: Record<string, unknown> = {}
      if (tokenizer === 'kmer') {
        params.k = partner ? this.partnerK : this.k
      }
      if (COVARIANCE_ENCODERS.has(encoder)) {
        params.k = 1
        if (encoder === 'ac') params.ac_lag = partner ? this.partnerAcLag : this.acLag
        if (encoder === 'cc') params.cc_lag = partner ? this.partnerCcLag : this.ccLag
        if (encoder === 'acc') params.acc_lag = partner ? this.partnerAccLag : this.accLag
        params.covariance_max_raw_length = partner
          ? this.partnerCovarianceMaxRawLength
          : this.covarianceMaxRawLength
      }
      if (PDT_ENCODERS.has(encoder)) {
        params.k = 1
        params.pdt_lag = partner ? this.partnerPdtLag : this.pdtLag
        params.pdt_max_raw_length = partner ? this.partnerPdtMaxRawLength : this.pdtMaxRawLength
      }
      if (PROFILE_DEPENDENT_ENCODERS.has(encoder)) {
        params.k = 1
      }
      if (encoder === 'ac_pssm') {
        params.ac_pssm_lag = partner ? this.partnerAcPssmLag : this.acPssmLag
      }
      if (encoder === 'cc_pssm') {
        params.cc_pssm_lag = partner ? this.partnerCcPssmLag : this.ccPssmLag
      }
      if (encoder === 'acc_pssm') {
        params.acc_pssm_lag = partner ? this.partnerAccPssmLag : this.accPssmLag
      }
      if (encoder === 'ac_pssm_blosum62_hybrid') {
        params.ac_pssm_blosum62_hybrid_lag = partner
          ? this.partnerAcPssmBlosum62HybridLag
          : this.acPssmBlosum62HybridLag
      }
      if (encoder === 'cc_pssm_blosum62_hybrid') {
        params.cc_pssm_blosum62_hybrid_lag = partner
          ? this.partnerCcPssmBlosum62HybridLag
          : this.ccPssmBlosum62HybridLag
      }
      if (encoder === 'acc_pssm_blosum62_hybrid') {
        params.acc_pssm_blosum62_hybrid_lag = partner
          ? this.partnerAccPssmBlosum62HybridLag
          : this.accPssmBlosum62HybridLag
      }
      if (tokenizer === 'dr') {
        params.dr_max_distance = partner ? this.partnerDrMaxDistance : this.drMaxDistance
        params.dr_max_raw_length = partner ? this.partnerDrMaxRawLength : this.drMaxRawLength
      }
      if (
        ['kmer', 'dr', 'top-n-gram', 'dt'].includes(tokenizer) &&
        ['bow', 'tfidf', 'tf-idf'].includes(encoder.toLowerCase())
      ) {
        params.frequency_max_raw_length = this.frequencyMaxRawLength
      }
      if (tokenizer === 'top-n-gram') {
        params.top_ngram_n = partner ? this.partnerTopNgramN : this.topNgramN
        params.topngram_dense_memory_limit_gib = this.topNgramDenseMemoryLimitGib
      }
      if (tokenizer === 'dt') {
        params.dt_max_distance = partner ? this.partnerDtMaxDistance : this.dtMaxDistance
        params.dt_dense_memory_limit_gib = partner
          ? this.partnerDtDenseMemoryLimitGib
          : this.dtDenseMemoryLimitGib
      }
      return {
        feature_route: featureRoute,
        tokenizer,
        encoder,
        stream_model: partner ? this.partnerStreamModel : this.streamModel,
        model_source:
          featureRoute === 'pretrained'
            ? partner
              ? this.partnerModelSource || PRETRAINED_MODEL_DEFAULTS[tokenizer] || ''
              : this.pretrainedModelSource(tokenizer)
            : '',
        max_length: partner ? this.partnerMaxLength : this.maxLength,
        params,
      }
    },
    toConfig(options: { isPairTask: boolean; encoderFamily: string }) {
      const encoder =
        this.featureRoute === 'pretrained'
          ? canonicalPretrainedEncoder(this.tokenizer, this.tokenizer)
          : this.encoder
      const recommendedPairStrategy =
        this.featureRoute === 'classic' && encoder === 'index' ? 'single_stream' : 'two_stream'
      const pairStrategy = this.mode === 'recommended' ? recommendedPairStrategy : this.pairStrategy
      const usesPartnerStream = options.isPairTask && pairStrategy === 'two_stream'
      const partnerTokenizer = usesPartnerStream
        ? this.streamsLinked
          ? this.tokenizer
          : this.partnerTokenizer
        : ''
      const partnerEncoder = usesPartnerStream
        ? this.streamsLinked
          ? encoder
          : this.partnerFeatureRoute === 'pretrained'
            ? canonicalPretrainedEncoder(this.partnerTokenizer, this.partnerTokenizer)
            : this.partnerEncoder
        : ''
      const effectiveReduction =
        options.isPairTask && this.reduction === 'lsa' ? 'none' : this.reduction
      const config: Record<string, unknown> = {
        ...hiddenConfigExtrasForRoute(this.clonedConfigExtras, {
          datasetId: this.datasetId,
          sourceDatasetId: this.cloneSourceDatasetId,
          tokenizer: this.tokenizer,
          encoder,
          partnerTokenizer,
          partnerEncoder,
          isPairTask: options.isPairTask,
        }),
        tokenizer: this.tokenizer,
        encoder,
        reduction: effectiveReduction,
      }
      const pretrainedModelField = this.pretrainedModelField(this.tokenizer)
      if (this.featureRoute === 'pretrained' && pretrainedModelField) {
        config[pretrainedModelField] = this.pretrainedModelSource(this.tokenizer)
      }

      if (!options.isPairTask) {
        if (this.featureRoute === 'pretrained') {
          const useResidueModeling =
            this.mode === 'custom' && this.pretrainedPredictorRoute === 'residue_modeling'
          if (useResidueModeling) {
            const architecture = MODELING_ARCHITECTURE_KEYS.has(this.modelingArchitecture)
              ? this.modelingArchitecture
              : 'cnn'
            config.model = architecture
            config.sequence_model = architecture
            config.training_mode = this.trainingMode
            config.classifier = serializedClassifierForTrainingMode(
              this.trainingMode,
              this.classifier,
            )
            config.residue_adapter_dim = this.residueAdapterDim
          } else {
            config.model = this.classifier
            config.classifier = this.classifier
          }
        } else if (
          this.mode === 'custom' &&
          this.classicPredictorRoute === 'flattened_classifier' &&
          FLATTENABLE_SEQUENCE_ENCODERS.has(this.encoder)
        ) {
          const classifier = CLASSICAL_CLASSIFIERS.has(this.classifier) ? this.classifier : 'svm'
          config.model = classifier
          config.classifier = classifier
        } else if (options.encoderFamily === 'sequence') {
          config.model = this.modelingArchitecture
          config.sequence_model = this.modelingArchitecture
          config.training_mode = this.mode === 'recommended' ? 'end_to_end' : this.trainingMode
          config.classifier = serializedClassifierForTrainingMode(
            config.training_mode,
            this.classifier,
          )
        } else {
          config.model = this.classifier
          config.classifier = this.classifier
        }
        return this.finalizeConfig(config, options)
      }

      config.pair_strategy = pairStrategy
      if (pairStrategy === 'single_stream') {
        config.model = this.modelingArchitecture
        config.sequence_model = this.modelingArchitecture
        config.training_mode = this.trainingMode
        config.classifier = serializedClassifierForTrainingMode(this.trainingMode, this.classifier)
        return this.finalizeConfig(config, options)
      }

      const peptideStream = this.buildStreamSpec('peptide')
      const partnerStream = this.streamsLinked
        ? { ...peptideStream, params: { ...peptideStream.params } }
        : this.buildStreamSpec('partner')
      config.streams = {
        peptide: peptideStream,
        partner: partnerStream,
      }

      const interactionLevel = this.mode === 'recommended' ? 'vector' : this.interactionLevel
      config.interaction_level = interactionLevel
      if (interactionLevel === 'token') {
        config.interaction_model = this.interactionModel
        config.model = this.interactionModel
        config.training_mode = this.trainingMode
        config.classifier = serializedClassifierForTrainingMode(this.trainingMode, this.classifier)
        return this.finalizeConfig(config, options)
      }

      const predictorType = this.mode === 'recommended' ? 'classifier' : this.predictorType
      config.predictor_type = predictorType
      const topicReduction = isPairTopicReduction(config.reduction)
      const fusionMode =
        this.mode === 'recommended' ? (topicReduction ? 'concat' : 'concat_all') : this.fusionMode
      if (fusionMode === 'aligned_gated') {
        config.stream_alignment_dim = this.streamAlignmentDim
      }
      if (predictorType === 'sample_graph') {
        config.graph_predictor = this.graphPredictor
        config.model = this.graphPredictor
        if (this.graphPredictor !== 'hgt') config.fusion_mode = fusionMode
      } else {
        config.fusion_mode = fusionMode
        config.classifier = this.classifier
        config.model = this.classifier
      }
      return this.finalizeConfig(config, options)
    },
    finalizeConfig(
      config: Record<string, unknown>,
      options: { isPairTask: boolean; encoderFamily: string },
    ) {
      config.checkpoint_objective =
        this.checkpointObjective || (options.isPairTask ? 'auc' : 'macro_f1')
      config.threshold_objective =
        this.thresholdObjective || (options.isPairTask ? 'f1' : 'macro_f1')
      for (const [key, id, params] of [
        ['checkpoint_objective_params', this.checkpointObjective, this.checkpointObjectiveParams],
        ['threshold_objective_params', this.thresholdObjective, this.thresholdObjectiveParams],
        ['tuning_objective_params', this.mlTuningObjective, this.tuningObjectiveParams],
      ] as const) {
        if (id === 'dual_f1_sacc') config[key] = { ...params }
      }
      const rawClassifier = normalizedConfigText(config, 'classifier')
      const rawModel = normalizedConfigText(config, 'model')
      if (rawClassifier === 'svm' || (!rawClassifier && rawModel === 'svm')) {
        config.svm_calibration = this.svmCalibration
        config.svm_calibration_cv_folds = 3
      }
      if (usesEpochTraining(config)) {
        config.epochs = this.epochs
        config.batch_size = this.batchSize
      }
      if (this.mode === 'custom') {
        this.applyAdvancedConfig(config, options)
        if (classificationLossApplies(config)) {
          for (const key of options.isPairTask ? PAIR_LOSS_FIELDS : SEQUENCE_LOSS_FIELDS) {
            if (Object.hasOwn(this.classificationLossOverrides, key)) {
              config[key] = this.classificationLossOverrides[key]
            }
          }
        }
      }
      const encoder = normalizedConfigText(config, 'encoder')
      if (!config.streams && COVARIANCE_ENCODERS.has(encoder)) {
        config.k = 1
        if (encoder === 'ac') config.ac_lag = this.acLag
        if (encoder === 'cc') config.cc_lag = this.ccLag
        if (encoder === 'acc') config.acc_lag = this.accLag
        config.covariance_max_raw_length = this.covarianceMaxRawLength
      }
      if (!config.streams && PDT_ENCODERS.has(encoder)) {
        config.k = 1
        config.pdt_lag = this.pdtLag
        config.pdt_max_raw_length = this.pdtMaxRawLength
      }
      if (!config.streams && PROFILE_DEPENDENT_ENCODERS.has(encoder)) {
        config.k = 1
      }
      if (!config.streams && encoder === 'ac_pssm') {
        config.ac_pssm_lag = this.acPssmLag
      }
      if (!config.streams && encoder === 'cc_pssm') {
        config.cc_pssm_lag = this.ccPssmLag
      }
      if (!config.streams && encoder === 'acc_pssm') {
        config.acc_pssm_lag = this.accPssmLag
      }
      if (!config.streams && encoder === 'ac_pssm_blosum62_hybrid') {
        config.ac_pssm_blosum62_hybrid_lag = this.acPssmBlosum62HybridLag
      }
      if (!config.streams && encoder === 'cc_pssm_blosum62_hybrid') {
        config.cc_pssm_blosum62_hybrid_lag = this.ccPssmBlosum62HybridLag
      }
      if (!config.streams && encoder === 'acc_pssm_blosum62_hybrid') {
        config.acc_pssm_blosum62_hybrid_lag = this.accPssmBlosum62HybridLag
      }
      const usesProfileTokenizer = configUsesTokenizer(config, PROFILE_DEPENDENT_TOKENIZERS)
      if (
        usesProfileTokenizer &&
        (config.profile_on_missing === undefined ||
          config.profile_on_missing === null ||
          config.profile_on_missing === '')
      ) {
        config.profile_on_missing = 'blosum62_target_frequency'
      }
      if (
        configUsesEncoder(
          config,
          new Set([
            'pssm_blosum62_hybrid',
            'psfm_blosum62_hybrid',
            ...HYBRID_PSSM_COVARIANCE_ENCODERS,
          ]),
        )
      ) {
        config.profile_on_missing = 'blosum62_hybrid'
      }
      const usesTopNgram = configUsesTokenizer(config, new Set(['top-n-gram']))
      if (usesTopNgram) {
        config.topngram_dense_memory_limit_gib = this.topNgramDenseMemoryLimitGib
      }
      return config
    },
    pretrainedModelField(tokenizer: string) {
      tokenizer = normalizedTokenizer(tokenizer)
      if (tokenizer === 'esm2') return 'esm2_model'
      if (tokenizer === 'protbert') return 'protbert_model'
      if (tokenizer === 'prott5') return 'prott5_model'
      if (tokenizer === 'esmc') return 'esmc_model'
      if (tokenizer === 'esm3') return 'esm3_model'
      return ''
    },
    pretrainedModelSource(tokenizer: string) {
      tokenizer = normalizedTokenizer(tokenizer)
      if (tokenizer === 'esm2') return this.esm2Model || PRETRAINED_MODEL_DEFAULTS.esm2
      if (tokenizer === 'protbert') return this.protbertModel || PRETRAINED_MODEL_DEFAULTS.protbert
      if (tokenizer === 'prott5') return this.prott5Model || PRETRAINED_MODEL_DEFAULTS.prott5
      if (tokenizer === 'esmc') return this.esmcModel || PRETRAINED_MODEL_DEFAULTS.esmc
      if (tokenizer === 'esm3') return this.esm3Model || PRETRAINED_MODEL_DEFAULTS.esm3
      return ''
    },
    applyPairTopicPreset(reduction: string) {
      const adjusted: PairTopicRouteField[] = []
      const mark = (field: PairTopicRouteField) => adjusted.push(field)

      this.reduction = String(reduction).trim().toLowerCase()
      if (!PAIR_TASKS.has(this.taskType) || !isPairTopicReduction(this.reduction)) {
        return adjusted
      }

      if (this.featureRoute !== 'classic') {
        this.featureRoute = 'classic'
        mark('featureRoute')
      }
      if (!PAIR_TOPIC_TOKENIZERS.has(this.tokenizer)) {
        this.tokenizer = 'kmer'
        mark('tokenizer')
      }
      if (this.encoder !== 'bow') {
        this.encoder = 'bow'
        mark('encoder')
      }
      const partnerAlreadyCompatible =
        this.streamsLinked ||
        (this.partnerFeatureRoute === 'classic' &&
          PAIR_TOPIC_TOKENIZERS.has(this.partnerTokenizer) &&
          this.partnerEncoder === 'bow')
      if (!partnerAlreadyCompatible) {
        this.streamsLinked = true
        mark('streamsLinked')
      }
      if (this.pairStrategy !== 'two_stream') {
        this.pairStrategy = 'two_stream'
        mark('pairStrategy')
      }
      if (this.interactionLevel !== 'vector') {
        this.interactionLevel = 'vector'
        mark('interactionLevel')
      }
      if (this.predictorType !== 'classifier') {
        this.predictorType = 'classifier'
        mark('predictorType')
      }
      if (this.fusionMode !== 'concat') {
        this.fusionMode = 'concat'
        mark('fusionMode')
      }
      if (!CLASSICAL_CLASSIFIERS.has(this.classifier)) {
        this.classifier = 'svm'
        mark('classifier')
      }
      return adjusted
    },
    applyAdvancedConfig(
      config: Record<string, unknown>,
      options: { isPairTask: boolean; encoderFamily: string },
    ) {
      const model = String(config.model ?? '')
      const classifier = String(config.classifier ?? '')
      const effectiveClassifier = classifier || (CLASSICAL_CLASSIFIERS.has(model) ? model : '')
      const isTrainableDlRoute =
        TRAINABLE_DL_MODELS.has(model) ||
        (model === 'mlp' && !CLASSICAL_CLASSIFIERS.has(effectiveClassifier))
      const isClassicalRoute = CLASSICAL_CLASSIFIERS.has(effectiveClassifier)
      const effectiveReduction = String(config.reduction ?? 'none').toLowerCase()

      config.max_length = this.maxLength
      if (usesEpochTraining(config)) {
        config.epochs = this.epochs
        config.batch_size = this.batchSize
      }

      if (this.tokenizer === 'kmer') {
        const encoder = String(config.encoder ?? '').toLowerCase()
        config.k = COVARIANCE_ENCODERS.has(encoder) || PDT_ENCODERS.has(encoder) ? 1 : this.k
      }
      if (this.tokenizer === 'dr') {
        config.dr_max_distance = this.drMaxDistance
        config.dr_max_raw_length = this.drMaxRawLength
      }
      if (
        ['kmer', 'dr', 'top-n-gram', 'dt'].includes(this.tokenizer) &&
        ['bow', 'tfidf', 'tf-idf'].includes(String(config.encoder ?? '').toLowerCase())
      ) {
        config.frequency_max_raw_length = this.frequencyMaxRawLength
      }
      if (this.tokenizer === 'dr' || (options.isPairTask && this.partnerTokenizer === 'dr')) {
        config.dr_dense_memory_limit_gib = this.drDenseMemoryLimitGib
      }
      if (this.tokenizer === 'top-n-gram') {
        config.top_ngram_n = this.topNgramN
        config.topngram_dense_memory_limit_gib = this.topNgramDenseMemoryLimitGib
      }
      if (this.tokenizer === 'dt') {
        config.dt_max_distance = this.dtMaxDistance
        config.dt_dense_memory_limit_gib = this.dtDenseMemoryLimitGib
      }

      const activeStaticEncoders = [
        this.featureRoute === 'classic' ? this.encoder.toLowerCase() : '',
        options.isPairTask && this.partnerFeatureRoute === 'classic'
          ? this.partnerEncoder.toLowerCase()
          : '',
      ]
      const usesStaticEmbedding = activeStaticEncoders.some((encoder) =>
        ['word2vec', 'glove', 'fasttext'].includes(encoder),
      )
      const usesGlove = activeStaticEncoders.includes('glove')
      const usesFasttext = activeStaticEncoders.includes('fasttext')
      const usesPredictiveStatic = activeStaticEncoders.some((encoder) =>
        ['word2vec', 'fasttext'].includes(encoder),
      )
      if (usesStaticEmbedding) {
        config.embedding_vector_size = this.embeddingVectorSize
        config.embedding_window = this.embeddingWindow
        config.embedding_min_count = this.embeddingMinCount
        config.embedding_epochs = this.embeddingEpochs
      }
      if (usesPredictiveStatic) {
        config.embedding_workers = this.embeddingWorkers
      }
      if (usesGlove) {
        config.glove_x_max = this.gloveXMax
        config.glove_alpha = this.gloveAlpha
        config.glove_learning_rate = this.gloveLearningRate
        config.glove_batch_size = this.gloveBatchSize
      }
      if (usesFasttext) {
        config.fasttext_bucket = this.fasttextBucket
        config.static_embedding_memory_limit_gib = this.staticEmbeddingMemoryLimitGib
      }

      if (['lsa', 'lda', 'plsa'].includes(effectiveReduction)) {
        if (effectiveReduction === 'lsa') {
          config.lsa_dim = this.lsaDim
          config.lsa_dense_memory_limit_gib = this.lsaDenseMemoryLimitGib
        }
        if (effectiveReduction === 'lda') {
          config.lda_topics = this.ldaTopics
          config.lda_dense_memory_limit_gib = this.ldaDenseMemoryLimitGib
        }
        if (effectiveReduction === 'plsa') {
          config.plsa_topics = this.plsaTopics
          config.plsa_memory_limit_gib = this.plsaMemoryLimitGib
          config.plsa_contract_version = 1
        }
      }

      if (isTrainableDlRoute) {
        config.lr = this.learningRate
        config.dropout = this.enableRegularization ? this.dropout : 0.0
        config.weight_decay = this.enableRegularization ? this.weightDecay : 0.0
        config.grad_clip_norm = this.enableRegularization ? this.gradClipNorm : 0.0
        config.early_stop_patience = this.enableRegularization ? this.earlyStopPatience : 0
        config.early_stop_min_delta = this.enableRegularization ? this.earlyStopMinDelta : 0.0
        if (this.encoder !== 'index' || this.featureRoute === 'pretrained' || options.isPairTask) {
          config.dl_scaler = this.dlScaler
        }
      }

      if (predictionHeadApplies(config)) {
        const hiddenDims = predictionHeadHiddenDims(this.headHiddenLayerCount)
        config.head_hidden_dims = hiddenDims
        if (hiddenDims.length > 0) {
          config.head_activation = this.headActivation
          config.head_dropout = this.enableRegularization ? this.headDropout : 0.0
        }
      }

      if (isClassicalRoute) {
        config.ml_scaler = this.mlScaler
        config.ml_reduction = this.mlReduction
        if (this.mlReduction !== 'none') config.ml_reduction_dim = this.mlReductionDim
        if (this.mlReduction === 'tsvd') {
          config.tsvd_dense_memory_limit_gib = this.tsvdDenseMemoryLimitGib
        }
        if (this.mlReduction === 'pca') {
          config.pca_dense_memory_limit_gib = this.pcaDenseMemoryLimitGib
        }
        if (this.mlReduction === 'kpca') {
          config.kpca_kernel = this.kpcaKernel
          config.kpca_memory_limit_gib = this.kpcaMemoryLimitGib
          if (['rbf', 'poly'].includes(this.kpcaKernel)) config.kpca_gamma = this.kpcaGamma
        }
        if (
          ['rf', 'svm', 'logreg', 'extratrees', 'histgb', 'xgb', 'lgb'].includes(
            effectiveClassifier,
          )
        ) {
          config.ml_auto_tune = this.mlAutoTune
          if (this.mlAutoTune) {
            const tuningBudgets = {
              quick: { maxCandidates: 6, cvFolds: 2 },
              standard: { maxCandidates: 12, cvFolds: 3 },
              thorough: { maxCandidates: 18, cvFolds: 5 },
            } as const
            const budgetName =
              this.mlTuningBudget in tuningBudgets
                ? (this.mlTuningBudget as keyof typeof tuningBudgets)
                : 'quick'
            const budget = tuningBudgets[budgetName]
            config.ml_tuning_budget = budgetName
            config.ml_tuning_objective = this.mlTuningObjective
            config.ml_tuning_mode = 'cv'
            config.ml_tuning_max_candidates = budget.maxCandidates
            config.ml_cv_folds = budget.cvFolds
          }
        }
      }

      if (effectiveClassifier === 'svm') {
        config.svm_c = this.svmC
        config.svm_class_weight = this.enableRegularization ? this.svmClassWeight : 'none'
        config.svm_max_iter = this.svmMaxIter
        config.svm_tol = this.svmTol
        config.svm_kernel = this.svmKernel
        config.svm_gamma = this.svmGamma
        config.svm_convergence_policy = this.svmConvergencePolicy
      }
      if (effectiveClassifier === 'logreg') {
        config.logreg_c = this.logregC
        config.logreg_class_weight = this.enableRegularization ? this.logregClassWeight : 'none'
        config.logreg_max_iter = this.logregMaxIter
        config.logreg_convergence_policy = this.logregConvergencePolicy
      }
      if (effectiveClassifier === 'rf') {
        config.rf_estimators = this.rfEstimators
        config.rf_max_depth = this.rfMaxDepth
        config.rf_min_samples_leaf = this.rfMinSamplesLeaf
      }
      if (effectiveClassifier === 'extratrees') {
        config.et_estimators = this.etEstimators
        config.et_max_depth = this.etMaxDepth
        config.et_min_samples_leaf = this.etMinSamplesLeaf
      }
      if (effectiveClassifier === 'histgb') {
        config.histgb_max_iter = this.histgbMaxIter
        config.histgb_max_leaf_nodes = this.histgbMaxLeafNodes
        config.histgb_lr = this.histgbLearningRate
        config.histgb_l2_regularization = this.enableRegularization
          ? this.histgbL2Regularization
          : 0.0
      }
      if (effectiveClassifier === 'xgb') {
        config.xgb_estimators = this.xgbEstimators
        config.xgb_max_depth = this.xgbMaxDepth
        config.xgb_lr = this.xgbLearningRate
        config.xgb_subsample = this.enableRegularization ? this.xgbSubsample : 1.0
        config.xgb_colsample = this.enableRegularization ? this.xgbColsample : 1.0
        config.xgb_reg_alpha = this.enableRegularization ? this.xgbRegAlpha : 0.0
        config.xgb_reg_lambda = this.enableRegularization ? this.xgbRegLambda : 1.0
      }
      if (effectiveClassifier === 'lgb') {
        config.lgb_estimators = this.lgbEstimators
        config.lgb_max_depth = this.lgbMaxDepth
        config.lgb_lr = this.lgbLearningRate
        config.lgb_subsample = this.enableRegularization ? this.lgbSubsample : 1.0
        config.lgb_colsample = this.enableRegularization ? this.lgbColsample : 1.0
        config.lgb_reg_alpha = this.enableRegularization ? this.lgbRegAlpha : 0.0
        config.lgb_reg_lambda = this.enableRegularization ? this.lgbRegLambda : 0.0
      }

      if (this.semanticAuxiliarySupported(options.isPairTask, model)) {
        config.enable_semantic_similarity = this.enableSemanticSimilarity
        config.semantic_similarity_weight = this.semanticSimilarityWeight
        config.semantic_zero_weight = this.semanticZeroWeight
        if (this.enableSemanticSimilarity) {
          config.semantic_positive_partners = this.semanticPositivePartners
          config.semantic_zero_partners = this.semanticZeroPartners
        }
        if (this.enableSemanticSimilarity && this.semanticLossType !== 'legacy') {
          config.semantic_loss_type = this.semanticLossType
        }
        if (
          this.enableSemanticSimilarity &&
          ['hla_context', 'hlai_anchor'].includes(this.interactionSemanticMode)
        ) {
          // Preserve removed modes from cloned historical configs so backend
          // preflight rejects them explicitly instead of silently migrating.
          config.interaction_semantic_mode = this.interactionSemanticMode
        }
      }

      if (this.enableEnsemble && !options.isPairTask) {
        const members = this.ensembleMembers
          .split(',')
          .map((value) => value.trim())
          .filter(Boolean)
        if (members.length >= 2) {
          config.ensemble_members = members
          config.ensemble_method = this.ensembleMethod
          config.ensemble_weight_source = this.ensembleWeightSource
          config.ensemble_grid_step = this.ensembleGridStep
        }
      }
    },
    semanticAuxiliarySupported(isPairTask: boolean, model: string) {
      if (
        this.trainingMode === 'export_embeddings' ||
        (isPairTask && this.predictorType === 'sample_graph')
      ) {
        return false
      }
      if (!isPairTask) {
        return ['cnn', 'rnn', 'gru', 'lstm', 'transformer', 'mlp'].includes(model)
      }
      return [
        'cnn',
        'rnn',
        'gru',
        'lstm',
        'transformer',
        'mlp',
        'cross_attn',
        'bi_cross_attn',
      ].includes(model)
    },
  },
})
