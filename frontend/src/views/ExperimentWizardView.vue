<script setup lang="ts">
import { normalizePairFields } from '../pairTerminology'
import { normalizeTaskType } from '../taskTerminology'
import '../experimentWizard.css'
import SelectionObjectives from '../components/SelectionObjectives.vue'
import ClassificationLossSettings from '../components/ClassificationLossSettings.vue'
import {
  classificationLossApplies,
  classificationLossParametersValid,
  defaultClassificationLoss,
} from '../classificationLoss'
import EnsembleEntry from '../components/EnsembleEntry.vue'
import { useMutation, useQuery } from '@tanstack/vue-query'
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  CircleCheck,
  FlaskConical,
  Link2Off,
  Play,
  Sparkles,
} from '@lucide/vue'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { api } from '../api/client'
import { useWorkspaceAccess } from '../workspaceAccess'
import { useLoginDraft } from '../loginDraft'
import EffectiveRouteDiagram from '../components/EffectiveRouteDiagram.vue'
import PageHeader from '../components/PageHeader.vue'
import RouteConflictDialog from '../components/RouteConflictDialog.vue'
import SelectControl from '../components/SelectControl.vue'
import type { SelectOption } from '../components/selectControl'
import { TASK_ENTRIES } from '../taskCatalog'
import { localizedApiError, localizedPreflightMessage } from '../errorLocalization'
import {
  CLASSICAL_CLASSIFIERS,
  TRAINABLE_DL_MODELS,
  dropoutScopeForModel,
  resolveClassicFeatureRoute,
  usesEpochTraining as routeUsesEpochTraining,
} from '../experimentRoutes'
import { buildModelDefaultName } from '../modelDisplay'
import { nextRerunNameForDisplay } from '../localization'
import {
  COVARIANCE_ENCODERS,
  HYBRID_PSSM_COVARIANCE_ENCODERS,
  PDT_ENCODERS,
  datasetLengthRoleMax,
  datasetLengthRoleMin,
  isPairTopicReduction,
  pairTopicRouteIssues,
  preflightAllowsTraining,
  predictionHeadApplies,
  predictionHeadHiddenDims,
  type PairTopicRouteField,
  type PairTopicRouteState,
  useExperimentDraftStore,
} from '../stores/experimentDraft'

const { isGuest, canUseWorkspace, requireLogin } = useWorkspaceAccess()

const router = useRouter()
const draft = useExperimentDraftStore()
const { t, te } = useI18n()
const capabilities = useQuery({ queryKey: ['capabilities'], queryFn: api.capabilities })
const kmerKConstraint = computed(
  () => capabilities.data.value?.tokenizer_constraints?.kmer?.k ?? { min: 1, max: 3 },
)
const fallbackResidueProfileEncoders = [
  'pssm',
  'pssm_blosum62_hybrid',
  'psfm',
  'psfm_blosum62_hybrid',
  'ac_pssm',
  'cc_pssm',
  'acc_pssm',
  ...HYBRID_PSSM_COVARIANCE_ENCODERS,
]
const residueProfileEncoders = computed(() => {
  const contract = capabilities.data.value?.sequence_profile_contract
  if (!contract) return new Set(fallbackResidueProfileEncoders)
  return new Set([
    ...(contract?.strict_features ?? []),
    ...(contract?.explicit_hybrid_encoders ?? []),
    ...(contract?.explicit_hybrid_feature ? [contract.explicit_hybrid_feature] : []),
  ])
})
const profileRequiredK = computed(
  () => capabilities.data.value?.sequence_profile_contract?.required_k ?? 1,
)
const covarianceEncoders = computed(
  () =>
    new Set(capabilities.data.value?.covariance_encoder_contract?.encoders ?? COVARIANCE_ENCODERS),
)
const covarianceMaxRawLength = computed(
  () =>
    capabilities.data.value?.covariance_encoder_contract?.max_raw_length?.absolute_max ?? 100000,
)
const pdtEncoders = computed(
  () => new Set(capabilities.data.value?.pdt_encoder_contract?.encoders ?? [...PDT_ENCODERS]),
)
const pdtRequiredK = computed(() => capabilities.data.value?.pdt_encoder_contract?.required_k ?? 1)
const pdtDefaultLag = computed(
  () => capabilities.data.value?.pdt_encoder_contract?.default_lag ?? 8,
)
const pdtMaxRawLength = computed(
  () => capabilities.data.value?.pdt_encoder_contract?.max_raw_length?.absolute_max ?? 100000,
)
const pssmCovarianceEncoders = computed(
  () =>
    new Set(
      capabilities.data.value?.pssm_covariance_encoder_contract?.encoders ?? [
        'ac_pssm',
        'cc_pssm',
        'acc_pssm',
      ],
    ),
)
function isCovarianceEncoder(encoder: string) {
  return covarianceEncoders.value.has(String(encoder).toLowerCase())
}
function isPssmCovarianceEncoder(encoder: string) {
  return pssmCovarianceEncoders.value.has(String(encoder).toLowerCase())
}
function isPdtEncoder(encoder: string) {
  return pdtEncoders.value.has(String(encoder).toLowerCase())
}
function isFixedKEncoder(encoder: string) {
  return (
    isCovarianceEncoder(encoder) ||
    isPdtEncoder(encoder) ||
    residueProfileEncoders.value.has(String(encoder).toLowerCase())
  )
}
function hasInvalidProfileK(tokenizer: string, encoder: string, k: number) {
  return (
    tokenizer === 'kmer' &&
    residueProfileEncoders.value.has(String(encoder).toLowerCase()) &&
    Number(k) !== profileRequiredK.value
  )
}
const primaryProfileKInvalid = computed(() =>
  hasInvalidProfileK(draft.tokenizer, draft.encoder, draft.k),
)
const primaryKValue = computed({
  get: () => (isFixedKEncoder(draft.encoder) ? 1 : draft.k),
  set: (value: number) => {
    draft.k = isFixedKEncoder(draft.encoder) ? 1 : value
  },
})
function maxLengthLabelForTokenizer(tokenizer: string) {
  return tokenizer === 'kmer'
    ? t('wizard.maximumKmerTokenCount')
    : t('wizard.maximumSequenceLength')
}
function maxLengthMinimumForTokenizer(tokenizer: string) {
  return 1
}
function kLabelForTokenizer(tokenizer: string) {
  return tokenizer === 'kmer' ? t('wizard.kmerWindowLength') : 'k'
}
const datasets = useQuery({
  queryKey: ['datasets'],
  queryFn: api.datasets,
  enabled: canUseWorkspace,
})
const matchingDatasets = computed(() =>
  datasets.data.value?.filter((dataset) => dataset.task_type === draft.taskType),
)
const selectedDataset = computed(() =>
  datasets.data.value?.find((dataset) => dataset.id === draft.datasetId),
)
const selectedDatasetSplits = computed(() => {
  const splits = selectedDataset.value?.metadata.splits
  if (!splits || typeof splits !== 'object' || Array.isArray(splits)) return {}
  return splits as Record<string, number>
})
const isPairTask = computed(() =>
  ['pep_protein_interaction', 'pep_hlaii', 'pep_hlai'].includes(draft.taskType),
)
const partnerProfileKInvalid = computed(
  () =>
    isPairTask.value &&
    !draft.streamsLinked &&
    hasInvalidProfileK(draft.partnerTokenizer, draft.partnerEncoder, draft.partnerK),
)
const partnerKValue = computed({
  get: () => (isFixedKEncoder(draft.partnerEncoder) ? 1 : draft.partnerK),
  set: (value: number) => {
    draft.partnerK = isFixedKEncoder(draft.partnerEncoder) ? 1 : value
  },
})
const primaryCovarianceLag = computed({
  get: () =>
    draft.encoder === 'ac' ? draft.acLag : draft.encoder === 'cc' ? draft.ccLag : draft.accLag,
  set: (value: number) => {
    if (draft.encoder === 'ac') draft.acLag = value
    if (draft.encoder === 'cc') draft.ccLag = value
    if (draft.encoder === 'acc') draft.accLag = value
  },
})
const partnerCovarianceLag = computed({
  get: () =>
    draft.partnerEncoder === 'ac'
      ? draft.partnerAcLag
      : draft.partnerEncoder === 'cc'
        ? draft.partnerCcLag
        : draft.partnerAccLag,
  set: (value: number) => {
    if (draft.partnerEncoder === 'ac') draft.partnerAcLag = value
    if (draft.partnerEncoder === 'cc') draft.partnerCcLag = value
    if (draft.partnerEncoder === 'acc') draft.partnerAccLag = value
  },
})
const primaryPdtLag = computed({
  get: () => draft.pdtLag,
  set: (value: number) => {
    draft.pdtLag = value
    draft.pdtLagExplicit = true
  },
})
const partnerPdtLag = computed({
  get: () => draft.partnerPdtLag,
  set: (value: number) => {
    draft.partnerPdtLag = value
    draft.partnerPdtLagExplicit = true
  },
})
const primaryPssmCovarianceLag = computed({
  get: () =>
    draft.encoder === 'ac_pssm'
      ? draft.acPssmLag
      : draft.encoder === 'cc_pssm'
        ? draft.ccPssmLag
        : draft.encoder === 'acc_pssm'
          ? draft.accPssmLag
          : draft.encoder === 'ac_pssm_blosum62_hybrid'
            ? draft.acPssmBlosum62HybridLag
            : draft.encoder === 'cc_pssm_blosum62_hybrid'
              ? draft.ccPssmBlosum62HybridLag
              : draft.accPssmBlosum62HybridLag,
  set: (value: number) => {
    if (draft.encoder === 'ac_pssm') draft.acPssmLag = value
    if (draft.encoder === 'cc_pssm') draft.ccPssmLag = value
    if (draft.encoder === 'acc_pssm') draft.accPssmLag = value
    if (draft.encoder === 'ac_pssm_blosum62_hybrid') draft.acPssmBlosum62HybridLag = value
    if (draft.encoder === 'cc_pssm_blosum62_hybrid') draft.ccPssmBlosum62HybridLag = value
    if (draft.encoder === 'acc_pssm_blosum62_hybrid') draft.accPssmBlosum62HybridLag = value
  },
})
const partnerPssmCovarianceLag = computed({
  get: () =>
    draft.partnerEncoder === 'ac_pssm'
      ? draft.partnerAcPssmLag
      : draft.partnerEncoder === 'cc_pssm'
        ? draft.partnerCcPssmLag
        : draft.partnerEncoder === 'acc_pssm'
          ? draft.partnerAccPssmLag
          : draft.partnerEncoder === 'ac_pssm_blosum62_hybrid'
            ? draft.partnerAcPssmBlosum62HybridLag
            : draft.partnerEncoder === 'cc_pssm_blosum62_hybrid'
              ? draft.partnerCcPssmBlosum62HybridLag
              : draft.partnerAccPssmBlosum62HybridLag,
  set: (value: number) => {
    if (draft.partnerEncoder === 'ac_pssm') draft.partnerAcPssmLag = value
    if (draft.partnerEncoder === 'cc_pssm') draft.partnerCcPssmLag = value
    if (draft.partnerEncoder === 'acc_pssm') draft.partnerAccPssmLag = value
    if (draft.partnerEncoder === 'ac_pssm_blosum62_hybrid')
      draft.partnerAcPssmBlosum62HybridLag = value
    if (draft.partnerEncoder === 'cc_pssm_blosum62_hybrid')
      draft.partnerCcPssmBlosum62HybridLag = value
    if (draft.partnerEncoder === 'acc_pssm_blosum62_hybrid')
      draft.partnerAccPssmBlosum62HybridLag = value
  },
})
const primaryCovarianceDatasetMinimum = computed(() => {
  if (!isCovarianceEncoder(draft.encoder)) return 0
  const metadata = selectedDataset.value?.metadata
  if (!isPairTask.value) return datasetLengthRoleMin(metadata, 'sequence')
  const peptide = datasetLengthRoleMin(metadata, 'peptide')
  if (!draft.streamsLinked) return peptide
  const partner = datasetLengthRoleMin(metadata, 'partner')
  return peptide > 0 && partner > 0 ? Math.min(peptide, partner) : 0
})
const primaryCovarianceDatasetMaximum = computed(() => {
  if (!isCovarianceEncoder(draft.encoder)) return 0
  const metadata = selectedDataset.value?.metadata
  if (!isPairTask.value) return datasetLengthRoleMax(metadata, 'sequence')
  const peptide = datasetLengthRoleMax(metadata, 'peptide')
  if (!draft.streamsLinked) return peptide
  const partner = datasetLengthRoleMax(metadata, 'partner')
  return peptide > 0 && partner > 0 ? Math.max(peptide, partner) : 0
})
const partnerCovarianceDatasetMinimum = computed(() =>
  isPairTask.value && !draft.streamsLinked && isCovarianceEncoder(draft.partnerEncoder)
    ? datasetLengthRoleMin(selectedDataset.value?.metadata, 'partner')
    : 0,
)
const partnerCovarianceDatasetMaximum = computed(() =>
  isPairTask.value && !draft.streamsLinked && isCovarianceEncoder(draft.partnerEncoder)
    ? datasetLengthRoleMax(selectedDataset.value?.metadata, 'partner')
    : 0,
)
const primaryPdtDatasetMinimum = computed(() => {
  if (!isPdtEncoder(draft.encoder)) return 0
  const metadata = selectedDataset.value?.metadata
  if (!isPairTask.value) return datasetLengthRoleMin(metadata, 'sequence')
  const peptide = datasetLengthRoleMin(metadata, 'peptide')
  if (!draft.streamsLinked) return peptide
  const partner = datasetLengthRoleMin(metadata, 'partner')
  return peptide > 0 && partner > 0 ? Math.min(peptide, partner) : 0
})
const primaryPdtDatasetMaximum = computed(() => {
  if (!isPdtEncoder(draft.encoder)) return 0
  const metadata = selectedDataset.value?.metadata
  if (!isPairTask.value) return datasetLengthRoleMax(metadata, 'sequence')
  const peptide = datasetLengthRoleMax(metadata, 'peptide')
  if (!draft.streamsLinked) return peptide
  const partner = datasetLengthRoleMax(metadata, 'partner')
  return peptide > 0 && partner > 0 ? Math.max(peptide, partner) : 0
})
const partnerPdtDatasetMinimum = computed(() =>
  isPairTask.value && !draft.streamsLinked && isPdtEncoder(draft.partnerEncoder)
    ? datasetLengthRoleMin(selectedDataset.value?.metadata, 'partner')
    : 0,
)
const partnerPdtDatasetMaximum = computed(() =>
  isPairTask.value && !draft.streamsLinked && isPdtEncoder(draft.partnerEncoder)
    ? datasetLengthRoleMax(selectedDataset.value?.metadata, 'partner')
    : 0,
)
const primaryPdtLagMaximum = computed(() =>
  primaryPdtDatasetMinimum.value > 1 ? primaryPdtDatasetMinimum.value - 1 : undefined,
)
const partnerPdtLagMaximum = computed(() =>
  partnerPdtDatasetMinimum.value > 1 ? partnerPdtDatasetMinimum.value - 1 : undefined,
)
const primaryPssmCovarianceDatasetMinimum = computed(() => {
  if (!isPssmCovarianceEncoder(draft.encoder)) return 0
  const metadata = selectedDataset.value?.metadata
  if (!isPairTask.value) return datasetLengthRoleMin(metadata, 'sequence')
  const peptide = datasetLengthRoleMin(metadata, 'peptide')
  if (!draft.streamsLinked) return peptide
  const partner = datasetLengthRoleMin(metadata, 'partner')
  return peptide > 0 && partner > 0 ? Math.min(peptide, partner) : 0
})
const partnerPssmCovarianceDatasetMinimum = computed(() =>
  isPairTask.value && !draft.streamsLinked && isPssmCovarianceEncoder(draft.partnerEncoder)
    ? datasetLengthRoleMin(selectedDataset.value?.metadata, 'partner')
    : 0,
)
const primaryTopNgramMinimum = computed(() => {
  if (draft.tokenizer !== 'top-n-gram') return 1
  if (!isPairTask.value) {
    return Math.max(1, datasetLengthRoleMax(selectedDataset.value?.metadata, 'sequence'))
  }
  const peptide = datasetLengthRoleMax(selectedDataset.value?.metadata, 'peptide')
  const partner = datasetLengthRoleMax(selectedDataset.value?.metadata, 'partner')
  return Math.max(1, draft.streamsLinked ? Math.max(peptide, partner) : peptide)
})
const partnerTopNgramMinimum = computed(() =>
  draft.partnerTokenizer === 'top-n-gram'
    ? Math.max(1, datasetLengthRoleMax(selectedDataset.value?.metadata, 'partner'))
    : 1,
)
const sequenceEncoders = new Set([
  'index',
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
const flattenableSequenceEncoders = new Set([
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
const regularizedClassifiers = new Set(['svm', 'logreg', 'histgb', 'xgb', 'lgb'])
const encoderFamily = computed(() => (sequenceEncoders.has(draft.encoder) ? 'sequence' : 'global'))
const tokenizerOptions = computed(() => {
  if (!capabilities.data.value) return []
  return draft.featureRoute === 'pretrained'
    ? capabilities.data.value.pretrained_tokenizers
    : capabilities.data.value.classic_tokenizers
})
const TOKENIZER_DISPLAY_NAMES: Record<string, string> = {
  esm2: 'ESM2',
  protbert: 'ProtBERT',
  prott5: 'ProtT5',
  esmc: 'ESMC',
  esm3: 'ESM3',
}
function tokenizerDisplayName(tokenizer: string) {
  return TOKENIZER_DISPLAY_NAMES[tokenizer] ?? tokenizer
}
const encoders = computed(() => {
  if (!capabilities.data.value || draft.featureRoute === 'pretrained') return []
  return capabilities.data.value.classic_encoder_options[draft.tokenizer] ?? []
})
const partnerTokenizerOptions = computed(() => {
  if (!capabilities.data.value) return []
  return draft.partnerFeatureRoute === 'pretrained'
    ? capabilities.data.value.pretrained_tokenizers
    : capabilities.data.value.classic_tokenizers
})
const partnerEncoders = computed(() => {
  if (!capabilities.data.value || draft.partnerFeatureRoute === 'pretrained') return []
  return capabilities.data.value.classic_encoder_options[draft.partnerTokenizer] ?? []
})
const transformations = computed(() => {
  if (!capabilities.data.value) return ['none']
  const catalog = isPairTask.value
    ? (capabilities.data.value.pair_feature_transformations ??
      capabilities.data.value.feature_transformations)
    : capabilities.data.value.feature_transformations
  return catalog[draft.encoder] ?? ['none']
})
const classifierOptions = computed(() => capabilities.data.value?.classifiers ?? [])
const classicalClassifierOptions = computed(() =>
  (capabilities.data.value?.classifiers ?? []).filter((value) => value !== 'mlp'),
)
const pairTopicActive = computed(() => isPairTask.value && isPairTopicReduction(draft.reduction))
const pairTopicPresetAdjustments = ref<PairTopicRouteField[]>([])
const pendingTopicConflict = ref<{
  field: PairTopicRouteField
  value: string | boolean
} | null>(null)

function pairTopicState(overrides: Partial<PairTopicRouteState> = {}): PairTopicRouteState {
  return {
    featureRoute: draft.featureRoute,
    tokenizer: draft.tokenizer,
    encoder: draft.encoder,
    streamsLinked: draft.streamsLinked,
    partnerFeatureRoute: draft.partnerFeatureRoute,
    partnerTokenizer: draft.partnerTokenizer,
    partnerEncoder: draft.partnerEncoder,
    pairStrategy: draft.pairStrategy,
    interactionLevel: draft.interactionLevel,
    predictorType: draft.predictorType,
    fusionMode: draft.fusionMode,
    classifier: draft.classifier,
    ...overrides,
  }
}

const activePairTopicIssues = computed(() =>
  pairTopicActive.value ? pairTopicRouteIssues(pairTopicState()) : [],
)
const pairTopicAdjustmentLabels = computed(() =>
  pairTopicPresetAdjustments.value.map((field) => t(`wizard.pairTopic.fields.${field}`)),
)
const pairTopicName = computed(() => draft.reduction.toUpperCase())

function setRouteField(field: PairTopicRouteField, value: string | boolean) {
  switch (field) {
    case 'featureRoute':
      draft.featureRoute = String(value) as 'classic' | 'pretrained'
      break
    case 'tokenizer':
      draft.tokenizer = String(value)
      break
    case 'encoder':
      draft.encoder = String(value)
      break
    case 'streamsLinked':
      draft.streamsLinked = Boolean(value)
      break
    case 'partnerFeatureRoute':
      draft.partnerFeatureRoute = String(value) as 'classic' | 'pretrained'
      break
    case 'partnerTokenizer':
      draft.partnerTokenizer = String(value)
      break
    case 'partnerEncoder':
      draft.partnerEncoder = String(value)
      break
    case 'pairStrategy':
      draft.pairStrategy = String(value)
      break
    case 'interactionLevel':
      draft.interactionLevel = String(value)
      break
    case 'predictorType':
      draft.predictorType = String(value)
      break
    case 'fusionMode':
      draft.fusionMode = String(value)
      break
    case 'classifier':
      draft.classifier = String(value)
      break
  }
}

function requestRouteFieldChange(field: PairTopicRouteField, value: string | number | boolean) {
  const normalizedValue = typeof value === 'number' ? String(value) : value
  const currentValue = field === 'streamsLinked' ? draft.streamsLinked : String(draft[field])
  if (Object.is(currentValue, normalizedValue)) return
  if (
    pairTopicActive.value &&
    pairTopicRouteIssues(pairTopicState({ [field]: normalizedValue })).length
  ) {
    pendingTopicConflict.value = { field, value: normalizedValue }
    return
  }
  setRouteField(field, normalizedValue)
}

function selectTransformation(value: string | number) {
  const reduction = String(value).trim().toLowerCase()
  pendingTopicConflict.value = null
  pairTopicPresetAdjustments.value =
    isPairTask.value && isPairTopicReduction(reduction) ? draft.applyPairTopicPreset(reduction) : []
  if (!isPairTask.value || !isPairTopicReduction(reduction)) draft.reduction = reduction
}

function repairPairTopicRoute() {
  pairTopicPresetAdjustments.value = draft.applyPairTopicPreset(draft.reduction)
}

function confirmTopicConflict() {
  const pending = pendingTopicConflict.value
  if (!pending) return
  draft.reduction = 'none'
  pairTopicPresetAdjustments.value = []
  pendingTopicConflict.value = null
  setRouteField(pending.field, pending.value)
}

function cancelTopicConflict() {
  pendingTopicConflict.value = null
}

function onStreamsLinkedChange(event: Event) {
  requestRouteFieldChange('streamsLinked', (event.target as HTMLInputElement).checked)
}
const sequenceArchitectureValues = computed(
  () =>
    capabilities.data.value?.modeling_architectures?.sequence ??
    (capabilities.data.value?.sequence_models ?? []).filter(
      (value) => !['gcn', 'gat'].includes(value),
    ),
)
const chainGraphArchitectureValues = computed(
  () =>
    capabilities.data.value?.modeling_architectures?.chain_graph ??
    (capabilities.data.value?.sequence_models ?? []).filter((value) =>
      ['gcn', 'gat'].includes(value),
    ),
)
const chainGraphCompatibleEncoders = computed(
  () =>
    capabilities.data.value?.route_constraints?.single_sequence?.classic?.chain_graph_encoders ??
    [],
)
const classicModelingArchitectureOptions = computed<SelectOption[]>(() => [
  ...sequenceArchitectureValues.value.map((value) => ({
    value,
    label: value.toUpperCase(),
    group: t('wizard.architectureGroups.sequence'),
  })),
  ...(chainGraphCompatibleEncoders.value.includes(draft.encoder)
    ? chainGraphArchitectureValues.value.map((value) => ({
        value,
        label: value.toUpperCase(),
        group: t('wizard.architectureGroups.sequenceChainGraph'),
      }))
    : []),
])
const pretrainedResidueArchitectureOptions = computed<SelectOption[]>(() => {
  const values =
    capabilities.data.value?.route_constraints?.single_sequence?.pretrained
      ?.residue_architectures?.[draft.tokenizer] ??
    capabilities.data.value?.sequence_models ??
    []
  return values.map((value) => ({
    value,
    label: value.toUpperCase(),
    group: ['gcn', 'gat'].includes(value)
      ? t('wizard.architectureGroups.residueChainGraph')
      : t('wizard.architectureGroups.residueSequence'),
  }))
})
const allowSingleStream = computed(
  () => draft.featureRoute === 'classic' && encoderFamily.value === 'sequence',
)
const allowTokenInteraction = computed(
  () =>
    (draft.featureRoute === 'pretrained' || encoderFamily.value === 'sequence') &&
    (draft.streamsLinked ||
      draft.partnerFeatureRoute === 'pretrained' ||
      sequenceEncoders.has(draft.partnerEncoder)),
)
const pairStrategyOptions = computed(() =>
  allowSingleStream.value ? ['single_stream', 'two_stream'] : ['two_stream'],
)
const interactionLevelOptions = computed(() =>
  allowTokenInteraction.value &&
  ((draft.featureRoute === 'classic' && draft.encoder === 'index') ||
    (!draft.streamsLinked &&
      draft.partnerFeatureRoute === 'classic' &&
      draft.partnerEncoder === 'index'))
    ? ['token']
    : allowTokenInteraction.value
      ? ['vector', 'token']
      : ['vector'],
)
const effectivePairStrategy = computed(() =>
  draft.mode === 'recommended'
    ? draft.featureRoute === 'classic' && draft.encoder === 'index'
      ? 'single_stream'
      : 'two_stream'
    : draft.pairStrategy,
)
const effectiveInteractionLevel = computed(() =>
  draft.mode === 'recommended' ? 'vector' : draft.interactionLevel,
)
const effectivePredictorType = computed(() =>
  draft.mode === 'recommended' ? 'classifier' : draft.predictorType,
)
const canUseFlattenedSequenceFeatures = computed(
  () =>
    draft.mode === 'custom' &&
    !isPairTask.value &&
    draft.featureRoute === 'classic' &&
    flattenableSequenceEncoders.has(draft.encoder),
)
const usesFlattenedClassifier = computed(
  () =>
    canUseFlattenedSequenceFeatures.value && draft.classicPredictorRoute === 'flattened_classifier',
)
const usesPretrainedResidueModeling = computed(
  () =>
    !isPairTask.value &&
    draft.mode === 'custom' &&
    draft.featureRoute === 'pretrained' &&
    draft.pretrainedPredictorRoute === 'residue_modeling',
)
const needsClassicModelingArchitecture = computed(
  () =>
    draft.featureRoute === 'classic' &&
    encoderFamily.value === 'sequence' &&
    (!isPairTask.value || effectivePairStrategy.value === 'single_stream') &&
    !usesFlattenedClassifier.value,
)
const usesTokenInteraction = computed(
  () =>
    isPairTask.value &&
    effectivePairStrategy.value === 'two_stream' &&
    effectiveInteractionLevel.value === 'token',
)
const usesVectorInteraction = computed(
  () =>
    isPairTask.value &&
    effectivePairStrategy.value === 'two_stream' &&
    effectiveInteractionLevel.value === 'vector',
)
const routeSummary = computed(() => {
  if (!isPairTask.value) {
    if (draft.featureRoute === 'pretrained') {
      if (usesPretrainedResidueModeling.value) {
        return `Tokenizer[${draft.tokenizer}] → Encoder[${draft.encoder}: ${t('wizard.routeSegments.residueEmbeddings')}] → Adapter[${draft.residueAdapterDim}] → Predictor[${draft.modelingArchitecture} · ${draft.trainingMode}]`
      }
      return `Tokenizer[${draft.tokenizer}] → Encoder[${draft.encoder}] → Predictor[${draft.classifier}]`
    }
    if (usesFlattenedClassifier.value) {
      return `Tokenizer[${draft.tokenizer}] → Encoder[${draft.encoder}: ${t('wizard.routeSegments.flattenedFeatures')}] → Predictor[${draft.classifier}]`
    }
    if (needsClassicModelingArchitecture.value) {
      const mode = draft.mode === 'recommended' ? 'end_to_end' : draft.trainingMode
      return `Tokenizer[${draft.tokenizer}] → Encoder[${draft.encoder}] → Predictor[${draft.modelingArchitecture} · ${mode}]`
    }
    return `Tokenizer[${draft.tokenizer}] → Encoder[${draft.encoder}] → Predictor[${draft.classifier}]`
  }
  if (effectivePairStrategy.value === 'single_stream') {
    return `Tokenizer[${draft.tokenizer}] → Encoder[${draft.encoder}] → Predictor[Single-stream · ${draft.modelingArchitecture} · ${draft.trainingMode}]`
  }
  if (effectiveInteractionLevel.value === 'token') {
    const partner = draft.streamsLinked
      ? `${draft.tokenizer}/${draft.encoder}`
      : `${draft.partnerTokenizer}/${draft.partnerEncoder}`
    return `Pep Tokenizer/Encoder[${draft.tokenizer}/${draft.encoder}] + Partner[${partner}] → Predictor[Two-stream · token · ${draft.interactionModel}]`
  }
  const partner = draft.streamsLinked
    ? `${draft.tokenizer}/${draft.encoder}`
    : `${draft.partnerTokenizer}/${draft.partnerEncoder}`
  return `Pep Tokenizer/Encoder[${draft.tokenizer}/${draft.encoder}] + Partner[${partner}] → Predictor[Two-stream · vector · ${effectivePredictorType.value}]`
})
const representationSummary = computed(() => {
  if (!isPairTask.value || effectivePairStrategy.value === 'single_stream') {
    return `${draft.tokenizer} → ${draft.encoder}`
  }
  const partner = draft.streamsLinked
    ? `${draft.tokenizer} / ${draft.encoder}`
    : `${draft.partnerTokenizer} / ${draft.partnerEncoder}`
  return `${draft.tokenizer} / ${draft.encoder} + ${partner}`
})
const pretrainedModelSource = computed({
  get() {
    return draft.pretrainedModelSource(draft.tokenizer)
  },
  set(value: string) {
    if (draft.tokenizer === 'esm2') draft.esm2Model = value
    if (draft.tokenizer === 'protbert') draft.protbertModel = value
    if (draft.tokenizer === 'prott5') draft.prott5Model = value
    if (draft.tokenizer === 'esmc') draft.esmcModel = value
    if (draft.tokenizer === 'esm3') draft.esm3Model = value
  },
})
const FALLBACK_ESMC_MODEL_OPTIONS: SelectOption[] = [
  {
    value: 'EvolutionaryScale/esmc-300m-2024-12',
    label: 'ESM-C 300M',
  },
  {
    value: 'EvolutionaryScale/esmc-600m-2024-12',
    label: 'ESM-C 600M',
  },
]
const esmcModelOptions = computed<SelectOption[]>(
  () =>
    capabilities.data.value?.pretrained_models?.esmc?.map((option) => ({
      value: option.id,
      label: option.label,
    })) ?? FALLBACK_ESMC_MODEL_OPTIONS,
)
const pretrainedModelSelectOptions = computed<SelectOption[]>(() =>
  draft.tokenizer === 'esmc' ? esmcModelOptions.value : [],
)
const partnerPretrainedModelSelectOptions = computed<SelectOption[]>(() =>
  draft.partnerTokenizer === 'esmc' ? esmcModelOptions.value : [],
)
const currentConfig = computed(() =>
  draft.toConfig({ isPairTask: isPairTask.value, encoderFamily: encoderFamily.value }),
)
const usesEpochSchedule = computed(() => routeUsesEpochTraining(currentConfig.value))
const effectiveModel = computed(() => String(currentConfig.value.model ?? ''))
const effectiveClassifier = computed(() => {
  const classifier = String(currentConfig.value.classifier ?? '')
  if (classifier) return classifier
  return CLASSICAL_CLASSIFIERS.has(effectiveModel.value) ? effectiveModel.value : ''
})
const effectiveHeadHiddenDims = computed(() => predictionHeadHiddenDims(draft.headHiddenLayerCount))
const headActivationLabel = computed(() =>
  draft.headActivation === 'silu' ? 'SiLU' : draft.headActivation.toUpperCase(),
)
const headStructurePreview = computed(() => {
  const nodes = [t('wizard.advanced.headStructure.input')]
  for (const width of effectiveHeadHiddenDims.value) {
    nodes.push(String(width), headActivationLabel.value)
  }
  nodes.push(t('wizard.advanced.headStructure.output'))
  return nodes.join(' → ')
})
const usesClassicalClassifier = computed(() => CLASSICAL_CLASSIFIERS.has(effectiveClassifier.value))
const usesTrainableDl = computed(() => TRAINABLE_DL_MODELS.has(effectiveModel.value))
const modelDropoutScope = computed(() => dropoutScopeForModel(effectiveModel.value))
const modelDropoutVisible = computed(
  () => usesTrainableDl.value && modelDropoutScope.value !== 'none',
)
const dropoutLabelKeys = {
  none: 'dropoutBackbone',
  cnn: 'dropoutCnn',
  sequence: 'dropoutSequence',
  transformer: 'dropoutTransformer',
  graph: 'dropoutGraph',
  interaction: 'dropoutInteraction',
  backbone: 'dropoutBackbone',
} as const
const modelDropoutLabel = computed(() =>
  modelDropoutScope.value === 'none'
    ? ''
    : t(`wizard.advanced.fields.${dropoutLabelKeys[modelDropoutScope.value]}`),
)
const regularizationRelevant = computed(
  () => usesTrainableDl.value || regularizedClassifiers.has(effectiveClassifier.value),
)
const tokenizerParameterVisible = computed(
  () =>
    draft.mode === 'custom' &&
    draft.featureRoute === 'classic' &&
    ['kmer', 'dr', 'top-n-gram', 'dt'].includes(draft.tokenizer),
)
const staticEmbeddingParameterVisible = computed(
  () =>
    draft.mode === 'custom' &&
    ((draft.featureRoute === 'classic' &&
      ['word2vec', 'glove', 'fasttext'].includes(draft.encoder.toLowerCase())) ||
      (isPairTask.value &&
        draft.partnerFeatureRoute === 'classic' &&
        ['word2vec', 'glove', 'fasttext'].includes(draft.partnerEncoder.toLowerCase()))),
)
const gloveParameterVisible = computed(
  () =>
    staticEmbeddingParameterVisible.value &&
    ((draft.featureRoute === 'classic' && draft.encoder.toLowerCase() === 'glove') ||
      (isPairTask.value &&
        draft.partnerFeatureRoute === 'classic' &&
        draft.partnerEncoder.toLowerCase() === 'glove')),
)
const fasttextParameterVisible = computed(
  () =>
    staticEmbeddingParameterVisible.value &&
    ((draft.featureRoute === 'classic' && draft.encoder.toLowerCase() === 'fasttext') ||
      (isPairTask.value &&
        draft.partnerFeatureRoute === 'classic' &&
        draft.partnerEncoder.toLowerCase() === 'fasttext')),
)
const predictiveStaticParameterVisible = computed(
  () =>
    staticEmbeddingParameterVisible.value &&
    ((draft.featureRoute === 'classic' &&
      ['word2vec', 'fasttext'].includes(draft.encoder.toLowerCase())) ||
      (isPairTask.value &&
        draft.partnerFeatureRoute === 'classic' &&
        ['word2vec', 'fasttext'].includes(draft.partnerEncoder.toLowerCase()))),
)
const featureTransformParameterVisible = computed(
  () => draft.mode === 'custom' && ['lsa', 'lda', 'plsa'].includes(draft.reduction),
)
const dlScalerVisible = computed(
  () =>
    draft.mode === 'custom' &&
    usesTrainableDl.value &&
    (draft.encoder !== 'index' || draft.featureRoute === 'pretrained' || isPairTask.value),
)
const mlProcessingVisible = computed(() => draft.mode === 'custom' && usesClassicalClassifier.value)
const predictionHeadVisible = computed(() => {
  if (draft.mode !== 'custom') return false
  return predictionHeadApplies(currentConfig.value)
})
const autoTunableClassifiers = computed(
  () =>
    new Set(
      capabilities.data.value?.classical_tuning?.supported_classifiers ?? [
        'rf',
        'svm',
        'logreg',
        'extratrees',
        'histgb',
        'xgb',
        'lgb',
      ],
    ),
)
const autoTuneVisible = computed(
  () => draft.mode === 'custom' && autoTunableClassifiers.value.has(effectiveClassifier.value),
)
const tuningBudgetContracts = computed(
  () =>
    capabilities.data.value?.classical_tuning?.budgets ?? {
      quick: { max_candidates: 6, cv_folds: 2, max_candidate_fold_evaluations: 12 },
      standard: { max_candidates: 12, cv_folds: 3, max_candidate_fold_evaluations: 36 },
      thorough: { max_candidates: 18, cv_folds: 5, max_candidate_fold_evaluations: 90 },
    },
)
const tuningBudgetNames = ['quick', 'standard', 'thorough'] as const
const semanticParameterVisible = computed(
  () =>
    draft.mode === 'custom' &&
    draft.semanticAuxiliarySupported(isPairTask.value, effectiveModel.value),
)
const semanticPositivePartnersInvalid = computed(
  () =>
    semanticParameterVisible.value &&
    draft.enableSemanticSimilarity &&
    (typeof draft.semanticPositivePartners !== 'number' ||
      !Number.isInteger(draft.semanticPositivePartners) ||
      draft.semanticPositivePartners < 1),
)
const semanticZeroPartnersInvalid = computed(
  () =>
    semanticParameterVisible.value &&
    draft.enableSemanticSimilarity &&
    (typeof draft.semanticZeroPartners !== 'number' ||
      !Number.isInteger(draft.semanticZeroPartners) ||
      draft.semanticZeroPartners < 0),
)
const semanticRelationType = computed(
  () =>
    capabilities.data.value?.task_semantic_relation_type?.[draft.taskType] ??
    (isPairTask.value ? 'interaction_profile_jaccard' : 'intrinsic_label_jaccard'),
)
const advancedSummary = computed(() => {
  if (draft.mode !== 'custom') return []
  const items = [`max_length=${draft.maxLength}`]
  if (usesTrainableDl.value) items.push(`lr=${draft.learningRate}`)
  if (draft.featureRoute === 'classic' && draft.tokenizer === 'kmer') {
    items.push(`k=${isFixedKEncoder(draft.encoder) ? 1 : draft.k}`)
  }
  if (draft.featureRoute === 'classic' && isCovarianceEncoder(draft.encoder)) {
    items.push(
      `${draft.encoder}_lag=${primaryCovarianceLag.value}`,
      `covariance_max_raw_length=${draft.covarianceMaxRawLength}`,
    )
  }
  if (draft.featureRoute === 'classic' && isPdtEncoder(draft.encoder)) {
    items.push(`pdt_lag=${draft.pdtLag}`, `pdt_max_raw_length=${draft.pdtMaxRawLength}`)
  }
  if (draft.featureRoute === 'classic' && isPssmCovarianceEncoder(draft.encoder)) {
    items.push(`${draft.encoder}_lag=${primaryPssmCovarianceLag.value}`)
  }
  if (draft.featureRoute === 'classic' && draft.tokenizer === 'dr') {
    items.push(
      `dr_max_distance=${draft.drMaxDistance}`,
      `dr_max_raw_length=${draft.drMaxRawLength}`,
      `dr_dense_memory_limit_gib=${draft.drDenseMemoryLimitGib}`,
    )
  }
  if (
    draft.featureRoute === 'classic' &&
    ['kmer', 'dr', 'top-n-gram', 'dt'].includes(draft.tokenizer) &&
    ['bow', 'tfidf', 'tf-idf'].includes(draft.encoder.toLowerCase())
  ) {
    items.push(`frequency_max_raw_length=${draft.frequencyMaxRawLength}`)
  }
  if (draft.featureRoute === 'classic' && draft.tokenizer === 'top-n-gram') {
    items.push(`top_ngram_n=${draft.topNgramN}`)
    items.push(`topngram_dense_memory_limit_gib=${draft.topNgramDenseMemoryLimitGib}`)
  }
  if (draft.featureRoute === 'classic' && draft.tokenizer === 'dt') {
    items.push(`dt_max_distance=${draft.dtMaxDistance}`)
    items.push(`dt_dense_memory_limit_gib=${draft.dtDenseMemoryLimitGib}`)
  }
  if (staticEmbeddingParameterVisible.value) {
    items.push(`embedding_dim=${draft.embeddingVectorSize}`)
    if (predictiveStaticParameterVisible.value) {
      items.push(`embedding_workers=${draft.embeddingWorkers}`)
    }
  }
  if (gloveParameterVisible.value) {
    items.push(`glove_alpha=${draft.gloveAlpha}`)
  }
  if (fasttextParameterVisible.value) {
    items.push(`fasttext_bucket=${draft.fasttextBucket}`)
  }
  if (regularizationRelevant.value) {
    items.push(
      draft.enableRegularization
        ? t('wizard.advanced.summary.regOn')
        : t('wizard.advanced.summary.regOff'),
    )
  }
  if (usesTrainableDl.value && draft.enableRegularization) {
    if (modelDropoutVisible.value) items.push(`${modelDropoutLabel.value}=${draft.dropout}`)
    items.push(`weight_decay=${draft.weightDecay}`)
  }
  if (predictionHeadVisible.value) {
    const hidden = Array.isArray(currentConfig.value.head_hidden_dims)
      ? (currentConfig.value.head_hidden_dims as number[])
      : []
    items.push(hidden.length ? `head=${hidden.join('x')} · ${draft.headActivation}` : 'head=linear')
    if (hidden.length && draft.enableRegularization) items.push(`head_dropout=${draft.headDropout}`)
  }
  if (semanticParameterVisible.value && draft.enableSemanticSimilarity) {
    items.push(`semantic_relation=${semanticRelationType.value}`)
    items.push(`semantic_loss=${draft.semanticLossType}`)
    items.push(`semantic_positive_partners=${draft.semanticPositivePartners}`)
    items.push(`semantic_zero_partners=${draft.semanticZeroPartners}`)
  }
  if (usesClassicalClassifier.value) {
    items.push(`ml_scaler=${draft.mlScaler}`)
    if (draft.mlReduction !== 'none') items.push(`ml_reduction=${draft.mlReduction}`)
    if (autoTuneVisible.value && draft.mlAutoTune) {
      items.push(`auto_tune=${draft.mlTuningObjective}`)
      const budget =
        tuningBudgetContracts.value[draft.mlTuningBudget] ?? tuningBudgetContracts.value.quick
      items.push(`budget=${draft.mlTuningBudget}`)
      items.push(`search≤${budget.max_candidates}×${budget.cv_folds}`)
    }
    if (effectiveClassifier.value === 'svm') {
      items.push(`svm_calibration=${draft.svmCalibration}`)
      items.push(`svm_kernel=${draft.svmKernel}`)
      items.push(`svm_gamma=${draft.svmGamma}`)
    }
  }
  return items
})
const agentHandoff = computed(() => (draft.handoff?.origin === 'agent' ? draft.handoff : null))
const isAgentHandoff = computed(() => Boolean(agentHandoff.value))
const experimentName = ref('')
const experimentNameEdited = ref(false)
const submissionError = ref('')
const suggestedExperimentName = computed(() =>
  buildModelDefaultName(
    currentConfig.value,
    draft.taskType,
    (key, params) => t(key, params ?? {}),
    agentHandoff.value?.agent_lane,
  ),
)
const defaultExperimentName = computed(() =>
  draft.cloneSourceExperimentName
    ? nextRerunNameForDisplay(
        draft.cloneSourceExperimentName,
        draft.cloneSourceGeneratedName,
        (key, params) => t(key, params ?? {}),
        currentConfig.value,
        draft.taskType,
      )
    : suggestedExperimentName.value,
)
const submittedExperimentName = computed(
  () => experimentName.value.trim() || defaultExperimentName.value,
)
const preflightPayload = computed(() => ({
  task_type: draft.taskType,
  dataset_id: draft.datasetId,
  config: currentConfig.value,
}))
interface ExperimentSubmissionSource {
  agentPlanId?: string
  agentActionId?: string
}

const submissionPayload = computed(() => ({
  name: submittedExperimentName.value,
  ...preflightPayload.value,
  ...(!experimentNameEdited.value || !experimentName.value.trim()
    ? draft.cloneSourceExperimentId
      ? {
          generated_name: {
            kind: 'rerun',
            source_experiment_id: draft.cloneSourceExperimentId,
          },
        }
      : {
          generated_name: {
            kind: 'config',
            ...(agentHandoff.value?.agent_lane ? { qualifier: agentHandoff.value.agent_lane } : {}),
          },
        }
    : {}),
}))
const createExperiment = useMutation({
  mutationFn: ({ agentPlanId, agentActionId }: ExperimentSubmissionSource) => {
    if (agentPlanId && agentActionId) {
      return api.confirmAgentExecutionAction(agentPlanId, agentActionId, {
        name: submittedExperimentName.value,
        ...preflightPayload.value,
      })
    }
    return api.createExperiment(submissionPayload.value)
  },
  onSuccess: (experiment, source) => {
    if (source.agentPlanId && source.agentActionId) {
      router.replace({ name: 'agent-design', query: { plan_id: source.agentPlanId } })
      return
    }
    router.push(`/experiments/${experiment.id}`)
  },
  onError: (error) => {
    submissionError.value = localizedApiError(error, t, te, 'wizard.submitError')
  },
})
const preflight = useQuery({
  queryKey: computed(() => ['preflight', preflightPayload.value]),
  queryFn: () => api.preflightExperiment(preflightPayload.value),
  enabled: computed(() => canUseWorkspace.value && Boolean(draft.datasetId)),
})
const currentTaskDescription = computed(() => t(`tasks.${draft.taskType}.description`))
const currentTaskRecommendation = computed(() => t(`tasks.${draft.taskType}.recommendation`))
const currentModeDescription = computed(() => t(`wizard.modeDescriptions.${draft.mode}`))
function stringSelectOptions(
  values: readonly string[] | null | undefined,
  label: (value: string) => string = (value) => value,
): SelectOption[] {
  return (values ?? []).map((value) => ({ value, label: label(value) }))
}
const taskDisplayOrder = new Map<string, number>(
  TASK_ENTRIES.map((task, index) => [task.key, index]),
)
const taskSelectOptions = computed<SelectOption[]>(() =>
  [...(capabilities.data.value?.tasks ?? [])]
    .sort(
      (left, right) =>
        (taskDisplayOrder.get(left.id) ?? taskDisplayOrder.size) -
        (taskDisplayOrder.get(right.id) ?? taskDisplayOrder.size),
    )
    .map((task) => ({
      value: task.id,
      label: taskLabel(task.id),
    })),
)
const datasetSelectOptions = computed<SelectOption[]>(() => [
  { value: '', label: t('wizard.selectDataset') },
  ...(matchingDatasets.value ?? []).map((dataset) => ({ value: dataset.id, label: dataset.name })),
])
const featureRouteSelectOptions = computed<SelectOption[]>(() => [
  { value: 'classic', label: t('wizard.featureRoutes.classic') },
  { value: 'pretrained', label: t('wizard.featureRoutes.pretrained') },
])
const classicPredictorRouteSelectOptions = computed<SelectOption[]>(() => [
  { value: 'modeling_architecture', label: t('wizard.predictorRoutes.modelingArchitecture') },
  { value: 'flattened_classifier', label: t('wizard.predictorRoutes.flattenedClassifier') },
])
const pretrainedPredictorRouteSelectOptions = computed<SelectOption[]>(() => [
  { value: 'pooled_classifier', label: t('wizard.predictorRoutes.pooledClassifier') },
  { value: 'residue_modeling', label: t('wizard.predictorRoutes.residueModeling') },
])
const trainingModeSelectOptions = stringSelectOptions(['end_to_end', 'export_embeddings'])
const fixedMlpSelectOptions = stringSelectOptions(['mlp'], () => 'MLP')
const predictorTypeSelectOptions = computed<SelectOption[]>(() => [
  { value: 'classifier', label: t('wizard.predictorTypes.classifier') },
  { value: 'sample_graph', label: t('wizard.predictorTypes.sampleGraph') },
])
const headLayerOptions = computed(() => [
  { value: 0 as const, label: t('wizard.advanced.headLayers.zero'), detail: 'Linear' },
  { value: 1 as const, label: t('wizard.advanced.headLayers.one'), detail: '128' },
  { value: 2 as const, label: t('wizard.advanced.headLayers.two'), detail: '256 → 128' },
])
const headActivationSelectOptions = stringSelectOptions(['relu', 'gelu', 'silu'], (value) =>
  value === 'silu' ? 'SiLU' : value.toUpperCase(),
)
const mlScalerSelectOptions = stringSelectOptions(['none', 'standard', 'minmax'])
const mlReductionSelectOptions = stringSelectOptions(['none', 'pca', 'kpca', 'tsvd'])
const kpcaKernelSelectOptions = stringSelectOptions(['rbf', 'linear', 'poly', 'cosine'])
const tuningObjectiveSelectOptions: SelectOption[] = [
  { value: 'f1', label: 'F1' },
  { value: 'auc', label: 'AUC' },
]
const classWeightSelectOptions = stringSelectOptions(['balanced', 'none'])

const svmKernelSelectOptions = stringSelectOptions(['linear', 'rbf'])
const svmCalibrationSelectOptions = computed<SelectOption[]>(() =>
  (capabilities.data.value?.svm_calibration?.options ?? ['none', 'sigmoid']).map((value) => ({
    value,
    label:
      value === 'sigmoid'
        ? t('wizard.advanced.fields.svmCalibrationSigmoid')
        : t('wizard.advanced.fields.svmCalibrationNone'),
  })),
)
const svmConvergencePolicySelectOptions = computed<SelectOption[]>(() => [
  { value: 'bounded_retry', label: t('wizard.advanced.fields.svmPolicyBounded') },
  { value: 'strict', label: t('wizard.advanced.fields.svmPolicyStrict') },
])
const logregConvergencePolicySelectOptions = computed<SelectOption[]>(() => [
  { value: 'bounded_retry', label: t('wizard.advanced.fields.logregPolicyBounded') },
  { value: 'strict', label: t('wizard.advanced.fields.logregPolicyStrict') },
])
const dlScalerSelectOptions = computed(() =>
  stringSelectOptions(isPairTask.value ? ['none', 'standard', 'minmax'] : ['none', 'standard']),
)
const fusionSelectOptions = computed<SelectOption[]>(() => {
  const options: SelectOption[] = [{ value: 'concat', label: 'concat' }]
  if (effectivePredictorType.value === 'classifier') {
    options.push({ value: 'aligned_gated', label: 'aligned_gated (learned projection)' })
  }
  if (draft.streamsLinked) {
    options.push(
      { value: 'concat_all', label: 'concat_all' },
      { value: 'concat_absdiff', label: 'concat_absdiff' },
      { value: 'concat_hadamard', label: 'concat_hadamard' },
      { value: 'gated', label: 'gated(static)' },
    )
  }
  return options
})
const localizedPreflightErrors = computed(() =>
  localizePreflightMessages(
    preflight.data.value?.errors ?? [],
    preflight.data.value?.error_codes ?? [],
  ),
)
const localizedPreflightWarnings = computed(() =>
  localizePreflightMessages(
    preflight.data.value?.warnings ?? [],
    preflight.data.value?.warning_codes ?? [],
  ),
)
const wizardEyebrow = computed(() => {
  if (isAgentHandoff.value) return t('wizard.agentEyebrow')
  return draft.cloneSourceExperimentName ? t('wizard.cloneEyebrow') : t('wizard.eyebrow')
})
const currentStep = ref(draft.initialStep)
const wizardSteps = computed(() => [
  {
    id: 1,
    title: t('wizard.steps.problem.title'),
    description: t('wizard.steps.problem.description'),
  },
  {
    id: 2,
    title: t('wizard.steps.representation.title'),
    description: t('wizard.steps.representation.description'),
  },
  {
    id: 3,
    title: t('wizard.steps.predictor.title'),
    description: t('wizard.steps.predictor.description'),
  },
  {
    id: 4,
    title: t('wizard.steps.review.title'),
    description: t('wizard.steps.review.description'),
  },
])
const activeStep = computed(() => wizardSteps.value[currentStep.value - 1] ?? wizardSteps.value[0])
const canAdvance = computed(() => {
  if (isGuest.value) return true
  if (currentStep.value === 1) return Boolean(draft.datasetId)
  return true
})
const canSubmitExperiment = computed(
  () =>
    !semanticPositivePartnersInvalid.value &&
    !semanticZeroPartnersInvalid.value &&
    classificationLossParametersValid(currentConfig.value) &&
    preflightAllowsTraining(
      draft.datasetId,
      preflight.data.value,
      preflight.isPending.value || preflight.isFetching.value || createExperiment.isPending.value,
      !isAgentHandoff.value ||
        Boolean(agentHandoff.value?.agent_plan_id && agentHandoff.value.agent_action_id),
    ),
)

function submit() {
  if (!requireLogin()) return
  if (!canSubmitExperiment.value) return
  submissionError.value = ''
  createExperiment.mutate({
    agentPlanId: agentHandoff.value?.agent_plan_id,
    agentActionId: agentHandoff.value?.agent_action_id,
  })
}

function detachAgentHandoff() {
  draft.detachAgentHandoff()
  submissionError.value = ''
}

function markExperimentNameEdited() {
  experimentNameEdited.value = true
}

function formatHandoffValue(value: unknown) {
  if (value === undefined) return t('wizard.aiHandoffUnset')
  if (typeof value === 'string') return value || t('wizard.aiHandoffEmpty')
  return JSON.stringify(value)
}

function taskLabel(taskType: string) {
  taskType = normalizeTaskType(taskType)
  return t(`tasks.${taskType}.label`)
}

function sourceLabel(source: string) {
  return t(`datasetSource.${source}`)
}

function localizePreflightMessages(messages: string[], codes: string[]) {
  return messages.map((message, index) => {
    const code = codes[index]
    return localizedPreflightMessage(code, message, t, te, {
      datasetTask: preflight.data.value?.dataset.task_type,
      requestedTask: submissionPayload.value.task_type,
      format: preflight.data.value?.dataset.format,
      task: submissionPayload.value.task_type,
    })
  })
}

function agentPreflightMessages(kind: 'errors' | 'warnings') {
  const snapshot = agentHandoff.value?.agent_preflight
  const codes = kind === 'errors' ? (snapshot?.error_codes ?? []) : (snapshot?.warning_codes ?? [])
  const messages = kind === 'errors' ? (snapshot?.errors ?? []) : (snapshot?.warnings ?? [])
  return messages.map((message, index) => localizedPreflightMessage(codes[index], message, t, te))
}

function goToStep(step: number) {
  if (isGuest.value || step <= currentStep.value || draft.datasetId) currentStep.value = step
}

function previousStep() {
  if (currentStep.value > 1) currentStep.value -= 1
}

function nextStep() {
  if (currentStep.value < wizardSteps.value.length && canAdvance.value) currentStep.value += 1
}

useLoginDraft(
  'experiment',
  () => ({
    state: { ...draft.$state },
    step: currentStep.value,
    name: experimentName.value,
    nameEdited: experimentNameEdited.value,
  }),
  (saved) => {
    draft.$patch((state) => Object.assign(state, normalizePairFields(saved.state)))
    currentStep.value = saved.step
    experimentName.value = saved.name
    experimentNameEdited.value = saved.nameEdited
  },
)

watch(
  () => draft.taskType,
  (taskType, previous) => {
    if (taskType !== previous) {
      draft.resetForTask(taskType)
      currentStep.value = 1
    }
  },
)

watch(
  [
    () => selectedDataset.value?.metadata,
    () => draft.tokenizer,
    () => draft.partnerTokenizer,
    () => draft.streamsLinked,
    isPairTask,
  ],
  () => {
    draft.applyProfileTokenizerDatasetLengthFloor(selectedDataset.value?.metadata, isPairTask.value)
  },
  { immediate: true },
)

watch(
  [
    () => selectedDataset.value?.metadata,
    () => draft.tokenizer,
    () => draft.encoder,
    () => draft.partnerTokenizer,
    () => draft.partnerEncoder,
    () => draft.streamsLinked,
    isPairTask,
    pdtDefaultLag,
  ],
  () => {
    draft.applyPdtDatasetLagDefaults(
      selectedDataset.value?.metadata,
      isPairTask.value,
      pdtDefaultLag.value,
    )
  },
  { immediate: true },
)

watch(
  defaultExperimentName,
  (name) => {
    if (!experimentNameEdited.value) experimentName.value = name
  },
  { immediate: true },
)

watch(
  () => draft.featureRoute,
  (route) => {
    if (!capabilities.data.value) return
    if (route === 'pretrained') {
      draft.tokenizer = capabilities.data.value.pretrained_tokenizers[0]
      draft.encoder = draft.tokenizer
      draft.reduction = 'none'
      draft.pretrainedPredictorRoute = 'pooled_classifier'
      return
    }
    const resolved = resolveClassicFeatureRoute(
      draft.tokenizer,
      draft.encoder,
      capabilities.data.value.classic_tokenizers,
      capabilities.data.value.classic_encoder_options,
    )
    draft.tokenizer = resolved.tokenizer
    draft.encoder = resolved.encoder
  },
)

watch(
  () => draft.tokenizer,
  (tokenizer) => {
    if (tokenizer === 'prott5' && draft.maxLength === 512) draft.maxLength = 511
    if (draft.featureRoute === 'pretrained') draft.encoder = draft.tokenizer
    if (!encoders.value.length) return
    if (!encoders.value.includes(draft.encoder)) {
      draft.encoder = encoders.value[0]
    }
  },
)

watch(
  [() => draft.tokenizer, () => draft.encoder, () => isFixedKEncoder(draft.encoder)],
  () => {
    if (draft.tokenizer === 'kmer' && isFixedKEncoder(draft.encoder)) {
      draft.k = 1
    }
  },
  { immediate: true },
)

watch(classicModelingArchitectureOptions, (options) => {
  if (
    needsClassicModelingArchitecture.value &&
    !options.some((option) => option.value === draft.modelingArchitecture)
  ) {
    draft.modelingArchitecture = String(options[0]?.value ?? 'cnn')
  }
})

watch(pretrainedResidueArchitectureOptions, (options) => {
  if (
    usesPretrainedResidueModeling.value &&
    !options.some((option) => option.value === draft.modelingArchitecture)
  ) {
    draft.modelingArchitecture = String(options[0]?.value ?? 'gcn')
  }
})

watch(
  () => draft.pretrainedPredictorRoute,
  (route) => {
    if (route !== 'residue_modeling') return
    draft.modelingArchitecture = String(
      pretrainedResidueArchitectureOptions.value[0]?.value ?? 'cnn',
    )
    draft.trainingMode = 'end_to_end'
    draft.classifier = 'mlp'
  },
)

watch(
  () => draft.partnerFeatureRoute,
  (route) => {
    if (!capabilities.data.value) return
    if (route === 'pretrained') {
      if (!capabilities.data.value.pretrained_tokenizers.includes(draft.partnerTokenizer)) {
        draft.partnerTokenizer = capabilities.data.value.pretrained_tokenizers[0]
      }
      draft.partnerEncoder = draft.partnerTokenizer
      if (!draft.partnerModelSource) {
        draft.partnerModelSource = draft.pretrainedModelSource(draft.partnerTokenizer)
      }
      return
    }
    if (!capabilities.data.value.classic_tokenizers.includes(draft.partnerTokenizer)) {
      draft.partnerTokenizer = capabilities.data.value.classic_tokenizers[0]
    }
    const validEncoders =
      capabilities.data.value.classic_encoder_options[draft.partnerTokenizer] ?? []
    if (!validEncoders.includes(draft.partnerEncoder)) {
      draft.partnerEncoder = validEncoders[0] ?? 'index'
    }
    draft.partnerModelSource = ''
  },
)

watch(
  () => draft.partnerTokenizer,
  (tokenizer, previous) => {
    if (tokenizer === 'prott5' && draft.partnerMaxLength === 512) draft.partnerMaxLength = 511
    if (draft.partnerFeatureRoute === 'pretrained') {
      draft.partnerEncoder = tokenizer
      const previousDefault = previous ? draft.pretrainedModelSource(previous) : ''
      if (!draft.partnerModelSource || draft.partnerModelSource === previousDefault) {
        draft.partnerModelSource = draft.pretrainedModelSource(tokenizer)
      }
      return
    }
    if (partnerEncoders.value.length && !partnerEncoders.value.includes(draft.partnerEncoder)) {
      draft.partnerEncoder = partnerEncoders.value[0]
    }
  },
)

watch(
  [
    () => draft.partnerTokenizer,
    () => draft.partnerEncoder,
    () => isFixedKEncoder(draft.partnerEncoder),
  ],
  () => {
    if (draft.partnerTokenizer === 'kmer' && isFixedKEncoder(draft.partnerEncoder)) {
      draft.partnerK = 1
    }
  },
  { immediate: true },
)

watch(
  () => draft.streamsLinked,
  (linked) => {
    if (!linked && !['concat', 'aligned_gated'].includes(draft.fusionMode)) {
      draft.fusionMode = 'concat'
    }
  },
)

watch(
  () => draft.predictorType,
  (predictor) => {
    if (predictor === 'sample_graph' && !draft.streamsLinked) {
      draft.fusionMode = 'concat'
    }
  },
)

watch(pairStrategyOptions, (values) => {
  if (!values.includes(draft.pairStrategy)) draft.pairStrategy = values[0]
})

watch(interactionLevelOptions, (values) => {
  if (!values.includes(draft.interactionLevel)) draft.interactionLevel = values[0]
})

watch(
  [
    () => draft.trainingMode,
    () => draft.pairStrategy,
    () => draft.classicPredictorRoute,
    needsClassicModelingArchitecture,
    usesPretrainedResidueModeling,
    canUseFlattenedSequenceFeatures,
    usesFlattenedClassifier,
    isPairTask,
    classicalClassifierOptions,
  ],
  () => {
    if (
      !canUseFlattenedSequenceFeatures.value &&
      draft.classicPredictorRoute === 'flattened_classifier'
    ) {
      draft.classicPredictorRoute = 'modeling_architecture'
    }
    if (
      usesFlattenedClassifier.value &&
      !classicalClassifierOptions.value.includes(draft.classifier)
    ) {
      draft.classifier = classicalClassifierOptions.value[0] ?? 'svm'
    }
    const requiresClassical =
      draft.trainingMode === 'export_embeddings' &&
      ((!isPairTask.value &&
        (needsClassicModelingArchitecture.value || usesPretrainedResidueModeling.value)) ||
        (isPairTask.value &&
          (effectivePairStrategy.value === 'single_stream' || usesTokenInteraction.value)))
    if (requiresClassical && !classicalClassifierOptions.value.includes(draft.classifier)) {
      draft.classifier = classicalClassifierOptions.value[0] ?? 'svm'
    }
  },
)
</script>

<template>
  <section class="panel experiment-wizard">
    <PageHeader :eyebrow="wizardEyebrow" :title="t('wizard.title')" compact>
      <template #actions>
        <EnsembleEntry />
        <div class="mode-toggle">
          <button
            class="mode-toggle-button"
            :class="draft.mode === 'recommended' ? 'mode-toggle-button-active' : ''"
            @click="draft.mode = 'recommended'"
          >
            {{ t('common.recommended') }}
          </button>
          <button
            class="mode-toggle-button"
            :class="draft.mode === 'custom' ? 'mode-toggle-button-active' : ''"
            @click="draft.mode = 'custom'"
          >
            {{ t('common.custom') }}
          </button>
        </div>
      </template>
    </PageHeader>
    <section
      v-if="draft.handoff"
      class="wizard-ai-handoff"
      aria-labelledby="wizard-ai-handoff-title"
    >
      <div class="wizard-ai-handoff-head">
        <span class="wizard-ai-handoff-mark"><Sparkles aria-hidden="true" /></span>
        <div>
          <p class="eyebrow">{{ t('wizard.aiHandoffEyebrow') }}</p>
          <h2 id="wizard-ai-handoff-title">{{ draft.handoff.title }}</h2>
          <p>{{ draft.handoff.rationale }}</p>
        </div>
      </div>
      <details v-if="draft.handoff.config_changes.length" class="wizard-ai-handoff-changes">
        <summary>{{
          t('wizard.aiHandoffChanges', { count: draft.handoff.config_changes.length })
        }}</summary>
        <dl>
          <div v-for="change in draft.handoff.config_changes" :key="change.path">
            <dt>{{ change.path }}</dt>
            <dd>
              <span>{{ formatHandoffValue(change.before) }}</span>
              <span aria-hidden="true">→</span>
              <strong>{{ formatHandoffValue(change.after) }}</strong>
            </dd>
          </div>
        </dl>
      </details>
      <div
        v-if="isAgentHandoff && agentHandoff?.agent_preflight"
        class="wizard-agent-preflight-snapshot"
      >
        <div class="wizard-agent-preflight-head">
          <div>
            <p class="eyebrow">{{ t('wizard.agentPreflightSnapshot') }}</p>
            <p>{{ t('wizard.agentPreflightSnapshotHint') }}</p>
          </div>
          <span
            class="status-pill"
            :class="
              agentHandoff.agent_preflight.valid ? 'status-pill-success' : 'status-pill-danger'
            "
          >
            {{
              agentHandoff.agent_preflight.valid
                ? t('wizard.readyToTrain')
                : t('wizard.needsAttention')
            }}
          </span>
        </div>
        <ul
          v-if="agentHandoff.agent_preflight.errors.length"
          class="wizard-agent-preflight-list text-rose-700"
        >
          <li v-for="message in agentPreflightMessages('errors')" :key="message">{{ message }}</li>
        </ul>
        <ul
          v-if="agentHandoff.agent_preflight.warnings.length"
          class="wizard-agent-preflight-list text-amber-700"
        >
          <li v-for="message in agentPreflightMessages('warnings')" :key="message">{{
            message
          }}</li>
        </ul>
      </div>
      <div class="wizard-ai-handoff-footer">
        <p class="wizard-ai-handoff-note">{{
          t(isAgentHandoff ? 'wizard.agentHandoffNote' : 'wizard.aiHandoffNote')
        }}</p>
        <button
          v-if="isAgentHandoff"
          type="button"
          class="tertiary-action compact-action"
          @click="detachAgentHandoff"
        >
          <Link2Off aria-hidden="true" />
          {{ t('wizard.detachAgentHandoff') }}
        </button>
      </div>
    </section>
    <div class="wizard-layout">
      <aside class="section-card wizard-progress">
        <div class="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
          {{ t('wizard.progress', { current: currentStep, total: wizardSteps.length }) }}
        </div>
        <div class="mt-4 space-y-3 wizard-steps">
          <button
            v-for="step in wizardSteps"
            :key="step.id"
            class="step-card"
            :class="
              step.id === currentStep
                ? 'step-card-active'
                : step.id < currentStep || draft.datasetId
                  ? 'step-card-complete'
                  : 'step-card-locked cursor-not-allowed'
            "
            :disabled="!isGuest && step.id > currentStep && !draft.datasetId"
            @click="goToStep(step.id)"
          >
            <div class="text-sm font-medium">{{ step.title }}</div>
            <p class="step-card-description mt-2 text-xs leading-5">
              {{ step.description }}
            </p>
          </button>
        </div>
      </aside>

      <div class="wizard-main">
        <div v-if="currentStep !== 3" class="subtle-card">
          <div class="eyebrow">{{ activeStep.title }}</div>
          <p class="muted-copy mt-3 text-sm leading-6">{{ activeStep.description }}</p>
        </div>

        <template v-if="currentStep === 1">
          <div class="experiment-brief mt-5">
            <article class="experiment-brief-panel experiment-brief-task">
              <header class="experiment-brief-header">
                <span class="experiment-brief-icon">
                  <FlaskConical aria-hidden="true" />
                </span>
                <span class="experiment-brief-kicker">{{ t('wizard.task') }}</span>
              </header>
              <h2 class="experiment-brief-title">{{ taskLabel(draft.taskType) }}</h2>
              <p class="experiment-brief-copy">{{ currentTaskDescription }}</p>
            </article>
            <article class="experiment-brief-panel experiment-brief-route">
              <header class="experiment-brief-header">
                <span class="experiment-brief-icon">
                  <Sparkles aria-hidden="true" />
                </span>
                <span class="experiment-brief-kicker">{{ t('wizard.currentPath') }}</span>
              </header>
              <h2 class="experiment-brief-title">
                {{ draft.mode === 'recommended' ? t('common.recommended') : t('common.custom') }}
              </h2>
              <p class="experiment-brief-copy">{{ currentModeDescription }}</p>
              <p v-if="draft.mode === 'recommended'" class="experiment-brief-recommendation">
                {{ currentTaskRecommendation }}
              </p>
            </article>
          </div>

          <div class="mt-5 grid gap-5 md:grid-cols-2 wizard-field-grid">
            <SelectControl
              v-model="draft.taskType"
              :label="t('wizard.task')"
              :options="taskSelectOptions"
              :disabled="isAgentHandoff"
              multiline
            />
            <SelectControl
              v-model="draft.datasetId"
              :label="t('wizard.dataset')"
              :options="datasetSelectOptions"
              :disabled="isAgentHandoff"
            />
            <p v-if="isAgentHandoff" class="text-xs leading-5 text-slate-500 md:col-span-2">
              {{ t('wizard.agentContextLocked') }}
            </p>
          </div>

          <div class="section-card mt-5">
            <p class="eyebrow">{{ t('wizard.datasetPanel') }}</p>
            <template v-if="selectedDataset">
              <div class="mt-4 font-medium">{{ selectedDataset.name }}</div>
              <div class="mt-1 text-sm text-slate-500">
                {{ selectedDataset.format }} · {{ sourceLabel(selectedDataset.source) }}
              </div>
              <div class="mt-4 flex flex-wrap gap-2">
                <span
                  v-for="(count, split) in selectedDatasetSplits"
                  :key="split"
                  class="dataset-chip"
                >
                  {{ split }}: {{ count }}
                </span>
              </div>
              <p
                v-if="!Object.keys(selectedDatasetSplits).length"
                class="mt-4 text-sm text-slate-500"
              >
                {{ t('wizard.noSplitMetadata') }}
              </p>
            </template>
            <p v-else class="mt-4 text-sm text-slate-500">{{ t('wizard.inspectDataset') }}</p>
          </div>
        </template>

        <template v-else-if="currentStep === 2">
          <div class="mt-5 grid gap-5 md:grid-cols-2 wizard-field-grid">
            <SelectControl
              :model-value="draft.featureRoute"
              :label="t('wizard.featureExtraction')"
              :options="featureRouteSelectOptions"
              @update:model-value="requestRouteFieldChange('featureRoute', $event)"
            />
            <label
              v-if="isPairTask && effectivePairStrategy === 'two_stream' && draft.mode === 'custom'"
              class="flex min-h-11 items-center gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm"
            >
              <input
                :checked="draft.streamsLinked"
                type="checkbox"
                class="h-5 w-5 accent-slate-900"
                @change="onStreamsLinkedChange"
              />
              <span>
                <span class="block font-medium">{{ t('wizard.linkStreams') }}</span>
                <span class="mt-0.5 block text-xs text-slate-500">
                  {{ t('wizard.linkStreamsHint') }}
                </span>
              </span>
            </label>
          </div>

          <div class="section-card mt-5">
            <p class="eyebrow">
              {{
                isPairTask && effectivePairStrategy === 'two_stream'
                  ? t('wizard.peptideStream')
                  : t('wizard.representation')
              }}
            </p>
            <div class="mt-4 grid items-start gap-5 md:grid-cols-3 wizard-field-grid">
              <SelectControl
                :model-value="draft.tokenizer"
                :label="
                  draft.featureRoute === 'pretrained'
                    ? t('wizard.pretrainedModel')
                    : t('wizard.tokenizer')
                "
                :options="stringSelectOptions(tokenizerOptions, tokenizerDisplayName)"
                @update:model-value="requestRouteFieldChange('tokenizer', $event)"
              />
              <SelectControl
                v-if="draft.featureRoute === 'classic'"
                :model-value="draft.encoder"
                :label="t('wizard.encoder')"
                :options="stringSelectOptions(encoders)"
                @update:model-value="requestRouteFieldChange('encoder', $event)"
              />
              <div v-else class="derived-route-field" aria-live="polite">
                <span class="derived-route-field-label">{{ t('wizard.encoder') }}</span>
                <strong>{{ tokenizerDisplayName(draft.encoder) }}</strong>
                <small>{{ t('wizard.pretrainedEncoderCoupled') }}</small>
              </div>
              <SelectControl
                v-if="draft.featureRoute === 'classic' && transformations.length > 1"
                :model-value="draft.reduction"
                :label="t('wizard.featureTransformation')"
                :options="stringSelectOptions(transformations)"
                @update:model-value="selectTransformation"
              />
              <SelectControl
                v-if="
                  draft.mode === 'custom' &&
                  draft.featureRoute === 'pretrained' &&
                  draft.tokenizer === 'esmc'
                "
                :model-value="pretrainedModelSource"
                :label="t('wizard.pretrainedModelSource')"
                :options="pretrainedModelSelectOptions"
                @update:model-value="pretrainedModelSource = String($event)"
              />
              <label
                v-else-if="draft.mode === 'custom' && draft.featureRoute === 'pretrained'"
                class="grid gap-2 text-sm md:col-span-2"
              >
                <span class="wizard-field-label">{{ t('wizard.pretrainedModelSource') }}</span>
                <input
                  v-model.trim="pretrainedModelSource"
                  class="field-control"
                  :placeholder="t('wizard.pretrainedModelSourcePlaceholder')"
                />
                <span class="text-xs leading-5 text-slate-500">
                  {{ t('wizard.pretrainedModelSourceHint') }}
                </span>
              </label>
              <label
                v-if="
                  isPairTask && effectivePairStrategy === 'two_stream' && draft.mode === 'custom'
                "
                class="grid gap-2 text-sm"
              >
                <span class="wizard-field-label">{{
                  maxLengthLabelForTokenizer(draft.tokenizer)
                }}</span>
                <input
                  v-model.number="draft.maxLength"
                  type="number"
                  :min="
                    Math.max(primaryTopNgramMinimum, maxLengthMinimumForTokenizer(draft.tokenizer))
                  "
                  :max="draft.tokenizer === 'prott5' ? 511 : undefined"
                  class="field-control"
                />
                <small v-if="draft.tokenizer === 'kmer'" class="leading-5 text-slate-500">
                  {{ t('wizard.kmerMaxLengthHint') }}
                </small>
                <small
                  v-else-if="draft.tokenizer === 'top-n-gram' && primaryTopNgramMinimum > 1"
                  class="leading-5 text-slate-500"
                >
                  {{ t('wizard.topNgramDatasetLengthHint', { length: primaryTopNgramMinimum }) }}
                </small>
                <small v-else-if="draft.tokenizer === 'prott5'" class="leading-5 text-slate-500">
                  ProtT5 reserves one position for EOS; 512 total tokens means 511 residues.
                </small>
              </label>
            </div>
          </div>

          <aside
            v-if="pairTopicActive"
            class="pair-topic-notice mt-5"
            :class="{ 'pair-topic-notice-warning': activePairTopicIssues.length }"
            aria-live="polite"
          >
            <AlertTriangle v-if="activePairTopicIssues.length" aria-hidden="true" />
            <CircleCheck v-else aria-hidden="true" />
            <div>
              <h3>
                {{
                  activePairTopicIssues.length
                    ? t('wizard.pairTopic.needsRepairTitle', { reduction: pairTopicName })
                    : t('wizard.pairTopic.activeTitle', { reduction: pairTopicName })
                }}
              </h3>
              <p>
                {{
                  activePairTopicIssues.length
                    ? t('wizard.pairTopic.needsRepairDescription')
                    : t('wizard.pairTopic.activeDescription')
                }}
              </p>
              <div v-if="pairTopicAdjustmentLabels.length" class="pair-topic-adjustments">
                <span>{{ t('wizard.pairTopic.adjusted') }}</span>
                <strong>{{ pairTopicAdjustmentLabels.join(' · ') }}</strong>
              </div>
            </div>
            <button
              v-if="activePairTopicIssues.length"
              type="button"
              class="secondary-action"
              @click="repairPairTopicRoute"
            >
              {{ t('wizard.pairTopic.applyPreset') }}
            </button>
          </aside>

          <div
            v-if="
              isPairTask &&
              effectivePairStrategy === 'two_stream' &&
              draft.mode === 'custom' &&
              !draft.streamsLinked
            "
            class="section-card mt-5"
          >
            <div class="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p class="eyebrow">{{ t('wizard.partnerStream') }}</p>
                <p class="mt-2 text-sm leading-6 text-slate-500">
                  {{ t('wizard.partnerStreamHint') }}
                </p>
              </div>
              <span class="status-pill status-pill-marker">heterogeneous</span>
            </div>
            <div class="mt-4 grid items-start gap-5 md:grid-cols-3 wizard-field-grid">
              <SelectControl
                :model-value="draft.partnerFeatureRoute"
                :label="t('wizard.featureExtraction')"
                :options="featureRouteSelectOptions"
                @update:model-value="requestRouteFieldChange('partnerFeatureRoute', $event)"
              />
              <SelectControl
                :model-value="draft.partnerTokenizer"
                :label="
                  draft.partnerFeatureRoute === 'pretrained'
                    ? t('wizard.pretrainedModel')
                    : t('wizard.tokenizer')
                "
                :options="stringSelectOptions(partnerTokenizerOptions, tokenizerDisplayName)"
                @update:model-value="requestRouteFieldChange('partnerTokenizer', $event)"
              />
              <SelectControl
                v-if="draft.partnerFeatureRoute === 'classic'"
                :model-value="draft.partnerEncoder"
                :label="t('wizard.encoder')"
                :options="stringSelectOptions(partnerEncoders)"
                @update:model-value="requestRouteFieldChange('partnerEncoder', $event)"
              />
              <div v-else class="derived-route-field" aria-live="polite">
                <span class="derived-route-field-label">{{ t('wizard.encoder') }}</span>
                <strong>{{ tokenizerDisplayName(draft.partnerEncoder) }}</strong>
                <small>{{ t('wizard.pretrainedEncoderCoupled') }}</small>
              </div>
              <SelectControl
                v-if="
                  draft.partnerFeatureRoute === 'pretrained' && draft.partnerTokenizer === 'esmc'
                "
                :model-value="draft.partnerModelSource"
                :label="t('wizard.pretrainedModelSource')"
                :options="partnerPretrainedModelSelectOptions"
                @update:model-value="draft.partnerModelSource = String($event)"
              />
              <label
                v-else-if="draft.partnerFeatureRoute === 'pretrained'"
                class="grid gap-2 text-sm md:col-span-2"
              >
                <span class="wizard-field-label">{{ t('wizard.pretrainedModelSource') }}</span>
                <input
                  v-model.trim="draft.partnerModelSource"
                  class="field-control"
                  :placeholder="t('wizard.pretrainedModelSourcePlaceholder')"
                />
              </label>
              <label class="grid gap-2 text-sm">
                <span class="wizard-field-label">{{
                  maxLengthLabelForTokenizer(draft.partnerTokenizer)
                }}</span>
                <input
                  v-model.number="draft.partnerMaxLength"
                  type="number"
                  :min="
                    Math.max(
                      partnerTopNgramMinimum,
                      maxLengthMinimumForTokenizer(draft.partnerTokenizer),
                    )
                  "
                  :max="draft.partnerTokenizer === 'prott5' ? 511 : undefined"
                  class="field-control"
                />
                <small v-if="draft.partnerTokenizer === 'kmer'" class="leading-5 text-slate-500">
                  {{ t('wizard.kmerMaxLengthHint') }}
                </small>
                <small
                  v-else-if="draft.partnerTokenizer === 'top-n-gram' && partnerTopNgramMinimum > 1"
                  class="leading-5 text-slate-500"
                >
                  {{ t('wizard.topNgramDatasetLengthHint', { length: partnerTopNgramMinimum }) }}
                </small>
                <small
                  v-else-if="draft.partnerTokenizer === 'prott5'"
                  class="leading-5 text-slate-500"
                >
                  ProtT5 reserves one position for EOS; 512 total tokens means 511 residues.
                </small>
              </label>
              <label v-if="draft.partnerTokenizer === 'kmer'" class="grid gap-2 text-sm">
                <span class="wizard-field-label">{{
                  kLabelForTokenizer(draft.partnerTokenizer)
                }}</span>
                <input
                  v-model.number="partnerKValue"
                  type="number"
                  :min="kmerKConstraint.min"
                  :max="kmerKConstraint.max"
                  :disabled="isFixedKEncoder(draft.partnerEncoder)"
                  :aria-invalid="partnerProfileKInvalid"
                  :aria-describedby="partnerProfileKInvalid ? 'partner-profile-k-error' : undefined"
                  class="field-control"
                />
                <small
                  v-if="partnerProfileKInvalid"
                  id="partner-profile-k-error"
                  class="leading-5 text-rose-700"
                  role="alert"
                >
                  {{
                    t('wizard.profileKRequired', {
                      encoder: draft.partnerEncoder,
                      requiredK: profileRequiredK,
                    })
                  }}
                </small>
                <small v-if="draft.partnerTokenizer === 'kmer'" class="leading-5 text-slate-500">
                  {{ t('wizard.kmerWindowHint', kmerKConstraint) }}
                </small>
                <small v-if="isPdtEncoder(draft.partnerEncoder)" class="leading-5 text-slate-500">
                  {{ t('wizard.pdtKFixed', { requiredK: pdtRequiredK }) }}
                </small>
                <small
                  v-else-if="isCovarianceEncoder(draft.partnerEncoder)"
                  class="leading-5 text-slate-500"
                >
                  {{ t('wizard.covarianceKFixed') }}
                </small>
                <small
                  v-else-if="residueProfileEncoders.has(draft.partnerEncoder)"
                  class="leading-5 text-slate-500"
                >
                  {{ t('wizard.profileKFixed') }}
                </small>
              </label>
              <label
                v-if="isPssmCovarianceEncoder(draft.partnerEncoder)"
                class="grid gap-2 text-sm"
              >
                <span class="wizard-field-label">{{
                  t('wizard.pssmCovarianceLag', { encoder: draft.partnerEncoder.toUpperCase() })
                }}</span>
                <input
                  v-model.number="partnerPssmCovarianceLag"
                  type="number"
                  min="1"
                  class="field-control"
                />
                <small
                  v-if="partnerPssmCovarianceDatasetMinimum > 0"
                  class="leading-5 text-slate-500"
                >
                  {{
                    t('wizard.pssmCovarianceLagDatasetHint', {
                      minimum: partnerPssmCovarianceDatasetMinimum,
                    })
                  }}
                </small>
              </label>
              <label v-if="isCovarianceEncoder(draft.partnerEncoder)" class="grid gap-2 text-sm">
                <span class="wizard-field-label">{{
                  t('wizard.advanced.fields.covarianceLag', {
                    encoder: draft.partnerEncoder.toUpperCase(),
                  })
                }}</span>
                <input
                  v-model.number="partnerCovarianceLag"
                  type="number"
                  min="1"
                  class="field-control"
                />
                <small v-if="partnerCovarianceDatasetMinimum > 0" class="leading-5 text-slate-500">
                  {{
                    t('wizard.covarianceLagDatasetHint', {
                      minimum: partnerCovarianceDatasetMinimum,
                    })
                  }}
                </small>
              </label>
              <label v-if="isPdtEncoder(draft.partnerEncoder)" class="grid gap-2 text-sm">
                <span class="wizard-field-label">{{ t('wizard.pdtLag') }}</span>
                <input
                  v-model.number="partnerPdtLag"
                  type="number"
                  min="1"
                  :max="partnerPdtLagMaximum"
                  class="field-control"
                />
                <small v-if="partnerPdtDatasetMinimum > 1" class="leading-5 text-slate-500">
                  {{
                    t('wizard.pdtLagDatasetHint', {
                      minimum: partnerPdtDatasetMinimum,
                      maximum: partnerPdtLagMaximum,
                    })
                  }}
                </small>
                <small
                  v-else-if="partnerPdtDatasetMinimum === 1"
                  class="leading-5 text-rose-700"
                  role="alert"
                >
                  {{ t('wizard.pdtLagUnavailableHint') }}
                </small>
              </label>
              <label v-if="isPdtEncoder(draft.partnerEncoder)" class="grid gap-2 text-sm">
                <span class="wizard-field-label">{{ t('wizard.pdtMaxRawLength') }}</span>
                <input
                  v-model.number="draft.partnerPdtMaxRawLength"
                  type="number"
                  min="1"
                  :max="pdtMaxRawLength"
                  class="field-control"
                />
                <small v-if="partnerPdtDatasetMaximum > 0" class="leading-5 text-slate-500">
                  {{ t('wizard.pdtRawLengthDatasetHint', { maximum: partnerPdtDatasetMaximum }) }}
                </small>
              </label>
              <label v-if="isCovarianceEncoder(draft.partnerEncoder)" class="grid gap-2 text-sm">
                <span class="wizard-field-label">{{
                  t('wizard.advanced.fields.covarianceMaxRawLength')
                }}</span>
                <input
                  v-model.number="draft.partnerCovarianceMaxRawLength"
                  type="number"
                  min="1"
                  :max="covarianceMaxRawLength"
                  class="field-control"
                />
                <small v-if="partnerCovarianceDatasetMaximum > 0" class="leading-5 text-slate-500">
                  {{
                    t('wizard.covarianceRawLengthDatasetHint', {
                      maximum: partnerCovarianceDatasetMaximum,
                    })
                  }}
                </small>
              </label>
              <label v-if="draft.partnerTokenizer === 'dr'" class="grid gap-2 text-sm">
                <span class="wizard-field-label">DR maximum distance</span>
                <input
                  v-model.number="draft.partnerDrMaxDistance"
                  type="number"
                  min="0"
                  max="200"
                  class="field-control"
                />
              </label>
              <label v-if="draft.partnerTokenizer === 'dr'" class="grid gap-2 text-sm">
                <span class="wizard-field-label">DR maximum raw sequence length</span>
                <input
                  v-model.number="draft.partnerDrMaxRawLength"
                  type="number"
                  min="1"
                  max="100000"
                  class="field-control"
                />
              </label>
              <label
                v-if="
                  draft.partnerFeatureRoute === 'classic' &&
                  ['kmer', 'dr', 'top-n-gram', 'dt'].includes(draft.partnerTokenizer) &&
                  ['bow', 'tfidf', 'tf-idf'].includes(draft.partnerEncoder.toLowerCase())
                "
                class="grid gap-2 text-sm"
              >
                <span class="wizard-field-label">{{
                  t('wizard.advanced.fields.frequencyMaxRawLength')
                }}</span>
                <input
                  v-model.number="draft.frequencyMaxRawLength"
                  type="number"
                  min="1"
                  max="100000"
                  class="field-control"
                />
              </label>
              <label
                v-if="draft.partnerTokenizer === 'dr' && draft.tokenizer !== 'dr'"
                class="grid gap-2 text-sm"
              >
                <span class="wizard-field-label">DR dense memory limit (GiB)</span>
                <input
                  v-model.number="draft.drDenseMemoryLimitGib"
                  type="number"
                  min="0.1"
                  step="0.1"
                  class="field-control"
                />
              </label>
              <label v-if="draft.partnerTokenizer === 'top-n-gram'" class="grid gap-2 text-sm">
                <span class="wizard-field-label">Top n-gram n</span>
                <input
                  v-model.number="draft.partnerTopNgramN"
                  type="number"
                  min="1"
                  max="3"
                  class="field-control"
                />
              </label>
              <label
                v-if="draft.partnerTokenizer === 'top-n-gram' && draft.tokenizer !== 'top-n-gram'"
                class="grid gap-2 text-sm"
              >
                <span class="wizard-field-label">Top-n-gram dense memory limit (GiB)</span>
                <input
                  v-model.number="draft.topNgramDenseMemoryLimitGib"
                  type="number"
                  min="0.1"
                  step="0.1"
                  class="field-control"
                />
              </label>
              <label v-if="draft.partnerTokenizer === 'dt'" class="grid gap-2 text-sm">
                <span class="wizard-field-label">DT maximum distance</span>
                <input
                  v-model.number="draft.partnerDtMaxDistance"
                  type="number"
                  min="0"
                  max="200"
                  class="field-control"
                />
              </label>
              <label v-if="draft.partnerTokenizer === 'dt'" class="grid gap-2 text-sm">
                <span class="wizard-field-label">DT dense memory limit (GiB)</span>
                <input
                  v-model.number="draft.partnerDtDenseMemoryLimitGib"
                  type="number"
                  min="0.1"
                  step="0.1"
                  class="field-control"
                />
              </label>
            </div>
          </div>

          <EffectiveRouteDiagram
            class="mt-5"
            variant="representation"
            :state="draft.mode === 'recommended' ? 'recommended' : 'draft'"
            :route="preflight.data.value?.effective_route"
            :fallback="representationSummary"
            :updating="preflight.isFetching.value"
          />
        </template>

        <template v-else-if="currentStep === 3">
          <div class="subtle-card">
            <div class="eyebrow">{{ activeStep.title }}</div>
            <p class="muted-copy mt-3 text-sm leading-6">{{ activeStep.description }}</p>

            <div class="mt-5 border-t pt-5" style="border-color: var(--line-soft)">
              <aside v-if="pairTopicActive" class="pair-topic-step-notice" aria-live="polite">
                <CircleCheck aria-hidden="true" />
                <p>
                  <strong>{{
                    t('wizard.pairTopic.predictorTitle', { reduction: pairTopicName })
                  }}</strong>
                  {{ t('wizard.pairTopic.predictorDescription') }}
                </p>
              </aside>
              <div v-if="!isPairTask" class="grid gap-5 md:grid-cols-3 wizard-field-grid">
                <SelectControl
                  v-if="canUseFlattenedSequenceFeatures"
                  v-model="draft.classicPredictorRoute"
                  :label="t('wizard.predictorRoute')"
                  :options="classicPredictorRouteSelectOptions"
                />
                <SelectControl
                  v-if="draft.mode === 'custom' && draft.featureRoute === 'pretrained'"
                  v-model="draft.pretrainedPredictorRoute"
                  :label="t('wizard.predictorRoute')"
                  :options="pretrainedPredictorRouteSelectOptions"
                />
                <SelectControl
                  v-if="needsClassicModelingArchitecture || usesPretrainedResidueModeling"
                  v-model="draft.modelingArchitecture"
                  :label="t('wizard.modelingArchitecture')"
                  :options="
                    usesPretrainedResidueModeling
                      ? pretrainedResidueArchitectureOptions
                      : classicModelingArchitectureOptions
                  "
                />
                <SelectControl
                  v-if="usesPretrainedResidueModeling"
                  v-model="draft.trainingMode"
                  :label="t('wizard.trainingMode')"
                  :options="trainingModeSelectOptions"
                />
                <SelectControl
                  v-else-if="needsClassicModelingArchitecture && draft.mode === 'custom'"
                  v-model="draft.trainingMode"
                  :label="t('wizard.trainingMode')"
                  :options="trainingModeSelectOptions"
                />
                <SelectControl
                  v-if="usesPretrainedResidueModeling && draft.trainingMode === 'end_to_end'"
                  :model-value="'mlp'"
                  :label="t('wizard.predictionHead')"
                  :options="fixedMlpSelectOptions"
                  disabled
                />
                <SelectControl
                  v-else-if="
                    draft.featureRoute === 'pretrained' ||
                    !needsClassicModelingArchitecture ||
                    (draft.mode === 'custom' && draft.trainingMode === 'export_embeddings')
                  "
                  v-model="draft.classifier"
                  :label="
                    draft.trainingMode === 'export_embeddings' &&
                    (needsClassicModelingArchitecture || usesPretrainedResidueModeling)
                      ? t('wizard.downstreamClassifier')
                      : t('wizard.classifier')
                  "
                  :options="
                    stringSelectOptions(
                      (draft.trainingMode === 'export_embeddings' &&
                        (needsClassicModelingArchitecture || usesPretrainedResidueModeling)) ||
                        usesFlattenedClassifier
                        ? classicalClassifierOptions
                        : classifierOptions,
                    )
                  "
                />
                <SelectControl
                  v-else-if="needsClassicModelingArchitecture && draft.mode === 'custom'"
                  :model-value="'mlp'"
                  :label="t('wizard.predictionHead')"
                  :options="fixedMlpSelectOptions"
                  disabled
                />
                <label v-if="usesPretrainedResidueModeling" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{ t('wizard.residueAdapterDim') }}</span>
                  <input
                    v-model.number="draft.residueAdapterDim"
                    type="number"
                    min="16"
                    max="1024"
                    step="4"
                    class="field-control"
                  />
                </label>
              </div>
              <div v-else class="grid gap-5 md:grid-cols-3 wizard-field-grid">
                <SelectControl
                  v-if="draft.mode === 'custom'"
                  :model-value="draft.pairStrategy"
                  :label="t('wizard.pairStrategy')"
                  :options="
                    stringSelectOptions(pairStrategyOptions, (value) =>
                      t(`wizard.pairStrategyLabels.${value}`),
                    )
                  "
                  @update:model-value="requestRouteFieldChange('pairStrategy', $event)"
                />
                <template v-if="effectivePairStrategy === 'single_stream'">
                  <SelectControl
                    v-model="draft.modelingArchitecture"
                    :label="t('wizard.modelingArchitecture')"
                    :options="classicModelingArchitectureOptions"
                  />
                  <SelectControl
                    v-model="draft.trainingMode"
                    :label="t('wizard.trainingMode')"
                    :options="trainingModeSelectOptions"
                  />
                </template>
                <template v-else>
                  <SelectControl
                    v-if="draft.mode === 'custom'"
                    :model-value="draft.interactionLevel"
                    :label="t('wizard.interactionLevel')"
                    :options="stringSelectOptions(interactionLevelOptions)"
                    @update:model-value="requestRouteFieldChange('interactionLevel', $event)"
                  />
                  <SelectControl
                    v-if="usesTokenInteraction"
                    v-model="draft.interactionModel"
                    :label="t('wizard.interactionArchitecture')"
                    :options="stringSelectOptions(capabilities.data.value?.interaction_models)"
                  />
                  <SelectControl
                    v-if="usesTokenInteraction"
                    v-model="draft.trainingMode"
                    :label="t('wizard.trainingMode')"
                    :options="trainingModeSelectOptions"
                  />
                  <SelectControl
                    v-if="usesVectorInteraction && draft.mode === 'custom'"
                    :model-value="draft.predictorType"
                    :label="t('wizard.predictorType')"
                    :options="predictorTypeSelectOptions"
                    @update:model-value="requestRouteFieldChange('predictorType', $event)"
                  />
                  <SelectControl
                    v-if="
                      usesVectorInteraction &&
                      draft.mode === 'custom' &&
                      draft.predictorType === 'sample_graph'
                    "
                    v-model="draft.graphPredictor"
                    :label="t('wizard.sampleGraphArchitecture')"
                    :options="stringSelectOptions(capabilities.data.value?.graph_predictors)"
                  />
                </template>
                <SelectControl
                  v-if="
                    (usesVectorInteraction && effectivePredictorType === 'classifier') ||
                    (effectivePairStrategy === 'single_stream' &&
                      draft.trainingMode === 'export_embeddings') ||
                    (usesTokenInteraction && draft.trainingMode === 'export_embeddings')
                  "
                  :model-value="draft.classifier"
                  :label="t('wizard.classifier')"
                  :options="
                    stringSelectOptions(
                      draft.trainingMode === 'export_embeddings' &&
                        (effectivePairStrategy === 'single_stream' || usesTokenInteraction)
                        ? classicalClassifierOptions
                        : classifierOptions,
                    )
                  "
                  @update:model-value="requestRouteFieldChange('classifier', $event)"
                />
                <SelectControl
                  v-if="
                    (effectivePairStrategy === 'single_stream' || usesTokenInteraction) &&
                    draft.trainingMode === 'end_to_end'
                  "
                  :model-value="'mlp'"
                  :label="t('wizard.predictionHead')"
                  :options="fixedMlpSelectOptions"
                  disabled
                />
                <SelectControl
                  v-if="
                    usesVectorInteraction &&
                    draft.mode === 'custom' &&
                    !(effectivePredictorType === 'sample_graph' && draft.graphPredictor === 'hgt')
                  "
                  :model-value="draft.fusionMode"
                  :label="t('wizard.fusion')"
                  :options="fusionSelectOptions"
                  @update:model-value="requestRouteFieldChange('fusionMode', $event)"
                />
                <label
                  v-if="
                    usesVectorInteraction &&
                    draft.mode === 'custom' &&
                    draft.fusionMode === 'aligned_gated' &&
                    !(effectivePredictorType === 'sample_graph' && draft.graphPredictor === 'hgt')
                  "
                  class="grid gap-2 text-sm"
                >
                  <span class="wizard-field-label">Alignment dimension</span>
                  <input
                    v-model.number="draft.streamAlignmentDim"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
              </div>
            </div>

            <div
              v-if="usesEpochSchedule"
              class="mt-5 grid gap-5 border-t pt-5 md:grid-cols-2 wizard-field-grid"
              style="border-color: var(--line-soft)"
            >
              <label class="grid gap-2 text-sm">
                <span class="wizard-field-label">{{ t('wizard.epochs') }}</span>
                <input v-model.number="draft.epochs" type="number" class="field-control" />
              </label>
              <label class="grid gap-2 text-sm">
                <span class="wizard-field-label">{{ t('wizard.batchSize') }}</span>
                <input v-model.number="draft.batchSize" type="number" class="field-control" />
              </label>
            </div>
          </div>

          <EffectiveRouteDiagram
            class="mt-5"
            variant="draft"
            :state="draft.mode === 'recommended' ? 'recommended' : 'draft'"
            :route="preflight.data.value?.effective_route"
            :fallback="routeSummary"
            :updating="preflight.isFetching.value"
          />

          <div v-if="draft.mode === 'custom'" class="advanced-panel mt-5">
            <div class="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p class="eyebrow">{{ t('wizard.advanced.title') }}</p>
                <p class="muted-copy mt-2 text-sm leading-6">{{ t('wizard.advanced.subtitle') }}</p>
              </div>
              <span class="status-pill status-pill-marker">{{
                t('wizard.advanced.customOnly')
              }}</span>
            </div>

            <div v-if="tokenizerParameterVisible" class="advanced-section">
              <div>
                <h3>{{ t('wizard.advanced.groups.tokenizer') }}</h3>
                <p>{{ t('wizard.advanced.groupHints.tokenizer') }}</p>
              </div>
              <div class="advanced-grid wizard-field-grid">
                <label v-if="draft.tokenizer === 'kmer'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{ kLabelForTokenizer(draft.tokenizer) }}</span>
                  <input
                    v-model.number="primaryKValue"
                    type="number"
                    :min="kmerKConstraint.min"
                    :max="kmerKConstraint.max"
                    :disabled="isFixedKEncoder(draft.encoder)"
                    :aria-invalid="primaryProfileKInvalid"
                    :aria-describedby="primaryProfileKInvalid ? 'profile-k-error' : undefined"
                    class="field-control"
                  />
                  <small
                    v-if="primaryProfileKInvalid"
                    id="profile-k-error"
                    class="leading-5 text-rose-700"
                    role="alert"
                  >
                    {{
                      t('wizard.profileKRequired', {
                        encoder: draft.encoder,
                        requiredK: profileRequiredK,
                      })
                    }}
                  </small>
                  <small v-if="draft.tokenizer === 'kmer'" class="leading-5 text-slate-500">
                    {{ t('wizard.kmerWindowHint', kmerKConstraint) }}
                  </small>
                  <small v-if="isPdtEncoder(draft.encoder)" class="leading-5 text-slate-500">
                    {{ t('wizard.pdtKFixed', { requiredK: pdtRequiredK }) }}
                  </small>
                  <small
                    v-else-if="isCovarianceEncoder(draft.encoder)"
                    class="leading-5 text-slate-500"
                  >
                    {{ t('wizard.covarianceKFixed') }}
                  </small>
                  <small
                    v-else-if="residueProfileEncoders.has(draft.encoder)"
                    class="leading-5 text-slate-500"
                  >
                    {{ t('wizard.profileKFixed') }}
                  </small>
                </label>
                <label v-if="isPssmCovarianceEncoder(draft.encoder)" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.pssmCovarianceLag', { encoder: draft.encoder.toUpperCase() })
                  }}</span>
                  <input
                    v-model.number="primaryPssmCovarianceLag"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                  <small
                    v-if="primaryPssmCovarianceDatasetMinimum > 0"
                    class="leading-5 text-slate-500"
                  >
                    {{
                      t('wizard.pssmCovarianceLagDatasetHint', {
                        minimum: primaryPssmCovarianceDatasetMinimum,
                      })
                    }}
                  </small>
                </label>
                <label v-if="isCovarianceEncoder(draft.encoder)" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.covarianceLag', {
                      encoder: draft.encoder.toUpperCase(),
                    })
                  }}</span>
                  <input
                    v-model.number="primaryCovarianceLag"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                  <small
                    v-if="primaryCovarianceDatasetMinimum > 0"
                    class="leading-5 text-slate-500"
                  >
                    {{
                      t('wizard.covarianceLagDatasetHint', {
                        minimum: primaryCovarianceDatasetMinimum,
                      })
                    }}
                  </small>
                </label>
                <label v-if="isPdtEncoder(draft.encoder)" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{ t('wizard.pdtLag') }}</span>
                  <input
                    v-model.number="primaryPdtLag"
                    type="number"
                    min="1"
                    :max="primaryPdtLagMaximum"
                    class="field-control"
                  />
                  <small v-if="primaryPdtDatasetMinimum > 1" class="leading-5 text-slate-500">
                    {{
                      t('wizard.pdtLagDatasetHint', {
                        minimum: primaryPdtDatasetMinimum,
                        maximum: primaryPdtLagMaximum,
                      })
                    }}
                  </small>
                  <small
                    v-else-if="primaryPdtDatasetMinimum === 1"
                    class="leading-5 text-rose-700"
                    role="alert"
                  >
                    {{ t('wizard.pdtLagUnavailableHint') }}
                  </small>
                </label>
                <label v-if="isPdtEncoder(draft.encoder)" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{ t('wizard.pdtMaxRawLength') }}</span>
                  <input
                    v-model.number="draft.pdtMaxRawLength"
                    type="number"
                    min="1"
                    :max="pdtMaxRawLength"
                    class="field-control"
                  />
                  <small v-if="primaryPdtDatasetMaximum > 0" class="leading-5 text-slate-500">
                    {{ t('wizard.pdtRawLengthDatasetHint', { maximum: primaryPdtDatasetMaximum }) }}
                  </small>
                </label>
                <label v-if="isCovarianceEncoder(draft.encoder)" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.covarianceMaxRawLength')
                  }}</span>
                  <input
                    v-model.number="draft.covarianceMaxRawLength"
                    type="number"
                    min="1"
                    :max="covarianceMaxRawLength"
                    class="field-control"
                  />
                  <small
                    v-if="primaryCovarianceDatasetMaximum > 0"
                    class="leading-5 text-slate-500"
                  >
                    {{
                      t('wizard.covarianceRawLengthDatasetHint', {
                        maximum: primaryCovarianceDatasetMaximum,
                      })
                    }}
                  </small>
                </label>
                <label v-if="draft.tokenizer === 'dr'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.drMaxDistance')
                  }}</span>
                  <input
                    v-model.number="draft.drMaxDistance"
                    type="number"
                    min="0"
                    max="200"
                    class="field-control"
                  />
                </label>
                <label
                  v-if="
                    draft.featureRoute === 'classic' &&
                    ['kmer', 'dr', 'top-n-gram', 'dt'].includes(draft.tokenizer) &&
                    ['bow', 'tfidf', 'tf-idf'].includes(draft.encoder.toLowerCase())
                  "
                  class="grid gap-2 text-sm"
                >
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.frequencyMaxRawLength')
                  }}</span>
                  <input
                    v-model.number="draft.frequencyMaxRawLength"
                    type="number"
                    min="1"
                    max="100000"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.tokenizer === 'dr'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.drMaxRawLength')
                  }}</span>
                  <input
                    v-model.number="draft.drMaxRawLength"
                    type="number"
                    min="1"
                    max="100000"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.tokenizer === 'dr'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.drDenseMemoryLimitGib')
                  }}</span>
                  <input
                    v-model.number="draft.drDenseMemoryLimitGib"
                    type="number"
                    min="0.1"
                    step="0.1"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.tokenizer === 'top-n-gram'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.topNgramN')
                  }}</span>
                  <input
                    v-model.number="draft.topNgramN"
                    type="number"
                    min="1"
                    max="3"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.tokenizer === 'top-n-gram'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.topNgramDenseMemoryLimitGib')
                  }}</span>
                  <input
                    v-model.number="draft.topNgramDenseMemoryLimitGib"
                    type="number"
                    min="0.1"
                    step="0.1"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.tokenizer === 'dt'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.dtMaxDistance')
                  }}</span>
                  <input
                    v-model.number="draft.dtMaxDistance"
                    type="number"
                    min="0"
                    max="200"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.tokenizer === 'dt'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.dtDenseMemoryLimitGib')
                  }}</span>
                  <input
                    v-model.number="draft.dtDenseMemoryLimitGib"
                    type="number"
                    min="0.1"
                    step="0.1"
                    class="field-control"
                  />
                </label>
              </div>
            </div>

            <div v-if="staticEmbeddingParameterVisible" class="advanced-section">
              <div>
                <h3>{{ t('wizard.advanced.groups.staticEmbedding') }}</h3>
                <p>{{ t('wizard.advanced.groupHints.staticEmbedding') }}</p>
              </div>
              <div class="advanced-grid wizard-field-grid">
                <label v-if="predictiveStaticParameterVisible" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.embeddingVectorSize')
                  }}</span>
                  <input
                    v-model.number="draft.embeddingVectorSize"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.embeddingWindow')
                  }}</span>
                  <input
                    v-model.number="draft.embeddingWindow"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.embeddingMinCount')
                  }}</span>
                  <input
                    v-model.number="draft.embeddingMinCount"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.embeddingEpochs')
                  }}</span>
                  <input
                    v-model.number="draft.embeddingEpochs"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.embeddingWorkers')
                  }}</span>
                  <input
                    v-model.number="draft.embeddingWorkers"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label v-if="gloveParameterVisible" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.gloveXMax')
                  }}</span>
                  <input
                    v-model.number="draft.gloveXMax"
                    type="number"
                    min="0.000001"
                    step="any"
                    class="field-control"
                  />
                </label>
                <label v-if="gloveParameterVisible" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.gloveAlpha')
                  }}</span>
                  <input
                    v-model.number="draft.gloveAlpha"
                    type="number"
                    min="0.000001"
                    max="1"
                    step="any"
                    class="field-control"
                  />
                </label>
                <label v-if="gloveParameterVisible" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.gloveLearningRate')
                  }}</span>
                  <input
                    v-model.number="draft.gloveLearningRate"
                    type="number"
                    min="0.000001"
                    step="any"
                    class="field-control"
                  />
                </label>
                <label v-if="gloveParameterVisible" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.gloveBatchSize')
                  }}</span>
                  <input
                    v-model.number="draft.gloveBatchSize"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label v-if="fasttextParameterVisible" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.fasttextBucket')
                  }}</span>
                  <input
                    v-model.number="draft.fasttextBucket"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label v-if="fasttextParameterVisible" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.staticEmbeddingMemoryLimitGib')
                  }}</span>
                  <input
                    v-model.number="draft.staticEmbeddingMemoryLimitGib"
                    type="number"
                    min="0.1"
                    step="0.1"
                    class="field-control"
                  />
                </label>
              </div>
            </div>

            <div v-if="predictionHeadVisible" class="advanced-section">
              <div>
                <h3>{{ t('wizard.advanced.groups.predictionHead') }}</h3>
                <p>{{ t('wizard.advanced.groupHints.predictionHead') }}</p>
              </div>
              <div class="advanced-grid wizard-field-grid">
                <fieldset class="head-layer-fieldset">
                  <legend>{{ t('wizard.advanced.fields.headPreset') }}</legend>
                  <div class="head-layer-segments">
                    <label
                      v-for="option in headLayerOptions"
                      :key="option.value"
                      class="head-layer-option"
                    >
                      <input
                        v-model.number="draft.headHiddenLayerCount"
                        type="radio"
                        name="prediction-head-layers"
                        :value="option.value"
                      />
                      <span>
                        <strong>{{ option.label }}</strong>
                        <small>{{ option.detail }}</small>
                      </span>
                    </label>
                  </div>
                </fieldset>
                <SelectControl
                  v-if="draft.headHiddenLayerCount !== 0"
                  v-model="draft.headActivation"
                  :label="t('wizard.advanced.fields.headActivation')"
                  :options="headActivationSelectOptions"
                />
              </div>
              <div class="head-structure-preview" aria-live="polite">
                <span>{{ t('wizard.advanced.headStructure.label') }}</span>
                <code>{{ headStructurePreview }}</code>
                <small
                  v-if="
                    effectiveHeadHiddenDims.length &&
                    draft.enableRegularization &&
                    draft.headDropout > 0
                  "
                >
                  {{ t('wizard.advanced.headStructure.dropout', { value: draft.headDropout }) }}
                </small>
              </div>
            </div>

            <div v-if="!isPairTask" class="advanced-section">
              <div>
                <h3>{{ t('wizard.advanced.groups.ensemble') }}</h3>
                <p>{{ t('wizard.advanced.groupHints.ensemble') }}</p>
              </div>
              <label class="advanced-toggle">
                <input v-model="draft.enableEnsemble" type="checkbox" />
                <span>{{ t('wizard.advanced.fields.enableEnsemble') }}</span>
              </label>
              <div v-if="draft.enableEnsemble" class="advanced-grid wizard-field-grid">
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.ensembleMembers')
                  }}</span>
                  <input
                    v-model="draft.ensembleMembers"
                    type="text"
                    class="field-control"
                    :placeholder="t('wizard.advanced.fields.ensembleMembersPlaceholder')"
                  />
                </label>
                <SelectControl
                  v-model="draft.ensembleMethod"
                  :label="t('wizard.advanced.fields.ensembleMethod')"
                  :options="[{ value: 'probability_mean', label: 'probability_mean' }]"
                />
                <SelectControl
                  v-model="draft.ensembleWeightSource"
                  :label="t('wizard.advanced.fields.ensembleWeightSource')"
                  :options="[
                    { value: 'equal', label: 'equal' },
                    { value: 'validation', label: 'validation' },
                  ]"
                />
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.ensembleGridStep')
                  }}</span>
                  <input
                    v-model.number="draft.ensembleGridStep"
                    type="number"
                    min="0.01"
                    max="0.5"
                    step="0.01"
                    class="field-control"
                  />
                </label>
              </div>
            </div>

            <div class="advanced-section">
              <div>
                <h3>{{ t('wizard.advanced.groups.optimization') }}</h3>
                <p>{{ t('wizard.advanced.groupHints.optimization') }}</p>
              </div>
              <div class="advanced-grid wizard-field-grid">
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    maxLengthLabelForTokenizer(draft.tokenizer)
                  }}</span>
                  <input
                    v-model.number="draft.maxLength"
                    type="number"
                    :min="primaryTopNgramMinimum"
                    class="field-control"
                  />
                  <small v-if="draft.tokenizer === 'kmer'" class="leading-5 text-slate-500">
                    {{ t('wizard.kmerMaxLengthHint') }}
                  </small>
                  <small
                    v-else-if="draft.tokenizer === 'top-n-gram' && primaryTopNgramMinimum > 1"
                    class="leading-5 text-slate-500"
                  >
                    {{ t('wizard.topNgramDatasetLengthHint', { length: primaryTopNgramMinimum }) }}
                  </small>
                </label>
                <label v-if="usesTrainableDl" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.learningRate')
                  }}</span>
                  <input
                    v-model.number="draft.learningRate"
                    type="number"
                    min="0"
                    step="0.0001"
                    class="field-control"
                  />
                </label>
                <SelectControl
                  v-if="dlScalerVisible"
                  v-model="draft.dlScaler"
                  :label="t('wizard.advanced.fields.dlScaler')"
                  :options="dlScalerSelectOptions"
                />
              </div>
            </div>

            <ClassificationLossSettings
              v-if="classificationLossApplies(currentConfig)"
              v-model="draft.classificationLossOverrides"
              :is-pair-task="isPairTask"
              :default-loss="defaultClassificationLoss(currentConfig, isPairTask)"
            />

            <div v-if="regularizationRelevant" class="advanced-section">
              <div>
                <h3>{{ t('wizard.advanced.groups.regularization') }}</h3>
                <p>{{ t('wizard.advanced.groupHints.regularization') }}</p>
              </div>
              <label class="advanced-toggle">
                <input v-model="draft.enableRegularization" type="checkbox" />
                <span>{{ t('wizard.advanced.fields.enableRegularization') }}</span>
              </label>
              <div
                v-if="usesTrainableDl && draft.enableRegularization"
                class="advanced-grid wizard-field-grid"
              >
                <label v-if="modelDropoutVisible" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{ modelDropoutLabel }}</span>
                  <input
                    v-model.number="draft.dropout"
                    type="number"
                    min="0"
                    max="0.99"
                    step="0.05"
                    class="field-control"
                  />
                  <small class="leading-5 text-slate-500">{{
                    t('wizard.advanced.fields.dropoutHelp')
                  }}</small>
                </label>
                <label
                  v-if="predictionHeadVisible && draft.headHiddenLayerCount !== 0"
                  class="grid gap-2 text-sm"
                >
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.headDropout')
                  }}</span>
                  <input
                    v-model.number="draft.headDropout"
                    type="number"
                    min="0"
                    max="0.99"
                    step="0.05"
                    class="field-control"
                  />
                  <small class="leading-5 text-slate-500">{{
                    t('wizard.advanced.fields.headDropoutHelp')
                  }}</small>
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.weightDecay')
                  }}</span>
                  <input
                    v-model.number="draft.weightDecay"
                    type="number"
                    min="0"
                    step="0.0001"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.gradClipNorm')
                  }}</span>
                  <input
                    v-model.number="draft.gradClipNorm"
                    type="number"
                    min="0"
                    step="0.1"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.earlyStopPatience')
                  }}</span>
                  <input
                    v-model.number="draft.earlyStopPatience"
                    type="number"
                    min="0"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.earlyStopMinDelta')
                  }}</span>
                  <input
                    v-model.number="draft.earlyStopMinDelta"
                    type="number"
                    min="0"
                    step="0.0001"
                    class="field-control"
                  />
                </label>
              </div>
            </div>

            <div v-if="semanticParameterVisible" class="advanced-section">
              <div>
                <h3>{{ t('wizard.advanced.groups.semantic') }}</h3>
                <p>{{ t('wizard.advanced.groupHints.semantic') }}</p>
              </div>
              <label class="advanced-toggle">
                <input v-model="draft.enableSemanticSimilarity" type="checkbox" />
                <span>{{ t('wizard.advanced.fields.enableSemanticSimilarity') }}</span>
              </label>
              <div v-if="draft.enableSemanticSimilarity" class="advanced-grid wizard-field-grid">
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.semanticLossType')
                  }}</span>
                  <select v-model="draft.semanticLossType" class="field-control">
                    <option value="legacy">{{
                      t('wizard.advanced.fields.semanticLossLegacy')
                    }}</option>
                    <option value="jaccard_supcon">{{
                      t('wizard.advanced.fields.semanticLossJaccard')
                    }}</option>
                  </select>
                  <span class="text-xs text-slate-500">{{
                    t('wizard.advanced.fields.semanticLossHint')
                  }}</span>
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.semanticSimilarityWeight')
                  }}</span>
                  <input
                    v-model.number="draft.semanticSimilarityWeight"
                    type="number"
                    min="0"
                    step="0.01"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.semanticZeroWeight')
                  }}</span>
                  <input
                    v-model.number="draft.semanticZeroWeight"
                    type="number"
                    min="0"
                    step="0.01"
                    class="field-control"
                  />
                </label>
                <label for="semantic-positive-partners" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.semanticPositivePartners')
                  }}</span>
                  <input
                    id="semantic-positive-partners"
                    v-model.number="draft.semanticPositivePartners"
                    type="number"
                    min="1"
                    step="1"
                    required
                    :aria-invalid="semanticPositivePartnersInvalid"
                    :aria-describedby="
                      semanticPositivePartnersInvalid
                        ? 'semantic-positive-partners-help semantic-positive-partners-error'
                        : 'semantic-positive-partners-help'
                    "
                    class="field-control"
                  />
                  <small id="semantic-positive-partners-help" class="leading-5 text-slate-500">
                    {{ t('wizard.advanced.fields.semanticPositivePartnersHelp') }}
                  </small>
                  <small
                    v-if="semanticPositivePartnersInvalid"
                    id="semantic-positive-partners-error"
                    role="alert"
                    class="text-rose-700"
                  >
                    {{ t('preflightMessages.semantic_positive_partners_invalid') }}
                  </small>
                </label>
                <label for="semantic-zero-partners" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.semanticZeroPartners')
                  }}</span>
                  <input
                    id="semantic-zero-partners"
                    v-model.number="draft.semanticZeroPartners"
                    type="number"
                    min="0"
                    step="1"
                    required
                    :aria-invalid="semanticZeroPartnersInvalid"
                    :aria-describedby="
                      semanticZeroPartnersInvalid
                        ? 'semantic-zero-partners-help semantic-zero-partners-error'
                        : 'semantic-zero-partners-help'
                    "
                    class="field-control"
                  />
                  <small id="semantic-zero-partners-help" class="leading-5 text-slate-500">
                    {{ t('wizard.advanced.fields.semanticZeroPartnersHelp') }}
                  </small>
                  <small
                    v-if="semanticZeroPartnersInvalid"
                    id="semantic-zero-partners-error"
                    role="alert"
                    class="text-rose-700"
                  >
                    {{ t('preflightMessages.semantic_zero_partners_invalid') }}
                  </small>
                </label>
                <p class="text-sm text-slate-500">{{ semanticRelationType }}</p>
              </div>
            </div>

            <div
              v-if="
                featureTransformParameterVisible || mlProcessingVisible || usesClassicalClassifier
              "
              class="advanced-section"
            >
              <div>
                <h3>{{ t('wizard.advanced.groups.classifier') }}</h3>
                <p>{{ t('wizard.advanced.groupHints.classifier') }}</p>
              </div>

              <div v-if="featureTransformParameterVisible" class="advanced-grid wizard-field-grid">
                <label v-if="draft.reduction === 'lsa'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{ t('wizard.advanced.fields.lsaDim') }}</span>
                  <input
                    v-model.number="draft.lsaDim"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.reduction === 'lsa'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.lsaDenseMemoryLimitGib')
                  }}</span>
                  <input
                    v-model.number="draft.lsaDenseMemoryLimitGib"
                    type="number"
                    min="0.01"
                    step="0.25"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.reduction === 'lda'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.ldaTopics')
                  }}</span>
                  <input
                    v-model.number="draft.ldaTopics"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.reduction === 'lda'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.ldaDenseMemoryLimitGib')
                  }}</span>
                  <input
                    v-model.number="draft.ldaDenseMemoryLimitGib"
                    type="number"
                    min="0.01"
                    step="0.25"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.reduction === 'plsa'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.plsaTopics')
                  }}</span>
                  <input
                    v-model.number="draft.plsaTopics"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.reduction === 'plsa'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.plsaMemoryLimitGib')
                  }}</span>
                  <input
                    v-model.number="draft.plsaMemoryLimitGib"
                    type="number"
                    min="0.01"
                    step="0.25"
                    class="field-control"
                  />
                </label>
              </div>

              <div v-if="mlProcessingVisible" class="advanced-grid wizard-field-grid">
                <SelectControl
                  v-model="draft.mlScaler"
                  :label="t('wizard.advanced.fields.mlScaler')"
                  :options="mlScalerSelectOptions"
                />
                <SelectControl
                  v-model="draft.mlReduction"
                  :label="t('wizard.advanced.fields.mlReduction')"
                  :options="mlReductionSelectOptions"
                />
                <label v-if="draft.mlReduction !== 'none'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.mlReductionDim')
                  }}</span>
                  <input
                    v-model.number="draft.mlReductionDim"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.mlReduction === 'tsvd'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.tsvdDenseMemoryLimitGib')
                  }}</span>
                  <input
                    v-model.number="draft.tsvdDenseMemoryLimitGib"
                    type="number"
                    min="0.01"
                    step="0.25"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.mlReduction === 'pca'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.pcaDenseMemoryLimitGib')
                  }}</span>
                  <input
                    v-model.number="draft.pcaDenseMemoryLimitGib"
                    type="number"
                    min="0.01"
                    step="0.25"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.mlReduction === 'kpca'" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.kpcaMemoryLimitGib')
                  }}</span>
                  <input
                    v-model.number="draft.kpcaMemoryLimitGib"
                    type="number"
                    min="0.01"
                    step="0.25"
                    class="field-control"
                  />
                </label>
                <SelectControl
                  v-if="draft.mlReduction === 'kpca'"
                  v-model="draft.kpcaKernel"
                  :label="t('wizard.advanced.fields.kpcaKernel')"
                  :options="kpcaKernelSelectOptions"
                />
                <label
                  v-if="draft.mlReduction === 'kpca' && ['rbf', 'poly'].includes(draft.kpcaKernel)"
                  class="grid gap-2 text-sm"
                >
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.kpcaGamma')
                  }}</span>
                  <input
                    v-model.number="draft.kpcaGamma"
                    type="number"
                    min="0.000000001"
                    step="0.01"
                    class="field-control"
                  />
                </label>
              </div>

              <label v-if="autoTuneVisible" class="advanced-toggle">
                <input v-model="draft.mlAutoTune" type="checkbox" />
                <span>{{ t('wizard.advanced.fields.mlAutoTune') }}</span>
              </label>
              <SelectionObjectives
                :draft="draft"
                :objectives="capabilities.data.value?.selection_objectives"
                :pair="isPairTask"
                :checkpoint="usesTrainableDl"
                :tuning="autoTuneVisible && draft.mlAutoTune"
              />
              <fieldset
                v-if="autoTuneVisible && draft.mlAutoTune"
                class="mt-4"
                aria-describedby="tuning-budget-hint"
              >
                <legend class="text-sm font-semibold">{{
                  t('wizard.advanced.fields.mlTuningBudget')
                }}</legend>
                <p id="tuning-budget-hint" class="mt-1 max-w-3xl text-sm leading-6 text-slate-500">
                  {{ t('wizard.advanced.fields.mlTuningBudgetHint') }}
                </p>
                <div class="mt-3 grid gap-3 md:grid-cols-3">
                  <label
                    v-for="budgetName in tuningBudgetNames"
                    :key="budgetName"
                    class="min-h-28 cursor-pointer rounded-xl border p-4 transition focus-within:ring-2 focus-within:ring-offset-2"
                    :class="
                      draft.mlTuningBudget === budgetName
                        ? 'border-amber-700 bg-amber-50/70'
                        : 'border-slate-300 bg-white/50'
                    "
                  >
                    <span class="flex items-center justify-between gap-3">
                      <span class="font-semibold">{{
                        t(`wizard.advanced.tuningBudgets.${budgetName}`)
                      }}</span>
                      <input
                        v-model="draft.mlTuningBudget"
                        type="radio"
                        name="ml-tuning-budget"
                        :value="budgetName"
                      />
                    </span>
                    <span class="mt-3 block text-sm text-slate-600">
                      {{
                        t('wizard.advanced.fields.mlTuningBudgetCost', {
                          candidates: tuningBudgetContracts[budgetName]?.max_candidates,
                          folds: tuningBudgetContracts[budgetName]?.cv_folds,
                          evaluations:
                            tuningBudgetContracts[budgetName]?.max_candidate_fold_evaluations,
                        })
                      }}
                    </span>
                    <span
                      v-if="budgetName === 'thorough'"
                      class="mt-2 inline-flex rounded-full bg-amber-100 px-2 py-1 text-xs font-semibold text-amber-800"
                    >
                      {{ t('wizard.advanced.fields.mlTuningHighCost') }}
                    </span>
                  </label>
                </div>
              </fieldset>

              <div v-if="effectiveClassifier === 'svm'" class="advanced-grid wizard-field-grid">
                <SelectControl
                  v-model="draft.svmCalibration"
                  :label="t('wizard.advanced.fields.svmCalibration')"
                  :options="svmCalibrationSelectOptions"
                />
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{ t('wizard.advanced.fields.svmC') }}</span>
                  <input
                    v-model.number="draft.svmC"
                    type="number"
                    min="0"
                    step="0.1"
                    class="field-control"
                  />
                </label>
                <SelectControl
                  v-model="draft.svmClassWeight"
                  :label="t('wizard.advanced.fields.svmClassWeight')"
                  :options="classWeightSelectOptions"
                  :disabled="!draft.enableRegularization"
                />
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.svmMaxIter')
                  }}</span>
                  <input
                    v-model.number="draft.svmMaxIter"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{ t('wizard.advanced.fields.svmTol') }}</span>
                  <input
                    v-model.number="draft.svmTol"
                    type="number"
                    min="0"
                    step="0.0001"
                    class="field-control"
                  />
                </label>
                <SelectControl
                  v-model="draft.svmKernel"
                  :label="t('wizard.advanced.fields.svmKernel')"
                  :options="svmKernelSelectOptions"
                />
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{ t('wizard.advanced.fields.svmGamma') }}</span>
                  <input v-model.trim="draft.svmGamma" type="text" class="field-control" />
                </label>
                <div
                  v-if="effectiveClassifier === 'svm'"
                  class="grid gap-2 text-sm wizard-field-grid"
                >
                  <SelectControl
                    v-model="draft.svmConvergencePolicy"
                    :label="t('wizard.advanced.fields.svmConvergencePolicy')"
                    :options="svmConvergencePolicySelectOptions"
                  />
                  <p class="text-xs leading-5 text-slate-500">
                    {{ t('wizard.advanced.fields.svmConvergencePolicyHint') }}
                  </p>
                </div>
              </div>

              <div v-if="effectiveClassifier === 'logreg'" class="advanced-grid wizard-field-grid">
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{ t('wizard.advanced.fields.logregC') }}</span>
                  <input
                    v-model.number="draft.logregC"
                    type="number"
                    min="0.000001"
                    step="0.1"
                    class="field-control"
                  />
                </label>
                <SelectControl
                  v-model="draft.logregClassWeight"
                  :label="t('wizard.advanced.fields.logregClassWeight')"
                  :options="classWeightSelectOptions"
                  :disabled="!draft.enableRegularization"
                />
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.logregMaxIter')
                  }}</span>
                  <input
                    v-model.number="draft.logregMaxIter"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <div class="grid gap-2 text-sm wizard-field-grid">
                  <SelectControl
                    v-model="draft.logregConvergencePolicy"
                    :label="t('wizard.advanced.fields.logregConvergencePolicy')"
                    :options="logregConvergencePolicySelectOptions"
                  />
                  <p class="text-xs leading-5 text-slate-500">
                    {{ t('wizard.advanced.fields.logregConvergencePolicyHint') }}
                  </p>
                </div>
              </div>

              <div v-if="effectiveClassifier === 'rf'" class="advanced-grid wizard-field-grid">
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.rfEstimators')
                  }}</span>
                  <input
                    v-model.number="draft.rfEstimators"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.rfMaxDepth')
                  }}</span>
                  <input
                    v-model.number="draft.rfMaxDepth"
                    type="number"
                    min="0"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.rfMinSamplesLeaf')
                  }}</span>
                  <input
                    v-model.number="draft.rfMinSamplesLeaf"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
              </div>

              <div
                v-if="effectiveClassifier === 'extratrees'"
                class="advanced-grid wizard-field-grid"
              >
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.etEstimators')
                  }}</span>
                  <input
                    v-model.number="draft.etEstimators"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.etMaxDepth')
                  }}</span>
                  <input
                    v-model.number="draft.etMaxDepth"
                    type="number"
                    min="0"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.etMinSamplesLeaf')
                  }}</span>
                  <input
                    v-model.number="draft.etMinSamplesLeaf"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
              </div>

              <div v-if="effectiveClassifier === 'histgb'" class="advanced-grid wizard-field-grid">
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.histgbMaxIter')
                  }}</span>
                  <input
                    v-model.number="draft.histgbMaxIter"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.histgbMaxLeafNodes')
                  }}</span>
                  <input
                    v-model.number="draft.histgbMaxLeafNodes"
                    type="number"
                    min="0"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.histgbLearningRate')
                  }}</span>
                  <input
                    v-model.number="draft.histgbLearningRate"
                    type="number"
                    min="0.000001"
                    step="0.01"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.enableRegularization" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.histgbL2Regularization')
                  }}</span>
                  <input
                    v-model.number="draft.histgbL2Regularization"
                    type="number"
                    min="0"
                    step="0.01"
                    class="field-control"
                  />
                </label>
              </div>

              <div v-if="effectiveClassifier === 'xgb'" class="advanced-grid wizard-field-grid">
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.xgbEstimators')
                  }}</span>
                  <input
                    v-model.number="draft.xgbEstimators"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.xgbMaxDepth')
                  }}</span>
                  <input
                    v-model.number="draft.xgbMaxDepth"
                    type="number"
                    min="0"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.xgbLearningRate')
                  }}</span>
                  <input
                    v-model.number="draft.xgbLearningRate"
                    type="number"
                    min="0"
                    step="0.01"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.enableRegularization" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.xgbSubsample')
                  }}</span>
                  <input
                    v-model.number="draft.xgbSubsample"
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.enableRegularization" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.xgbColsample')
                  }}</span>
                  <input
                    v-model.number="draft.xgbColsample"
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.enableRegularization" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.xgbRegAlpha')
                  }}</span>
                  <input
                    v-model.number="draft.xgbRegAlpha"
                    type="number"
                    min="0"
                    step="0.01"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.enableRegularization" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.xgbRegLambda')
                  }}</span>
                  <input
                    v-model.number="draft.xgbRegLambda"
                    type="number"
                    min="0"
                    step="0.01"
                    class="field-control"
                  />
                </label>
              </div>

              <div v-if="effectiveClassifier === 'lgb'" class="advanced-grid wizard-field-grid">
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.lgbEstimators')
                  }}</span>
                  <input
                    v-model.number="draft.lgbEstimators"
                    type="number"
                    min="1"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.lgbMaxDepth')
                  }}</span>
                  <input
                    v-model.number="draft.lgbMaxDepth"
                    type="number"
                    min="0"
                    class="field-control"
                  />
                </label>
                <label class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.lgbLearningRate')
                  }}</span>
                  <input
                    v-model.number="draft.lgbLearningRate"
                    type="number"
                    min="0"
                    step="0.01"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.enableRegularization" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.lgbSubsample')
                  }}</span>
                  <input
                    v-model.number="draft.lgbSubsample"
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.enableRegularization" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.lgbColsample')
                  }}</span>
                  <input
                    v-model.number="draft.lgbColsample"
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.enableRegularization" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.lgbRegAlpha')
                  }}</span>
                  <input
                    v-model.number="draft.lgbRegAlpha"
                    type="number"
                    min="0"
                    step="0.01"
                    class="field-control"
                  />
                </label>
                <label v-if="draft.enableRegularization" class="grid gap-2 text-sm">
                  <span class="wizard-field-label">{{
                    t('wizard.advanced.fields.lgbRegLambda')
                  }}</span>
                  <input
                    v-model.number="draft.lgbRegLambda"
                    type="number"
                    min="0"
                    step="0.01"
                    class="field-control"
                  />
                </label>
              </div>
            </div>
          </div>
        </template>

        <template v-else>
          <aside
            v-if="pairTopicActive && activePairTopicIssues.length"
            class="pair-topic-notice pair-topic-notice-warning mt-5"
            aria-live="polite"
          >
            <AlertTriangle aria-hidden="true" />
            <div>
              <h3>{{ t('wizard.pairTopic.needsRepairTitle', { reduction: pairTopicName }) }}</h3>
              <p>{{ t('wizard.pairTopic.needsRepairDescription') }}</p>
            </div>
            <button type="button" class="secondary-action" @click="repairPairTopicRoute">
              {{ t('wizard.pairTopic.applyPreset') }}
            </button>
          </aside>

          <div class="section-card mt-5">
            <div class="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
              <label class="grid flex-1 gap-2 text-sm">
                <span class="wizard-field-label">{{ t('wizard.experimentName') }}</span>
                <input
                  v-model="experimentName"
                  type="text"
                  maxlength="120"
                  class="field-control"
                  :placeholder="defaultExperimentName"
                  @input="markExperimentNameEdited"
                />
              </label>
              <div
                class="rounded-lg border px-4 py-3 text-sm md:max-w-xs"
                style="background: var(--surface-muted); border-color: var(--line-soft)"
              >
                <div class="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">
                  {{ t('wizard.suggestedName') }}
                </div>
                <div class="mt-2 font-semibold">{{ defaultExperimentName }}</div>
              </div>
            </div>
            <p class="mt-3 text-sm text-slate-500">{{ t('wizard.experimentNameHint') }}</p>
          </div>

          <div class="mt-5 grid gap-4 md:grid-cols-2">
            <div class="section-card">
              <div class="flex items-center justify-between gap-3">
                <p class="eyebrow">{{ t('wizard.review.problem') }}</p>
                <button class="text-xs text-slate-500" @click="goToStep(1)">{{
                  t('wizard.edit')
                }}</button>
              </div>
              <div class="mt-4 font-medium">{{ taskLabel(draft.taskType) }}</div>
              <div class="mt-1 text-sm text-slate-500">
                {{ selectedDataset?.name ?? t('wizard.notSelected') }}
              </div>
            </div>
            <div class="section-card">
              <div class="flex items-center justify-between gap-3">
                <p class="eyebrow">{{ t('wizard.review.representation') }}</p>
                <button class="text-xs text-slate-500" @click="goToStep(2)">{{
                  t('wizard.edit')
                }}</button>
              </div>
              <div class="mt-4 font-medium">
                {{ t(`wizard.featureRoutes.${draft.featureRoute}`) }}
              </div>
              <div class="mt-1 text-sm text-slate-500">
                {{ draft.tokenizer }} · {{ draft.encoder }}
              </div>
            </div>
            <div class="section-card">
              <div class="flex items-center justify-between gap-3">
                <p class="eyebrow">{{ t('wizard.review.predictor') }}</p>
                <button class="text-xs text-slate-500" @click="goToStep(3)">{{
                  t('wizard.edit')
                }}</button>
              </div>
              <div class="mt-4 font-medium">{{ routeSummary }}</div>
            </div>
            <div class="section-card">
              <div class="flex items-center justify-between gap-3">
                <p class="eyebrow">{{ t('wizard.review.training') }}</p>
                <button class="text-xs text-slate-500" @click="goToStep(3)">{{
                  t('wizard.edit')
                }}</button>
              </div>
              <div class="mt-4 font-medium">
                <template v-if="usesEpochSchedule">
                  {{
                    t(draft.enableRegularization && draft.earlyStopPatience > 0 ? 'wizard.maxEpochs' : 'wizard.epochs')
                  }} {{ draft.epochs }} · {{ t('wizard.batchSize') }}
                  {{ draft.batchSize }}
                </template>
                <template v-else>
                  {{ t('wizard.singleFitSummary') }}
                </template>
              </div>
              <div v-if="advancedSummary.length" class="mt-3 flex flex-wrap gap-2">
                <span v-for="item in advancedSummary" :key="item" class="config-chip">{{
                  item
                }}</span>
              </div>
            </div>
          </div>

          <EffectiveRouteDiagram
            class="mt-5"
            variant="review"
            :state="isGuest ? 'draft' : 'resolved'"
            :route="preflight.data.value?.effective_route"
            :fallback="routeSummary"
            :updating="preflight.isFetching.value"
          />

          <div
            class="mt-5 rounded-lg border p-5"
            :class="preflight.data.value?.valid ? 'success-card' : 'subtle-card'"
          >
            <p class="eyebrow">{{ t('wizard.preflight') }}</p>
            <p v-if="isGuest" class="mt-4 text-sm text-slate-500">{{ t('guest.preflight') }}</p>
            <p v-else-if="!draft.datasetId" class="mt-4 text-sm text-slate-500">
              {{ t('wizard.chooseDatasetForChecks') }}
            </p>
            <p v-else-if="preflight.isPending.value" class="mt-4 text-sm text-slate-500">
              {{ t('wizard.checkingCompatibility') }}
            </p>
            <template v-else-if="preflight.data.value">
              <div class="mt-4 flex flex-wrap items-center gap-3">
                <span
                  class="status-pill"
                  :class="preflight.data.value.valid ? 'status-pill-success' : 'status-pill-danger'"
                >
                  {{
                    preflight.data.value.valid
                      ? t('wizard.readyToTrain')
                      : t('wizard.needsAttention')
                  }}
                </span>
                <span class="text-sm text-slate-600">{{
                  preflight.data.value.execution.script
                }}</span>
              </div>
              <ul
                v-if="preflight.data.value.errors.length"
                class="mt-4 space-y-2 text-sm text-rose-700"
              >
                <li v-for="message in localizedPreflightErrors" :key="message">• {{ message }}</li>
              </ul>
              <ul
                v-if="preflight.data.value.warnings.length"
                class="mt-4 space-y-2 text-sm text-amber-700"
              >
                <li v-for="message in localizedPreflightWarnings" :key="message"
                  >• {{ message }}</li
                >
              </ul>
              <div class="mt-4 text-sm text-slate-600">
                {{ t('wizard.reusableInference') }}:
                <span class="font-medium">
                  {{
                    !preflight.data.value.valid
                      ? t('wizard.pendingValidation')
                      : preflight.data.value.execution.supports_reusable_inference
                        ? t('wizard.available')
                        : t('wizard.notYetAvailable')
                  }}
                </span>
              </div>
            </template>
            <p v-else-if="preflight.error.value" class="mt-4 text-sm text-rose-700">
              {{ localizedApiError(preflight.error.value, t, te) }}
            </p>
          </div>
          <p v-if="submissionError" class="error-card mt-4" role="alert">{{ submissionError }}</p>
        </template>

        <div class="mt-6 flex flex-wrap items-center justify-between gap-3">
          <button v-if="currentStep > 1" class="tertiary-action" @click="previousStep">
            <ArrowLeft aria-hidden="true" />
            {{ t('wizard.back') }}
          </button>
          <div v-else />
          <button
            v-if="currentStep < wizardSteps.length"
            class="primary-action"
            :disabled="!canAdvance"
            @click="nextStep"
          >
            {{
              currentStep === wizardSteps.length - 1
                ? t('wizard.reviewAndConfirm')
                : t('wizard.next')
            }}
            <ArrowRight aria-hidden="true" />
          </button>
          <button
            v-else
            class="primary-action"
            :disabled="!isGuest && !canSubmitExperiment"
            @click="submit"
          >
            <Play aria-hidden="true" />
            {{
              createExperiment.isPending.value
                ? t('wizard.startingTraining')
                : t('wizard.startTraining')
            }}
          </button>
        </div>
      </div>
    </div>

    <RouteConflictDialog
      :open="Boolean(pendingTopicConflict)"
      :title="t('wizard.pairTopic.conflictTitle', { reduction: pairTopicName })"
      :description="
        t('wizard.pairTopic.conflictDescription', {
          reduction: pairTopicName,
          field: pendingTopicConflict
            ? t(`wizard.pairTopic.fields.${pendingTopicConflict.field}`)
            : '',
        })
      "
      :confirm-label="t('wizard.pairTopic.confirmChange')"
      :cancel-label="t('common.cancel')"
      :close-label="t('common.close')"
      @confirm="confirmTopicConflict"
      @cancel="cancelTopicConflict"
    />
  </section>
</template>
