<script setup lang="ts">
import { normalizeTaskType } from '../taskTerminology'
import SelectionSummary from '../components/SelectionSummary.vue'
import ExperimentVersions from '../components/ExperimentVersions.vue'
import EnsembleSummary from '../components/EnsembleSummary.vue'
import { Cpu, RefreshCw, Scale, SquareStop } from '@lucide/vue'
import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { api } from '../api/client'
import EffectiveRouteDiagram from '../components/EffectiveRouteDiagram.vue'
import ExperimentRunStages, {
  type ExperimentRunStageItem,
} from '../components/ExperimentRunStages.vue'
import { localizedApiError } from '../errorLocalization'
import PageHeader from '../components/PageHeader.vue'
import CurveAnalysisChart from '../components/CurveAnalysisChart.vue'
import FeatureImportanceChart from '../components/FeatureImportanceChart.vue'
import InferencePanel from '../components/InferencePanel.vue'
import LiveExperimentLog from '../components/LiveExperimentLog.vue'
import MetricChart from '../components/MetricChart.vue'
import PerformanceSummary from '../components/PerformanceSummary.vue'
import Projection3DChart from '../components/Projection3DChart.vue'
import ProjectionChart from '../components/ProjectionChart.vue'
import ScoreDiagnosticsChart from '../components/ScoreDiagnosticsChart.vue'
import SplitMetricChart from '../components/SplitMetricChart.vue'
import SvmResumeDialog from '../components/SvmResumeDialog.vue'
import TuningChart from '../components/TuningChart.vue'
import TrainingFailureCard from '../components/TrainingFailureCard.vue'
import {
  classifierDisplayName,
  configText,
  effectiveClassifier,
  isPureClassicalClassifierRoute,
} from '../experimentRoutes'
import { deriveExperimentRunProgress, resolveExperimentProgressTotal } from '../agentWorkspace'
import { modelFeatureSignature } from '../modelDisplay'
import { experimentDisplayName } from '../localization'
import { useExperimentDraftStore } from '../stores/experimentDraft'
import type { FeatureImportancePayload, MetricEvent } from '../types'

const props = defineProps<{ id: string }>()
const router = useRouter()
const draft = useExperimentDraftStore()
const { t, te } = useI18n()
const queryClient = useQueryClient()
const experiment = useQuery({
  queryKey: ['experiment', props.id],
  queryFn: () => api.experiment(props.id),
  refetchInterval: 2000,
})
const datasets = useQuery({ queryKey: ['datasets'], queryFn: api.datasets })
const logs = useQuery({
  queryKey: ['experiment-logs', props.id],
  queryFn: () => api.experimentLogs(props.id),
  refetchInterval: 2000,
})
const analysis = useQuery({
  queryKey: ['experiment-analysis', props.id],
  queryFn: () => api.experimentAnalysis(props.id),
  refetchInterval: 2000,
})
const events = computed(() => experiment.data.value?.metrics.events ?? [])
const epochEvents = computed<MetricEvent[]>(() =>
  events.value.filter((event) => event.kind === 'epoch'),
)
const svmProgressEvents = computed<MetricEvent[]>(() =>
  events.value.filter((event) => event.kind === 'stage' && event.stage === 'svm_fit'),
)
const latestSvmProgress = computed(() => svmProgressEvents.value.at(-1))
const svmRecovered = computed(() =>
  svmProgressEvents.value.some((event) => event.status === 'complete' && event.recovered === true),
)
const summary = computed(() => experiment.data.value?.metrics.summary ?? {})
const currentConfig = computed(() => experiment.data.value?.config ?? {})
const isWebEnsemble = computed(() => Boolean(currentConfig.value.web_ensemble))
const runProgress = computed(() => deriveExperimentRunProgress(experiment.data.value))
const featureSignature = computed(() => modelFeatureSignature(currentConfig.value))
const hasStreamConfig = computed(
  () => currentConfig.value.streams && typeof currentConfig.value.streams === 'object',
)
const isPureMlRoute = computed(() => isPureClassicalClassifierRoute(currentConfig.value))
const isEmbeddingClassifierRoute = computed(
  () => configText(currentConfig.value, 'training_mode') === 'export_embeddings',
)
const usesEpochSchedule = computed(() => runProgress.value.usesEpochs)
const graphRouteModels = new Set(['gcn', 'gat', 'hgt', 'sage', 'graph_transformer'])
const profileEncoders = new Set([
  'pssm',
  'pssm_blosum62_hybrid',
  'psfm',
  'psfm_blosum62_hybrid',
  'ac_pssm',
  'cc_pssm',
  'acc_pssm',
  'ac_pssm_blosum62_hybrid',
  'cc_pssm_blosum62_hybrid',
  'acc_pssm_blosum62_hybrid',
])
const profileTokenizers = new Set(['top-n-gram', 'dt'])
const profileFeatureNames: Record<string, string> = {
  pssm: 'PSSM',
  pssm_blosum62_hybrid: 'PSSM–BLOSUM62 Hybrid',
  ac_pssm_blosum62_hybrid: 'AC-PSSM–BLOSUM62 Hybrid',
  cc_pssm_blosum62_hybrid: 'CC-PSSM–BLOSUM62 Hybrid',
  acc_pssm_blosum62_hybrid: 'ACC-PSSM–BLOSUM62 Hybrid',
  psfm: 'PSFM',
  psfm_blosum62_hybrid: 'PSFM–BLOSUM62 Hybrid',
  target_frequency: 'Target Frequency',
}
const isGraphRoute = computed(
  () =>
    graphRouteModels.has(configText(currentConfig.value, 'model')) ||
    graphRouteModels.has(configText(currentConfig.value, 'graph_predictor')) ||
    configText(currentConfig.value, 'predictor_type') === 'sample_graph',
)
const hasFinalMetrics = computed(() =>
  isWebEnsemble.value
    ? ['val_f1', 'val_auc', 'test_f1'].some((key) => typeof summary.value[key] === 'number')
    : isPureMlRoute.value
      ? Boolean(summary.value.val_f1 ?? summary.value.test_f1)
      : Boolean(summary.value.best_val_f1 ?? summary.value.best_val_auc ?? summary.value.test_f1),
)
const modelId = computed(() => experiment.data.value?.model_id)
const model = useQuery({
  queryKey: ['model-for-experiment', modelId],
  queryFn: () => api.model(modelId.value as string),
  enabled: computed(() => Boolean(modelId.value)),
})
const displayName = computed(() => {
  const record = experiment.data.value
  return record ? experimentDisplayName(record, t) : ''
})
const resolvedMlConfig = computed(() => {
  const metadataArgs = model.data.value?.metadata.args
  const finalArgs =
    metadataArgs && typeof metadataArgs === 'object' && !Array.isArray(metadataArgs)
      ? (metadataArgs as Record<string, unknown>)
      : {}
  return { ...currentConfig.value, ...finalArgs }
})
const tuningMetadata = computed(() => {
  const tuning = model.data.value?.metadata.tuning
  return tuning && typeof tuning === 'object' && !Array.isArray(tuning)
    ? (tuning as Record<string, unknown>)
    : {}
})
const tuningCandidates = computed(() =>
  Array.isArray(tuningMetadata.value.candidates) ? tuningMetadata.value.candidates : [],
)
const hasTuningResults = computed(() => tuningCandidates.value.length > 0)
const analysisData = computed(() => analysis.data.value)
const hasCurveAnalysis = computed(() =>
  ['validation', 'test'].some((split) =>
    Boolean(
      analysisData.value?.curves?.[split]?.roc?.available ||
      analysisData.value?.curves?.[split]?.pr?.available,
    ),
  ),
)
const hasMicroEvaluationCurves = computed(() =>
  ['validation', 'test'].some((split) => analysisData.value?.curves?.[split]?.average === 'micro'),
)
const ensembleValidationOnlyCurves = computed(
  () =>
    isWebEnsemble.value &&
    !analysisData.value?.curves?.test?.roc?.available &&
    !analysisData.value?.curves?.test?.pr?.available,
)
const hasScoreDiagnostics = computed(() =>
  ['validation', 'test'].some((split) => {
    const curves = analysisData.value?.curves?.[split]
    return Boolean(
      curves?.confusion_matrix?.available ||
      curves?.threshold_sweep?.available ||
      curves?.score_distribution?.available,
    )
  }),
)
const hasCurveAnalysisArtifact = computed(
  () => Object.keys(analysisData.value?.curves ?? {}).length > 0,
)
const globalImportance = computed(() => analysisData.value?.feature_importance)
const validationSplitImportance = computed(
  () =>
    analysisData.value?.split_importance?.validation ??
    analysisData.value?.shap_importance?.validation,
)
const testSplitImportance = computed(
  () => analysisData.value?.split_importance?.test ?? analysisData.value?.shap_importance?.test,
)
const legacySplitImportanceMethod = computed(() => {
  const classifier = effectiveClassifier(resolvedMlConfig.value)
  return classifier === 'xgb' || classifier === 'lgb' ? 'shap_mean_abs' : undefined
})
function hasImportancePayload(payload?: FeatureImportancePayload) {
  return Boolean(payload?.available && payload.items?.length)
}
const hasImportanceCharts = computed(() =>
  [globalImportance.value, validationSplitImportance.value, testSplitImportance.value].some(
    (payload) => hasImportancePayload(payload),
  ),
)
function inferredImportanceMethod(payload?: FeatureImportancePayload, defaultMethod?: string) {
  const method = String(payload?.method ?? '').trim()
  const methodLabel = String(payload?.method_label ?? '')
    .trim()
    .toLowerCase()
  if (method) return method
  if (methodLabel.startsWith('shap')) return 'shap_mean_abs'
  if (methodLabel === 'permutation importance') return 'permutation'
  if (payload?.metric || Number.isFinite(Number(payload?.baseline_score))) return 'permutation'
  return defaultMethod ?? ''
}
function importanceMethodLabel(payload?: FeatureImportancePayload, defaultMethod?: string) {
  const method = inferredImportanceMethod(payload, defaultMethod)
  if (method === 'built_in') return t('detail.importanceMethodBuiltIn')
  if (method === 'shap_mean_abs') return t('detail.importanceMethodShapMeanAbs')
  if (method === 'permutation') return t('detail.importanceMethodPermutation')
  const methodLabel = String(payload?.method_label ?? '').trim()
  return methodLabel && !['value', 'importance'].includes(methodLabel.toLowerCase())
    ? methodLabel
    : t('detail.importanceValue')
}
const hasRepresentationCharts = computed(() =>
  Boolean(
    analysisData.value?.projection_2d?.available ||
    analysisData.value?.projection_3d?.available ||
    hasImportanceCharts.value,
  ),
)
const clusterSummary = computed(() => {
  const clusters = analysisData.value?.clusters
  if (!clusters?.available) return ''
  const method =
    String(clusters.method ?? '').toLowerCase() === 'kmeans'
      ? 'K-means'
      : String(clusters.method ?? '')
  const count = Number(clusters.n_clusters)
  if (method && Number.isFinite(count)) return `${method} · ${t('detail.clusterCount', { count })}`
  return method
})
const tuningSelectionMessage = computed(() =>
  tuningMetadata.value.selected_matches_baseline
    ? t('detail.tuningDefaultSelected')
    : t('detail.tuningBestChanged'),
)
const dataset = computed(() =>
  datasets.data.value?.find((item) => item.id === experiment.data.value?.dataset_id),
)
const isActive = computed(() => ['queued', 'running'].includes(experiment.data.value?.status ?? ''))
const isTerminal = computed(() =>
  ['succeeded', 'failed', 'canceled'].includes(experiment.data.value?.status ?? ''),
)
const cancelMutation = useMutation({
  mutationFn: () => api.cancelExperiment(props.id),
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ['experiments'] })
    queryClient.invalidateQueries({ queryKey: ['experiment', props.id] })
    queryClient.invalidateQueries({ queryKey: ['experiment-logs', props.id] })
  },
})
const resumeDialogOpen = ref(false)
const resumeMutation = useMutation({
  mutationFn: (overrides: Record<string, unknown>) => api.resumeDownstreamSvm(props.id, overrides),
  onSuccess: (record) => {
    resumeDialogOpen.value = false
    queryClient.invalidateQueries({ queryKey: ['experiments'] })
    router.push(`/experiments/${record.id}`)
  },
})
const canResumeSvm = computed(() =>
  Boolean(experiment.data.value?.resume_capabilities?.downstream_svm?.available),
)
const resumeError = computed(() =>
  resumeMutation.error.value ? localizedApiError(resumeMutation.error.value, t, te) : '',
)
const latestEpoch = computed(() => epochEvents.value.at(-1))
const completedEpoch = computed(
  () =>
    Number(latestEpoch.value?.epoch) ||
    (runProgress.value.runStage ? 0 : (runProgress.value.current ?? 0)),
)
const scheduledEpochTotal = computed(() =>
  resolveExperimentProgressTotal(runProgress.value, currentConfig.value, latestEpoch.value?.total),
)
const configuredEarlyStopPatience = computed(
  () => Number(currentConfig.value.early_stop_patience) || 0,
)
const bestEpoch = computed(() => Number(summary.value.best_epoch) || 0)
const earlyStoppedSucceeded = computed(() => runProgress.value.earlyStopped)
const tuningActive = computed(() => runProgress.value.preparation === 'hyperparameter_tuning')
const profilePreparationActive = computed(() => {
  if (epochEvents.value.length) return false
  if (runProgress.value.preparation) return !tuningActive.value
  if (experiment.data.value?.status !== 'running') return false
  const tokenizer = configText(currentConfig.value, 'tokenizer')
  const encoder = configText(currentConfig.value, 'encoder')
  return profileTokenizers.has(tokenizer) || profileEncoders.has(encoder)
})
const progressIndeterminate = computed(
  () => experiment.data.value?.status === 'running' && runProgress.value.mode === 'indeterminate',
)
const progressPercent = computed(() => {
  if (experiment.data.value?.status === 'succeeded') return 100
  return runProgress.value.percent ?? 0
})
const profileFeatureName = computed(() => {
  const feature = runProgress.value.preparationFeature
  if (!feature) return 'PSSM/PSFM'
  return feature
    .split('+')
    .map((item) => profileFeatureNames[item] ?? item)
    .join(' + ')
})
function durationLabel(value?: number) {
  if (value === undefined || !Number.isFinite(value)) return ''
  const totalSeconds = Math.max(0, Math.round(value))
  if (totalSeconds < 60) return t('detail.durationSeconds', { count: totalSeconds })
  const totalMinutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (totalMinutes < 60) return t('detail.durationMinutes', { minutes: totalMinutes, seconds })
  const hours = Math.floor(totalMinutes / 60)
  const minutes = totalMinutes % 60
  return t('detail.durationHours', { hours, minutes })
}
const profileElapsedLabel = computed(() => {
  const elapsed = durationLabel(runProgress.value.preparationElapsedSeconds)
  return elapsed ? t('detail.profileElapsed', { time: elapsed }) : ''
})
const profileRateLabel = computed(() => {
  const rate = runProgress.value.preparationRatePerSecond
  if (rate === undefined) return ''
  const formatted = rate.toLocaleString(undefined, {
    maximumFractionDigits: rate < 1 ? 2 : 1,
  })
  return t('detail.profileRate', { rate: formatted })
})
const profileEtaLabel = computed(() => {
  const eta = durationLabel(runProgress.value.preparationEtaSeconds)
  return eta ? t('detail.profileEta', { time: eta }) : ''
})
const profileProgressStats = computed(() =>
  [profileElapsedLabel.value, profileRateLabel.value, profileEtaLabel.value].filter(Boolean),
)
const structuredProgressStats = computed(() => {
  if (!runProgress.value.runStage) return []
  const elapsed = durationLabel(runProgress.value.preparationElapsedSeconds)
  const eta = durationLabel(runProgress.value.preparationEtaSeconds)
  return [
    elapsed ? t('detail.runElapsed', { time: elapsed }) : '',
    eta ? t('detail.runEta', { time: eta }) : '',
    runProgress.value.runStageThreads
      ? t('detail.runThreads', { count: runProgress.value.runStageThreads })
      : '',
  ].filter(Boolean)
})
const runStageItems = computed<ExperimentRunStageItem[]>(() => {
  const keys = ['representation', 'embedding_export', 'classifier_fit', 'finalization'] as const
  const terminalStatus = experiment.data.value?.status
  let activeIndex = runProgress.value.runStage
    ? keys.indexOf(runProgress.value.runStage)
    : ['running', 'failed', 'canceled'].includes(terminalStatus ?? '')
      ? 0
      : -1
  if (runProgress.value.runStageStatus === 'complete' && activeIndex < keys.length - 1) {
    activeIndex += 1
  }
  return keys.map((key, index) => {
    let state: ExperimentRunStageItem['state'] = 'pending'
    if (terminalStatus === 'succeeded' || index < activeIndex) state = 'complete'
    else if (index === activeIndex) {
      state = ['failed', 'canceled'].includes(terminalStatus ?? '') ? 'stopped' : 'active'
      if (
        runProgress.value.runStage === 'finalization' &&
        runProgress.value.runStageStatus === 'complete'
      ) {
        state = 'complete'
      }
    }
    return {
      key,
      label: t(`detail.runStages.${key}`),
      state,
    }
  })
})
const progressLabel = computed(() => {
  if (runProgress.value.runStage) {
    const params = {
      current: runProgress.value.current ?? 0,
      total: runProgress.value.total ?? 1,
      split: runProgress.value.runStageSplit || '-',
      output: runProgress.value.runStageOutput ?? 0,
      outputs: runProgress.value.runStageOutputs ?? 1,
      iteration: runProgress.value.runStageIteration ?? 0,
      iterations: runProgress.value.runStageIterations ?? 1,
    }
    if (runProgress.value.runStage === 'representation') {
      return t('detail.runRepresentationProgress', params)
    }
    if (runProgress.value.runStage === 'embedding_export') {
      return t('detail.runEmbeddingProgress', params)
    }
    if (runProgress.value.runStage === 'classifier_fit') {
      return t('detail.runClassifierProgress', params)
    }
    return t('detail.runFinalizationProgress', params)
  }
  const svm = latestSvmProgress.value
  if (svm && experiment.data.value?.status === 'running') {
    const params = {
      current: svm.current ?? 1,
      total: svm.total ?? 1,
      attempt: svm.attempt ?? 1,
      attempts: svm.attempts ?? 1,
      maxIter: svm.max_iter ?? 0,
    }
    return svm.status === 'retrying'
      ? t('detail.svmFitRetrying', params)
      : t('detail.svmFitRunning', params)
  }
  if (tuningActive.value) {
    const classifier = classifierDisplayName(runProgress.value.preparationFeature ?? '')
    const status = experiment.data.value?.status
    if (status === 'canceled' || status === 'failed') {
      return t('detail.tuningStoppedAt', {
        classifier,
        current: completedEpoch.value,
        total: scheduledEpochTotal.value,
      })
    }
    if (runProgress.value.preparationStatus === 'complete') {
      return t('detail.tuningComplete', { classifier })
    }
    return t('detail.tuningProgress', {
      classifier,
      current: completedEpoch.value,
      total: scheduledEpochTotal.value,
    })
  }
  if (profilePreparationActive.value) {
    const status = experiment.data.value?.status
    if (status === 'canceled' || status === 'failed') {
      return scheduledEpochTotal.value
        ? t('detail.profileStoppedAt', {
            feature: profileFeatureName.value,
            current: completedEpoch.value,
            total: scheduledEpochTotal.value,
          })
        : t('detail.profileStopped', { feature: profileFeatureName.value })
    }
    if (runProgress.value.preparation === 'profile_database') {
      return t('detail.profileDatabaseBuilding', { feature: profileFeatureName.value })
    }
    if (runProgress.value.preparationStatus === 'complete') {
      return t('detail.profileReadyStartingTraining', { feature: profileFeatureName.value })
    }
    return runProgress.value.preparation === 'profile_generation' && scheduledEpochTotal.value
      ? t('detail.profileProgress', {
          feature: profileFeatureName.value,
          current: completedEpoch.value,
          total: scheduledEpochTotal.value,
        })
      : t('detail.profilePreparing', { feature: profileFeatureName.value })
  }
  if (usesEpochSchedule.value) {
    if (earlyStoppedSucceeded.value) {
      const params = {
        best: bestEpoch.value,
        current: completedEpoch.value,
        total: scheduledEpochTotal.value,
      }
      return bestEpoch.value
        ? t('detail.earlyStoppedProgressWithBest', params)
        : t('detail.earlyStoppedProgress', params)
    }
    if (experiment.data.value?.status === 'canceled') {
      return completedEpoch.value && scheduledEpochTotal.value
        ? t('detail.stoppedAtEpoch', {
            current: completedEpoch.value,
            total: scheduledEpochTotal.value,
          })
        : t('detail.stoppedNoEpochs')
    }
    if (!completedEpoch.value && isTerminal.value) return t('detail.noParsedEpochMetricsShort')
    return completedEpoch.value && scheduledEpochTotal.value
      ? t('detail.epochProgress', {
          current: completedEpoch.value,
          total: scheduledEpochTotal.value,
        })
      : t('detail.noEpochsYet')
  }
  const status = experiment.data.value?.status
  if (status === 'queued') return t('detail.mlFitQueued')
  if (status === 'running') return t('detail.mlFitRunning')
  if (status === 'succeeded') return t('detail.mlFitComplete')
  return t('detail.mlFitStopped')
})
const trainingMessage = computed(() => {
  const status = experiment.data.value?.status
  if (status === 'running' && runProgress.value.runStage) {
    return t(`detail.runStageMessages.${runProgress.value.runStage}`)
  }
  if (status === 'running' && tuningActive.value) {
    return t('detail.tuningRunning', {
      classifier: classifierDisplayName(runProgress.value.preparationFeature ?? ''),
    })
  }
  if (status === 'running' && profilePreparationActive.value) {
    return runProgress.value.preparation === 'profile_database'
      ? t('detail.profileDatabaseBuilding', { feature: profileFeatureName.value })
      : t('detail.profilePreparing', { feature: profileFeatureName.value })
  }
  if (isPureMlRoute.value) {
    if (status === 'queued') return t('detail.mlQueued')
    if (status === 'running') return t('detail.mlRunning')
    if (status === 'succeeded') return t('detail.mlSucceeded')
    if (status === 'canceled') return t('detail.trainingCanceled')
    return t('detail.trainingFailed')
  }
  if (status === 'queued') return t('detail.trainingQueued')
  if (status === 'running') return t('detail.trainingRunning')
  if (status === 'succeeded' && earlyStoppedSucceeded.value)
    return t('detail.trainingEarlyStoppedSucceeded')
  if (status === 'succeeded') return t('detail.trainingSucceeded')
  if (status === 'canceled') return t('detail.trainingCanceled')
  return t('detail.trainingFailed')
})
const modelReadinessMessage = computed(() => {
  if (experiment.data.value?.status === 'canceled' && !model.data.value)
    return t('detail.modelCanceled')
  if (!model.data.value) return t('detail.modelPending')
  if (model.data.value.supports_inference) return t('detail.inferenceReady')
  return isGraphRoute.value
    ? t('detail.graphInferenceUnavailable')
    : t('detail.inferenceUnavailable')
})
const mlClassifier = computed(() => classifierDisplayName(effectiveClassifier(currentConfig.value)))
const mlPipeline = computed(() => {
  const config = currentConfig.value
  const steps = [
    String(config.tokenizer ?? '-'),
    String(config.encoder ?? '-'),
    t('detail.pipelineFeatureMatrix'),
  ]
  const scaler = String(config.ml_scaler ?? '')
  const reduction = String(config.ml_reduction ?? '')
  if (scaler && scaler !== 'none') steps.push(`${t('detail.pipelineScaler')}: ${scaler}`)
  if (reduction && reduction !== 'none')
    steps.push(`${t('detail.pipelineReduction')}: ${reduction}`)
  steps.push(mlClassifier.value)
  return steps
})
const mlHyperparameters = computed(() => {
  const config = resolvedMlConfig.value
  const classifier = effectiveClassifier(config)
  const items: string[] = []
  if (classifier === 'rf') {
    items.push(`estimators=${config.rf_estimators ?? '-'}`)
    items.push(`max_depth=${config.rf_max_depth ?? '-'}`)
    items.push(`min_leaf=${config.rf_min_samples_leaf ?? '-'}`)
  } else if (classifier === 'svm') {
    items.push(`svm_calibration=${config.svm_calibration ?? 'none'}`)
    items.push(`C=${config.svm_c ?? '-'}`)
    items.push(`kernel=${config.svm_kernel ?? 'linear'}`)
    if (config.svm_kernel === 'rbf') items.push(`gamma=${config.svm_gamma ?? '-'}`)
    items.push(`class_weight=${config.svm_class_weight ?? '-'}`)
    items.push(`max_iter=${config.svm_max_iter ?? '-'}`)
    items.push(`tol=${config.svm_tol ?? '-'}`)
    items.push(`convergence_policy=${config.svm_convergence_policy ?? 'strict'}`)
  } else if (classifier === 'logreg') {
    items.push(`C=${config.logreg_c ?? '-'}`)
    items.push(`class_weight=${config.logreg_class_weight ?? '-'}`)
    items.push(`max_iter=${config.logreg_max_iter ?? '-'}`)
    items.push(`convergence_policy=${config.logreg_convergence_policy ?? 'strict'}`)
  } else if (classifier === 'xgb') {
    items.push(`estimators=${config.xgb_estimators ?? '-'}`)
    items.push(`max_depth=${config.xgb_max_depth ?? '-'}`)
    items.push(`lr=${config.xgb_lr ?? '-'}`)
  } else if (classifier === 'lgb') {
    items.push(`estimators=${config.lgb_estimators ?? '-'}`)
    items.push(`max_depth=${config.lgb_max_depth ?? '-'}`)
    items.push(`lr=${config.lgb_lr ?? '-'}`)
    items.push(`threads=${config.lgb_n_jobs ?? 4}`)
    items.push(`deterministic=${config.lgb_deterministic ?? true}`)
    items.push(`histogram=${config.lgb_histogram_mode ?? 'row_wise'}`)
  }
  const scaler = String(config.ml_scaler ?? '')
  const reduction = String(config.ml_reduction ?? '')
  if (scaler) items.push(`ml_scaler=${scaler}`)
  if (reduction) items.push(`ml_reduction=${reduction}`)
  if (config.ml_auto_tune || tuningMetadata.value.best_params) {
    items.push(`auto_tune=${config.ml_tuning_objective ?? tuningMetadata.value.objective ?? 'f1'}`)
  }
  return items
})
const mlStages = computed(() => {
  const status = experiment.data.value?.status
  const finished = status === 'succeeded' || status === 'failed' || status === 'canceled'
  return [
    {
      label: t('detail.mlStageFeatures'),
      state: status === 'queued' ? 'pending' : 'done',
    },
    {
      label: t('detail.mlStageFit'),
      state:
        status === 'running'
          ? 'active'
          : status === 'succeeded'
            ? 'done'
            : finished
              ? 'active'
              : 'pending',
    },
    {
      label: t('detail.mlStageEvaluate'),
      state: status === 'succeeded' ? 'done' : status === 'failed' ? 'active' : 'pending',
    },
  ]
})
const metricsEmptyMessage = computed(() => {
  if (isPureMlRoute.value) return t('detail.noMlMetricsYet')
  if (isTerminal.value && usesEpochSchedule.value) return t('detail.noParsedEpochMetrics')
  return t('detail.noMetricsYet')
})
const curveAnalysisEmptyMessage = computed(() => {
  if (hasCurveAnalysisArtifact.value) return t('detail.noCurveAnalysisAvailable')
  return isTerminal.value ? t('detail.noCurveAnalysisArtifact') : t('detail.noCurveAnalysisYet')
})
const representationEmptyMessage = computed(() =>
  isGraphRoute.value ? t('detail.graphInterpretabilityUnavailable') : t('detail.noAnalysisYet'),
)

function cloneAndRerun() {
  if (!dataset.value || !experiment.data.value) return
  if (isWebEnsemble.value) {
    router.push({ path: '/ensembles/new', query: { clone: props.id } })
    return
  }
  draft.cloneFromExperiment(experiment.data.value)
  router.push('/experiments/new')
}

function viewFailureLogs() {
  const panel = document.getElementById('experiment-logs')
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  panel?.scrollIntoView({
    behavior: reduceMotion ? 'auto' : 'smooth',
    block: 'start',
  })
  const log = panel?.querySelector<HTMLElement>('pre[tabindex]')
  window.setTimeout(() => log?.focus({ preventScroll: true }), reduceMotion ? 0 : 350)
}

function openSvmResume() {
  resumeMutation.reset()
  resumeDialogOpen.value = true
}

function confirmStop() {
  if (!window.confirm(t('detail.stopConfirm', { name: displayName.value }))) return
  cancelMutation.mutate()
}

function statusLabel(status: string) {
  return t(`status.${status}`)
}

function statusClass(status: string) {
  if (status === 'succeeded') return 'status-pill-success'
  if (status === 'failed') return 'status-pill-danger'
  if (status === 'queued' || status === 'running') return 'status-pill-warning'
  return 'status-pill-neutral'
}

function taskLabel(taskType: string) {
  taskType = normalizeTaskType(taskType)
  return t(`tasks.${taskType}.label`)
}
</script>

<template>
  <section
    v-if="experiment.data.value"
    id="experiment-overview"
    class="experiment-detail grid gap-6"
  >
    <PageHeader :eyebrow="t('detail.eyebrow')" :title="displayName" compact>
      <template #actions>
        <span class="status-pill" :class="statusClass(experiment.data.value.status)">
          {{ statusLabel(experiment.data.value.status) }}
        </span>
      </template>
    </PageHeader>
    <EnsembleSummary :metadata="model.data.value?.metadata" :summary="summary" />
    <SelectionSummary :metadata="model.data.value?.metadata" />
    <ExperimentVersions :experiment-id="props.id" />
    <div class="panel">
      <p v-if="dataset" class="text-sm text-slate-500">
        {{ t('detail.datasetLabel') }}: {{ dataset.name }}
      </p>
      <div class="mt-4 grid gap-4 md:grid-cols-4">
        <div class="metric-chip">
          <div class="text-xs text-slate-500">{{ t('detail.task') }}</div>
          <div class="mt-1 font-medium">{{ taskLabel(experiment.data.value.task_type) }}</div>
        </div>
        <div class="metric-chip">
          <div class="text-xs text-slate-500">
            {{ hasStreamConfig ? t('wizard.representation') : t('detail.tokenizer') }}
          </div>
          <div class="mt-1 font-medium">{{ featureSignature }}</div>
        </div>
        <div class="metric-chip">
          <div class="text-xs text-slate-500">
            {{ hasStreamConfig ? t('detail.streamMode') : t('detail.encoder') }}
          </div>
          <div class="mt-1 font-medium">
            {{
              hasStreamConfig
                ? experiment.data.value.config.stream_mode
                : experiment.data.value.config.encoder
            }}
          </div>
        </div>
        <div class="metric-chip">
          <div class="text-xs text-slate-500">{{ t('detail.model') }}</div>
          <div class="mt-1 font-medium">{{ experiment.data.value.config.model }}</div>
        </div>
      </div>
    </div>

    <div class="grid gap-6 lg:grid-cols-[1fr_1fr]">
      <div class="panel">
        <p class="eyebrow">{{ t('detail.runStatus') }}</p>
        <p class="mt-4 text-sm leading-6 text-slate-600">{{ trainingMessage }}</p>
        <div class="section-card mt-5">
          <ExperimentRunStages
            v-if="isEmbeddingClassifierRoute"
            class="mb-4"
            :items="runStageItems"
            :label="t('detail.runStagesAria')"
          />
          <div class="flex flex-wrap items-center justify-between gap-3 text-sm" aria-live="polite">
            <span class="font-medium">{{ t('detail.progress') }}</span>
            <span class="text-right text-slate-500">{{ progressLabel }}</span>
          </div>
          <div
            class="progress-track mt-3"
            role="progressbar"
            :aria-label="t('detail.progress')"
            aria-valuemin="0"
            aria-valuemax="100"
            :aria-valuenow="progressIndeterminate ? undefined : progressPercent"
            :aria-valuetext="progressLabel"
          >
            <div
              class="progress-fill"
              :class="{
                'progress-fill-active': isActive && !progressIndeterminate,
                'progress-fill-indeterminate': progressIndeterminate,
              }"
              :style="{ width: progressIndeterminate ? '38%' : `${progressPercent}%` }"
            />
          </div>
          <div
            v-if="
              structuredProgressStats.length ||
              (profilePreparationActive && profileProgressStats.length)
            "
            class="mt-3 flex flex-wrap gap-2"
            aria-live="polite"
          >
            <span
              v-for="item in structuredProgressStats.length
                ? structuredProgressStats
                : profileProgressStats"
              :key="item"
              class="config-chip"
            >
              {{ item }}
            </span>
          </div>
          <details
            v-if="earlyStoppedSucceeded"
            class="mt-4 rounded-md border px-3 py-3 text-sm"
            :style="{
              background: 'var(--color-warning-bg)',
              borderColor: 'var(--color-warning-border)',
              color: 'var(--color-warning-text)',
            }"
          >
            <summary class="cursor-pointer font-semibold">
              {{ t('detail.earlyStopHelpTitle') }}
            </summary>
            <p class="mt-2 leading-6">
              {{
                t('detail.earlyStopHelpBody', {
                  current: completedEpoch,
                  patience: configuredEarlyStopPatience,
                  total: scheduledEpochTotal,
                })
              }}
            </p>
          </details>
          <div
            v-if="svmRecovered"
            class="mt-4 rounded-md border px-3 py-3 text-sm leading-6"
            :style="{
              background: 'var(--color-success-bg)',
              borderColor: 'var(--color-success-border)',
              color: 'var(--color-success-text)',
            }"
            role="status"
          >
            <strong>{{ t('detail.svmRecoveredTitle') }}</strong>
            {{ t('detail.svmRecoveredBody') }}
          </div>
          <div v-if="isPureMlRoute" class="mt-4 grid gap-3 sm:grid-cols-3">
            <div
              v-for="stage in mlStages"
              :key="stage.label"
              class="rounded-md border px-3 py-2 text-xs font-semibold"
              :style="{
                background:
                  stage.state === 'done'
                    ? 'var(--color-success-bg)'
                    : stage.state === 'active'
                      ? 'var(--color-warning-bg)'
                      : 'var(--surface-muted)',
                borderColor:
                  stage.state === 'done'
                    ? 'var(--color-success-border)'
                    : stage.state === 'active'
                      ? 'var(--color-warning-border)'
                      : 'var(--line-soft)',
                color:
                  stage.state === 'done'
                    ? 'var(--color-success-text)'
                    : stage.state === 'active'
                      ? 'var(--color-warning-text)'
                      : 'var(--color-muted)',
              }"
            >
              {{ stage.label }}
            </div>
          </div>
        </div>
        <TrainingFailureCard
          v-if="experiment.data.value.status === 'failed'"
          :failure="experiment.data.value.failure"
          :legacy-error="experiment.data.value.error"
          :can-rerun="Boolean(dataset)"
          :can-resume-svm="canResumeSvm"
          @view-logs="viewFailureLogs"
          @rerun="cloneAndRerun"
          @resume-svm="openSvmResume"
        />
        <p v-if="cancelMutation.error.value" class="mt-4 text-sm text-rose-600">
          {{
            t('experiments.stopError', {
              message: localizedApiError(cancelMutation.error.value, t, te),
            })
          }}
        </p>
      </div>

      <div class="panel">
        <p class="eyebrow">{{ t('detail.modelReadiness') }}</p>
        <p class="mt-4 text-sm leading-6 text-slate-600">{{ modelReadinessMessage }}</p>
        <div v-if="isActive" class="mt-5 flex flex-wrap gap-3">
          <button
            type="button"
            class="danger-action"
            :disabled="cancelMutation.isPending.value"
            @click="confirmStop"
          >
            <SquareStop aria-hidden="true" />
            {{
              cancelMutation.isPending.value
                ? t('experiments.stopping')
                : t('detail.cancelExperiment')
            }}
          </button>
        </div>
        <div v-if="isTerminal" class="mt-5 flex flex-wrap gap-3">
          <RouterLink
            :to="{ path: '/experiments/compare', query: { ids: experiment.data.value.id } }"
            class="secondary-action compact-action"
          >
            <Scale aria-hidden="true" />
            {{ t('compare.addToCompare') }}
          </RouterLink>
          <button v-if="dataset" class="secondary-action compact-action" @click="cloneAndRerun">
            <RefreshCw aria-hidden="true" />
            {{ t('detail.rerunOnDataset') }}
          </button>
          <RouterLink v-if="model.data.value" to="/models" class="secondary-action compact-action">
            <Cpu aria-hidden="true" />
            {{ t('detail.openModelLibrary') }}
          </RouterLink>
        </div>
      </div>
    </div>

    <EffectiveRouteDiagram
      v-if="!isWebEnsemble"
      variant="actual"
      state="actual"
      collapsible
      :default-expanded="false"
      :route="experiment.data.value.effective_route"
    />

    <div class="grid gap-6 lg:grid-cols-[1fr_0.9fr]">
      <div class="panel">
        <p class="eyebrow">{{ t('detail.metrics') }}</p>
        <p class="mt-3 text-sm text-slate-500">
          {{
            ensembleValidationOnlyCurves
              ? t('ensemble.validationCurvesHint')
              : isPureMlRoute || isWebEnsemble
                ? t('detail.splitComparisonHint')
                : earlyStoppedSucceeded
                  ? progressLabel
                  : t('detail.curveHint')
          }}
        </p>
        <details
          v-if="(isPureMlRoute || isWebEnsemble) && hasFinalMetrics"
          class="section-card mt-4"
        >
          <summary class="cursor-pointer text-sm font-bold">{{
            t('detail.fullSplitComparison')
          }}</summary>
          <p class="mt-2 text-xs leading-5 text-slate-500">{{
            t('detail.fullSplitComparisonHint')
          }}</p>
          <SplitMetricChart :summary="summary" />
        </details>
        <MetricChart
          v-else-if="epochEvents.length"
          :events="events"
          :multilabel="experiment.data.value.task_type === 'pep_function'"
        />
        <p v-else class="section-card mt-4 text-sm text-stone-500">
          {{ metricsEmptyMessage }}
        </p>
        <div v-if="isPureMlRoute && hasTuningResults" class="section-card mt-5">
          <div class="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p class="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
                {{ t('detail.tuningResults') }}
              </p>
              <p class="mt-2 text-sm leading-6 text-slate-500">{{
                t('detail.tuningResultsHint')
              }}</p>
            </div>
            <span class="config-chip">{{ tuningSelectionMessage }}</span>
          </div>
          <TuningChart class="mt-4" :tuning="tuningMetadata" />
        </div>
      </div>
      <LiveExperimentLog :content="logs.data.value?.content" :active="isActive" />
    </div>

    <div class="panel flex flex-wrap items-center justify-between gap-4">
      <div
        ><p class="eyebrow">{{ t('heatmap.title') }}</p
        ><p class="mt-2 text-sm text-slate-500">{{ t('heatmap.entry') }}</p></div
      >
      <RouterLink :to="`/experiments/${id}/visualizations`" class="secondary-action">{{
        t('heatmap.open')
      }}</RouterLink>
    </div>
    <div v-if="analysisData" id="experiment-analysis-artifacts" class="panel">
      <div class="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p class="eyebrow">{{ t('detail.evaluationCurves') }}</p>
          <p class="mt-3 text-sm text-slate-500">
            {{
              t(
                ensembleValidationOnlyCurves
                  ? 'ensemble.validationCurvesHint'
                  : hasMicroEvaluationCurves
                    ? 'detail.microEvaluationCurvesHint'
                    : 'detail.evaluationCurvesHint',
              )
            }}
          </p>
        </div>
      </div>
      <CurveAnalysisChart
        v-if="hasCurveAnalysis"
        class="mt-5"
        :analysis="analysisData"
        :aria-labels="
          ensembleValidationOnlyCurves
            ? { roc: t('ensemble.validationRocAria'), pr: t('ensemble.validationPrAria') }
            : undefined
        "
      />
      <p v-else class="section-card mt-5 text-sm text-stone-500">
        {{ curveAnalysisEmptyMessage }}
      </p>
      <div v-if="hasScoreDiagnostics" class="mt-6">
        <div>
          <p class="eyebrow">{{ t('detail.scoreDiagnostics') }}</p>
          <p class="mt-3 text-sm text-slate-500">{{ t('detail.scoreDiagnosticsHint') }}</p>
        </div>
        <ScoreDiagnosticsChart
          class="mt-5"
          :analysis="analysisData"
          :threshold-description="isWebEnsemble ? t('ensemble.thresholdsHint') : undefined"
        />
      </div>
    </div>

    <div class="panel">
      <div class="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p class="eyebrow">{{ t('detail.representationDiagnostics') }}</p>
          <p class="mt-3 text-sm text-slate-500">{{ t('detail.representationDiagnosticsHint') }}</p>
        </div>
        <div class="flex flex-wrap gap-2">
          <span v-if="clusterSummary" class="config-chip">{{ clusterSummary }}</span>
          <span v-if="hasRepresentationCharts" class="config-chip">{{
            t('detail.analysisAvailable')
          }}</span>
        </div>
      </div>
      <div v-if="analysisData && hasRepresentationCharts" class="mt-5 grid gap-5 xl:grid-cols-2">
        <div class="section-card">
          <div class="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
            {{ t('detail.validationProjection2d') }}
          </div>
          <ProjectionChart class="mt-4" :analysis="analysisData" split="validation" />
        </div>
        <div class="section-card">
          <div class="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
            {{ t('detail.testProjection2d') }}
          </div>
          <ProjectionChart class="mt-4" :analysis="analysisData" split="test" />
        </div>
        <div v-if="analysisData.projection_3d?.available" class="section-card">
          <div class="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
            {{ t('detail.validationProjection3d') }}
          </div>
          <Projection3DChart class="mt-4" :analysis="analysisData" split="validation" />
        </div>
        <div v-if="analysisData.projection_3d?.available" class="section-card">
          <div class="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
            {{ t('detail.testProjection3d') }}
          </div>
          <Projection3DChart class="mt-4" :analysis="analysisData" split="test" />
        </div>
        <div v-if="hasImportancePayload(globalImportance)" class="section-card">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <div class="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
              {{ t('detail.globalModelImportance') }}
            </div>
            <span class="config-chip">{{ importanceMethodLabel(globalImportance) }}</span>
          </div>
          <FeatureImportanceChart
            class="mt-4"
            :importance="globalImportance"
            :context-label="t('detail.globalModelImportance')"
            :method-label="importanceMethodLabel(globalImportance)"
            variant="global"
          />
        </div>
        <div v-if="hasImportancePayload(validationSplitImportance)" class="section-card">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <div class="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
              {{ t('detail.validationSetExplanation') }}
            </div>
            <span class="config-chip">
              {{ importanceMethodLabel(validationSplitImportance, legacySplitImportanceMethod) }}
            </span>
          </div>
          <FeatureImportanceChart
            class="mt-4"
            :importance="validationSplitImportance"
            :context-label="t('detail.validationSetExplanation')"
            :method-label="
              importanceMethodLabel(validationSplitImportance, legacySplitImportanceMethod)
            "
            variant="validation"
          />
        </div>
        <div v-if="hasImportancePayload(testSplitImportance)" class="section-card">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <div class="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
              {{ t('detail.testSetExplanation') }}
            </div>
            <span class="config-chip">
              {{ importanceMethodLabel(testSplitImportance, legacySplitImportanceMethod) }}
            </span>
          </div>
          <FeatureImportanceChart
            class="mt-4"
            :importance="testSplitImportance"
            :context-label="t('detail.testSetExplanation')"
            :method-label="importanceMethodLabel(testSplitImportance, legacySplitImportanceMethod)"
            variant="test"
          />
        </div>
      </div>
      <p v-else-if="analysisData" class="section-card mt-5 text-sm text-stone-500">
        {{ representationEmptyMessage }}
      </p>
      <p v-else class="section-card mt-5 text-sm text-stone-500">
        {{ t('detail.noAnalysisYet') }}
      </p>
      <div v-if="analysisData?.warnings?.length" class="mt-4 grid gap-2">
        <p v-for="warning in analysisData.warnings" :key="warning" class="error-card text-sm">
          {{ warning }}
        </p>
      </div>
    </div>

    <div class="grid gap-6 lg:grid-cols-[0.8fr_1.2fr]">
      <div id="experiment-model" class="panel">
        <PerformanceSummary
          v-if="hasFinalMetrics && (!isWebEnsemble || typeof summary.test_f1 === 'number')"
          :summary="summary"
          :analysis="analysisData"
          :task-type="experiment.data.value.task_type"
        />
        <p v-else-if="!hasFinalMetrics" class="section-card mt-4 text-sm text-stone-500">
          {{ t('detail.noFinalMetricsYet') }}
        </p>
        <div v-if="isPureMlRoute" class="section-card mt-5">
          <div class="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
            {{ t('detail.featurePipeline') }}
          </div>
          <div class="mt-3 flex flex-wrap items-center gap-2">
            <template v-for="(step, index) in mlPipeline" :key="`${step}-${index}`">
              <span class="config-chip">{{ step }}</span>
              <span v-if="index < mlPipeline.length - 1" class="text-slate-400">→</span>
            </template>
          </div>
          <div class="mt-5 text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
            {{ t('detail.fitSummary') }}
          </div>
          <div class="mt-3 flex flex-wrap gap-2">
            <span class="config-chip">{{ t('detail.classifier') }}={{ mlClassifier }}</span>
            <span v-for="item in mlHyperparameters" :key="item" class="config-chip">{{
              item
            }}</span>
          </div>
        </div>
        <details class="section-card mt-5">
          <summary class="cursor-pointer text-sm font-medium">{{
            t('detail.configurationSnapshot')
          }}</summary>
          <pre class="mt-3 overflow-auto text-xs text-slate-700">{{
            JSON.stringify(experiment.data.value.config, null, 2)
          }}</pre>
        </details>
      </div>
      <InferencePanel
        v-if="model.data.value"
        id="experiment-inference"
        :model-id="model.data.value.id"
        :disabled="!model.data.value.supports_inference"
        :reason="String(model.data.value.metadata.reason ?? '')"
        :reason-code="String(model.data.value.metadata.reason_code ?? '')"
        :metadata="model.data.value.metadata"
      />
      <div v-else id="experiment-inference" class="panel">
        <p class="eyebrow">{{ t('inference.eyebrow') }}</p>
        <p class="mt-4 text-sm text-slate-500">{{ t('detail.modelPending') }}</p>
      </div>
    </div>
    <SvmResumeDialog
      :open="resumeDialogOpen"
      :failure="experiment.data.value.failure"
      :config="experiment.data.value.config"
      :pending="resumeMutation.isPending.value"
      :error="resumeError"
      @close="resumeDialogOpen = false"
      @submit="resumeMutation.mutate"
    />
  </section>
</template>
