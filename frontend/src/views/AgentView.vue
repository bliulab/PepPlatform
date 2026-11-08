<script setup lang="ts">
import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { api } from '../api/client'
import { useWorkspaceAccess } from '../workspaceAccess'
import { useLoginDraft } from '../loginDraft'
import {
  completeAgentGeneration,
  streamAgentExecutionPlan,
  streamAgentExecutionPlanReview,
} from '../api/chatStream'
import { localizedApiError, localizedPreflightMessage } from '../errorLocalization'
import { experimentDisplayName } from '../localization'
import { useExperimentDraftStore } from '../stores/experimentDraft'
import AgentCandidateCompare from '../components/AgentCandidateCompare.vue'
import WorkspacePanel from '../components/WorkspacePanel.vue'
import '../agentPlanning.css'
import {
  Activity,
  AlertTriangle,
  Check,
  CheckCircle2,
  ChevronRight,
  Database,
  FileText,
  FlaskConical,
  History,
  List,
  SlidersHorizontal,
  Info,
  Pencil,
  RefreshCw,
  RotateCcw,
  PlugZap,
  ShieldCheck,
  Square,
  Sparkles,
  X,
} from '@lucide/vue'
import {
  agentWorkspaceStageIndex,
  agentWorkspaceStages,
  deriveAgentWorkspaceStage,
  deriveExperimentRunProgress,
} from '../agentWorkspace'
import type { ExperimentRunProgress } from '../agentWorkspace'
import type {
  AgentExecutionAction,
  AgentExecutionPlan,
  AgentExecutionPlanPayload,
  AgentExecutionPlanReviewPayload,
  AgentGenerationStage,
  AgentStrategyPreference,
  ExperimentAnalysisMessage,
  PreflightReport,
} from '../types'

const { isGuest, canUseWorkspace, requireLogin } = useWorkspaceAccess()

const { t, te, locale } = useI18n()
const route = useRoute()
const router = useRouter()
const queryClient = useQueryClient()
const draft = useExperimentDraftStore()

const loadedPlanQueryId = ref('')
const loadedDiagnosticSourceKey = ref('')
const viewMode = ref<'cards' | 'compare'>('cards')
const strategyOptions: AgentStrategyPreference[] = ['conservative', 'balanced', 'aggressive']

const providersQuery = useQuery({
  queryKey: ['ai-providers'],
  queryFn: api.aiProviders,
  enabled: canUseWorkspace,
})
const capabilitiesQuery = useQuery({ queryKey: ['capabilities'], queryFn: api.capabilities })
const datasetsQuery = useQuery({
  queryKey: ['datasets'],
  queryFn: api.datasets,
  enabled: canUseWorkspace,
})
const experimentsQuery = useQuery({
  queryKey: ['experiments'],
  queryFn: api.experiments,
  enabled: canUseWorkspace,
})
const planFromQuery = computed(() => {
  const planId = String(route.query.plan_id || route.query.reuse_plan_id || '')
  return planId.trim()
})
const isReuseMode = computed(() => Boolean(route.query.reuse_plan_id))
const diagnosticSourceExperimentId = computed(() =>
  String(route.query.source_experiment_id || '').trim(),
)
const diagnosticSourceMessageId = computed(() =>
  String(route.query.analysis_message_id || '').trim(),
)
const diagnosticSourceRequested = computed(
  () =>
    !planFromQuery.value &&
    Boolean(diagnosticSourceExperimentId.value || diagnosticSourceMessageId.value),
)
const planQuery = useQuery({
  queryKey: computed(() => ['agent-execution-plan', planFromQuery.value]),
  queryFn: () => api.agentExecutionPlan(planFromQuery.value),
  enabled: computed(() => canUseWorkspace.value && Boolean(planFromQuery.value)),
})
const diagnosticExperimentQuery = useQuery({
  queryKey: computed(() => ['diagnostic-source-experiment', diagnosticSourceExperimentId.value]),
  queryFn: () => api.experiment(diagnosticSourceExperimentId.value),
  enabled: computed(
    () =>
      canUseWorkspace.value &&
      diagnosticSourceRequested.value &&
      Boolean(diagnosticSourceExperimentId.value && diagnosticSourceMessageId.value),
  ),
})
const diagnosticMessageQuery = useQuery({
  queryKey: computed(() => [
    'diagnostic-source-message',
    diagnosticSourceExperimentId.value,
    diagnosticSourceMessageId.value,
  ]),
  queryFn: () =>
    api.experimentAnalysisMessage(
      diagnosticSourceExperimentId.value,
      diagnosticSourceMessageId.value,
    ),
  enabled: computed(
    () =>
      canUseWorkspace.value &&
      diagnosticSourceRequested.value &&
      Boolean(diagnosticSourceExperimentId.value && diagnosticSourceMessageId.value),
  ),
})

const selectedProviderId = ref('')
const taskType = ref('pep_function')
const datasetId = ref('')
const objective = ref('')
const strategyPreference = ref<AgentStrategyPreference>('balanced')
const includeDatasetPreview = ref(false)
const baselineExperimentIds = ref<string[]>([])
const errorMessage = ref('')
const reviewErrorMessage = ref('')
const editErrorMessage = ref('')
const activePlan = ref<AgentExecutionPlan | null>(null)
const editingActionId = ref('')
const editingConfigText = ref('')
const autoReportRequestedForPlanId = ref('')
const activePanel = ref<'context' | 'inspector' | null>(null)
const contextRailOpen = computed(() => activePanel.value === 'context')
const detailRailOpen = computed(() => activePanel.value === 'inspector')
function togglePanel(panel: 'context' | 'inspector') {
  activePanel.value = activePanel.value === panel ? null : panel
}
const baselineSearch = ref('')
const activeCandidateId = ref('')
const configEditorTextarea = ref<HTMLTextAreaElement | null>(null)
const detailRailScroll = ref<HTMLElement | null>(null)
const generationStage = ref<AgentGenerationStage | ''>('')
const stoppedGeneration = ref<'plan' | 'review' | ''>('')
let activeGeneration: { kind: 'plan' | 'review'; controller: AbortController } | null = null
let refreshTimer: ReturnType<typeof window.setTimeout> | undefined

useLoginDraft(
  'agent-design',
  () => ({
    taskType: taskType.value,
    objective: objective.value,
    strategyPreference: strategyPreference.value,
    includeDatasetPreview: includeDatasetPreview.value,
  }),
  (saved) => {
    taskType.value = saved.taskType
    objective.value = saved.objective
    strategyPreference.value = saved.strategyPreference
    includeDatasetPreview.value = saved.includeDatasetPreview
  },
)

const enabledProviders = computed(() =>
  (providersQuery.data.value ?? []).filter((provider) => provider.enabled),
)
const selectedProvider = computed(() =>
  enabledProviders.value.find((provider) => provider.id === selectedProviderId.value),
)
const taskOptions = computed(() => capabilitiesQuery.data.value?.tasks ?? [])
const matchingDatasets = computed(() =>
  (datasetsQuery.data.value ?? []).filter((dataset) => dataset.task_type === taskType.value),
)
const matchingExperiments = computed(() =>
  (experimentsQuery.data.value ?? []).filter(
    (experiment) =>
      experiment.task_type === taskType.value && experiment.dataset_id === datasetId.value,
  ),
)
const diagnosticSourceMessage = computed<ExperimentAnalysisMessage | undefined>(
  () => diagnosticMessageQuery.data.value,
)
const diagnosticSourceReport = computed(() => diagnosticSourceMessage.value?.payload?.report)
const diagnosticSourceReady = computed(() =>
  Boolean(
    diagnosticSourceRequested.value &&
    diagnosticExperimentQuery.data.value &&
    diagnosticSourceMessage.value?.kind === 'report' &&
    diagnosticSourceMessage.value?.status === 'complete' &&
    diagnosticSourceReport.value,
  ),
)
const diagnosticSourceError = computed(() => {
  if (!diagnosticSourceRequested.value) return ''
  if (!diagnosticSourceExperimentId.value || !diagnosticSourceMessageId.value) {
    return t('agent.diagnosticSourceInvalid')
  }
  if (diagnosticExperimentQuery.isError.value || diagnosticMessageQuery.isError.value) {
    return t('agent.diagnosticSourceLoadFailed')
  }
  if (
    diagnosticMessageQuery.data.value &&
    (diagnosticMessageQuery.data.value.kind !== 'report' ||
      diagnosticMessageQuery.data.value.status !== 'complete' ||
      !diagnosticSourceReport.value)
  ) {
    return t('agent.diagnosticSourceNotReady')
  }
  return ''
})
const diagnosticSourceIsStale = computed(() => {
  const experimentUpdatedAt = diagnosticExperimentQuery.data.value?.updated_at || ''
  const capturedUpdatedAt =
    diagnosticSourceMessage.value?.context_manifest?.experiment_updated_at || ''
  return Boolean(
    experimentUpdatedAt && capturedUpdatedAt && experimentUpdatedAt > capturedUpdatedAt,
  )
})
const isLoadingContext = computed(
  () =>
    providersQuery.isLoading.value ||
    capabilitiesQuery.isLoading.value ||
    datasetsQuery.isLoading.value ||
    (diagnosticSourceRequested.value &&
      (diagnosticExperimentQuery.isLoading.value || diagnosticMessageQuery.isLoading.value)),
)

const designMutation = useMutation({
  mutationFn: ({
    payload,
    controller,
  }: {
    payload: AgentExecutionPlanPayload
    controller: AbortController
  }) =>
    completeAgentGeneration(streamAgentExecutionPlan(payload, controller.signal), (stage) => {
      generationStage.value = stage
    }),
  onSuccess: (data) => {
    activePlan.value = data
    errorMessage.value = ''
  },
  onError: (error, variables) => {
    if (variables.controller.signal.aborted) {
      stoppedGeneration.value = 'plan'
      errorMessage.value = ''
      return
    }
    errorMessage.value = localizedApiError(error, t, te, 'agent.genericError')
  },
  onSettled: (_data, _error, variables) => {
    if (activeGeneration?.controller === variables.controller) activeGeneration = null
    generationStage.value = ''
  },
})

const result = computed(() => activePlan.value)
const workspaceStage = computed(() => deriveAgentWorkspaceStage(result.value))
const workspaceStagePosition = computed(() => agentWorkspaceStageIndex(workspaceStage.value))
const workspaceStageItems = computed(() =>
  agentWorkspaceStages.map((stage, index) => ({
    stage,
    label: t(`agent.workspaceStage.${stage}`),
    complete: index < workspaceStagePosition.value,
    active: index === workspaceStagePosition.value,
  })),
)
const selectedDataset = computed(() =>
  matchingDatasets.value.find((dataset) => dataset.id === datasetId.value),
)
const selectedBaselines = computed(() =>
  matchingExperiments.value.filter((experiment) =>
    baselineExperimentIds.value.includes(experiment.id),
  ),
)
function displayExperimentName(experiment: Parameters<typeof experimentDisplayName>[0]) {
  return experimentDisplayName(experiment, t)
}

function preflightMessages(report: PreflightReport | undefined, kind: 'errors' | 'warnings') {
  const codes = kind === 'errors' ? (report?.error_codes ?? []) : (report?.warning_codes ?? [])
  const messages = kind === 'errors' ? (report?.errors ?? []) : (report?.warnings ?? [])
  return messages.map((message, index) => localizedPreflightMessage(codes[index], message, t, te))
}
const baselineOptions = computed(() => {
  const query = baselineSearch.value.trim().toLocaleLowerCase()
  return matchingExperiments.value
    .filter((experiment) => {
      if (!query) return true
      return `${displayExperimentName(experiment)} ${experiment.status}`
        .toLocaleLowerCase()
        .includes(query)
    })
    .slice(0, 12)
})
const activeCandidate = computed(
  () =>
    result.value?.actions.find((candidate) => candidate.id === activeCandidateId.value) ??
    result.value?.actions[0],
)
const workspaceTitle = computed(
  () => result.value?.objective || objective.value.trim() || t('agent.workspaceNewTask'),
)
const canGenerate = computed(() =>
  Boolean(
    selectedProviderId.value &&
    datasetId.value &&
    (!diagnosticSourceRequested.value || diagnosticSourceReady.value) &&
    !designMutation.isPending.value,
  ),
)
const usageLabel = computed(() => {
  const usage = result.value?.usage ?? {}
  const total = usage.total_tokens ?? usage.totalTokens
  const input = usage.prompt_tokens ?? usage.promptTokens
  const output = usage.completion_tokens ?? usage.completionTokens
  const parts = []
  if (input !== undefined) parts.push(`in=${input}`)
  if (output !== undefined) parts.push(`out=${output}`)
  if (total !== undefined) parts.push(`total=${total}`)
  return parts.join(' · ')
})
const manifestSections = computed(() => result.value?.context_manifest?.sections ?? [])
const contextSummary = computed(() => result.value?.context_summary)

function retryContext() {
  if (!requireLogin()) return
  void providersQuery.refetch()
  void datasetsQuery.refetch()
  void capabilitiesQuery.refetch()
}

function taskLabel(value: string) {
  return t(`tasks.${value}.label`)
}

function addBaselineExperiment(experimentId: string) {
  if (baselineExperimentIds.value.includes(experimentId) || baselineExperimentIds.value.length >= 6)
    return
  baselineExperimentIds.value = [...baselineExperimentIds.value, experimentId]
  baselineSearch.value = ''
}

function removeBaselineExperiment(experimentId: string) {
  baselineExperimentIds.value = baselineExperimentIds.value.filter((id) => id !== experimentId)
}

function detachDiagnosticSource() {
  const query = { ...route.query }
  delete query.source_experiment_id
  delete query.analysis_message_id
  loadedDiagnosticSourceKey.value = ''
  void router.replace({ path: route.path, query })
}

function focusCandidate(candidate: AgentExecutionAction) {
  activeCandidateId.value = candidate.id
  activePanel.value = 'inspector'
}

function laneLabel(candidate: AgentExecutionAction) {
  if (['conservative', 'balanced', 'exploratory'].includes(candidate.lane)) {
    return t(`agent.lanes.${candidate.lane}`)
  }
  return candidate.lane
}

function laneClass(candidate: AgentExecutionAction) {
  if (candidate.lane === 'conservative') return 'agent-lane-moss'
  if (candidate.lane === 'balanced') return 'agent-lane-brass'
  return 'agent-lane-accent'
}

function candidateList(value: string[] | undefined) {
  return Array.isArray(value) ? value : []
}

function riskKey(candidate: AgentExecutionAction) {
  return ['low', 'medium', 'high'].includes(candidate.risk_level) ? candidate.risk_level : 'medium'
}

function riskLabel(candidate: AgentExecutionAction) {
  return t(`agent.risk.${riskKey(candidate)}`)
}

function riskClass(candidate: AgentExecutionAction) {
  const risk = riskKey(candidate)
  if (risk === 'low') return 'status-pill-success'
  if (risk === 'high') return 'status-pill-danger'
  return 'status-pill-warning'
}

function actionStatusLabel(candidate: AgentExecutionAction) {
  if (candidate.status === 'pending') return t('agent.actionStatus.pending')
  if (candidate.status === 'executed') return t('agent.actionStatus.executed')
  return candidate.status
}

function candidateConfig(candidate: AgentExecutionAction) {
  return candidate.preflight?.normalized_config ?? candidate.config ?? {}
}

function configText(candidate: AgentExecutionAction) {
  return JSON.stringify(candidateConfig(candidate), null, 2)
}

function hasConfig(candidate: AgentExecutionAction) {
  return (
    candidate.type === 'create_experiment' && Object.keys(candidateConfig(candidate)).length > 0
  )
}

function generateDesigns() {
  if (!requireLogin()) return
  if (!canGenerate.value) return
  errorMessage.value = ''
  reviewErrorMessage.value = ''
  activePlan.value = null
  reviewMutation.reset()
  editErrorMessage.value = ''
  editingActionId.value = ''
  stoppedGeneration.value = ''
  generationStage.value = 'generating'
  const controller = new AbortController()
  activeGeneration = { kind: 'plan', controller }
  designMutation.mutate({
    controller,
    payload: {
      provider_config_id: selectedProviderId.value,
      task_type: taskType.value,
      dataset_id: datasetId.value,
      objective: objective.value.trim(),
      strategy_preference: strategyPreference.value,
      baseline_experiment_ids: baselineExperimentIds.value.slice(0, 6),
      language: locale.value,
      include: {
        dataset_preview: includeDatasetPreview.value,
      },
      ...(diagnosticSourceReady.value
        ? {
            diagnostic_source: {
              experiment_id: diagnosticSourceExperimentId.value,
              message_id: diagnosticSourceMessageId.value,
            },
          }
        : {}),
    },
  })
}

function stopGeneration(kind: 'plan' | 'review') {
  if (activeGeneration?.kind === kind) activeGeneration.controller.abort()
}

function generationStatusLabel(kind: 'plan' | 'review') {
  return t(`agent.generationStage.${kind}.${generationStage.value || 'generating'}`)
}

function applyCandidate(candidate: AgentExecutionAction) {
  if (
    !result.value?.id ||
    !candidate.valid ||
    candidate.status !== 'pending' ||
    candidate.type !== 'create_experiment'
  )
    return
  draft.applyAgentCandidate(
    taskType.value,
    datasetId.value,
    candidateConfig(candidate),
    candidate.title,
    {
      rationale: candidate.rationale,
      planId: result.value.id,
      actionId: candidate.id,
      lane: ['conservative', 'balanced', 'exploratory'].includes(candidate.lane)
        ? (candidate.lane as 'conservative' | 'balanced' | 'exploratory')
        : undefined,
      preflight: candidate.preflight,
    },
  )
  router.push('/experiments/new')
}

function startEditingCandidate(candidate: AgentExecutionAction) {
  if (candidate.type !== 'create_experiment' || candidate.status !== 'pending') return
  editingActionId.value = candidate.id
  editingConfigText.value = JSON.stringify(candidateConfig(candidate), null, 2)
  editErrorMessage.value = ''
}

function openCandidateEditor(candidate: AgentExecutionAction, event?: Event) {
  event?.stopPropagation()
  if (candidate.type !== 'create_experiment' || candidate.status !== 'pending') return
  activeCandidateId.value = candidate.id
  activePanel.value = 'inspector'
  startEditingCandidate(candidate)
  void nextTick(() => configEditorTextarea.value?.focus())
}

function cancelEditingCandidate() {
  editingActionId.value = ''
  editingConfigText.value = ''
  editErrorMessage.value = ''
}

function restoreOriginalCandidate(candidate: AgentExecutionAction) {
  const original = candidate.original_config ?? candidate.config ?? {}
  editingActionId.value = candidate.id
  editingConfigText.value = JSON.stringify(original, null, 2)
  editErrorMessage.value = ''
}

function saveEditedCandidate(candidate: AgentExecutionAction) {
  if (!result.value?.id || candidate.type !== 'create_experiment') return
  let parsed: Record<string, unknown>
  try {
    const value = JSON.parse(editingConfigText.value)
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error('config must be an object')
    }
    parsed = value as Record<string, unknown>
  } catch {
    editErrorMessage.value = t('agent.editInvalidJson')
    return
  }
  editErrorMessage.value = ''
  actionEditMutation.mutate({ actionId: candidate.id, config: parsed })
}

function addCreatedAsBaselines() {
  const createdIds = createdExperiments.value.map((item) => item.experiment.id)
  baselineExperimentIds.value = Array.from(
    new Set([...baselineExperimentIds.value, ...createdIds]),
  ).slice(0, 6)
}

const actionEditMutation = useMutation({
  mutationFn: ({ actionId, config }: { actionId: string; config: Record<string, unknown> }) =>
    api.updateAgentExecutionAction(result.value?.id ?? '', actionId, { config }),
  onSuccess: (data) => {
    activePlan.value = data
    editErrorMessage.value = ''
    const editedAction = data.actions.find((action) => action.id === editingActionId.value)
    if (editedAction?.valid) {
      editingActionId.value = ''
      editingConfigText.value = ''
      return
    }
    if (editedAction) {
      editingConfigText.value = JSON.stringify(candidateConfig(editedAction), null, 2)
      void nextTick(() => configEditorTextarea.value?.focus())
    }
  },
  onError: (error) => {
    editErrorMessage.value = localizedApiError(error, t, te, 'agent.editGenericError')
  },
})

const createdExperiments = computed(() => result.value?.execution?.created_experiments ?? [])
const canceledExperiments = computed(() => result.value?.execution?.canceled_experiments ?? [])
const progress = computed(() => result.value?.execution?.progress ?? {})
const reportReady = computed(() => Boolean(result.value?.execution?.report_ready))
const review = computed(() => result.value?.review ?? {})
const hasReview = computed(() => Boolean(review.value.summary))
const totalExperiments = computed(() =>
  Number(progress.value.total ?? createdExperiments.value.length ?? 0),
)
const completedExperiments = computed(
  () =>
    progressCount('succeeded') +
    progressCount('failed') +
    progressCount('canceled') +
    progressCount('missing'),
)
const experimentRuns = computed(() =>
  createdExperiments.value.map((item) => ({
    ...item,
    progress: deriveExperimentRunProgress(item.experiment),
  })),
)
const metricEvidence = computed(() =>
  candidateList(review.value.metric_evidence ?? review.value.evidence),
)
const analysisFindings = computed(() => candidateList(review.value.analysis_findings))
const recommendations = computed(() =>
  candidateList(review.value.recommendations ?? review.value.next_recommendations),
)
const failedOrIncomplete = computed(() =>
  Array.isArray(review.value.failed_or_incomplete) ? review.value.failed_or_incomplete : [],
)
const bestExperiment = computed(() => review.value.best_experiment ?? {})
const bestExperimentName = computed(() => {
  const best = bestExperiment.value
  return String(best.name || best.experiment_id || '').trim()
})
const bestExperimentReason = computed(() => String(bestExperiment.value.reason || '').trim())

const reviewMutation = useMutation({
  mutationFn: ({
    planId,
    payload,
    controller,
  }: {
    planId: string
    payload: AgentExecutionPlanReviewPayload
    controller: AbortController
  }) =>
    completeAgentGeneration(
      streamAgentExecutionPlanReview(planId, payload, controller.signal),
      (stage) => {
        generationStage.value = stage
      },
    ),
  onSuccess: (data) => {
    activePlan.value = data
    reviewErrorMessage.value = ''
  },
  onError: (error, variables) => {
    if (variables.controller.signal.aborted) {
      stoppedGeneration.value = 'review'
      reviewErrorMessage.value = ''
      return
    }
    reviewErrorMessage.value = localizedApiError(error, t, te, 'agent.reviewGenericError')
  },
  onSettled: (_data, _error, variables) => {
    if (activeGeneration?.controller === variables.controller) activeGeneration = null
    generationStage.value = ''
  },
})

const canReview = computed(() =>
  Boolean(
    result.value?.id &&
    createdExperiments.value.length &&
    reportReady.value &&
    !reviewMutation.isPending.value,
  ),
)

function reviewPlan() {
  if (!requireLogin()) return
  if (!canReview.value) return
  reviewErrorMessage.value = ''
  stoppedGeneration.value = ''
  generationStage.value = 'generating'
  const controller = new AbortController()
  activeGeneration = { kind: 'review', controller }
  reviewMutation.mutate({
    planId: result.value?.id ?? '',
    controller,
    payload: {
      provider_config_id: selectedProviderId.value,
      prompt: objective.value.trim(),
      language: locale.value,
      include: {
        dataset_preview: includeDatasetPreview.value,
      },
    },
  })
}

function statusLabel(status: string | undefined) {
  if (!status) return ''
  if (['queued', 'running', 'succeeded', 'failed', 'canceled'].includes(status)) {
    return t(`status.${status}`)
  }
  if (status === 'draft') return t('agent.planStatus.draft')
  if (status === 'ready_for_report') return t('agent.planStatus.ready_for_report')
  if (status === 'reported' || status === 'reviewed') return t('agent.planStatus.reported')
  return status
}

function progressCount(
  key: 'queued' | 'running' | 'succeeded' | 'failed' | 'canceled' | 'missing',
) {
  return Number(progress.value[key] ?? 0)
}

function experimentRunStatusLabel(run: ExperimentRunProgress) {
  return t(`agent.runStatus.${run.status}`)
}

function experimentRunProgressLabel(run: ExperimentRunProgress) {
  const epochParams = { current: run.current, total: run.total, percent: run.percent }
  if (run.status === 'queued') return t('agent.runProgress.waiting')
  if (run.status === 'running') {
    if (run.mode === 'determinate') return t('agent.runProgress.epochRunning', epochParams)
    return run.usesEpochs
      ? t('agent.runProgress.trainingStarting')
      : t('agent.runProgress.fitRunning')
  }
  if (run.status === 'succeeded') {
    if (run.earlyStopped) return t('agent.runProgress.earlyStopped', epochParams)
    if (run.usesEpochs && run.current && run.total)
      return t('agent.runProgress.epochCompleted', epochParams)
    return run.usesEpochs
      ? t('agent.runProgress.trainingCompleted')
      : t('agent.runProgress.fitCompleted')
  }
  if (run.status === 'failed') {
    return run.usesEpochs && run.current && run.total
      ? t('agent.runProgress.failedAtEpoch', epochParams)
      : t('agent.runProgress.failed')
  }
  if (run.status === 'canceled') {
    return run.usesEpochs && run.current && run.total
      ? t('agent.runProgress.stoppedAtEpoch', epochParams)
      : t('agent.runProgress.stopped')
  }
  return t('agent.runProgress.unavailable')
}

function experimentRunProgressWidth(run: ExperimentRunProgress) {
  if (run.mode === 'indeterminate') return '38%'
  if (run.status === 'succeeded') return '100%'
  if (['failed', 'canceled'].includes(run.status)) {
    return run.percent === undefined ? '100%' : `${run.percent}%`
  }
  return `${run.percent ?? 0}%`
}

function objectText(value: Record<string, unknown>, key: string) {
  return String(value[key] ?? '').trim()
}

function schedulePlanRefresh() {
  if (refreshTimer) window.clearTimeout(refreshTimer)
  if (!result.value?.id || hasReview.value) return
  if (!['running', 'executed', 'ready_for_report'].includes(result.value.status)) return
  refreshTimer = window.setTimeout(() => {
    refreshMutation.mutate()
  }, 4000)
}

const refreshMutation = useMutation({
  mutationFn: () => api.refreshAgentExecutionPlan(result.value?.id ?? ''),
  onSuccess: (data) => {
    activePlan.value = data
    queryClient.invalidateQueries({ queryKey: ['experiments'] })
  },
})

watch(
  enabledProviders,
  (providers) => {
    if (
      selectedProviderId.value &&
      providers.some((provider) => provider.id === selectedProviderId.value)
    )
      return
    selectedProviderId.value = providers[0]?.id ?? ''
  },
  { immediate: true },
)

watch(
  matchingDatasets,
  (datasets) => {
    if (datasetId.value && datasets.some((dataset) => dataset.id === datasetId.value)) return
    datasetId.value = datasets[0]?.id ?? ''
  },
  { immediate: true },
)

watch([() => taskType.value, () => datasetId.value], () => {
  baselineExperimentIds.value = baselineExperimentIds.value.filter((id) =>
    matchingExperiments.value.some((experiment) => experiment.id === id),
  )
})

watch(result, (value) => {
  if (!value?.actions.some((candidate) => candidate.id === activeCandidateId.value)) {
    activeCandidateId.value = value?.actions[0]?.id ?? ''
  }
  schedulePlanRefresh()
})

watch(
  () => planQuery.data.value,
  (plan) => {
    if (!plan || loadedPlanQueryId.value === plan.id) return
    loadedPlanQueryId.value = plan.id
    taskType.value = plan.task_type || taskType.value
    datasetId.value = plan.dataset_id || datasetId.value
    objective.value = plan.objective || ''
    const storedStrategy = plan.trace?.strategy_preference
    strategyPreference.value = ['conservative', 'balanced', 'aggressive'].includes(
      storedStrategy ?? '',
    )
      ? (storedStrategy as AgentStrategyPreference)
      : 'balanced'
    selectedProviderId.value = plan.provider_config_id || selectedProviderId.value
    activePlan.value = isReuseMode.value ? null : plan
    if (isReuseMode.value) {
      autoReportRequestedForPlanId.value = ''
    }
  },
  { immediate: true },
)

watch(
  [() => diagnosticExperimentQuery.data.value, () => diagnosticMessageQuery.data.value],
  ([sourceExperiment, sourceMessage]) => {
    if (!diagnosticSourceReady.value || !sourceExperiment || !sourceMessage) return
    const sourceKey = `${sourceExperiment.id}:${sourceMessage.id}`
    if (loadedDiagnosticSourceKey.value === sourceKey) return
    loadedDiagnosticSourceKey.value = sourceKey
    taskType.value = sourceExperiment.task_type
    datasetId.value = sourceExperiment.dataset_id
    baselineExperimentIds.value = Array.from(
      new Set([sourceExperiment.id, ...baselineExperimentIds.value]),
    ).slice(0, 6)
    if (!objective.value.trim()) {
      objective.value = t('agent.diagnosticObjective', {
        name: displayExperimentName(sourceExperiment),
      })
    }
    activePlan.value = null
  },
  { immediate: true },
)

watch(
  [diagnosticSourceMessage, enabledProviders],
  ([sourceMessage, providers]) => {
    if (!diagnosticSourceRequested.value || !sourceMessage?.provider_config_id) return
    if (providers.some((provider) => provider.id === sourceMessage.provider_config_id)) {
      selectedProviderId.value = sourceMessage.provider_config_id
    }
  },
  { immediate: true },
)

watch(
  () => route.query.plan_id || route.query.reuse_plan_id,
  () => {
    loadedPlanQueryId.value = ''
  },
)

watch([diagnosticSourceExperimentId, diagnosticSourceMessageId], () => {
  loadedDiagnosticSourceKey.value = ''
})

watch([reportReady, hasReview, () => result.value?.id], ([ready, reviewed, planId]) => {
  if (!ready || reviewed || !planId || autoReportRequestedForPlanId.value === planId) return
  autoReportRequestedForPlanId.value = planId
  reviewPlan()
})

watch(
  [() => result.value?.id, () => createdExperiments.value.length, hasReview],
  (
    [planId, createdCount, reviewed],
    [previousPlanId, previousCreatedCount, previouslyReviewed],
  ) => {
    const openedExecution = Boolean(
      planId && createdCount && (planId !== previousPlanId || !previousCreatedCount),
    )
    const openedReview = Boolean(reviewed && !previouslyReviewed)
    if (!openedExecution && !openedReview) return
    void nextTick(() => detailRailScroll.value?.scrollTo({ top: 0, behavior: 'auto' }))
  },
)

onBeforeUnmount(() => {
  if (refreshTimer) window.clearTimeout(refreshTimer)
  activeGeneration?.controller.abort()
})
</script>

<template>
  <section class="agent-design-page">
    <header class="agent-design-toolbar">
      <div class="agent-design-toolbar-title">
        <button
          type="button"
          class="agent-toolbar-icon agent-context-trigger"
          :title="
            contextRailOpen ? t('agent.workspaceCloseContext') : t('agent.workspaceOpenContext')
          "
          :aria-label="
            contextRailOpen ? t('agent.workspaceCloseContext') : t('agent.workspaceOpenContext')
          "
          :aria-expanded="contextRailOpen"
          aria-controls="agent-context-rail"
          @click="togglePanel('context')"
        >
          <SlidersHorizontal aria-hidden="true" />
          <span>{{ t('agent.workspaceContextShort') }}</span>
        </button>
        <div class="min-w-0">
          <h1>{{ workspaceTitle }}</h1>
        </div>
      </div>

      <ol class="agent-stage-track" :aria-label="t('agent.workspaceProgress')">
        <li
          v-for="item in workspaceStageItems"
          :key="item.stage"
          class="agent-stage-item"
          :class="{
            'agent-stage-item-active': item.active,
            'agent-stage-item-complete': item.complete,
          }"
          :aria-current="item.active ? 'step' : undefined"
        >
          <span class="agent-stage-marker">
            <Check v-if="item.complete" aria-hidden="true" />
          </span>
          <span>{{ item.label }}</span>
        </li>
      </ol>

      <div class="agent-design-toolbar-actions">
        <label v-if="enabledProviders.length" class="agent-model-picker">
          <span class="sr-only">{{ t('agent.provider') }}</span>
          <select
            v-model="selectedProviderId"
            :title="
              selectedProvider
                ? `${selectedProvider.display_name} · ${selectedProvider.model}`
                : t('agent.provider')
            "
          >
            <option v-for="provider in enabledProviders" :key="provider.id" :value="provider.id">
              {{ provider.display_name }} · {{ provider.model }}
            </option>
          </select>
        </label>
        <RouterLink
          to="/agent/plans"
          class="agent-toolbar-history"
          :title="t('agent.history')"
          :aria-label="t('agent.history')"
        >
          <History aria-hidden="true" />
          <span>{{ t('agent.history') }}</span>
        </RouterLink>
        <RouterLink
          to="/account/ai-services"
          class="agent-toolbar-service"
          :title="t('aiServices.openSettings')"
          :aria-label="t('aiServices.openSettings')"
        >
          <PlugZap aria-hidden="true" />
          <span class="agent-toolbar-service-label">{{ t('aiServices.shortLabel') }}</span>
          <span class="agent-toolbar-service-label-compact">{{ t('aiServices.mobileLabel') }}</span>
        </RouterLink>
        <button
          type="button"
          class="agent-toolbar-icon"
          v-if="result"
          :title="
            detailRailOpen ? t('agent.workspaceCloseInspector') : t('agent.workspaceOpenInspector')
          "
          :aria-label="
            detailRailOpen ? t('agent.workspaceCloseInspector') : t('agent.workspaceOpenInspector')
          "
          :aria-expanded="detailRailOpen"
          aria-controls="agent-detail-rail"
          @click="togglePanel('inspector')"
        >
          <Info aria-hidden="true" />
        </button>
      </div>
    </header>

    <div
      class="agent-design-shell"
      :class="{
        'agent-design-shell-no-context': !contextRailOpen,
        'agent-design-shell-no-detail': !detailRailOpen,
      }"
    >
      <WorkspacePanel
        v-if="contextRailOpen"
        id="agent-context-rail"
        class="agent-context-rail"
        side="left"
        :label="t('agent.workspaceContext')"
        @close="activePanel = null"
      >
        <div class="agent-rail-heading">
          <div>
            <p class="agent-rail-kicker">{{ t('agent.workspaceContext') }}</p>
            <h2>{{ t('agent.workspaceResearchInputs') }}</h2>
          </div>
          <button
            type="button"
            class="agent-rail-close"
            :title="t('agent.workspaceCloseContext')"
            :aria-label="t('agent.workspaceCloseContext')"
            @click="activePanel = null"
          >
            <X aria-hidden="true" />
          </button>
        </div>

        <div class="agent-rail-scroll">
          <div v-if="isGuest || enabledProviders.length" class="agent-context-fields">
            <div v-if="result" class="agent-input-selectors agent-input-selectors--rail">
              <label class="form-field">
                <span>{{ t('agent.task') }}</span>
                <select v-model="taskType" class="field-control">
                  <option v-for="task in taskOptions" :key="task.id" :value="task.id">
                    {{ taskLabel(task.id) }}
                  </option>
                </select>
              </label>
              <label class="form-field">
                <span>{{ t('agent.dataset') }}</span>
                <select v-model="datasetId" class="field-control">
                  <option value="">{{ t('agent.selectDataset') }}</option>
                  <option
                    v-for="dataset in matchingDatasets"
                    :key="dataset.id"
                    :value="dataset.id"
                    :title="dataset.name"
                  >
                    {{ dataset.name }}
                  </option>
                </select>
              </label>
            </div>
            <div class="agent-baseline-field">
              <div class="agent-field-label-row">
                <span>{{ t('agent.baselineExperiments') }}</span>
                <span>{{ baselineExperimentIds.length }}/6</span>
              </div>
              <div v-if="selectedBaselines.length" class="agent-baseline-selected">
                <span
                  v-for="experiment in selectedBaselines"
                  :key="experiment.id"
                  class="agent-baseline-chip"
                >
                  <span>{{ displayExperimentName(experiment) }}</span>
                  <button
                    type="button"
                    :title="
                      t('agent.workspaceRemoveBaseline', {
                        name: displayExperimentName(experiment),
                      })
                    "
                    :aria-label="
                      t('agent.workspaceRemoveBaseline', {
                        name: displayExperimentName(experiment),
                      })
                    "
                    @click="removeBaselineExperiment(experiment.id)"
                  >
                    <X aria-hidden="true" />
                  </button>
                </span>
              </div>
              <input
                v-model="baselineSearch"
                type="search"
                class="field-control"
                :placeholder="t('agent.workspaceSearchBaselines')"
              />
              <div v-if="matchingExperiments.length" class="agent-baseline-options">
                <button
                  v-for="experiment in baselineOptions"
                  :key="experiment.id"
                  type="button"
                  :class="{
                    'agent-baseline-option-selected': baselineExperimentIds.includes(experiment.id),
                  }"
                  :disabled="
                    !baselineExperimentIds.includes(experiment.id) &&
                    baselineExperimentIds.length >= 6
                  "
                  @click="
                    baselineExperimentIds.includes(experiment.id)
                      ? removeBaselineExperiment(experiment.id)
                      : addBaselineExperiment(experiment.id)
                  "
                >
                  <span>
                    <strong>{{ displayExperimentName(experiment) }}</strong>
                    <small>{{ statusLabel(experiment.status) }}</small>
                  </span>
                  <Check v-if="baselineExperimentIds.includes(experiment.id)" aria-hidden="true" />
                  <ChevronRight v-else aria-hidden="true" />
                </button>
              </div>
              <p class="muted-copy text-xs leading-5">{{ t('agent.baselineHint') }}</p>
            </div>

            <label class="agent-context-toggle">
              <span>
                <Database aria-hidden="true" />
                <span>
                  <strong>{{ t('agent.includeDatasetPreview') }}</strong>
                  <small>{{ t('agent.workspaceDatasetPreviewHint') }}</small>
                </span>
              </span>
              <input v-model="includeDatasetPreview" type="checkbox" />
            </label>
          </div>

          <p v-else class="agent-rail-empty">
            {{
              providersQuery.isLoading.value
                ? t('detail.aiLoadingProviders')
                : t('agent.noProviders')
            }}
          </p>
        </div>

        <div class="agent-context-footer">
          <span>{{
            isLoadingContext
              ? t('agent.loading')
              : t('agent.workspaceDatasetCount', { count: matchingDatasets.length })
          }}</span>
          <span v-if="selectedProvider">{{ selectedProvider.model }}</span>
        </div>
      </WorkspacePanel>

      <section class="agent-plan-canvas">
        <p
          v-if="planQuery.isLoading.value && planFromQuery"
          class="agent-input-status"
          role="status"
        >
          {{ t('common.loading') }}
        </p>
        <p v-if="planQuery.isError.value" class="error-card" role="alert">
          {{ t('agent.workspaceLoadError') }}
          <button type="button" class="tertiary-action compact-action" @click="planQuery.refetch()">
            {{ t('home.retry') }}
          </button>
        </p>
        <div v-if="result" class="agent-plan-surface">
          <div class="agent-plan-head">
            <div>
              <p class="agent-rail-kicker">{{ t('agent.workspaceGeneratedPlan') }}</p>
              <h2>{{ result.objective || t('agent.workspaceUntitledObjective') }}</h2>
              <p class="muted-copy mt-2 text-xs">
                {{ t('agent.workspaceOptionCount', { count: result.actions.length }) }}
              </p>
            </div>
            <div class="agent-plan-head-actions">
              <span class="config-chip">{{ statusLabel(result.status) }}</span>
              <div class="mode-toggle mode-toggle-compact inline-flex">
                <button
                  type="button"
                  class="mode-toggle-button"
                  :class="{ 'mode-toggle-button-active': viewMode === 'cards' }"
                  @click="viewMode = 'cards'"
                >
                  <List aria-hidden="true" />
                  {{ t('agent.viewMode.cards') }}
                </button>
                <button
                  type="button"
                  class="mode-toggle-button"
                  :class="{ 'mode-toggle-button-active': viewMode === 'compare' }"
                  @click="viewMode = 'compare'"
                >
                  <FileText aria-hidden="true" />
                  {{ t('agent.viewMode.compare') }}
                </button>
              </div>
            </div>
          </div>

          <section v-if="createdExperiments.length" class="agent-execution-overview">
            <div class="agent-inspector-title">
              <Activity aria-hidden="true" />
              <h3>{{ t('agent.workspaceRunStatus') }}</h3>
            </div>
            <div class="agent-run-summary" aria-live="polite" aria-atomic="true">
              <span>{{ t('agent.finishedExperiments') }}</span>
              <strong>{{ completedExperiments }}/{{ totalExperiments }}</strong>
            </div>
            <div class="agent-run-list">
              <RouterLink
                v-for="run in experimentRuns"
                :key="run.experiment.id"
                :to="`/experiments/${run.experiment.id}`"
                class="agent-run-row"
                :class="`agent-run-row-${run.progress.status}`"
                :aria-label="
                  t('agent.openExperimentRun', {
                    name: displayExperimentName(run.experiment),
                    status: experimentRunStatusLabel(run.progress),
                    progress: experimentRunProgressLabel(run.progress),
                  })
                "
              >
                <span class="agent-run-row-head">
                  <strong>{{ displayExperimentName(run.experiment) }}</strong>
                  <span class="agent-run-status">{{ experimentRunStatusLabel(run.progress) }}</span>
                </span>
                <span class="agent-run-row-detail">
                  <span>{{ experimentRunProgressLabel(run.progress) }}</span>
                  <ChevronRight aria-hidden="true" />
                </span>
                <span
                  class="agent-run-progress-track"
                  :role="
                    ['determinate', 'indeterminate'].includes(run.progress.mode)
                      ? 'progressbar'
                      : undefined
                  "
                  :aria-label="
                    t('agent.runProgressAriaLabel', { name: displayExperimentName(run.experiment) })
                  "
                  :aria-valuemin="run.progress.mode === 'determinate' ? 0 : undefined"
                  :aria-valuemax="run.progress.mode === 'determinate' ? 100 : undefined"
                  :aria-valuenow="
                    run.progress.mode === 'determinate' ? run.progress.percent : undefined
                  "
                  :aria-valuetext="
                    ['determinate', 'indeterminate'].includes(run.progress.mode)
                      ? experimentRunProgressLabel(run.progress)
                      : undefined
                  "
                >
                  <span
                    class="agent-run-progress-fill"
                    :class="[
                      `agent-run-progress-fill-${run.progress.status}`,
                      {
                        'agent-run-progress-fill-indeterminate':
                          run.progress.mode === 'indeterminate',
                      },
                    ]"
                    :style="{ width: experimentRunProgressWidth(run.progress) }"
                  />
                </span>
              </RouterLink>
            </div>
            <div class="agent-inspector-actions">
              <button
                type="button"
                class="secondary-action compact-action"
                @click="addCreatedAsBaselines"
              >
                <Database aria-hidden="true" />{{ t('agent.useCreatedAsBaselines') }}
              </button>
              <button
                type="button"
                class="secondary-action compact-action"
                :disabled="refreshMutation.isPending.value || hasReview"
                @click="refreshMutation.mutate()"
              >
                <RefreshCw aria-hidden="true" />{{
                  refreshMutation.isPending.value ? t('agent.refreshing') : t('agent.refreshPlan')
                }}
              </button>
              <button
                v-if="reviewMutation.isPending.value"
                type="button"
                class="secondary-action compact-action agent-generation-stop"
                @click="stopGeneration('review')"
              >
                <Square aria-hidden="true" />{{ t('agent.stopGenerating') }}
              </button>
              <button
                v-else
                type="button"
                class="primary-action compact-action"
                :disabled="!canReview"
                @click="reviewPlan"
              >
                <ShieldCheck aria-hidden="true" />{{ t('agent.reviewPlan') }}
              </button>
            </div>
            <p
              v-if="reviewMutation.isPending.value"
              class="agent-generation-status"
              role="status"
              aria-live="polite"
            >
              <span class="agent-generation-pulse" /><span>{{
                generationStatusLabel('review')
              }}</span>
            </p>
            <p
              v-else-if="stoppedGeneration === 'review'"
              class="agent-generation-stopped"
              role="status"
            >
              {{ t('agent.reportGenerationStopped') }}
            </p>
            <p v-if="reviewErrorMessage" class="error-card text-sm" role="alert">
              {{ reviewErrorMessage }}
            </p>
          </section>

          <div v-if="canceledExperiments.length" class="section-card mt-5 text-sm">
            <p class="font-bold">
              {{ t('agent.canceledExperiments', { count: canceledExperiments.length }) }}
            </p>
          </div>

          <p v-if="reviewErrorMessage" class="error-card mt-5" role="alert">
            {{ reviewErrorMessage }}
          </p>

          <div v-if="hasReview" class="agent-review-overview">
            <p class="font-bold">{{ t('agent.reviewSummary') }}</p>
            <p class="muted-copy leading-6">{{ review.summary }}</p>
            <div v-if="bestExperimentName || bestExperimentReason">
              <p class="font-bold">{{ t('agent.bestExperiment') }}</p>
              <p v-if="bestExperimentName" class="muted-copy mt-1 leading-6">
                {{ bestExperimentName }}
              </p>
              <p v-if="bestExperimentReason" class="muted-copy mt-1 leading-6">
                {{ bestExperimentReason }}
              </p>
            </div>
            <div v-if="recommendations.length">
              <p class="font-bold">{{ t('agent.nextRecommendations') }}</p>
              <ul class="muted-copy mt-1 list-disc space-y-1 pl-5">
                <li v-for="item in recommendations" :key="item">{{ item }}</li>
              </ul>
            </div>
            <details class="agent-review-evidence">
              <summary>{{ t('agent.workspaceReviewDetails') }}</summary>
              <div v-if="metricEvidence.length">
                <p class="font-bold">{{ t('agent.reviewEvidence') }}</p>
                <ul class="muted-copy mt-1 list-disc space-y-1 pl-5">
                  <li v-for="item in metricEvidence" :key="item">{{ item }}</li>
                </ul>
              </div>
              <div v-if="analysisFindings.length">
                <p class="font-bold">{{ t('agent.analysisFindings') }}</p>
                <ul class="muted-copy mt-1 list-disc space-y-1 pl-5">
                  <li v-for="item in analysisFindings" :key="item">{{ item }}</li>
                </ul>
              </div>
              <div v-if="failedOrIncomplete.length">
                <p class="font-bold">{{ t('agent.failedOrIncomplete') }}</p>
                <ul class="muted-copy mt-1 list-disc space-y-1 pl-5">
                  <li
                    v-for="item in failedOrIncomplete"
                    :key="`${objectText(item, 'experiment_id')}-${objectText(item, 'reason')}`"
                  >
                    {{ objectText(item, 'experiment_id') || objectText(item, 'name') }}
                    <span v-if="objectText(item, 'status')">
                      · {{ statusLabel(objectText(item, 'status')) }}</span
                    >
                    <span v-if="objectText(item, 'reason')">
                      · {{ objectText(item, 'reason') }}</span
                    >
                  </li>
                </ul>
              </div>
              <div v-if="candidateList(review.risks).length">
                <p class="font-bold">{{ t('agent.remainingRisks') }}</p>
                <ul class="muted-copy mt-1 list-disc space-y-1 pl-5">
                  <li v-for="item in candidateList(review.risks)" :key="item">{{ item }}</li>
                </ul>
              </div>
            </details>
          </div>

          <div class="agent-candidate-list">
            <AgentCandidateCompare
              v-if="viewMode === 'compare'"
              :actions="result.actions"
              @apply="applyCandidate"
              @edit="openCandidateEditor"
              @inspect="focusCandidate"
            />
            <article
              v-for="candidate in result.actions"
              v-else
              :key="candidate.id"
              class="agent-candidate-card grid gap-4"
              :class="[
                `agent-card-lane-${candidate.lane}`,
                { 'agent-candidate-card-active': activeCandidate?.id === candidate.id },
              ]"
            >
              <div class="flex flex-wrap items-start justify-between gap-3">
                <div class="min-w-0">
                  <div class="flex flex-wrap items-center gap-2">
                    <span class="status-pill" :class="laneClass(candidate)">{{
                      laneLabel(candidate)
                    }}</span>
                    <span v-if="candidate.type === 'cancel_experiment'" class="config-chip">{{
                      t('agent.actionTypes.cancel_experiment')
                    }}</span>
                    <span v-if="candidate.status !== 'pending'" class="config-chip">{{
                      actionStatusLabel(candidate)
                    }}</span>
                    <span
                      class="status-pill"
                      :class="candidate.valid ? 'status-pill-success' : 'status-pill-danger'"
                    >
                      {{ candidate.valid ? t('agent.valid') : t('agent.invalid') }}
                    </span>
                    <span class="status-pill" :class="riskClass(candidate)">
                      {{ t('agent.riskLabel') }}: {{ riskLabel(candidate) }}
                    </span>
                  </div>
                  <h3 class="mt-3 text-xl font-black">{{ candidate.title }}</h3>
                  <p v-if="candidate.rationale" class="muted-copy mt-2 text-sm leading-6">
                    {{ candidate.rationale }}
                  </p>
                </div>
                <div class="flex flex-wrap items-center justify-end gap-2">
                  <button
                    v-if="candidate.type === 'create_experiment'"
                    type="button"
                    class="primary-action compact-action"
                    :disabled="!candidate.valid || candidate.status !== 'pending'"
                    @click="applyCandidate(candidate)"
                  >
                    <ChevronRight aria-hidden="true" />
                    {{ t('agent.reviewInWizard') }}
                  </button>
                  <button
                    v-if="candidate.type === 'create_experiment' && candidate.status === 'pending'"
                    type="button"
                    class="secondary-action compact-action"
                    @click="openCandidateEditor(candidate, $event)"
                  >
                    <Pencil aria-hidden="true" />
                    {{ candidate.edited ? t('agent.editAgain') : t('agent.editConfig') }}
                  </button>
                </div>
              </div>

              <div
                v-if="candidate.preflight?.errors?.length"
                class="agent-candidate-errors error-card"
                role="alert"
              >
                <strong>{{ t('agent.errors') }}</strong>
                <ul>
                  <li
                    v-for="error in preflightMessages(candidate.preflight, 'errors')"
                    :key="error"
                  >
                    {{ error }}
                  </li>
                </ul>
              </div>
              <div class="agent-candidate-footer">
                <span v-if="candidate.preflight?.warnings?.length" class="agent-candidate-warning"
                  ><AlertTriangle aria-hidden="true" />{{
                    t('agent.workspaceWarningCount', { count: candidate.preflight.warnings.length })
                  }}</span
                ><button
                  type="button"
                  class="agent-details-trigger"
                  @click="focusCandidate(candidate)"
                >
                  {{ t('agent.workspaceViewDetails') }}<ChevronRight aria-hidden="true" />
                </button>
              </div>
            </article>
          </div>
        </div>

        <div
          v-else
          class="agent-objective-canvas"
          :class="{ 'agent-objective-canvas-busy': designMutation.isPending.value }"
        >
          <div class="agent-objective-intro">
            <span class="agent-objective-mark"><FlaskConical aria-hidden="true" /></span>
            <div>
              <p class="agent-rail-kicker">
                {{ isReuseMode ? t('agent.workspaceReusePlan') : t('agent.eyebrow') }}
              </p>
              <h2>{{ t('agent.workspaceObjectiveTitle') }}</h2>
              <p class="agent-objective-subtitle">{{ t('agent.workspaceObjectiveHint') }}</p>
            </div>
          </div>

          <section
            v-if="diagnosticSourceRequested"
            class="agent-diagnostic-source"
            :class="{ 'agent-diagnostic-source-error': diagnosticSourceError }"
          >
            <div>
              <FileText aria-hidden="true" />
              <span>
                <strong>{{ t('agent.diagnosticSourceTitle') }}</strong>
                <small v-if="diagnosticSourceReady">
                  {{
                    diagnosticExperimentQuery.data.value
                      ? displayExperimentName(diagnosticExperimentQuery.data.value)
                      : ''
                  }}
                  <template v-if="diagnosticSourceIsStale">
                    · {{ t('agent.diagnosticSourceStale') }}</template
                  >
                </small>
                <small v-else>{{
                  diagnosticSourceError || t('agent.diagnosticSourceLoading')
                }}</small>
              </span>
            </div>
            <button
              type="button"
              :title="t('agent.diagnosticSourceDetach')"
              :aria-label="t('agent.diagnosticSourceDetach')"
              @click="detachDiagnosticSource"
            >
              <X aria-hidden="true" />
            </button>
            <p v-if="diagnosticSourceReady">{{ diagnosticSourceReport?.summary }}</p>
          </section>

          <div class="agent-input-selectors">
            <label class="form-field">
              <span>{{ t('agent.task') }}</span>
              <select v-model="taskType" class="field-control">
                <option v-for="task in taskOptions" :key="task.id" :value="task.id">
                  {{ taskLabel(task.id) }}
                </option>
              </select>
            </label>
            <label class="form-field">
              <span>{{ t('agent.dataset') }}</span>
              <select v-model="datasetId" class="field-control">
                <option value="">{{ t('agent.selectDataset') }}</option>
                <option
                  v-for="dataset in matchingDatasets"
                  :key="dataset.id"
                  :value="dataset.id"
                  :title="dataset.name"
                >
                  {{ dataset.name }}
                </option>
              </select>
            </label>
          </div>
          <p v-if="isLoadingContext" class="agent-input-status" role="status">
            {{ t('agent.loading') }}
          </p>
          <p
            v-else-if="
              providersQuery.isError.value ||
              datasetsQuery.isError.value ||
              capabilitiesQuery.isError.value
            "
            class="error-card agent-input-status"
            role="alert"
          >
            {{ t('agent.workspaceLoadError') }}
            <button type="button" class="tertiary-action compact-action" @click="retryContext">
              {{ t('home.retry') }}
            </button>
          </p>
          <p v-else-if="!isGuest && !enabledProviders.length" class="agent-input-notice">
            <PlugZap aria-hidden="true" /><span>{{ t('agent.noProviders') }}</span
            ><RouterLink to="/account/ai-services">{{ t('aiServices.connect') }}</RouterLink>
          </p>
          <p v-else-if="!isGuest && !matchingDatasets.length" class="agent-input-notice">
            <Database aria-hidden="true" /><span>{{ t('agent.workspaceNoDataset') }}</span
            ><RouterLink to="/datasets">{{ t('nav.datasets') }}</RouterLink>
          </p>
          <label class="agent-objective-field">
            <span class="sr-only">{{ t('agent.objective') }}</span>
            <textarea
              v-model="objective"
              :placeholder="t('agent.objectivePlaceholder')"
              :disabled="designMutation.isPending.value"
            />
          </label>

          <fieldset class="agent-strategy-field" :disabled="designMutation.isPending.value">
            <legend>{{ t('agent.strategyPreference') }}</legend>
            <div class="agent-strategy-control">
              <label v-for="option in strategyOptions" :key="option">
                <input
                  v-model="strategyPreference"
                  type="radio"
                  name="agent-strategy"
                  :value="option"
                />
                <span>{{ t(`agent.strategy.${option}`) }}</span>
              </label>
            </div>
          </fieldset>

          <div class="agent-objective-actions">
            <button
              type="button"
              class="agent-context-summary-trigger"
              :aria-expanded="contextRailOpen"
              aria-controls="agent-context-rail"
              @click="togglePanel('context')"
            >
              <SlidersHorizontal aria-hidden="true" />{{ t('agent.workspaceContextShort')
              }}<span v-if="baselineExperimentIds.length"
                >·
                {{
                  t('agent.workspaceBaselineCount', { count: baselineExperimentIds.length })
                }}</span
              ><span v-if="includeDatasetPreview">· {{ t('agent.includeDatasetPreview') }}</span>
            </button>
            <button
              type="button"
              class="inline-flex items-center gap-2"
              :class="
                designMutation.isPending.value
                  ? 'secondary-action agent-generation-stop'
                  : 'primary-action'
              "
              :disabled="!isGuest && (designMutation.isPending.value ? false : !canGenerate)"
              @click="designMutation.isPending.value ? stopGeneration('plan') : generateDesigns()"
            >
              <Square v-if="designMutation.isPending.value" class="h-4 w-4" aria-hidden="true" />
              <Sparkles v-else class="h-4 w-4" aria-hidden="true" />
              {{ designMutation.isPending.value ? t('agent.stopGenerating') : t('agent.generate') }}
            </button>
          </div>

          <div
            v-if="designMutation.isPending.value"
            class="agent-generation-status"
            role="status"
            aria-live="polite"
          >
            <span class="agent-generation-pulse" />
            <span>{{ generationStatusLabel('plan') }}</span>
          </div>
          <p
            v-else-if="stoppedGeneration === 'plan'"
            class="agent-generation-stopped"
            role="status"
          >
            {{ t('agent.planGenerationStopped') }}
          </p>
          <p v-if="errorMessage" class="error-card" role="alert">{{ errorMessage }}</p>
        </div>

        <div v-if="result?.warnings?.length" class="grid gap-2">
          <p v-for="warning in result.warnings" :key="warning" class="error-card text-sm">
            {{ warning }}
          </p>
        </div>
      </section>

      <WorkspacePanel
        v-if="detailRailOpen"
        id="agent-detail-rail"
        class="agent-detail-rail"
        side="right"
        :label="t('agent.workspaceCandidateDetail')"
        @close="activePanel = null"
      >
        <div class="agent-rail-heading">
          <div>
            <p class="agent-rail-kicker">{{ t('agent.workspaceStageLabel') }}</p>
            <h2>{{ t(`agent.workspaceStage.${workspaceStage}`) }}</h2>
          </div>
          <button
            type="button"
            class="agent-rail-close"
            :title="t('agent.workspaceCloseInspector')"
            :aria-label="t('agent.workspaceCloseInspector')"
            @click="activePanel = null"
          >
            <X aria-hidden="true" />
          </button>
        </div>

        <div ref="detailRailScroll" class="agent-rail-scroll">
          <template v-if="result">
            <section
              v-if="activeCandidate"
              class="agent-inspector-section agent-inspector-candidate"
            >
              <div class="agent-inspector-title">
                <FileText aria-hidden="true" />
                <h3>{{ t('agent.workspaceCandidateDetail') }}</h3>
              </div>
              <div class="agent-candidate-inspector-head">
                <div class="flex flex-wrap gap-2">
                  <span class="status-pill" :class="laneClass(activeCandidate)">{{
                    laneLabel(activeCandidate)
                  }}</span>
                  <span class="status-pill" :class="riskClass(activeCandidate)"
                    >{{ t('agent.riskLabel') }}: {{ riskLabel(activeCandidate) }}</span
                  >
                </div>
                <h4>{{ activeCandidate.title }}</h4>
                <p>{{ activeCandidate.rationale }}</p>
              </div>

              <div class="agent-inspector-actions">
                <button
                  v-if="activeCandidate.type === 'create_experiment'"
                  type="button"
                  class="primary-action compact-action"
                  :disabled="!activeCandidate.valid || activeCandidate.status !== 'pending'"
                  @click="applyCandidate(activeCandidate)"
                >
                  <ChevronRight aria-hidden="true" />{{ t('agent.reviewInWizard') }}
                </button>
                <button
                  v-if="
                    activeCandidate.type === 'create_experiment' &&
                    activeCandidate.status === 'pending'
                  "
                  type="button"
                  class="secondary-action compact-action"
                  @click="openCandidateEditor(activeCandidate, $event)"
                >
                  <Pencil aria-hidden="true" />{{
                    activeCandidate.edited ? t('agent.editAgain') : t('agent.editConfig')
                  }}
                </button>
              </div>

              <div v-if="editingActionId === activeCandidate.id" class="agent-config-editor">
                <textarea
                  ref="configEditorTextarea"
                  v-model="editingConfigText"
                  :aria-label="t('agent.editConfig')"
                  class="field-control font-mono text-xs"
                  spellcheck="false"
                />
                <div class="agent-config-editor-actions">
                  <button
                    type="button"
                    class="agent-toolbar-icon"
                    :title="t('agent.restoreOriginal')"
                    @click="restoreOriginalCandidate(activeCandidate)"
                  >
                    <RotateCcw aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    class="tertiary-action compact-action"
                    @click="cancelEditingCandidate"
                  >
                    <X aria-hidden="true" />{{ t('common.cancel') }}
                  </button>
                  <button
                    type="button"
                    class="primary-action compact-action"
                    :disabled="actionEditMutation.isPending.value"
                    @click="saveEditedCandidate(activeCandidate)"
                  >
                    <Check aria-hidden="true" />
                    {{
                      actionEditMutation.isPending.value
                        ? t('common.saving')
                        : t('agent.saveAndPreflight')
                    }}
                  </button>
                </div>
                <p v-if="editErrorMessage" class="error-card text-sm" role="alert">
                  {{ editErrorMessage }}
                </p>
              </div>

              <div v-if="activeCandidate.objective_alignment" class="agent-inspector-copy">
                <strong>{{ t('agent.objectiveAlignment') }}</strong>
                <p>{{ activeCandidate.objective_alignment }}</p>
              </div>
              <div
                v-if="candidateList(activeCandidate.baseline_changes).length"
                class="agent-inspector-copy"
              >
                <strong>{{ t('agent.baselineChanges') }}</strong>
                <ul>
                  <li v-for="item in candidateList(activeCandidate.baseline_changes)" :key="item">
                    {{ item }}
                  </li>
                </ul>
              </div>
              <div
                v-if="candidateList(activeCandidate.verification_steps).length"
                class="agent-inspector-copy"
              >
                <strong>{{ t('agent.verificationSteps') }}</strong>
                <ul>
                  <li v-for="item in candidateList(activeCandidate.verification_steps)" :key="item">
                    {{ item }}
                  </li>
                </ul>
              </div>
              <div v-if="activeCandidate.preflight?.errors?.length" class="error-card text-sm">
                <strong>{{ t('agent.errors') }}</strong>
                <ul class="mt-2 list-disc space-y-1 pl-5">
                  <li
                    v-for="item in preflightMessages(activeCandidate.preflight, 'errors')"
                    :key="item"
                  >
                    {{ item }}
                  </li>
                </ul>
              </div>
              <div
                v-if="candidateList(activeCandidate.expected_tradeoffs).length"
                class="agent-inspector-copy"
              >
                <strong>{{ t('agent.expectedTradeoffs') }}</strong>
                <ul>
                  <li v-for="item in candidateList(activeCandidate.expected_tradeoffs)" :key="item">
                    {{ item }}
                  </li>
                </ul>
              </div>
              <div v-if="activeCandidate.preflight?.warnings?.length" class="agent-inspector-copy">
                <strong>{{ t('agent.warnings') }}</strong>
                <ul>
                  <li
                    v-for="item in preflightMessages(activeCandidate.preflight, 'warnings')"
                    :key="item"
                  >
                    {{ item }}
                  </li>
                </ul>
              </div>
              <details v-if="hasConfig(activeCandidate)" class="agent-config-details">
                <summary>{{ t('agent.config') }}</summary>
                <pre>{{ configText(activeCandidate) }}</pre>
              </details>
            </section>

            <section class="agent-inspector-section">
              <div class="agent-inspector-title">
                <Database aria-hidden="true" />
                <h3>{{ t('agent.contextSent') }}</h3>
              </div>
              <dl class="agent-context-summary-list">
                <div>
                  <dt>{{ t('agent.provider') }}</dt>
                  <dd>{{ result.model }}</dd>
                </div>
                <div>
                  <dt>{{ t('agent.dataset') }}</dt>
                  <dd>{{ selectedDataset?.name || result.dataset_id }}</dd>
                </div>
                <div v-if="contextSummary">
                  <dt>{{ t('agent.baselineExperiments') }}</dt>
                  <dd>{{ contextSummary.baseline_experiment_count }}</dd>
                </div>
              </dl>
              <p v-if="contextSummary" class="muted-copy mt-2 text-xs">
                {{
                  t('agent.contextSummary', {
                    recent: contextSummary.recent_experiment_count,
                    baseline: contextSummary.baseline_experiment_count,
                    leaderboard: contextSummary.leaderboard_count,
                  })
                }}
              </p>
              <p v-if="usageLabel" class="muted-copy text-xs">{{ usageLabel }}</p>
              <div v-if="manifestSections.length" class="agent-manifest-list">
                <div v-for="section in manifestSections" :key="section.id">
                  <span>{{ section.id }}</span>
                  <small>{{ section.sent ? section.detail : t('agent.notSent') }}</small>
                </div>
              </div>
            </section>
          </template>
        </div>
      </WorkspacePanel>
    </div>
  </section>
</template>
