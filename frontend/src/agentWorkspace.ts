import { usesEpochTraining } from './experimentRoutes'
import type {
  AgentExecutionAction,
  AgentExecutionPlan,
  ExperimentRecord,
  MetricEvent,
} from './types'

export const agentWorkspaceStages = ['objective', 'approval', 'execution', 'review'] as const

export type AgentWorkspaceStage = (typeof agentWorkspaceStages)[number]

type PlanStageInput = Pick<AgentExecutionPlan, 'status' | 'review'>

export type ExperimentRunStatus =
  'queued' | 'running' | 'succeeded' | 'failed' | 'canceled' | 'missing'
export type ExperimentRunProgressMode =
  'idle' | 'determinate' | 'indeterminate' | 'terminal' | 'unavailable'
export type StructuredRunStage =
  'representation' | 'embedding_export' | 'classifier_fit' | 'finalization'

export interface ExperimentRunProgress {
  status: ExperimentRunStatus
  mode: ExperimentRunProgressMode
  usesEpochs: boolean
  current?: number
  total?: number
  percent?: number
  earlyStopped: boolean
  preparation?: 'profile_database' | 'profile_generation' | 'hyperparameter_tuning'
  preparationStatus?: 'running' | 'complete'
  preparationFeature?: string
  preparationElapsedSeconds?: number
  preparationRatePerSecond?: number
  preparationEtaSeconds?: number
  runStage?: StructuredRunStage
  runStageStatus?: 'running' | 'complete' | 'failed'
  runStageFeature?: string
  runStageUnit?: string
  runStageSplit?: string
  runStageOutput?: number
  runStageOutputs?: number
  runStageIteration?: number
  runStageIterations?: number
  runStageThreads?: number
}

type ExperimentProgressInput = Pick<ExperimentRecord, 'status'> &
  Partial<Pick<ExperimentRecord, 'config' | 'metrics'>>

function finitePositiveInteger(value: unknown) {
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : undefined
}

function finiteNonNegativeInteger(value: unknown) {
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed >= 0 ? Math.floor(parsed) : undefined
}

function finiteNonNegativeNumber(value: unknown) {
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : undefined
}

export function resolveExperimentProgressTotal(
  progress: Pick<ExperimentRunProgress, 'preparation' | 'total'>,
  config: Record<string, unknown>,
  epochTotal?: unknown,
) {
  const parsedEpochTotal = finitePositiveInteger(epochTotal)
  if (parsedEpochTotal !== undefined) return parsedEpochTotal
  if (progress.preparation) return progress.total ?? 0
  return finitePositiveInteger(config.epochs) ?? progress.total ?? 0
}

function normalizeExperimentStatus(status: string): ExperimentRunStatus {
  if (['queued', 'running', 'succeeded', 'failed', 'canceled'].includes(status)) {
    return status as ExperimentRunStatus
  }
  return 'missing'
}

function latestEpochEvent(events: MetricEvent[]) {
  return [...events]
    .reverse()
    .find((event) => event.kind === 'epoch' && finitePositiveInteger(event.epoch) !== undefined)
}

function latestPreparationStageEvent(events: MetricEvent[]) {
  return [...events]
    .reverse()
    .find(
      (event) =>
        event.kind === 'stage' &&
        ['database', 'profiles', 'hyperparameter_tuning'].includes(String(event.stage ?? '')),
    )
}

const structuredRunStages = ['representation', 'embedding_export', 'classifier_fit', 'finalization']

function latestStructuredRunStageEvent(events: MetricEvent[]) {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]
    if (event.kind === 'stage' && structuredRunStages.includes(String(event.stage ?? ''))) {
      return { event, index }
    }
  }
  return undefined
}

type ProfilePreparationProgress = Pick<
  ExperimentRunProgress,
  | 'mode'
  | 'current'
  | 'total'
  | 'percent'
  | 'preparation'
  | 'preparationStatus'
  | 'preparationFeature'
  | 'preparationElapsedSeconds'
  | 'preparationRatePerSecond'
  | 'preparationEtaSeconds'
>

function deriveProfilePreparation(event: MetricEvent): ProfilePreparationProgress {
  const stage = String(event.stage)
  const preparation =
    stage === 'database'
      ? 'profile_database'
      : stage === 'profiles'
        ? 'profile_generation'
        : 'hyperparameter_tuning'
  const preparationStatus = String(event.status) === 'complete' ? 'complete' : 'running'
  const current = finiteNonNegativeInteger(event.current)
  const total = finitePositiveInteger(event.total)
  const preparationElapsedSeconds = finiteNonNegativeNumber(event.elapsed_seconds)
  const percent =
    preparation !== 'profile_database' && current !== undefined && total
      ? Math.min(100, Math.max(0, Math.round((current / total) * 100)))
      : undefined
  const preparationRatePerSecond =
    preparation === 'profile_generation' &&
    current !== undefined &&
    current >= 5 &&
    preparationElapsedSeconds !== undefined &&
    preparationElapsedSeconds >= 10
      ? current / preparationElapsedSeconds
      : undefined
  const preparationEtaSeconds =
    preparationRatePerSecond && total && current !== undefined
      ? Math.max(0, (total - current) / preparationRatePerSecond)
      : undefined
  return {
    mode: percent === undefined ? 'indeterminate' : 'determinate',
    current,
    total,
    percent,
    preparation,
    preparationStatus,
    preparationFeature: String(event.feature ?? ''),
    preparationElapsedSeconds,
    preparationRatePerSecond,
    preparationEtaSeconds,
  }
}

function deriveStructuredRunProgress(event: MetricEvent) {
  const current = finiteNonNegativeInteger(event.current)
  const total = finitePositiveInteger(event.total)
  const percent =
    current !== undefined && total
      ? Math.min(100, Math.max(0, Math.round((current / total) * 100)))
      : undefined
  const status = ['running', 'complete', 'failed'].includes(String(event.status ?? ''))
    ? (String(event.status) as 'running' | 'complete' | 'failed')
    : 'running'
  return {
    mode: percent === undefined ? ('indeterminate' as const) : ('determinate' as const),
    current,
    total,
    percent,
    runStage: String(event.stage) as StructuredRunStage,
    runStageStatus: status,
    runStageFeature: String(event.feature ?? ''),
    runStageUnit: String(event.unit ?? ''),
    runStageSplit: String(event.split ?? ''),
    runStageOutput: finiteNonNegativeInteger(event.output),
    runStageOutputs: finitePositiveInteger(event.outputs),
    runStageIteration: finiteNonNegativeInteger(event.iteration),
    runStageIterations: finitePositiveInteger(event.iterations),
    runStageThreads: finitePositiveInteger(event.threads),
    preparationElapsedSeconds: finiteNonNegativeNumber(event.elapsed_seconds),
    preparationEtaSeconds: finiteNonNegativeNumber(event.eta_seconds),
  }
}

export function deriveExperimentRunProgress(
  experiment?: ExperimentProgressInput | null,
): ExperimentRunProgress {
  const status = normalizeExperimentStatus(String(experiment?.status ?? 'missing'))
  const config = experiment?.config ?? {}
  const events = experiment?.metrics?.events ?? []
  const epochEvent = latestEpochEvent(events)
  const epochEventIndex = epochEvent ? events.lastIndexOf(epochEvent) : -1
  const structuredStageMatch = latestStructuredRunStageEvent(events)
  const structuredProgress =
    structuredStageMatch && structuredStageMatch.index >= epochEventIndex
      ? deriveStructuredRunProgress(structuredStageMatch.event)
      : undefined
  const preparationStageEvent = latestPreparationStageEvent(events)
  const usesEpochs = Boolean(epochEvent) || usesEpochTraining(config)
  const current = finitePositiveInteger(epochEvent?.epoch)
  const total = finitePositiveInteger(epochEvent?.total) ?? finitePositiveInteger(config.epochs)
  const percent =
    current && total ? Math.min(100, Math.max(0, Math.round((current / total) * 100))) : undefined
  const earlyStopPatience = finitePositiveInteger(config.early_stop_patience)
  const earlyStopped = Boolean(
    status === 'succeeded' &&
    usesEpochs &&
    earlyStopPatience &&
    current &&
    total &&
    current < total,
  )
  const preparationProgress =
    !epochEvent && preparationStageEvent
      ? deriveProfilePreparation(preparationStageEvent)
      : undefined

  if (status === 'queued') {
    return { status, mode: 'idle', usesEpochs, current, total, percent, earlyStopped }
  }
  if (status === 'running') {
    if (structuredProgress) {
      return {
        status,
        usesEpochs,
        earlyStopped,
        ...structuredProgress,
      }
    }
    if (preparationProgress) {
      return {
        status,
        usesEpochs,
        earlyStopped,
        ...preparationProgress,
      }
    }
    return {
      status,
      mode: percent === undefined ? 'indeterminate' : 'determinate',
      usesEpochs,
      current,
      total,
      percent,
      earlyStopped,
    }
  }
  if (['succeeded', 'failed', 'canceled'].includes(status)) {
    if (structuredProgress) {
      return {
        status,
        usesEpochs,
        earlyStopped,
        ...structuredProgress,
        mode: 'terminal',
      }
    }
    if (status !== 'succeeded' && preparationProgress) {
      return {
        status,
        usesEpochs,
        earlyStopped,
        ...preparationProgress,
        mode: 'terminal',
      }
    }
    return { status, mode: 'terminal', usesEpochs, current, total, percent, earlyStopped }
  }
  return { status, mode: 'unavailable', usesEpochs, current, total, percent, earlyStopped }
}

export function deriveAgentWorkspaceStage(plan?: PlanStageInput | null): AgentWorkspaceStage {
  if (!plan) return 'objective'
  if (plan.review?.summary || ['reported', 'reviewed'].includes(plan.status)) return 'review'
  if (['executed', 'running', 'ready_for_report'].includes(plan.status)) return 'execution'
  return 'approval'
}

export function agentWorkspaceStageIndex(stage: AgentWorkspaceStage) {
  return agentWorkspaceStages.indexOf(stage)
}

export function reconcileSelectedActionIds(
  selectedIds: string[],
  actions: Pick<AgentExecutionAction, 'id' | 'type' | 'valid' | 'status'>[],
  maxSelectable = Number.POSITIVE_INFINITY,
) {
  const selectableIds = new Set(
    actions
      .filter(
        (action) =>
          action.type === 'create_experiment' && action.valid && action.status === 'pending',
      )
      .map((action) => action.id),
  )
  return selectedIds.filter((id) => selectableIds.has(id)).slice(0, Math.max(0, maxSelectable))
}
