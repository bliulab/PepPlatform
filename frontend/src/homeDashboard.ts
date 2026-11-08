import { normalizeTaskType } from './taskTerminology'
import { TASK_ENTRIES, type TaskKey } from './taskCatalog'
import type { ExperimentRecord, ModelRecord } from './types'

/** Window of the home activity strip, in local calendar days (today included). */
export const ACTIVITY_DAYS = 14

export interface HomeActivityDay {
  /** Local calendar day, `YYYY-MM-DD`. */
  date: string
  count: number
}

export interface HomeTaskFamily {
  /** Task key shared with `tasks.*` i18n entries. */
  key: TaskKey
  /** Station id (`T01`–`T04`) used by the landing scale band. */
  id: string
  runs: number
  readyModels: number
}

export interface HomeDashboardData {
  recentExperiments: ExperimentRecord[]
  recentModels: ModelRecord[]
  latestCompleted?: ExperimentRecord
  experimentCount: number
  completedCount: number
  /** `completedCount / experimentCount`, zero when nothing has run yet. */
  completionRate: number
  reusableModelCount: number
  metricsOnlyModelCount: number
  /** Always four entries, in `T01`–`T04` order. */
  taskFamilies: HomeTaskFamily[]
  /** Families with at least one run. */
  familiesStarted: number
  /** Fourteen local calendar days, ascending, ending today. */
  activity: HomeActivityDay[]
  activityTotal: number
  activityPeak: number
}

export interface HomeDashboardOptions {
  /** Injectable clock, so tests and screenshots agree on "today". */
  now?: Date
}

function timestamp(value?: string | null) {
  if (!value) return 0
  const parsed = new Date(value).getTime()
  return Number.isNaN(parsed) ? 0 : parsed
}

function experimentTimestamp(experiment: ExperimentRecord) {
  return (
    timestamp(experiment.finished_at) ||
    timestamp(experiment.updated_at) ||
    timestamp(experiment.created_at)
  )
}

function localDayKey(value: Date) {
  const month = String(value.getMonth() + 1).padStart(2, '0')
  const day = String(value.getDate()).padStart(2, '0')
  return `${value.getFullYear()}-${month}-${day}`
}

function startOfLocalDay(value: Date) {
  return new Date(value.getFullYear(), value.getMonth(), value.getDate())
}

/**
 * Runs created per local calendar day. Only `created_at` is used: the strip
 * describes new activity, not how long a run stayed open.
 */
function buildActivity(experiments: ExperimentRecord[], now: Date): HomeActivityDay[] {
  const today = startOfLocalDay(now)
  const days: HomeActivityDay[] = []
  const indexByDay = new Map<string, number>()
  for (let offset = ACTIVITY_DAYS - 1; offset >= 0; offset -= 1) {
    const day = new Date(today)
    day.setDate(today.getDate() - offset)
    const date = localDayKey(day)
    indexByDay.set(date, days.length)
    days.push({ date, count: 0 })
  }
  for (const experiment of experiments) {
    if (!experiment.created_at) continue
    const parsed = new Date(experiment.created_at)
    if (Number.isNaN(parsed.getTime())) continue
    const index = indexByDay.get(localDayKey(parsed))
    if (index === undefined) continue
    days[index].count += 1
  }
  return days
}

/** Normalised bar height in `[0, 1]`; an empty window is a flat baseline. */
export function barRatio(count: number, peak: number): number {
  if (!Number.isFinite(count) || !Number.isFinite(peak) || peak <= 0) return 0
  return Math.min(1, Math.max(0, count / peak))
}

export function buildHomeDashboard(
  experiments: ExperimentRecord[] = [],
  models: ModelRecord[] = [],
  options: HomeDashboardOptions = {},
): HomeDashboardData {
  const now = options.now ?? new Date()
  const sortedExperiments = [...experiments].sort(
    (left, right) => experimentTimestamp(right) - experimentTimestamp(left),
  )
  const sortedModels = [...models].sort(
    (left, right) => timestamp(right.created_at) - timestamp(left.created_at),
  )
  const completed = sortedExperiments.filter((experiment) => experiment.status === 'succeeded')
  const reusableModelCount = models.filter((model) => model.supports_inference).length

  const runsByTask = new Map<string, number>()
  for (const experiment of experiments) {
    runsByTask.set(normalizeTaskType(experiment.task_type), (runsByTask.get(normalizeTaskType(experiment.task_type)) ?? 0) + 1)
  }
  const readyByTask = new Map<string, number>()
  for (const model of models) {
    if (!model.supports_inference) continue
    readyByTask.set(normalizeTaskType(model.task_type), (readyByTask.get(normalizeTaskType(model.task_type)) ?? 0) + 1)
  }
  const taskFamilies: HomeTaskFamily[] = TASK_ENTRIES.map((entry) => ({
    key: entry.key,
    id: entry.id,
    runs: runsByTask.get(entry.key) ?? 0,
    readyModels: readyByTask.get(entry.key) ?? 0,
  }))

  const activity = buildActivity(experiments, now)

  return {
    recentExperiments: sortedExperiments.slice(0, 4),
    recentModels: sortedModels.slice(0, 3),
    latestCompleted: completed[0],
    experimentCount: experiments.length,
    completedCount: completed.length,
    completionRate: experiments.length ? completed.length / experiments.length : 0,
    reusableModelCount,
    metricsOnlyModelCount: models.length - reusableModelCount,
    taskFamilies,
    familiesStarted: taskFamilies.filter((family) => family.runs > 0).length,
    activity,
    activityTotal: activity.reduce((total, day) => total + day.count, 0),
    activityPeak: activity.reduce((peak, day) => Math.max(peak, day.count), 0),
  }
}
