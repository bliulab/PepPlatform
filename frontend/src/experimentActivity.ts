import type { ExperimentRecord } from './types'

export const EXPERIMENT_ACTIVITY_DAYS = 365
export interface ActivityDay {
  date: string
  count: number
  level: number
}
export interface ActivityMonth {
  date: string
  column: number
}
export interface ExperimentActivity {
  days: ActivityDay[]
  cells: (ActivityDay | null)[]
  months: ActivityMonth[]
  weeks: number
  total: number
  activeDays: number
  leading: number
}

export function activityLevel(count: number) {
  if (count <= 0) return 0
  if (count === 1) return 1
  if (count <= 3) return 2
  if (count <= 6) return 3
  return 4
}

function dayKey(date: Date) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`
}

/** Existing records only; status/updates never move a contribution's creation day. */
export function buildExperimentActivity(
  experiments: ExperimentRecord[] = [],
  options: { now?: Date; ownerId?: string | null } = {},
): ExperimentActivity {
  const now = options.now ?? new Date()
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  start.setDate(start.getDate() - EXPERIMENT_ACTIVITY_DAYS + 1)
  const leading = start.getDay()
  const days: ActivityDay[] = []
  const byDate = new Map<string, ActivityDay>()
  const months: ActivityMonth[] = []
  for (let offset = 0; offset < EXPERIMENT_ACTIVITY_DAYS; offset++) {
    const date = new Date(start)
    date.setDate(start.getDate() + offset)
    const day = { date: dayKey(date), count: 0, level: 0 }
    days.push(day)
    byDate.set(day.date, day)
    if (date.getDate() === 1)
      months.push({ date: day.date, column: Math.floor((leading + offset) / 7) + 1 })
  }
  for (const experiment of experiments) {
    if (options.ownerId && experiment.owner_user_id !== options.ownerId) continue
    if (!experiment.created_at) continue
    const created = new Date(experiment.created_at)
    if (!Number.isFinite(created.getTime()) || created.getTime() > now.getTime()) continue
    const day = byDate.get(dayKey(created))
    if (day) day.count++
  }
  for (const day of days) day.level = activityLevel(day.count)
  const weeks = Math.ceil((leading + days.length) / 7)
  const cells: (ActivityDay | null)[] = Array(weeks * 7).fill(null)
  days.forEach((day, index) => {
    cells[leading + index] = day
  })
  return {
    days,
    cells,
    months,
    weeks,
    leading,
    total: days.reduce((sum, day) => sum + day.count, 0),
    activeDays: days.filter((day) => day.count > 0).length,
  }
}
