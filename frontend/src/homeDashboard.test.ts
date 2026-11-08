import { describe, expect, it } from 'vitest'
import { ACTIVITY_DAYS, barRatio, buildHomeDashboard } from './homeDashboard'
import type { ExperimentRecord, ModelRecord } from './types'

function experiment(
  id: string,
  status: string,
  options: { finishedAt?: string; createdAt?: string; taskType?: string } = {},
): ExperimentRecord {
  const record: ExperimentRecord = {
    id,
    dataset_id: 'dataset',
    name: id,
    task_type: options.taskType ?? 'pep_function',
    config: {},
    status,
    log_path: '',
    artifact_dir: '',
    metrics: { events: [], summary: {} },
  }
  if (options.createdAt) record.created_at = options.createdAt
  if (options.finishedAt) record.finished_at = options.finishedAt
  return record
}

function model(
  id: string,
  ready: boolean,
  createdAt: string,
  taskType = 'pep_function',
): ModelRecord {
  return {
    id,
    experiment_id: 'experiment',
    name: id,
    task_type: taskType,
    bundle_path: '',
    supports_inference: ready,
    metadata: {},
    created_at: createdAt,
  }
}

/** Local wall-clock instant, so the calendar-day buckets stay timezone independent. */
function localIso(year: number, month: number, day: number, hour = 12): string {
  return new Date(year, month - 1, day, hour, 0, 0).toISOString()
}

const NOW = new Date(2026, 4, 10, 18, 0, 0)

describe('home dashboard data', () => {
  it('sorts recent records and derives research-path counts', () => {
    const dashboard = buildHomeDashboard(
      [
        experiment('old', 'succeeded', {
          finishedAt: '2025-05-01T00:00:00Z',
          createdAt: '2025-05-01T00:00:00Z',
        }),
        experiment('running', 'running', { createdAt: localIso(2026, 5, 9) }),
        experiment('latest', 'succeeded', {
          finishedAt: '2026-06-01T00:00:00Z',
          createdAt: localIso(2026, 5, 8),
        }),
      ],
      [
        model('older-model', false, '2025-01-01T00:00:00Z'),
        model('ready-model', true, localIso(2026, 5, 7)),
      ],
      { now: NOW },
    )

    expect(dashboard.experimentCount).toBe(3)
    expect(dashboard.completedCount).toBe(2)
    expect(dashboard.completionRate).toBeCloseTo(2 / 3, 10)
    expect(dashboard.reusableModelCount).toBe(1)
    expect(dashboard.metricsOnlyModelCount).toBe(1)
    expect(dashboard.latestCompleted?.id).toBe('latest')
    expect(dashboard.recentExperiments.map((item) => item.id)).toEqual(['latest', 'running', 'old'])
    expect(dashboard.recentModels.map((item) => item.id)).toEqual(['ready-model', 'older-model'])
  })

  it('returns a stable empty state', () => {
    const dashboard = buildHomeDashboard([], [], { now: NOW })

    expect(dashboard.experimentCount).toBe(0)
    expect(dashboard.completedCount).toBe(0)
    expect(dashboard.completionRate).toBe(0)
    expect(dashboard.reusableModelCount).toBe(0)
    expect(dashboard.metricsOnlyModelCount).toBe(0)
    expect(dashboard.latestCompleted).toBeUndefined()
    expect(dashboard.familiesStarted).toBe(0)
    expect(dashboard.activityTotal).toBe(0)
    expect(dashboard.activityPeak).toBe(0)
    expect(dashboard.recentExperiments).toEqual([])
    expect(dashboard.recentModels).toEqual([])
    expect(dashboard.activity).toHaveLength(ACTIVITY_DAYS)
    expect(dashboard.taskFamilies).toEqual([
      { key: 'pep_function', id: 'T01', runs: 0, readyModels: 0 },
      { key: 'pep_protein_interaction', id: 'T02', runs: 0, readyModels: 0 },
      { key: 'pep_hlai', id: 'T03', runs: 0, readyModels: 0 },
      { key: 'pep_hlaii', id: 'T04', runs: 0, readyModels: 0 },
    ])
  })

  it('aggregates runs and ready models per task family in T01-T04 order', () => {
    const dashboard = buildHomeDashboard(
      [
        experiment('f1', 'succeeded', { taskType: 'pep_function' }),
        experiment('f2', 'running', { taskType: 'pep_function' }),
        experiment('h1', 'succeeded', { taskType: 'pep_hlaii' }),
        experiment('unknown', 'succeeded', { taskType: 'pep_something_new' }),
      ],
      [
        model('m1', true, localIso(2026, 5, 1), 'pep_function'),
        model('m2', false, localIso(2026, 5, 2), 'pep_function'),
        model('m3', true, localIso(2026, 5, 3), 'pep_hlai'),
        model('m4', true, localIso(2026, 5, 4), 'pep_something_new'),
      ],
      { now: NOW },
    )

    expect(dashboard.taskFamilies).toEqual([
      { key: 'pep_function', id: 'T01', runs: 2, readyModels: 1 },
      { key: 'pep_protein_interaction', id: 'T02', runs: 0, readyModels: 0 },
      { key: 'pep_hlai', id: 'T03', runs: 0, readyModels: 1 },
      { key: 'pep_hlaii', id: 'T04', runs: 1, readyModels: 0 },
    ])
    // A family that only has ready models is not "started"; unknown tasks stay out of the rows.
    expect(dashboard.familiesStarted).toBe(2)
    expect(dashboard.experimentCount).toBe(4)
    expect(dashboard.reusableModelCount).toBe(3)
  })

  it('counts new runs per local day over a 14 day window', () => {
    const dashboard = buildHomeDashboard(
      [
        experiment('today-morning', 'succeeded', { createdAt: localIso(2026, 5, 10, 9) }),
        experiment('today-evening', 'running', { createdAt: localIso(2026, 5, 10, 17) }),
        experiment('first-day', 'failed', { createdAt: localIso(2026, 4, 27, 0) }),
        experiment('too-old', 'succeeded', { createdAt: localIso(2026, 4, 26, 23) }),
        experiment('no-date', 'queued'),
      ],
      [],
      { now: NOW },
    )

    expect(dashboard.activity).toHaveLength(ACTIVITY_DAYS)
    expect(dashboard.activity[0]).toEqual({ date: '2026-04-27', count: 1 })
    expect(dashboard.activity[ACTIVITY_DAYS - 1]).toEqual({ date: '2026-05-10', count: 2 })
    expect(dashboard.activityTotal).toBe(3)
    expect(dashboard.activityPeak).toBe(2)

    const dates = dashboard.activity.map((day) => day.date)
    expect([...dates].sort()).toEqual(dates)
  })

  it('normalises activity bar heights without dividing by zero', () => {
    expect(barRatio(0, 0)).toBe(0)
    expect(barRatio(3, 0)).toBe(0)
    expect(barRatio(0, 4)).toBe(0)
    expect(barRatio(3, 6)).toBe(0.5)
    expect(barRatio(9, 6)).toBe(1)
    expect(barRatio(Number.NaN, 6)).toBe(0)
  })
})
