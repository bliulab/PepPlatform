import { describe, expect, it } from 'vitest'
import { activityLevel, buildExperimentActivity } from './experimentActivity'
import type { ExperimentRecord } from './types'

function run(id: string, created_at?: string, owner_user_id = 'me', status = 'succeeded') {
  return { id, created_at, owner_user_id, status } as ExperimentRecord
}
const now = new Date(2026, 9, 26, 18)
const local = (y: number, m: number, d: number, hour = 12) =>
  new Date(y, m - 1, d, hour).toISOString()

describe('annual experiment activity', () => {
  it('fills exactly 365 local days and pads complete Sunday-first weeks', () => {
    const result = buildExperimentActivity([], { now })
    expect(result.days).toHaveLength(365)
    expect(result.days[0]?.date).toBe('2025-10-27')
    expect(result.days.at(-1)?.date).toBe('2026-10-26')
    expect(result.cells.length % 7).toBe(0)
    expect(result.cells.filter(Boolean)).toHaveLength(365)
    expect(result.cells.slice(0, result.leading).every((day) => day === null)).toBe(true)
    expect(result.total).toBe(0)
    expect(result.activeDays).toBe(0)
  })

  it('counts creations in every state and only the current owner, including for an admin', () => {
    const records = ['queued', 'running', 'failed', 'succeeded', 'cancelled'].map((status, i) =>
      run(String(i), local(2026, 10, 25), 'me', status),
    )
    records.push(run('someone-else', local(2026, 10, 25), 'other'))
    records.push(run('legacy', local(2026, 10, 25), ''))
    const result = buildExperimentActivity(records, { now, ownerId: 'me' })
    expect(result.total).toBe(5)
    expect(result.activeDays).toBe(1)
    expect(result.days.find((day) => day.date === '2026-10-25')).toEqual({
      date: '2026-10-25',
      count: 5,
      level: 3,
    })
    expect(buildExperimentActivity(records, { now }).total).toBe(7)
  })

  it('reduces a day after deletion and does not use finish or update times', () => {
    const old = {
      ...run('a', local(2026, 10, 25)),
      finished_at: local(2026, 10, 26),
      updated_at: local(2026, 10, 26),
    }
    const records = [old, run('b', local(2026, 10, 25))]
    expect(buildExperimentActivity(records, { now }).days.at(-2)?.count).toBe(2)
    const result = buildExperimentActivity(records.slice(1), { now })
    expect(result.days.at(-2)?.count).toBe(1)
    expect(result.days.at(-1)?.count).toBe(0)
  })

  it('excludes missing/invalid/future/out-of-window dates without losing the first day', () => {
    const result = buildExperimentActivity(
      [
        run('first', local(2025, 10, 27, 0)),
        run('before', local(2025, 10, 26, 23)),
        run('future', local(2026, 10, 27)),
        run('later-today', local(2026, 10, 26, 23)),
        run('invalid', 'invalid'),
        run('missing'),
      ],
      { now },
    )
    expect(result.total).toBe(1)
    expect(result.days[0]?.count).toBe(1)
  })

  it('retains leap day and consecutive dates across year and DST boundaries', () => {
    const result = buildExperimentActivity([], { now: new Date(2024, 11, 31, 18) })
    expect(result.days.some((day) => day.date === '2024-02-29')).toBe(true)
    expect(new Set(result.days.map((day) => day.date)).size).toBe(365)
    expect(
      result.months.every((item) =>
        result.cells
          .slice((item.column - 1) * 7, item.column * 7)
          .some((day) => day?.date === item.date),
      ),
    ).toBe(true)
  })

  it('uses stable green thresholds rather than recolouring all days when a peak is deleted', () => {
    expect([0, 1, 2, 3, 4, 6, 7, 100].map(activityLevel)).toEqual([0, 1, 2, 2, 3, 3, 4, 4])
  })
})
