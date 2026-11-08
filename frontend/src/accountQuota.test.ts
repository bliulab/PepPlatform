import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  accountQuotaReadouts,
  nextQuotaReset,
  notifyQuotaUsageChanged,
  onQuotaUsageChanged,
  quotaReadout,
} from './accountQuota'
import type { AdminResourceCounts, QuotaUsage, UserQuota } from './types'

const now = new Date('2026-10-26T23:59:59Z')
const quota = {
  user_id: 'me',
  daily_experiment_limit: 10,
  daily_concurrent_experiment_limit: 2,
  daily_inference_request_limit: 20,
  daily_inference_input_limit: 100,
} as UserQuota
const usage = {
  user_id: 'me',
  usage_date: '2026-10-26',
  experiments_created: 3,
  inference_requests: 2,
  inference_inputs: 12,
} as QuotaUsage
const resources = { active_experiments: 2 } as AdminResourceCounts
afterEach(() => vi.restoreAllMocks())

describe('account quota readouts', () => {
  it('keeps unlimited administrator usage distinct from unavailable data and zero quota', () => {
    const readouts = accountQuotaReadouts(
      'me',
      { ...quota, enforcement: 'exempt_admin', daily_experiment_limit: 0 },
      usage,
      resources,
      now,
    )
    expect(readouts.experiments).toEqual({ used: 3, limit: null, remaining: null, unlimited: true })
    expect(readouts.concurrent?.unlimited).toBe(true)
    expect(quotaReadout(undefined, 0, true)).toBeNull()
    expect(quotaReadout(3, 0)?.remaining).toBe(0)
  })
  it('distinguishes daily remaining submissions from occupied concurrent slots', () => {
    const result = accountQuotaReadouts('me', quota, usage, resources, now)
    expect(result.experiments).toEqual({ used: 3, limit: 10, remaining: 7 })
    expect(result.concurrent).toEqual({ used: 2, limit: 2, remaining: 0 })
    expect(result.inferenceRequests?.remaining).toBe(18)
    expect(result.inferenceInputs?.remaining).toBe(88)
  })
  it('treats zero as disabled and clamps an administrator-reduced limit to zero remaining', () => {
    expect(quotaReadout(0, 0)?.remaining).toBe(0)
    expect(quotaReadout(12, 10)?.remaining).toBe(0)
    expect(quotaReadout(undefined, 10)).toBeNull()
    expect(quotaReadout(0, NaN)).toBeNull()
  })
  it('does not turn stale or another user’s daily usage into an apparent fresh allowance', () => {
    expect(accountQuotaReadouts('other', quota, usage, resources, now).experiments).toBeNull()
    expect(
      accountQuotaReadouts('me', quota, usage, resources, new Date('2026-10-27T00:00:00Z'))
        .experiments,
    ).toBeNull()
    expect(accountQuotaReadouts('me', null, usage, resources, now).concurrent).toBeNull()
    expect(nextQuotaReset(now).toISOString()).toBe('2026-10-27T00:00:00.000Z')
    expect(nextQuotaReset(new Date('2026-12-31T23:59:59Z')).toISOString()).toBe(
      '2027-01-01T00:00:00.000Z',
    )
  })
  it('refreshes after each submission route, deletion and quota rejection, without refreshing on preflight/reads', () => {
    const refresh = vi.fn()
    const unsubscribe = onQuotaUsageChanged(refresh)
    for (const path of [
      '/api/experiments',
      '/api/experiments/e/cancel',
      '/api/experiments/e/resume-downstream',
      '/api/ensembles',
      '/api/agent/experiment-batches',
      '/api/agent/execution-plans/p/execute',
      '/api/agent/execution-plans/p/actions/a/confirm',
      '/api/models/m/predict',
      '/api/models/m/predict/stream',
    ]) {
      notifyQuotaUsageChanged(path, 'POST', 200)
    }
    notifyQuotaUsageChanged('/api/experiments/e', 'DELETE', 200)
    notifyQuotaUsageChanged('/api/experiments', 'POST', 429)
    notifyQuotaUsageChanged('/api/models/m', 'DELETE', 200)
    notifyQuotaUsageChanged('/api/models/m/protection', 'PATCH', 200)
    expect(refresh).toHaveBeenCalledTimes(13)
    for (const path of [
      '/api/auth/me',
      '/api/experiments/preflight',
      '/api/agent/execution-plans/p/refresh',
    ])
      notifyQuotaUsageChanged(path, 'POST', 200)
    notifyQuotaUsageChanged('/api/experiments', 'GET', 200)
    expect(refresh).toHaveBeenCalledTimes(13)
    unsubscribe()
    notifyQuotaUsageChanged('/api/experiments', 'POST')
    expect(refresh).toHaveBeenCalledTimes(13)
  })
})
