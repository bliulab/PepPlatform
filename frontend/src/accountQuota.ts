import type { AdminResourceCounts, QuotaUsage, UserQuota } from './types'

export interface QuotaReadout {
  used: number
  limit: number
  remaining: number
}
export type QuotaKind = 'experiments' | 'concurrent' | 'inferenceRequests' | 'inferenceInputs'

export function quotaReadout(
  used: number | undefined,
  limit: number | undefined,
): QuotaReadout | null {
  if (
    used == null ||
    limit == null ||
    !Number.isFinite(used) ||
    !Number.isFinite(limit) ||
    used < 0 ||
    limit < 0
  )
    return null
  return { used, limit, remaining: Math.max(0, limit - used) }
}

export function accountQuotaReadouts(
  userId: string,
  quota: UserQuota | null,
  usage: QuotaUsage | null,
  resources: AdminResourceCounts | null,
  now = new Date(),
): Record<QuotaKind, QuotaReadout | null> {
  const ownQuota = quota?.user_id === userId ? quota : null
  const today =
    usage?.user_id === userId && usage.usage_date === now.toISOString().slice(0, 10) ? usage : null
  return {
    experiments: quotaReadout(today?.experiments_created, ownQuota?.daily_experiment_limit),
    concurrent: quotaReadout(
      resources?.active_experiments,
      ownQuota?.daily_concurrent_experiment_limit,
    ),
    inferenceRequests: quotaReadout(
      today?.inference_requests,
      ownQuota?.daily_inference_request_limit,
    ),
    inferenceInputs: quotaReadout(today?.inference_inputs, ownQuota?.daily_inference_input_limit),
  }
}

export function nextQuotaReset(now = new Date()) {
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1))
}

const usageListeners = new Set<() => void>()
export function onQuotaUsageChanged(listener: () => void) {
  usageListeners.add(listener)
  return () => {
    usageListeners.delete(listener)
  }
}

/** Keep account refreshes out of the API module's dependency graph. */
export function notifyQuotaUsageChanged(path: string, method = 'GET', status = 200) {
  if (!['POST', 'DELETE'].includes(method.toUpperCase()) || status < 200) return
  const pathname = path.split('?')[0]
  const affectsUsage =
    pathname === '/api/experiments' ||
    (method.toUpperCase() === 'DELETE' && /^\/api\/experiments\/[^/]+$/.test(pathname)) ||
    /^\/api\/experiments\/[^/]+\/(?:cancel|resume-downstream)$/.test(pathname) ||
    pathname === '/api/ensembles' ||
    pathname === '/api/agent/experiment-batches' ||
    /^\/api\/agent\/execution-plans\/[^/]+\/(?:execute|actions\/[^/]+\/confirm)$/.test(pathname) ||
    /^\/api\/models\/[^/]+\/predict$/.test(pathname)
  if (affectsUsage) for (const listener of usageListeners) listener()
}
