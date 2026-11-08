import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, api } from '../api/client'
import type { AuthResponse, AuthUser } from '../types'
import { useAuthStore } from './auth'

const user = { id: 'me', email: 'me@example.com', email_verified: true } as AuthUser
function response(id = 'me', used = 3): AuthResponse {
  return {
    auth_enabled: true,
    user: { ...user, id },
    quota: {
      user_id: id,
      daily_experiment_limit: 10,
      daily_concurrent_experiment_limit: 1,
      daily_inference_request_limit: 20,
      daily_inference_input_limit: 100,
    },
    usage: {
      user_id: id,
      usage_date: '2026-10-26',
      experiments_created: used,
      inference_requests: 0,
      inference_inputs: 0,
    },
  }
}
function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}
beforeEach(() => {
  setActivePinia(createPinia())
  const auth = useAuthStore()
  auth.clearCurrentUser()
  auth.applyAuthResponse(response())
})
afterEach(() => vi.restoreAllMocks())

describe('live account usage', () => {
  it('deduplicates concurrent refreshes and applies the latest server usage', async () => {
    const result = deferred<AuthResponse>()
    const me = vi.spyOn(api, 'me').mockReturnValue(result.promise)
    const auth = useAuthStore()
    const first = auth.refreshUsage()
    const second = auth.refreshUsage()
    expect(me).toHaveBeenCalledTimes(1)
    result.resolve(response('me', 4))
    await Promise.all([first, second])
    expect(auth.usage?.experiments_created).toBe(4)
    expect(auth.loading).toBe(false)
  })
  it('marks failed refreshes unavailable without losing a valid session, and recovers on retry', async () => {
    const me = vi.spyOn(api, 'me').mockRejectedValue(new Error('offline'))
    const auth = useAuthStore()
    await auth.refreshUsage()
    expect(auth.usageUnavailable).toBe(true)
    expect(auth.user?.id).toBe('me')
    me.mockResolvedValue(response())
    await auth.refreshUsage()
    expect(auth.usageUnavailable).toBe(false)
  })
  it('never lets an old usage request restore an account after logout', async () => {
    const old = deferred<AuthResponse>()
    vi.spyOn(api, 'me').mockReturnValue(old.promise)
    vi.spyOn(api, 'logout').mockResolvedValue({ ok: true })
    const auth = useAuthStore()
    const pending = auth.refreshUsage()
    await auth.logout()
    old.resolve(response())
    await pending
    expect(auth.user).toBeNull()
    expect(auth.quota).toBeNull()
  })
  it('ignores a previous account’s refresh after a different account signs in', async () => {
    const old = deferred<AuthResponse>()
    vi.spyOn(api, 'me').mockReturnValue(old.promise)
    vi.spyOn(api, 'login').mockResolvedValue(response('new-user'))
    const auth = useAuthStore()
    const pending = auth.refreshUsage()
    await auth.login('new@example.com', 'password')
    old.resolve(response())
    await pending
    expect(auth.user?.id).toBe('new-user')
    expect(auth.quota?.user_id).toBe('new-user')
  })
  it('clears an expired session instead of showing cached quota', async () => {
    vi.spyOn(api, 'me').mockRejectedValue(new ApiError('expired', 401))
    const auth = useAuthStore()
    await auth.refreshUsage()
    expect(auth.user).toBeNull()
    expect(auth.usage).toBeNull()
    expect(auth.loaded).toBe(true)
  })
})
