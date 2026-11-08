import { afterEach, describe, expect, it, vi } from 'vitest'
import { notifyUnauthorized, setUnauthorizedHandler } from './authSession'
import { request, requestBlob } from './api/client'

afterEach(() => {
  vi.unstubAllGlobals()
  setUnauthorizedHandler(() => {})
})

describe('expired sessions', () => {
  it.each([
    '/api/auth/me',
    '/api/auth/login',
    '/api/auth/register',
    '/api/auth/logout',
    '/api/auth/verify-email',
    '/api/auth/resend-verification',
    '/api/auth/password-reset/request',
    '/api/auth/password-reset/confirm',
  ])('keeps expected authentication failures local: %s', (path) => {
    const handler = vi.fn()
    setUnauthorizedHandler(handler)
    notifyUnauthorized(path, 401)
    expect(handler).not.toHaveBeenCalled()
  })

  it('handles expired private JSON and download requests without hiding their errors', async () => {
    const handler = vi.fn()
    setUnauthorizedHandler(handler)
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () => new Response(JSON.stringify({ detail: 'session expired' }), { status: 401 }),
      ),
    )
    await expect(request('/api/experiments')).rejects.toMatchObject({ status: 401 })
    await expect(requestBlob('/api/models/owned/export')).rejects.toMatchObject({ status: 401 })
    expect(handler).toHaveBeenCalledTimes(2)
  })

  it('preserves authorization failures and catches private account endpoints', () => {
    const handler = vi.fn()
    setUnauthorizedHandler(handler)
    notifyUnauthorized('/api/experiments', 403)
    expect(handler).not.toHaveBeenCalled()
    notifyUnauthorized('/api/auth/profile', 401)
    expect(handler).toHaveBeenCalledOnce()
  })
})
