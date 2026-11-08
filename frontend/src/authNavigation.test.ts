import { describe, expect, it } from 'vitest'
import { allowsGuestPreview, loginLocation, postAuthRedirect } from './authNavigation'

describe('authentication return addresses', () => {
  it.each([
    undefined,
    null,
    [],
    '',
    'https://example.org',
    '//example.org',
    '/\\example.org',
    '/%2f%2fexample.org',
    '/%5cexample.org',
    '/%0a/login',
    '/%',
    '/login',
    '/login?redirect=/login',
    '/login/',
    '/%6cogin',
    '/register#form',
    '/verify-email?email=user@example.org',
    '/password-reset',
  ])('uses the dashboard for an unsafe or recursive return address: %s', (value) => {
    expect(postAuthRedirect(value)).toBe('/dashboard')
  })

  it.each([
    '/',
    '/dashboard',
    '/experiments/new?task=pep_hlai#settings',
    '/models?model=owned-model',
  ])('keeps an internal destination: %s', (value) => {
    expect(postAuthRedirect(value)).toBe(value)
    expect(loginLocation(value)).toEqual({ path: '/login', query: { redirect: value } })
  })
})

describe('guest route boundaries', () => {
  const preview = { meta: { guestPreview: true }, params: {}, query: {} }
  it('allows generic tools but requires authentication for resource details', () => {
    expect(allowsGuestPreview(preview)).toBe(true)
    expect(allowsGuestPreview({ ...preview, meta: {} })).toBe(false)
    expect(allowsGuestPreview({ ...preview, params: { conversationId: 'private-chat' } })).toBe(
      false,
    )
  })
  it.each([
    'model',
    'ids',
    'clone',
    'plan_id',
    'reuse_plan_id',
    'source_experiment_id',
    'analysis_message_id',
  ])('guards resource query %s', (key) => {
    expect(allowsGuestPreview({ ...preview, query: { [key]: 'private-id' } })).toBe(false)
    expect(allowsGuestPreview({ ...preview, query: { [key]: ['private-id'] } })).toBe(false)
    expect(allowsGuestPreview({ ...preview, query: { [key]: '' } })).toBe(true)
  })
})
