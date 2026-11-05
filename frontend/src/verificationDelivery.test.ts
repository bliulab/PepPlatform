import { beforeEach, describe, expect, it } from 'vitest'
import {
  cacheDevelopmentCode,
  clearDevelopmentCode,
  deliveryState,
  getDevelopmentCode,
  recordVerificationDelivery,
  verificationDelivery,
  verificationNotice,
} from './verificationDelivery'
import type { AuthResponse } from './types'

const response = { auth_enabled: true, user: null, email: 'user@example.com' } as AuthResponse
beforeEach(() => {
  verificationDelivery.value = null
})
describe('verification delivery feedback', () => {
  it('only claims delivery when the server confirms it', () => {
    expect(deliveryState({ ...response, verification_sent: true })).toBe('sent')
    expect(deliveryState({ ...response, verification_sent: false })).toBe('failed')
    expect(deliveryState(response)).toBe('failed')
    expect(deliveryState({ ...response, verification_sent: false, debug_code: '123456' })).toBe(
      'development',
    )
  })
  it('keeps registration feedback across navigation without leaking it to a different email', () => {
    recordVerificationDelivery(
      { ...response, verification_sent: false },
      'ignored@example.com',
      true,
    )
    expect(verificationNotice(' USER@example.com ')).toEqual({
      email: 'user@example.com',
      state: 'failed',
      registered: true,
    })
    expect(verificationNotice('other@example.com')).toBeNull()
    expect(verificationDelivery.value).not.toHaveProperty('debug_code')
  })
  it('replaces the previous failure after resend succeeds and handles an already verified account', () => {
    recordVerificationDelivery({ ...response, verification_sent: false }, response.email!)
    recordVerificationDelivery({ ...response, verification_sent: true }, response.email!)
    expect(verificationNotice(response.email!)?.state).toBe('sent')
    expect(
      deliveryState({
        ...response,
        user: { email_verified: true } as NonNullable<AuthResponse['user']>,
        verification_sent: false,
      }),
    ).toBe('verified')
  })
  it('updates a resent development code reactively even when browser storage is unavailable', () => {
    cacheDevelopmentCode({ ...response, debug_code: '123456' }, response.email!)
    expect(getDevelopmentCode('USER@example.com')).toBe('123456')
    cacheDevelopmentCode({ ...response, debug_code: '654321' }, response.email!)
    expect(getDevelopmentCode(response.email!)).toBe('654321')
    clearDevelopmentCode(response.email!)
    expect(getDevelopmentCode(response.email!)).toBe('')
  })
})
