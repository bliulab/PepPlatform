import { shallowReactive, shallowRef } from 'vue'
import type { AuthResponse } from './types'

export type DeliveryState = 'sent' | 'failed' | 'development' | 'verified'
export interface VerificationDelivery {
  email: string
  state: DeliveryState
  registered: boolean
}

/** Navigation feedback only: never apply an unverified registration as a session. */
export const verificationDelivery = shallowRef<VerificationDelivery | null>(null)
const developmentCodes = shallowReactive(new Map<string, string>())

function codeKey(email: string, prefix: string) {
  return `${prefix}:${email.trim().toLowerCase()}`
}

/** Only server-provided development codes use this existing opt-in path. */
export function cacheDevelopmentCode(
  response: AuthResponse,
  email: string,
  prefix = 'pepplatform-code',
) {
  const key = codeKey(response.email || email, prefix)
  if (response.debug_code) developmentCodes.set(key, response.debug_code)
  else developmentCodes.set(key, '')
  try {
    if (response.debug_code) window.sessionStorage.setItem(key, response.debug_code)
    else window.sessionStorage.removeItem(key)
  } catch {
    /* Feedback remains usable when browser storage is unavailable. */
  }
}

export function getDevelopmentCode(email: string, prefix = 'pepplatform-code') {
  const key = codeKey(email, prefix)
  const cached = developmentCodes.get(key)
  if (cached !== undefined) return cached
  try {
    return window.sessionStorage.getItem(key) || ''
  } catch {
    return ''
  }
}

export function clearDevelopmentCode(email: string, prefix = 'pepplatform-code') {
  const key = codeKey(email, prefix)
  developmentCodes.set(key, '')
  try {
    window.sessionStorage.removeItem(key)
  } catch {
    /* Storage is optional. */
  }
}

export function deliveryState(response: AuthResponse): DeliveryState {
  if (response.user?.email_verified) return 'verified'
  if (response.verification_sent === true) return 'sent'
  if (response.debug_code) return 'development'
  return 'failed'
}

export function recordVerificationDelivery(
  response: AuthResponse,
  email: string,
  registered = false,
) {
  verificationDelivery.value = {
    email: (response.email || email).trim().toLowerCase(),
    state: deliveryState(response),
    registered,
  }
}

export function verificationNotice(email: string) {
  const delivery = verificationDelivery.value
  return delivery?.email === email.trim().toLowerCase() ? delivery : null
}
