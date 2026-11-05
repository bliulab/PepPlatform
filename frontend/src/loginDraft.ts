import { onScopeDispose } from 'vue'
import { useRoute } from 'vue-router'
import { useAuthStore } from './stores/auth'

const storageKey = 'pepplatform:login-drafts:v1'
type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue }
interface DraftEntry {
  owner: string | null
  data: JsonValue
}
const readers = new Map<string, () => unknown>()

function storage(): Storage | undefined {
  try {
    return window.sessionStorage
  } catch {
    return undefined
  }
}

/** Keep JSON-compatible form values and omit known sensitive field names. */
export function draftValues(value: unknown): JsonValue | undefined {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value
  if (typeof value === 'number') return Number.isFinite(value) ? value : undefined
  if (Array.isArray(value))
    return value.map(draftValues).filter((v): v is JsonValue => v !== undefined)
  if (typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype)
    return undefined
  const result: Record<string, JsonValue> = {}
  for (const [key, item] of Object.entries(value)) {
    if (
      /^(password|api_?key|secret|access_?token|refresh_?token|verification_?code|file|files|archive)$/i.test(
        key,
      )
    )
      continue
    if (key === '__proto__' || key === 'constructor' || key === 'prototype') continue
    const safe = draftValues(item)
    if (safe !== undefined) result[key] = safe
  }
  return result
}

function readEntries(): Record<string, DraftEntry> {
  try {
    const parsed = JSON.parse(storage()?.getItem(storageKey) || '{}')
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}
  } catch {
    return {}
  }
}

function writeEntries(entries: Record<string, DraftEntry>) {
  try {
    if (Object.keys(entries).length) storage()?.setItem(storageKey, JSON.stringify(entries))
    else storage()?.removeItem(storageKey)
  } catch {
    /* Browsing and signing in also work when session storage is unavailable. */
  }
}

export function saveLoginDrafts(owner: string | null) {
  const entries: Record<string, DraftEntry> = {}
  for (const [key, read] of readers) {
    const data = draftValues(read())
    if (data !== undefined) entries[key] = { owner, data }
  }
  writeEntries(entries)
}

export function clearLoginDrafts() {
  writeEntries({})
}

export function takeLoginDraft(key: string, owner: string | null): JsonValue | undefined {
  const entries = readEntries()
  const entry = entries[key]
  // A session-expiry remount must not consume the previous account's draft.
  if (entry?.owner && owner === null) return undefined
  delete entries[key]
  writeEntries(entries)
  if (!entry || (entry.owner !== null && entry.owner !== owner)) return undefined
  return draftValues(entry.data)
}

/** Register explicitly selected form fields; restoration happens during view setup. */
export function useLoginDraft<T extends object>(
  scope: string,
  read: () => T,
  restore: (value: T) => void,
) {
  const route = useRoute()
  const auth = useAuthStore()
  const key = `${route.fullPath}:${scope}`
  const saved = takeLoginDraft(key, auth.user?.id ?? null)
  if (saved && typeof saved === 'object' && !Array.isArray(saved)) {
    try {
      restore(saved as T)
    } catch {
      /* Discard an incomplete or obsolete form snapshot. */
    }
  }
  readers.set(key, read)
  onScopeDispose(() => {
    if (readers.get(key) === read) readers.delete(key)
  })
}
