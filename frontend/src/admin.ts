import { onBeforeUnmount, ref, type ComputedRef } from 'vue'
import { onBeforeRouteLeave } from 'vue-router'

export type AdminAuditRange = '24h' | '7d' | '30d' | 'all'
export type AdminUserSort = 'created_at' | 'display_name' | 'email' | 'role' | 'status'
export type SortOrder = 'asc' | 'desc'

export function positiveInteger(value: unknown, fallback: number) {
  const parsed = Number.parseInt(String(value ?? ''), 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

export function adminPageSize(value: unknown) {
  const parsed = positiveInteger(value, 25)
  return [25, 50, 100].includes(parsed) ? parsed : 25
}

export function adminAuditRange(value: unknown): AdminAuditRange {
  return ['24h', '7d', '30d', 'all'].includes(String(value))
    ? (String(value) as AdminAuditRange)
    : '24h'
}

export function auditRangeStart(range: AdminAuditRange, now = new Date()) {
  if (range === 'all') return undefined
  const hours = range === '24h' ? 24 : range === '7d' ? 24 * 7 : 24 * 30
  return new Date(now.getTime() - hours * 60 * 60 * 1000).toISOString()
}

export function adminUserSort(value: unknown): AdminUserSort {
  return ['created_at', 'display_name', 'email', 'role', 'status'].includes(String(value))
    ? (String(value) as AdminUserSort)
    : 'created_at'
}

export function sortOrder(value: unknown): SortOrder {
  return value === 'asc' ? 'asc' : 'desc'
}

export function safeAdminUsersReturnPath(value: unknown) {
  const path = String(value || '')
  return path.startsWith('/admin/users') && !path.startsWith('//') ? path : '/admin/users'
}

export function confirmationMatches(value: string, requiredText: string) {
  return value.trim() === requiredText.trim()
}

export function useDirtyRouteGuard(isDirty: ComputedRef<boolean>) {
  const discardDialogOpen = ref(false)
  let settleNavigation: ((allow: boolean) => void) | null = null

  onBeforeRouteLeave(() => {
    if (!isDirty.value) return true
    discardDialogOpen.value = true
    return new Promise<boolean>((resolve) => {
      settleNavigation = resolve
    })
  })

  function settle(allow: boolean) {
    discardDialogOpen.value = false
    const resolve = settleNavigation
    settleNavigation = null
    resolve?.(allow)
  }

  onBeforeUnmount(() => settle(false))

  return {
    discardDialogOpen,
    confirmDiscard: () => settle(true),
    keepEditing: () => settle(false),
  }
}
