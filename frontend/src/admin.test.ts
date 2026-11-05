import { describe, expect, it } from 'vitest'
import {
  adminAuditRange,
  adminPageSize,
  adminUserSort,
  auditRangeStart,
  confirmationMatches,
  positiveInteger,
  safeAdminUsersReturnPath,
  sortOrder,
} from './admin'

describe('admin workspace helpers', () => {
  it('normalizes pagination and sorting query values', () => {
    expect(positiveInteger('3', 1)).toBe(3)
    expect(positiveInteger('-2', 1)).toBe(1)
    expect(adminPageSize('50')).toBe(50)
    expect(adminPageSize('17')).toBe(25)
    expect(adminUserSort('email')).toBe('email')
    expect(adminUserSort('quota')).toBe('created_at')
    expect(sortOrder('asc')).toBe('asc')
    expect(sortOrder('sideways')).toBe('desc')
  })

  it('turns audit ranges into stable UTC cutoffs', () => {
    const now = new Date('2026-07-20T12:00:00.000Z')
    expect(adminAuditRange('7d')).toBe('7d')
    expect(adminAuditRange('invalid')).toBe('24h')
    expect(auditRangeStart('24h', now)).toBe('2026-07-19T12:00:00.000Z')
    expect(auditRangeStart('7d', now)).toBe('2026-07-13T12:00:00.000Z')
    expect(auditRangeStart('all', now)).toBeUndefined()
  })

  it('accepts only internal user-list return paths and exact confirmations', () => {
    expect(safeAdminUsersReturnPath('/admin/users?page=2')).toBe('/admin/users?page=2')
    expect(safeAdminUsersReturnPath('//example.com/admin/users')).toBe('/admin/users')
    expect(safeAdminUsersReturnPath('/experiments')).toBe('/admin/users')
    expect(confirmationMatches(' user@example.com ', 'user@example.com')).toBe(true)
    expect(confirmationMatches('USER@example.com', 'user@example.com')).toBe(false)
  })
})
