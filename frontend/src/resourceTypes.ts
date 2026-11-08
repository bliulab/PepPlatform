import type { QuotaKey, QuotaUsage, UserQuota } from './types'

export type DailyReductionMode = 'next_reset' | 'immediate'

export interface ResourcePolicy {
  revision: number
  updated_at: string
  defaults: Record<QuotaKey, number>
  limits: {
    daily_registration_limit: number
    daily_verification_email_limit: number
    queue_capacity: number
    disk_reserve_bytes: number
    disk_reserve_percent: number
    disk_critical_bytes: number
    disk_critical_percent: number
    inference_timeout_seconds: number
    allowed_pretrained_models: string[]
  }
  retention: {
    enabled: boolean
    delete_unprotected_models: boolean
    publish_announcements: boolean
    notice_hours: number
    cache_idle_hours: number
  }
}
export interface EmailDeliveryUsage {
  usage_date: string
  used: number
  limit: number
  remaining: number
  reset_at: string | null
}
export interface PolicyDraft {
  revision: number
  defaults?: Partial<ResourcePolicy['defaults']>
  limits?: Partial<ResourcePolicy['limits']>
  retention?: Partial<ResourcePolicy['retention']>
  apply_to_users?: boolean
  preview_token?: string
  daily_reduction_mode?: DailyReductionMode
  publish_announcements?: boolean
}
export interface PolicyPreview {
  affected_count: number
  skipped_count: number
  affected: { user_id: string; email: string; fields: QuotaKey[] }[]
  skipped: { user_id: string; email: string; fields: QuotaKey[] }[]
  preview_token: string
  immediate_count: number
  deferred_count: number
  blocked_count: number
  exempt_count: number
  announcement_count: number
  impacts: { user_id: string; email: string; changes: QuotaChanges; blocks_new: boolean }[]
}
export type QuotaChanges = Partial<
  Record<
    QuotaKey,
    {
      from: number
      to: number
      effective_at: string
      effective_from?: number
      effective_to?: number
      daily_reduction_mode?: DailyReductionMode
    }
  >
>
export interface StorageUsage {
  used_bytes: number
  protected_bytes: number
  reclaimable_bytes: number
  reserved_bytes: number
  scanned_at: string | null
}
export interface ResourceEntry {
  owner_email?: string | null
  id: string
  name: string
  kind: string
  owner_user_id: string | null
  model_id: string | null
  size_bytes: number
  protected: boolean
  reason: string
  state: string
  scheduled_at?: string | null
  notified_at?: string | null
  delete_after: string | null
}
export interface ResourceOverview {
  gpus: {
    index: string
    name: string
    total_bytes: number
    used_bytes: number
    utilization_percent: number | null
  }[]
  disk: {
    total_bytes: number
    used_bytes: number
    free_bytes: number
    reserve_bytes: number
    critical_bytes: number
    state: 'ready' | 'paused' | 'critical'
  }
  storage: StorageUsage
  workload: { queued: number; running: number }
  registration: { used: number; limit: number }
  items: ResourceEntry[]
  cleanup_logs: {
    id: string
    resource_id: string
    status: string
    released_bytes: number
    error: string | null
    created_at: string
  }[]
}
export interface OwnResources {
  models: ModelUsage
  storage: StorageUsage
  quota: UserQuota
  usage: QuotaUsage
  items: ResourceEntry[]
  disk_state: string
}
export interface ModelUsage {
  retained: number
  protected: number
  reserved: number
  remaining: number | null
  protect_remaining: number | null
  over_limit: number
  protection_over_limit: number
}
export interface ModelCleanupPreview {
  enabled: boolean
  notice_hours: number
  eligible_count: number
  items: {
    model_id: string
    name: string
    owner_user_id: string
    owner_email?: string
    state: string
    pinned: boolean
    eligible: boolean
    skip_reason: string | null
    expires_at: string
    scheduled_at?: string | null
    notified_at?: string | null
    delete_after: string | null
    estimated_bytes?: number
  }[]
}
export interface Announcement {
  audience_labels?: Record<string, string>
  id: string
  title: string
  body: string
  audience: 'all' | string[]
  pinned: boolean
  kind: 'manual' | 'cleanup' | 'quota_change'
  quota_change?: QuotaChanges
  publish_at: string
  expires_at: string | null
  read: boolean
}
export type AnnouncementPayload = Pick<Announcement, 'title' | 'body' | 'audience' | 'pinned'> & {
  publish_at?: string
  expires_at?: string | null
}
