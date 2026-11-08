import { describe, expect, it } from 'vitest'
import {
  quotaValue,
  formatResourceBytes,
  pendingQuotaChanges,
  pendingQuotaUpdates,
} from './resourcePolicy'
import { localizedApiError } from './errorLocalization'
import { ApiError } from './api/client'
import { i18n } from './i18n'

describe('resource quota input', () => {
  it('describes effective and scheduled values without rewriting the configured quota', () => {
    const quota = {
      user_id: 'me',
      daily_experiment_limit: 10,
      daily_concurrent_experiment_limit: 1,
      daily_inference_request_limit: 200,
      daily_inference_input_limit: 10000,
      pending: { daily_experiment_limit: { value: 5, effective_at: '2026-11-08T00:00:00+00:00' } },
    }
    expect(pendingQuotaChanges(quota)).toEqual({
      daily_experiment_limit: { from: 10, to: 5, effective_at: '2026-11-08T00:00:00+00:00' },
    })
    expect(pendingQuotaChanges({ ...quota, enforcement: 'exempt_admin' })).toEqual({})
    expect(pendingQuotaChanges(null)).toEqual({})
    expect(quota.daily_experiment_limit).toBe(10)
  })
  it('converts displayed minutes and fractional GiB without rounding away a quota', () => {
    expect(quotaValue(0.5, 1024 ** 3)).toBe(536870912)
    expect(quotaValue(0.1, 1024 ** 3)).toBe(107374182)
    expect(quotaValue(1 / 60, 60)).toBe(1)
    expect(quotaValue(30, 60)).toBe(1800)
  })
  it('applies only pending daily fields and preserves their inheritance', () => {
    const quota = {
      user_id: 'me',
      daily_experiment_limit: 10,
      daily_concurrent_experiment_limit: 1,
      daily_inference_request_limit: 200,
      daily_inference_input_limit: 10000,
      sources: {
        daily_experiment_limit: 'global',
        daily_inference_request_limit: 'personal',
      } as const,
      overrides: { daily_inference_request_limit: 5 },
      pending: {
        daily_experiment_limit: { value: 2, effective_at: '2026-11-08T00:00:00Z' },
        daily_inference_request_limit: { value: 5, effective_at: '2026-11-08T00:00:00Z' },
        max_model_count: { value: 1, effective_at: '2026-11-08T00:00:00Z' },
      },
    }
    expect(pendingQuotaUpdates(quota)).toEqual({
      daily_experiment_limit: null,
      daily_inference_request_limit: 5,
    })
    expect(quota.sources.daily_experiment_limit).toBe('global')
    expect(quota.overrides).toEqual({ daily_inference_request_limit: 5 })
    expect(pendingQuotaUpdates({ ...quota, enforcement: 'exempt_admin' })).toEqual({})
    expect(pendingQuotaUpdates({ ...quota, pending: {} })).toEqual({})
  })
  it('rejects invalid, fractional base units and unsafe integers', () => {
    for (const value of [NaN, Infinity, -1, 0.1, Number.MAX_SAFE_INTEGER + 1]) {
      expect(quotaValue(value, 1)).toBeNull()
    }
    expect(quotaValue(0, 1)).toBe(0)
    expect(quotaValue(0, 1, 1)).toBeNull()
    expect(quotaValue('' as unknown as number, 1)).toBeNull()
  })
  it('labels byte quantities with binary units', () => {
    expect(formatResourceBytes(1024 ** 3)).toBe('1 GiB')
    expect(formatResourceBytes(1024 ** 2)).toBe('1 MiB')
  })
})

describe('resource error localization', () => {
  it('explains the shared email budget and distinguishes manual pause from daily exhaustion', () => {
    const previous = i18n.global.locale.value
    try {
      for (const locale of ['zh', 'en'] as const) {
        i18n.global.locale.value = locale
        const reset_at = '2026-11-09T00:00:00+00:00'
        const message = (limit: number) => localizedApiError(
          new ApiError('quota', 429, {
            code: 'quota_exceeded', quota_type: 'daily_verification_email_limit',
            limit, used: 20, reset_at,
          }), i18n.global.t, i18n.global.te,
        )
        expect(message(20)).toContain(new Date(reset_at).toLocaleString())
        expect(message(0)).not.toContain(new Date(reset_at).toLocaleString())
        expect(message(20)).toContain(locale === 'zh' ? '平台今日验证码邮件' : 'platform’s daily verification email')
        expect(message(0)).toContain(locale === 'zh' ? '管理员已暂停' : 'administrator paused')
      }
    } finally {
      i18n.global.locale.value = previous
    }
  })
  it('distinguishes inference quotas, disabled access, GPU contention and rate limiting in both languages', () => {
    const previous = i18n.global.locale.value
    try {
      for (const locale of ['zh', 'en'] as const) {
        i18n.global.locale.value = locale
        const message = (detail: ConstructorParameters<typeof ApiError>[2]) =>
          localizedApiError(new ApiError('error', 429, detail), i18n.global.t, i18n.global.te)
        const reset_at = '2026-11-08T00:00:00+00:00'
        const request = message({
          code: 'quota_exceeded',
          quota_type: 'daily_inference_request_limit',
          limit: 10,
          used: 10,
          requested: 1,
          reset_at,
        })
        const input = message({
          code: 'quota_exceeded',
          quota_type: 'daily_inference_input_limit',
          limit: 20,
          used: 17,
          requested: 5,
          reset_at,
        })
        const disabled = message({
          code: 'quota_exceeded',
          quota_type: 'daily_inference_request_limit',
          limit: 0,
        })
        const busy = message({ code: 'gpu_busy_training' })
        const generic = message(undefined)
        expect(new Set([request, input, disabled, busy, generic]).size).toBe(5)
        expect(request).toContain('10')
        expect(request).toContain(new Date(reset_at).toLocaleString())
        expect(input).toContain('3')
        expect(input).toContain('5')
        expect(disabled).not.toContain(new Date(reset_at).toLocaleString())
        expect(request).toContain(locale === 'zh' ? '今日推理次数' : 'daily inference allowance')
      }
    } finally {
      i18n.global.locale.value = previous
    }
  })
  it('explains time and storage limits in both supported languages', () => {
    const previous = i18n.global.locale.value
    try {
      for (const locale of ['zh', 'en'] as const) {
        i18n.global.locale.value = locale
        for (const quota of [
          'daily_training_seconds_limit',
          'storage_limit_bytes',
          'daily_registration_limit',
        ]) {
          const message = localizedApiError(
            new ApiError('quota', 429, { code: 'quota_exceeded', quota_type: quota }),
            i18n.global.t,
            i18n.global.te,
          )
          expect(message).not.toContain('resource.errors.')
          expect(message).not.toBe(i18n.global.t('apiErrors.generic'))
        }
      }
    } finally {
      i18n.global.locale.value = previous
    }
  })
})
