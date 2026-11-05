import { describe, expect, it } from 'vitest'

import { ApiError } from './api/client'
import { i18n } from './i18n'
import {
  localizedApiError,
  localizedDatasetIssue,
  localizedPreflightMessage,
  localizedReadinessCheck,
} from './errorLocalization'

const messages: Record<string, string> = {
  'apiErrors.generic': 'generic',
  'apiErrors.invalid_credentials': 'localized credentials',
  'apiErrors.quota_daily_concurrent_experiment_limit': 'slots {used}/{limit}',
  'apiErrors.quota_exceeded': 'generic quota',
  'datasetIssues.duplicate_sample_id': '{count} duplicates',
  'preflightMessages.task_mismatch': 'localized mismatch',
  'preflightMessages.unknown': 'localized preflight fallback',
  'readiness.codes.blast_binary_missing': '{tool} missing',
  'readiness.unknown': 'unknown readiness',
}
const te = (key: string) => key in messages
const t = (key: string, params: Record<string, unknown> = {}) =>
  (messages[key] ?? key).replace(/\{(\w+)\}/g, (_match, name) => String(params[name] ?? ''))

describe('deterministic error localization', () => {
  it.each(['en', 'zh'])('explains upload limits in %s with server-configured values', (locale) => {
    const composer = i18n.global
    const translate = (key: string, params?: Record<string, unknown>) =>
      composer.t(key, params ?? {}, { locale })
    const exists = (key: string) => composer.te(key, locale)
    for (const code of [
      'dataset_upload_request_too_large',
      'dataset_upload_archive_too_large',
      'dataset_upload_extracted_too_large',
      'dataset_upload_file_too_large',
      'dataset_upload_too_many_entries',
    ]) {
      const error = new ApiError('backend detail', 413, {
        code,
        params: { limit_mib: 17, limit_entries: 123 },
      })
      const message = localizedApiError(error, translate, exists)
      expect(message).toContain(code.endsWith('entries') ? '123' : '17')
      expect(message).not.toContain('backend detail')
      expect(message).not.toContain('apiErrors.')
    }
    expect(localizedApiError(new ApiError('raw', 400, {
      code: 'dataset_upload_invalid_archive',
    }), translate, exists)).toMatch(locale === 'zh' ? /ZIP 文件/ : /ZIP is/)
  })

  it('uses API codes and never the backend message', () => {
    const error = new ApiError('后端保存的文本', 401, { code: 'invalid_credentials' })
    expect(localizedApiError(error, t, te)).toBe('localized credentials')
  })

  it('explains occupied concurrent slots separately from a daily quota limit', () => {
    const concurrent = new ApiError('raw quota error', 429, {
      code: 'quota_exceeded',
      quota_type: 'daily_concurrent_experiment_limit',
      used: 2,
      limit: 2,
      requested: 1,
    })
    expect(localizedApiError(concurrent, t, te)).toBe('slots 2/2')
    expect(localizedApiError(new ApiError('raw', 429, { code: 'quota_exceeded' }), t, te)).toBe(
      'generic quota',
    )
  })

  it('renders dataset, Preflight, and readiness codes', () => {
    expect(
      localizedDatasetIssue(
        { severity: 'warning', code: 'duplicate_sample_id', message: 'raw', params: { count: 3 } },
        t,
        te,
      ),
    ).toBe('3 duplicates')
    expect(localizedPreflightMessage('task_mismatch', 'raw', t, te)).toBe('localized mismatch')
    expect(localizedPreflightMessage('missing-code', 'backend details', t, te)).toBe(
      'backend details',
    )
    expect(localizedPreflightMessage('missing-code', '  ', t, te)).toBe(
      'localized preflight fallback',
    )
    expect(
      localizedReadinessCheck(
        {
          id: 'psiblast',
          ok: false,
          status: 'error',
          code: 'blast_binary_missing',
          message: 'raw',
        },
        t,
        te,
      ),
    ).toBe('psiblast missing')
  })

  it('provides Chinese and English messages for semantic partner count failures', () => {
    const codes = ['semantic_positive_partners_invalid', 'semantic_zero_partners_invalid']
    for (const locale of ['zh', 'en']) {
      const localeMessages = i18n.global.getLocaleMessage(locale)
      const preflightMessages = localeMessages.preflightMessages as Record<string, unknown>
      for (const code of codes) {
        expect(String(preflightMessages[code] ?? '').trim(), `${locale}.${code}`).not.toBe('')
      }
    }
  })

  it('provides Chinese and English messages for every DT preflight code', () => {
    const codes = [
      'dt_fallback_unsupported',
      'dt_hash_size_removed',
      'dt_length_stats_unavailable',
      'dt_max_length_auto_adjusted',
      'dt_profile_fallback_enabled',
    ]

    for (const locale of ['zh', 'en']) {
      const localeMessages = i18n.global.getLocaleMessage(locale)
      const preflightMessages = localeMessages.preflightMessages as Record<string, unknown>
      for (const code of codes) {
        expect(preflightMessages[code], `${locale}.${code}`).toBeTypeOf('string')
        expect(String(preflightMessages[code]).trim(), `${locale}.${code}`).not.toBe('')
      }
    }
  })

  it('provides Chinese and English messages for covariance preflight failures', () => {
    const codes = [
      'covariance_encoder_requires_k1',
      'covariance_encoder_requires_kmer',
      'covariance_length_stats_unavailable',
      'covariance_lag_too_large',
      'covariance_raw_length_exceeded',
    ]

    for (const locale of ['zh', 'en']) {
      const localeMessages = i18n.global.getLocaleMessage(locale)
      const preflightMessages = localeMessages.preflightMessages as Record<string, unknown>
      for (const code of codes) {
        expect(preflightMessages[code], `${locale}.${code}`).toBeTypeOf('string')
        expect(String(preflightMessages[code]).trim(), `${locale}.${code}`).not.toBe('')
      }
    }
  })

  it('provides Chinese and English messages for PDT preflight failures', () => {
    const codes = [
      'pdt_encoder_requires_k1',
      'pdt_encoder_requires_kmer',
      'pdt_length_stats_unavailable',
      'pdt_lag_too_large',
      'pdt_raw_length_exceeded',
    ]

    for (const locale of ['zh', 'en']) {
      const localeMessages = i18n.global.getLocaleMessage(locale)
      const preflightMessages = localeMessages.preflightMessages as Record<string, unknown>
      for (const code of codes) {
        expect(preflightMessages[code], `${locale}.${code}`).toBeTypeOf('string')
        expect(String(preflightMessages[code]).trim(), `${locale}.${code}`).not.toBe('')
      }
    }
  })
})
