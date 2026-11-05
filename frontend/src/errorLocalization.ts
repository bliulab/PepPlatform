import { ApiError } from './api/client'
import type { DatasetReportIssue, InferenceReadinessCheck, InferenceReadinessIssue } from './types'

type Translate = (key: string, params?: Record<string, unknown>) => string
type TranslationExists = (key: string) => boolean

function apiErrorParams(error: ApiError) {
  return {
    ...(error.params ?? {}),
    limit: error.limit ?? '',
    used: error.used ?? '',
    requested: error.requested ?? '',
    quotaType: error.quotaType ?? '',
  }
}

export function localizedApiError(
  error: unknown,
  translate: Translate,
  translationExists: TranslationExists,
  fallbackKey = 'apiErrors.generic',
) {
  if (error instanceof ApiError) {
    const quotaKey =
      error.code === 'quota_exceeded' && error.quotaType ? `apiErrors.quota_${error.quotaType}` : ''
    if (quotaKey && translationExists(quotaKey)) return translate(quotaKey, apiErrorParams(error))
    const codeKey = error.code ? `apiErrors.${error.code}` : ''
    if (codeKey && translationExists(codeKey)) {
      return translate(codeKey, apiErrorParams(error))
    }
    const statusKey = `apiErrors.http${error.status}`
    if (translationExists(statusKey)) return translate(statusKey)
  }
  return translate(fallbackKey)
}

export function localizedDatasetIssue(
  issue: DatasetReportIssue,
  translate: Translate,
  translationExists: TranslationExists,
) {
  const variant = String(issue.params?.variant ?? '')
  const variantKey = variant ? `datasetIssues.${issue.code}.${variant}` : ''
  const codeKey = `datasetIssues.${issue.code}`
  const params = issue.params ?? {}
  if (variantKey && translationExists(variantKey)) return translate(variantKey, params)
  if (translationExists(codeKey)) return translate(codeKey, params)
  return translate('datasetIssues.unknown')
}

export function localizedPreflightMessage(
  code: string | undefined,
  message: string | undefined,
  translate: Translate,
  translationExists: TranslationExists,
  params: Record<string, unknown> = {},
) {
  const key = code ? `preflightMessages.${code}` : ''
  return key && translationExists(key)
    ? translate(key, params)
    : message?.trim() || translate('preflightMessages.unknown')
}

function readinessParams(check: Partial<InferenceReadinessCheck>) {
  return {
    tool: check.id ?? '',
    feature: check.feature ?? '',
    value: check.value ?? '',
  }
}

export function localizedReadinessCheck(
  check: InferenceReadinessCheck,
  translate: Translate,
  translationExists: TranslationExists,
) {
  const codeKey = check.code ? `readiness.codes.${check.code}` : ''
  if (codeKey && translationExists(codeKey)) {
    return translate(codeKey, readinessParams(check))
  }
  const stateKey = `readiness.checks.${check.id}.${check.ok ? 'ok' : check.status}`
  if (translationExists(stateKey)) return translate(stateKey, readinessParams(check))
  return translate('readiness.unknown')
}

export function localizedReadinessIssue(
  issue: InferenceReadinessIssue,
  translate: Translate,
  translationExists: TranslationExists,
) {
  const codeKey = `readiness.codes.${issue.code}`
  return translationExists(codeKey) ? translate(codeKey) : translate('readiness.unknown')
}

export function localizedReadinessHint(
  issue: InferenceReadinessIssue | InferenceReadinessCheck,
  translate: Translate,
  translationExists: TranslationExists,
) {
  const code = issue.code ?? ''
  const hintKey = code ? `readiness.hints.${code}` : ''
  return hintKey && translationExists(hintKey) ? translate(hintKey) : ''
}
