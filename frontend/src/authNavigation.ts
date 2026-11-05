export const authRoutePaths = new Set(['/login', '/register', '/verify-email', '/password-reset'])

/** A single return-address policy for every step of authentication. */
export function postAuthRedirect(value: unknown): string {
  const fallback = '/dashboard'
  if (typeof value !== 'string' || !value.startsWith('/')) return fallback
  try {
    const decoded = decodeURIComponent(value)
    if (decoded.startsWith('//') || /[\\\u0000-\u001f\u007f]/.test(decoded)) return fallback
    const url = new URL(value, 'https://pepplatform.invalid')
    if (url.origin !== 'https://pepplatform.invalid') return fallback
    const path = decodeURIComponent(url.pathname).replace(/\/+$/, '') || '/'
    if (authRoutePaths.has(path)) return fallback
    return `${url.pathname}${url.search}${url.hash}`
  } catch {
    return fallback
  }
}

export function loginLocation(returnTo: unknown) {
  return { path: '/login', query: { redirect: postAuthRedirect(returnTo) } }
}

const resourceQueryKeys = [
  'model',
  'ids',
  'clone',
  'plan_id',
  'reuse_plan_id',
  'source_experiment_id',
  'analysis_message_id',
]

export function allowsGuestPreview(to: {
  meta: { guestPreview?: unknown }
  params: Record<string, unknown>
  query: Record<string, unknown>
}) {
  return (
    Boolean(to.meta.guestPreview) &&
    !to.params.conversationId &&
    !resourceQueryKeys.some((key) => {
      const value = to.query[key]
      return Array.isArray(value) ? value.some(Boolean) : Boolean(value)
    })
  )
}
