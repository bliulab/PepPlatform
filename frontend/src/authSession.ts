type UnauthorizedHandler = () => void
let unauthorizedHandler: UnauthorizedHandler | undefined

export function setUnauthorizedHandler(handler: UnauthorizedHandler) {
  unauthorizedHandler = handler
}

/** Login errors and the anonymous session probe must remain on their own pages. */
export function notifyUnauthorized(path: string, status: number) {
  if (status !== 401) return
  const pathname = path.split('?')[0]
  if (
    /^\/api\/auth\/(me|login|logout|register|verify-email|resend-verification|password-reset\/.*)$/.test(
      pathname,
    )
  )
    return
  unauthorizedHandler?.()
}
