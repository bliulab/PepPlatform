/**
 * WebGL capability probe for the landing hero.
 *
 * three.js r163+ requires WebGL2, so the 3D stage only mounts when a context can
 * actually be created; `LandingView` falls back to the 2D molecular lattice
 * otherwise. Guarded for non-browser environments (unit tests, SSR).
 */
export function supportsWebgl2(): boolean {
  if (typeof document === 'undefined') return false
  try {
    const canvas = document.createElement('canvas')
    const context = canvas.getContext('webgl2')
    if (!context) return false
    context.getExtension('WEBGL_lose_context')?.loseContext()
    return true
  } catch {
    return false
  }
}
