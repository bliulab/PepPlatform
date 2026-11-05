/** Playback rules are independent of rendering, pointer events and browser clocks. */
export const INTRO_DURATION_MS = 2000
export const TASK_DWELL_MS = 7000
export const TASK_TRANSITION_MS = 800

export interface TaskPlayback {
  index: number
  elapsedMs: number
  transitionMs: number
  manual: boolean
}

export function createTaskPlayback(): TaskPlayback {
  return { index: 0, elapsedMs: 0, transitionMs: 0, manual: false }
}

export function selectTask(state: TaskPlayback, index: number, count = 4): TaskPlayback {
  const next = Math.max(0, Math.min(count - 1, Math.trunc(index)))
  return {
    index: Number.isFinite(next) ? next : 0,
    elapsedMs: 0,
    transitionMs: next === state.index ? state.transitionMs : TASK_TRANSITION_MS,
    manual: true,
  }
}

export function advanceTaskPlayback(
  state: TaskPlayback,
  dtMs: number,
  blocked: boolean,
  count = 4,
): TaskPlayback {
  if (blocked || state.manual || !Number.isFinite(dtMs) || dtMs <= 0 || count < 1) return state
  let remaining = dtMs
  let next = { ...state }
  // Consume transitions separately: every task gets seven full seconds to read.
  while (remaining > 0) {
    if (next.transitionMs > 0) {
      const step = Math.min(remaining, next.transitionMs)
      next.transitionMs -= step
      remaining -= step
    } else {
      const step = Math.min(remaining, TASK_DWELL_MS - next.elapsedMs)
      next.elapsedMs += step
      remaining -= step
      if (next.elapsedMs >= TASK_DWELL_MS) {
        next = {
          index: (next.index + 1) % count,
          elapsedMs: 0,
          transitionMs: TASK_TRANSITION_MS,
          manual: false,
        }
      }
    }
  }
  return next
}

export function introProgress(elapsedMs: number): number {
  if (!Number.isFinite(elapsedMs)) return 1
  return Math.min(1, Math.max(0, elapsedMs / INTRO_DURATION_MS))
}
