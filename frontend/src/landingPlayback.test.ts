import { describe, expect, it } from 'vitest'
import {
  advanceTaskPlayback,
  createTaskPlayback,
  introProgress,
  selectTask,
  TASK_DWELL_MS,
  TASK_TRANSITION_MS,
} from './landingPlayback'

describe('welcome playback', () => {
  it('gives each task a full dwell, accounts for transitions and wraps all four tasks', () => {
    let state = createTaskPlayback()
    state = advanceTaskPlayback(state, TASK_DWELL_MS, false)
    expect(state.index).toBe(1)
    expect(state.transitionMs).toBe(TASK_TRANSITION_MS)
    state = advanceTaskPlayback(state, TASK_TRANSITION_MS + TASK_DWELL_MS - 1, false)
    expect(state.index).toBe(1)
    state = advanceTaskPlayback(state, 1, false)
    expect(state.index).toBe(2)
    state = advanceTaskPlayback(state, 2 * (TASK_TRANSITION_MS + TASK_DWELL_MS), false)
    expect(state.index).toBe(0)
  })

  it('freezes elapsed time while offscreen, hidden, hovered or paused', () => {
    const state = advanceTaskPlayback(createTaskPlayback(), 3200, false)
    expect(advanceTaskPlayback(state, 60000, true)).toBe(state)
    expect(advanceTaskPlayback(state, 1000, false).elapsedMs).toBe(4200)
  })

  it('keeps manual selection fixed until explicitly resumed', () => {
    const state = selectTask(createTaskPlayback(), 3)
    expect(advanceTaskPlayback(state, 60000, false)).toBe(state)
    const resumed = advanceTaskPlayback(
      { ...state, manual: false },
      TASK_TRANSITION_MS + TASK_DWELL_MS,
      false,
    )
    expect(resumed.index).toBe(0)
    expect(selectTask(selectTask(state, 0), 2).index).toBe(2)
  })

  it('integrates the same visible time independently of frame rate', () => {
    let small = createTaskPlayback()
    for (let i = 0; i < 320; i++) small = advanceTaskPlayback(small, 50, false)
    expect(small).toEqual(advanceTaskPlayback(createTaskPlayback(), 16000, false))
    expect(advanceTaskPlayback(small, Number.NaN, false)).toBe(small)
  })

  it('finishes the opening at two seconds and safely clamps interruptions', () => {
    expect(introProgress(-1)).toBe(0)
    expect(introProgress(1000)).toBe(0.5)
    expect(introProgress(2000)).toBe(1)
    expect(introProgress(Number.NaN)).toBe(1)
  })
})
