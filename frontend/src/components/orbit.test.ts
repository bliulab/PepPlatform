import { describe, expect, it } from 'vitest'

import {
  CLICK_SLOP_PX,
  ORBIT_PITCH_LIMIT,
  ORBIT_SENSITIVITY,
  TOUCH_PAN_THRESHOLD_PX,
  applyDrag,
  canOrbit,
  createOrbit,
  decayOrbit,
  isClick,
  orbitPose,
  pointerTravel,
  resolveTouchGesture,
  restOrbit,
} from './orbit'

describe('pointer orbit', () => {
  it('starts at rest', () => {
    const state = createOrbit()
    expect(state).toEqual({ yaw: 0, pitch: 0, velocity: { yaw: 0, pitch: 0 } })
  })

  it('only offers dragging to fine pointers without reduced motion', () => {
    expect(canOrbit()).toBe(true)
    expect(canOrbit({ reducedMotion: true })).toBe(false)
    expect(canOrbit({ coarsePointer: true })).toBe(false)
    expect(canOrbit({ reducedMotion: false, coarsePointer: false })).toBe(true)
  })

  it('classifies a touch travel before taking the gesture over', () => {
    expect(resolveTouchGesture({ dx: 2, dy: 1 })).toBe('pending')
    expect(resolveTouchGesture({ dx: 0, dy: 0 })).toBe('pending')
    expect(resolveTouchGesture({ dx: TOUCH_PAN_THRESHOLD_PX, dy: 0 })).toBe('drag')
    expect(resolveTouchGesture({ dx: -30, dy: 4 })).toBe('drag')
    expect(resolveTouchGesture({ dx: 3, dy: 24 })).toBe('scroll')
    expect(resolveTouchGesture({ dx: 9, dy: 9 })).toBe('scroll')
    expect(resolveTouchGesture({ dx: 12, dy: 0, threshold: 20 })).toBe('pending')
    expect(Number.isFinite(resolveTouchGesture({ dx: Number.NaN, dy: 1 }) ? 1 : 0)).toBe(true)
  })

  it('accumulates a drag without mutating the input state', () => {
    const state = createOrbit()
    const dragged = applyDrag(state, 100, 0)
    expect(state).toEqual(createOrbit())
    expect(dragged.yaw).toBeCloseTo(100 * ORBIT_SENSITIVITY, 10)
    expect(dragged.pitch).toBe(0)
    expect(dragged.velocity.yaw).toBeCloseTo(100 * ORBIT_SENSITIVITY, 10)
  })

  it('clamps pitch on both sides', () => {
    let state = createOrbit()
    for (let index = 0; index < 20; index += 1) state = applyDrag(state, 0, 400)
    expect(state.pitch).toBeCloseTo(ORBIT_PITCH_LIMIT, 10)
    for (let index = 0; index < 40; index += 1) state = applyDrag(state, 0, -400)
    expect(state.pitch).toBeCloseTo(-ORBIT_PITCH_LIMIT, 10)
  })

  it('bounds yaw only when the stage asks for it', () => {
    let unlimited = createOrbit()
    for (let index = 0; index < 10; index += 1) unlimited = applyDrag(unlimited, 800, 0)
    expect(unlimited.yaw).toBeGreaterThan(Math.PI)

    let bounded = createOrbit()
    for (let index = 0; index < 10; index += 1)
      bounded = applyDrag(bounded, 800, 0, { yawLimit: 1 })
    expect(bounded.yaw).toBeCloseTo(1, 10)
    for (let index = 0; index < 30; index += 1)
      bounded = applyDrag(bounded, -800, 0, { yawLimit: 1 })
    expect(bounded.yaw).toBeCloseTo(-1, 10)

    // Released inertia respects the same wall.
    const thrown = applyDrag(createOrbit(), 800, 0, { yawLimit: 1 })
    let settled = thrown
    for (let frame = 0; frame < 200; frame += 1)
      settled = decayOrbit(settled, 1 / 60, { yawLimit: 1 })
    expect(settled.yaw).toBeLessThanOrEqual(1 + 1e-9)
  })

  it('ignores non-finite pointer deltas', () => {
    const dragged = applyDrag(createOrbit(), Number.NaN, Number.POSITIVE_INFINITY)
    expect(dragged.yaw).toBe(0)
    expect(dragged.pitch).toBe(0)
    expect(Number.isNaN(dragged.velocity.yaw)).toBe(false)
  })

  it('decays released inertia to a full stop', () => {
    const thrown = applyDrag(createOrbit(), 120, 30)
    let state = thrown
    let previous = Math.abs(state.velocity.yaw)
    for (let frame = 0; frame < 400; frame += 1) {
      state = decayOrbit(state, 1 / 60)
      const current = Math.abs(state.velocity.yaw)
      expect(current).toBeLessThanOrEqual(previous + 1e-12)
      previous = current
    }
    expect(state.velocity.yaw).toBe(0)
    expect(state.velocity.pitch).toBe(0)
    // The throw keeps travelling past the release point.
    expect(state.yaw).toBeGreaterThan(thrown.yaw)
    expect(Math.abs(state.pitch)).toBeLessThanOrEqual(ORBIT_PITCH_LIMIT)
  })

  it('returns the same state object once settled', () => {
    const settled = { yaw: 1, pitch: 0.2, velocity: { yaw: 0, pitch: 0 } }
    expect(decayOrbit(settled, 1 / 60)).toBe(settled)
  })

  it('integrates across frame rates identically', () => {
    const start = applyDrag(createOrbit(), 90, 0)
    let sixty = start
    for (let frame = 0; frame < 30; frame += 1) sixty = decayOrbit(sixty, 1 / 60)
    let halved = start
    for (let frame = 0; frame < 15; frame += 1) halved = decayOrbit(halved, 1 / 30)
    // Same elapsed time (0.5s), different step counts: same angle travelled.
    expect(halved.yaw).toBeCloseTo(sixty.yaw, 6)
    expect(sixty.yaw).toBeGreaterThan(start.yaw)
    // A longer single step travels further than one 60 Hz frame, but not unbounded.
    const oneFrame = decayOrbit(start, 1 / 60)
    const oneStep = decayOrbit(start, 1 / 30)
    expect(oneStep.yaw).toBeGreaterThan(oneFrame.yaw)
  })

  it('caps the simulated stretch so a restored tab does not spin the model', () => {
    const start = applyDrag(createOrbit(), 90, 0)
    const stalled = decayOrbit(start, 5)
    const fourFrames = decayOrbit(applyDrag(createOrbit(), 90, 0), 4 / 60)
    expect(stalled.yaw).toBeCloseTo(fourFrames.yaw, 10)
  })

  it('drops inertia when the pointer grabs the model again', () => {
    const thrown = decayOrbit(applyDrag(createOrbit(), 90, 0), 1 / 60)
    const grabbed = restOrbit(thrown)
    expect(grabbed.yaw).toBe(thrown.yaw)
    expect(grabbed.velocity).toEqual({ yaw: 0, pitch: 0 })
    expect(restOrbit(grabbed)).toBe(grabbed)
  })

  it('reports the pose the camera should use', () => {
    const state = applyDrag(createOrbit(), 40, -20)
    expect(orbitPose(state)).toEqual({ yaw: state.yaw, pitch: state.pitch })
  })

  it('separates a click from a drag by pointer travel', () => {
    expect(pointerTravel({ x: 0, y: 0 }, { x: 3, y: 4 })).toBeCloseTo(5, 10)
    expect(isClick({ x: 0, y: 0 }, { x: 2, y: 2 })).toBe(true)
    expect(isClick({ x: 0, y: 0 }, { x: CLICK_SLOP_PX + 6, y: 0 })).toBe(false)
  })
})
