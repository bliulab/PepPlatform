/**
 * Pointer-orbit math with closed-form inertia and capped simulation steps.
 * Velocities use radians per 60 Hz frame; each update is capped by
 * ORBIT_MAX_FRAMES to prevent jumps after a suspended tab resumes.
 */

export interface OrbitVelocity {
  yaw: number
  pitch: number
}

export interface OrbitState {
  yaw: number
  pitch: number
  velocity: OrbitVelocity
}

export interface OrbitOptions {
  /** Radians of rotation per pixel dragged. */
  sensitivity?: number
  /** Hard limit for pitch, in radians, on both sides. */
  pitchLimit?: number
  /** Hard limit for yaw, in radians, on both sides. */
  yawLimit?: number
}

export interface OrbitInput {
  reducedMotion?: boolean
  coarsePointer?: boolean
}

/** Radians per dragged pixel. ~0.005 keeps a 300px drag under a quarter turn. */
export const ORBIT_SENSITIVITY = 0.005
/** Stop before the poles: past this the helix reads as a flat disc. */
export const ORBIT_PITCH_LIMIT = 0.5
/** Yaw is free by default; a stage can bound it so the model stays off its copy. */
export const ORBIT_YAW_LIMIT = Number.POSITIVE_INFINITY
/** Fraction of the velocity that survives one 60 Hz frame. */
export const ORBIT_DECAY_PER_FRAME = 0.94
/** Angular speed below which the inertia is considered settled. */
export const ORBIT_SETTLE_EPSILON = 1e-4
/** Pointer travel (px) under which a press/release counts as a click. */
export const CLICK_SLOP_PX = 4
/** Travel (px) before a touch press is classified as drag or scroll. */
export const TOUCH_PAN_THRESHOLD_PX = 8
/** Longest simulated stretch, so a restored tab does not spin the model. */
export const ORBIT_MAX_FRAMES = 4

export interface OrbitPoint {
  x: number
  y: number
}

function clamp(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}

export function createOrbit(): OrbitState {
  return { yaw: 0, pitch: 0, velocity: { yaw: 0, pitch: 0 } }
}

/**
 * Dragging is only offered where it makes sense: a touch stage owns the gesture
 * for scrolling, and a visitor who asked for reduced motion should get a still
 * frame instead of a manipulator.
 */
export function canOrbit(input: OrbitInput = {}): boolean {
  return !input.reducedMotion && !input.coarsePointer
}

/** Accumulate one pointer delta into a new orbit state (input is not mutated). */
export function applyDrag(
  state: OrbitState,
  deltaX: number,
  deltaY: number,
  options: OrbitOptions = {},
): OrbitState {
  const sensitivity = options.sensitivity ?? ORBIT_SENSITIVITY
  const pitchLimit = options.pitchLimit ?? ORBIT_PITCH_LIMIT
  const yawLimit = options.yawLimit ?? ORBIT_YAW_LIMIT
  const dx = Number.isFinite(deltaX) ? deltaX : 0
  const dy = Number.isFinite(deltaY) ? deltaY : 0
  const stepYaw = dx * sensitivity
  const stepPitch = dy * sensitivity
  return {
    yaw: clamp(state.yaw + stepYaw, -yawLimit, yawLimit),
    pitch: clamp(state.pitch + stepPitch, -pitchLimit, pitchLimit),
    // The last delta is the release velocity: cheap, and it matches the feel of
    // a "throw" better than averaging the whole gesture.
    velocity: { yaw: stepYaw, pitch: stepPitch },
  }
}

/**
 * Advance the release inertia by `dtSeconds`. Returns the same object when the
 * inertia has already settled, so callers can skip a re-render.
 */
export function decayOrbit(
  state: OrbitState,
  dtSeconds: number,
  limits: OrbitOptions = {},
): OrbitState {
  const { yaw: vYaw, pitch: vPitch } = state.velocity
  if (vYaw === 0 && vPitch === 0) return state
  const dt = Number.isFinite(dtSeconds) ? clamp(dtSeconds, 0, 1) : 0
  if (dt === 0) return state

  const pitchLimit = limits.pitchLimit ?? ORBIT_PITCH_LIMIT
  const yawLimit = limits.yawLimit ?? ORBIT_YAW_LIMIT
  const frames = Math.min(dt * 60, ORBIT_MAX_FRAMES)
  const survival = Math.pow(ORBIT_DECAY_PER_FRAME, frames)
  // Closed-form sum of the geometric series: v * (1 - r) / (1 - decay).
  const travel = (1 - survival) / (1 - ORBIT_DECAY_PER_FRAME)
  const nextYawVelocity = Math.abs(vYaw * survival) < ORBIT_SETTLE_EPSILON ? 0 : vYaw * survival
  const nextPitchVelocity =
    Math.abs(vPitch * survival) < ORBIT_SETTLE_EPSILON ? 0 : vPitch * survival
  return {
    yaw: clamp(state.yaw + vYaw * travel, -yawLimit, yawLimit),
    pitch: clamp(state.pitch + vPitch * travel, -pitchLimit, pitchLimit),
    velocity: { yaw: nextYawVelocity, pitch: nextPitchVelocity },
  }
}

/** Drop any residual inertia, e.g. when the pointer grabs the model again. */
export function restOrbit(state: OrbitState): OrbitState {
  if (state.velocity.yaw === 0 && state.velocity.pitch === 0) return state
  return { yaw: state.yaw, pitch: state.pitch, velocity: { yaw: 0, pitch: 0 } }
}

/** Camera-facing view of the state; the scenes add their own scroll offsets. */
export function orbitPose(state: OrbitState): { yaw: number; pitch: number } {
  return { yaw: state.yaw, pitch: state.pitch }
}

export type TouchGesture = 'pending' | 'drag' | 'scroll'

/**
 * Classify a touch press by its travel so the stage can share the gesture with the
 * page: horizontal movement orbits the model, vertical movement stays a scroll.
 * Below the threshold the gesture is still undecided.
 */
export function resolveTouchGesture(input: {
  dx: number
  dy: number
  threshold?: number
}): TouchGesture {
  const threshold = input.threshold ?? TOUCH_PAN_THRESHOLD_PX
  const dx = Number.isFinite(input.dx) ? input.dx : 0
  const dy = Number.isFinite(input.dy) ? input.dy : 0
  if (Math.hypot(dx, dy) < threshold) return 'pending'
  return Math.abs(dx) > Math.abs(dy) ? 'drag' : 'scroll'
}

/** Total pointer travel between press and release, used to tell drag from click. */
export function pointerTravel(from: OrbitPoint, to: OrbitPoint): number {
  return Math.hypot(to.x - from.x, to.y - from.y)
}

export function isClick(from: OrbitPoint, to: OrbitPoint, slop = CLICK_SLOP_PX): boolean {
  return pointerTravel(from, to) <= slop
}
