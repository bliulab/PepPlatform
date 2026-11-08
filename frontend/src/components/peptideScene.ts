/**
 * Pure geometry, camera and scroll-progress math for the landing 3D stage.
 *
 * Deliberately free of any `three` import so it can be unit-tested in Node
 * (vitest runs without a DOM or WebGL). `PeptideScene.vue` only turns these
 * numbers into meshes and matrices.
 */

import { clampPointer, createRandom, type PointerState } from './molecularField'

export type { PointerState }

export interface Vec3 {
  x: number
  y: number
  z: number
}

export interface HelixOptions {
  count: number
  /** Helix radius in scene units. */
  radius: number
  /** Total height along Y. */
  height: number
  /** Full rotations over the height. */
  turns: number
  /** Deterministic radial jitter, in scene units. */
  jitter: number
}

export interface CameraPose {
  position: Vec3
  /** Where the camera should look: the stage centre, or a focused station. */
  target: Vec3
  yaw: number
  pitch: number
  fov: number
}

export interface CameraFocus {
  /** World-space point to zoom toward (a task station). */
  target: Vec3
  /** 0 = no focus, 1 = fully focused. */
  amount: number
}

/** Yaw/pitch added by pointer dragging, on top of the scroll and pointer offsets. */
export interface CameraOrbit {
  yaw: number
  pitch: number
}

export const HELIX_SEED = 20260214
export const DEFAULT_HELIX: HelixOptions = {
  count: 48,
  radius: 1.35,
  height: 6,
  turns: 3,
  jitter: 0.085,
}

/** Camera travel driven by the pinned hero scroll progress. */
export const CAMERA_Z_NEAR = 7
export const CAMERA_Z_FAR = 12.5
export const CAMERA_YAW_SWEEP = 0.5
/** Pointer parallax: the whole stage reacts, not just the four stations. */
export const POINTER_YAW = 0.16
export const POINTER_PITCH = 0.11
export const CAMERA_FOV = 38
/** Camera target when nothing is focused: slightly above the stage centre. */
export const CAMERA_TARGET: Vec3 = { x: 0, y: 0.1, z: 0 }
/** Fraction of the orbit distance removed at full focus. */
export const FOCUS_DOLLY = 0.15

/** Number of task-family stations marked on the helix (T01-T04). */
export const STATION_COUNT = 4

/**
 * Idle spin of the helix, in radians per second. The first pass used 0.06 (a full
 * turn every 105 s), which read as a still image; 0.18 is a turn every 35 s.
 */
export const IDLE_SPIN_RATE = 0.18
/** Spin multiplier while a station is focused: the visitor is reading, not watching. */
export const FOCUSED_SPIN_SCALE = 0.15
/** Seconds after an interaction before the idle spin has fully resumed. */
export const SPIN_RESUME_SECONDS = 1.2
/** Gentle idle sway around the X axis, so the model breathes between turns. */
export const IDLE_SWAY_AMPLITUDE = 0.025
export const IDLE_SWAY_RATE = 0.22
/**
 * Hover pick radius as a fraction of the stage's longest side. Anywhere inside
 * it resolves to the nearest station, so the stage no longer needs a direct hit
 * on a 0.12-unit sphere.
 */
export const STATION_PICK_RADIUS_RATIO = 0.34

/** Screen-space point, in CSS pixels relative to the stage origin. */
export interface ScreenPoint {
  x: number
  y: number
}

export interface StationPick {
  index: number
  distance: number
}

/**
 * Spin multiplier after an interaction: 0.15 while focused, easing back to 1 over
 * `SPIN_RESUME_SECONDS`. Frame-rate independent by construction (elapsed time in).
 */
export function spinScale(secondsSinceInteraction: number): number {
  const elapsed = Number.isFinite(secondsSinceInteraction)
    ? Math.max(0, secondsSinceInteraction)
    : 0
  return lerp(FOCUSED_SPIN_SCALE, 1, smoothstep(0, SPIN_RESUME_SECONDS, elapsed))
}

/**
 * Nearest station to a pointer position, in screen space. The helix spins and the
 * camera dollies, so this is recomputed from the projected station positions each
 * frame instead of raycasting four small spheres.
 */
export function nearestStationIndex(
  points: readonly ScreenPoint[],
  pointer: ScreenPoint,
): StationPick | null {
  let best: StationPick | null = null
  points.forEach((point, index) => {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return
    const distance = Math.hypot(point.x - pointer.x, point.y - pointer.y)
    if (!best || distance < best.distance) best = { index, distance }
  })
  return best
}

/** Fraction of the scroll range used by the per-residue dissolve stagger. */
export const DISSOLVE_STAGGER = 0.35

function clamp(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}

function smoothstep(edge0: number, edge1: number, value: number) {
  if (edge1 === edge0) return value < edge0 ? 0 : 1
  const t = clamp((value - edge0) / (edge1 - edge0), 0, 1)
  return t * t * (3 - 2 * t)
}

function lerp(from: number, to: number, t: number) {
  return from + (to - from) * t
}

function lerpVec3(from: Vec3, to: Vec3, t: number): Vec3 {
  return {
    x: lerp(from.x, to.x, t),
    y: lerp(from.y, to.y, t),
    z: lerp(from.z, to.z, t),
  }
}

/**
 * Residue anchors along an alpha helix. Deterministic: the same options always
 * produce the same positions, which keeps screenshots and reviews stable.
 */
export function helixLayout(options: HelixOptions = DEFAULT_HELIX): Vec3[] {
  const count = Math.max(1, Math.round(options.count))
  const random = createRandom(HELIX_SEED)
  const points: Vec3[] = []
  for (let index = 0; index < count; index += 1) {
    const t = count === 1 ? 0.5 : index / (count - 1)
    const angle = t * options.turns * Math.PI * 2
    const radius = options.radius + (random() - 0.5) * options.jitter * 2
    points.push({
      x: Math.cos(angle) * radius,
      y: (t - 0.5) * options.height,
      z: Math.sin(angle) * radius,
    })
  }
  return points
}

/** Backbone samples for the ribbon tube: the same helix, denser and smoother. */
export function helixBackbone(options: HelixOptions = DEFAULT_HELIX, samples = 160): Vec3[] {
  const points: Vec3[] = []
  const count = Math.max(2, Math.round(samples))
  for (let index = 0; index < count; index += 1) {
    const t = index / (count - 1)
    const angle = t * options.turns * Math.PI * 2
    points.push({
      x: Math.cos(angle) * options.radius,
      y: (t - 0.5) * options.height,
      z: Math.sin(angle) * options.radius,
    })
  }
  return points
}

/**
 * Residue indices that carry the four task-family stations. Even quarters of the
 * sequence, rounded to the nearest residue, so the markers read as a ruler along
 * the helix instead of four adjacent beads.
 */
export function stationIndices(count: number = DEFAULT_HELIX.count): number[] {
  const total = Math.max(STATION_COUNT, Math.round(count))
  const indices: number[] = []
  for (let slot = 0; slot < STATION_COUNT; slot += 1) {
    const index = Math.round(((slot + 0.5) / STATION_COUNT) * (total - 1))
    if (!indices.includes(index)) indices.push(index)
  }
  return indices
}

/** Station positions in the helix group's own coordinate space. */
export function stationPositions(options: HelixOptions = DEFAULT_HELIX): Vec3[] {
  const layout = helixLayout(options)
  return stationIndices(options.count).map((index) => layout[index])
}

/** Laps per second of the packet stream travelling up the backbone. */
export const FLOW_SPEED = 0.07
/** Packets ride just outside the tube so they do not Z-fight with the ribbon. */
export const FLOW_RADIUS_RATIO = 1.08

/**
 * Bright packets that stream along the backbone. Deterministic given `time`, and
 * phase-shifted by exactly one slot per `1 / (count * FLOW_SPEED)` seconds, which
 * is what makes the stream read as travel instead of flicker.
 */
export function backboneFlow(
  count: number,
  time: number,
  options: HelixOptions = DEFAULT_HELIX,
): Vec3[] {
  const total = Math.max(1, Math.round(count))
  const seconds = Number.isFinite(time) ? time : 0
  const points: Vec3[] = []
  for (let index = 0; index < total; index += 1) {
    const phase = (((index / total + seconds * FLOW_SPEED) % 1) + 1) % 1
    const angle = phase * options.turns * Math.PI * 2
    const radius = options.radius * FLOW_RADIUS_RATIO
    points.push({
      x: Math.cos(angle) * radius,
      y: (phase - 0.5) * options.height,
      z: Math.sin(angle) * radius,
    })
  }
  return points
}

/** Pocket ligands: three fixed, slightly offset anchors that read as bound peptides. */
export function ligandPositions(radius = DEFAULT_HELIX.radius): Vec3[] {
  return [
    { x: radius * 1.42, y: 1.55, z: radius * 0.42 },
    { x: -radius * 1.28, y: -0.35, z: radius * 0.72 },
    { x: radius * 0.38, y: -1.85, z: -radius * 1.18 },
  ]
}

/**
 * Planar lattice that the helix dissolves into. Spread inside `bounds` with
 * depth jitter so the result still reads as a 3D field after the morph.
 */
export function latticeTargets(
  count: number,
  bounds: { width: number; height: number },
  seed = HELIX_SEED,
): Vec3[] {
  const total = Math.max(1, Math.round(count))
  const random = createRandom(seed)
  const columns = Math.max(
    1,
    Math.round(Math.sqrt((total * bounds.width) / Math.max(1, bounds.height))),
  )
  const rows = Math.max(1, Math.ceil(total / columns))
  const cellWidth = bounds.width / columns
  const cellHeight = bounds.height / rows
  const targets: Vec3[] = []
  for (let index = 0; index < total; index += 1) {
    const column = index % columns
    const row = Math.floor(index / columns)
    targets.push({
      x: (column + 0.5) * cellWidth - bounds.width / 2 + (random() - 0.5) * cellWidth * 0.55,
      y: (row + 0.5) * cellHeight - bounds.height / 2 + (random() - 0.5) * cellHeight * 0.55,
      z: (random() - 0.5) * 1.4,
    })
  }
  return targets
}

/**
 * Per-residue dissolve. `progress` is the hero scroll progress; residues leave
 * the helix in a staggered wave so the morph reads as motion, not a fade.
 */
export function dissolvePose(home: Vec3, target: Vec3, progress: number, index: number): Vec3 {
  const p = clamp(progress, 0, 1)
  const slot = index % 12
  const offset = (slot / 12) * DISSOLVE_STAGGER
  const local = clamp((p - offset) / (1 - DISSOLVE_STAGGER), 0, 1)
  return lerpVec3(home, target, smoothstep(0, 1, local))
}

/** Combined pitch is allowed a little further than a bare drag before it flattens. */
export const CAMERA_PITCH_LIMIT = 0.9

/** Camera orbit + dolly for a given scroll progress and normalised pointer. */
export function cameraPose(
  progress: number,
  pointer?: PointerState | null,
  focus?: CameraFocus | null,
  orbit?: CameraOrbit | null,
): CameraPose {
  const p = clamp(progress, 0, 1)
  const pointerState = clampPointer(pointer)
  const eased = smoothstep(0, 1, p)
  const focusAmount = clamp(focus?.amount ?? 0, 0, 1)
  const z = lerp(CAMERA_Z_NEAR, CAMERA_Z_FAR, eased) * (1 - FOCUS_DOLLY * focusAmount)
  const yaw = CAMERA_YAW_SWEEP * eased + pointerState.x * POINTER_YAW + (orbit?.yaw ?? 0)
  const pitch = clamp(
    pointerState.y * POINTER_PITCH + (orbit?.pitch ?? 0),
    -CAMERA_PITCH_LIMIT,
    CAMERA_PITCH_LIMIT,
  )
  return {
    position: {
      x: Math.sin(yaw) * Math.cos(pitch) * z,
      y: -Math.sin(pitch) * z,
      z: Math.cos(yaw) * Math.cos(pitch) * z,
    },
    target:
      focus && focusAmount > 0 ? lerpVec3(CAMERA_TARGET, focus.target, focusAmount) : CAMERA_TARGET,
    yaw,
    pitch,
    fov: CAMERA_FOV,
  }
}

/**
 * Scroll progress of the pinned hero: 0 while the stage is fully in view, 1 once
 * the pin has been consumed. Degenerate inputs collapse to 0 instead of NaN.
 */
export function heroProgress(scrollY: number, viewportHeight: number, pinHeight: number): number {
  const travel = pinHeight - viewportHeight
  if (!Number.isFinite(travel) || travel <= 0) return 0
  return clamp(scrollY / travel, 0, 1)
}

/** Residue count for the current viewport; mobile halves the load. */
export function sphereBudget(viewportWidth: number, mobile: boolean): number {
  const base = mobile || viewportWidth < 720 ? 26 : DEFAULT_HELIX.count
  return clamp(Math.round(base), 12, DEFAULT_HELIX.count)
}

/** Render-quality knobs derived from the device profile. */
export function sceneQuality(mobile: boolean, viewportWidth: number) {
  return {
    /** Residues + lattice points. */
    spheres: sphereBudget(viewportWidth, mobile),
    pixelRatio: mobile ? 1 : Math.min(1.5, viewportWidth / 960),
    bloom: !mobile,
    antialias: !mobile,
  }
}

/** Opacity curves for the three scene layers across the scroll range. */
export function sceneOpacity(progress: number, layer: 'tube' | 'lattice' | 'ligand'): number {
  const p = clamp(progress, 0, 1)
  if (layer === 'tube') return 1 - smoothstep(0.35, 0.8, p)
  if (layer === 'lattice') return smoothstep(0.4, 0.95, p)
  return 1 - smoothstep(0.5, 0.85, p) * 0.65
}
