import { describe, expect, it } from 'vitest'

import {
  CAMERA_FOV,
  CAMERA_PITCH_LIMIT,
  CAMERA_TARGET,
  CAMERA_Z_FAR,
  CAMERA_Z_NEAR,
  DEFAULT_HELIX,
  DISSOLVE_STAGGER,
  FLOW_RADIUS_RATIO,
  FLOW_SPEED,
  FOCUSED_SPIN_SCALE,
  FOCUS_DOLLY,
  IDLE_SPIN_RATE,
  SPIN_RESUME_SECONDS,
  STATION_COUNT,
  backboneFlow,
  helixBackbone,
  helixLayout,
  heroProgress,
  latticeTargets,
  ligandPositions,
  cameraPose,
  dissolvePose,
  nearestStationIndex,
  sceneOpacity,
  sceneQuality,
  sphereBudget,
  spinScale,
  stationIndices,
  stationPositions,
  type Vec3,
} from './peptideScene'

function distance(a: Vec3, b: Vec3) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)
}

describe('peptide scene geometry', () => {
  it('builds a deterministic helix', () => {
    const first = helixLayout()
    const second = helixLayout()
    expect(first).toHaveLength(DEFAULT_HELIX.count)
    expect(first).toEqual(second)
    for (let index = 0; index < first.length; index += 1) {
      expect(first[index]).toEqual(second[index])
    }
  })

  it('keeps residues inside the helix envelope', () => {
    const points = helixLayout()
    const maxRadius = DEFAULT_HELIX.radius + DEFAULT_HELIX.jitter
    const halfHeight = DEFAULT_HELIX.height / 2
    for (const point of points) {
      const radial = Math.hypot(point.x, point.z)
      expect(radial).toBeGreaterThan(DEFAULT_HELIX.radius - DEFAULT_HELIX.jitter - 1e-9)
      expect(radial).toBeLessThanOrEqual(maxRadius + 1e-9)
      expect(Math.abs(point.y)).toBeLessThanOrEqual(halfHeight + 1e-9)
    }
  })

  it('advances monotonically along the helix axis', () => {
    const points = helixLayout()
    for (let index = 1; index < points.length; index += 1) {
      expect(points[index].y).toBeGreaterThan(points[index - 1].y)
    }
  })

  it('keeps the backbone on the helix radius', () => {
    const backbone = helixBackbone()
    expect(backbone.length).toBe(160)
    for (const point of backbone) {
      expect(Math.hypot(point.x, point.z)).toBeCloseTo(DEFAULT_HELIX.radius, 6)
    }
  })

  it('places ligands outside the helix envelope', () => {
    const ligands = ligandPositions()
    expect(ligands).toHaveLength(3)
    for (const ligand of ligands) {
      expect(Math.hypot(ligand.x, ligand.z)).toBeGreaterThan(DEFAULT_HELIX.radius)
    }
  })

  it('produces deterministic lattice targets inside the bounds', () => {
    const bounds = { width: 7, height: 5 }
    const first = latticeTargets(48, bounds)
    const second = latticeTargets(48, bounds)
    expect(first).toEqual(second)
    expect(first).toHaveLength(48)
    for (const point of first) {
      expect(Math.abs(point.x)).toBeLessThanOrEqual(bounds.width / 2)
      expect(Math.abs(point.y)).toBeLessThanOrEqual(bounds.height / 2)
      expect(Math.abs(point.z)).toBeLessThanOrEqual(0.7)
    }
  })

  it('lands exactly on home and target at the dissolve endpoints', () => {
    const home = { x: 1, y: 2, z: 3 }
    const target = { x: -4, y: 0.5, z: -1.5 }
    for (let index = 0; index < 12; index += 1) {
      expect(dissolvePose(home, target, 0, index)).toEqual(home)
      expect(dissolvePose(home, target, 1, index)).toEqual(target)
    }
  })

  it('staggers the dissolve so residues do not move in lockstep', () => {
    const home = { x: 0, y: 0, z: 0 }
    const target = { x: 10, y: 0, z: 0 }
    const early = dissolvePose(home, target, 0.4, 0).x
    const late = dissolvePose(home, target, 0.4, 11).x
    expect(early).toBeGreaterThan(late)
    expect(DISSOLVE_STAGGER).toBeGreaterThan(0)
  })

  it('clamps dissolve progress outside 0..1', () => {
    const home = { x: 0, y: 0, z: 0 }
    const target = { x: 4, y: 4, z: 4 }
    expect(dissolvePose(home, target, -3, 0)).toEqual(home)
    expect(dissolvePose(home, target, 4, 0)).toEqual(target)
    expect(dissolvePose(home, target, Number.NaN, 0)).toEqual(home)
  })

  it('dollies the camera within the designed range and clamps the pointer', () => {
    const near = cameraPose(0, { x: 0, y: 0 })
    const far = cameraPose(1, { x: 0, y: 0 })
    const nearDistance = Math.hypot(near.position.x, near.position.y, near.position.z)
    const farDistance = Math.hypot(far.position.x, far.position.y, far.position.z)
    expect(nearDistance).toBeCloseTo(CAMERA_Z_NEAR, 6)
    expect(farDistance).toBeCloseTo(CAMERA_Z_FAR, 6)
    expect(near.fov).toBe(CAMERA_FOV)

    const wild = cameraPose(1, { x: 40, y: -40 })
    const clamped = cameraPose(1, { x: 1, y: -1 })
    expect(wild.position).toEqual(clamped.position)

    const beyond = cameraPose(9, { x: 0, y: 0 })
    expect(beyond.position).toEqual(far.position)
  })

  it('reports finite scene layer opacity across the scroll range', () => {
    for (let step = 0; step <= 20; step += 1) {
      const p = step / 20
      for (const layer of ['tube', 'lattice', 'ligand'] as const) {
        const value = sceneOpacity(p, layer)
        expect(Number.isFinite(value)).toBe(true)
        expect(value).toBeGreaterThanOrEqual(0)
        expect(value).toBeLessThanOrEqual(1)
      }
    }
    expect(sceneOpacity(0, 'tube')).toBe(1)
    expect(sceneOpacity(1, 'lattice')).toBe(1)
    expect(sceneOpacity(1, 'tube')).toBe(0)
  })

  it('maps scroll position onto hero progress with safe degenerate cases', () => {
    expect(heroProgress(0, 900, 1980)).toBe(0)
    expect(heroProgress(1080, 900, 1980)).toBe(1)
    expect(heroProgress(540, 900, 1980)).toBeCloseTo(0.5, 6)
    expect(heroProgress(5000, 900, 1980)).toBe(1)
    expect(heroProgress(-200, 900, 1980)).toBe(0)
    expect(heroProgress(500, 900, 900)).toBe(0)
    expect(heroProgress(Number.NaN, 900, 1980)).toBe(0)
  })

  it('scales the residue budget down on mobile', () => {
    expect(sphereBudget(1440, false)).toBe(DEFAULT_HELIX.count)
    expect(sphereBudget(390, true)).toBeLessThan(sphereBudget(1440, false))
    expect(sphereBudget(320, true)).toBeGreaterThanOrEqual(12)

    const mobile = sceneQuality(true, 390)
    const desktop = sceneQuality(false, 1440)
    expect(mobile.bloom).toBe(false)
    expect(mobile.pixelRatio).toBe(1)
    expect(desktop.bloom).toBe(true)
    expect(desktop.pixelRatio).toBeLessThanOrEqual(1.5)
    expect(desktop.spheres).toBeGreaterThan(mobile.spheres)
  })

  it('spreads the four stations evenly along the sequence', () => {
    const desktop = stationIndices(DEFAULT_HELIX.count)
    expect(desktop).toHaveLength(STATION_COUNT)
    expect(desktop).toEqual(stationIndices(DEFAULT_HELIX.count))
    expect(new Set(desktop).size).toBe(STATION_COUNT)
    for (const index of desktop) {
      expect(index).toBeGreaterThanOrEqual(0)
      expect(index).toBeLessThan(DEFAULT_HELIX.count)
    }
    for (let slot = 1; slot < desktop.length; slot += 1) {
      expect(desktop[slot]).toBeGreaterThan(desktop[slot - 1])
      // Even quarters: no gap is wildly larger than the others.
      expect(desktop[slot] - desktop[slot - 1]).toBeGreaterThan(2)
    }

    const mobile = stationIndices(26)
    expect(mobile).toHaveLength(STATION_COUNT)
    expect(new Set(mobile).size).toBe(STATION_COUNT)
    expect(Math.max(...mobile)).toBeLessThan(26)
  })

  it('keeps the stations on the helix envelope', () => {
    const positions = stationPositions()
    const layout = helixLayout()
    const expected = stationIndices().map((index) => layout[index])
    expect(positions).toEqual(expected)
    const maxRadius = DEFAULT_HELIX.radius + DEFAULT_HELIX.jitter
    for (const point of positions) {
      expect(Math.hypot(point.x, point.z)).toBeLessThanOrEqual(maxRadius + 1e-9)
      expect(Math.abs(point.y)).toBeLessThanOrEqual(DEFAULT_HELIX.height / 2 + 1e-9)
    }
    for (let slot = 1; slot < positions.length; slot += 1) {
      // Consecutive stations are far apart, so four markers never collapse into one.
      expect(distance(positions[slot], positions[slot - 1])).toBeGreaterThan(0.5)
    }
  })

  it('focuses the camera on a station without moving the unfocused pose', () => {
    const unfocused = cameraPose(0.2, { x: 0.3, y: -0.2 })
    expect(unfocused.target).toEqual(CAMERA_TARGET)
    const zero = cameraPose(0.2, { x: 0.3, y: -0.2 }, { target: { x: 1, y: 2, z: 3 }, amount: 0 })
    expect(zero).toEqual(unfocused)

    const station = stationPositions()[1]
    const focused = cameraPose(0, { x: 0, y: 0 }, { target: station, amount: 1 })
    expect(focused.target).toEqual(station)
    const nearDistance = Math.hypot(focused.position.x, focused.position.y, focused.position.z)
    expect(nearDistance).toBeCloseTo(CAMERA_Z_NEAR * (1 - FOCUS_DOLLY), 6)

    const half = cameraPose(0, { x: 0, y: 0 }, { target: station, amount: 0.5 })
    expect(half.target.x).toBeCloseTo(station.x / 2 + CAMERA_TARGET.x / 2, 6)
    expect(half.target.y).toBeCloseTo(station.y / 2 + CAMERA_TARGET.y / 2, 6)
  })

  it('streams backbone packets deterministically along the helix', () => {
    const first = backboneFlow(60, 0)
    expect(first).toHaveLength(60)
    expect(backboneFlow(60, 0)).toEqual(first)

    const radius = DEFAULT_HELIX.radius * FLOW_RADIUS_RATIO
    for (const point of first) {
      expect(Math.hypot(point.x, point.z)).toBeCloseTo(radius, 8)
      expect(Math.abs(point.y)).toBeLessThanOrEqual(DEFAULT_HELIX.height / 2 + 1e-9)
    }

    // A packet advances exactly one slot per 1 / (count * FLOW_SPEED) seconds.
    const step = 1 / (60 * FLOW_SPEED)
    const later = backboneFlow(60, step)
    expect(later[0].x).toBeCloseTo(first[1].x, 8)
    expect(later[0].y).toBeCloseTo(first[1].y, 8)
    expect(later[59].x).toBeCloseTo(first[0].x, 8)

    // Negative and non-finite times stay inside the helix instead of NaN.
    expect(Number.isFinite(backboneFlow(60, -3)[0].y)).toBe(true)
    expect(Number.isFinite(backboneFlow(60, Number.NaN)[0].y)).toBe(true)
  })

  it('adds drag orbit on top of the scroll and pointer offsets', () => {
    const stage = cameraPose(0, { x: 0, y: 0 })
    expect(cameraPose(0, { x: 0, y: 0 }, null, { yaw: 0, pitch: 0 })).toEqual(stage)
    expect(cameraPose(0, { x: 0, y: 0 }, null, null)).toEqual(stage)

    const dragged = cameraPose(0, { x: 0, y: 0 }, null, { yaw: 0.6, pitch: 0.2 })
    expect(dragged.yaw).toBeCloseTo(stage.yaw + 0.6, 6)
    expect(dragged.pitch).toBeCloseTo(stage.pitch + 0.2, 6)
    // The camera stays on the orbit sphere: a dragged camera is a rotation, not a zoom.
    expect(Math.hypot(dragged.position.x, dragged.position.y, dragged.position.z)).toBeCloseTo(
      Math.hypot(stage.position.x, stage.position.y, stage.position.z),
      6,
    )

    const clamped = cameraPose(0, { x: 0, y: 0 }, null, { yaw: 0, pitch: 8 })
    expect(clamped.pitch).toBeCloseTo(CAMERA_PITCH_LIMIT, 6)
  })

  it('spins visibly at rest and settles back after an interaction', () => {
    // One full turn takes roughly 35 seconds.
    expect((Math.PI * 2) / IDLE_SPIN_RATE).toBeGreaterThan(30)
    expect((Math.PI * 2) / IDLE_SPIN_RATE).toBeLessThan(40)

    expect(spinScale(0)).toBeCloseTo(FOCUSED_SPIN_SCALE, 10)
    expect(spinScale(-1)).toBeCloseTo(FOCUSED_SPIN_SCALE, 10)
    expect(spinScale(SPIN_RESUME_SECONDS)).toBeCloseTo(1, 10)
    expect(spinScale(SPIN_RESUME_SECONDS * 4)).toBeCloseTo(1, 10)
    expect(Number.isFinite(spinScale(Number.NaN))).toBe(true)

    let previous = spinScale(0)
    for (let step = 1; step <= 12; step += 1) {
      const next = spinScale((step / 12) * SPIN_RESUME_SECONDS)
      expect(next).toBeGreaterThanOrEqual(previous - 1e-9)
      previous = next
    }
  })

  it('resolves the nearest station from projected screen points', () => {
    const points = [
      { x: 100, y: 100 },
      { x: 400, y: 100 },
      { x: 400, y: 400 },
      { x: 100, y: 400 },
    ]

    expect(nearestStationIndex(points, { x: 380, y: 130 })).toEqual({
      index: 1,
      distance: Math.hypot(20, 30),
    })
    // A pointer far from every station still resolves to exactly one of them.
    expect(nearestStationIndex(points, { x: 900, y: -400 })?.index).toBe(1)
    expect(nearestStationIndex([], { x: 0, y: 0 })).toBeNull()
    expect(nearestStationIndex([{ x: Number.NaN, y: 0 }], { x: 0, y: 0 })).toBeNull()
  })
})
