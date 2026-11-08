/** Deterministic particle geometry and critically damped interaction, without WebGL. */
export interface ParticlePoint {
  x: number
  y: number
  z: number
}
export interface ParticleFeedback {
  dt: number
  pointer: ParticlePoint | null
  radius: number
  displacement: number
  wave: { origin: number; age: number } | null
  pointScale: number
  instant?: boolean
}
export interface PeptideParticles {
  count: number
  home: Float32Array
  scattered: Float32Array
  positions: Float32Array
  velocities: Float32Array
  arc: Float32Array
  phase: Float32Array
  glow: Float32Array
  sizes: Float32Array
  initialized: boolean
}

const clamp = (n: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, n))
const smooth = (n: number) => {
  const p = clamp(n, 0, 1)
  return p * p * (3 - 2 * p)
}
const random = (i: number) => {
  const v = Math.sin(i * 127.1 + 31.7) * 43758.5453
  return v - Math.floor(v)
}

export function createPeptideParticles(count: number): PeptideParticles {
  const n = Math.max(1, Math.floor(count))
  const cloud: PeptideParticles = {
    count: n,
    home: new Float32Array(n * 3),
    scattered: new Float32Array(n * 3),
    positions: new Float32Array(n * 3),
    velocities: new Float32Array(n * 3),
    arc: new Float32Array(n),
    phase: new Float32Array(n),
    glow: new Float32Array(n),
    sizes: new Float32Array(n),
    initialized: false,
  }
  for (let i = 0; i < n; i++) {
    const s = (i + 0.5) / n
    const angle = s * Math.PI * 5.1
    const phase = random(i + 1) * Math.PI * 2
    // Dense tubular cloud plus small side-chain clusters; always a single chain.
    const thickness = (i % 7 === 0 ? 0.26 : 0.12) * Math.sqrt(random(i + 37))
    const radius = 1.14 + Math.cos(phase) * thickness
    const x = Math.cos(angle) * radius
    const y = (s - 0.5) * 6.5 + Math.sin(phase) * thickness
    const z = Math.sin(angle) * radius
    const roll = -0.28
    cloud.home[i * 3] = x * Math.cos(roll) - y * Math.sin(roll)
    cloud.home[i * 3 + 1] = x * Math.sin(roll) + y * Math.cos(roll)
    cloud.home[i * 3 + 2] = z
    const spread = 3.2 + random(i + 53) * 2.5
    cloud.scattered[i * 3] = Math.cos(phase + s * 9) * spread
    cloud.scattered[i * 3 + 1] = (random(i + 71) - 0.5) * 8.5
    cloud.scattered[i * 3 + 2] = Math.sin(phase + s * 9) * spread * 0.65
    cloud.arc[i] = s
    cloud.phase[i] = phase
    cloud.sizes[i] = 0.025 + random(i + 101) * 0.035 + (i % 23 === 0 ? 0.035 : 0)
  }
  cloud.positions.set(cloud.scattered)
  return cloud
}

export function nearestParticleArc(cloud: PeptideParticles, point: ParticlePoint): number {
  let nearest = Number.POSITIVE_INFINITY
  let arc = 0.5
  for (let i = 0; i < cloud.count; i++) {
    const p = i * 3
    const d =
      (cloud.positions[p] - point.x) ** 2 +
      (cloud.positions[p + 1] - point.y) ** 2 +
      (cloud.positions[p + 2] - point.z) ** 2
    if (d < nearest) {
      nearest = d
      arc = cloud.arc[i]
    }
  }
  return arc
}

/** Returns maximum deformation, useful for checking that feedback has settled. */
export function stepPeptideParticles(
  cloud: PeptideParticles,
  time: number,
  entrance: number,
  input: ParticleFeedback,
): number {
  const dt = Number.isFinite(input.dt) ? clamp(input.dt, 0, 0.25) : 0
  const omega = 12
  const decay = Math.exp(-omega * dt)
  const radius = Math.max(0.001, input.radius)
  let maximum = 0
  for (let i = 0; i < cloud.count; i++) {
    const p = i * 3
    const s = cloud.arc[i]
    const phase = cloud.phase[i]
    const progress = smooth((entrance - s * 0.12) / 0.78)
    const remaining = 1 - progress
    const twist = remaining * 2.4
    const sx = cloud.scattered[p] * Math.cos(twist) - cloud.scattered[p + 2] * Math.sin(twist)
    const sz = cloud.scattered[p] * Math.sin(twist) + cloud.scattered[p + 2] * Math.cos(twist)
    let x = cloud.home[p] * progress + sx * remaining
    let y = cloud.home[p + 1] * progress + cloud.scattered[p + 1] * remaining
    let z = cloud.home[p + 2] * progress + sz * remaining
    let proximity = 0
    if (input.pointer && entrance >= 0.8) {
      const dx = x - input.pointer.x
      const dy = y - input.pointer.y
      const dz = z - input.pointer.z
      // A shallow depth slab keeps the back of the rotating chain reachable.
      const distance = Math.hypot(dx, dy, dz * 0.35)
      proximity = Math.max(0, 1 - distance / radius)
      const force = proximity * proximity * input.displacement
      const length = Math.hypot(dx, dy) || 1
      x += (distance < 0.01 ? Math.cos(phase) : dx / length) * force
      y += (distance < 0.01 ? Math.sin(phase) : dy / length) * force
      z += Math.sin(phase) * force * 0.3
    }
    let wave = 0
    if (input.wave && input.wave.age >= 0 && input.wave.age < 1.25) {
      const distance = Math.abs(s - input.wave.origin) - input.wave.age * 0.85
      wave = Math.exp(-(distance * distance) / 0.0025) * (1 - input.wave.age / 1.25)
      x += Math.cos(phase) * wave * 0.2
      y += Math.sin(phase) * wave * 0.2
    }
    for (let axis = 0; axis < 3; axis++) {
      const k = p + axis
      const target = axis === 0 ? x : axis === 1 ? y : z
      if (!cloud.initialized || entrance < 1 || input.instant) {
        cloud.positions[k] = target
        cloud.velocities[k] = 0
      } else {
        // Closed-form critical damping is stable and frame-rate independent.
        const offset = cloud.positions[k] - target
        const c = cloud.velocities[k] + omega * offset
        cloud.positions[k] = target + (offset + c * dt) * decay
        cloud.velocities[k] = (cloud.velocities[k] - omega * c * dt) * decay
      }
      maximum = Math.max(maximum, Math.abs(cloud.positions[k] - cloud.home[k]))
    }
    const flow = Math.exp(-((s - ((time * 0.12) % 1)) ** 2) / 0.0018)
    cloud.glow[i] = clamp(0.12 + flow * 0.7 + proximity * 0.9 + wave * 1.5, 0, 1.7)
  }
  cloud.initialized = true
  return maximum
}
