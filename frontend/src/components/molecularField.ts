/**
 * Deterministic geometry for the landing-page molecular field.
 *
 * The lattice is generated from a fixed seed, so one viewport always renders the
 * same layout (stable screenshots and reviews). Motion is a pure function of
 * time and pointer position, so it never accumulates drift and can be sampled
 * frame-by-frame by the canvas painter.
 */

export const MIN_NODES = 18
export const MAX_NODES = 84
export const NODE_AREA = 16000
export const EDGE_RADIUS_RATIO = 0.16
export const MAX_EDGES_PER_NODE = 3
export const ACCENT_NODE_COUNT = 4
export const DRIFT_AMPLITUDE_RATIO = 0.035
export const MAX_PARALLAX_PX = 6

const TAU = Math.PI * 2

export interface MolecularNode {
  x: number
  y: number
  radius: number
  phase: number
  /** 0.5–1 multiplier applied to the drift amplitude. */
  drift: number
  /** 0 (far) … 1 (near); scales radius, alpha and pointer parallax. */
  depth: number
  accent: boolean
}

export interface MolecularEdge {
  from: number
  to: number
}

export interface MolecularField {
  seed: number
  width: number
  height: number
  nodes: MolecularNode[]
  edges: MolecularEdge[]
}

export interface PointerState {
  x: number
  y: number
}

export interface SampledNode {
  x: number
  y: number
  radius: number
  alpha: number
  accent: boolean
}

export interface SampledEdge {
  x1: number
  y1: number
  x2: number
  y2: number
  alpha: number
}

export interface SampledMolecularField {
  nodes: SampledNode[]
  edges: SampledEdge[]
  parallax: { x: number; y: number }
}

function clamp(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}

/** mulberry32 — small deterministic PRNG, identical output for identical seeds. */
export function createRandom(seed: number) {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
}

export function clampPointer(pointer?: PointerState | null): PointerState {
  const finite = (value: number | undefined) => (Number.isFinite(value) ? (value as number) : 0)
  return {
    x: clamp(finite(pointer?.x), -1, 1),
    y: clamp(finite(pointer?.y), -1, 1),
  }
}

/** Node count grows with the stage area and stays inside `[MIN_NODES, MAX_NODES]`. */
export function nodeBudget(width: number, height: number) {
  const area = Math.max(0, width) * Math.max(0, height)
  return clamp(Math.round(area / NODE_AREA), MIN_NODES, MAX_NODES)
}

function markAccentNodes(nodes: MolecularNode[]) {
  if (nodes.length < ACCENT_NODE_COUNT) {
    for (const node of nodes) node.accent = true
    return
  }
  for (let slot = 0; slot < ACCENT_NODE_COUNT; slot += 1) {
    const ratio = (slot + 0.5) / ACCENT_NODE_COUNT
    const index = clamp(Math.floor(ratio * nodes.length), 0, nodes.length - 1)
    nodes[index].accent = true
  }
}

function buildEdges(nodes: MolecularNode[], width: number, height: number): MolecularEdge[] {
  const radius = EDGE_RADIUS_RATIO * Math.min(width, height)
  const edges: MolecularEdge[] = []
  if (!(radius > 0)) return edges

  const degree = nodes.map(() => 0)
  const seen = new Set<string>()

  for (let from = 0; from < nodes.length; from += 1) {
    if (degree[from] >= MAX_EDGES_PER_NODE) continue
    const candidates = nodes
      .map((node, index) => ({
        index,
        distance: Math.hypot(node.x - nodes[from].x, node.y - nodes[from].y),
      }))
      .filter((candidate) => candidate.index !== from && candidate.distance <= radius)
      .sort((a, b) => a.distance - b.distance || a.index - b.index)

    for (const candidate of candidates) {
      if (degree[from] >= MAX_EDGES_PER_NODE) break
      if (degree[candidate.index] >= MAX_EDGES_PER_NODE) continue
      const key =
        from < candidate.index ? `${from}:${candidate.index}` : `${candidate.index}:${from}`
      if (seen.has(key)) continue
      seen.add(key)
      degree[from] += 1
      degree[candidate.index] += 1
      edges.push({ from, to: candidate.index })
    }
  }

  return edges
}

export function createMolecularField(options: {
  seed: number
  width: number
  height: number
}): MolecularField {
  const width = Math.max(0, options.width)
  const height = Math.max(0, options.height)
  const count = nodeBudget(width, height)
  const random = createRandom(options.seed)
  const columns = Math.max(1, Math.round(Math.sqrt((count * width) / Math.max(1, height))))
  const rows = Math.max(1, Math.ceil(count / columns))
  const cellWidth = width / columns
  const cellHeight = height / rows
  const padding = Math.min(width, height) * 0.04
  const nodes: MolecularNode[] = []

  for (let index = 0; index < count; index += 1) {
    const column = index % columns
    const row = Math.floor(index / columns)
    const jitterX = (random() - 0.5) * cellWidth * 0.7
    const jitterY = (random() - 0.5) * cellHeight * 0.7
    const depth = random()
    nodes.push({
      x: clamp((column + 0.5) * cellWidth + jitterX, padding, Math.max(padding, width - padding)),
      y: clamp((row + 0.5) * cellHeight + jitterY, padding, Math.max(padding, height - padding)),
      radius: 0.7 + depth * 1.4,
      phase: random() * TAU,
      drift: 0.5 + random() * 0.5,
      depth,
      accent: false,
    })
  }

  markAccentNodes(nodes)
  return { seed: options.seed, width, height, nodes, edges: buildEdges(nodes, width, height) }
}

/**
 * Pure sampler: returns node/edge positions for a given time and pointer.
 * `pointer` is normalised to `[-1, 1]`; parallax is clamped to `MAX_PARALLAX_PX`.
 */
export function sampleMolecularField(
  field: MolecularField,
  time: number,
  pointer?: PointerState | null,
): SampledMolecularField {
  const pointerState = clampPointer(pointer)
  const parallax = {
    x: pointerState.x * MAX_PARALLAX_PX,
    y: pointerState.y * MAX_PARALLAX_PX,
  }
  const seconds = Number.isFinite(time) ? time : 0
  const amplitude = DRIFT_AMPLITUDE_RATIO * Math.min(field.width, field.height)

  const nodes: SampledNode[] = field.nodes.map((node) => {
    const drift = amplitude * node.drift
    return {
      x: clamp(
        node.x + Math.sin(seconds * 0.22 + node.phase) * drift + parallax.x * node.depth,
        0,
        field.width,
      ),
      y: clamp(
        node.y + Math.cos(seconds * 0.18 + node.phase * 1.3) * drift + parallax.y * node.depth,
        0,
        field.height,
      ),
      radius: node.radius,
      alpha: 0.34 + node.depth * 0.56,
      accent: node.accent,
    }
  })

  const edges: SampledEdge[] = field.edges.map((edge) => {
    const from = nodes[edge.from]
    const to = nodes[edge.to]
    return {
      x1: from.x,
      y1: from.y,
      x2: to.x,
      y2: to.y,
      alpha: 0.45 + 0.35 * Math.min(from.alpha, to.alpha),
    }
  })

  return { nodes, edges, parallax }
}

/** Stable signature of a layout — used by tests and by reviewers to spot drift. */
export function fieldSignature(field: MolecularField) {
  return `${field.nodes.length}/${field.edges.length}|${field.nodes
    .map((node) => `${node.x.toFixed(3)},${node.y.toFixed(3)},${node.depth.toFixed(3)}`)
    .join(';')}`
}
