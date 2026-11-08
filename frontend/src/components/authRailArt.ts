/**
 * Line-art for the authentication brand rail.
 *
 * Deterministic SVG traces avoid a WebGL context. `AuthShell.vue` renders the
 * returned paths; the geometry can be tested in Node.
 */

import { createRandom } from './molecularField'

export interface RailPoint {
  x: number
  y: number
}

export interface RailOptions {
  /** Number of traces. */
  count?: number
  /** Samples per trace; more samples, smoother line. */
  samples?: number
  /** Deterministic seed. */
  seed?: number
}

/** The SVG viewBox every coordinate is clamped into. */
export const RAIL_VIEWBOX = { width: 320, height: 180 }
export const RAIL_SEED = 20261023
export const RAIL_RIBBON_COUNT = 3
/** Default spacing between sampled nodes along a trace. */
export const RAIL_NODE_SPACING = 7

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function round1(value: number) {
  return Math.round(value * 10) / 10
}

/**
 * Three (by default) horizontal traces across the viewBox, each a sine with its
 * own amplitude, phase and frequency, plus a slight linear drift so the family
 * does not look like a copy-paste. Everything stays inside the viewBox.
 */
export function railRibbons(options: RailOptions = {}): RailPoint[][] {
  const count = Math.max(1, Math.round(options.count ?? RAIL_RIBBON_COUNT))
  const samples = Math.max(4, Math.round(options.samples ?? 48))
  const random = createRandom(options.seed ?? RAIL_SEED)
  const innerWidth = RAIL_VIEWBOX.width - 12
  return Array.from({ length: count }, (_, ribbon) => {
    const base = RAIL_VIEWBOX.height * (0.3 + (0.4 * ribbon) / Math.max(1, count - 1))
    const amplitude = 9 + random() * 6
    const phase = random() * Math.PI * 2
    const frequency = 1.3 + random() * 0.9
    const drift = (random() - 0.5) * 10
    return Array.from({ length: samples }, (_, index) => {
      const t = index / (samples - 1)
      return {
        x: 6 + t * innerWidth,
        y: clamp(
          base + drift * t + Math.sin(phase + t * Math.PI * 2 * frequency) * amplitude,
          4,
          RAIL_VIEWBOX.height - 4,
        ),
      }
    })
  })
}

/** SVG path for a trace: `M x y L x y ...`, rounded to one decimal. */
export function railPath(points: RailPoint[]): string {
  if (points.length === 0) return ''
  const [first, ...rest] = points.map((point) => `${round1(point.x)} ${round1(point.y)}`)
  return `M ${first}${rest.map((point) => ` L ${point}`).join('')}`
}

/**
 * Every `spacing`-th point of a trace, always including both ends, so nodes land
 * on the same visual rhythm regardless of how many samples were drawn.
 */
export function railNodes(points: RailPoint[], spacing = RAIL_NODE_SPACING): RailPoint[] {
  if (points.length === 0) return []
  const step = Math.max(1, Math.round(spacing))
  const nodes: RailPoint[] = []
  for (let index = 0; index < points.length; index += step) nodes.push(points[index])
  const last = points[points.length - 1]
  if (nodes[nodes.length - 1] !== last) nodes.push(last)
  return nodes.map((point) => ({ x: round1(point.x), y: round1(point.y) }))
}
