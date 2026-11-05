import { describe, expect, it } from 'vitest'

import {
  ACCENT_NODE_COUNT,
  MAX_EDGES_PER_NODE,
  MAX_NODES,
  MAX_PARALLAX_PX,
  MIN_NODES,
  clampPointer,
  createMolecularField,
  fieldSignature,
  nodeBudget,
  sampleMolecularField,
} from './molecularField'

const VIEWPORT = { width: 960, height: 640 }

function build(seed = 42, width = VIEWPORT.width, height = VIEWPORT.height) {
  return createMolecularField({ seed, width, height })
}

describe('molecular field geometry', () => {
  it('is deterministic for a fixed seed and viewport', () => {
    expect(fieldSignature(build(7))).toBe(fieldSignature(build(7)))
  })

  it('changes layout when the seed changes', () => {
    expect(fieldSignature(build(7))).not.toBe(fieldSignature(build(8)))
  })

  it('keeps the node budget inside the documented range', () => {
    expect(nodeBudget(VIEWPORT.width, VIEWPORT.height)).toBeGreaterThanOrEqual(MIN_NODES)
    expect(nodeBudget(VIEWPORT.width, VIEWPORT.height)).toBeLessThanOrEqual(MAX_NODES)
    expect(nodeBudget(200, 140)).toBe(MIN_NODES)
    expect(nodeBudget(6000, 3000)).toBe(MAX_NODES)
    expect(nodeBudget(1440, 780)).toBeGreaterThanOrEqual(nodeBudget(720, 390))
  })

  it('places every node inside the stage', () => {
    const field = build()
    for (const node of field.nodes) {
      expect(node.x).toBeGreaterThanOrEqual(0)
      expect(node.y).toBeGreaterThanOrEqual(0)
      expect(node.x).toBeLessThanOrEqual(field.width)
      expect(node.y).toBeLessThanOrEqual(field.height)
    }
  })

  it('keeps drifted nodes inside the stage for a long session', () => {
    const field = build()
    for (let time = 0; time <= 3600; time += 7) {
      for (const node of sampleMolecularField(field, time, { x: 1, y: -1 }).nodes) {
        expect(node.x).toBeGreaterThanOrEqual(0)
        expect(node.y).toBeGreaterThanOrEqual(0)
        expect(node.x).toBeLessThanOrEqual(field.width)
        expect(node.y).toBeLessThanOrEqual(field.height)
      }
    }
  })

  it('produces valid edges only', () => {
    const field = build()
    const degree = field.nodes.map(() => 0)
    const seen = new Set<string>()
    for (const edge of field.edges) {
      expect(edge.from).not.toBe(edge.to)
      expect(edge.from).toBeGreaterThanOrEqual(0)
      expect(edge.to).toBeGreaterThanOrEqual(0)
      expect(edge.from).toBeLessThan(field.nodes.length)
      expect(edge.to).toBeLessThan(field.nodes.length)
      const key = edge.from < edge.to ? `${edge.from}:${edge.to}` : `${edge.to}:${edge.from}`
      expect(seen.has(key)).toBe(false)
      seen.add(key)
      degree[edge.from] += 1
      degree[edge.to] += 1
    }
    expect(Math.max(0, ...degree)).toBeLessThanOrEqual(MAX_EDGES_PER_NODE)
  })

  it('marks exactly four accent nodes', () => {
    const field = build()
    expect(field.nodes.filter((node) => node.accent)).toHaveLength(ACCENT_NODE_COUNT)
  })

  it('clamps pointer parallax to the documented maximum', () => {
    const field = build()
    expect(clampPointer({ x: 9, y: -9 })).toEqual({ x: 1, y: -1 })
    expect(clampPointer(undefined)).toEqual({ x: 0, y: 0 })
    expect(clampPointer({ x: Number.NaN, y: Number.NaN })).toEqual({ x: 0, y: 0 })

    const centred = sampleMolecularField(field, 3, { x: 0, y: 0 })
    expect(centred.parallax).toEqual({ x: 0, y: 0 })

    const pushed = sampleMolecularField(field, 3, { x: 5, y: -5 })
    expect(Math.abs(pushed.parallax.x)).toBeLessThanOrEqual(MAX_PARALLAX_PX)
    expect(Math.abs(pushed.parallax.y)).toBeLessThanOrEqual(MAX_PARALLAX_PX)
  })

  it('moves over time without a pointer', () => {
    const field = build()
    const start = sampleMolecularField(field, 0).nodes
    const later = sampleMolecularField(field, 6).nodes
    const moved = start.some((node, index) => Math.abs(node.x - later[index].x) > 0.01)
    expect(moved).toBe(true)
  })

  it('survives degenerate viewports', () => {
    for (const [width, height] of [
      [0, 0],
      [1, 1],
      [8, 600],
      [4000, 2000],
    ]) {
      const field = build(3, width, height)
      expect(field.nodes.length).toBeGreaterThanOrEqual(MIN_NODES)
      expect(sampleMolecularField(field, 12).nodes).toHaveLength(field.nodes.length)
    }
  })
})
