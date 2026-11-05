import { describe, expect, it } from 'vitest'

import {
  RAIL_NODE_SPACING,
  RAIL_RIBBON_COUNT,
  RAIL_VIEWBOX,
  railNodes,
  railPath,
  railRibbons,
} from './authRailArt'

describe('auth rail line art', () => {
  it('is deterministic', () => {
    expect(railRibbons()).toEqual(railRibbons())
    expect(railRibbons()).toHaveLength(RAIL_RIBBON_COUNT)
  })

  it('keeps every trace inside the viewBox', () => {
    for (const ribbon of railRibbons()) {
      expect(ribbon.length).toBeGreaterThan(4)
      for (const point of ribbon) {
        expect(point.x).toBeGreaterThanOrEqual(0)
        expect(point.x).toBeLessThanOrEqual(RAIL_VIEWBOX.width)
        expect(point.y).toBeGreaterThanOrEqual(0)
        expect(point.y).toBeLessThanOrEqual(RAIL_VIEWBOX.height)
      }
      // Traces run left to right, so the family reads as a field, not a scribble.
      for (let index = 1; index < ribbon.length; index += 1) {
        expect(ribbon[index].x).toBeGreaterThan(ribbon[index - 1].x)
      }
    }
  })

  it('separates the traces vertically', () => {
    const [first, second, third] = railRibbons().map(
      (ribbon) => ribbon.reduce((total, point) => total + point.y, 0) / ribbon.length,
    )
    expect(first).toBeLessThan(second)
    expect(second).toBeLessThan(third)
  })

  it('renders a trace as a rounded SVG path', () => {
    const path = railPath([
      { x: 1.24, y: 2.06 },
      { x: 3, y: 4 },
    ])
    expect(path).toBe('M 1.2 2.1 L 3 4')
    expect(railPath([])).toBe('')
    expect(railPath([{ x: 0, y: 0 }])).toBe('M 0 0')
  })

  it('samples nodes on a fixed rhythm including both ends', () => {
    const ribbon = railRibbons({ count: 1, samples: 30 })[0]
    const nodes = railNodes(ribbon)
    expect(nodes[0]).toEqual({
      x: Math.round(ribbon[0].x * 10) / 10,
      y: Math.round(ribbon[0].y * 10) / 10,
    })
    expect(nodes[nodes.length - 1]).toEqual({
      x: Math.round(ribbon[ribbon.length - 1].x * 10) / 10,
      y: Math.round(ribbon[ribbon.length - 1].y * 10) / 10,
    })
    expect(nodes).toHaveLength(Math.ceil(30 / RAIL_NODE_SPACING) + 1)
    expect(railNodes([])).toEqual([])
    expect(railNodes(ribbon, 1)).toHaveLength(30)
  })
})
