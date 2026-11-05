import { describe, expect, it } from 'vitest'

import {
  ANCHOR_INDICES,
  GROOVE_HELIX_OFFSET,
  GROOVE_HELIX_RADIUS,
  GROOVE_LENGTH,
  GROOVE_SAMPLES,
  GROOVE_STRANDS,
  PEPTIDE_RESIDUES,
  anchorIndices,
  grooveBounds,
  grooveHelices,
  grooveSheet,
  grooveSheetMesh,
  peptideChain,
} from './hlaiGroove'

const BOX = { x: 4.2, y: 1, z: 1.6 }

function expectInsideBox(points: { x: number; y: number; z: number }[], label: string) {
  for (const point of points) {
    expect(Math.abs(point.x), `${label} x`).toBeLessThanOrEqual(BOX.x)
    expect(Math.abs(point.y), `${label} y`).toBeLessThanOrEqual(BOX.y)
    expect(Math.abs(point.z), `${label} z`).toBeLessThanOrEqual(BOX.z)
  }
}

describe('peptide-HLA-I groove geometry', () => {
  it('is deterministic', () => {
    expect(grooveHelices()).toEqual(grooveHelices())
    expect(grooveSheet()).toEqual(grooveSheet())
    expect(peptideChain()).toEqual(peptideChain())
  })

  it('draws two alpha helices that stay inside the stage box', () => {
    const helices = grooveHelices()
    expect(helices).toHaveLength(2)
    expect(helices[0]).toHaveLength(GROOVE_SAMPLES)
    expect(helices[1]).toHaveLength(GROOVE_SAMPLES)
    expectInsideBox(helices.flat(), 'helix')

    // The two strands sit on opposite sides of the groove, never overlapping.
    for (let index = 0; index < GROOVE_SAMPLES; index += 1) {
      expect(helices[0][index].z).toBeLessThan(0)
      expect(helices[1][index].z).toBeGreaterThan(0)
      expect(helices[0][index].x).toBeCloseTo(helices[1][index].x, 10)
      // Each spine stays near its offset, inside one coil radius.
      expect(Math.abs(helices[1][index].z - GROOVE_HELIX_OFFSET)).toBeLessThanOrEqual(
        GROOVE_HELIX_RADIUS + 1e-9,
      )
    }

    // The helices rise toward the ends: the groove has walls.
    expect(Math.abs(helices[1][0].y)).toBeGreaterThan(
      Math.abs(helices[1][Math.floor(GROOVE_SAMPLES / 2)].y),
    )
    expect(Math.abs(helices[1][0].x)).toBeCloseTo(GROOVE_LENGTH / 2, 6)
  })

  it('builds an eight-strand beta sheet floor below the peptide', () => {
    const sheet = grooveSheet()
    expect(sheet).toHaveLength(GROOVE_STRANDS)
    for (const strand of sheet) expect(strand).toHaveLength(9)
    expectInsideBox(sheet.flat(), 'sheet')

    for (const point of sheet.flat()) expect(point.y).toBeLessThan(0)
    // Strands are spread across the groove, not stacked on one line.
    const xs = sheet.map((strand) => strand[0].x)
    expect(new Set(xs).size).toBe(GROOVE_STRANDS)
    expect(Math.max(...xs) - Math.min(...xs)).toBeGreaterThan(GROOVE_LENGTH / 2)
  })

  it('lays a nine-residue peptide across the groove', () => {
    const chain = peptideChain()
    expect(chain).toHaveLength(PEPTIDE_RESIDUES)
    expectInsideBox(chain, 'peptide')
    // Above the sheet floor, below the top of the helices.
    for (const point of chain) {
      expect(point.y).toBeGreaterThan(-0.28)
      expect(point.y).toBeLessThan(GROOVE_HELIX_OFFSET)
    }
    // Monotonic along the groove, so the chain reads as one stretched peptide.
    for (let index = 1; index < chain.length; index += 1) {
      expect(chain[index].x).toBeGreaterThan(chain[index - 1].x)
    }
  })

  it('anchors P2 and the C-terminal position', () => {
    expect(anchorIndices()).toEqual([...ANCHOR_INDICES])
    expect(anchorIndices()).toEqual([1, PEPTIDE_RESIDUES - 1])
    expect(anchorIndices()).not.toBe(ANCHOR_INDICES)
  })

  it('stitches the beta floor into one triangle mesh', () => {
    const mesh = grooveSheetMesh()
    const strands = grooveSheet()
    expect(mesh.positions).toHaveLength(strands.flat().length * 3)
    // Two triangles per cell of an (strands - 1) x (rows - 1) grid.
    expect(mesh.indices).toHaveLength((strands.length - 1) * (strands[0].length - 1) * 6)
    const vertexCount = mesh.positions.length / 3
    for (const index of mesh.indices) {
      expect(Number.isInteger(index)).toBe(true)
      expect(index).toBeGreaterThanOrEqual(0)
      expect(index).toBeLessThan(vertexCount)
    }
    // Every triangle is non-degenerate: no repeated corners.
    for (let index = 0; index < mesh.indices.length; index += 3) {
      const [a, b, c] = mesh.indices.slice(index, index + 3)
      expect(a).not.toBe(b)
      expect(b).not.toBe(c)
      expect(a).not.toBe(c)
    }
  })

  it('reports bounds that cover every strand', () => {
    const bounds = grooveBounds()
    const all = [...grooveHelices().flat(), ...grooveSheet().flat(), ...peptideChain()]
    for (const point of all) {
      expect(point.x).toBeGreaterThanOrEqual(bounds.min.x)
      expect(point.x).toBeLessThanOrEqual(bounds.max.x)
      expect(point.y).toBeGreaterThanOrEqual(bounds.min.y)
      expect(point.y).toBeLessThanOrEqual(bounds.max.y)
      expect(point.z).toBeGreaterThanOrEqual(bounds.min.z)
      expect(point.z).toBeLessThanOrEqual(bounds.max.z)
    }
  })
})
