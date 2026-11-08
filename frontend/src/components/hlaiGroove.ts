/**
 * Deterministic geometry for the peptide-HLA-I groove stage.
 *
 * Procedural illustration of two alpha helices over a beta sheet, with a peptide
 * anchored at P2 and its C-terminus.
 *
 * Like `peptideScene.ts` this module is free of any `three` import so the shape
 * can be unit-tested in Node; the component only turns the points into tubes,
 * ribbons and instanced spheres.
 */

export type Vec3Point = { x: number; y: number; z: number }

/** Groove extent along X, in scene units. */
export const GROOVE_LENGTH = 8
/** Samples per alpha helix. */
export const GROOVE_SAMPLES = 200
/** Coil radius of each alpha helix around its own spine. */
export const GROOVE_HELIX_RADIUS = 0.22
/** Distance of each helix spine from the groove centre line. */
export const GROOVE_HELIX_OFFSET = 1.2
/** How far the helices rise at their ends, forming the groove walls. */
export const GROOVE_ARCH = 0.4
/** Turns of each alpha helix along the groove. */
export const GROOVE_TURNS = 14
/** Antiparallel strands of the beta sheet floor. */
export const GROOVE_STRANDS = 8
/** Residues of the modelled peptide. */
export const PEPTIDE_RESIDUES = 9
/** Anchor positions, as peptide indices: P2 and the C-terminal P-omega. */
export const ANCHOR_INDICES: readonly number[] = [1, PEPTIDE_RESIDUES - 1]

function normalised(index: number, count: number): number {
  return count <= 1 ? 0.5 : index / (count - 1)
}

/**
 * The two alpha helices. Each sample spirals around a spine that runs along X at
 * `z = ±GROOVE_HELIX_OFFSET`, and the spine arches up toward both ends so the
 * pair reads as a groove rather than two parallel rods.
 */
export function grooveHelices(): Vec3Point[][] {
  return [-1, 1].map((side) =>
    Array.from({ length: GROOVE_SAMPLES }, (_, index) => {
      const t = normalised(index, GROOVE_SAMPLES)
      const arch = GROOVE_ARCH * Math.pow(2 * t - 1, 2)
      const coil = t * GROOVE_TURNS * Math.PI * 2
      return {
        x: (t - 0.5) * GROOVE_LENGTH,
        y: arch + Math.cos(coil) * GROOVE_HELIX_RADIUS,
        z: side * (GROOVE_HELIX_OFFSET + Math.sin(coil) * GROOVE_HELIX_RADIUS),
      }
    }),
  )
}

/**
 * Beta sheet floor: eight antiparallel strands running across Z, each a shallow
 * zig-zag in Y so the ribbon reads as a pleat instead of a flat plane.
 */
export function grooveSheet(): Vec3Point[][] {
  const span = GROOVE_HELIX_OFFSET * 2 - 0.25
  return Array.from({ length: GROOVE_STRANDS }, (_, strand) => {
    const x = (normalised(strand, GROOVE_STRANDS) - 0.5) * (GROOVE_LENGTH - 1.1)
    return Array.from({ length: 9 }, (_, index) => {
      const t = normalised(index, 9)
      return {
        x,
        y: -0.28 + (index % 2 === 0 ? 0.07 : -0.07),
        z: (t - 0.5) * span,
      }
    })
  })
}

/**
 * The peptide itself: a stretched chain lying across the groove floor, arched up
 * in the middle and zig-zagging slightly as an extended backbone does.
 */
export function peptideChain(): Vec3Point[] {
  return Array.from({ length: PEPTIDE_RESIDUES }, (_, index) => {
    const t = normalised(index, PEPTIDE_RESIDUES)
    return {
      x: (t - 0.5) * GROOVE_LENGTH * 0.72,
      y: 0.1 + Math.sin(t * Math.PI) * 0.3,
      z: Math.sin(t * Math.PI * 2) * 0.2,
    }
  })
}

/** Peptide indices that anchor into the groove pockets (P2 and P-omega). */
export function anchorIndices(): number[] {
  return [...ANCHOR_INDICES]
}

/**
 * The beta floor as one triangle mesh instead of eight loose tubes: the strands
 * are stitched left to right so the floor reads as a continuous corrugated sheet.
 * Returns raw arrays so the caller can build a BufferGeometry without three.
 */
export function grooveSheetMesh(): { positions: number[]; indices: number[] } {
  const strands = grooveSheet()
  const columns = strands.length
  const rows = strands[0]?.length ?? 0
  const positions: number[] = []
  for (const strand of strands) {
    for (const point of strand) positions.push(point.x, point.y, point.z)
  }
  const indices: number[] = []
  for (let column = 0; column < columns - 1; column += 1) {
    for (let row = 0; row < rows - 1; row += 1) {
      const a = column * rows + row
      const b = a + 1
      const c = a + rows
      const d = c + 1
      indices.push(a, c, b, b, c, d)
    }
  }
  return { positions, indices }
}

/** Bounding box of everything the stage draws, used by the tests and the camera. */
export function grooveBounds(): { min: Vec3Point; max: Vec3Point } {
  const points = [...grooveHelices().flat(), ...grooveSheet().flat(), ...peptideChain()]
  const min = { x: Infinity, y: Infinity, z: Infinity }
  const max = { x: -Infinity, y: -Infinity, z: -Infinity }
  for (const point of points) {
    min.x = Math.min(min.x, point.x)
    min.y = Math.min(min.y, point.y)
    min.z = Math.min(min.z, point.z)
    max.x = Math.max(max.x, point.x)
    max.y = Math.max(max.y, point.y)
    max.z = Math.max(max.z, point.z)
  }
  return { min, max }
}
