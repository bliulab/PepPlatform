import * as THREE from 'three'
import type { TaskKey } from '../taskCatalog'
import { createParticleSculpture } from './particleSculpture'
import type { ParticleFeedback } from './peptideParticles'
import { grooveHelices, grooveSheetMesh, peptideChain } from './hlaiGroove'

export type SculptureKind = 'hero' | TaskKey
export interface Sculpture {
  root: THREE.Group
  radius: number
  update: (time: number, entrance: number, feedback?: ParticleFeedback) => void
  dispose: () => void
}

const TEAL = 0x0f9488
const MINT = 0x99f6e4

/** Procedural, reproducible concept sculptures. No model files or remote textures. */
export function createSculpture(kind: SculptureKind, mobile: boolean): Sculpture {
  if (kind === 'hero') return createParticleSculpture(mobile)
  const root = new THREE.Group()
  const body = new THREE.Group()
  root.add(body)
  const resources = new Set<{ dispose: () => void }>()
  const reveals: Array<{ object: THREE.Object3D; start: number; size: number }> = []
  const animations: Array<(time: number, entrance: number) => void> = []
  const track = <T extends { dispose: () => void }>(item: T): T => {
    resources.add(item)
    return item
  }
  const material = (color: number, emissive = 0, roughness = 0.3) =>
    track(
      new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: emissive,
        roughness,
        metalness: 0.25,
        transparent: true,
        opacity: 1,
      }),
    )
  const skin = material(TEAL, 0.08)
  const bead = material(0x5eead4, 0.025, 0.36)
  const accent = material(MINT, 0.12)
  const dim = material(0x115e59, 0.03, 0.4)
  const sphere = track(new THREE.SphereGeometry(1, mobile ? 12 : 20, mobile ? 10 : 16))
  const vector = (p: { x: number; y: number; z: number }) => new THREE.Vector3(p.x, p.y, p.z)

  function tube(
    points: THREE.Vector3[],
    width: number,
    mat: THREE.MeshStandardMaterial,
    parent = body,
    reveal = false,
  ) {
    const geometry = track(
      new THREE.TubeGeometry(
        new THREE.CatmullRomCurve3(points),
        Math.max(48, points.length * 3),
        width,
        mobile ? 6 : 10,
        false,
      ),
    )
    const mesh = new THREE.Mesh(geometry, mat)
    parent.add(mesh)
    if (reveal)
      animations.push((_time, entrance) =>
        geometry.setDrawRange(
          0,
          Math.floor((geometry.index!.count * Math.min(1, entrance * 1.2)) / 3) * 3,
        ),
      )
    return mesh
  }

  function ball(p: THREE.Vector3, size: number, mat = bead, parent = body, start = -1) {
    const object = new THREE.Mesh(sphere, mat)
    object.position.copy(p)
    object.scale.setScalar(size)
    parent.add(object)
    if (start >= 0) reveals.push({ object, start, size })
    return object
  }

  function helix(
    center: THREE.Vector3,
    radius: number,
    height: number,
    turns: number,
    count: number,
  ) {
    return Array.from({ length: count }, (_, i) => {
      const t = i / (count - 1)
      const angle = t * Math.PI * 2 * turns
      return new THREE.Vector3(
        Math.cos(angle) * radius,
        (t - 0.5) * height,
        Math.sin(angle) * radius,
      ).add(center)
    })
  }

  function strand(points: THREE.Vector3[], parent = body, highlight = false) {
    tube(points, 0.055, highlight ? accent : skin, parent)
    points.forEach((p, i) =>
      ball(
        p,
        highlight && (i === 1 || i === points.length - 1) ? 0.17 : 0.115,
        highlight ? accent : bead,
        parent,
      ),
    )
  }

  if (kind === 'pep_function') {
    const points = Array.from(
      { length: 22 },
      (_, i) =>
        new THREE.Vector3((i - 10.5) * 0.34, Math.sin(i * 0.38) * 0.5, Math.cos(i * 0.38) * 0.45),
    )
    strand(points)
    body.rotation.z = 0.12
    const scan = ball(new THREE.Vector3(), 0.23, accent)
    const sequence = new THREE.CatmullRomCurve3(points)
    animations.push((time) => {
      sequence.getPointAt((time * 0.12) % 1, scan.position)
    })
    // Three curved paths branch from the same sequence: function is multi-label.
    for (let i = 0; i < 3; i++) {
      const from = points[6 + i * 5]
      const to = new THREE.Vector3(-2.2 + i * 2.2, 2.05 + (i % 2) * 0.35, -0.1)
      const mid = from
        .clone()
        .lerp(to, 0.5)
        .add(new THREE.Vector3(0, 0.35, 0.2))
      tube([from, mid, to], 0.018, dim)
      const label = new THREE.Mesh(track(new THREE.BoxGeometry(1.35, 0.52, 0.12)), skin)
      label.position.copy(to)
      body.add(label)
      const mark = ball(to.clone().add(new THREE.Vector3(-0.43, 0, 0.1)), 0.06, accent)
      animations.push((time) => {
        mark.scale.setScalar(0.06 * (1 + Math.sin(time * 2 - i) * 0.15))
      })
    }
  } else if (kind === 'pep_protein_interaction') {
    // Connected secondary-structure helices read as a folded protein and leave
    // an exposed pocket on the right for the approaching peptide.
    let previousEnd: THREE.Vector3 | null = null
    for (let i = 0; i < 3; i++) {
      const rotation = new THREE.Matrix4().makeRotationZ([-0.38, 0.35, -0.2][i])
      const offset = new THREE.Vector3(-1.9 + i * 0.9, i === 1 ? -0.15 : 0.1, i === 1 ? -0.65 : 0)
      const fold = helix(new THREE.Vector3(), 0.3, 3.2, 5.5, mobile ? 75 : 110).map((p) =>
        p.applyMatrix4(rotation).add(offset),
      )
      if (i % 2) fold.reverse()
      tube(fold, 0.1, i === 1 ? dim : skin)
      if (previousEnd) {
        const middle = previousEnd
          .clone()
          .lerp(fold[0], 0.5)
          .add(new THREE.Vector3(0, i === 1 ? 0.65 : -0.65, 0.15))
        tube([previousEnd, middle, fold[0]], 0.085, skin)
      }
      previousEnd = fold[fold.length - 1]
    }
    const peptide = new THREE.Group()
    body.add(peptide)
    strand(
      Array.from(
        { length: 10 },
        (_, i) => new THREE.Vector3(0.55 + i * 0.22, 0.15 + Math.sin(i * 0.55) * 0.4, 0.6),
      ),
      peptide,
      true,
    )
    animations.push((_time, entrance) => {
      peptide.position.x = (1 - entrance) * 2.7
    })
    const pocket = ball(new THREE.Vector3(0.6, 0.22, 0.48), 0.23, accent)
    animations.push((time) => pocket.scale.setScalar(0.19 + Math.sin(time * 1.5) * 0.025))
    body.rotation.z = -0.16
  } else {
    const classII = kind === 'pep_hlaii'
    for (const [i, raw] of grooveHelices().entries()) {
      tube(raw.map(vector), 0.13, classII && i === 1 ? dim : skin)
    }
    const sheet = grooveSheetMesh()
    const geometry = track(new THREE.BufferGeometry())
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(sheet.positions, 3))
    geometry.setIndex(sheet.indices)
    geometry.computeVertexNormals()
    const floor = material(0x0f766e, 0.04, 0.5)
    floor.side = THREE.DoubleSide
    body.add(new THREE.Mesh(geometry, floor))
    const peptide = new THREE.Group()
    body.add(peptide)
    const chain = classII
      ? Array.from(
          { length: 17 },
          (_, i) =>
            new THREE.Vector3(
              (i - 8) * 0.57,
              0.25 + Math.sin((i / 16) * Math.PI) * 0.28,
              Math.sin(i * 0.65) * 0.16,
            ),
        )
      : peptideChain().map(vector)
    strand(chain, peptide, true)
    if (!classII) {
      // End pockets make the short, bounded class-I binding site distinct.
      for (const x of [-3.35, 3.35]) {
        tube(
          [
            new THREE.Vector3(x, 0.15, -1.15),
            new THREE.Vector3(x, -0.1, 0),
            new THREE.Vector3(x, 0.15, 1.15),
          ],
          0.18,
          skin,
        )
      }
    } else {
      chain
        .slice(4, 13)
        .forEach((p) => ball(p.clone().add(new THREE.Vector3(0, -0.12, 0)), 0.08, bead, peptide))
    }
    animations.push((_time, entrance) => {
      peptide.position.y = (1 - entrance) * 1.55
    })
    body.rotation.set(0.42, 0, -0.13)
    body.scale.setScalar(0.8)
  }

  // Thin orbital marks provide depth without competing with the molecule.
  const ringMaterial = track(
    new THREE.LineBasicMaterial({ color: TEAL, transparent: true, opacity: 0.13 }),
  )
  for (let i = 0; i < 2; i++) {
    const points = Array.from({ length: 121 }, (_, j) => {
      const t = (j / 120) * Math.PI * 2
      return new THREE.Vector3(Math.cos(t) * (3.5 + i * 0.45), 0, Math.sin(t) * (3.5 + i * 0.45))
    })
    const ring = new THREE.Line(
      track(new THREE.BufferGeometry().setFromPoints(points)),
      ringMaterial,
    )
    ring.rotation.set(0.35 + i * 0.42, 0.2, 0.18 + i * 0.65)
    root.add(ring)
  }
  root.updateMatrixWorld(true)
  const bounds = new THREE.Box3().setFromObject(body)
  body.position.sub(bounds.getCenter(new THREE.Vector3()))
  const radius = bounds.getBoundingSphere(new THREE.Sphere()).radius
  return {
    root,
    radius,
    update(time, entrance) {
      const p = THREE.MathUtils.smoothstep(entrance, 0, 1)
      for (const { object, start, size } of reveals)
        object.scale.setScalar(
          size * THREE.MathUtils.smoothstep(entrance, start, Math.min(1, start + 0.18)),
        )
      for (const animate of animations) animate(time, p)
    },
    dispose() {
      for (const item of resources) item.dispose()
    },
  }
}
