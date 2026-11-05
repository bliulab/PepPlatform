import * as THREE from 'three'
import { describe, expect, it } from 'vitest'
import { TASK_ENTRIES } from '../taskCatalog'
import { createSculpture } from './landingSculptures'

describe('welcome concept sculptures', () => {
  it('builds every task on both quality levels with finite geometry', () => {
    for (const mobile of [false, true])
      for (const kind of ['hero' as const, ...TASK_ENTRIES.map((task) => task.key)]) {
        const sculpture = createSculpture(kind, mobile)
        expect(sculpture.radius).toBeGreaterThan(1)
        expect(sculpture.radius).toBeLessThan(8)
        for (const p of [0, 0.5, 1]) {
          sculpture.update(2, p)
          sculpture.root.updateMatrixWorld(true)
          const box = new THREE.Box3().setFromObject(sculpture.root)
          expect([...box.min.toArray(), ...box.max.toArray()].every(Number.isFinite)).toBe(true)
        }
        sculpture.dispose()
      }
  })

  it('releases each GPU resource once even when many meshes share materials', () => {
    const sculpture = createSculpture('hero', false)
    const resources = new Set<THREE.BufferGeometry | THREE.Material>()
    sculpture.root.traverse((object) => {
      if (!(
        object instanceof THREE.Mesh ||
        object instanceof THREE.Line ||
        object instanceof THREE.Points
      ))
        return
      resources.add(object.geometry)
      for (const material of Array.isArray(object.material) ? object.material : [object.material])
        resources.add(material)
    })
    const counts = new Map<object, number>()
    for (const resource of resources)
      resource.addEventListener('dispose', () =>
        counts.set(resource, (counts.get(resource) ?? 0) + 1),
      )
    sculpture.dispose()
    expect(counts.size).toBe(resources.size)
    expect([...counts.values()].every((count) => count === 1)).toBe(true)
  })
})
