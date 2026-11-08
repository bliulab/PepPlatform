import { describe, expect, it } from 'vitest'
import {
  createPeptideParticles,
  nearestParticleArc,
  stepPeptideParticles,
  type ParticleFeedback,
} from './peptideParticles'

const input = (dt = 1 / 60): ParticleFeedback => ({
  dt,
  pointer: null,
  radius: 1.8,
  displacement: 0.6,
  wave: null,
  pointScale: 1000,
})

describe('interactive peptide particles', () => {
  it('forms one deterministic bounded chain from a wider particle field', () => {
    const a = createPeptideParticles(1800)
    expect(a.home).toEqual(createPeptideParticles(1800).home)
    expect(a.positions).not.toEqual(a.home)
    stepPeptideParticles(a, 0, 1, input())
    expect(a.positions).toEqual(a.home)
    expect([...a.positions].every(Number.isFinite)).toBe(true)
    expect(Math.max(...a.home.map(Math.abs))).toBeLessThan(4)
  })
  it('repels nearby particles visibly and returns to the original chain', () => {
    const a = createPeptideParticles(450)
    stepPeptideParticles(a, 0, 1, input())
    const i = 200 * 3
    const feedback = {
      ...input(),
      pointer: { x: a.home[i] - 0.15, y: a.home[i + 1], z: a.home[i + 2] },
    }
    let displacement = 0
    for (let n = 0; n < 30; n++) displacement = stepPeptideParticles(a, 0, 1, feedback)
    expect(displacement).toBeGreaterThan(0.25)
    for (let n = 0; n < 90; n++) displacement = stepPeptideParticles(a, 0, 1, input())
    expect(displacement).toBeLessThan(0.001)
  })
  it('integrates the same spring motion at 30 and 60 frames per second', () => {
    const a = createPeptideParticles(100),
      b = createPeptideParticles(100)
    stepPeptideParticles(a, 0, 1, input())
    stepPeptideParticles(b, 0, 1, input())
    const pointer = { x: 0, y: 0, z: 0 }
    for (let n = 0; n < 60; n++) stepPeptideParticles(a, 0, 1, { ...input(1 / 60), pointer })
    for (let n = 0; n < 30; n++) stepPeptideParticles(b, 0, 1, { ...input(1 / 30), pointer })
    expect(Math.max(...a.positions.map((p, i) => Math.abs(p - b.positions[i])))).toBeLessThan(
      0.00001,
    )
  })
  it('runs a bounded click wave and maps clicks to the chain', () => {
    const a = createPeptideParticles(450)
    stepPeptideParticles(a, 0, 1, input())
    const p = { x: a.home[600], y: a.home[601], z: a.home[602] }
    const origin = nearestParticleArc(a, p)
    expect(origin).toBeCloseTo(a.arc[200])
    stepPeptideParticles(a, 0, 1, { ...input(), wave: { origin, age: 0.1 } })
    expect(Math.max(...a.glow)).toBeGreaterThan(1)
    stepPeptideParticles(a, 0, 1, { ...input(), wave: { origin, age: 2 } })
    expect(Math.max(...a.glow)).toBeLessThan(1)
  })
})
