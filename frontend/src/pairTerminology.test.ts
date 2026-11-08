import { describe, expect, it } from 'vitest'
import { reactive } from 'vue'
import { normalizePairFields, normalizePairStrategy, pairStrategyLabel } from './pairTerminology'

describe('stream migration compatibility', () => {
  it('maps legacy spellings into canonical runtime values', () => {
    for (const value of ['single_tower', 'single-tower', 'single_stream', 'single-stream'])
      expect(normalizePairStrategy(value)).toBe('single_stream')
    for (const value of ['dual_tower', 'dual-tower', 'two_stream', 'two-stream'])
      expect(normalizePairStrategy(value)).toBe('two_stream')
    expect(pairStrategyLabel('dual_tower')).toBe('Two-stream')
    expect(normalizePairFields({ pair_strategy: ' DUAL_TOWER ' }).pair_strategy).toBe('two_stream')
  })
  it('converts historical views without changing identities or the original record', () => {
    const original = { config: { pair_strategy: 'dual_tower', towers: {
      peptide: { tower_model: 'identity' },
    } }, id: 'c008_single_dual_tower_prott5_esm2_b16',
    sources: { 'pepplatform/tower_spec.py': 'sha256' } }
    const copy = structuredClone(original)
    const view = normalizePairFields(original) as unknown as Record<string, any>
    expect(view.config.pair_strategy).toBe('two_stream')
    expect(view.config.streams.peptide.stream_model).toBe('identity')
    expect(view.id).toBe(original.id)
    expect(view.sources).toEqual(original.sources)
    expect(original).toEqual(copy)
  })
  it('restores historical browser draft field names and strategy values', () => {
    const state = normalizePairFields({ pairStrategy: 'dual_tower', towerAlignmentDim: 32, towersLinked: false }) as unknown as Record<string, unknown>
    expect(state).toEqual({ pairStrategy: 'two_stream', streamAlignmentDim: 32, streamsLinked: false })
  })
  it('preserves historical protocol payloads for version-aware validation', () => {
    const contract = { contract_version: 1, strategy: 'dual_tower', towers: { peptide: { rank: 2 } } }
    const original = { pair_strategy: 'dual_tower', pair_strategy_contract: contract }
    const view = normalizePairFields(reactive(original))
    expect(view.pair_strategy).toBe('two_stream')
    expect(view.pair_strategy_contract).toEqual(contract)
    expect(view.pair_strategy_contract).not.toBe(contract)
  })
  it('rejects conflicting old and new settings', () => {
    expect(() => normalizePairFields({ tower_alignment_dim: 32, stream_alignment_dim: 64 }))
      .toThrow('Conflicting')
  })
})
