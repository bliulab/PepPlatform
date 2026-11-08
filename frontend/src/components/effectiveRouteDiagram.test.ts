import { describe, expect, it } from 'vitest'
import type { EffectiveRoute } from '../types'
import {
  buildEffectiveRouteDiagramModel,
  isResolvedEffectiveRoute,
  routeStageFingerprint,
} from './effectiveRouteDiagram'

function singleRoute(): EffectiveRoute {
  return {
    schema_version: 2,
    route_id: 'sequence.classic_positional',
    task_type: 'pep_function',
    training_mode: 'end_to_end',
    streams: [
      {
        role: 'sequence',
        feature_route: 'classic',
        tokenizer: { id: 'kmer', label: 'Kmer' },
        encoder: { id: 'index', label: 'Index', output_kind: 'token_ids' },
      },
    ],
    predictor: {
      input_adapter: { id: 'learned_embedding', label: 'Learned Embedding' },
      architecture: { id: 'gru', label: 'GRU', learnable: true },
      readout: { id: 'last_hidden', label: 'Last Hidden State' },
      head: { id: 'mlp', label: 'MLP', hidden_dims: [256, 128], activation: 'silu', dropout: 0.1 },
    },
  }
}

describe('effective route diagram model', () => {
  it('builds an ordered single-sequence topology and concise summary', () => {
    const model = buildEffectiveRouteDiagramModel(singleRoute())

    expect(model.summary).toBe('Kmer → Index → GRU → MLP')
    expect(model.representationSummary).toBe('Kmer → Index')
    expect(model.streams[0]?.stages.map((stage) => stage.kind)).toEqual(['tokenizer', 'encoder'])
    expect(model.predictorStages.map((stage) => stage.kind)).toEqual([
      'inputAdapter',
      'architecture',
      'readout',
      'head',
    ])
  })

  it('joins heterogeneous streams without presenting either route as the only input', () => {
    const route = singleRoute()
    route.route_id = 'pair.dual_vector'
    route.streams = [
      { ...route.streams[0], role: 'peptide' },
      {
        role: 'partner',
        feature_route: 'pretrained',
        tokenizer: { id: 'prott5', label: 'ProtT5' },
        encoder: { id: 'prott5', label: 'ProtT5', output_kind: 'sequence_embedding' },
      },
    ]
    route.predictor.architecture = { id: 'identity', label: 'Identity' }

    const model = buildEffectiveRouteDiagramModel(route)

    expect(model.representationSummary).toBe('Kmer / Index + ProtT5 / ProtT5')
    expect(model.summary).toBe('Kmer / Index + ProtT5 / ProtT5 → MLP')
    expect(model.predictorStages[1]?.muted).toBe(true)
  })

  it('guards against incomplete preflight payloads and fingerprints topology changes', () => {
    expect(isResolvedEffectiveRoute({})).toBe(false)
    expect(isResolvedEffectiveRoute(singleRoute())).toBe(true)

    const model = buildEffectiveRouteDiagramModel(singleRoute())
    const head = model.predictorStages.at(-1)!
    const original = routeStageFingerprint(head)
    head.component.hidden_dims = [64]
    expect(routeStageFingerprint(head)).not.toBe(original)
  })
})
