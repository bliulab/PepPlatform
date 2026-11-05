import { describe, expect, it } from 'vitest'
import { dropoutScopeForModel, resolveClassicFeatureRoute } from './experimentRoutes'

describe('dropoutScopeForModel', () => {
  it.each([
    ['cnn', 'cnn'],
    ['gru', 'sequence'],
    ['transformer', 'transformer'],
    ['gcn', 'graph'],
    ['cross_attn', 'interaction'],
    ['mlp', 'none'],
  ] as const)('maps %s to its effective scope', (model, scope) => {
    expect(dropoutScopeForModel(model)).toBe(scope)
  })
})

describe('resolveClassicFeatureRoute', () => {
  const classicTokenizers = ['kmer', 'dt']
  const classicEncoderOptions = {
    kmer: ['index', 'bow', 'tfidf'],
    dt: ['index', 'bow'],
  }

  it('preserves an already-valid BoW selection when applying a classic route', () => {
    expect(
      resolveClassicFeatureRoute('kmer', 'bow', classicTokenizers, classicEncoderOptions),
    ).toEqual({
      tokenizer: 'kmer',
      encoder: 'bow',
    })
  })

  it('falls back only when the current tokenizer or encoder is unavailable', () => {
    expect(
      resolveClassicFeatureRoute('esm2', 'esm2', classicTokenizers, classicEncoderOptions),
    ).toEqual({
      tokenizer: 'kmer',
      encoder: 'index',
    })
    expect(
      resolveClassicFeatureRoute('dt', 'tfidf', classicTokenizers, classicEncoderOptions),
    ).toEqual({
      tokenizer: 'dt',
      encoder: 'index',
    })
  })
})
