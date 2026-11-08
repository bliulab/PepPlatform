import { describe, expect, it } from 'vitest'
import { cloneEnsemblePayload, ensemblePayloadError, memberUnavailableReason } from './ensemble'
import type { EnsemblePayload } from './ensemble'
import type { ExperimentRecord, ModelRecord } from './types'

const payload: EnsemblePayload = {
  name: 'test',
  task_type: 'pep_hlai',
  dataset_id: 'd',
  member_ids: ['a', 'b'],
  method: 'rank_mean',
  weight_source: 'equal',
  grid_step: 0.05,
  objective: 'auc',
  evaluate_test: false,
  seed: 42,
}
describe('ensemble configuration', () => {
  it('clones the versioned definition without sharing mutable member arrays', () => {
    const source = {
      name: 'source',
      dataset_id: 'd',
      config: { web_ensemble: { ...payload, weights: [2, 1] } },
    } as unknown as ExperimentRecord
    const clone = cloneEnsemblePayload(source)!
    expect(clone).toMatchObject({ ...payload, name: 'source', weights: [2, 1] })
    clone.member_ids.reverse()
    clone.weights![0] = 3
    expect(cloneEnsemblePayload(source)?.member_ids).toEqual(['a', 'b'])
    expect(cloneEnsemblePayload(source)?.weights).toEqual([2, 1])
    expect(cloneEnsemblePayload({ ...source, config: {} })).toBeUndefined()
  })
  it('does not silently accept an empty, duplicate, or invalid weighted selection', () => {
    expect(ensemblePayloadError(payload)).toBe('')
    expect(ensemblePayloadError({ ...payload, member_ids: ['a'] })).toBe('selectMembers')
    expect(ensemblePayloadError({ ...payload, member_ids: ['a', 'a'] })).toBe('selectMembers')
    expect(ensemblePayloadError({ ...payload, weight_source: 'custom', weights: [0, 0] })).toBe(
      'invalidWeights',
    )
    expect(ensemblePayloadError({ ...payload, weight_source: 'custom', weights: [NaN, 1] })).toBe(
      'invalidWeights',
    )
    expect(ensemblePayloadError({ ...payload, weight_source: 'custom', weights: [2, 1] })).toBe('')
  })
  it('requires compatible raw inference models without nesting', () => {
    const model = {
      task_type: 'pep_hlai',
      supports_inference: true,
      metadata: {
        evaluation_protocol_version: 3,
        feature_fit_scope: 'train_only',
        bundle_type: 'pair_vector_nn',
      },
    } as ModelRecord
    expect(memberUnavailableReason(model, 'pep_hlai')).toBe('')
    expect(memberUnavailableReason(model, 'pep_function')).toBe('taskMismatch')
    expect(
      memberUnavailableReason(
        { ...model, metadata: { ...model.metadata, bundle_type: 'web_ensemble' } },
        'pep_hlai',
      ),
    ).toBe('nested')
    expect(memberUnavailableReason({ ...model, metadata: {} }, 'pep_hlai')).toBe('protocolRequired')
  })
})
