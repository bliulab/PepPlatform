import { describe, expect, it } from 'vitest'
import {
  classificationLossApplies,
  classificationLossParametersValid,
  defaultClassificationLoss,
} from './classificationLoss'

describe('classification loss route behavior', () => {
  it.each(['esm2', 'protbert', 'prott5', 'esmc', 'esm3'])(
    'keeps WBCE for %s pooled MLP and BCE for its residue backbone',
    (encoder) => {
      const pooled = { encoder, model: 'mlp' }
      expect(classificationLossApplies(pooled)).toBe(true)
      expect(defaultClassificationLoss(pooled, false)).toBe('WBCE')
      expect(defaultClassificationLoss({ ...pooled, model: 'cnn', sequence_model: 'cnn' }, false)).toBe('BCE')
    },
  )

  it('keeps generic and pair defaults and includes neural representation training', () => {
    const config = { encoder: 'onehot', model: 'cnn', sequence_model: 'cnn' }
    expect(defaultClassificationLoss(config, false)).toBe('BCE')
    expect(defaultClassificationLoss(config, true)).toBe('WBCE')
    expect(classificationLossApplies({ ...config, training_mode: 'export_embeddings', classifier: 'svm' })).toBe(true)
    expect(classificationLossApplies({ encoder: 'bow', model: 'svm' })).toBe(false)
    expect(classificationLossApplies({ ...config, ensemble_members: ['a', 'b'] })).toBe(false)
  })

  it.each([
    { asl_clip: 1 }, { asl_clip: -0.01 }, { asl_clip: Infinity },
    { asl_gamma_pos: -1 }, { asl_gamma_neg: '' }, { asl_gamma_neg: 'nan' },
    { pos_weight_max: 0.9 }, { multilabel_pos_weight_cap: true },
    { multilabel_pos_weight_cap: null },
  ])('rejects invalid numeric overrides %j', (config) => {
    expect(classificationLossParametersValid(config)).toBe(false)
  })

  it('accepts omitted defaults, finite numeric strings, and zero ASL parameters', () => {
    expect(classificationLossParametersValid({})).toBe(true)
    expect(classificationLossParametersValid({ asl_clip: 0, asl_gamma_pos: 0, asl_gamma_neg: '0', pos_weight_max: '1' })).toBe(true)
  })
})
