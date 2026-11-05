import { describe, expect, it } from 'vitest'
import { normalizeInputSchema, normalizeTaskFields, normalizeTaskType } from './taskTerminology'
import { useExperimentDraftStore } from './stores/experimentDraft'
import { createPinia, setActivePinia } from 'pinia'

describe('HLA-I compatibility', () => {
  it('decodes historical records without changing record IDs or source objects', () => {
    const old = { id: 'builtin-mhci', task_type: 'pep_mhci', metadata: { input_schema: 'peptide_mhc_pairs' } }
    const view = normalizeTaskFields(old)
    expect(view).toEqual({ id: 'builtin-mhci', task_type: 'pep_hlai', metadata: { input_schema: 'peptide_hlai_pairs' } })
    expect(old.task_type).toBe('pep_mhci')
    expect(old.metadata.input_schema).toBe('peptide_mhc_pairs')
    expect(normalizeInputSchema('peptide_hla_pairs')).toBe('peptide_hla_pairs')
    expect(normalizeTaskFields({ warning_codes: ['mhci_route_inference_unavailable'] }))
      .toEqual({ warning_codes: ['hlai_route_inference_unavailable'] })
  })
  it('restores old task links with the HLA-I defaults', () => {
    setActivePinia(createPinia())
    const draft = useExperimentDraftStore()
    draft.resetForTask('pep_mhci')
    expect(draft.taskType).toBe('pep_hlai')
    expect(draft.tokenizer).toBe('esm2')
    expect(normalizeTaskType('peptide_mhci_interaction')).toBe('pep_hlai')
  })
})
