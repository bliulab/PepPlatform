import { describe, expect, it } from 'vitest'

import { trainingFailurePresentation } from './trainingFailure'
import type { TrainingFailure } from './types'

function failure(code: string, context: Record<string, unknown> = {}): TrainingFailure {
  return {
    schema_version: 1,
    code,
    category: 'runtime',
    stage: 'training',
    message: code,
    context,
    exit_code: 1,
  }
}

describe('training failure presentation', () => {
  it('gives strict PSSM a scientifically explicit hybrid option', () => {
    const presentation = trainingFailurePresentation(
      failure('profile_no_nonself_hits', { feature: 'pssm' }),
    )

    expect(presentation.id).toBe('profileNoHomologs')
    expect(presentation.remediations).toContain('choosePssmHybrid')
    expect(presentation.remediations).not.toContain('chooseNonProfileEncoder')
  })

  it('does not imply that strict PSFM has a hybrid fallback', () => {
    const presentation = trainingFailurePresentation(
      failure('profile_no_nonself_hits', { feature: 'psfm' }),
    )

    expect(presentation.remediations).toContain('chooseNonProfileEncoder')
    expect(presentation.remediations).not.toContain('choosePssmHybrid')
  })

  it('maps CUDA OOM and unknown failures to actionable fallbacks', () => {
    expect(trainingFailurePresentation(failure('cuda_out_of_memory')).id).toBe('cudaOom')
    expect(trainingFailurePresentation(failure('new_failure_code')).id).toBe('unknown')
    expect(trainingFailurePresentation(null).remediations).toContain('inspectLogs')
  })

  it('distinguishes reusable downstream SVM convergence failures', () => {
    const downstream = trainingFailurePresentation(
      failure('svm_convergence_failed', { pipeline_stage: 'downstream_classifier' }),
    )
    const direct = trainingFailurePresentation(failure('svm_convergence_failed'))

    expect(downstream.id).toBe('svmConvergenceDownstream')
    expect(downstream.remediations).toContain('increaseSvmBudget')
    expect(direct.id).toBe('svmConvergence')
  })

  it('explains a post-fit SVM artifact contract failure', () => {
    const presentation = trainingFailurePresentation(failure('svm_artifact_contract_failed'))

    expect(presentation.id).toBe('svmArtifactContract')
    expect(presentation.remediations).toContain('resumeExistingSvm')
  })
})
