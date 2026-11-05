import type { TrainingFailure } from './types'

export type TrainingFailurePresentationId =
  | 'profileNoHomologs'
  | 'profileDependency'
  | 'profileTimeout'
  | 'profileRuntime'
  | 'profileOutput'
  | 'psfmResidue'
  | 'cudaOom'
  | 'hostOom'
  | 'dependency'
  | 'missingFile'
  | 'configuration'
  | 'data'
  | 'terminated'
  | 'svmConvergence'
  | 'svmConvergenceDownstream'
  | 'svmArtifactContract'
  | 'unknown'

export type TrainingFailureRemediationId =
  | 'inspectProfileSample'
  | 'expandProfileDatabase'
  | 'choosePssmHybrid'
  | 'chooseNonProfileEncoder'
  | 'installBlast'
  | 'retryProfileSearch'
  | 'checkProfileOutputs'
  | 'fixPsFmResidues'
  | 'reduceBatchSize'
  | 'freeGpuMemory'
  | 'reduceModelMemory'
  | 'freeHostMemory'
  | 'installDependency'
  | 'restoreMissingFile'
  | 'reviewConfiguration'
  | 'reviewDataset'
  | 'inspectLogs'
  | 'retryRun'
  | 'increaseSvmBudget'
  | 'reviewSvmOptimization'
  | 'resumeExistingSvm'

export interface TrainingFailurePresentation {
  id: TrainingFailurePresentationId
  remediations: TrainingFailureRemediationId[]
}

const PROFILE_RUNTIME_CODES = new Set([
  'profile_search_failed',
  'profile_database_failed',
  'profile_binary_version_failed',
  'profile_hits_unreadable',
  'profile_hits_invalid',
])

const PROFILE_OUTPUT_CODES = new Set([
  'profile_output_missing',
  'profile_output_unreadable',
  'profile_checkpoint_field_missing',
  'profile_checkpoint_invalid',
])

function contextString(failure: TrainingFailure | null | undefined, key: string) {
  const value = failure?.context?.[key]
  return typeof value === 'string' ? value.trim().toLowerCase() : ''
}

export function trainingFailurePresentation(
  failure?: TrainingFailure | null,
): TrainingFailurePresentation {
  const code = String(failure?.code || '')
  if (code === 'profile_no_nonself_hits') {
    const feature = contextString(failure, 'feature')
    return {
      id: 'profileNoHomologs',
      remediations: [
        'inspectProfileSample',
        'expandProfileDatabase',
        feature === 'pssm' ? 'choosePssmHybrid' : 'chooseNonProfileEncoder',
      ],
    }
  }
  if (code === 'profile_binary_missing') {
    return { id: 'profileDependency', remediations: ['installBlast', 'retryRun'] }
  }
  if (code === 'profile_search_timeout' || code === 'profile_database_timeout') {
    return { id: 'profileTimeout', remediations: ['retryProfileSearch', 'inspectLogs'] }
  }
  if (PROFILE_RUNTIME_CODES.has(code)) {
    return { id: 'profileRuntime', remediations: ['inspectLogs', 'retryProfileSearch'] }
  }
  if (PROFILE_OUTPUT_CODES.has(code)) {
    return { id: 'profileOutput', remediations: ['checkProfileOutputs', 'inspectLogs'] }
  }
  if (code === 'psfm_query_residue_invalid') {
    return { id: 'psfmResidue', remediations: ['fixPsFmResidues', 'reviewDataset'] }
  }
  if (code === 'cuda_out_of_memory') {
    return {
      id: 'cudaOom',
      remediations: ['reduceBatchSize', 'freeGpuMemory', 'reduceModelMemory'],
    }
  }
  if (code === 'host_out_of_memory' || code.includes('memory_limit_exceeded')) {
    return {
      id: 'hostOom',
      remediations: ['reduceBatchSize', 'freeHostMemory', 'reduceModelMemory'],
    }
  }
  if (code === 'dependency_missing') {
    return { id: 'dependency', remediations: ['installDependency', 'retryRun'] }
  }
  if (code === 'file_not_found') {
    return { id: 'missingFile', remediations: ['restoreMissingFile', 'inspectLogs'] }
  }
  if (code === 'training_configuration_error') {
    return { id: 'configuration', remediations: ['reviewConfiguration', 'inspectLogs'] }
  }
  if (code === 'training_data_invalid') {
    return { id: 'data', remediations: ['reviewDataset', 'inspectLogs'] }
  }
  if (code === 'process_terminated_by_signal') {
    return { id: 'terminated', remediations: ['inspectLogs', 'retryRun'] }
  }
  if (code === 'svm_convergence_failed') {
    return {
      id:
        contextString(failure, 'pipeline_stage') === 'downstream_classifier'
          ? 'svmConvergenceDownstream'
          : 'svmConvergence',
      remediations: ['increaseSvmBudget', 'reviewSvmOptimization', 'inspectLogs'],
    }
  }
  if (code === 'svm_artifact_contract_failed') {
    return {
      id: 'svmArtifactContract',
      remediations: ['resumeExistingSvm', 'inspectLogs'],
    }
  }
  return { id: 'unknown', remediations: ['inspectLogs', 'retryRun'] }
}
