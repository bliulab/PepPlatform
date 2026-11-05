import { describe, expect, it } from 'vitest'
import {
  agentWorkspaceStageIndex,
  deriveAgentWorkspaceStage,
  deriveExperimentRunProgress,
  reconcileSelectedActionIds,
  resolveExperimentProgressTotal,
} from './agentWorkspace'
import type { ExperimentRecord } from './types'

function experiment(
  status: string,
  config: Record<string, unknown>,
  events: ExperimentRecord['metrics']['events'] = [],
) {
  return { status, config, metrics: { events, summary: {} } }
}

describe('deriveAgentWorkspaceStage', () => {
  it('starts at the objective stage without a plan', () => {
    expect(deriveAgentWorkspaceStage()).toBe('objective')
  })

  it('maps a draft or unknown plan state to approval', () => {
    expect(deriveAgentWorkspaceStage({ status: 'draft', review: {} })).toBe('approval')
    expect(deriveAgentWorkspaceStage({ status: 'custom', review: {} })).toBe('approval')
  })

  it.each(['executed', 'running', 'ready_for_report'])('maps %s to execution', (status) => {
    expect(deriveAgentWorkspaceStage({ status, review: {} })).toBe('execution')
  })

  it('prefers review content over the persisted status', () => {
    expect(deriveAgentWorkspaceStage({ status: 'running', review: { summary: 'Ready' } })).toBe(
      'review',
    )
    expect(deriveAgentWorkspaceStage({ status: 'reported', review: {} })).toBe('review')
  })
})

describe('agentWorkspaceStageIndex', () => {
  it('preserves the lifecycle order', () => {
    expect(agentWorkspaceStageIndex('objective')).toBe(0)
    expect(agentWorkspaceStageIndex('approval')).toBe(1)
    expect(agentWorkspaceStageIndex('execution')).toBe(2)
    expect(agentWorkspaceStageIndex('review')).toBe(3)
  })
})

describe('reconcileSelectedActionIds', () => {
  const actions = [
    { id: 'ready', type: 'create_experiment', valid: true, status: 'pending' },
    { id: 'invalid', type: 'create_experiment', valid: false, status: 'pending' },
    { id: 'executed', type: 'create_experiment', valid: true, status: 'executed' },
  ]

  it('never adds pending actions that the user did not select', () => {
    expect(reconcileSelectedActionIds([], actions)).toEqual([])
  })

  it('keeps only explicitly selected actions that remain executable', () => {
    expect(reconcileSelectedActionIds(['ready', 'invalid', 'executed'], actions)).toEqual(['ready'])
  })

  it('applies a reduced quota without selecting replacements', () => {
    const twoReady = [
      ...actions,
      { id: 'ready-2', type: 'create_experiment', valid: true, status: 'pending' },
    ]
    expect(reconcileSelectedActionIds(['ready', 'ready-2'], twoReady, 1)).toEqual(['ready'])
  })
})

describe('deriveExperimentRunProgress', () => {
  it('derives determinate progress from the latest real epoch event', () => {
    const progress = deriveExperimentRunProgress(
      experiment('running', { model: 'cnn', epochs: 30 }, [
        { kind: 'epoch', epoch: 4, total: 30 },
        { kind: 'epoch', epoch: 7, total: 30 },
      ]),
    )

    expect(progress).toMatchObject({
      status: 'running',
      mode: 'determinate',
      usesEpochs: true,
      current: 7,
      total: 30,
      percent: 23,
    })
  })

  it('falls back to configured epochs when the event omits its total', () => {
    const progress = deriveExperimentRunProgress(
      experiment('running', { model: 'lstm', epochs: 12 }, [{ kind: 'epoch', epoch: 3 }]),
    )

    expect(progress).toMatchObject({ mode: 'determinate', current: 3, total: 12, percent: 25 })
  })

  it('keeps an epoch route indeterminate until the first metric arrives', () => {
    const progress = deriveExperimentRunProgress(
      experiment('running', { model: 'transformer', epochs: 20 }),
    )

    expect(progress).toMatchObject({ mode: 'indeterminate', usesEpochs: true })
    expect(progress.percent).toBeUndefined()
  })

  it('uses profile preparation events before the first epoch metric', () => {
    const database = deriveExperimentRunProgress(
      experiment('running', { model: 'cnn', encoder: 'pssm', epochs: 20 }, [
        {
          kind: 'stage',
          stage: 'database',
          status: 'running',
          current: 0,
          total: 1,
          feature: 'pssm+psfm',
        },
      ]),
    )
    expect(database).toMatchObject({ mode: 'indeterminate', preparation: 'profile_database' })

    const profiles = deriveExperimentRunProgress(
      experiment('running', { model: 'cnn', encoder: 'pssm', epochs: 20 }, [
        {
          kind: 'stage',
          stage: 'profiles',
          status: 'running',
          current: 42,
          total: 125,
          feature: 'pssm+psfm',
          elapsed_seconds: 21,
        },
      ]),
    )
    expect(profiles).toMatchObject({
      mode: 'determinate',
      preparation: 'profile_generation',
      current: 42,
      total: 125,
      percent: 34,
      preparationElapsedSeconds: 21,
      preparationRatePerSecond: 2,
      preparationEtaSeconds: 41.5,
    })
  })

  it('withholds an unstable profile ETA during warm-up', () => {
    const progress = deriveExperimentRunProgress(
      experiment('running', { model: 'cnn', encoder: 'pssm', epochs: 20 }, [
        {
          kind: 'stage',
          stage: 'profiles',
          status: 'running',
          current: 4,
          total: 125,
          feature: 'pssm',
          elapsed_seconds: 9,
        },
      ]),
    )

    expect(progress).toMatchObject({
      preparationElapsedSeconds: 9,
    })
    expect(progress.preparationRatePerSecond).toBeUndefined()
    expect(progress.preparationEtaSeconds).toBeUndefined()
  })

  it('returns to epoch progress once an epoch event arrives', () => {
    const progress = deriveExperimentRunProgress(
      experiment('running', { model: 'cnn', encoder: 'pssm', epochs: 20 }, [
        {
          kind: 'stage',
          stage: 'profiles',
          status: 'complete',
          current: 125,
          total: 125,
          feature: 'pssm+psfm',
        },
        { kind: 'epoch', epoch: 2, total: 20 },
      ]),
    )
    expect(progress).toMatchObject({ mode: 'determinate', current: 2, total: 20, percent: 10 })
    expect(progress.preparation).toBeUndefined()
  })

  it('switches from epoch progress to structured downstream classifier progress', () => {
    const progress = deriveExperimentRunProgress(
      experiment(
        'running',
        { model: 'cnn', classifier: 'lgb', training_mode: 'export_embeddings', epochs: 8 },
        [
          { kind: 'epoch', epoch: 8, total: 8 },
          {
            kind: 'stage',
            stage: 'classifier_fit',
            status: 'running',
            current: 240,
            total: 800,
            output: 3,
            outputs: 8,
            iteration: 40,
            iterations: 100,
            threads: 4,
            elapsed_seconds: 12.5,
            eta_seconds: 29,
          },
        ],
      ),
    )

    expect(progress).toMatchObject({
      mode: 'determinate',
      runStage: 'classifier_fit',
      runStageStatus: 'running',
      current: 240,
      total: 800,
      percent: 30,
      runStageOutput: 3,
      runStageOutputs: 8,
      runStageIteration: 40,
      runStageIterations: 100,
      runStageThreads: 4,
      preparationElapsedSeconds: 12.5,
      preparationEtaSeconds: 29,
    })
  })

  it('preserves the final structured stage for a completed run', () => {
    const progress = deriveExperimentRunProgress(
      experiment(
        'succeeded',
        { model: 'cnn', classifier: 'lgb', training_mode: 'export_embeddings' },
        [{ kind: 'stage', stage: 'finalization', status: 'complete', current: 3, total: 3 }],
      ),
    )

    expect(progress).toMatchObject({
      status: 'succeeded',
      mode: 'terminal',
      runStage: 'finalization',
      runStageStatus: 'complete',
      current: 3,
      total: 3,
      percent: 100,
    })
  })

  it('never invents epoch progress for a traditional classifier', () => {
    const progress = deriveExperimentRunProgress(
      experiment('running', { model: 'rf', classifier: 'rf', epochs: 30 }),
    )

    expect(progress).toMatchObject({ mode: 'indeterminate', usesEpochs: false })
    expect(progress.current).toBeUndefined()
    expect(progress.percent).toBeUndefined()
  })

  it('derives determinate progress for bounded hyperparameter search', () => {
    const progress = deriveExperimentRunProgress(
      experiment('running', { model: 'rf', classifier: 'rf' }, [
        {
          kind: 'stage',
          stage: 'hyperparameter_tuning',
          status: 'running',
          current: 6,
          total: 24,
          feature: 'rf',
        },
      ]),
    )

    expect(progress).toMatchObject({
      mode: 'determinate',
      usesEpochs: false,
      preparation: 'hyperparameter_tuning',
      preparationFeature: 'rf',
      current: 6,
      total: 24,
      percent: 25,
    })
  })

  it('recognizes an early-stopped successful run without marking its schedule complete', () => {
    const progress = deriveExperimentRunProgress(
      experiment('succeeded', { model: 'cnn', epochs: 30, early_stop_patience: 5 }, [
        { kind: 'epoch', epoch: 8, total: 30 },
      ]),
    )

    expect(progress).toMatchObject({
      mode: 'terminal',
      earlyStopped: true,
      current: 8,
      total: 30,
      percent: 27,
    })
  })

  it.each(['failed', 'canceled'])('preserves the last real epoch for a %s run', (status) => {
    const progress = deriveExperimentRunProgress(
      experiment(status, { model: 'gru', epochs: 10 }, [{ kind: 'epoch', epoch: 4, total: 10 }]),
    )

    expect(progress).toMatchObject({ status, mode: 'terminal', current: 4, total: 10, percent: 40 })
  })

  it('preserves the last profile progress when a run stops before training', () => {
    const progress = deriveExperimentRunProgress(
      experiment('canceled', { model: 'cnn', encoder: 'pssm', epochs: 10 }, [
        {
          kind: 'stage',
          stage: 'profiles',
          status: 'running',
          current: 25,
          total: 100,
          feature: 'pssm',
          elapsed_seconds: 50,
        },
      ]),
    )

    expect(progress).toMatchObject({
      status: 'canceled',
      mode: 'terminal',
      preparation: 'profile_generation',
      current: 25,
      total: 100,
      percent: 25,
    })
  })

  it('treats missing and unknown records as unavailable', () => {
    expect(deriveExperimentRunProgress()).toMatchObject({ status: 'missing', mode: 'unavailable' })
    expect(deriveExperimentRunProgress(experiment('custom', {}))).toMatchObject({
      status: 'missing',
      mode: 'unavailable',
    })
  })
})

describe('resolveExperimentProgressTotal', () => {
  it('prefers the profile stage total over configured training epochs', () => {
    expect(
      resolveExperimentProgressTotal(
        { preparation: 'profile_generation', total: 375 },
        { epochs: 10 },
      ),
    ).toBe(375)
  })

  it('prefers a real epoch event total when one is available', () => {
    expect(
      resolveExperimentProgressTotal({ preparation: undefined, total: 30 }, { epochs: 20 }, 30),
    ).toBe(30)
  })

  it('falls back to configured epochs when no stage or epoch total exists', () => {
    expect(
      resolveExperimentProgressTotal({ preparation: undefined, total: undefined }, { epochs: 12 }),
    ).toBe(12)
  })
})
