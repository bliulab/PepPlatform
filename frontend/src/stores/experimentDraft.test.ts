import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import {
  datasetLengthRoleMin,
  pairTopicRouteIssues,
  preflightAllowsTraining,
  useExperimentDraftStore,
} from './experimentDraft'
import type { ExperimentRecord } from '../types'
import { CLASSIFICATION_LOSS_FIELDS } from '../classificationLoss'

function experiment(
  config: Record<string, unknown>,
  overrides: Partial<ExperimentRecord> = {},
): ExperimentRecord {
  return {
    id: 'source-run',
    dataset_id: 'dataset-1',
    name: 'source experiment',
    task_type: 'pep_function',
    config,
    status: 'succeeded',
    log_path: '',
    artifact_dir: '',
    metrics: { events: [], summary: {} },
    model_id: null,
    error: '',
    ...overrides,
  }
}

describe('experiment draft clone rerun config', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('keeps loss fields absent for old clones even after cloning a configured run', () => {
    const draft = useExperimentDraftStore()
    const base = { tokenizer: 'kmer', encoder: 'onehot', model: 'cnn', classifier: 'mlp' }
    draft.cloneFromExperiment(experiment({ ...base, multilabel_loss: 'asymmetric_focal', asl_clip: 0 }))
    expect(draft.classificationLossOverrides.asl_clip).toBe(0)
    draft.cloneFromExperiment(experiment(base))
    const config = draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })
    expect(draft.classificationLossOverrides).toEqual({})
    for (const key of CLASSIFICATION_LOSS_FIELDS) expect(config).not.toHaveProperty(key)
  })

  it.each(['esm2', 'protbert', 'prott5', 'esmc', 'esm3'])(
    'round-trips explicit losses and only supplied parameters for %s pooled MLP',
    (encoder) => {
      const draft = useExperimentDraftStore()
      const base = { tokenizer: encoder, encoder, model: 'mlp', classifier: 'mlp' }
      for (const overrides of [
        { multilabel_loss: 'auto' },
        { multilabel_loss: 'bce' },
        { multilabel_loss: 'balanced_bce', multilabel_pos_weight_cap: 7 },
        { multilabel_loss: 'asymmetric_focal', asl_gamma_pos: 0, asl_gamma_neg: 3, asl_clip: 0 },
      ]) {
        draft.cloneFromExperiment(experiment({ ...base, ...overrides }))
        const config = draft.toConfig({ isPairTask: false, encoderFamily: 'pretrained' })
        expect(config).toMatchObject(overrides)
        expect(Object.fromEntries(CLASSIFICATION_LOSS_FIELDS.filter((key) => key in config).map((key) => [key, config[key]]))).toEqual(overrides)
      }
    },
  )

  it.each(['pep_protein_interaction', 'pep_hlai', 'pep_hlaii'])(
    'round-trips pair loss options and excludes sequence options for %s',
    (task) => {
      const draft = useExperimentDraftStore()
      for (const loss of ['bce', 'wbce']) {
        draft.cloneFromExperiment(experiment({
          tokenizer: 'kmer', encoder: 'onehot', model: 'cnn', classifier: 'mlp',
          pair_strategy: 'single_stream', loss_type: loss, pos_weight_max: 9,
        }, { task_type: task }))
        const config = draft.toConfig({ isPairTask: true, encoderFamily: 'sequence' })
        expect(config).toMatchObject({ loss_type: loss, pos_weight_max: 9 })
        expect(config).not.toHaveProperty('multilabel_loss')
      }
      draft.resetForTask(task)
      expect(draft.classificationLossOverrides).toEqual({})
    },
  )

  it('omits latent loss settings for recommended, pure classical, and ensemble routes', () => {
    const draft = useExperimentDraftStore()
    draft.cloneFromExperiment(experiment({ tokenizer: 'esm2', encoder: 'esm2', model: 'mlp', multilabel_loss: 'asymmetric_focal' }))
    const options = { isPairTask: false, encoderFamily: 'pretrained' }
    draft.mode = 'recommended'
    expect(draft.toConfig(options)).not.toHaveProperty('multilabel_loss')
    draft.mode = 'custom'
    draft.classifier = 'svm'
    expect(draft.toConfig(options)).not.toHaveProperty('multilabel_loss')
    draft.classifier = 'mlp'
    draft.enableEnsemble = true
    draft.ensembleMembers = 'a,b'
    expect(draft.toConfig(options)).not.toHaveProperty('multilabel_loss')
  })

  it('retains loss selection for the neural stage of embedding export', () => {
    const draft = useExperimentDraftStore()
    draft.cloneFromExperiment(experiment({ tokenizer: 'kmer', encoder: 'onehot', model: 'cnn', sequence_model: 'cnn', training_mode: 'export_embeddings', classifier: 'svm', multilabel_loss: 'balanced_bce' }))
    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' }).multilabel_loss).toBe('balanced_bce')
  })

  it('round-trips explicit dual parameters and historical checkpoint thresholds', () => {
    const draft = useExperimentDraftStore()
    const params = { reference_f1: 0.4, reference_sacc: 0.6 }
    draft.cloneFromExperiment(
      experiment({
        tokenizer: 'esm2',
        encoder: 'esm2',
        model: 'mlp',
        classifier: 'mlp',
        checkpoint_objective: 'macro_f1',
        threshold_objective: 'dual_f1_sacc',
        threshold_objective_params: params,
        checkpoint_threshold_objective: 'macro_f1',
        checkpoint_threshold_search_profile: 'f1_exact_v1',
      }),
    )
    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'pretrained' })).toMatchObject({
      checkpoint_objective: 'macro_f1',
      threshold_objective: 'dual_f1_sacc',
      threshold_objective_params: params,
      checkpoint_threshold_objective: 'macro_f1',
      checkpoint_threshold_search_profile: 'f1_exact_v1',
    })
    draft.resetForTask('pep_hlai')
    const pair = draft.toConfig({ isPairTask: true, encoderFamily: 'sequence' })
    expect(pair.checkpoint_objective).toBe('auc')
    expect(pair.threshold_objective).toBe('f1')
    expect(pair).not.toHaveProperty('threshold_objective_params')
  })

  it('round-trips the opt-in semantic loss and restores legacy when cloning an old run', () => {
    const draft = useExperimentDraftStore()
    const base = {
      tokenizer: 'kmer',
      encoder: 'onehot',
      model: 'cnn',
      classifier: 'mlp',
      enable_semantic_similarity: true,
    }
    draft.cloneFromExperiment(experiment({ ...base, semantic_loss_type: 'jaccard_supcon' }))
    expect(draft.semanticLossType).toBe('jaccard_supcon')
    expect(
      draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' }).semantic_loss_type,
    ).toBe('jaccard_supcon')
    draft.enableSemanticSimilarity = false
    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })).not.toHaveProperty(
      'semantic_loss_type',
    )
    draft.cloneFromExperiment(experiment(base))
    expect(draft.semanticLossType).toBe('legacy')
    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })).not.toHaveProperty(
      'semantic_loss_type',
    )
    draft.semanticLossType = 'jaccard_supcon'
    draft.resetForTask('pep_hlai')
    expect(draft.semanticLossType).toBe('legacy')
  })

  it.each(['pep_protein_interaction', 'pep_hlaii', 'pep_hlai'])(
    'preserves the new semantic loss when cloning %s',
    (task) => {
      const draft = useExperimentDraftStore()
      draft.cloneFromExperiment(
        experiment(
          {
            tokenizer: 'kmer',
            encoder: 'onehot',
            model: 'cnn',
            classifier: 'mlp',
            pair_strategy: 'single_stream',
            predictor_type: 'classifier',
            enable_semantic_similarity: true,
            semantic_loss_type: 'jaccard_supcon',
          },
          { task_type: task },
        ),
      )
      expect(
        draft.toConfig({ isPairTask: true, encoderFamily: 'sequence' }).semantic_loss_type,
      ).toBe('jaccard_supcon')
    },
  )

  it.each(['pep_function', 'pep_protein_interaction', 'pep_hlaii', 'pep_hlai'])(
    'round-trips semantic partner counts and historical defaults for %s',
    (task) => {
      const draft = useExperimentDraftStore()
      const options = { isPairTask: task !== 'pep_function', encoderFamily: 'sequence' }
      const base = {
        tokenizer: 'kmer',
        encoder: 'onehot',
        model: 'cnn',
        classifier: 'mlp',
        pair_strategy: 'single_stream',
        predictor_type: 'classifier',
        enable_semantic_similarity: true,
      }
      for (const loss of ['legacy', 'jaccard_supcon']) {
        for (const [positive, zero] of [
          [1, 8],
          [4, 16],
          [4, 0],
        ]) {
          draft.cloneFromExperiment(
            experiment(
              {
                ...base,
                semantic_loss_type: loss,
                semantic_positive_partners: positive,
                semantic_zero_partners: zero,
              },
              { task_type: task },
            ),
          )
          expect(draft.semanticPositivePartners).toBe(positive)
          expect(draft.semanticZeroPartners).toBe(zero)
          expect(draft.toConfig(options)).toMatchObject({
            semantic_positive_partners: positive,
            semantic_zero_partners: zero,
          })
        }
      }
      draft.cloneFromExperiment(experiment(base, { task_type: task }))
      expect(draft.toConfig(options)).toMatchObject({
        semantic_positive_partners: 1,
        semantic_zero_partners: 8,
      })
      expect(draft.toConfig(options)).not.toHaveProperty('semantic_loss_type')
      draft.enableSemanticSimilarity = false
      expect(draft.toConfig(options)).not.toHaveProperty('semantic_positive_partners')
      expect(draft.toConfig(options)).not.toHaveProperty('semantic_zero_partners')
      draft.semanticPositivePartners = 4
      draft.semanticZeroPartners = 0
      draft.resetForTask(task)
      expect(draft.semanticPositivePartners).toBe(1)
      expect(draft.semanticZeroPartners).toBe(8)
    },
  )

  it('retains zero and edited counts in an analysis handoff', () => {
    const draft = useExperimentDraftStore()
    draft.applyHandoff({
      origin: 'experiment_analysis',
      task_type: 'pep_function',
      dataset_id: 'dataset-1',
      config: {
        tokenizer: 'kmer',
        encoder: 'onehot',
        model: 'cnn',
        classifier: 'mlp',
        enable_semantic_similarity: true,
        semantic_positive_partners: 4,
        semantic_zero_partners: 0,
      },
      title: 'Multi-positive semantic run',
      rationale: 'Compare partner counts.',
      config_changes: [],
    })
    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })).toMatchObject({
      semantic_positive_partners: 4,
      semantic_zero_partners: 0,
    })
  })

  it('omits latent counts for classical, export, and pair sample-graph routes', () => {
    const draft = useExperimentDraftStore()
    draft.mode = 'custom'
    draft.encoder = 'bow'
    draft.classifier = 'rf'
    draft.enableSemanticSimilarity = true
    draft.semanticPositivePartners = 4
    draft.semanticZeroPartners = 0
    let config = draft.toConfig({ isPairTask: false, encoderFamily: 'global' })
    expect(config).not.toHaveProperty('semantic_positive_partners')
    expect(config).not.toHaveProperty('semantic_zero_partners')
    draft.encoder = 'onehot'
    draft.trainingMode = 'export_embeddings'
    config = draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })
    expect(config).not.toHaveProperty('semantic_positive_partners')
    expect(config).not.toHaveProperty('semantic_zero_partners')
    draft.resetForTask('pep_hlai')
    draft.mode = 'custom'
    draft.predictorType = 'sample_graph'
    draft.enableSemanticSimilarity = true
    config = draft.toConfig({ isPairTask: true, encoderFamily: 'sequence' })
    expect(config).not.toHaveProperty('semantic_positive_partners')
    expect(config).not.toHaveProperty('semantic_zero_partners')
  })

  it('applies an experiment-analysis handoff with provenance intact', () => {
    const draft = useExperimentDraftStore()
    const handoff = {
      origin: 'experiment_analysis' as const,
      task_type: 'pep_function',
      dataset_id: 'dataset-1',
      config: {
        tokenizer: 'kmer',
        encoder: 'onehot',
        model: 'rf',
        classifier: 'rf',
        rf_estimators: 500,
      },
      title: 'Run a random-forest baseline',
      rationale: 'Adds a reusable classical comparison.',
      config_changes: [{ path: 'model', before: 'cnn', after: 'rf' }],
      source_experiment_id: 'source-run',
      source_experiment_name: 'CNN baseline',
      conversation_id: 'conversation-1',
      message_id: 'message-2',
      recommendation_id: 'recommendation-1',
    }

    draft.applyHandoff(handoff)

    expect(draft.taskType).toBe('pep_function')
    expect(draft.datasetId).toBe('dataset-1')
    expect(draft.classifier).toBe('rf')
    expect(draft.rfEstimators).toBe(500)
    expect(draft.initialStep).toBe(4)
    expect(draft.cloneSourceExperimentId).toBe('source-run')
    expect(draft.handoff).toEqual(handoff)
  })

  it('applies and detaches an Agent candidate handoff with plan provenance', () => {
    const draft = useExperimentDraftStore()
    const preflight = {
      valid: true,
      errors: [],
      error_codes: [],
      warnings: ['No test split'],
      warning_codes: ['no_test_split'],
      normalized_config: {
        tokenizer: 'kmer',
        encoder: 'index',
        model: 'cnn',
        epochs: 12,
      },
      parameter_sources: { epochs: 'explicit' as const },
      dataset: {
        id: 'dataset-1',
        name: 'Agent dataset',
        task_type: 'pep_function',
        format: 'sequence_dir',
        metadata: {},
      },
      execution: {
        script: 'scripts/train_sequence_pipeline.py',
        supports_reusable_inference: true,
      },
    }

    draft.applyAgentCandidate(
      'pep_function',
      'dataset-1',
      preflight.normalized_config,
      'Reviewed CNN',
      {
        rationale: 'A low-risk baseline.',
        planId: 'plan-1',
        actionId: 'action-1',
        lane: 'conservative',
        preflight,
      },
    )

    expect(draft.initialStep).toBe(4)
    expect(draft.handoff).toMatchObject({
      origin: 'agent',
      agent_plan_id: 'plan-1',
      agent_action_id: 'action-1',
      agent_lane: 'conservative',
      agent_preflight: preflight,
    })
    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })).toMatchObject({
      tokenizer: 'kmer',
      encoder: 'index',
      model: 'cnn',
      epochs: 12,
    })

    draft.detachAgentHandoff()
    expect(draft.handoff).toBeNull()
    expect(draft.datasetId).toBe('dataset-1')
  })

  it('uses task-aware tuning defaults while preserving cloned explicit values', () => {
    const draft = useExperimentDraftStore()

    draft.resetForTask('pep_function')
    expect(draft.mlTuningObjective).toBe('f1')

    draft.resetForTask('pep_hlai')
    expect(draft.mlTuningObjective).toBe('auc')

    draft.cloneFromExperiment(
      experiment(
        { model: 'rf', classifier: 'rf', ml_tuning_objective: 'f1' },
        { task_type: 'pep_hlai' },
      ),
    )
    expect(draft.mlTuningObjective).toBe('f1')
  })

  it('serializes bounded tuning and canonical calibration fields for SVM', () => {
    const draft = useExperimentDraftStore()
    draft.resetForTask('pep_function')
    draft.mode = 'custom'
    draft.classifier = 'svm'
    draft.svmCalibration = 'sigmoid'
    draft.sequenceModel = ''
    draft.encoder = 'bow'
    draft.mlAutoTune = true
    draft.mlTuningBudget = 'standard'
    draft.mlTuningObjective = 'auc'

    const config = draft.toConfig({ isPairTask: false, encoderFamily: 'global' })

    expect(config).toMatchObject({
      model: 'svm',
      classifier: 'svm',
      svm_calibration: 'sigmoid',
      svm_calibration_cv_folds: 3,
      ml_auto_tune: true,
      ml_tuning_budget: 'standard',
      ml_tuning_mode: 'cv',
      ml_tuning_max_candidates: 12,
      ml_cv_folds: 3,
      ml_tuning_objective: 'auc',
    })
  })

  it('loads the legacy calibrated SVM alias as canonical sigmoid SVM', () => {
    const draft = useExperimentDraftStore()
    draft.cloneFromExperiment(
      experiment({
        tokenizer: 'kmer',
        encoder: 'bow',
        model: 'calibrated_svm',
        classifier: 'calibrated_svm',
      }),
    )

    const config = draft.toConfig({ isPairTask: false, encoderFamily: 'global' })

    expect(draft.classifier).toBe('svm')
    expect(draft.svmCalibration).toBe('sigmoid')
    expect(config).toMatchObject({
      model: 'svm',
      classifier: 'svm',
      svm_calibration: 'sigmoid',
      svm_calibration_cv_folds: 3,
    })
  })

  it('round-trips the BoW/TF-IDF raw-sequence safety limit', () => {
    const draft = useExperimentDraftStore()
    draft.cloneFromExperiment(
      experiment({
        tokenizer: 'kmer',
        encoder: 'tfidf',
        model: 'rf',
        classifier: 'rf',
        frequency_max_raw_length: 4321,
      }),
    )

    const config = draft.toConfig({
      isPairTask: false,
      encoderFamily: 'global',
    })

    expect(draft.frequencyMaxRawLength).toBe(4321)
    expect(config.frequency_max_raw_length).toBe(4321)
  })

  it('rebuilds a legacy pair single-stream route from model-only config', () => {
    const draft = useExperimentDraftStore()

    draft.cloneFromExperiment(
      experiment(
        {
          tokenizer: 'kmer',
          encoder: 'index',
          model: 'cnn',
          epochs: 7,
          batch_size: 12,
        },
        { task_type: 'pep_hlaii' },
      ),
    )

    expect(draft.pairStrategy).toBe('single_stream')
    expect(draft.modelingArchitecture).toBe('cnn')
    expect(draft.initialStep).toBe(4)

    const config = draft.toConfig({ isPairTask: true, encoderFamily: 'sequence' })
    expect(config).toMatchObject({
      tokenizer: 'kmer',
      encoder: 'index',
      model: 'cnn',
      pair_strategy: 'single_stream',
      sequence_model: 'cnn',
      training_mode: 'end_to_end',
      classifier: 'mlp',
      epochs: 7,
      batch_size: 12,
    })
  })

  it('rebuilds a legacy token-interaction route from interaction model aliases', () => {
    const draft = useExperimentDraftStore()

    draft.cloneFromExperiment(
      experiment(
        {
          tokenizer: 'esm2',
          encoder: 'ppi_fusion',
          model: 'cross_attn+mlp',
          training_mode: 'export_embeddings',
          classifier: 'rf',
        },
        { task_type: 'pep_protein_interaction' },
      ),
    )

    expect(draft.pairStrategy).toBe('two_stream')
    expect(draft.interactionLevel).toBe('token')
    expect(draft.interactionModel).toBe('cross_attn')

    const config = draft.toConfig({ isPairTask: true, encoderFamily: 'global' })
    expect(config).toMatchObject({
      tokenizer: 'esm2',
      encoder: 'esm2',
      model: 'cross_attn',
      pair_strategy: 'two_stream',
      interaction_level: 'token',
      interaction_model: 'cross_attn',
      training_mode: 'export_embeddings',
      classifier: 'rf',
    })
  })

  it('rebuilds a legacy graph pair route from graph model config', () => {
    const draft = useExperimentDraftStore()

    draft.cloneFromExperiment(
      experiment(
        {
          tokenizer: 'esm2',
          encoder: 'ppi_fusion',
          model: 'gcn',
          fusion_mode: 'gated',
        },
        { task_type: 'pep_protein_interaction' },
      ),
    )

    expect(draft.pairStrategy).toBe('two_stream')
    expect(draft.interactionLevel).toBe('vector')
    expect(draft.predictorType).toBe('sample_graph')
    expect(draft.graphPredictor).toBe('gcn')

    const config = draft.toConfig({ isPairTask: true, encoderFamily: 'global' })
    expect(config).toMatchObject({
      model: 'gcn',
      pair_strategy: 'two_stream',
      interaction_level: 'vector',
      predictor_type: 'sample_graph',
      graph_predictor: 'gcn',
      fusion_mode: 'gated',
    })
  })

  it('serializes pretrained residue modeling with export mode and adapter dimension', () => {
    const draft = useExperimentDraftStore()
    draft.mode = 'custom'
    draft.featureRoute = 'pretrained'
    draft.tokenizer = 'esm2'
    draft.pretrainedPredictorRoute = 'residue_modeling'
    draft.modelingArchitecture = 'gat'
    draft.trainingMode = 'export_embeddings'
    draft.classifier = 'rf'
    draft.residueAdapterDim = 256

    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })).toMatchObject({
      tokenizer: 'esm2',
      encoder: 'esm2',
      model: 'gat',
      sequence_model: 'gat',
      training_mode: 'export_embeddings',
      classifier: 'rf',
      residue_adapter_dim: 256,
    })
  })

  it('falls back to a classical classifier for Pair token export serialization', () => {
    const draft = useExperimentDraftStore()
    draft.resetForTask('pep_protein_interaction')
    draft.mode = 'custom'
    draft.featureRoute = 'classic'
    draft.tokenizer = 'kmer'
    draft.encoder = 'index'
    draft.pairStrategy = 'two_stream'
    draft.interactionLevel = 'token'
    draft.interactionModel = 'cross_attn'
    draft.trainingMode = 'export_embeddings'
    draft.classifier = 'mlp'

    const config = draft.toConfig({ isPairTask: true, encoderFamily: 'sequence' })

    expect(config).toMatchObject({
      training_mode: 'export_embeddings',
      classifier: 'svm',
    })
  })

  it('serializes the selected official ESM-C parameter scale', () => {
    const draft = useExperimentDraftStore()
    draft.mode = 'custom'
    draft.featureRoute = 'pretrained'
    draft.tokenizer = 'esmc'
    draft.encoder = 'esmc'
    draft.esmcModel = 'EvolutionaryScale/esmc-300m-2024-12'

    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'pretrained' })).toMatchObject({
      tokenizer: 'esmc',
      encoder: 'esmc',
      esmc_model: 'EvolutionaryScale/esmc-300m-2024-12',
    })
  })

  it('keeps recommended pretrained routes on pooled classification', () => {
    const draft = useExperimentDraftStore()
    draft.mode = 'recommended'
    draft.featureRoute = 'pretrained'
    draft.tokenizer = 'esm2'
    draft.pretrainedPredictorRoute = 'residue_modeling'
    draft.modelingArchitecture = 'gcn'
    draft.classifier = 'mlp'

    const config = draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })
    expect(config).toMatchObject({ model: 'mlp', classifier: 'mlp' })
    expect(config).not.toHaveProperty('sequence_model')
  })

  it('keeps recommended classic index pair routes behind a learned embedding', () => {
    const draft = useExperimentDraftStore()
    draft.resetForTask('pep_hlaii')
    draft.mode = 'recommended'

    const config = draft.toConfig({ isPairTask: true, encoderFamily: 'sequence' })

    expect(config).toMatchObject({
      tokenizer: 'kmer',
      encoder: 'index',
      pair_strategy: 'single_stream',
      model: 'cnn',
      sequence_model: 'cnn',
      classifier: 'mlp',
    })
    expect(config).not.toHaveProperty('streams')
    expect(config).not.toHaveProperty('interaction_level')
  })

  it('migrates legacy ESM2 residue graph configs to unified residue modeling', () => {
    const draft = useExperimentDraftStore()
    draft.cloneFromExperiment(
      experiment({
        tokenizer: 'esm2',
        encoder: 'index',
        model: 'gcn',
        sequence_model: 'gcn',
        training_mode: 'end_to_end',
        classifier: 'mlp',
      }),
    )

    expect(draft.pretrainedPredictorRoute).toBe('residue_modeling')
    expect(draft.modelingArchitecture).toBe('gcn')
    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })).toMatchObject({
      model: 'gcn',
      sequence_model: 'gcn',
      training_mode: 'end_to_end',
      classifier: 'mlp',
      residue_adapter_dim: 128,
    })
  })

  it('round-trips ProtT5 residue LSTM routes', () => {
    const draft = useExperimentDraftStore()
    draft.cloneFromExperiment(
      experiment({
        tokenizer: 'prott5',
        encoder: 'prott5',
        model: 'lstm',
        sequence_model: 'lstm',
        training_mode: 'end_to_end',
        classifier: 'mlp',
        residue_adapter_dim: 192,
      }),
    )

    expect(draft.pretrainedPredictorRoute).toBe('residue_modeling')
    expect(draft.residueAdapterDim).toBe(192)
    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })).toMatchObject({
      tokenizer: 'prott5',
      encoder: 'prott5',
      sequence_model: 'lstm',
      residue_adapter_dim: 192,
    })
  })

  it('keeps hidden profile fields only while they still match the rerun route', () => {
    const draft = useExperimentDraftStore()

    draft.cloneFromExperiment(
      experiment({
        tokenizer: 'top-n-gram',
        encoder: 'bow',
        model: 'rf',
        classifier: 'rf',
        profile_iterations: 4,
        profile_evalue: 0.01,
        profile_on_missing: 'error',
        frequency_profile_index_path: '/data/source/target_frequency/index.json',
        feature_dir: '/data/source/features',
        report_aupr: false,
        cache_features: false,
        unknown_knob: 'drop-me',
      }),
    )

    const clonedConfig = draft.toConfig({ isPairTask: false, encoderFamily: 'global' })
    expect(clonedConfig).toMatchObject({
      profile_iterations: 4,
      profile_evalue: 0.01,
      profile_on_missing: 'error',
      frequency_profile_index_path: '/data/source/target_frequency/index.json',
      report_aupr: false,
      cache_features: false,
    })
    expect(clonedConfig).not.toHaveProperty('feature_dir')
    expect(clonedConfig).not.toHaveProperty('unknown_knob')

    draft.tokenizer = 'kmer'
    draft.encoder = 'index'
    const changedRouteConfig = draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })
    expect(changedRouteConfig).not.toHaveProperty('profile_iterations')
    expect(changedRouteConfig).not.toHaveProperty('profile_evalue')
    expect(changedRouteConfig).not.toHaveProperty('profile_on_missing')
    expect(changedRouteConfig).not.toHaveProperty('frequency_profile_index_path')
    expect(changedRouteConfig).toMatchObject({ report_aupr: false, cache_features: false })
  })

  it('drops dataset-scoped hidden paths when the rerun dataset changes', () => {
    const draft = useExperimentDraftStore()

    draft.cloneFromExperiment(
      experiment(
        {
          tokenizer: 'kmer',
          encoder: 'pssm',
          model: 'rf',
          classifier: 'rf',
          profile_threads: 2,
          pssm_index_path: '/data/source/pssm/index.json',
          feature_dir: '/data/source/features',
        },
        { task_type: 'pep_protein_interaction' },
      ),
    )
    draft.datasetId = 'dataset-2'

    const config = draft.toConfig({ isPairTask: true, encoderFamily: 'sequence' })
    expect(config).toMatchObject({ profile_threads: 2 })
    expect(config).not.toHaveProperty('pssm_index_path')
    expect(config).not.toHaveProperty('feature_dir')
  })

  it('restores and submits standard GloVe training parameters', () => {
    const draft = useExperimentDraftStore()

    draft.cloneFromExperiment(
      experiment({
        tokenizer: 'kmer',
        encoder: 'glove',
        model: 'cnn',
        embedding_vector_size: 64,
        embedding_window: 7,
        embedding_min_count: 2,
        embedding_epochs: 30,
        glove_x_max: 10,
        glove_alpha: 0.5,
        glove_learning_rate: 0.1,
        glove_batch_size: 512,
      }),
    )
    draft.mode = 'custom'

    const config = draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })

    expect(config).toMatchObject({
      embedding_vector_size: 64,
      embedding_window: 7,
      embedding_min_count: 2,
      embedding_epochs: 30,
      glove_x_max: 10,
      glove_alpha: 0.5,
      glove_learning_rate: 0.1,
      glove_batch_size: 512,
    })
    expect(config).not.toHaveProperty('embedding_workers')
  })

  it('restores and submits complete fastText training parameters', () => {
    const draft = useExperimentDraftStore()

    draft.cloneFromExperiment(
      experiment({
        tokenizer: 'kmer',
        encoder: 'fasttext',
        model: 'cnn',
        embedding_vector_size: 48,
        embedding_window: 4,
        embedding_min_count: 2,
        embedding_epochs: 12,
        embedding_workers: 2,
        fasttext_bucket: 500000,
        static_embedding_memory_limit_gib: 3,
      }),
    )
    draft.mode = 'custom'

    const config = draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })

    expect(config).toMatchObject({
      embedding_vector_size: 48,
      embedding_window: 4,
      embedding_min_count: 2,
      embedding_epochs: 12,
      embedding_workers: 2,
      fasttext_bucket: 500000,
      static_embedding_memory_limit_gib: 3,
    })
    expect(config).not.toHaveProperty('glove_x_max')
  })

  it('round-trips independent peptide and partner stream specs', () => {
    const draft = useExperimentDraftStore()
    const streams = {
      peptide: {
        feature_route: 'pretrained',
        tokenizer: 'esm2',
        encoder: 'ppi_fusion',
        stream_model: '',
        model_source: '/models/esm2',
        max_length: 64,
        params: {},
      },
      partner: {
        feature_route: 'pretrained',
        tokenizer: 'prott5',
        encoder: 'ppi_fusion',
        stream_model: '',
        model_source: '/models/prott5',
        max_length: 256,
        params: {},
      },
    }
    draft.cloneFromExperiment(
      experiment(
        {
          tokenizer: 'esm2',
          encoder: 'ppi_fusion',
          model: 'mlp',
          classifier: 'mlp',
          pair_strategy: 'two_stream',
          interaction_level: 'vector',
          predictor_type: 'classifier',
          fusion_mode: 'concat',
          streams,
        },
        { task_type: 'pep_protein_interaction' },
      ),
    )

    expect(draft.streamsLinked).toBe(false)
    expect(draft.tokenizer).toBe('esm2')
    expect(draft.partnerTokenizer).toBe('prott5')
    expect(draft.maxLength).toBe(64)
    expect(draft.partnerMaxLength).toBe(256)
    expect(draft.toConfig({ isPairTask: true, encoderFamily: 'global' })).toMatchObject({
      pair_strategy: 'two_stream',
      fusion_mode: 'concat',
      streams: {
        peptide: { ...streams.peptide, encoder: 'esm2' },
        partner: { ...streams.partner, encoder: 'prott5' },
      },
    })
  })

  it('serializes HGT with typed role inputs and no fusion mode', () => {
    const draft = useExperimentDraftStore()
    draft.taskType = 'pep_protein_interaction'
    draft.mode = 'custom'
    draft.featureRoute = 'pretrained'
    draft.tokenizer = 'esm2'
    draft.encoder = 'esm2'
    draft.pairStrategy = 'two_stream'
    draft.interactionLevel = 'vector'
    draft.predictorType = 'sample_graph'
    draft.graphPredictor = 'hgt'
    draft.fusionMode = 'concat_all'

    const config = draft.toConfig({ isPairTask: true, encoderFamily: 'global' })

    expect(config).toMatchObject({
      predictor_type: 'sample_graph',
      graph_predictor: 'hgt',
      model: 'hgt',
    })
    expect(config).not.toHaveProperty('fusion_mode')
  })

  it('round-trips a configurable end-to-end prediction head', () => {
    const draft = useExperimentDraftStore()
    draft.cloneFromExperiment(
      experiment({
        tokenizer: 'kmer',
        encoder: 'index',
        model: 'cnn',
        sequence_model: 'cnn',
        training_mode: 'end_to_end',
        classifier: 'mlp',
        head_hidden_dims: [256, 128],
        head_activation: 'silu',
        head_dropout: 0.15,
      }),
    )
    draft.mode = 'custom'

    expect(draft.headHiddenLayerCount).toBe(2)
    expect(draft.headActivation).toBe('silu')
    expect(draft.headDropout).toBe(0.15)
    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })).toMatchObject({
      head_hidden_dims: [256, 128],
      head_activation: 'silu',
      head_dropout: 0.15,
    })
  })

  it('serializes the linear end-to-end prediction head without inactive fields', () => {
    const draft = useExperimentDraftStore()
    draft.mode = 'custom'
    draft.headHiddenLayerCount = 0
    const config = draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })
    expect(config.head_hidden_dims).toEqual([])
    expect(config).not.toHaveProperty('head_activation')
    expect(config).not.toHaveProperty('head_dropout')
  })

  it('maps arbitrary cloned head widths to the fixed Web layer-count presets', () => {
    const draft = useExperimentDraftStore()
    draft.cloneFromExperiment(
      experiment({
        tokenizer: 'esm2',
        encoder: 'esm2',
        model: 'mlp',
        classifier: 'mlp',
        head_hidden_dims: [512, 256, 64],
      }),
    )
    draft.mode = 'custom'

    expect(draft.headHiddenLayerCount).toBe(2)
    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'pretrained' })).toMatchObject({
      head_hidden_dims: [256, 128],
    })
  })

  it('serializes DT profile defaults for sequence and pair routes', () => {
    const draft = useExperimentDraftStore()
    draft.mode = 'custom'
    draft.featureRoute = 'classic'
    draft.tokenizer = 'dt'
    draft.encoder = 'bow'
    draft.classifier = 'mlp'

    const sequenceConfig = draft.toConfig({ isPairTask: false, encoderFamily: 'global' })
    expect(sequenceConfig).toMatchObject({
      tokenizer: 'dt',
      profile_on_missing: 'blosum62_target_frequency',
      dt_max_distance: 5,
      dt_dense_memory_limit_gib: 4,
    })
    expect(sequenceConfig).not.toHaveProperty('dt_hash_size')

    draft.resetForTask('pep_protein_interaction')
    draft.mode = 'custom'
    draft.featureRoute = 'classic'
    draft.tokenizer = 'dt'
    draft.encoder = 'bow'
    draft.pairStrategy = 'two_stream'
    draft.interactionLevel = 'vector'
    draft.predictorType = 'classifier'
    draft.classifier = 'mlp'
    draft.streamsLinked = true

    const linkedPairConfig = draft.toConfig({ isPairTask: true, encoderFamily: 'global' })
    expect(linkedPairConfig).toMatchObject({
      profile_on_missing: 'blosum62_target_frequency',
      streams: {
        peptide: { tokenizer: 'dt' },
        partner: { tokenizer: 'dt' },
      },
    })

    draft.tokenizer = 'kmer'
    draft.encoder = 'bow'
    draft.streamsLinked = false
    draft.partnerFeatureRoute = 'classic'
    draft.partnerTokenizer = 'dt'
    draft.partnerEncoder = 'bow'

    const unlinkedPairConfig = draft.toConfig({ isPairTask: true, encoderFamily: 'global' })
    expect(unlinkedPairConfig).toMatchObject({
      profile_on_missing: 'blosum62_target_frequency',
      streams: {
        peptide: { tokenizer: 'kmer' },
        partner: { tokenizer: 'dt' },
      },
    })
  })

  it('preserves an explicit strict DT profile fallback when cloning', () => {
    const draft = useExperimentDraftStore()
    draft.cloneFromExperiment(
      experiment({
        tokenizer: 'dt',
        encoder: 'bow',
        model: 'mlp',
        classifier: 'mlp',
        profile_on_missing: 'error',
      }),
    )

    const config = draft.toConfig({ isPairTask: false, encoderFamily: 'global' })
    expect(config.profile_on_missing).toBe('error')

    draft.tokenizer = 'kmer'
    draft.encoder = 'index'
    const unrelatedConfig = draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })
    expect(unrelatedConfig).not.toHaveProperty('profile_on_missing')
  })

  it('forces k=1 for profile encoders in sequence and pair configs', () => {
    const draft = useExperimentDraftStore()
    draft.mode = 'custom'
    draft.featureRoute = 'classic'
    draft.tokenizer = 'kmer'
    draft.encoder = 'pssm_blosum62_hybrid'
    draft.k = 2

    const sequenceConfig = draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })
    expect(sequenceConfig).toMatchObject({
      tokenizer: 'kmer',
      encoder: 'pssm_blosum62_hybrid',
      k: 1,
      profile_on_missing: 'blosum62_hybrid',
    })

    draft.resetForTask('pep_protein_interaction')
    draft.mode = 'custom'
    draft.featureRoute = 'classic'
    draft.tokenizer = 'kmer'
    draft.encoder = 'pssm'
    draft.k = 2
    draft.pairStrategy = 'two_stream'
    draft.interactionLevel = 'vector'
    draft.streamsLinked = true

    const pairConfig = draft.toConfig({ isPairTask: true, encoderFamily: 'sequence' })
    expect(pairConfig).toMatchObject({
      streams: {
        peptide: { params: { k: 1 } },
        partner: { params: { k: 1 } },
      },
    })

    draft.streamsLinked = false
    draft.partnerFeatureRoute = 'classic'
    draft.partnerTokenizer = 'kmer'
    draft.partnerEncoder = 'psfm'
    draft.partnerK = 3

    const heterogeneousConfig = draft.toConfig({ isPairTask: true, encoderFamily: 'sequence' })
    expect(heterogeneousConfig).toMatchObject({
      streams: {
        peptide: { params: { k: 1 } },
        partner: { params: { k: 1 } },
      },
    })
  })

  it('round-trips the independent PSFM-BLOSUM62 Hybrid route', () => {
    const draft = useExperimentDraftStore()
    draft.cloneFromExperiment(
      experiment({
        tokenizer: 'kmer',
        encoder: 'psfm_blosum62_hybrid',
        model: 'cnn',
        classifier: 'mlp',
        k: 3,
        psfm_index_path: '/data/source/psfm-hybrid/index.json',
        profile_on_missing: 'blosum62_hybrid',
      }),
    )

    const config = draft.toConfig({ isPairTask: false, encoderFamily: 'sequence' })
    expect(config).toMatchObject({
      tokenizer: 'kmer',
      encoder: 'psfm_blosum62_hybrid',
      k: 1,
      psfm_index_path: '/data/source/psfm-hybrid/index.json',
      profile_on_missing: 'blosum62_hybrid',
    })
  })

  it('raises profile-tokenizer lengths to dataset-derived floors', () => {
    const draft = useExperimentDraftStore()
    const metadata = {
      length_stats: {
        version: 1,
        by_split: {},
        overall: {
          sequence: { max: 207 },
          peptide: { max: 50 },
          partner: { max: 5037 },
        },
      },
    }

    draft.tokenizer = 'top-n-gram'
    draft.maxLength = 60
    draft.applyProfileTokenizerDatasetLengthFloor(metadata, false)
    expect(draft.maxLength).toBe(207)

    draft.maxLength = 60
    draft.partnerMaxLength = 512
    draft.streamsLinked = true
    draft.applyProfileTokenizerDatasetLengthFloor(metadata, true)
    expect(draft.maxLength).toBe(5037)
    expect(draft.partnerMaxLength).toBe(5037)

    draft.streamsLinked = false
    draft.maxLength = 60
    draft.partnerMaxLength = 512
    draft.partnerTokenizer = 'top-n-gram'
    draft.applyProfileTokenizerDatasetLengthFloor(metadata, true)
    expect(draft.maxLength).toBe(60)
    expect(draft.partnerMaxLength).toBe(5037)

    draft.tokenizer = 'dt'
    draft.partnerTokenizer = 'kmer'
    draft.maxLength = 20
    draft.partnerMaxLength = 512
    draft.applyProfileTokenizerDatasetLengthFloor(metadata, true)
    expect(draft.maxLength).toBe(50)
    expect(draft.partnerMaxLength).toBe(512)
  })

  it('serializes covariance lag and forces k=1 for every fixed-vector encoder', () => {
    const draft = useExperimentDraftStore()
    draft.mode = 'custom'
    draft.featureRoute = 'classic'
    draft.tokenizer = 'kmer'
    draft.classifier = 'svm'
    draft.k = 3
    draft.acLag = 2
    draft.ccLag = 3
    draft.accLag = 4
    draft.covarianceMaxRawLength = 321

    for (const [encoder, lagKey, lag] of [
      ['ac', 'ac_lag', 2],
      ['cc', 'cc_lag', 3],
      ['acc', 'acc_lag', 4],
    ] as const) {
      draft.encoder = encoder
      const config = draft.toConfig({ isPairTask: false, encoderFamily: 'global' })
      expect(config).toMatchObject({
        tokenizer: 'kmer',
        encoder,
        model: 'svm',
        classifier: 'svm',
        k: 1,
        [lagKey]: lag,
        covariance_max_raw_length: 321,
      })
    }
  })

  it('serializes role-specific covariance parameters for heterogeneous streams', () => {
    const draft = useExperimentDraftStore()
    draft.resetForTask('pep_protein_interaction')
    draft.mode = 'custom'
    draft.featureRoute = 'classic'
    draft.tokenizer = 'kmer'
    draft.encoder = 'ac'
    draft.k = 3
    draft.acLag = 7
    draft.covarianceMaxRawLength = 50
    draft.pairStrategy = 'two_stream'
    draft.interactionLevel = 'vector'
    draft.streamsLinked = false
    draft.partnerFeatureRoute = 'classic'
    draft.partnerTokenizer = 'kmer'
    draft.partnerEncoder = 'acc'
    draft.partnerK = 2
    draft.partnerAccLag = 9
    draft.partnerCovarianceMaxRawLength = 512

    const config = draft.toConfig({ isPairTask: true, encoderFamily: 'global' })
    expect(config).toMatchObject({
      streams: {
        peptide: {
          encoder: 'ac',
          params: {
            k: 1,
            ac_lag: 7,
            covariance_max_raw_length: 50,
          },
        },
        partner: {
          encoder: 'acc',
          params: {
            k: 1,
            acc_lag: 9,
            covariance_max_raw_length: 512,
          },
        },
      },
    })
  })

  it('adapts only untouched PDT lag defaults to dataset length constraints', () => {
    const draft = useExperimentDraftStore()
    draft.tokenizer = 'kmer'
    draft.encoder = 'pdt'
    const sequenceMetadata = {
      length_stats: {
        overall: { sequence: { min: 5 } },
      },
    }

    draft.applyPdtDatasetLagDefaults(sequenceMetadata, false)
    expect(draft.pdtLag).toBe(4)

    draft.pdtLag = 3
    draft.pdtLagExplicit = true
    draft.applyPdtDatasetLagDefaults(
      {
        length_stats: { overall: { sequence: { min: 2 } } },
      },
      false,
    )
    expect(draft.pdtLag).toBe(3)

    draft.resetForTask('pep_protein_interaction')
    draft.tokenizer = 'kmer'
    draft.encoder = 'pdt'
    draft.streamsLinked = true
    const pairMetadata = {
      length_stats: {
        overall: {
          peptide: { min: 7 },
          partner: { min: 4 },
        },
      },
    }
    draft.applyPdtDatasetLagDefaults(pairMetadata, true)
    expect(draft.pdtLag).toBe(3)

    draft.streamsLinked = false
    draft.partnerTokenizer = 'kmer'
    draft.partnerEncoder = 'pdt'
    draft.applyPdtDatasetLagDefaults(pairMetadata, true)
    expect(draft.pdtLag).toBe(6)
    expect(draft.partnerPdtLag).toBe(3)

    draft.applyPdtDatasetLagDefaults({}, true)
    expect(draft.pdtLag).toBe(8)
    expect(draft.partnerPdtLag).toBe(8)

    draft.applyPdtDatasetLagDefaults(
      {
        length_stats: {
          overall: {
            peptide: { min: 1 },
            partner: { min: 1 },
          },
        },
      },
      true,
    )
    expect(draft.pdtLag).toBe(8)
    expect(draft.partnerPdtLag).toBe(8)
  })

  it('round-trips PDT parameters and forces k=1 for sequence and heterogeneous streams', () => {
    const draft = useExperimentDraftStore()
    draft.cloneFromExperiment(
      experiment({
        tokenizer: 'kmer',
        encoder: 'pdt',
        model: 'svm',
        classifier: 'svm',
        k: 3,
        pdt_lag: 4,
        pdt_max_raw_length: 321,
      }),
    )

    expect(draft.pdtLag).toBe(4)
    expect(draft.pdtLagExplicit).toBe(true)
    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'global' })).toMatchObject({
      tokenizer: 'kmer',
      encoder: 'pdt',
      k: 1,
      pdt_lag: 4,
      pdt_max_raw_length: 321,
    })

    draft.resetForTask('pep_protein_interaction')
    draft.mode = 'custom'
    draft.featureRoute = 'classic'
    draft.tokenizer = 'kmer'
    draft.encoder = 'pdt'
    draft.k = 3
    draft.pdtLag = 2
    draft.pdtMaxRawLength = 100
    draft.pairStrategy = 'two_stream'
    draft.interactionLevel = 'vector'
    draft.streamsLinked = false
    draft.partnerFeatureRoute = 'classic'
    draft.partnerTokenizer = 'kmer'
    draft.partnerEncoder = 'pdt'
    draft.partnerK = 2
    draft.partnerPdtLag = 5
    draft.partnerPdtMaxRawLength = 500

    expect(draft.toConfig({ isPairTask: true, encoderFamily: 'global' })).toMatchObject({
      streams: {
        peptide: {
          encoder: 'pdt',
          params: { k: 1, pdt_lag: 2, pdt_max_raw_length: 100 },
        },
        partner: {
          encoder: 'pdt',
          params: { k: 1, pdt_lag: 5, pdt_max_raw_length: 500 },
        },
      },
    })
  })

  it('hydrates and serializes role-specific PSSM covariance lag controls', () => {
    const draft = useExperimentDraftStore()
    draft.cloneFromExperiment(
      experiment({
        tokenizer: 'kmer',
        encoder: 'ac_pssm',
        model: 'svm',
        classifier: 'svm',
        k: 3,
        ac_pssm_lag: 4,
      }),
    )
    expect(draft.acPssmLag).toBe(4)
    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'global' })).toMatchObject({
      k: 1,
      ac_pssm_lag: 4,
    })

    draft.resetForTask('pep_protein_interaction')
    draft.mode = 'custom'
    draft.featureRoute = 'classic'
    draft.tokenizer = 'kmer'
    draft.encoder = 'cc_pssm'
    draft.ccPssmLag = 5
    draft.pairStrategy = 'two_stream'
    draft.interactionLevel = 'vector'
    draft.streamsLinked = false
    draft.partnerFeatureRoute = 'classic'
    draft.partnerTokenizer = 'kmer'
    draft.partnerEncoder = 'acc_pssm'
    draft.partnerAccPssmLag = 7

    const pair = draft.toConfig({ isPairTask: true, encoderFamily: 'global' })
    expect(pair).toMatchObject({
      streams: {
        peptide: { params: { k: 1, cc_pssm_lag: 5 } },
        partner: { params: { k: 1, acc_pssm_lag: 7 } },
      },
    })
  })

  it('round-trips an independent hybrid PSSM covariance sequence route', () => {
    const draft = useExperimentDraftStore()
    draft.cloneFromExperiment(
      experiment({
        tokenizer: 'kmer',
        encoder: 'ac_pssm_blosum62_hybrid',
        model: 'svm',
        classifier: 'svm',
        k: 3,
        ac_pssm_blosum62_hybrid_lag: 4,
        ac_pssm_blosum62_hybrid_index_path: '/data/source/ac-hybrid/index.json',
        pssm_index_path: '/data/source/generic-pssm/index.json',
        profile_on_missing: 'blosum62_hybrid',
      }),
    )

    const config = draft.toConfig({ isPairTask: false, encoderFamily: 'global' })
    expect(draft.acPssmBlosum62HybridLag).toBe(4)
    expect(config).toMatchObject({
      tokenizer: 'kmer',
      encoder: 'ac_pssm_blosum62_hybrid',
      k: 1,
      ac_pssm_blosum62_hybrid_lag: 4,
      ac_pssm_blosum62_hybrid_index_path: '/data/source/ac-hybrid/index.json',
      profile_on_missing: 'blosum62_hybrid',
    })
    expect(config).not.toHaveProperty('pssm_index_path')
  })

  it('round-trips independent hybrid PSSM covariance heterogeneous streams', () => {
    const draft = useExperimentDraftStore()
    draft.cloneFromExperiment(
      experiment(
        {
          tokenizer: 'kmer',
          encoder: 'cc_pssm_blosum62_hybrid',
          model: 'mlp',
          classifier: 'mlp',
          pair_strategy: 'two_stream',
          interaction_level: 'vector',
          predictor_type: 'classifier',
          profile_on_missing: 'blosum62_hybrid',
          streams: {
            peptide: {
              feature_route: 'classic',
              tokenizer: 'kmer',
              encoder: 'cc_pssm_blosum62_hybrid',
              stream_model: '',
              model_source: '',
              max_length: 60,
              params: { k: 1, cc_pssm_blosum62_hybrid_lag: 5 },
            },
            partner: {
              feature_route: 'classic',
              tokenizer: 'kmer',
              encoder: 'acc_pssm_blosum62_hybrid',
              stream_model: '',
              model_source: '',
              max_length: 512,
              params: { k: 1, acc_pssm_blosum62_hybrid_lag: 7 },
            },
          },
        },
        { task_type: 'pep_protein_interaction' },
      ),
    )

    const config = draft.toConfig({ isPairTask: true, encoderFamily: 'global' })
    expect(draft.streamsLinked).toBe(false)
    expect(draft.ccPssmBlosum62HybridLag).toBe(5)
    expect(draft.partnerAccPssmBlosum62HybridLag).toBe(7)
    expect(config).toMatchObject({
      profile_on_missing: 'blosum62_hybrid',
      streams: {
        peptide: {
          encoder: 'cc_pssm_blosum62_hybrid',
          params: { k: 1, cc_pssm_blosum62_hybrid_lag: 5 },
        },
        partner: {
          encoder: 'acc_pssm_blosum62_hybrid',
          params: { k: 1, acc_pssm_blosum62_hybrid_lag: 7 },
        },
      },
    })
  })

  it('keeps training disabled for failed or pending preflight', () => {
    expect(preflightAllowsTraining('dataset-1', { valid: false }, false)).toBe(false)
    expect(preflightAllowsTraining('dataset-1', { valid: true }, true)).toBe(false)
    expect(preflightAllowsTraining('', { valid: true }, false)).toBe(false)
    expect(preflightAllowsTraining('dataset-1', { valid: true }, false, false)).toBe(false)
    expect(preflightAllowsTraining('dataset-1', { valid: true }, false)).toBe(true)
  })

  it('serializes the LSA memory guard for single-sequence configs', () => {
    const draft = useExperimentDraftStore()
    draft.taskType = 'pep_function'
    draft.mode = 'custom'
    draft.encoder = 'tfidf'
    draft.reduction = 'lsa'
    draft.lsaDim = 32
    draft.lsaDenseMemoryLimitGib = 1.5

    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'global' })).toMatchObject({
      reduction: 'lsa',
      lsa_dim: 32,
      lsa_dense_memory_limit_gib: 1.5,
    })
  })

  it('serializes TSVD memory controls only when generic TSVD is selected', () => {
    const draft = useExperimentDraftStore()
    draft.taskType = 'pep_function'
    draft.mode = 'custom'
    draft.featureRoute = 'classic'
    draft.encoder = 'bow'
    draft.classifier = 'rf'
    draft.model = 'rf'
    draft.mlReduction = 'tsvd'
    draft.mlReductionDim = 32
    draft.tsvdDenseMemoryLimitGib = 1.5

    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'global' })).toMatchObject({
      ml_reduction: 'tsvd',
      ml_reduction_dim: 32,
      tsvd_dense_memory_limit_gib: 1.5,
    })

    draft.mlReduction = 'pca'
    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'global' })).not.toHaveProperty(
      'tsvd_dense_memory_limit_gib',
    )
  })

  it('serializes only the controls used by the selected PCA or KPCA reducer', () => {
    const draft = useExperimentDraftStore()
    draft.taskType = 'pep_function'
    draft.mode = 'custom'
    draft.featureRoute = 'classic'
    draft.encoder = 'bow'
    draft.classifier = 'rf'
    draft.model = 'rf'
    draft.mlReductionDim = 16
    draft.pcaDenseMemoryLimitGib = 1.5
    draft.kpcaMemoryLimitGib = 2.5
    draft.kpcaKernel = 'rbf'
    draft.kpcaGamma = 0.2

    draft.mlReduction = 'pca'
    const pca = draft.toConfig({ isPairTask: false, encoderFamily: 'global' })
    expect(pca).toMatchObject({
      ml_reduction: 'pca',
      ml_reduction_dim: 16,
      pca_dense_memory_limit_gib: 1.5,
    })
    expect(pca).not.toHaveProperty('kpca_memory_limit_gib')
    expect(pca).not.toHaveProperty('kpca_kernel')
    expect(pca).not.toHaveProperty('kpca_gamma')

    draft.mlReduction = 'kpca'
    const rbf = draft.toConfig({ isPairTask: false, encoderFamily: 'global' })
    expect(rbf).toMatchObject({
      ml_reduction: 'kpca',
      kpca_memory_limit_gib: 2.5,
      kpca_kernel: 'rbf',
      kpca_gamma: 0.2,
    })
    expect(rbf).not.toHaveProperty('pca_dense_memory_limit_gib')

    draft.kpcaKernel = 'linear'
    const linear = draft.toConfig({ isPairTask: false, encoderFamily: 'global' })
    expect(linear).not.toHaveProperty('kpca_gamma')
  })

  it('never generates a new Pair LSA config from stale draft state', () => {
    const draft = useExperimentDraftStore()
    draft.taskType = 'pep_protein_interaction'
    draft.encoder = 'tfidf'
    draft.reduction = 'lsa'

    const config = draft.toConfig({ isPairTask: true, encoderFamily: 'global' })
    expect(config.reduction).toBe('none')
    expect(config).not.toHaveProperty('lsa_dim')
    expect(config).not.toHaveProperty('lsa_dense_memory_limit_gib')
  })

  it('serializes aligned_gated as the canonical learned Pair fusion', () => {
    const draft = useExperimentDraftStore()
    draft.resetForTask('pep_protein_interaction')
    draft.mode = 'custom'
    draft.pairStrategy = 'two_stream'
    draft.interactionLevel = 'vector'
    draft.predictorType = 'classifier'
    draft.classifier = 'mlp'
    draft.fusionMode = 'aligned_gated'
    draft.streamAlignmentDim = 192

    expect(draft.toConfig({ isPairTask: true, encoderFamily: 'global' })).toMatchObject({
      predictor_type: 'classifier',
      classifier: 'mlp',
      fusion_mode: 'aligned_gated',
      stream_alignment_dim: 192,
    })
  })

  it('serializes the LDA contract controls for single-sequence configs', () => {
    const draft = useExperimentDraftStore()
    draft.taskType = 'pep_function'
    draft.mode = 'custom'
    draft.encoder = 'bow'
    draft.reduction = 'lda'
    draft.ldaTopics = 64
    draft.ldaDenseMemoryLimitGib = 1.5

    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'global' })).toMatchObject({
      reduction: 'lda',
      lda_topics: 64,
      lda_dense_memory_limit_gib: 1.5,
    })
  })

  it('preserves a stale Pair LDA selection so preflight can explain the incompatibility', () => {
    const draft = useExperimentDraftStore()
    draft.taskType = 'pep_protein_interaction'
    draft.encoder = 'bow'
    draft.reduction = 'lda'

    const config = draft.toConfig({ isPairTask: true, encoderFamily: 'global' })
    expect(config.reduction).toBe('lda')
    expect(config).toMatchObject({ classifier: 'mlp', fusion_mode: 'concat' })
    expect(config).not.toHaveProperty('lda_topics')
    expect(config).not.toHaveProperty('lda_dense_memory_limit_gib')
  })

  it('serializes the PLSA contract controls for single-sequence configs', () => {
    const draft = useExperimentDraftStore()
    draft.taskType = 'pep_function'
    draft.mode = 'custom'
    draft.encoder = 'bow'
    draft.reduction = 'plsa'
    draft.plsaTopics = 32
    draft.plsaMemoryLimitGib = 1.5

    expect(draft.toConfig({ isPairTask: false, encoderFamily: 'global' })).toMatchObject({
      reduction: 'plsa',
      plsa_topics: 32,
      plsa_memory_limit_gib: 1.5,
      plsa_contract_version: 1,
    })
  })

  it('preserves a stale Pair PLSA selection so preflight can explain the incompatibility', () => {
    const draft = useExperimentDraftStore()
    draft.taskType = 'pep_protein_interaction'
    draft.encoder = 'bow'
    draft.reduction = 'plsa'

    const config = draft.toConfig({ isPairTask: true, encoderFamily: 'global' })
    expect(config.reduction).toBe('plsa')
    expect(config).toMatchObject({ classifier: 'mlp', fusion_mode: 'concat' })
    expect(config).not.toHaveProperty('plsa_topics')
    expect(config).not.toHaveProperty('plsa_memory_limit_gib')
    expect(config).not.toHaveProperty('plsa_contract_version')
  })

  it('applies a valid Pair topic preset in Recommended mode', () => {
    const draft = useExperimentDraftStore()
    draft.resetForTask('pep_protein_interaction')
    draft.encoder = 'bow'

    expect(draft.applyPairTopicPreset('plsa')).toEqual([
      'featureRoute',
      'tokenizer',
      'fusionMode',
      'classifier',
    ])
    expect(
      pairTopicRouteIssues({
        featureRoute: draft.featureRoute,
        tokenizer: draft.tokenizer,
        encoder: draft.encoder,
        streamsLinked: draft.streamsLinked,
        partnerFeatureRoute: draft.partnerFeatureRoute,
        partnerTokenizer: draft.partnerTokenizer,
        partnerEncoder: draft.partnerEncoder,
        pairStrategy: draft.pairStrategy,
        interactionLevel: draft.interactionLevel,
        predictorType: draft.predictorType,
        fusionMode: draft.fusionMode,
        classifier: draft.classifier,
      }),
    ).toEqual([])
    expect(draft.toConfig({ isPairTask: true, encoderFamily: 'global' })).toMatchObject({
      reduction: 'plsa',
      pair_strategy: 'two_stream',
      interaction_level: 'vector',
      predictor_type: 'classifier',
      fusion_mode: 'concat',
      classifier: 'svm',
      model: 'svm',
      streams: {
        peptide: { encoder: 'bow' },
        partner: { encoder: 'bow' },
      },
    })
  })

  it('keeps a compatible heterogeneous Pair BoW route when applying LDA', () => {
    const draft = useExperimentDraftStore()
    draft.resetForTask('pep_protein_interaction')
    draft.mode = 'custom'
    draft.featureRoute = 'classic'
    draft.tokenizer = 'kmer'
    draft.encoder = 'bow'
    draft.streamsLinked = false
    draft.partnerFeatureRoute = 'classic'
    draft.partnerTokenizer = 'dt'
    draft.partnerEncoder = 'bow'
    draft.classifier = 'rf'

    const adjusted = draft.applyPairTopicPreset('lda')

    expect(adjusted).toEqual(['fusionMode'])
    expect(draft.streamsLinked).toBe(false)
    expect(draft.classifier).toBe('rf')
    expect(draft.toConfig({ isPairTask: true, encoderFamily: 'global' })).toMatchObject({
      reduction: 'lda',
      fusion_mode: 'concat',
      classifier: 'rf',
      streams: {
        peptide: { tokenizer: 'kmer', encoder: 'bow' },
        partner: { tokenizer: 'dt', encoder: 'bow' },
      },
    })
  })

  it('links an incompatible partner stream without erasing its latent fields', () => {
    const draft = useExperimentDraftStore()
    draft.resetForTask('pep_protein_interaction')
    draft.mode = 'custom'
    draft.featureRoute = 'classic'
    draft.tokenizer = 'kmer'
    draft.encoder = 'bow'
    draft.streamsLinked = false
    draft.partnerFeatureRoute = 'pretrained'
    draft.partnerTokenizer = 'prott5'
    draft.partnerEncoder = 'prott5'

    const adjusted = draft.applyPairTopicPreset('lda')

    expect(adjusted).toEqual(['streamsLinked', 'fusionMode', 'classifier'])
    expect(draft.streamsLinked).toBe(true)
    expect(draft.partnerFeatureRoute).toBe('pretrained')
    expect(draft.partnerTokenizer).toBe('prott5')
    expect(draft.partnerEncoder).toBe('prott5')
  })

  it('reads dataset minimum lengths for covariance hints', () => {
    const metadata = {
      length_stats: {
        version: 1,
        by_split: {},
        overall: {
          sequence: { min: 8 },
          peptide: { min: 9 },
          partner: { min: 10 },
        },
      },
    }

    expect(datasetLengthRoleMin(metadata, 'sequence')).toBe(8)
    expect(datasetLengthRoleMin(metadata, 'peptide')).toBe(9)
    expect(datasetLengthRoleMin(metadata, 'partner')).toBe(10)
    expect(datasetLengthRoleMin({}, 'sequence')).toBe(0)
  })
})
