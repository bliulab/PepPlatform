import { describe, expect, it } from 'vitest'

import { buildModelDefaultName, modelRouteSummary } from './modelDisplay'

const esm2Stream = {
  feature_route: 'pretrained',
  tokenizer: 'esm2',
  encoder: 'ppi_fusion',
}

describe('generated model names with run identifiers', () => {
  it('names pooled pretrained and classical routes in pipeline order', () => {
    expect(
      buildModelDefaultName({
        tokenizer: 'esm2',
        encoder: 'index',
        model: 'mlp',
        classifier: 'mlp',
      }),
    ).toBe('ESM2-Embedding-MLP')

    expect(
      buildModelDefaultName({
        tokenizer: 'kmer',
        encoder: 'bow',
        model: 'rf',
        classifier: 'rf',
      }),
    ).toBe('Kmer-BOW-RF')
  })

  it('names the zero-hidden-layer compatibility route as Linear', () => {
    const config = {
      tokenizer: 'esm2',
      encoder: 'esm2',
      model: 'mlp',
      classifier: 'mlp',
      head_hidden_dims: [],
    }
    expect(buildModelDefaultName(config)).toBe('ESM2-Embedding-Linear')
    expect(modelRouteSummary(config)).toBe('ESM2 embeddings · Linear')
    expect(
      buildModelDefaultName({
        ...config,
        head_hidden_dims: undefined,
        prediction_head: { kind: 'linear', hidden_dims: [] },
      }),
    ).toBe('ESM2-Embedding-Linear')
  })

  it('keeps sequence encoders and predictors as separate method components', () => {
    expect(
      buildModelDefaultName({
        tokenizer: 'kmer',
        encoder: 'index',
        model: 'cnn',
        sequence_model: 'cnn',
        training_mode: 'end_to_end',
        classifier: 'mlp',
      }),
    ).toBe('Kmer-Index-CNN')
  })

  it('marks k-mer BLOSUM62 averaging as a derived method', () => {
    expect(
      buildModelDefaultName({
        tokenizer: 'kmer',
        encoder: 'blosum62',
        k: 1,
        model: 'cnn',
        sequence_model: 'cnn',
      }),
    ).toBe('Kmer-BLOSUM62-CNN')

    expect(
      buildModelDefaultName({
        tokenizer: 'kmer',
        encoder: 'blosum62',
        k: 2,
        model: 'cnn',
        sequence_model: 'cnn',
      }),
    ).toBe('Kmer-BLOSUM62DerivedMean-CNN')
  })

  it('names ESM2 residue-chain graph routes from token embeddings', () => {
    const config = {
      tokenizer: 'esm2',
      encoder: 'index',
      model: 'gat',
      sequence_model: 'gat',
      training_mode: 'end_to_end',
      classifier: 'mlp',
    }
    expect(buildModelDefaultName(config)).toBe('ESM2-TokenEmbedding-GAT')
    expect(modelRouteSummary(config)).toBe('ESM2 residue embeddings · GAT')
  })

  it('includes explicit feature transformations', () => {
    expect(
      buildModelDefaultName({
        tokenizer: 'kmer',
        encoder: 'tfidf',
        reduction: 'lsa',
        model: 'xgb',
        classifier: 'xgb',
      }),
    ).toBe('Kmer-TF-IDF-LSA-XGB')
  })

  it('marks historical pair LSA method names as legacy', () => {
    expect(
      buildModelDefaultName({
        task_type: 'pep_protein_interaction',
        tokenizer: 'kmer',
        encoder: 'tfidf',
        reduction: 'lsa',
        model: 'rf',
        classifier: 'rf',
      }),
    ).toBe('Kmer-TF-IDF-LegacyLSA-RF')
  })

  it('marks historical pair LDA method names as legacy', () => {
    expect(
      buildModelDefaultName({
        task_type: 'pep_protein_interaction',
        tokenizer: 'kmer',
        encoder: 'bow',
        reduction: 'lda',
        model: 'rf',
        classifier: 'rf',
      }),
    ).toBe('Kmer-BOW-LegacyLDA-RF')
  })

  it.each([
    ['lda', 'LDA'],
    ['plsa', 'PLSA'],
  ])('recognizes submitted strict Pair %s routes as current', (reduction, display) => {
    const stream = {
      feature_route: 'classic',
      tokenizer: 'kmer',
      encoder: 'bow',
    }
    expect(
      buildModelDefaultName({
        task_type: 'pep_protein_interaction',
        tokenizer: 'kmer',
        encoder: 'bow',
        reduction,
        model: 'rf',
        classifier: 'rf',
        pair_strategy: 'two_stream',
        interaction_level: 'vector',
        predictor_type: 'classifier',
        fusion_mode: 'concat',
        streams: { peptide: stream, partner: stream },
      }),
    ).toBe(`Kmer-BOW-${display}-RF`)
  })

  it('distinguishes current PLSA from historical KL-NMF configs', () => {
    expect(
      buildModelDefaultName({
        tokenizer: 'kmer',
        encoder: 'bow',
        reduction: 'plsa',
        plsa_contract_version: 1,
        model: 'rf',
        classifier: 'rf',
      }),
    ).toBe('Kmer-BOW-PLSA-RF')
    expect(
      buildModelDefaultName({
        tokenizer: 'kmer',
        encoder: 'bow',
        reduction: 'plsa',
        model: 'rf',
        classifier: 'rf',
      }),
    ).toBe('Kmer-BOW-LegacyKLNMF-RF')
  })

  it('names exported learned representations before their downstream classifier', () => {
    expect(
      buildModelDefaultName({
        tokenizer: 'kmer',
        encoder: 'index',
        model: 'mlp',
        sequence_model: 'gru',
        training_mode: 'export_embeddings',
        classifier: 'xgb',
      }),
    ).toBe('Kmer-GRUEmb-XGB')
  })

  it('collapses homogeneous pair streams and preserves heterogeneous roles', () => {
    expect(
      buildModelDefaultName({
        streams: { peptide: esm2Stream, partner: { ...esm2Stream } },
        interaction_level: 'vector',
        predictor_type: 'classifier',
        model: 'mlp',
        classifier: 'mlp',
      }),
    ).toBe('ESM2-Embedding-MLP')

    expect(
      buildModelDefaultName({
        streams: {
          peptide: esm2Stream,
          partner: { feature_route: 'classic', tokenizer: 'kmer', encoder: 'bow' },
        },
        interaction_level: 'vector',
        predictor_type: 'classifier',
        model: 'mlp',
        classifier: 'mlp',
      }),
    ).toBe('Pep[ESM2-Embedding]+Partner[Kmer-BOW]-MLP')
  })

  it('uses interaction and graph models as predictors', () => {
    expect(
      buildModelDefaultName({
        streams: { peptide: esm2Stream, partner: { ...esm2Stream } },
        interaction_level: 'token',
        interaction_model: 'cross_attn',
        model: 'cross_attn',
        classifier: 'mlp',
      }),
    ).toBe('ESM2-Token-CrossAttn')

    expect(
      buildModelDefaultName({
        tokenizer: 'prott5',
        encoder: 'ppi_fusion',
        predictor_type: 'sample_graph',
        graph_predictor: 'gat',
        model: 'gat',
      }),
    ).toBe('ProtT5-Embedding-GAT')
  })

  it('normalizes legacy tokenizer aliases and paper abbreviations', () => {
    expect(
      buildModelDefaultName({
        tokenizer: 'probert',
        encoder: 'index',
        model: 'calibrated_svm',
        classifier: 'calibrated_svm',
      }),
    ).toBe('ProtBERT-Embedding-CalibratedSVM')

    expect(
      buildModelDefaultName({
        tokenizer: 'prott5',
        encoder: 'prott5',
        model: 'svm',
        classifier: 'svm',
        svm_calibration: 'sigmoid',
      }),
    ).toBe('ProtT5-Embedding-CalibratedSVM')

    expect(
      buildModelDefaultName({
        tokenizer: 'esmc',
        encoder: 'index',
        model: 'lgb',
        classifier: 'lgb',
      }),
    ).toBe('ESM-C-600M-Embedding-LightGBM')

    expect(
      buildModelDefaultName({
        tokenizer: 'esmc',
        esmc_model: 'Synthyra/ESMplusplus_small',
        encoder: 'index',
        model: 'lgb',
        classifier: 'lgb',
      }),
    ).toBe('ESM++-300M-Embedding-LightGBM')
    expect(
      buildModelDefaultName({
        tokenizer: 'esmc',
        esmc_model: 'EvolutionaryScale/esmc-300m-2024-12',
        encoder: 'index',
        model: 'lgb',
        classifier: 'lgb',
      }),
    ).toBe('ESM-C-300M-Embedding-LightGBM')
    expect(
      modelRouteSummary({
        tokenizer: 'esmc',
        esmc_model: 'EvolutionaryScale/esmc-600m-2024-12',
        encoder: 'index',
        model: 'mlp',
        classifier: 'mlp',
      }),
    ).toBe('ESM-C 600M embeddings · MLP')
  })

  it('uses a compact fallback for future method components', () => {
    expect(
      buildModelDefaultName({
        tokenizer: 'custom-token',
        encoder: 'custom_encoder',
        model: 'custom_model',
      }),
    ).toBe('CustomToken-CustomEncoder-CustomModel')
  })

  it('does not change the descriptive route summary', () => {
    expect(
      modelRouteSummary({
        tokenizer: 'esm2',
        encoder: 'index',
        model: 'mlp',
        classifier: 'mlp',
      }),
    ).toBe('ESM2 embeddings · MLP')
  })
})
