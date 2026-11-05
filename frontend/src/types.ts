export interface DatasetRecord {
  id: string
  name: string
  task_type: string
  source: string
  format: string
  root_path: string
  metadata: Record<string, unknown>
  owner_user_id?: string | null
  created_at?: string
}

export interface DatasetReportIssue {
  severity: 'error' | 'warning' | 'info' | string
  code: string
  message: string
  split?: string
  file?: string
  params?: Record<string, string | number>
}

export type GeneratedNameQualifier = 'conservative' | 'balanced' | 'exploratory'

export type GeneratedExperimentName =
  | {
      kind: 'config'
      qualifier?: GeneratedNameQualifier
    }
  | {
      kind: 'rerun'
      base_name: string
      base_kind?: 'config' | 'literal'
      qualifier?: GeneratedNameQualifier
      count: number
      source_experiment_id?: string | null
    }

export interface DatasetPreviewRow {
  sample_id?: string
  sequence?: string
  sequence_length?: number
  labels?: string[]
  peptide?: string
  partner?: string
  peptide_length?: number
  partner_length?: number
  label?: string
  invalid_residues?: string[]
}

export interface DatasetLengthStats {
  count: number
  min?: number | null
  max?: number | null
  mean?: number | null
}

export interface DatasetSplitReport {
  name: string
  sample_count: number
  files: Record<string, boolean>
  columns: {
    labels?: string[]
    pairs?: string[]
  }
  label_mode: 'single_label' | 'multi_label' | 'unknown' | string
  label_counts: {
    positive: Record<string, number>
    values: Record<string, number>
  }
  sequence_count?: number | null
  sequence_lengths: DatasetLengthStats
  peptide_lengths: DatasetLengthStats
  partner_lengths: DatasetLengthStats
  preview: DatasetPreviewRow[]
}

export interface DatasetReport {
  dataset_id: string
  generated_at: string
  valid: boolean
  status: 'valid' | 'warning' | 'error' | string
  task_type: string
  format: string
  summary: {
    total_samples: number
    splits: Record<string, number>
    has_test_split: boolean
    label_mode: 'single_label' | 'multi_label' | 'unknown' | string
    label_columns: string[]
  }
  splits: DatasetSplitReport[]
  issues: DatasetReportIssue[]
}

export interface MetricEvent {
  kind: 'epoch' | 'best' | 'validation' | 'test' | 'stage'
  [key: string]: string | number | boolean | undefined
}

export type EncoderOutputKind =
  | 'token_ids'
  | 'position_features'
  | 'global_features'
  | 'sequence_embedding'
  | 'residue_embeddings'

export interface RouteComponent {
  id: string
  label: string
  output_kind?: EncoderOutputKind
  transformation?: string
  frozen?: boolean
  learnable?: boolean
  normalization?: string
  output_dimension?: number
  fusion_mode?: string
  join_mode?: string
  hidden_dims?: number[]
  activation?: string
  dropout?: number
}

export interface EffectiveRouteStream {
  role: 'sequence' | 'peptide' | 'partner' | string
  feature_route: 'classic' | 'pretrained' | string
  tokenizer: RouteComponent
  encoder: RouteComponent
}

export interface EffectiveRoute {
  schema_version: number
  route_id: string
  task_type: string
  streams: EffectiveRouteStream[]
  predictor: {
    input_adapter: RouteComponent
    architecture: RouteComponent
    readout: RouteComponent
    head: RouteComponent
  }
  training_mode: string
}

export interface TrainingFailure {
  schema_version: number
  code: string
  category:
    | 'profile'
    | 'resource'
    | 'dependency'
    | 'data'
    | 'configuration'
    | 'runtime'
    | 'unknown'
    | string
  stage:
    | 'setup'
    | 'profile_database'
    | 'profile_generation'
    | 'feature_encoding'
    | 'training'
    | 'evaluation'
    | 'artifact_export'
    | 'unknown'
    | string
  message: string
  context: Record<string, unknown>
  exit_code?: number | null
}

export interface ExperimentRecord {
  id: string
  dataset_id: string
  owner_user_id?: string | null
  name: string
  generated_name?: GeneratedExperimentName | null
  task_type: string
  config: Record<string, unknown>
  requested_config?: Record<string, unknown>
  config_parameter_sources?: Record<string, 'explicit' | 'derived' | 'default' | string>
  effective_route?: EffectiveRoute
  status: string
  log_path: string
  artifact_dir: string
  metrics: {
    events: MetricEvent[]
    summary: Record<string, unknown>
  }
  created_at?: string
  updated_at?: string
  started_at?: string | null
  finished_at?: string | null
  model_id?: string | null
  error?: string
  failure?: TrainingFailure | null
  resume?: {
    source_experiment_id: string
    stage: 'downstream_svm' | string
    bundle_schema_version: number
    bundle_digest: string
  } | null
  resume_capabilities?: {
    downstream_svm?: {
      available: boolean
      reason?: string
    }
  }
}

export interface AnalysisCurvePoint {
  fpr?: number
  tpr?: number
  recall?: number
  precision?: number
}

export interface MetricDefinition {
  metric_namespace_version?: number
  aupr_method?: string
}

export interface AnalysisCurve extends MetricDefinition {
  available: boolean
  reason?: string
  auc?: number
  aupr?: number
  points: AnalysisCurvePoint[]
}

export interface ConfusionMatrixPayload {
  available: boolean
  reason?: string
  threshold?: number
  threshold_source?: string
  threshold_mode?: 'scalar' | 'per_label'
  threshold_count?: number
  average?: string
  tp?: number
  fp?: number
  fn?: number
  tn?: number
  total?: number
  accuracy?: number
  balanced_accuracy?: number
  precision?: number
  recall?: number
  specificity?: number
  f1?: number
  mcc?: number
}

export interface ThresholdSweepPoint {
  threshold: number
  accuracy?: number
  precision?: number
  recall?: number
  specificity?: number
  f1?: number
  mcc?: number
}

export interface ThresholdSweepPayload {
  available: boolean
  reason?: string
  selected_threshold?: number
  threshold_source?: string
  points: ThresholdSweepPoint[]
}

export interface ScoreDistributionBin {
  lower: number
  upper: number
  positive_count: number
  negative_count: number
}

export interface ScoreDistributionSummary {
  count: number
  min?: number
  max?: number
  mean?: number
  median?: number
}

export interface ScoreDistributionPayload {
  available: boolean
  reason?: string
  bin_count?: number
  score_min?: number
  score_max?: number
  positive?: ScoreDistributionSummary
  negative?: ScoreDistributionSummary
  bins: ScoreDistributionBin[]
}

export interface AnalysisSplitCurves {
  average?: string
  positive_count?: number
  sample_count?: number
  label_count?: number
  decision_count?: number
  roc: AnalysisCurve
  pr: AnalysisCurve
  confusion_matrix?: ConfusionMatrixPayload
  threshold_sweep?: ThresholdSweepPayload
  score_distribution?: ScoreDistributionPayload
}

export interface AnalysisOperatingPoint {
  values?: number[]
  threshold?: number
  thresholds?: number[]
  threshold_by_label?: Record<string, number>
  label_names?: string[]
  threshold_source?: string
  source?: string
  objective?: string
}

export interface AnalysisMetricSummaries {
  operating_point?: AnalysisOperatingPoint
  validation?: Record<string, unknown>
  test?: Record<string, unknown>
}

export interface ProjectionPoint {
  x: number
  y: number
  z?: number
  split: string
  label: string
  positive_count?: number
  cluster?: number
}

export interface FeatureImportanceItem {
  name: string
  importance: number
  rank: number
}

export interface FeatureImportancePayload {
  available: boolean
  reason?: string
  source?: string
  method?: string
  method_label?: string
  metric?: string
  baseline_score?: number
  sample_count?: number
  candidate_count?: number
  feature_count?: number
  items: FeatureImportanceItem[]
}

export interface ExperimentAnalysis {
  schema_version: number
  curves: Record<string, AnalysisSplitCurves>
  metrics?: AnalysisMetricSummaries
  projection_2d: {
    available: boolean
    reason?: string
    method?: string
    method_details?: Record<string, unknown>
    points: ProjectionPoint[]
  }
  projection_3d: {
    available: boolean
    reason?: string
    method?: string
    method_details?: Record<string, unknown>
    points: ProjectionPoint[]
  }
  clusters: {
    available: boolean
    reason?: string
    method?: string
    n_clusters?: number
    source?: string
    dimension?: number
    sample_count?: number
  }
  feature_importance: FeatureImportancePayload
  split_importance?: Record<string, FeatureImportancePayload>
  shap_importance?: Record<string, FeatureImportancePayload>
  warnings: string[]
}

export interface ModelRecord {
  id: string
  experiment_id: string
  owner_user_id?: string | null
  name: string
  name_is_custom?: boolean
  task_type: string
  bundle_path: string
  supports_inference: boolean
  metadata: Record<string, unknown>
  created_at?: string
}

export interface InferenceReadinessCheck {
  id: string
  ok: boolean
  status: 'ok' | 'warning' | 'error' | 'skipped' | string
  code?: string
  message: string
  hint?: string
  version?: string
  required?: boolean
  mode?: string
  feature?: string
  value?: number
}

export interface InferenceReadinessIssue {
  code: string
  message: string
  hint?: string
}

export interface InferenceReadiness {
  model_id: string
  ready: boolean
  profile_required: boolean
  status: 'ready' | 'warning' | 'blocked' | 'not_required' | string
  mode: 'standard' | 'prebuilt_db' | 'build_on_demand' | 'unavailable' | string
  feature?: string
  fallback_mode?: string
  checks: InferenceReadinessCheck[]
  errors: InferenceReadinessIssue[]
  warnings: InferenceReadinessIssue[]
  environment_checked_at?: string
  runtime?: {
    cache_status: 'cold' | 'loading' | 'warm' | 'suspended' | string
    suspended_reason?: string
    idle_ttl_seconds?: number
  }
}

export type InferenceStage =
  | 'validating_bundle'
  | 'waiting_for_runtime'
  | 'loading_encoder'
  | 'encoder_ready'
  | 'extracting_features'
  | 'running_predictor'
  | 'saving_result'
  | string

export interface InferenceProgressEvent {
  stage: InferenceStage
  completed?: number
  total?: number
  batch_completed?: number
  batch_total?: number
  elapsed_ms?: number
  cache_hit?: boolean
  [key: string]: unknown
}

export type InferenceStreamEvent =
  | { type: 'stage'; data: InferenceProgressEvent }
  | { type: 'done'; data: PredictionResponse }
  | { type: 'error'; data: { code: string; stage?: string; message: string; hint?: string } }
  | { type: 'heartbeat'; data: null }

export interface ProfileInferenceDiagnostics {
  status: 'ready' | 'degraded' | string
  feature: string
  input_count: number
  cache_hits: number
  generated: number
  fallbacks: number
  fallback_mode: string
  warning_codes: string[]
}

export interface PredictionRecord {
  sequence?: string
  peptide?: string
  protein?: string
  score?: number
  score_kind?: 'probability' | 'margin' | string
  score_range?: number[]
  threshold?: number
  is_interaction?: boolean
  prediction_label?: string
  predicted_labels?: string[]
  scores: number[]
}

export interface PredictionResponse {
  score_kind?: string
  ensemble?: {
    method: string
    batch_dependent: boolean
    batch_digest: string
    input_count: number
    numpy_version: string
    contract_version: number
  }
  model_id: string
  run_id?: string
  labels: string[]
  thresholds?: number[]
  evaluation_protocol_version?: number
  legacy_evaluation_protocol?: boolean
  predictions: PredictionRecord[]
  diagnostics?: {
    profile?: ProfileInferenceDiagnostics
    inference?: {
      device: string
      encoder_cache_hit?: boolean | null
      encoder_load_ms?: number
      total_ms?: number
    }
  }
}

export interface PredictionPair {
  peptide: string
  protein: string
}

export interface InferenceRunRecord {
  id: string
  model_id: string
  owner_user_id?: string | null
  input_kind: 'sequences' | 'pairs' | string
  input_count: number
  prediction_count: number
  request: {
    sequences?: string[]
    pairs?: PredictionPair[]
  }
  response: PredictionResponse
  created_at?: string
}

export interface DeleteInferenceRunResponse {
  run_id: string
  deleted: boolean
}

export interface AiProviderConfig {
  id: string
  owner_user_id?: string | null
  provider: 'deepseek' | string
  display_name: string
  base_url: string
  model: string
  api_key_last4: string
  has_api_key: boolean
  capabilities: AiProviderCapabilities
  enabled: boolean
  created_at?: string
  updated_at?: string
}

export interface AiProviderCapabilities {
  supports_json_object?: boolean
  supports_reasoning_toggle?: boolean
  reasoning_control?: 'none' | 'thinking_disabled' | string
  default_max_tokens?: number
}

export interface AiProviderPreset {
  provider: string
  display_name: string
  base_url: string
  model: string
  description?: string
  capabilities: AiProviderCapabilities
}

export interface AiProviderPayload {
  provider?: string
  display_name: string
  base_url: string
  model: string
  api_key?: string
  capabilities: AiProviderCapabilities
  enabled: boolean
}

export interface AiProviderTestResponse {
  ok: boolean
  provider: string
  model: string
  answer_preview?: string
  usage: Record<string, unknown>
}

export interface AiContextManifestSection {
  id: string
  sent: boolean
  detail: string
}

export interface AiContextManifest {
  sections: AiContextManifestSection[]
  include: {
    dataset_preview?: boolean
    inference_summary?: boolean
    extended_logs?: boolean
    [key: string]: boolean | undefined
  }
  redactions: string[]
  context_chars?: number
}

export interface ExperimentAiAnalysisResponse {
  answer: string
  provider: string
  model: string
  usage: Record<string, unknown>
  context_manifest: AiContextManifest
  warnings: string[]
}

export type AgentStrategyPreference = 'conservative' | 'balanced' | 'aggressive'

export interface AgentExperimentDesignPayload {
  provider_config_id: string
  task_type: string
  dataset_id: string
  objective?: string
  baseline_experiment_ids?: string[]
  language?: string
  include?: {
    dataset_preview?: boolean
  }
  max_tokens?: number
  strategy_preference?: AgentStrategyPreference
}

export interface AgentExperimentCandidate {
  id: string
  lane: 'conservative' | 'balanced' | 'exploratory' | string
  title: string
  rationale: string
  expected_tradeoffs: string[]
  objective_alignment: string
  baseline_changes: string[]
  risk_level: 'low' | 'medium' | 'high' | string
  verification_steps: string[]
  config: Record<string, unknown>
  preflight: PreflightReport
  valid: boolean
}

export interface AgentContextSummary {
  recent_experiment_count: number
  baseline_experiment_count: number
  leaderboard_count: number
  previous_round_count?: number
  review_count?: number
  recommendation_count?: number
  avoided_config_count?: number
}

export interface AgentExperimentDesignResponse {
  candidates: AgentExperimentCandidate[]
  provider: string
  model: string
  usage: Record<string, unknown>
  context_manifest: AiContextManifest
  context_summary: AgentContextSummary
  repair_attempted: boolean
  repair_succeeded: boolean
  strategy_preference: AgentStrategyPreference
  attempts: number
  warnings: string[]
}

export interface AgentExperimentBatchCandidate {
  id?: string
  lane: string
  title: string
  config: Record<string, unknown>
}

export interface AgentExperimentBatchPayload {
  task_type?: string
  dataset_id: string
  objective?: string
  candidates: AgentExperimentBatchCandidate[]
}

export interface AgentExperimentBatchCreatedExperiment {
  candidate_id: string
  lane: string
  title: string
  experiment: ExperimentRecord
}

export interface AgentExperimentBatchResponse {
  batch_id: string
  created_experiments: AgentExperimentBatchCreatedExperiment[]
  warnings: string[]
}

export interface AgentChatPayload {
  provider_config_id: string
  messages: { role: 'user' | 'assistant'; content: string }[]
  language?: string
  max_tokens?: number
  temperature?: number
  web_search?: boolean
}

export interface AgentWebSearchSource {
  index: number
  title: string
  url: string
  domain: string
  snippet: string
  favicon?: string
}

export interface AgentWebSearch {
  requested: boolean
  status: 'disabled' | 'complete' | 'no_results' | 'unavailable' | string
  query: string
  provider: string
  credits: number
  reason: string
  sources: AgentWebSearchSource[]
  remaining?: number | null
  request_id?: string
}

export interface AgentWebSearchStatus {
  configured: boolean
  available: boolean
  reason: string
  remaining?: number | null
  limit?: number | null
  resets_on: string
}

export interface AgentChatResponse {
  reply: string
  finish_reason?: string
  provider?: string
  model?: string
  usage?: {
    prompt_tokens?: number
    completion_tokens?: number
    total_tokens?: number
  }
  web_search?: AgentWebSearch
}

export type AgentChatMessageStatus = 'streaming' | 'complete' | 'stopped' | 'error'

export interface AgentChatConversation {
  id: string
  owner_user_id?: string | null
  scope?: 'agent' | 'experiment_analysis' | string
  experiment_id?: string | null
  title: string
  pinned: boolean
  pinned_at?: string | null
  last_provider_config_id?: string | null
  last_provider: string
  last_model: string
  last_message_preview: string
  created_at: string
  updated_at: string
}

export interface AgentChatMessage {
  id: string
  conversation_id: string
  ordinal: number
  role: 'user' | 'assistant'
  content: string
  status: AgentChatMessageStatus
  kind?: 'chat' | 'report' | string
  provider_config_id?: string | null
  provider: string
  model: string
  finish_reason: string
  error: string
  usage: Record<string, unknown>
  web_search?: AgentWebSearch
  payload?: Record<string, unknown>
  context_manifest?: AiContextManifest & {
    captured_at?: string
    experiment_status?: string
    experiment_updated_at?: string
    context_hash?: string
  }
  created_at: string
  updated_at: string
}

export interface AgentChatConversationListResponse {
  items: AgentChatConversation[]
  total: number
  limit: number
  offset: number
}

export interface AgentChatConversationResponse {
  conversation: AgentChatConversation
  messages: AgentChatMessage[]
  has_more: boolean
  next_before_ordinal?: number | null
}

export interface AgentChatStreamPayload {
  conversation_id?: string
  provider_config_id: string
  content: string
  language?: string
  max_tokens?: number
  temperature?: number
  web_search?: boolean
}

export interface AgentChatTurnMeta {
  conversation: AgentChatConversation
  user_message: AgentChatMessage
  assistant_message: AgentChatMessage
}

export type AgentChatStreamEvent =
  | { type: 'meta'; data: AgentChatTurnMeta }
  | { type: 'delta'; data: { message_id: string; content: string } }
  | {
      type: 'search'
      data: {
        message_id: string
        status: 'checking' | 'searching' | 'complete' | 'unavailable'
        query?: string
        web_search?: AgentWebSearch
      }
    }
  | { type: 'done'; data: { message: AgentChatMessage } }
  | {
      type: 'error'
      data: { code: string; detail: string; message?: AgentChatMessage }
    }
  | { type: 'heartbeat'; data: null }

export interface ExperimentDiagnosticPreflight {
  valid: boolean
  errors: string[]
  error_codes: string[]
  warnings: string[]
  warning_codes: string[]
  execution: {
    script: string
    supports_reusable_inference: boolean
  }
}

export interface ExperimentDiagnosticEvidence {
  label: string
  value: string
  interpretation: string
  source_section: 'experiment' | 'dataset' | 'analysis' | 'model' | 'logs' | 'inference_runs'
}

export interface ExperimentDiagnosticFinding {
  severity: 'info' | 'warning' | 'critical'
  title: string
  detail: string
  source_sections: string[]
}

export interface ExperimentConfigChange {
  path: string
  before: unknown
  after: unknown
}

export interface ExperimentDraftHandoff {
  origin: 'experiment_analysis' | 'agent'
  task_type: string
  dataset_id: string
  config: Record<string, unknown>
  title: string
  rationale: string
  config_changes: ExperimentConfigChange[]
  source_experiment_id?: string
  source_experiment_name?: string
  conversation_id?: string
  message_id?: string
  recommendation_id?: string
  agent_plan_id?: string
  agent_action_id?: string
  agent_lane?: GeneratedNameQualifier
  agent_preflight?: PreflightReport
}

export interface ExperimentDiagnosticRecommendation {
  id: string
  title: string
  rationale: string
  priority: 'high' | 'medium' | 'low'
  success_criteria?: string[]
  // Legacy report fields remain optional so persisted reports continue to deserialize.
  proposed_config?: Record<string, unknown>
  config_changes?: ExperimentConfigChange[]
  preflight?: ExperimentDiagnosticPreflight
}

export interface ExperimentDiagnosticReport {
  status: 'provisional' | 'final'
  summary: string
  evidence: ExperimentDiagnosticEvidence[]
  findings: ExperimentDiagnosticFinding[]
  recommendations: ExperimentDiagnosticRecommendation[]
  limitations: string[]
}

export interface ExperimentAnalysisMessage extends AgentChatMessage {
  kind?: 'chat' | 'report'
  payload?: {
    report?: ExperimentDiagnosticReport
    warnings?: string[]
    repair_attempted?: boolean
  }
}

export interface ExperimentAnalysisConversationResponse {
  conversation: AgentChatConversation
  messages: ExperimentAnalysisMessage[]
  has_more: boolean
  next_before_ordinal?: number | null
}

export interface ExperimentAnalysisStreamPayload {
  conversation_id?: string
  provider_config_id: string
  content: string
  language?: string
  max_tokens?: number
  include?: {
    dataset_preview?: boolean
    inference_summary?: boolean
    extended_logs?: boolean
  }
}

export type ExperimentAnalysisStreamEvent =
  | {
      type: 'meta'
      data: {
        conversation: AgentChatConversation
        user_message: ExperimentAnalysisMessage
        assistant_message: ExperimentAnalysisMessage
      }
    }
  | { type: 'stage'; data: { message_id: string; stage: string } }
  | { type: 'delta'; data: { message_id: string; content: string } }
  | { type: 'done'; data: { message: ExperimentAnalysisMessage } }
  | {
      type: 'error'
      data: { code: string; detail: string; message?: ExperimentAnalysisMessage }
    }
  | { type: 'heartbeat'; data: null }

export interface AgentExecutionPlanPayload {
  provider_config_id: string
  task_type: string
  dataset_id: string
  objective?: string
  baseline_experiment_ids?: string[]
  language?: string
  include?: {
    dataset_preview?: boolean
  }
  max_tokens?: number
  strategy_preference?: AgentStrategyPreference
  diagnostic_source?: {
    experiment_id: string
    message_id: string
  }
}

export interface AgentCampaignPayload {
  task_type: string
  dataset_id: string
  objective?: string
  max_rounds?: number
}

export interface AgentCampaignRoundPayload {
  provider_config_id?: string
  objective?: string
  baseline_experiment_ids?: string[]
  language?: string
  include?: {
    dataset_preview?: boolean
  }
  max_tokens?: number
  strategy_preference?: AgentStrategyPreference
}

export type AgentGenerationStage = 'generating' | 'validating' | 'repairing' | 'saving'

export type AgentGenerationStreamEvent =
  | {
      type: 'stage'
      data: {
        stage: AgentGenerationStage
        attempt: number
        max_attempts: number
      }
    }
  | { type: 'done'; data: { plan: AgentExecutionPlan } }
  | { type: 'error'; data: { code: string; detail: string } }
  | { type: 'heartbeat'; data: null }

export interface AgentExecutionAction {
  id: string
  type: 'create_experiment' | 'cancel_experiment' | string
  lane: string
  title: string
  rationale: string
  expected_tradeoffs: string[]
  objective_alignment: string
  baseline_changes: string[]
  risk_level: 'low' | 'medium' | 'high' | string
  verification_steps: string[]
  requires_approval: boolean
  status: 'pending' | 'executed' | string
  result: Record<string, unknown>
  valid: boolean
  config?: Record<string, unknown>
  original_config?: Record<string, unknown>
  edited?: boolean
  preflight?: PreflightReport
  original_preflight?: PreflightReport
  target_experiment_id?: string
  source_action_id?: string
  created_by_plan_id?: string
}

export interface AgentPlanReview {
  summary: string
  best_experiment?: Record<string, unknown>
  metric_evidence?: string[]
  failed_or_incomplete?: Record<string, unknown>[]
  analysis_findings?: string[]
  evidence: string[]
  action_outcomes: Record<string, unknown>[]
  recommendations?: string[]
  next_recommendations: string[]
  risks: string[]
}

export interface AgentExecutionPlan {
  id: string
  owner_user_id?: string | null
  campaign_id?: string
  round_index?: number
  provider_config_id: string
  provider: string
  model: string
  task_type: string
  dataset_id: string
  objective: string
  status: 'draft' | 'executed' | 'running' | 'ready_for_report' | 'reported' | 'failed' | string
  actions: AgentExecutionAction[]
  execution: {
    executed_action_ids?: string[]
    created_experiments?: Array<{
      action_id: string
      experiment_id: string
      experiment: ExperimentRecord
    }>
    canceled_experiments?: Array<{
      action_id: string
      experiment_id: string
      experiment: ExperimentRecord
    }>
    warnings?: string[]
    report_ready?: boolean
    progress?: {
      total?: number
      queued?: number
      running?: number
      succeeded?: number
      failed?: number
      canceled?: number
      missing?: number
      completed?: boolean
      unfinished?: Array<{ id?: string; name?: string; status?: string }>
    }
  }
  review: Partial<AgentPlanReview>
  usage: Record<string, unknown>
  context_manifest: AiContextManifest
  trace?: {
    repair_attempted?: boolean
    repair_succeeded?: boolean
    repair_issues?: string[]
    action_count?: number
    valid_action_count?: number
    edited_action_count?: number
    context_chars?: number
    attempts?: number
    strategy_preference?: AgentStrategyPreference
    diagnostic_source?: {
      experiment_id?: string
      message_id?: string
      conversation_id?: string
    }
  }
  context_summary?: AgentContextSummary
  repair_attempted?: boolean
  repair_succeeded?: boolean
  warnings: string[]
  created_at: string
  updated_at: string
  executed_at?: string | null
  reviewed_at?: string | null
}

export interface AgentExecutionPlanSummary {
  id: string
  campaign_id?: string
  round_index?: number
  status: string
  task_type: string
  dataset_id: string
  objective: string
  model: string
  provider: string
  provider_config_id?: string
  action_count: number
  valid_action_count?: number
  created_experiment_count: number
  canceled_experiment_count: number
  has_review: boolean
  trace?: Record<string, unknown>
  created_at: string
  updated_at: string
  executed_at?: string | null
  reviewed_at?: string | null
}

export interface AgentExecutionPlanListParams {
  dataset_id?: string
  task_type?: string
  status?: string
  campaign_id?: string
  limit?: number
  offset?: number
}

export interface AgentExecutionPlanListResponse {
  plans: AgentExecutionPlanSummary[]
  total: number
  limit: number
  offset: number
}

export interface AgentExecutionPlanExecutePayload {
  approved_action_ids: string[]
}

export interface AgentExecutionPlanReviewPayload {
  provider_config_id?: string
  prompt?: string
  language?: string
  include?: {
    dataset_preview?: boolean
  }
  max_tokens?: number
}

export interface AgentCampaign {
  id: string
  owner_user_id?: string | null
  task_type: string
  dataset_id: string
  objective: string
  status: 'active' | 'completed' | string
  max_rounds: number
  current_round_index: number
  plans?: AgentExecutionPlanSummary[]
  created_at: string
  updated_at: string
}

export interface AgentCampaignListParams {
  dataset_id?: string
  task_type?: string
  status?: string
  limit?: number
  offset?: number
}

export interface AgentCampaignListResponse {
  campaigns: AgentCampaign[]
  total: number
  limit: number
  offset: number
}

export interface AgentExecutionActionUpdatePayload {
  config: Record<string, unknown>
  title?: string
}

export interface AgentExecutionActionConfirmPayload {
  name: string
  task_type: string
  dataset_id: string
  config: Record<string, unknown>
  generated_name?: {
    kind: 'config'
    qualifier?: GeneratedNameQualifier
  }
}

export interface LeaderboardRow {
  rank: number
  task_type: string
  experiment: {
    id: string
    name: string
    generated_name?: GeneratedExperimentName | null
    config: Record<string, unknown>
    task_type: string
    status: string
    created_at?: string
    updated_at?: string
    started_at?: string | null
    finished_at?: string | null
  }
  dataset: {
    id: string
    name: string
    source: string
    format: string
  } | null
  model: {
    id: string
    name: string
    name_is_custom?: boolean
    supports_inference: boolean
    evaluation_protocol_version?: number
    created_at?: string
  } | null
  metrics: {
    summary: Record<string, unknown>
    validation: Record<string, number>
    test: Record<string, number>
  }
  config_summary: Record<string, unknown>
  duration_seconds?: number | null
  score?: number | null
  score_source?: string
}

export interface LeaderboardResponse {
  rows: LeaderboardRow[]
  total: number
  metric: string
  split: string
  filters: {
    task_type: string
    dataset_id: string
    supports_inference?: boolean | null
    status: string
    limit: number
    global_view?: boolean
  }
}

export type ComparisonMetricKey =
  | 'f1'
  | 'auc'
  | 'aupr'
  | 'acc'
  | 'sacc'
  | 'balanced_accuracy'
  | 'precision'
  | 'recall'
  | 'specificity'
  | 'mcc'

export type ComparisonSplitKey = 'test' | 'validation'

export interface ComparisonMetricCell {
  experiment_id: string
  value?: number
  source?: string
}

export interface ComparisonMetricRow {
  metric: ComparisonMetricKey
  validation: ComparisonMetricCell[]
  test: ComparisonMetricCell[]
}

export interface ConfigDiffRow {
  key: string
  changed: boolean
  highlighted: boolean
  values: Record<string, string>
  sources: Record<string, string>
}

export interface ComparisonReportPayload {
  generated_at: string
  experiments: ExperimentRecord[]
  datasets: Record<string, DatasetRecord | undefined>
  metric_rows: ComparisonMetricRow[]
  config_diff: ConfigDiffRow[]
  analyses: Record<string, ExperimentAnalysis | undefined>
}

export interface AuthUser {
  id: string
  email: string
  display_name: string
  preferred_locale: 'en' | 'zh'
  email_verified: boolean
  role: 'user' | 'admin' | string
  status: 'active' | 'disabled' | string
}

export interface AuthResponse {
  auth_enabled: boolean
  user: AuthUser | null
  quota?: UserQuota
  usage?: QuotaUsage
  resource_counts?: AdminResourceCounts
  email?: string
  verification_sent?: boolean
  reset_sent?: boolean
  debug_code?: string
}

export interface UserQuota {
  user_id: string
  daily_experiment_limit: number
  daily_concurrent_experiment_limit: number
  daily_inference_request_limit: number
  daily_inference_input_limit: number
  updated_at?: string | null
}

export interface QuotaUsage {
  user_id: string
  usage_date: string
  experiments_created: number
  inference_requests: number
  inference_inputs: number
  updated_at?: string | null
}

export interface AdminResourceCounts {
  datasets: number
  experiments: number
  active_experiments: number
  models: number
  inference_runs: number
  agent_campaigns?: number
  agent_execution_plans?: number
  agent_chat_conversations?: number
}

export interface AdminOverview {
  generated_at: string
  usage_date: string
  users: {
    total: number
    active: number
    disabled: number
    unverified: number
    new_last_7_days: number
  }
  workload: {
    experiments_created_today: number
    active_experiments: number
    inference_requests_today: number
    inference_inputs_today: number
  }
  security_24h: {
    failed_logins: number
    admin_actions: number
    session_revoke_events: number
    user_deletions: number
  }
}

export interface AdminSession {
  id: string
  user_id: string
  expires_at: string
  user_agent: string
  created_at: string
  last_seen_at: string
  revoked_at?: string | null
  active: boolean
}

export interface AdminVerificationCode {
  id: string
  user_id: string
  purpose: string
  expires_at: string
  attempts: number
  consumed_at?: string | null
  created_at: string
  active: boolean
}

export interface AdminUser extends AuthUser {
  created_at?: string
  updated_at?: string
  quota: UserQuota
  usage: QuotaUsage
  resource_counts: AdminResourceCounts
  sessions?: AdminSession[]
  verification_codes?: AdminVerificationCode[]
  can_delete?: boolean
  delete_block_reason?: string
}

export interface AdminUsersResponse {
  items: AdminUser[]
  total: number
  limit: number
  offset: number
}

export interface AdminAuditUserRef {
  id: string
  email: string
  display_name: string
  role: string
  status: string
}

export interface AdminAuditLog {
  id: string
  actor_user_id?: string | null
  target_user_id?: string | null
  event_type: string
  ip_address: string
  user_agent: string
  metadata: Record<string, unknown>
  created_at: string
  actor?: AdminAuditUserRef | null
  target?: AdminAuditUserRef | null
}

export interface AdminAuditLogsResponse {
  items: AdminAuditLog[]
  total: number
  limit: number
  offset: number
}

export interface AdminAuditEventType {
  event_type: string
  count: number
}

export interface AdminAuditEventTypesResponse {
  items: AdminAuditEventType[]
}

export interface AdminResendVerificationResponse {
  user: AdminUser
  verification_sent: boolean
  debug_code?: string
}

export interface AdminRevokeSessionsResponse {
  user: AdminUser
  revoked_sessions: number
}

export interface AdminDeleteUserResponse {
  user_id: string
  email: string
  deleted: boolean
  deleted_counts: Record<string, number>
}

export interface DeleteModelResponse {
  model_id: string
  artifact_deleted: boolean
}

export interface DeleteDatasetResponse {
  dataset_id: string
  artifact_deleted: boolean
}

export interface DeleteExperimentResponse {
  experiment_id: string
  artifact_deleted: boolean
  model_artifacts_deleted: number
}

export interface Capabilities {
  selection_objectives?: Array<{
    id: string
    version: number
    direction: string
    requires_predictions: boolean
    tasks: string[]
    requires_both_classes: boolean
    parameters: string[]
  }>
  route_schema_version: number
  output_kinds: EncoderOutputKind[]
  route_families: Array<{ id: string; task_scope: string; label: string }>
  encoder_specs: Array<{ id: string; label: string; family: string }>
  predictor_parts: string[]
  defaults: Record<string, Record<string, string>>
  tasks: Array<{ id: string; label: string }>
  feature_routes: string[]
  pretrained_tokenizers: string[]
  pretrained_models?: Record<
    string,
    Array<{
      id: string
      label: string
      variant?: string
      parameter_count?: number
      hidden_size?: number
      runtime_key?: string
      license_url?: string
      download_policy?: string
    }>
  >
  classic_tokenizers: string[]
  classic_encoder_options: Record<string, string[]>
  svm_calibration?: {
    options: string[]
    default: string
    cv_folds: number
  }
  classical_tuning?: {
    supported_classifiers: string[]
    objectives: string[]
    mode: 'cv'
    default_budget: 'quick' | 'standard' | 'thorough'
    space_version: string
    semantics: 'best_observed_candidate'
    budgets: Record<
      string,
      {
        max_candidates: number
        cv_folds: number
        max_candidate_fold_evaluations: number
      }
    >
  }
  sequence_profile_contract: {
    version: number
    strict_features: string[]
    explicit_hybrid_feature: string
    explicit_hybrid_encoders?: string[]
    required_k: number
    fallback_policies: Record<string, string>
  }
  covariance_encoder_contract: {
    version: number
    encoders: string[]
    strict_encoders?: string[]
    hybrid_encoders?: string[]
    tokenizer: 'kmer'
    required_k: number
    properties: string[]
    max_raw_length: {
      default: number
      absolute_max: number
    }
    stream_policy: 'complete_raw_sequence'
  }
  pdt_encoder_contract: {
    version: number
    encoders: string[]
    tokenizer: 'kmer'
    required_k: number
    default_lag: number
    dimensions_per_lag: number
    method_source_doi: string
    max_raw_length: {
      default: number
      absolute_max: number
    }
    stream_policy: 'complete_raw_sequence'
  }
  pssm_covariance_encoder_contract: {
    version: number
    encoders: string[]
    tokenizer: 'kmer'
    required_k: number
    default_lag: number
    dimensions_per_lag: Record<string, number>
    lag_keys?: Record<string, string>
    profile_feature_types?: Record<string, string>
    stream_policy: 'complete_pssm_profile'
  }
  tokenizer_constraints: {
    kmer: {
      k: {
        min: number
        max: number
      }
      max_length_unit: 'tokens'
    }
    wordpiece?: {
      max_length_unit: 'tokens_including_special_tokens'
      max_raw_length: {
        default: number
        absolute_max: number
      }
    }
    unigram?: {
      max_length_unit: 'tokens_including_special_tokens'
      max_raw_length: {
        default: number
        absolute_max: number
      }
      sampling: {
        default_enabled: boolean
        default_nbest_size: number
        min_nbest_size: number
        max_nbest_size: number
        default_alpha: number
        inference_mode: 'viterbi'
      }
    }
    prott5?: {
      max_length_unit: 'residues'
      max_residues: number
      total_token_compatibility_adjustment: {
        requested_total_tokens: number
        effective_residues: number
        reason: string
      }
    }
  }
  feature_transformations: Record<string, string[]>
  pair_feature_transformations?: Record<string, string[]>
  lsa_contract?: {
    version: number
    variant: string
    route: string
    pair_supported: boolean
    dense_memory_limit_gib_default: number
  }
  lda_contract?: {
    version: number
    variant: string
    route: string
    pair_route?: string
    pair_supported: boolean
    dense_memory_limit_gib_default: number
  }
  plsa_contract?: {
    version: number
    variant: string
    route: string
    pair_route?: string
    pair_supported: boolean
    memory_limit_gib_default: number
  }
  classifiers: string[]
  sequence_models: string[]
  modeling_architectures: {
    sequence: string[]
    chain_graph: string[]
  }
  pair_strategies: string[]
  pair_strategy_aliases?: Record<string, string>
  pair_strategy_labels?: Record<string, { en: string; zh: string }>
  interaction_levels: string[]
  interaction_models: string[]
  semantic_relation_types: string[]
  task_semantic_relation_type: Record<string, string>
  semantic_defaults: Record<string, number>
  predictor_types: string[]
  graph_predictors: string[]
  training_modes: string[]
  route_constraints: {
    single_sequence: {
      classic: {
        chain_graph_encoders: string[]
      }
      pretrained: {
        pooled_predictors: string[]
        residue_architectures: Record<string, string[]>
        residue_training_modes: Record<string, string[]>
        residue_adapter_dim: {
          default: number
          min: number
          max: number
          multiple_of: number
        }
        residue_encoder_frozen: boolean
        residue_graph_predictors: Record<string, string[]>
        residue_graph_training_modes: Record<string, string[]>
        residue_graph_prediction_head: string
        unsupported_post_encoders: string[]
      }
    }
    pair: {
      pretrained_token_interaction: {
        pair_strategy: string
        interaction_level: string
        interaction_models: string[]
        classifier: string
      }
    }
  }
}

export interface PreflightReport {
  valid: boolean
  errors: string[]
  error_codes: string[]
  warnings: string[]
  warning_codes: string[]
  normalized_config: Record<string, unknown>
  effective_route: EffectiveRoute
  parameter_sources: Record<string, 'explicit' | 'derived' | 'default' | string>
  dataset: {
    id: string
    name: string
    task_type: string
    format: string
    metadata: Record<string, unknown>
  }
  execution: {
    script: string
    supports_reusable_inference: boolean
  }
}
