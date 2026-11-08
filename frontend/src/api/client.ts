import { normalizePairFields } from '../pairTerminology'
import { normalizeTaskFields } from '../taskTerminology'
import type { EnsembleCapabilities, EnsemblePayload, EnsemblePreflight } from '../ensemble'
import { notifyUnauthorized } from '../authSession'
import { notifyQuotaUsageChanged } from '../accountQuota'
import type {
  Announcement,
  AnnouncementPayload,
  DailyReductionMode,
  EmailDeliveryUsage,
  OwnResources,
  ModelCleanupPreview,
  PolicyDraft,
  PolicyPreview,
  ResourceOverview,
  ResourcePolicy,
} from '../resourceTypes'
import type {
  AdminAuditEventTypesResponse,
  AdminAuditLogsResponse,
  AdminDeleteUserResponse,
  AdminRevokeSessionsResponse,
  AdminResendVerificationResponse,
  AdminOverview,
  AdminUser,
  AdminUsersResponse,
  AgentCampaign,
  AgentCampaignListParams,
  AgentCampaignListResponse,
  AgentCampaignPayload,
  AgentCampaignRoundPayload,
  AgentChatPayload,
  AgentChatResponse,
  AgentChatConversation,
  AgentChatConversationListResponse,
  AgentChatConversationResponse,
  AgentWebSearchStatus,
  AgentExecutionActionConfirmPayload,
  AgentExecutionActionUpdatePayload,
  AgentExecutionPlan,
  AgentExecutionPlanExecutePayload,
  AgentExecutionPlanListParams,
  AgentExecutionPlanListResponse,
  AgentExecutionPlanPayload,
  AgentExecutionPlanReviewPayload,
  AgentExperimentBatchPayload,
  AgentExperimentBatchResponse,
  AgentExperimentDesignPayload,
  AgentExperimentDesignResponse,
  AiProviderConfig,
  AiProviderPayload,
  AiProviderPreset,
  AiProviderTestResponse,
  AuthResponse,
  Capabilities,
  DatasetRecord,
  DatasetReport,
  DeleteDatasetResponse,
  DeleteExperimentResponse,
  DeleteInferenceRunResponse,
  ExperimentAnalysis,
  ExperimentAnalysisConversationResponse,
  ExperimentAnalysisMessage,
  ExperimentAiAnalysisResponse,
  ExperimentRecord,
  InferenceReadiness,
  InferenceRunRecord,
  LeaderboardResponse,
  DeleteModelResponse,
  ModelRecord,
  PredictionPair,
  PredictionResponse,
  PreflightReport,
  UserQuota,
  QuotaKey,
} from '../types'

export class ApiError extends Error {
  status: number
  code?: string
  quotaType?: string
  limit?: number
  used?: number
  requested?: number
  stage?: string
  hint?: string
  technicalDetail?: string
  params?: Record<string, string | number>
  detail?: Record<string, unknown>
  resetAt?: string

  constructor(
    message: string,
    status: number,
    detail?: {
      code?: string
      quota_type?: string
      limit?: number
      used?: number
      requested?: number
      stage?: string
      hint?: string
      technical_detail?: string
      params?: Record<string, string | number>
      [key: string]: unknown
    },
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = detail?.code
    this.quotaType = detail?.quota_type
    this.limit = detail?.limit
    this.used = detail?.used
    this.requested = detail?.requested
    this.stage = detail?.stage
    this.hint = detail?.hint
    this.technicalDetail = detail?.technical_detail
    this.params = detail?.params
    this.detail = detail
    this.resetAt = typeof detail?.reset_at === 'string' ? detail.reset_at : undefined
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers =
    init?.body instanceof FormData
      ? init?.headers
      : { 'Content-Type': 'application/json', ...(init?.headers || {}) }
  const response = await fetch(path, {
    ...init,
    headers,
    credentials: 'include',
  })
  notifyQuotaUsageChanged(path, init?.method, response.status)
  if (!response.ok) {
    notifyUnauthorized(path, response.status)
    const payload = await response.json().catch(() => ({}))
    const detail = payload.detail
    const message =
      typeof detail === 'string'
        ? detail
        : typeof detail?.message === 'string'
          ? detail.message
          : Array.isArray(detail?.errors)
            ? detail.errors.join('; ')
            : `Request failed: ${response.status}`
    throw new ApiError(message, response.status, typeof detail === 'object' ? detail : undefined)
  }
  return normalizePairFields(normalizeTaskFields(await response.json())) as T
}

export async function requestBlob(path: string): Promise<Blob> {
  const response = await fetch(path, { credentials: 'include' })
  if (!response.ok) {
    notifyUnauthorized(path, response.status)
    const payload = await response.json().catch(() => ({}))
    const detail = payload.detail
    throw new ApiError(
      typeof detail === 'string'
        ? detail
        : typeof detail?.message === 'string'
          ? detail.message
          : `Request failed: ${response.status}`,
      response.status,
      typeof detail === 'object' ? detail : undefined,
    )
  }
  return response.blob()
}

export interface LeaderboardParams {
  task_type?: string
  dataset_id?: string
  metric?: string
  split?: string
  supports_inference?: boolean
  status?: string
  limit?: number
  global_view?: boolean
}

export interface AdminUsersParams {
  search?: string
  role?: string
  status?: string
  email_verified?: boolean | string
  sort?: 'created_at' | 'display_name' | 'email' | 'role' | 'status'
  order?: 'asc' | 'desc'
  limit?: number
  offset?: number
}

export interface AdminAuditLogParams {
  search?: string
  event_type?: string
  actor_user_id?: string
  target_user_id?: string
  user_id?: string
  created_from?: string
  created_to?: string
  limit?: number
  offset?: number
}

export type AdminUserUpdatePayload = Partial<Pick<AdminUser, 'role' | 'status'>>
export type UserQuotaUpdatePayload = Partial<Record<QuotaKey, number | null>> & {
  allow_global_override?: boolean
  daily_reduction_mode?: DailyReductionMode
  publish_announcements?: boolean
}

export interface ExperimentAiAnalysisPayload {
  provider_config_id: string
  prompt: string
  language?: string
  include?: {
    dataset_preview?: boolean
    inference_summary?: boolean
    extended_logs?: boolean
  }
}

function queryString<T extends object>(params: T) {
  const query = new URLSearchParams()
  Object.entries(params as Record<string, unknown>).forEach(([key, value]) => {
    if (value === undefined || value === '') return
    query.set(key, String(value))
  })
  const text = query.toString()
  return text ? `?${text}` : ''
}

export const api = {
  emailDeliveryUsage: () => request<EmailDeliveryUsage>('/api/admin/email-delivery-usage'),
  resourcePolicy: () => request<ResourcePolicy>('/api/admin/resource-policies'),
  previewResourcePolicy: (payload: PolicyDraft) =>
    request<PolicyPreview>('/api/admin/resource-policies/preview', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  saveResourcePolicy: (payload: PolicyDraft) =>
    request<{
      policy: ResourcePolicy
      application: PolicyPreview | null
      impacts: PolicyPreview['impacts']
    }>('/api/admin/resource-policies', { method: 'PATCH', body: JSON.stringify(payload) }),
  adminResources: () => request<ResourceOverview>('/api/admin/resources'),
  modelCleanupPreview: () =>
    request<ModelCleanupPreview>('/api/admin/model-cleanup/preview', { method: 'POST' }),
  cleanupModel: (id: string) =>
    request<{ deleted: boolean; released_bytes: number; error?: string }>(
      `/api/admin/model-cleanup/${encodeURIComponent(id)}`,
      { method: 'POST' },
    ),
  scanResources: () => request<ResourceOverview>('/api/admin/resources/scan', { method: 'POST' }),
  checkRetention: (payload: { publish_announcements?: boolean } = {}) =>
    request<ResourceOverview>('/api/admin/resources/retention', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  cleanupResource: (id: string) =>
    request<{ deleted: boolean; released_bytes: number; error?: string }>(
      `/api/admin/resources/${id}/cleanup`,
      { method: 'POST' },
    ),
  ownResources: () => request<OwnResources>('/api/resources/me'),
  announcements: () => request<{ items: Announcement[] }>('/api/announcements'),
  adminAnnouncements: () => request<{ items: Announcement[] }>('/api/admin/announcements'),
  publishAnnouncement: (payload: AnnouncementPayload) =>
    request<Announcement>('/api/admin/announcements', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  updateAnnouncement: (id: string, payload: AnnouncementPayload) =>
    request<Announcement>(`/api/admin/announcements/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  deleteAnnouncement: (id: string) =>
    request(`/api/admin/announcements/${id}`, { method: 'DELETE' }),
  readAnnouncement: (id: string) => request(`/api/announcements/${id}/read`, { method: 'POST' }),
  modelProtection: (id: string) =>
    request<{ model_id: string; pinned: boolean; reproducibility_retained: boolean }>(
      `/api/models/${id}/protection`,
    ),
  protectModel: (id: string, pinned: boolean) =>
    request(`/api/models/${id}/protection`, { method: 'PATCH', body: JSON.stringify({ pinned }) }),
  ensembleCapabilities: () => request<EnsembleCapabilities>('/api/ensembles/capabilities'),
  preflightEnsemble: (payload: EnsemblePayload) =>
    request<EnsemblePreflight>('/api/ensembles/preflight', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  createEnsemble: (payload: EnsemblePayload) =>
    request<ExperimentRecord>('/api/ensembles', { method: 'POST', body: JSON.stringify(payload) }),
  capabilities: () => request<Capabilities>('/api/capabilities'),
  datasets: () => request<DatasetRecord[]>('/api/datasets'),
  datasetReport: (id: string, limit = 5) =>
    request<DatasetReport>(`/api/datasets/${id}/report?limit=${limit}`),
  uploadDataset: (formData: FormData) =>
    request<DatasetRecord>('/api/datasets/upload', {
      method: 'POST',
      body: formData,
    }),
  deleteDataset: (id: string) =>
    request<DeleteDatasetResponse>(`/api/datasets/${id}`, {
      method: 'DELETE',
    }),
  experiments: () => request<ExperimentRecord[]>('/api/experiments'),
  preflightExperiment: (payload: Record<string, unknown>) =>
    request<PreflightReport>('/api/experiments/preflight', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  experiment: (id: string) => request<ExperimentRecord>(`/api/experiments/${id}`),
  experimentLogs: (id: string) => request<{ content: string }>(`/api/experiments/${id}/logs`),
  experimentAnalysis: (id: string) =>
    request<ExperimentAnalysis>(`/api/experiments/${id}/analysis`),
  leaderboard: (params: LeaderboardParams = {}) =>
    request<LeaderboardResponse>(`/api/leaderboard${queryString(params)}`),
  createExperiment: (payload: Record<string, unknown>) =>
    request<ExperimentRecord>('/api/experiments', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  resumeDownstreamSvm: (id: string, overrides: Record<string, unknown>) =>
    request<ExperimentRecord>(`/api/experiments/${id}/resume-downstream`, {
      method: 'POST',
      body: JSON.stringify({ overrides }),
    }),
  updateExperiment: (id: string, payload: { name: string }) =>
    request<ExperimentRecord>(`/api/experiments/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  deleteExperiment: (id: string) =>
    request<DeleteExperimentResponse>(`/api/experiments/${id}`, {
      method: 'DELETE',
    }),
  cancelExperiment: (id: string) =>
    request<ExperimentRecord>(`/api/experiments/${id}/cancel`, {
      method: 'POST',
    }),
  models: () => request<ModelRecord[]>('/api/models'),
  model: (id: string) => request<ModelRecord>(`/api/models/${id}`),
  inferenceReadiness: (id: string) =>
    request<InferenceReadiness>(`/api/models/${id}/inference-readiness`),
  updateModel: (id: string, payload: { name: string }) =>
    request<ModelRecord>(`/api/models/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  deleteModel: (id: string) =>
    request<DeleteModelResponse>(`/api/models/${id}`, {
      method: 'DELETE',
    }),
  inferenceRuns: (modelId: string) =>
    request<InferenceRunRecord[]>(`/api/models/${modelId}/inference-runs`),
  inferenceRunCsv: (runId: string) => requestBlob(`/api/inference-runs/${runId}/csv`),
  deleteInferenceRun: (runId: string) =>
    request<DeleteInferenceRunResponse>(`/api/inference-runs/${runId}`, {
      method: 'DELETE',
    }),
  predict: (modelId: string, payload: { sequences?: string[]; pairs?: PredictionPair[] }) =>
    request<PredictionResponse>(`/api/models/${modelId}/predict`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  aiProviders: () => request<AiProviderConfig[]>('/api/ai/providers'),
  aiProviderPresets: () => request<AiProviderPreset[]>('/api/ai/provider-presets'),
  createAiProvider: (payload: AiProviderPayload) =>
    request<AiProviderConfig>('/api/ai/providers', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  updateAiProvider: (id: string, payload: Partial<AiProviderPayload>) =>
    request<AiProviderConfig>(`/api/ai/providers/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  deleteAiProvider: (id: string) =>
    request<{ provider_id: string; deleted: boolean }>(`/api/ai/providers/${id}`, {
      method: 'DELETE',
    }),
  testAiProvider: (id: string) =>
    request<AiProviderTestResponse>(`/api/ai/providers/${id}/test`, {
      method: 'POST',
    }),
  agentChat: (payload: AgentChatPayload) =>
    request<AgentChatResponse>('/api/agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  agentWebSearchStatus: () => request<AgentWebSearchStatus>('/api/agent/web-search/status'),
  agentChatConversations: (params: { q?: string; limit?: number; offset?: number } = {}) =>
    request<AgentChatConversationListResponse>(
      `/api/agent/chat/conversations${queryString(params)}`,
    ),
  agentChatConversation: (id: string, params: { limit?: number; before_ordinal?: number } = {}) =>
    request<AgentChatConversationResponse>(
      `/api/agent/chat/conversations/${id}${queryString(params)}`,
    ),
  updateAgentChatConversation: (id: string, payload: { title?: string; pinned?: boolean }) =>
    request<AgentChatConversation>(`/api/agent/chat/conversations/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  deleteAgentChatConversation: (id: string) =>
    request<{ deleted: boolean; conversation: AgentChatConversation }>(
      `/api/agent/chat/conversations/${id}`,
      { method: 'DELETE' },
    ),
  agentExperimentDesign: (payload: AgentExperimentDesignPayload) =>
    request<AgentExperimentDesignResponse>('/api/agent/experiment-design', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  agentExperimentBatch: (payload: AgentExperimentBatchPayload) =>
    request<AgentExperimentBatchResponse>('/api/agent/experiment-batches', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  createAgentCampaign: (payload: AgentCampaignPayload) =>
    request<AgentCampaign>('/api/agent/campaigns', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  agentCampaigns: (params: AgentCampaignListParams = {}) =>
    request<AgentCampaignListResponse>(`/api/agent/campaigns${queryString(params)}`),
  agentCampaign: (id: string) => request<AgentCampaign>(`/api/agent/campaigns/${id}`),
  createAgentCampaignRound: (id: string, payload: AgentCampaignRoundPayload) =>
    request<AgentExecutionPlan>(`/api/agent/campaigns/${id}/rounds`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  createAgentExecutionPlan: (payload: AgentExecutionPlanPayload) =>
    request<AgentExecutionPlan>('/api/agent/execution-plans', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  agentExecutionPlans: (params: AgentExecutionPlanListParams = {}) =>
    request<AgentExecutionPlanListResponse>(`/api/agent/execution-plans${queryString(params)}`),
  deleteAgentExecutionPlan: (id: string) =>
    request<{ plan_id: string; deleted: boolean }>(`/api/agent/execution-plans/${id}`, {
      method: 'DELETE',
    }),
  agentExecutionPlan: (id: string) =>
    request<AgentExecutionPlan>(`/api/agent/execution-plans/${id}`),
  refreshAgentExecutionPlan: (id: string) =>
    request<AgentExecutionPlan>(`/api/agent/execution-plans/${id}/refresh`, {
      method: 'POST',
    }),
  updateAgentExecutionAction: (
    planId: string,
    actionId: string,
    payload: AgentExecutionActionUpdatePayload,
  ) =>
    request<AgentExecutionPlan>(`/api/agent/execution-plans/${planId}/actions/${actionId}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  confirmAgentExecutionAction: (
    planId: string,
    actionId: string,
    payload: AgentExecutionActionConfirmPayload,
  ) =>
    request<ExperimentRecord>(`/api/agent/execution-plans/${planId}/actions/${actionId}/confirm`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  executeAgentExecutionPlan: (id: string, payload: AgentExecutionPlanExecutePayload) =>
    request<AgentExecutionPlan>(`/api/agent/execution-plans/${id}/execute`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  reviewAgentExecutionPlan: (id: string, payload: AgentExecutionPlanReviewPayload) =>
    request<AgentExecutionPlan>(`/api/agent/execution-plans/${id}/review`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  experimentAiAnalysis: (id: string, payload: ExperimentAiAnalysisPayload) =>
    request<ExperimentAiAnalysisResponse>(`/api/experiments/${id}/ai-analysis`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  experimentAnalysisConversations: (
    id: string,
    params: { q?: string; limit?: number; offset?: number } = {},
  ) =>
    request<AgentChatConversationListResponse>(
      `/api/experiments/${id}/analysis/conversations${queryString(params)}`,
    ),
  experimentAnalysisConversation: (
    id: string,
    conversationId: string,
    params: { limit?: number; before_ordinal?: number } = {},
  ) =>
    request<ExperimentAnalysisConversationResponse>(
      `/api/experiments/${id}/analysis/conversations/${conversationId}${queryString(params)}`,
    ),
  experimentAnalysisMessage: (id: string, messageId: string) =>
    request<ExperimentAnalysisMessage>(`/api/experiments/${id}/analysis/messages/${messageId}`),
  updateExperimentAnalysisConversation: (
    id: string,
    conversationId: string,
    payload: { title?: string; pinned?: boolean },
  ) =>
    request<AgentChatConversation>(
      `/api/experiments/${id}/analysis/conversations/${conversationId}`,
      { method: 'PATCH', body: JSON.stringify(payload) },
    ),
  deleteExperimentAnalysisConversation: (id: string, conversationId: string) =>
    request<{ deleted: boolean; conversation: AgentChatConversation }>(
      `/api/experiments/${id}/analysis/conversations/${conversationId}`,
      { method: 'DELETE' },
    ),
  me: () => request<AuthResponse>('/api/auth/me'),
  updateProfile: (payload: { display_name: string }) =>
    request<AuthResponse>('/api/auth/profile', {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  updatePreferences: (payload: { preferred_locale: 'en' | 'zh' }) =>
    request<AuthResponse>('/api/auth/preferences', {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  register: (payload: { email: string; password: string; preferred_locale?: 'en' | 'zh' }) =>
    request<AuthResponse>('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  verifyEmail: (payload: { email: string; code: string }) =>
    request<AuthResponse>('/api/auth/verify-email', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  resendVerification: (payload: { email: string }) =>
    request<AuthResponse>('/api/auth/resend-verification', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  login: (payload: { email: string; password: string }) =>
    request<AuthResponse>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  logout: () =>
    request<{ ok: boolean }>('/api/auth/logout', {
      method: 'POST',
    }),
  requestPasswordReset: (payload: { email: string }) =>
    request<AuthResponse>('/api/auth/password-reset/request', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  confirmPasswordReset: (payload: { email: string; code: string; password: string }) =>
    request<AuthResponse>('/api/auth/password-reset/confirm', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  adminUsers: (params: AdminUsersParams = {}) =>
    request<AdminUsersResponse>(`/api/admin/users${queryString(params)}`),
  adminOverview: () => request<AdminOverview>('/api/admin/overview'),
  adminUser: (id: string) => request<AdminUser>(`/api/admin/users/${id}`),
  adminUpdateUser: (id: string, payload: AdminUserUpdatePayload) =>
    request<AdminUser>(`/api/admin/users/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  adminUpdateUserQuotas: (id: string, payload: UserQuotaUpdatePayload) =>
    request<UserQuota>(`/api/admin/users/${id}/quotas`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  adminRevokeUserSessions: (id: string) =>
    request<AdminRevokeSessionsResponse>(`/api/admin/users/${id}/revoke-sessions`, {
      method: 'POST',
    }),
  adminResendVerification: (id: string) =>
    request<AdminResendVerificationResponse>(`/api/admin/users/${id}/resend-verification`, {
      method: 'POST',
    }),
  adminDeleteUser: (id: string) =>
    request<AdminDeleteUserResponse>(`/api/admin/users/${id}`, {
      method: 'DELETE',
    }),
  adminAuditLogs: (params: AdminAuditLogParams = {}) =>
    request<AdminAuditLogsResponse>(`/api/admin/audit-logs${queryString(params)}`),
  adminAuditEventTypes: () => request<AdminAuditEventTypesResponse>('/api/admin/audit-event-types'),
}
