<script setup lang="ts">
import { useQuery } from '@tanstack/vue-query'
import {
  AlertTriangle,
  ArrowUpRight,
  Bot,
  Check,
  CheckCircle2,
  Clock3,
  Copy,
  Database,
  FileSearch,
  FlaskConical,
  Info,
  Menu,
  MessageSquarePlus,
  MoreHorizontal,
  Pencil,
  Pin,
  Search,
  Send,
  PlugZap,
  Sparkles,
  Square,
  Trash2,
  X,
} from '@lucide/vue'
import { computed, nextTick, onActivated, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { api } from '../api/client'
import { createAgentChatMarkdownRenderer } from '../agentChatMarkdown'
import { experimentDisplayName } from '../localization'
import { useExperimentAnalysisStore } from '../stores/experimentAnalysis'
import type {
  AgentChatConversation,
  ExperimentAnalysisMessage,
  ExperimentDiagnosticReport,
} from '../types'

const props = defineProps<{ id: string; conversationId?: string }>()
const { t, te, locale } = useI18n()
const router = useRouter()
const analysis = useExperimentAnalysisStore()
const markdownRenderer = createAgentChatMarkdownRenderer()
const experimentQuery = useQuery({
  queryKey: computed(() => ['experiment', props.id]),
  queryFn: () => api.experiment(props.id),
  refetchInterval: (query) => (query.state.data?.status === 'running' ? 2000 : false),
})
const providersQuery = useQuery({ queryKey: ['ai-providers'], queryFn: api.aiProviders })
const enabledProviders = computed(() =>
  (providersQuery.data.value ?? []).filter((provider) => provider.enabled),
)
const selectedProviderId = ref('')
const input = ref('')
const composer = ref<HTMLTextAreaElement | null>(null)
const thread = ref<HTMLElement | null>(null)
const search = ref('')
const sidebarOpen = ref(false)
const contextOpen = ref(false)
const openMenuId = ref('')
const editingId = ref('')
const editingTitle = ref('')
const localError = ref('')
const copiedMessageId = ref('')
const includeDatasetPreview = ref(false)
const includeInferenceSummary = ref(false)
const includeExtendedLogs = ref(false)
let pendingNewConversation = false
let searchTimer: number | undefined

const conversationId = computed(() => String(props.conversationId || ''))
const experiment = computed(() => experimentQuery.data.value)
const currentConversation = computed(() =>
  analysis.conversations.find((item) => item.id === conversationId.value),
)
const messages = computed(() => analysis.messagesByConversation[conversationId.value] || [])
const assistantMessages = computed(() =>
  messages.value.filter((message) => message.role === 'assistant'),
)
const latestAssistantMessage = computed(() => assistantMessages.value.at(-1))
const canAnalyze = computed(() =>
  ['running', 'succeeded', 'failed', 'canceled'].includes(experiment.value?.status || ''),
)
const canSend = computed(() =>
  Boolean(
    input.value.trim() &&
    selectedProviderId.value &&
    enabledProviders.value.length &&
    canAnalyze.value &&
    !analysis.generating,
  ),
)
const isCurrentGenerating = computed(
  () => analysis.generating && analysis.activeGenerationConversationId === conversationId.value,
)
const anotherConversationGenerating = computed(
  () => analysis.generating && analysis.activeGenerationConversationId !== conversationId.value,
)
const visibleError = computed(() => {
  if (localError.value) return localError.value
  if (analysis.streamErrorStatus) {
    const statusKey = `apiErrors.http${analysis.streamErrorStatus}`
    if (te(statusKey)) return t(statusKey)
  }
  if (!analysis.streamError) return ''
  const codeKey = `apiErrors.${analysis.streamError}`
  if (te(codeKey)) return t(codeKey)
  if (['stream_failed', 'experiment_analysis_failed'].includes(analysis.streamError)) {
    return t('detail.analysisError')
  }
  return analysis.streamError
})

const suggestions = computed(() => [
  {
    title: t('detail.analysisSuggestionSummary'),
    prompt: t('detail.aiPromptSummaryText'),
    icon: Sparkles,
  },
  {
    title: t('detail.analysisSuggestionWeakness'),
    prompt: t('detail.aiPromptWeaknessText'),
    icon: FileSearch,
  },
  {
    title: t('detail.analysisSuggestionNext'),
    prompt: t('detail.aiPromptNextText'),
    icon: FlaskConical,
  },
])

const stageItems = computed(() => [
  { id: 'context_ready', label: t('detail.analysisStageContext') },
  { id: 'analyzing_evidence', label: t('detail.analysisStageEvidence') },
  { id: 'validating_report', label: t('detail.analysisStageReport') },
])

const stageOrder: Record<string, number> = {
  context_ready: 0,
  analyzing_evidence: 1,
  validating_report: 2,
  repairing_report: 2,
  answering_follow_up: 1,
}

const groupedConversations = computed(() => {
  const now = new Date()
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const weekStart = todayStart - 6 * 24 * 60 * 60 * 1000
  const groups = [
    { key: 'today', label: t('detail.analysisHistoryToday'), items: [] as AgentChatConversation[] },
    { key: 'week', label: t('detail.analysisHistoryWeek'), items: [] as AgentChatConversation[] },
    { key: 'older', label: t('detail.analysisHistoryOlder'), items: [] as AgentChatConversation[] },
  ]
  for (const conversation of analysis.conversations) {
    const updated = new Date(conversation.updated_at).getTime()
    if (updated >= todayStart) groups[0].items.push(conversation)
    else if (updated >= weekStart) groups[1].items.push(conversation)
    else groups[2].items.push(conversation)
  }
  return groups.filter((group) => group.items.length)
})

const requiredContextSections = computed(() => [
  t('detail.analysisContextExperiment'),
  t('detail.analysisContextDataset'),
  t('detail.analysisContextArtifacts'),
  t('detail.analysisContextModel'),
  t('detail.analysisContextLogs'),
])

function reportFor(message: ExperimentAnalysisMessage): ExperimentDiagnosticReport | null {
  return message.payload?.report || null
}

function renderMarkdown(message: ExperimentAnalysisMessage) {
  return markdownRenderer.render(message)
}

function providerLabel(message: ExperimentAnalysisMessage) {
  const provider = enabledProviders.value.find((item) => item.id === message.provider_config_id)
  return provider?.display_name || message.provider || t('detail.analysisAssistant')
}

function statusClass(status: string) {
  if (status === 'succeeded') return 'status-pill-success'
  if (status === 'failed') return 'status-pill-danger'
  return 'status-pill-warning'
}

function reportIsStale(message: ExperimentAnalysisMessage) {
  const currentUpdatedAt = experiment.value?.updated_at || ''
  const capturedUpdatedAt = message.context_manifest?.experiment_updated_at || ''
  return Boolean(currentUpdatedAt && capturedUpdatedAt && currentUpdatedAt > capturedUpdatedAt)
}

function sourceLabel(source: string) {
  return t(`detail.analysisSource.${source}`)
}

function sourceTarget(source: string) {
  const targets: Record<string, string> = {
    experiment: 'experiment-overview',
    dataset: 'experiment-overview',
    analysis: 'experiment-analysis-artifacts',
    model: 'experiment-model',
    logs: 'experiment-logs',
    inference_runs: 'experiment-inference',
  }
  return `/experiments/${props.id}#${targets[source] || 'experiment-overview'}`
}

function priorityLabel(priority: string) {
  return t(`detail.analysisPriority.${priority}`)
}

function findingIcon(severity: string) {
  return severity === 'critical' || severity === 'warning' ? AlertTriangle : Info
}

function stageState(stageId: string) {
  const current = stageOrder[analysis.activeStage] ?? -1
  const target = stageOrder[stageId] ?? 0
  if (current > target) return 'complete'
  if (current === target) return 'active'
  return 'pending'
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat(locale.value, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

function resizeComposer() {
  const element = composer.value
  if (!element) return
  element.style.height = 'auto'
  element.style.height = `${Math.min(element.scrollHeight, 180)}px`
}

function scrollToBottom() {
  void nextTick(() => {
    if (thread.value) thread.value.scrollTop = thread.value.scrollHeight
  })
}

function startNewConversation() {
  openMenuId.value = ''
  sidebarOpen.value = false
  localError.value = ''
  input.value = ''
  router.push(`/experiments/${props.id}/analysis`)
  void nextTick(() => composer.value?.focus())
}

function openConversation(id: string) {
  sidebarOpen.value = false
  localError.value = ''
  router.push(`/experiments/${props.id}/analysis/${id}`)
}

async function sendMessage(contentOverride?: string) {
  const content = (contentOverride ?? input.value).trim()
  if (!content || !selectedProviderId.value || !canAnalyze.value || analysis.generating) return
  localError.value = ''
  pendingNewConversation = !conversationId.value
  input.value = ''
  resizeComposer()
  try {
    await analysis.send({
      conversation_id: conversationId.value || undefined,
      provider_config_id: selectedProviderId.value,
      content,
      language: locale.value,
      include: {
        dataset_preview: includeDatasetPreview.value,
        inference_summary: includeInferenceSummary.value,
        extended_logs: includeExtendedLogs.value,
      },
    })
  } catch {
    localError.value = t('detail.analysisError')
  }
}

function handleComposerKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault()
    void sendMessage()
  }
}

function beginRename(conversation: AgentChatConversation) {
  editingId.value = conversation.id
  editingTitle.value = conversation.title
  openMenuId.value = ''
}

async function saveRename() {
  const title = editingTitle.value.trim()
  if (!editingId.value || !title) return
  try {
    await analysis.renameConversation(editingId.value, title)
    editingId.value = ''
  } catch {
    localError.value = t('detail.analysisRenameFailed')
  }
}

async function togglePin(conversation: AgentChatConversation) {
  try {
    await analysis.togglePin(conversation)
    openMenuId.value = ''
  } catch {
    localError.value = t('detail.analysisPinFailed')
  }
}

async function removeConversation(conversation: AgentChatConversation) {
  if (!window.confirm(t('detail.analysisDeleteConfirm', { title: conversation.title }))) return
  try {
    await analysis.deleteConversation(conversation.id)
    openMenuId.value = ''
    if (conversationId.value === conversation.id)
      await router.replace(`/experiments/${props.id}/analysis`)
  } catch {
    localError.value = t('detail.analysisDeleteFailed')
  }
}

async function copyMessage(message: ExperimentAnalysisMessage) {
  try {
    await navigator.clipboard.writeText(message.content)
    copiedMessageId.value = message.id
    window.setTimeout(() => {
      if (copiedMessageId.value === message.id) copiedMessageId.value = ''
    }, 1600)
  } catch {
    localError.value = t('detail.analysisCopyFailed')
  }
}

function planNextRound(message: ExperimentAnalysisMessage) {
  if (!experiment.value || !reportFor(message)) return
  router.push({
    path: '/agent/design',
    query: {
      source_experiment_id: experiment.value.id,
      analysis_message_id: message.id,
    },
  })
}

watch(
  [enabledProviders, currentConversation],
  ([providers, conversation]) => {
    if (
      selectedProviderId.value &&
      providers.some((provider) => provider.id === selectedProviderId.value)
    )
      return
    const preferred = conversation?.last_provider_config_id
    selectedProviderId.value =
      providers.find((provider) => provider.id === preferred)?.id || providers[0]?.id || ''
  },
  { immediate: true },
)

watch(
  () => analysis.activeGenerationConversationId,
  (id) => {
    if (!id || !pendingNewConversation) return
    pendingNewConversation = false
    void router.replace(`/experiments/${props.id}/analysis/${id}`)
  },
)

watch(
  conversationId,
  async (id) => {
    localError.value = ''
    if (id) await analysis.loadConversation(id)
    scrollToBottom()
  },
  { immediate: true },
)

watch(search, (value) => {
  if (searchTimer) window.clearTimeout(searchTimer)
  searchTimer = window.setTimeout(() => void analysis.loadConversations(value.trim()), 240)
})

watch(
  () =>
    messages.value
      .map((message) => `${message.id}:${message.content.length}:${message.status}`)
      .join('|'),
  scrollToBottom,
)

watch(messages, (items) => markdownRenderer.prune(items.map((message) => message.id)))

onMounted(() => void analysis.loadConversations())
onActivated(() => void analysis.loadConversations(search.value.trim()))
onBeforeUnmount(() => {
  if (searchTimer) window.clearTimeout(searchTimer)
})
</script>

<template>
  <section class="experiment-analysis-shell">
    <button
      class="experiment-analysis-mobile-menu"
      type="button"
      :aria-label="t('detail.analysisOpenHistory')"
      @click="sidebarOpen = true"
    >
      <Menu aria-hidden="true" />
    </button>
    <div v-if="sidebarOpen" class="experiment-analysis-scrim" @click="sidebarOpen = false" />

    <aside
      class="experiment-analysis-sidebar"
      :class="{ 'experiment-analysis-sidebar-open': sidebarOpen }"
      :aria-label="t('detail.analysisHistory')"
    >
      <div class="experiment-analysis-sidebar-head">
        <button type="button" class="experiment-analysis-new" @click="startNewConversation">
          <MessageSquarePlus aria-hidden="true" />
          <span>{{ t('detail.analysisNew') }}</span>
        </button>
        <button
          type="button"
          class="experiment-analysis-sidebar-close"
          :aria-label="t('common.close')"
          @click="sidebarOpen = false"
        >
          <X aria-hidden="true" />
        </button>
      </div>
      <label class="experiment-analysis-search">
        <Search aria-hidden="true" />
        <input v-model="search" type="search" :placeholder="t('detail.analysisSearch')" />
      </label>
      <div class="experiment-analysis-history-list">
        <div v-if="analysis.historyError" class="experiment-analysis-history-error" role="alert">
          <AlertTriangle aria-hidden="true" />
          <p>{{ t('detail.analysisHistoryLoadFailed') }}</p>
          <button
            type="button"
            :disabled="analysis.historyLoading"
            @click="analysis.loadConversations(search.trim())"
          >
            {{ analysis.historyLoading ? t('common.loading') : t('detail.analysisRetryHistory') }}
          </button>
        </div>
        <p
          v-if="analysis.historyLoading && !analysis.conversations.length"
          class="experiment-analysis-history-state"
          >{{ t('common.loading') }}</p
        >
        <p
          v-else-if="!analysis.conversations.length && !analysis.historyError"
          class="experiment-analysis-history-state"
          >{{ t('detail.analysisHistoryEmpty') }}</p
        >
        <section
          v-for="group in groupedConversations"
          :key="group.key"
          class="experiment-analysis-history-group"
        >
          <h2>{{ group.label }}</h2>
          <div
            v-for="conversation in group.items"
            :key="conversation.id"
            class="experiment-analysis-history-item"
            :class="{
              'experiment-analysis-history-item-active': conversationId === conversation.id,
            }"
          >
            <form
              v-if="editingId === conversation.id"
              class="experiment-analysis-rename"
              @submit.prevent="saveRename"
            >
              <input
                v-model="editingTitle"
                maxlength="120"
                :aria-label="t('detail.analysisRename')"
                autofocus
              />
              <button type="submit" :aria-label="t('common.save')"
                ><Check aria-hidden="true"
              /></button>
              <button type="button" :aria-label="t('common.cancel')" @click="editingId = ''"
                ><X aria-hidden="true"
              /></button>
            </form>
            <template v-else>
              <button
                type="button"
                class="experiment-analysis-history-link"
                @click="openConversation(conversation.id)"
              >
                <span class="experiment-analysis-history-title"
                  ><Pin v-if="conversation.pinned" aria-hidden="true" />{{
                    conversation.title
                  }}</span
                >
                <span>{{ conversation.last_model || formatDate(conversation.updated_at) }}</span>
              </button>
              <button
                type="button"
                class="experiment-analysis-history-more"
                :aria-label="t('detail.analysisConversationActions')"
                :aria-expanded="openMenuId === conversation.id"
                @click="openMenuId = openMenuId === conversation.id ? '' : conversation.id"
              >
                <MoreHorizontal aria-hidden="true" />
              </button>
              <div v-if="openMenuId === conversation.id" class="experiment-analysis-history-menu">
                <button type="button" @click="togglePin(conversation)"
                  ><Pin aria-hidden="true" />{{
                    conversation.pinned ? t('detail.analysisUnpin') : t('detail.analysisPin')
                  }}</button
                >
                <button type="button" @click="beginRename(conversation)"
                  ><Pencil aria-hidden="true" />{{ t('detail.analysisRename') }}</button
                >
                <button type="button" class="danger" @click="removeConversation(conversation)"
                  ><Trash2 aria-hidden="true" />{{ t('common.delete') }}</button
                >
              </div>
            </template>
          </div>
        </section>
        <button
          v-if="analysis.conversations.length < analysis.total"
          type="button"
          class="experiment-analysis-load-more"
          :disabled="analysis.historyLoading"
          @click="analysis.loadMoreConversations()"
        >
          {{ analysis.historyLoading ? t('common.loading') : t('detail.analysisLoadMore') }}
        </button>
      </div>
    </aside>

    <main class="experiment-analysis-main">
      <header class="experiment-analysis-head">
        <div class="experiment-analysis-title">
          <p>{{ t('detail.aiEyebrow') }}</p>
          <div>
            <h1>{{ experiment ? experimentDisplayName(experiment, t) : t('common.loading') }}</h1>
            <span v-if="experiment" class="status-pill" :class="statusClass(experiment.status)">{{
              t(`status.${experiment.status}`)
            }}</span>
          </div>
        </div>
        <div class="experiment-analysis-head-actions">
          <label class="experiment-analysis-model-picker">
            <span>{{ t('detail.aiProvider') }}</span>
            <select
              v-model="selectedProviderId"
              :disabled="analysis.generating || !enabledProviders.length"
            >
              <option v-for="provider in enabledProviders" :key="provider.id" :value="provider.id"
                >{{ provider.display_name }} · {{ provider.model }}</option
              >
            </select>
          </label>
          <button
            type="button"
            class="experiment-analysis-toolbar-action"
            :class="{ 'experiment-analysis-toolbar-action-active': contextOpen }"
            :title="t('detail.analysisContext')"
            :aria-label="t('detail.analysisContext')"
            :aria-expanded="contextOpen"
            aria-controls="experiment-analysis-context"
            @click="contextOpen = true"
          >
            <Database aria-hidden="true" />
            <span class="experiment-analysis-action-label-full">{{
              t('detail.analysisContext')
            }}</span>
            <span class="experiment-analysis-action-label-compact">{{
              t('detail.analysisContextCompact')
            }}</span>
          </button>
          <RouterLink
            to="/account/ai-services"
            class="experiment-analysis-toolbar-action"
            :title="t('aiServices.openSettings')"
            :aria-label="t('aiServices.openSettings')"
          >
            <PlugZap aria-hidden="true" />
            <span>{{ t('aiServices.shortLabel') }}</span>
          </RouterLink>
        </div>
      </header>

      <div ref="thread" class="experiment-analysis-thread">
        <div
          v-if="!enabledProviders.length && !providersQuery.isLoading.value"
          class="experiment-analysis-empty"
        >
          <Bot aria-hidden="true" />
          <h2>{{ t('detail.aiNoProviders') }}</h2>
          <RouterLink to="/account/ai-services" class="primary-action compact-action">
            <PlugZap aria-hidden="true" />{{ t('aiServices.connect') }}
          </RouterLink>
        </div>
        <div v-else-if="experiment?.status === 'queued'" class="experiment-analysis-empty">
          <Clock3 aria-hidden="true" />
          <h2>{{ t('detail.analysisQueuedTitle') }}</h2>
          <p>{{ t('detail.analysisQueuedCopy') }}</p>
        </div>
        <div v-else-if="!conversationId && !analysis.generating" class="experiment-analysis-empty">
          <span class="experiment-analysis-empty-mark"><Sparkles aria-hidden="true" /></span>
          <p class="experiment-analysis-kicker">PepPlatform Diagnostics</p>
          <h2>{{ t('detail.analysisEmptyTitle') }}</h2>
          <div class="experiment-analysis-suggestions">
            <button
              v-for="suggestion in suggestions"
              :key="suggestion.title"
              type="button"
              @click="sendMessage(suggestion.prompt)"
            >
              <component :is="suggestion.icon" aria-hidden="true" />
              <span>{{ suggestion.title }}</span>
              <ArrowUpRight aria-hidden="true" />
            </button>
          </div>
        </div>
        <div v-else class="experiment-analysis-messages">
          <button
            v-if="analysis.hasMoreMessages[conversationId]"
            type="button"
            class="experiment-analysis-load-earlier"
            :disabled="analysis.messageLoading"
            @click="analysis.loadEarlierMessages(conversationId)"
          >
            {{ analysis.messageLoading ? t('common.loading') : t('detail.analysisLoadEarlier') }}
          </button>
          <article
            v-for="message in messages"
            :key="message.id"
            class="experiment-analysis-message"
            :class="`experiment-analysis-message-${message.role}`"
          >
            <div v-if="message.role === 'assistant'" class="experiment-analysis-assistant-mark"
              ><Sparkles aria-hidden="true"
            /></div>
            <div class="experiment-analysis-message-body">
              <div class="experiment-analysis-message-meta">
                <strong>{{
                  message.role === 'user' ? t('agent.chatYou') : providerLabel(message)
                }}</strong>
                <span v-if="message.role === 'assistant' && message.model">{{
                  message.model
                }}</span>
                <span>{{ formatDate(message.created_at) }}</span>
                <button
                  v-if="message.role === 'assistant' && message.content"
                  type="button"
                  :title="t('common.copy')"
                  :aria-label="t('common.copy')"
                  @click="copyMessage(message)"
                >
                  <Check v-if="copiedMessageId === message.id" aria-hidden="true" />
                  <Copy v-else aria-hidden="true" />
                </button>
              </div>

              <p v-if="message.role === 'user'" class="experiment-analysis-user-copy">{{
                message.content
              }}</p>

              <div
                v-else-if="
                  message.kind === 'report' && message.status === 'streaming' && !reportFor(message)
                "
                class="experiment-analysis-stage"
                aria-live="polite"
              >
                <div
                  v-for="item in stageItems"
                  :key="item.id"
                  :class="`experiment-analysis-stage-${stageState(item.id)}`"
                >
                  <CheckCircle2 v-if="stageState(item.id) === 'complete'" aria-hidden="true" />
                  <span v-else class="experiment-analysis-stage-dot" />
                  <span>{{ item.label }}</span>
                </div>
              </div>

              <section v-else-if="reportFor(message)" class="experiment-diagnostic-report">
                <div v-if="reportIsStale(message)" class="experiment-analysis-stale"
                  ><AlertTriangle aria-hidden="true" />{{ t('detail.analysisStale') }}</div
                >
                <header class="experiment-diagnostic-report-head">
                  <span class="config-chip">{{
                    reportFor(message)?.status === 'provisional'
                      ? t('detail.analysisProvisional')
                      : t('detail.analysisFinal')
                  }}</span>
                  <span v-if="message.context_manifest?.captured_at">{{
                    t('detail.analysisCapturedAt', {
                      date: formatDate(message.context_manifest.captured_at),
                    })
                  }}</span>
                </header>
                <p class="experiment-diagnostic-summary">{{ reportFor(message)?.summary }}</p>

                <section
                  v-if="reportFor(message)?.evidence.length"
                  class="experiment-diagnostic-section"
                >
                  <h2>{{ t('detail.analysisEvidence') }}</h2>
                  <div class="experiment-diagnostic-evidence-grid">
                    <article
                      v-for="item in reportFor(message)?.evidence"
                      :key="`${item.label}-${item.value}`"
                    >
                      <div
                        ><strong>{{ item.label }}</strong
                        ><span>{{ item.value }}</span></div
                      >
                      <p>{{ item.interpretation }}</p>
                      <RouterLink :to="sourceTarget(item.source_section)"
                        >{{ sourceLabel(item.source_section) }}<ArrowUpRight aria-hidden="true"
                      /></RouterLink>
                    </article>
                  </div>
                </section>

                <section
                  v-if="reportFor(message)?.findings.length"
                  class="experiment-diagnostic-section"
                >
                  <h2>{{ t('detail.analysisFindings') }}</h2>
                  <div class="experiment-diagnostic-findings">
                    <article
                      v-for="item in reportFor(message)?.findings"
                      :key="item.title"
                      :class="`experiment-diagnostic-finding-${item.severity}`"
                    >
                      <component :is="findingIcon(item.severity)" aria-hidden="true" />
                      <div
                        ><h3>{{ item.title }}</h3
                        ><p>{{ item.detail }}</p
                        ><div
                          ><RouterLink
                            v-for="source in item.source_sections"
                            :key="source"
                            :to="sourceTarget(source)"
                            >{{ sourceLabel(source) }}</RouterLink
                          ></div
                        ></div
                      >
                    </article>
                  </div>
                </section>

                <section
                  v-if="reportFor(message)?.recommendations.length"
                  class="experiment-diagnostic-section"
                >
                  <h2>{{ t('detail.analysisRecommendations') }}</h2>
                  <div class="experiment-diagnostic-recommendations">
                    <article
                      v-for="recommendation in reportFor(message)?.recommendations"
                      :key="recommendation.id"
                    >
                      <div class="experiment-diagnostic-recommendation-head">
                        <div
                          ><span>{{ priorityLabel(recommendation.priority) }}</span
                          ><h3>{{ recommendation.title }}</h3></div
                        >
                      </div>
                      <p>{{ recommendation.rationale }}</p>
                      <div
                        v-if="recommendation.success_criteria?.length"
                        class="experiment-diagnostic-success-criteria"
                      >
                        <strong>{{ t('detail.analysisSuccessCriteria') }}</strong>
                        <ul
                          ><li
                            v-for="criterion in recommendation.success_criteria"
                            :key="criterion"
                            >{{ criterion }}</li
                          ></ul
                        >
                      </div>
                    </article>
                  </div>
                </section>

                <details
                  v-if="reportFor(message)?.limitations.length"
                  class="experiment-diagnostic-limitations"
                >
                  <summary>{{ t('detail.analysisLimitations') }}</summary>
                  <ul
                    ><li v-for="item in reportFor(message)?.limitations" :key="item">{{
                      item
                    }}</li></ul
                  >
                </details>

                <button
                  type="button"
                  class="primary-action compact-action experiment-diagnostic-plan-action"
                  @click="planNextRound(message)"
                >
                  <FlaskConical aria-hidden="true" />{{ t('detail.analysisPlanNextRound') }}
                </button>
              </section>

              <div
                v-else-if="message.content"
                class="experiment-analysis-markdown ai-markdown"
                v-html="renderMarkdown(message)"
              />
              <div
                v-else-if="message.status === 'streaming'"
                class="experiment-analysis-caret"
                :aria-label="t('detail.aiGenerating')"
                ><span
              /></div>
              <p v-if="message.status === 'error'" class="experiment-analysis-message-error">{{
                message.error || t('detail.analysisError')
              }}</p>
            </div>
          </article>
          <p
            v-if="analysis.messageLoading && !messages.length"
            class="experiment-analysis-history-state"
            >{{ t('common.loading') }}</p
          >
        </div>
      </div>

      <div v-if="anotherConversationGenerating" class="experiment-analysis-background-banner">
        <span>{{ t('detail.analysisBackgroundGenerating') }}</span>
        <button type="button" @click="analysis.stopGeneration()"
          ><Square aria-hidden="true" />{{ t('agent.chatStop') }}</button
        >
      </div>
      <p v-if="visibleError" class="experiment-analysis-error" role="alert">{{ visibleError }}</p>

      <footer class="experiment-analysis-composer-wrap">
        <div
          class="experiment-analysis-composer"
          :class="{ 'experiment-analysis-composer-active': input.trim() }"
        >
          <textarea
            ref="composer"
            v-model="input"
            rows="1"
            :placeholder="
              anotherConversationGenerating
                ? t('agent.chatWaitForAnswer')
                : t('detail.analysisPlaceholder')
            "
            :disabled="analysis.generating || !enabledProviders.length || !canAnalyze"
            @input="resizeComposer"
            @keydown="handleComposerKeydown"
          />
          <button
            type="button"
            class="experiment-analysis-context-shortcut"
            :class="{
              active: includeDatasetPreview || includeInferenceSummary || includeExtendedLogs,
            }"
            :title="t('detail.analysisContext')"
            :aria-label="t('detail.analysisContext')"
            @click="contextOpen = true"
            ><Database aria-hidden="true"
          /></button>
          <button
            v-if="isCurrentGenerating"
            type="button"
            class="experiment-analysis-send"
            :aria-label="t('agent.chatStop')"
            @click="analysis.stopGeneration()"
            ><Square aria-hidden="true"
          /></button>
          <button
            v-else
            type="button"
            class="experiment-analysis-send"
            :disabled="!canSend"
            :aria-label="t('agent.chatSend')"
            @click="sendMessage()"
            ><Send aria-hidden="true"
          /></button>
        </div>
        <p>{{ t('detail.analysisDisclaimer') }}</p>
      </footer>
    </main>

    <div
      v-if="contextOpen"
      class="experiment-analysis-context-scrim"
      @click="contextOpen = false"
    />
    <aside
      id="experiment-analysis-context"
      class="experiment-analysis-context"
      :class="{ 'experiment-analysis-context-open': contextOpen }"
      :aria-hidden="!contextOpen"
      :aria-label="t('detail.analysisContext')"
    >
      <header
        ><div
          ><Database aria-hidden="true" /><h2>{{ t('detail.analysisContext') }}</h2></div
        ><button type="button" :aria-label="t('common.close')" @click="contextOpen = false"
          ><X aria-hidden="true" /></button
      ></header>
      <div class="experiment-analysis-context-body">
        <section>
          <h3>{{ t('detail.analysisRequiredContext') }}</h3>
          <div class="experiment-analysis-required-context"
            ><div v-for="item in requiredContextSections" :key="item"
              ><Check aria-hidden="true" /><span>{{ item }}</span></div
            ></div
          >
        </section>
        <section>
          <h3>{{ t('detail.analysisOptionalContext') }}</h3>
          <label class="experiment-analysis-context-toggle"
            ><span
              ><strong>{{ t('detail.aiIncludeDatasetPreview') }}</strong
              ><small>{{ t('detail.analysisDatasetPreviewHint') }}</small></span
            ><input v-model="includeDatasetPreview" type="checkbox"
          /></label>
          <label class="experiment-analysis-context-toggle"
            ><span
              ><strong>{{ t('detail.aiIncludeInferenceSummary') }}</strong
              ><small>{{ t('detail.analysisInferenceHint') }}</small></span
            ><input v-model="includeInferenceSummary" type="checkbox"
          /></label>
          <label class="experiment-analysis-context-toggle"
            ><span
              ><strong>{{ t('detail.aiIncludeExtendedLogs') }}</strong
              ><small>{{ t('detail.analysisExtendedLogsHint') }}</small></span
            ><input v-model="includeExtendedLogs" type="checkbox"
          /></label>
        </section>
        <section v-if="latestAssistantMessage?.context_manifest">
          <h3>{{ t('detail.analysisLatestSnapshot') }}</h3>
          <dl class="experiment-analysis-snapshot">
            <div
              ><dt>{{ t('detail.analysisCaptured') }}</dt
              ><dd>{{
                latestAssistantMessage.context_manifest.captured_at
                  ? formatDate(latestAssistantMessage.context_manifest.captured_at)
                  : '—'
              }}</dd></div
            >
            <div
              ><dt>{{ t('detail.analysisContextSize') }}</dt
              ><dd>{{ latestAssistantMessage.context_manifest.context_chars || 0 }}</dd></div
            >
            <div
              ><dt>{{ t('detail.analysisContextHash') }}</dt
              ><dd>{{ latestAssistantMessage.context_manifest.context_hash || '—' }}</dd></div
            >
          </dl>
        </section>
        <section class="experiment-analysis-redactions"
          ><h3>{{ t('detail.analysisProtectedData') }}</h3
          ><p>{{ t('detail.analysisProtectedDataCopy') }}</p></section
        >
      </div>
    </aside>
  </section>
</template>
