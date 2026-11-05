<script setup lang="ts">
import { computed, nextTick, onActivated, onBeforeUnmount, ref, watch } from 'vue'
import { useQuery } from '@tanstack/vue-query'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import {
  ArrowUp,
  Bot,
  Check,
  ExternalLink,
  Globe2,
  Menu,
  MessageSquarePlus,
  MoreHorizontal,
  Pencil,
  Pin,
  Search,
  PlugZap,
  Square,
  Trash2,
  X,
} from '@lucide/vue'
import { api } from '../api/client'
import { useWorkspaceAccess } from '../workspaceAccess'
import { useLoginDraft } from '../loginDraft'
import { createAgentChatMarkdownRenderer } from '../agentChatMarkdown'
import { localizedApiError } from '../errorLocalization'
import { formatLocalizedDate } from '../localization'
import { useAgentChatStore } from '../stores/agentChat'
import type { AgentChatConversation, AgentChatMessage, AgentWebSearchSource } from '../types'

const { isGuest, canUseWorkspace, requireLogin } = useWorkspaceAccess()

const { t, te, locale } = useI18n()
const route = useRoute()
const router = useRouter()
const chat = useAgentChatStore()
const providersQuery = useQuery({
  queryKey: ['ai-providers'],
  queryFn: api.aiProviders,
  enabled: canUseWorkspace,
})
const webSearchStatusQuery = useQuery({
  enabled: canUseWorkspace,
  queryKey: ['agent-web-search-status'],
  queryFn: api.agentWebSearchStatus,
  retry: false,
  staleTime: 5 * 60 * 1000,
})
const markdownRenderer = createAgentChatMarkdownRenderer()
const input = ref('')
useLoginDraft(
  'agent-chat',
  () => ({ input: input.value }),
  (saved) => {
    input.value = saved.input
  },
)
const selectedProviderId = ref('')
const webSearchEnabled = ref(false)
const search = ref('')
const sidebarOpen = ref(false)
const openMenuId = ref('')
const editingId = ref('')
const editingTitle = ref('')
const localError = ref('')
const messageScroll = ref<HTMLElement | null>(null)
const composer = ref<HTMLTextAreaElement | null>(null)
const completionAnnouncement = ref('')
const preservingEarlierScroll = ref(false)
const sourceDialog = ref<HTMLDialogElement | null>(null)
const openSourceMessageId = ref('')
const highlightedSourceIndex = ref<number | null>(null)
const sourceReturnFocus = ref<HTMLElement | null>(null)
const failedFavicons = ref<Set<string>>(new Set())
let searchTimer: ReturnType<typeof window.setTimeout> | undefined
let pendingNewConversation = false

const conversationId = computed(() => String(route.params.conversationId || ''))
const enabledProviders = computed(() =>
  (providersQuery.data.value || []).filter((provider) => provider.enabled),
)
const currentConversation = computed(() =>
  chat.conversations.find((item) => item.id === conversationId.value),
)
const messages = computed(() => chat.messagesByConversation[conversationId.value] || [])
const activeSourceMessage = computed(() =>
  messages.value.find((message) => message.id === openSourceMessageId.value),
)
const activeSources = computed(() => activeSourceMessage.value?.web_search?.sources || [])
const webSearchStatus = computed(() => webSearchStatusQuery.data.value)
const webSearchAvailable = computed(() => Boolean(webSearchStatus.value?.available))
const isCurrentGenerating = computed(
  () => chat.generating && chat.activeGenerationConversationId === conversationId.value,
)
const anotherConversationGenerating = computed(
  () => chat.generating && chat.activeGenerationConversationId !== conversationId.value,
)
const canSend = computed(
  () =>
    isGuest.value || Boolean(input.value.trim() && selectedProviderId.value && !chat.generating),
)
const webSearchReasonKeys = new Set([
  'not_configured',
  'disabled',
  'quota_exhausted',
  'paid_plan_rejected',
  'paygo_blocked',
  'rate_limited',
  'invalid_key',
  'usage_unverified',
  'usage_unavailable',
  'provider_unreachable',
  'provider_timeout',
  'provider_error',
  'no_results',
  'empty_query',
])

const webSearchButtonTitle = computed(() => {
  const status = webSearchStatus.value
  if (!status?.available) return webSearchReason(status?.reason)
  return t('agent.chatWebSearchHint')
})

const groupedConversations = computed(() => {
  const now = new Date()
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const day = 24 * 60 * 60 * 1000
  const groups = new Map<string, AgentChatConversation[]>()
  const append = (key: string, item: AgentChatConversation) => {
    groups.set(key, [...(groups.get(key) || []), item])
  }
  chat.conversations.forEach((item) => {
    if (item.pinned) return append('pinned', item)
    const timestamp = new Date(item.updated_at).getTime()
    if (timestamp >= today) append('today', item)
    else if (timestamp >= today - day) append('yesterday', item)
    else if (timestamp >= today - day * 7) append('week', item)
    else append('older', item)
  })
  return ['pinned', 'today', 'yesterday', 'week', 'older']
    .filter((key) => groups.has(key))
    .map((key) => ({
      key,
      label: t(`agent.chatHistoryGroups.${key}`),
      items: groups.get(key) || [],
    }))
})

const suggestions = computed(() => [
  t('agent.chatSuggestionOne'),
  t('agent.chatSuggestionTwo'),
  t('agent.chatSuggestionThree'),
])

function renderMarkdown(message: AgentChatMessage) {
  return markdownRenderer.render(message)
}

function webSearchReason(reason?: string) {
  const normalized = String(reason || 'unavailable')
  const key = webSearchReasonKeys.has(normalized) ? normalized : 'unavailable'
  return t(`agent.chatWebReason.${key}`)
}

function messageWebSearchStatus(message: AgentChatMessage) {
  if (message.id === chat.activeAssistantMessageId) {
    if (chat.activeSearchStatus === 'checking') return t('agent.chatWebSearchChecking')
    if (chat.activeSearchStatus === 'searching') return t('agent.chatWebSearching')
  }
  const webSearch = message.web_search
  if (!webSearch?.requested) return ''
  if (webSearch.status === 'complete') return ''
  return t('agent.chatWebSearchUnavailable', { reason: webSearchReason(webSearch.reason) })
}

function messageWebSearchUnavailable(message: AgentChatMessage) {
  return Boolean(message.web_search?.requested && message.web_search.status !== 'complete')
}

function providerLabel(message: AgentChatMessage) {
  const provider = enabledProviders.value.find((item) => item.id === message.provider_config_id)
  return provider?.display_name || message.provider || t('agent.chatAi')
}

function sourcePreview(message: AgentChatMessage) {
  return (message.web_search?.sources || []).slice(0, 3)
}

function sourceInitial(source: AgentWebSearchSource) {
  return (source.domain || source.title || '?').trim().charAt(0).toUpperCase() || '?'
}

function faviconKey(source: AgentWebSearchSource) {
  return `${source.domain}:${source.favicon || ''}`
}

function canShowFavicon(source: AgentWebSearchSource) {
  return Boolean(source.favicon && !failedFavicons.value.has(faviconKey(source)))
}

function markFaviconFailed(source: AgentWebSearchSource) {
  failedFavicons.value = new Set([...failedFavicons.value, faviconKey(source)])
}

async function openSources(
  messageId: string,
  sourceIndex?: number,
  returnFocus?: HTMLElement | null,
) {
  if (returnFocus && !sourceDialog.value?.open) sourceReturnFocus.value = returnFocus
  openSourceMessageId.value = messageId
  highlightedSourceIndex.value = sourceIndex ?? null
  await nextTick()
  const dialog = sourceDialog.value
  if (!dialog) return
  if (!dialog.open) dialog.showModal()
  if (sourceIndex === undefined) return
  await nextTick()
  const row = dialog.querySelector<HTMLElement>(`[data-source-index="${sourceIndex}"]`)
  row?.scrollIntoView({ block: 'center' })
  row?.focus({ preventScroll: true })
}

function openSourcesFromTrigger(messageId: string, event: MouseEvent) {
  void openSources(messageId, undefined, event.currentTarget as HTMLElement)
}

function handleMessageClick(event: MouseEvent) {
  const element =
    event.target instanceof Element
      ? event.target.closest<HTMLAnchorElement>('.agent-chat-citation')
      : null
  if (!element) return
  const messageId = element.dataset.messageId || ''
  const sourceIndex = Number(element.dataset.sourceIndex)
  if (!messageId || !Number.isInteger(sourceIndex)) return
  event.preventDefault()
  void openSources(messageId, sourceIndex, element)
}

function closeSources() {
  sourceDialog.value?.close()
}

function handleSourceDialogClose() {
  const returnFocus = sourceReturnFocus.value
  openSourceMessageId.value = ''
  highlightedSourceIndex.value = null
  sourceReturnFocus.value = null
  nextTick(() => returnFocus?.focus())
}

function formatTime(value: string) {
  return formatLocalizedDate(value, locale.value, {
    month: 'short',
    day: 'numeric',
  })
}

function chatErrorCodeText(code: string) {
  const key = `agent.chatErrors.${code}`
  return te(key) ? t(key) : t('agent.genericError')
}

function chatActionError(error: unknown, fallbackCode = 'action_failed') {
  if (error instanceof Error) {
    const key = `agent.chatErrors.${error.message}`
    if (te(key)) return t(key)
  }
  return localizedApiError(error, t, te, `agent.chatErrors.${fallbackCode}`)
}

function messageError(message: AgentChatMessage) {
  return message.finish_reason === 'interrupted'
    ? t('agent.chatErrors.ai_stream_interrupted')
    : t('agent.chatErrors.ai_provider_error')
}

const visibleChatError = computed(() => {
  if (localError.value) return localError.value
  if (chat.streamError) return chatErrorCodeText(chat.streamError)
  if (chat.historyError) return chatErrorCodeText(chat.historyError)
  return ''
})

async function scrollToBottom() {
  await nextTick()
  if (messageScroll.value) messageScroll.value.scrollTop = messageScroll.value.scrollHeight
}

async function loadEarlierMessages() {
  if (!conversationId.value || !messageScroll.value) return
  const previousHeight = messageScroll.value.scrollHeight
  const previousTop = messageScroll.value.scrollTop
  preservingEarlierScroll.value = true
  try {
    await chat.loadEarlierMessages(conversationId.value)
    await nextTick()
    if (messageScroll.value) {
      messageScroll.value.scrollTop =
        previousTop + messageScroll.value.scrollHeight - previousHeight
    }
  } finally {
    preservingEarlierScroll.value = false
  }
}

function resizeComposer() {
  if (!composer.value) return
  composer.value.style.height = 'auto'
  composer.value.style.height = `${Math.min(composer.value.scrollHeight, 200)}px`
}

function startNewConversation() {
  sidebarOpen.value = false
  openMenuId.value = ''
  localError.value = ''
  void router.push('/agent/chat')
  nextTick(() => composer.value?.focus())
}

function openConversation(id: string) {
  sidebarOpen.value = false
  openMenuId.value = ''
  void router.push(`/agent/chat/${id}`)
}

function useSuggestion(value: string) {
  input.value = value
  nextTick(() => {
    resizeComposer()
    composer.value?.focus()
  })
}

async function sendMessage() {
  if (!requireLogin()) return
  const content = input.value.trim()
  if (!content || !canSend.value) return
  localError.value = ''
  chat.streamError = ''
  input.value = ''
  pendingNewConversation = !conversationId.value
  const useWebSearch = webSearchEnabled.value && webSearchAvailable.value
  webSearchEnabled.value = false
  await nextTick()
  resizeComposer()
  void chat.send({
    conversation_id: conversationId.value || undefined,
    provider_config_id: selectedProviderId.value,
    content,
    language: locale.value,
    web_search: useWebSearch,
  })
}

function handleComposerKeydown(event: KeyboardEvent) {
  if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return
  event.preventDefault()
  void sendMessage()
}

function beginRename(conversation: AgentChatConversation) {
  openMenuId.value = ''
  editingId.value = conversation.id
  editingTitle.value = conversation.title
}

async function saveRename() {
  const title = editingTitle.value.trim()
  if (!editingId.value || !title) return
  try {
    await chat.renameConversation(editingId.value, title)
    editingId.value = ''
  } catch (error) {
    localError.value = chatActionError(error, 'rename_failed')
  }
}

async function togglePin(conversation: AgentChatConversation) {
  openMenuId.value = ''
  try {
    await chat.togglePin(conversation)
  } catch (error) {
    localError.value = chatActionError(error, 'pin_failed')
  }
}

async function removeConversation(conversation: AgentChatConversation) {
  openMenuId.value = ''
  if (!window.confirm(t('agent.chatDeleteConfirm', { title: conversation.title }))) return
  try {
    await chat.deleteConversation(conversation.id)
    if (conversationId.value === conversation.id) await router.replace('/agent/chat')
  } catch (error) {
    localError.value = chatActionError(error, 'delete_failed')
  }
}

watch(
  conversationId,
  async (id) => {
    localError.value = ''
    openMenuId.value = ''
    if (sourceDialog.value?.open) closeSources()
    if (!id || !canUseWorkspace.value) {
      await nextTick()
      composer.value?.focus()
      return
    }
    try {
      await chat.loadConversation(id)
    } catch (error) {
      localError.value = chatActionError(error, 'conversation_load_failed')
    }
  },
  { immediate: true },
)

watch(webSearchAvailable, (available) => {
  if (!available) webSearchEnabled.value = false
})

watch(
  () => chat.activeGenerationConversationId,
  (id) => {
    if (!id || !pendingNewConversation || route.name !== 'agent-chat') return
    pendingNewConversation = false
    void router.replace(`/agent/chat/${id}`)
  },
)

watch(
  [enabledProviders, currentConversation],
  ([providers, conversation]) => {
    if (!providers.length) {
      selectedProviderId.value = ''
      return
    }
    const preferred = conversation?.last_provider_config_id
    if (preferred && providers.some((provider) => provider.id === preferred)) {
      selectedProviderId.value = preferred
    } else if (!providers.some((provider) => provider.id === selectedProviderId.value)) {
      selectedProviderId.value = providers[0].id
    }
  },
  { immediate: true },
)

watch(search, (value) => {
  if (searchTimer) window.clearTimeout(searchTimer)
  searchTimer = window.setTimeout(() => {
    if (canUseWorkspace.value) void chat.loadConversations(value.trim())
  }, 300)
})

watch(
  () =>
    messages.value
      .map((message) => `${message.id}:${message.content.length}:${message.status}`)
      .join('|'),
  () => {
    if (!preservingEarlierScroll.value) void scrollToBottom()
  },
)

watch(messages, (items) => markdownRenderer.prune(items.map((message) => message.id)))

watch(
  () => chat.generating,
  (generating, wasGenerating) => {
    if (!generating && wasGenerating) {
      completionAnnouncement.value = chat.streamError
        ? chatErrorCodeText(chat.streamError)
        : t('agent.chatAnswerFinished')
      void webSearchStatusQuery.refetch()
    }
  },
)

watch(input, () => nextTick(resizeComposer))

onActivated(() => {
  if (!canUseWorkspace.value) return
  void chat.loadConversations(search.value.trim())
  void webSearchStatusQuery.refetch()
})

onBeforeUnmount(() => {
  if (searchTimer) window.clearTimeout(searchTimer)
})
</script>

<template>
  <section class="agent-chat-shell">
    <button
      class="agent-chat-mobile-menu"
      type="button"
      :aria-label="t('agent.chatOpenHistory')"
      @click="sidebarOpen = true"
    >
      <Menu aria-hidden="true" />
    </button>

    <div v-if="sidebarOpen" class="agent-chat-sidebar-scrim" @click="sidebarOpen = false" />
    <aside
      class="agent-chat-sidebar"
      :class="{ 'agent-chat-sidebar-open': sidebarOpen }"
      :aria-label="t('agent.chatHistory')"
    >
      <div class="agent-chat-sidebar-head">
        <button type="button" class="agent-chat-new" @click="startNewConversation">
          <MessageSquarePlus aria-hidden="true" />
          <span>{{ t('agent.chatNew') }}</span>
        </button>
        <button
          class="agent-chat-sidebar-close"
          type="button"
          :aria-label="t('common.close')"
          @click="sidebarOpen = false"
        >
          <X aria-hidden="true" />
        </button>
      </div>
      <label class="agent-chat-search">
        <Search aria-hidden="true" />
        <input v-model="search" type="search" :placeholder="t('agent.chatSearch')" />
      </label>

      <div class="agent-chat-history-list">
        <p
          v-if="chat.historyLoading && !chat.conversations.length"
          class="agent-chat-history-state"
          >{{ t('common.loading') }}</p
        >
        <p v-else-if="!chat.conversations.length" class="agent-chat-history-state">{{
          t('agent.chatHistoryEmpty')
        }}</p>
        <section
          v-for="group in groupedConversations"
          :key="group.key"
          class="agent-chat-history-group"
        >
          <h2>{{ group.label }}</h2>
          <div
            v-for="conversation in group.items"
            :key="conversation.id"
            class="agent-chat-history-item"
            :class="{ 'agent-chat-history-item-active': conversationId === conversation.id }"
          >
            <form
              v-if="editingId === conversation.id"
              class="agent-chat-rename"
              @submit.prevent="saveRename"
            >
              <input
                v-model="editingTitle"
                maxlength="120"
                :aria-label="t('agent.chatRename')"
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
                class="agent-chat-history-link"
                @click="openConversation(conversation.id)"
              >
                <span class="agent-chat-history-title"
                  ><Pin v-if="conversation.pinned" aria-hidden="true" />{{
                    conversation.title
                  }}</span
                >
                <span>{{ conversation.last_model || formatTime(conversation.updated_at) }}</span>
              </button>
              <button
                type="button"
                class="agent-chat-history-more"
                :aria-label="t('agent.chatConversationActions')"
                :aria-expanded="openMenuId === conversation.id"
                @click="openMenuId = openMenuId === conversation.id ? '' : conversation.id"
              >
                <MoreHorizontal aria-hidden="true" />
              </button>
              <div v-if="openMenuId === conversation.id" class="agent-chat-history-menu">
                <button type="button" @click="togglePin(conversation)"
                  ><Pin aria-hidden="true" />{{
                    conversation.pinned ? t('agent.chatUnpin') : t('agent.chatPin')
                  }}</button
                >
                <button type="button" @click="beginRename(conversation)"
                  ><Pencil aria-hidden="true" />{{ t('agent.chatRename') }}</button
                >
                <button
                  type="button"
                  class="agent-chat-history-delete"
                  @click="removeConversation(conversation)"
                  ><Trash2 aria-hidden="true" />{{ t('common.delete') }}</button
                >
              </div>
            </template>
          </div>
        </section>
        <button
          v-if="chat.conversations.length < chat.total"
          type="button"
          class="agent-chat-load-more"
          :disabled="chat.historyLoading"
          @click="chat.loadMoreConversations()"
        >
          {{ chat.historyLoading ? t('common.loading') : t('agent.chatLoadMore') }}
        </button>
      </div>
    </aside>

    <main class="agent-chat-main">
      <header class="agent-chat-main-head">
        <div>
          <p class="agent-chat-kicker">PepPlatform Agent</p>
          <h1>{{ currentConversation?.title || t('agent.chatNewTitle') }}</h1>
        </div>
        <label class="agent-chat-model-picker">
          <span>{{ t('agent.chatModel') }}</span>
          <select
            v-model="selectedProviderId"
            :disabled="chat.generating || (!isGuest && !enabledProviders.length)"
          >
            <option v-for="provider in enabledProviders" :key="provider.id" :value="provider.id">
              {{ provider.display_name }} · {{ provider.model }}
            </option>
          </select>
        </label>
      </header>

      <div ref="messageScroll" class="agent-chat-thread">
        <div
          v-if="!isGuest && !enabledProviders.length && !providersQuery.isLoading.value"
          class="agent-chat-empty agent-chat-no-provider"
        >
          <Bot aria-hidden="true" />
          <h2>{{ t('agent.noProviders') }}</h2>
          <p>{{ t('agent.chatNeedProvider') }}</p>
          <RouterLink to="/account/ai-services" class="primary-action compact-action">
            <PlugZap aria-hidden="true" />{{ t('aiServices.connect') }}
          </RouterLink>
        </div>

        <div v-else-if="!conversationId && !chat.generating" class="agent-chat-empty">
          <span class="agent-chat-orbit"><Bot aria-hidden="true" /></span>
          <p class="agent-chat-kicker">PepPlatform AI</p>
          <h2>{{ t('agent.chatEmptyTitle') }}</h2>
          <p>{{ t('agent.chatHint') }}</p>
          <div class="agent-chat-suggestions">
            <button
              v-for="suggestion in suggestions"
              :key="suggestion"
              type="button"
              @click="useSuggestion(suggestion)"
            >
              {{ suggestion }}
              <ArrowUp aria-hidden="true" />
            </button>
          </div>
        </div>

        <div v-else class="agent-chat-messages" @click="handleMessageClick">
          <button
            v-if="chat.hasMoreMessages[conversationId]"
            type="button"
            class="agent-chat-load-earlier"
            :disabled="chat.messageLoading"
            @click="loadEarlierMessages"
          >
            {{ chat.messageLoading ? t('common.loading') : t('agent.chatLoadEarlier') }}
          </button>
          <article
            v-for="message in messages"
            :key="message.id"
            class="agent-chat-message"
            :class="`agent-chat-message-${message.role}`"
          >
            <div v-if="message.role === 'assistant'" class="agent-chat-assistant-mark">
              <Bot aria-hidden="true" />
            </div>
            <div class="agent-chat-message-body">
              <div class="agent-chat-message-meta">
                <strong>{{
                  message.role === 'user' ? t('agent.chatYou') : providerLabel(message)
                }}</strong>
                <span v-if="message.role === 'assistant' && message.model">{{
                  message.model
                }}</span>
                <span v-if="message.status === 'stopped'">{{ t('agent.chatStopped') }}</span>
              </div>
              <div
                v-if="message.role === 'assistant' && messageWebSearchStatus(message)"
                class="agent-chat-web-status"
                :class="{
                  'agent-chat-web-status-unavailable': messageWebSearchUnavailable(message),
                }"
              >
                <Globe2 aria-hidden="true" />
                <span>{{ messageWebSearchStatus(message) }}</span>
              </div>
              <div
                v-if="message.role === 'assistant' && message.content"
                class="agent-chat-markdown ai-markdown"
                v-html="renderMarkdown(message)"
              />
              <p v-else-if="message.role === 'user'" class="agent-chat-user-copy">{{
                message.content
              }}</p>
              <div
                v-else-if="message.status === 'streaming'"
                class="agent-chat-caret"
                :aria-label="t('agent.chatGenerating')"
                ><span
              /></div>
              <button
                v-if="message.role === 'assistant' && message.web_search?.sources?.length"
                type="button"
                class="agent-chat-source-trigger"
                :aria-label="
                  t('agent.chatWebSourceCount', { count: message.web_search.sources.length })
                "
                @click="openSourcesFromTrigger(message.id, $event)"
              >
                <span class="agent-chat-source-stack" aria-hidden="true">
                  <span
                    v-for="source in sourcePreview(message)"
                    :key="`${message.id}-preview-${source.index}`"
                    class="agent-chat-source-favicon"
                  >
                    <img
                      v-if="canShowFavicon(source)"
                      :src="source.favicon"
                      alt=""
                      loading="lazy"
                      referrerpolicy="no-referrer"
                      @error="markFaviconFailed(source)"
                    />
                    <span v-else>{{ sourceInitial(source) }}</span>
                  </span>
                </span>
                <span>{{ t('agent.chatWebSources') }}</span>
              </button>
              <p
                v-if="message.status === 'error'"
                class="agent-chat-message-error"
                :class="{ 'agent-chat-message-warning': message.finish_reason === 'interrupted' }"
              >
                {{ messageError(message) }}
              </p>
            </div>
          </article>
          <div v-if="chat.messageLoading && !messages.length" class="agent-chat-history-state">{{
            t('common.loading')
          }}</div>
        </div>
      </div>

      <div v-if="anotherConversationGenerating" class="agent-chat-background-banner">
        <span>{{ t('agent.chatBackgroundGenerating') }}</span>
        <button type="button" @click="chat.stopGeneration()"
          ><Square aria-hidden="true" />{{ t('agent.chatStop') }}</button
        >
      </div>
      <p v-if="visibleChatError" class="agent-chat-error" role="alert">{{ visibleChatError }}</p>

      <footer class="agent-chat-composer-wrap">
        <div class="agent-chat-composer" :class="{ 'agent-chat-composer-active': input.trim() }">
          <div class="agent-chat-composer-main">
            <textarea
              ref="composer"
              v-model="input"
              rows="1"
              :placeholder="
                anotherConversationGenerating
                  ? t('agent.chatWaitForAnswer')
                  : t('agent.chatPlaceholder')
              "
              :disabled="chat.generating || (!isGuest && !enabledProviders.length)"
              @keydown="handleComposerKeydown"
            />
            <div class="agent-chat-tools">
              <button
                type="button"
                class="agent-chat-tool"
                :class="{ 'agent-chat-tool-active': webSearchEnabled }"
                :aria-pressed="webSearchEnabled"
                :title="webSearchButtonTitle"
                :disabled="
                  chat.generating || webSearchStatusQuery.isLoading.value || !webSearchAvailable
                "
                @click="webSearchEnabled = !webSearchEnabled"
              >
                <Globe2 aria-hidden="true" />
                <span>{{ t('agent.chatWebSearch') }}</span>
              </button>
              <span
                v-if="!webSearchAvailable && !webSearchStatusQuery.isLoading.value"
                class="agent-chat-tool-unavailable"
              >
                {{ webSearchReason(webSearchStatus?.reason) }}
              </span>
            </div>
          </div>
          <button
            v-if="isCurrentGenerating"
            type="button"
            class="agent-chat-send agent-chat-stop"
            :aria-label="t('agent.chatStop')"
            @click="chat.stopGeneration()"
          >
            <Square aria-hidden="true" />
          </button>
          <button
            v-else
            type="button"
            class="agent-chat-send"
            :disabled="!canSend"
            :aria-label="t('agent.chatSend')"
            @click="sendMessage"
          >
            <ArrowUp aria-hidden="true" />
          </button>
        </div>
        <p>{{ t('agent.chatDisclaimer') }}</p>
      </footer>
      <p class="sr-only" aria-live="polite">{{ completionAnnouncement }}</p>
    </main>
  </section>

  <Teleport to="body">
    <dialog
      id="agent-chat-source-dialog"
      ref="sourceDialog"
      class="agent-chat-source-dialog"
      aria-labelledby="agent-chat-source-dialog-title"
      @click.self="closeSources"
      @close="handleSourceDialogClose"
    >
      <div class="agent-chat-source-panel">
        <header>
          <div>
            <h2 id="agent-chat-source-dialog-title">{{ t('agent.chatWebSources') }}</h2>
            <span>{{ t('agent.chatWebSourceCount', { count: activeSources.length }) }}</span>
          </div>
          <button type="button" :aria-label="t('common.close')" @click="closeSources">
            <X aria-hidden="true" />
          </button>
        </header>
        <ol class="agent-chat-source-list">
          <li
            v-for="source in activeSources"
            :key="`${openSourceMessageId}-${source.index}`"
            :data-source-index="source.index"
            :class="{ 'agent-chat-source-highlighted': highlightedSourceIndex === source.index }"
            tabindex="-1"
          >
            <a
              :href="source.url"
              target="_blank"
              rel="noopener noreferrer"
              :aria-label="t('agent.chatOpenWebSource', { title: source.title })"
            >
              <span class="agent-chat-source-index">{{ source.index }}</span>
              <span class="agent-chat-source-favicon" aria-hidden="true">
                <img
                  v-if="canShowFavicon(source)"
                  :src="source.favicon"
                  alt=""
                  loading="lazy"
                  referrerpolicy="no-referrer"
                  @error="markFaviconFailed(source)"
                />
                <span v-else>{{ sourceInitial(source) }}</span>
              </span>
              <span class="agent-chat-source-copy">
                <strong>{{ source.title }}</strong>
                <small>{{ source.domain }}</small>
                <span v-if="source.snippet" class="agent-chat-source-snippet">{{
                  source.snippet
                }}</span>
              </span>
              <ExternalLink aria-hidden="true" />
            </a>
          </li>
        </ol>
      </div>
    </dialog>
  </Teleport>
</template>
