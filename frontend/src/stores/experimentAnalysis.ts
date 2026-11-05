import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { api } from '../api/client'
import { AgentChatDeltaBuffer } from '../agentChatDeltaBuffer'
import { streamExperimentAnalysis } from '../api/chatStream'
import type {
  AgentChatConversation,
  ExperimentAnalysisMessage,
  ExperimentAnalysisStreamPayload,
} from '../types'

function sortedConversations(items: AgentChatConversation[]) {
  return [...items].sort((left, right) => {
    if (left.pinned !== right.pinned) return left.pinned ? -1 : 1
    const leftDate = left.pinned_at || left.updated_at
    const rightDate = right.pinned_at || right.updated_at
    return rightDate.localeCompare(leftDate)
  })
}

export const useExperimentAnalysisStore = defineStore('experimentAnalysis', () => {
  const experimentId = ref('')
  const conversations = ref<AgentChatConversation[]>([])
  const messagesByConversation = ref<Record<string, ExperimentAnalysisMessage[]>>({})
  const hasMoreMessages = ref<Record<string, boolean>>({})
  const total = ref(0)
  const searchQuery = ref('')
  const historyLoading = ref(false)
  const historyError = ref('')
  const messageLoading = ref(false)
  const generating = ref(false)
  const activeGenerationConversationId = ref('')
  const activeAssistantMessageId = ref('')
  const activeStage = ref('')
  const streamError = ref('')
  const streamErrorStatus = ref(0)
  let controller: AbortController | null = null
  let generationVersion = 0

  const activeGenerationMessages = computed(
    () => messagesByConversation.value[activeGenerationConversationId.value] || [],
  )

  function upsertConversation(conversation: AgentChatConversation) {
    const next = conversations.value.filter((item) => item.id !== conversation.id)
    next.push(conversation)
    conversations.value = sortedConversations(next)
  }

  function upsertMessage(message: ExperimentAnalysisMessage) {
    const current = messagesByConversation.value[message.conversation_id] || []
    const next = current.filter((item) => item.id !== message.id)
    next.push(message)
    next.sort((left, right) => left.ordinal - right.ordinal)
    messagesByConversation.value = {
      ...messagesByConversation.value,
      [message.conversation_id]: next,
    }
  }

  const deltaBuffer = new AgentChatDeltaBuffer((deltas) => {
    for (const delta of deltas) {
      const messages = messagesByConversation.value[delta.conversationId] || []
      const message = messages.find((item) => item.id === delta.messageId)
      if (message) {
        upsertMessage({ ...message, content: message.content + delta.content, status: 'streaming' })
      }
    }
  })

  function selectExperiment(nextExperimentId: string) {
    if (experimentId.value === nextExperimentId) return
    reset()
    experimentId.value = nextExperimentId
  }

  async function loadConversations(query = searchQuery.value) {
    if (!experimentId.value) return
    searchQuery.value = query
    historyLoading.value = true
    historyError.value = ''
    try {
      const result = await api.experimentAnalysisConversations(experimentId.value, {
        q: query,
        limit: 50,
        offset: 0,
      })
      conversations.value = result.items
      total.value = result.total
    } catch {
      historyError.value = 'history_load_failed'
    } finally {
      historyLoading.value = false
    }
  }

  async function loadMoreConversations() {
    if (!experimentId.value || historyLoading.value || conversations.value.length >= total.value)
      return
    historyLoading.value = true
    try {
      const result = await api.experimentAnalysisConversations(experimentId.value, {
        q: searchQuery.value,
        limit: 50,
        offset: conversations.value.length,
      })
      const merged = new Map(conversations.value.map((item) => [item.id, item]))
      result.items.forEach((item) => merged.set(item.id, item))
      conversations.value = sortedConversations([...merged.values()])
      total.value = result.total
    } catch {
      historyError.value = 'history_load_failed'
    } finally {
      historyLoading.value = false
    }
  }

  async function loadConversation(conversationId: string, force = false) {
    if (!experimentId.value || !conversationId) return null
    if (
      !force &&
      activeGenerationConversationId.value === conversationId &&
      (messagesByConversation.value[conversationId]?.length || 0) > 0
    ) {
      return {
        conversation: conversations.value.find((item) => item.id === conversationId) || null,
        messages: messagesByConversation.value[conversationId],
      }
    }
    messageLoading.value = true
    try {
      const result = await api.experimentAnalysisConversation(experimentId.value, conversationId, {
        limit: 200,
      })
      upsertConversation(result.conversation)
      messagesByConversation.value = {
        ...messagesByConversation.value,
        [conversationId]: result.messages,
      }
      hasMoreMessages.value = {
        ...hasMoreMessages.value,
        [conversationId]: result.has_more,
      }
      return result
    } finally {
      messageLoading.value = false
    }
  }

  async function loadEarlierMessages(conversationId: string) {
    if (!experimentId.value) return
    const current = messagesByConversation.value[conversationId] || []
    const beforeOrdinal = current[0]?.ordinal
    if (!beforeOrdinal || !hasMoreMessages.value[conversationId] || messageLoading.value) return
    messageLoading.value = true
    try {
      const result = await api.experimentAnalysisConversation(experimentId.value, conversationId, {
        limit: 200,
        before_ordinal: beforeOrdinal,
      })
      const merged = new Map(result.messages.map((message) => [message.id, message]))
      current.forEach((message) => merged.set(message.id, message))
      messagesByConversation.value = {
        ...messagesByConversation.value,
        [conversationId]: [...merged.values()].sort((left, right) => left.ordinal - right.ordinal),
      }
      hasMoreMessages.value = {
        ...hasMoreMessages.value,
        [conversationId]: result.has_more,
      }
    } finally {
      messageLoading.value = false
    }
  }

  async function send(payload: ExperimentAnalysisStreamPayload) {
    if (!experimentId.value) throw new Error('experiment_required')
    if (generating.value) throw new Error('generation_in_progress')
    const version = ++generationVersion
    generating.value = true
    streamError.value = ''
    streamErrorStatus.value = 0
    activeStage.value = ''
    activeGenerationConversationId.value = payload.conversation_id || ''
    activeAssistantMessageId.value = ''
    deltaBuffer.cancel()
    const activeController = new AbortController()
    controller = activeController
    try {
      for await (const event of streamExperimentAnalysis(
        experimentId.value,
        payload,
        activeController.signal,
      )) {
        if (version !== generationVersion) break
        if (event.type === 'heartbeat') continue
        if (event.type === 'meta') {
          activeGenerationConversationId.value = event.data.conversation.id
          activeAssistantMessageId.value = event.data.assistant_message.id
          upsertConversation(event.data.conversation)
          upsertMessage(event.data.user_message)
          upsertMessage(event.data.assistant_message)
          continue
        }
        if (event.type === 'stage') {
          activeStage.value = event.data.stage
          continue
        }
        if (event.type === 'delta') {
          deltaBuffer.append(
            activeGenerationConversationId.value,
            event.data.message_id,
            event.data.content,
          )
          continue
        }
        if (event.type === 'done') {
          deltaBuffer.flush()
          upsertMessage(event.data.message)
          continue
        }
        if (event.type === 'error') {
          deltaBuffer.flush()
          streamError.value = event.data.detail || event.data.code || 'stream_failed'
          if (event.data.message) upsertMessage(event.data.message)
        }
      }
    } catch (error) {
      deltaBuffer.flush()
      if (activeController.signal.aborted) {
        const messages = messagesByConversation.value[activeGenerationConversationId.value] || []
        const message = messages.find((item) => item.id === activeAssistantMessageId.value)
        if (message) upsertMessage({ ...message, status: 'stopped', finish_reason: 'stopped' })
      } else {
        const status = Number((error as { status?: unknown } | null)?.status)
        streamErrorStatus.value = Number.isFinite(status) ? status : 0
        streamError.value =
          error instanceof Error && error.message ? error.message : 'stream_failed'
      }
    } finally {
      deltaBuffer.flush()
      if (version !== generationVersion) return activeGenerationConversationId.value
      controller = null
      generating.value = false
      activeAssistantMessageId.value = ''
      activeStage.value = ''
      void loadConversations(searchQuery.value)
    }
    return activeGenerationConversationId.value
  }

  function stopGeneration() {
    controller?.abort()
  }

  async function renameConversation(conversationId: string, title: string) {
    const updated = await api.updateExperimentAnalysisConversation(
      experimentId.value,
      conversationId,
      { title },
    )
    upsertConversation(updated)
    return updated
  }

  async function togglePin(conversation: AgentChatConversation) {
    const updated = await api.updateExperimentAnalysisConversation(
      experimentId.value,
      conversation.id,
      { pinned: !conversation.pinned },
    )
    upsertConversation(updated)
    return updated
  }

  async function deleteConversation(conversationId: string) {
    if (generating.value && activeGenerationConversationId.value === conversationId) {
      throw new Error('stop_before_delete')
    }
    await api.deleteExperimentAnalysisConversation(experimentId.value, conversationId)
    conversations.value = conversations.value.filter((item) => item.id !== conversationId)
    const nextMessages = { ...messagesByConversation.value }
    delete nextMessages[conversationId]
    messagesByConversation.value = nextMessages
    total.value = Math.max(0, total.value - 1)
  }

  function reset() {
    generationVersion += 1
    controller?.abort()
    deltaBuffer.cancel()
    controller = null
    generating.value = false
    experimentId.value = ''
    conversations.value = []
    messagesByConversation.value = {}
    hasMoreMessages.value = {}
    total.value = 0
    searchQuery.value = ''
    historyError.value = ''
    streamError.value = ''
    streamErrorStatus.value = 0
    activeGenerationConversationId.value = ''
    activeAssistantMessageId.value = ''
    activeStage.value = ''
  }

  return {
    experimentId,
    conversations,
    messagesByConversation,
    hasMoreMessages,
    total,
    searchQuery,
    historyLoading,
    historyError,
    messageLoading,
    generating,
    activeGenerationConversationId,
    activeAssistantMessageId,
    activeGenerationMessages,
    activeStage,
    streamError,
    streamErrorStatus,
    selectExperiment,
    loadConversations,
    loadMoreConversations,
    loadConversation,
    loadEarlierMessages,
    send,
    stopGeneration,
    renameConversation,
    togglePin,
    deleteConversation,
    reset,
  }
})
