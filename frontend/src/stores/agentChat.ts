import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { api } from '../api/client'
import { AgentChatDeltaBuffer } from '../agentChatDeltaBuffer'
import { streamAgentChat } from '../api/chatStream'
import type { AgentChatConversation, AgentChatMessage, AgentChatStreamPayload } from '../types'

function sortedConversations(items: AgentChatConversation[]) {
  return [...items].sort((left, right) => {
    if (left.pinned !== right.pinned) return left.pinned ? -1 : 1
    const leftDate = left.pinned_at || left.updated_at
    const rightDate = right.pinned_at || right.updated_at
    return rightDate.localeCompare(leftDate)
  })
}

export const useAgentChatStore = defineStore('agentChat', () => {
  const conversations = ref<AgentChatConversation[]>([])
  const messagesByConversation = ref<Record<string, AgentChatMessage[]>>({})
  const hasMoreMessages = ref<Record<string, boolean>>({})
  const total = ref(0)
  const searchQuery = ref('')
  const historyLoading = ref(false)
  const historyError = ref('')
  const messageLoading = ref(false)
  const generating = ref(false)
  const activeGenerationConversationId = ref('')
  const activeAssistantMessageId = ref('')
  const activeSearchStatus = ref<'checking' | 'searching' | 'complete' | 'unavailable' | ''>('')
  const streamError = ref('')
  let controller: AbortController | null = null
  let generationVersion = 0
  let sessionVersion = 0

  const activeGenerationMessages = computed(
    () => messagesByConversation.value[activeGenerationConversationId.value] || [],
  )

  function upsertConversation(conversation: AgentChatConversation) {
    const next = conversations.value.filter((item) => item.id !== conversation.id)
    next.push(conversation)
    conversations.value = sortedConversations(next)
  }

  function upsertMessage(message: AgentChatMessage) {
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

  async function loadConversations(query = searchQuery.value) {
    const version = sessionVersion
    searchQuery.value = query
    historyLoading.value = true
    historyError.value = ''
    try {
      const result = await api.agentChatConversations({ q: query, limit: 50, offset: 0 })
      if (version !== sessionVersion) return
      conversations.value = result.items
      total.value = result.total
    } catch (error) {
      if (version === sessionVersion) historyError.value = 'history_load_failed'
    } finally {
      if (version === sessionVersion) historyLoading.value = false
    }
  }

  async function loadMoreConversations() {
    const version = sessionVersion
    if (historyLoading.value || conversations.value.length >= total.value) return
    historyLoading.value = true
    historyError.value = ''
    try {
      const result = await api.agentChatConversations({
        q: searchQuery.value,
        limit: 50,
        offset: conversations.value.length,
      })
      if (version !== sessionVersion) return
      const merged = new Map(conversations.value.map((item) => [item.id, item]))
      result.items.forEach((item) => merged.set(item.id, item))
      conversations.value = sortedConversations([...merged.values()])
      total.value = result.total
    } catch (error) {
      if (version === sessionVersion) historyError.value = 'history_load_failed'
    } finally {
      if (version === sessionVersion) historyLoading.value = false
    }
  }

  async function loadConversation(conversationId: string, force = false) {
    const version = sessionVersion
    if (!conversationId) return null
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
      const result = await api.agentChatConversation(conversationId, { limit: 200 })
      if (version !== sessionVersion) return
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
      if (version === sessionVersion) messageLoading.value = false
    }
  }

  async function loadEarlierMessages(conversationId: string) {
    const version = sessionVersion
    const current = messagesByConversation.value[conversationId] || []
    const beforeOrdinal = current[0]?.ordinal
    if (!beforeOrdinal || !hasMoreMessages.value[conversationId] || messageLoading.value) return
    messageLoading.value = true
    try {
      const result = await api.agentChatConversation(conversationId, {
        limit: 200,
        before_ordinal: beforeOrdinal,
      })
      if (version !== sessionVersion) return
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
      if (version === sessionVersion) messageLoading.value = false
    }
  }

  async function send(payload: AgentChatStreamPayload) {
    if (generating.value) throw new Error('generation_in_progress')
    const version = ++generationVersion
    generating.value = true
    streamError.value = ''
    historyError.value = ''
    activeGenerationConversationId.value = payload.conversation_id || ''
    activeAssistantMessageId.value = ''
    activeSearchStatus.value = ''
    deltaBuffer.cancel()
    const activeController = new AbortController()
    controller = activeController
    try {
      for await (const event of streamAgentChat(payload, activeController.signal)) {
        if (version !== generationVersion) break
        if (event.type === 'heartbeat') continue
        if (event.type === 'meta') {
          const turn = event.data
          activeGenerationConversationId.value = turn.conversation.id
          activeAssistantMessageId.value = turn.assistant_message.id
          upsertConversation(turn.conversation)
          upsertMessage(turn.user_message)
          upsertMessage(turn.assistant_message)
          continue
        }
        if (event.type === 'delta') {
          const conversationId = activeGenerationConversationId.value
          deltaBuffer.append(conversationId, event.data.message_id, event.data.content)
          continue
        }
        if (event.type === 'search') {
          activeSearchStatus.value = event.data.status
          if (event.data.web_search) {
            const conversationId = activeGenerationConversationId.value
            const messages = messagesByConversation.value[conversationId] || []
            const message = messages.find((item) => item.id === event.data.message_id)
            if (message) upsertMessage({ ...message, web_search: event.data.web_search })
          }
          continue
        }
        if (event.type === 'done') {
          deltaBuffer.flush()
          upsertMessage(event.data.message)
          continue
        }
        if (event.type === 'error') {
          deltaBuffer.flush()
          if (!event.data.message) streamError.value = event.data.code || 'stream_failed'
          if (event.data.message) upsertMessage(event.data.message)
        }
      }
    } catch (error) {
      deltaBuffer.flush()
      if (activeController.signal.aborted) {
        const conversationId = activeGenerationConversationId.value
        const messages = messagesByConversation.value[conversationId] || []
        const message = messages.find((item) => item.id === activeAssistantMessageId.value)
        if (message) upsertMessage({ ...message, status: 'stopped', finish_reason: 'stopped' })
      } else {
        streamError.value = 'stream_failed'
      }
    } finally {
      deltaBuffer.flush()
      if (version !== generationVersion) return activeGenerationConversationId.value
      controller = null
      generating.value = false
      activeAssistantMessageId.value = ''
      activeSearchStatus.value = ''
      void loadConversations(searchQuery.value)
    }
    return activeGenerationConversationId.value
  }

  function stopGeneration() {
    controller?.abort()
  }

  async function renameConversation(conversationId: string, title: string) {
    const version = sessionVersion
    const updated = await api.updateAgentChatConversation(conversationId, { title })
    if (version !== sessionVersion) return
    upsertConversation(updated)
    return updated
  }

  async function togglePin(conversation: AgentChatConversation) {
    const version = sessionVersion
    const updated = await api.updateAgentChatConversation(conversation.id, {
      pinned: !conversation.pinned,
    })
    if (version !== sessionVersion) return
    upsertConversation(updated)
    return updated
  }

  async function deleteConversation(conversationId: string) {
    const version = sessionVersion
    if (generating.value && activeGenerationConversationId.value === conversationId) {
      throw new Error('stop_before_delete')
    }
    await api.deleteAgentChatConversation(conversationId)
    if (version !== sessionVersion) return
    conversations.value = conversations.value.filter((item) => item.id !== conversationId)
    const nextMessages = { ...messagesByConversation.value }
    delete nextMessages[conversationId]
    messagesByConversation.value = nextMessages
    const nextHasMore = { ...hasMoreMessages.value }
    delete nextHasMore[conversationId]
    hasMoreMessages.value = nextHasMore
    total.value = Math.max(0, total.value - 1)
  }

  function reset() {
    sessionVersion += 1
    generationVersion += 1
    controller?.abort()
    deltaBuffer.cancel()
    controller = null
    generating.value = false
    conversations.value = []
    messagesByConversation.value = {}
    hasMoreMessages.value = {}
    total.value = 0
    searchQuery.value = ''
    streamError.value = ''
    historyError.value = ''
    historyLoading.value = false
    messageLoading.value = false
    activeGenerationConversationId.value = ''
    activeAssistantMessageId.value = ''
    activeSearchStatus.value = ''
  }

  return {
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
    activeSearchStatus,
    streamError,
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
