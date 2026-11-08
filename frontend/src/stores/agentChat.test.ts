import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import { useAgentChatStore } from './agentChat'
import type { AgentChatConversation, AgentChatConversationListResponse } from '../types'

beforeEach(() => {
  setActivePinia(createPinia())
})
afterEach(() => {
  vi.restoreAllMocks()
})

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const privateConversation = {
  id: 'private-chat',
  title: 'Private research',
  pinned: false,
  updated_at: '2026-10-26T00:00:00Z',
} as AgentChatConversation

describe('chat data after account reset', () => {
  it('drops history that returns after sign-out without overwriting a new account loading state', async () => {
    const oldRequest = deferred<AgentChatConversationListResponse>()
    const newRequest = deferred<AgentChatConversationListResponse>()
    vi.spyOn(api, 'agentChatConversations')
      .mockReturnValueOnce(oldRequest.promise)
      .mockReturnValueOnce(newRequest.promise)
    const chat = useAgentChatStore()
    const oldLoad = chat.loadConversations()
    chat.reset()
    const newLoad = chat.loadConversations()
    oldRequest.resolve({ items: [privateConversation], total: 1, limit: 50, offset: 0 })
    await oldLoad
    expect(chat.conversations).toEqual([])
    expect(chat.historyLoading).toBe(true)
    newRequest.resolve({ items: [], total: 0, limit: 50, offset: 0 })
    await newLoad
    expect(chat.historyLoading).toBe(false)
  })

  it('drops a conversation detail response that arrives after reset', async () => {
    const response = deferred<Awaited<ReturnType<typeof api.agentChatConversation>>>()
    vi.spyOn(api, 'agentChatConversation').mockReturnValue(response.promise)
    const chat = useAgentChatStore()
    const load = chat.loadConversation(privateConversation.id)
    chat.reset()
    response.resolve({ conversation: privateConversation, messages: [], has_more: false })
    await load
    expect(chat.conversations).toEqual([])
    expect(chat.messagesByConversation).toEqual({})
    expect(chat.messageLoading).toBe(false)
  })

  it('does not reinsert a late rename result into the signed-out history', async () => {
    const response = deferred<AgentChatConversation>()
    vi.spyOn(api, 'updateAgentChatConversation').mockReturnValue(response.promise)
    const chat = useAgentChatStore()
    const rename = chat.renameConversation(privateConversation.id, 'Renamed private research')
    chat.reset()
    response.resolve(privateConversation)
    await rename
    expect(chat.conversations).toEqual([])
  })
})
