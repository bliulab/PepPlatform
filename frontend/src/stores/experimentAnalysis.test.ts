import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type {
  AgentChatConversation,
  ExperimentAnalysisMessage,
  ExperimentAnalysisStreamEvent,
  ExperimentDiagnosticReport,
} from '../types'

const mocks = vi.hoisted(() => ({
  stream: vi.fn(),
  listConversations: vi.fn(),
}))

vi.mock('../api/chatStream', () => ({
  streamExperimentAnalysis: mocks.stream,
}))

vi.mock('../api/client', () => ({
  api: {
    experimentAnalysisConversations: mocks.listConversations,
  },
}))

import { useExperimentAnalysisStore } from './experimentAnalysis'

const timestamp = '2026-07-20T10:00:00+00:00'

function conversation(): AgentChatConversation {
  return {
    id: 'conversation-1',
    scope: 'experiment_analysis',
    experiment_id: 'experiment-1',
    title: 'Diagnose validation behavior',
    pinned: false,
    last_provider: 'deepseek',
    last_model: 'deepseek-v4-flash',
    last_message_preview: 'Diagnose validation behavior',
    created_at: timestamp,
    updated_at: timestamp,
  }
}

function message(
  id: string,
  role: 'user' | 'assistant',
  ordinal: number,
  overrides: Partial<ExperimentAnalysisMessage> = {},
): ExperimentAnalysisMessage {
  return {
    id,
    conversation_id: 'conversation-1',
    ordinal,
    role,
    content: role === 'user' ? 'Diagnose validation behavior' : '',
    status: role === 'user' ? 'complete' : 'streaming',
    kind: 'chat',
    provider_config_id: role === 'assistant' ? 'provider-1' : null,
    provider: role === 'assistant' ? 'deepseek' : '',
    model: role === 'assistant' ? 'deepseek-v4-flash' : '',
    finish_reason: '',
    error: '',
    usage: {},
    created_at: timestamp,
    updated_at: timestamp,
    ...overrides,
  }
}

const report: ExperimentDiagnosticReport = {
  status: 'final',
  summary: 'The run is stable enough for a baseline comparison.',
  evidence: [],
  findings: [],
  recommendations: [],
  limitations: [],
}

async function* analysisEvents(): AsyncGenerator<ExperimentAnalysisStreamEvent> {
  yield {
    type: 'meta',
    data: {
      conversation: conversation(),
      user_message: message('user-1', 'user', 1),
      assistant_message: message('assistant-1', 'assistant', 2, { kind: 'report' }),
    },
  }
  yield { type: 'stage', data: { message_id: 'assistant-1', stage: 'validating_report' } }
  yield { type: 'delta', data: { message_id: 'assistant-1', content: 'temporary content' } }
  yield {
    type: 'done',
    data: {
      message: message('assistant-1', 'assistant', 2, {
        kind: 'report',
        status: 'complete',
        content: '# Experiment diagnostic report',
        finish_reason: 'stop',
        payload: { report, repair_attempted: false, warnings: [] },
      }),
    },
  }
}

describe('experiment analysis store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    mocks.stream.mockReset()
    mocks.listConversations.mockReset()
    mocks.stream.mockImplementation(() => analysisEvents())
    mocks.listConversations.mockResolvedValue({
      items: [conversation()],
      total: 1,
      limit: 50,
      offset: 0,
    })
  })

  it('merges stream events and keeps the terminal structured report', async () => {
    const store = useExperimentAnalysisStore()
    store.selectExperiment('experiment-1')

    const conversationId = await store.send({
      provider_config_id: 'provider-1',
      content: 'Diagnose validation behavior',
      language: 'en',
    })

    expect(conversationId).toBe('conversation-1')
    expect(mocks.stream).toHaveBeenCalledWith(
      'experiment-1',
      expect.objectContaining({ provider_config_id: 'provider-1' }),
      expect.any(AbortSignal),
    )
    expect(store.generating).toBe(false)
    expect(store.activeStage).toBe('')
    expect(store.messagesByConversation['conversation-1']).toHaveLength(2)
    expect(store.messagesByConversation['conversation-1'][1]).toMatchObject({
      id: 'assistant-1',
      kind: 'report',
      status: 'complete',
      content: '# Experiment diagnostic report',
      payload: { report },
    })
  })

  it('distinguishes a failed history request from an empty history', async () => {
    const store = useExperimentAnalysisStore()
    store.selectExperiment('experiment-1')
    mocks.listConversations.mockRejectedValueOnce(new Error('database unavailable'))

    await store.loadConversations()

    expect(store.conversations).toEqual([])
    expect(store.historyError).toBe('history_load_failed')
  })

  it('keeps the HTTP status and message from a failed stream request', async () => {
    const store = useExperimentAnalysisStore()
    store.selectExperiment('experiment-1')
    mocks.stream.mockImplementationOnce(async function* () {
      if (false) yield {} as ExperimentAnalysisStreamEvent
      throw Object.assign(new Error('Request failed: 500'), { status: 500 })
    })

    await store.send({
      provider_config_id: 'provider-1',
      content: 'Diagnose validation behavior',
    })

    expect(store.streamError).toBe('Request failed: 500')
    expect(store.streamErrorStatus).toBe(500)
    expect(store.generating).toBe(false)
  })
})
