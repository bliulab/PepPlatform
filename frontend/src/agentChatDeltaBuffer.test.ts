import { afterEach, describe, expect, it, vi } from 'vitest'
import { AgentChatDeltaBuffer } from './agentChatDeltaBuffer'

afterEach(() => vi.useRealTimers())

describe('AgentChatDeltaBuffer', () => {
  it('combines rapid deltas into one ordered update', () => {
    vi.useFakeTimers()
    const flushed: string[] = []
    const buffer = new AgentChatDeltaBuffer(
      (items) => flushed.push(items.map((item) => item.content).join('|')),
      40,
    )

    buffer.append('conversation-1', 'message-1', '你')
    buffer.append('conversation-1', 'message-1', '好')
    buffer.append('conversation-1', 'message-1', '。')
    vi.advanceTimersByTime(39)
    expect(flushed).toEqual([])
    vi.advanceTimersByTime(1)
    expect(flushed).toEqual(['你好。'])
  })

  it('flushes terminal content immediately and cancels stale work', () => {
    vi.useFakeTimers()
    const flushed: string[] = []
    const buffer = new AgentChatDeltaBuffer(
      (items) => flushed.push(items.map((item) => item.content).join('|')),
      40,
    )

    buffer.append('conversation-1', 'message-1', 'final')
    buffer.flush()
    buffer.append('conversation-1', 'message-1', 'stale')
    buffer.cancel()
    vi.runAllTimers()
    expect(flushed).toEqual(['final'])
  })
})
