export interface AgentChatBufferedDelta {
  conversationId: string
  messageId: string
  content: string
}

export class AgentChatDeltaBuffer {
  private pending = new Map<string, AgentChatBufferedDelta>()
  private timer: ReturnType<typeof setTimeout> | undefined

  constructor(
    private readonly onFlush: (deltas: AgentChatBufferedDelta[]) => void,
    private readonly intervalMs = 40,
  ) {}

  append(conversationId: string, messageId: string, content: string) {
    if (!conversationId || !messageId || !content) return
    const current = this.pending.get(messageId)
    this.pending.set(messageId, {
      conversationId,
      messageId,
      content: `${current?.content || ''}${content}`,
    })
    if (!this.timer) this.timer = setTimeout(() => this.flush(), this.intervalMs)
  }

  flush() {
    if (this.timer) clearTimeout(this.timer)
    this.timer = undefined
    if (!this.pending.size) return
    const deltas = [...this.pending.values()]
    this.pending.clear()
    this.onFlush(deltas)
  }

  cancel() {
    if (this.timer) clearTimeout(this.timer)
    this.timer = undefined
    this.pending.clear()
  }
}
