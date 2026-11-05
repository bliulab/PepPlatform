import { notifyUnauthorized } from '../authSession'
import type {
  AgentCampaignRoundPayload,
  AgentChatStreamEvent,
  AgentChatStreamPayload,
  AgentExecutionPlan,
  AgentExecutionPlanPayload,
  AgentExecutionPlanReviewPayload,
  AgentGenerationStage,
  AgentGenerationStreamEvent,
  ExperimentAnalysisStreamEvent,
  ExperimentAnalysisStreamPayload,
  InferenceStreamEvent,
  PredictionPair,
} from '../types'

interface ParsedSseFrame {
  event: string
  data: string
}

export class SseFrameParser {
  private buffer = ''
  private eventName = 'message'
  private dataLines: string[] = []

  feed(chunk: string) {
    this.buffer += chunk
    const frames: ParsedSseFrame[] = []
    let newline = this.buffer.indexOf('\n')
    while (newline >= 0) {
      let line = this.buffer.slice(0, newline)
      this.buffer = this.buffer.slice(newline + 1)
      if (line.endsWith('\r')) line = line.slice(0, -1)
      this.consumeLine(line, frames)
      newline = this.buffer.indexOf('\n')
    }
    return frames
  }

  finish() {
    const frames: ParsedSseFrame[] = []
    if (this.buffer) {
      this.consumeLine(this.buffer.replace(/\r$/, ''), frames)
      this.buffer = ''
    }
    this.dispatch(frames)
    return frames
  }

  private consumeLine(line: string, frames: ParsedSseFrame[]) {
    if (!line) {
      this.dispatch(frames)
      return
    }
    if (line.startsWith(':')) {
      frames.push({ event: 'heartbeat', data: '' })
      return
    }
    const separator = line.indexOf(':')
    const field = separator >= 0 ? line.slice(0, separator) : line
    let value = separator >= 0 ? line.slice(separator + 1) : ''
    if (value.startsWith(' ')) value = value.slice(1)
    if (field === 'event') this.eventName = value || 'message'
    if (field === 'data') this.dataLines.push(value)
  }

  private dispatch(frames: ParsedSseFrame[]) {
    if (!this.dataLines.length) {
      this.eventName = 'message'
      return
    }
    frames.push({ event: this.eventName, data: this.dataLines.join('\n') })
    this.eventName = 'message'
    this.dataLines = []
  }
}

export class AgentChatStreamError extends Error {
  status: number

  constructor(message: string, status = 0) {
    super(message)
    this.name = 'AgentChatStreamError'
    this.status = status
  }
}

function normalizedEvent<T extends { type: string }>(
  frame: ParsedSseFrame,
  allowedEvents: Set<string>,
): T | null {
  if (frame.event === 'heartbeat') return { type: 'heartbeat', data: null } as unknown as T
  if (!allowedEvents.has(frame.event)) return null
  try {
    return { type: frame.event, data: JSON.parse(frame.data) } as unknown as T
  } catch {
    throw new AgentChatStreamError('The AI stream returned an invalid event.')
  }
}

async function* streamChat<T extends { type: string }>(
  endpoint: string,
  payload: object,
  allowedEvents: Set<string>,
  signal?: AbortSignal,
): AsyncGenerator<T> {
  const response = await fetch(endpoint, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(payload),
    signal,
  })
  if (!response.ok) {
    notifyUnauthorized(endpoint, response.status)
    const body = await response.json().catch(() => ({}))
    const detail = body?.detail
    const message =
      typeof detail === 'string' ? detail : detail?.message || `Request failed: ${response.status}`
    throw new AgentChatStreamError(message, response.status)
  }
  if (!response.body) throw new AgentChatStreamError('Streaming is not supported by this browser.')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  const parser = new SseFrameParser()
  let terminalEvent = false
  try {
    while (true) {
      const { value, done } = await reader.read()
      const frames = parser.feed(decoder.decode(value, { stream: !done }))
      for (const frame of frames) {
        const event = normalizedEvent<T>(frame, allowedEvents)
        if (!event) continue
        const eventType = event.type
        if (eventType === 'done' || eventType === 'error') terminalEvent = true
        yield event
      }
      if (done) break
    }
    for (const frame of parser.finish()) {
      const event = normalizedEvent<T>(frame, allowedEvents)
      if (!event) continue
      const eventType = event.type
      if (eventType === 'done' || eventType === 'error') terminalEvent = true
      yield event
    }
    if (!terminalEvent && !signal?.aborted) {
      throw new AgentChatStreamError('The AI stream ended before the answer was complete.')
    }
  } finally {
    reader.releaseLock()
  }
}

export function streamAgentChat(
  payload: AgentChatStreamPayload,
  signal?: AbortSignal,
): AsyncGenerator<AgentChatStreamEvent> {
  return streamChat<AgentChatStreamEvent>(
    '/api/agent/chat/stream',
    payload,
    new Set(['meta', 'search', 'delta', 'done', 'error']),
    signal,
  )
}

export function streamExperimentAnalysis(
  experimentId: string,
  payload: ExperimentAnalysisStreamPayload,
  signal?: AbortSignal,
): AsyncGenerator<ExperimentAnalysisStreamEvent> {
  return streamChat<ExperimentAnalysisStreamEvent>(
    `/api/experiments/${encodeURIComponent(experimentId)}/analysis/stream`,
    payload,
    new Set(['meta', 'stage', 'delta', 'done', 'error']),
    signal,
  )
}

export function streamAgentExecutionPlan(
  payload: AgentExecutionPlanPayload,
  signal?: AbortSignal,
): AsyncGenerator<AgentGenerationStreamEvent> {
  return streamChat<AgentGenerationStreamEvent>(
    '/api/agent/execution-plans/stream',
    payload,
    new Set(['stage', 'done', 'error']),
    signal,
  )
}

export function streamPrediction(
  modelId: string,
  payload: { sequences?: string[]; pairs?: PredictionPair[] },
  signal?: AbortSignal,
): AsyncGenerator<InferenceStreamEvent> {
  return streamChat<InferenceStreamEvent>(
    `/api/models/${encodeURIComponent(modelId)}/predict/stream`,
    payload,
    new Set(['stage', 'done', 'error']),
    signal,
  )
}

export function streamAgentExecutionPlanReview(
  planId: string,
  payload: AgentExecutionPlanReviewPayload,
  signal?: AbortSignal,
): AsyncGenerator<AgentGenerationStreamEvent> {
  return streamChat<AgentGenerationStreamEvent>(
    `/api/agent/execution-plans/${encodeURIComponent(planId)}/review/stream`,
    payload,
    new Set(['stage', 'done', 'error']),
    signal,
  )
}

export function streamAgentCampaignRound(
  campaignId: string,
  payload: AgentCampaignRoundPayload,
  signal?: AbortSignal,
): AsyncGenerator<AgentGenerationStreamEvent> {
  return streamChat<AgentGenerationStreamEvent>(
    `/api/agent/campaigns/${encodeURIComponent(campaignId)}/rounds/stream`,
    payload,
    new Set(['stage', 'done', 'error']),
    signal,
  )
}

export async function completeAgentGeneration(
  events: AsyncGenerator<AgentGenerationStreamEvent>,
  onStage?: (stage: AgentGenerationStage) => void,
): Promise<AgentExecutionPlan> {
  for await (const event of events) {
    if (event.type === 'heartbeat') continue
    if (event.type === 'stage') {
      onStage?.(event.data.stage)
      continue
    }
    if (event.type === 'done') return event.data.plan
    throw new AgentChatStreamError(event.data.detail || event.data.code)
  }
  throw new AgentChatStreamError('The AI stream ended before generation was complete.')
}
