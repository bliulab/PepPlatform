import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  completeAgentGeneration,
  SseFrameParser,
  streamAgentExecutionPlan,
  streamPrediction,
} from './chatStream'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('SseFrameParser', () => {
  it('preserves events split across arbitrary transport chunks', () => {
    const parser = new SseFrameParser()
    const frames = [
      ...parser.feed('event: me'),
      ...parser.feed('ta\ndata: {"conversation"'),
      ...parser.feed(': {"id": "one"}}\n\n'),
      ...parser.feed(
        ': ping\n\nevent: search\ndata: {"status":"searching"}\n\nevent: delta\ndata: {"content":"你',
      ),
      ...parser.feed('好"}\n\n'),
      ...parser.finish(),
    ]

    expect(frames).toEqual([
      { event: 'meta', data: '{"conversation": {"id": "one"}}' },
      { event: 'heartbeat', data: '' },
      { event: 'search', data: '{"status":"searching"}' },
      { event: 'delta', data: '{"content":"你好"}' },
    ])
  })

  it('joins multi-line data fields and flushes the final frame', () => {
    const parser = new SseFrameParser()
    expect(parser.feed('event: error\ndata: first\ndata: second')).toEqual([])
    expect(parser.finish()).toEqual([{ event: 'error', data: 'first\nsecond' }])
  })
})

describe('agent generation stream', () => {
  it('returns the completed plan and reports generation stages', async () => {
    const responseBody = [
      'event: stage\ndata: {"stage":"generating","attempt":1,"max_attempts":3}\n\n',
      ': ping\n\n',
      'event: stage\ndata: {"stage":"saving","attempt":1,"max_attempts":3}\n\n',
      'event: done\ndata: {"plan":{"id":"plan-1","actions":[]}}\n\n',
    ].join('')
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(responseBody, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)
    const stages: string[] = []

    const plan = await completeAgentGeneration(
      streamAgentExecutionPlan({
        provider_config_id: 'provider-1',
        task_type: 'pep_function',
        dataset_id: 'dataset-1',
      }),
      (stage) => stages.push(stage),
    )

    expect(plan.id).toBe('plan-1')
    expect(stages).toEqual(['generating', 'saving'])
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/agent/execution-plans/stream',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('passes AbortSignal through to fetch', async () => {
    const fetchMock = vi.fn(
      (_path: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () =>
            reject(new DOMException('Aborted', 'AbortError')),
          )
        }),
    )
    vi.stubGlobal('fetch', fetchMock)
    const controller = new AbortController()
    const result = completeAgentGeneration(
      streamAgentExecutionPlan(
        {
          provider_config_id: 'provider-1',
          task_type: 'pep_function',
          dataset_id: 'dataset-1',
        },
        controller.signal,
      ),
    )

    controller.abort()

    await expect(result).rejects.toMatchObject({ name: 'AbortError' })
    expect(fetchMock.mock.calls[0]?.[1]?.signal).toBe(controller.signal)
  })
})

describe('prediction stream', () => {
  it('parses progress and terminal prediction events', async () => {
    const responseBody = [
      'event: stage\ndata: {"stage":"loading_encoder","elapsed_ms":10}\n\n',
      'event: done\ndata: {"model_id":"model/one","labels":[],"predictions":[]}\n\n',
    ].join('')
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(responseBody, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const eventTypes: string[] = []
    for await (const event of streamPrediction('model/one', { sequences: ['ACD'] })) {
      eventTypes.push(event.type)
    }

    expect(eventTypes).toEqual(['stage', 'done'])
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/models/model%2Fone/predict/stream',
      expect.objectContaining({ method: 'POST' }),
    )
  })
})
