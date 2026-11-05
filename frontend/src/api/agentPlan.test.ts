import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from './client'

describe('createAgentExecutionPlan', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('sends the explicit strategy preference in the generation payload', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ id: 'plan-1', actions: [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await api.createAgentExecutionPlan({
      provider_config_id: 'provider-1',
      task_type: 'pep_function',
      dataset_id: 'dataset-1',
      strategy_preference: 'aggressive',
    })

    expect(fetchMock).toHaveBeenCalledOnce()
    const [path, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(path).toBe('/api/agent/execution-plans')
    expect(JSON.parse(String(init.body))).toMatchObject({ strategy_preference: 'aggressive' })
  })

  it('confirms one Agent action with the wizard final configuration', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ id: 'experiment-1', status: 'queued' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await api.confirmAgentExecutionAction('plan-1', 'action-1', {
      name: 'Reviewed candidate',
      task_type: 'pep_function',
      dataset_id: 'dataset-1',
      config: { tokenizer: 'kmer', encoder: 'index', model: 'cnn', epochs: 12 },
      generated_name: { kind: 'config', qualifier: 'conservative' },
    })

    const [path, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(path).toBe('/api/agent/execution-plans/plan-1/actions/action-1/confirm')
    expect(init.method).toBe('POST')
    expect(JSON.parse(String(init.body))).toMatchObject({
      name: 'Reviewed candidate',
      config: { epochs: 12 },
      generated_name: { kind: 'config', qualifier: 'conservative' },
    })
  })
})
