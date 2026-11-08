import { describe, expect, it } from 'vitest'
import { createAgentChatMarkdownRenderer } from './agentChatMarkdown'
import type { AgentChatMessage } from './types'

function message(content: string): AgentChatMessage {
  return {
    id: 'message-1',
    conversation_id: 'conversation-1',
    ordinal: 1,
    role: 'assistant',
    content,
    status: 'complete',
    provider: 'deepseek',
    model: 'deepseek-v4-flash',
    finish_reason: 'stop',
    error: '',
    usage: {},
    created_at: '2026-07-19T00:00:00Z',
    updated_at: '2026-07-19T00:00:00Z',
  }
}

describe('agent chat Markdown renderer', () => {
  it('renders bracket and dollar math delimiters', () => {
    const renderer = createAgentChatMarkdownRenderer((html) => html)
    const html = renderer.render(
      message('Inline \\(x^2\\) and $y^2$.\n\n\\[\\text{RoPE}(x_m) = R_m \\cdot x_m\\]\n\n$$z^2$$'),
    )

    expect(html.match(/class="katex"/g)?.length).toBe(4)
    expect(html).toContain('katex-display')
    expect(html).toContain('RoPE')
  })

  it('keeps math-like code literal and sanitizes raw HTML', () => {
    const renderer = createAgentChatMarkdownRenderer((html) => html)
    const html = renderer.render(
      message('`$x$`\n\n```latex\n\\[x\\]\n```\n\n<script>alert(1)</script>'),
    )

    expect(html).toContain('<code>$x$</code>')
    expect(html).toContain('\\[x\\]')
    expect(html).not.toContain('<script>')
  })

  it('adds source metadata only for known citations', () => {
    const renderer = createAgentChatMarkdownRenderer((html) => html)
    const cited = message('Supported [1], unknown [2].')
    cited.web_search = {
      requested: true,
      status: 'complete',
      query: 'test',
      provider: 'tavily',
      credits: 1,
      reason: '',
      sources: [
        {
          index: 1,
          title: 'Source',
          url: 'https://example.test',
          domain: 'example.test',
          snippet: '',
        },
      ],
    }

    const html = renderer.render(cited)
    expect(html).toContain('data-message-id="message-1"')
    expect(html).toContain('data-source-index="1"')
    expect(html).toContain('unknown [2]')
  })
})
