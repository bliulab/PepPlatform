import { katex } from '@mdit/plugin-katex'
import DOMPurify from 'dompurify'
import MarkdownIt from 'markdown-it'
import 'katex/dist/katex.min.css'
import type { AgentChatMessage } from './types'

interface CitationEnvironment {
  citationIndexes?: number[]
  citationMessageId?: string
}

interface CachedMarkdown {
  signature: string
  html: string
}

export function createAgentChatMarkdownRenderer(
  sanitize: (html: string) => string = (html) => DOMPurify.sanitize(html),
) {
  const md = new MarkdownIt({ html: false, linkify: true, breaks: true })
  const cache = new Map<string, CachedMarkdown>()

  md.use(katex, {
    delimiters: 'all',
    throwOnError: false,
    strict: 'ignore',
    trust: false,
  })

  md.inline.ruler.before('link', 'agent-source-citation', (state, silent) => {
    const match = state.src.slice(state.pos).match(/^\[(\d+)\]/)
    if (!match || state.src[state.pos + match[0].length] === '(') return false
    const sourceIndex = Number(match[1])
    const env = state.env as CitationEnvironment
    if (!env.citationIndexes?.includes(sourceIndex) || !env.citationMessageId) return false
    if (!silent) {
      const linkOpen = state.push('link_open', 'a', 1)
      linkOpen.attrSet('href', '#agent-chat-source-dialog')
      linkOpen.attrSet('class', 'agent-chat-citation')
      linkOpen.attrSet('data-message-id', env.citationMessageId)
      linkOpen.attrSet('data-source-index', String(sourceIndex))
      const textToken = state.push('text', '', 0)
      textToken.content = match[0]
      state.push('link_close', 'a', -1)
    }
    state.pos += match[0].length
    return true
  })

  return {
    render(message: AgentChatMessage) {
      const citationIndexes = (message.web_search?.sources || []).map((source) => source.index)
      const signature = `${message.content}\u0000${citationIndexes.join(',')}`
      const cached = cache.get(message.id)
      if (cached?.signature === signature) return cached.html
      const html = sanitize(
        md.render(message.content || '', { citationIndexes, citationMessageId: message.id }),
      )
      cache.set(message.id, { signature, html })
      return html
    },
    prune(messageIds: Iterable<string>) {
      const retained = new Set(messageIds)
      for (const messageId of cache.keys()) {
        if (!retained.has(messageId)) cache.delete(messageId)
      }
    },
  }
}
