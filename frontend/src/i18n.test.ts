import { readdirSync, readFileSync } from 'node:fs'
import { extname, join } from 'node:path'
import { describe, expect, it } from 'vitest'

import { i18n } from './i18n'

type Messages = Record<string, unknown>

function flatten(value: Messages, prefix = ''): Record<string, string> {
  const result: Record<string, string> = {}
  for (const [key, child] of Object.entries(value)) {
    const path = prefix ? `${prefix}.${key}` : key
    if (child && typeof child === 'object' && !Array.isArray(child)) {
      Object.assign(result, flatten(child as Messages, path))
    } else {
      result[path] = String(child ?? '')
    }
  }
  return result
}

function sourceFiles(root: string): string[] {
  return readdirSync(root, { withFileTypes: true }).flatMap((entry) => {
    const path = join(root, entry.name)
    if (entry.isDirectory()) return sourceFiles(path)
    return ['.ts', '.vue'].includes(extname(entry.name)) && !entry.name.endsWith('.test.ts')
      ? [path]
      : []
  })
}

describe('translation catalog', () => {
  const zh = flatten(i18n.global.getLocaleMessage('zh') as Messages)
  const en = flatten(i18n.global.getLocaleMessage('en') as Messages)

  it('keeps Chinese and English key sets identical', () => {
    expect(Object.keys(zh).sort()).toEqual(Object.keys(en).sort())
  })

  it('does not leak Chinese UI copy into English translations', () => {
    const leaked = Object.entries(en).filter(([, value]) => /[\u3400-\u9fff]/u.test(value))
    expect(leaked).toEqual([])
  })

  it('defines every literal translation key used by the frontend', () => {
    const literalKeys = new Set<string>()
    for (const path of sourceFiles(join(process.cwd(), 'src'))) {
      const source = readFileSync(path, 'utf8')
      for (const match of source.matchAll(/\bt\(\s*['"]([^'"]+)['"]/g)) literalKeys.add(match[1])
    }
    const missing = [...literalKeys].filter((key) => !(key in en)).sort()
    expect(missing).toEqual([])
  })

  it('does not persist locale state in browser storage', () => {
    const sources = sourceFiles(join(process.cwd(), 'src'))
      .map((path) => readFileSync(path, 'utf8'))
      .join('\n')
    expect(sources).not.toMatch(
      /localStorage[^\n]*(locale|language)|(locale|language)[^\n]*localStorage/i,
    )
  })

  it('names the class I receptor family HLA-I, never MHC-I', () => {
    // Backend-provided labels and dataset names are outside this UI-copy check.
    const labels = { ...zh, ...en }
    const stale = Object.entries(labels).filter(([, value]) => /MHC-I/.test(value))
    expect(stale).toEqual([])
  })

  it('keeps the inference batch markers that the input format depends on', () => {
    for (const catalog of [zh, en]) {
      // Published examples use HLA names; historical tokens are accepted by the compatibility parser.
      expect(catalog['inference.hlaiPairBatchPlaceholder']).toContain('>hla')
      expect(catalog['inference.hlaPairBatchPlaceholder']).toContain('>hla')
    }
  })
})
