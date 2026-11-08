/**
 * Splits a hero headline into reveal units.
 *
 * English (and any space-separated text) splits on whitespace so the masked
 * reveal reads word by word; Chinese has no spaces, so it splits per character
 * with latin/numeric runs kept intact (e.g. `PepPlatform`, `HLA-I`, `1e-6`).
 * Punctuation becomes its own unit so it never clips its neighbour.
 */
const TOKEN_PATTERN = /[\u3400-\u9fff]|[A-Za-z0-9]+(?:[.'’-][A-Za-z0-9]+)*|[^\s]/gu

export function splitHeadline(text: string): string[] {
  const source = String(text ?? '').trim()
  if (!source) return []
  if (!/[\u3400-\u9fff]/.test(source)) return source.split(/\s+/).filter(Boolean)
  return source.match(TOKEN_PATTERN) ?? []
}

/** True when the headline should be laid out with word gaps (non-CJK). */
export function usesWordSpacing(text: string): boolean {
  return !/[\u3400-\u9fff]/.test(text)
}

/**
 * Groups the reveal units into line-break-safe clusters.
 *
 * Browsers happily break between inline-block characters, which splits Chinese
 * words apart and leaves gaps around full-width punctuation. Each cluster is one
 * inline-block, so breaks only happen between clusters: latin words for English,
 * punctuation-delimited clauses for Chinese (comma stays with its clause).
 */
export function groupHeadline(parts: string[], locale: string): string[][] {
  if (!parts.length) return []
  if (locale !== 'zh') return parts.map((part) => [part])
  const groups: string[][] = []
  let current: string[] = []
  for (const part of parts) {
    current.push(part)
    if (/^[\u3000-\u303f\uff00-\uffef]$/.test(part)) {
      groups.push(current)
      current = []
    }
  }
  if (current.length) groups.push(current)
  return groups
}
