/** Canonical pair names and read compatibility for historical API/config views. */
const aliases: Record<string, string> = {
  single_stream: 'single_stream',
  'single-stream': 'single_stream',
  two_stream: 'two_stream',
  'two-stream': 'two_stream',
  single_tower: 'single_stream',
  'single-tower': 'single_stream',
  dual_tower: 'two_stream',
  'dual-tower': 'two_stream',
}
export function normalizePairStrategy(value: unknown): string {
  const key = String(value ?? '').trim().toLowerCase()
  return aliases[key] ?? key
}
export function pairStrategyLabel(value: unknown): string {
  const key = normalizePairStrategy(value)
  return ({ single_stream: 'Single-stream', two_stream: 'Two-stream' } as Record<string, string>)[key] ?? key
}
function currentField(key: string): string {
  if (!/^[A-Za-z_][A-Za-z_0-9]*$/.test(key) || /^c\d{3}_/.test(key)) return key
  return key.replace(/dual_tower/g, 'two_stream').replace(/single_tower/g, 'single_stream')
    .replace(/DualTower/g, 'TwoStream').replace(/dualTower/g, 'twoStream')
    .replace(/Tower/g, 'Stream').replace(/tower/g, 'stream')
}
function copyProtocol<T>(value: T): T {
  if (Array.isArray(value)) return value.map(copyProtocol) as T
  if (!value || typeof value !== 'object') return value
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, copyProtocol(item)])) as T
}
export function normalizePairFields<T>(value: T): T {
  if (Array.isArray(value)) return value.map(normalizePairFields) as T
  if (!value || typeof value !== 'object') return value
  const result: Record<string, unknown> = {}
  for (const [key, raw] of Object.entries(value)) {
    const name = currentField(key)
    // Keep versioned protocol and manifest payloads byte-for-byte interpretable.
    // Their original validators run on the backend before any adaptation.
    let item = key.endsWith('_contract') || key.endsWith('_manifest')
      ? copyProtocol(raw) : normalizePairFields(raw)
    if ((key === 'pair_strategy' || key === 'pairStrategy' || key === 'strategy') && typeof raw === 'string' && normalizePairStrategy(raw) in aliases) item = normalizePairStrategy(raw)
    else if (['input_adapter', 'route', 'pair_route', 'code', 'reason_code'].includes(key) && typeof raw === 'string') item = currentField(raw)
    if (name in result && JSON.stringify(result[name]) !== JSON.stringify(item))
      throw new Error('Conflicting historical and current stream fields: ' + name)
    result[name] = item
  }
  return result as T
}
/** Render old literal descriptions without rewriting stored names or identifiers. */
export function streamTerminologyForDisplay(value: string): string {
  return value.replace(/single[-_ ]tower/gi, 'Single-stream')
    .replace(/dual[-_ ]tower/gi, 'Two-stream').replace(/单塔/g, '单流').replace(/双塔/g, '双流')
}
