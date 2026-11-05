/** Current public names; historical records are normalized only in memory. */
const taskAliases: Record<string, string> = {
  pep_mhci: 'pep_hlai',
  pep_mhci_interaction: 'pep_hlai',
  peptide_mhci_interaction: 'pep_hlai',
  pep_hlai_interaction: 'pep_hlai',
  peptide_hlai_interaction: 'pep_hlai',
}

export function normalizeTaskType(value: unknown): string {
  const key = String(value ?? '').trim().toLowerCase()
  return taskAliases[key] ?? key
}

export function normalizeInputSchema(value: unknown): string {
  const key = String(value ?? '')
  return key === 'peptide_mhc_pairs' ? 'peptide_hlai_pairs' : key
}

export const HLAI_PARTNER_HEADERS = [
  'hla', 'hlai', 'hla_i', 'hla-i', 'mhc', 'mhci', 'mhc_i', 'mhc-i', 'protein', 'pair_pseudo',
]

export function isHlaiFastaHeader(value: string): boolean {
  return ['>hla', '>hlai', '>hla-i', '>mhc', '>mhci', '>mhc-i'].includes(value)
}

/** Decode API views without modifying stored metadata or renaming record IDs. */
export function normalizeTaskFields<T>(value: T): T {
  if (value === 'mhci_route_inference_unavailable') return 'hlai_route_inference_unavailable' as T
  if (Array.isArray(value)) return value.map(normalizeTaskFields) as T
  if (!value || typeof value !== 'object') return value
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [key,
      key === 'task_type' || key === 'taskType' ? normalizeTaskType(item)
      : key === 'input_schema' ? normalizeInputSchema(item)
      : item === 'mhci_route_inference_unavailable' ? 'hlai_route_inference_unavailable'
      : normalizeTaskFields(item),
    ]),
  ) as T
}
