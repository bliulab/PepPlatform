/**
 * Shared ordering of the four peptide task families.
 *
 * Used by the landing scale band / task grid and by the auth brand rail so both
 * surfaces present the families in the same order with the same `T01—T04`
 * station ids. Labels and descriptions come from the `tasks.*` i18n catalog.
 */

export type TaskKey = 'pep_function' | 'pep_protein_interaction' | 'pep_hlaii' | 'pep_hlai'

export interface TaskEntry {
  /** Station id used by the landing scale band and the auth brand rail. */
  id: string
  key: TaskKey
}

export const TASK_ENTRIES: readonly TaskEntry[] = [
  { id: 'T01', key: 'pep_function' },
  { id: 'T02', key: 'pep_protein_interaction' },
  { id: 'T03', key: 'pep_hlai' },
  { id: 'T04', key: 'pep_hlaii' },
]
