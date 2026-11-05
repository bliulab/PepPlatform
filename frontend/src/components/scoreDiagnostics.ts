import type { AnalysisOperatingPoint, ExperimentAnalysis } from '../types'

export interface ThresholdEntry {
  label: string
  value: number
}

export function analysisOperatingPoint(
  analysis?: ExperimentAnalysis,
): AnalysisOperatingPoint | undefined {
  return analysis?.metrics?.operating_point
}

export function operatingPointValues(point?: AnalysisOperatingPoint): number[] {
  const raw =
    point?.thresholds ?? point?.values ?? (point?.threshold == null ? [] : [point.threshold])
  return raw.map(Number).filter(Number.isFinite)
}

export function isMultilabelOperatingPoint(point?: AnalysisOperatingPoint): boolean {
  return operatingPointValues(point).length > 1
}

export function thresholdEntries(point?: AnalysisOperatingPoint): ThresholdEntry[] {
  const values = operatingPointValues(point)
  const labels = Array.isArray(point?.label_names) ? point.label_names : []
  const byLabel = point?.threshold_by_label ?? {}
  if (labels.length === values.length) {
    return values.map((value, index) => ({ label: String(labels[index]), value }))
  }
  const mappedEntries = Object.entries(byLabel)
    .map(([label, value]) => ({ label, value: Number(value) }))
    .filter((entry) => Number.isFinite(entry.value))
  if (mappedEntries.length) return mappedEntries
  return values.map((value, index) => ({ label: String(index + 1), value }))
}
