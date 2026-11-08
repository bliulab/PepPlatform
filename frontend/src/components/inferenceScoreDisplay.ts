export function formatIndependentScore(score: number) {
  return Number.isFinite(score) ? score.toFixed(3) : '-'
}

export function independentScoreWidth(score: number) {
  if (!Number.isFinite(score)) return '0%'
  return `${Math.max(0, Math.min(1, score)) * 100}%`
}

export function scoresUseUnitInterval(scores: number[]) {
  return (
    scores.length > 0 && scores.every((score) => Number.isFinite(score) && score >= 0 && score <= 1)
  )
}

export interface IndependentLabelScore {
  index: number
  label: string
  score: number
  threshold?: number
  isPredicted: boolean
}

export function rankIndependentLabelScores(
  scores: number[],
  labels: string[],
  thresholds: number[] = [],
  predictedLabels: string[] = [],
  fallbackLabel: (index: number) => string = (index) => `Label ${index + 1}`,
): IndependentLabelScore[] {
  const predicted = new Set(predictedLabels)

  return scores
    .map((score, index) => {
      const apiLabel = labels[index] || `label_${index + 1}`
      const threshold = thresholds[index]

      return {
        index,
        label: labels[index] || fallbackLabel(index),
        score,
        threshold: Number.isFinite(threshold) ? threshold : undefined,
        isPredicted: predicted.has(apiLabel),
      }
    })
    .sort((left, right) => right.score - left.score || left.index - right.index)
}
