import type { FeatureImportanceItem } from '../types'

export const DEFAULT_VISIBLE_IMPORTANCE_LIMIT = 15

const MAX_FIXED_DECIMALS = 8
const MAX_EXPONENTIAL_DIGITS = 6
const MEANINGFUL_GAP_RATIO = 0.0001

interface IndexedImportanceItem {
  item: FeatureImportanceItem
  index: number
}

export interface ImportanceFormatter {
  mode: 'fixed' | 'exponential'
  decimals?: number
  exponentialDigits?: number
  format: (value: unknown) => string
}

function finiteNumber(value: unknown) {
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? numberValue : undefined
}

function finiteRank(item: FeatureImportanceItem) {
  return finiteNumber(item.rank) ?? Number.POSITIVE_INFINITY
}

export function selectVisibleImportanceItems(
  items: FeatureImportanceItem[] = [],
  limit = DEFAULT_VISIBLE_IMPORTANCE_LIMIT,
) {
  return items
    .map<IndexedImportanceItem>((item, index) => ({ item, index }))
    .filter(({ item }) => finiteNumber(item.importance) !== undefined)
    .sort((left, right) => {
      const leftRank = finiteRank(left.item)
      const rightRank = finiteRank(right.item)
      if (leftRank !== rightRank) return leftRank - rightRank

      const importanceDelta = right.item.importance - left.item.importance
      if (importanceDelta !== 0) return importanceDelta

      return left.index - right.index
    })
    .slice(0, Math.max(0, limit))
    .map(({ item }) => item)
}

function minFixedDecimals(maxAbs: number) {
  if (maxAbs >= 1) return 4
  if (maxAbs >= 0.01) return 5
  if (maxAbs >= 0.0001) return 6
  return MAX_FIXED_DECIMALS
}

function hasMeaningfulCollision(values: number[], formatter: (value: number) => string) {
  if (values.length < 2) return false

  const maxAbs = Math.max(...values.map((value) => Math.abs(value)))
  const meaningfulGap = Math.max(maxAbs * MEANINGFUL_GAP_RATIO, Number.EPSILON)

  for (let leftIndex = 0; leftIndex < values.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < values.length; rightIndex += 1) {
      if (
        Math.abs(values[leftIndex] - values[rightIndex]) >= meaningfulGap &&
        formatter(values[leftIndex]) === formatter(values[rightIndex])
      ) {
        return true
      }
    }
  }

  return false
}

function fixedFormatter(decimals: number) {
  return (value: number) => (value === 0 ? '0' : value.toFixed(decimals))
}

function exponentialFormatter(digits: number) {
  return (value: number) => (value === 0 ? '0' : value.toExponential(digits))
}

function chooseFixedDecimals(values: number[], maxAbs: number) {
  for (let decimals = minFixedDecimals(maxAbs); decimals <= MAX_FIXED_DECIMALS; decimals += 1) {
    if (!hasMeaningfulCollision(values, fixedFormatter(decimals))) return decimals
  }
  return MAX_FIXED_DECIMALS
}

function chooseExponentialDigits(values: number[]) {
  for (let digits = 2; digits <= MAX_EXPONENTIAL_DIGITS; digits += 1) {
    if (!hasMeaningfulCollision(values, exponentialFormatter(digits))) return digits
  }
  return MAX_EXPONENTIAL_DIGITS
}

export function createImportanceFormatter(values: unknown[]): ImportanceFormatter {
  const finiteValues = values
    .map(finiteNumber)
    .filter((value): value is number => value !== undefined)
  const maxAbs = finiteValues.length ? Math.max(...finiteValues.map((value) => Math.abs(value))) : 0

  if (maxAbs > 0 && maxAbs < 0.000001) {
    const exponentialDigits = chooseExponentialDigits(finiteValues)
    return {
      mode: 'exponential',
      exponentialDigits,
      format: (value: unknown) => {
        const numberValue = finiteNumber(value)
        return numberValue === undefined
          ? '-'
          : exponentialFormatter(exponentialDigits)(numberValue)
      },
    }
  }

  const decimals = chooseFixedDecimals(finiteValues, maxAbs)
  return {
    mode: 'fixed',
    decimals,
    format: (value: unknown) => {
      const numberValue = finiteNumber(value)
      return numberValue === undefined ? '-' : fixedFormatter(decimals)(numberValue)
    },
  }
}

function trimInsignificantZeros(value: string): string {
  if (value.includes('e')) {
    const [mantissa, exponent] = value.split('e')
    return `${trimInsignificantZeros(mantissa)}e${exponent}`
  }
  if (!value.includes('.')) return value
  return value.replace(/0+$/, '').replace(/\.$/, '')
}

export function formatFullImportance(value: unknown) {
  const numberValue = finiteNumber(value)
  if (numberValue === undefined) return '-'
  if (numberValue === 0) return '0'

  const abs = Math.abs(numberValue)
  if (abs < 0.000001) return trimInsignificantZeros(numberValue.toExponential(6))
  return trimInsignificantZeros(numberValue.toPrecision(10))
}
