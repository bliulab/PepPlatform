export type SelectValue = string | number

export interface SelectOption {
  value: SelectValue
  label: string
  group?: string
  disabled?: boolean
}

export interface SelectMenuPlacementInput {
  trigger: {
    top: number
    bottom: number
    left: number
    width: number
  }
  menuHeight: number
  viewportWidth: number
  viewportHeight: number
  gap?: number
  margin?: number
  maxHeight?: number
}

export interface SelectMenuPlacement {
  placement: 'top' | 'bottom'
  top: number
  left: number
  width: number
  maxHeight: number
}

export function selectedOptionIndex(options: SelectOption[], value: SelectValue) {
  return options.findIndex((option) => Object.is(option.value, value))
}

export function boundaryEnabledOptionIndex(options: SelectOption[], edge: 'first' | 'last') {
  if (edge === 'first') return options.findIndex((option) => !option.disabled)
  for (let index = options.length - 1; index >= 0; index -= 1) {
    if (!options[index]?.disabled) return index
  }
  return -1
}

export function nextEnabledOptionIndex(
  options: SelectOption[],
  currentIndex: number,
  direction: 1 | -1,
) {
  if (!options.length) return -1
  const origin =
    currentIndex >= 0 && currentIndex < options.length ? currentIndex : direction === 1 ? -1 : 0
  for (let offset = 1; offset <= options.length; offset += 1) {
    const index = (origin + direction * offset + options.length) % options.length
    if (!options[index]?.disabled) return index
  }
  return -1
}

export function findTypeaheadOptionIndex(
  options: SelectOption[],
  query: string,
  currentIndex: number,
) {
  const normalizedQuery = query.trim().toLocaleLowerCase()
  if (!normalizedQuery || !options.length) return -1
  const origin = currentIndex >= 0 && currentIndex < options.length ? currentIndex : -1
  for (let offset = 1; offset <= options.length; offset += 1) {
    const index = (origin + offset + options.length) % options.length
    const option = options[index]
    if (!option?.disabled && option.label.trim().toLocaleLowerCase().startsWith(normalizedQuery)) {
      return index
    }
  }
  return -1
}

export function resolveSelectMenuPlacement({
  trigger,
  menuHeight,
  viewportWidth,
  viewportHeight,
  gap = 6,
  margin = 8,
  maxHeight = 288,
}: SelectMenuPlacementInput): SelectMenuPlacement {
  const availableBelow = Math.max(0, viewportHeight - trigger.bottom - gap - margin)
  const availableAbove = Math.max(0, trigger.top - gap - margin)
  const desiredHeight = Math.min(maxHeight, Math.max(44, menuHeight))
  const placement =
    availableBelow >= Math.min(desiredHeight, 160) || availableBelow >= availableAbove
      ? 'bottom'
      : 'top'
  const availableHeight = placement === 'bottom' ? availableBelow : availableAbove
  const visibleHeight = Math.min(desiredHeight, Math.max(44, availableHeight))
  const width = Math.max(0, Math.min(trigger.width, viewportWidth - margin * 2))
  const leftLimit = Math.max(margin, viewportWidth - margin - width)
  const left = Math.min(Math.max(trigger.left, margin), leftLimit)
  const top =
    placement === 'bottom'
      ? Math.min(trigger.bottom + gap, viewportHeight - margin - visibleHeight)
      : Math.max(margin, trigger.top - gap - visibleHeight)

  return { placement, top, left, width, maxHeight: visibleHeight }
}
