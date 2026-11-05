import { describe, expect, it } from 'vitest'
import {
  boundaryEnabledOptionIndex,
  findTypeaheadOptionIndex,
  nextEnabledOptionIndex,
  resolveSelectMenuPlacement,
  selectedOptionIndex,
  type SelectOption,
} from './selectControl'

const options: SelectOption[] = [
  { value: 'classic', label: 'Classic', group: 'Feature routes' },
  { value: 'blocked', label: 'Blocked', group: 'Feature routes', disabled: true },
  { value: 'pretrained', label: 'Pretrained', group: 'Learned routes' },
]

describe('select control helpers', () => {
  it('finds selected and boundary options while skipping disabled entries', () => {
    expect(options.map((option) => option.group)).toEqual([
      'Feature routes',
      'Feature routes',
      'Learned routes',
    ])
    expect(selectedOptionIndex(options, 'pretrained')).toBe(2)
    expect(boundaryEnabledOptionIndex(options, 'first')).toBe(0)
    expect(boundaryEnabledOptionIndex(options, 'last')).toBe(2)
  })

  it('wraps keyboard navigation and skips disabled entries', () => {
    expect(nextEnabledOptionIndex(options, 0, 1)).toBe(2)
    expect(nextEnabledOptionIndex(options, 2, 1)).toBe(0)
    expect(nextEnabledOptionIndex(options, 0, -1)).toBe(2)
  })

  it('matches typeahead labels from the option after the current one', () => {
    const repeated: SelectOption[] = [
      { value: 'pep', label: 'Peptide' },
      { value: 'protein', label: 'Protein' },
      { value: 'pair', label: 'Pair' },
    ]
    expect(findTypeaheadOptionIndex(repeated, 'p', 0)).toBe(1)
    expect(findTypeaheadOptionIndex(repeated, 'pai', 1)).toBe(2)
  })

  it('places the menu below when space is available', () => {
    expect(
      resolveSelectMenuPlacement({
        trigger: { top: 100, bottom: 148, left: 24, width: 320 },
        menuHeight: 220,
        viewportWidth: 1024,
        viewportHeight: 768,
      }),
    ).toEqual({ placement: 'bottom', top: 154, left: 24, width: 320, maxHeight: 220 })
  })

  it('flips above and constrains the menu near the viewport edge', () => {
    expect(
      resolveSelectMenuPlacement({
        trigger: { top: 650, bottom: 698, left: 760, width: 320 },
        menuHeight: 260,
        viewportWidth: 900,
        viewportHeight: 720,
      }),
    ).toEqual({ placement: 'top', top: 384, left: 572, width: 320, maxHeight: 260 })
  })
})
