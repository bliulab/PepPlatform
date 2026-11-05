import { describe, expect, it } from 'vitest'
import { sameParams, shortGroup, visibleCellValue } from './heatmaps'

describe('heatmap display contracts', () => {
  it('keeps missing and low-support values distinct from measured zeros', () => {
    const cell = { row: 'A', column: '9', value: 0, count: 2, reason: '', low_support: true }
    expect(visibleCellValue(cell, 'stratified', false)).toBeNull()
    expect(visibleCellValue(cell, 'stratified', true)).toBe(0)
    expect(visibleCellValue(cell, 'labels', false)).toBe(0)
    expect(visibleCellValue({ ...cell, value: null }, 'labels', true)).toBeNull()
  })
  it('detects changes in the displayed scientific question', () => {
    const params = {
      split: 'validation',
      kind: 'patterns' as const,
      pattern: 'truth',
      scale: 'count',
    }
    expect(sameParams(params, { ...params, scale: 'jaccard' })).toBe(false)
    expect(sameParams(params, { ...params, split: 'test' })).toBe(false)
    expect(sameParams(params, { ...params })).toBe(true)
  })
  it('shortens target identifiers only for presentation', () => {
    const id = 'protein:' + 'a'.repeat(64)
    expect(shortGroup(id)).toContain('…')
    expect(shortGroup('DPA1*01:03|DPB1*02:01')).toBe('DPA1*01:03|DPB1*02:01')
  })
})
