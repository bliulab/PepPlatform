import { describe, expect, it } from 'vitest'

import { groupHeadline, splitHeadline, usesWordSpacing } from './headline'

describe('headline splitting', () => {
  it('splits english into words', () => {
    expect(splitHeadline('Four peptide tasks, one experiment workspace.')).toEqual([
      'Four',
      'peptide',
      'tasks,',
      'one',
      'experiment',
      'workspace.',
    ])
  })

  it('collapses repeated whitespace and trims', () => {
    expect(splitHeadline('  Four   peptide \n tasks  ')).toEqual(['Four', 'peptide', 'tasks'])
  })

  it('splits chinese per character', () => {
    const parts = splitHeadline('四类肽任务，一个实验工作台。')
    expect(parts[0]).toBe('四')
    expect(parts).toHaveLength(14)
    expect(parts.join('')).toBe('四类肽任务，一个实验工作台。')
  })

  it('keeps latin runs intact inside chinese headlines', () => {
    const parts = splitHeadline('肽–HLA-I 结合预测')
    expect(parts).toContain('HLA-I')
    expect(parts).toContain('肽')
    expect(parts.join('')).toBe('肽–HLA-I结合预测')
  })

  it('returns an empty list for empty input', () => {
    expect(splitHeadline('')).toEqual([])
    expect(splitHeadline('   ')).toEqual([])
  })

  it('reports when word spacing applies', () => {
    expect(usesWordSpacing('Four peptide tasks')).toBe(true)
    expect(usesWordSpacing('四类肽任务')).toBe(false)
  })

  it('groups chinese clauses so lines break only at punctuation', () => {
    const groups = groupHeadline(splitHeadline('四类肽任务，一个实验工作台。'), 'zh')
    expect(groups).toHaveLength(2)
    expect(groups[0].join('')).toBe('四类肽任务，')
    expect(groups[1].join('')).toBe('一个实验工作台。')
  })

  it('groups english per word', () => {
    const groups = groupHeadline(splitHeadline('Four peptide tasks'), 'en')
    expect(groups).toEqual([['Four'], ['peptide'], ['tasks']])
  })
})
