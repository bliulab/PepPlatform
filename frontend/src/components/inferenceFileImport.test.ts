import { describe, expect, it } from 'vitest'
import {
  parsePairFile,
  parseSequenceFile,
  parseTableRows,
  tableDelimiter,
} from './inferenceFileImport'

const messages = { columns: 'Missing pair columns', empty: 'No usable rows' }
const missingSequence = 'Missing sequence column'

describe('inference file import', () => {
  it.each(['csv', 'tsv', 'TSV', 'txt', ''])('imports sequence tables from %s', (extension) => {
    const delimiter = extension === 'csv' ? ',' : '\t'
    const text = ` Sequence${delimiter}note\r\n ac d${delimiter}first\r\nEFG${delimiter}second\r\n`
    expect(
      parseSequenceFile(text, `input${extension ? '.' + extension : ''}`, missingSequence),
    ).toEqual(['ACD', 'EFG'])
  })

  it.each(['protein', 'mhc', 'pair_pseudo', 'hla'])('imports ordered pairs using %s', (partner) => {
    const rows = [
      ['peptide', partner],
      ['ac d', 'ef g'],
      ['HIJ', 'KLM'],
    ]
    const expected = [
      { peptide: 'ACD', protein: 'EFG' },
      { peptide: 'HIJ', protein: 'KLM' },
    ]
    for (const [extension, delimiter] of [
      ['csv', ','],
      ['tsv', '\t'],
      ['txt', '\t'],
    ]) {
      const text = rows.map((row) => row.join(delimiter)).join('\r\n')
      expect(parsePairFile(text, `input.${extension}`, [partner], messages)).toEqual(expected)
    }
  })

  it('keeps CSV delimiters even when headers and fields contain tabs', () => {
    expect(parseTableRows('sequence,"note\tname"\nACD,"x\ty"', 'input.csv')).toEqual({
      headers: ['sequence', 'note\tname'],
      rows: [['ACD', 'x\ty']],
    })
    expect(tableDelimiter('sequence\tnote\nACD\tx', 'input.csv')).toBe(',')
    expect(tableDelimiter('sequence,note\nACD,x', 'input.tsv')).toBe('\t')
  })

  it('uses only the first nonempty header and ignores quoted delimiters', () => {
    expect(tableDelimiter('\r\nsequence,"note\tname"\nACD,x', 'input.txt')).toBe(',')
    expect(tableDelimiter('sequence\t"note,name"\nACD\tx', 'input.txt')).toBe('\t')
    expect(tableDelimiter('sequence,note\textra\nACD,x\ty', 'input.txt')).toBe(',')
    expect(tableDelimiter('sequence\n"ACD\tEFG"', 'input.txt')).toBe(',')
    expect(tableDelimiter('sequence\tnote\nACD\tx', 'input.dat')).toBe(',')
    expect(tableDelimiter('sequence,note\nACD,x', 'tsv')).toBe(',')
  })

  it('keeps one delimiter for all rows and handles escaped quotes in TSV', () => {
    expect(
      parseTableRows(
        'sequence\tnote\r\nACD\t"comma, tab\t and ""quote"""\r\nEFG,x\r\n',
        'input.tsv',
      ),
    ).toEqual({
      headers: ['sequence', 'note'],
      rows: [['ACD', 'comma, tab\t and "quote"'], ['EFG,x']],
    })
  })

  it.each(['input.tsv', 'input.txt', 'input'])(
    'rejects missing TSV sequence columns in %s',
    (filename) => {
      expect(() => parseSequenceFile('id\tnote\n1\tACD', filename, missingSequence)).toThrow(
        missingSequence,
      )
    },
  )

  it('does not treat a single-column TSV with the wrong header as raw sequences', () => {
    expect(() => parseSequenceFile('id\nACD', 'input.tsv', missingSequence)).toThrow(
      missingSequence,
    )
    expect(parseSequenceFile('seq\nACD\nEFG', 'input.tsv', missingSequence)).toEqual(['ACD', 'EFG'])
  })

  it('reports missing pair columns and empty pairs without making any requests', () => {
    expect(() =>
      parsePairFile('peptide\tother\nACD\tEFG', 'input.tsv', ['protein'], messages),
    ).toThrow(messages.columns)
    expect(() =>
      parsePairFile('peptide\tprotein\nACD\t', 'input.tsv', ['protein'], messages),
    ).toThrow(messages.empty)
  })

  it('preserves FASTA, plain text, and CSV sequence behavior', () => {
    expect(
      parseSequenceFile('>one\nac d\nef\n>two\nGHI\n', 'input.fasta', missingSequence),
    ).toEqual(['ACDEF', 'GHI'])
    expect(parseSequenceFile(' acd \n\n efg\n', 'input.txt', missingSequence)).toEqual([
      'ACD',
      'EFG',
    ])
    expect(
      parseSequenceFile('sequence,note\n"ac d","say ""hi"""\nEFG,x', 'input.csv', missingSequence),
    ).toEqual(['ACD', 'EFG'])
    expect(parseSequenceFile('ACD\nEFG\n', 'input.csv', missingSequence)).toEqual(['ACD', 'EFG'])
  })
})
