export function cleanSequence(value: string) {
  return value.replace(/\s+/g, '').toUpperCase()
}

export function splitTableLine(line: string, delimiter: ',' | '\t') {
  const cells: string[] = []
  let cell = ''
  let quoted = false
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index]
    const next = line[index + 1]
    if (char === '"' && quoted && next === '"') {
      cell += '"'
      index += 1
    } else if (char === '"') {
      quoted = !quoted
    } else if (char === delimiter && !quoted) {
      cells.push(cell.trim())
      cell = ''
    } else {
      cell += char
    }
  }
  cells.push(cell.trim())
  return cells
}

export function tableDelimiter(text: string, filename: string): ',' | '\t' {
  const extension = filename.toLowerCase().match(/\.([^.]+)$/)?.[1]
  if (extension === 'csv') return ','
  if (extension === 'tsv') return '\t'
  if (extension && extension !== 'txt') return ','
  const header = text.split(/\r?\n/).find((line) => line.trim()) || ''
  // Scan only the header, respecting the same quoting rules as the row parser.
  if (splitTableLine(header, ',').length > 1) return ','
  return splitTableLine(header, '\t').length > 1 ? '\t' : ','
}

export function parseTableRows(text: string, filename: string) {
  const delimiter = tableDelimiter(text, filename)
  const lines = text.split(/\r?\n/).filter((line) => line.trim())
  if (lines.length < 2) return { headers: [] as string[], rows: [] as string[][] }
  return {
    headers: splitTableLine(lines[0], delimiter).map((header) => header.trim().toLowerCase()),
    rows: lines.slice(1).map((line) => splitTableLine(line, delimiter)),
  }
}

function parseFastaSequences(text: string) {
  const rows: string[] = []
  let current: string[] = []
  text.split(/\r?\n/).forEach((rawLine) => {
    const line = rawLine.trim()
    if (!line) return
    if (line.startsWith('>')) {
      if (current.length) rows.push(cleanSequence(current.join('')))
      current = []
      return
    }
    current.push(line)
  })
  if (current.length) rows.push(cleanSequence(current.join('')))
  return rows.filter(Boolean)
}

export function parseSequenceFile(text: string, filename: string, missingColumnsMessage: string) {
  if (text.trimStart().startsWith('>')) return parseFastaSequences(text)
  const csv = parseTableRows(text, filename)
  const sequenceColumn = csv.headers.findIndex((header) =>
    ['sequence', 'seq', 'peptide'].includes(header),
  )
  if (sequenceColumn >= 0) {
    return csv.rows.map((row) => cleanSequence(row[sequenceColumn] || '')).filter(Boolean)
  }
  if (tableDelimiter(text, filename) === '\t') throw new Error(missingColumnsMessage)
  return text
    .split('\n')
    .map((row) => row.trim())
    .filter((line) => !line.startsWith('>') && !line.includes(','))
    .map(cleanSequence)
    .filter(Boolean)
}

export function parsePairFile(
  text: string,
  filename: string,
  partnerCandidates: string[],
  messages: { columns: string; empty: string },
) {
  const csv = parseTableRows(text, filename)
  const peptideColumn = csv.headers.findIndex((header) => ['peptide', 'pep'].includes(header))
  const proteinColumn = csv.headers.findIndex((header) => partnerCandidates.includes(header))
  if (peptideColumn < 0 || proteinColumn < 0) {
    throw new Error(messages.columns)
  }
  const pairs = csv.rows
    .map((row) => ({
      peptide: cleanSequence(row[peptideColumn] || ''),
      protein: cleanSequence(row[proteinColumn] || ''),
    }))
    .filter((row) => row.peptide && row.protein)
  if (!pairs.length) throw new Error(messages.empty)
  return pairs
}
