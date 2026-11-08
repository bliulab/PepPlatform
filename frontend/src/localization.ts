import type { ExperimentRecord, GeneratedExperimentName, ModelRecord } from './types'
import { buildModelDefaultName } from './modelDisplay'

export type SupportedLocale = 'en' | 'zh'
export type Translate = (key: string, params?: Record<string, unknown>) => string

const RERUN_SUFFIXES = [' · rerun', ' · \u91cd\u8dd1'] as const

export function normalizeLocale(value: unknown): SupportedLocale {
  return String(value || '')
    .toLowerCase()
    .startsWith('zh')
    ? 'zh'
    : 'en'
}

export function intlLocale(value: unknown) {
  return normalizeLocale(value) === 'zh' ? 'zh-CN' : 'en-US'
}

export function formatLocalizedDate(
  value: string | Date | null | undefined,
  locale: unknown,
  options: Intl.DateTimeFormatOptions,
  fallback = '',
) {
  if (!value) return fallback
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return fallback
  return new Intl.DateTimeFormat(intlLocale(locale), options).format(date)
}

export function legacyGeneratedName(
  name: string,
): Extract<GeneratedExperimentName, { kind: 'rerun' }> | null {
  let baseName = String(name || '')
  let count = 0
  while (true) {
    const suffix = RERUN_SUFFIXES.find((candidate) => baseName.endsWith(candidate))
    if (!suffix) break
    baseName = baseName.slice(0, -suffix.length)
    count += 1
  }
  baseName = baseName.trim()
  return baseName && count
    ? { kind: 'rerun', base_name: baseName, base_kind: 'literal', count }
    : null
}

export function generatedNameForDisplay(
  rawName: string,
  generatedName: GeneratedExperimentName | null | undefined,
  translate: Translate,
  config?: Record<string, unknown>,
  taskType?: string,
) {
  if (generatedName?.kind === 'config') {
    return config
      ? buildModelDefaultName(config, taskType, translate, generatedName.qualifier)
      : rawName
  }
  const generated = generatedName?.kind === 'rerun' ? generatedName : legacyGeneratedName(rawName)
  if (!generated?.base_name) return rawName
  const baseName =
    generated.base_kind === 'config' && config
      ? buildModelDefaultName(config, taskType, translate, generated.qualifier)
      : generated.base_name
  const count = Math.max(1, Number(generated.count) || 1)
  return count === 1
    ? translate('generatedNames.rerun', { name: baseName })
    : translate('generatedNames.rerunCount', { name: baseName, count })
}

export function experimentDisplayName(experiment: ExperimentRecord, translate: Translate) {
  return generatedNameForDisplay(
    experiment.name,
    experiment.generated_name,
    translate,
    experiment.config,
    experiment.task_type,
  )
}

export function nextRerunNameForDisplay(
  rawName: string,
  generatedName: GeneratedExperimentName | null | undefined,
  translate: Translate,
  config?: Record<string, unknown>,
  taskType?: string,
) {
  const sourceGenerated = generatedName || legacyGeneratedName(rawName)
  const configBaseName = config
    ? buildModelDefaultName(config, taskType, translate, sourceGenerated?.qualifier)
    : rawName
  const baseName =
    sourceGenerated?.kind === 'config'
      ? configBaseName
      : sourceGenerated?.kind === 'rerun' && sourceGenerated.base_kind === 'config'
        ? configBaseName
        : sourceGenerated?.kind === 'rerun'
          ? sourceGenerated.base_name
          : rawName
  const count =
    Math.max(0, Number(sourceGenerated?.kind === 'rerun' ? sourceGenerated.count : 0) || 0) + 1
  return count === 1
    ? translate('generatedNames.rerun', { name: baseName })
    : translate('generatedNames.rerunCount', { name: baseName, count })
}

export function modelDisplayName(
  model: ModelRecord,
  experiment: ExperimentRecord | undefined,
  translate: Translate,
) {
  const inheritsExperimentName =
    model.name_is_custom === false ||
    (model.name_is_custom === undefined && experiment?.name === model.name)
  if (inheritsExperimentName && experiment) {
    return experimentDisplayName(experiment, translate)
  }
  return model.name_is_custom
    ? model.name
    : generatedNameForDisplay(model.name, undefined, translate)
}
