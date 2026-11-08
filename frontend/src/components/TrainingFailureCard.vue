<script setup lang="ts">
import { AlertTriangle, RefreshCw, ScrollText } from '@lucide/vue'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import { trainingFailurePresentation } from '../trainingFailure'
import type { TrainingFailure } from '../types'

const props = withDefaults(
  defineProps<{
    failure?: TrainingFailure | null
    legacyError?: string
    canRerun?: boolean
    canResumeSvm?: boolean
  }>(),
  {
    failure: null,
    legacyError: '',
    canRerun: false,
    canResumeSvm: false,
  },
)

const emit = defineEmits<{
  viewLogs: []
  rerun: []
  resumeSvm: []
}>()

const { t } = useI18n()
const normalizedFailure = computed<TrainingFailure>(
  () =>
    props.failure ?? {
      schema_version: 1,
      code: 'legacy_failure',
      category: 'unknown',
      stage: 'unknown',
      message: props.legacyError || t('detail.failureFallback'),
      context: {},
    },
)
const presentation = computed(() => trainingFailurePresentation(normalizedFailure.value))

function contextValue(key: string) {
  return normalizedFailure.value.context?.[key]
}

function contextText(key: string) {
  const value = contextValue(key)
  return typeof value === 'string' ? value.trim() : ''
}

function contextNumber(key: string) {
  const value = contextValue(key)
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

const feature = computed(() => {
  const value = contextText('feature')
  if (!value) return t('detail.trainingFailure.valueUnknown')
  const names: Record<string, string> = {
    pssm: 'PSSM',
    psfm: 'PSFM',
    psfm_blosum62_hybrid: 'PSFM–BLOSUM62 Hybrid',
    pssm_blosum62_hybrid: 'PSSM–BLOSUM62 Hybrid',
    ac_pssm_blosum62_hybrid: 'AC-PSSM–BLOSUM62 Hybrid',
    cc_pssm_blosum62_hybrid: 'CC-PSSM–BLOSUM62 Hybrid',
    acc_pssm_blosum62_hybrid: 'ACC-PSSM–BLOSUM62 Hybrid',
    target_frequency: 'Target Frequency',
  }
  return names[value.toLowerCase()] || value
})
const sampleId = computed(
  () => contextText('sample_id') || t('detail.trainingFailure.sampleUnknown'),
)
const copyParams = computed(() => ({
  feature: feature.value,
  sampleId: sampleId.value,
  dependency: contextText('dependency') || t('detail.trainingFailure.valueUnknown'),
  outputNumber: contextNumber('output_number') ?? t('detail.trainingFailure.valueUnknown'),
  outputCount: contextNumber('output_count') ?? t('detail.trainingFailure.valueUnknown'),
  budgets: contextText('attempted_max_iters') || t('detail.trainingFailure.valueUnknown'),
  tolerance: contextNumber('tol') ?? t('detail.trainingFailure.valueUnknown'),
}))
const title = computed(() =>
  t(`detail.trainingFailure.codes.${presentation.value.id}.title`, copyParams.value),
)
const description = computed(() =>
  t(`detail.trainingFailure.codes.${presentation.value.id}.description`, copyParams.value),
)
const category = computed(() => {
  const value = String(normalizedFailure.value.category || 'unknown')
  const supported = [
    'profile',
    'resource',
    'dependency',
    'data',
    'configuration',
    'optimization',
    'runtime',
  ]
  return supported.includes(value) ? value : 'unknown'
})
const stage = computed(() => {
  const value = String(normalizedFailure.value.stage || 'unknown')
  const supported = [
    'setup',
    'profile_database',
    'profile_generation',
    'feature_encoding',
    'training',
    'evaluation',
    'artifact_export',
  ]
  return supported.includes(value) ? value : 'unknown'
})
const evidence = computed(() => {
  const items = [
    {
      key: 'stage',
      label: t('detail.trainingFailure.evidence.stage'),
      value: t(`detail.trainingFailure.stages.${stage.value}`),
    },
  ]
  const sample = contextText('sample_id')
  const profileFeature = contextText('feature')
  const sequenceLength = contextValue('sequence_length')
  const hitCount = contextValue('nonself_hit_count')
  const dependency = contextText('dependency')
  const classifier = contextText('classifier')
  const kernel = contextText('kernel')
  const outputNumber = contextNumber('output_number')
  const outputCount = contextNumber('output_count')
  const labelName = contextText('label_name')
  const attemptedBudgets = contextText('attempted_max_iters')
  const actualIterations = contextText('actual_n_iters')
  const tolerance = contextNumber('tol')
  const scaler = contextText('scaler')
  if (sample) {
    items.push({
      key: 'sample',
      label: t('detail.trainingFailure.evidence.sample'),
      value: sample,
    })
  }
  if (profileFeature) {
    items.push({
      key: 'feature',
      label: t('detail.trainingFailure.evidence.feature'),
      value: feature.value,
    })
  }
  if (typeof sequenceLength === 'number') {
    items.push({
      key: 'sequenceLength',
      label: t('detail.trainingFailure.evidence.sequenceLength'),
      value: String(sequenceLength),
    })
  }
  if (typeof hitCount === 'number') {
    items.push({
      key: 'hitCount',
      label: t('detail.trainingFailure.evidence.nonselfHits'),
      value: String(hitCount),
    })
  }
  if (dependency) {
    items.push({
      key: 'dependency',
      label: t('detail.trainingFailure.evidence.dependency'),
      value: dependency,
    })
  }
  if (classifier) {
    items.push({
      key: 'classifier',
      label: t('detail.trainingFailure.evidence.classifier'),
      value: classifier.toUpperCase(),
    })
  }
  if (outputNumber !== undefined) {
    items.push({
      key: 'output',
      label: t('detail.trainingFailure.evidence.output'),
      value: outputCount
        ? t('detail.trainingFailure.evidence.outputValue', {
            current: outputNumber,
            total: outputCount,
            label: labelName || t('detail.trainingFailure.valueUnknown'),
          })
        : String(outputNumber),
    })
  }
  if (kernel) {
    items.push({
      key: 'kernel',
      label: t('detail.trainingFailure.evidence.kernel'),
      value: kernel,
    })
  }
  if (attemptedBudgets) {
    items.push({
      key: 'budgets',
      label: t('detail.trainingFailure.evidence.attemptedBudgets'),
      value: attemptedBudgets,
    })
  }
  if (actualIterations) {
    items.push({
      key: 'actualIterations',
      label: t('detail.trainingFailure.evidence.actualIterations'),
      value: actualIterations,
    })
  }
  if (tolerance !== undefined) {
    items.push({
      key: 'tolerance',
      label: t('detail.trainingFailure.evidence.tolerance'),
      value: String(tolerance),
    })
  }
  if (scaler) {
    items.push({
      key: 'scaler',
      label: t('detail.trainingFailure.evidence.scaler'),
      value: scaler,
    })
  }
  if (normalizedFailure.value.exit_code != null) {
    items.push({
      key: 'exitCode',
      label: t('detail.trainingFailure.evidence.exitCode'),
      value: String(normalizedFailure.value.exit_code),
    })
  }
  return items
})
</script>

<template>
  <section class="training-failure-card" role="alert" aria-labelledby="training-failure-title">
    <header class="training-failure-head">
      <span class="training-failure-mark" aria-hidden="true">
        <AlertTriangle />
      </span>
      <div>
        <p class="training-failure-kicker">
          {{ t(`detail.trainingFailure.categories.${category}`) }}
        </p>
        <h3 id="training-failure-title">{{ title }}</h3>
      </div>
    </header>

    <p class="training-failure-description">{{ description }}</p>

    <dl v-if="evidence.length" class="training-failure-evidence">
      <div v-for="item in evidence" :key="item.key">
        <dt>{{ item.label }}</dt>
        <dd>{{ item.value }}</dd>
      </div>
    </dl>

    <div class="training-failure-next">
      <h4>{{ t('detail.trainingFailure.nextSteps') }}</h4>
      <ul>
        <li v-for="item in presentation.remediations" :key="item">
          {{ t(`detail.trainingFailure.remediations.${item}`) }}
        </li>
      </ul>
    </div>

    <div class="training-failure-actions">
      <button
        v-if="canResumeSvm"
        type="button"
        class="primary-action compact-action"
        @click="emit('resumeSvm')"
      >
        <RefreshCw aria-hidden="true" />
        {{ t('detail.trainingFailure.resumeSvm') }}
      </button>
      <button type="button" class="secondary-action compact-action" @click="emit('viewLogs')">
        <ScrollText aria-hidden="true" />
        {{ t('detail.trainingFailure.viewLogs') }}
      </button>
      <button
        v-if="canRerun"
        type="button"
        class="secondary-action compact-action"
        @click="emit('rerun')"
      >
        <RefreshCw aria-hidden="true" />
        {{ t('detail.rerunOnDataset') }}
      </button>
    </div>

    <details class="training-failure-technical">
      <summary>{{ t('detail.trainingFailure.technicalDetails') }}</summary>
      <dl>
        <div>
          <dt>{{ t('detail.trainingFailure.evidence.errorCode') }}</dt>
          <dd
            ><code>{{ normalizedFailure.code }}</code></dd
          >
        </div>
        <div>
          <dt>{{ t('detail.trainingFailure.evidence.originalMessage') }}</dt>
          <dd>{{ normalizedFailure.message }}</dd>
        </div>
      </dl>
    </details>
  </section>
</template>

<style scoped>
.training-failure-card {
  position: relative;
  overflow: hidden;
  margin-top: 1.25rem;
  border: 1px solid var(--color-danger-border);
  border-left: 4px solid var(--color-danger);
  border-radius: 1rem;
  background: linear-gradient(
    135deg,
    color-mix(in srgb, var(--color-danger-bg) 86%, var(--surface-strong)),
    var(--surface-strong) 72%
  );
  padding: 1.15rem;
  color: var(--color-danger);
}

.training-failure-card::after {
  position: absolute;
  top: -3rem;
  right: -2rem;
  width: 9rem;
  height: 9rem;
  border-radius: 999px;
  background: color-mix(in srgb, var(--color-danger) 8%, transparent);
  content: '';
  pointer-events: none;
}

.training-failure-head {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: flex-start;
  gap: 0.8rem;
}

.training-failure-mark {
  display: inline-grid;
  flex: 0 0 auto;
  width: 2.25rem;
  height: 2.25rem;
  place-items: center;
  border: 1px solid var(--color-danger-border);
  border-radius: 0.75rem;
  background: var(--surface-strong);
}

.training-failure-mark :deep(svg) {
  width: 1.15rem;
  height: 1.15rem;
}

.training-failure-kicker {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.training-failure-head h3 {
  margin-top: 0.2rem;
  color: var(--color-ink);
  font-family: var(--font-display);
  font-size: 1.05rem;
  line-height: 1.35;
}

.training-failure-description {
  position: relative;
  z-index: 1;
  margin-top: 0.85rem;
  max-width: 68ch;
  color: var(--color-ink);
  font-size: 0.875rem;
  line-height: 1.65;
}

.training-failure-evidence {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.55rem;
  margin-top: 0.9rem;
}

.training-failure-evidence > div {
  min-width: 0;
  border: 1px solid color-mix(in srgb, var(--color-danger-border) 72%, var(--surface-strong));
  border-radius: 0.7rem;
  background: color-mix(in srgb, var(--surface-strong) 82%, transparent);
  padding: 0.65rem 0.75rem;
}

.training-failure-evidence dt,
.training-failure-technical dt {
  color: var(--color-muted);
  font-size: 0.68rem;
  font-weight: 700;
}

.training-failure-evidence dd {
  margin-top: 0.15rem;
  overflow-wrap: anywhere;
  color: var(--color-ink);
  font-size: 0.8rem;
  font-weight: 700;
}

.training-failure-next {
  position: relative;
  z-index: 1;
  margin-top: 1rem;
  border-top: 1px solid color-mix(in srgb, var(--color-danger-border) 62%, var(--surface-strong));
  padding-top: 0.85rem;
}

.training-failure-next h4 {
  color: var(--color-ink);
  font-size: 0.78rem;
  font-weight: 700;
}

.training-failure-next ul {
  display: grid;
  gap: 0.35rem;
  margin-top: 0.45rem;
  padding-left: 1.1rem;
  color: var(--color-ink);
  font-size: 0.78rem;
  line-height: 1.55;
}

.training-failure-actions {
  position: relative;
  z-index: 1;
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  margin-top: 1rem;
}

.training-failure-technical {
  position: relative;
  z-index: 1;
  margin-top: 0.9rem;
  border-top: 1px dashed color-mix(in srgb, var(--color-danger-border) 76%, var(--surface-strong));
  padding-top: 0.7rem;
  color: var(--color-muted);
  font-size: 0.75rem;
}

.training-failure-technical summary {
  width: fit-content;
  cursor: pointer;
  font-weight: 700;
}

.training-failure-technical summary:focus-visible {
  border-radius: 0.25rem;
  outline: 2px solid var(--color-accent);
  outline-offset: 3px;
}

.training-failure-technical dl {
  display: grid;
  gap: 0.65rem;
  margin-top: 0.65rem;
}

.training-failure-technical dd {
  margin-top: 0.15rem;
  overflow-wrap: anywhere;
  color: var(--color-ink);
  line-height: 1.55;
}

@media (max-width: 640px) {
  .training-failure-card {
    padding: 1rem;
  }

  .training-failure-evidence {
    grid-template-columns: 1fr;
  }

  .training-failure-actions {
    display: grid;
  }

  .training-failure-actions :deep(.compact-action) {
    min-height: 2.75rem;
    justify-content: center;
  }
}
</style>
