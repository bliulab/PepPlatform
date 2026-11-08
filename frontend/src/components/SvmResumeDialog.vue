<script setup lang="ts">
import { RotateCcw, X } from '@lucide/vue'
import { computed, nextTick, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import type { TrainingFailure } from '../types'

const props = withDefaults(
  defineProps<{
    open: boolean
    failure?: TrainingFailure | null
    config?: Record<string, unknown>
    pending?: boolean
    error?: string
  }>(),
  {
    failure: null,
    config: () => ({}),
    pending: false,
    error: '',
  },
)

const emit = defineEmits<{
  close: []
  submit: [overrides: Record<string, unknown>]
}>()

const { t } = useI18n()
const dialog = ref<HTMLElement>()
const firstField = ref<HTMLInputElement>()
const previouslyFocused = ref<HTMLElement | null>(null)
const svmMaxIter = ref(50000)
const svmTol = ref(0.001)
const mlScaler = ref('standard')
const svmC = ref(1)
const svmKernel = ref('linear')
const svmGamma = ref('scale')
const svmClassWeight = ref('balanced')
const mlReduction = ref('none')
const mlReductionDim = ref(128)
const pcaMemoryLimitGib = ref(4.0)
const kpcaMemoryLimitGib = ref(4.0)
const kpcaKernel = ref('rbf')
const kpcaGamma = ref(0.1)
const mlAutoTune = ref(false)

function finiteNumber(value: unknown, fallback: number) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function textValue(value: unknown, fallback: string) {
  const text = typeof value === 'string' ? value.trim() : ''
  return text || fallback
}

function resetForm() {
  const failedBudget = finiteNumber(props.failure?.context?.max_iter, 0)
  const configuredBudget = finiteNumber(props.config.svm_max_iter, 2000)
  svmMaxIter.value = Math.max(1, Math.ceil(Math.max(failedBudget, configuredBudget) * 2))
  svmTol.value = finiteNumber(
    props.failure?.context?.tol,
    finiteNumber(props.config.svm_tol, 0.001),
  )
  mlScaler.value = textValue(
    props.failure?.context?.scaler,
    textValue(props.config.ml_scaler, 'standard'),
  )
  svmC.value = finiteNumber(props.config.svm_c, 1)
  svmKernel.value = textValue(props.config.svm_kernel, 'linear')
  svmGamma.value = textValue(props.config.svm_gamma, 'scale')
  svmClassWeight.value = textValue(props.config.svm_class_weight, 'balanced')
  mlReduction.value = textValue(props.config.ml_reduction, 'none')
  mlReductionDim.value = finiteNumber(props.config.ml_reduction_dim, 128)
  pcaMemoryLimitGib.value = finiteNumber(props.config.pca_dense_memory_limit_gib, 4.0)
  kpcaMemoryLimitGib.value = finiteNumber(props.config.kpca_memory_limit_gib, 4.0)
  kpcaKernel.value = textValue(props.config.kpca_kernel, 'rbf')
  kpcaGamma.value = finiteNumber(props.config.kpca_gamma, 0.1)
  mlAutoTune.value = Boolean(props.config.ml_auto_tune)
}

watch(
  () => props.open,
  async (open) => {
    if (!open) {
      await nextTick()
      previouslyFocused.value?.focus()
      previouslyFocused.value = null
      return
    }
    previouslyFocused.value = document.activeElement as HTMLElement | null
    resetForm()
    await nextTick()
    firstField.value?.focus()
  },
)

const sourceBudget = computed(() =>
  finiteNumber(
    props.failure?.context?.initial_max_iter,
    finiteNumber(props.config.svm_max_iter, 2000),
  ),
)

function close() {
  if (!props.pending) emit('close')
}

function submit() {
  const overrides: Record<string, unknown> = {
    svm_max_iter: svmMaxIter.value,
    svm_tol: svmTol.value,
    ml_scaler: mlScaler.value,
    svm_c: svmC.value,
    svm_kernel: svmKernel.value,
    svm_gamma: svmGamma.value,
    svm_class_weight: svmClassWeight.value,
    ml_reduction: mlReduction.value,
    ml_reduction_dim: mlReductionDim.value,
    ml_auto_tune: mlAutoTune.value,
    svm_convergence_policy: 'bounded_retry',
  }
  if (mlReduction.value === 'pca') {
    overrides.pca_dense_memory_limit_gib = pcaMemoryLimitGib.value
  } else if (mlReduction.value === 'kpca') {
    overrides.kpca_memory_limit_gib = kpcaMemoryLimitGib.value
    overrides.kpca_kernel = kpcaKernel.value
    if (kpcaKernel.value === 'rbf' || kpcaKernel.value === 'poly') {
      overrides.kpca_gamma = kpcaGamma.value
    }
  }
  emit('submit', overrides)
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape') {
    event.preventDefault()
    close()
    return
  }
  if (event.key !== 'Tab' || !dialog.value) return
  const focusable = Array.from(
    dialog.value.querySelectorAll<HTMLElement>(
      'button:not([disabled]), input:not([disabled]), select:not([disabled]), summary, [tabindex]:not([tabindex="-1"])',
    ),
  )
  if (!focusable.length) return
  const first = focusable[0]
  const last = focusable[focusable.length - 1]
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault()
    last.focus()
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault()
    first.focus()
  }
}
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="svm-resume-backdrop" @click.self="close">
      <section
        ref="dialog"
        class="svm-resume-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="svm-resume-title"
        aria-describedby="svm-resume-description"
        @keydown="onKeydown"
      >
        <header class="dialog-head">
          <div>
            <p class="eyebrow">{{ t('detail.svmResume.eyebrow') }}</p>
            <h2 id="svm-resume-title">{{ t('detail.svmResume.title') }}</h2>
          </div>
          <button
            type="button"
            class="icon-button"
            :aria-label="t('detail.svmResume.close')"
            :disabled="pending"
            @click="close"
          >
            <X aria-hidden="true" />
          </button>
        </header>

        <p id="svm-resume-description" class="dialog-intro">
          {{ t('detail.svmResume.description') }}
        </p>
        <div class="lineage-note">
          <strong>{{ t('detail.svmResume.reusedTitle') }}</strong>
          <span>{{ t('detail.svmResume.reusedBody') }}</span>
        </div>

        <form class="dialog-form" @submit.prevent="submit">
          <div class="field-grid">
            <label>
              <span>{{ t('detail.svmResume.maxIter') }}</span>
              <input
                ref="firstField"
                v-model.number="svmMaxIter"
                class="field-control"
                type="number"
                min="1"
                required
              />
              <small>{{ t('detail.svmResume.maxIterHint', { previous: sourceBudget }) }}</small>
            </label>
            <label>
              <span>{{ t('detail.svmResume.tolerance') }}</span>
              <input
                v-model.number="svmTol"
                class="field-control"
                type="number"
                min="0.000000001"
                step="any"
                required
              />
              <small>{{ t('detail.svmResume.toleranceHint') }}</small>
            </label>
            <label>
              <span>{{ t('detail.svmResume.scaler') }}</span>
              <select v-model="mlScaler" class="field-control">
                <option value="standard">standard</option>
                <option value="minmax">minmax</option>
                <option value="none">none</option>
              </select>
              <small>{{ t('detail.svmResume.scalerHint') }}</small>
            </label>
          </div>

          <details class="advanced-fields">
            <summary>{{ t('detail.svmResume.advanced') }}</summary>
            <div class="field-grid">
              <label
                ><span>C</span
                ><input
                  v-model.number="svmC"
                  class="field-control"
                  type="number"
                  min="0.000000001"
                  step="any"
                  required
              /></label>
              <label
                ><span>kernel</span
                ><select v-model="svmKernel" class="field-control"
                  ><option value="linear">linear</option
                  ><option value="rbf">rbf</option></select
                ></label
              >
              <label v-if="svmKernel === 'rbf'"
                ><span>gamma</span
                ><input v-model.trim="svmGamma" class="field-control" type="text" required
              /></label>
              <label
                ><span>class_weight</span
                ><select v-model="svmClassWeight" class="field-control"
                  ><option value="balanced">balanced</option
                  ><option value="none">none</option></select
                ></label
              >
              <label
                ><span>ml_reduction</span
                ><select v-model="mlReduction" class="field-control"
                  ><option value="none">none</option
                  ><option value="pca">pca</option
                  ><option value="kpca">kpca</option
                  ><option value="tsvd">tsvd</option></select
                ></label
              >
              <label v-if="mlReduction !== 'none'"
                ><span>ml_reduction_dim</span
                ><input
                  v-model.number="mlReductionDim"
                  class="field-control"
                  type="number"
                  min="1"
                  required
              /></label>
              <label v-if="mlReduction === 'pca'"
                ><span>pca_dense_memory_limit_gib</span
                ><input
                  v-model.number="pcaMemoryLimitGib"
                  class="field-control"
                  type="number"
                  min="0.01"
                  step="0.25"
                  required
              /></label>
              <label v-if="mlReduction === 'kpca'"
                ><span>kpca_memory_limit_gib</span
                ><input
                  v-model.number="kpcaMemoryLimitGib"
                  class="field-control"
                  type="number"
                  min="0.01"
                  step="0.25"
                  required
              /></label>
              <label v-if="mlReduction === 'kpca'"
                ><span>kpca_kernel</span
                ><select v-model="kpcaKernel" class="field-control"
                  ><option value="rbf">rbf</option
                  ><option value="linear">linear</option
                  ><option value="poly">poly</option
                  ><option value="cosine">cosine</option></select
                ></label
              >
              <label v-if="mlReduction === 'kpca' && ['rbf', 'poly'].includes(kpcaKernel)"
                ><span>kpca_gamma</span
                ><input
                  v-model.number="kpcaGamma"
                  class="field-control"
                  type="number"
                  min="0.000000001"
                  step="any"
                  required
              /></label>
            </div>
            <label class="check-field"
              ><input v-model="mlAutoTune" type="checkbox" /><span>{{
                t('detail.svmResume.autoTune')
              }}</span></label
            >
          </details>

          <p class="policy-note">{{ t('detail.svmResume.policyNote') }}</p>
          <p v-if="error" class="dialog-error" role="alert">{{ error }}</p>

          <footer class="dialog-actions">
            <button
              type="button"
              class="secondary-action compact-action"
              :disabled="pending"
              @click="close"
            >
              {{ t('detail.svmResume.cancel') }}
            </button>
            <button type="submit" class="primary-action compact-action" :disabled="pending">
              <RotateCcw aria-hidden="true" />
              {{ pending ? t('detail.svmResume.creating') : t('detail.svmResume.submit') }}
            </button>
          </footer>
        </form>
      </section>
    </div>
  </Teleport>
</template>

<style scoped>
.svm-resume-backdrop {
  position: fixed;
  inset: 0;
  z-index: 80;
  display: grid;
  place-items: center;
  overflow-y: auto;
  background: color-mix(in srgb, var(--color-ink) 55%, transparent);
  padding: 1rem;
}
.svm-resume-dialog {
  width: min(46rem, 100%);
  max-height: calc(100vh - 2rem);
  overflow-y: auto;
  border: 1px solid var(--color-line);
  border-radius: 1rem;
  background: var(--surface-strong);
  box-shadow: 0 24px 70px rgb(28 25 23 / 28%);
  padding: clamp(1rem, 3vw, 1.6rem);
  color: var(--color-ink);
}
.dialog-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
}
.dialog-head h2 {
  margin-top: 0.35rem;
  font-family: var(--font-display);
  font-size: clamp(1.25rem, 3vw, 1.65rem);
}
.icon-button {
  display: grid;
  flex: 0 0 auto;
  width: 2.4rem;
  height: 2.4rem;
  place-items: center;
  border: 1px solid var(--line-soft);
  border-radius: 0.65rem;
  background: var(--surface-muted);
}
.icon-button svg {
  width: 1rem;
}
.dialog-intro {
  margin-top: 1rem;
  max-width: 65ch;
  color: var(--color-muted);
  font-size: 0.875rem;
  line-height: 1.65;
}
.lineage-note {
  display: grid;
  gap: 0.2rem;
  margin-top: 1rem;
  border-left: 3px solid var(--color-accent);
  background: var(--surface-muted);
  padding: 0.8rem 1rem;
  font-size: 0.8rem;
  line-height: 1.5;
}
.lineage-note span {
  color: var(--color-muted);
}
.dialog-form {
  margin-top: 1.2rem;
}
.field-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.9rem;
}
.field-grid label {
  display: grid;
  align-content: start;
  gap: 0.4rem;
  font-size: 0.8rem;
  font-weight: 700;
}
.field-grid small {
  color: var(--color-muted);
  font-size: 0.7rem;
  font-weight: 400;
  line-height: 1.45;
}
.advanced-fields {
  margin-top: 1rem;
  border: 1px solid var(--line-soft);
  border-radius: 0.75rem;
  padding: 0.8rem;
}
.advanced-fields summary {
  cursor: pointer;
  font-size: 0.8rem;
  font-weight: 700;
}
.advanced-fields .field-grid {
  margin-top: 0.9rem;
}
.check-field {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin-top: 0.9rem;
  font-size: 0.8rem;
  font-weight: 700;
}
.policy-note {
  margin-top: 1rem;
  color: var(--color-muted);
  font-size: 0.75rem;
  line-height: 1.55;
}
.dialog-error {
  margin-top: 0.8rem;
  border: 1px solid var(--color-danger-border);
  border-radius: 0.65rem;
  background: var(--color-danger-bg);
  padding: 0.7rem 0.8rem;
  color: var(--color-danger);
  font-size: 0.78rem;
}
.dialog-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.65rem;
  margin-top: 1.2rem;
  border-top: 1px solid var(--line-soft);
  padding-top: 1rem;
}
button:focus-visible,
input:focus-visible,
select:focus-visible,
summary:focus-visible {
  outline: 2px solid var(--color-accent);
  outline-offset: 2px;
}
@media (max-width: 640px) {
  .field-grid {
    grid-template-columns: 1fr;
  }
  .dialog-actions {
    display: grid;
  }
  .dialog-actions button {
    justify-content: center;
  }
}
@media (prefers-reduced-motion: reduce) {
  .svm-resume-backdrop,
  .svm-resume-dialog {
    scroll-behavior: auto;
  }
}
</style>
