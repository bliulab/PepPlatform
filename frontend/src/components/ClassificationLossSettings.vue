<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import SelectControl from './SelectControl.vue'
import { classificationLossParametersValid } from '../classificationLoss'

const props = defineProps<{ isPairTask: boolean; defaultLoss: string }>()
const overrides = defineModel<Record<string, unknown>>({ required: true })
const { t } = useI18n()
const lossKey = computed(() => (props.isPairTask ? 'loss_type' : 'multilabel_loss'))
const capKey = computed(() => (props.isPairTask ? 'pos_weight_max' : 'multilabel_pos_weight_cap'))
const selectedLoss = computed({
  get: () => {
    const value = overrides.value[lossKey.value]
    return value === undefined || value === 'auto' ? 'inherit' : String(value)
  },
  set: (value: string) => {
    if (value === 'inherit') {
      overrides.value = {}
      return
    }
    const keys =
      value === 'asymmetric_focal'
        ? ['asl_gamma_pos', 'asl_gamma_neg', 'asl_clip']
        : value === 'wbce' || value === 'balanced_bce'
          ? [capKey.value]
          : []
    overrides.value = {
      [lossKey.value]: value,
      ...Object.fromEntries(
        keys.filter((key) => key in overrides.value).map((key) => [key, overrides.value[key]]),
      ),
    }
  },
})
const options = computed(() => [
  { value: 'inherit', label: t('wizard.classificationLoss.inherit', { loss: props.defaultLoss }) },
  { value: 'bce', label: 'BCE' },
  { value: props.isPairTask ? 'wbce' : 'balanced_bce', label: 'WBCE' },
  ...(!props.isPairTask ? [{ value: 'asymmetric_focal', label: 'ASL' }] : []),
])
function parameter(key: () => string, fallback: () => number) {
  return computed({
    get: () => (overrides.value[key()] ?? fallback()) as number | string,
    set: (value: number | string) => {
      overrides.value = { ...overrides.value, [key()]: value }
    },
  })
}
const cap = parameter(() => capKey.value, () => (props.isPairTask ? 50 : 20))
const gammaPos = parameter(() => 'asl_gamma_pos', () => 1)
const gammaNeg = parameter(() => 'asl_gamma_neg', () => 4)
const clip = parameter(() => 'asl_clip', () => 0.05)
const valid = computed(() => classificationLossParametersValid(overrides.value))
</script>

<template>
  <details
    class="advanced-section classification-loss-settings"
    data-testid="classification-loss-settings"
  >
    <summary>{{ t('wizard.classificationLoss.title') }}</summary>
    <p>{{ t('wizard.classificationLoss.hint') }}</p>
    <div class="advanced-grid wizard-field-grid">
      <SelectControl
        v-model="selectedLoss"
        :label="t('wizard.classificationLoss.label')"
        :options="options"
      />
      <label
        v-if="selectedLoss === 'wbce' || selectedLoss === 'balanced_bce'"
        class="grid gap-2 text-sm"
      >
        <span class="wizard-field-label">{{ t('wizard.classificationLoss.cap') }}</span>
        <input v-model.number="cap" type="number" min="1" step="1" class="field-control" />
        <small class="leading-5 text-slate-500">{{ t('wizard.classificationLoss.trainOnly') }}</small>
      </label>
      <template v-if="selectedLoss === 'asymmetric_focal'">
        <label class="grid gap-2 text-sm">
          <span class="wizard-field-label">{{ t('wizard.classificationLoss.gammaPos') }}</span>
          <input v-model.number="gammaPos" type="number" min="0" step="0.1" class="field-control" />
        </label>
        <label class="grid gap-2 text-sm">
          <span class="wizard-field-label">{{ t('wizard.classificationLoss.gammaNeg') }}</span>
          <input v-model.number="gammaNeg" type="number" min="0" step="0.1" class="field-control" />
        </label>
        <label class="grid gap-2 text-sm">
          <span class="wizard-field-label">{{ t('wizard.classificationLoss.clip') }}</span>
          <input v-model.number="clip" type="number" min="0" max="1" step="0.01" class="field-control" />
        </label>
      </template>
    </div>
    <p v-if="!valid" role="alert" class="loss-validation-error">
      {{ t('wizard.classificationLoss.invalidParameters') }}
    </p>
    <p v-if="isPairTask" class="text-xs">{{ t('wizard.classificationLoss.pairHint') }}</p>
  </details>
</template>

<style scoped>
.classification-loss-settings > summary {
  width: fit-content;
  cursor: pointer;
  color: var(--color-ink);
  font-size: 0.95rem;
  font-weight: 650;
  letter-spacing: -0.015em;
}
.classification-loss-settings > summary:focus-visible {
  outline: 2px solid var(--color-brass);
  outline-offset: 4px;
  border-radius: 4px;
}
.classification-loss-settings .loss-validation-error {
  color: var(--color-danger, #b91c1c);
}
</style>
