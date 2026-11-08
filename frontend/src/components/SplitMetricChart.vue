<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { chartPalette, readChartMode } from '../chartTheme'

const props = defineProps<{ summary: Record<string, unknown> }>()
const { t } = useI18n()
const theme = ref<'light' | 'dark'>('light')
let themeObserver: MutationObserver | undefined

const palette = computed(() => chartPalette(theme.value))

onMounted(() => {
  theme.value = readChartMode()
  themeObserver = new MutationObserver(() => {
    theme.value = readChartMode()
  })
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  })
})

onBeforeUnmount(() => {
  themeObserver?.disconnect()
})

function metricValue(key: string) {
  const value = Number(props.summary[key])
  return Number.isFinite(value) ? value : undefined
}

function formatMetric(value?: number) {
  return value === undefined ? '-' : value.toFixed(4)
}

function barWidth(value?: number) {
  if (value === undefined) return '0%'
  if (value <= 0) return '0%'
  return `${Math.max(2, Math.min(100, Math.round(value * 100)))}%`
}

const rows = computed(() =>
  [
    { label: 'F1', validation: metricValue('val_f1'), test: metricValue('test_f1') },
    { label: 'AUC (primary)', validation: metricValue('val_auc'), test: metricValue('test_auc') },
    {
      label: 'Micro AUC',
      validation: metricValue('val_auc_micro'),
      test: metricValue('test_auc_micro'),
    },
    {
      label: 'Micro AUPR',
      validation: metricValue('val_aupr_micro'),
      test: metricValue('test_aupr_micro'),
    },
    { label: 'Accuracy', validation: metricValue('val_acc'), test: metricValue('test_acc') },
    {
      label: 'Balanced Accuracy',
      validation: metricValue('val_balanced_accuracy'),
      test: metricValue('test_balanced_accuracy'),
    },
    {
      label: t('detail.sampleAccuracy'),
      validation: metricValue('val_sacc'),
      test: metricValue('test_sacc'),
    },
    {
      label: 'Precision',
      validation: metricValue('val_precision'),
      test: metricValue('test_precision'),
    },
    {
      label: 'Recall (Sensitivity)',
      validation: metricValue('val_recall'),
      test: metricValue('test_recall'),
    },
    {
      label: 'Specificity',
      validation: metricValue('val_specificity'),
      test: metricValue('test_specificity'),
    },
    { label: 'MCC', validation: metricValue('val_mcc'), test: metricValue('test_mcc') },
  ].filter((row) => row.validation !== undefined || row.test !== undefined),
)
</script>

<template>
  <div class="mt-5 space-y-4">
    <div
      v-for="row in rows"
      :key="row.label"
      class="rounded-lg border p-4"
      style="background: var(--surface-bg); border-color: var(--line-soft)"
    >
      <div class="flex items-center justify-between gap-3">
        <div class="min-w-0 break-words text-sm font-black">{{ row.label }}</div>
        <div class="flex flex-wrap items-center justify-end gap-3 text-xs text-slate-500">
          <span>{{ t('detail.validation') }} {{ formatMetric(row.validation) }}</span>
          <span>{{ t('detail.test') }} {{ formatMetric(row.test) }}</span>
        </div>
      </div>
      <div class="mt-3 grid gap-2">
        <div class="h-3 rounded-full" style="background: var(--surface-muted)">
          <div
            class="h-3 rounded-full"
            :style="{ width: barWidth(row.validation), background: palette.splitValidation }"
          />
        </div>
        <div class="h-3 rounded-full" style="background: var(--surface-muted)">
          <div
            class="h-3 rounded-full"
            :style="{ width: barWidth(row.test), background: palette.splitTest }"
          />
        </div>
      </div>
    </div>
    <div class="flex flex-wrap gap-3 text-xs font-semibold">
      <span class="config-chip">
        <span
          class="mr-2 inline-block h-2 w-2 rounded-full"
          :style="{ background: palette.splitValidation }"
        />
        {{ t('detail.validation') }}
      </span>
      <span class="config-chip">
        <span
          class="mr-2 inline-block h-2 w-2 rounded-full"
          :style="{ background: palette.splitTest }"
        />
        {{ t('detail.test') }}
      </span>
    </div>
  </div>
</template>
