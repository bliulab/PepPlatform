<script setup lang="ts">
import * as echarts from 'echarts/core'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { BarChart, LineChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import type {
  AnalysisSplitCurves,
  ConfusionMatrixPayload,
  ExperimentAnalysis,
  ScoreDistributionBin,
  ThresholdSweepPayload,
} from '../types'
import {
  analysisOperatingPoint,
  isMultilabelOperatingPoint,
  thresholdEntries,
} from './scoreDiagnostics'
import { chartPalette, readChartMode } from '../chartTheme'

echarts.use([GridComponent, LegendComponent, TooltipComponent, BarChart, LineChart, CanvasRenderer])

const props = defineProps<{ analysis: ExperimentAnalysis; thresholdDescription?: string }>()
const { t } = useI18n()
const theme = ref<'light' | 'dark'>('light')
const sweepRoots = new Map<string, HTMLElement>()
const distributionRoots = new Map<string, HTMLElement>()
const sweepCharts = new Map<string, echarts.ECharts>()
const distributionCharts = new Map<string, echarts.ECharts>()
let themeObserver: MutationObserver | undefined

const splitKeys = ['validation', 'test'] as const
const metricKeys = ['precision', 'recall', 'f1', 'specificity', 'mcc'] as const

type SplitKey = (typeof splitKeys)[number]

function splitLabel(split: string) {
  return split === 'test' ? t('detail.test') : t('detail.validation')
}

function hasDiagnostics(payload?: AnalysisSplitCurves) {
  return Boolean(
    payload?.confusion_matrix?.available ||
    payload?.threshold_sweep?.available ||
    payload?.score_distribution?.available,
  )
}

const splitItems = computed(() =>
  splitKeys
    .map((key) => ({
      key,
      label: splitLabel(key),
      payload: props.analysis.curves?.[key],
    }))
    .filter((item): item is { key: SplitKey; label: string; payload: AnalysisSplitCurves } =>
      hasDiagnostics(item.payload),
    ),
)

const hasScoreDiagnostics = computed(() => splitItems.value.length > 0)
const operatingPoint = computed(() => analysisOperatingPoint(props.analysis))
const perLabelThresholds = computed(() => thresholdEntries(operatingPoint.value))
const isMultilabel = computed(() => isMultilabelOperatingPoint(operatingPoint.value))
const thresholdCount = computed(() => perLabelThresholds.value.length)

function setSweepRoot(split: string, el: unknown) {
  if (el instanceof HTMLElement) sweepRoots.set(split, el)
  else sweepRoots.delete(split)
}

function setDistributionRoot(split: string, el: unknown) {
  if (el instanceof HTMLElement) distributionRoots.set(split, el)
  else distributionRoots.delete(split)
}

function formatMetric(value: unknown, digits = 3) {
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? numberValue.toFixed(digits) : '-'
}

function formatCount(value: unknown) {
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? String(Math.round(numberValue)) : '0'
}

function f1LabelForAverage(average?: string) {
  return average === 'micro' ? t('detail.microF1') : 'F1'
}

function metricLabel(metric: (typeof metricKeys)[number], average?: string) {
  if (metric === 'f1') return f1LabelForAverage(average)
  if (metric === 'mcc') return 'MCC'
  if (metric === 'specificity') return t('detail.specificity')
  if (metric === 'recall') return t('detail.recall')
  return t('detail.precision')
}

function thresholdSourceLabel(source?: string, average?: string) {
  if (source === 'validation_micro_f1') return t('detail.thresholdSourceValidationMicro')
  if (source === 'split_micro_f1') return t('detail.thresholdSourceSplitMicro')
  if (average === 'micro') {
    if (source === 'validation_f1') return t('detail.thresholdSourceValidationMicro')
    if (source === 'split_f1') return t('detail.thresholdSourceSplitMicro')
  }
  if (source === 'validation_f1') return t('detail.thresholdSourceValidation')
  if (source === 'split_f1') return t('detail.thresholdSourceSplit')
  if (source === 'provided') return t('detail.thresholdSourceProvided')
  return source || t('detail.thresholdSourceSplit')
}

function confusionCells(payload?: ConfusionMatrixPayload) {
  const total = Math.max(0, Number(payload?.total ?? 0))
  const cells = [
    { key: 'tn', label: t('detail.trueNegative'), value: payload?.tn, correct: true },
    { key: 'fp', label: t('detail.falsePositive'), value: payload?.fp, correct: false },
    { key: 'fn', label: t('detail.falseNegative'), value: payload?.fn, correct: false },
    { key: 'tp', label: t('detail.truePositive'), value: payload?.tp, correct: true },
  ]
  return cells.map((cell) => {
    const count = Math.max(0, Number(cell.value ?? 0))
    return {
      ...cell,
      count,
      percent: total > 0 ? count / total : 0,
    }
  })
}

function cellStyle(correct: boolean) {
  return correct
    ? {
        background: 'var(--color-success-bg)',
        borderColor: 'var(--color-success-border)',
        color: 'var(--color-success-text)',
      }
    : {
        background: 'var(--color-danger-bg)',
        borderColor: 'var(--color-danger-border)',
        color: 'var(--color-danger)',
      }
}

function chartKeys(kind: 'sweep' | 'distribution') {
  return new Set(
    splitItems.value
      .filter((item) =>
        kind === 'sweep'
          ? !isMultilabel.value && item.payload.threshold_sweep?.available
          : item.payload.score_distribution?.available,
      )
      .map((item) => item.key),
  )
}

function disposeMissingCharts() {
  const activeSweeps = chartKeys('sweep')
  const activeDistributions = chartKeys('distribution')
  for (const [split, chart] of sweepCharts) {
    if (!activeSweeps.has(split as SplitKey)) {
      chart.dispose()
      sweepCharts.delete(split)
    }
  }
  for (const [split, chart] of distributionCharts) {
    if (!activeDistributions.has(split as SplitKey)) {
      chart.dispose()
      distributionCharts.delete(split)
    }
  }
}

function ensureCharts() {
  disposeMissingCharts()
  for (const item of splitItems.value) {
    if (
      !isMultilabel.value &&
      item.payload.threshold_sweep?.available &&
      sweepRoots.get(item.key) &&
      !sweepCharts.has(item.key)
    ) {
      sweepCharts.set(item.key, echarts.init(sweepRoots.get(item.key) as HTMLElement))
    }
    if (
      item.payload.score_distribution?.available &&
      distributionRoots.get(item.key) &&
      !distributionCharts.has(item.key)
    ) {
      distributionCharts.set(item.key, echarts.init(distributionRoots.get(item.key) as HTMLElement))
    }
  }
}

function sweepSeries(sweep: ThresholdSweepPayload, average?: string) {
  const palette = chartPalette(theme.value)
  const points = Array.isArray(sweep.points) ? sweep.points : []
  const series: Array<Record<string, unknown>> = metricKeys.map((metric, index) => ({
    id: metric,
    name: metricLabel(metric, average),
    type: 'line',
    smooth: true,
    showSymbol: false,
    lineStyle: { width: 2 },
    emphasis: { focus: 'series' },
    color: palette.series[index],
    data: points
      .map((point) => [Number(point.threshold), Number(point[metric])])
      .filter(([threshold, value]) => Number.isFinite(threshold) && Number.isFinite(value)),
  }))
  const selected = Number(sweep.selected_threshold)
  if (Number.isFinite(selected)) {
    series.push({
      id: 'selected-threshold',
      name: t('detail.selectedThreshold'),
      type: 'line',
      smooth: false,
      showSymbol: false,
      silent: true,
      lineStyle: { width: 2, type: 'dashed' },
      emphasis: { disabled: true },
      color: palette.selected,
      data: [
        [selected, -1],
        [selected, 1],
      ],
    })
  }
  return series
}

function renderSweep(split: string, payload?: AnalysisSplitCurves) {
  const chart = sweepCharts.get(split)
  const sweep = payload?.threshold_sweep
  if (isMultilabel.value || !chart || !payload || !sweep?.available) return
  const palette = chartPalette(theme.value)
  chart.setOption({
    color: palette.series,
    legend: {
      top: 0,
      right: 0,
      textStyle: { color: palette.axisLabel, fontWeight: 600 },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: palette.tooltipBg,
      borderColor: palette.tooltipBg,
      textStyle: { color: palette.tooltipText },
      valueFormatter: (value: unknown) => formatMetric(value),
    },
    grid: { left: 42, right: 18, top: 48, bottom: 36 },
    xAxis: {
      type: 'value',
      name: t('detail.threshold'),
      axisLine: { lineStyle: { color: palette.axis } },
      axisLabel: { color: palette.axisLabel },
      splitLine: { lineStyle: { color: palette.split } },
    },
    yAxis: {
      type: 'value',
      min: -1,
      max: 1,
      axisLine: { lineStyle: { color: palette.axis } },
      axisLabel: { color: palette.axisLabel },
      splitLine: { lineStyle: { color: palette.split } },
    },
    series: sweepSeries(sweep, payload.average),
  })
}

function binLabel(bin: ScoreDistributionBin) {
  return `${formatMetric(bin.lower, 2)}-${formatMetric(bin.upper, 2)}`
}

function renderDistribution(split: string, payload?: AnalysisSplitCurves['score_distribution']) {
  const chart = distributionCharts.get(split)
  if (!chart || !payload?.available) return
  const palette = chartPalette(theme.value)
  const bins = Array.isArray(payload.bins) ? payload.bins : []
  chart.setOption({
    color: [palette.positive, palette.negative],
    legend: {
      top: 0,
      right: 0,
      textStyle: { color: palette.axisLabel, fontWeight: 600 },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: palette.tooltipBg,
      borderColor: palette.tooltipBg,
      textStyle: { color: palette.tooltipText },
    },
    grid: { left: 44, right: 18, top: 44, bottom: 42 },
    xAxis: {
      type: 'category',
      data: bins.map(binLabel),
      axisLine: { lineStyle: { color: palette.axis } },
      axisLabel: { color: palette.axisLabel, interval: 3 },
    },
    yAxis: {
      type: 'value',
      name: t('detail.count'),
      minInterval: 1,
      axisLine: { lineStyle: { color: palette.axis } },
      axisLabel: { color: palette.axisLabel },
      splitLine: { lineStyle: { color: palette.split } },
    },
    series: [
      {
        id: 'positive',
        name: t('detail.positiveClass'),
        type: 'bar',
        barGap: '-35%',
        data: bins.map((bin) => Number(bin.positive_count ?? 0)),
      },
      {
        id: 'negative',
        name: t('detail.negativeClass'),
        type: 'bar',
        data: bins.map((bin) => Number(bin.negative_count ?? 0)),
      },
    ],
  })
}

function renderCharts() {
  ensureCharts()
  for (const item of splitItems.value) {
    renderSweep(item.key, item.payload)
    renderDistribution(item.key, item.payload.score_distribution)
  }
}

function scheduleRender() {
  void nextTick(() => {
    renderCharts()
    for (const chart of sweepCharts.values()) chart.resize()
    for (const chart of distributionCharts.values()) chart.resize()
  })
}

onMounted(() => {
  theme.value = readChartMode()
  themeObserver = new MutationObserver(() => {
    theme.value = readChartMode()
  })
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  })
  scheduleRender()
})

watch(() => props.analysis, scheduleRender, { deep: true })
watch(theme, scheduleRender)

onBeforeUnmount(() => {
  themeObserver?.disconnect()
  for (const chart of sweepCharts.values()) chart.dispose()
  for (const chart of distributionCharts.values()) chart.dispose()
  sweepCharts.clear()
  distributionCharts.clear()
})
</script>

<template>
  <div>
    <div v-if="hasScoreDiagnostics" class="grid gap-5">
      <section
        v-if="isMultilabel"
        class="section-card overflow-hidden"
        :aria-label="t('detail.perLabelThresholdTitle')"
      >
        <div class="flex flex-wrap items-start justify-between gap-4">
          <div class="max-w-3xl">
            <div class="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
              {{ t('detail.perLabelThresholdTitle') }}
            </div>
            <div class="mt-2 text-lg font-black text-slate-800 dark:text-slate-100">
              {{ t('detail.perLabelThresholdSummary', { count: thresholdCount }) }}
            </div>
            <p class="mt-2 text-sm leading-6 text-slate-500">
              {{ thresholdDescription ?? t('detail.perLabelThresholdDescription') }}
            </p>
          </div>
          <span class="config-chip">
            {{ thresholdSourceLabel(operatingPoint?.threshold_source) }}
          </span>
        </div>

        <details
          class="group mt-4 rounded-lg border border-stone-200 bg-white/50 dark:border-slate-700 dark:bg-slate-900/30"
        >
          <summary
            class="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 text-sm font-bold text-slate-700 outline-none transition hover:text-slate-950 focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-slate-500 dark:text-slate-200 dark:hover:text-white"
          >
            <span>{{ t('detail.viewPerLabelThresholds') }}</span>
            <span aria-hidden="true" class="text-lg transition-transform group-open:rotate-45"
              >＋</span
            >
          </summary>
          <dl
            class="grid gap-px border-t border-stone-200 bg-stone-200 sm:grid-cols-2 lg:grid-cols-3 dark:border-slate-700 dark:bg-slate-700"
          >
            <div
              v-for="entry in perLabelThresholds"
              :key="entry.label"
              class="flex items-center justify-between gap-4 bg-white px-4 py-3 dark:bg-slate-900"
            >
              <dt class="truncate text-sm font-semibold text-slate-600 dark:text-slate-300">
                {{ entry.label }}
              </dt>
              <dd class="font-mono text-sm font-black tabular-nums text-slate-900 dark:text-white">
                {{ formatMetric(entry.value) }}
              </dd>
            </div>
          </dl>
        </details>
      </section>

      <div v-for="item in splitItems" :key="item.key" class="section-card">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div class="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
              {{ item.label }}
            </div>
            <div
              v-if="isMultilabel && item.payload.confusion_matrix?.available"
              class="mt-2 text-sm text-slate-500"
            >
              {{ t('detail.perLabelThresholdAggregate', { count: thresholdCount }) }}
              · {{ t('detail.microAggregate') }}
            </div>
            <div
              v-else-if="item.payload.confusion_matrix?.available"
              class="mt-2 text-sm text-slate-500"
            >
              {{ t('detail.selectedThreshold') }}
              {{ formatMetric(item.payload.confusion_matrix.threshold) }}
              ·
              {{
                thresholdSourceLabel(
                  item.payload.confusion_matrix.threshold_source,
                  item.payload.average,
                )
              }}
            </div>
          </div>
          <div v-if="isMultilabel" class="flex flex-wrap justify-end gap-2">
            <span v-if="item.payload.sample_count" class="config-chip">
              {{ t('detail.sampleCountChip', { count: formatCount(item.payload.sample_count) }) }}
            </span>
            <span v-if="item.payload.label_count" class="config-chip">
              {{ t('detail.labelCountChip', { count: formatCount(item.payload.label_count) }) }}
            </span>
            <span v-if="item.payload.decision_count" class="config-chip">
              {{
                t('detail.decisionCountChip', { count: formatCount(item.payload.decision_count) })
              }}
            </span>
          </div>
          <span v-else-if="item.payload.sample_count" class="config-chip">
            n={{ formatCount(item.payload.sample_count) }}
          </span>
        </div>

        <div
          class="mt-5 grid gap-5"
          :class="
            !isMultilabel && item.payload.threshold_sweep?.available
              ? 'xl:grid-cols-[0.85fr_1.15fr]'
              : 'xl:grid-cols-1'
          "
        >
          <div v-if="item.payload.confusion_matrix?.available" class="min-w-0">
            <div class="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
              {{
                isMultilabel
                  ? t('detail.confusionMatrixMicroAggregate')
                  : t('detail.confusionMatrix')
              }}
            </div>
            <div
              class="mt-3 grid grid-cols-[minmax(4.25rem,auto)_repeat(2,minmax(0,1fr))] gap-2 text-xs"
              role="img"
              :aria-label="t('detail.confusionMatrixAria')"
            >
              <div />
              <div class="break-words text-center font-semibold text-slate-500">
                {{ t('detail.predictedNegative') }}
              </div>
              <div class="break-words text-center font-semibold text-slate-500">
                {{ t('detail.predictedPositive') }}
              </div>
              <div class="flex items-center break-words font-semibold text-slate-500">
                {{ t('detail.actualNegative') }}
              </div>
              <div
                v-for="cell in confusionCells(item.payload.confusion_matrix).slice(0, 2)"
                :key="`${item.key}-${cell.key}`"
                class="min-h-[5rem] rounded-md border p-3"
                :style="cellStyle(cell.correct)"
              >
                <div class="font-bold">{{ cell.label }}</div>
                <div class="mt-2 text-2xl font-black">{{ formatCount(cell.count) }}</div>
                <div class="mt-1 text-xs opacity-80"
                  >{{ formatMetric(cell.percent * 100, 1) }}%</div
                >
              </div>
              <div class="flex items-center break-words font-semibold text-slate-500">
                {{ t('detail.actualPositive') }}
              </div>
              <div
                v-for="cell in confusionCells(item.payload.confusion_matrix).slice(2, 4)"
                :key="`${item.key}-${cell.key}`"
                class="min-h-[5rem] rounded-md border p-3"
                :style="cellStyle(cell.correct)"
              >
                <div class="font-bold">{{ cell.label }}</div>
                <div class="mt-2 text-2xl font-black">{{ formatCount(cell.count) }}</div>
                <div class="mt-1 text-xs opacity-80"
                  >{{ formatMetric(cell.percent * 100, 1) }}%</div
                >
              </div>
            </div>
            <div class="mt-4 grid gap-2 sm:grid-cols-2">
              <span class="config-chip"
                >{{ t('detail.precision') }}
                {{ formatMetric(item.payload.confusion_matrix.precision) }}</span
              >
              <span class="config-chip"
                >{{ t('detail.recall') }}
                {{ formatMetric(item.payload.confusion_matrix.recall) }}</span
              >
              <span class="config-chip"
                >{{ t('detail.specificity') }}
                {{ formatMetric(item.payload.confusion_matrix.specificity) }}</span
              >
              <span class="config-chip"
                >MCC {{ formatMetric(item.payload.confusion_matrix.mcc) }}</span
              >
            </div>
          </div>

          <div v-if="!isMultilabel && item.payload.threshold_sweep?.available" class="min-w-0">
            <div class="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
              {{ t('detail.thresholdSweep') }}
            </div>
            <div
              :ref="(el) => setSweepRoot(item.key, el)"
              class="mt-3 h-72 w-full"
              role="img"
              :aria-label="t('detail.thresholdSweepAria')"
            />
          </div>

          <div v-if="item.payload.score_distribution?.available" class="min-w-0 xl:col-span-2">
            <div class="flex flex-wrap items-center justify-between gap-3">
              <div class="text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
                {{
                  isMultilabel
                    ? t('detail.scoreDistributionMicroAggregate')
                    : t('detail.scoreDistribution')
                }}
              </div>
              <div class="flex flex-wrap gap-2 text-xs font-semibold">
                <span class="config-chip">
                  {{ t('detail.positiveClass') }}
                  mean={{ formatMetric(item.payload.score_distribution.positive?.mean) }}
                </span>
                <span class="config-chip">
                  {{ t('detail.negativeClass') }}
                  mean={{ formatMetric(item.payload.score_distribution.negative?.mean) }}
                </span>
              </div>
            </div>
            <div
              :ref="(el) => setDistributionRoot(item.key, el)"
              class="mt-3 h-64 w-full"
              role="img"
              :aria-label="t('detail.scoreDistributionAria')"
            />
          </div>
        </div>
      </div>
    </div>
    <p v-else class="section-card text-sm text-stone-500">
      {{ t('detail.noScoreDiagnosticsYet') }}
    </p>
  </div>
</template>
