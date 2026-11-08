<script setup lang="ts">
import * as echarts from 'echarts/core'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { LineChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import type { AnalysisCurvePoint, ExperimentAnalysis } from '../types'
import { chartPalette, readChartMode } from '../chartTheme'

echarts.use([GridComponent, LegendComponent, TooltipComponent, LineChart, CanvasRenderer])

const props = defineProps<{
  analysis: ExperimentAnalysis
  ariaLabels?: { roc: string; pr: string }
}>()
const { t } = useI18n()
const rocRoot = ref<HTMLElement>()
const prRoot = ref<HTMLElement>()
const theme = ref<'light' | 'dark'>('light')
let rocChart: echarts.ECharts | undefined
let prChart: echarts.ECharts | undefined
let themeObserver: MutationObserver | undefined

const splitKeys = ['validation', 'test'] as const

function splitLabel(split: string) {
  return split === 'test' ? t('detail.test') : t('detail.validation')
}

function pointPair(kind: 'roc' | 'pr', point: AnalysisCurvePoint) {
  if (kind === 'roc') return [Number(point.fpr), Number(point.tpr)]
  return [Number(point.recall), Number(point.precision)]
}

function seriesFor(kind: 'roc' | 'pr') {
  const palette = chartPalette(theme.value)
  return splitKeys
    .map((split) => {
      const curve = props.analysis.curves?.[split]?.[kind]
      const isTest = split === 'test'
      const data =
        curve?.available && Array.isArray(curve.points)
          ? curve.points
              .map((point) => pointPair(kind, point))
              .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
          : []
      return {
        id: `${kind}-${split}`,
        name: splitLabel(split),
        type: 'line',
        smooth: false,
        showSymbol: false,
        lineStyle: {
          width: isTest ? 3.5 : 2,
          type: isTest ? 'solid' : 'dashed',
          opacity: isTest ? 1 : 0.78,
        },
        emphasis: { focus: 'series' },
        z: isTest ? 3 : 2,
        data,
        color: isTest ? palette.splitTest : palette.splitValidation,
      }
    })
    .filter((series) => series.data.length > 0)
}

const hasCurves = computed(() =>
  splitKeys.some((split) => {
    const curves = props.analysis.curves?.[split]
    return Boolean(curves?.roc?.available || curves?.pr?.available)
  }),
)

const isMicroAveraged = computed(() =>
  splitKeys.some((split) => props.analysis.curves?.[split]?.average === 'micro'),
)

function curveTitle(kind: 'roc' | 'pr') {
  const label = kind.toUpperCase()
  return isMicroAveraged.value ? `Micro ${label}` : label
}

function curveAriaLabel(kind: 'roc' | 'pr') {
  if (props.ariaLabels) return props.ariaLabels[kind]
  if (isMicroAveraged.value) {
    return t(kind === 'roc' ? 'detail.microRocCurveAria' : 'detail.microPrCurveAria')
  }
  return t(kind === 'roc' ? 'detail.rocCurveAria' : 'detail.prCurveAria')
}

function renderChart(chart: echarts.ECharts | undefined, kind: 'roc' | 'pr') {
  if (!chart) return
  const palette = chartPalette(theme.value)
  const series = seriesFor(kind)
  chart.setOption({
    color: [palette.splitValidation, palette.splitTest],
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
      valueFormatter: (value: unknown) => Number(value).toFixed(3),
    },
    grid: { left: 42, right: 18, top: 44, bottom: 34 },
    xAxis: {
      type: 'value',
      min: 0,
      max: 1,
      name: kind === 'roc' ? 'FPR' : t('detail.recall'),
      axisLine: { lineStyle: { color: palette.axis } },
      axisLabel: { color: palette.axisLabel },
      splitLine: { lineStyle: { color: palette.split } },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      name: kind === 'roc' ? 'TPR' : t('detail.precision'),
      axisLine: { lineStyle: { color: palette.axis } },
      axisLabel: { color: palette.axisLabel },
      splitLine: { lineStyle: { color: palette.split } },
    },
    series,
  })
}

function render() {
  renderChart(rocChart, 'roc')
  renderChart(prChart, 'pr')
}

function scheduleRender() {
  void nextTick(() => {
    rocChart?.resize()
    prChart?.resize()
    render()
  })
}

onMounted(() => {
  theme.value = readChartMode()
  if (rocRoot.value) rocChart = echarts.init(rocRoot.value)
  if (prRoot.value) prChart = echarts.init(prRoot.value)
  themeObserver = new MutationObserver(() => {
    theme.value = readChartMode()
  })
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  })
  render()
})

watch(() => props.analysis, scheduleRender, { deep: true })
watch(theme, scheduleRender)

onBeforeUnmount(() => {
  themeObserver?.disconnect()
  rocChart?.dispose()
  prChart?.dispose()
})
</script>

<template>
  <div>
    <div v-show="hasCurves" class="grid gap-5 xl:grid-cols-2">
      <div class="section-card">
        <div class="flex items-center justify-between gap-3">
          <div class="text-xs font-bold uppercase text-slate-500">{{ curveTitle('roc') }}</div>
          <span v-if="isMicroAveraged" class="config-chip">{{ t('detail.microAveraged') }}</span>
        </div>
        <div
          ref="rocRoot"
          class="mt-3 h-72 w-full"
          role="img"
          :aria-label="curveAriaLabel('roc')"
        />
      </div>
      <div class="section-card">
        <div class="flex items-center justify-between gap-3">
          <div class="text-xs font-bold uppercase text-slate-500">{{ curveTitle('pr') }}</div>
          <span v-if="isMicroAveraged" class="config-chip">{{ t('detail.microAveraged') }}</span>
        </div>
        <div ref="prRoot" class="mt-3 h-72 w-full" role="img" :aria-label="curveAriaLabel('pr')" />
      </div>
    </div>
    <p v-if="!hasCurves" class="section-card text-sm text-stone-500">
      {{ t('detail.noCurveAnalysisYet') }}
    </p>
  </div>
</template>
