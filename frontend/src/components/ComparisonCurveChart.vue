<script setup lang="ts">
import * as echarts from 'echarts/core'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { LineChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { experimentDisplayName } from '../localization'
import type {
  AnalysisCurvePoint,
  ComparisonSplitKey,
  ExperimentAnalysis,
  ExperimentRecord,
} from '../types'
import { chartPalette, readChartMode } from '../chartTheme'

echarts.use([GridComponent, LegendComponent, TooltipComponent, LineChart, CanvasRenderer])

const props = defineProps<{
  experiments: ExperimentRecord[]
  analyses: Record<string, ExperimentAnalysis | undefined>
  split: ComparisonSplitKey
  kind: 'roc' | 'pr'
}>()
const { t } = useI18n()
const root = ref<HTMLElement>()
const theme = ref<'light' | 'dark'>('light')
let chart: echarts.ECharts | undefined
let themeObserver: MutationObserver | undefined

function pointPair(point: AnalysisCurvePoint) {
  if (props.kind === 'roc') return [Number(point.fpr), Number(point.tpr)]
  return [Number(point.recall), Number(point.precision)]
}

const seriesData = computed(() =>
  props.experiments.map((experiment) => {
    const curve = props.analyses[experiment.id]?.curves?.[props.split]?.[props.kind]
    const points =
      curve?.available && Array.isArray(curve.points)
        ? curve.points.map(pointPair).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
        : []
    return { experiment, points }
  }),
)

const hasData = computed(() => seriesData.value.some((item) => item.points.length > 0))

function render() {
  if (!chart) return
  const palette = chartPalette(theme.value)
  chart.setOption({
    color: palette.series,
    legend: {
      top: 0,
      right: 0,
      type: 'scroll',
      textStyle: { color: palette.axisLabel, fontWeight: 600 },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: palette.tooltipBg,
      borderColor: palette.tooltipBg,
      textStyle: { color: palette.tooltipText },
      valueFormatter: (value: unknown) => Number(value).toFixed(4),
    },
    grid: { left: 42, right: 18, top: 52, bottom: 34 },
    xAxis: {
      type: 'value',
      min: 0,
      max: 1,
      name: props.kind === 'roc' ? 'FPR' : t('detail.recall'),
      axisLine: { lineStyle: { color: palette.axis } },
      axisLabel: { color: palette.axisLabel },
      splitLine: { lineStyle: { color: palette.split } },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      name: props.kind === 'roc' ? 'TPR' : t('detail.precision'),
      axisLine: { lineStyle: { color: palette.axis } },
      axisLabel: { color: palette.axisLabel },
      splitLine: { lineStyle: { color: palette.split } },
    },
    series: seriesData.value
      .filter((item) => item.points.length > 0)
      .map((item) => ({
        id: item.experiment.id,
        name: experimentDisplayName(item.experiment, t),
        type: 'line',
        smooth: false,
        showSymbol: false,
        lineStyle: { width: 3 },
        emphasis: { focus: 'series' },
        data: item.points,
      })),
  })
}

function scheduleRender() {
  void nextTick(() => {
    chart?.resize()
    render()
  })
}

onMounted(() => {
  theme.value = readChartMode()
  if (root.value) chart = echarts.init(root.value)
  themeObserver = new MutationObserver(() => {
    theme.value = readChartMode()
  })
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  })
  render()
})

watch(() => [props.experiments, props.analyses, props.split, props.kind], scheduleRender, {
  deep: true,
})
watch(theme, scheduleRender)

onBeforeUnmount(() => {
  themeObserver?.disconnect()
  chart?.dispose()
})
</script>

<template>
  <div>
    <div
      v-show="hasData"
      ref="root"
      class="h-80 w-full"
      role="img"
      :aria-label="t('compare.curveChartAria')"
    />
    <p v-if="!hasData" class="section-card text-sm text-stone-500">
      {{ t('compare.noAnalysisCurves') }}
    </p>
  </div>
</template>
