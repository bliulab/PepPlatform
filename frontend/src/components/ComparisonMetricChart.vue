<script setup lang="ts">
import * as echarts from 'echarts/core'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { LineChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { experimentDisplayName } from '../localization'
import type { ExperimentRecord, MetricEvent } from '../types'
import { chartPalette, readChartMode } from '../chartTheme'

echarts.use([GridComponent, LegendComponent, TooltipComponent, LineChart, CanvasRenderer])

const props = defineProps<{ experiments: ExperimentRecord[]; metricKey: string }>()
const { t } = useI18n()
const root = ref<HTMLElement>()
const theme = ref<'light' | 'dark'>('light')
let chart: echarts.ECharts | undefined
let themeObserver: MutationObserver | undefined

function metricValue(event: MetricEvent) {
  const value = Number(event[props.metricKey])
  return Number.isFinite(value) ? value : null
}

const seriesData = computed(() =>
  props.experiments.map((experiment) => ({
    experiment,
    points: (experiment.metrics?.events ?? [])
      .filter((event) => event.kind === 'epoch' && Number.isFinite(Number(event.epoch)))
      .map((event) => [Number(event.epoch), metricValue(event)])
      .filter((point) => point[1] !== null),
  })),
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
    grid: { left: 44, right: 20, top: 54, bottom: 34 },
    xAxis: {
      type: 'value',
      minInterval: 1,
      name: t('compare.epoch'),
      axisLine: { lineStyle: { color: palette.axis } },
      axisLabel: { color: palette.axisLabel },
      splitLine: { lineStyle: { color: palette.split } },
    },
    yAxis: {
      type: 'value',
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
        smooth: true,
        symbol: 'circle',
        symbolSize: 6,
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

watch(() => [props.experiments, props.metricKey], scheduleRender, { deep: true })
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
      :aria-label="t('compare.metricChartAria')"
    />
    <p v-if="!hasData" class="section-card text-sm text-stone-500">
      {{ t('compare.noEpochMetrics') }}
    </p>
  </div>
</template>
