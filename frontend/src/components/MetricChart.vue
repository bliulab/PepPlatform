<script setup lang="ts">
import * as echarts from 'echarts/core'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { LineChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import type { MetricEvent } from '../types'
import { chartPalette, readChartMode } from '../chartTheme'

echarts.use([GridComponent, LegendComponent, TooltipComponent, LineChart, CanvasRenderer])

const props = defineProps<{ events: MetricEvent[]; multilabel?: boolean }>()
const { t } = useI18n()
const root = ref<HTMLElement>()
const theme = ref<'light' | 'dark'>('light')
let chart: echarts.ECharts | undefined
let themeObserver: MutationObserver | undefined

const metricSeries = computed(
  () =>
    [
      {
        key: 'val_f1',
        label: props.multilabel ? t('detail.validationMacroF1') : t('detail.validationF1'),
        symbol: 'rect',
      },
      {
        key: 'val_auc',
        label: props.multilabel ? t('detail.validationMacroAuc') : t('detail.validationAuc'),
        symbol: 'diamond',
      },
    ] as const,
)

function metricValue(event: MetricEvent, key: string) {
  const value = Number(event[key])
  return Number.isFinite(value) ? value : null
}

function render() {
  const rows = props.events.filter((event) => event.kind === 'epoch')
  const palette = chartPalette(theme.value)
  const legendData = metricSeries.value.map((metric) => ({
    name: metric.label,
    icon: metric.symbol,
  }))

  chart?.setOption({
    color: palette.series,
    legend: {
      data: legendData,
      top: 0,
      right: 0,
      itemWidth: 13,
      itemHeight: 13,
      textStyle: {
        color: palette.axisLabel,
        fontWeight: 600,
      },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: palette.tooltipBg,
      borderColor: palette.tooltipBg,
      textStyle: { color: palette.tooltipText },
    },
    grid: { left: 36, right: 18, top: 52, bottom: 28 },
    xAxis: {
      type: 'category',
      data: rows.map((event) => event.epoch),
      axisLine: { lineStyle: { color: palette.axis } },
      axisLabel: {
        color: palette.axisLabel,
        showMinLabel: true,
        showMaxLabel: true,
        hideOverlap: true,
      },
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: palette.split } },
      axisLabel: { color: palette.axisLabel },
    },
    series: metricSeries.value.map((metric) => ({
      id: metric.key,
      name: metric.label,
      type: 'line',
      smooth: true,
      symbol: metric.symbol,
      symbolSize: 8,
      lineStyle: { width: 3 },
      emphasis: { focus: 'series' },
      data: rows.map((event) => metricValue(event, metric.key)),
    })),
  })
}

onMounted(() => {
  if (!root.value) return
  theme.value = readChartMode()
  chart = echarts.init(root.value)
  themeObserver = new MutationObserver(() => {
    theme.value = readChartMode()
  })
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  })
  render()
})
watch(() => props.events, render, { deep: true })
watch(theme, render)
onBeforeUnmount(() => {
  themeObserver?.disconnect()
  chart?.dispose()
})
</script>

<template>
  <div ref="root" class="h-72 w-full" role="img" :aria-label="t('detail.metricCurveAria')" />
</template>
