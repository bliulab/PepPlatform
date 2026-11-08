<script setup lang="ts">
import * as echarts from 'echarts/core'
import { GridComponent, TooltipComponent } from 'echarts/components'
import { BarChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import type { ExperimentAnalysis, FeatureImportancePayload } from '../types'
import {
  createImportanceFormatter,
  formatFullImportance,
  selectVisibleImportanceItems,
} from './featureImportanceChart'
import { chartPalette, readChartMode, type ChartPalette } from '../chartTheme'

echarts.use([GridComponent, TooltipComponent, BarChart, CanvasRenderer])

const props = defineProps<{
  analysis?: ExperimentAnalysis
  importance?: FeatureImportancePayload
  contextLabel?: string
  methodLabel?: string
  variant?: 'global' | 'validation' | 'test'
}>()
const { t } = useI18n()
const root = ref<HTMLElement>()
const theme = ref<'light' | 'dark'>('light')
let chart: echarts.ECharts | undefined
let themeObserver: MutationObserver | undefined

const payload = computed(() => props.importance ?? props.analysis?.feature_importance)
const visibleItems = computed(() =>
  payload.value?.available ? selectVisibleImportanceItems(payload.value.items ?? []) : [],
)
const items = computed(() => [...visibleItems.value].reverse())
const hasImportance = computed(() => items.value.length > 0)
const chartVariant = computed(() => props.variant ?? 'global')
const importanceValueLabel = computed(() => props.methodLabel || t('detail.importanceValue'))
const importanceFormatter = computed(() =>
  createImportanceFormatter(visibleItems.value.map((item) => item.importance)),
)
const chartAriaLabel = computed(() =>
  props.contextLabel
    ? `${props.contextLabel}: ${t('detail.featureImportanceAria')}`
    : t('detail.featureImportanceAria'),
)

function displayFeatureName(name: string) {
  const featureMatch = /^feature_(\d+)$/i.exec(name)
  if (featureMatch) return `F${featureMatch[1]}`
  const componentMatch = /^component_(\d+)$/i.exec(name)
  if (componentMatch) return `C${componentMatch[1]}`
  return name
}

function importanceColor(palette: ChartPalette, variant: 'global' | 'validation' | 'test') {
  if (variant === 'validation') return palette.splitValidation
  if (variant === 'test') return palette.splitTest
  return palette.accent
}

function escapeHtml(value: string) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function render() {
  if (!chart) return
  const palette = chartPalette(theme.value)
  const formatImportance = importanceFormatter.value.format
  chart.setOption({
    color: [importanceColor(palette, chartVariant.value)],
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: palette.tooltipBg,
      borderColor: palette.tooltipBg,
      textStyle: { color: palette.tooltipText },
      formatter: (params: unknown) => {
        const row = Array.isArray(params) ? params[0] : params
        const dataIndex = Number((row as { dataIndex?: unknown } | undefined)?.dataIndex)
        const item = items.value[dataIndex]
        if (!item) return ''
        const lines = [
          escapeHtml(item.name),
          `${escapeHtml(importanceValueLabel.value)}: ${formatFullImportance(item.importance)}`,
        ]
        if (props.contextLabel)
          lines.push(
            `${escapeHtml(t('detail.importanceContext'))}: ${escapeHtml(props.contextLabel)}`,
          )
        return lines.join('<br/>')
      },
    },
    grid: { left: 8, right: 112, top: 14, bottom: 34, containLabel: true },
    xAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: palette.axis } },
      axisLabel: {
        color: palette.axisLabel,
        formatter: (value: number) => formatImportance(value),
        hideOverlap: true,
      },
      splitLine: { lineStyle: { color: palette.split } },
    },
    yAxis: {
      type: 'category',
      data: items.value.map((item) => item.name),
      axisLine: { lineStyle: { color: palette.axis } },
      axisLabel: {
        color: palette.axisLabel,
        fontSize: 12,
        hideOverlap: false,
        interval: 0,
        margin: 10,
        formatter: (value: string) => displayFeatureName(value),
      },
    },
    series: [
      {
        name: importanceValueLabel.value,
        type: 'bar',
        data: items.value.map((item) => item.importance),
        barWidth: 10,
        label: {
          show: true,
          position: 'right',
          color: palette.axisLabel,
          fontSize: 11,
          formatter: (params: { value?: unknown }) => formatImportance(params.value),
        },
        itemStyle: { borderRadius: 4 },
      },
    ],
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

watch(
  () => [props.analysis, props.importance, props.contextLabel, props.methodLabel, props.variant],
  scheduleRender,
  {
    deep: true,
  },
)
watch(theme, scheduleRender)

onBeforeUnmount(() => {
  themeObserver?.disconnect()
  chart?.dispose()
})
</script>

<template>
  <div
    v-show="hasImportance"
    ref="root"
    class="h-80 w-full"
    role="img"
    :aria-label="chartAriaLabel"
  />
  <p v-if="!hasImportance" class="text-sm text-stone-500">
    {{ t('detail.noFeatureImportanceYet') }}
  </p>
</template>
