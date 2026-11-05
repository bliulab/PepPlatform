<script setup lang="ts">
import * as echarts from 'echarts/core'
import { TooltipComponent } from 'echarts/components'
import { Scatter3DChart } from 'echarts-gl/charts'
import { Grid3DComponent } from 'echarts-gl/components'
import { CanvasRenderer } from 'echarts/renderers'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import type { ExperimentAnalysis, ProjectionPoint } from '../types'
import { chartPalette, readChartMode } from '../chartTheme'

echarts.use([TooltipComponent, Scatter3DChart, Grid3DComponent, CanvasRenderer])

const props = defineProps<{
  analysis: ExperimentAnalysis
  split: string
}>()
const { t } = useI18n()
const root = ref<HTMLElement>()
const theme = ref<'light' | 'dark'>('light')
let chart: echarts.ECharts | undefined
let themeObserver: MutationObserver | undefined

const points = computed(() =>
  props.analysis.projection_3d?.available ? (props.analysis.projection_3d.points ?? []) : [],
)

function splitLabel(split: string) {
  if (split === 'test') return t('detail.test')
  if (split === 'validation' || split === 'val') return t('detail.validation')
  return split
}

function splitMatches(actual: string, requested: string) {
  if (requested === 'validation') return actual === 'validation' || actual === 'val'
  if (requested === 'val') return actual === 'validation' || actual === 'val'
  return actual === requested
}

const splitPoints = computed(() =>
  points.value.filter((point) => splitMatches(point.split, props.split)),
)
const hasProjection = computed(() => splitPoints.value.length > 0)

function clusterKey(point: ProjectionPoint) {
  return point.cluster === undefined || point.cluster === null ? '-' : String(point.cluster)
}

function clusterLabel(cluster: string) {
  return `${t('detail.cluster')} ${cluster}`
}

function clusterSort(a: string, b: string) {
  const left = Number(a)
  const right = Number(b)
  if (Number.isFinite(left) && Number.isFinite(right)) return left - right
  return a.localeCompare(b)
}

const clusters = computed(() =>
  Array.from(new Set(points.value.map((point) => clusterKey(point)))).sort(clusterSort),
)

function clusterColor(cluster: string) {
  const colors = chartPalette(theme.value).series
  const index = Math.max(0, clusters.value.indexOf(cluster))
  return colors[index % colors.length]
}

function paddedExtent(values: number[]) {
  const finite = values.filter((value) => Number.isFinite(value))
  if (!finite.length) return { min: -1, max: 1 }
  let min = Math.min(...finite)
  let max = Math.max(...finite)
  if (min === max) {
    min -= 1
    max += 1
  }
  const padding = (max - min) * 0.08
  return { min: min - padding, max: max + padding }
}

const axisExtent = computed(() => ({
  x: paddedExtent(points.value.map((point) => Number(point.x))),
  y: paddedExtent(points.value.map((point) => Number(point.y))),
  z: paddedExtent(points.value.map((point) => Number(point.z ?? 0))),
}))

function formatAxisNumber(value: unknown) {
  const numberValue = Number(value)
  if (!Number.isFinite(numberValue)) return ''
  if (Math.abs(numberValue) < 0.005) return '0'
  return numberValue.toFixed(2).replace(/\.?0+$/, '')
}

function seriesData(cluster: string) {
  return splitPoints.value
    .filter((point) => clusterKey(point) === cluster)
    .map((point) => ({
      value: [point.x, point.y, point.z ?? 0, point.label, clusterKey(point), point.split],
    }))
}

function valueData(data: unknown) {
  if (Array.isArray(data)) return data
  if (data && typeof data === 'object' && Array.isArray((data as { value?: unknown }).value)) {
    return (data as { value: unknown[] }).value
  }
  return []
}

function render() {
  if (!chart) return
  const current = chartPalette(theme.value)
  chart.setOption(
    {
      color: current.series,
      tooltip: {
        trigger: 'item',
        backgroundColor: current.tooltipBg,
        borderColor: current.tooltipBg,
        textStyle: { color: current.tooltipText },
        formatter: (params: { data?: unknown }) => {
          const data = valueData(params.data)
          const x = Number(data[0])
          const y = Number(data[1])
          const z = Number(data[2])
          const label = String(data[3] ?? '-')
          const cluster = String(data[4] ?? '-')
          const splitValue = String(data[5] ?? props.split)
          return `${splitLabel(splitValue)}<br/>${t('detail.label')}: ${label}<br/>${t('detail.cluster')}: ${cluster}<br/>x ${x.toFixed(3)}<br/>y ${y.toFixed(3)}<br/>z ${z.toFixed(3)}`
        },
      },
      grid3D: {
        boxWidth: 110,
        boxHeight: 86,
        boxDepth: 86,
        axisLine: { lineStyle: { color: current.axis } },
        axisPointer: { lineStyle: { color: current.axisLabel } },
        splitLine: { lineStyle: { color: current.split } },
        viewControl: {
          projection: 'perspective',
          distance: 190,
          alpha: 24,
          beta: 38,
          rotateSensitivity: 1.2,
          zoomSensitivity: 1,
          panSensitivity: 0.6,
        },
      },
      xAxis3D: {
        type: 'value',
        min: axisExtent.value.x.min,
        max: axisExtent.value.x.max,
        axisLabel: { color: current.axisLabel, formatter: formatAxisNumber },
      },
      yAxis3D: {
        type: 'value',
        min: axisExtent.value.y.min,
        max: axisExtent.value.y.max,
        axisLabel: { color: current.axisLabel, formatter: formatAxisNumber },
      },
      zAxis3D: {
        type: 'value',
        min: axisExtent.value.z.min,
        max: axisExtent.value.z.max,
        axisLabel: { color: current.axisLabel, formatter: formatAxisNumber },
      },
      series: clusters.value.map((cluster) => ({
        id: `cluster-3d-${cluster}`,
        name: clusterLabel(cluster),
        type: 'scatter3D',
        symbol: 'circle',
        symbolSize: 7,
        emphasis: { focus: 'series' },
        itemStyle: {
          color: clusterColor(cluster),
          opacity: 0.82,
          borderColor: current.surface,
          borderWidth: 0.6,
        },
        data: seriesData(cluster),
      })),
    },
    true,
  )
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

watch(() => props.analysis, scheduleRender, { deep: true })
watch(theme, scheduleRender)

onBeforeUnmount(() => {
  themeObserver?.disconnect()
  chart?.dispose()
})
</script>

<template>
  <div v-if="hasProjection">
    <div
      v-if="clusters.length"
      class="mb-3 flex flex-wrap items-center gap-3 text-xs font-semibold text-slate-500"
    >
      <span v-for="cluster in clusters" :key="cluster" class="inline-flex items-center gap-2">
        <span
          class="inline-block h-2.5 w-2.5 rounded-full"
          :style="{ backgroundColor: clusterColor(cluster) }"
        />
        {{ clusterLabel(cluster) }}
      </span>
    </div>
    <div
      ref="root"
      class="mt-3 h-80 w-full"
      role="img"
      :aria-label="`${t('detail.projection3dAria')} ${splitLabel(split)}`"
    />
  </div>
  <p v-else class="text-sm text-stone-500">
    {{ t('detail.noProjection3dYet') }}
  </p>
</template>
