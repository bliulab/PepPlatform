<script setup lang="ts">
import * as echarts from 'echarts/core'
import {
  GridComponent,
  LegendComponent,
  MarkPointComponent,
  TooltipComponent,
} from 'echarts/components'
import { BarChart, LineChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { chartPalette, readChartMode } from '../chartTheme'

echarts.use([
  GridComponent,
  LegendComponent,
  MarkPointComponent,
  TooltipComponent,
  BarChart,
  LineChart,
  CanvasRenderer,
])

interface TuningMetricSet {
  f1?: number
  auc?: number
  acc?: number
  sacc?: number
  [key: string]: number | undefined
}

interface TuningCandidate {
  index?: number
  rank?: number
  selected?: boolean
  params?: Record<string, unknown>
  metrics?: TuningMetricSet
  fold_metrics?: Array<TuningMetricSet & { fold?: number }>
}

const props = defineProps<{ tuning: Record<string, unknown> }>()
const { t } = useI18n()
const root = ref<HTMLElement>()
const theme = ref<'light' | 'dark'>('light')
let chart: echarts.ECharts | undefined
let themeObserver: MutationObserver | undefined

const candidates = computed(() =>
  Array.isArray(props.tuning.candidates) ? (props.tuning.candidates as TuningCandidate[]) : [],
)
const objective = computed(() => String(props.tuning.objective ?? 'f1').toLowerCase())
const selectedIndex = computed(() => candidates.value.findIndex((candidate) => candidate.selected))
const maxFold = computed(() =>
  Math.max(
    0,
    ...candidates.value.flatMap((candidate) =>
      (candidate.fold_metrics ?? []).map((fold) => Number(fold.fold ?? 0)),
    ),
  ),
)

function numberValue(value: unknown) {
  if (value === null || value === undefined || value === '') return null
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function metricValue(candidate: TuningCandidate, key: string) {
  return numberValue(
    candidate.metrics?.[key === 'dual_f1_sacc' ? 'dual_score' : key === 'macro_f1' ? 'f1' : key],
  )
}

function foldMetricValue(candidate: TuningCandidate, foldNumber: number, key: string) {
  const fold = candidate.fold_metrics?.find((item) => Number(item.fold) === foldNumber)
  return numberValue(
    fold?.[key === 'dual_f1_sacc' ? 'dual_score' : key === 'macro_f1' ? 'f1' : key],
  )
}

function formatParams(params?: Record<string, unknown>) {
  if (!params) return ''
  return Object.entries(params)
    .map(([key, value]) => `${key}=${value}`)
    .join(', ')
}

function render() {
  if (!chart) return
  if (candidates.value.length === 0) {
    chart.clear()
    return
  }
  const palette = chartPalette(theme.value)
  const key = objective.value
  const labels = candidates.value.map((candidate, index) => `C${candidate.index ?? index + 1}`)
  const foldSeries = Array.from({ length: maxFold.value }, (_, foldIndex) => {
    const foldNumber = foldIndex + 1
    return {
      id: `fold-${foldNumber}`,
      name: `${t('detail.tuningFold')} ${foldNumber}`,
      type: 'line',
      smooth: true,
      symbol: 'circle',
      symbolSize: 7,
      lineStyle: { width: 2 },
      emphasis: { focus: 'series' },
      data: candidates.value.map((candidate) => foldMetricValue(candidate, foldNumber, key)),
    }
  })

  chart.clear()
  chart.setOption({
    color: [palette.accent, ...palette.series.slice(1)],
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
      formatter(params: unknown) {
        const rows = Array.isArray(params) ? params : [params]
        const first = rows[0] as { dataIndex?: number }
        const candidate = candidates.value[Number(first?.dataIndex ?? 0)]
        const metric = metricValue(candidate, key)
        const lines = [
          `<strong>C${candidate?.index ?? ''}${candidate?.selected ? ' *' : ''}</strong>`,
          `${t('detail.tuningMean')}: ${metric === null ? '-' : metric.toFixed(4)}`,
        ]
        for (const item of rows as Array<{
          marker?: string
          seriesName?: string
          value?: number
        }>) {
          if (item.seriesName === t('detail.tuningMean')) continue
          const value = Number(item.value)
          if (Number.isFinite(value))
            lines.push(`${item.marker ?? ''}${item.seriesName}: ${value.toFixed(4)}`)
        }
        const paramsText = formatParams(candidate?.params)
        if (paramsText) lines.push(`<span style="opacity:.75">${paramsText}</span>`)
        return lines.join('<br/>')
      },
    },
    grid: { left: 38, right: 18, top: 54, bottom: 34 },
    xAxis: {
      type: 'category',
      data: labels,
      axisLine: { lineStyle: { color: palette.axis } },
      axisLabel: { color: palette.axisLabel },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: key === 'dual_f1_sacc' ? undefined : 1,
      splitLine: { lineStyle: { color: palette.split } },
      axisLabel: { color: palette.axisLabel },
    },
    series: [
      {
        id: 'mean',
        name: t('detail.tuningMean'),
        type: 'bar',
        barMaxWidth: 34,
        data: candidates.value.map((candidate) => metricValue(candidate, key)),
        itemStyle: {
          borderRadius: [4, 4, 0, 0],
          color(params: { dataIndex: number }) {
            return params.dataIndex === selectedIndex.value ? palette.accent : `${palette.accent}99`
          },
        },
        markPoint:
          selectedIndex.value >= 0
            ? {
                symbol: 'pin',
                symbolSize: 44,
                label: { formatter: 'best', fontSize: 10 },
                data: [
                  {
                    coord: [
                      selectedIndex.value,
                      metricValue(candidates.value[selectedIndex.value], key),
                    ],
                  },
                ],
              }
            : undefined,
      },
      ...foldSeries,
    ],
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
watch(() => props.tuning, render, { deep: true })
watch(theme, render)
onBeforeUnmount(() => {
  themeObserver?.disconnect()
  chart?.dispose()
})
</script>

<template>
  <div>
    <div ref="root" class="h-80 w-full" role="img" :aria-label="t('detail.tuningChartAria')" />
    <details class="mt-4 rounded-lg border border-slate-200 p-4">
      <summary class="cursor-pointer text-sm font-semibold">{{
        t('detail.tuningCandidateTable')
      }}</summary>
      <div class="mt-3 overflow-x-auto">
        <table class="w-full min-w-[36rem] text-left text-sm">
          <thead>
            <tr class="border-b border-slate-200 text-slate-500">
              <th class="px-2 py-2">{{ t('detail.tuningCandidate') }}</th>
              <th class="px-2 py-2">{{ t('detail.tuningParameters') }}</th>
              <th class="px-2 py-2">{{ objective.toUpperCase() }}</th>
              <th class="px-2 py-2">{{ t('detail.tuningRank') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(candidate, index) in candidates"
              :key="candidate.index ?? index"
              class="border-b border-slate-100"
            >
              <td class="px-2 py-2 font-semibold">
                C{{ candidate.index ?? index + 1 }}
                <span v-if="candidate.selected" class="ml-2 text-amber-700">{{
                  t('detail.tuningSelected')
                }}</span>
              </td>
              <td class="px-2 py-2 font-mono text-xs">{{ formatParams(candidate.params) }}</td>
              <td class="px-2 py-2">{{
                metricValue(candidate, objective) === null
                  ? '-'
                  : metricValue(candidate, objective)?.toFixed(4)
              }}</td>
              <td class="px-2 py-2">{{ candidate.rank ?? '-' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </details>
  </div>
</template>
