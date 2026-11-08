<script setup lang="ts">
import * as echarts from 'echarts/core'
import { HeatmapChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, VisualMapComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { shortGroup, visibleCellValue, type HeatmapCell, type HeatmapResult } from '../heatmaps'
import { chartPalette, readChartMode } from '../chartTheme'

echarts.use([HeatmapChart, GridComponent, TooltipComponent, VisualMapComponent, CanvasRenderer])
const props = defineProps<{ result: HeatmapResult; showSmall: boolean }>()
const emit = defineEmits<{ select: [cell: HeatmapCell] }>()
const { t, te, locale } = useI18n()
const root = ref<HTMLElement>()
let chart: echarts.ECharts | undefined
let resize: ResizeObserver | undefined
let themeObserver: MutationObserver | undefined
let cells: HeatmapCell[] = []
const label = (value: string) =>
  te(`heatmap.${value}`) ? t(`heatmap.${value}`) : shortGroup(value)
const escape = (value: string) =>
  value.replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!,
  )

async function render() {
  await nextTick()
  if (!root.value || !chart) return
  const mode = readChartMode()
  const palette = chartPalette(mode)
  const foreground = palette.foreground
  const muted = palette.mutedCell
  root.value.style.height = `${Math.max(360, props.result.rows.length * 27 + 125)}px`
  const lookup = new Map(
    props.result.cells.map((cell) => [JSON.stringify([cell.row, cell.column]), cell]),
  )
  cells = props.result.rows.flatMap((row) =>
    props.result.columns.map(
      (column) =>
        lookup.get(JSON.stringify([row, column])) ?? {
          row,
          column,
          value: null,
          count: 0,
          reason: 'noData',
        },
    ),
  )
  chart.resize()
  chart.setOption(
    {
      animation: false,
      grid: { top: 22, bottom: 100, left: 175, right: 30 },
      xAxis: {
        type: 'category',
        data: props.result.columns.map(label),
        axisLabel: { color: foreground, rotate: props.result.columns.length > 15 ? 45 : 0 },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'category',
        inverse: true,
        data: props.result.rows.map(shortGroup),
        axisLabel: { color: foreground, width: 155, overflow: 'truncate' },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      visualMap: {
        min: 0,
        max: props.result.color_max,
        calculable: false,
        orient: 'horizontal',
        left: 'center',
        bottom: 6,
        text: [String(props.result.color_max), '0'],
        inRange: { color: [...palette.heatmapRange] },
        textStyle: { color: foreground },
      },
      tooltip: {
        formatter: (raw: unknown) => {
          const cell = cells[(raw as { dataIndex: number }).dataIndex]
          if (!cell) return ''
          const value = visibleCellValue(cell, props.result.params.kind, props.showSmall)
          return `${escape(cell.row)} · ${escape(label(cell.column))}<br>${escape(t('heatmap.value'))}: ${value === null ? '—' : value.toFixed(props.result.count_scale ? 0 : 4)}<br>${escape(t('heatmap.count'))}: ${cell.count}`
        },
      },
      series: [
        {
          type: 'heatmap',
          data: cells.map((cell) => {
            const value = visibleCellValue(cell, props.result.params.kind, props.showSmall)
            return {
              value: [
                props.result.columns.indexOf(cell.column),
                props.result.rows.indexOf(cell.row),
                value ?? 0,
              ],
              itemStyle: {
                borderWidth: 2,
                borderColor: palette.surface,
                ...(value === null ? { color: muted } : {}),
              },
              label: {
                show: props.result.columns.length <= 15 && props.result.rows.length <= 25,
                formatter: value === null ? '—' : value.toFixed(props.result.count_scale ? 0 : 2),
              },
            }
          }),
          emphasis: { itemStyle: { borderColor: palette.emphasisBorder, borderWidth: 3 } },
        },
      ],
    },
    true,
  )
}
onMounted(() => {
  root.value!.style.height = `${Math.max(360, props.result.rows.length * 27 + 125)}px`
  chart = echarts.init(root.value!)
  chart.on('click', (raw) => {
    const cell = cells[raw.dataIndex]
    if (cell) emit('select', cell)
  })
  resize = new ResizeObserver(() => chart?.resize())
  resize.observe(root.value!)
  themeObserver = new MutationObserver(render)
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  })
  void render()
})
watch(() => [props.result, props.showSmall, locale.value], render)
onBeforeUnmount(() => {
  resize?.disconnect()
  themeObserver?.disconnect()
  chart?.dispose()
  chart = undefined
})
defineExpose({
  image: () =>
    chart?.getDataURL({
      type: 'png',
      pixelRatio: 2,
      backgroundColor: chartPalette(readChartMode()).surface,
    }),
})
</script>

<template>
  <div
    ref="root"
    role="img"
    :aria-label="`${t('heatmap.title')} · ${t('heatmap.table')}`"
    class="min-w-[540px] w-full"
  />
</template>
