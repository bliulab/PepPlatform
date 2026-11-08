<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useQuery } from '@tanstack/vue-query'
import { useI18n } from 'vue-i18n'
import { api, ApiError } from '../api/client'
import { heatmapApi } from '../api/heatmaps'
import HeatmapChart from '../components/HeatmapChart.vue'
import {
  sameParams,
  shortGroup,
  visibleCellValue,
  type HeatmapCell,
  type HeatmapKind,
  type HeatmapParams,
  type HeatmapResult,
} from '../heatmaps'

const props = defineProps<{ id: string }>()
const { t, te } = useI18n()
const split = ref('')
const kind = ref<HeatmapKind>('stratified')
const metric = ref('aupr')
const pattern = ref('truth')
const scale = ref('count')
const jobId = ref('')
const resultId = ref('')
const displayedJobId = ref('')
const result = ref<HeatmapResult>()
const selected = ref<HeatmapCell>()
const rowOffset = ref(0)
const columnOffset = ref(0)
const search = ref('')
const showSmall = ref(false)
const busy = ref(false)
const error = ref('')
const chart = ref<InstanceType<typeof HeatmapChart>>()
const experiment = useQuery({
  queryKey: computed(() => ['experiment', props.id]),
  queryFn: () => api.experiment(props.id),
})
const capabilities = useQuery({
  queryKey: computed(() => ['heatmap-capabilities', props.id]),
  queryFn: () => heatmapApi.capabilities(props.id),
})
const selectedSplit = computed(() =>
  capabilities.data.value?.splits.find((s) => s.key === split.value),
)
const availability = computed(() =>
  selectedSplit.value?.types.find((item) => item.kind === kind.value),
)
const params = computed<HeatmapParams>(() => ({
  split: split.value,
  kind: kind.value,
  ...(kind.value === 'stratified' ? { metric: metric.value } : {}),
  ...(kind.value === 'patterns' ? { pattern: pattern.value, scale: scale.value } : {}),
}))
const job = useQuery({
  queryKey: computed(() => ['heatmap-job', props.id, jobId.value]),
  queryFn: () => heatmapApi.job(props.id, jobId.value),
  enabled: computed(() => Boolean(jobId.value)),
  refetchInterval: (query) =>
    ['queued', 'running'].includes(query.state.data?.status ?? '') ? 1000 : false,
})
const active = computed(() => ['queued', 'running'].includes(job.data.value?.status ?? ''))
const resultQuery = useQuery({
  queryKey: computed(() => [
    'heatmap-result',
    props.id,
    resultId.value,
    rowOffset.value,
    columnOffset.value,
    search.value,
  ]),
  queryFn: () =>
    heatmapApi.result(props.id, resultId.value, rowOffset.value, columnOffset.value, search.value),
  enabled: computed(() => Boolean(resultId.value)),
  retry: false,
})
const stale = computed(() => result.value && !sameParams(result.value.params, params.value))
const canGenerate = computed(
  () => Boolean(split.value && availability.value?.available) && !busy.value && !active.value,
)
const needsPoint = computed(
  () =>
    (kind.value === 'patterns' && pattern.value !== 'truth') ||
    (kind.value === 'stratified' && ['acc', 'f1'].includes(metric.value)),
)
const explain = (code: string) =>
  te(`heatmap.errors.${code}`) ? t(`heatmap.errors.${code}`) : code
const label = (value: string) => (te(`heatmap.${value}`) ? t(`heatmap.${value}`) : value)
const format = (value: unknown) =>
  typeof value === 'number' ? (Number.isInteger(value) ? String(value) : value.toFixed(4)) : '—'
function report(exc: unknown) {
  error.value = exc instanceof ApiError ? exc.code || exc.message : 'analysis_failed'
}

watch(
  () => props.id,
  () => {
    split.value = ''
    result.value = undefined
    resultId.value = ''
    displayedJobId.value = ''
    selected.value = undefined
    jobId.value = sessionStorage.getItem(`heatmap-job:${props.id}`) || ''
    rowOffset.value = columnOffset.value = 0
    search.value = ''
    error.value = ''
  },
  { immediate: true },
)
watch(
  () => capabilities.data.value,
  (value) => {
    if (value && !split.value) {
      split.value = value.splits.some((s) => s.key === 'validation') ? 'validation' : ''
      kind.value =
        value.task_type === 'pep_function' && value.label_names.length > 1 ? 'labels' : 'stratified'
    }
  },
  { immediate: true },
)
watch(
  () => job.data.value,
  (value) => {
    if (value?.status === 'succeeded' && resultId.value !== value.id) {
      rowOffset.value = columnOffset.value = 0
      search.value = ''
      resultId.value = value.id
    }
  },
  { immediate: true },
)
watch(
  () => resultQuery.data.value,
  (value) => {
    if (value) {
      result.value = value
      displayedJobId.value = resultId.value
      selected.value = undefined
    }
  },
  { immediate: true },
)
watch(search, () => {
  rowOffset.value = 0
})
async function generate() {
  busy.value = true
  error.value = ''
  const id = props.id
  try {
    const response = await heatmapApi.submit(id, params.value)
    if (props.id !== id) return
    jobId.value = response.id
    sessionStorage.setItem(`heatmap-job:${id}`, response.id)
    await job.refetch()
    if (response.status === 'succeeded') await resultQuery.refetch()
  } catch (exc) {
    report(exc)
  } finally {
    busy.value = false
  }
}
async function cancel() {
  try {
    await heatmapApi.cancel(props.id, jobId.value)
    await job.refetch()
  } catch (exc) {
    report(exc)
  }
}
function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = name
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
async function exportResult(format: 'csv' | 'png') {
  error.value = ''
  try {
    if (format === 'csv')
      download(await heatmapApi.export(props.id, displayedJobId.value, 'csv'), 'heatmap.csv')
    else {
      // Recheck the input identity before exporting the currently displayed page.
      const metadata = JSON.parse(
        await (await heatmapApi.export(props.id, displayedJobId.value, 'json')).text(),
      )
      const image = chart.value?.image()
      if (image) {
        const link = document.createElement('a')
        link.href = image
        link.download = 'heatmap.png'
        link.click()
      }
      download(
        new Blob(
          [
            JSON.stringify(
              {
                ...metadata,
                view: {
                  row_offset: result.value?.row_offset,
                  column_offset: result.value?.column_offset,
                  search: result.value?.search,
                  show_small: showSmall.value,
                  rows: result.value?.rows,
                  columns: result.value?.columns,
                },
              },
              null,
              2,
            ),
          ],
          { type: 'application/json' },
        ),
        'heatmap.provenance.json',
      )
    }
  } catch (exc) {
    report(exc)
  }
}
const renderedCells = computed(() => {
  if (!result.value) return []
  const map = new Map(
    result.value.cells.map((cell) => [JSON.stringify([cell.row, cell.column]), cell]),
  )
  return result.value.rows.flatMap((row) =>
    result.value!.columns.map(
      (column) =>
        map.get(JSON.stringify([row, column])) ??
        ({ row, column, value: null, count: 0, reason: 'noData' } as HeatmapCell),
    ),
  )
})
</script>

<template>
  <div class="heatmap-workspace space-y-5">
    <header class="panel heatmap-heading">
      <div
        ><p class="eyebrow">{{ experiment.data.value?.name }}</p
        ><h1 class="mt-2 text-3xl font-semibold">{{ t('heatmap.title') }}</h1>
        <p class="mt-3 text-sm text-slate-500">{{ t('heatmap.intro') }}</p></div
      >
      <RouterLink :to="`/experiments/${id}`" class="secondary-action compact-action">{{
        t('detail.overviewTab')
      }}</RouterLink>
    </header>
    <section class="panel" :aria-label="t('heatmap.kind')">
      <div class="heatmap-controls">
        <label
          >{{ t('heatmap.kind')
          }}<select v-model="kind"
            ><option
              v-for="key in ['stratified', 'labels', 'patterns'] as const"
              :key="key"
              :value="key"
              >{{ t(`heatmap.${key}`) }}</option
            ></select
          ></label
        >
        <label
          >{{ t('heatmap.split')
          }}<select v-model="split"
            ><option value="" disabled>{{ t('heatmap.chooseSplit') }}</option
            ><option
              v-for="item in capabilities.data.value?.splits"
              :key="item.key"
              :value="item.key"
              >{{ label(item.key) }}</option
            ></select
          ></label
        >
        <label v-if="kind === 'stratified'"
          >{{ t('heatmap.metric')
          }}<select v-model="metric"
            ><option
              v-for="key in ['aupr', 'auc', 'f1', 'acc', 'count', 'prevalence']"
              :key="key"
              :value="key"
              >{{ label(key) }}</option
            ></select
          ></label
        >
        <template v-if="kind === 'patterns'">
          <label
            >{{ t('heatmap.pattern')
            }}<select v-model="pattern"
              ><option
                v-for="key in ['truth', 'prediction', 'false_negative', 'false_positive']"
                :key="key"
                :value="key"
                >{{ label(key) }}</option
              ></select
            ></label
          >
          <label
            >{{ t('heatmap.scale')
            }}<select v-model="scale"
              ><option value="count">{{ t('heatmap.count') }}</option
              ><option value="jaccard">Jaccard</option></select
            ></label
          >
        </template>
        <button
          class="primary-action"
          :disabled="!canGenerate || (needsPoint && !selectedSplit?.threshold_available)"
          @click="generate"
          >{{ busy ? t('heatmap.loading') : t('heatmap.generate') }}</button
        >
      </div>
      <p v-if="capabilities.isPending.value" class="mt-4 text-sm" role="status">{{
        t('heatmap.loading')
      }}</p>
      <p
        v-else-if="capabilities.data.value?.reason || (split && !availability?.available)"
        class="mt-4 text-sm text-slate-500"
        >{{ t('heatmap.unavailable') }}:
        {{
          explain(capabilities.data.value?.reason || availability?.reason || 'evidence_missing')
        }}</p
      >
      <p
        v-if="selectedSplit && !selectedSplit.threshold_available"
        class="mt-4 text-sm text-slate-500"
        >{{ explain('threshold_missing') }}</p
      >
      <div v-if="job.data.value" class="mt-4 flex items-center gap-4 text-sm" role="status"
        ><span>{{ label(job.data.value.status) }}</span>
        <button v-if="active" class="secondary-action compact-action" @click="cancel">{{
          t('heatmap.cancel')
        }}</button>
        <span v-if="job.data.value.error">{{ explain(job.data.value.error) }}</span>
      </div>
      <p
        v-if="error || capabilities.error.value || job.error.value || resultQuery.error.value"
        class="error-card mt-4"
        role="alert"
        >{{
          explain(
            error ||
              (capabilities.error.value as ApiError)?.code ||
              (job.error.value as ApiError)?.code ||
              (resultQuery.error.value as ApiError)?.code ||
              'analysis_failed',
          )
        }}</p
      >
    </section>
    <section v-if="result" class="panel">
      <div class="heatmap-heading gap-4"
        ><div
          ><h2 class="text-xl font-semibold">{{ t(`heatmap.${result.params.kind}`) }}</h2>
          <p class="mt-2 text-sm text-slate-500"
            >{{ label(result.params.split) }} · {{ result.sample_count }} {{ t('heatmap.samples') }}
            <span v-if="result.params.metric"> · {{ label(result.params.metric) }}</span
            ><span v-if="result.params.pattern">
              · {{ label(result.params.pattern) }} · {{ label(result.params.scale!) }}</span
            ></p
          ></div
        >
        <div class="flex flex-wrap gap-2"
          ><button
            class="secondary-action compact-action"
            :disabled="resultQuery.isFetching.value"
            @click="exportResult('png')"
            >{{ t('heatmap.png') }}</button
          ><button
            class="secondary-action compact-action"
            :disabled="resultQuery.isFetching.value"
            @click="exportResult('csv')"
            >{{ t('heatmap.csv') }}</button
          ></div
        >
      </div>
      <p v-if="stale" class="mt-4 text-sm" role="status">{{ t('heatmap.stale') }}</p>
      <p class="mt-4 text-sm text-slate-500">{{
        t(result.params.kind === 'patterns' ? 'heatmap.patternHint' : 'heatmap.metricHint')
      }}</p>
      <p v-if="result.params.split !== 'validation'" class="mt-2 text-sm text-slate-500">{{
        t('heatmap.postHoc')
      }}</p>
      <div class="heatmap-controls mt-5"
        ><label>{{ t('heatmap.filter') }}<input v-model.lazy="search" type="search" /></label>
        <label v-if="result.params.kind === 'stratified'" class="heatmap-checkbox"
          ><input v-model="showSmall" type="checkbox" />{{ t('heatmap.small') }}</label
        >
      </div>
      <p v-if="resultQuery.isFetching.value" class="mt-3 text-sm" role="status">{{
        t('heatmap.loading')
      }}</p>
      <div class="mt-4 overflow-x-auto"
        ><HeatmapChart
          ref="chart"
          :result="result"
          :show-small="showSmall"
          @select="selected = $event"
      /></div>
      <p class="mb-2 text-xs text-slate-500 sm:hidden">{{ t('heatmap.scrollHint') }}</p>
      <p class="text-xs text-slate-500">{{ t('heatmap.smallHint') }}</p>
      <div class="mt-5 flex flex-wrap gap-5 text-sm">
        <div v-for="axis in ['rows', 'columns']" :key="axis" class="flex items-center gap-3"
          ><span
            >{{ t(`heatmap.${axis}`) }}:
            {{ axis === 'rows' ? result.row_offset + 1 : result.column_offset + 1 }}–{{
              axis === 'rows'
                ? result.row_offset + result.rows.length
                : result.column_offset + result.columns.length
            }}
            / {{ axis === 'rows' ? result.row_total : result.column_total }}</span
          >
          <button
            class="secondary-action compact-action"
            :aria-label="`${t(`heatmap.${axis}`)} ${t('heatmap.previous')}`"
            :disabled="
              resultQuery.isFetching.value ||
              (axis === 'rows' ? rowOffset === 0 : columnOffset === 0)
            "
            @click="axis === 'rows' ? (rowOffset -= 50) : (columnOffset -= 50)"
            >←</button
          >
          <button
            class="secondary-action compact-action"
            :aria-label="`${t(`heatmap.${axis}`)} ${t('heatmap.next')}`"
            :disabled="
              resultQuery.isFetching.value ||
              (axis === 'rows'
                ? rowOffset + 50 >= result.row_total
                : columnOffset + 50 >= result.column_total)
            "
            @click="axis === 'rows' ? (rowOffset += 50) : (columnOffset += 50)"
            >→</button
          >
        </div>
      </div>
      <div class="section-card mt-5" aria-live="polite"
        ><h3 class="font-semibold">{{ t('heatmap.details') }}</h3>
        <p v-if="!selected" class="mt-2 text-sm text-slate-500">{{ t('heatmap.selectCell') }}</p>
        <template v-else
          ><p class="mt-2 break-all">{{ selected.row }} · {{ label(selected.column) }}</p
          ><dl class="heatmap-stats mt-3">
            <template
              v-for="key in [
                'value',
                'count',
                'positive',
                'negative',
                'prevalence',
                'intersection',
                'union',
                'row_support',
                'column_support',
              ] as const"
              :key="key"
              ><div v-if="selected[key] !== undefined"
                ><dt>{{ label(key) }}</dt
                ><dd>{{ format(selected[key]) }}</dd></div
              ></template
            > </dl
          ><p v-if="selected.reason" class="mt-2 text-sm">{{
            selected.reason === 'noData' ? t('heatmap.noData') : explain(selected.reason)
          }}</p
          ><p v-if="selected.low_support" class="mt-2 text-sm">{{ t('heatmap.low') }}</p
          ><p v-if="selected.imbalanced_support" class="mt-2 text-sm">{{
            t('heatmap.imbalanced')
          }}</p></template
        >
      </div>
      <details class="mt-5"
        ><summary class="cursor-pointer font-medium">{{ t('heatmap.table') }}</summary
        ><div class="mt-3 max-h-96 overflow-auto"
          ><table class="w-full text-left text-sm"
            ><thead
              ><tr
                ><th>{{ t('heatmap.rows') }}</th
                ><th>{{ t('heatmap.columns') }}</th
                ><th>{{ t('heatmap.value') }}</th
                ><th>{{ t('heatmap.count') }}</th></tr
              ></thead
            ><tbody
              ><tr v-for="cell in renderedCells" :key="JSON.stringify([cell.row, cell.column])"
                ><td :title="cell.row">{{ shortGroup(cell.row) }}</td
                ><td>{{ label(cell.column) }}</td
                ><td
                  ><button class="underline" @click="selected = cell">{{
                    format(visibleCellValue(cell, result.params.kind, showSmall))
                  }}</button></td
                ><td>{{ cell.count }}</td></tr
              ></tbody
            ></table
          ></div
        ></details
      >
      <details class="mt-5"
        ><summary class="cursor-pointer font-medium">{{ t('heatmap.source') }}</summary
        ><pre class="mt-3 overflow-auto whitespace-pre-wrap break-all text-xs">{{
          JSON.stringify(
            {
              provenance: result.provenance,
              operating_point: result.operating_point,
              definitions: result.definitions,
            },
            null,
            2,
          )
        }}</pre>
      </details>
    </section>
    <section v-else class="panel py-16 text-center text-sm text-slate-500">{{
      t('heatmap.empty')
    }}</section>
  </div>
</template>

<style scoped>
.heatmap-heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
}
.heatmap-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: end;
  gap: 1rem;
}
.heatmap-controls label {
  display: grid;
  gap: 0.5rem;
  font-size: 0.8rem;
  font-weight: 600;
}
.heatmap-controls select,
.heatmap-controls input[type='search'] {
  min-height: 2.7rem;
  min-width: 11rem;
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--line-soft);
  border-radius: 0.5rem;
  background: var(--surface-muted);
  color: inherit;
}
.heatmap-controls .heatmap-checkbox {
  display: flex;
  align-items: center;
  min-height: 2.7rem;
}
.heatmap-stats {
  display: flex;
  gap: 1rem 2rem;
  flex-wrap: wrap;
}
.heatmap-stats dt {
  font-size: 0.75rem;
  color: var(--color-muted);
}
.heatmap-stats dd {
  font-variant-numeric: tabular-nums;
  margin-top: 0.25rem;
}
th,
td {
  padding: 0.5rem;
  border-bottom: 1px solid var(--line-soft);
}
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
@media (max-width: 640px) {
  .heatmap-controls > label {
    width: 100%;
  }
}
</style>
