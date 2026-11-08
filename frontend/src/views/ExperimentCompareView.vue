<script setup lang="ts">
import { normalizeTaskType } from '../taskTerminology'
import { ArrowLeft, Download, X } from '@lucide/vue'
import { useQuery } from '@tanstack/vue-query'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { api } from '../api/client'
import { useWorkspaceAccess } from '../workspaceAccess'
import PageHeader from '../components/PageHeader.vue'
import ComparisonCurveChart from '../components/ComparisonCurveChart.vue'
import ComparisonMetricChart from '../components/ComparisonMetricChart.vue'
import {
  buildConfigDiff,
  buildJsonReport,
  buildMarkdownReport,
  buildMetricRows,
  comparisonFilename,
  downloadTextFile,
  experimentScore,
  normalizeComparisonIds,
} from '../experimentComparison'
import { modelRouteSummary } from '../modelDisplay'
import { experimentDisplayName, formatLocalizedDate } from '../localization'
import type {
  ComparisonMetricKey,
  ComparisonMetricRow,
  ComparisonSplitKey,
  ConfigDiffRow,
  DatasetRecord,
  ExperimentAnalysis,
  ExperimentRecord,
} from '../types'

type CurveKind = 'roc' | 'pr'

const { isGuest, canUseWorkspace, requireLogin } = useWorkspaceAccess()

const route = useRoute()
const router = useRouter()
const { locale, t } = useI18n()
const experiments = useQuery({
  queryKey: ['experiments'],
  queryFn: api.experiments,
  enabled: canUseWorkspace,
})
const datasets = useQuery({
  queryKey: ['datasets'],
  queryFn: api.datasets,
  enabled: canUseWorkspace,
})

const sortMetric = ref<ComparisonMetricKey>('f1')
const sortSplit = ref<ComparisonSplitKey>('test')
const metricChartKey = ref('val_f1')
const curveSplit = ref<ComparisonSplitKey>('test')
const curveKind = ref<CurveKind>('roc')
const showOnlyDiff = ref(true)

const metricOptions: ComparisonMetricKey[] = [
  'f1',
  'auc',
  'aupr',
  'acc',
  'balanced_accuracy',
  'precision',
  'recall',
  'specificity',
  'mcc',
]
const splitOptions: ComparisonSplitKey[] = ['test', 'validation']
const metricChartOptions = ['val_f1', 'val_auc', 'train_loss', 'val_loss']

const selectedIds = computed(() => normalizeComparisonIds(route.query.ids))
const experimentById = computed(
  () => new Map((experiments.data.value ?? []).map((item) => [item.id, item])),
)
const datasetById = computed(
  () => new Map((datasets.data.value ?? []).map((item) => [item.id, item])),
)
const selectedExperiments = computed(() =>
  selectedIds.value
    .map((id) => experimentById.value.get(id))
    .filter((experiment): experiment is ExperimentRecord => Boolean(experiment)),
)
const missingIds = computed(() => selectedIds.value.filter((id) => !experimentById.value.has(id)))
const metricRows = computed(() => buildMetricRows(selectedExperiments.value))
const configRows = computed(() => buildConfigDiff(selectedExperiments.value))
const visibleConfigRows = computed(() =>
  showOnlyDiff.value ? configRows.value.filter((row) => row.changed) : configRows.value,
)
const taskCount = computed(
  () => new Set(selectedExperiments.value.map((experiment) => experiment.task_type)).size,
)
const datasetCount = computed(
  () => new Set(selectedExperiments.value.map((experiment) => experiment.dataset_id)).size,
)
const hasCompatibilityWarning = computed(() => taskCount.value > 1 || datasetCount.value > 1)
const sortedByScore = computed(() =>
  selectedExperiments.value
    .map((experiment, index) => ({
      experiment,
      index,
      score: experimentScore(experiment, sortSplit.value, sortMetric.value).value,
    }))
    .sort((left, right) => {
      if (left.score === undefined && right.score === undefined) return left.index - right.index
      if (left.score === undefined) return 1
      if (right.score === undefined) return -1
      return right.score - left.score || left.index - right.index
    })
    .map((item) => item.experiment),
)
const bestExperiment = computed(() => {
  const first = sortedByScore.value[0]
  if (!first) return undefined
  const score = experimentScore(first, sortSplit.value, sortMetric.value).value
  return score === undefined ? undefined : first
})

const comparisonAnalyses = useQuery({
  queryKey: computed(() => ['comparison-analysis', selectedIds.value]),
  queryFn: async () => {
    const pairs = await Promise.all(
      selectedExperiments.value.map(async (experiment) => {
        const analysis = await api.experimentAnalysis(experiment.id).catch(() => undefined)
        return [experiment.id, analysis] as const
      }),
    )
    return Object.fromEntries(pairs) as Record<string, ExperimentAnalysis | undefined>
  },
  enabled: computed(() => canUseWorkspace.value && selectedExperiments.value.length > 0),
})

const analysisMap = computed(() => comparisonAnalyses.data.value ?? {})

function reportPayload() {
  const datasetRecord = Object.fromEntries(
    selectedExperiments.value.map((experiment) => [
      experiment.dataset_id,
      datasetById.value.get(experiment.dataset_id) as DatasetRecord | undefined,
    ]),
  )
  return {
    generated_at: new Date().toISOString(),
    experiments: selectedExperiments.value,
    datasets: datasetRecord,
    metric_rows: metricRows.value,
    config_diff: configRows.value,
    analyses: analysisMap.value,
  }
}

function downloadMarkdown() {
  if (!requireLogin()) return
  const content = buildMarkdownReport(reportPayload(), { locale: locale.value, translate: t })
  downloadTextFile(comparisonFilename('md'), content, 'text/markdown;charset=utf-8')
}

function downloadJson() {
  if (!requireLogin()) return
  const content = JSON.stringify(buildJsonReport(reportPayload()), null, 2)
  downloadTextFile(comparisonFilename('json'), content, 'application/json;charset=utf-8')
}

function setIds(ids: string[]) {
  router.replace({
    path: '/experiments/compare',
    query: ids.length ? { ids: ids.join(',') } : {},
  })
}

function removeExperiment(id: string) {
  setIds(selectedIds.value.filter((item) => item !== id))
}

function formatMetric(value: unknown) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric.toFixed(4) : '-'
}

function metricLabel(metric: string) {
  return t(`leaderboard.metrics.${metric}`)
}

function chartMetricLabel(metric: string) {
  if (metric === 'train_loss') return t('compare.trainLoss')
  if (metric === 'val_loss') return t('compare.validationLoss')
  const match = /^val_(.+)$/.exec(metric)
  return match ? `${t('detail.validation')} ${metricLabel(match[1])}` : metric
}

function splitLabel(split: ComparisonSplitKey) {
  return split === 'test' ? t('detail.test') : t('detail.validation')
}

function statusLabel(status: string) {
  return t(`status.${status}`)
}

function statusClass(status: string) {
  if (status === 'succeeded') return 'status-pill-success'
  if (status === 'failed') return 'status-pill-danger'
  if (status === 'queued' || status === 'running') return 'status-pill-warning'
  return 'status-pill-neutral'
}

function taskLabel(taskType: string) {
  taskType = normalizeTaskType(taskType)
  return t(`tasks.${taskType}.label`)
}

function datasetName(experiment: ExperimentRecord) {
  return datasetById.value.get(experiment.dataset_id)?.name ?? t('experiments.unknownDataset')
}

function formatDate(value?: string | null) {
  return formatLocalizedDate(
    value,
    locale.value,
    {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    },
    t('experiments.notAvailable'),
  )
}

function displayExperimentName(experiment: ExperimentRecord) {
  return experimentDisplayName(experiment, t)
}

function durationSeconds(experiment: ExperimentRecord) {
  if (!experiment.started_at || !experiment.finished_at) return undefined
  const started = new Date(experiment.started_at).getTime()
  const finished = new Date(experiment.finished_at).getTime()
  if (!Number.isFinite(started) || !Number.isFinite(finished) || finished < started)
    return undefined
  return Math.round((finished - started) / 1000)
}

function formatDuration(seconds?: number) {
  if (seconds === undefined) return t('experiments.notAvailable')
  if (seconds < 60) return t('leaderboard.seconds', { count: seconds })
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds % 60
  if (minutes < 60) {
    return remainder
      ? t('leaderboard.minutesSeconds', { minutes, seconds: remainder })
      : t('leaderboard.minutes', { count: minutes })
  }
  const hours = Math.floor(minutes / 60)
  const restMinutes = minutes % 60
  return restMinutes
    ? t('leaderboard.hoursMinutes', { hours, minutes: restMinutes })
    : t('leaderboard.hours', { count: hours })
}

function cellValue(row: ComparisonMetricRow, split: ComparisonSplitKey, experimentId: string) {
  return row[split].find((cell) => cell.experiment_id === experimentId)?.value
}

function bestMetricValue(row: ComparisonMetricRow, split: ComparisonSplitKey) {
  const values = row[split]
    .map((cell) => cell.value)
    .filter((value): value is number => value !== undefined && Number.isFinite(value))
  return values.length ? Math.max(...values) : undefined
}

function isBestCell(row: ComparisonMetricRow, split: ComparisonSplitKey, experimentId: string) {
  const value = cellValue(row, split, experimentId)
  const best = bestMetricValue(row, split)
  return value !== undefined && best !== undefined && Math.abs(value - best) < 1e-12
}

function isSortBasisRow(metric: ComparisonMetricKey, split: ComparisonSplitKey) {
  return metric === sortMetric.value && split === sortSplit.value
}

function configValue(row: ConfigDiffRow, experimentId: string) {
  return row.values[experimentId] ?? '-'
}

function configSource(row: ConfigDiffRow, experimentId: string) {
  return row.sources[experimentId] ?? ''
}
</script>

<template>
  <section class="grid gap-6">
    <PageHeader :eyebrow="t('compare.eyebrow')" :title="t('compare.title')" compact>
      <template #actions>
        <RouterLink to="/experiments" class="tertiary-action">
          <ArrowLeft aria-hidden="true" />{{ t('compare.backToExperiments') }}
        </RouterLink>
        <button
          type="button"
          class="secondary-action compact-action"
          :disabled="!isGuest && selectedExperiments.length < 2"
          @click="downloadMarkdown"
        >
          <Download aria-hidden="true" />
          {{ t('compare.exportMarkdown') }}
        </button>
        <button
          type="button"
          class="secondary-action compact-action"
          :disabled="!isGuest && selectedExperiments.length < 2"
          @click="downloadJson"
        >
          <Download aria-hidden="true" />
          {{ t('compare.exportJson') }}
        </button>
      </template>
    </PageHeader>

    <div class="grid gap-3 sm:grid-cols-3">
      <div class="metric-chip">
        <div class="text-xs font-semibold uppercase tracking-[0.16em] text-stone-500">
          {{ t('compare.selected') }}
        </div>
        <div class="stat-number mt-1 text-3xl">{{ selectedExperiments.length }}</div>
      </div>
      <div class="metric-chip">
        <div class="text-xs font-semibold uppercase tracking-[0.16em] text-stone-500">
          {{ t('experiments.task') }}
        </div>
        <div class="mt-1 text-base font-bold leading-9">
          {{
            taskCount === 1 && selectedExperiments[0]
              ? taskLabel(selectedExperiments[0].task_type)
              : t('compare.mixed')
          }}
        </div>
      </div>
      <div class="metric-chip">
        <div class="text-xs font-semibold uppercase tracking-[0.16em] text-stone-500">
          {{ t('experiments.dataset') }}
        </div>
        <div class="mt-1 text-base font-bold leading-9">
          {{
            datasetCount === 1 && selectedExperiments[0]
              ? datasetName(selectedExperiments[0])
              : t('compare.mixed')
          }}
        </div>
      </div>
    </div>

    <p v-if="missingIds.length" class="error-card">
      {{ t('compare.missingExperiments', { count: missingIds.length }) }}
    </p>
    <p
      v-if="hasCompatibilityWarning"
      class="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800"
    >
      {{ t('compare.compatibilityWarning') }}
    </p>

    <div v-if="experiments.isLoading.value" class="panel text-sm text-slate-500">
      {{ t('compare.loading') }}
    </div>
    <div v-else-if="selectedExperiments.length < 2" class="panel">
      <p class="eyebrow">{{ t('compare.emptyEyebrow') }}</p>
      <p class="mt-4 text-sm leading-6 text-slate-600">{{ t('compare.needTwoExperiments') }}</p>
      <RouterLink to="/experiments" class="primary-action mt-5 inline-flex">
        <ArrowLeft aria-hidden="true" />
        {{ t('compare.chooseExperiments') }}
      </RouterLink>
    </div>

    <template v-else>
      <div class="panel">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <p class="eyebrow">{{ t('compare.experimentSet') }}</p>
          <span v-if="bestExperiment" class="dataset-chip text-brass">
            {{
              t('compare.bestBy', {
                name: displayExperimentName(bestExperiment),
                metric: metricLabel(sortMetric),
                split: splitLabel(sortSplit),
              })
            }}
          </span>
        </div>
        <div class="mt-5 overflow-x-auto">
          <table class="min-w-full border-separate border-spacing-0 text-left text-sm">
            <thead>
              <tr class="text-xs font-bold uppercase tracking-[0.16em] text-stone-500">
                <th class="border-b border-[var(--line-soft)] px-3 py-3">{{
                  t('compare.experiment')
                }}</th>
                <th class="border-b border-[var(--line-soft)] px-3 py-3">{{
                  t('experiments.task')
                }}</th>
                <th class="border-b border-[var(--line-soft)] px-3 py-3">{{
                  t('experiments.dataset')
                }}</th>
                <th class="border-b border-[var(--line-soft)] px-3 py-3">{{
                  t('leaderboard.route')
                }}</th>
                <th class="border-b border-[var(--line-soft)] px-3 py-3">{{
                  t('leaderboard.finishedAt')
                }}</th>
                <th class="border-b border-[var(--line-soft)] px-3 py-3">{{
                  t('leaderboard.actions')
                }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="experiment in selectedExperiments" :key="experiment.id" class="align-top">
                <td class="min-w-[14rem] border-b border-[var(--line-soft)] px-3 py-4">
                  <div class="grid justify-items-start gap-2">
                    <RouterLink
                      :to="`/experiments/${experiment.id}`"
                      class="font-semibold leading-6 hover:text-[var(--color-accent-dark)]"
                    >
                      {{ displayExperimentName(experiment) }}
                    </RouterLink>
                    <span class="status-pill" :class="statusClass(experiment.status)">
                      {{ statusLabel(experiment.status) }}
                    </span>
                  </div>
                </td>
                <td class="min-w-[11rem] border-b border-[var(--line-soft)] px-3 py-4">{{
                  taskLabel(experiment.task_type)
                }}</td>
                <td
                  class="min-w-[12rem] max-w-[14rem] truncate border-b border-[var(--line-soft)] px-3 py-4"
                  >{{ datasetName(experiment) }}</td
                >
                <td
                  class="min-w-[16rem] border-b border-[var(--line-soft)] px-3 py-4 text-slate-600"
                >
                  {{ modelRouteSummary(experiment.config) || t('models.configurationUnavailable') }}
                </td>
                <td class="min-w-[10rem] border-b border-[var(--line-soft)] px-3 py-4">
                  <div>{{ formatDate(experiment.finished_at ?? experiment.updated_at) }}</div>
                  <div class="mt-1 text-xs text-slate-500">{{
                    formatDuration(durationSeconds(experiment))
                  }}</div>
                </td>
                <td class="border-b border-[var(--line-soft)] px-3 py-4">
                  <button
                    type="button"
                    class="tertiary-action compact-action"
                    @click="removeExperiment(experiment.id)"
                  >
                    <X aria-hidden="true" />
                    {{ t('compare.remove') }}
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="panel">
        <div class="grid gap-4 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
          <div class="max-w-3xl">
            <p class="eyebrow">{{ t('compare.metricMatrix') }}</p>
            <p class="mt-3 text-sm text-slate-500">{{ t('compare.metricMatrixHint') }}</p>
            <p class="mt-2 text-xs font-semibold text-stone-500">
              {{
                t('compare.metricMatrixSortBasis', {
                  metric: metricLabel(sortMetric),
                  split: splitLabel(sortSplit),
                })
              }}
            </p>
          </div>
          <div class="grid w-full gap-3 sm:grid-cols-2 lg:w-[30rem]">
            <label class="grid gap-2 text-sm font-semibold">
              {{ t('leaderboard.sortMetric') }}
              <select v-model="sortMetric" class="field-control">
                <option v-for="metric in metricOptions" :key="metric" :value="metric">
                  {{ metricLabel(metric) }}
                </option>
              </select>
            </label>
            <label class="grid gap-2 text-sm font-semibold">
              {{ t('leaderboard.split') }}
              <select v-model="sortSplit" class="field-control">
                <option v-for="split in splitOptions" :key="split" :value="split">
                  {{ splitLabel(split) }}
                </option>
              </select>
            </label>
          </div>
        </div>
        <div class="mt-5 overflow-x-auto">
          <table class="min-w-full border-separate border-spacing-0 text-left text-sm">
            <thead>
              <tr class="text-xs font-bold uppercase tracking-[0.16em] text-stone-500">
                <th class="border-b border-[var(--line-soft)] px-3 py-3">{{
                  t('compare.metric')
                }}</th>
                <th class="border-b border-[var(--line-soft)] px-3 py-3">{{
                  t('leaderboard.split')
                }}</th>
                <th
                  v-for="experiment in sortedByScore"
                  :key="experiment.id"
                  class="min-w-[10rem] border-b border-[var(--line-soft)] px-3 py-3"
                >
                  {{ displayExperimentName(experiment) }}
                </th>
              </tr>
            </thead>
            <tbody>
              <template v-for="row in metricRows" :key="row.metric">
                <tr
                  v-for="split in splitOptions"
                  :key="`${row.metric}-${split}`"
                  :class="isSortBasisRow(row.metric, split) ? 'bg-[var(--color-success-bg)]' : ''"
                >
                  <td class="border-b border-[var(--line-soft)] px-3 py-3 font-semibold">{{
                    metricLabel(row.metric)
                  }}</td>
                  <td
                    class="border-b border-[var(--line-soft)] px-3 py-3"
                    :class="
                      isSortBasisRow(row.metric, split)
                        ? 'font-semibold text-[var(--color-success-text)]'
                        : 'text-slate-500'
                    "
                  >
                    {{ splitLabel(split) }}
                  </td>
                  <td
                    v-for="experiment in sortedByScore"
                    :key="experiment.id"
                    class="border-b border-[var(--line-soft)] px-3 py-3"
                  >
                    <span
                      class="rounded-md px-2 py-1 font-semibold"
                      :class="
                        isBestCell(row, split, experiment.id)
                          ? 'bg-emerald-50 text-emerald-700'
                          : 'text-slate-700'
                      "
                    >
                      {{ formatMetric(cellValue(row, split, experiment.id)) }}
                    </span>
                  </td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
      </div>

      <div class="grid gap-6 xl:grid-cols-2">
        <div class="panel">
          <div class="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p class="eyebrow">{{ t('compare.trainingCurves') }}</p>
              <p class="mt-3 text-sm text-slate-500">{{ t('compare.trainingCurvesHint') }}</p>
            </div>
            <div class="mode-toggle flex flex-wrap gap-1">
              <button
                v-for="metric in metricChartOptions"
                :key="metric"
                type="button"
                class="mode-toggle-button"
                :class="{ 'mode-toggle-button-active': metricChartKey === metric }"
                :aria-pressed="metricChartKey === metric"
                @click="metricChartKey = metric"
              >
                {{ chartMetricLabel(metric) }}
              </button>
            </div>
          </div>
          <ComparisonMetricChart
            class="mt-5"
            :experiments="selectedExperiments"
            :metric-key="metricChartKey"
          />
        </div>

        <div class="panel">
          <div class="flex flex-wrap items-end justify-between gap-4">
            <div>
              <p class="eyebrow">{{ t('compare.analysisCurves') }}</p>
              <p class="mt-3 text-sm text-slate-500">{{ t('compare.analysisCurvesHint') }}</p>
            </div>
            <div class="grid gap-3 sm:grid-cols-2">
              <label class="grid gap-2 text-sm font-semibold">
                {{ t('leaderboard.split') }}
                <select v-model="curveSplit" class="field-control">
                  <option v-for="split in splitOptions" :key="split" :value="split">
                    {{ splitLabel(split) }}
                  </option>
                </select>
              </label>
              <label class="grid gap-2 text-sm font-semibold">
                {{ t('compare.curveType') }}
                <select v-model="curveKind" class="field-control">
                  <option value="roc">ROC</option>
                  <option value="pr">PR</option>
                </select>
              </label>
            </div>
          </div>
          <p
            v-if="comparisonAnalyses.isLoading.value"
            class="section-card mt-5 text-sm text-stone-500"
          >
            {{ t('compare.loadingAnalysis') }}
          </p>
          <ComparisonCurveChart
            v-else
            class="mt-5"
            :experiments="selectedExperiments"
            :analyses="analysisMap"
            :split="curveSplit"
            :kind="curveKind"
          />
        </div>
      </div>

      <div class="panel">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p class="eyebrow">{{ t('compare.configDiff') }}</p>
            <p class="mt-3 text-sm text-slate-500">{{ t('compare.configDiffHint') }}</p>
          </div>
          <label class="inline-flex items-center gap-2 text-sm font-semibold">
            <input
              v-model="showOnlyDiff"
              type="checkbox"
              class="h-4 w-4 accent-[var(--color-accent-dark)]"
            />
            {{ t('compare.showOnlyDifferences') }}
          </label>
        </div>
        <div v-if="visibleConfigRows.length" class="mt-5 overflow-x-auto">
          <table class="min-w-full border-separate border-spacing-0 text-left text-sm">
            <thead>
              <tr class="text-xs font-bold uppercase tracking-[0.16em] text-stone-500">
                <th class="border-b border-[var(--line-soft)] px-3 py-3">{{
                  t('compare.configKey')
                }}</th>
                <th
                  v-for="experiment in selectedExperiments"
                  :key="experiment.id"
                  class="min-w-[13rem] border-b border-[var(--line-soft)] px-3 py-3"
                >
                  {{ displayExperimentName(experiment) }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in visibleConfigRows" :key="row.key" class="align-top">
                <td class="border-b border-[var(--line-soft)] px-3 py-3">
                  <span class="font-semibold">{{ row.key }}</span>
                  <span v-if="row.highlighted" class="config-chip ml-2">{{
                    t('compare.keyParameter')
                  }}</span>
                </td>
                <td
                  v-for="experiment in selectedExperiments"
                  :key="experiment.id"
                  class="max-w-xs border-b border-[var(--line-soft)] px-3 py-3"
                >
                  <span
                    class="block truncate rounded-md px-2 py-1 text-xs"
                    :class="row.changed ? 'bg-amber-50 text-amber-800' : 'text-slate-600'"
                    :title="configValue(row, experiment.id)"
                  >
                    {{ configValue(row, experiment.id) }}
                    <span
                      v-if="configSource(row, experiment.id) === 'default'"
                      class="config-chip ml-1"
                    >
                      {{ t('compare.defaultValue') }}
                    </span>
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p v-else class="section-card mt-5 text-sm text-stone-500">
          {{ t('compare.noConfigDifferences') }}
        </p>
      </div>
    </template>
  </section>
</template>
