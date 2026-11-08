<script setup lang="ts">
import { normalizeTaskType } from '../taskTerminology'
import { useQuery } from '@tanstack/vue-query'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { api } from '../api/client'
import { useWorkspaceAccess } from '../workspaceAccess'
import PageHeader from '../components/PageHeader.vue'
import { COMPARISON_MAX_EXPERIMENTS } from '../experimentComparison'
import { localizedApiError } from '../errorLocalization'
import { formatLocalizedDate, generatedNameForDisplay } from '../localization'
import { modelRouteSummary } from '../modelDisplay'
import { useAuthStore } from '../stores/auth'
import {
  ArrowDownWideNarrow,
  ArrowRight,
  Cpu,
  FlaskConical,
  LayoutList,
  RefreshCw,
  Scale,
  Zap,
} from '@lucide/vue'
import type { LeaderboardRow } from '../types'

const { isGuest, canUseWorkspace, requireLogin } = useWorkspaceAccess()

const { locale, t, te } = useI18n()
const router = useRouter()
const auth = useAuthStore()
const capabilities = useQuery({ queryKey: ['capabilities'], queryFn: api.capabilities })
const datasets = useQuery({
  queryKey: ['datasets'],
  queryFn: api.datasets,
  enabled: canUseWorkspace,
})

const selectedTask = ref('')
const selectedDataset = ref('')
const selectedMetric = ref('f1')
const selectedSplit = ref('auto')
const selectedStatus = ref('succeeded')
const inferenceFilter = ref('all')
const globalView = ref(false)
const selectedCompareIds = ref<string[]>([])

const metricOptions = [
  'f1',
  'auc',
  'aupr',
  'acc',
  'sacc',
  'balanced_accuracy',
  'precision',
  'recall',
  'specificity',
  'mcc',
  'auc_micro',
  'aupr_micro',
]
const splitOptions = ['auto', 'test', 'validation']
const statusOptions = ['succeeded', 'failed', 'canceled', 'running', 'queued', 'all']

const filteredDatasets = computed(() =>
  (datasets.data.value ?? []).filter(
    (dataset) => !selectedTask.value || dataset.task_type === selectedTask.value,
  ),
)

const leaderboardParams = computed(() => ({
  task_type: selectedTask.value || undefined,
  dataset_id: selectedDataset.value || undefined,
  metric: selectedMetric.value,
  split: selectedSplit.value,
  supports_inference: inferenceFilter.value === 'ready' ? true : undefined,
  status: selectedStatus.value,
  limit: 200,
  global_view: auth.user?.role === 'admin' ? globalView.value : false,
}))

const leaderboard = useQuery({
  queryKey: computed(() => ['leaderboard', leaderboardParams.value]),
  queryFn: () => api.leaderboard(leaderboardParams.value),
  enabled: canUseWorkspace,
})

const rows = computed(() => leaderboard.data.value?.rows ?? [])
const readyCount = computed(() => rows.value.filter((row) => row.model?.supports_inference).length)
const modelCount = computed(() => rows.value.filter((row) => row.model).length)
const compareSelectionFull = computed(
  () => selectedCompareIds.value.length >= COMPARISON_MAX_EXPERIMENTS,
)
const selectedCompareCount = computed(() => selectedCompareIds.value.length)
const compareCtaLabel = computed(() =>
  selectedCompareCount.value < 2 ? t('leaderboard.compareNeedTwo') : t('compare.compareSelected'),
)

const BINARY_DEFAULT_METRICS: Record<string, string> = {
  pep_protein_interaction: 'auc',
  pep_hlaii: 'auc',
  pep_hlai: 'auc',
}

watch(selectedTask, () => {
  if (selectedTask.value && selectedTask.value in BINARY_DEFAULT_METRICS) {
    selectedMetric.value = BINARY_DEFAULT_METRICS[selectedTask.value]
  } else if (selectedTask.value === 'pep_function') {
    selectedMetric.value = 'f1'
  } else {
    selectedMetric.value = 'f1'
  }
  if (!selectedDataset.value) return
  const dataset = datasets.data.value?.find((item) => item.id === selectedDataset.value)
  if (dataset && selectedTask.value && dataset.task_type !== selectedTask.value) {
    selectedDataset.value = ''
  }
})

watch(rows, (items) => {
  const ids = new Set(items.map((row) => row.experiment.id))
  selectedCompareIds.value = selectedCompareIds.value.filter((id) => ids.has(id))
})

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
    t('leaderboard.notAvailable'),
  )
}

function experimentName(row: LeaderboardRow) {
  return generatedNameForDisplay(
    row.experiment.name,
    row.experiment.generated_name,
    t,
    row.experiment.config,
    row.experiment.task_type,
  )
}

function modelName(row: LeaderboardRow) {
  if (!row.model) return t('leaderboard.noModel')
  if (row.model.name_is_custom) return row.model.name
  if (row.model.name_is_custom === false || row.model.name === row.experiment.name)
    return experimentName(row)
  return generatedNameForDisplay(row.model.name, undefined, t)
}

function loadError(error: unknown) {
  return localizedApiError(error, t, te)
}

function formatMetric(value: unknown) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric.toFixed(4) : t('leaderboard.noScore')
}

function formatDuration(seconds?: number | null) {
  if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) {
    return t('leaderboard.notAvailable')
  }
  const total = Math.round(Number(seconds))
  if (total < 60) return t('leaderboard.seconds', { count: total })
  const minutes = Math.floor(total / 60)
  const remainder = total % 60
  if (minutes < 60)
    return remainder
      ? t('leaderboard.minutesSeconds', { minutes, seconds: remainder })
      : t('leaderboard.minutes', { count: minutes })
  const hours = Math.floor(minutes / 60)
  const restMinutes = minutes % 60
  return restMinutes
    ? t('leaderboard.hoursMinutes', { hours, minutes: restMinutes })
    : t('leaderboard.hours', { count: hours })
}

function taskLabel(taskType: string) {
  taskType = normalizeTaskType(taskType)
  return t(`tasks.${taskType}.label`)
}

function metricLabel(metric: string) {
  return t(`leaderboard.metrics.${metric}`)
}

function splitLabel(split: string) {
  if (split === 'test') return t('detail.test')
  if (split === 'validation' || split === 'val') return t('detail.validation')
  return t('leaderboard.autoSplit')
}

function scoreSourceLabel(source?: string) {
  if (!source) return t('leaderboard.noScore')
  if (source === 'best_val_f1') return t('leaderboard.bestValidationF1')
  const match = /^(test|val)_(.+)$/.exec(source)
  if (!match) return source
  return `${splitLabel(match[1] === 'val' ? 'validation' : 'test')} ${metricLabel(match[2])}`
}

function medalClass(rank: number) {
  if (rank === 1) return 'leaderboard-medal leaderboard-medal-gold'
  if (rank === 2) return 'leaderboard-medal leaderboard-medal-silver'
  if (rank === 3) return 'leaderboard-medal leaderboard-medal-bronze'
  return 'leaderboard-rank-block'
}

function scoreWidth(value: unknown) {
  const n = Number(value)
  if (!Number.isFinite(n)) return '0%'
  return `${Math.max(0, Math.min(1, n)) * 100}%`
}

function statusLabel(status: string) {
  return status === 'all' ? t('leaderboard.allStatuses') : t(`status.${status}`)
}

function statusClass(status: string) {
  if (status === 'succeeded') return 'status-pill-success'
  if (status === 'failed') return 'status-pill-danger'
  if (status === 'queued' || status === 'running') return 'status-pill-warning'
  return 'status-pill-neutral'
}

function metricItems(row: LeaderboardRow, split: 'test' | 'validation') {
  const values = row.metrics?.[split] ?? {}
  return ['f1', 'auc', 'aupr']
    .filter((metric) => values[metric] !== undefined)
    .map((metric) => ({
      metric,
      label: metricLabel(metric),
      value: formatMetric(values[metric]),
    }))
}

function routeSummary(row: LeaderboardRow) {
  return modelRouteSummary(row.config_summary) || t('models.configurationUnavailable')
}

function resetFilters() {
  selectedTask.value = ''
  selectedDataset.value = ''
  selectedMetric.value = 'f1'
  selectedSplit.value = 'auto'
  selectedStatus.value = 'succeeded'
  inferenceFilter.value = 'all'
  globalView.value = false
}

function isCompareSelected(experimentId: string) {
  return selectedCompareIds.value.includes(experimentId)
}

function compareDisabled(experimentId: string) {
  return !isCompareSelected(experimentId) && compareSelectionFull.value
}

function compareToggleLabel(experimentId: string) {
  return isCompareSelected(experimentId)
    ? t('leaderboard.selectedForCompare')
    : t('compare.compare')
}

function rankTone(rank: number) {
  if (rank === 1) return 'leaderboard-row-gold'
  if (rank === 2) return 'leaderboard-row-silver'
  if (rank === 3) return 'leaderboard-row-bronze'
  return ''
}

function toggleCompareExperiment(experimentId: string, selected: boolean) {
  if (!selected) {
    selectedCompareIds.value = selectedCompareIds.value.filter((id) => id !== experimentId)
    return
  }
  if (isCompareSelected(experimentId) || compareSelectionFull.value) return
  selectedCompareIds.value = [...selectedCompareIds.value, experimentId]
}

function handleCompareCheckbox(experimentId: string, event: Event) {
  const target = event.target as HTMLInputElement | null
  toggleCompareExperiment(experimentId, Boolean(target?.checked))
}

function openComparison() {
  if (!requireLogin()) return
  if (selectedCompareIds.value.length < 2) return
  router.push({ path: '/experiments/compare', query: { ids: selectedCompareIds.value.join(',') } })
}
</script>

<template>
  <section class="grid gap-6">
    <PageHeader :eyebrow="t('leaderboard.eyebrow')" :title="t('leaderboard.title')" compact>
      <template #actions>
        <RouterLink to="/experiments/new" class="primary-action">
          <FlaskConical aria-hidden="true" />
          {{ t('experiments.newExperiment') }}
        </RouterLink>
      </template>
    </PageHeader>

    <div class="grid gap-3 sm:grid-cols-3">
      <div class="metric-chip">
        <div class="flex items-center gap-3">
          <div class="leaderboard-stat-icon leaderboard-stat-icon-rows">
            <LayoutList class="h-4 w-4" />
          </div>
          <div>
            <div class="text-xs font-semibold uppercase tracking-[0.16em] text-stone-500">
              {{ t('leaderboard.visibleRows') }}
            </div>
            <div class="stat-number mt-1 text-3xl">{{ rows.length }}</div>
          </div>
        </div>
      </div>
      <div class="metric-chip">
        <div class="flex items-center gap-3">
          <div class="leaderboard-stat-icon leaderboard-stat-icon-models">
            <Cpu class="h-4 w-4" />
          </div>
          <div>
            <div class="text-xs font-semibold uppercase tracking-[0.16em] text-stone-500">
              {{ t('leaderboard.models') }}
            </div>
            <div class="stat-number mt-1 text-3xl">{{ modelCount }}</div>
          </div>
        </div>
      </div>
      <div class="metric-chip">
        <div class="flex items-center gap-3">
          <div class="leaderboard-stat-icon leaderboard-stat-icon-infer">
            <Zap class="h-4 w-4" />
          </div>
          <div>
            <div class="text-xs font-semibold uppercase tracking-[0.16em] text-stone-500">
              {{ t('leaderboard.inferenceReady') }}
            </div>
            <div class="stat-number mt-1 text-3xl">{{ readyCount }}</div>
          </div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="grid gap-4 md:grid-cols-3 xl:grid-cols-6">
        <label class="grid gap-2 text-sm font-semibold">
          {{ t('experiments.task') }}
          <select v-model="selectedTask" class="field-control">
            <option value="">{{ t('leaderboard.allTasks') }}</option>
            <option
              v-for="task in capabilities.data.value?.tasks ?? []"
              :key="task.id"
              :value="task.id"
            >
              {{ taskLabel(task.id) }}
            </option>
          </select>
        </label>
        <label class="grid gap-2 text-sm font-semibold">
          {{ t('experiments.dataset') }}
          <select v-model="selectedDataset" class="field-control">
            <option value="">{{ t('leaderboard.allDatasets') }}</option>
            <option
              v-for="dataset in filteredDatasets"
              :key="dataset.id"
              :value="dataset.id"
              :title="dataset.name"
            >
              {{ dataset.name }}
            </option>
          </select>
        </label>
        <label class="grid gap-2 text-sm font-semibold">
          {{ t('leaderboard.sortMetric') }}
          <select v-model="selectedMetric" class="field-control">
            <option v-for="metric in metricOptions" :key="metric" :value="metric">
              {{ metricLabel(metric) }}
            </option>
          </select>
        </label>
        <label class="grid gap-2 text-sm font-semibold">
          {{ t('leaderboard.split') }}
          <select v-model="selectedSplit" class="field-control">
            <option v-for="split in splitOptions" :key="split" :value="split">
              {{ splitLabel(split) }}
            </option>
          </select>
        </label>
        <label class="grid gap-2 text-sm font-semibold">
          {{ t('leaderboard.status') }}
          <select v-model="selectedStatus" class="field-control">
            <option v-for="status in statusOptions" :key="status" :value="status">
              {{ statusLabel(status) }}
            </option>
          </select>
        </label>
        <label class="grid gap-2 text-sm font-semibold">
          {{ t('leaderboard.inference') }}
          <select v-model="inferenceFilter" class="field-control">
            <option value="all">{{ t('leaderboard.allModels') }}</option>
            <option value="ready">{{ t('common.readyForInference') }}</option>
          </select>
        </label>
      </div>
      <div class="mt-4 flex flex-wrap items-center justify-between gap-3">
        <div class="flex flex-wrap items-center gap-3">
          <span class="leaderboard-sort-badge">
            <ArrowDownWideNarrow class="h-3.5 w-3.5" />
            {{
              t('leaderboard.sortedBy', {
                metric: metricLabel(selectedMetric),
                split: splitLabel(selectedSplit),
              })
            }}
          </span>
          <label
            v-if="auth.user?.role === 'admin'"
            class="inline-flex items-center gap-2 text-sm font-semibold"
          >
            <input
              v-model="globalView"
              type="checkbox"
              class="h-4 w-4 accent-[var(--color-accent-dark)]"
            />
            {{ t('leaderboard.globalView') }}
          </label>
          <span v-if="auth.user?.role === 'admin'" class="dataset-chip text-brass">
            {{ globalView ? t('leaderboard.globalScope') : t('leaderboard.myScope') }}
          </span>
        </div>
        <button type="button" class="tertiary-action" @click="resetFilters">
          <RefreshCw aria-hidden="true" />
          {{ t('leaderboard.resetFilters') }}
        </button>
      </div>
    </div>

    <div class="panel">
      <div class="leaderboard-toolbar">
        <div>
          <p class="eyebrow">{{ t('leaderboard.tableTitle') }}</p>
          <p class="mt-2 text-sm text-slate-500">{{ t('leaderboard.compareHint') }}</p>
        </div>
        <div class="leaderboard-compare-tray">
          <span class="dataset-chip text-brass">{{
            t('leaderboard.count', { count: rows.length })
          }}</span>
          <span class="dataset-chip text-brass" aria-live="polite">
            {{
              t('compare.selectedCount', {
                count: selectedCompareCount,
                max: COMPARISON_MAX_EXPERIMENTS,
              })
            }}
          </span>
          <button
            type="button"
            class="primary-action compact-action disabled:cursor-not-allowed disabled:opacity-50"
            :disabled="!isGuest && selectedCompareCount < 2"
            @click="openComparison"
          >
            <Scale aria-hidden="true" />
            {{ compareCtaLabel }}
          </button>
        </div>
      </div>

      <p v-if="leaderboard.isLoading.value" class="mt-5 text-sm text-slate-500">
        {{ t('leaderboard.loading') }}
      </p>
      <p v-else-if="leaderboard.error.value" class="mt-5 text-sm text-rose-600">
        {{ t('leaderboard.loadError', { message: loadError(leaderboard.error.value) }) }}
      </p>
      <div v-else-if="rows.length" class="mt-5">
        <div class="hidden overflow-x-auto lg:block">
          <table
            class="leaderboard-table min-w-full border-separate border-spacing-0 text-left text-sm"
          >
            <thead>
              <tr class="text-xs font-bold uppercase tracking-[0.16em] text-stone-500">
                <th class="px-3 py-3">{{ t('leaderboard.rank') }}</th>
                <th class="px-3 py-3">{{ t('leaderboard.score') }}</th>
                <th class="px-3 py-3">{{ t('leaderboard.experimentModel') }}</th>
                <th class="px-3 py-3">{{ t('experiments.task') }}</th>
                <th class="px-3 py-3">{{ t('detail.test') }}</th>
                <th class="px-3 py-3">{{ t('detail.validation') }}</th>
                <th class="px-3 py-3">{{ t('leaderboard.route') }}</th>
                <th class="px-3 py-3">{{ t('leaderboard.inference') }}</th>
                <th class="px-3 py-3">{{ t('leaderboard.finishedAt') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="row in rows"
                :key="row.experiment.id"
                class="leaderboard-row align-top"
                :class="rankTone(row.rank)"
              >
                <td class="px-3 py-4">
                  <div :class="medalClass(row.rank)">#{{ row.rank }}</div>
                </td>
                <td class="px-3 py-4">
                  <div class="leaderboard-score-block">
                    <div class="text-lg font-black">{{ formatMetric(row.score) }}</div>
                    <div class="mt-1 text-xs text-slate-500">{{
                      scoreSourceLabel(row.score_source)
                    }}</div>
                    <div class="leaderboard-score-bar">
                      <div
                        class="leaderboard-score-bar-fill"
                        :class="{ 'leaderboard-score-bar-fill-gold': row.rank === 1 }"
                        :style="{ width: scoreWidth(row.score) }"
                      />
                    </div>
                  </div>
                </td>
                <td class="min-w-[21rem] px-3 py-4">
                  <RouterLink
                    :to="`/experiments/${row.experiment.id}`"
                    class="leaderboard-experiment-name hover:text-[var(--color-accent-dark)]"
                  >
                    {{ experimentName(row) }}
                  </RouterLink>
                  <div class="mt-1 text-sm text-slate-500">
                    {{ modelName(row) }}
                  </div>
                  <div class="leaderboard-row-actions">
                    <label
                      class="leaderboard-compare-toggle"
                      :class="{
                        'leaderboard-compare-toggle-active': isCompareSelected(row.experiment.id),
                        'leaderboard-compare-toggle-disabled': compareDisabled(row.experiment.id),
                      }"
                    >
                      <input
                        type="checkbox"
                        class="sr-only"
                        :checked="isCompareSelected(row.experiment.id)"
                        :disabled="compareDisabled(row.experiment.id)"
                        @change="handleCompareCheckbox(row.experiment.id, $event)"
                      />
                      <span class="leaderboard-compare-box" aria-hidden="true" />
                      {{ compareToggleLabel(row.experiment.id) }}
                    </label>
                    <div class="leaderboard-link-actions">
                      <RouterLink
                        :to="`/experiments/${row.experiment.id}`"
                        class="emphasis-action compact-action leaderboard-mini-action"
                      >
                        <ArrowRight aria-hidden="true" />
                        {{ t('experiments.openDetail') }}
                      </RouterLink>
                      <RouterLink
                        v-if="row.model"
                        :to="{ path: '/models', query: { model: row.model.id } }"
                        class="secondary-action compact-action leaderboard-mini-action"
                      >
                        <Cpu aria-hidden="true" />
                        {{ t('leaderboard.openModel') }}
                      </RouterLink>
                    </div>
                  </div>
                  <span class="status-pill mt-3" :class="statusClass(row.experiment.status)">
                    {{ statusLabel(row.experiment.status) }}
                  </span>
                </td>
                <td class="min-w-[12rem] px-3 py-4">
                  <div class="font-semibold">{{ taskLabel(row.task_type) }}</div>
                  <div class="mt-1 min-w-0 truncate text-sm text-slate-500">
                    {{ row.dataset?.name ?? t('experiments.unknownDataset') }}
                  </div>
                </td>
                <td class="min-w-[10rem] px-3 py-4">
                  <div v-if="metricItems(row, 'test').length" class="leaderboard-metric-stack">
                    <span
                      v-for="item in metricItems(row, 'test')"
                      :key="`test-${row.experiment.id}-${item.metric}`"
                      class="leaderboard-metric-pill"
                      :class="{ 'leaderboard-metric-pill-primary': item.metric === selectedMetric }"
                    >
                      <span>{{ item.label }}</span>
                      <strong>{{ item.value }}</strong>
                    </span>
                  </div>
                  <span v-else class="text-sm text-slate-500">{{ t('leaderboard.noScore') }}</span>
                </td>
                <td class="min-w-[10rem] px-3 py-4">
                  <div
                    v-if="metricItems(row, 'validation').length"
                    class="leaderboard-metric-stack"
                  >
                    <span
                      v-for="item in metricItems(row, 'validation')"
                      :key="`validation-${row.experiment.id}-${item.metric}`"
                      class="leaderboard-metric-pill"
                      :class="{ 'leaderboard-metric-pill-primary': item.metric === selectedMetric }"
                    >
                      <span>{{ item.label }}</span>
                      <strong>{{ item.value }}</strong>
                    </span>
                  </div>
                  <span v-else class="text-sm text-slate-500">{{ t('leaderboard.noScore') }}</span>
                </td>
                <td class="min-w-[14rem] px-3 py-4 text-slate-600">
                  {{ routeSummary(row) }}
                </td>
                <td class="px-3 py-4">
                  <span
                    class="leaderboard-inference-chip"
                    :class="{ 'leaderboard-inference-ready': row.model?.supports_inference }"
                  >
                    {{
                      row.model?.supports_inference
                        ? t('common.readyForInference')
                        : t('common.metricsOnly')
                    }}
                  </span>
                  <span v-if="row.model" class="mt-2 block text-xs text-slate-500">
                    {{
                      Number(row.model.evaluation_protocol_version ?? 1) >= 2
                        ? t('models.currentEvaluationMethod')
                        : t('models.legacyProtocol')
                    }}
                  </span>
                </td>
                <td class="min-w-[8rem] px-3 py-4">
                  <div>{{
                    formatDate(row.experiment.finished_at ?? row.experiment.updated_at)
                  }}</div>
                  <div class="mt-1 text-xs text-slate-500">{{
                    formatDuration(row.duration_seconds)
                  }}</div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="grid gap-3 lg:hidden">
          <article
            v-for="row in rows"
            :key="`card-${row.experiment.id}`"
            class="leaderboard-mobile-card"
            :class="rankTone(row.rank)"
          >
            <div class="flex items-start justify-between gap-3">
              <div>
                <div :class="medalClass(row.rank)">#{{ row.rank }}</div>
                <div class="mt-2 text-2xl font-black">{{ formatMetric(row.score) }}</div>
                <div class="mt-1 text-xs text-slate-500">{{
                  scoreSourceLabel(row.score_source)
                }}</div>
                <div class="leaderboard-score-bar mt-2">
                  <div
                    class="leaderboard-score-bar-fill"
                    :class="{ 'leaderboard-score-bar-fill-gold': row.rank === 1 }"
                    :style="{ width: scoreWidth(row.score) }"
                  />
                </div>
              </div>
              <span class="status-pill" :class="statusClass(row.experiment.status)">
                {{ statusLabel(row.experiment.status) }}
              </span>
            </div>

            <RouterLink
              :to="`/experiments/${row.experiment.id}`"
              class="leaderboard-experiment-name mt-4 block hover:text-[var(--color-accent-dark)]"
            >
              {{ experimentName(row) }}
            </RouterLink>
            <div class="mt-1 text-sm text-slate-500">{{
              row.model?.name ?? t('leaderboard.noModel')
            }}</div>

            <div class="leaderboard-row-actions">
              <label
                class="leaderboard-compare-toggle"
                :class="{
                  'leaderboard-compare-toggle-active': isCompareSelected(row.experiment.id),
                  'leaderboard-compare-toggle-disabled': compareDisabled(row.experiment.id),
                }"
              >
                <input
                  type="checkbox"
                  class="sr-only"
                  :checked="isCompareSelected(row.experiment.id)"
                  :disabled="compareDisabled(row.experiment.id)"
                  @change="handleCompareCheckbox(row.experiment.id, $event)"
                />
                <span class="leaderboard-compare-box" aria-hidden="true" />
                {{ compareToggleLabel(row.experiment.id) }}
              </label>
              <div class="leaderboard-link-actions">
                <RouterLink
                  :to="`/experiments/${row.experiment.id}`"
                  class="emphasis-action compact-action leaderboard-mini-action"
                >
                  <ArrowRight aria-hidden="true" />
                  {{ t('experiments.openDetail') }}
                </RouterLink>
                <RouterLink
                  v-if="row.model"
                  :to="{ path: '/models', query: { model: row.model.id } }"
                  class="secondary-action compact-action leaderboard-mini-action"
                >
                  <Cpu aria-hidden="true" />
                  {{ t('leaderboard.openModel') }}
                </RouterLink>
              </div>
            </div>

            <div class="mt-4 grid gap-3 sm:grid-cols-2">
              <div>
                <div class="text-xs font-bold uppercase tracking-[0.14em] text-stone-500">{{
                  t('detail.test')
                }}</div>
                <div v-if="metricItems(row, 'test').length" class="leaderboard-metric-stack mt-2">
                  <span
                    v-for="item in metricItems(row, 'test')"
                    :key="`mobile-test-${row.experiment.id}-${item.metric}`"
                    class="leaderboard-metric-pill"
                    :class="{ 'leaderboard-metric-pill-primary': item.metric === selectedMetric }"
                  >
                    <span>{{ item.label }}</span>
                    <strong>{{ item.value }}</strong>
                  </span>
                </div>
                <span v-else class="mt-2 block text-sm text-slate-500">{{
                  t('leaderboard.noScore')
                }}</span>
              </div>
              <div>
                <div class="text-xs font-bold uppercase tracking-[0.14em] text-stone-500">{{
                  t('detail.validation')
                }}</div>
                <div
                  v-if="metricItems(row, 'validation').length"
                  class="leaderboard-metric-stack mt-2"
                >
                  <span
                    v-for="item in metricItems(row, 'validation')"
                    :key="`mobile-validation-${row.experiment.id}-${item.metric}`"
                    class="leaderboard-metric-pill"
                    :class="{ 'leaderboard-metric-pill-primary': item.metric === selectedMetric }"
                  >
                    <span>{{ item.label }}</span>
                    <strong>{{ item.value }}</strong>
                  </span>
                </div>
                <span v-else class="mt-2 block text-sm text-slate-500">{{
                  t('leaderboard.noScore')
                }}</span>
              </div>
            </div>

            <div class="mt-4 grid gap-3 text-sm sm:grid-cols-2">
              <div>
                <span class="text-xs font-bold uppercase tracking-[0.14em] text-stone-500">{{
                  t('experiments.task')
                }}</span>
                <div class="mt-1 font-semibold">{{ taskLabel(row.task_type) }}</div>
                <div class="min-w-0 truncate text-slate-500">{{
                  row.dataset?.name ?? t('experiments.unknownDataset')
                }}</div>
              </div>
              <div>
                <span class="text-xs font-bold uppercase tracking-[0.14em] text-stone-500">{{
                  t('leaderboard.route')
                }}</span>
                <div class="mt-1 text-slate-600">{{ routeSummary(row) }}</div>
              </div>
              <div>
                <span class="text-xs font-bold uppercase tracking-[0.14em] text-stone-500">{{
                  t('leaderboard.inference')
                }}</span>
                <div class="mt-2">
                  <span
                    class="leaderboard-inference-chip"
                    :class="{ 'leaderboard-inference-ready': row.model?.supports_inference }"
                  >
                    {{
                      row.model?.supports_inference
                        ? t('common.readyForInference')
                        : t('common.metricsOnly')
                    }}
                  </span>
                </div>
              </div>
              <div>
                <span class="text-xs font-bold uppercase tracking-[0.14em] text-stone-500">{{
                  t('leaderboard.finishedAt')
                }}</span>
                <div class="mt-1">{{
                  formatDate(row.experiment.finished_at ?? row.experiment.updated_at)
                }}</div>
                <div class="text-xs text-slate-500">{{ formatDuration(row.duration_seconds) }}</div>
              </div>
            </div>
          </article>
        </div>
      </div>

      <div v-else class="section-card mt-5">
        <p class="font-semibold">{{ t('leaderboard.emptyTitle') }}</p>
        <p class="muted-copy mt-2 text-sm leading-6">{{
          t(isGuest ? 'guest.resources' : 'leaderboard.emptyHint')
        }}</p>
      </div>
    </div>
  </section>
</template>
