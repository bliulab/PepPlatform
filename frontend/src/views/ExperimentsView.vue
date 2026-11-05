<script setup lang="ts">
import { normalizeTaskType } from '../taskTerminology'
import {
  ArrowRight,
  Check,
  FlaskConical,
  Pencil,
  RefreshCw,
  Scale,
  SquareStop,
  Trash2,
  X,
} from '@lucide/vue'
import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import { computed, nextTick, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { api } from '../api/client'
import { useWorkspaceAccess } from '../workspaceAccess'
import PageHeader from '../components/PageHeader.vue'
import { COMPARISON_MAX_EXPERIMENTS } from '../experimentComparison'
import { localizedApiError } from '../errorLocalization'
import { experimentDisplayName, formatLocalizedDate } from '../localization'
import { useExperimentDraftStore } from '../stores/experimentDraft'
import type { ExperimentRecord } from '../types'

const { isGuest, canUseWorkspace, requireLogin } = useWorkspaceAccess()

const { locale, t, te } = useI18n()
const queryClient = useQueryClient()
const router = useRouter()
const draft = useExperimentDraftStore()
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
const deletingExperimentId = ref('')
const stoppingExperimentId = ref('')
const selectedCompareIds = ref<string[]>([])
const editingExperimentId = ref('')
const editingName = ref('')
const renameError = ref('')
const renameInput = ref<HTMLInputElement | null>(null)

const datasetById = computed(
  () => new Map((datasets.data.value ?? []).map((dataset) => [dataset.id, dataset])),
)
const sortedExperiments = computed(() =>
  [...(experiments.data.value ?? [])].sort(
    (left, right) => timestamp(right.created_at) - timestamp(left.created_at),
  ),
)
const activeCount = computed(
  () =>
    sortedExperiments.value.filter((experiment) =>
      ['queued', 'running'].includes(experiment.status),
    ).length,
)
const compareSelectionFull = computed(
  () => selectedCompareIds.value.length >= COMPARISON_MAX_EXPERIMENTS,
)

const deleteMutation = useMutation({
  mutationFn: (experimentId: string) => api.deleteExperiment(experimentId),
  onSuccess: (_result, experimentId) => {
    queryClient.invalidateQueries({ queryKey: ['experiments'] })
    queryClient.invalidateQueries({ queryKey: ['models'] })
    queryClient.removeQueries({ queryKey: ['experiment', experimentId] })
    queryClient.removeQueries({ queryKey: ['experiment-logs', experimentId] })
  },
  onSettled: () => {
    deletingExperimentId.value = ''
  },
})

const cancelMutation = useMutation({
  mutationFn: (experimentId: string) => api.cancelExperiment(experimentId),
  onSuccess: (_result, experimentId) => {
    queryClient.invalidateQueries({ queryKey: ['experiments'] })
    queryClient.invalidateQueries({ queryKey: ['experiment', experimentId] })
    queryClient.invalidateQueries({ queryKey: ['experiment-logs', experimentId] })
  },
  onSettled: () => {
    stoppingExperimentId.value = ''
  },
})

const renameMutation = useMutation({
  mutationFn: ({ id, name }: { id: string; name: string }) => api.updateExperiment(id, { name }),
  onSuccess: (updated) => {
    queryClient.setQueryData<ExperimentRecord[]>(['experiments'], (items) =>
      (items ?? []).map((item) => (item.id === updated.id ? updated : item)),
    )
    queryClient.setQueryData(['experiment', updated.id], updated)
    queryClient.invalidateQueries({ queryKey: ['models'] })
    queryClient.invalidateQueries({ queryKey: ['leaderboard'] })
    queryClient.invalidateQueries({ queryKey: ['agent-execution-plan'] })
    queryClient.invalidateQueries({ queryKey: ['agent-execution-plans'] })
    queryClient.invalidateQueries({ queryKey: ['agent-campaign'] })
    cancelRename()
  },
  onError: (error) => {
    renameError.value = t('experiments.renameError', { message: errorText(error) })
  },
})

watch(sortedExperiments, (items) => {
  const ids = new Set(items.map((experiment) => experiment.id))
  selectedCompareIds.value = selectedCompareIds.value.filter((id) => ids.has(id))
})

function timestamp(value?: string | null) {
  if (!value) return 0
  const parsed = new Date(value).getTime()
  return Number.isNaN(parsed) ? 0 : parsed
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

function displayName(experiment: ExperimentRecord) {
  return experimentDisplayName(experiment, (key, params) => t(key, params ?? {}))
}

async function startRename(experiment: ExperimentRecord) {
  editingExperimentId.value = experiment.id
  editingName.value = displayName(experiment)
  renameError.value = ''
  await nextTick()
  renameInput.value?.focus()
  renameInput.value?.select()
}

function setRenameInput(element: unknown) {
  renameInput.value = element instanceof HTMLInputElement ? element : null
}

function cancelRename() {
  editingExperimentId.value = ''
  editingName.value = ''
  renameError.value = ''
}

function saveRename(experiment: ExperimentRecord) {
  const name = editingName.value.trim()
  if (!name) {
    renameError.value = t('experiments.nameRequired')
    return
  }
  if (name === displayName(experiment)) {
    cancelRename()
    return
  }
  renameError.value = ''
  renameMutation.mutate({ id: experiment.id, name })
}

function errorText(error: unknown) {
  return localizedApiError(error, t, te)
}

function canDelete(experiment: ExperimentRecord) {
  return ['succeeded', 'failed', 'canceled'].includes(experiment.status)
}

function canStop(experiment: ExperimentRecord) {
  return ['queued', 'running'].includes(experiment.status)
}

function datasetName(experiment: ExperimentRecord) {
  return datasetById.value.get(experiment.dataset_id)?.name ?? t('experiments.unknownDataset')
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

function startNewExperiment() {
  draft.resetForTask('pep_function')
  router.push('/experiments/new')
}

function isCompareSelected(experimentId: string) {
  return selectedCompareIds.value.includes(experimentId)
}

function toggleCompareExperiment(experiment: ExperimentRecord, selected: boolean) {
  if (!selected) {
    selectedCompareIds.value = selectedCompareIds.value.filter((id) => id !== experiment.id)
    return
  }
  if (isCompareSelected(experiment.id) || compareSelectionFull.value) return
  selectedCompareIds.value = [...selectedCompareIds.value, experiment.id]
}

function handleCompareCheckbox(experiment: ExperimentRecord, event: Event) {
  const target = event.target as HTMLInputElement | null
  toggleCompareExperiment(experiment, Boolean(target?.checked))
}

function openComparison() {
  if (!requireLogin()) return
  if (selectedCompareIds.value.length < 2) return
  router.push({ path: '/experiments/compare', query: { ids: selectedCompareIds.value.join(',') } })
}

function rerunExperiment(experiment: ExperimentRecord) {
  if (!datasetById.value.get(experiment.dataset_id)) return
  draft.cloneFromExperiment(experiment)
  router.push('/experiments/new')
}

function confirmDelete(experiment: ExperimentRecord) {
  if (!canDelete(experiment)) return
  if (!window.confirm(t('experiments.deleteConfirm', { name: displayName(experiment) }))) return
  deletingExperimentId.value = experiment.id
  deleteMutation.mutate(experiment.id)
}

function confirmStop(experiment: ExperimentRecord) {
  if (!canStop(experiment)) return
  if (!window.confirm(t('experiments.stopConfirm', { name: displayName(experiment) }))) return
  stoppingExperimentId.value = experiment.id
  cancelMutation.mutate(experiment.id)
}
</script>

<template>
  <section class="grid gap-6">
    <PageHeader :eyebrow="t('experiments.eyebrow')" :title="t('experiments.title')" compact>
      <template #actions>
        <button type="button" class="primary-action" @click="startNewExperiment">
          <FlaskConical aria-hidden="true" />
          {{ t('experiments.newExperiment') }}
        </button>
      </template>
    </PageHeader>

    <div class="panel">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <p class="eyebrow">{{ t('experiments.listTitle') }}</p>
        <div class="flex flex-wrap items-center justify-end gap-2">
          <span class="dataset-chip text-brass">
            {{ t('experiments.count', { count: sortedExperiments.length }) }}
          </span>
          <span v-if="activeCount" class="dataset-chip text-moss">
            {{ t('experiments.active') }} · {{ activeCount }}
          </span>
          <span class="dataset-chip text-brass">
            {{
              t('compare.selectedCount', {
                count: selectedCompareIds.length,
                max: COMPARISON_MAX_EXPERIMENTS,
              })
            }}
          </span>
          <button
            type="button"
            class="secondary-action compact-action"
            :disabled="!isGuest && selectedCompareIds.length < 2"
            @click="openComparison"
          >
            <Scale aria-hidden="true" />
            {{ t('compare.compareSelected') }}
          </button>
        </div>
      </div>

      <div v-if="sortedExperiments.length" class="mt-5 grid gap-3">
        <article
          v-for="experiment in sortedExperiments"
          :key="experiment.id"
          class="experiment-row"
        >
          <div class="min-w-0 flex-1">
            <div class="flex flex-wrap items-start gap-3">
              <div v-if="editingExperimentId === experiment.id" class="experiment-name-editor">
                <label :for="`experiment-name-${experiment.id}`" class="sr-only">
                  {{ t('experiments.nameLabel') }}
                </label>
                <input
                  :id="`experiment-name-${experiment.id}`"
                  :ref="setRenameInput"
                  v-model="editingName"
                  class="field-control"
                  type="text"
                  maxlength="120"
                  :disabled="renameMutation.isPending.value"
                  @keydown.enter.prevent="saveRename(experiment)"
                  @keydown.esc.prevent="cancelRename"
                />
                <button
                  type="button"
                  class="experiment-title-action experiment-title-action-save"
                  :disabled="renameMutation.isPending.value"
                  :title="t('common.save')"
                  :aria-label="t('common.save')"
                  @click="saveRename(experiment)"
                >
                  <Check aria-hidden="true" />
                </button>
                <button
                  type="button"
                  class="experiment-title-action"
                  :disabled="renameMutation.isPending.value"
                  :title="t('common.cancel')"
                  :aria-label="t('common.cancel')"
                  @click="cancelRename"
                >
                  <X aria-hidden="true" />
                </button>
                <p v-if="renameError" class="experiment-name-error" role="alert" aria-live="polite">
                  {{ renameError }}
                </p>
              </div>
              <div v-else class="experiment-title-cluster">
                <RouterLink
                  :to="`/experiments/${experiment.id}`"
                  class="min-w-0 break-words text-lg font-semibold hover:text-[var(--color-accent-dark)]"
                >
                  {{ displayName(experiment) }}
                </RouterLink>
                <button
                  type="button"
                  class="experiment-title-action"
                  :title="t('experiments.rename')"
                  :aria-label="t('experiments.rename')"
                  @click="startRename(experiment)"
                >
                  <Pencil aria-hidden="true" />
                </button>
              </div>
              <span class="status-pill" :class="statusClass(experiment.status)">
                {{ statusLabel(experiment.status) }}
              </span>
            </div>
            <div class="experiment-meta-grid mt-4">
              <div>
                <span>{{ t('experiments.task') }}</span>
                <strong>{{ taskLabel(experiment.task_type) }}</strong>
              </div>
              <div>
                <span>{{ t('experiments.dataset') }}</span>
                <strong>{{ datasetName(experiment) }}</strong>
              </div>
              <div>
                <span>{{ t('experiments.updatedAt') }}</span>
                <strong>{{ formatDate(experiment.updated_at ?? experiment.created_at) }}</strong>
              </div>
            </div>
          </div>
          <div class="flex shrink-0 flex-wrap justify-end gap-2">
            <label class="selection-control">
              <input
                type="checkbox"
                :checked="isCompareSelected(experiment.id)"
                :disabled="!isCompareSelected(experiment.id) && compareSelectionFull"
                @change="handleCompareCheckbox(experiment, $event)"
              />
              {{ t('compare.compare') }}
            </label>
            <button
              v-if="canStop(experiment)"
              type="button"
              class="danger-action compact-action"
              :disabled="cancelMutation.isPending.value && stoppingExperimentId === experiment.id"
              @click="confirmStop(experiment)"
            >
              <SquareStop aria-hidden="true" />
              {{
                cancelMutation.isPending.value && stoppingExperimentId === experiment.id
                  ? t('experiments.stopping')
                  : t('experiments.stop')
              }}
            </button>
            <button
              type="button"
              class="secondary-action compact-action experiment-rerun-action"
              :disabled="!datasetById.get(experiment.dataset_id)"
              @click="rerunExperiment(experiment)"
            >
              <RefreshCw aria-hidden="true" />
              {{ t('experiments.rerun') }}
            </button>
            <RouterLink
              :to="`/experiments/${experiment.id}`"
              class="emphasis-action compact-action"
            >
              <ArrowRight aria-hidden="true" />
              {{ t('experiments.openDetail') }}
            </RouterLink>
            <button
              type="button"
              class="danger-action compact-action"
              :disabled="
                !canDelete(experiment) ||
                (deleteMutation.isPending.value && deletingExperimentId === experiment.id)
              "
              :title="canDelete(experiment) ? '' : t('experiments.deleteBlocked')"
              @click="confirmDelete(experiment)"
            >
              <Trash2 aria-hidden="true" />
              {{
                deleteMutation.isPending.value && deletingExperimentId === experiment.id
                  ? t('common.deleting')
                  : t('common.delete')
              }}
            </button>
          </div>
        </article>
      </div>

      <div v-else class="section-card mt-5">
        <p class="font-semibold">{{ t('experiments.emptyTitle') }}</p>
        <p class="muted-copy mt-2 text-sm leading-6">{{
          t(isGuest ? 'guest.resources' : 'experiments.emptyHint')
        }}</p>
        <button type="button" class="primary-action mt-4" @click="startNewExperiment">
          <FlaskConical aria-hidden="true" />
          {{ t('experiments.newExperiment') }}
        </button>
      </div>

      <p v-if="deleteMutation.error.value" class="mt-4 text-sm text-rose-600">
        {{ t('experiments.deleteError', { message: errorText(deleteMutation.error.value) }) }}
      </p>
      <p v-if="cancelMutation.error.value" class="mt-4 text-sm text-rose-600">
        {{ t('experiments.stopError', { message: errorText(cancelMutation.error.value) }) }}
      </p>
    </div>
  </section>
</template>
