<script setup lang="ts">
import { normalizeTaskType } from '../taskTerminology'
import {
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  FileArchive,
  FlaskConical,
  FolderOpen,
  ScanSearch,
  Trash2,
  Upload,
  X,
} from '@lucide/vue'
import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { api } from '../api/client'
import { useWorkspaceAccess } from '../workspaceAccess'
import { useLoginDraft } from '../loginDraft'
import PageHeader from '../components/PageHeader.vue'
import { localizedApiError, localizedDatasetIssue } from '../errorLocalization'
import { useExperimentDraftStore } from '../stores/experimentDraft'
import type { DatasetRecord, DatasetPreviewRow, DatasetReport, DatasetSplitReport } from '../types'

const { isGuest, canUseWorkspace, requireLogin } = useWorkspaceAccess()

const queryClient = useQueryClient()
const router = useRouter()
const draft = useExperimentDraftStore()
const { t, te } = useI18n()
const datasets = useQuery({
  queryKey: ['datasets'],
  queryFn: api.datasets,
  enabled: canUseWorkspace,
})
const name = ref('')
const taskType = ref('pep_function')
useLoginDraft(
  'dataset-upload',
  () => ({ name: name.value, taskType: taskType.value }),
  (saved) => {
    name.value = saved.name
    taskType.value = saved.taskType
  },
)
const file = ref<File>()
const fileInput = ref<HTMLInputElement>()
const fileIssue = ref(false)
const isDragging = ref(false)
const uploadedDataset = ref<DatasetRecord>()
const deletingDatasetId = ref('')
const reportDatasetId = ref('')
const datasetPageSize = 4
const datasetPage = ref(1)
const datasetItems = computed(() => datasets.data.value ?? [])
const datasetTotal = computed(() => datasetItems.value.length)
const datasetTotalPages = computed(() =>
  Math.max(1, Math.ceil(datasetTotal.value / datasetPageSize)),
)
const paginatedDatasets = computed(() => {
  const start = (datasetPage.value - 1) * datasetPageSize
  return datasetItems.value.slice(start, start + datasetPageSize)
})
const paginationStart = computed(() =>
  datasetTotal.value === 0 ? 0 : (datasetPage.value - 1) * datasetPageSize + 1,
)
const paginationEnd = computed(() =>
  Math.min(datasetPage.value * datasetPageSize, datasetTotal.value),
)
const selectedFileSize = computed(() => (file.value ? formatFileSize(file.value.size) : ''))
const uploadGuide = computed(() => {
  if (taskType.value === 'pep_function') {
    return t('datasets.guides.pep_function')
  }
  if (taskType.value === 'pep_hlaii') {
    return t('datasets.guides.pep_hlaii')
  }
  return t('datasets.guides.pair')
})
const upload = useMutation({
  mutationFn: api.uploadDataset,
  onSuccess: (dataset) => {
    uploadedDataset.value = dataset
    file.value = undefined
    fileIssue.value = false
    if (fileInput.value) {
      fileInput.value.value = ''
    }
    datasetPage.value = 1
    queryClient.invalidateQueries({ queryKey: ['datasets'] })
  },
})
const deleteMutation = useMutation({
  mutationFn: (datasetId: string) => api.deleteDataset(datasetId),
  onSuccess: () => {
    deletingDatasetId.value = ''
    queryClient.invalidateQueries({ queryKey: ['datasets'] })
  },
  onError: () => {
    deletingDatasetId.value = ''
  },
})
const datasetReport = useQuery({
  queryKey: computed(() => ['dataset-report', reportDatasetId.value]),
  queryFn: () => api.datasetReport(reportDatasetId.value, 5),
  enabled: computed(() => canUseWorkspace.value && Boolean(reportDatasetId.value)),
})

watch(datasetTotalPages, (totalPages) => {
  if (datasetPage.value > totalPages) {
    datasetPage.value = totalPages
  }
})

function submit() {
  if (!requireLogin()) return
  if (!file.value || !name.value) return
  const form = new FormData()
  form.append('name', name.value)
  form.append('task_type', taskType.value)
  form.append('archive', file.value)
  upload.mutate(form)
}

function openFilePicker() {
  if (!requireLogin()) return
  fileInput.value?.click()
}

function handleFileChange(event: Event) {
  if (!requireLogin()) return
  const target = event.target as HTMLInputElement
  setSelectedFile(target.files?.[0])
}

function handleDrop(event: DragEvent) {
  if (!requireLogin()) return
  isDragging.value = false
  setSelectedFile(event.dataTransfer?.files?.[0])
}

function handleDragLeave(event: DragEvent) {
  const currentTarget = event.currentTarget as HTMLElement | null
  const relatedTarget = event.relatedTarget as Node | null
  if (!currentTarget || !relatedTarget || !currentTarget.contains(relatedTarget)) {
    isDragging.value = false
  }
}

function setSelectedFile(nextFile?: File) {
  fileIssue.value = false
  if (!nextFile) return
  if (!nextFile.name.toLowerCase().endsWith('.zip')) {
    file.value = undefined
    fileIssue.value = true
    return
  }
  file.value = nextFile
  uploadedDataset.value = undefined
  upload.reset()
}

function clearFile() {
  file.value = undefined
  fileIssue.value = false
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

function formatFileSize(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let value = bytes
  let unitIndex = 0
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024
    unitIndex += 1
  }
  const digits = value >= 10 || unitIndex === 0 ? 0 : 1
  return `${value.toFixed(digits)} ${units[unitIndex]}`
}

function splitEntries(dataset: DatasetRecord) {
  const splits = dataset.metadata.splits
  if (!splits || typeof splits !== 'object' || Array.isArray(splits)) return []
  return Object.entries(splits as Record<string, number>)
}

function startExperiment(dataset: DatasetRecord) {
  draft.resetForTask(dataset.task_type)
  draft.datasetId = dataset.id
  router.push('/experiments/new')
}

function canDeleteDataset(dataset: DatasetRecord) {
  return dataset.source === 'upload'
}

function confirmDeleteDataset(dataset: DatasetRecord) {
  if (!requireLogin()) return
  if (!canDeleteDataset(dataset)) return
  if (!window.confirm(t('datasets.deleteConfirm', { name: dataset.name }))) return
  deletingDatasetId.value = dataset.id
  deleteMutation.mutate(dataset.id)
}

function toggleReport(dataset: DatasetRecord) {
  reportDatasetId.value = reportDatasetId.value === dataset.id ? '' : dataset.id
}

function reportFor(dataset: DatasetRecord) {
  if (reportDatasetId.value !== dataset.id) return undefined
  return datasetReport.data.value
}

function reportStatusLabel(report?: DatasetReport) {
  return t(`datasets.reportStatus.${report?.status ?? 'warning'}`)
}

function reportStatusClass(status: string) {
  if (status === 'valid') return 'status-pill-success'
  if (status === 'error') return 'status-pill-danger'
  return 'status-pill-warning'
}

function labelModeLabel(mode: string) {
  return t(`datasets.labelModes.${mode}`)
}

function lengthSummary(stats?: { min?: number | null; max?: number | null; mean?: number | null }) {
  if (
    !stats ||
    stats.min === null ||
    stats.min === undefined ||
    stats.max === null ||
    stats.max === undefined
  ) {
    return t('datasets.notAvailable')
  }
  return t('datasets.lengthSummary', {
    min: stats.min,
    max: stats.max,
    mean: stats.mean ?? stats.max,
  })
}

function topCounts(values: Record<string, number>, limit = 6) {
  return Object.entries(values ?? {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
}

function splitIssues(report: DatasetReport | undefined, split: DatasetSplitReport) {
  return (report?.issues ?? []).filter((issue) => issue.split === split.name)
}

function globalIssues(report?: DatasetReport) {
  return (report?.issues ?? []).filter((issue) => !issue.split)
}

function previewLabel(row: DatasetPreviewRow) {
  if (row.labels?.length) return row.labels.join(', ')
  return row.label || t('datasets.notAvailable')
}

function taskLabel(taskType: string) {
  taskType = normalizeTaskType(taskType)
  return t(`tasks.${taskType}.label`)
}

function sourceLabel(source: string) {
  return t(`datasetSource.${source}`)
}

function errorText(error: unknown) {
  return localizedApiError(error, t, te)
}

function issueText(issue: DatasetReport['issues'][number]) {
  return localizedDatasetIssue(issue, t, te)
}
</script>

<template>
  <section class="grid gap-6">
    <PageHeader :title="t('datasets.listEyebrow')" compact />

    <div class="grid gap-6 lg:grid-cols-[0.82fr_1.18fr]">
      <div class="bio-panel panel">
        <p class="eyebrow">{{ t('datasets.uploadEyebrow') }}</p>
        <p class="muted-copy mt-2 text-sm leading-6">{{ t('datasets.uploadSubtitle') }}</p>

        <div class="mt-6 space-y-5">
          <label class="grid gap-2 text-sm font-medium text-slate-700">
            {{ t('datasets.nameLabel') }}
            <input
              v-model="name"
              class="field-control"
              :placeholder="t('datasets.namePlaceholder')"
            />
          </label>

          <label class="grid gap-2 text-sm font-medium text-slate-700">
            {{ t('datasets.taskLabel') }}
            <select v-model="taskType" class="field-control">
              <option value="pep_function">{{ taskLabel('pep_function') }}</option>
              <option value="pep_protein_interaction">{{
                taskLabel('pep_protein_interaction')
              }}</option>
              <option value="pep_hlaii">{{ taskLabel('pep_hlaii') }}</option>
              <option value="pep_hlai">{{ taskLabel('pep_hlai') }}</option>
            </select>
          </label>

          <div class="protocol-card">
            <div class="flex items-center gap-2 font-semibold">
              <span class="protocol-dot" />
              {{ t('datasets.protocolTitle') }}
            </div>
            <p class="mt-3 leading-6">{{ uploadGuide }}</p>
          </div>

          <div class="space-y-3">
            <input
              ref="fileInput"
              class="sr-only"
              type="file"
              :disabled="isGuest"
              accept=".zip"
              @change="handleFileChange"
            />
            <div
              class="dataset-dropzone"
              :class="{
                'dataset-dropzone-active': isDragging,
                'dataset-dropzone-ready': file,
              }"
              role="button"
              tabindex="0"
              :aria-label="file ? t('datasets.replaceFile') : t('datasets.chooseZip')"
              @click="openFilePicker"
              @keydown.enter.prevent="openFilePicker"
              @keydown.space.prevent="openFilePicker"
              @dragenter.prevent="isDragging = true"
              @dragover.prevent="isDragging = true"
              @dragleave.prevent="handleDragLeave"
              @drop.prevent="handleDrop"
            >
              <div class="dataset-dropzone-layout">
                <div class="file-orbit" :class="file ? 'file-orbit-ready' : ''">
                  <FileArchive aria-hidden="true" />
                  <span>ZIP</span>
                </div>
                <div class="dataset-dropzone-copy">
                  <div class="truncate text-base font-semibold text-slate-950">
                    {{ file?.name ?? t('datasets.dropzoneTitle') }}
                  </div>
                  <p class="mt-2 text-sm leading-6 text-slate-500">
                    {{
                      file
                        ? t('datasets.fileReady', { size: selectedFileSize })
                        : t('datasets.dropzoneSubtitle')
                    }}
                  </p>
                  <div class="dataset-dropzone-meta">
                    <span class="dataset-chip text-brass">.zip</span>
                    <span class="dataset-chip">{{ t('datasets.archiveFormat') }}</span>
                  </div>
                </div>
              </div>
            </div>

            <div class="dataset-picker-actions">
              <button type="button" class="secondary-action" @click="openFilePicker">
                <FolderOpen aria-hidden="true" />
                {{ file ? t('datasets.replaceFile') : t('datasets.chooseZip') }}
              </button>
              <button v-if="file" type="button" class="tertiary-action" @click="clearFile">
                <X aria-hidden="true" />
                {{ t('datasets.removeFile') }}
              </button>
            </div>
          </div>

          <button
            class="primary-action w-full"
            :disabled="!isGuest && (!name || !file || upload.isPending.value)"
            @click="submit"
          >
            <Upload aria-hidden="true" />
            {{ upload.isPending.value ? t('datasets.uploading') : t('datasets.upload') }}
          </button>

          <p v-if="fileIssue" class="error-card">
            {{ t('datasets.zipOnly') }}
          </p>
          <p v-if="upload.error.value" class="error-card">
            {{ errorText(upload.error.value) }}
          </p>

          <div v-if="uploadedDataset" class="success-card">
            <div class="flex items-start gap-3">
              <span class="mt-1 h-2.5 w-2.5 rounded-full bg-[color:var(--color-moss)]" />
              <div>
                <div class="text-sm font-semibold">
                  {{ t('datasets.ready', { name: uploadedDataset.name }) }}
                </div>
                <p class="mt-1 text-sm">{{ t('datasets.readyHint') }}</p>
              </div>
            </div>
            <button class="primary-action mt-4" @click="startExperiment(uploadedDataset)">
              <FlaskConical aria-hidden="true" />
              {{ t('datasets.configureExperiment') }}
            </button>
            <button
              class="secondary-action mt-3"
              :class="{ 'secondary-action-active': reportDatasetId === uploadedDataset.id }"
              :aria-expanded="reportDatasetId === uploadedDataset.id"
              @click="toggleReport(uploadedDataset)"
            >
              <ScanSearch aria-hidden="true" />
              {{
                reportDatasetId === uploadedDataset.id
                  ? t('datasets.hideReport')
                  : t('datasets.viewReport')
              }}
            </button>
          </div>
        </div>
      </div>
      <div class="bio-panel panel">
        <div class="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p class="eyebrow">{{ t('datasets.listEyebrow') }}</p>
            <h2 class="mt-2 text-xl font-semibold tracking-tight text-slate-950">{{
              t('datasets.libraryTitle')
            }}</h2>
          </div>
          <span class="dataset-chip text-brass">
            {{ t('datasets.datasetCount', { count: datasetTotal }) }}
          </span>
        </div>
        <div class="mt-5 grid gap-4">
          <article v-for="dataset in paginatedDatasets" :key="dataset.id" class="dataset-card">
            <div class="flex flex-wrap items-start justify-between gap-3">
              <div class="min-w-0">
                <div class="truncate font-semibold text-slate-950">{{ dataset.name }}</div>
                <div class="mt-2 flex flex-wrap gap-2">
                  <span class="dataset-chip text-moss">{{ taskLabel(dataset.task_type) }}</span>
                  <span class="dataset-chip">
                    {{ t('datasets.formatLabel') }} {{ dataset.format }}
                  </span>
                </div>
              </div>
              <span class="dataset-chip text-brass">
                {{ sourceLabel(dataset.source) }}
              </span>
            </div>
            <div v-if="splitEntries(dataset).length" class="mt-4 flex flex-wrap gap-2">
              <span
                v-for="[split, count] in splitEntries(dataset)"
                :key="split"
                class="dataset-chip"
              >
                {{ split }}: {{ count }}
              </span>
            </div>
            <div class="dataset-card-actions">
              <button class="secondary-action compact-action" @click="startExperiment(dataset)">
                <FlaskConical aria-hidden="true" />
                {{ t('datasets.useInNewExperiment') }}
              </button>
              <button
                class="tertiary-action compact-action"
                :class="{ 'secondary-action-active': reportDatasetId === dataset.id }"
                :aria-expanded="reportDatasetId === dataset.id"
                @click="toggleReport(dataset)"
              >
                <ScanSearch aria-hidden="true" />
                {{
                  reportDatasetId === dataset.id
                    ? t('datasets.hideReport')
                    : t('datasets.viewReport')
                }}
              </button>
              <button
                v-if="canDeleteDataset(dataset)"
                class="danger-action compact-action"
                :disabled="deleteMutation.isPending.value && deletingDatasetId === dataset.id"
                @click="confirmDeleteDataset(dataset)"
              >
                <Trash2 aria-hidden="true" />
                {{
                  deleteMutation.isPending.value && deletingDatasetId === dataset.id
                    ? t('common.deleting')
                    : t('common.delete')
                }}
              </button>
            </div>
            <div v-if="reportDatasetId === dataset.id" class="mt-5 border-t border-slate-100 pt-5">
              <div class="mb-4 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div class="text-sm font-black uppercase tracking-[0.16em] text-slate-500">
                    {{ t('datasets.reportTitle') }}
                  </div>
                  <div class="mt-1 text-sm text-slate-600">{{ dataset.name }}</div>
                </div>
              </div>
              <p v-if="datasetReport.isLoading.value" class="text-sm text-slate-500">
                {{ t('datasets.reportLoading') }}
              </p>
              <p v-else-if="datasetReport.error.value" class="text-sm text-rose-600">
                {{ t('datasets.reportError', { message: errorText(datasetReport.error.value) }) }}
              </p>
              <div v-else-if="reportFor(dataset)" class="grid gap-5">
                <div class="flex flex-wrap items-center gap-2">
                  <span class="dataset-chip">
                    {{
                      t('datasets.totalSamples', {
                        count: reportFor(dataset)?.summary.total_samples ?? 0,
                      })
                    }}
                  </span>
                  <span class="dataset-chip">
                    {{ labelModeLabel(reportFor(dataset)?.summary.label_mode ?? 'unknown') }}
                  </span>
                  <span
                    class="status-pill"
                    :class="reportStatusClass(reportFor(dataset)?.status ?? 'warning')"
                  >
                    {{ reportStatusLabel(reportFor(dataset)) }}
                  </span>
                </div>

                <div v-if="globalIssues(reportFor(dataset)).length" class="grid gap-2">
                  <div
                    v-for="issue in globalIssues(reportFor(dataset))"
                    :key="`${issue.code}-${JSON.stringify(issue.params ?? {})}`"
                    class="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800"
                  >
                    {{ issueText(issue) }}
                  </div>
                </div>

                <div
                  v-for="split in reportFor(dataset)?.splits ?? []"
                  :key="split.name"
                  class="rounded-lg border border-slate-100 bg-white/70 p-4"
                >
                  <div class="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <div class="text-sm font-black uppercase tracking-[0.16em] text-slate-500">{{
                        split.name
                      }}</div>
                      <div class="mt-1 text-sm text-slate-600">
                        {{ t('datasets.splitSamples', { count: split.sample_count }) }}
                      </div>
                    </div>
                    <div class="flex flex-wrap gap-2">
                      <span
                        v-for="[fileName, present] in Object.entries(split.files)"
                        :key="fileName"
                        class="dataset-chip"
                        :class="present ? 'dataset-chip-success' : 'dataset-chip-danger'"
                      >
                        {{ fileName }}
                      </span>
                    </div>
                  </div>

                  <div class="mt-4 grid gap-3 text-sm sm:grid-cols-3">
                    <div>
                      <div class="font-semibold text-slate-500">{{
                        t('datasets.sequenceLengths')
                      }}</div>
                      <div class="mt-1 text-slate-700">{{
                        lengthSummary(split.sequence_lengths)
                      }}</div>
                    </div>
                    <div>
                      <div class="font-semibold text-slate-500">{{
                        t('datasets.peptideLengths')
                      }}</div>
                      <div class="mt-1 text-slate-700">{{
                        lengthSummary(split.peptide_lengths)
                      }}</div>
                    </div>
                    <div>
                      <div class="font-semibold text-slate-500">{{
                        t('datasets.partnerLengths')
                      }}</div>
                      <div class="mt-1 text-slate-700">{{
                        lengthSummary(split.partner_lengths)
                      }}</div>
                    </div>
                  </div>

                  <div class="mt-4 flex flex-wrap gap-2">
                    <span
                      v-for="[label, count] in topCounts(split.label_counts.positive).length
                        ? topCounts(split.label_counts.positive)
                        : topCounts(split.label_counts.values)"
                      :key="label"
                      class="dataset-chip"
                    >
                      {{ label }}: {{ count }}
                    </span>
                    <span
                      v-if="
                        !topCounts(split.label_counts.positive).length &&
                        !topCounts(split.label_counts.values).length
                      "
                      class="text-sm text-slate-500"
                    >
                      {{ t('datasets.noLabelStats') }}
                    </span>
                  </div>

                  <div v-if="splitIssues(reportFor(dataset), split).length" class="mt-4 grid gap-2">
                    <div
                      v-for="issue in splitIssues(reportFor(dataset), split)"
                      :key="`${split.name}-${issue.code}-${issue.file ?? ''}`"
                      class="rounded-md border px-3 py-2 text-sm"
                      :class="
                        issue.severity === 'error'
                          ? 'border-rose-200 bg-rose-50 text-rose-700'
                          : 'border-amber-200 bg-amber-50 text-amber-800'
                      "
                    >
                      {{ issue.file ? `${issue.file}: ` : '' }}{{ issueText(issue) }}
                    </div>
                  </div>

                  <div v-if="split.preview.length" class="mt-4 overflow-x-auto">
                    <table class="min-w-full border-separate border-spacing-0 text-left text-xs">
                      <thead class="text-slate-500">
                        <tr>
                          <th class="border-b border-slate-100 py-2 pr-4">{{
                            t('datasets.sampleId')
                          }}</th>
                          <th class="border-b border-slate-100 py-2 pr-4">{{
                            t('datasets.sequenceOrPair')
                          }}</th>
                          <th class="border-b border-slate-100 py-2 pr-4">{{
                            t('datasets.label')
                          }}</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr
                          v-for="row in split.preview"
                          :key="`${split.name}-${row.sample_id}-${row.sequence ?? row.peptide}`"
                        >
                          <td
                            class="border-b border-slate-100 py-2 pr-4 align-top font-semibold text-slate-700"
                            >{{ row.sample_id }}</td
                          >
                          <td
                            class="max-w-md break-all whitespace-normal border-b border-slate-100 py-2 pr-4 align-top text-slate-600"
                          >
                            <span v-if="row.sequence">{{ row.sequence }}</span>
                            <span v-else>{{ row.peptide }} / {{ row.partner }}</span>
                          </td>
                          <td
                            class="border-b border-slate-100 py-2 pr-4 align-top text-slate-600"
                            >{{ previewLabel(row) }}</td
                          >
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
                <div class="flex justify-end border-t border-slate-100 pt-4">
                  <button class="tertiary-action compact-action" @click="toggleReport(dataset)">
                    <ChevronUp aria-hidden="true" />
                    {{ t('datasets.hideReport') }}
                  </button>
                </div>
              </div>
            </div>
          </article>
          <p v-if="deleteMutation.error.value" class="mt-1 text-sm text-rose-600">
            {{ t('datasets.deleteError', { message: errorText(deleteMutation.error.value) }) }}
          </p>
          <div
            v-if="!datasets.isLoading.value && datasetTotal === 0"
            class="rounded-lg border border-dashed border-stone-300 bg-white/70 p-6 text-sm text-stone-500"
          >
            {{ t(isGuest ? 'guest.resources' : 'datasets.empty') }}
          </div>
          <div
            v-if="datasetTotal > datasetPageSize"
            class="flex flex-col gap-3 border-t border-slate-100 pt-4 text-sm text-slate-500 sm:flex-row sm:items-center sm:justify-between"
          >
            <span>
              {{
                t('datasets.paginationRange', {
                  start: paginationStart,
                  end: paginationEnd,
                  total: datasetTotal,
                })
              }}
            </span>
            <div class="flex items-center gap-2">
              <button
                type="button"
                class="secondary-action icon-action compact-action"
                :disabled="datasetPage <= 1"
                :aria-label="t('datasets.previousPage')"
                :title="t('datasets.previousPage')"
                @click="datasetPage -= 1"
              >
                <ChevronLeft aria-hidden="true" />
              </button>
              <span class="min-w-16 text-center">
                {{ t('datasets.pageStatus', { page: datasetPage, total: datasetTotalPages }) }}
              </span>
              <button
                type="button"
                class="secondary-action icon-action compact-action"
                :disabled="datasetPage >= datasetTotalPages"
                :aria-label="t('datasets.nextPage')"
                :title="t('datasets.nextPage')"
                @click="datasetPage += 1"
              >
                <ChevronRight aria-hidden="true" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
