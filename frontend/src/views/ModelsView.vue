<script setup lang="ts">
import EnsembleSummary from '../components/EnsembleSummary.vue'
import EnsembleEntry from '../components/EnsembleEntry.vue'
import { Pencil, Save, Trash2, X } from '@lucide/vue'
import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'
import { api } from '../api/client'
import { useWorkspaceAccess } from '../workspaceAccess'
import PageHeader from '../components/PageHeader.vue'
import InferencePanel from '../components/InferencePanel.vue'
import { localizedApiError } from '../errorLocalization'
import { formatLocalizedDate, modelDisplayName } from '../localization'
import { modelConfigForDisplay, modelMetricSummary, modelRouteSummary } from '../modelDisplay'
import type { ModelRecord } from '../types'

const { isGuest, canUseWorkspace, requireLogin } = useWorkspaceAccess()

const { locale, t, te } = useI18n()
const queryClient = useQueryClient()
const route = useRoute()
const models = useQuery({ queryKey: ['models'], queryFn: api.models, enabled: canUseWorkspace })
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
const selectedModel = ref(String(route.query.model ?? ''))
const deletingModelId = ref('')
const editingModelId = ref('')
const editingName = ref('')
const renameError = ref('')
const currentModel = computed(() =>
  models.data.value?.find((model) => model.id === selectedModel.value),
)
const experimentLookup = computed(
  () => new Map((experiments.data.value ?? []).map((experiment) => [experiment.id, experiment])),
)
const datasetLookup = computed(
  () => new Map((datasets.data.value ?? []).map((dataset) => [dataset.id, dataset])),
)
const modelRows = computed(() =>
  (models.data.value ?? []).map((model) => {
    const experiment = experimentLookup.value.get(model.experiment_id)
    const dataset = experiment ? datasetLookup.value.get(experiment.dataset_id) : undefined
    const config = modelConfigForDisplay(model, experiment)
    const completedEpochs =
      Number(experiment?.metrics.events.filter((event) => event.kind === 'epoch').at(-1)?.epoch) ||
      undefined
    const routeSummary = modelRouteSummary(config, completedEpochs)
    return {
      model,
      experiment,
      dataset,
      routeSummary: routeSummary || t('models.configurationUnavailable'),
      metricSummary: modelMetricSummary(experiment),
      createdAt: formatDate(model.created_at),
      displayName: modelDisplayName(model, experiment, (key, params) => t(key, params ?? {})),
    }
  }),
)

watch(
  () => route.query.model,
  (modelId) => {
    if (typeof modelId === 'string') selectedModel.value = modelId
  },
)

const renameMutation = useMutation({
  mutationFn: ({ id, name }: { id: string; name: string }) => api.updateModel(id, { name }),
  onSuccess: () => {
    editingModelId.value = ''
    editingName.value = ''
    renameError.value = ''
    queryClient.invalidateQueries({ queryKey: ['models'] })
  },
})

const deleteMutation = useMutation({
  mutationFn: (modelId: string) => api.deleteModel(modelId),
  onSuccess: (_result, modelId) => {
    if (selectedModel.value === modelId) {
      selectedModel.value = ''
    }
    queryClient.invalidateQueries({ queryKey: ['models'] })
  },
  onSettled: () => {
    deletingModelId.value = ''
  },
})

function confirmDelete(model: ModelRecord) {
  const experiment = experimentLookup.value.get(model.experiment_id)
  const name = modelDisplayName(model, experiment, (key, params) => t(key, params ?? {}))
  if (!window.confirm(t('models.deleteConfirm', { name }))) return
  deletingModelId.value = model.id
  deleteMutation.mutate(model.id)
}

function startRename(model: ModelRecord) {
  editingModelId.value = model.id
  editingName.value = modelDisplayName(
    model,
    experimentLookup.value.get(model.experiment_id),
    (key, params) => t(key, params ?? {}),
  )
  renameError.value = ''
}

function cancelRename() {
  editingModelId.value = ''
  editingName.value = ''
  renameError.value = ''
}

function saveRename(model: ModelRecord) {
  const name = editingName.value.trim()
  if (!name) {
    renameError.value = t('models.nameRequired')
    return
  }
  renameError.value = ''
  renameMutation.mutate({ id: model.id, name })
}

function formatDate(value?: string) {
  return formatLocalizedDate(value, locale.value, { month: 'short', day: 'numeric' })
}

function errorText(error: unknown) {
  return localizedApiError(error, t, te)
}

function metricText(summary: ReturnType<typeof modelMetricSummary>) {
  if (!summary.split) return ''
  const parts = []
  if (summary.f1) parts.push(`F1 ${summary.f1}`)
  if (summary.auc) parts.push(`AUC ${summary.auc}`)
  const split = summary.split === 'test' ? t('detail.test') : t('detail.validation')
  return `${split} ${parts.join(' · ')}`
}
</script>

<template>
  <section class="grid gap-6">
    <PageHeader :title="t('models.title')" compact>
      <template #actions>
        <EnsembleEntry />
        <span class="dataset-chip text-brass">{{ models.data.value?.length ?? 0 }}</span>
      </template>
    </PageHeader>

    <EnsembleSummary :metadata="currentModel?.metadata" class="mb-6" />
    <div class="grid gap-6 lg:grid-cols-[0.8fr_1.2fr]">
      <div class="panel">
        <div v-if="modelRows.length" class="space-y-3">
          <div
            v-for="row in modelRows"
            :key="row.model.id"
            class="model-row"
            :class="selectedModel === row.model.id ? 'model-row-selected' : ''"
          >
            <div
              v-if="editingModelId === row.model.id"
              class="flex min-w-0 flex-1 items-start gap-3"
            >
              <input v-model="selectedModel" type="radio" :value="row.model.id" class="mt-3" />
              <div class="min-w-0 flex-1">
                <label class="grid gap-2 text-sm">
                  {{ t('models.nameLabel') }}
                  <input
                    v-model="editingName"
                    type="text"
                    maxlength="120"
                    class="field-control"
                    :placeholder="row.displayName"
                  />
                </label>
                <p v-if="renameError" class="mt-2 text-xs text-rose-600">{{ renameError }}</p>
                <div class="mt-3 flex flex-wrap gap-2">
                  <button
                    type="button"
                    class="primary-action compact-action"
                    :disabled="renameMutation.isPending.value"
                    @click.stop="saveRename(row.model)"
                  >
                    <Save aria-hidden="true" />
                    {{ renameMutation.isPending.value ? t('common.saving') : t('common.save') }}
                  </button>
                  <button
                    type="button"
                    class="tertiary-action compact-action"
                    :disabled="renameMutation.isPending.value"
                    @click.stop="cancelRename"
                  >
                    <X aria-hidden="true" />
                    {{ t('common.cancel') }}
                  </button>
                </div>
              </div>
            </div>
            <label v-else class="flex min-w-0 flex-1 cursor-pointer items-start gap-3">
              <input v-model="selectedModel" type="radio" :value="row.model.id" class="mt-1" />
              <span class="min-w-0">
                <span class="block truncate font-medium">{{ row.displayName }}</span>
                <span class="mt-1 block text-sm text-slate-500">{{ row.routeSummary }}</span>
                <span class="mt-3 flex flex-wrap gap-2">
                  <span class="config-chip">
                    {{
                      row.model.supports_inference
                        ? t('common.readyForInference')
                        : t('common.metricsOnly')
                    }}
                  </span>
                  <span class="config-chip">
                    {{
                      Number(row.model.metadata.evaluation_protocol_version ?? 1) >= 2
                        ? t('models.currentEvaluationMethod')
                        : t('models.legacyProtocol')
                    }}
                  </span>
                  <span v-if="row.dataset" class="config-chip">{{ row.dataset.name }}</span>
                  <span v-if="metricText(row.metricSummary)" class="config-chip">
                    {{ metricText(row.metricSummary) }}
                  </span>
                  <span v-if="row.createdAt" class="config-chip">{{ row.createdAt }}</span>
                </span>
              </span>
            </label>
            <div class="flex shrink-0 flex-col gap-2 sm:flex-row">
              <button
                v-if="editingModelId !== row.model.id"
                type="button"
                class="secondary-action compact-action"
                :disabled="renameMutation.isPending.value"
                @click.stop="startRename(row.model)"
              >
                <Pencil aria-hidden="true" />
                {{ t('common.edit') }}
              </button>
              <button
                type="button"
                class="danger-action compact-action"
                :disabled="deleteMutation.isPending.value && deletingModelId === row.model.id"
                @click.stop="confirmDelete(row.model)"
              >
                <Trash2 aria-hidden="true" />
                {{
                  deleteMutation.isPending.value && deletingModelId === row.model.id
                    ? t('common.deleting')
                    : t('common.delete')
                }}
              </button>
            </div>
          </div>
        </div>
        <p v-else class="mt-4 text-sm text-slate-500">{{
          t(isGuest ? 'guest.resources' : 'common.empty')
        }}</p>
        <p v-if="deleteMutation.error.value" class="mt-4 text-sm text-rose-600">
          {{ errorText(deleteMutation.error.value) }}
        </p>
        <p v-if="renameMutation.error.value" class="mt-4 text-sm text-rose-600">
          {{ errorText(renameMutation.error.value) }}
        </p>
      </div>
      <InferencePanel
        v-if="currentModel"
        :model-id="currentModel.id"
        :disabled="!currentModel.supports_inference"
        :reason="String(currentModel.metadata.reason ?? '')"
        :reason-code="String(currentModel.metadata.reason_code ?? '')"
        :metadata="currentModel.metadata"
      />
      <div v-else class="panel">
        <p class="eyebrow">{{ t('models.inference') }}</p>
        <button v-if="isGuest" type="button" class="primary-action mt-4" @click="requireLogin">{{
          t('guest.signInToUse')
        }}</button>
        <p class="mt-4 text-sm text-slate-500">{{
          t(isGuest ? 'guest.resources' : 'models.selectModel')
        }}</p>
      </div>
    </div>
  </section>
</template>
