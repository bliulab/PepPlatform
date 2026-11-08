<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useQuery, useQueryClient } from '@tanstack/vue-query'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { api } from '../api/client'
import { useWorkspaceAccess } from '../workspaceAccess'
import { useLoginDraft } from '../loginDraft'
import { cloneEnsemblePayload, ensemblePayloadError, memberUnavailableReason } from '../ensemble'
import type {
  EnsembleCapabilities,
  EnsemblePayload,
  EnsemblePreflight,
  FusionMethod,
  WeightSource,
} from '../ensemble'
import PageHeader from '../components/PageHeader.vue'
import { TASK_ENTRIES } from '../taskCatalog'

const { isGuest, canUseWorkspace, requireLogin } = useWorkspaceAccess()

const { t } = useI18n()
const router = useRouter()
const route = useRoute()
const queryClient = useQueryClient()
const capabilities = useQuery({
  queryKey: ['ensemble-capabilities'],
  queryFn: api.ensembleCapabilities,
  enabled: canUseWorkspace,
})
const publicCapabilities = useQuery({
  queryKey: ['capabilities'],
  queryFn: api.capabilities,
  enabled: isGuest,
})
const formCapabilities = computed<EnsembleCapabilities | undefined>(() =>
  isGuest.value
    ? {
        enabled: true,
        tasks: TASK_ENTRIES.map((entry) => entry.key),
        methods: ['probability_mean', 'logit_mean', 'rank_mean'],
        weight_sources: ['equal', 'custom', 'validation'],
        objectives: publicCapabilities.data.value?.selection_objectives,
      }
    : capabilities.data.value,
)
const models = useQuery({ queryKey: ['models'], queryFn: api.models, enabled: canUseWorkspace })
const datasets = useQuery({
  queryKey: ['datasets'],
  queryFn: api.datasets,
  enabled: canUseWorkspace,
})
const task = ref('pep_function')
const datasetId = ref('')
const name = ref('')
const members = ref<string[]>([])
const weights = ref<Record<string, number>>({})
const method = ref<FusionMethod>('probability_mean')
const weightSource = ref<WeightSource>('equal')
const objective = ref('macro_f1')
const objectiveParams = ref<Record<string, number>>({})
const thresholdObjective = ref('macro_f1')
const thresholdParams = ref<Record<string, number>>({})
const objectiveOptions = computed(() =>
  (formCapabilities.value?.objectives ?? []).filter(
    (o) =>
      o.tasks.includes(task.value === 'pep_function' ? 'multilabel' : 'binary') &&
      (task.value === 'pep_function' || o.id !== 'macro_f1'),
  ),
)
const thresholdOptions = computed(() =>
  objectiveOptions.value.filter((o) => o.requires_predictions),
)
const gridStep = ref(0.05)
const evaluateTest = ref(false)
const seed = ref(42)
const report = ref<EnsemblePreflight>()
const error = ref('')
const busy = ref(false)
const submittedFingerprint = ref('')
const availableDatasets = computed(() =>
  (datasets.data.value ?? []).filter((x) => x.task_type === task.value),
)
const rows = computed(() =>
  (models.data.value ?? [])
    .filter((x) => x.task_type === task.value)
    .map((model) => ({ model, reason: memberUnavailableReason(model, task.value) })),
)
const payload = computed<EnsemblePayload>(() => ({
  name: name.value || t('ensemble.summary'),
  task_type: task.value,
  dataset_id: datasetId.value,
  member_ids: [...members.value],
  method: method.value,
  weight_source: weightSource.value,
  weights:
    weightSource.value === 'custom' ? members.value.map((id) => weights.value[id] ?? 1) : undefined,
  grid_step: gridStep.value,
  objective: objective.value,
  objective_params: objective.value === 'dual_f1_sacc' ? objectiveParams.value : {},
  threshold_objective: thresholdObjective.value,
  threshold_objective_params:
    thresholdObjective.value === 'dual_f1_sacc' ? thresholdParams.value : {},
  evaluate_test: evaluateTest.value,
  seed: seed.value,
}))
const fingerprint = computed(() => JSON.stringify(payload.value))
const valid = computed(
  () => report.value?.valid && submittedFingerprint.value === fingerprint.value,
)
useLoginDraft(
  'ensemble',
  () => ({
    task: task.value,
    name: name.value,
    method: method.value,
    weightSource: weightSource.value,
    objective: objective.value,
    objectiveParams: objectiveParams.value,
    thresholdObjective: thresholdObjective.value,
    thresholdParams: thresholdParams.value,
    gridStep: gridStep.value,
    evaluateTest: evaluateTest.value,
    seed: seed.value,
  }),
  (saved) => {
    task.value = saved.task
    name.value = saved.name
    method.value = saved.method
    weightSource.value = saved.weightSource
    objective.value = saved.objective
    objectiveParams.value = saved.objectiveParams
    thresholdObjective.value = saved.thresholdObjective
    thresholdParams.value = saved.thresholdParams
    gridStep.value = saved.gridStep
    evaluateTest.value = saved.evaluateTest
    seed.value = saved.seed
  },
)
watch(task, () => {
  members.value = []
  datasetId.value = ''
  objective.value = task.value === 'pep_function' ? 'macro_f1' : 'auc'
  thresholdObjective.value = task.value === 'pep_function' ? 'macro_f1' : 'f1'
  objectiveParams.value = {}
  thresholdParams.value = {}
})
watch(fingerprint, () => {
  report.value = undefined
  error.value = ''
})
onMounted(async () => {
  if (typeof route.query.clone !== 'string') return
  busy.value = true
  try {
    const cloned = cloneEnsemblePayload(await api.experiment(route.query.clone))
    if (!cloned) throw new Error(t('ensemble.unavailable'))
    task.value = cloned.task_type
    // Let the task watcher reset task-specific state before restoring the clone.
    await nextTick()
    name.value = cloned.name
    datasetId.value = cloned.dataset_id
    members.value = cloned.member_ids
    weights.value = Object.fromEntries(
      cloned.member_ids.map((id, i) => [id, cloned.weights?.[i] ?? 1]),
    )
    method.value = cloned.method
    weightSource.value = cloned.weight_source
    objective.value = cloned.objective
    objectiveParams.value = { ...cloned.objective_params }
    thresholdObjective.value =
      cloned.threshold_objective ?? (cloned.task_type === 'pep_function' ? 'macro_f1' : 'f1')
    thresholdParams.value = { ...cloned.threshold_objective_params }
    gridStep.value = cloned.grid_step
    evaluateTest.value = cloned.evaluate_test
    seed.value = cloned.seed
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
})
async function check() {
  if (!requireLogin()) return
  const issue = ensemblePayloadError(payload.value)
  if (issue) {
    error.value = t(`ensemble.${issue}`)
    return
  }
  busy.value = true
  error.value = ''
  const submitted = fingerprint.value
  try {
    const result = await api.preflightEnsemble(payload.value)
    if (fingerprint.value === submitted) {
      report.value = result
      submittedFingerprint.value = submitted
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
async function create() {
  if (!requireLogin()) return
  if (!valid.value) return
  busy.value = true
  error.value = ''
  try {
    const result = await api.createEnsemble(payload.value)
    queryClient.invalidateQueries({ queryKey: ['experiments'] })
    await router.push(`/experiments/${result.id}`)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <section class="grid gap-6">
    <PageHeader :eyebrow="t('ensemble.eyebrow')" :title="t('ensemble.title')" compact />
    <p class="max-w-3xl text-sm leading-7 text-slate-600">{{ t('ensemble.intro') }}</p>
    <p
      v-if="capabilities.isError.value || models.isError.value || datasets.isError.value"
      role="alert"
      class="text-rose-700"
      >{{
        capabilities.error.value?.message ||
        models.error.value?.message ||
        datasets.error.value?.message
      }}</p
    >
    <p v-if="formCapabilities && !formCapabilities.enabled" class="panel">{{
      t('ensemble.disabled')
    }}</p>
    <form
      v-else
      class="grid gap-6 lg:grid-cols-[1.5fr_1fr]"
      :novalidate="isGuest"
      @submit.prevent="create"
    >
      <fieldset
        :disabled="busy || !formCapabilities?.enabled"
        class="grid min-w-0 content-start gap-6"
      >
        <div class="panel grid gap-5 sm:grid-cols-2">
          <label class="grid gap-2 text-sm sm:col-span-2"
            >{{ t('ensemble.name') }}<input v-model="name" class="field-control" maxlength="120"
          /></label>
          <label class="grid gap-2 text-sm"
            >{{ t('ensemble.task')
            }}<select v-model="task" class="field-control"
              ><option v-for="value in formCapabilities?.tasks" :key="value" :value="value">{{
                t(`ensemble.tasks.${value}`)
              }}</option></select
            ></label
          >
          <label class="grid gap-2 text-sm"
            >{{ t('ensemble.dataset')
            }}<select v-model="datasetId" class="field-control"
              ><option value="">{{ t('ensemble.selectDataset') }}</option
              ><option v-for="dataset in availableDatasets" :key="dataset.id" :value="dataset.id">{{
                dataset.name
              }}</option></select
            ></label
          >
        </div>
        <div class="panel">
          <div class="flex flex-wrap items-center justify-between gap-3"
            ><h2 class="font-display text-xl">{{ t('ensemble.members') }}</h2
            ><span class="config-chip">{{
              t('ensemble.selected', { count: members.length })
            }}</span></div
          >
          <p class="my-4 text-sm leading-6 text-slate-500">{{ t('ensemble.membersHint') }}</p>
          <div class="grid gap-3">
            <div
              v-for="row in rows"
              :key="row.model.id"
              class="ensemble-member"
              :class="{ 'ensemble-member-selected': members.includes(row.model.id) }"
            >
              <label class="flex min-w-0 flex-1 items-start gap-3">
                <input
                  v-model="members"
                  type="checkbox"
                  :value="row.model.id"
                  :disabled="Boolean(row.reason)"
                  class="mt-1"
                />
                <span class="min-w-0"
                  ><strong class="block break-words text-sm">{{ row.model.name }}</strong
                  ><small v-if="row.reason" class="mt-1 block text-slate-500">{{
                    t(`ensemble.${row.reason}`)
                  }}</small
                  ><small v-else-if="members.includes(row.model.id)" class="mt-1 block text-brass"
                    >#{{ members.indexOf(row.model.id) + 1 }}</small
                  ></span
                >
              </label>
              <label
                v-if="weightSource === 'custom' && members.includes(row.model.id)"
                class="grid w-24 gap-1 text-xs"
                >{{ t('ensemble.weight')
                }}<input
                  :value="weights[row.model.id] ?? 1"
                  type="number"
                  min="0"
                  step="any"
                  class="field-control"
                  @input="
                    weights[row.model.id] = Number(($event.target as HTMLInputElement).value)
                  "
              /></label>
            </div>
            <p v-if="!rows.length" class="py-8 text-sm text-slate-500">{{ t('ensemble.empty') }}</p>
          </div>
        </div>
      </fieldset>
      <div class="grid content-start gap-6">
        <fieldset :disabled="busy || !formCapabilities?.enabled" class="panel grid gap-5">
          <label class="grid gap-2 text-sm"
            >{{ t('ensemble.method')
            }}<select v-model="method" class="field-control"
              ><option v-for="value in formCapabilities?.methods" :key="value" :value="value">{{
                t(`ensemble.${value}`)
              }}</option></select
            ></label
          >
          <p v-if="method === 'rank_mean'" class="ensemble-note" role="note">{{
            t('ensemble.rankHint')
          }}</p>
          <label class="grid gap-2 text-sm"
            >{{ t('ensemble.weightSource')
            }}<select v-model="weightSource" class="field-control"
              ><option
                v-for="value in formCapabilities?.weight_sources"
                :key="value"
                :value="value"
                >{{ t(`ensemble.${value}`) }}</option
              ></select
            ></label
          >
          <template v-if="weightSource === 'validation'">
            <label class="grid gap-2 text-sm"
              >{{ t('ensemble.objective')
              }}<select v-model="objective" class="field-control"
                ><option v-for="option in objectiveOptions" :key="option.id" :value="option.id">{{
                  t(`ensemble.${option.id}`)
                }}</option></select
              ></label
            >
            <label
              v-for="key in objective === 'dual_f1_sacc' ? ['reference_f1', 'reference_sacc'] : []"
              :key="key"
              class="grid gap-2 text-sm"
              >{{ key === 'reference_f1' ? 'Macro-F1' : 'SAcc' }} {{ t('ensemble.reference')
              }}<input
                v-model.number="objectiveParams[key]"
                type="number"
                min="0.000000001"
                step="any"
                required
                class="field-control"
            /></label>
            <label class="grid gap-2 text-sm"
              >{{ t('ensemble.thresholdObjective')
              }}<select v-model="thresholdObjective" class="field-control"
                ><option v-for="option in thresholdOptions" :key="option.id" :value="option.id">{{
                  t(`ensemble.${option.id}`)
                }}</option></select
              ></label
            >
            <label
              v-for="key in thresholdObjective === 'dual_f1_sacc'
                ? ['reference_f1', 'reference_sacc']
                : []"
              :key="key"
              class="grid gap-2 text-sm"
              >{{ key === 'reference_f1' ? 'Macro-F1' : 'SAcc' }} {{ t('ensemble.reference')
              }}<input
                v-model.number="thresholdParams[key]"
                type="number"
                min="0.000000001"
                step="any"
                required
                class="field-control"
            /></label>
            <label class="grid gap-2 text-sm"
              >{{ t('ensemble.grid')
              }}<select v-model.number="gridStep" class="field-control"
                ><option
                  v-for="value in [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5]"
                  :key="value"
                  :value="value"
                  >{{ value }}</option
                ></select
              ></label
            >
          </template>
          <label class="flex items-start gap-3 text-sm"
            ><input v-model="evaluateTest" type="checkbox" class="mt-1" /><span>{{
              t('ensemble.evaluateTest')
            }}</span></label
          >
          <p class="text-xs leading-6 text-slate-500">{{ t('ensemble.testHint') }}</p>
        </fieldset>
        <div class="panel grid gap-4" aria-live="polite">
          <p class="text-xs leading-6 text-slate-500">{{ t('ensemble.evidenceHint') }}</p>
          <p v-if="error" role="alert" class="text-sm text-rose-700">{{ error }}</p>
          <ul v-if="report && !report.valid" class="grid gap-2 text-sm text-rose-700"
            ><li v-for="issue in report.errors" :key="issue">{{ issue }}</li></ul
          >
          <p v-if="valid" class="text-sm text-emerald-700">{{
            t('ensemble.passed', { count: report?.validation_count })
          }}</p>
          <button
            type="button"
            class="secondary-action"
            :disabled="busy || !formCapabilities?.enabled"
            @click="check"
            >{{ busy ? t('ensemble.checking') : t('ensemble.preflight') }}</button
          >
          <button
            type="submit"
            class="primary-action"
            :disabled="busy || (!isGuest && (!valid || !formCapabilities?.enabled))"
            >{{ busy ? t('ensemble.creating') : t('ensemble.create') }}</button
          >
        </div>
      </div>
    </form>
  </section>
</template>

<style scoped>
.ensemble-member {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  border: 1px solid var(--color-line);
  border-radius: 1rem;
  transition:
    border-color 150ms,
    background 150ms;
}
.ensemble-member-selected {
  border-color: color-mix(in srgb, var(--color-accent) 55%, var(--color-line));
  background: color-mix(in srgb, var(--color-accent) 6%, var(--surface-strong));
}
.ensemble-note {
  border-left: 3px solid var(--color-accent);
  padding: 0.75rem 1rem;
  background: color-mix(in srgb, var(--color-accent) 5%, var(--surface-strong));
  color: var(--color-ink);
  font-size: 0.8rem;
  line-height: 1.7;
}
@media (prefers-reduced-motion: reduce) {
  .ensemble-member {
    transition: none;
  }
}
</style>
