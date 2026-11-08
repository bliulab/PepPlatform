<script setup lang="ts">
import { normalizeTaskType } from '../taskTerminology'
import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { api } from '../api/client'
import { useWorkspaceAccess } from '../workspaceAccess'
import { formatLocalizedDate } from '../localization'
import { useAuthStore } from '../stores/auth'
import { ArrowRight, ChevronLeft, ChevronRight, Copy, Layers3, Plus, Trash2 } from '@lucide/vue'
import type { AgentExecutionPlanSummary } from '../types'

const { isGuest, canUseWorkspace } = useWorkspaceAccess()

const { locale, t } = useI18n()
const router = useRouter()
const queryClient = useQueryClient()
const auth = useAuthStore()

const filterTaskType = ref('')
const filterDatasetId = ref('')
const filterStatus = ref('')
const limit = ref(20)
const offset = ref(0)
const deletingPlanId = ref('')

const plansQuery = useQuery({
  enabled: canUseWorkspace,
  queryKey: computed(() => [
    'agent-execution-plans',
    {
      task_type: filterTaskType.value,
      dataset_id: filterDatasetId.value,
      status: filterStatus.value,
      limit: limit.value,
      offset: offset.value,
    },
  ]),
  queryFn: () =>
    api.agentExecutionPlans({
      task_type: filterTaskType.value || undefined,
      dataset_id: filterDatasetId.value || undefined,
      status: filterStatus.value || undefined,
      limit: limit.value,
      offset: offset.value,
    }),
})
const campaignsQuery = useQuery({
  enabled: canUseWorkspace,
  queryKey: computed(() => [
    'agent-campaigns',
    {
      task_type: filterTaskType.value,
      dataset_id: filterDatasetId.value,
      status:
        filterStatus.value === 'draft' || filterStatus.value === 'reported'
          ? ''
          : filterStatus.value,
      limit: 20,
      offset: 0,
    },
  ]),
  queryFn: () =>
    api.agentCampaigns({
      task_type: filterTaskType.value || undefined,
      dataset_id: filterDatasetId.value || undefined,
      status:
        filterStatus.value === 'active' || filterStatus.value === 'completed'
          ? filterStatus.value
          : undefined,
      limit: 20,
      offset: 0,
    }),
})
const datasetsQuery = useQuery({
  queryKey: ['datasets'],
  queryFn: api.datasets,
  enabled: canUseWorkspace,
})
const capabilitiesQuery = useQuery({ queryKey: ['capabilities'], queryFn: api.capabilities })

const plans = computed(() => plansQuery.data.value?.plans ?? [])
const campaigns = computed(() => campaignsQuery.data.value?.campaigns ?? [])
const total = computed(() => plansQuery.data.value?.total ?? 0)
const currentLimit = computed(() => plansQuery.data.value?.limit ?? limit.value)
const currentOffset = computed(() => plansQuery.data.value?.offset ?? offset.value)

const datasetById = computed(
  () => new Map((datasetsQuery.data.value ?? []).map((dataset) => [dataset.id, dataset])),
)
const taskOptions = computed(() => capabilitiesQuery.data.value?.tasks ?? [])
const statusOptions = computed(() => [
  { id: '', label: t('agent.allStatuses') },
  { id: 'draft', label: t('agent.planStatus.draft') },
  { id: 'running', label: t('status.running') },
  { id: 'ready_for_report', label: t('agent.planStatus.ready_for_report') },
  { id: 'reported', label: t('agent.planStatus.reported') },
  { id: 'active', label: t('agent.campaignStatus.active') },
  { id: 'completed', label: t('agent.campaignStatus.completed') },
  { id: 'failed', label: t('status.failed') },
])

const totalPages = computed(() => Math.max(1, Math.ceil(total.value / currentLimit.value)))
const currentPage = computed(() => Math.floor(currentOffset.value / currentLimit.value) + 1)

const deleteMutation = useMutation({
  mutationFn: (planId: string) => api.deleteAgentExecutionPlan(planId),
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ['agent-execution-plans'] })
  },
  onSettled: () => {
    deletingPlanId.value = ''
  },
})

watch([filterTaskType, filterDatasetId, filterStatus], () => {
  offset.value = 0
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

function datasetName(datasetId: string) {
  return datasetById.value.get(datasetId)?.name ?? t('experiments.unknownDataset')
}

function taskLabel(taskType: string) {
  taskType = normalizeTaskType(taskType)
  return t(`tasks.${taskType}.label`)
}

function statusLabel(status: string) {
  if (['queued', 'running', 'succeeded', 'failed', 'canceled'].includes(status)) {
    return t(`status.${status}`)
  }
  if (status === 'draft') return t('agent.planStatus.draft')
  if (status === 'ready_for_report') return t('agent.planStatus.ready_for_report')
  if (status === 'reported') return t('agent.planStatus.reported')
  if (status === 'active') return t('agent.campaignStatus.active')
  if (status === 'completed') return t('agent.campaignStatus.completed')
  return status
}

function statusClass(status: string) {
  if (status === 'reported' || status === 'succeeded' || status === 'completed')
    return 'status-pill-success'
  if (status === 'failed') return 'status-pill-danger'
  if (status === 'running' || status === 'ready_for_report') return 'status-pill-warning'
  return 'status-pill-neutral'
}

function canDelete(plan: AgentExecutionPlanSummary) {
  return plan.status !== 'running'
}

function viewPlan(plan: AgentExecutionPlanSummary) {
  router.push(`/agent/design?plan_id=${plan.id}`)
}

function reusePlan(plan: AgentExecutionPlanSummary) {
  router.push(`/agent/design?reuse_plan_id=${plan.id}`)
}

function viewCampaign(campaignId: string) {
  router.push(`/agent/campaigns/${campaignId}`)
}

function confirmDelete(plan: AgentExecutionPlanSummary) {
  if (!canDelete(plan)) return
  if (!window.confirm(t('agent.deletePlanConfirm', { objective: plan.objective }))) return
  deletingPlanId.value = plan.id
  deleteMutation.mutate(plan.id)
}

function goToPage(page: number) {
  if (page < 1 || page > totalPages.value) return
  offset.value = (page - 1) * currentLimit.value
}

function goToNextPage() {
  goToPage(currentPage.value + 1)
}

function goToPreviousPage() {
  goToPage(currentPage.value - 1)
}

void auth.loadCurrentUser()
</script>

<template>
  <section class="agent-library-page">
    <header class="agent-library-toolbar">
      <div>
        <p class="agent-rail-kicker">{{ t('agent.eyebrow') }}</p>
        <h1>{{ t('agent.plansTitle') }}</h1>
        <p>{{ t('agent.plansSubtitle') }}</p>
      </div>
      <RouterLink to="/agent" class="primary-action compact-action inline-flex items-center gap-2">
        <Plus aria-hidden="true" />{{ t('agent.newPlan') }}
      </RouterLink>
    </header>

    <div class="agent-library-filters">
      <div class="grid gap-4 md:grid-cols-4">
        <label class="form-field">
          <span>{{ t('experiments.task') }}</span>
          <select v-model="filterTaskType" class="field-control">
            <option value="">{{ t('agent.allTasks') }}</option>
            <option v-for="task in taskOptions" :key="task.id" :value="task.id">
              {{ taskLabel(task.id) }}
            </option>
          </select>
        </label>
        <label class="form-field">
          <span>{{ t('experiments.dataset') }}</span>
          <select v-model="filterDatasetId" class="field-control">
            <option value="">{{ t('agent.allDatasets') }}</option>
            <option
              v-for="dataset in datasetsQuery.data.value ?? []"
              :key="dataset.id"
              :value="dataset.id"
            >
              {{ dataset.name }}
            </option>
          </select>
        </label>
        <label class="form-field">
          <span>{{ t('agent.status') }}</span>
          <select v-model="filterStatus" class="field-control">
            <option v-for="status in statusOptions" :key="status.id" :value="status.id">
              {{ status.label }}
            </option>
          </select>
        </label>
        <label class="form-field">
          <span>{{ t('agent.pageSize') }}</span>
          <select v-model="limit" class="field-control">
            <option :value="10">10</option>
            <option :value="20">20</option>
            <option :value="50">50</option>
          </select>
        </label>
      </div>

      <p class="agent-library-result-count">
        {{ t('agent.totalPlans', { total }) }}
      </p>
    </div>

    <div v-if="plansQuery.isLoading.value" class="agent-library-empty">
      {{ t('agent.loadingPlans') }}
    </div>

    <div v-if="campaigns.length" class="agent-library-section">
      <div class="agent-library-section-head">
        <div
          ><Layers3 aria-hidden="true" /><h2>{{ t('agent.campaignsTitle') }}</h2></div
        >
        <span class="muted-copy text-sm">{{
          t('agent.totalCampaigns', { total: campaigns.length })
        }}</span>
      </div>
      <article
        v-for="campaign in campaigns"
        :key="campaign.id"
        class="agent-library-row agent-library-row-campaign"
      >
        <div class="agent-library-row-main">
          <div class="agent-library-row-copy">
            <div class="flex flex-wrap items-center gap-2">
              <span class="status-pill" :class="statusClass(campaign.status)">{{
                statusLabel(campaign.status)
              }}</span>
              <span class="config-chip">{{ taskLabel(campaign.task_type) }}</span>
              <span class="config-chip">
                {{
                  t('agent.campaignRoundProgress', {
                    current: campaign.current_round_index,
                    max: campaign.max_rounds,
                  })
                }}
              </span>
            </div>
            <h3>{{ campaign.objective }}</h3>
            <p class="muted-copy mt-1">
              {{ datasetName(campaign.dataset_id) }} ·
              {{ t('agent.createdAt', { date: formatDate(campaign.created_at) }) }}
            </p>
          </div>
          <button type="button" class="agent-library-open" @click="viewCampaign(campaign.id)">
            <span>{{ t('agent.viewCampaign') }}</span
            ><ArrowRight aria-hidden="true" />
          </button>
        </div>
      </article>
    </div>

    <div v-if="!plansQuery.isLoading.value && plans.length === 0" class="agent-library-empty">
      {{ t(isGuest ? 'guest.resources' : 'agent.noPlans') }}
    </div>

    <div
      v-if="!plansQuery.isLoading.value && plans.length"
      class="agent-library-section agent-library-plan-list"
    >
      <article v-for="plan in plans" :key="plan.id" class="agent-library-row">
        <div class="agent-library-row-main">
          <div class="agent-library-row-copy">
            <div class="flex flex-wrap items-center gap-2">
              <span class="status-pill" :class="statusClass(plan.status)">{{
                statusLabel(plan.status)
              }}</span>
              <span class="config-chip">{{ taskLabel(plan.task_type) }}</span>
              <span class="config-chip">{{ plan.model }}</span>
              <span v-if="plan.has_review" class="config-chip">{{ t('agent.hasReview') }}</span>
            </div>
            <h3>{{ plan.objective || t('agent.noObjective') }}</h3>
            <p class="muted-copy mt-1">
              {{ datasetName(plan.dataset_id) }} ·
              {{ t('agent.actionCount', { count: plan.action_count }) }} ·
              {{ t('agent.createdExperimentCount', { count: plan.created_experiment_count }) }}
              <span v-if="plan.canceled_experiment_count">
                ·
                {{
                  t('agent.canceledExperimentCount', { count: plan.canceled_experiment_count })
                }}</span
              >
            </p>
            <p class="muted-copy mt-1 text-xs">
              {{ t('agent.createdAt', { date: formatDate(plan.created_at) }) }}
              <span v-if="plan.executed_at">
                · {{ t('agent.executedAt', { date: formatDate(plan.executed_at) }) }}</span
              >
              <span v-if="plan.reviewed_at">
                · {{ t('agent.reviewedAt', { date: formatDate(plan.reviewed_at) }) }}</span
              >
            </p>
          </div>
          <div class="agent-library-row-actions">
            <button type="button" class="secondary-action compact-action" @click="viewPlan(plan)">
              <ArrowRight aria-hidden="true" />{{ t('agent.viewPlan') }}
            </button>
            <button type="button" class="secondary-action compact-action" @click="reusePlan(plan)">
              <Copy aria-hidden="true" />{{ t('agent.reusePlan') }}
            </button>
            <button
              type="button"
              class="danger-action compact-action"
              :disabled="!canDelete(plan) || deletingPlanId === plan.id"
              @click="confirmDelete(plan)"
            >
              <Trash2 aria-hidden="true" />{{
                deletingPlanId === plan.id ? t('agent.deletingPlan') : t('agent.deletePlan')
              }}
            </button>
          </div>
        </div>
      </article>

      <div v-if="totalPages > 1" class="agent-library-pagination">
        <button
          type="button"
          class="secondary-action icon-action compact-action"
          :disabled="currentPage <= 1"
          :aria-label="t('agent.previousPage')"
          :title="t('agent.previousPage')"
          @click="goToPreviousPage"
        >
          <ChevronLeft aria-hidden="true" />
        </button>
        <span class="muted-copy text-sm">
          {{ t('agent.pageInfo', { current: currentPage, total: totalPages }) }}
        </span>
        <button
          type="button"
          class="secondary-action icon-action compact-action"
          :disabled="currentPage >= totalPages"
          :aria-label="t('agent.nextPage')"
          :title="t('agent.nextPage')"
          @click="goToNextPage"
        >
          <ChevronRight aria-hidden="true" />
        </button>
      </div>
    </div>
  </section>
</template>
