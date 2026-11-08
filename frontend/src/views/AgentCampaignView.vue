<script setup lang="ts">
import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { api } from '../api/client'
import { completeAgentGeneration, streamAgentCampaignRound } from '../api/chatStream'
import { localizedApiError } from '../errorLocalization'
import { formatLocalizedDate } from '../localization'
import {
  ArrowLeft,
  ArrowRight,
  Database,
  History,
  Layers3,
  Plus,
  RefreshCw,
  Square,
} from '@lucide/vue'
import type {
  AgentCampaignRoundPayload,
  AgentExecutionPlan,
  AgentExecutionPlanSummary,
  AgentGenerationStage,
} from '../types'

const { locale, t, te } = useI18n()
const route = useRoute()
const router = useRouter()
const queryClient = useQueryClient()

const campaignId = computed(() => String(route.params.id || '').trim())
const selectedProviderId = ref('')
const includeDatasetPreview = ref(false)
const errorMessage = ref('')
const generationStage = ref<AgentGenerationStage | ''>('')
const generationStopped = ref(false)
let generationController: AbortController | null = null

const campaignQuery = useQuery({
  queryKey: computed(() => ['agent-campaign', campaignId.value]),
  queryFn: () => api.agentCampaign(campaignId.value),
  enabled: computed(() => Boolean(campaignId.value)),
})
const providersQuery = useQuery({ queryKey: ['ai-providers'], queryFn: api.aiProviders })
const datasetsQuery = useQuery({ queryKey: ['datasets'], queryFn: api.datasets })

const campaign = computed(() => campaignQuery.data.value)
const plans = computed(() => campaign.value?.plans ?? [])
const orderedPlans = computed(() =>
  [...plans.value].sort((left, right) => {
    const roundDiff = (left.round_index ?? 0) - (right.round_index ?? 0)
    if (roundDiff) return roundDiff
    return String(left.created_at || '').localeCompare(String(right.created_at || ''))
  }),
)
const latestPlan = computed(() => orderedPlans.value[orderedPlans.value.length - 1])
const enabledProviders = computed(() =>
  (providersQuery.data.value ?? []).filter((provider) => provider.enabled),
)
const dataset = computed(() =>
  (datasetsQuery.data.value ?? []).find((item) => item.id === campaign.value?.dataset_id),
)
const planDetailsQuery = useQuery({
  queryKey: computed(() => [
    'agent-campaign-plan-details',
    campaignId.value,
    orderedPlans.value.map((plan) => plan.id).join(','),
  ]),
  queryFn: () => Promise.all(orderedPlans.value.map((plan) => api.agentExecutionPlan(plan.id))),
  enabled: computed(() => Boolean(orderedPlans.value.length)),
})
const planDetailsById = computed(() => {
  const map = new Map<string, AgentExecutionPlan>()
  for (const plan of planDetailsQuery.data.value ?? []) {
    map.set(plan.id, plan)
  }
  return map
})
const latestPlanDetail = computed(() => {
  const plan = latestPlan.value
  return plan ? planDetailsById.value.get(plan.id) : undefined
})
const nextRoundBlockReason = computed(() => {
  if (!campaign.value || campaign.value.status !== 'active') return ''
  if (!orderedPlans.value.length) return ''
  if (
    (campaign.value.current_round_index ?? orderedPlans.value.length) >= campaign.value.max_rounds
  ) {
    const latest = latestPlan.value
    const detail = latestPlanDetail.value
    if (latest?.has_review || detail?.review?.summary) return t('agent.maxRoundsReached')
  }
  if (planDetailsQuery.isLoading.value || !latestPlanDetail.value)
    return t('agent.loadingRoundState')
  const progress = latestPlanDetail.value.execution?.progress ?? {}
  if (Array.isArray(progress.unfinished) && progress.unfinished.length)
    return t('agent.previousRoundUnfinished')
  if (!latestPlanDetail.value.review?.summary) return t('agent.previousRoundNeedsReview')
  return ''
})
const canCreateRound = computed(() =>
  Boolean(
    campaign.value?.id &&
    selectedProviderId.value &&
    campaign.value.status === 'active' &&
    !nextRoundBlockReason.value &&
    !roundMutation.isPending.value,
  ),
)
const roundProgressLabel = computed(() => {
  const current = campaign.value?.current_round_index ?? plans.value.length
  const max = campaign.value?.max_rounds ?? 0
  return t('agent.campaignRoundProgress', { current, max })
})

const roundMutation = useMutation({
  mutationFn: ({
    payload,
    controller,
  }: {
    payload: AgentCampaignRoundPayload
    controller: AbortController
  }) =>
    completeAgentGeneration(
      streamAgentCampaignRound(campaignId.value, payload, controller.signal),
      (stage) => {
        generationStage.value = stage
      },
    ),
  onSuccess: (plan) => {
    errorMessage.value = ''
    queryClient.invalidateQueries({ queryKey: ['agent-campaign', campaignId.value] })
    queryClient.invalidateQueries({ queryKey: ['agent-execution-plans'] })
    router.push(`/agent/design?plan_id=${plan.id}`)
  },
  onError: (error, variables) => {
    if (variables.controller.signal.aborted) {
      generationStopped.value = true
      errorMessage.value = ''
      return
    }
    errorMessage.value = localizedApiError(error, t, te, 'agent.campaignRoundGenericError')
  },
  onSettled: (_data, _error, variables) => {
    if (generationController === variables.controller) generationController = null
    generationStage.value = ''
  },
})

function createRound() {
  if (!canCreateRound.value) return
  errorMessage.value = ''
  generationStopped.value = false
  generationStage.value = 'generating'
  const controller = new AbortController()
  generationController = controller
  roundMutation.mutate({
    controller,
    payload: {
      provider_config_id: selectedProviderId.value,
      language: locale.value,
      include: {
        dataset_preview: includeDatasetPreview.value,
      },
    },
  })
}

function stopRoundGeneration() {
  generationController?.abort()
}

const roundGenerationStatus = computed(() =>
  t(`agent.generationStage.round.${generationStage.value || 'generating'}`),
)

watch(
  [enabledProviders, plans],
  ([providers, planItems]) => {
    if (
      selectedProviderId.value &&
      providers.some((provider) => provider.id === selectedProviderId.value)
    )
      return
    const lastProviderId = [...planItems]
      .reverse()
      .find((plan) => plan.provider_config_id)?.provider_config_id
    selectedProviderId.value =
      lastProviderId && providers.some((provider) => provider.id === lastProviderId)
        ? lastProviderId
        : (providers[0]?.id ?? '')
  },
  { immediate: true },
)

onBeforeUnmount(() => generationController?.abort())

function statusLabel(status: string) {
  if (['queued', 'running', 'succeeded', 'failed', 'canceled'].includes(status))
    return t(`status.${status}`)
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

function qualityLabel(plan: AgentExecutionPlanSummary) {
  return t('agent.planQuality', {
    valid: plan.valid_action_count ?? 0,
    total: plan.action_count,
  })
}

function planDetail(plan: AgentExecutionPlanSummary) {
  return planDetailsById.value.get(plan.id)
}

function stringList(value: unknown) {
  return Array.isArray(value) ? value.map((item) => String(item).trim()).filter(Boolean) : []
}

function reviewRecommendations(plan: AgentExecutionPlanSummary) {
  const review = planDetail(plan)?.review
  return stringList(
    review?.recommendations?.length ? review.recommendations : review?.next_recommendations,
  )
}

function reviewRisks(plan: AgentExecutionPlanSummary) {
  return stringList(planDetail(plan)?.review?.risks)
}

function reviewSummary(plan: AgentExecutionPlanSummary) {
  return String(planDetail(plan)?.review?.summary || '').trim()
}

function bestExperimentLabel(plan: AgentExecutionPlanSummary) {
  const best = planDetail(plan)?.review?.best_experiment ?? {}
  return String(best.name || best.experiment_id || '').trim()
}
</script>

<template>
  <section class="agent-campaign-page">
    <header class="agent-library-toolbar">
      <div>
        <p class="agent-rail-kicker">{{ t('agent.campaignEyebrow') }}</p>
        <h1>{{ campaign?.objective || t('agent.campaignTitle') }}</h1>
        <p>{{ dataset?.name || campaign?.dataset_id }} · {{ roundProgressLabel }}</p>
      </div>
      <div class="agent-library-toolbar-actions">
        <RouterLink
          to="/agent/plans"
          class="secondary-action compact-action inline-flex items-center gap-2"
        >
          <History aria-hidden="true" />{{ t('agent.history') }}
        </RouterLink>
        <RouterLink
          to="/agent"
          class="primary-action compact-action inline-flex items-center gap-2"
        >
          <Plus aria-hidden="true" />{{ t('agent.newPlan') }}
        </RouterLink>
      </div>
    </header>

    <div class="agent-campaign-workbench">
      <aside class="agent-campaign-context">
        <div class="agent-campaign-panel-head">
          <Database aria-hidden="true" />
          <div
            ><p class="agent-rail-kicker">{{ t('agent.workspaceContext') }}</p
            ><h2>{{ t('agent.workspaceResearchInputs') }}</h2></div
          >
        </div>
        <div v-if="campaign" class="agent-campaign-meta">
          <span class="status-pill" :class="statusClass(campaign.status)">{{
            statusLabel(campaign.status)
          }}</span>
          <dl>
            <div
              ><dt>{{ t('agent.task') }}</dt
              ><dd>{{ campaign.task_type }}</dd></div
            >
            <div
              ><dt>{{ t('agent.dataset') }}</dt
              ><dd>{{ dataset?.name || campaign.dataset_id }}</dd></div
            >
            <div
              ><dt>{{ t('agent.workspaceCreated') }}</dt
              ><dd>{{ formatDate(campaign.created_at) }}</dd></div
            >
          </dl>
        </div>

        <div class="agent-campaign-next-round">
          <label class="form-field">
            <span>{{ t('agent.provider') }}</span>
            <select v-model="selectedProviderId" class="field-control">
              <option v-for="provider in enabledProviders" :key="provider.id" :value="provider.id">
                {{ provider.display_name }} · {{ provider.model }}
              </option>
            </select>
          </label>
          <label class="agent-context-toggle">
            <span
              ><Database aria-hidden="true" /><span
                ><strong>{{ t('agent.includeDatasetPreview') }}</strong></span
              ></span
            >
            <input v-model="includeDatasetPreview" type="checkbox" />
          </label>
          <button
            type="button"
            class="inline-flex items-center justify-center gap-2"
            :class="
              roundMutation.isPending.value
                ? 'secondary-action agent-generation-stop'
                : 'primary-action'
            "
            :disabled="roundMutation.isPending.value ? false : !canCreateRound"
            @click="roundMutation.isPending.value ? stopRoundGeneration() : createRound()"
          >
            <Square v-if="roundMutation.isPending.value" aria-hidden="true" />
            <RefreshCw v-else aria-hidden="true" />
            {{
              roundMutation.isPending.value ? t('agent.stopGenerating') : t('agent.createNextRound')
            }}
          </button>
          <p
            v-if="roundMutation.isPending.value"
            class="agent-generation-status"
            role="status"
            aria-live="polite"
          >
            <span class="agent-generation-pulse" /><span>{{ roundGenerationStatus }}</span>
          </p>
          <p v-else-if="generationStopped" class="agent-generation-stopped" role="status">
            {{ t('agent.roundGenerationStopped') }}
          </p>
          <p v-if="nextRoundBlockReason" class="muted-copy text-xs">{{ nextRoundBlockReason }}</p>
          <p v-if="errorMessage" class="error-card text-sm" role="alert">{{ errorMessage }}</p>
        </div>
      </aside>

      <section class="agent-campaign-timeline">
        <div class="agent-campaign-panel-head">
          <Layers3 aria-hidden="true" />
          <div
            ><p class="agent-rail-kicker">{{ t('agent.workspaceExecutionActivity') }}</p
            ><h2>{{ roundProgressLabel }}</h2></div
          >
        </div>
        <div v-if="campaignQuery.isLoading.value" class="agent-library-empty">{{
          t('agent.loadingCampaign')
        }}</div>
        <div v-else-if="!orderedPlans.length" class="agent-library-empty">{{
          t('agent.noCampaignRounds')
        }}</div>
        <div v-else class="agent-round-list">
          <article v-for="plan in orderedPlans" :key="plan.id" class="agent-round-row">
            <span class="agent-round-marker">{{ plan.round_index || 0 }}</span>
            <div class="agent-round-content">
              <div class="agent-round-head">
                <div>
                  <div class="flex flex-wrap gap-2">
                    <span class="config-chip">{{ statusLabel(plan.status) }}</span>
                    <span class="config-chip">{{ plan.model }}</span>
                    <span class="config-chip">{{ qualityLabel(plan) }}</span>
                  </div>
                  <h3>{{ plan.objective }}</h3>
                  <p
                    >{{ t('agent.createdExperimentCount', { count: plan.created_experiment_count })
                    }}<span v-if="plan.has_review"> · {{ t('agent.hasReview') }}</span></p
                  >
                </div>
                <RouterLink :to="`/agent/design?plan_id=${plan.id}`" class="agent-library-open">
                  <span>{{ t('agent.viewPlan') }}</span
                  ><ArrowRight aria-hidden="true" />
                </RouterLink>
              </div>
              <p v-if="reviewSummary(plan)" class="agent-round-review">{{ reviewSummary(plan) }}</p>
            </div>
          </article>
        </div>
      </section>

      <aside class="agent-campaign-review">
        <div class="agent-campaign-panel-head">
          <ArrowLeft aria-hidden="true" />
          <div
            ><p class="agent-rail-kicker">{{ t('agent.latestReview') }}</p
            ><h2>{{ latestPlan?.objective || t('agent.noCampaignRounds') }}</h2></div
          >
        </div>
        <template v-if="latestPlan && reviewSummary(latestPlan)">
          <p class="agent-review-lead">{{ reviewSummary(latestPlan) }}</p>
          <div v-if="bestExperimentLabel(latestPlan)" class="agent-review-highlight">
            <small>{{ t('agent.campaignBestExperiment') }}</small
            ><strong>{{ bestExperimentLabel(latestPlan) }}</strong>
          </div>
          <div v-if="reviewRecommendations(latestPlan).length" class="agent-inspector-copy">
            <strong>{{ t('agent.campaignRecommendations') }}</strong>
            <ul
              ><li v-for="item in reviewRecommendations(latestPlan)" :key="item">{{ item }}</li></ul
            >
          </div>
          <div v-if="reviewRisks(latestPlan).length" class="agent-inspector-copy">
            <strong>{{ t('agent.campaignRisks') }}</strong>
            <ul
              ><li v-for="item in reviewRisks(latestPlan)" :key="item">{{ item }}</li></ul
            >
          </div>
        </template>
        <p v-else class="agent-rail-empty">{{
          nextRoundBlockReason || t('agent.noCampaignRounds')
        }}</p>
      </aside>
    </div>
  </section>
</template>
