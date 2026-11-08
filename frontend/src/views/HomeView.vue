<script setup lang="ts">
import { normalizeTaskType } from '../taskTerminology'
import {
  ArrowUpRight,
  ArrowRight,
  FlaskConical,
  SlidersHorizontal,
  Activity,
  ChartNoAxesCombined,
  Boxes,
  RefreshCw,
} from '@lucide/vue'
import WorkbenchArt from '../components/WorkbenchArt.vue'
import ExperimentActivityHeatmap from '../components/ExperimentActivityHeatmap.vue'
import { useAuthStore } from '../stores/auth'
import { useQuery } from '@tanstack/vue-query'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { api } from '../api/client'
import { useWorkspaceAccess } from '../workspaceAccess'
import { buildHomeDashboard } from '../homeDashboard'
import { experimentDisplayName, formatLocalizedDate, modelDisplayName } from '../localization'
import { useExperimentDraftStore } from '../stores/experimentDraft'
import { TASK_ENTRIES } from '../taskCatalog'

const { canUseWorkspace, requireLogin } = useWorkspaceAccess()

const { locale, t } = useI18n()
const router = useRouter()
const draft = useExperimentDraftStore()
const auth = useAuthStore()
const experiments = useQuery({
  queryKey: ['experiments'],
  queryFn: api.experiments,
  enabled: canUseWorkspace,
})
const models = useQuery({ queryKey: ['models'], queryFn: api.models, enabled: canUseWorkspace })
const dashboard = computed(() => buildHomeDashboard(experiments.data.value, models.data.value))
const experimentById = computed(
  () => new Map((experiments.data.value ?? []).map((item) => [item.id, item])),
)
const familyIdByTask = computed(() => new Map(TASK_ENTRIES.map((entry) => [entry.key, entry.id])))

const todayLabel = computed(() =>
  formatLocalizedDate(new Date(), locale.value, {
    month: 'short',
    day: 'numeric',
    weekday: 'short',
  }),
)

const experimentUnavailable = computed(
  () => experiments.isPending.value || experiments.isError.value,
)
const modelUnavailable = computed(() => models.isPending.value || models.isError.value)
const modelRatio = computed(() => {
  const total = dashboard.value.reusableModelCount + dashboard.value.metricsOnlyModelCount
  return total ? dashboard.value.reusableModelCount / total : 0
})

const researchStages = computed(() => [
  {
    number: '01',
    icon: SlidersHorizontal,
    label: t('home.pathConfigure'),
    detail: t('home.pathConfigureHint'),
    to: '/experiments/new',
    fresh: true,
    filled: true,
  },
  {
    number: '02',
    icon: Activity,
    label: t('home.pathTrain'),
    detail: t('home.pathTrainHint', { count: dashboard.value.experimentCount }),
    to: '/experiments',
    fresh: false,
    filled: dashboard.value.experimentCount > 0,
  },
  {
    number: '03',
    icon: ChartNoAxesCombined,
    label: t('home.pathEvaluate'),
    detail: t('home.pathEvaluateHint', { count: dashboard.value.completedCount }),
    to: '/leaderboard',
    fresh: false,
    filled: dashboard.value.completedCount > 0,
  },
  {
    number: '04',
    icon: Boxes,
    label: t('home.pathInfer'),
    detail: t('home.pathInferHint', { count: dashboard.value.reusableModelCount }),
    to: '/models',
    fresh: false,
    filled: dashboard.value.reusableModelCount > 0,
  },
])

const families = computed(() =>
  dashboard.value.taskFamilies.map((family) => ({
    ...family,
    label: t(`tasks.${family.key}.label`),
    started: family.runs > 0,
  })),
)

function retryData() {
  if (!requireLogin()) return
  void experiments.refetch()
  void models.refetch()
}

function countLabel(value: number) {
  return new Intl.NumberFormat(locale.value).format(value)
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

function familyId(taskType: string) {
  return familyIdByTask.value.get(taskType as (typeof TASK_ENTRIES)[number]['key']) ?? ''
}

function formatDate(value?: string | null) {
  return formatLocalizedDate(value, locale.value, { month: 'short', day: 'numeric' })
}

function displayExperimentName(experiment: Parameters<typeof experimentDisplayName>[0]) {
  return experimentDisplayName(experiment, t)
}

function displayModelName(model: Parameters<typeof modelDisplayName>[0]) {
  return modelDisplayName(model, experimentById.value.get(model.experiment_id), t)
}

function experimentDate(experiment: {
  finished_at?: string | null
  updated_at?: string
  created_at?: string
}) {
  return formatDate(experiment.finished_at || experiment.updated_at || experiment.created_at)
}

function startNewExperiment() {
  draft.resetForTask('pep_function')
  router.push('/experiments/new')
}

function startFamily(taskKey: string) {
  draft.resetForTask(taskKey)
  router.push('/experiments/new')
}

function enterStage(stage: { fresh: boolean }) {
  if (stage.fresh) {
    draft.resetForTask('pep_function')
  }
}
</script>

<template>
  <section class="home-dashboard">
    <header class="home-overview">
      <div class="home-overview-copy">
        <p class="home-kicker">
          <span class="home-kicker-dot" aria-hidden="true" />{{ t('home.overviewKicker')
          }}<span class="home-kicker-date">{{ todayLabel }}</span>
        </p>
        <h1>{{ t('home.title') }}</h1>
        <p class="home-overview-lead">{{ t('home.subtitle') }}</p>
        <div class="home-overview-actions">
          <button type="button" class="primary-action" @click="startNewExperiment">
            <FlaskConical aria-hidden="true" />{{ t('home.newExperiment')
            }}<ArrowRight aria-hidden="true" />
          </button>
          <RouterLink to="/experiments" class="home-text-action"
            >{{ t('home.viewExperiments') }}<ArrowUpRight aria-hidden="true"
          /></RouterLink>
        </div>
      </div>
      <div class="home-overview-art">
        <WorkbenchArt /><span class="home-art-caption">{{ t('home.artCaption') }}</span>
      </div>
    </header>

    <div
      v-if="experiments.isError.value || models.isError.value"
      class="home-data-note"
      role="alert"
    >
      <span>{{ t('home.loadError') }}</span>
      <button type="button" class="tertiary-action compact-action" @click="retryData">
        <RefreshCw aria-hidden="true" />{{ t('home.retry') }}
      </button>
    </div>

    <dl class="home-readout">
      <div class="home-stat home-stat--activity" :aria-busy="experiments.isLoading.value">
        <dt>
          <span class="home-stat-icon"><FlaskConical aria-hidden="true" /></span
          >{{ t('home.totalExperiments') }}
        </dt>
        <dd class="home-stat-main">
          <span class="home-stat-number">{{
            experimentUnavailable ? '—' : countLabel(dashboard.experimentCount)
          }}</span>
          <FlaskConical class="home-model-mark" aria-hidden="true" />
        </dd>
        <dd class="home-stat-foot">
          {{ t('home.contextAllTasks') }}
        </dd>
      </div>
      <div class="home-stat" :aria-busy="experiments.isLoading.value">
        <dt>
          <span class="home-stat-icon"><ChartNoAxesCombined aria-hidden="true" /></span
          >{{ t('home.completedExperiments') }}
        </dt>
        <dd class="home-stat-main">
          <span class="home-stat-number">{{
            experimentUnavailable ? '—' : countLabel(dashboard.completedCount)
          }}</span>
          <svg class="home-completion-ring" viewBox="0 0 64 64" aria-hidden="true">
            <circle class="home-ring-track" cx="32" cy="32" r="25" />
            <circle
              class="home-ring-value"
              cx="32"
              cy="32"
              r="25"
              pathLength="100"
              :stroke-dasharray="`${experimentUnavailable ? 0 : dashboard.completionRate * 100} 100`"
            />
            <text x="32" y="35">
              {{
                experimentUnavailable || !dashboard.experimentCount
                  ? '—'
                  : `${Math.round(dashboard.completionRate * 100)}%`
              }}
            </text>
          </svg>
        </dd>
        <dd class="home-stat-foot">
          {{
            !experimentUnavailable && dashboard.experimentCount
              ? t('home.contextCompletionRate', {
                  rate: Math.round(dashboard.completionRate * 100),
                })
              : t('home.awaitingRuns')
          }}
        </dd>
      </div>
      <div class="home-stat" :aria-busy="models.isLoading.value">
        <dt>
          <span class="home-stat-icon"><Boxes aria-hidden="true" /></span
          >{{ t('home.reusableModels') }}
        </dt>
        <dd class="home-stat-main">
          <span class="home-stat-number">{{
            modelUnavailable ? '—' : countLabel(dashboard.reusableModelCount)
          }}</span
          ><Boxes class="home-model-mark" aria-hidden="true" />
        </dd>
        <dd class="home-stat-foot">
          <span>{{
            !modelUnavailable && dashboard.metricsOnlyModelCount
              ? t('home.contextMetricsOnly', { count: dashboard.metricsOnlyModelCount })
              : !modelUnavailable && dashboard.reusableModelCount
                ? t('home.modelsReady')
                : t('home.awaitingModels')
          }}</span
          ><span class="home-model-ratio" aria-hidden="true"
            ><i :style="{ width: `${modelUnavailable ? 0 : modelRatio * 100}%` }"
          /></span>
        </dd>
      </div>
    </dl>

    <p v-if="!experimentUnavailable && dashboard.experimentCount === 0" class="home-first-run">
      <span>{{ t('home.firstRunHint') }}</span
      ><button type="button" @click="startNewExperiment">
        {{ t('home.newExperiment') }}<ArrowUpRight aria-hidden="true" />
      </button>
    </p>

    <div class="home-main">
      <section class="panel home-panel" aria-labelledby="research-path-title">
        <header class="home-panel-head">
          <div>
            <p class="home-panel-kicker">{{ t('home.pathKicker') }}</p>
            <h2 id="research-path-title">{{ t('home.pathTitle') }}</h2>
          </div>
          <span class="home-panel-meta" aria-hidden="true">01—04</span>
        </header>

        <ol class="home-rail">
          <li
            v-for="stage in researchStages"
            :key="stage.number"
            class="home-rail-stage"
            :class="{ 'is-void': !stage.filled }"
          >
            <RouterLink :to="stage.to" class="home-rail-link" @click="enterStage(stage)">
              <span class="home-rail-node" aria-hidden="true"><component :is="stage.icon" /></span>
              <span class="home-rail-index" aria-hidden="true">{{ stage.number }}</span>
              <h3>{{ stage.label }}</h3>
              <p>{{ stage.detail }}</p>
            </RouterLink>
          </li>
        </ol>

        <RouterLink
          v-if="dashboard.latestCompleted"
          :to="`/experiments/${dashboard.latestCompleted.id}`"
          class="latest-research"
        >
          <span class="latest-research-label">{{ t('home.latestCompleted') }}</span>
          <span class="latest-research-main">
            <strong>{{ displayExperimentName(dashboard.latestCompleted) }}</strong>
            <span class="latest-research-date">{{
              experimentDate(dashboard.latestCompleted)
            }}</span>
          </span>
          <ArrowUpRight class="latest-research-arrow" aria-hidden="true" />
        </RouterLink>
        <div v-else class="latest-research latest-research--empty">
          <span class="latest-research-label">{{ t('home.latestCompleted') }}</span>
          <strong>{{ t('home.noCompletedExperiment') }}</strong>
        </div>
      </section>

      <section class="panel home-panel" aria-labelledby="task-families-title">
        <header class="home-panel-head">
          <div>
            <p class="home-panel-kicker">T01—T04</p>
            <h2 id="task-families-title">{{ t('home.taskFamilies') }}</h2>
          </div>
          <span class="home-panel-meta">
            {{
              t('home.taskFamiliesMeta', {
                started: dashboard.familiesStarted,
                total: dashboard.taskFamilies.length,
              })
            }}
          </span>
        </header>

        <ul class="home-families">
          <li v-for="family in families" :key="family.key">
            <RouterLink
              :to="{ path: '/experiments/new' }"
              class="home-family"
              @click="startFamily(family.key)"
            >
              <span class="home-family-id">{{ family.id }}</span>
              <span class="home-family-label">{{ family.label }}</span>
              <span class="home-family-metrics">
                {{
                  family.started
                    ? `${t('home.familyRuns', { count: family.runs })} · ${t('home.familyReady', {
                        count: family.readyModels,
                      })}`
                    : t('home.familyNotStarted')
                }}
              </span>
              <span class="home-family-action">
                {{ t('home.startFromFamily') }}
                <ArrowUpRight aria-hidden="true" />
              </span>
            </RouterLink>
          </li>
        </ul>
      </section>
    </div>

    <aside class="home-feed" :aria-label="t('home.activityFeed')">
      <section class="panel home-feed-panel">
        <header class="home-feed-head">
          <p class="eyebrow">{{ t('home.recentExperiments') }}</p>
          <span class="home-feed-count">{{ countLabel(dashboard.recentExperiments.length) }}</span>
          <RouterLink to="/experiments" class="home-feed-more">
            {{ t('home.viewAll') }}
            <ArrowUpRight aria-hidden="true" />
          </RouterLink>
        </header>
        <ul class="home-feed-list">
          <li v-for="experiment in dashboard.recentExperiments" :key="experiment.id">
            <RouterLink :to="`/experiments/${experiment.id}`" class="home-feed-row">
              <span class="home-feed-row-top">
                <span v-if="familyId(experiment.task_type)" class="home-feed-id">
                  {{ familyId(experiment.task_type) }}
                </span>
                <span class="home-feed-name">{{ displayExperimentName(experiment) }}</span>
                <ArrowUpRight class="home-feed-arrow" aria-hidden="true" />
              </span>
              <span class="home-feed-meta">
                <span>{{ taskLabel(experiment.task_type) }}</span>
                <template v-if="experimentDate(experiment)">
                  <span aria-hidden="true">·</span>
                  <span>{{ experimentDate(experiment) }}</span>
                </template>
                <span class="status-pill" :class="statusClass(experiment.status)">
                  {{ statusLabel(experiment.status) }}
                </span>
              </span>
            </RouterLink>
          </li>
          <li v-if="!dashboard.recentExperiments.length" class="home-feed-empty">
            <p>
              {{
                experiments.isLoading.value
                  ? t('common.loading')
                  : experiments.isError.value
                    ? t('home.loadError')
                    : t('common.empty')
              }}
            </p>
            <RouterLink to="/experiments/new" @click="startNewExperiment">
              {{ t('home.newExperiment') }}
            </RouterLink>
          </li>
        </ul>
      </section>

      <section class="panel home-feed-panel">
        <header class="home-feed-head">
          <p class="eyebrow">{{ t('home.recentModels') }}</p>
          <span class="home-feed-count">{{ countLabel(dashboard.recentModels.length) }}</span>
          <RouterLink to="/models" class="home-feed-more">
            {{ t('home.viewAll') }}
            <ArrowUpRight aria-hidden="true" />
          </RouterLink>
        </header>
        <ul class="home-feed-list">
          <li v-for="model in dashboard.recentModels" :key="model.id">
            <RouterLink :to="{ path: '/models', query: { model: model.id } }" class="home-feed-row">
              <span class="home-feed-row-top">
                <span v-if="familyId(model.task_type)" class="home-feed-id">
                  {{ familyId(model.task_type) }}
                </span>
                <span class="home-feed-name">{{ displayModelName(model) }}</span>
                <ArrowUpRight class="home-feed-arrow" aria-hidden="true" />
              </span>
              <span class="home-feed-meta">
                <span>{{ taskLabel(model.task_type) }}</span>
                <template v-if="formatDate(model.created_at)">
                  <span aria-hidden="true">·</span>
                  <span>{{ formatDate(model.created_at) }}</span>
                </template>
                <span
                  class="status-pill"
                  :class="model.supports_inference ? 'status-pill-success' : 'status-pill-neutral'"
                >
                  {{
                    model.supports_inference
                      ? t('common.readyForInference')
                      : t('common.metricsOnly')
                  }}
                </span>
              </span>
            </RouterLink>
          </li>
          <li v-if="!dashboard.recentModels.length" class="home-feed-empty">
            <p>
              {{
                models.isLoading.value
                  ? t('common.loading')
                  : models.isError.value
                    ? t('home.loadError')
                    : t('common.empty')
              }}
            </p>
          </li>
        </ul>
      </section>
    </aside>

    <ExperimentActivityHeatmap
      class="home-calendar"
      :experiments="experiments.data.value"
      :owner-id="auth.user?.id"
      :workspace="!auth.authEnabled"
      :loading="experiments.isPending.value"
      :failed="experiments.isError.value"
      :guest="!canUseWorkspace"
      @retry="retryData"
    />
  </section>
</template>

<style scoped>
.home-dashboard {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(19rem, 0.43fr);
  gap: 1.35rem 1.5rem;
}
.home-main {
  display: grid;
  min-width: 0;
  align-content: start;
  gap: 1.5rem;
}
.home-dashboard > .home-calendar {
  margin-top: 0;
}
.home-overview {
  position: relative;
  display: grid;
  grid-column: 1 / -1;
  grid-template-columns: 1.15fr 1fr;
  align-items: center;
  min-height: 17.5rem;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--color-accent) 12%, var(--color-line));
  border-radius: 1.25rem;
  background: linear-gradient(
    112deg,
    var(--surface-strong) 15%,
    color-mix(in srgb, var(--color-accent) 5%, var(--surface-strong))
  );
}
.home-overview-copy {
  position: relative;
  z-index: 1;
  padding: 2rem 0 2rem 2.25rem;
}
.home-kicker {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.6rem;
  color: var(--color-accent-dark);
  font-size: 0.67rem;
  font-weight: 600;
  letter-spacing: 0.11em;
  text-transform: uppercase;
}
.home-kicker-dot {
  width: 0.4rem;
  height: 0.4rem;
  border-radius: 50%;
  background: var(--color-accent);
  box-shadow: 0 0 0 4px var(--color-accent-soft);
}
.home-kicker-date {
  margin-left: 0.25rem;
  border-left: 1px solid var(--color-line);
  padding-left: 0.8rem;
  color: var(--color-muted);
  font-weight: 400;
  letter-spacing: 0.02em;
}
.home-overview h1 {
  max-width: 37rem;
  margin-top: 1.1rem;
  font-size: clamp(1.9rem, 3vw, 2.8rem);
  font-weight: 650;
  line-height: 1.16;
  letter-spacing: -0.04em;
  text-wrap: balance;
}
.home-overview-lead {
  max-width: 33rem;
  margin-top: 0.8rem;
  color: var(--color-muted);
  font-size: 0.9rem;
  line-height: 1.75;
  text-wrap: pretty;
}
.home-overview-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.8rem 1.35rem;
  margin-top: 1.5rem;
}
.home-overview-actions .primary-action {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
}
.home-overview-actions svg {
  width: 1rem;
  height: 1rem;
}
.home-overview-actions .primary-action svg:last-child {
  margin-left: 0.6rem;
}
.home-text-action {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  color: var(--color-accent-dark);
  font-size: 0.78rem;
  font-weight: 600;
}
.home-text-action:hover {
  text-decoration: underline;
  text-underline-offset: 0.25rem;
}
.home-overview-art {
  position: relative;
  align-self: stretch;
  display: grid;
  place-content: center;
  padding: 0.1rem 1.5rem 0.8rem 0;
}
.home-art-caption {
  position: absolute;
  bottom: 1.4rem;
  right: 2rem;
  color: var(--color-muted);
  font-size: 0.61rem;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}
.home-readout {
  display: grid;
  grid-column: 1 / -1;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1.2rem;
}
.home-stat {
  min-width: 0;
  overflow: hidden;
  border: 1px solid var(--color-line);
  border-radius: 0.9rem;
  background: var(--surface-strong);
  padding: 1.1rem 1.3rem 1rem;
  box-shadow: 0 2px 6px color-mix(in srgb, var(--color-ink) 2%, transparent);
}
.home-stat dt {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--color-muted);
  font-size: 0.75rem;
  font-weight: 500;
}
.home-stat-icon {
  display: grid;
  width: 1.65rem;
  height: 1.65rem;
  place-items: center;
  border-radius: 0.5rem;
  background: var(--color-accent-soft);
  color: var(--color-accent);
}
.home-stat-icon svg {
  width: 0.95rem;
  height: 0.95rem;
}
.home-stat-main {
  display: flex;
  min-width: 0;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin: 0.6rem 0 0.25rem;
}
.home-stat-number {
  min-width: 0;
  overflow-wrap: anywhere;
  font-size: 2.55rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  line-height: 1.15;
  letter-spacing: -0.05em;
}
.home-stat-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin: 0.35rem 0 0;
  color: var(--color-muted);
  font-size: 0.7rem;
}
.home-activity-bars {
  width: 8.75rem;
  max-width: 48%;
  height: 3rem;
  flex: 0 1 auto;
  fill: var(--color-accent);
}
.home-activity-bars .is-idle {
  opacity: 0.17;
}
.home-activity-bars.is-unavailable {
  opacity: 0.15;
}
.home-completion-ring {
  flex: 0 0 auto;
  width: 3.4rem;
  height: 3.4rem;
}
.home-completion-ring circle {
  fill: none;
  stroke-width: 4;
}
.home-ring-track {
  stroke: var(--line-soft);
}
.home-ring-value {
  stroke: var(--color-accent);
  stroke-linecap: round;
  transform: rotate(-90deg);
  transform-origin: center;
}
.home-completion-ring text {
  fill: var(--color-accent-dark);
  font-family: var(--font-body);
  font-size: 11px;
  font-weight: 600;
  text-anchor: middle;
}
.home-model-mark {
  width: 3rem;
  height: 3rem;
  margin-right: 0.4rem;
  color: var(--color-accent);
  stroke-width: 0.8;
  opacity: 0.65;
}
.home-model-ratio {
  display: block;
  width: 6rem;
  max-width: 35%;
  height: 4px;
  flex: 0 0 auto;
  overflow: hidden;
  border-radius: 9px;
  background: var(--color-line);
}
.home-model-ratio i {
  display: block;
  height: 100%;
  background: var(--color-accent);
}
.home-first-run,
.home-data-note {
  grid-column: 1 / -1;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.6rem 1rem;
  margin: 0;
  color: var(--color-muted);
  font-size: 0.78rem;
}
.home-first-run button {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  color: var(--color-accent-dark);
  font-weight: 600;
}
.home-first-run svg,
.home-data-note svg {
  width: 0.9rem;
  height: 0.9rem;
}
.home-data-note {
  border: 1px solid var(--color-line);
  border-radius: 0.7rem;
  background: var(--surface-strong);
  padding: 0.6rem 1rem;
}
.home-panel {
  display: grid;
  gap: 1rem;
}

.home-panel-head {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  justify-content: space-between;
  gap: 0.4rem 1rem;
}

.home-panel-kicker {
  color: var(--color-accent);
  font-family: var(--font-mono);
  font-size: 0.64rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.home-panel-head h2 {
  margin-top: 0.28rem;
  font-size: 1.02rem;
  font-weight: 600;
  letter-spacing: -0.01em;
}

.home-panel-meta {
  color: var(--color-muted);
  font-size: 0.7rem;
  font-variant-numeric: tabular-nums;
}

/* --- Research rail --- */

.home-rail {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.home-rail-stage {
  min-width: 0;
}

.home-rail-stage + .home-rail-stage {
  border-left: 1px solid var(--line-soft);
}

.home-rail-link {
  display: grid;
  height: 100%;
  align-content: start;
  gap: 0.32rem;
  border-radius: var(--radius-control);
  padding: 0.15rem 1rem 0.35rem;
}

.home-rail-stage:first-child .home-rail-link {
  padding-left: 0;
}

.home-rail-stage:last-child .home-rail-link {
  padding-right: 0;
}

.home-rail-link:hover {
  background: color-mix(in srgb, var(--color-accent) 3%, transparent);
}

.home-rail-link:focus-visible {
  outline: 3px solid var(--focus-ring);
  outline-offset: 2px;
}

.home-rail-node {
  display: inline-flex;
  width: 1.6rem;
  height: 1.6rem;
  align-items: center;
  justify-content: center;
  border: 1px solid transparent;
  border-radius: 50%;
  background: var(--color-accent-soft);
  color: var(--color-accent-dark);
  font-size: 0.62rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

:root[data-theme='dark'] .home-rail-node {
  color: var(--color-accent);
}

.home-rail-stage.is-void .home-rail-node {
  border-color: var(--color-line);
  border-style: dashed;
  background: transparent;
  color: var(--color-muted);
}

.home-rail-link h3 {
  margin-top: 0.35rem;
  font-size: 0.85rem;
  font-weight: 600;
}

.home-rail-link:hover h3 {
  color: var(--color-accent-dark);
}

.home-rail-link p {
  color: var(--color-muted);
  font-size: 0.72rem;
  line-height: 1.55;
}

.latest-research {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 0.35rem 0.9rem;
  border: 1px solid var(--color-line);
  border-left: 2px solid var(--color-accent);
  border-radius: var(--radius-control);
  padding: 0.7rem 0.9rem;
}

.latest-research:hover {
  background: color-mix(in srgb, var(--color-accent) 3%, var(--surface-strong));
}

.latest-research-label {
  color: var(--color-muted);
  font-family: var(--font-mono);
  font-size: 0.64rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.latest-research-main {
  display: flex;
  min-width: 0;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.15rem 0.6rem;
}

.latest-research strong {
  min-width: 0;
  overflow: hidden;
  font-size: 0.82rem;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.latest-research-date {
  color: var(--color-muted);
  font-size: 0.7rem;
  font-variant-numeric: tabular-nums;
}

.latest-research-arrow {
  width: 0.95rem;
  height: 0.95rem;
  color: var(--color-accent);
}

.latest-research--empty {
  grid-template-columns: auto minmax(0, 1fr);
  border-left-color: var(--color-line);
}

.latest-research--empty strong {
  color: var(--color-muted);
  font-size: 0.78rem;
  font-weight: 500;
}

/* --- Task families --- */

.home-families {
  display: grid;
  border-top: 1px solid var(--line-soft);
}

.home-families li + li {
  border-top: 1px solid var(--line-soft);
}

.home-family {
  display: grid;
  grid-template-columns: 2.6rem minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 0.4rem 1rem;
  border-left: 2px solid transparent;
  border-radius: var(--radius-control);
  padding: 0.7rem 0.6rem;
}

.home-family:hover {
  border-left-color: var(--color-accent);
  background: color-mix(in srgb, var(--color-accent) 3%, transparent);
}

.home-family:focus-visible {
  outline: 3px solid var(--focus-ring);
  outline-offset: 2px;
}

.home-family-id {
  color: var(--color-muted);
  font-family: var(--font-mono);
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.06em;
}

.home-family-label {
  min-width: 0;
  overflow-wrap: anywhere;
  font-size: 0.85rem;
  font-weight: 600;
  line-height: 1.5;
}

.home-family-metrics {
  color: var(--color-muted);
  font-size: 0.7rem;
  font-variant-numeric: tabular-nums;
}

.home-family-action {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  color: var(--color-muted);
  font-size: 0.72rem;
  font-weight: 600;
}

.home-family-action svg {
  width: 0.85rem;
  height: 0.85rem;
}

.home-family:hover .home-family-action {
  color: var(--color-accent-dark);
}

/* --- Activity feed --- */

.home-feed {
  display: grid;
  min-width: 0;
  align-content: start;
  gap: 1.5rem;
}

.home-feed-panel {
  display: grid;
  gap: 0.4rem;
}

.home-feed-head {
  display: flex;
  align-items: center;
  gap: 0.55rem;
}

.home-feed-count {
  color: var(--color-muted);
  font-family: var(--font-mono);
  font-size: 0.68rem;
  font-variant-numeric: tabular-nums;
}

.home-feed-more {
  display: inline-flex;
  align-items: center;
  gap: 0.2rem;
  margin-left: auto;
  color: var(--color-muted);
  font-size: 0.7rem;
  font-weight: 600;
}

.home-feed-more svg {
  width: 0.85rem;
  height: 0.85rem;
}

.home-feed-more:hover {
  color: var(--color-accent-dark);
}

.home-feed-list {
  display: grid;
  border-top: 1px solid var(--line-soft);
}

.home-feed-list li + li {
  border-top: 1px solid var(--line-soft);
}

.home-feed-row {
  display: grid;
  gap: 0.22rem;
  border-radius: var(--radius-control);
  padding: 0.65rem 0.45rem;
}

.home-feed-row:hover {
  background: color-mix(in srgb, var(--color-accent) 3%, transparent);
}

.home-feed-row-top {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 0.5rem;
}

.home-feed-arrow {
  flex: 0 0 auto;
  margin-left: auto;
  width: 0.9rem;
  height: 0.9rem;
  color: color-mix(in srgb, var(--color-muted) 55%, transparent);
}

.home-feed-id {
  flex: 0 0 auto;
  color: var(--color-muted);
  font-family: var(--font-mono);
  font-size: 0.66rem;
  letter-spacing: 0.04em;
}

.home-feed-name {
  min-width: 0;
  overflow: hidden;
  font-size: 0.84rem;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.home-feed-row-top :deep(.status-pill) {
  flex: 0 0 auto;
  margin-left: auto;
}

.home-feed-arrow {
  flex: 0 0 auto;
  width: 0.9rem;
  height: 0.9rem;
  color: color-mix(in srgb, var(--color-muted) 55%, transparent);
}

.home-feed-row:hover .home-feed-arrow {
  color: var(--color-accent);
}

.home-feed-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.15rem 0.35rem;
  color: var(--color-muted);
  font-size: 0.68rem;
  font-variant-numeric: tabular-nums;
  line-height: 1.4;
}

.home-feed-meta :deep(.status-pill) {
  flex: 0 0 auto;
  margin-left: auto;
}

.home-feed-empty {
  display: grid;
  justify-items: center;
  gap: 0.45rem;
  border: 1px dashed var(--color-line);
  border-radius: var(--radius-control);
  margin-top: 0.7rem;
  padding: 1.15rem;
  color: var(--color-muted);
  font-size: 0.78rem;
  text-align: center;
}

.home-feed-empty a {
  color: var(--color-accent-dark);
  font-weight: 600;
}

/* --- Breakpoints --- */

@media (min-width: 1024px) {
  .home-dashboard {
    grid-template-columns: minmax(0, 1fr) minmax(19rem, 0.42fr);
    align-items: start;
  }
}

@media (max-width: 900px) {
  .home-rail {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .home-rail-stage:nth-child(3) {
    border-left: 0;
  }

  .home-rail-stage:nth-child(n + 3) {
    border-top: 0;
    padding-top: 0;
  }

  .home-rail-stage:first-child .home-rail-link,
  .home-rail-link {
    padding-left: 0.2rem;
    padding-right: 0.6rem;
  }
}

@media (max-width: 560px) {
  .home-rail {
    grid-template-columns: 1fr;
  }

  .home-rail-stage + .home-rail-stage {
    border-left: 0;
    border-top: 0;
    padding-top: 0;
  }

  .home-rail-link,
  .home-rail-stage:first-child .home-rail-link {
    padding-left: 0;
    padding-right: 0;
  }

  .home-family {
    grid-template-columns: 2rem minmax(0, 1fr) auto;
    gap: 0.35rem 0.6rem;
  }
  .home-family-id {
    grid-column: 1;
    grid-row: 1;
  }
  .home-family-label {
    grid-column: 2 / -1;
    grid-row: 1;
  }
  .home-family-metrics {
    grid-column: 2;
    grid-row: 2;
  }
  .home-family-action {
    grid-column: 3;
    grid-row: 2;
    justify-self: end;
  }

  .latest-research {
    grid-template-columns: minmax(0, 1fr) auto;
  }

  .latest-research-label {
    grid-column: 1 / -1;
  }
}

/* The overview and readouts span the grid; the daily work stays compact below. */
.home-main,
.home-feed {
  grid-row: auto;
}
.home-feed {
  align-content: start;
}
.home-panel,
.home-feed-panel {
  border-radius: 0.9rem;
  box-shadow: 0 2px 8px color-mix(in srgb, var(--color-ink) 3%, transparent);
}
.home-rail {
  gap: 0;
}
.home-rail-stage {
  position: relative;
  border: 0;
}
.home-rail-stage + .home-rail-stage {
  border-left: 0;
}
.home-rail-stage:not(:last-child)::after {
  position: absolute;
  top: 1.15rem;
  left: 3.1rem;
  right: 0.65rem;
  height: 1px;
  background: var(--color-line);
  content: '';
}
.home-rail-link {
  position: relative;
  gap: 0.35rem;
  padding: 0 0.65rem 0.5rem 0;
}
.home-rail-stage:first-child .home-rail-link,
.home-rail-stage:last-child .home-rail-link {
  padding: 0 0.65rem 0.5rem 0;
}
.home-rail-node {
  position: relative;
  z-index: 1;
  width: 2.3rem;
  height: 2.3rem;
  border-radius: 0.7rem;
}
.home-rail-node svg {
  width: 1.1rem;
  height: 1.1rem;
  stroke-width: 1.5;
}
.home-rail-index {
  position: absolute;
  right: 0.7rem;
  top: 0.45rem;
  padding-left: 0.25rem;
  background: var(--surface-strong);
  color: var(--color-muted);
  font-family: var(--font-mono);
  font-size: 0.6rem;
}
.home-rail-link h3 {
  margin-top: 0.6rem;
}
.home-rail-link p {
  max-width: 11rem;
  font-size: 0.7rem;
}
.latest-research {
  border: 0;
  border-radius: 0.6rem;
  background: color-mix(in srgb, var(--color-accent) 4%, var(--surface-strong));
}
.home-feed-head .eyebrow {
  letter-spacing: 0.08em;
  font-size: 0.68rem;
}
.home-panel-kicker {
  font-size: 0.6rem;
  letter-spacing: 0.1em;
}
@media (max-width: 1099px) {
  .home-dashboard {
    grid-template-columns: minmax(0, 1fr);
  }
  .home-feed {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .home-overview-copy {
    padding-left: 1.65rem;
  }
}
@media (max-width: 700px) {
  .home-overview {
    grid-template-columns: 1fr;
  }
  .home-overview-copy {
    padding: 1.65rem 1.5rem 0;
  }
  .home-overview-art {
    max-width: 27rem;
    width: 100%;
    justify-self: center;
    padding: 0 1rem 0.6rem;
  }
  .home-art-caption {
    bottom: 0.8rem;
    right: 1.5rem;
    font-size: 0.55rem;
  }
  .home-readout {
    grid-template-columns: 1fr;
    gap: 0.75rem;
  }
  .home-stat {
    display: grid;
    grid-template-columns: 1fr 1fr;
    align-items: center;
    padding: 1rem 1.1rem;
  }
  .home-stat-main {
    grid-row: 1 / 3;
    grid-column: 2;
    margin: 0;
  }
  .home-stat-foot {
    justify-content: start;
  }
  .home-stat-number {
    font-size: 2.1rem;
  }
  .home-model-ratio {
    display: none;
  }
  .home-feed {
    grid-template-columns: 1fr;
  }
  .home-rail-stage:nth-child(2)::after {
    display: none;
  }
}
@media (max-width: 900px) {
  .home-rail {
    row-gap: 1.2rem;
  }
  .home-rail-stage:nth-child(2)::after {
    display: none;
  }
}
@media (max-width: 560px) {
  .home-dashboard {
    gap: 1rem;
  }
  .home-overview h1 {
    font-size: 2rem;
  }
  .home-overview-lead {
    font-size: 0.82rem;
  }
  .home-overview-actions {
    gap: 0.8rem;
  }
  .home-rail {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  .home-rail-link p {
    font-size: 0.68rem;
  }
}
</style>
