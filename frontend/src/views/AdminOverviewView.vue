<script setup lang="ts">
import { useQuery } from '@tanstack/vue-query'
import {
  Activity,
  ArrowRight,
  FlaskConical,
  Gauge,
  MailWarning,
  RefreshCw,
  ShieldAlert,
  UserCheck,
  Users,
  UserX,
} from '@lucide/vue'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { api } from '../api/client'
import { localizedApiError } from '../errorLocalization'
import { formatLocalizedDate } from '../localization'

const { locale, t, te } = useI18n()
const overviewQuery = useQuery({ queryKey: ['admin-overview'], queryFn: api.adminOverview })
const recentAuditQuery = useQuery({
  queryKey: ['admin-audit-logs', 'overview'],
  queryFn: () => api.adminAuditLogs({ limit: 6, offset: 0 }),
})

const overview = computed(() => overviewQuery.data.value)
const userMetrics = computed(() => [
  {
    key: 'total',
    label: t('admin.totalUsers'),
    value: overview.value?.users.total,
    icon: Users,
    tone: 'cool',
    to: '/admin/users',
  },
  {
    key: 'active',
    label: t('admin.activeUsers'),
    value: overview.value?.users.active,
    icon: UserCheck,
    tone: 'moss',
    to: '/admin/users?status=active',
  },
  {
    key: 'disabled',
    label: t('admin.disabledUsers'),
    value: overview.value?.users.disabled,
    icon: UserX,
    tone: 'danger',
    to: '/admin/users?status=disabled',
  },
  {
    key: 'unverified',
    label: t('admin.unverifiedUsers'),
    value: overview.value?.users.unverified,
    icon: MailWarning,
    tone: 'brass',
    to: '/admin/users?verification=unverified',
  },
  {
    key: 'new',
    label: t('admin.newUsersSevenDays'),
    value: overview.value?.users.new_last_7_days,
    icon: Activity,
    tone: 'accent',
    to: '/admin/users?sort=created_at&order=desc',
  },
])
const workloadMetrics = computed(() => [
  {
    label: t('admin.experimentsCreatedToday'),
    value: overview.value?.workload.experiments_created_today,
    icon: FlaskConical,
    tone: 'accent',
  },
  {
    label: t('admin.activeTraining'),
    value: overview.value?.workload.active_experiments,
    icon: Activity,
    tone: 'moss',
  },
  {
    label: t('admin.inferenceRequestsToday'),
    value: overview.value?.workload.inference_requests_today,
    icon: Gauge,
    tone: 'cool',
  },
  {
    label: t('admin.inferenceInputsToday'),
    value: overview.value?.workload.inference_inputs_today,
    icon: Gauge,
    tone: 'brass',
  },
])
const securityMetrics = computed(() => [
  {
    label: t('admin.failedLogins24h'),
    value: overview.value?.security_24h.failed_logins,
    to: '/admin/audit?event=auth.login_failed&range=24h',
  },
  {
    label: t('admin.adminActions24h'),
    value: overview.value?.security_24h.admin_actions,
    to: '/admin/audit?q=admin.&range=24h',
  },
  {
    label: t('admin.sessionRevokes24h'),
    value: overview.value?.security_24h.session_revoke_events,
    to: '/admin/audit?event=admin.sessions_revoked&range=24h',
  },
  {
    label: t('admin.userDeletions24h'),
    value: overview.value?.security_24h.user_deletions,
    to: '/admin/audit?event=admin.user_deleted&range=24h',
  },
])
const recentLogs = computed(() => recentAuditQuery.data.value?.items ?? [])

function refresh() {
  void Promise.all([overviewQuery.refetch(), recentAuditQuery.refetch()])
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
    t('admin.notAvailable'),
  )
}

function eventLabel(value: string) {
  const key = `admin.eventLabels.${value.replaceAll('.', '_')}`
  return te(key) ? t(key) : value
}

function actorTarget(log: (typeof recentLogs.value)[number]) {
  const actor = log.actor?.email || t('admin.systemActor')
  const target = log.target?.email || String(log.metadata?.target_email || t('admin.noTarget'))
  return t('admin.auditActorTarget', { actor, target })
}

function errorMessage(error: unknown) {
  return localizedApiError(error, t, te, 'admin.genericError')
}
</script>

<template>
  <section class="admin-page admin-overview-page">
    <header class="admin-page-header">
      <div
        ><p>{{ t('admin.eyebrow') }}</p
        ><h1>{{ t('admin.overviewTitle') }}</h1
        ><span>{{ t('admin.overviewSubtitle') }}</span></div
      >
      <button
        type="button"
        class="icon-action"
        :title="t('admin.refresh')"
        :aria-label="t('admin.refresh')"
        @click="refresh"
      >
        <RefreshCw
          :class="{
            'animate-spin': overviewQuery.isFetching.value || recentAuditQuery.isFetching.value,
          }"
          aria-hidden="true"
        />
      </button>
    </header>

    <div v-if="overviewQuery.isLoading.value" class="admin-overview-skeleton" aria-live="polite"
      ><span v-for="index in 9" :key="index"
    /></div>
    <template v-else-if="overview">
      <section class="admin-overview-section">
        <header
          ><div
            ><Users class="admin-icon-tone-cool" aria-hidden="true" /><h2>{{
              t('admin.accountHealth')
            }}</h2></div
          ><span>{{ t('admin.globalData') }}</span></header
        >
        <div class="admin-metric-strip admin-metric-strip-five">
          <RouterLink v-for="metric in userMetrics" :key="metric.key" :to="metric.to">
            <component
              :is="metric.icon"
              :class="`admin-icon-tone-${metric.tone}`"
              aria-hidden="true"
            />
            <span>{{ metric.label }}</span
            ><strong>{{ metric.value ?? 0 }}</strong
            ><ArrowRight class="admin-link-arrow" aria-hidden="true" />
          </RouterLink>
        </div>
      </section>

      <section class="admin-overview-section">
        <header
          ><div
            ><FlaskConical class="admin-icon-tone-brass" aria-hidden="true" /><h2>{{
              t('admin.todayWorkload')
            }}</h2></div
          ><span>{{ overview.usage_date }} UTC</span></header
        >
        <div class="admin-metric-strip">
          <div v-for="metric in workloadMetrics" :key="metric.label">
            <component
              :is="metric.icon"
              :class="`admin-icon-tone-${metric.tone}`"
              aria-hidden="true"
            /><span>{{ metric.label }}</span
            ><strong>{{ metric.value ?? 0 }}</strong>
          </div>
        </div>
      </section>

      <section class="admin-overview-grid">
        <div class="admin-overview-section">
          <header
            ><div
              ><ShieldAlert class="admin-icon-tone-danger" aria-hidden="true" /><h2>{{
                t('admin.securitySignals')
              }}</h2></div
            ><span>{{ t('admin.last24Hours') }}</span></header
          >
          <div class="admin-security-list">
            <RouterLink v-for="metric in securityMetrics" :key="metric.label" :to="metric.to">
              <span>{{ metric.label }}</span
              ><strong>{{ metric.value ?? 0 }}</strong
              ><ArrowRight class="admin-link-arrow" aria-hidden="true" />
            </RouterLink>
          </div>
        </div>

        <div class="admin-overview-section">
          <header
            ><div
              ><Activity class="admin-icon-tone-accent" aria-hidden="true" /><h2>{{
                t('admin.recentActivity')
              }}</h2></div
            ><RouterLink to="/admin/audit"
              >{{ t('admin.viewAll')
              }}<ArrowRight class="admin-link-arrow" aria-hidden="true" /></RouterLink
          ></header>
          <div v-if="recentLogs.length" class="admin-recent-list">
            <RouterLink
              v-for="log in recentLogs"
              :key="log.id"
              :to="`/admin/audit?q=${encodeURIComponent(log.event_type)}`"
            >
              <span
                ><strong>{{ eventLabel(log.event_type) }}</strong
                ><small>{{ actorTarget(log) }}</small></span
              ><time>{{ formatDate(log.created_at) }}</time>
            </RouterLink>
          </div>
          <div v-else class="admin-empty-state admin-empty-state-compact"
            ><strong>{{ t('admin.noAuditLogs') }}</strong></div
          >
        </div>
      </section>
    </template>

    <p v-if="overviewQuery.error.value" class="error-copy admin-inline-feedback">{{
      errorMessage(overviewQuery.error.value)
    }}</p>
  </section>
</template>
