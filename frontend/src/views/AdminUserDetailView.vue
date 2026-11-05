<script setup lang="ts">
import { useQuery } from '@tanstack/vue-query'
import { ArrowLeft, Database, FlaskConical, Gauge, ShieldCheck, UserRound } from '@lucide/vue'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'
import { safeAdminUsersReturnPath } from '../admin'
import { api } from '../api/client'
import AdminAuditTable from '../components/AdminAuditTable.vue'
import AdminUserAccountPanel from '../components/AdminUserAccountPanel.vue'
import AdminUserSecurityPanel from '../components/AdminUserSecurityPanel.vue'
import AdminUserUsagePanel from '../components/AdminUserUsagePanel.vue'
import { localizedApiError } from '../errorLocalization'
import { formatLocalizedDate } from '../localization'

const { locale, t, te } = useI18n()
const route = useRoute()
const userId = computed(() => String(route.params.id || ''))
const activeTab = computed(() => String(route.meta.adminUserTab || 'account'))
const returnPath = computed(() => safeAdminUsersReturnPath(route.query.return_to))
const detailQuery = useQuery({
  queryKey: computed(() => ['admin-user', userId.value]),
  queryFn: () => api.adminUser(userId.value),
  enabled: computed(() => Boolean(userId.value)),
})
const user = computed(() => detailQuery.data.value)
const tabs = computed(() => [
  { id: 'account', label: t('admin.accountTab') },
  { id: 'usage', label: t('admin.usageTab') },
  { id: 'security', label: t('admin.securityTab') },
  { id: 'activity', label: t('admin.activityTab') },
])
const resources = computed(() =>
  user.value
    ? [
        {
          label: t('admin.datasetsShort'),
          value: user.value.resource_counts.datasets,
          icon: Database,
          tone: 'cool',
        },
        {
          label: t('admin.experimentsShort'),
          value: user.value.resource_counts.experiments,
          icon: FlaskConical,
          tone: 'accent',
        },
        {
          label: t('admin.activeTraining'),
          value: user.value.resource_counts.active_experiments,
          icon: Gauge,
          tone: 'moss',
        },
        {
          label: t('admin.modelsShort'),
          value: user.value.resource_counts.models,
          icon: ShieldCheck,
          tone: 'brass',
        },
        {
          label: t('admin.inferenceRunsShort'),
          value: user.value.resource_counts.inference_runs,
          icon: UserRound,
          tone: 'cool',
        },
      ]
    : [],
)

function tabTarget(id: string) {
  return { path: `/admin/users/${userId.value}/${id}`, query: { return_to: returnPath.value } }
}

function formatDate(value?: string | null) {
  return formatLocalizedDate(
    value,
    locale.value,
    { year: 'numeric', month: 'short', day: '2-digit' },
    t('admin.notAvailable'),
  )
}

function errorMessage(error: unknown) {
  return localizedApiError(error, t, te, 'admin.genericError')
}
</script>

<template>
  <section class="admin-page admin-user-detail-page">
    <RouterLink :to="returnPath" class="admin-back-link"
      ><ArrowLeft aria-hidden="true" />{{ t('admin.backToUsers') }}</RouterLink
    >

    <div v-if="detailQuery.isLoading.value" class="admin-user-detail-skeleton" aria-live="polite"
      ><span /><span /><span
    /></div>
    <template v-else-if="user">
      <header class="admin-user-header">
        <div class="admin-user-header-main">
          <span class="admin-user-avatar">{{ user.display_name.charAt(0).toUpperCase() }}</span>
          <div
            ><p
              >{{ user.role === 'admin' ? t('account.admin') : t('account.user') }} ·
              {{ t('admin.createdOn', { date: formatDate(user.created_at) }) }}</p
            ><h1>{{ user.display_name }}</h1
            ><span>{{ user.email }}</span></div
          >
        </div>
        <div class="admin-user-header-status"
          ><span
            class="status-pill"
            :class="user.status === 'active' ? 'status-pill-success' : 'status-pill-danger'"
            >{{
              user.status === 'active' ? t('admin.statusActive') : t('admin.statusDisabled')
            }}</span
          ><span
            class="status-pill"
            :class="user.email_verified ? 'status-pill-neutral' : 'status-pill-warning'"
            >{{ user.email_verified ? t('account.verified') : t('account.unverified') }}</span
          ></div
        >
      </header>

      <div class="admin-resource-strip">
        <div v-for="item in resources" :key="item.label"
          ><component
            :is="item.icon"
            :class="`admin-icon-tone-${item.tone}`"
            aria-hidden="true"
          /><span>{{ item.label }}</span
          ><strong>{{ item.value }}</strong></div
        >
      </div>

      <nav class="admin-detail-tabs" :aria-label="t('admin.userDetailTabs')">
        <RouterLink
          v-for="tab in tabs"
          :key="tab.id"
          :to="tabTarget(tab.id)"
          :class="{ active: activeTab === tab.id }"
          >{{ tab.label }}</RouterLink
        >
      </nav>

      <AdminUserAccountPanel v-if="activeTab === 'account'" :user="user" />
      <AdminUserUsagePanel v-else-if="activeTab === 'usage'" :user="user" />
      <AdminUserSecurityPanel
        v-else-if="activeTab === 'security'"
        :user="user"
        :return-path="returnPath"
      />
      <section v-else class="admin-settings-panel"
        ><header
          ><div
            ><h2>{{ t('admin.activityTitle') }}</h2
            ><p>{{ t('admin.activityHint') }}</p></div
          ></header
        ><AdminAuditTable :user-id="user.id"
      /></section>
    </template>

    <div v-else class="admin-empty-state"
      ><UserRound class="admin-icon-tone-cool" aria-hidden="true" /><strong>{{
        t('admin.userNotFound')
      }}</strong
      ><RouterLink :to="returnPath" class="secondary-action">{{
        t('admin.backToUsers')
      }}</RouterLink></div
    >
    <p v-if="detailQuery.error.value" class="error-copy admin-inline-feedback">{{
      errorMessage(detailQuery.error.value)
    }}</p>
  </section>
</template>
