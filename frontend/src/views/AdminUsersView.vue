<script setup lang="ts">
import { useQuery } from '@tanstack/vue-query'
import {
  ArrowDown,
  ArrowRight,
  ArrowUp,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  Search,
  Users,
} from '@lucide/vue'
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import {
  adminPageSize,
  adminUserSort,
  positiveInteger,
  sortOrder,
  type AdminUserSort,
} from '../admin'
import { api } from '../api/client'
import { localizedApiError } from '../errorLocalization'
import { formatLocalizedDate } from '../localization'
import type { AdminUser } from '../types'

const { locale, t, te } = useI18n()
const route = useRoute()
const router = useRouter()
const search = ref(String(route.query.q || ''))
const appliedSearch = ref(search.value.trim())
const roleFilter = ref(String(route.query.role || ''))
const statusFilter = ref(String(route.query.status || ''))
const verificationFilter = ref(String(route.query.verification || ''))
const sort = ref<AdminUserSort>(adminUserSort(route.query.sort))
const order = ref(sortOrder(route.query.order))
const pageSize = ref(adminPageSize(route.query.per_page))
const page = ref(positiveInteger(route.query.page, 1))
let searchTimer: ReturnType<typeof window.setTimeout> | undefined

const userParams = computed(() => ({
  search: appliedSearch.value || undefined,
  role: roleFilter.value || undefined,
  status: statusFilter.value || undefined,
  email_verified:
    verificationFilter.value === 'verified'
      ? true
      : verificationFilter.value === 'unverified'
        ? false
        : undefined,
  sort: sort.value,
  order: order.value,
  limit: pageSize.value,
  offset: (page.value - 1) * pageSize.value,
}))
const usersQuery = useQuery({
  queryKey: computed(() => ['admin-users', userParams.value]),
  queryFn: () => api.adminUsers(userParams.value),
})
const users = computed(() => usersQuery.data.value?.items ?? [])
const total = computed(() => usersQuery.data.value?.total ?? 0)
const totalPages = computed(() => Math.max(1, Math.ceil(total.value / pageSize.value)))

watch(search, (value) => {
  if (searchTimer) window.clearTimeout(searchTimer)
  searchTimer = window.setTimeout(() => {
    appliedSearch.value = value.trim()
    page.value = 1
  }, 300)
})
watch([roleFilter, statusFilter, verificationFilter, sort, order, pageSize], () => {
  page.value = 1
})
watch(totalPages, (value) => {
  if (page.value > value) page.value = value
})
watch(
  [appliedSearch, roleFilter, statusFilter, verificationFilter, sort, order, pageSize, page],
  () => {
    const query = { ...route.query }
    const values: Record<string, string> = {
      q: appliedSearch.value,
      role: roleFilter.value,
      status: statusFilter.value,
      verification: verificationFilter.value,
      sort: sort.value,
      order: order.value,
      page: String(page.value),
      per_page: String(pageSize.value),
    }
    for (const [key, value] of Object.entries(values)) {
      const isDefault =
        (key === 'sort' && value === 'created_at') ||
        (key === 'order' && value === 'desc') ||
        (key === 'page' && value === '1') ||
        (key === 'per_page' && value === '25')
      if (value && !isDefault) query[key] = value
      else delete query[key]
    }
    void router.replace({ query })
  },
)

function toggleSort(nextSort: AdminUserSort) {
  if (sort.value === nextSort) order.value = order.value === 'asc' ? 'desc' : 'asc'
  else {
    sort.value = nextSort
    order.value = nextSort === 'created_at' ? 'desc' : 'asc'
  }
}

function openUser(user: AdminUser) {
  void router.push({
    path: `/admin/users/${user.id}/account`,
    query: { return_to: route.fullPath },
  })
}

function formatDate(value?: string | null) {
  return formatLocalizedDate(
    value,
    locale.value,
    { year: 'numeric', month: '2-digit', day: '2-digit' },
    t('admin.notAvailable'),
  )
}

function statusLabel(status: string) {
  return status === 'active'
    ? t('admin.statusActive')
    : status === 'disabled'
      ? t('admin.statusDisabled')
      : status
}

function roleLabel(role: string) {
  return role === 'admin' ? t('account.admin') : t('account.user')
}

function quotaUse(used: number, limit: number) {
  return t('admin.quotaUse', { used, limit })
}

function resourceSummary(user: AdminUser) {
  return t('admin.resourceSummary', {
    datasets: user.resource_counts.datasets,
    experiments: user.resource_counts.experiments,
    models: user.resource_counts.models,
  })
}

function errorMessage(error: unknown) {
  return localizedApiError(error, t, te, 'admin.genericError')
}

onBeforeUnmount(() => {
  if (searchTimer) window.clearTimeout(searchTimer)
})
</script>

<template>
  <section class="admin-page admin-users-page">
    <header class="admin-page-header">
      <div
        ><p>{{ t('admin.eyebrow') }}</p
        ><h1>{{ t('admin.usersTitle') }}</h1
        ><span>{{ t('admin.usersSubtitle') }}</span></div
      >
      <button
        type="button"
        class="icon-action"
        :title="t('admin.refresh')"
        :aria-label="t('admin.refresh')"
        @click="usersQuery.refetch()"
      >
        <RefreshCw :class="{ 'animate-spin': usersQuery.isFetching.value }" aria-hidden="true" />
      </button>
    </header>

    <div class="admin-filter-bar admin-user-filters">
      <label class="admin-search-control">
        <Search aria-hidden="true" /><span class="sr-only">{{ t('admin.search') }}</span>
        <input v-model="search" type="search" :placeholder="t('admin.searchPlaceholder')" />
      </label>
      <label class="admin-compact-select"
        ><span>{{ t('admin.role') }}</span
        ><select v-model="roleFilter"
          ><option value="">{{ t('admin.allRoles') }}</option
          ><option value="admin">{{ t('account.admin') }}</option
          ><option value="user">{{ t('account.user') }}</option></select
        ></label
      >
      <label class="admin-compact-select"
        ><span>{{ t('admin.status') }}</span
        ><select v-model="statusFilter"
          ><option value="">{{ t('admin.allStatuses') }}</option
          ><option value="active">{{ t('admin.statusActive') }}</option
          ><option value="disabled">{{ t('admin.statusDisabled') }}</option></select
        ></label
      >
      <label class="admin-compact-select"
        ><span>{{ t('admin.emailVerification') }}</span
        ><select v-model="verificationFilter"
          ><option value="">{{ t('admin.allVerification') }}</option
          ><option value="verified">{{ t('account.verified') }}</option
          ><option value="unverified">{{ t('account.unverified') }}</option></select
        ></label
      >
    </div>

    <div v-if="usersQuery.isLoading.value" class="admin-loading-list" aria-live="polite"
      ><span v-for="index in 8" :key="index"
    /></div>

    <div v-else-if="users.length" class="admin-data-table-wrap admin-users-table-wrap">
      <table class="admin-data-table admin-users-table">
        <thead
          ><tr>
            <th
              ><button type="button" @click="toggleSort('display_name')"
                >{{ t('admin.user')
                }}<ArrowUp
                  v-if="sort === 'display_name' && order === 'asc'"
                  aria-hidden="true" /><ArrowDown
                  v-else-if="sort === 'display_name'"
                  aria-hidden="true" /></button
            ></th>
            <th>{{ t('admin.access') }}</th
            ><th>{{ t('admin.status') }}</th
            ><th>{{ t('admin.todayUsage') }}</th
            ><th>{{ t('admin.resources') }}</th>
            <th
              ><button type="button" @click="toggleSort('created_at')"
                >{{ t('admin.createdAt')
                }}<ArrowUp
                  v-if="sort === 'created_at' && order === 'asc'"
                  aria-hidden="true" /><ArrowDown
                  v-else-if="sort === 'created_at'"
                  aria-hidden="true" /></button
            ></th> </tr
        ></thead>
        <tbody>
          <tr
            v-for="user in users"
            :key="user.id"
            tabindex="0"
            @click="openUser(user)"
            @keydown.enter="openUser(user)"
          >
            <td
              ><div class="admin-user-identity"
                ><span>{{ user.display_name.charAt(0).toUpperCase() }}</span
                ><div
                  ><strong>{{ user.display_name }}</strong
                  ><small>{{ user.email }}</small></div
                ></div
              ></td
            >
            <td
              ><strong>{{ roleLabel(user.role) }}</strong
              ><small :class="user.email_verified ? 'admin-text-success' : 'admin-text-warning'">{{
                user.email_verified ? t('account.verified') : t('account.unverified')
              }}</small></td
            >
            <td
              ><span
                class="status-pill"
                :class="user.status === 'active' ? 'status-pill-success' : 'status-pill-danger'"
                >{{ statusLabel(user.status) }}</span
              ></td
            >
            <td
              ><span
                >{{ quotaUse(user.usage.experiments_created, user.quota.daily_experiment_limit) }}
                {{ t('admin.experimentsShort') }}</span
              ><small
                >{{ quotaUse(user.usage.inference_inputs, user.quota.daily_inference_input_limit) }}
                {{ t('admin.inferenceInputsShort') }}</small
              ></td
            >
            <td
              ><span>{{ resourceSummary(user) }}</span></td
            >
            <td
              ><span>{{ formatDate(user.created_at) }}</span
              ><ArrowRight aria-hidden="true"
            /></td>
          </tr>
        </tbody>
      </table>

      <div class="admin-user-mobile-list">
        <button v-for="user in users" :key="user.id" type="button" @click="openUser(user)">
          <span class="admin-user-identity"
            ><span>{{ user.display_name.charAt(0).toUpperCase() }}</span
            ><span
              ><strong>{{ user.display_name }}</strong
              ><small>{{ user.email }}</small></span
            ></span
          >
          <span
            class="status-pill"
            :class="user.status === 'active' ? 'status-pill-success' : 'status-pill-danger'"
            >{{ statusLabel(user.status) }}</span
          >
          <span
            >{{ roleLabel(user.role) }} ·
            {{ user.email_verified ? t('account.verified') : t('account.unverified') }}</span
          >
          <small>{{ resourceSummary(user) }}</small
          ><ArrowRight aria-hidden="true" />
        </button>
      </div>
    </div>

    <div v-else class="admin-empty-state"
      ><Users class="admin-icon-tone-cool" aria-hidden="true" /><strong>{{
        t('admin.noUsers')
      }}</strong
      ><span>{{ t('admin.noUsersHint') }}</span></div
    >
    <p v-if="usersQuery.error.value" class="error-copy admin-inline-feedback">{{
      errorMessage(usersQuery.error.value)
    }}</p>

    <footer class="admin-pagination">
      <span>{{ t('admin.resultCount', { total }) }}</span>
      <label
        ><span>{{ t('admin.pageSize') }}</span
        ><select v-model.number="pageSize"
          ><option :value="25">25</option
          ><option :value="50">50</option
          ><option :value="100">100</option></select
        ></label
      >
      <button
        type="button"
        class="icon-action"
        :disabled="page <= 1"
        :title="t('admin.previousPage')"
        :aria-label="t('admin.previousPage')"
        @click="page -= 1"
        ><ChevronLeft aria-hidden="true"
      /></button>
      <strong>{{ page }} / {{ totalPages }}</strong>
      <button
        type="button"
        class="icon-action"
        :disabled="page >= totalPages"
        :title="t('admin.nextPage')"
        :aria-label="t('admin.nextPage')"
        @click="page += 1"
        ><ChevronRight aria-hidden="true"
      /></button>
    </footer>
  </section>
</template>
