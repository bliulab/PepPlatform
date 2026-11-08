<script setup lang="ts">
import { useQuery } from '@tanstack/vue-query'
import { ChevronLeft, ChevronRight, Search, X } from '@lucide/vue'
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import {
  adminAuditRange,
  adminPageSize,
  auditRangeStart,
  positiveInteger,
  type AdminAuditRange,
} from '../admin'
import { api } from '../api/client'
import { localizedApiError } from '../errorLocalization'
import { formatLocalizedDate } from '../localization'
import type { AdminAuditLog } from '../types'

const props = withDefaults(
  defineProps<{
    userId?: string
  }>(),
  {
    userId: '',
  },
)

const { locale, t, te } = useI18n()
const route = useRoute()
const router = useRouter()
const detailDialog = ref<HTMLDialogElement | null>(null)
const detailLog = ref<AdminAuditLog | null>(null)
const search = ref(String(route.query.q || ''))
const appliedSearch = ref(search.value.trim())
const eventType = ref(String(route.query.event || ''))
const range = ref<AdminAuditRange>(adminAuditRange(route.query.range))
const pageSize = ref(adminPageSize(route.query.per_page))
const page = ref(positiveInteger(route.query.page, 1))
let searchTimer: ReturnType<typeof window.setTimeout> | undefined

const eventTypesQuery = useQuery({
  queryKey: ['admin-audit-event-types'],
  queryFn: api.adminAuditEventTypes,
})
const auditParams = computed(() => ({
  search: appliedSearch.value || undefined,
  event_type: eventType.value || undefined,
  user_id: props.userId || undefined,
  created_from: auditRangeStart(range.value),
  limit: pageSize.value,
  offset: (page.value - 1) * pageSize.value,
}))
const auditQuery = useQuery({
  queryKey: computed(() => ['admin-audit-logs', auditParams.value]),
  queryFn: () => api.adminAuditLogs(auditParams.value),
})

const logs = computed(() => auditQuery.data.value?.items ?? [])
const total = computed(() => auditQuery.data.value?.total ?? 0)
const totalPages = computed(() => Math.max(1, Math.ceil(total.value / pageSize.value)))
const eventTypes = computed(() => eventTypesQuery.data.value?.items ?? [])
const ranges: AdminAuditRange[] = ['24h', '7d', '30d', 'all']

watch(search, (value) => {
  if (searchTimer) window.clearTimeout(searchTimer)
  searchTimer = window.setTimeout(() => {
    appliedSearch.value = value.trim()
    page.value = 1
  }, 300)
})

watch([eventType, range, pageSize], () => {
  page.value = 1
})
watch(totalPages, (value) => {
  if (page.value > value) page.value = value
})
watch([appliedSearch, eventType, range, pageSize, page], () => {
  const query = { ...route.query }
  const values: Record<string, string> = {
    q: appliedSearch.value,
    event: eventType.value,
    range: range.value,
    page: String(page.value),
    per_page: String(pageSize.value),
  }
  for (const [key, value] of Object.entries(values)) {
    if (
      value &&
      !(key === 'range' && value === '24h') &&
      !(key === 'page' && value === '1') &&
      !(key === 'per_page' && value === '25')
    ) {
      query[key] = value
    } else {
      delete query[key]
    }
  }
  void router.replace({ query })
})

function formatDate(value?: string | null) {
  return formatLocalizedDate(
    value,
    locale.value,
    {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    },
    t('admin.notAvailable'),
  )
}

function eventLabel(value: string) {
  const key = `admin.eventLabels.${value.replaceAll('.', '_')}`
  return te(key) ? t(key) : value
}

function actorLabel(log: AdminAuditLog) {
  return log.actor?.email || t('admin.systemActor')
}

function targetLabel(log: AdminAuditLog) {
  return (
    log.target?.email ||
    String(log.metadata?.target_email || log.target_user_id || t('admin.noTarget'))
  )
}

function detailEntries(log: AdminAuditLog) {
  return Object.entries(log.metadata || {}).map(([key, value]) => ({
    key,
    value: typeof value === 'string' ? value : JSON.stringify(value, null, 2),
  }))
}

async function openDetail(log: AdminAuditLog) {
  detailLog.value = log
  await nextTick()
  detailDialog.value?.showModal()
}

function closeDetail() {
  detailDialog.value?.close()
  detailLog.value = null
}

function errorMessage(error: unknown) {
  return localizedApiError(error, t, te, 'admin.genericError')
}

onBeforeUnmount(() => {
  if (searchTimer) window.clearTimeout(searchTimer)
})
</script>

<template>
  <section class="admin-audit-workbench">
    <div class="admin-filter-bar admin-audit-filters">
      <label class="admin-search-control">
        <Search aria-hidden="true" />
        <span class="sr-only">{{ t('admin.auditSearch') }}</span>
        <input v-model="search" type="search" :placeholder="t('admin.auditSearchPlaceholder')" />
      </label>

      <label class="admin-compact-select">
        <span>{{ t('admin.eventType') }}</span>
        <select v-model="eventType">
          <option value="">{{ t('admin.allEvents') }}</option>
          <option v-for="item in eventTypes" :key="item.event_type" :value="item.event_type">
            {{ eventLabel(item.event_type) }} ({{ item.count }})
          </option>
        </select>
      </label>

      <div class="admin-segmented" :aria-label="t('admin.auditRange')">
        <button
          v-for="option in ranges"
          :key="option"
          type="button"
          :aria-pressed="range === option"
          @click="range = option"
        >
          {{ t(`admin.auditRanges.${option}`) }}
        </button>
      </div>
    </div>

    <div v-if="auditQuery.isLoading.value" class="admin-loading-list" aria-live="polite">
      <span v-for="index in 6" :key="index" />
    </div>

    <div v-else-if="logs.length" class="admin-data-table-wrap">
      <table class="admin-data-table admin-audit-table">
        <thead>
          <tr>
            <th>{{ t('admin.time') }}</th>
            <th>{{ t('admin.eventType') }}</th>
            <th>{{ t('admin.actor') }}</th>
            <th>{{ t('admin.target') }}</th>
            <th>{{ t('admin.origin') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="log in logs" :key="log.id">
            <td class="admin-mono-cell" :data-label="t('admin.time')">{{
              formatDate(log.created_at)
            }}</td>
            <td :data-label="t('admin.eventType')"
              ><button type="button" class="admin-table-link" @click="openDetail(log)">{{
                eventLabel(log.event_type)
              }}</button></td
            >
            <td :data-label="t('admin.actor')">{{ actorLabel(log) }}</td>
            <td :data-label="t('admin.target')">{{ targetLabel(log) }}</td>
            <td class="admin-mono-cell" :data-label="t('admin.origin')">{{
              log.ip_address || t('admin.notAvailable')
            }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-else class="admin-empty-state">
      <strong>{{ t('admin.noAuditLogs') }}</strong>
      <span>{{ t('admin.noAuditLogsHint') }}</span>
    </div>

    <p v-if="auditQuery.error.value" class="error-copy admin-inline-feedback">{{
      errorMessage(auditQuery.error.value)
    }}</p>

    <footer class="admin-pagination">
      <span>{{ t('admin.resultCount', { total }) }}</span>
      <label>
        <span>{{ t('admin.pageSize') }}</span>
        <select v-model.number="pageSize">
          <option :value="25">25</option
          ><option :value="50">50</option
          ><option :value="100">100</option>
        </select>
      </label>
      <button
        type="button"
        class="icon-action"
        :disabled="page <= 1"
        :title="t('admin.previousPage')"
        :aria-label="t('admin.previousPage')"
        @click="page -= 1"
      >
        <ChevronLeft aria-hidden="true" />
      </button>
      <strong>{{ page }} / {{ totalPages }}</strong>
      <button
        type="button"
        class="icon-action"
        :disabled="page >= totalPages"
        :title="t('admin.nextPage')"
        :aria-label="t('admin.nextPage')"
        @click="page += 1"
      >
        <ChevronRight aria-hidden="true" />
      </button>
    </footer>
  </section>

  <dialog ref="detailDialog" class="admin-audit-dialog" @cancel.prevent="closeDetail">
    <section v-if="detailLog">
      <header>
        <div
          ><span>{{ t('admin.auditRecord') }}</span
          ><h2>{{ eventLabel(detailLog.event_type) }}</h2></div
        >
        <button
          type="button"
          class="icon-action"
          :title="t('common.close')"
          :aria-label="t('common.close')"
          @click="closeDetail"
          ><X aria-hidden="true"
        /></button>
      </header>
      <dl class="admin-audit-detail-grid">
        <div
          ><dt>{{ t('admin.time') }}</dt
          ><dd>{{ formatDate(detailLog.created_at) }}</dd></div
        >
        <div
          ><dt>{{ t('admin.actor') }}</dt
          ><dd>{{ actorLabel(detailLog) }}</dd></div
        >
        <div
          ><dt>{{ t('admin.target') }}</dt
          ><dd>{{ targetLabel(detailLog) }}</dd></div
        >
        <div
          ><dt>{{ t('admin.origin') }}</dt
          ><dd>{{ detailLog.ip_address || t('admin.notAvailable') }}</dd></div
        >
        <div class="admin-audit-detail-wide"
          ><dt>User-Agent</dt><dd>{{ detailLog.user_agent || t('admin.notAvailable') }}</dd></div
        >
      </dl>
      <div v-if="detailEntries(detailLog).length" class="admin-audit-metadata-list">
        <div v-for="entry in detailEntries(detailLog)" :key="entry.key"
          ><strong>{{ entry.key }}</strong
          ><pre>{{ entry.value }}</pre>
        </div>
      </div>
    </section>
  </dialog>
</template>
