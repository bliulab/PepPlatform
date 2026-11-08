<script setup lang="ts">
import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import { HardDrive, RefreshCw } from '@lucide/vue'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { api } from '../api/client'
import { localizedApiError } from '../errorLocalization'
import { formatResourceBytes } from '../resourcePolicy'
import { formatLocalizedDate } from '../localization'
import { useAuthStore } from '../stores/auth'
import type { ResourceEntry } from '../resourceTypes'

const { t, te, locale } = useI18n()
const client = useQueryClient()
const auth = useAuthStore()
const query = useQuery({
  queryKey: ['admin-resources'],
  queryFn: api.adminResources,
  refetchInterval: 60000,
})
const showProtected = ref(true)
const publishAnnouncements = ref(false)
const search = ref('')
const modelPreview = useQuery({
  queryKey: ['model-cleanup-preview'],
  queryFn: api.modelCleanupPreview,
  enabled: false,
})
const cleanupModel = useMutation({
  mutationFn: async (id: string) => {
    const result = await api.cleanupModel(id)
    if (!result.deleted) throw new Error(result.error || t('resource.error'))
    return result
  },
  onSuccess: () =>
    Promise.all([
      modelPreview.refetch(),
      client.invalidateQueries({ queryKey: ['admin-resources'] }),
      client.invalidateQueries({ queryKey: ['models'] }),
      client.invalidateQueries({ queryKey: ['own-resources'] }),
      auth.refreshUsage(),
    ]),
})
const rows = computed(() =>
  (query.data.value?.items ?? [])
    .filter(
      (row) =>
        (showProtected.value || !row.protected) &&
        row.name.toLowerCase().includes(search.value.toLowerCase()),
    )
    .sort((a, b) => b.size_bytes - a.size_bytes),
)
const mutation = useMutation({
  mutationFn: async (action: string) => {
    if (action === 'scan') return api.scanResources()
    if (action === 'retention')
      return api.checkRetention({ publish_announcements: publishAnnouncements.value })
    const result = await api.cleanupResource(action)
    if (!result.deleted) throw new Error(result.error || t('resource.error'))
    return api.adminResources()
  },
  onSuccess: (result, action) => {
    if (action === 'retention') publishAnnouncements.value = false
    client.setQueryData(['admin-resources'], result)
    void client.invalidateQueries({ queryKey: ['announcements'] })
    void client.invalidateQueries({ queryKey: ['admin-announcements'] })
    void client.invalidateQueries({ queryKey: ['admin-audit-logs'] })
    if (action === 'retention') {
      void client.invalidateQueries({ queryKey: ['models'] })
      void client.invalidateQueries({ queryKey: ['own-resources'] })
      void client.invalidateQueries({ queryKey: ['admin-users'] })
      void client.invalidateQueries({ queryKey: ['admin-user'] })
      void auth.refreshUsage()
    }
    if (modelPreview.data.value) void modelPreview.refetch()
  },
})
function due(row: ResourceEntry) {
  return (
    !row.protected &&
    Boolean(
      row.scheduled_at &&
      (!row.notified_at || (row.delete_after && Date.parse(row.delete_after) <= Date.now())),
    )
  )
}
function date(value: string | null) {
  return value
    ? formatLocalizedDate(value, locale.value, { dateStyle: 'medium', timeStyle: 'short' })
    : '—'
}
</script>

<template>
  <section class="admin-page resource-page">
    <header class="admin-page-header"
      ><div
        ><p>{{ t('resource.storage') }}</p
        ><h1>{{ t('resource.storage') }}</h1
        ><span>{{ t('resource.storageSubtitle') }}</span></div
      ><HardDrive class="resource-heading-icon" aria-hidden="true"
    /></header>
    <template v-if="query.data.value">
      <div class="resource-metric-grid">
        <div class="resource-metric"
          ><span>{{ t('resource.free') }}</span
          ><strong>{{ formatResourceBytes(query.data.value.disk.free_bytes) }}</strong
          ><small :class="query.data.value.disk.state !== 'ready' ? 'error-copy' : ''">{{
            t(`resource.${query.data.value.disk.state}`)
          }}</small></div
        >
        <div class="resource-metric"
          ><span>{{ t('resource.retained') }}</span
          ><strong>{{ formatResourceBytes(query.data.value.storage.protected_bytes) }}</strong
          ><small>{{ t('resource.modelRetained') }}</small></div
        >
        <div class="resource-metric"
          ><span>{{ t('resource.reclaimable') }}</span
          ><strong>{{ formatResourceBytes(query.data.value.storage.reclaimable_bytes) }}</strong
          ><small
            >{{ t('resource.usedOf', { used: query.data.value.workload.running, limit: 1 }) }}
            {{ t('resource.running') }} · {{ query.data.value.workload.queued }}
            {{ t('resource.queued') }}</small
          ></div
        >
        <div class="resource-metric"
          ><span>{{ t('resource.registrations') }}</span
          ><strong
            >{{ query.data.value.registration.used }} /
            {{ query.data.value.registration.limit }}</strong
          ><small>UTC</small></div
        >
      </div>
      <div v-if="query.data.value.gpus?.length" class="resource-gpu-strip">
        <div v-for="gpu in query.data.value.gpus" :key="gpu.index" class="resource-log"
          ><strong>{{ gpu.name }}</strong
          ><span
            >{{ t('resource.gpuMemory') }} {{ formatResourceBytes(gpu.used_bytes) }} /
            {{ formatResourceBytes(gpu.total_bytes) }}</span
          ><small v-if="gpu.utilization_percent !== null"
            >{{ t('resource.gpuUtilization') }} {{ gpu.utilization_percent }}%</small
          ></div
        >
      </div>
      <section class="admin-settings-panel">
        <header
          ><h2>{{ t('resource.modelCleanup') }}</h2
          ><button
            class="secondary-action"
            :disabled="modelPreview.isFetching.value"
            @click="modelPreview.refetch()"
            >{{ t('resource.preview') }}</button
          ></header
        >
        <p class="resource-help">{{ t('resource.modelCleanupNotice') }}</p>
        <template v-if="modelPreview.data.value">
          <p v-if="!modelPreview.data.value.enabled" class="resource-help">{{
            t('resource.modelCleanupOff')
          }}</p>
          <div class="resource-table-wrap"
            ><table class="resource-table"
              ><thead
                ><tr
                  ><th>{{ t('resource.name') }}</th
                  ><th>{{ t('resource.estimatedSize') }}</th
                  ><th>{{ t('resource.state') }}</th
                  ><th>{{ t('resource.deleteAfter') }}</th
                  ><th>{{ t('resource.cleanup') }}</th></tr
                ></thead
              ><tbody>
                <tr v-for="model in modelPreview.data.value.items" :key="model.model_id"
                  ><td
                    ><strong>{{ model.name }}</strong
                    ><small>{{ model.owner_email }}</small></td
                  ><td>{{ formatResourceBytes(model.estimated_bytes ?? 0) }}</td
                  ><td>{{
                    model.skip_reason
                      ? t(`resource.reason.${model.skip_reason}`)
                      : t(`resource.${model.state === 'active' ? 'available' : model.state}`)
                  }}</td
                  ><td>{{
                    model.scheduled_at && !model.notified_at
                      ? t('resource.immediateCleanup')
                      : date(model.delete_after)
                  }}</td
                  ><td
                    ><button
                      v-if="
                        model.eligible &&
                        (!model.notified_at ||
                          (model.delete_after && Date.parse(model.delete_after) <= Date.now()))
                      "
                      class="secondary-action"
                      :disabled="!modelPreview.data.value.enabled || cleanupModel.isPending.value"
                      @click="cleanupModel.mutate(model.model_id)"
                      >{{ t('resource.cleanup') }}</button
                    ></td
                  ></tr
                >
              </tbody></table
            ></div
          >
        </template>
        <p v-if="modelPreview.error.value || cleanupModel.error.value" class="error-copy">{{
          localizedApiError(
            modelPreview.error.value || cleanupModel.error.value,
            t,
            te,
            'resource.error',
          )
        }}</p>
      </section>
      <section class="admin-settings-panel">
        <div class="resource-toolbar"
          ><input
            v-model="search"
            :aria-label="t('resource.name')"
            :placeholder="t('resource.name')"
            type="search"
            class="field-control"
          /><label class="resource-check"
            ><input v-model="showProtected" type="checkbox" />{{ t('resource.protected') }}</label
          ><button
            class="secondary-action"
            :disabled="mutation.isPending.value"
            @click="mutation.mutate('scan')"
            ><RefreshCw aria-hidden="true" />{{ t('resource.scan') }}</button
          ><label class="resource-check">
            <input
              v-model="publishAnnouncements"
              type="checkbox"
              :disabled="mutation.isPending.value"
            />
            {{ t('resource.publishCleanupAnnouncements') }} </label
          ><button
            class="secondary-action"
            :disabled="mutation.isPending.value"
            @click="mutation.mutate('retention')"
            >{{ t('resource.checkRetention') }}</button
          ></div
        >
        <p class="resource-help">{{
          t(
            publishAnnouncements
              ? 'resource.manualCleanupNoticeHint'
              : 'resource.silentCleanupHint',
          )
        }}</p>
        <div class="resource-table-wrap"
          ><table class="resource-table"
            ><thead
              ><tr
                ><th>{{ t('resource.name') }}</th
                ><th>{{ t('resource.size') }}</th
                ><th>{{ t('resource.state') }}</th
                ><th>{{ t('resource.deleteAfter') }}</th
                ><th
                  ><span class="sr-only">{{ t('resource.cleanup') }}</span></th
                ></tr
              ></thead
            ><tbody
              ><tr v-for="row in rows" :key="row.id"
                ><td
                  ><strong :title="row.name">{{ row.name }}</strong
                  ><small v-if="row.owner_email">{{ row.owner_email }}</small
                  ><small>{{
                    te(`resource.reason.${row.reason}`)
                      ? t(`resource.reason.${row.reason}`)
                      : row.reason
                  }}</small></td
                ><td>{{ formatResourceBytes(row.size_bytes) }}</td
                ><td
                  ><span
                    class="resource-state"
                    :class="{ 'resource-state-protected': row.protected }"
                    >{{ t(`resource.${row.protected ? 'protected' : row.state}`) }}</span
                  ></td
                ><td>{{
                  row.scheduled_at && !row.notified_at
                    ? t('resource.immediateCleanup')
                    : date(row.delete_after)
                }}</td
                ><td
                  ><button
                    v-if="due(row)"
                    class="secondary-action"
                    :disabled="mutation.isPending.value"
                    @click="mutation.mutate(row.id)"
                    >{{ t('resource.cleanup') }}</button
                  ></td
                ></tr
              ></tbody
            ></table
          ><p v-if="!rows.length" class="resource-help">{{ t('resource.empty') }}</p></div
        >
      </section>
      <section v-if="query.data.value.cleanup_logs.length" class="admin-settings-panel"
        ><header
          ><h2>{{ t('resource.recentCleanup') }}</h2></header
        ><div v-for="log in query.data.value.cleanup_logs" :key="log.id" class="resource-log"
          ><span>{{ date(log.created_at) }} · {{ t(`resource.${log.status}`) }}</span
          ><strong
            >{{ t('resource.released') }} {{ formatResourceBytes(log.released_bytes) }}</strong
          ><small v-if="log.error" class="error-copy">{{ log.error }}</small></div
        ></section
      >
    </template>
    <p v-if="query.error.value || mutation.error.value" class="error-copy" role="alert">{{
      localizedApiError(query.error.value || mutation.error.value, t, te, 'resource.error')
    }}</p>
  </section>
</template>
