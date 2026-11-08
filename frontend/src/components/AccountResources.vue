<script setup lang="ts">
import { useQuery } from '@tanstack/vue-query'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { api } from '../api/client'
import { useAuthStore } from '../stores/auth'
import { formatResourceBytes } from '../resourcePolicy'
import { localizedApiError } from '../errorLocalization'
import { pendingQuotaChanges } from '../resourcePolicy'
import QuotaChangeDetails from './QuotaChangeDetails.vue'

const { t, te } = useI18n()
const auth = useAuthStore()
const query = useQuery({
  queryKey: computed(() => ['own-resources', auth.user?.id]),
  queryFn: api.ownResources,
  enabled: computed(() => Boolean(auth.user)),
  refetchInterval: 60000,
})
const usage = computed(() => query.data.value)
</script>

<template>
  <section v-if="auth.user" class="admin-settings-panel resource-account-panel">
    <header
      ><h2>{{ t('resource.userStorage') }}</h2
      ><button
        class="secondary-action"
        :disabled="query.isFetching.value"
        @click="query.refetch()"
        >{{ t('resource.refresh') }}</button
      ></header
    >
    <template v-if="usage"
      ><dl class="resource-account-readouts"
        ><div
          ><dt>{{ t('resource.userStorage') }}</dt
          ><dd
            >{{ formatResourceBytes(usage.storage.used_bytes) }} /
            {{
              usage.quota.enforcement === 'exempt_admin'
                ? t('resource.unlimited')
                : formatResourceBytes(usage.quota.storage_limit_bytes ?? 0)
            }}</dd
          ></div
        ><div
          ><dt>{{ t('resource.trainingTime') }}</dt
          ><dd
            >{{ Math.round((usage.usage.training_seconds ?? 0) / 60) }} /
            {{
              usage.quota.enforcement === 'exempt_admin'
                ? t('resource.unlimited')
                : (usage.quota.daily_training_seconds_limit ?? 0) / 60
            }}
            {{ t('resource.minute') }}</dd
          ></div
        ></dl
      ><p v-if="usage.disk_state !== 'ready'" class="error-copy">{{
        t(`resource.${usage.disk_state}`)
      }}</p
      ><div
        v-for="row in usage.items.filter((item) => item.delete_after)"
        :key="row.id"
        class="resource-log"
        ><strong>{{ row.name }}</strong
        ><small>{{
          row.notified_at
            ? t('resource.deleteAfter') + ' ' + new Date(row.delete_after!).toLocaleString()
            : t('resource.immediateCleanup')
        }}</small></div
      ><QuotaChangeDetails :changes="pendingQuotaChanges(usage.quota)" pending
    /></template>
    <p v-if="query.error.value" class="error-copy">{{
      localizedApiError(query.error.value, t, te, 'resource.error')
    }}</p>
  </section>
</template>
