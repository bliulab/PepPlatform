<script setup lang="ts">
import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import { Bell } from '@lucide/vue'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { api } from '../api/client'
import { useAuthStore } from '../stores/auth'
import QuotaChangeDetails from './QuotaChangeDetails.vue'
import { localizedApiError } from '../errorLocalization'
import { formatLocalizedDate } from '../localization'

const { t, te, locale } = useI18n()
const auth = useAuthStore()
const client = useQueryClient()
const query = useQuery({
  queryKey: computed(() => ['announcements', auth.user?.id]),
  queryFn: api.announcements,
  enabled: computed(() => Boolean(auth.user)),
  refetchInterval: 60000,
})
const unread = computed(() => query.data.value?.items.filter((row) => !row.read).length ?? 0)
const read = useMutation({
  mutationFn: api.readAnnouncement,
  onSuccess: () => client.invalidateQueries({ queryKey: ['announcements'] }),
})
</script>

<template>
  <details v-if="query.data.value?.items.length" class="resource-notices">
    <summary
      ><Bell aria-hidden="true" /><strong>{{ t('resource.notices') }}</strong
      ><span v-if="unread" class="resource-state">{{ unread }}</span></summary
    >
    <div class="resource-notice-scroll">
      <article
        v-for="row in query.data.value.items"
        :key="row.id"
        class="resource-notice"
        :class="{ 'resource-notice-unread': !row.read }"
      >
        <header
          ><h2>{{ row.kind === 'quota_change' ? t('resource.quotaChangeTitle') : row.title }}</h2
          ><span v-if="row.pinned" class="resource-state">{{ t('resource.pinned') }}</span></header
        >
        <time>{{
          formatLocalizedDate(row.publish_at, locale, { dateStyle: 'medium', timeStyle: 'short' })
        }}</time>
        <QuotaChangeDetails v-if="row.quota_change" :changes="row.quota_change" />
        <p v-else class="resource-notice-body">{{ row.body }}</p>
        <button
          v-if="!row.read"
          class="secondary-action"
          :disabled="read.isPending.value"
          @click="read.mutate(row.id)"
          >{{ t('resource.markRead') }}</button
        ><span v-else class="resource-help">{{ t('resource.read') }}</span>
      </article>
    </div>
    <p v-if="read.error.value" class="error-copy">{{
      localizedApiError(read.error.value, t, te, 'resource.error')
    }}</p>
  </details>
</template>
