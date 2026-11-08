<script setup lang="ts">
import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import { Download, ShieldCheck } from '@lucide/vue'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { api } from '../api/client'
import { localizedApiError } from '../errorLocalization'
import type { ModelRecord } from '../types'
import { useAuthStore } from '../stores/auth'

const props = defineProps<{
  modelId: string
  retention?: ModelRecord['retention']
  compact?: boolean
}>()
const { t, te } = useI18n()
const client = useQueryClient()
const auth = useAuthStore()
const query = useQuery({
  queryKey: computed(() => ['model-protection', props.modelId]),
  queryFn: () => api.modelProtection(props.modelId),
  enabled: computed(() => !props.retention),
})
const pinned = computed(() => props.retention?.pinned ?? query.data.value?.pinned ?? false)
const available = computed(
  () => !['removed', 'cleaning', 'failed'].includes(props.retention?.state ?? 'active'),
)
const mutation = useMutation({
  mutationFn: () => api.protectModel(props.modelId, !pinned.value),
  onSuccess: async () => {
    await Promise.all([
      client.invalidateQueries({ queryKey: ['model-protection', props.modelId] }),
      client.invalidateQueries({ queryKey: ['models'] }),
      client.invalidateQueries({ queryKey: ['own-resources'] }),
      auth.refreshUsage(),
    ])
  },
})
</script>

<template>
  <section class="resource-model-actions" :class="{ 'resource-model-actions--compact': compact }">
    <div class="resource-actions"
      ><a
        v-if="!compact && available"
        :href="`/api/models/${encodeURIComponent(modelId)}/download`"
        class="secondary-action"
        ><Download aria-hidden="true" />{{ t('resource.download') }}</a
      ><button
        class="secondary-action"
        :disabled="(!retention && !query.data.value) || mutation.isPending.value || !available"
        :aria-pressed="pinned"
        @click.stop="mutation.mutate()"
        ><ShieldCheck aria-hidden="true" />{{
          t(pinned ? 'resource.unpin' : 'resource.pin')
        }}</button
      ><span v-if="pinned" class="resource-state resource-state-protected">{{
        t('resource.protected')
      }}</span></div
    >
    <p v-if="!compact" class="resource-help">{{ t('resource.modelProtectionHint') }}</p>
    <p v-if="retention?.delete_after" class="resource-help">{{
      retention.notified_at
        ? t('resource.deleteAfter') + ' ' + new Date(retention.delete_after).toLocaleString()
        : t('resource.immediateCleanup')
    }}</p>
    <p v-if="query.error.value || mutation.error.value" class="error-copy">{{
      localizedApiError(query.error.value || mutation.error.value, t, te, 'resource.error')
    }}</p>
  </section>
</template>
