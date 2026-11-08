<script setup lang="ts">
import { useQuery } from '@tanstack/vue-query'
import { computed, onBeforeUnmount, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { api } from '../api/client'
import { useAuthStore } from '../stores/auth'
import { localizedApiError } from '../errorLocalization'
import { onQuotaUsageChanged } from '../accountQuota'

const auth = useAuthStore()
const { t, te } = useI18n()
const query = useQuery({
  queryKey: computed(() => ['own-resources', auth.user?.id]),
  queryFn: api.ownResources,
  enabled: computed(() => Boolean(auth.user)),
  refetchInterval: 60000,
})
const data = computed(() => query.data.value)
const exempt = computed(() => data.value?.quota.enforcement === 'exempt_admin')
let unsubscribe: (() => void) | undefined
onMounted(() => {
  unsubscribe = onQuotaUsageChanged(() => {
    void query.refetch()
  })
})
onBeforeUnmount(() => unsubscribe?.())
</script>

<template>
  <section
    v-if="auth.user"
    class="admin-settings-panel resource-model-quota"
    :aria-label="t('resource.modelQuota')"
  >
    <header
      ><h2>{{ t('resource.modelQuota') }}</h2
      ><button
        class="secondary-action"
        :disabled="query.isFetching.value"
        @click="query.refetch()"
        >{{ t('resource.refresh') }}</button
      ></header
    >
    <div class="resource-model-quota-grid" aria-live="polite">
      <div
        ><span>{{ t('resource.retainedModels') }}</span
        ><strong
          >{{ data?.models?.retained ?? '—' }}
          <small>
            / {{ exempt ? t('resource.unlimited') : (data?.quota.max_model_count ?? '—') }}</small
          ></strong
        ><p>{{
          exempt
            ? t('resource.unlimited')
            : data?.models
              ? t('resource.modelRemaining', { count: data.models.remaining ?? 0 })
              : '—'
        }}</p
        ><small v-if="data?.models?.reserved">{{
          t('resource.modelReserved', { count: data.models.reserved })
        }}</small></div
      >
      <div
        ><span>{{ t('resource.protectedModels') }}</span
        ><strong
          >{{ data?.models?.protected ?? '—' }}
          <small>
            /
            {{
              exempt ? t('resource.unlimited') : (data?.quota.protected_model_limit ?? '—')
            }}</small
          ></strong
        ><p>{{
          exempt
            ? t('resource.unlimited')
            : data?.models
              ? t('resource.protectionRemaining', { count: data.models.protect_remaining ?? 0 })
              : '—'
        }}</p></div
      >
    </div>
    <p class="resource-help">{{
      t(exempt ? 'resource.adminExempt' : 'resource.modelQuotaHint')
    }}</p>
    <p v-if="!exempt && data?.models?.over_limit" class="resource-policy-note" role="note">{{
      t('resource.modelOverLimit')
    }}</p>
    <p
      v-if="!exempt && data?.models?.protection_over_limit"
      class="resource-policy-note"
      role="note"
      >{{ t('resource.protectionOverLimit') }}</p
    >
    <p v-if="query.error.value" class="error-copy">{{
      localizedApiError(query.error.value, t, te, 'resource.error')
    }}</p>
  </section>
</template>
