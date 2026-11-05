<script setup lang="ts">
import { useMutation, useQueryClient } from '@tanstack/vue-query'
import { Save } from '@lucide/vue'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useDirtyRouteGuard } from '../admin'
import { api } from '../api/client'
import { localizedApiError } from '../errorLocalization'
import type { AdminUser, UserQuota } from '../types'
import AdminConfirmDialog from './AdminConfirmDialog.vue'

const props = defineProps<{ user: AdminUser }>()
const { t, te } = useI18n()
const queryClient = useQueryClient()
type QuotaKey = keyof Pick<
  UserQuota,
  | 'daily_experiment_limit'
  | 'daily_concurrent_experiment_limit'
  | 'daily_inference_request_limit'
  | 'daily_inference_input_limit'
>
const quotaKeys: QuotaKey[] = [
  'daily_experiment_limit',
  'daily_concurrent_experiment_limit',
  'daily_inference_request_limit',
  'daily_inference_input_limit',
]
const quotaDraft = ref<Record<QuotaKey, number>>(quotaValues(props.user.quota))
const message = ref('')
const dirty = computed(() =>
  quotaKeys.some((key) => normalize(quotaDraft.value[key]) !== props.user.quota[key]),
)
const dirtyGuard = useDirtyRouteGuard(dirty)

function quotaValues(quota: UserQuota) {
  return Object.fromEntries(quotaKeys.map((key) => [key, quota[key]])) as Record<QuotaKey, number>
}

function normalize(value: number) {
  const number = Number(value)
  return Number.isFinite(number) && number > 0 ? Math.floor(number) : 0
}

watch(
  () => props.user.quota,
  (quota) => {
    quotaDraft.value = quotaValues(quota)
    message.value = ''
  },
)

const updateMutation = useMutation({
  mutationFn: () =>
    api.adminUpdateUserQuotas(
      props.user.id,
      Object.fromEntries(quotaKeys.map((key) => [key, normalize(quotaDraft.value[key])])),
    ),
  onSuccess: async () => {
    message.value = t('admin.quotaSaved')
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['admin-user', props.user.id] }),
      queryClient.invalidateQueries({ queryKey: ['admin-users'] }),
      queryClient.invalidateQueries({ queryKey: ['admin-overview'] }),
      queryClient.invalidateQueries({ queryKey: ['admin-audit-logs'] }),
    ])
  },
})

const usageRows = computed(() => [
  {
    label: t('admin.dailyExperimentLimit'),
    used: props.user.usage.experiments_created,
    limit: props.user.quota.daily_experiment_limit,
  },
  {
    label: t('admin.dailyInferenceRequestLimit'),
    used: props.user.usage.inference_requests,
    limit: props.user.quota.daily_inference_request_limit,
  },
  {
    label: t('admin.dailyInferenceInputLimit'),
    used: props.user.usage.inference_inputs,
    limit: props.user.quota.daily_inference_input_limit,
  },
])

function save() {
  if (!dirty.value) return
  message.value = ''
  updateMutation.mutate()
}

function percentage(used: number, limit: number) {
  if (limit <= 0) return used > 0 ? 100 : 0
  return Math.min(100, Math.round((used / limit) * 100))
}

function errorMessage(error: unknown) {
  return localizedApiError(error, t, te, 'admin.genericError')
}
</script>

<template>
  <section class="admin-settings-panel">
    <header
      ><div
        ><h2>{{ t('admin.usageAndQuotas') }}</h2
        ><p>{{ t('admin.usageAndQuotasHint') }}</p></div
      ><span class="admin-date-chip">{{ user.usage.usage_date }} UTC</span></header
    >

    <div class="admin-usage-bars">
      <div v-for="row in usageRows" :key="row.label">
        <span
          ><strong>{{ row.label }}</strong
          ><small>{{ t('admin.quotaUse', { used: row.used, limit: row.limit }) }}</small></span
        >
        <div><i :style="{ width: `${percentage(row.used, row.limit)}%` }" /></div>
      </div>
    </div>

    <section class="admin-settings-subsection admin-quota-editor">
      <header
        ><div
          ><h3>{{ t('admin.quotaLimits') }}</h3
          ><p>{{ t('admin.zeroQuotaHint') }}</p></div
        ></header
      >
      <div class="admin-quota-grid">
        <label
          ><span>{{ t('admin.dailyExperimentLimit') }}</span
          ><input
            v-model.number="quotaDraft.daily_experiment_limit"
            min="0"
            step="1"
            type="number"
            class="field-control"
        /></label>
        <label
          ><span>{{ t('admin.concurrentExperimentLimit') }}</span
          ><input
            v-model.number="quotaDraft.daily_concurrent_experiment_limit"
            min="0"
            step="1"
            type="number"
            class="field-control"
        /></label>
        <label
          ><span>{{ t('admin.dailyInferenceRequestLimit') }}</span
          ><input
            v-model.number="quotaDraft.daily_inference_request_limit"
            min="0"
            step="1"
            type="number"
            class="field-control"
        /></label>
        <label
          ><span>{{ t('admin.dailyInferenceInputLimit') }}</span
          ><input
            v-model.number="quotaDraft.daily_inference_input_limit"
            min="0"
            step="1"
            type="number"
            class="field-control"
        /></label>
      </div>
    </section>

    <div class="admin-settings-actions"
      ><button
        type="button"
        class="primary-action"
        :disabled="!dirty || updateMutation.isPending.value"
        @click="save"
        ><Save aria-hidden="true" />{{
          updateMutation.isPending.value ? t('common.saving') : t('admin.saveQuotas')
        }}</button
      ></div
    >
    <div class="admin-feedback" aria-live="polite"
      ><p v-if="message" class="success-copy">{{ message }}</p
      ><p v-if="updateMutation.error.value" class="error-copy">{{
        errorMessage(updateMutation.error.value)
      }}</p></div
    >
  </section>

  <AdminConfirmDialog
    :open="dirtyGuard.discardDialogOpen.value"
    :title="t('admin.discardTitle')"
    :description="t('admin.discardDescription')"
    :confirm-label="t('admin.discardChanges')"
    :cancel-label="t('admin.keepEditing')"
    @confirm="dirtyGuard.confirmDiscard"
    @cancel="dirtyGuard.keepEditing"
  />
</template>
