<script setup lang="ts">
import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import { Save } from '@lucide/vue'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useDirtyRouteGuard } from '../admin'
import { api, type UserQuotaUpdatePayload } from '../api/client'
import { localizedApiError } from '../errorLocalization'
import {
  dailyQuotaFields,
  pendingQuotaChanges,
  pendingQuotaUpdates,
  quotaFields,
  quotaValue,
} from '../resourcePolicy'
import type { DailyReductionMode } from '../resourceTypes'
import type { AdminUser, QuotaKey, UserQuota } from '../types'
import AdminConfirmDialog from './AdminConfirmDialog.vue'
import QuotaChangeDetails from './QuotaChangeDetails.vue'
import QuotaReductionMode from './QuotaReductionMode.vue'

const props = defineProps<{ user: AdminUser }>()
const { t, te } = useI18n()
const queryClient = useQueryClient()
const policy = useQuery({ queryKey: ['resource-policy'], queryFn: api.resourcePolicy })
const draft = ref<Record<QuotaKey, number>>({} as Record<QuotaKey, number>)
const inherited = ref<Record<QuotaKey, boolean>>({} as Record<QuotaKey, boolean>)
const allowGlobal = ref(false)
const reductionMode = ref<DailyReductionMode>('next_reset')
const publishAnnouncements = ref(false)
const publishPendingAnnouncements = ref(false)
const message = ref('')
const exempt = computed(() => props.user.quota.enforcement === 'exempt_admin')
function inherits(quota: UserQuota, key: QuotaKey) {
  return quota.sources?.[key] === 'global' || quota[key] === undefined
}
watch(
  () => props.user.quota,
  (quota) => {
    draft.value = Object.fromEntries(
      quotaFields.map((field) => [
        field.key,
        (quota.configured?.[field.key] ?? quota[field.key] ?? 0) / field.divisor,
      ]),
    ) as Record<QuotaKey, number>
    inherited.value = Object.fromEntries(
      quotaFields.map((field) => [field.key, inherits(quota, field.key)]),
    ) as Record<QuotaKey, boolean>
    allowGlobal.value = quota.allow_global_override ?? false
    reductionMode.value = 'next_reset'
    publishAnnouncements.value = false
    publishPendingAnnouncements.value = false
    message.value = ''
  },
  { immediate: true },
)
const payload = computed(() => {
  const changes: UserQuotaUpdatePayload = {}
  if (exempt.value) return changes
  for (const field of quotaFields) {
    const value = quotaValue(draft.value[field.key], field.divisor, field.minimum)
    if (
      inherited.value[field.key] !== inherits(props.user.quota, field.key) ||
      (!inherited.value[field.key] &&
        value !== (props.user.quota.configured?.[field.key] ?? props.user.quota[field.key]))
    ) {
      changes[field.key] = inherited.value[field.key] ? null : value
    }
  }
  if (allowGlobal.value !== (props.user.quota.allow_global_override ?? false))
    changes.allow_global_override = allowGlobal.value
  return changes
})
const modelLimitsValid = computed(() => {
  const effective = (key: 'max_model_count' | 'protected_model_limit') =>
    inherited.value[key]
      ? (policy.data.value?.defaults[key] ?? props.user.quota[key])
      : draft.value[key]
  return Number(effective('protected_model_limit')) <= Number(effective('max_model_count'))
})
const valid = computed(
  () =>
    modelLimitsValid.value &&
    quotaFields.every(
      (field) =>
        inherited.value[field.key] ||
        quotaValue(draft.value[field.key], field.divisor, field.minimum) !== null,
    ),
)
const dirty = computed(() => Object.keys(payload.value).length > 0)
const pendingUpdates = computed(() => pendingQuotaUpdates(props.user.quota))
const hasPending = computed(() => Object.keys(pendingUpdates.value).length > 0)
const dirtyGuard = useDirtyRouteGuard(dirty)
const mutation = useMutation({
  mutationFn: (updates: UserQuotaUpdatePayload) =>
    api.adminUpdateUserQuotas(props.user.id, updates),
  onSuccess: async () => {
    reductionMode.value = 'next_reset'
    publishAnnouncements.value = false
    publishPendingAnnouncements.value = false
    message.value = t('resource.saved')
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['admin-user', props.user.id] }),
      queryClient.invalidateQueries({ queryKey: ['admin-users'] }),
      queryClient.invalidateQueries({ queryKey: ['admin-audit-logs'] }),
      queryClient.invalidateQueries({ queryKey: ['own-resources'] }),
      queryClient.invalidateQueries({ queryKey: ['announcements'] }),
      queryClient.invalidateQueries({ queryKey: ['admin-announcements'] }),
    ])
  },
})
function save() {
  message.value = ''
  mutation.mutate({
    ...payload.value,
    publish_announcements: publishAnnouncements.value,
    ...(reductionMode.value === 'immediate' && dailyQuotaFields.some((key) => key in payload.value)
      ? { daily_reduction_mode: reductionMode.value }
      : {}),
  })
}
function applyPendingNow() {
  if (dirty.value || !hasPending.value || mutation.isPending.value) return
  message.value = ''
  mutation.mutate({
    ...pendingUpdates.value,
    daily_reduction_mode: 'immediate',
    publish_announcements: publishPendingAnnouncements.value,
  })
}
const usageRows = computed(() => [
  {
    label: t('resource.retainedModels'),
    used:
      (props.user.model_usage?.retained ?? props.user.resource_counts.models) +
      (props.user.model_usage?.reserved ?? 0),
    limit: props.user.quota.max_model_count ?? 0,
  },
  {
    label: t('resource.protectedModels'),
    used: props.user.model_usage?.protected ?? 0,
    limit: props.user.quota.protected_model_limit ?? 0,
  },
  {
    label: t('resource.userStorage') + ' (GiB)',
    used: Number(
      (
        ((props.user.storage?.used_bytes ?? 0) + (props.user.storage?.reserved_bytes ?? 0)) /
        1024 ** 3
      ).toFixed(2),
    ),
    limit: (props.user.quota.storage_limit_bytes ?? 0) / 1024 ** 3,
  },
  {
    label: t('resource.fields.daily_experiment_limit'),
    used: props.user.usage.experiments_created,
    limit: props.user.quota.daily_experiment_limit,
  },
  {
    label: t('resource.trainingTime') + ' (min)',
    used: Math.round((props.user.usage.training_seconds ?? 0) / 60),
    limit: (props.user.quota.daily_training_seconds_limit ?? 0) / 60,
  },
  {
    label: t('resource.fields.daily_inference_request_limit'),
    used: props.user.usage.inference_requests,
    limit: props.user.quota.daily_inference_request_limit,
  },
  {
    label: t('resource.fields.daily_inference_input_limit'),
    used: props.user.usage.inference_inputs,
    limit: props.user.quota.daily_inference_input_limit,
  },
])
function percentage(used: number, limit: number) {
  return limit > 0 ? Math.min(100, Math.round((used / limit) * 100)) : used > 0 ? 100 : 0
}
</script>

<template>
  <section class="admin-settings-panel">
    <header
      ><div
        ><h2>{{ t('admin.usageAndQuotas') }}</h2
        ><p>{{ t(exempt ? 'resource.adminExempt' : 'resource.zeroHint') }}</p></div
      ><span class="admin-date-chip">{{ user.usage.usage_date }} UTC</span></header
    >
    <div class="admin-usage-bars">
      <div v-for="row in usageRows" :key="row.label">
        <span
          ><strong>{{ row.label }}</strong
          ><small>{{
            exempt
              ? t('resource.usedUnlimited', { used: row.used })
              : t('resource.usedOf', { used: row.used, limit: row.limit })
          }}</small></span
        >
        <div v-if="!exempt"><i :style="{ width: `${percentage(row.used, row.limit)}%` }" /></div>
      </div>
    </div>
    <p v-if="user.model_usage?.reserved" class="resource-help">{{
      t('resource.modelReserved', { count: user.model_usage.reserved })
    }}</p>
    <QuotaChangeDetails :changes="pendingQuotaChanges(user.quota)" pending />
    <div v-if="hasPending" class="resource-actions resource-pending-actions">
      <label class="resource-check">
        <input
          v-model="publishPendingAnnouncements"
          type="checkbox"
          :disabled="mutation.isPending.value"
        />
        {{ t('resource.publishQuotaAnnouncement') }}
      </label>
      <button
        type="button"
        class="secondary-action"
        :disabled="dirty || mutation.isPending.value"
        @click="applyPendingNow"
        >{{ t('resource.applyPendingNow') }}</button
      >
      <p v-if="dirty" class="resource-help">{{ t('resource.pendingEditHint') }}</p>
    </div>
    <p v-if="!exempt && user.model_usage?.over_limit" class="resource-policy-note" role="note">{{
      t('resource.modelOverLimit')
    }}</p>
    <section v-if="!exempt" class="admin-settings-subsection admin-quota-editor">
      <header
        ><div
          ><h3>{{ t('admin.quotaLimits') }}</h3></div
        ></header
      >
      <div class="admin-quota-grid">
        <div
          v-for="field in quotaFields"
          :key="field.key"
          class="resource-quota-field resource-quota-field--personal"
        >
          <label class="resource-quota-label" :for="`quota-${field.key}`"
            >{{ t(`resource.fields.${field.key}`) }}
            <small v-if="field.unit"
              >({{ field.unit === 'days' ? t('resource.days') : field.unit }})</small
            ></label
          >
          <input
            :id="`quota-${field.key}`"
            v-model.number="draft[field.key]"
            :disabled="inherited[field.key]"
            :min="field.minimum / field.divisor"
            step="any"
            type="number"
            class="field-control"
          />
          <label class="resource-check"
            ><input v-model="inherited[field.key]" type="checkbox" />{{
              t('resource.inherit')
            }}</label
          >
        </div>
      </div>
      <p class="resource-help">{{ t('resource.modelLimitHint') }}</p>
      <QuotaReductionMode
        v-model="reductionMode"
        name="personal-daily-reduction-mode"
        :disabled="mutation.isPending.value"
      />
      <p v-if="!modelLimitsValid" class="error-copy" role="alert">{{
        t('resource.modelLimitsInvalid')
      }}</p>
      <label class="resource-check resource-override-check"
        ><input v-model="allowGlobal" type="checkbox" />{{ t('resource.allowGlobal') }}</label
      >
      <p class="resource-help">{{ t('resource.allowGlobalHint') }}</p>
    </section>
    <div v-if="!exempt" class="admin-settings-actions resource-actions"
      ><label class="resource-check">
        <input
          v-model="publishAnnouncements"
          type="checkbox"
          :disabled="mutation.isPending.value"
        />
        {{ t('resource.publishQuotaAnnouncement') }}
      </label>
      <button
        type="button"
        class="primary-action"
        :disabled="!dirty || !valid || mutation.isPending.value"
        @click="save"
        ><Save aria-hidden="true" />{{ t('admin.saveQuotas') }}</button
      ></div
    >
    <p v-if="!exempt" class="resource-help">{{ t('resource.quotaAnnouncementHint') }}</p>
    <div class="admin-feedback" aria-live="polite">
      <p v-if="message" class="success-copy">{{ message }}</p>
      <p v-if="!valid" class="error-copy">{{ t('resource.invalid') }}</p>
      <p v-if="mutation.error.value" class="error-copy">{{
        localizedApiError(mutation.error.value, t, te, 'resource.error')
      }}</p>
    </div>
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
