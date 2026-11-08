<script setup lang="ts">
import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import { Eye, Save, ShieldCheck } from '@lucide/vue'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { api } from '../api/client'
import { localizedApiError } from '../errorLocalization'
import { quotaFields, quotaValue } from '../resourcePolicy'
import type { DailyReductionMode, PolicyDraft, ResourcePolicy } from '../resourceTypes'
import type { QuotaKey } from '../types'
import { useDirtyRouteGuard } from '../admin'
import AdminConfirmDialog from '../components/AdminConfirmDialog.vue'
import QuotaChangeDetails from '../components/QuotaChangeDetails.vue'
import QuotaReductionMode from '../components/QuotaReductionMode.vue'

const { t, te, locale } = useI18n()
const client = useQueryClient()
const query = useQuery({ queryKey: ['resource-policy'], queryFn: api.resourcePolicy })
const emailUsageQuery = useQuery({
  queryKey: ['email-delivery-usage'],
  queryFn: api.emailDeliveryUsage,
  refetchInterval: 30_000,
})
const emailResetTime = computed(() => {
  const reset = emailUsageQuery.data.value?.reset_at
  return reset ? new Date(reset).toLocaleString(locale.value === 'zh' ? 'zh-CN' : 'en-US') : ''
})
const draft = ref<ResourcePolicy | null>(null)
const values = ref<Record<QuotaKey, number>>({} as Record<QuotaKey, number>)
const selected = ref<QuotaKey[]>([])
const synchronize = ref(false)
const reductionMode = ref<DailyReductionMode>('next_reset')
const publishAnnouncements = ref(false)
const allowed = ref('')
const message = ref('')
const gib = 1024 ** 3
const limitFields: {
  key: Exclude<keyof ResourcePolicy['limits'], 'allowed_pretrained_models'>
  label: string
  divisor: number
}[] = [
  { key: 'daily_registration_limit', label: 'dailyRegistration', divisor: 1 },
  { key: 'daily_verification_email_limit', label: 'dailyVerificationEmails', divisor: 1 },
  { key: 'queue_capacity', label: 'queue', divisor: 1 },
  { key: 'disk_reserve_bytes', label: 'diskReserve', divisor: gib },
  { key: 'disk_reserve_percent', label: 'diskReservePercent', divisor: 1 },
  { key: 'disk_critical_bytes', label: 'diskCritical', divisor: gib },
  { key: 'disk_critical_percent', label: 'diskCriticalPercent', divisor: 1 },
  { key: 'inference_timeout_seconds', label: 'inferenceTimeout', divisor: 1 },
]
watch(
  query.data,
  (policy) => {
    if (!policy) return
    draft.value = JSON.parse(JSON.stringify(policy)) as ResourcePolicy
    values.value = Object.fromEntries(
      quotaFields.map((field) => [field.key, policy.defaults[field.key] / field.divisor]),
    ) as Record<QuotaKey, number>
    allowed.value = policy.limits.allowed_pretrained_models.join('\n')
    selected.value = []
    reductionMode.value = 'next_reset'
    publishAnnouncements.value = false
  },
  { immediate: true },
)
const payload = computed<PolicyDraft>(() => ({
  revision: draft.value?.revision ?? 0,
  defaults: Object.fromEntries(
    quotaFields
      .filter((field) => selected.value.includes(field.key))
      .map((field) => [
        field.key,
        quotaValue(values.value[field.key], field.divisor, field.minimum) ?? -1,
      ]),
  ),
  limits: draft.value
    ? {
        ...draft.value.limits,
        allowed_pretrained_models: allowed.value
          .split('\n')
          .map((value) => value.trim())
          .filter(Boolean),
      }
    : undefined,
  retention: draft.value?.retention,
  apply_to_users: synchronize.value,
  daily_reduction_mode: reductionMode.value,
  publish_announcements: publishAnnouncements.value,
}))
watch(
  () => JSON.stringify(payload.value),
  () => {
    message.value = ''
  },
)
const modelLimitsValid = computed(() => {
  const defaults = { ...draft.value?.defaults, ...payload.value.defaults }
  return Number(defaults.protected_model_limit) <= Number(defaults.max_model_count)
})
const valid = computed(
  () =>
    modelLimitsValid.value &&
    (!draft.value || quotaValue(draft.value.limits.daily_verification_email_limit, 1) !== null) &&
    quotaFields.every(
      (field) =>
        !selected.value.includes(field.key) ||
        quotaValue(values.value[field.key], field.divisor, field.minimum) !== null,
    ),
)
const dirty = computed(() =>
  Boolean(
    draft.value &&
    query.data.value &&
    (selected.value.length ||
      JSON.stringify(draft.value.limits) !== JSON.stringify(query.data.value.limits) ||
      JSON.stringify(draft.value.retention) !== JSON.stringify(query.data.value.retention) ||
      allowed.value !== query.data.value.limits.allowed_pretrained_models.join('\n')),
  ),
)
const guard = useDirtyRouteGuard(dirty)
const previewMutation = useMutation({
  mutationFn: async () => {
    const snapshot = JSON.parse(JSON.stringify(payload.value)) as PolicyDraft
    const result = await api.previewResourcePolicy(snapshot)
    return { result, fingerprint: JSON.stringify(snapshot) }
  },
})
const preview = computed(() => {
  const data = previewMutation.data.value
  return data?.fingerprint === JSON.stringify(payload.value) ? data.result : null
})
const saveMutation = useMutation({
  mutationFn: () =>
    api.saveResourcePolicy({ ...payload.value, preview_token: preview.value?.preview_token }),
  onSuccess: async () => {
    reductionMode.value = 'next_reset'
    publishAnnouncements.value = false
    await Promise.all([
      client.invalidateQueries({ queryKey: ['resource-policy'] }),
      client.invalidateQueries({ queryKey: ['email-delivery-usage'] }),
      client.invalidateQueries({ queryKey: ['admin-users'] }),
      client.invalidateQueries({ queryKey: ['admin-user'] }),
      client.invalidateQueries({ queryKey: ['admin-audit-logs'] }),
      client.invalidateQueries({ queryKey: ['own-resources'] }),
      client.invalidateQueries({ queryKey: ['announcements'] }),
      client.invalidateQueries({ queryKey: ['admin-announcements'] }),
    ])
    message.value = t('resource.saved')
  },
})
function updateLimit(key: (typeof limitFields)[number]['key'], value: Event, divisor: number) {
  if (draft.value)
    draft.value.limits[key] =
      quotaValue(Number((value.target as HTMLInputElement).value), divisor) ?? Number.NaN
}
</script>

<template>
  <section class="admin-page resource-page">
    <header class="admin-page-header"
      ><div
        ><p>{{ t('resource.policies') }}</p
        ><h1>{{ t('resource.policyTitle') }}</h1
        ><span>{{ t('resource.policySubtitle') }}</span></div
      ><ShieldCheck class="resource-heading-icon" aria-hidden="true"
    /></header>
    <p v-if="query.error.value" class="error-copy">{{
      localizedApiError(query.error.value, t, te, 'resource.error')
    }}</p>
    <form v-if="draft" class="resource-form" @submit.prevent="previewMutation.mutate()">
      <section class="admin-settings-panel">
        <header
          ><div
            ><h2>{{ t('resource.defaults') }}</h2
            ><p>{{ t('resource.selectFields') }}</p></div
          ></header
        >
        <div class="admin-quota-grid">
          <div v-for="field in quotaFields" :key="field.key" class="resource-quota-field">
            <label class="resource-check"
              ><input v-model="selected" type="checkbox" :value="field.key" /><span
                class="resource-quota-caption"
                >{{ t(`resource.fields.${field.key}`) }}
                <small v-if="field.unit"
                  >({{ field.unit === 'days' ? t('resource.days') : field.unit }})</small
                ></span
              ></label
            >
            <input
              v-model.number="values[field.key]"
              :aria-label="t(`resource.fields.${field.key}`)"
              type="number"
              step="any"
              :min="field.minimum / field.divisor"
              :disabled="!selected.includes(field.key)"
              class="field-control"
              required
            />
          </div> </div
        ><p class="resource-help">{{ t('resource.zeroHint') }}</p>
        <QuotaReductionMode
          v-model="reductionMode"
          name="global-daily-reduction-mode"
          :disabled="saveMutation.isPending.value"
        />
        <p class="resource-help">{{ t('resource.modelLimitHint') }}</p>
        <p v-if="!modelLimitsValid" class="error-copy" role="alert">{{
          t('resource.modelLimitsInvalid')
        }}</p>
      </section>
      <section class="admin-settings-panel">
        <header
          ><h2>{{ t('resource.limits') }}</h2></header
        >
        <div class="admin-quota-grid">
          <label v-for="field in limitFields" :key="field.key"
            ><span>{{ t(`resource.${field.label}`) }}</span
            ><input
              :value="draft.limits[field.key] / field.divisor"
              type="number"
              min="0"
              :step="field.key === 'daily_verification_email_limit' ? 1 : 'any'"
              required
              class="field-control"
              @input="updateLimit(field.key, $event, field.divisor)"
          /></label>
        </div>
        <p class="resource-help">{{ t('resource.emailLimitHint') }}</p>
        <div aria-live="polite">
          <p v-if="emailUsageQuery.data.value" class="resource-help">{{
            t('resource.emailUsage', {
              used: emailUsageQuery.data.value.used,
              limit: emailUsageQuery.data.value.limit,
              remaining: emailUsageQuery.data.value.remaining,
            })
          }}</p>
          <p v-if="emailUsageQuery.data.value" class="resource-help">{{
            emailUsageQuery.data.value.limit === 0
              ? t('resource.emailPaused')
              : t('resource.emailReset', { time: emailResetTime })
          }}</p>
          <p v-if="emailUsageQuery.error.value" class="error-copy">{{
            localizedApiError(emailUsageQuery.error.value, t, te, 'resource.error')
          }}</p>
        </div>
        <button
          type="button"
          class="secondary-action"
          :disabled="emailUsageQuery.isFetching.value"
          @click="emailUsageQuery.refetch()"
          >{{ t('resource.refreshEmailUsage') }}</button
        >
        <p class="resource-help">{{ t('resource.emailCountingHint') }}</p>
        <label class="resource-full-field"
          ><span>{{ t('resource.allowedModels') }}</span
          ><textarea v-model="allowed" rows="6" class="field-control" spellcheck="false" />
        </label>
      </section>
      <section class="admin-settings-panel">
        <header
          ><div
            ><h2>{{ t('resource.retention') }}</h2
            ><p>{{ t('resource.cleanupHint') }}</p></div
          ></header
        >
        <label class="resource-check"
          ><input v-model="draft.retention.enabled" type="checkbox" />{{
            t('resource.enableRetention')
          }}</label
        >
        <label class="resource-check"
          ><input v-model="draft.retention.delete_unprotected_models" type="checkbox" />{{
            t('resource.deleteModels')
          }}</label
        >
        <label class="resource-check">
          <input v-model="draft.retention.publish_announcements" type="checkbox" />
          {{ t('resource.publishAutomaticCleanupAnnouncements') }}
        </label>
        <p class="resource-help">{{
          t(
            draft.retention.publish_announcements
              ? 'resource.cleanupAnnouncementHint'
              : 'resource.silentCleanupHint',
          )
        }}</p>
        <div class="admin-quota-grid">
          <label v-if="draft.retention.publish_announcements"
            ><span>{{ t('resource.noticeHours') }}</span
            ><input
              v-model.number="draft.retention.notice_hours"
              type="number"
              min="72"
              required
              class="field-control"
          /></label>
          <label
            ><span>{{ t('resource.idleHours') }}</span
            ><input
              v-model.number="draft.retention.cache_idle_hours"
              type="number"
              min="1"
              required
              class="field-control"
          /></label>
        </div>
      </section>
      <section class="admin-settings-panel">
        <div class="resource-mode"
          ><label class="resource-check"
            ><input v-model="synchronize" type="radio" :value="false" />{{
              t('resource.defaultOnly')
            }}</label
          ><label class="resource-check"
            ><input v-model="synchronize" type="radio" :value="true" />{{
              t('resource.applyUsers')
            }}</label
          ></div
        >
        <label class="resource-check">
          <input
            v-model="publishAnnouncements"
            type="checkbox"
            :disabled="saveMutation.isPending.value"
          />
          {{ t('resource.publishQuotaAnnouncement') }}
        </label>
        <p class="resource-help">{{ t('resource.quotaAnnouncementHint') }}</p>
        <p class="resource-help">{{ t('resource.platformAnnouncementHint') }}</p>
        <div v-if="preview" class="resource-preview" aria-live="polite"
          ><strong>{{
            t('resource.affected', { count: synchronize ? preview.affected_count : 0 })
          }}</strong
          ><span>{{ t('resource.skipped', { count: preview.skipped_count }) }}</span
          ><span>{{ t('resource.announcementCount', { count: preview.announcement_count }) }}</span
          ><p>{{
            t('resource.impactSummary', {
              immediate: preview.immediate_count,
              deferred: preview.deferred_count,
              blocked: preview.blocked_count,
              exempt: preview.exempt_count,
            })
          }}</p>
          <details v-if="preview.impacts?.length"
            ><summary>{{ t('resource.preview') }}</summary>
            <div v-for="user in preview.impacts" :key="user.user_id"
              ><strong>{{ user.email }}</strong
              ><QuotaChangeDetails :changes="user.changes"
            /></div> </details
          ><ul v-if="synchronize"
            ><li v-for="user in preview.affected" :key="user.user_id"
              >{{ user.email }} ·
              {{ user.fields.map((field) => t(`resource.fields.${field}`)).join(' / ') }}</li
            ></ul
          ></div
        >
        <p v-else class="resource-help">{{ t('resource.previewRequired') }}</p>
        <div class="resource-actions"
          ><button
            type="submit"
            class="secondary-action"
            :disabled="!valid || previewMutation.isPending.value"
            ><Eye aria-hidden="true" />{{ t('resource.preview') }}</button
          ><button
            type="button"
            class="primary-action"
            :disabled="!preview || !valid || saveMutation.isPending.value"
            @click="saveMutation.mutate()"
            ><Save aria-hidden="true" />{{ t('resource.save') }}</button
          ></div
        >
        <p v-if="message" class="success-copy" aria-live="polite">{{ message }}</p>
        <p
          v-if="previewMutation.error.value || saveMutation.error.value"
          class="error-copy"
          role="alert"
          >{{
            localizedApiError(
              previewMutation.error.value || saveMutation.error.value,
              t,
              te,
              'resource.error',
            )
          }}</p
        >
      </section>
    </form>
    <AdminConfirmDialog
      :open="guard.discardDialogOpen.value"
      :title="t('admin.discardTitle')"
      :description="t('admin.discardDescription')"
      :confirm-label="t('admin.discardChanges')"
      :cancel-label="t('admin.keepEditing')"
      @confirm="guard.confirmDiscard"
      @cancel="guard.keepEditing"
    />
  </section>
</template>
