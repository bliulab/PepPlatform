<script setup lang="ts">
import { useMutation, useQueryClient } from '@tanstack/vue-query'
import { Mail, Save } from '@lucide/vue'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useDirtyRouteGuard } from '../admin'
import { api } from '../api/client'
import { localizedApiError } from '../errorLocalization'
import { useAuthStore } from '../stores/auth'
import type { AdminUser } from '../types'
import AdminConfirmDialog from './AdminConfirmDialog.vue'

const props = defineProps<{ user: AdminUser }>()
const { t, te } = useI18n()
const auth = useAuthStore()
const queryClient = useQueryClient()
const roleDraft = ref(props.user.role)
const statusDraft = ref(props.user.status)
const message = ref('')
const verificationMessage = ref('')
const debugCode = ref('')
const selectedIsSelf = computed(() => auth.user?.id === props.user.id)
const dirty = computed(
  () => roleDraft.value !== props.user.role || statusDraft.value !== props.user.status,
)
const dirtyGuard = useDirtyRouteGuard(dirty)

watch(
  () => props.user,
  (user) => {
    roleDraft.value = user.role
    statusDraft.value = user.status
    message.value = ''
    verificationMessage.value = ''
    debugCode.value = ''
  },
)

const updateMutation = useMutation({
  mutationFn: () =>
    api.adminUpdateUser(props.user.id, { role: roleDraft.value, status: statusDraft.value }),
  onSuccess: async () => {
    message.value = t('admin.lifecycleSaved')
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['admin-user', props.user.id] }),
      queryClient.invalidateQueries({ queryKey: ['admin-users'] }),
      queryClient.invalidateQueries({ queryKey: ['admin-overview'] }),
      queryClient.invalidateQueries({ queryKey: ['admin-audit-logs'] }),
    ])
  },
})
const resendMutation = useMutation({
  mutationFn: () => api.adminResendVerification(props.user.id),
  onSuccess: async (result) => {
    verificationMessage.value = result.verification_sent
      ? t('admin.verificationSent')
      : t('admin.verificationQueued')
    debugCode.value = result.debug_code || ''
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['admin-user', props.user.id] }),
      queryClient.invalidateQueries({ queryKey: ['admin-users'] }),
      queryClient.invalidateQueries({ queryKey: ['admin-audit-logs'] }),
    ])
  },
})

function save() {
  if (!dirty.value || selectedIsSelf.value) return
  message.value = ''
  updateMutation.mutate()
}

function resendVerification() {
  if (props.user.email_verified) return
  verificationMessage.value = ''
  debugCode.value = ''
  resendMutation.mutate()
}

function errorMessage(error: unknown) {
  return localizedApiError(error, t, te, 'admin.genericError')
}
</script>

<template>
  <section class="admin-settings-panel">
    <header
      ><div
        ><h2>{{ t('admin.accountSettings') }}</h2
        ><p>{{ t('admin.accountSettingsHint') }}</p></div
      ></header
    >

    <div class="admin-settings-row">
      <div
        ><strong>{{ t('admin.role') }}</strong
        ><span>{{ t('admin.roleHint') }}</span></div
      >
      <select
        v-model="roleDraft"
        class="field-control"
        :disabled="selectedIsSelf || updateMutation.isPending.value"
      >
        <option value="admin">{{ t('account.admin') }}</option
        ><option value="user">{{ t('account.user') }}</option>
      </select>
    </div>

    <div class="admin-settings-row">
      <div
        ><strong>{{ t('admin.status') }}</strong
        ><span>{{ t('admin.statusHint') }}</span></div
      >
      <div class="admin-segmented admin-status-segmented">
        <button
          type="button"
          :aria-pressed="statusDraft === 'active'"
          :disabled="selectedIsSelf || updateMutation.isPending.value"
          @click="statusDraft = 'active'"
          >{{ t('admin.statusActive') }}</button
        >
        <button
          type="button"
          :aria-pressed="statusDraft === 'disabled'"
          :disabled="selectedIsSelf || updateMutation.isPending.value"
          @click="statusDraft = 'disabled'"
          >{{ t('admin.statusDisabled') }}</button
        >
      </div>
    </div>

    <p v-if="selectedIsSelf" class="admin-context-note">{{ t('admin.selfProtection') }}</p>
    <div class="admin-settings-actions"
      ><button
        type="button"
        class="primary-action"
        :disabled="selectedIsSelf || !dirty || updateMutation.isPending.value"
        @click="save"
        ><Save aria-hidden="true" />{{
          updateMutation.isPending.value ? t('common.saving') : t('common.save')
        }}</button
      ></div
    >

    <section class="admin-settings-subsection">
      <header
        ><div
          ><h3>{{ t('admin.emailVerification') }}</h3
          ><p>{{ t('admin.emailVerificationHint') }}</p></div
        ><span
          class="status-pill"
          :class="user.email_verified ? 'status-pill-success' : 'status-pill-warning'"
          >{{ user.email_verified ? t('account.verified') : t('account.unverified') }}</span
        ></header
      >
      <button
        type="button"
        class="secondary-action"
        :disabled="user.email_verified || resendMutation.isPending.value"
        @click="resendVerification"
        ><Mail aria-hidden="true" />{{
          resendMutation.isPending.value ? t('admin.sending') : t('admin.resendVerification')
        }}</button
      >
      <p v-if="debugCode" class="dev-code">{{ t('admin.debugCode', { code: debugCode }) }}</p>
    </section>

    <div class="admin-feedback" aria-live="polite">
      <p v-if="message" class="success-copy">{{ message }}</p
      ><p v-if="verificationMessage" class="success-copy">{{ verificationMessage }}</p>
      <p v-if="updateMutation.error.value" class="error-copy">{{
        errorMessage(updateMutation.error.value)
      }}</p>
      <p v-if="resendMutation.error.value" class="error-copy">{{
        errorMessage(resendMutation.error.value)
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
