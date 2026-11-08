<script setup lang="ts">
import { useMutation, useQueryClient } from '@tanstack/vue-query'
import { ShieldOff, Trash2 } from '@lucide/vue'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { api } from '../api/client'
import { localizedApiError } from '../errorLocalization'
import { formatLocalizedDate } from '../localization'
import { useAuthStore } from '../stores/auth'
import type { AdminUser } from '../types'
import AdminConfirmDialog from './AdminConfirmDialog.vue'

const props = defineProps<{ user: AdminUser; returnPath: string }>()
const { locale, t, te } = useI18n()
const auth = useAuthStore()
const router = useRouter()
const queryClient = useQueryClient()
const revokeDialogOpen = ref(false)
const deleteDialogOpen = ref(false)
const message = ref('')
const selectedIsSelf = computed(() => auth.user?.id === props.user.id)
const activeSessions = computed(
  () => (props.user.sessions ?? []).filter((session) => session.active).length,
)

const revokeMutation = useMutation({
  mutationFn: () => api.adminRevokeUserSessions(props.user.id),
  onSuccess: async (result) => {
    revokeDialogOpen.value = false
    message.value = t('admin.sessionsRevoked', { count: result.revoked_sessions })
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['admin-user', props.user.id] }),
      queryClient.invalidateQueries({ queryKey: ['admin-audit-logs'] }),
      queryClient.invalidateQueries({ queryKey: ['admin-overview'] }),
    ])
  },
})
const deleteMutation = useMutation({
  mutationFn: () => api.adminDeleteUser(props.user.id),
  onSuccess: async () => {
    deleteDialogOpen.value = false
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['admin-users'] }),
      queryClient.invalidateQueries({ queryKey: ['admin-overview'] }),
      queryClient.invalidateQueries({ queryKey: ['admin-audit-logs'] }),
    ])
    await router.push(props.returnPath)
  },
})

function formatDate(value?: string | null) {
  return formatLocalizedDate(
    value,
    locale.value,
    {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    },
    t('admin.notAvailable'),
  )
}

function codePurpose(value: string) {
  if (value === 'email_verify') return t('admin.emailVerifyCode')
  if (value === 'password_reset') return t('admin.passwordResetCode')
  return value
}

function errorMessage(error: unknown) {
  return localizedApiError(error, t, te, 'admin.genericError')
}
</script>

<template>
  <section class="admin-settings-panel">
    <header
      ><div
        ><h2>{{ t('admin.securityTitle') }}</h2
        ><p>{{ t('admin.securityHint') }}</p></div
      ></header
    >

    <section class="admin-settings-subsection">
      <header
        ><div
          ><h3>{{ t('admin.sessions') }}</h3
          ><p>{{ t('admin.sessionsHint') }}</p></div
        ><span class="admin-date-chip"
          >{{ activeSessions }} {{ t('admin.activeSessions') }}</span
        ></header
      >
      <div v-if="user.sessions?.length" class="admin-record-table-wrap">
        <table class="admin-record-table"
          ><thead
            ><tr
              ><th>{{ t('admin.status') }}</th
              ><th>{{ t('admin.lastSeen') }}</th
              ><th>{{ t('admin.createdAt') }}</th
              ><th>User-Agent</th></tr
            ></thead
          ><tbody>
            <tr v-for="session in user.sessions" :key="session.id"
              ><td :data-label="t('admin.status')"
                ><span
                  class="status-pill"
                  :class="session.active ? 'status-pill-success' : 'status-pill-neutral'"
                  >{{ session.active ? t('admin.activeRecord') : t('admin.inactiveRecord') }}</span
                ></td
              ><td :data-label="t('admin.lastSeen')">{{ formatDate(session.last_seen_at) }}</td
              ><td :data-label="t('admin.createdAt')">{{ formatDate(session.created_at) }}</td
              ><td data-label="User-Agent">{{
                session.user_agent || t('admin.noUserAgent')
              }}</td></tr
            >
          </tbody></table
        >
      </div>
      <p v-else class="admin-context-note">{{ t('admin.noSessions') }}</p>
      <button
        type="button"
        class="secondary-action"
        :disabled="selectedIsSelf || activeSessions === 0 || revokeMutation.isPending.value"
        @click="revokeDialogOpen = true"
        ><ShieldOff aria-hidden="true" />{{ t('admin.revokeSessions') }}</button
      >
      <p v-if="selectedIsSelf" class="admin-context-note">{{ t('admin.selfSessionProtection') }}</p>
    </section>

    <section class="admin-settings-subsection">
      <header
        ><div
          ><h3>{{ t('admin.verificationCodes') }}</h3
          ><p>{{ t('admin.verificationCodesHint') }}</p></div
        ><span class="admin-date-chip">{{ user.verification_codes?.length ?? 0 }}</span></header
      >
      <div v-if="user.verification_codes?.length" class="admin-record-table-wrap">
        <table class="admin-record-table"
          ><thead
            ><tr
              ><th>{{ t('admin.codePurpose') }}</th
              ><th>{{ t('admin.status') }}</th
              ><th>{{ t('admin.attempts') }}</th
              ><th>{{ t('admin.createdAt') }}</th></tr
            ></thead
          ><tbody>
            <tr v-for="code in user.verification_codes" :key="code.id"
              ><td :data-label="t('admin.codePurpose')">{{ codePurpose(code.purpose) }}</td
              ><td :data-label="t('admin.status')">{{
                code.active ? t('admin.activeRecord') : t('admin.inactiveRecord')
              }}</td
              ><td :data-label="t('admin.attempts')">{{ code.attempts }}</td
              ><td :data-label="t('admin.createdAt')">{{ formatDate(code.created_at) }}</td></tr
            >
          </tbody></table
        >
      </div>
      <p v-else class="admin-context-note">{{ t('admin.noVerificationCodes') }}</p>
    </section>

    <section class="admin-danger-zone">
      <div
        ><h3>{{ t('admin.deleteUserTitle') }}</h3
        ><p>{{ t('admin.deleteUserHint') }}</p
        ><span v-if="user.delete_block_reason">{{
          t(`admin.deleteReasons.${user.delete_block_reason}`)
        }}</span></div
      >
      <button
        type="button"
        class="danger-action"
        :disabled="selectedIsSelf || !user.can_delete || deleteMutation.isPending.value"
        @click="deleteDialogOpen = true"
        ><Trash2 aria-hidden="true" />{{ t('admin.deleteUser') }}</button
      >
    </section>

    <div class="admin-feedback" aria-live="polite"
      ><p v-if="message" class="success-copy">{{ message }}</p
      ><p v-if="revokeMutation.error.value" class="error-copy">{{
        errorMessage(revokeMutation.error.value)
      }}</p
      ><p v-if="deleteMutation.error.value" class="error-copy">{{
        errorMessage(deleteMutation.error.value)
      }}</p></div
    >
  </section>

  <AdminConfirmDialog
    :open="revokeDialogOpen"
    :title="t('admin.revokeDialogTitle')"
    :description="t('admin.revokeDialogDescription', { count: activeSessions })"
    :confirm-label="t('admin.revokeSessions')"
    :cancel-label="t('common.cancel')"
    @confirm="revokeMutation.mutate()"
    @cancel="revokeDialogOpen = false"
  />
  <AdminConfirmDialog
    :open="deleteDialogOpen"
    :title="t('admin.deleteDialogTitle')"
    :description="t('admin.deleteDialogDescription')"
    :confirm-label="t('admin.deleteUser')"
    :cancel-label="t('common.cancel')"
    :required-text="user.email"
    :required-label="t('admin.deleteConfirmationLabel')"
    danger
    @confirm="deleteMutation.mutate()"
    @cancel="deleteDialogOpen = false"
  />
</template>
