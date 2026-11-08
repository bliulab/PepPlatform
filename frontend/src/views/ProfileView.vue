<script setup lang="ts">
import { Save } from '@lucide/vue'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { localizedApiError } from '../errorLocalization'
import PageHeader from '../components/PageHeader.vue'
import AccountQuota from '../components/AccountQuota.vue'
import AccountResources from '../components/AccountResources.vue'
import ModelQuotaSummary from '../components/ModelQuotaSummary.vue'
import { useAuthStore } from '../stores/auth'

const { t, te } = useI18n()
const auth = useAuthStore()
const displayName = ref(auth.user?.display_name || '')
const savingProfile = ref(false)
const message = ref('')
const errorMessage = ref('')
const profileMeta = computed(() => {
  if (!auth.user) return ''
  const role = auth.user.role === 'admin' ? t('account.admin') : t('account.user')
  const verification = auth.user.email_verified ? t('account.verified') : t('account.unverified')
  return `${role} · ${verification}`
})
const displayNameChanged = computed(
  () => displayName.value.trim() !== (auth.user?.display_name || '').trim(),
)

function errorText(error: unknown) {
  return localizedApiError(error, t, te, 'profile.genericError')
}

function clearFeedback() {
  message.value = ''
  errorMessage.value = ''
}

async function saveProfile() {
  clearFeedback()
  savingProfile.value = true
  try {
    await auth.updateProfile(displayName.value.trim())
    message.value = t('profile.profileSaved')
  } catch (error) {
    errorMessage.value = errorText(error)
  } finally {
    savingProfile.value = false
  }
}

watch(
  () => auth.user?.display_name,
  (nextName) => {
    displayName.value = nextName || ''
  },
)
</script>

<template>
  <section class="profile-shell">
    <PageHeader
      :eyebrow="t('profile.eyebrow')"
      :title="t('profile.title')"
      :subtitle="t('profile.subtitle')"
    />

    <div class="profile-grid">
      <section class="panel" :aria-label="t('profile.detailsSection')">
        <form class="auth-form" @submit.prevent="saveProfile">
          <label class="form-field">
            <span>{{ t('profile.displayName') }}</span>
            <input
              v-model="displayName"
              type="text"
              maxlength="40"
              autocomplete="nickname"
              required
            />
          </label>

          <div class="profile-readonly-grid">
            <div class="profile-readonly-field">
              <span>{{ t('profile.email') }}</span>
              <strong>{{ auth.user?.email }}</strong>
            </div>
            <div class="profile-readonly-field">
              <span>{{ t('profile.status') }}</span>
              <strong>{{ profileMeta }}</strong>
            </div>
          </div>

          <div class="profile-actions">
            <button
              class="primary-action"
              type="submit"
              :disabled="savingProfile || !displayNameChanged"
            >
              <Save aria-hidden="true" />
              {{ savingProfile ? t('profile.saving') : t('profile.saveProfile') }}
            </button>
          </div>
        </form>

        <p v-if="message" class="success-card mt-5" role="status">{{ message }}</p>
        <p v-if="errorMessage" class="error-card mt-5" role="alert">{{ errorMessage }}</p>
      </section>
      <AccountQuota />
      <ModelQuotaSummary />
      <AccountResources />
    </div>
  </section>
</template>
