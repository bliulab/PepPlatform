<script setup lang="ts">
import { BadgeCheck, Mail, MailWarning } from '@lucide/vue'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { api } from '../api/client'
import AuthShell from '../components/AuthShell.vue'
import { postAuthRedirect } from '../authNavigation'
import { localizedApiError } from '../errorLocalization'
import { useAuthStore } from '../stores/auth'
import {
  cacheDevelopmentCode,
  clearDevelopmentCode,
  getDevelopmentCode,
  recordVerificationDelivery,
  verificationDelivery,
  verificationNotice,
} from '../verificationDelivery'

const route = useRoute()
const router = useRouter()
const redirect = computed(() => postAuthRedirect(route.query.redirect))
const auth = useAuthStore()
const { t, te } = useI18n()

const initialEmail = typeof route.query.email === 'string' ? route.query.email : ''
const email = ref(initialEmail)
const code = ref('')
const error = ref('')
const notice = computed(() => verificationNotice(email.value))
const noticeKey = computed(() => {
  if (notice.value?.state === 'failed')
    return notice.value.registered ? 'auth.registeredMailFailed' : 'auth.deliveryFailed'
  return `auth.delivery_${notice.value?.state}`
})
const loading = ref(false)
const resendLoading = ref(false)
const debugCode = computed(() => getDevelopmentCode(email.value))

async function submit() {
  error.value = ''
  loading.value = true
  try {
    await auth.verifyEmail(email.value, code.value)
    verificationDelivery.value = null
    clearDevelopmentCode(email.value)
    router.push(redirect.value)
  } catch (err) {
    error.value = localizedApiError(err, t, te, 'auth.verifyFailed')
  } finally {
    loading.value = false
  }
}

async function resend() {
  error.value = ''
  verificationDelivery.value = null
  resendLoading.value = true
  try {
    const response = await api.resendVerification({ email: email.value })
    recordVerificationDelivery(response, email.value)
    cacheDevelopmentCode(response, email.value)
  } catch (err) {
    error.value = localizedApiError(err, t, te, 'auth.resendFailed')
  } finally {
    resendLoading.value = false
  }
}
</script>

<template>
  <AuthShell>
    <h1>{{ t('auth.verifyTitle') }}</h1>
    <p class="muted-copy">{{ t('auth.verifySubtitle') }}</p>
    <div
      v-if="notice"
      class="verification-notice"
      :class="{ 'verification-notice--warning': notice.state === 'failed' }"
      :role="notice.state === 'failed' ? 'alert' : 'status'"
    >
      <MailWarning v-if="notice.state === 'failed'" aria-hidden="true" />
      <Mail v-else aria-hidden="true" />
      <p>{{ t(noticeKey, { email: notice.email }) }}</p>
    </div>
    <form class="auth-form" @submit.prevent="submit">
      <label class="form-field">
        <span>{{ t('auth.email') }}</span>
        <input v-model="email" type="email" autocomplete="email" required />
      </label>
      <label class="form-field">
        <span>{{ t('auth.verificationCode') }}</span>
        <input
          v-model="code"
          inputmode="numeric"
          maxlength="6"
          autocomplete="one-time-code"
          required
        />
      </label>
      <p v-if="debugCode" class="dev-code">{{ t('auth.developmentCode', { code: debugCode }) }}</p>
      <p v-if="error" class="error-copy">{{ error }}</p>
      <button class="primary-action w-full" type="submit" :disabled="loading">
        <BadgeCheck aria-hidden="true" />
        {{ loading ? t('auth.verifying') : t('auth.verifyAndContinue') }}
      </button>
    </form>
    <div class="auth-links">
      <button class="text-link" type="button" :disabled="resendLoading || !email" @click="resend">
        {{ resendLoading ? t('auth.resending') : t('auth.resendCode') }}
      </button>
      <RouterLink :to="{ path: '/login', query: { redirect } }">{{
        t('auth.backToLogin')
      }}</RouterLink>
    </div>
  </AuthShell>
</template>
