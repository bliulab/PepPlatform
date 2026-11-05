<script setup lang="ts">
import { KeyRound, Send } from '@lucide/vue'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { api } from '../api/client'
import AuthShell from '../components/AuthShell.vue'
import PasswordInput from '../components/PasswordInput.vue'
import {
  cacheDevelopmentCode,
  clearDevelopmentCode,
  getDevelopmentCode,
} from '../verificationDelivery'
import { postAuthRedirect } from '../authNavigation'
import { localizedApiError } from '../errorLocalization'

const route = useRoute()
const router = useRouter()
const redirect = computed(() => postAuthRedirect(route.query.redirect))
const { t, te } = useI18n()

const email = ref('')
const code = ref('')
const password = ref('')
const requested = ref(false)
const error = ref('')
const message = ref('')
const loading = ref(false)
const debugCode = computed(() => getDevelopmentCode(email.value, 'pepplatform-reset'))

async function requestReset() {
  error.value = ''
  message.value = ''
  loading.value = true
  try {
    const response = await api.requestPasswordReset({ email: email.value })
    cacheDevelopmentCode(response, email.value, 'pepplatform-reset')
    requested.value = true
    message.value = t('auth.resetRequested')
  } catch (err) {
    error.value = localizedApiError(err, t, te, 'auth.resetRequestFailed')
  } finally {
    loading.value = false
  }
}

async function confirmReset() {
  error.value = ''
  loading.value = true
  try {
    await api.confirmPasswordReset({
      email: email.value,
      code: code.value,
      password: password.value,
    })
    clearDevelopmentCode(email.value, 'pepplatform-reset')
    router.push({ path: '/login', query: { redirect: redirect.value } })
  } catch (err) {
    error.value = localizedApiError(err, t, te, 'auth.resetFailed')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <AuthShell>
    <h1>{{ t('auth.resetTitle') }}</h1>
    <p class="muted-copy">{{ t('auth.resetSubtitle') }}</p>
    <form v-if="!requested" class="auth-form" @submit.prevent="requestReset">
      <label class="form-field">
        <span>{{ t('auth.email') }}</span>
        <input v-model="email" type="email" autocomplete="email" required />
      </label>
      <p v-if="error" class="error-copy">{{ error }}</p>
      <button class="primary-action w-full" type="submit" :disabled="loading">
        <Send aria-hidden="true" />
        {{ loading ? t('auth.sending') : t('auth.sendCode') }}
      </button>
    </form>
    <form v-else class="auth-form" @submit.prevent="confirmReset">
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
      <label class="form-field">
        <span>{{ t('auth.newPassword') }}</span>
        <PasswordInput v-model="password" minlength="8" autocomplete="new-password" required />
      </label>
      <p v-if="debugCode" class="dev-code">{{ t('auth.developmentCode', { code: debugCode }) }}</p>
      <p v-if="message" class="success-copy">{{ message }}</p>
      <p v-if="error" class="error-copy">{{ error }}</p>
      <button class="primary-action w-full" type="submit" :disabled="loading">
        <KeyRound aria-hidden="true" />
        {{ loading ? t('auth.resetting') : t('auth.resetPassword') }}
      </button>
    </form>
    <div class="auth-links auth-links--single">
      <RouterLink :to="{ path: '/login', query: { redirect } }">{{
        t('auth.backToLogin')
      }}</RouterLink>
    </div>
  </AuthShell>
</template>
