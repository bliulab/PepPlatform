<script setup lang="ts">
import { LogIn } from '@lucide/vue'
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import AuthShell from '../components/AuthShell.vue'
import PasswordInput from '../components/PasswordInput.vue'
import { postAuthRedirect } from '../authNavigation'
import { localizedApiError } from '../errorLocalization'
import { useAuthStore } from '../stores/auth'

const REMEMBER_EMAIL_KEY = 'pepplatform:remember-email'
const REMEMBER_EMAIL_VALUE_KEY = 'pepplatform:remember-email:value'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const { t, te } = useI18n()

const email = ref('')
const password = ref('')
const rememberEmail = ref(false)
const error = ref('')
const loading = ref(false)
const redirect = computed(() => postAuthRedirect(route.query.redirect))

function getStoredValue(key: string) {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

function setStoredValue(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value)
  } catch {
    // Login should still work when local storage is disabled.
  }
}

function removeStoredValue(key: string) {
  try {
    window.localStorage.removeItem(key)
  } catch {
    // Login should still work when local storage is disabled.
  }
}

onMounted(() => {
  rememberEmail.value = getStoredValue(REMEMBER_EMAIL_KEY) === '1'
  if (rememberEmail.value) {
    email.value = getStoredValue(REMEMBER_EMAIL_VALUE_KEY) || ''
  }
})

function persistRememberedEmail(value: string) {
  if (rememberEmail.value) {
    setStoredValue(REMEMBER_EMAIL_KEY, '1')
    setStoredValue(REMEMBER_EMAIL_VALUE_KEY, value)
    return
  }
  removeStoredValue(REMEMBER_EMAIL_KEY)
  removeStoredValue(REMEMBER_EMAIL_VALUE_KEY)
}

async function submit() {
  error.value = ''
  loading.value = true
  const normalizedEmail = email.value.trim().toLowerCase()
  try {
    await auth.login(normalizedEmail, password.value)
    email.value = normalizedEmail
    persistRememberedEmail(normalizedEmail)
    router.push(redirect.value)
  } catch (err) {
    error.value = localizedApiError(err, t, te, 'auth.loginFailed')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <AuthShell>
    <h1>{{ t('auth.signInTitle') }}</h1>
    <p class="muted-copy">{{ t('auth.signInSubtitle') }}</p>
    <form class="auth-form" @submit.prevent="submit">
      <label class="form-field">
        <span>{{ t('auth.email') }}</span>
        <input v-model="email" type="email" autocomplete="email" required />
      </label>
      <label class="form-field">
        <span>{{ t('auth.password') }}</span>
        <PasswordInput v-model="password" autocomplete="current-password" required />
      </label>
      <div class="auth-form-foot">
        <label class="remember-row">
          <input v-model="rememberEmail" type="checkbox" />
          <span>{{ t('auth.rememberEmail') }}</span>
        </label>
        <RouterLink class="auth-inline-link" :to="{ path: '/password-reset', query: { redirect } }">
          {{ t('auth.forgotPassword') }}
        </RouterLink>
      </div>
      <p v-if="error" class="error-copy">{{ error }}</p>
      <button class="primary-action w-full" type="submit" :disabled="loading">
        <LogIn aria-hidden="true" />
        {{ loading ? t('auth.signingIn') : t('auth.signIn') }}
      </button>
    </form>
    <div class="auth-links auth-links--single">
      <RouterLink :to="{ path: '/register', query: { redirect } }">{{
        t('auth.registerLink')
      }}</RouterLink>
    </div>
  </AuthShell>
</template>
