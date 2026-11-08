<script setup lang="ts">
import { UserPlus } from '@lucide/vue'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { ApiError, api } from '../api/client'
import AuthShell from '../components/AuthShell.vue'
import PasswordInput from '../components/PasswordInput.vue'
import { cacheDevelopmentCode, recordVerificationDelivery } from '../verificationDelivery'
import { postAuthRedirect } from '../authNavigation'
import { localizedApiError } from '../errorLocalization'
import { normalizeLocale } from '../localization'

const route = useRoute()
const router = useRouter()
const redirect = computed(() => postAuthRedirect(route.query.redirect))
const { locale, t, te } = useI18n()

const email = ref('')
const password = ref('')
const error = ref<unknown>(null)
const errorText = computed(() =>
  error.value ? localizedApiError(error.value, t, te, 'auth.registerFailed') : '',
)
const loading = ref(false)
const registrationFull = ref(false)
const deploymentUrl = computed(
  () =>
    `https://bliulab.github.io/pepplatform-docs/${locale.value === 'en' ? 'en/' : ''}install/docker`,
)

async function submit() {
  error.value = null
  registrationFull.value = false
  loading.value = true
  try {
    const response = await api.register({
      email: email.value,
      password: password.value,
      preferred_locale: normalizeLocale(locale.value),
    })
    recordVerificationDelivery(response, email.value, true)
    cacheDevelopmentCode(response, email.value)
    router.push({
      path: '/verify-email',
      query: { email: response.email || email.value, redirect: redirect.value },
    })
  } catch (err) {
    registrationFull.value = err instanceof ApiError && err.quotaType === 'daily_registration_limit'
    error.value = err
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <AuthShell>
    <h1>{{ t('auth.createTitle') }}</h1>
    <p class="muted-copy">{{ t('auth.createSubtitle') }}</p>
    <form id="register-form" class="auth-form" @submit.prevent="submit">
      <label class="form-field" for="register-email">
        <span>{{ t('auth.email') }}</span>
        <input
          id="register-email"
          v-model="email"
          name="email"
          type="email"
          autocomplete="username"
          required
        />
      </label>
      <label class="form-field" for="register-password">
        <span>{{ t('auth.password') }}</span>
        <PasswordInput
          id="register-password"
          v-model="password"
          name="password"
          autocomplete="new-password"
          minlength="8"
          required
        />
      </label>
      <p v-if="errorText" class="error-copy">{{ errorText }}</p>
      <aside v-if="registrationFull" class="resource-self-host"
        ><p>{{ t('resource.selfHostHint') }}</p
        ><a :href="deploymentUrl" target="_blank" rel="noopener noreferrer">{{
          t('resource.selfHostLink')
        }}</a></aside
      >
      <button class="primary-action w-full" type="submit" :disabled="loading">
        <UserPlus aria-hidden="true" />
        {{ loading ? t('auth.registering') : t('auth.registerAndSend') }}
      </button>
    </form>
    <div class="auth-links auth-links--single">
      <RouterLink :to="{ path: '/login', query: { redirect } }">{{
        t('auth.existingAccount')
      }}</RouterLink>
    </div>
  </AuthShell>
</template>
