<script setup lang="ts">
import { UserPlus } from '@lucide/vue'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { api } from '../api/client'
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
const error = ref('')
const loading = ref(false)

async function submit() {
  error.value = ''
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
    error.value = localizedApiError(err, t, te, 'auth.registerFailed')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <AuthShell>
    <h1>{{ t('auth.createTitle') }}</h1>
    <p class="muted-copy">{{ t('auth.createSubtitle') }}</p>
    <form class="auth-form" @submit.prevent="submit">
      <label class="form-field">
        <span>{{ t('auth.email') }}</span>
        <input v-model="email" type="email" autocomplete="email" required />
      </label>
      <label class="form-field">
        <span>{{ t('auth.password') }}</span>
        <PasswordInput v-model="password" autocomplete="new-password" minlength="8" required />
      </label>
      <p v-if="error" class="error-copy">{{ error }}</p>
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
