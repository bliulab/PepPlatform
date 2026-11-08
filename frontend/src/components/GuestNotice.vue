<script setup lang="ts">
import { ArrowRight } from '@lucide/vue'
import { useI18n } from 'vue-i18n'
import { useWorkspaceAccess } from '../workspaceAccess'

const { t } = useI18n()
const { isGuest, requireLogin } = useWorkspaceAccess()
</script>

<template>
  <aside v-if="isGuest" class="guest-notice" aria-labelledby="guest-notice-title">
    <div>
      <p id="guest-notice-title" class="guest-notice-title">{{ t('guest.title') }}</p>
      <p class="guest-notice-copy">{{ t('guest.description') }}</p>
    </div>
    <button type="button" class="secondary-action compact-action" @click="requireLogin">
      {{ t('account.login') }}<ArrowRight aria-hidden="true" />
    </button>
  </aside>
</template>
