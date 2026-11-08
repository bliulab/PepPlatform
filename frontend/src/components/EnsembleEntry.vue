<script setup lang="ts">
import { useQuery } from '@tanstack/vue-query'
import { useI18n } from 'vue-i18n'
import { api } from '../api/client'
import { useWorkspaceAccess } from '../workspaceAccess'
const { t } = useI18n()
const { isGuest, canUseWorkspace } = useWorkspaceAccess()
const capabilities = useQuery({
  queryKey: ['ensemble-capabilities'],
  queryFn: api.ensembleCapabilities,
  enabled: canUseWorkspace,
  staleTime: 30_000,
  retry: false,
})
</script>
<template>
  <RouterLink
    v-if="isGuest || capabilities.data.value?.enabled"
    to="/ensembles/new"
    class="secondary-action compact-action"
    >{{ t('ensemble.title') }}</RouterLink
  >
</template>
