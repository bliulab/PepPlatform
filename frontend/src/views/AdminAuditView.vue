<script setup lang="ts">
import { RefreshCw } from '@lucide/vue'
import { useQueryClient } from '@tanstack/vue-query'
import { useI18n } from 'vue-i18n'
import AdminAuditTable from '../components/AdminAuditTable.vue'

const { t } = useI18n()
const queryClient = useQueryClient()

function refresh() {
  void queryClient.invalidateQueries({ queryKey: ['admin-audit-logs'] })
  void queryClient.invalidateQueries({ queryKey: ['admin-audit-event-types'] })
}
</script>

<template>
  <section class="admin-page admin-audit-page">
    <header class="admin-page-header"
      ><div
        ><p>{{ t('admin.eyebrow') }}</p
        ><h1>{{ t('admin.auditTitle') }}</h1
        ><span>{{ t('admin.auditSubtitle') }}</span></div
      ><button
        type="button"
        class="icon-action"
        :title="t('admin.refresh')"
        :aria-label="t('admin.refresh')"
        @click="refresh"
        ><RefreshCw aria-hidden="true" /></button
    ></header>
    <AdminAuditTable />
  </section>
</template>
