<script setup lang="ts">
import { Activity, LayoutDashboard, Users, Gauge, HardDrive, Bell } from '@lucide/vue'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'

const { t } = useI18n()
const route = useRoute()
const section = computed(() => String(route.meta.adminSection || 'overview'))
const navigation = computed(() => [
  {
    id: 'overview',
    to: '/admin/overview',
    label: t('admin.overviewNav'),
    icon: LayoutDashboard,
    tone: 'accent',
  },
  { id: 'users', to: '/admin/users', label: t('admin.usersNav'), icon: Users, tone: 'cool' },
  {
    id: 'policies',
    to: '/admin/resources/policies',
    label: t('resource.policies'),
    icon: Gauge,
    tone: 'brass',
  },
  {
    id: 'storage',
    to: '/admin/resources/storage',
    label: t('resource.storage'),
    icon: HardDrive,
    tone: 'moss',
  },
  {
    id: 'announcements',
    to: '/admin/announcements',
    label: t('resource.announcements'),
    icon: Bell,
    tone: 'cool',
  },
  { id: 'audit', to: '/admin/audit', label: t('admin.auditNav'), icon: Activity, tone: 'brass' },
])
</script>

<template>
  <section class="admin-workspace">
    <aside class="admin-sidebar">
      <div class="admin-sidebar-brand">
        <span class="admin-sidebar-logo"><img src="/logo-mark.png" alt="PepPlatform" /></span>
        <div
          ><strong>{{ t('admin.console') }}</strong
          ><small>{{ t('admin.workspace') }}</small></div
        >
      </div>
      <nav :aria-label="t('admin.workspaceNavigation')">
        <RouterLink
          v-for="item in navigation"
          :key="item.id"
          :to="item.to"
          :class="{ active: section === item.id }"
        >
          <component
            :is="item.icon"
            :class="`admin-icon-tone-${item.tone}`"
            aria-hidden="true"
          /><span>{{ item.label }}</span>
        </RouterLink>
      </nav>
      <div class="admin-sidebar-foot"
        ><span>{{ t('admin.adminOnly') }}</span
        ><small>{{ t('admin.utcData') }}</small></div
      >
    </aside>
    <main class="admin-workspace-content"><RouterView /></main>
  </section>
</template>
