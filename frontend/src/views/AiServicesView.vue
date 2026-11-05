<script setup lang="ts">
import { KeyRound } from '@lucide/vue'
import { useI18n } from 'vue-i18n'
import AiProviderSettings from '../components/AiProviderSettings.vue'
import PageHeader from '../components/PageHeader.vue'

const { t } = useI18n()
</script>

<template>
  <section class="ai-services-page">
    <PageHeader
      :eyebrow="t('aiServices.eyebrow')"
      :title="t('aiServices.title')"
      :subtitle="t('aiServices.subtitle')"
    >
      <template #actions>
        <span class="ai-services-security-note">
          <KeyRound aria-hidden="true" />
          {{ t('aiServices.keyPrivacy') }}
        </span>
      </template>
    </PageHeader>

    <AiProviderSettings />
  </section>
</template>

<style scoped>
.ai-services-page {
  display: grid;
  gap: 1.25rem;
  padding: 1rem 0;
}

.ai-services-security-note {
  display: inline-flex;
  min-height: 2.25rem;
  align-items: center;
  gap: 0.5rem;
  border: 1px solid var(--line-soft);
  border-radius: 0.375rem;
  background: var(--surface-muted);
  padding: 0.45rem 0.7rem;
  color: var(--color-muted);
  font-size: 0.75rem;
  font-weight: 700;
}

.ai-services-security-note svg {
  width: 0.95rem;
  height: 0.95rem;
  color: var(--color-moss);
}
</style>
