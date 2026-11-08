<script setup lang="ts">
defineProps<{
  eyebrow?: string
  title: string
  subtitle?: string
  compact?: boolean
}>()
</script>

<template>
  <header class="page-header" :class="{ 'page-header--compact': compact }">
    <div class="page-header-copy">
      <p v-if="eyebrow" class="eyebrow">{{ eyebrow }}</p>
      <h1>{{ title }}</h1>
      <p v-if="subtitle && !compact" class="page-header-subtitle">{{ subtitle }}</p>
    </div>
    <div v-if="$slots.actions" class="page-header-actions">
      <slot name="actions" />
    </div>
  </header>
</template>

<style scoped>
.page-header {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  justify-content: space-between;
  gap: 0.75rem 1.5rem;
  border-bottom: 1px solid var(--color-line);
  padding-bottom: 0.9rem;
}

.page-header h1 {
  margin-top: 0.25rem;
  font-size: clamp(1.35rem, 2vw, 1.6rem);
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.25;
  color: var(--color-ink);
}

.page-header-subtitle {
  margin-top: 0.4rem;
  max-width: 44rem;
  color: var(--color-muted);
  font-size: 0.82rem;
  line-height: 1.65;
}

.page-header-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
}

.page-header--compact {
  align-items: center;
}
</style>
