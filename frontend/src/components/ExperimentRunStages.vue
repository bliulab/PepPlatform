<script setup lang="ts">
import { Check } from '@lucide/vue'

export interface ExperimentRunStageItem {
  key: string
  label: string
  state: 'complete' | 'active' | 'pending' | 'stopped'
}

defineProps<{
  items: ExperimentRunStageItem[]
  label: string
}>()
</script>

<template>
  <ol class="run-stage-rail" :aria-label="label">
    <li
      v-for="(item, index) in items"
      :key="item.key"
      class="run-stage-item"
      :class="`run-stage-${item.state}`"
      :aria-current="item.state === 'active' ? 'step' : undefined"
    >
      <span class="run-stage-marker" aria-hidden="true">
        <Check v-if="item.state === 'complete'" />
        <span v-else>{{ index + 1 }}</span>
      </span>
      <span class="run-stage-label">{{ item.label }}</span>
    </li>
  </ol>
</template>

<style scoped>
.run-stage-rail {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.5rem;
  margin: 0;
  padding: 0;
  list-style: none;
}

.run-stage-item {
  position: relative;
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  align-items: center;
  gap: 0.5rem;
  min-width: 0;
  border: 1px solid var(--color-line);
  border-radius: 0.75rem;
  background: color-mix(in srgb, var(--surface-strong) 92%, var(--color-line));
  padding: 0.55rem 0.65rem;
  color: var(--color-muted);
  transition:
    border-color 180ms ease,
    background-color 180ms ease,
    color 180ms ease;
}

.run-stage-marker {
  display: inline-grid;
  width: 1.55rem;
  height: 1.55rem;
  place-items: center;
  border-radius: 999px;
  background: var(--color-line);
  color: var(--color-muted);
  font-size: 0.7rem;
  font-weight: 700;
}

.run-stage-marker :deep(svg) {
  width: 0.9rem;
  height: 0.9rem;
}

.run-stage-label {
  overflow: hidden;
  font-size: 0.72rem;
  font-weight: 600;
  line-height: 1.25;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.run-stage-complete {
  border-color: var(--color-success-border);
  background: var(--color-success-bg);
  color: var(--color-success-text);
}

.run-stage-complete .run-stage-marker {
  background: var(--color-success-text);
  color: white;
}

.run-stage-active {
  border-color: var(--color-accent);
  background: color-mix(in srgb, var(--color-accent) 8%, var(--surface-strong));
  color: var(--color-ink);
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--color-accent) 35%, transparent);
}

.run-stage-active .run-stage-marker {
  background: var(--color-accent);
  color: white;
  box-shadow: 0 0 0 0.2rem color-mix(in srgb, var(--color-accent) 16%, transparent);
}

.run-stage-stopped {
  border-color: var(--color-warning-border);
  background: var(--color-warning-bg);
  color: var(--color-warning-text);
}

@media (max-width: 720px) {
  .run-stage-rail {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (prefers-reduced-motion: reduce) {
  .run-stage-item {
    transition: none;
  }
}
</style>
