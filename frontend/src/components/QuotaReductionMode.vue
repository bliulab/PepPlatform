<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import type { DailyReductionMode } from '../resourceTypes'

withDefaults(defineProps<{ name: string; disabled?: boolean }>(), { disabled: false })
const mode = defineModel<DailyReductionMode>({ required: true })
const { t } = useI18n()
</script>

<template>
  <fieldset class="resource-reduction-mode" :disabled="disabled" :aria-describedby="`${name}-hint`">
    <legend>{{ t('resource.reductionTiming') }}</legend>
    <label class="resource-check">
      <input v-model="mode" type="radio" :name="name" value="next_reset" />
      <span>{{ t('resource.nextResetMode') }}</span>
    </label>
    <label class="resource-check">
      <input v-model="mode" type="radio" :name="name" value="immediate" />
      <span>{{ t('resource.forceImmediateMode') }}</span>
    </label>
    <p :id="`${name}-hint`" class="resource-help">{{ t('resource.dailyChangeHint') }}</p>
  </fieldset>
</template>

<style scoped>
.resource-reduction-mode {
  display: grid;
  gap: 0.75rem;
  min-width: 0;
  margin: 1rem 0;
  padding: 1rem;
  border: 1px solid var(--color-line);
  border-radius: var(--radius-control);
}
.resource-reduction-mode legend {
  padding: 0 0.25rem;
  font-size: 0.85rem;
  font-weight: 600;
}
.resource-reduction-mode .resource-check {
  align-items: start;
  line-height: 1.5;
}
.resource-reduction-mode input {
  margin-top: 0.2rem;
}
.resource-reduction-mode .resource-help {
  margin: 0;
}
</style>
