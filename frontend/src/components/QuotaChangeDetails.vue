<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { formatLocalizedDate } from '../localization'
import { quotaFields } from '../resourcePolicy'
import type { QuotaChanges } from '../resourceTypes'

const props = defineProps<{ changes: QuotaChanges; pending?: boolean }>()
const { t, locale } = useI18n()
const rows = computed(() =>
  quotaFields.flatMap((field) => {
    const change = props.changes[field.key]
    return change ? [{ field, change }] : []
  }),
)
function amount(value: number, divisor: number) {
  return new Intl.NumberFormat(locale.value, { maximumFractionDigits: 4 }).format(value / divisor)
}
</script>

<template>
  <section v-if="rows.length" class="resource-quota-change" role="note">
    <h3 v-if="pending">{{ t('resource.pendingTitle') }}</h3>
    <dl>
      <div v-for="{ field, change } in rows" :key="field.key">
        <dt>{{ t(`resource.fields.${field.key}`) }}</dt>
        <dd>
          <strong
            >{{
              t('resource.changeValues', {
                before: amount(change.effective_from ?? change.from, field.divisor),
                after: amount(change.effective_to ?? change.to, field.divisor),
              })
            }}<small v-if="field.unit">
              {{ field.unit === 'days' ? t('resource.days') : field.unit }}</small
            ></strong
          >
          <span>{{
            t('resource.effectiveAt', {
              time: formatLocalizedDate(change.effective_at, locale, {
                dateStyle: 'medium',
                timeStyle: 'short',
              }),
            })
          }}</span>
          <span v-if="change.daily_reduction_mode === 'immediate'">{{
            t('resource.effectiveImmediately')
          }}</span>
        </dd>
      </div>
    </dl>
    <p>{{ t(pending ? 'resource.dailyChangeHint' : 'resource.quotaChangeHint') }}</p>
  </section>
</template>

<style scoped>
.resource-quota-change {
  margin-top: 1rem;
  padding: 1rem;
  border: 1px solid var(--color-line);
  border-radius: var(--radius-control);
  background: var(--surface-muted);
  font-size: 0.8rem;
}
.resource-quota-change h3 {
  margin: 0 0 0.75rem;
  font-size: 0.85rem;
}
.resource-quota-change dl {
  display: grid;
  gap: 0.75rem;
}
.resource-quota-change dl > div {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  align-items: start;
  gap: 0.5rem 1rem;
}
.resource-quota-change dt {
  color: var(--color-muted);
}
.resource-quota-change dd {
  display: grid;
  gap: 0.25rem;
  margin: 0;
}
.resource-quota-change dd span,
.resource-quota-change p {
  color: var(--color-muted);
  line-height: 1.6;
}
.resource-quota-change p {
  margin-top: 0.75rem;
}
@media (max-width: 640px) {
  .resource-quota-change dl > div {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
