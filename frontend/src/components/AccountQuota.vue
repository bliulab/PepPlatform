<script setup lang="ts">
import { FlaskConical, RefreshCw } from '@lucide/vue'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { accountQuotaReadouts, nextQuotaReset, type QuotaKind } from '../accountQuota'
import { useAuthStore } from '../stores/auth'
import { useCalendarClock } from '../useCalendarClock'
import { pendingQuotaChanges } from '../resourcePolicy'
import QuotaChangeDetails from './QuotaChangeDetails.vue'

const props = withDefaults(defineProps<{ mode?: 'compact' | 'full' | 'experiment' }>(), {
  mode: 'full',
})
const auth = useAuthStore()
const { t, locale } = useI18n()
const now = useCalendarClock()
const visible = computed(() => auth.authEnabled && Boolean(auth.user))
const exempt = computed(() => auth.quota?.enforcement === 'exempt_admin')
const readouts = computed(() =>
  accountQuotaReadouts(auth.user?.id ?? '', auth.quota, auth.usage, auth.resourceCounts, now.value),
)
const remaining = computed(() =>
  auth.usageUnavailable ? null : (readouts.value.experiments?.remaining ?? null),
)
const keys: QuotaKind[] = ['experiments', 'concurrent', 'inferenceRequests', 'inferenceInputs']
const rows = computed(() =>
  (props.mode === 'experiment' ? keys.slice(0, 2) : keys).map((key) => ({
    key,
    value: auth.usageUnavailable ? null : readouts.value[key],
  })),
)
const resetTime = computed(() =>
  new Intl.DateTimeFormat(locale.value, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(nextQuotaReset(now.value)),
)
</script>

<template>
  <RouterLink
    v-if="visible && mode === 'compact'"
    to="/account/profile#usage"
    class="quota-compact"
    :class="{ 'quota-compact--empty': remaining === 0 }"
    :aria-label="
      exempt
        ? t('resource.adminExempt')
        : remaining === null
          ? t('quota.unavailable')
          : t('quota.remainingToday', { count: remaining })
    "
    :title="
      exempt
        ? t('resource.adminExempt')
        : remaining === null
          ? t('quota.unavailable')
          : t('quota.remainingToday', { count: remaining })
    "
  >
    <FlaskConical aria-hidden="true" />
    <span>{{ t('quota.todayShort') }}</span
    ><strong>{{ exempt ? '∞' : (remaining ?? '—') }}</strong>
  </RouterLink>
  <section
    v-else-if="visible"
    :id="mode === 'full' ? 'usage' : undefined"
    class="account-quota"
    :class="{ 'account-quota--experiment': mode === 'experiment' }"
    :aria-label="t('quota.title')"
    :aria-busy="auth.loading"
  >
    <header class="quota-heading">
      <div
        ><p class="eyebrow">{{ t('quota.eyebrow') }}</p
        ><h2>{{
          t(
            exempt
              ? 'resource.adminUsage'
              : mode === 'experiment'
                ? 'quota.beforeStart'
                : 'quota.title',
          )
        }}</h2></div
      >
      <button
        type="button"
        class="quota-refresh"
        :disabled="auth.loading"
        :aria-label="t('quota.refresh')"
        :title="t('quota.refresh')"
        @click="auth.refreshUsage()"
      >
        <RefreshCw aria-hidden="true" />
      </button>
    </header>
    <dl class="quota-readouts">
      <div v-for="row in rows" :key="row.key" class="quota-readout">
        <dt>{{ t(`quota.${row.key}`) }}</dt>
        <dd
          ><strong :class="{ 'quota-zero': row.value?.remaining === 0 }">{{
            row.value?.unlimited ? '∞' : (row.value?.remaining ?? '—')
          }}</strong
          ><span>{{
            t(
              row.value?.unlimited
                ? 'resource.unlimited'
                : row.key === 'concurrent'
                  ? 'quota.slotsLeft'
                  : 'quota.left',
            )
          }}</span></dd
        >
        <dd class="quota-usage">{{
          row.value
            ? row.value.unlimited
              ? t('resource.usedUnlimited', { used: row.value.used })
              : t('quota.usedOf', { used: row.value.used, limit: row.value.limit })
            : t('quota.unavailable')
        }}</dd>
        <dd class="quota-track" aria-hidden="true"
          ><i
            :style="{
              width: row.value?.limit
                ? `${Math.min(100, (row.value.used / row.value.limit) * 100)}%`
                : '0%',
            }"
        /></dd>
      </div>
    </dl>
    <footer class="quota-foot">
      <span v-if="!exempt">{{ t('quota.nextReset', { time: resetTime }) }}</span>
      <p>{{
        t(
          exempt
            ? 'resource.adminExempt'
            : mode === 'experiment'
              ? 'quota.submissionHint'
              : 'quota.usageHint',
        )
      }}</p>
    </footer>
    <QuotaChangeDetails :changes="pendingQuotaChanges(auth.quota)" pending />
  </section>
</template>

<style scoped>
.quota-compact {
  display: inline-flex;
  min-height: 2.5rem;
  align-items: center;
  gap: 0.4rem;
  padding: 0.4rem 0.65rem;
  border: 1px solid var(--marker-border);
  border-radius: var(--radius-control);
  color: var(--color-accent-dark);
  background: var(--marker-bg);
  white-space: nowrap;
  font-size: 0.7rem;
}
.quota-compact svg {
  width: 0.95rem;
  height: 0.95rem;
}
.quota-compact strong {
  font-size: 0.95rem;
  font-variant-numeric: tabular-nums;
}
.quota-compact:hover {
  border-color: var(--color-accent);
}
.quota-compact:focus-visible,
.quota-refresh:focus-visible {
  outline: 2px solid var(--color-accent);
  outline-offset: 3px;
}
.quota-compact--empty {
  color: var(--color-danger);
  border-color: var(--color-danger-border);
  background: var(--color-danger-bg);
}
.account-quota {
  min-width: 0;
  scroll-margin-top: 6rem;
  border: 1px solid var(--color-line);
  border-radius: var(--radius-panel);
  background: var(--surface-strong);
  padding: 1.35rem;
}
.quota-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}
.quota-heading h2 {
  margin-top: 0.35rem;
  font-size: 1rem;
  font-weight: 650;
  color: var(--color-ink);
}
.quota-refresh {
  display: grid;
  width: 2.75rem;
  height: 2.75rem;
  place-items: center;
  color: var(--color-muted);
  border: 1px solid var(--color-line);
  border-radius: var(--radius-control);
}
.quota-refresh:hover {
  color: var(--color-accent);
  background: var(--surface-muted);
}
.quota-refresh:disabled {
  opacity: 0.5;
  cursor: wait;
}
.quota-refresh svg {
  width: 1rem;
  height: 1rem;
}
.quota-readouts {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 1.15rem;
  gap: 1.4rem 1.75rem;
}
.quota-readout {
  min-width: 0;
}
.quota-readout dt {
  font-size: 0.75rem;
  color: var(--color-muted);
}
.quota-readout dd {
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
  margin-top: 0.4rem;
}
.quota-readout strong {
  font-size: 1.85rem;
  font-weight: 600;
  line-height: 1.15;
  letter-spacing: -0.04em;
  font-variant-numeric: tabular-nums;
  color: var(--color-accent-dark);
}
.quota-readout strong.quota-zero {
  color: var(--color-danger);
}
.quota-readout dd > span {
  font-size: 0.7rem;
  color: var(--color-muted);
}
.quota-readout .quota-usage {
  min-height: 1.1rem;
  font-size: 0.7rem;
  line-height: 1.5;
  color: var(--color-muted);
}
.quota-readout .quota-track {
  overflow: hidden;
  height: 3px;
  margin-top: 0.65rem;
  background: var(--line-soft);
  border-radius: 2px;
}
.quota-track i {
  display: block;
  height: 100%;
  background: var(--color-accent);
  border-radius: inherit;
}
.quota-foot {
  margin-top: 1.2rem;
  border-top: 1px solid var(--line-soft);
  padding-top: 0.85rem;
  font-size: 0.7rem;
  line-height: 1.7;
  color: var(--color-muted);
}
.quota-foot p {
  margin-top: 0.2rem;
}
.account-quota--experiment {
  margin-top: 1.25rem;
  border-color: var(--marker-border);
  background: var(--surface-bg);
}
@media (max-width: 640px) {
  .quota-compact {
    gap: 0.3rem;
    padding-inline: 0.5rem;
  }
  .quota-compact > span {
    display: none;
  }
  .account-quota {
    padding: 1rem;
  }
  .quota-readouts {
    gap: 1.15rem;
  }
}
</style>
