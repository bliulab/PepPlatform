<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import type { ExperimentAnalysis } from '../types'
import { buildPerformanceSummary } from './performanceSummary'

const props = defineProps<{
  summary: Record<string, unknown>
  analysis?: ExperimentAnalysis
  taskType: string
}>()

const { t } = useI18n()
const model = computed(() => buildPerformanceSummary(props.summary, props.analysis, props.taskType))
const hasSupplemental = computed(
  () => model.value.testSupplemental.length > 0 || model.value.validationSupplemental.length > 0,
)

const RADIUS = 42
const CIRCUMFERENCE = 2 * Math.PI * RADIUS
const ANIMATION_MS = 600

const animatedValues = ref<number[]>([])

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3)
}

function targetValues(): number[] {
  return model.value.primary.map((m) => (m.value !== undefined ? m.value : 0))
}

function syncLength() {
  animatedValues.value = targetValues().map(() => 0)
}

function animateGauges() {
  const targets = targetValues()
  syncLength()
  const starts = animatedValues.value.slice()
  const startTime = performance.now()

  function tick(now: number) {
    const elapsed = now - startTime
    const progress = Math.min(elapsed / ANIMATION_MS, 1)
    const eased = easeOutCubic(progress)
    animatedValues.value = starts.map((s, i) => s + (targets[i] - s) * eased)
    if (progress < 1) requestAnimationFrame(tick)
  }
  requestAnimationFrame(tick)
}

onMounted(() => {
  animateGauges()
})

watch(
  () => props.summary,
  () => {
    animateGauges()
  },
  { deep: true },
)

function gaugeOffset(index: number): number {
  const v = animatedValues.value[index] ?? 0
  return CIRCUMFERENCE * (1 - Math.max(0, Math.min(1, v)))
}

function formatMetric(value?: number) {
  return value === undefined ? '—' : value.toFixed(4)
}
</script>

<template>
  <div class="performance-summary">
    <header class="summary-header">
      <div>
        <p class="eyebrow">{{ t('detail.performanceSummary') }}</p>
        <h2 class="summary-title">{{ t('detail.independentTestResults') }}</h2>
        <p class="summary-copy">{{ t('detail.finalGeneralizationHint') }}</p>
      </div>
      <span class="summary-stamp" aria-hidden="true">{{ t('detail.finalTestStamp') }}</span>
    </header>

    <section class="selection-rail" :aria-label="t('detail.modelSelectionBasis')">
      <div class="selection-rule" aria-hidden="true" />
      <div>
        <div class="selection-kicker">{{ t('detail.modelSelectionBasis') }}</div>
        <div class="selection-metric">
          <span>{{ t(model.selection.labelKey) }}</span>
          <strong>{{ formatMetric(model.selection.value) }}</strong>
        </div>
      </div>
      <div v-if="model.selection.bestEpoch !== undefined" class="epoch-block">
        <span>{{ t('detail.bestEpoch') }}</span>
        <strong>{{ model.selection.bestEpoch }}</strong>
      </div>
      <p class="selection-note">{{ t('detail.selectionOnlyHint') }}</p>
    </section>

    <div class="gauge-grid" :aria-label="t('detail.independentTestResults')">
      <article
        v-for="(metric, index) in model.primary"
        :key="metric.id"
        class="gauge-card"
        :class="{ 'gauge-card--primary': index === 0 }"
        :style="{ '--gauge-delay': `${index * 80}ms` }"
      >
        <div class="gauge-ring-wrap">
          <svg class="gauge-ring" viewBox="0 0 100 100" aria-hidden="true">
            <circle class="gauge-track" :cx="50" :cy="50" :r="RADIUS" />
            <circle
              class="gauge-fill"
              :cx="50"
              :cy="50"
              :r="RADIUS"
              :stroke-dasharray="CIRCUMFERENCE"
              :stroke-dashoffset="gaugeOffset(index)"
            />
          </svg>
          <div class="gauge-value">{{ formatMetric(metric.value) }}</div>
        </div>
        <div class="gauge-label">{{ t(metric.labelKey) }}</div>
      </article>
    </div>

    <details v-if="hasSupplemental" class="supplemental-details">
      <summary>
        <span>
          <strong>{{ t('detail.supplementalMetrics') }}</strong>
          <small>{{ t('detail.supplementalMetricsHint') }}</small>
        </span>
        <span class="details-icon" aria-hidden="true">＋</span>
      </summary>
      <div class="supplemental-grid">
        <section v-if="model.testSupplemental.length" class="supplemental-group">
          <h3>{{ t('detail.independentTestSupplement') }}</h3>
          <dl>
            <div v-for="metric in model.testSupplemental" :key="metric.id" class="supplemental-row">
              <dt>{{ t(metric.labelKey) }}</dt>
              <dd>{{ formatMetric(metric.value) }}</dd>
            </div>
          </dl>
        </section>
        <section
          v-if="model.validationSupplemental.length"
          class="supplemental-group validation-group"
        >
          <h3>{{ t('detail.validationDiagnostics') }}</h3>
          <dl>
            <div
              v-for="metric in model.validationSupplemental"
              :key="metric.id"
              class="supplemental-row"
            >
              <dt>{{ t(metric.labelKey) }}</dt>
              <dd>{{ formatMetric(metric.value) }}</dd>
            </div>
          </dl>
        </section>
      </div>
    </details>
  </div>
</template>

<style scoped>
.performance-summary {
  position: relative;
}

.summary-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1.25rem;
}

.summary-title {
  margin-top: 0.65rem;
  font-family: Georgia, 'Times New Roman', serif;
  font-size: clamp(1.7rem, 3.4vw, 2.55rem);
  font-weight: 700;
  letter-spacing: -0.035em;
  line-height: 1;
}

.summary-copy {
  max-width: 38rem;
  margin-top: 0.75rem;
  color: var(--color-muted);
  font-size: 0.875rem;
  line-height: 1.65;
}

.summary-stamp {
  flex: 0 0 auto;
  border: 1px solid var(--color-accent);
  padding: 0.45rem 0.65rem;
  color: var(--color-accent);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  transform: rotate(2deg);
}

/* --- Selection Rail --- */
.selection-rail {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 0.75rem 1.5rem;
  align-items: center;
  margin-top: 1.4rem;
  overflow: hidden;
  border: 1px solid var(--line-soft);
  border-left: 3px solid var(--color-accent);
  border-radius: 0.65rem;
  padding: 0.85rem 1rem 0.85rem 1.2rem;
  background: var(--surface-bg);
}

.selection-rule {
  display: none;
}

.selection-kicker,
.epoch-block span {
  color: var(--color-muted);
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.selection-metric {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.45rem 0.8rem;
  margin-top: 0.25rem;
  font-size: 0.82rem;
  font-weight: 700;
}

.selection-metric strong,
.epoch-block strong {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: var(--color-ink);
  font-size: 1.1rem;
  font-variant-numeric: tabular-nums;
}

.epoch-block {
  display: grid;
  justify-items: end;
  gap: 0.15rem;
}

.selection-note {
  grid-column: 1 / -1;
  color: var(--color-muted);
  font-size: 0.72rem;
  line-height: 1.5;
  margin-top: 0.2rem;
}

/* --- Gauge Grid --- */
.gauge-grid {
  display: grid;
  gap: 0.7rem;
  margin-top: 1rem;
}

.gauge-card {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.55rem;
  border: 1px solid var(--line-soft);
  border-radius: 0.75rem;
  background: var(--surface-bg);
  padding: 1rem 0.8rem;
  min-height: 8rem;
  transition:
    transform 180ms ease,
    border-color 180ms ease,
    box-shadow 180ms ease;
  animation: gauge-arrive 520ms both cubic-bezier(0.22, 1, 0.36, 1);
  animation-delay: var(--gauge-delay, 0ms);
}

.gauge-card:hover {
  border-color: color-mix(in srgb, var(--color-accent) 62%, var(--color-line));
  box-shadow: var(--shadow-soft);
  transform: translateY(-2px);
}

.gauge-card--primary {
  border-top: 0.22rem solid var(--color-accent);
  background:
    linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px),
    radial-gradient(
      circle at 50% 0%,
      color-mix(in srgb, var(--color-accent) 10%, transparent),
      transparent 60%
    ),
    var(--surface-bg);
  background-size:
    18px 18px,
    18px 18px,
    auto,
    auto;
}

/* --- Gauge Ring --- */
.gauge-ring-wrap {
  position: relative;
  width: 5.5rem;
  height: 5.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.gauge-ring {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  transform: rotate(-90deg);
}

.gauge-track {
  fill: none;
  stroke: var(--line-soft);
  stroke-width: 5;
}

.gauge-fill {
  fill: none;
  stroke: var(--color-brass);
  stroke-width: 5;
  stroke-linecap: round;
  transition: stroke-dashoffset 0.6s cubic-bezier(0.22, 1, 0.36, 1);
  transition-delay: var(--gauge-delay, 0ms);
}

.gauge-card--primary .gauge-fill {
  stroke: var(--color-accent);
}

.gauge-card:hover .gauge-fill {
  stroke-width: 6;
}

.gauge-value {
  position: relative;
  z-index: 1;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: clamp(1rem, 2.2vw, 1.25rem);
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.04em;
  color: var(--color-ink);
}

.gauge-label {
  color: var(--color-muted);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  text-align: center;
  line-height: 1.3;
}

/* --- Supplemental --- */
.supplemental-details {
  margin-top: 1rem;
  border: 1px solid var(--line-soft);
  border-radius: 0.7rem;
  background: var(--surface-bg);
}

.supplemental-details summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  cursor: pointer;
  list-style: none;
  padding: 1rem 1.1rem;
  outline: none;
}

.supplemental-details summary::-webkit-details-marker {
  display: none;
}

.supplemental-details summary:focus-visible {
  box-shadow: inset 0 0 0 2px var(--color-accent);
}

.supplemental-details summary span:first-child {
  display: grid;
  gap: 0.2rem;
}

.supplemental-details summary strong {
  font-size: 0.83rem;
}

.supplemental-details summary small {
  color: var(--color-muted);
  font-size: 0.72rem;
}

.details-icon {
  color: var(--color-accent);
  font-size: 1.2rem;
  transition: transform 180ms ease;
}

.supplemental-details[open] .details-icon {
  transform: rotate(45deg);
}

.supplemental-grid {
  display: grid;
  gap: 1rem;
  border-top: 1px solid var(--line-soft);
  padding: 1rem;
}

.supplemental-group h3 {
  margin-bottom: 0.55rem;
  color: var(--color-accent);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.validation-group h3 {
  color: var(--color-brass);
}

.supplemental-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  border-bottom: 1px solid var(--line-soft);
  padding: 0.55rem 0;
}

.supplemental-row dt {
  color: var(--color-muted);
  font-size: 0.75rem;
}

.supplemental-row dd {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.78rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

/* --- Responsive --- */
@media (min-width: 460px) {
  .gauge-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (min-width: 768px) {
  .gauge-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }
}

@media (max-width: 480px) {
  .summary-stamp {
    display: none;
  }

  .selection-rail {
    grid-template-columns: 1fr;
  }

  .epoch-block {
    justify-items: start;
  }
}

@keyframes gauge-arrive {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (prefers-reduced-motion: reduce) {
  .gauge-card,
  .details-icon,
  .gauge-fill {
    animation: none;
    transition: none;
  }
}
</style>
