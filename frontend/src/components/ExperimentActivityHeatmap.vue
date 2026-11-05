<script setup lang="ts">
import { Activity, RefreshCw } from '@lucide/vue'
import { computed, ref, useId } from 'vue'
import { useI18n } from 'vue-i18n'
import { buildExperimentActivity, type ActivityDay } from '../experimentActivity'
import type { ExperimentRecord } from '../types'
import { useCalendarClock } from '../useCalendarClock'

const props = defineProps<{
  experiments?: ExperimentRecord[]
  ownerId?: string | null
  workspace?: boolean
  loading?: boolean
  failed?: boolean
  guest?: boolean
}>()
defineEmits<{ retry: [] }>()
const { t, locale } = useI18n()
const now = useCalendarClock()
const activity = computed(() =>
  buildExperimentActivity(props.experiments, { ownerId: props.ownerId, now: now.value }),
)
const calendar = ref<HTMLElement>()
const focusedIndex = ref(-1)
const inspected = ref<ActivityDay | null>(null)
const hintId = useId()
const ready = computed(() => !props.guest && !props.loading && !props.failed)
const weekdays = computed(() =>
  Array.from({ length: 7 }, (_, index) =>
    new Intl.DateTimeFormat(locale.value, { weekday: 'short' }).format(
      new Date(2026, 0, 4 + index),
    ),
  ),
)
const currentIndex = computed(() =>
  Math.max(
    activity.value.leading,
    Math.min(
      focusedIndex.value < 0
        ? activity.value.leading + activity.value.days.length - 1
        : focusedIndex.value,
      activity.value.leading + activity.value.days.length - 1,
    ),
  ),
)
const month = (date: string) =>
  new Intl.DateTimeFormat(locale.value, { month: 'short' }).format(new Date(date + 'T12:00:00'))
function describe(day: ActivityDay) {
  const date = new Intl.DateTimeFormat(locale.value, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    weekday: 'short',
  }).format(new Date(day.date + 'T12:00:00'))
  return t('activity.dayCount', { date, count: day.count })
}
function inspect(day: ActivityDay, index: number) {
  inspected.value = day
  focusedIndex.value = index
}
function navigate(event: KeyboardEvent, index: number) {
  const delta: Record<string, number> = { ArrowLeft: -7, ArrowRight: 7, ArrowUp: -1, ArrowDown: 1 }
  let target: number
  if (event.key === 'Home') target = activity.value.leading
  else if (event.key === 'End') target = activity.value.leading + activity.value.days.length - 1
  else if (event.key in delta) target = index + delta[event.key]!
  else return
  event.preventDefault()
  target = Math.max(
    activity.value.leading,
    Math.min(activity.value.leading + activity.value.days.length - 1, target),
  )
  calendar.value?.querySelector<HTMLButtonElement>(`[data-day-index="${target}"]`)?.focus()
}
</script>

<template>
  <section class="experiment-activity" :aria-label="t('activity.title')" :aria-busy="loading">
    <header class="activity-heading">
      <div class="activity-heading-copy">
        <p class="activity-eyebrow"><Activity aria-hidden="true" />{{ t('activity.eyebrow') }}</p>
        <h2>{{ t(workspace ? 'activity.workspaceTitle' : 'activity.title') }}</h2>
        <p>{{ t('activity.subtitle') }}</p>
      </div>
      <dl v-if="ready" class="activity-totals">
        <div
          ><dt>{{ t('activity.yearTotal') }}</dt
          ><dd>{{ activity.total }}</dd></div
        >
        <div
          ><dt>{{ t('activity.activeDays') }}</dt
          ><dd>{{ activity.activeDays }}</dd></div
        >
      </dl>
    </header>
    <p v-if="guest" class="activity-state">{{ t('activity.signIn') }}</p>
    <div v-else-if="failed" class="activity-state activity-state--error" role="alert">
      <span>{{ t('home.loadError') }}</span>
      <button type="button" class="tertiary-action compact-action" @click="$emit('retry')"
        ><RefreshCw aria-hidden="true" />{{ t('home.retry') }}</button
      >
    </div>
    <p v-else-if="loading" class="activity-state" role="status">{{ t('common.loading') }}</p>
    <template v-else>
      <div class="activity-scroll">
        <div class="activity-calendar">
          <div
            class="activity-months"
            :style="{ gridTemplateColumns: `repeat(${activity.weeks}, minmax(0, 1fr))` }"
            aria-hidden="true"
          >
            <span
              v-for="item in activity.months"
              :key="item.date"
              :style="{
                gridColumn: `${item.column} / span ${Math.min(3, activity.weeks - item.column + 1)}`,
              }"
              >{{ month(item.date) }}</span
            >
          </div>
          <div class="activity-weekdays" aria-hidden="true"
            ><span v-for="(day, index) in weekdays" :key="index">{{
              index % 2 === 1 ? day : ''
            }}</span></div
          >
          <div
            ref="calendar"
            class="activity-days"
            role="group"
            :aria-label="t('activity.calendarLabel')"
            :aria-describedby="hintId"
            :style="{ gridTemplateColumns: `repeat(${activity.weeks}, minmax(0, 1fr))` }"
          >
            <template v-for="(day, index) in activity.cells" :key="index">
              <button
                v-if="day"
                type="button"
                class="activity-day"
                :data-level="day.level"
                :data-date="day.date"
                :data-day-index="index"
                :aria-label="describe(day)"
                :title="describe(day)"
                :tabindex="index === currentIndex ? 0 : -1"
                @mouseenter="inspected = day"
                @focus="inspect(day, index)"
                @click="inspect(day, index)"
                @keydown="navigate($event, index)"
              />
              <span v-else class="activity-padding" aria-hidden="true" />
            </template>
          </div>
        </div>
      </div>
      <footer class="activity-footer">
        <p :id="hintId" class="activity-inspection" aria-live="polite">{{
          inspected
            ? describe(inspected)
            : activity.total
              ? t('activity.inspectHint')
              : t('activity.empty')
        }}</p>
        <div class="activity-legend" :aria-label="t('activity.legend')"
          ><span>{{ t('activity.less') }}</span
          ><i
            v-for="level in 5"
            :key="level"
            :data-level="level - 1"
            :title="['0', '1', '2–3', '4–6', '7+'][level - 1]"
          /><span>{{ t('activity.more') }}</span></div
        >
      </footer>
      <p class="activity-policy">{{ t('activity.policy') }}</p>
    </template>
  </section>
</template>

<style scoped>
.experiment-activity {
  --activity-0: var(--surface-muted);
  --activity-1: #cde9d9;
  --activity-2: #8ac5a4;
  --activity-3: #429b72;
  --activity-4: #176b4a;
  grid-column: 1 / -1;
  margin-top: 1.5rem;
  padding: 1.5rem 1.75rem 1.1rem;
  border: 1px solid var(--color-line);
  border-radius: var(--radius-panel);
  background: var(--surface-strong);
  color: var(--color-ink);
}
:root[data-theme='dark'] .experiment-activity {
  --activity-0: #172a29;
  --activity-1: #244b3b;
  --activity-2: #2e7855;
  --activity-3: #4aab73;
  --activity-4: #8ad3a5;
}
.activity-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.25rem;
}
.activity-eyebrow {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  color: var(--color-accent-dark);
  font-size: 0.65rem;
  font-weight: 650;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.activity-eyebrow svg {
  width: 0.9rem;
  height: 0.9rem;
}
.activity-heading h2 {
  margin-top: 0.4rem;
  font-size: 1.25rem;
  font-weight: 600;
  letter-spacing: -0.03em;
}
.activity-heading-copy > p:last-child {
  margin-top: 0.35rem;
  color: var(--color-muted);
  font-size: 0.75rem;
}
.activity-totals {
  display: flex;
  gap: 1.7rem;
}
.activity-totals > div {
  display: flex;
  flex-direction: column-reverse;
  gap: 0.3rem;
}
.activity-totals dt {
  font-size: 0.65rem;
  color: var(--color-muted);
  white-space: nowrap;
}
.activity-totals dd {
  font-size: 1.75rem;
  line-height: 1;
  font-weight: 550;
  letter-spacing: -0.05em;
  font-variant-numeric: tabular-nums;
}
.activity-totals > div + div {
  border-left: 1px solid var(--line-soft);
  padding-left: 1.7rem;
}
.activity-scroll {
  direction: rtl;
  margin-top: 1.35rem;
  overflow-x: auto;
  padding: 3px 3px 8px;
  scrollbar-width: thin;
  scrollbar-color: var(--color-line) transparent;
}
.activity-calendar {
  direction: ltr;
  display: grid;
  min-width: 680px;
  grid-template-columns: 2rem minmax(0, 1fr);
  gap: 0.45rem 0.55rem;
}
.activity-months {
  grid-column: 2;
  display: grid;
  gap: 3px;
  min-height: 1rem;
  color: var(--color-muted);
  font-size: 0.65rem;
}
.activity-months span {
  overflow: hidden;
  white-space: nowrap;
}
.activity-weekdays {
  display: grid;
  grid-template-rows: repeat(7, 1fr);
  gap: 3px;
  font-size: 0.6rem;
  color: var(--color-muted);
}
.activity-weekdays span {
  display: flex;
  align-items: center;
}
.activity-days {
  display: grid;
  grid-auto-flow: column;
  grid-template-rows: repeat(7, auto);
  gap: 3px;
}
.activity-day,
.activity-padding {
  display: block;
  min-width: 0;
  width: 100%;
  aspect-ratio: 1;
  padding: 0;
  border: 1px solid color-mix(in srgb, var(--color-ink) 5%, transparent);
  border-radius: 3px;
}
.activity-padding {
  visibility: hidden;
}
.activity-day {
  cursor: pointer;
  transition: outline-color 120ms ease;
}
[data-level='0'] {
  background: var(--activity-0);
}
[data-level='1'] {
  background: var(--activity-1);
}
[data-level='2'] {
  background: var(--activity-2);
}
[data-level='3'] {
  background: var(--activity-3);
}
[data-level='4'] {
  background: var(--activity-4);
}
.activity-day:hover,
.activity-day:focus-visible {
  position: relative;
  outline: 2px solid var(--color-accent-dark);
  outline-offset: 1px;
  z-index: 1;
}
.activity-footer {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.65rem;
  margin-top: 0.5rem;
}
.activity-inspection {
  min-height: 1.2rem;
  color: var(--color-muted);
  font-size: 0.7rem;
  line-height: 1.6;
}
.activity-legend {
  display: flex;
  align-items: center;
  gap: 4px;
  color: var(--color-muted);
  font-size: 0.65rem;
}
.activity-legend i {
  width: 11px;
  height: 11px;
  border: 1px solid color-mix(in srgb, var(--color-ink) 5%, transparent);
  border-radius: 2px;
}
.activity-legend span:first-child {
  margin-right: 3px;
}
.activity-legend span:last-child {
  margin-left: 3px;
}
.activity-policy {
  margin-top: 0.6rem;
  font-size: 0.65rem;
  line-height: 1.6;
  color: var(--color-muted);
}
.activity-state {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  min-height: 7rem;
  gap: 1rem;
  padding-top: 1rem;
  color: var(--color-muted);
  font-size: 0.85rem;
}
.activity-state--error {
  color: var(--color-danger);
}
@media (max-width: 640px) {
  .experiment-activity {
    padding: 1.15rem 1rem 1rem;
  }
  .activity-heading {
    align-items: flex-start;
    flex-direction: column;
    gap: 1rem;
  }
  .activity-totals {
    gap: 1.25rem;
  }
  .activity-totals > div {
    flex-direction: row-reverse;
    align-items: baseline;
    gap: 0.5rem;
  }
  .activity-totals > div + div {
    padding-left: 1.25rem;
  }
  .activity-totals dd {
    font-size: 1.4rem;
  }
}
@media (prefers-reduced-motion: reduce) {
  .activity-day {
    transition: none;
  }
}
</style>
