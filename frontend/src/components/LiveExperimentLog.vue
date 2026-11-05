<script setup lang="ts">
import { Check, Copy, MoveDown } from '@lucide/vue'
import { nextTick, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

const props = withDefaults(
  defineProps<{
    content?: string
    active?: boolean
  }>(),
  {
    content: '',
    active: false,
  },
)

const { t } = useI18n()
const logElement = ref<HTMLElement | null>(null)
const followTail = ref(true)
const copied = ref(false)
let copiedTimer: ReturnType<typeof setTimeout> | undefined

function scrollToLatest() {
  followTail.value = true
  void nextTick(() => {
    if (!logElement.value) return
    logElement.value.scrollTop = logElement.value.scrollHeight
  })
}

function handleScroll() {
  const element = logElement.value
  if (!element) return
  const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight
  followTail.value = distanceFromBottom <= 24
}

async function copyLogs() {
  if (!props.content || !navigator.clipboard) return
  try {
    await navigator.clipboard.writeText(props.content)
  } catch {
    return
  }
  copied.value = true
  if (copiedTimer) clearTimeout(copiedTimer)
  copiedTimer = setTimeout(() => {
    copied.value = false
  }, 1800)
}

watch(
  () => props.content,
  () => {
    if (followTail.value) scrollToLatest()
  },
  { immediate: true },
)
</script>

<template>
  <div id="experiment-logs" class="panel">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div class="flex items-center gap-3">
        <p class="eyebrow">{{ t('detail.logs') }}</p>
        <span class="log-live-status" aria-live="polite">
          <span
            class="log-live-dot"
            :class="{ 'log-live-dot-active': active }"
            aria-hidden="true"
          />
          {{ active ? t('detail.logsLive') : t('detail.logsIdle') }}
        </span>
      </div>
      <div class="flex flex-wrap gap-2">
        <button
          v-if="!followTail"
          type="button"
          class="tertiary-action compact-action"
          @click="scrollToLatest"
        >
          <MoveDown aria-hidden="true" />
          {{ t('detail.logsFollowLatest') }}
        </button>
        <button
          type="button"
          class="tertiary-action compact-action"
          :disabled="!content"
          @click="copyLogs"
        >
          <Check v-if="copied" aria-hidden="true" />
          <Copy v-else aria-hidden="true" />
          {{ copied ? t('detail.logsCopied') : t('detail.logsCopy') }}
        </button>
      </div>
    </div>
    <pre
      ref="logElement"
      class="experiment-live-log mt-4"
      tabindex="0"
      :aria-label="t('detail.logs')"
      aria-live="off"
      @scroll.passive="handleScroll"
      >{{ content || t('detail.logsEmpty') }}</pre>
  </div>
</template>

<style scoped>
.log-live-status {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  color: var(--color-muted);
  font-size: 0.75rem;
  font-weight: 700;
}

.log-live-dot {
  width: 0.5rem;
  height: 0.5rem;
  border-radius: 999px;
  background: var(--color-line);
}

.log-live-dot-active {
  background: var(--color-success-text);
  box-shadow: 0 0 0 0.2rem var(--color-success-bg);
  animation: log-live-pulse 1.8s ease-in-out infinite;
}

.experiment-live-log {
  max-height: 18rem;
  min-height: 9rem;
  overflow: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  border-radius: 0.75rem;
  background: var(--color-log-bg);
  padding: 1rem;
  color: var(--color-log-text);
  font-size: 0.75rem;
  line-height: 1.55;
  tab-size: 2;
}

.experiment-live-log:focus-visible {
  outline: 2px solid var(--color-accent);
  outline-offset: 2px;
}

@keyframes log-live-pulse {
  0%,
  100% {
    opacity: 0.55;
  }
  50% {
    opacity: 1;
  }
}

@media (prefers-reduced-motion: reduce) {
  .log-live-dot-active {
    animation: none;
  }
}
</style>
