<script setup lang="ts">
import { ArrowDown, ArrowUpRight, LogIn, Pause, Play, RotateCcw, UserPlus } from '@lucide/vue'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef } from 'vue'
import { useI18n } from 'vue-i18n'
import { TASK_ENTRIES } from '../taskCatalog'
import {
  advanceTaskPlayback,
  createTaskPlayback,
  selectTask,
  TASK_DWELL_MS,
} from '../landingPlayback'
import PeptideScene from '../components/PeptideScene.vue'
import MolecularStage from '../components/MolecularStage.vue'
import MolecularSchematic from '../components/MolecularSchematic.vue'
import ResearchStory from '../components/ResearchStory.vue'
import { supportsWebgl2 } from '../components/webglSupport'
import { useAuthStore } from '../stores/auth'

const { t } = useI18n()
const auth = useAuthStore()
const motionPaused = ref(false)
const heroRef = ref<HTMLElement | null>(null)
const taskSection = ref<HTMLElement | null>(null)
const sceneRef = ref<InstanceType<typeof PeptideScene> | null>(null)
const hasWebgl = supportsWebgl2()
const heroFallback = ref(!hasWebgl)
const taskFallback = ref(!hasWebgl)
const heroReplayKey = ref(0)
const taskReady = ref(!hasWebgl)
const stageMounted = ref(false)
const taskVisible = ref(false)
const hovered = ref(false)
const focused = ref(false)
const interacting = ref(false)
const hidden = ref(document.hidden)
const playback = shallowRef(createTaskPlayback())
const activeTask = computed(() => TASK_ENTRIES[playback.value.index]!)
const taskKey = computed(() => activeTask.value.key)
const activeProgress = computed(() => playback.value.elapsedMs / TASK_DWELL_MS)
const blocked = computed(
  () =>
    motionPaused.value ||
    !taskVisible.value ||
    !taskReady.value ||
    hidden.value ||
    hovered.value ||
    focused.value ||
    interacting.value,
)
const slideshowPaused = computed(() => playback.value.manual)
// Each visit plays the intro; page controls are independent of the OS preference.
const playIntro = hasWebgl
const inWorkspace = computed(() => Boolean(auth.user) || !auth.authEnabled)
const primaryCta = computed(() => ({
  to: '/dashboard',
  label: t(inWorkspace.value ? 'landing.footerCta' : 'guest.explore'),
}))

function chooseTask(index: number) {
  playback.value = selectTask(playback.value, index)
}
function toggleSlideshow() {
  if (playback.value.manual) focused.value = false
  playback.value = {
    ...playback.value,
    manual: !playback.value.manual,
    elapsedMs: 0,
    transitionMs: 0,
  }
}
function focusIn(event: FocusEvent) {
  focused.value = event.target instanceof Element && event.target.matches(':focus-visible')
}
function taskKeydown(event: KeyboardEvent, index: number) {
  let next = index
  if (['ArrowRight', 'ArrowDown'].includes(event.key)) next = (index + 1) % TASK_ENTRIES.length
  else if (['ArrowLeft', 'ArrowUp'].includes(event.key))
    next = (index + TASK_ENTRIES.length - 1) % TASK_ENTRIES.length
  else if (event.key === 'Home') next = 0
  else if (event.key === 'End') next = TASK_ENTRIES.length - 1
  else return
  event.preventDefault()
  chooseTask(next)
  document.getElementById(`welcome-tab-${TASK_ENTRIES[next]!.id}`)?.focus()
}
function focusOut(event: FocusEvent) {
  if (!event.relatedTarget || !taskSection.value?.contains(event.relatedTarget as Node))
    focused.value = false
}
function handleVisibility() {
  hidden.value = document.hidden
}
function onScroll() {
  if (window.scrollY > 50) sceneRef.value?.finishIntro()
}
async function replayIntro() {
  motionPaused.value = false
  await nextTick()
  if (heroFallback.value) heroReplayKey.value++
  else sceneRef.value?.replayIntro()
}
function failHero() {
  heroFallback.value = true
}
function failTask() {
  taskFallback.value = true
  taskReady.value = true
  interacting.value = false
}
let observers: IntersectionObserver[] = []
let timer = 0

onMounted(() => {
  document.addEventListener('visibilitychange', handleVisibility)
  window.addEventListener('scroll', onScroll, { passive: true })
  if (typeof IntersectionObserver === 'undefined') {
    stageMounted.value = true
    taskVisible.value = true
  } else if (taskSection.value) {
    const mountObserver = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          stageMounted.value = true
          mountObserver.disconnect()
        }
      },
      { rootMargin: '320px' },
    )
    const visibilityObserver = new IntersectionObserver(
      (entries) => {
        taskVisible.value = entries.some(
          (entry) => entry.isIntersecting && entry.intersectionRatio >= 0.35,
        )
      },
      { threshold: [0, 0.35] },
    )
    mountObserver.observe(taskSection.value)
    visibilityObserver.observe(taskSection.value)
    observers = [mountObserver, visibilityObserver]
  }
  let previous = performance.now()
  timer = window.setInterval(() => {
    const now = performance.now()
    const dt = Math.min(250, now - previous)
    previous = now
    playback.value = advanceTaskPlayback(playback.value, dt, blocked.value)
  }, 80)
})
onBeforeUnmount(() => {
  window.clearInterval(timer)
  for (const observer of observers) observer.disconnect()
  document.removeEventListener('visibilitychange', handleVisibility)
  window.removeEventListener('scroll', onScroll)
})
</script>

<template>
  <div class="welcome" :data-motion="motionPaused ? 'paused' : 'playing'">
    <section id="welcome-top" ref="heroRef" class="welcome-hero" aria-labelledby="welcome-title">
      <div class="welcome-hero-atmosphere" aria-hidden="true" />
      <div class="welcome-hero-copy">
        <p class="welcome-eyebrow"><span class="welcome-live-dot" />{{ t('landing.eyebrow') }}</p>
        <h1 id="welcome-title" class="welcome-brand"> Pep<span>Platform</span> </h1>
        <p class="welcome-tagline">{{ t('landing.title') }}</p>
        <div class="welcome-actions">
          <RouterLink :to="primaryCta.to" class="primary-action welcome-primary"
            ><LogIn aria-hidden="true" />{{ primaryCta.label }}<ArrowUpRight aria-hidden="true"
          /></RouterLink>
          <RouterLink v-if="!inWorkspace" to="/register" class="welcome-register"
            ><UserPlus aria-hidden="true" />{{ t('auth.registerLink') }}</RouterLink
          >
        </div>
        <div class="welcome-motion-controls">
          <button type="button" class="welcome-quiet-button" @click="replayIntro">
            <RotateCcw aria-hidden="true" />{{ t('landing.replay') }}
          </button>
          <button
            type="button"
            class="welcome-quiet-button"
            :aria-pressed="motionPaused"
            @click="motionPaused = !motionPaused"
          >
            <Play v-if="motionPaused" aria-hidden="true" /><Pause v-else aria-hidden="true" />{{
              t(motionPaused ? 'landing.resumeMotion' : 'landing.pauseMotion')
            }}
          </button>
        </div>
        <p class="welcome-interaction-hint">
          {{ t(heroFallback ? 'landing.selectTask' : 'landing.particleHint') }}
        </p>
        <a class="welcome-explore" href="#welcome-tasks"
          ><span class="welcome-explore-line" /><ArrowDown aria-hidden="true" />{{
            t('landing.explore')
          }}</a
        >
      </div>
      <div
        class="welcome-hero-visual"
        :aria-label="t(heroFallback ? 'landing.staticDescription' : 'landing.heroDescription')"
        role="img"
      >
        <MolecularSchematic
          v-if="heroFallback"
          :key="heroReplayKey"
          class="welcome-hero-schematic"
          kind="hero"
          :label="t('landing.staticDescription')"
        />
        <PeptideScene
          v-if="hasWebgl && !heroFallback"
          ref="sceneRef"
          :paused="motionPaused"
          :play-intro="playIntro"
          @fallback="failHero"
        />
        <span class="welcome-coordinate welcome-coordinate-top" aria-hidden="true"
          >PEPTIDE / 001</span
        >
        <span class="welcome-coordinate welcome-coordinate-bottom" aria-hidden="true">N — C</span>
      </div>
      <div class="welcome-hero-bottom">
        <p class="welcome-hero-footnote">{{ t('landing.heroNote') }}</p>
      </div>
      <div class="welcome-hero-rule" aria-hidden="true">
        <span>01</span><span>PEPPLATFORM</span><span>04</span>
      </div>
    </section>

    <section
      id="welcome-tasks"
      ref="taskSection"
      class="welcome-showcase"
      aria-labelledby="welcome-tasks-title"
      @pointerenter="hovered = $event.pointerType !== 'touch'"
      @pointerleave="hovered = false"
      @focusin="focusIn"
      @focusout="focusOut"
    >
      <header class="welcome-section-header">
        <div>
          <p class="welcome-eyebrow">{{ t('landing.showcaseEyebrow') }}</p>
          <h2 id="welcome-tasks-title">{{ t('landing.showcaseTitle') }}</h2>
        </div>
        <span class="welcome-section-index" aria-hidden="true">01 — 04</span>
      </header>
      <div class="welcome-showcase-grid">
        <div
          class="welcome-task-list"
          role="tablist"
          :aria-label="t('landing.taskSectionTitle')"
          aria-orientation="vertical"
        >
          <button
            v-for="(task, index) in TASK_ENTRIES"
            :id="`welcome-tab-${task.id}`"
            :key="task.key"
            class="welcome-task-tab"
            :class="{ 'is-active': playback.index === index }"
            type="button"
            role="tab"
            :aria-selected="playback.index === index"
            aria-controls="welcome-task-panel"
            :tabindex="playback.index === index ? 0 : -1"
            @click="chooseTask(index)"
            @keydown="taskKeydown($event, index)"
          >
            <span class="welcome-task-number">{{ task.id }}</span>
            <span class="welcome-task-name">{{ t(`tasks.${task.key}.label`) }}</span>
            <span class="welcome-task-arrow" aria-hidden="true">✓</span>
            <span
              class="welcome-task-progress"
              :style="{
                transform: `scaleX(${playback.index === index ? (slideshowPaused ? 1 : activeProgress) : 0})`,
              }"
              aria-hidden="true"
            />
          </button>
        </div>
        <div
          id="welcome-task-panel"
          class="welcome-task-panel"
          role="tabpanel"
          :aria-labelledby="`welcome-tab-${activeTask.id}`"
          tabindex="0"
        >
          <div class="welcome-task-stage">
            <MolecularStage
              v-if="stageMounted && hasWebgl && !taskFallback"
              :kind="taskKey"
              :paused="motionPaused"
              @ready="taskReady = true"
              @fallback="failTask"
              @interactionstart="interacting = true"
              @interactionend="interacting = false"
            />
            <MolecularSchematic
              v-if="taskFallback || !stageMounted"
              :kind="taskKey"
              :label="t(`landing.scenes.${taskKey}`)"
            />
            <div
              v-if="taskKey === 'pep_function'"
              class="welcome-function-labels"
              aria-hidden="true"
            >
              <span>{{ t('landing.functionAntimicrobial') }}</span
              ><span>{{ t('landing.functionAntiviral') }}</span
              ><span>{{ t('landing.functionPenetrating') }}</span>
            </div>
            <span class="welcome-stage-label">{{ t('landing.concept') }}</span>
          </div>
          <div class="welcome-task-caption">
            <span class="welcome-caption-number">{{ activeTask.id }}</span>
            <p>{{ t(`landing.scenes.${taskKey}`) }}</p>
          </div>
        </div>
      </div>
      <div class="welcome-showcase-bottom">
        <p class="welcome-drag-note">
          {{ t(taskFallback ? 'landing.selectTask' : 'landing.dragHint') }}
        </p>
        <div class="welcome-playback">
          <span aria-hidden="true"
            >{{ String(playback.index + 1).padStart(2, '0') }} <i>/ 04</i></span
          ><button
            type="button"
            class="welcome-quiet-button"
            :aria-pressed="slideshowPaused"
            @click="toggleSlideshow"
          >
            <Play v-if="slideshowPaused" aria-hidden="true" /><Pause v-else aria-hidden="true" />{{
              t(slideshowPaused ? 'landing.resumeSlideshow' : 'landing.pauseSlideshow')
            }}
          </button>
        </div>
      </div>
    </section>

    <ResearchStory :paused="motionPaused || hidden" />

    <footer class="welcome-closing">
      <div class="welcome-closing-inner">
        <div>
          <p class="welcome-eyebrow">PEPPLATFORM</p>
          <h2>{{ t('landing.closingTitle') }}</h2>
        </div>
        <RouterLink :to="primaryCta.to" class="welcome-closing-link"
          >{{ primaryCta.label }}<ArrowUpRight aria-hidden="true"
        /></RouterLink>
      </div>
      <div class="welcome-footer-line">
        <span>PepPlatform</span><span>{{ t('landing.footerSignature') }}</span
        ><a href="#welcome-top" :aria-label="t('landing.backToTop')">↑</a>
      </div>
    </footer>
  </div>
</template>

<style src="../landing.css"></style>
