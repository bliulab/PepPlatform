<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import ResearchGlyph from './ResearchGlyph.vue'

defineProps<{ paused: boolean }>()
const { t } = useI18n()
const root = ref<HTMLElement | null>(null)
const steps = ['sequence', 'representation', 'learning', 'inference'] as const
const methods = ['combine', 'compare', 'reuse'] as const
let observer: IntersectionObserver | null = null
onMounted(() => {
  const sections = root.value?.querySelectorAll('.research-panel') ?? []
  if (typeof IntersectionObserver === 'undefined') {
    for (const section of sections) section.classList.add('is-visible')
    return
  }
  observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) entry.target.classList.toggle('is-visible', entry.isIntersecting)
    },
    { threshold: 0.08 },
  )
  for (const section of sections) observer.observe(section)
})
onBeforeUnmount(() => observer?.disconnect())
</script>

<template>
  <div ref="root" class="research-story" :data-running="!paused">
    <section class="welcome-research research-panel" aria-labelledby="research-path-title">
      <header class="welcome-section-header">
        <div>
          <p class="welcome-eyebrow">{{ t('landing.research.eyebrow') }}</p>
          <h2 id="research-path-title">{{ t('landing.research.title') }}</h2>
        </div>
        <span class="welcome-section-index" aria-hidden="true">SEQUENCE → DISCOVERY</span>
      </header>
      <p class="research-section-lead">{{ t('landing.research.lead') }}</p>
      <ol class="research-flow">
        <li
          v-for="(step, index) in steps"
          :key="step"
          class="research-flow-step"
          :style="{ '--step': index }"
        >
          <span class="research-step-index">{{ String(index + 1).padStart(2, '0') }}</span>
          <ResearchGlyph :kind="step" />
          <div class="research-step-copy">
            <h3>{{ t(`landing.research.steps.${step}.title`) }}</h3>
            <p>{{ t(`landing.research.steps.${step}.body`) }}</p>
          </div>
        </li>
      </ol>
      <div class="research-path-foot">
        <span>{{ t('landing.research.record') }}</span
        ><span class="research-foot-line" aria-hidden="true" /><span>{{
          t('landing.research.result')
        }}</span>
      </div>
    </section>

    <section class="welcome-methods research-panel" aria-labelledby="research-methods-title">
      <header class="welcome-section-header">
        <div>
          <p class="welcome-eyebrow">{{ t('landing.methods.eyebrow') }}</p>
          <h2 id="research-methods-title">{{ t('landing.methods.title') }}</h2>
        </div>
        <span class="welcome-section-index" aria-hidden="true">DESIGNED FOR RESEARCH</span>
      </header>
      <div class="research-method-grid">
        <article
          v-for="(method, index) in methods"
          :key="method"
          class="research-method"
          :style="{ '--step': index }"
        >
          <div class="research-method-number">
            <span>{{ String(index + 1).padStart(2, '0') }}</span
            ><span aria-hidden="true">↗</span>
          </div>
          <ResearchGlyph :kind="method" />
          <h3>{{ t(`landing.methods.items.${method}.title`) }}</h3>
          <p>{{ t(`landing.methods.items.${method}.body`) }}</p>
          <div class="research-method-tags">
            <span v-for="tag in [0, 1, 2]" :key="tag">{{
              t(`landing.methods.items.${method}.tag${tag}`)
            }}</span>
          </div>
        </article>
      </div>
    </section>
  </div>
</template>
