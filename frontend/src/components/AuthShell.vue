<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import { TASK_ENTRIES } from '../taskCatalog'
import { railNodes, railPath, railRibbons } from './authRailArt'

/**
 * Shared shell for the four authentication pages: a brand rail on the left
 * (identity, claim, a deterministic molecular trace, the T01-T04 family stations,
 * a way back to the landing page) and the form card on the right.
 *
 * The rail owns the only logo on these pages - the sticky header drops its lockup
 * when an auth route is active (see App.vue).
 *
 * Motion budget stays low on purpose: the landing page owns the 3D and the scroll
 * choreography, so the trace only drifts its dash pattern and breathes its nodes,
 * and both are declared under `prefers-reduced-motion: no-preference`.
 */
const { t } = useI18n()

const railPaths = computed(() => railRibbons().map((points) => railPath(points)))
const railNodePoints = computed(() => railRibbons().flatMap((points) => railNodes(points)))
</script>

<template>
  <section class="auth-shell">
    <div class="auth-brandrail">
      <RouterLink to="/" class="auth-brand-link">
        <span class="auth-brand-mark">
          <img src="/logo-mark.png" alt="" />
        </span>
        <span class="auth-brand-word">PepPlatform</span>
      </RouterLink>
      <p class="auth-brand-claim">{{ t('landing.title') }}</p>
      <figure class="auth-brand-art-wrap">
        <p class="auth-brand-kicker">{{ t('auth.railKicker') }}</p>
        <svg class="auth-brand-art" :viewBox="`0 0 320 180`" aria-hidden="true" focusable="false">
          <g class="auth-brand-art-lines">
            <path
              v-for="(path, index) in railPaths"
              :key="`line-${index}`"
              class="auth-brand-art-line"
              :d="path"
              :style="{ '--i': index }"
            />
          </g>
          <g class="auth-brand-art-nodes">
            <circle
              v-for="(node, index) in railNodePoints"
              :key="`node-${index}`"
              :cx="node.x"
              :cy="node.y"
              r="2.2"
              :style="{ '--i': index % 12 }"
            />
          </g>
        </svg>
      </figure>
      <ol class="auth-brand-tasks">
        <li v-for="task in TASK_ENTRIES" :key="task.id" class="auth-brand-task">
          <span class="auth-brand-task-id">{{ task.id }}</span>
          <span class="auth-brand-task-label">{{ t(`tasks.${task.key}.label`) }}</span>
        </li>
      </ol>
      <div class="auth-brand-foot">
        <p class="auth-brand-note">{{ t('auth.brandFoot') }}</p>
        <RouterLink to="/" class="auth-brand-back">← {{ t('auth.backToHome') }}</RouterLink>
      </div>
    </div>
    <div class="auth-formrail">
      <div class="auth-card panel">
        <slot />
      </div>
    </div>
  </section>
</template>
