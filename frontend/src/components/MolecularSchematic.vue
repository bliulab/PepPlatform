<script setup lang="ts">
import { computed, useId } from 'vue'
import type { SculptureKind } from './landingSculptures'

const props = defineProps<{ kind: SculptureKind; label: string }>()
const id = useId().replace(/:/g, '')
const nodes = computed(() =>
  Array.from({ length: props.kind === 'hero' ? 42 : 20 }, (_, i) => {
    const t = i / (props.kind === 'hero' ? 41 : 19)
    return props.kind === 'hero'
      ? { x: 300 + Math.cos(t * Math.PI * 5) * 76, y: 60 + t * 300 }
      : { x: 100 + t * 400, y: 235 + Math.sin(t * Math.PI * 3) * 20 }
  }),
)
const path = computed(() => nodes.value.map((p, i) => `${i ? 'L' : 'M'}${p.x},${p.y}`).join(' '))
</script>

<template>
  <svg class="molecular-schematic" viewBox="0 0 600 420" role="img" :aria-label="label">
    <defs>
      <radialGradient :id="`${id}-glow`">
        <stop stop-color="#0d9488" stop-opacity=".27" />
        <stop offset="1" stop-color="#050f0e" stop-opacity="0" />
      </radialGradient>
      <linearGradient :id="`${id}-line`" x2="1" y2="1">
        <stop stop-color="#5eead4" />
        <stop offset="1" stop-color="#3a9f8c" />
      </linearGradient>
    </defs>
    <ellipse cx="300" cy="220" rx="285" ry="190" :fill="`url(#${id}-glow)`" />
    <g fill="none" stroke="#0d9488" stroke-opacity=".2">
      <ellipse cx="300" cy="215" rx="240" ry="130" transform="rotate(-15 300 215)" />
      <ellipse cx="300" cy="215" rx="225" ry="160" transform="rotate(28 300 215)" />
    </g>
    <g v-if="kind === 'hero'" transform="rotate(17 300 210)">
      <path :d="path" fill="none" :stroke="`url(#${id}-line)`" stroke-width="5" />
      <circle
        v-for="(p, i) in nodes"
        :key="i"
        :cx="p.x"
        :cy="p.y"
        :r="i % 7 === 0 ? 7 : 4"
        :fill="i % 7 === 0 ? '#99f6e4' : '#5eead4'"
      />
    </g>
    <g v-else-if="kind === 'pep_function'">
      <path :d="path" fill="none" stroke="#2dd4bf" stroke-width="4" />
      <circle v-for="(p, i) in nodes" :key="i" :cx="p.x" :cy="p.y" r="5" fill="#5eead4" />
      <g v-for="x in [170, 300, 430]" :key="x">
        <path :d="`M${x} 220 V145`" stroke="#0d9488" />
        <rect :x="x - 38" y="112" width="76" height="30" rx="5" fill="#0a322e" stroke="#5eead4" />
        <circle :cx="x" cy="127" r="4" fill="#99f6e4" />
      </g>
    </g>
    <g v-else-if="kind === 'pep_protein_interaction'" fill="none" stroke-linecap="round">
      <path
        d="M280 105C160 65 110 140 165 205S265 175 235 135 115 170 150 250 305 335 310 235 210 275 220 320"
        stroke="#0d9488"
        stroke-width="13"
      />
      <path d="M305 211Q340 165 365 198T425 185 480 205" stroke="#99f6e4" stroke-width="6" />
      <circle cx="310" cy="211" r="11" fill="#99f6e4" stroke="none" />
      <circle
        v-for="(p, i) in [
          [350, 192],
          [390, 198],
          [430, 188],
          [470, 203],
        ]"
        :key="i"
        :cx="p[0]"
        :cy="p[1]"
        r="7"
        fill="#99f6e4"
        stroke="none"
      />
    </g>
    <g v-else transform="translate(0 -5)">
      <path
        d="M105 200Q300 120 495 200L495 270Q300 190 105 270Z"
        fill="#0f766e"
        fill-opacity=".6"
        stroke="#0d9488"
      />
      <path
        v-for="y in [192, 264]"
        :key="y"
        :d="`M105 ${y} Q300 ${y - 80} 495 ${y}`"
        fill="none"
        stroke="#5eead4"
        stroke-width="14"
        stroke-linecap="round"
      />
      <path
        v-if="kind === 'pep_hlai'"
        d="M120 193V266M480 193V266"
        stroke="#5eead4"
        stroke-width="14"
        stroke-linecap="round"
      />
      <path
        :d="kind === 'pep_hlaii' ? 'M55 231Q300 143 545 231' : 'M145 211Q300 170 455 211'"
        fill="none"
        stroke="#99f6e4"
        stroke-width="6"
      />
      <circle
        v-for="i in kind === 'pep_hlaii' ? 17 : 9"
        :key="i"
        :cx="kind === 'pep_hlaii' ? 55 + (i - 1) * 30.6 : 145 + (i - 1) * 38.75"
        :cy="
          kind === 'pep_hlaii'
            ? 231 - Math.sin(((i - 1) / 16) * Math.PI) * 44
            : 211 - Math.sin(((i - 1) / 8) * Math.PI) * 21
        "
        r="6"
        fill="#99f6e4"
      />
    </g>
  </svg>
</template>
