<script setup lang="ts">
import { useId } from 'vue'

// A conceptual peptide trace, independent of research data and the Welcome renderer.
const id = useId().replace(/:/g, '')
const beads = Array.from({ length: 38 }, (_, index) => {
  const angle = index * 0.37
  return {
    x: 42 + index * 6.1,
    y: 141 + Math.sin(angle) * 42 - index * 0.65,
    r: 3.2 + (Math.cos(angle) + 1) * 1.35,
  }
})
const trace = beads
  .map((point, index) => `${index ? 'L' : 'M'}${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
  .join(' ')
</script>

<template>
  <svg class="workbench-art" viewBox="0 0 480 270" fill="none" aria-hidden="true" focusable="false">
    <defs>
      <linearGradient
        :id="`${id}-line`"
        x1="40"
        y1="70"
        x2="330"
        y2="220"
        gradientUnits="userSpaceOnUse"
      >
        <stop class="art-stop-light" />
        <stop offset=".6" class="art-stop-main" />
        <stop offset="1" class="art-stop-light" />
      </linearGradient>
      <linearGradient
        :id="`${id}-face`"
        x1="310"
        y1="70"
        x2="430"
        y2="210"
        gradientUnits="userSpaceOnUse"
      >
        <stop class="art-stop-light" stop-opacity=".42" />
        <stop offset="1" class="art-stop-main" stop-opacity=".08" />
      </linearGradient>
      <radialGradient :id="`${id}-wash`">
        <stop class="art-stop-light" stop-opacity=".24" />
        <stop offset="1" class="art-stop-light" stop-opacity="0" />
      </radialGradient>
    </defs>
    <ellipse cx="270" cy="149" rx="211" ry="116" :fill="`url(#${id}-wash)`" />
    <g class="art-guides">
      <path d="M29 205L207 247 457 157M28 165L207 207 457 117M71 104L250 146 438 78" />
      <path d="M98 218L348 128M163 233L413 143M209 54V245M331 37V205" />
      <ellipse
        cx="245"
        cy="151"
        rx="191"
        ry="69"
        transform="rotate(-14 245 151)"
        stroke-dasharray="3 8"
      />
    </g>
    <g class="art-orbits">
      <path d="M62 201C142 240 274 219 351 189" />
      <path d="M153 58C229 24 350 43 400 83" />
      <circle cx="62" cy="201" r="3" />
      <circle cx="153" cy="58" r="3" />
    </g>
    <path
      :d="trace"
      :stroke="`url(#${id}-line)`"
      stroke-width="7"
      stroke-linecap="round"
      stroke-linejoin="round"
      opacity=".15"
    />
    <path
      :d="trace"
      :stroke="`url(#${id}-line)`"
      stroke-width="1.8"
      stroke-linecap="round"
      stroke-linejoin="round"
    />
    <circle
      v-for="(bead, index) in beads"
      :key="index"
      :cx="bead.x"
      :cy="bead.y"
      :r="bead.r"
      :fill="`url(#${id}-line)`"
      class="art-bead"
    />
    <path d="M271 107C296 105 293 144 320 144" class="art-connection" stroke-dasharray="3 5" />
    <g :fill="`url(#${id}-face)`" class="art-model">
      <path d="M319 112L372 84 425 112 372 141Z" />
      <path d="M319 112V177L372 207V141Z" />
      <path d="M372 141L425 112V177L372 207Z" />
      <path d="M319 96L372 67 425 96 372 126Z" opacity=".55" />
      <path d="M319 81L372 52 425 81 372 111Z" opacity=".3" />
    </g>
    <g class="art-model-lines">
      <path d="M336 137V165L357 177M387 163L408 151M387 176L408 164" />
      <circle cx="372" cy="84" r="3" />
      <circle cx="319" cy="112" r="3" />
      <circle cx="425" cy="112" r="3" />
    </g>
    <g class="art-sequence">
      <rect
        v-for="index in 7"
        :key="index"
        :x="63 + index * 15"
        y="222"
        width="9"
        height="9"
        rx="2"
        :opacity="0.2 + (index % 3) * 0.2"
      />
      <path d="M66 225H50V212M198 225H214V212" />
    </g>
  </svg>
</template>

<style scoped>
.workbench-art {
  display: block;
  width: 100%;
  height: auto;
  color: var(--color-accent);
}
.art-stop-light {
  stop-color: var(--color-accent);
}
.art-stop-main {
  stop-color: var(--color-accent-dark);
}
.art-guides {
  stroke: currentColor;
  stroke-width: 0.65;
  opacity: 0.12;
}
.art-orbits {
  stroke: currentColor;
  stroke-width: 1;
  opacity: 0.24;
}
.art-orbits circle {
  fill: var(--surface-strong);
}
.art-bead {
  stroke: var(--surface-strong);
  stroke-width: 1.4;
}
.art-connection {
  stroke: currentColor;
  opacity: 0.45;
}
.art-model {
  stroke: currentColor;
  stroke-width: 1.1;
  stroke-linejoin: round;
  stroke-opacity: 0.6;
}
.art-model-lines {
  stroke: currentColor;
  stroke-width: 1.5;
  opacity: 0.6;
}
.art-model-lines circle {
  fill: var(--surface-strong);
}
.art-sequence {
  fill: currentColor;
}
.art-sequence path {
  fill: none;
  stroke: currentColor;
  stroke-width: 1;
  opacity: 0.3;
}
</style>
