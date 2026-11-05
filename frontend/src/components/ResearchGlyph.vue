<script setup lang="ts">
defineProps<{ kind: string }>()
</script>

<template>
  <svg class="research-glyph" viewBox="0 0 240 140" fill="none" aria-hidden="true">
    <g v-if="kind === 'sequence'">
      <path d="M18 85Q52 35 86 75T154 70 222 55" class="glyph-line" />
      <circle
        v-for="(p, i) in [
          [23, 78],
          [52, 60],
          [81, 70],
          [111, 87],
          [141, 80],
          [170, 65],
          [201, 60],
          [221, 55],
        ]"
        :key="i"
        :cx="p[0]"
        :cy="p[1]"
        r="5"
        class="glyph-bead"
      />
      <text
        v-for="(label, i) in ['A', 'L', 'W', 'K', 'G', 'S']"
        :key="i"
        :x="26 + i * 35"
        y="122"
        class="glyph-text"
      >
        {{ label }}
      </text>
      <circle cx="28" cy="73" r="4" class="story-packet" />
    </g>
    <g v-else-if="kind === 'representation'">
      <path d="M12 70H49M192 70H228" class="glyph-dim" />
      <g v-for="row in 5" :key="row">
        <rect
          v-for="col in 8"
          :key="col"
          :x="55 + (col - 1) * 17"
          :y="27 + (row - 1) * 18"
          width="10"
          height="10"
          rx="1"
          :opacity="0.18 + ((col * 3 + row * 7) % 8) * 0.1"
          class="glyph-cell"
        />
      </g>
      <path d="M45 18H38V121H45M199 18H206V121H199" class="glyph-line" />
    </g>
    <g v-else-if="kind === 'learning'">
      <path
        d="M16 70H45M45 70V28H80M45 70H80M45 70V112H80M125 28H151V70M125 70H151M125 112H151V70H188"
        class="glyph-line"
      />
      <g v-for="(y, i) in [15, 57, 99]" :key="i">
        <rect x="80" :y="y" width="45" height="26" rx="3" class="glyph-panel" />
        <text x="102" :y="y + 17" text-anchor="middle" class="glyph-text">
          {{ ['A', 'B', 'C'][i] }}
        </text>
      </g>
      <path d="M188 70L207 49 228 70 207 91Z" class="glyph-panel" />
      <circle cx="207" cy="70" r="4" class="method-pulse glyph-bead" />
    </g>
    <g v-else-if="kind === 'inference'">
      <path d="M15 70H66M119 70H156" class="glyph-line" />
      <path
        d="M66 43L93 28 120 43V94L93 110 66 94ZM66 43L93 60 120 43M93 60V110"
        class="glyph-panel"
      />
      <rect x="156" y="28" width="65" height="84" rx="4" class="glyph-panel" />
      <path d="M170 47H205M170 59H195M170 88L179 97 199 76" class="glyph-line" />
      <circle cx="20" cy="70" r="3" class="story-packet" />
    </g>
    <g v-else-if="kind === 'combine'">
      <path d="M26 45H76V70H115M26 98H76V70M145 70H208" class="glyph-line" />
      <rect x="14" y="25" width="44" height="34" rx="3" class="glyph-panel" />
      <path d="M24 35H48M24 43H40M24 51H45" class="glyph-dim" />
      <g v-for="i in 4" :key="i">
        <circle :cx="20 + (i - 1) * 10" cy="96" r="3" class="glyph-bead" />
      </g>
      <path d="M113 70L131 52 149 70 131 88Z" class="glyph-panel" />
      <g v-for="i in 3" :key="i">
        <rect
          x="177"
          :y="38 + (i - 1) * 23"
          width="45"
          height="12"
          rx="2"
          :opacity="0.35 + i * 0.18"
          class="glyph-cell"
        />
      </g>
    </g>
    <g v-else-if="kind === 'compare'">
      <path d="M28 20V118M28 118H222" class="glyph-dim" />
      <g v-for="(y, i) in [37, 71, 105]" :key="i">
        <path :d="`M30 ${y}H213`" class="glyph-dim" />
        <rect
          x="45"
          :y="y - 8"
          :width="[112, 143, 126][i]"
          height="8"
          rx="1"
          :opacity="0.35 + i * 0.2"
          class="glyph-cell"
        />
      </g>
      <path d="M198 15V119" class="glyph-line glyph-dashed" />
      <circle cx="198" cy="15" r="3" class="method-pulse glyph-bead" />
    </g>
    <g v-else>
      <path d="M50 42H113M50 70H113M50 98H113M113 42V98M113 70H163" class="glyph-line" />
      <g v-for="y in [29, 57, 85]" :key="y">
        <rect x="16" :y="y" width="34" height="26" rx="2" class="glyph-panel" />
        <path :d="`M23 ${y + 8}H42M23 ${y + 16}H37`" class="glyph-dim" />
      </g>
      <path
        d="M163 45L191 29 219 45V95L191 111 163 95ZM163 45L191 61 219 45M191 61V111"
        class="glyph-panel"
      />
      <circle cx="122" cy="70" r="3" class="story-packet" style="--packet-travel: 70px" />
    </g>
  </svg>
</template>
