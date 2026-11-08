<script setup lang="ts">
import { ref } from 'vue'
import MolecularStage from './MolecularStage.vue'

withDefaults(defineProps<{ paused?: boolean; playIntro?: boolean }>(), {
  paused: false,
  playIntro: false,
})
const emit = defineEmits<{ ready: []; fallback: []; introcomplete: [] }>()
const stage = ref<InstanceType<typeof MolecularStage> | null>(null)
defineExpose({
  replayIntro: () => stage.value?.replayIntro(),
  finishIntro: () => stage.value?.finishIntro(),
})
</script>

<template>
  <MolecularStage
    ref="stage"
    kind="hero"
    :paused="paused"
    :play-intro="playIntro"
    @ready="emit('ready')"
    @fallback="emit('fallback')"
    @introcomplete="emit('introcomplete')"
  />
</template>
