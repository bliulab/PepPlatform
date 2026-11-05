<script setup lang="ts">
import { defineAsyncComponent } from 'vue'

const LandingView = defineAsyncComponent(() => import('./LandingView.vue'))
</script>

<template>
  <LandingView />
</template>
