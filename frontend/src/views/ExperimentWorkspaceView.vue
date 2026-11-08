<script setup lang="ts">
import { Grid2X2, LayoutDashboard, Sparkles } from '@lucide/vue'
import { computed, onBeforeUnmount, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'
import { useExperimentAnalysisStore } from '../stores/experimentAnalysis'

const props = defineProps<{ id: string }>()
const route = useRoute()
const { t } = useI18n()
const analysis = useExperimentAnalysisStore()
const activeTab = computed(() => String(route.meta.experimentTab || 'overview'))
const analysisTarget = computed(() => {
  const conversationId = analysis.activeGenerationConversationId
  return conversationId
    ? `/experiments/${props.id}/analysis/${conversationId}`
    : `/experiments/${props.id}/analysis`
})

watch(
  () => props.id,
  (experimentId) => analysis.selectExperiment(experimentId),
  { immediate: true },
)

onBeforeUnmount(() => analysis.reset())
</script>

<template>
  <section class="experiment-workspace">
    <nav class="experiment-workspace-tabs" role="tablist" :aria-label="t('detail.workspaceTabs')">
      <RouterLink
        :to="`/experiments/${id}`"
        class="experiment-workspace-tab"
        :class="{ 'experiment-workspace-tab-active': activeTab === 'overview' }"
        role="tab"
        :aria-selected="activeTab === 'overview'"
      >
        <LayoutDashboard aria-hidden="true" />
        <span>{{ t('detail.overviewTab') }}</span>
      </RouterLink>
      <RouterLink
        :to="analysisTarget"
        class="experiment-workspace-tab"
        :class="{ 'experiment-workspace-tab-active': activeTab === 'analysis' }"
        role="tab"
        :aria-selected="activeTab === 'analysis'"
      >
        <Sparkles aria-hidden="true" />
        <span>{{ t('detail.aiAnalysisTab') }}</span>
        <span
          v-if="analysis.generating"
          class="agent-stream-indicator"
          :title="t('detail.aiGenerating')"
        />
      </RouterLink>
      <RouterLink
        :to="`/experiments/${id}/visualizations`"
        class="experiment-workspace-tab whitespace-nowrap"
        :class="{ 'experiment-workspace-tab-active': activeTab === 'visualizations' }"
        role="tab"
        :aria-selected="activeTab === 'visualizations'"
      >
        <Grid2X2 aria-hidden="true" />
        <span>{{ t('heatmap.title') }}</span>
      </RouterLink>
    </nav>
    <RouterView />
  </section>
</template>
