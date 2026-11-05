<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'
import { FlaskConical, MessageSquareText } from '@lucide/vue'
import { useAgentChatStore } from '../stores/agentChat'

const { t } = useI18n()
const route = useRoute()
const chat = useAgentChatStore()
const activeTab = computed(() => String(route.meta.agentTab || 'design'))
const lastChatPath = ref('/agent/chat')
const chatTarget = computed(() =>
  chat.generating && chat.activeGenerationConversationId
    ? `/agent/chat/${chat.activeGenerationConversationId}`
    : lastChatPath.value,
)

watch(
  () => route.fullPath,
  (path) => {
    if (route.meta.agentTab === 'chat') lastChatPath.value = path
  },
  { immediate: true },
)

onBeforeUnmount(() => chat.stopGeneration())
</script>

<template>
  <section class="agent-workspace">
    <nav class="agent-workspace-tabs" role="tablist" :aria-label="t('agent.workspaceTabs')">
      <RouterLink
        to="/agent/design"
        class="agent-workspace-tab"
        :class="{ 'agent-workspace-tab-active': activeTab === 'design' }"
        role="tab"
        :aria-selected="activeTab === 'design'"
      >
        <FlaskConical aria-hidden="true" />
        <span>{{ t('agent.designMode') }}</span>
      </RouterLink>
      <RouterLink
        :to="chatTarget"
        class="agent-workspace-tab"
        :class="{ 'agent-workspace-tab-active': activeTab === 'chat' }"
        role="tab"
        :aria-selected="activeTab === 'chat'"
      >
        <MessageSquareText aria-hidden="true" />
        <span>{{ t('agent.chatMode') }}</span>
        <span
          v-if="chat.generating"
          class="agent-stream-indicator"
          :title="t('agent.chatGenerating')"
        />
      </RouterLink>
    </nav>

    <RouterView v-slot="{ Component }">
      <KeepAlive>
        <component :is="Component" />
      </KeepAlive>
    </RouterView>
  </section>
</template>
