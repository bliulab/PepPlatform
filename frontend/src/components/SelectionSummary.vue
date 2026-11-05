<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
const props = defineProps<{ metadata?: Record<string, unknown> }>()
const { locale } = useI18n()
const zh = computed(() => locale.value.startsWith('zh'))
const contract = computed(
  () => props.metadata?.selection_contract as Record<string, unknown> | undefined,
)
const entries = computed(() => {
  if (!contract.value) return []
  const keys = ['checkpoint', 'threshold', 'tuning', 'weight_selection']
  const epochThreshold = contract.value.checkpoint_threshold as { digest?: string } | undefined
  const finalThreshold = contract.value.threshold as { digest?: string } | undefined
  if (epochThreshold && epochThreshold.digest !== finalThreshold?.digest)
    keys.push('checkpoint_threshold')
  return keys
    .filter((k) => contract.value?.[k])
    .map((k) => ({
      stage: k,
      definition: contract.value?.[k] as {
        id: string
        version: number
        params?: Record<string, number>
      },
    }))
})
</script>
<template>
  <div v-if="metadata?.evaluation_protocol_version" class="panel grid gap-3">
    <h2 class="text-sm font-semibold">{{ zh ? '选择规则' : 'Selection rules' }}</h2>
    <dl v-if="contract" class="grid gap-3 sm:grid-cols-2">
      <div v-for="entry in entries" :key="entry.stage" class="text-sm">
        <dt class="text-slate-500">{{ entry.stage }}</dt>
        <dd
          >{{ entry.definition.id }} · v{{ entry.definition.version }}
          <span
            v-if="entry.stage === 'checkpoint' && contract.checkpoint_scope === 'not_applicable'"
            >{{ zh ? '（不适用）' : ' (not applicable)' }}</span
          >
          <span v-if="entry.stage === 'tuning' && !contract.tuning_enabled">{{
            zh ? '（未运行）' : ' (not run)'
          }}</span>
          <span v-for="(value, key) in entry.definition.params" :key="key" class="ml-2"
            >{{ key }}={{ value }}</span
          >
        </dd>
      </div>
    </dl>
    <p v-else class="text-sm text-slate-500">{{
      zh
        ? '历史 artifact：使用保存的阈值与版本兼容定义。'
        : 'Historical artifact: uses its saved thresholds and compatibility definitions.'
    }}</p>
  </div>
</template>
