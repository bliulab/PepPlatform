<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
const props = defineProps<{
  metadata?: Record<string, unknown>
  summary?: Record<string, unknown>
}>()
const { t } = useI18n()
const definition = computed(
  () =>
    props.metadata?.web_ensemble as
      | {
          method: string
          weight_source: string
          weights: number[]
          objective: string
          members: { id: string; name: string }[]
          batch_dependent: boolean
          test_splits: Record<string, Record<string, number | null>>
        }
      | undefined,
)
const thresholds = computed(
  () => (props.metadata?.operating_point as { values?: number[] } | undefined)?.values ?? [],
)
const validationMetrics = computed(() =>
  ['f1', 'sacc', 'auc', 'aupr'].flatMap((key) => {
    const value = props.summary?.[`val_${key}`]
    return typeof value === 'number'
      ? [{ name: key === 'sacc' ? 'SAcc' : key.toUpperCase(), value }]
      : []
  }),
)
</script>
<template>
  <section v-if="definition" class="panel grid gap-4">
    <h2 class="font-display text-xl">{{ t('ensemble.summary') }}</h2>
    <div class="flex flex-wrap gap-2"
      ><span class="config-chip">{{ t(`ensemble.${definition.method}`) }}</span
      ><span class="config-chip">{{ t(`ensemble.${definition.weight_source}`) }}</span
      ><span class="config-chip">{{ t(`ensemble.${definition.objective}`) }}</span></div
    >
    <p v-if="definition.batch_dependent" class="text-sm leading-6 text-amber-900">{{
      t('ensemble.rankHint')
    }}</p>
    <ol class="grid gap-2"
      ><li
        v-for="(member, index) in definition.members"
        :key="member.id"
        class="flex justify-between gap-4 text-sm"
        ><span>{{ index + 1 }}. {{ member.name }}</span
        ><strong>{{ definition.weights[index]?.toFixed(6) }}</strong></li
      ></ol
    >
    <details
      ><summary class="cursor-pointer text-sm">{{ t('ensemble.thresholds') }}</summary
      ><p class="mt-2 break-words font-mono text-xs">{{
        thresholds.map((x) => x.toFixed(6)).join(', ')
      }}</p></details
    >
    <div v-if="validationMetrics.length">
      <h3 class="mb-3 text-sm">{{ t('ensemble.validationMetrics') }}</h3>
      <dl class="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div v-for="metric in validationMetrics" :key="metric.name" class="section-card">
          <dt class="text-xs text-slate-500">{{ metric.name }}</dt
          ><dd class="mt-1 font-mono text-lg">{{ metric.value.toFixed(4) }}</dd>
        </div>
      </dl>
    </div>
    <p
      v-if="!Object.keys(definition.test_splits ?? {}).length"
      class="text-xs leading-6 text-slate-500"
      >{{ t('ensemble.validationOnly') }}</p
    >
    <div v-if="Object.keys(definition.test_splits ?? {}).length" class="overflow-auto"
      ><h3 class="mb-3 text-sm">{{ t('ensemble.testSplits') }}</h3
      ><table class="w-full text-left text-sm"
        ><thead
          ><tr
            ><th>{{ t('ensemble.dataset') }}</th
            ><th>AUC</th><th>AUPR</th><th>F1</th></tr
          ></thead
        ><tbody
          ><tr v-for="(metrics, split) in definition.test_splits" :key="split"
            ><td class="py-2">{{ split }}</td
            ><td>{{ metrics.auc?.toFixed(4) ?? '—' }}</td
            ><td>{{ metrics.aupr?.toFixed(4) ?? '—' }}</td
            ><td>{{ metrics.f1?.toFixed(4) ?? '—' }}</td></tr
          ></tbody
        ></table
      ></div
    >
  </section>
</template>
