<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { Capabilities } from '../types'

type Params = Record<string, number>
const props = defineProps<{
  draft: {
    checkpointObjective: string
    thresholdObjective: string
    mlTuningObjective: string
    checkpointObjectiveParams: Params
    thresholdObjectiveParams: Params
    tuningObjectiveParams: Params
  }
  objectives: Capabilities['selection_objectives']
  pair: boolean
  checkpoint: boolean
  tuning: boolean
}>()
const { locale } = useI18n()
const zh = computed(() => locale.value.startsWith('zh'))
const stages = computed(() =>
  [
    {
      id: 'checkpointObjective' as const,
      params: 'checkpointObjectiveParams' as const,
      title: zh.value ? 'Checkpoint 选择目标' : 'Checkpoint objective',
      visible: props.checkpoint,
    },
    {
      id: 'thresholdObjective' as const,
      params: 'thresholdObjectiveParams' as const,
      title: zh.value ? '阈值选择目标' : 'Threshold objective',
      visible: true,
    },
    {
      id: 'mlTuningObjective' as const,
      params: 'tuningObjectiveParams' as const,
      title: zh.value ? '超参数选择目标' : 'Tuning objective',
      visible: props.tuning,
    },
  ].filter((s) => s.visible),
)
function options(stage: string) {
  return (props.objectives ?? []).filter(
    (o) =>
      o.tasks.includes(props.pair ? 'binary' : 'multilabel') &&
      (stage !== 'thresholdObjective' || o.requires_predictions) &&
      (!props.pair || o.id !== 'macro_f1'),
  )
}
function change(stage: (typeof stages.value)[number], event: Event) {
  props.draft[stage.id] = (event.target as HTMLSelectElement).value
  props.draft[stage.params] = {}
}
function selected(stage: (typeof stages.value)[number]) {
  const id =
    props.draft[stage.id] ||
    (props.pair ? (stage.id === 'thresholdObjective' ? 'f1' : 'auc') : 'macro_f1')
  return id === 'f1' && !props.pair ? 'macro_f1' : id
}
</script>

<template>
  <fieldset class="mt-5 grid gap-4">
    <legend class="text-sm font-semibold">{{
      zh ? '模型与阈值选择' : 'Model and threshold selection'
    }}</legend>
    <div class="advanced-grid">
      <div v-for="stage in stages" :key="stage.id" class="grid content-start gap-2">
        <label class="grid gap-2 text-sm">
          {{ stage.title }}
          <select class="field-control" :value="selected(stage)" @change="change(stage, $event)">
            <option v-for="option in options(stage.id)" :key="option.id" :value="option.id">{{
              option.id
            }}</option>
          </select>
        </label>
        <small class="text-slate-500">{{
          (objectives ?? []).find((o) => o.id === selected(stage))?.requires_predictions
            ? zh
              ? '在所配置的阈值下计算'
              : 'Computed using the configured thresholds'
            : zh
              ? '使用连续预测分数；验证标签需包含两类'
              : 'Uses prediction scores; validation labels need both classes'
        }}</small>
        <template v-if="selected(stage) === 'dual_f1_sacc'">
          <label
            v-for="key in ['reference_f1', 'reference_sacc']"
            :key="key"
            class="grid gap-2 text-sm"
          >
            {{ key === 'reference_f1' ? 'Macro-F1' : 'SAcc' }} {{ zh ? '参考值' : 'reference' }}
            <input
              v-model.number="draft[stage.params][key]"
              type="number"
              min="0.000000001"
              step="any"
              required
              class="field-control"
            />
          </label>
        </template>
      </div>
    </div>
  </fieldset>
</template>
