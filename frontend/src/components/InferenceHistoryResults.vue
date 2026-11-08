<script setup lang="ts">
import { ChevronLeft, ChevronRight } from '@lucide/vue'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import type { InferenceRunRecord } from '../types'
import { formatIndependentScore } from './inferenceScoreDisplay'
import {
  buildInferenceHistoryTable,
  pageInferenceHistoryRows,
  type InferenceHistoryColumn,
  type InferenceHistoryRow,
} from './inferenceHistoryResults'

const props = defineProps<{
  run: InferenceRunRecord
  partnerLabel: string
}>()

const PAGE_SIZE = 20
const { t } = useI18n()
const page = ref(1)
const table = computed(() => buildInferenceHistoryTable(props.run))
const pagedRows = computed(() => pageInferenceHistoryRows(table.value.rows, page.value, PAGE_SIZE))

watch(
  () => props.run.id,
  () => {
    page.value = 1
  },
)

function columnLabel(column: InferenceHistoryColumn) {
  if (column.kind === 'index') return '#'
  if (column.kind === 'sequence') return t('inference.sequenceInput')
  if (column.kind === 'peptide') return t('inference.peptideSequence')
  if (column.kind === 'partner') return props.partnerLabel
  if (column.kind === 'prediction_label') return t('inference.predictionLabel')
  if (column.kind === 'interaction_score') return t('inference.score')
  if (column.kind === 'score_kind') return t('inference.scoreKind')
  if (column.kind === 'threshold') return t('inference.threshold')
  if (column.kind === 'is_interaction') return t('inference.interactionDecision')
  if (column.kind === 'top_label') return t('inference.topLabel')
  if (column.kind === 'top_score') return t('inference.topScore')
  return column.label || t('inference.labelFallback', { index: (column.scoreIndex ?? 0) + 1 })
}

function formatInteractionScore(value?: number) {
  if (!Number.isFinite(value)) return '-'
  return Math.abs(value as number) < 10
    ? (value as number).toFixed(4)
    : (value as number).toFixed(3)
}

function cellValue(row: InferenceHistoryRow, column: InferenceHistoryColumn) {
  if (column.kind === 'index') return String(row.index)
  if (column.kind === 'sequence') return row.sequence || '-'
  if (column.kind === 'peptide') return row.peptide || '-'
  if (column.kind === 'partner') return row.partner || '-'
  if (column.kind === 'prediction_label') return row.prediction.prediction_label || '-'
  if (column.kind === 'interaction_score') return formatInteractionScore(row.prediction.score)
  if (column.kind === 'score_kind') {
    const kind = row.prediction.score_kind || props.run.response.score_kind
    return kind === 'rank' ? t('ensemble.rankScore') : kind || '-'
  }
  if (column.kind === 'threshold')
    return formatIndependentScore(row.prediction.threshold ?? Number.NaN)
  if (column.kind === 'is_interaction') {
    if (typeof row.prediction.is_interaction !== 'boolean') return '-'
    return row.prediction.is_interaction ? t('inference.interact') : t('inference.noInteraction')
  }
  if (column.kind === 'top_label') return row.topLabel || '-'
  if (column.kind === 'top_score') return formatIndependentScore(row.topScore ?? Number.NaN)
  return formatIndependentScore(row.prediction.scores?.[column.scoreIndex ?? -1] ?? Number.NaN)
}

function isSequenceColumn(column: InferenceHistoryColumn) {
  return ['sequence', 'peptide', 'partner'].includes(column.kind)
}
</script>

<template>
  <section class="mt-3 border-t border-slate-100 pt-3" :aria-label="t('inference.historyResults')">
    <p
      v-if="run.response.ensemble?.batch_dependent"
      role="note"
      class="mb-3 text-xs leading-6 text-slate-500"
    >
      {{ t('ensemble.rankHint') }}
    </p>
    <div class="mb-3 flex flex-wrap items-center justify-between gap-2">
      <div>
        <p class="text-xs font-semibold text-slate-800">{{ t('inference.historyResults') }}</p>
        <p class="mt-0.5 text-xs text-slate-500">
          {{ t('inference.completeResultHint') }}
        </p>
      </div>
      <span class="text-xs tabular-nums text-slate-500">
        {{
          t('inference.resultRange', {
            start: pagedRows.start,
            end: pagedRows.end,
            total: table.rows.length,
          })
        }}
      </span>
    </div>

    <div v-if="table.rows.length" class="overflow-x-auto border-y border-slate-100">
      <table class="w-full min-w-max border-collapse text-left text-xs">
        <thead class="bg-slate-50 text-slate-600">
          <tr>
            <th
              v-for="column in table.columns"
              :key="column.key"
              scope="col"
              class="whitespace-nowrap border-b border-slate-100 px-3 py-2 font-semibold"
            >
              {{ columnLabel(column) }}
            </th>
          </tr>
        </thead>
        <tbody class="divide-y divide-slate-100">
          <tr v-for="row in pagedRows.rows" :key="row.index" class="align-top hover:bg-slate-50/70">
            <td
              v-for="column in table.columns"
              :key="column.key"
              class="px-3 py-2 text-slate-700"
              :class="isSequenceColumn(column) ? 'max-w-xs' : 'whitespace-nowrap tabular-nums'"
            >
              <code
                v-if="isSequenceColumn(column)"
                class="block whitespace-normal break-all font-mono text-[11px] leading-5"
              >
                {{ cellValue(row, column) }}
              </code>
              <span v-else>{{ cellValue(row, column) }}</span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <p v-else class="border-y border-slate-100 py-4 text-sm text-slate-500">
      {{ t('inference.noSavedResults') }}
    </p>

    <div v-if="pagedRows.totalPages > 1" class="mt-3 flex items-center justify-end gap-2">
      <button
        type="button"
        class="secondary-action icon-action compact-action"
        :disabled="pagedRows.page <= 1"
        :aria-label="t('inference.previousPage')"
        :title="t('inference.previousPage')"
        @click="page -= 1"
      >
        <ChevronLeft :size="16" aria-hidden="true" />
      </button>
      <span class="min-w-20 text-center text-xs tabular-nums text-slate-600">
        {{ t('inference.pageInfo', { current: pagedRows.page, total: pagedRows.totalPages }) }}
      </span>
      <button
        type="button"
        class="secondary-action icon-action compact-action"
        :disabled="pagedRows.page >= pagedRows.totalPages"
        :aria-label="t('inference.nextPage')"
        :title="t('inference.nextPage')"
        @click="page += 1"
      >
        <ChevronRight :size="16" aria-hidden="true" />
      </button>
    </div>
  </section>
</template>
