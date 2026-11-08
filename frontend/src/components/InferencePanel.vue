<script setup lang="ts">
import { normalizeInputSchema, HLAI_PARTNER_HEADERS, isHlaiFastaHeader } from '../taskTerminology'
import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import {
  CheckCircle,
  Download,
  Eye,
  EyeOff,
  FileText,
  LoaderCircle,
  Play,
  RefreshCw,
  RotateCcw,
  Tags,
  Trash2,
  Upload,
  XCircle,
} from '@lucide/vue'
import { computed, onUnmounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { ApiError, api } from '../api/client'
import { AgentChatStreamError, streamPrediction } from '../api/chatStream'
import {
  localizedApiError,
  localizedReadinessCheck,
  localizedReadinessHint,
  localizedReadinessIssue,
} from '../errorLocalization'
import { formatLocalizedDate } from '../localization'
import type { InferenceProgressEvent, InferenceRunRecord, PredictionPair } from '../types'
import InferenceHistoryResults from './InferenceHistoryResults.vue'
import { cleanSequence, parsePairFile, parseSequenceFile } from './inferenceFileImport'
import {
  formatIndependentScore,
  independentScoreWidth,
  rankIndependentLabelScores,
  scoresUseUnitInterval,
} from './inferenceScoreDisplay'

const props = defineProps<{
  modelId: string
  disabled?: boolean
  reason?: string
  reasonCode?: string
  metadata?: Record<string, unknown>
}>()

type PairMode = 'single' | 'batch'
type PairSection = 'peptide' | 'protein'
type PairPartnerKind = 'protein' | 'hlai' | 'hlaii'

interface PairBlock {
  peptide: string[]
  protein: string[]
  seenPeptide: boolean
  seenProtein: boolean
  startLine: number
}

const EXAMPLE_PEPTIDE = 'NLSALGIFST'
const EXAMPLE_PROTEIN =
  'GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDCETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVPSGQEQRYTCHVQHEGLPKPLTLRWEP'
const EXAMPLE_FUNCTION_SEQUENCE = 'FLPILGNLLSGL'

const { locale, t, te } = useI18n()
const queryClient = useQueryClient()
const sequences = ref('')
const peptide = ref('')
const protein = ref('')
const pairBlocks = ref('')
const pairMode = ref<PairMode>('single')
const fileInput = ref<HTMLInputElement | null>(null)
const expandedRows = ref<Record<string, boolean>>({})
const activeHistoryRunId = ref('')
const downloadingRunId = ref('')
const deletingRunId = ref('')
const inputError = ref('')
const inferenceStage = ref('')
const inferenceCompleted = ref<number | undefined>(undefined)
const inferenceTotal = ref<number | undefined>(undefined)
const inferenceBatchCompleted = ref<number | undefined>(undefined)
const inferenceBatchTotal = ref<number | undefined>(undefined)
const inferenceElapsedMs = ref(0)
let inferenceTimer: number | undefined
const rankEnsemble = computed(() =>
  Boolean(
    (props.metadata?.web_ensemble as { batch_dependent?: boolean } | undefined)?.batch_dependent,
  ),
)
const inputSchema = computed(() => normalizeInputSchema(props.metadata?.input_schema))
const usesPairInput = computed(() =>
  ['peptide_protein_pairs', 'peptide_hlai_pairs', 'peptide_hla_pairs'].includes(inputSchema.value),
)
const partnerKind = computed<PairPartnerKind>(() => {
  if (inputSchema.value === 'peptide_hlai_pairs') return 'hlai'
  if (inputSchema.value === 'peptide_hla_pairs') return 'hlaii'
  return 'protein'
})
const partnerSequenceLabel = computed(() => {
  if (partnerKind.value === 'hlai') return t('inference.hlaiSequence')
  if (partnerKind.value === 'hlaii') return t('inference.hlaSequence')
  return t('inference.proteinSequence')
})
const singlePartnerPlaceholder = computed(() => {
  if (partnerKind.value === 'hlai') return t('inference.singleHlaiPlaceholder')
  if (partnerKind.value === 'hlaii') return t('inference.singleHlaPlaceholder')
  return t('inference.singleProteinPlaceholder')
})
const pairBatchPlaceholder = computed(() => {
  if (partnerKind.value === 'hlai') return t('inference.hlaiPairBatchPlaceholder')
  if (partnerKind.value === 'hlaii') return t('inference.hlaPairBatchPlaceholder')
  return t('inference.pairBatchPlaceholder')
})
const partnerBatchHeader = computed(() => {
  if (partnerKind.value === 'hlai') return 'hla'
  if (partnerKind.value === 'hlaii') return 'hla'
  return 'protein'
})
const partnerBatchMarker = computed(() => `>${partnerBatchHeader.value}`)
const pairBatchExample = computed(
  () => `>pair example_1
>peptide
${EXAMPLE_PEPTIDE}
>${partnerBatchHeader.value}
${EXAMPLE_PROTEIN.slice(0, 90)}
${EXAMPLE_PROTEIN.slice(90, 180)}
${EXAMPLE_PROTEIN.slice(180)}
//`,
)
const unavailableReason = computed(() => {
  const key = props.reasonCode ? `inference.${props.reasonCode}` : ''
  if (key && te(key)) return t(key)
  return t('inference.unavailable')
})
const readiness = useQuery({
  queryKey: computed(() => ['inference-readiness', props.modelId]),
  queryFn: () => api.inferenceReadiness(props.modelId),
  enabled: computed(() => Boolean(props.modelId && !props.disabled)),
  retry: false,
})
const runtimeBlocked = computed(() => readiness.data.value?.ready === false)
const readinessVisible = computed(
  () =>
    !props.disabled &&
    (readiness.isPending.value ||
      readiness.isError.value ||
      runtimeBlocked.value ||
      Boolean(readiness.data.value?.profile_required)),
)
const readinessTitle = computed(() => {
  if (readiness.isPending.value) return t('inference.checkingReadiness')
  if (readiness.isError.value) return t('inference.readinessCheckFailed')
  if (runtimeBlocked.value) return t('inference.readinessBlocked')
  if (readiness.data.value?.mode === 'prebuilt_db') return t('inference.readinessPrebuilt')
  if (readiness.data.value?.mode === 'build_on_demand') return t('inference.readinessOnDemand')
  if (readiness.data.value?.status === 'warning') return t('inference.readinessWarning')
  return t('inference.readinessReady')
})
const readinessMessage = computed(() => {
  const report = readiness.data.value
  if (!report) return ''
  if (!report.ready && report.errors[0]) return localizedReadinessIssue(report.errors[0], t, te)
  if (report.warnings[0]) return localizedReadinessIssue(report.warnings[0], t, te)
  if (report.mode === 'prebuilt_db') return t('inference.readinessPrebuiltHint')
  if (report.mode === 'build_on_demand') return t('inference.readinessOnDemandHint')
  return ''
})
const profileDiagnostics = computed(() => prediction.data.value?.diagnostics?.profile)
const predictionErrorHint = computed(() => {
  const error = prediction.error.value
  if (!(error instanceof ApiError) || !error.code) return ''
  const key = `apiErrorHints.${error.code}`
  return te(key) ? t(key) : ''
})
const submitDisabled = computed(
  () =>
    Boolean(props.disabled) ||
    readiness.isPending.value ||
    readiness.isError.value ||
    runtimeBlocked.value ||
    prediction.isPending.value,
)

const progressStageKeys = [
  'validating_bundle',
  'waiting_for_runtime',
  'loading_encoder',
  'encoder_ready',
  'extracting_features',
  'running_predictor',
  'saving_result',
]
const progressStageIndex = computed(() => {
  const index = progressStageKeys.indexOf(inferenceStage.value)
  return index < 0 ? 0 : index
})
const progressIndeterminate = computed(
  () =>
    inferenceStage.value === 'loading_encoder' ||
    !Number.isFinite(inferenceBatchTotal.value ?? inferenceTotal.value),
)
const progressStageLabel = (stage: string) => {
  const key = `inference.progress.${stage}`
  return te(key) ? t(key) : stage
}
const progressCountCompleted = computed(
  () => inferenceBatchCompleted.value ?? inferenceCompleted.value ?? 0,
)
const progressCountTotal = computed(() => inferenceBatchTotal.value ?? inferenceTotal.value ?? 0)
const progressElapsedLabel = computed(() =>
  t('inference.elapsed', { seconds: (inferenceElapsedMs.value / 1000).toFixed(1) }),
)

const inferenceRuns = useQuery({
  queryKey: computed(() => ['inference-runs', props.modelId]),
  queryFn: () => api.inferenceRuns(props.modelId),
  enabled: computed(() => Boolean(props.modelId && !props.disabled)),
})

const prediction = useMutation({
  mutationFn: async ({
    modelId,
    payload,
  }: {
    modelId: string
    payload: { sequences?: string[]; pairs?: PredictionPair[] }
  }) => {
    inferenceStage.value = 'waiting_for_runtime'
    inferenceCompleted.value = undefined
    inferenceTotal.value = undefined
    inferenceBatchCompleted.value = undefined
    inferenceBatchTotal.value = undefined
    inferenceElapsedMs.value = 0
    const started = performance.now()
    inferenceTimer = window.setInterval(() => {
      inferenceElapsedMs.value = performance.now() - started
    }, 100)
    let result
    try {
      for await (const event of streamPrediction(modelId, payload)) {
        if (event.type === 'heartbeat') continue
        if (event.type === 'stage') {
          const progress = event.data as InferenceProgressEvent
          inferenceStage.value = progress.stage
          if (typeof progress.completed === 'number') inferenceCompleted.value = progress.completed
          if (typeof progress.total === 'number') inferenceTotal.value = progress.total
          if (typeof progress.batch_completed === 'number')
            inferenceBatchCompleted.value = progress.batch_completed
          if (typeof progress.batch_total === 'number')
            inferenceBatchTotal.value = progress.batch_total
          if (typeof progress.elapsed_ms === 'number')
            inferenceElapsedMs.value = progress.elapsed_ms
        } else if (event.type === 'done') {
          result = event.data
        } else if (event.type === 'error') {
          throw new ApiError(event.data.message, Number(event.data.status_code) || 409, event.data)
        }
      }
    } catch (error) {
      if (error instanceof AgentChatStreamError) {
        if (error.status > 0) throw error
        throw new ApiError(t('inference.streamDisconnected'), 0, {
          code: 'inference_stream_disconnected',
        })
      }
      throw error
    }
    if (!result) {
      throw new ApiError(t('inference.streamEnded'), 0, { code: 'inference_stream_disconnected' })
    }
    return result
  },
  onMutate: () => {
    inputError.value = ''
  },
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ['inference-runs', props.modelId] })
  },
  onSettled: () => {
    if (inferenceTimer !== undefined) window.clearInterval(inferenceTimer)
    inferenceTimer = undefined
  },
})

const downloadMutation = useMutation({
  mutationFn: ({ runId }: { runId: string }) => api.inferenceRunCsv(runId),
  onMutate: ({ runId }) => {
    downloadingRunId.value = runId
  },
  onSuccess: (blob, variables) => {
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `inference-${variables.runId}.csv`
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
  },
  onSettled: () => {
    downloadingRunId.value = ''
  },
})

const deleteRunMutation = useMutation({
  mutationFn: ({ runId }: { runId: string }) => api.deleteInferenceRun(runId),
  onMutate: ({ runId }) => {
    deletingRunId.value = runId
  },
  onSuccess: (_response, variables) => {
    if (activeHistoryRunId.value === variables.runId) activeHistoryRunId.value = ''
    queryClient.invalidateQueries({ queryKey: ['inference-runs', props.modelId] })
  },
  onSettled: () => {
    deletingRunId.value = ''
  },
})

const predictionResults = computed(() => {
  const data = prediction.data.value
  if (!data) return []
  return data.predictions.map((row, rowIndex) => {
    const rankedScores = rankIndependentLabelScores(
      row.scores,
      data.labels,
      data.thresholds,
      row.predicted_labels,
      (scoreIndex) => t('inference.labelFallback', { index: scoreIndex + 1 }),
    )
    const rawInteractionScore = typeof row.score === 'number' ? row.score : row.scores[0]
    const threshold =
      typeof row.threshold === 'number' && Number.isFinite(row.threshold) ? row.threshold : 0.5
    const scoreKind =
      row.score_kind === 'rank' || data.score_kind === 'rank'
        ? 'rank'
        : row.score_kind === 'margin'
          ? 'margin'
          : row.score_kind === 'probability'
            ? 'probability'
            : rawInteractionScore >= 0 &&
                rawInteractionScore <= 1 &&
                threshold >= 0 &&
                threshold <= 1
              ? 'probability'
              : 'margin'
    const hasInteractionDecision =
      Boolean(row.peptide || row.protein) && Number.isFinite(rawInteractionScore)
    const isInteraction =
      typeof row.is_interaction === 'boolean'
        ? row.is_interaction
        : hasInteractionDecision
          ? rawInteractionScore >= threshold
          : false

    return {
      id: String(rowIndex),
      sequence: row.sequence ?? '',
      peptide: row.peptide ?? '',
      protein: row.protein ?? '',
      topScores: rankedScores.slice(0, 3),
      allScores: rankedScores,
      predictedLabels: rankedScores
        .filter((score) => score.isPredicted)
        .map((score) => score.label),
      labelScoreTitle:
        data.score_kind === 'rank'
          ? t('ensemble.rankScore')
          : scoresUseUnitInterval(row.scores)
            ? t('inference.independentLabelScoresUnitInterval')
            : t('inference.independentLabelScores'),
      hasInteractionDecision,
      interactionScore: Number.isFinite(rawInteractionScore) ? rawInteractionScore : 0,
      scoreKind,
      scoreTitle:
        scoreKind === 'rank'
          ? t('ensemble.rankScore')
          : scoreKind === 'margin'
            ? t('inference.interactionMarginScore')
            : t('inference.interactionProbability'),
      threshold,
      isInteraction,
      interactionLabel: isInteraction ? t('inference.interact') : t('inference.noInteraction'),
    }
  })
})

function formatInteractionScore(score: number) {
  if (!Number.isFinite(score)) return '-'
  return Math.abs(score) < 10 ? score.toFixed(4) : score.toFixed(3)
}

function formatDateTime(value?: string) {
  return formatLocalizedDate(value, locale.value, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function apiErrorText(error: unknown) {
  return localizedApiError(error, t, te)
}

function readinessCheckText(check: Parameters<typeof localizedReadinessCheck>[0]) {
  return localizedReadinessCheck(check, t, te)
}

function readinessCheckHint(check: Parameters<typeof localizedReadinessHint>[0]) {
  return localizedReadinessHint(check, t, te)
}

function runKindLabel(kind: string) {
  return kind === 'pairs' ? t('inference.pairConnector') : t('inference.sequenceInput')
}

function downloadRun(runId?: string) {
  if (!runId) return
  downloadMutation.mutate({ runId })
}

function toggleHistoryRun(runId: string) {
  activeHistoryRunId.value = activeHistoryRunId.value === runId ? '' : runId
}

function confirmDeleteRun(run: InferenceRunRecord) {
  const confirmed = window.confirm(
    t('inference.deleteHistoryConfirm', {
      date: formatDateTime(run.created_at),
      count: run.prediction_count,
    }),
  )
  if (!confirmed) return
  deleteRunMutation.mutate({ runId: run.id })
}

function toggleDetails(id: string) {
  expandedRows.value = {
    ...expandedRows.value,
    [id]: !expandedRows.value[id],
  }
}

function normalizedLines(value: string) {
  const rows = value.split('\n').map((row) => row.trim())
  while (rows.length && !rows[rows.length - 1]) rows.pop()
  return rows
}

function partnerColumnCandidates() {
  if (partnerKind.value === 'hlai') return HLAI_PARTNER_HEADERS
  if (partnerKind.value === 'hlaii')
    return ['hla', 'hlaii', 'hla_ii', 'hla-ii', 'pair_pseudo', 'protein']
  return ['protein', 'partner', 'target']
}

function pairsToBlocks(pairs: PredictionPair[]) {
  return pairs
    .map(
      (row, index) => `>pair imported_${index + 1}
>peptide
${row.peptide}
>${partnerBatchHeader.value}
${row.protein}
//`,
    )
    .join('\n')
}

function resetPredictionState() {
  inputError.value = ''
  expandedRows.value = {}
  prediction.reset()
}

function resetInput() {
  sequences.value = ''
  peptide.value = ''
  protein.value = ''
  pairBlocks.value = ''
  resetPredictionState()
}

function loadExample() {
  resetPredictionState()
  if (usesPairInput.value) {
    if (pairMode.value === 'batch') {
      pairBlocks.value = pairBatchExample.value
    } else {
      peptide.value = EXAMPLE_PEPTIDE
      protein.value = EXAMPLE_PROTEIN
    }
    return
  }
  sequences.value = EXAMPLE_FUNCTION_SEQUENCE
}

function openFilePicker() {
  fileInput.value?.click()
}

async function importFile(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  resetPredictionState()
  try {
    const text = await file.text()
    if (usesPairInput.value) {
      const pairs = parsePairFile(text, file.name, partnerColumnCandidates(), {
        columns: t('inference.filePairColumns', { partner: partnerBatchHeader.value }),
        empty: t('inference.fileNoRows'),
      })
      pairMode.value = 'batch'
      pairBlocks.value = pairsToBlocks(pairs)
      return
    }
    const parsedSequences = parseSequenceFile(text, file.name, t('inference.fileSequenceColumns'))
    if (!parsedSequences.length) throw new Error(t('inference.fileNoRows'))
    sequences.value = parsedSequences.join('\n')
  } catch (error) {
    inputError.value = error instanceof Error ? error.message : t('inference.fileImportFailed')
  } finally {
    input.value = ''
  }
}

function parsePairBlocks(value: string) {
  const pairs: PredictionPair[] = []
  let block: PairBlock | null = null
  let section: PairSection | null = null
  const lines = value.split(/\r?\n/)

  function requireBlock(lineNumber: number) {
    if (!block) throw new Error(t('inference.pairBatchStart', { line: lineNumber }))
    return block
  }

  function finalizeBlock(lineNumber: number) {
    const current = requireBlock(lineNumber)
    const parsedPeptide = cleanSequence(current.peptide.join(''))
    const parsedProtein = cleanSequence(current.protein.join(''))
    if (!current.seenPeptide || !current.seenProtein || !parsedPeptide || !parsedProtein) {
      throw new Error(
        t('inference.pairBatchMissingSequence', {
          line: current.startLine,
          partner: partnerSequenceLabel.value,
        }),
      )
    }
    pairs.push({ peptide: parsedPeptide, protein: parsedProtein })
    block = null
    section = null
  }

  lines.forEach((rawLine, index) => {
    const lineNumber = index + 1
    const line = rawLine.trim()
    if (!line) return
    const lower = line.toLowerCase()
    if (line === '//') {
      finalizeBlock(lineNumber)
      return
    }
    if (lower.startsWith('>pair')) {
      if (block) throw new Error(t('inference.pairBatchEndRequired'))
      block = {
        peptide: [],
        protein: [],
        seenPeptide: false,
        seenProtein: false,
        startLine: lineNumber,
      }
      section = null
      return
    }
    if (lower === '>peptide') {
      const current = requireBlock(lineNumber)
      if (current.seenPeptide)
        throw new Error(t('inference.pairBatchDuplicatePeptide', { line: lineNumber }))
      current.seenPeptide = true
      section = 'peptide'
      return
    }
    if (lower === '>protein') {
      const current = requireBlock(lineNumber)
      if (current.seenProtein)
        throw new Error(t('inference.pairBatchDuplicateProtein', { line: lineNumber }))
      current.seenProtein = true
      section = 'protein'
      return
    }
    if (
      (partnerKind.value === 'hlai' && isHlaiFastaHeader(lower)) ||
      (partnerKind.value === 'hlaii' &&
        ['>hla', '>hlaii', '>hla-ii', '>pair_pseudo'].includes(lower))
    ) {
      const current = requireBlock(lineNumber)
      if (current.seenProtein)
        throw new Error(
          t('inference.pairBatchDuplicatePartner', {
            line: lineNumber,
            partner: partnerSequenceLabel.value,
          }),
        )
      current.seenProtein = true
      section = 'protein'
      return
    }
    if (line.startsWith('>')) {
      throw new Error(t('inference.pairBatchUnknownHeader', { line: lineNumber }))
    }
    const current = requireBlock(lineNumber)
    if (!section)
      throw new Error(
        t('inference.pairBatchSection', { line: lineNumber, partner: partnerBatchMarker.value }),
      )
    current[section].push(line)
  })

  if (block) throw new Error(t('inference.pairBatchEndRequired'))
  if (!pairs.length) throw new Error(t('inference.pairBatchEmpty'))
  return pairs
}

function submit() {
  inputError.value = ''
  if (
    !props.modelId ||
    props.disabled ||
    runtimeBlocked.value ||
    readiness.isPending.value ||
    readiness.isError.value
  )
    return
  let payload: { sequences?: string[]; pairs?: PredictionPair[] }
  if (usesPairInput.value) {
    if (pairMode.value === 'single') {
      const parsedPeptide = cleanSequence(peptide.value)
      const parsedProtein = cleanSequence(protein.value)
      if (!parsedPeptide || !parsedProtein) {
        inputError.value = t('inference.pairRequired', { partner: partnerSequenceLabel.value })
        return
      }
      payload = { pairs: [{ peptide: parsedPeptide, protein: parsedProtein }] }
    } else {
      try {
        payload = { pairs: parsePairBlocks(pairBlocks.value) }
      } catch (error) {
        inputError.value = error instanceof Error ? error.message : t('inference.pairBatchInvalid')
        return
      }
    }
  } else {
    const rows = normalizedLines(sequences.value).filter(Boolean)
    if (!rows.length) return
    payload = { sequences: rows }
  }
  expandedRows.value = {}
  prediction.mutate({ modelId: props.modelId, payload })
}

watch(
  () => [props.modelId, usesPairInput.value],
  () => {
    expandedRows.value = {}
    activeHistoryRunId.value = ''
    inputError.value = ''
    prediction.reset()
    inferenceStage.value = ''
    inferenceCompleted.value = undefined
    inferenceTotal.value = undefined
    inferenceBatchCompleted.value = undefined
    inferenceBatchTotal.value = undefined
    inferenceElapsedMs.value = 0
  },
)

onUnmounted(() => {
  if (inferenceTimer !== undefined) window.clearInterval(inferenceTimer)
})
</script>

<template>
  <div class="panel">
    <p v-if="rankEnsemble" class="mb-4 text-sm leading-6 text-amber-900" role="note">{{
      t('ensemble.rankHint')
    }}</p>
    <p class="eyebrow">{{ t('inference.eyebrow') }}</p>
    <p v-if="disabled" class="warning-card mt-4">
      {{ unavailableReason }}
    </p>
    <template v-else>
      <div
        v-if="readinessVisible"
        class="mt-4 rounded-md border px-3 py-3"
        :class="
          runtimeBlocked || readiness.isError.value
            ? 'border-rose-200 bg-rose-50/70'
            : readiness.data.value?.status === 'warning'
              ? 'border-amber-200 bg-amber-50/70'
              : 'border-emerald-200 bg-emerald-50/60'
        "
        aria-live="polite"
      >
        <div class="flex flex-wrap items-start justify-between gap-3">
          <div class="min-w-0">
            <p class="text-sm font-semibold text-slate-900">{{ readinessTitle }}</p>
            <p v-if="readinessMessage" class="mt-1 text-xs leading-5 text-slate-600">
              {{ readinessMessage }}
            </p>
          </div>
          <button
            type="button"
            class="tertiary-action compact-action"
            :disabled="readiness.isFetching.value"
            @click="readiness.refetch()"
          >
            <RefreshCw aria-hidden="true" />
            {{ t('inference.readinessRetry') }}
          </button>
        </div>
        <details v-if="readiness.data.value?.checks.length" class="mt-2">
          <summary class="cursor-pointer text-xs font-semibold text-slate-600">
            {{ t('inference.readinessDetails') }}
          </summary>
          <div class="mt-2 grid gap-1.5">
            <div
              v-for="check in readiness.data.value.checks"
              :key="check.id"
              class="grid grid-cols-[0.65rem_minmax(0,1fr)] items-start gap-2 text-xs text-slate-600"
            >
              <span
                class="mt-1 h-2 w-2 rounded-full"
                :class="
                  check.status === 'error'
                    ? 'bg-rose-500'
                    : check.status === 'warning' || check.status === 'skipped'
                      ? 'bg-amber-500'
                      : 'bg-emerald-500'
                "
              />
              <span>
                {{ readinessCheckText(check) }}
                <span v-if="check.version" class="text-slate-400"> · {{ check.version }}</span>
                <span v-if="readinessCheckHint(check)" class="mt-0.5 block text-slate-500">
                  {{ readinessCheckHint(check) }}
                </span>
              </span>
            </div>
          </div>
        </details>
      </div>
      <div class="mt-4 flex flex-wrap items-center gap-3">
        <div v-if="usesPairInput" class="mode-toggle mode-toggle-compact inline-flex">
          <button
            type="button"
            class="mode-toggle-button"
            :class="{ 'mode-toggle-button-active': pairMode === 'single' }"
            :aria-pressed="pairMode === 'single'"
            @click="pairMode = 'single'"
          >
            {{ t('inference.singlePair') }}
          </button>
          <button
            type="button"
            class="mode-toggle-button"
            :class="{ 'mode-toggle-button-active': pairMode === 'batch' }"
            :aria-pressed="pairMode === 'batch'"
            @click="pairMode = 'batch'"
          >
            {{ t('inference.batchPairs') }}
          </button>
        </div>
        <div class="ml-auto flex flex-wrap gap-2">
          <input
            ref="fileInput"
            type="file"
            accept=".csv,.tsv,.fa,.fasta,.txt"
            class="sr-only"
            @change="importFile"
          />
          <button type="button" class="secondary-action compact-action" @click="openFilePicker">
            <Upload aria-hidden="true" />
            {{ t('inference.importFile') }}
          </button>
          <button type="button" class="secondary-action compact-action" @click="loadExample">
            <FileText aria-hidden="true" />
            {{ t('inference.examples') }}
          </button>
          <button type="button" class="tertiary-action compact-action" @click="resetInput">
            <RotateCcw aria-hidden="true" />
            {{ t('inference.reset') }}
          </button>
        </div>
      </div>

      <div v-if="usesPairInput" class="inference-input-zone mt-4">
        <div v-if="pairMode === 'single'" class="grid gap-4">
          <label class="grid gap-2 text-sm">
            <span class="font-medium text-slate-700">{{ t('inference.peptideSequence') }}</span>
            <input
              v-model="peptide"
              class="field-control"
              :placeholder="t('inference.singlePeptidePlaceholder')"
            />
          </label>
          <label class="grid gap-2 text-sm">
            <span class="font-medium text-slate-700">{{ partnerSequenceLabel }}</span>
            <textarea
              v-model="protein"
              class="field-control h-44 resize-y"
              :placeholder="singlePartnerPlaceholder"
            />
          </label>
        </div>

        <label v-else class="grid gap-2 text-sm">
          <span class="font-medium text-slate-700">{{ t('inference.batchPairs') }}</span>
          <textarea
            v-model="pairBlocks"
            class="field-control h-72 resize-y font-mono text-xs leading-5"
            spellcheck="false"
            :placeholder="pairBatchPlaceholder"
          />
        </label>
      </div>
      <div v-else class="inference-input-zone mt-4">
        <textarea
          v-model="sequences"
          class="field-control h-40 resize-y"
          :placeholder="t('inference.placeholder')"
        />
      </div>
      <div class="inference-actions mt-4">
        <button class="primary-action inference-submit" :disabled="submitDisabled" @click="submit">
          <LoaderCircle v-if="prediction.isPending.value" class="animate-spin" aria-hidden="true" />
          <Play v-else aria-hidden="true" />
          {{ prediction.isPending.value ? t('inference.predicting') : t('common.infer') }}
        </button>
      </div>
      <section
        v-if="prediction.isPending.value"
        class="inference-progress mt-4"
        aria-live="polite"
        :aria-label="progressStageLabel(inferenceStage)"
      >
        <div class="flex flex-wrap items-center justify-between gap-2">
          <p class="text-sm font-semibold text-slate-900">
            {{ progressStageLabel(inferenceStage) }}
          </p>
          <p class="text-xs tabular-nums text-slate-500">{{ progressElapsedLabel }}</p>
        </div>
        <div class="mt-3 flex flex-wrap gap-1.5" role="list">
          <span
            v-for="(stage, index) in progressStageKeys"
            :key="stage"
            role="listitem"
            class="inference-stage-dot"
            :class="{
              'inference-stage-dot--complete': index < progressStageIndex,
              'inference-stage-dot--active': index === progressStageIndex,
            }"
            :title="progressStageLabel(stage)"
          />
        </div>
        <div class="progress-track mt-3" aria-hidden="true">
          <div
            class="score-fill inference-runtime-progress"
            :class="{ 'inference-runtime-progress--indeterminate': progressIndeterminate }"
            :style="
              progressIndeterminate
                ? undefined
                : {
                    width: `${Math.min(
                      100,
                      Math.max(
                        0,
                        ((inferenceBatchCompleted ?? inferenceCompleted ?? 0) /
                          Math.max(1, inferenceBatchTotal ?? inferenceTotal ?? 1)) *
                          100,
                      ),
                    )}%`,
                  }
            "
          />
        </div>
        <p v-if="progressCountTotal > 0" class="mt-2 text-xs text-slate-500">
          {{
            t('inference.progressCount', {
              completed: progressCountCompleted,
              total: progressCountTotal,
            })
          }}
        </p>
      </section>
      <p v-if="inputError" class="mt-3 text-sm text-rose-600" aria-live="polite">
        {{ inputError }}
      </p>
      <div v-if="prediction.data.value" class="mt-5 space-y-4">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p class="text-sm font-semibold text-slate-900">{{ t('inference.latestResults') }}</p>
            <p class="text-xs text-slate-500">
              {{
                t('inference.predictionCount', { count: prediction.data.value.predictions.length })
              }}
            </p>
          </div>
          <button
            v-if="prediction.data.value.run_id"
            type="button"
            class="secondary-action compact-action"
            :disabled="downloadMutation.isPending.value"
            @click="downloadRun(prediction.data.value.run_id)"
          >
            <Download aria-hidden="true" />
            {{
              downloadMutation.isPending.value
                ? t('inference.downloading')
                : t('inference.downloadCsv')
            }}
          </button>
        </div>
        <div
          v-if="profileDiagnostics"
          class="rounded-md border px-3 py-2.5 text-xs"
          :class="
            profileDiagnostics.fallbacks
              ? 'border-amber-200 bg-amber-50/70 text-amber-900'
              : 'border-slate-200 bg-slate-50/80 text-slate-600'
          "
        >
          <p class="font-semibold">
            {{
              profileDiagnostics.fallbacks
                ? t('inference.profileFallbackWarning', {
                    count: profileDiagnostics.fallbacks,
                    mode: profileDiagnostics.fallback_mode,
                  })
                : t('inference.profileDiagnostics')
            }}
          </p>
          <p class="mt-1">
            {{
              t('inference.profileDiagnosticCounts', {
                generated: profileDiagnostics.generated,
                cached: profileDiagnostics.cache_hits,
                fallback: profileDiagnostics.fallbacks,
              })
            }}
          </p>
        </div>
        <p
          v-if="prediction.data.value?.legacy_evaluation_protocol"
          class="mb-4 rounded border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800"
        >
          {{ t('inference.legacyEvaluationWarning') }}
        </p>
        <article
          v-for="result in predictionResults"
          :key="result.id"
          class="dataset-card inference-result"
          :class="{
            'inference-result--positive': result.hasInteractionDecision && result.isInteraction,
            'inference-result--negative': result.hasInteractionDecision && !result.isInteraction,
            'inference-result--multilabel': !result.hasInteractionDecision,
          }"
        >
          <!-- Verdict badge -->
          <div
            v-if="result.hasInteractionDecision"
            class="inference-verdict"
            :class="
              result.isInteraction ? 'inference-verdict--positive' : 'inference-verdict--negative'
            "
          >
            <div class="inference-verdict-icon">
              <CheckCircle v-if="result.isInteraction" :size="26" aria-hidden="true" />
              <XCircle v-else :size="26" aria-hidden="true" />
            </div>
            <div class="inference-verdict-body">
              <p class="inference-verdict-label">{{ result.interactionLabel }}</p>
              <div class="inference-verdict-stats">
                <span
                  >{{ result.scoreTitle }}
                  <strong>{{ formatInteractionScore(result.interactionScore) }}</strong></span
                >
                <span
                  >{{ t('inference.threshold') }}
                  <strong>{{ formatInteractionScore(result.threshold) }}</strong></span
                >
              </div>
            </div>
          </div>
          <div v-else class="inference-verdict inference-verdict--multilabel">
            <div class="inference-verdict-icon">
              <Tags :size="26" aria-hidden="true" />
            </div>
            <div class="inference-verdict-body">
              <p class="inference-verdict-title">{{
                t('inference.positiveLabelsAboveThreshold')
              }}</p>
              <p class="inference-verdict-label">
                {{
                  result.predictedLabels.length
                    ? result.predictedLabels.join(', ')
                    : t('inference.noPositiveLabels')
                }}
              </p>
              <p class="inference-verdict-stats">{{ t('inference.perLabelThresholdHint') }}</p>
            </div>
          </div>

          <!-- Sequence info -->
          <div class="mt-4">
            <code v-if="result.sequence" class="sequence-code">{{ result.sequence }}</code>
            <div v-else class="grid min-w-0 gap-2">
              <div class="grid gap-1">
                <span class="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">
                  {{ t('inference.peptideSequence') }}
                </span>
                <code class="sequence-code block min-w-0">{{ result.peptide }}</code>
              </div>
              <div class="grid gap-1">
                <span class="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">
                  {{ partnerSequenceLabel }}
                </span>
                <code class="sequence-code block max-h-24 min-w-0 overflow-auto">{{
                  result.protein
                }}</code>
              </div>
            </div>
          </div>

          <!-- Multi-label scores with animated progress bars -->
          <div v-if="!result.hasInteractionDecision" class="mt-4 space-y-3">
            <div>
              <p class="text-xs font-semibold text-slate-700">{{
                t('inference.topIndependentLabelScores')
              }}</p>
              <p class="mt-0.5 text-xs text-slate-500">{{ result.labelScoreTitle }}</p>
            </div>
            <div
              v-for="(score, scoreIndex) in result.topScores"
              :key="score.index"
              class="independent-score independent-score--featured"
            >
              <div class="inference-score-heading">
                <span class="font-medium text-slate-800">{{ score.label }}</span>
                <span
                  class="inference-threshold-status"
                  :class="{
                    'inference-threshold-status--positive':
                      score.isPredicted && score.threshold !== undefined,
                    'inference-threshold-status--negative':
                      !score.isPredicted && score.threshold !== undefined,
                    'inference-threshold-status--unavailable': score.threshold === undefined,
                  }"
                >
                  <CheckCircle
                    v-if="score.isPredicted && score.threshold !== undefined"
                    :size="13"
                    aria-hidden="true"
                  />
                  <XCircle
                    v-else-if="score.threshold !== undefined"
                    :size="13"
                    aria-hidden="true"
                  />
                  {{
                    score.threshold === undefined
                      ? t('inference.thresholdUnavailable')
                      : score.isPredicted
                        ? t('inference.aboveThreshold')
                        : t('inference.belowThreshold')
                  }}
                </span>
              </div>
              <div class="inference-score-meta">
                <span
                  >{{ t('inference.score') }}
                  <strong>{{ formatIndependentScore(score.score) }}</strong></span
                >
                <span v-if="score.threshold !== undefined">
                  {{ t('inference.threshold') }}
                  <strong>{{ formatIndependentScore(score.threshold) }}</strong>
                </span>
                <span v-else>{{ t('inference.thresholdUnavailable') }}</span>
              </div>
              <div class="progress-track inference-score-track">
                <div
                  class="score-fill inference-score-bar"
                  :style="{
                    width: independentScoreWidth(score.score),
                    '--score-delay': `${scoreIndex * 80}ms`,
                  }"
                />
                <span
                  v-if="score.threshold !== undefined"
                  class="inference-threshold-marker"
                  :style="{ left: independentScoreWidth(score.threshold) }"
                  aria-hidden="true"
                />
              </div>
            </div>
          </div>

          <button
            v-if="
              !result.hasInteractionDecision && result.allScores.length > result.topScores.length
            "
            type="button"
            class="tertiary-action compact-action mt-4"
            @click="toggleDetails(result.id)"
          >
            <EyeOff v-if="expandedRows[result.id]" aria-hidden="true" />
            <Eye v-else aria-hidden="true" />
            {{
              expandedRows[result.id]
                ? t('inference.hideAllLabelScores')
                : t('inference.showAllLabelScores')
            }}
          </button>

          <div v-if="expandedRows[result.id]" class="mt-4 space-y-2 border-t border-slate-100 pt-4">
            <div
              v-for="score in result.allScores"
              :key="score.index"
              class="independent-score independent-score--compact"
            >
              <div class="inference-score-heading">
                <span class="truncate font-medium text-slate-700">{{ score.label }}</span>
                <span
                  class="inference-threshold-status"
                  :class="{
                    'inference-threshold-status--positive':
                      score.isPredicted && score.threshold !== undefined,
                    'inference-threshold-status--negative':
                      !score.isPredicted && score.threshold !== undefined,
                    'inference-threshold-status--unavailable': score.threshold === undefined,
                  }"
                >
                  {{
                    score.threshold === undefined
                      ? t('inference.thresholdUnavailable')
                      : score.isPredicted
                        ? t('inference.aboveThreshold')
                        : t('inference.belowThreshold')
                  }}
                </span>
              </div>
              <div class="inference-score-meta">
                <span
                  >{{ t('inference.score') }}
                  <strong>{{ formatIndependentScore(score.score) }}</strong></span
                >
                <span v-if="score.threshold !== undefined">
                  {{ t('inference.threshold') }}
                  <strong>{{ formatIndependentScore(score.threshold) }}</strong>
                </span>
              </div>
              <div class="progress-track inference-score-track">
                <div
                  class="score-fill-muted"
                  :style="{ width: independentScoreWidth(score.score) }"
                />
                <span
                  v-if="score.threshold !== undefined"
                  class="inference-threshold-marker inference-threshold-marker--muted"
                  :style="{ left: independentScoreWidth(score.threshold) }"
                  aria-hidden="true"
                />
              </div>
            </div>
          </div>
        </article>
      </div>
      <p v-if="prediction.error.value" class="mt-4 text-sm text-rose-600">
        {{ apiErrorText(prediction.error.value) }}
      </p>
      <p v-if="predictionErrorHint" class="mt-1 text-xs leading-5 text-rose-500">
        {{ predictionErrorHint }}
      </p>
      <div class="mt-6 border-t border-slate-100 pt-5">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p class="text-sm font-semibold text-slate-900">{{ t('inference.historyTitle') }}</p>
            <p class="text-xs text-slate-500">{{ t('inference.historySubtitle') }}</p>
          </div>
          <button
            type="button"
            class="secondary-action icon-action compact-action"
            :aria-label="t('inference.refreshHistory')"
            :title="t('inference.refreshHistory')"
            @click="inferenceRuns.refetch()"
          >
            <RefreshCw :size="16" aria-hidden="true" />
          </button>
        </div>
        <p v-if="inferenceRuns.isPending.value" class="mt-3 text-sm text-slate-500">
          {{ t('inference.loadingHistory') }}
        </p>
        <div v-else-if="(inferenceRuns.data.value ?? []).length" class="mt-3 space-y-2">
          <article
            v-for="run in inferenceRuns.data.value"
            :key="run.id"
            class="flex flex-wrap items-center justify-between gap-3 rounded-md border border-slate-100 bg-white/70 px-3 py-2"
          >
            <div class="min-w-0">
              <p class="text-sm font-medium text-slate-900">
                {{ runKindLabel(run.input_kind) }}
                <span class="text-slate-400">·</span>
                {{ t('inference.predictionCount', { count: run.prediction_count }) }}
              </p>
              <p class="text-xs text-slate-500">
                {{ formatDateTime(run.created_at) }}
                <span class="text-slate-300">·</span>
                {{ t('inference.inputCount', { count: run.input_count }) }}
                <template v-if="run.response.diagnostics?.profile?.fallbacks">
                  <span class="text-slate-300">·</span>
                  <span class="text-amber-700">
                    {{
                      t('inference.profileHistoryFallback', {
                        count: run.response.diagnostics.profile.fallbacks,
                      })
                    }}
                  </span>
                </template>
              </p>
            </div>
            <div class="flex shrink-0 items-center gap-2">
              <button
                type="button"
                class="secondary-action icon-action compact-action"
                :aria-label="
                  activeHistoryRunId === run.id
                    ? t('inference.hideHistoryResults')
                    : t('inference.viewHistoryResults')
                "
                :title="
                  activeHistoryRunId === run.id
                    ? t('inference.hideHistoryResults')
                    : t('inference.viewHistoryResults')
                "
                :aria-expanded="activeHistoryRunId === run.id"
                @click="toggleHistoryRun(run.id)"
              >
                <EyeOff v-if="activeHistoryRunId === run.id" :size="16" aria-hidden="true" />
                <Eye v-else :size="16" aria-hidden="true" />
              </button>
              <button
                type="button"
                class="secondary-action icon-action compact-action"
                :disabled="downloadMutation.isPending.value"
                :aria-label="t('inference.downloadCsv')"
                :title="t('inference.downloadCsv')"
                @click="downloadRun(run.id)"
              >
                <LoaderCircle
                  v-if="downloadingRunId === run.id"
                  :size="16"
                  class="animate-spin"
                  aria-hidden="true"
                />
                <Download v-else :size="16" aria-hidden="true" />
              </button>
              <button
                type="button"
                class="danger-action icon-action compact-action"
                :disabled="deleteRunMutation.isPending.value"
                :aria-label="t('inference.deleteHistory')"
                :title="t('inference.deleteHistory')"
                @click="confirmDeleteRun(run)"
              >
                <LoaderCircle
                  v-if="deletingRunId === run.id"
                  :size="16"
                  class="animate-spin"
                  aria-hidden="true"
                />
                <Trash2 v-else :size="16" aria-hidden="true" />
              </button>
            </div>
            <InferenceHistoryResults
              v-if="activeHistoryRunId === run.id"
              class="w-full basis-full"
              :run="run"
              :partner-label="partnerSequenceLabel"
            />
          </article>
        </div>
        <p v-else class="mt-3 text-sm text-slate-500">{{ t('inference.noHistory') }}</p>
        <p v-if="inferenceRuns.error.value" class="mt-3 text-sm text-rose-600">
          {{ apiErrorText(inferenceRuns.error.value) }}
        </p>
        <p v-if="downloadMutation.error.value" class="mt-3 text-sm text-rose-600">
          {{ apiErrorText(downloadMutation.error.value) }}
        </p>
        <p v-if="deleteRunMutation.error.value" class="mt-3 text-sm text-rose-600">
          {{
            t('inference.deleteHistoryFailed', {
              message: apiErrorText(deleteRunMutation.error.value),
            })
          }}
        </p>
      </div>
    </template>
  </div>
</template>

<style scoped>
/* --- Result card left accent stripe --- */

.inference-result {
  border-left-width: 3px;
  border-left-style: solid;
}

.inference-result--positive {
  border-left-color: var(--color-moss);
}

.inference-result--negative {
  border-left-color: var(--color-danger);
}

.inference-result--multilabel {
  border-left-color: var(--color-brass);
}

.inference-progress {
  border: 1px solid var(--color-line);
  border-radius: 0.6rem;
  background: color-mix(in srgb, var(--surface-bg) 88%, var(--color-moss) 12%);
  padding: 0.85rem 1rem;
}

.inference-stage-dot {
  width: 0.55rem;
  height: 0.55rem;
  border-radius: 999px;
  background: var(--color-line);
}

.inference-stage-dot--complete {
  background: var(--color-moss);
}

.inference-stage-dot--active {
  background: var(--color-brass);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--color-brass) 20%, transparent);
}

.inference-runtime-progress {
  min-width: 0.5rem;
  transition: width 180ms ease;
}

.inference-runtime-progress--indeterminate {
  width: 38%;
  animation: inference-progress-sweep 1.4s ease-in-out infinite;
}

@keyframes inference-progress-sweep {
  0% {
    transform: translateX(-115%);
  }
  100% {
    transform: translateX(285%);
  }
}

@media (prefers-reduced-motion: reduce) {
  .inference-runtime-progress--indeterminate {
    animation: none;
    width: 55%;
  }
}

/* --- Verdict badge --- */

.inference-verdict {
  display: flex;
  align-items: flex-start;
  gap: 0.85rem;
  border-radius: 0.6rem;
  padding: 0.9rem 1rem;
}

.inference-verdict--positive {
  background: linear-gradient(
    135deg,
    var(--color-success-bg),
    color-mix(in srgb, var(--color-moss) 15%, var(--surface-bg)) 70%
  );
}

.inference-verdict--negative {
  background: linear-gradient(
    135deg,
    var(--color-danger-bg),
    color-mix(in srgb, var(--color-danger) 10%, var(--surface-bg)) 70%
  );
}

.inference-verdict--multilabel {
  background: linear-gradient(
    135deg,
    var(--color-warning-bg),
    color-mix(in srgb, var(--color-brass) 12%, var(--surface-bg)) 70%
  );
}

.inference-verdict-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.6rem;
  height: 2.6rem;
  border-radius: 0.55rem;
  flex-shrink: 0;
}

.inference-verdict--positive .inference-verdict-icon {
  background: var(--color-moss);
  color: white;
}

.inference-verdict--negative .inference-verdict-icon {
  background: var(--color-danger);
  color: white;
}

.inference-verdict--multilabel .inference-verdict-icon {
  background: var(--color-brass);
  color: white;
}

.inference-verdict-body {
  min-width: 0;
  flex: 1;
}

.inference-verdict-title {
  margin-bottom: 0.2rem;
  color: var(--color-muted);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.inference-verdict-label {
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.25;
  word-break: break-word;
}

.inference-verdict--positive .inference-verdict-label {
  color: var(--color-moss);
}

.inference-verdict--negative .inference-verdict-label {
  color: var(--color-danger);
}

.inference-verdict--multilabel .inference-verdict-label {
  color: var(--color-warning-text);
}

.inference-verdict-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1rem;
  margin-top: 0.35rem;
  color: var(--color-muted);
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.inference-verdict-stats strong {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-variant-numeric: tabular-nums;
  color: var(--color-ink);
}

/* --- Score bar entrance animation --- */

@keyframes inference-score-enter {
  from {
    width: 0 !important;
  }
}

.inference-score-bar {
  animation: inference-score-enter 0.6s both cubic-bezier(0.22, 1, 0.36, 1);
  animation-delay: var(--score-delay, 0ms);
}

/* --- Independent label score and threshold comparison --- */

.independent-score {
  display: grid;
  gap: 0.4rem;
}

.independent-score--featured {
  border: 1px solid var(--line-soft);
  border-radius: 0.55rem;
  background: color-mix(in srgb, var(--surface-bg) 82%, transparent);
  padding: 0.7rem 0.8rem;
}

.independent-score--compact {
  padding: 0.4rem 0;
}

.inference-score-heading,
.inference-score-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.35rem 0.8rem;
}

.inference-score-meta {
  justify-content: flex-start;
  color: var(--color-muted);
  font-size: 0.72rem;
}

.inference-score-meta strong {
  color: var(--color-ink);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-variant-numeric: tabular-nums;
}

.inference-threshold-status {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  border-radius: 999px;
  padding: 0.18rem 0.5rem;
  font-size: 0.68rem;
  font-weight: 600;
  line-height: 1.2;
  white-space: nowrap;
}

.inference-threshold-status--positive {
  background: var(--color-success-bg);
  color: var(--color-moss);
}

.inference-threshold-status--negative {
  background: var(--color-danger-bg);
  color: var(--color-danger);
}

.inference-threshold-status--unavailable {
  background: var(--surface-muted);
  color: var(--color-muted);
}

.inference-score-track {
  position: relative;
  overflow: visible;
}

.inference-threshold-marker {
  position: absolute;
  z-index: 2;
  top: -0.2rem;
  bottom: -0.2rem;
  width: 2px;
  border-radius: 999px;
  background: var(--color-ink);
  box-shadow: 0 0 0 1px var(--surface-bg);
  transform: translateX(-1px);
}

.inference-threshold-marker--muted {
  opacity: 0.72;
}

/* --- Input zone --- */

.inference-input-zone {
  border: 1px solid var(--line-soft);
  border-radius: 0.7rem;
  background: var(--surface-bg);
  padding: 1rem;
}

/* --- Submit button --- */

.inference-submit {
  min-height: 3rem;
  padding: 0.85rem 1.6rem;
  font-size: 0.95rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  box-shadow: var(--shadow-lift);
}

.inference-submit:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: var(--shadow-lift);
}

.inference-actions {
  display: flex;
  justify-content: flex-end;
}

/* --- Dark mode --- */

:root[data-theme='dark'] .inference-verdict--multilabel .inference-verdict-label {
  color: var(--color-brass);
}

/* --- Reduced motion --- */

@media (prefers-reduced-motion: reduce) {
  .inference-score-bar {
    animation: none !important;
  }

  .inference-submit {
    transition: none !important;
  }
}
</style>
