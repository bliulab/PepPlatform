<script setup lang="ts">
import { AlertTriangle, ChevronRight, Pencil } from '@lucide/vue'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { localizedPreflightMessage } from '../errorLocalization'
import type { AgentExecutionAction } from '../types'

const props = defineProps<{
  actions: AgentExecutionAction[]
}>()

const emit = defineEmits<{
  (event: 'apply', action: AgentExecutionAction): void
  (event: 'edit', action: AgentExecutionAction, sourceEvent?: Event): void
  (event: 'inspect', action: AgentExecutionAction): void
}>()

const { t, te } = useI18n()

const lanes = ['conservative', 'balanced', 'exploratory']

const candidates = computed(() => {
  const byLane = new Map<string, AgentExecutionAction>()
  for (const action of props.actions) {
    const lane = action.lane || 'unknown'
    if (!byLane.has(lane) && lanes.includes(lane)) {
      byLane.set(lane, action)
    }
  }
  return lanes.map((lane) => byLane.get(lane))
})

const hasUnknownLane = computed(() =>
  props.actions.some((action) => !lanes.includes(action.lane || '')),
)

function candidateConfig(candidate?: AgentExecutionAction) {
  if (!candidate) return {}
  return candidate.preflight?.normalized_config ?? candidate.config ?? {}
}

function configKeys() {
  const keys = new Set<string>()
  for (const candidate of candidates.value) {
    for (const key of Object.keys(candidateConfig(candidate))) {
      keys.add(key)
    }
  }
  return Array.from(keys).sort()
}

function configValue(candidate: AgentExecutionAction | undefined, key: string) {
  const value = candidateConfig(candidate)[key]
  return value === undefined ? undefined : value
}

function configSource(candidate: AgentExecutionAction | undefined, key: string) {
  if (!candidate) return undefined
  return candidate.preflight?.parameter_sources?.[key]
}

function preflightMessages(candidate: AgentExecutionAction, kind: 'errors' | 'warnings') {
  const report = candidate.preflight
  const codes = kind === 'errors' ? (report?.error_codes ?? []) : (report?.warning_codes ?? [])
  const messages = kind === 'errors' ? (report?.errors ?? []) : (report?.warnings ?? [])
  return messages.map((message, index) => localizedPreflightMessage(codes[index], message, t, te))
}

function valueString(value: unknown) {
  if (value === undefined) return ''
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function keyDiffers(key: string) {
  const values = candidates.value.map((candidate) => JSON.stringify(configValue(candidate, key)))
  return new Set(values).size > 1
}

function applyCandidate(candidate?: AgentExecutionAction) {
  if (!candidate) return
  emit('apply', candidate)
}

function riskClass(candidate?: AgentExecutionAction) {
  const risk = candidate?.risk_level || 'medium'
  if (risk === 'low') return 'status-pill-success'
  if (risk === 'high') return 'status-pill-danger'
  return 'status-pill-warning'
}
</script>

<template>
  <div class="agent-compare grid gap-4">
    <div class="grid gap-4 xl:grid-cols-3">
      <div
        v-for="(candidate, index) in candidates"
        :key="lanes[index]"
        class="section-card grid content-start gap-4"
      >
        <div v-if="!candidate" class="muted-copy text-sm">
          {{ t('agent.noCandidateForLane', { lane: t(`agent.lanes.${lanes[index]}`) }) }}
        </div>

        <template v-else>
          <div>
            <div class="flex flex-wrap items-center gap-2">
              <span class="status-pill status-pill-marker">{{
                t(`agent.lanes.${lanes[index]}`)
              }}</span>
              <span
                class="status-pill"
                :class="candidate.valid ? 'status-pill-success' : 'status-pill-danger'"
              >
                {{ candidate.valid ? t('agent.valid') : t('agent.invalid') }}
              </span>
              <span class="status-pill" :class="riskClass(candidate)">
                {{ t('agent.riskLabel') }}:
                {{ t(`agent.risk.${candidate.risk_level || 'medium'}`) }}
              </span>
            </div>
            <span v-if="candidate.type === 'cancel_experiment'" class="config-chip">{{
              t('agent.actionTypes.cancel_experiment')
            }}</span>
            <h3 class="mt-3 text-lg font-black tracking-tight">{{ candidate.title }}</h3>
            <p v-if="candidate.rationale" class="muted-copy mt-2 text-sm leading-6">
              {{ candidate.rationale }}
            </p>
          </div>

          <div class="flex flex-wrap items-center gap-2">
            <button
              v-if="candidate.type === 'create_experiment'"
              type="button"
              class="primary-action compact-action"
              :disabled="!candidate.valid || candidate.status !== 'pending'"
              @click="applyCandidate(candidate)"
            >
              <ChevronRight aria-hidden="true" />
              {{ t('agent.reviewInWizard') }}
            </button>
            <button
              v-if="candidate.type === 'create_experiment' && candidate.status === 'pending'"
              type="button"
              class="secondary-action compact-action"
              @click="emit('edit', candidate, $event)"
            >
              <Pencil aria-hidden="true" />
              {{ candidate.edited ? t('agent.editAgain') : t('agent.editConfig') }}
            </button>
          </div>

          <div
            v-if="candidate.preflight?.errors?.length"
            class="agent-candidate-errors error-card"
            role="alert"
          >
            <strong>{{ t('agent.errors') }}</strong>
            <ul>
              <li v-for="error in preflightMessages(candidate, 'errors')" :key="error">
                {{ error }}
              </li>
            </ul>
          </div>
          <div class="agent-candidate-footer">
            <span v-if="candidate.preflight?.warnings?.length" class="agent-candidate-warning"
              ><AlertTriangle aria-hidden="true" />{{
                t('agent.workspaceWarningCount', { count: candidate.preflight.warnings.length })
              }}</span
            ><button
              type="button"
              class="agent-details-trigger"
              @click="emit('inspect', candidate)"
            >
              {{ t('agent.workspaceViewDetails') }}<ChevronRight aria-hidden="true" />
            </button>
          </div>
        </template>
      </div>
    </div>

    <details class="section-card agent-compare-config">
      <summary class="text-xs font-semibold">
        {{ t('agent.compare.configDiff') }}
      </summary>
      <div class="mt-3 overflow-x-auto">
        <table class="w-full text-left text-sm">
          <thead>
            <tr class="border-b border-slate-200">
              <th class="py-2 pr-4 font-semibold">{{ t('agent.config') }}</th>
              <th
                v-for="(candidate, index) in candidates"
                :key="lanes[index]"
                class="py-2 px-2 font-semibold"
              >
                {{
                  candidate
                    ? t(`agent.lanes.${candidate.lane || lanes[index]}`)
                    : t(`agent.lanes.${lanes[index]}`)
                }}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="key in configKeys()"
              :key="key"
              class="border-b border-slate-100"
              :class="keyDiffers(key) ? 'bg-amber-50/50' : ''"
            >
              <td
                class="py-2 pr-4 font-mono text-xs"
                :class="keyDiffers(key) ? 'font-bold' : 'text-slate-500'"
              >
                {{ key }}
              </td>
              <td
                v-for="(candidate, index) in candidates"
                :key="`${lanes[index]}-${key}`"
                class="py-2 px-2 font-mono text-xs"
                :class="keyDiffers(key) ? 'font-semibold' : 'text-slate-500'"
              >
                <span v-if="configValue(candidate, key) === undefined" class="text-slate-300">{{
                  t('agent.compare.noValue')
                }}</span>
                <span v-else>
                  {{ valueString(configValue(candidate, key)) }}
                  <span v-if="configSource(candidate, key) === 'default'" class="config-chip ml-1">
                    {{ t('agent.compare.defaultValue') }}
                  </span>
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </details>

    <div v-if="hasUnknownLane" class="text-xs text-stone-500">
      {{ t('agent.unknownLaneHint') }}
    </div>
  </div>
</template>
