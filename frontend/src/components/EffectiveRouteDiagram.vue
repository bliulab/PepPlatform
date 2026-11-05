<script setup lang="ts">
import {
  Binary,
  Braces,
  CheckCircle2,
  ChevronDown,
  Combine,
  GitMerge,
  LoaderCircle,
  Target,
  Waypoints,
  Workflow,
} from '@lucide/vue'
import { computed, onBeforeUnmount, ref, useId, watch, type Component } from 'vue'
import { useI18n } from 'vue-i18n'
import type { EffectiveRoute } from '../types'
import {
  buildEffectiveRouteDiagramModel,
  isResolvedEffectiveRoute,
  routeStageFingerprint,
  type RouteDiagramStage,
  type RouteDiagramState,
  type RouteDiagramVariant,
  type RouteStageKind,
} from './effectiveRouteDiagram'

const props = withDefaults(
  defineProps<{
    route?: EffectiveRoute | null
    fallback?: string
    variant?: RouteDiagramVariant
    state?: RouteDiagramState
    collapsible?: boolean
    defaultExpanded?: boolean
    updating?: boolean
  }>(),
  {
    fallback: '',
    variant: 'draft',
    state: 'draft',
    collapsible: false,
    defaultExpanded: true,
    updating: false,
  },
)

const { t, te } = useI18n()
const contentId = `effective-route-${useId()}`
const expanded = ref(!props.collapsible || props.defaultExpanded)
const resolvedRoute = computed(() => (isResolvedEffectiveRoute(props.route) ? props.route : null))
const model = computed(() =>
  resolvedRoute.value ? buildEffectiveRouteDiagramModel(resolvedRoute.value) : null,
)
const showPredictor = computed(() => props.variant !== 'representation')
const summary = computed(() => {
  if (!model.value) return props.fallback || t('routeDiagram.fallback')
  return showPredictor.value ? model.value.summary : model.value.representationSummary
})
const stateKey = computed(() => (props.updating ? 'updating' : props.state))
const titleKey = computed(() => `routeDiagram.variants.${props.variant}`)
const stateLabel = computed(() => t(`routeDiagram.states.${stateKey.value}`))
const toggleLabel = computed(() =>
  expanded.value ? t('routeDiagram.collapse') : t('routeDiagram.expand'),
)

const stageIcons: Record<RouteStageKind, Component> = {
  tokenizer: Braces,
  encoder: Binary,
  inputAdapter: Waypoints,
  architecture: Workflow,
  readout: Combine,
  head: Target,
}

const changedStageKeys = ref(new Set<string>())
let changeTimer: number | undefined
let previousStageFingerprints = new Map<string, string>()

function allStages() {
  if (!model.value) return []
  return [
    ...model.value.streams.flatMap((stream) => stream.stages),
    ...(showPredictor.value ? model.value.predictorStages : []),
  ]
}

function clearChangeTimer() {
  if (changeTimer === undefined) return
  window.clearTimeout(changeTimer)
  changeTimer = undefined
}

watch(
  () => allStages().map((stage) => [stage.key, routeStageFingerprint(stage)] as const),
  (entries) => {
    const next = new Map(entries)
    if (previousStageFingerprints.size) {
      changedStageKeys.value = new Set(
        entries
          .filter(([key, fingerprint]) => previousStageFingerprints.get(key) !== fingerprint)
          .map(([key]) => key),
      )
      clearChangeTimer()
      changeTimer = window.setTimeout(() => {
        changedStageKeys.value = new Set()
        changeTimer = undefined
      }, 700)
    }
    previousStageFingerprints = next
  },
  { immediate: true },
)

watch(
  () => props.defaultExpanded,
  (value) => {
    if (props.collapsible) expanded.value = value
  },
)

onBeforeUnmount(clearChangeTimer)

function roleLabel(role: string) {
  if (role === 'peptide') return t('routeDiagram.peptide')
  if (role === 'partner') return t('routeDiagram.partner')
  return t('routeDiagram.sequence')
}

function outputLabel(value?: string) {
  if (!value) return ''
  const key = `routeDiagram.outputKinds.${value}`
  return te(key) ? t(key) : value
}

function activationLabel(value?: string) {
  const normalized = String(value || 'relu').toLowerCase()
  return normalized === 'silu' ? 'SiLU' : normalized.toUpperCase()
}

function stageDetail(stage: RouteDiagramStage) {
  const component = stage.component
  if (stage.kind === 'tokenizer') {
    const key = `routeDiagram.featureRoutes.${stage.featureRoute}`
    return stage.featureRoute && te(key) ? t(key) : String(stage.featureRoute || '')
  }
  if (stage.kind === 'encoder') {
    return [
      outputLabel(component.output_kind),
      component.frozen === true ? t('routeDiagram.fixed') : '',
    ]
      .filter(Boolean)
      .join(' · ')
  }
  if (stage.kind === 'inputAdapter') {
    return [
      String(component.normalization || component.fusion_mode || component.transformation || ''),
      component.output_dimension ? `${component.output_dimension}d` : '',
    ]
      .filter(Boolean)
      .join(' · ')
  }
  if (stage.kind === 'architecture' && component.learnable !== undefined) {
    return t(component.learnable ? 'routeDiagram.trainable' : 'routeDiagram.fixed')
  }
  if (stage.kind === 'readout') {
    return String(component.join_mode || '')
  }
  if (stage.kind === 'head') {
    if (component.id === 'legacy_gated_residual')
      return t('routeDiagram.legacyGatedResidualDetails')
    const hidden = component.hidden_dims ?? []
    if (!hidden.length) return t('routeDiagram.noHiddenLayers')
    return t('routeDiagram.headDetails', {
      widths: hidden.join(' → '),
      activation: activationLabel(component.activation),
      dropout: component.dropout ?? 0,
    })
  }
  return ''
}

function stageLabel(kind: RouteStageKind) {
  return t(`routeDiagram.stageLabels.${kind}`)
}

function toggleExpanded() {
  if (props.collapsible) expanded.value = !expanded.value
}
</script>

<template>
  <section
    class="effective-route"
    :class="[
      `effective-route-${variant}`,
      `effective-route-state-${stateKey}`,
      { 'effective-route-collapsible': collapsible },
    ]"
    :aria-label="t('routeDiagram.aria')"
    :aria-busy="updating || undefined"
  >
    <header class="effective-route-header">
      <div class="effective-route-heading">
        <span class="effective-route-brand-mark" aria-hidden="true">
          <Workflow />
        </span>
        <div class="effective-route-heading-copy">
          <div class="effective-route-context">
            <span class="effective-route-kicker">{{ t(titleKey) }}</span>
            <span class="effective-route-state">
              <LoaderCircle v-if="updating" class="effective-route-spinner" aria-hidden="true" />
              <CheckCircle2
                v-else-if="stateKey === 'resolved' || stateKey === 'actual'"
                aria-hidden="true"
              />
              <span v-else class="effective-route-state-dot" aria-hidden="true" />
              {{ stateLabel }}
            </span>
          </div>
          <h2 class="effective-route-title">{{ summary }}</h2>
        </div>
      </div>
      <button
        v-if="collapsible"
        type="button"
        class="effective-route-toggle"
        :aria-controls="contentId"
        :aria-expanded="expanded"
        :aria-label="toggleLabel"
        :title="toggleLabel"
        @click="toggleExpanded"
      >
        <ChevronDown :class="{ 'effective-route-toggle-open': expanded }" aria-hidden="true" />
      </button>
    </header>

    <Transition name="effective-route-reveal">
      <div v-if="expanded" :id="contentId" class="effective-route-content">
        <div
          v-if="model"
          class="effective-route-canvas"
          :class="{
            'effective-route-canvas-pair': model.streams.length > 1,
            'effective-route-canvas-representation': !showPredictor,
          }"
        >
          <div class="effective-route-representation-group">
            <div class="effective-route-group-heading">
              <h3><span aria-hidden="true">01</span>{{ t('routeDiagram.representation') }}</h3>
              <small>{{
                model.streams.length > 1
                  ? t('routeDiagram.source')
                  : roleLabel(model.streams[0]?.role || 'sequence')
              }}</small>
            </div>
            <div
              class="effective-route-streams"
              :class="{ 'effective-route-streams-pair': model.streams.length > 1 }"
            >
              <div v-for="stream in model.streams" :key="stream.key" class="effective-route-stream">
                <div v-if="model.streams.length > 1" class="effective-route-stream-role">
                  {{ roleLabel(stream.role) }}
                </div>
                <ol class="effective-route-stage-run effective-route-stage-run-stream">
                  <li
                    v-for="stage in stream.stages"
                    :key="stage.key"
                    class="effective-route-stage"
                    :class="[
                      `effective-route-stage-${stage.kind}`,
                      {
                        'effective-route-stage-muted': stage.muted,
                        'effective-route-stage-changed': changedStageKeys.has(stage.key),
                      },
                    ]"
                  >
                    <div class="effective-route-stage-heading">
                      <span class="effective-route-stage-icon" aria-hidden="true">
                        <component :is="stageIcons[stage.kind]" />
                      </span>
                      <span>{{ stageLabel(stage.kind) }}</span>
                    </div>
                    <strong>{{ stage.component.label || stage.component.id }}</strong>
                    <small v-if="stageDetail(stage)">{{ stageDetail(stage) }}</small>
                  </li>
                </ol>
              </div>
            </div>
          </div>

          <div v-if="showPredictor" class="effective-route-merge" aria-hidden="true">
            <span v-if="model.streams.length > 1" class="effective-route-merge-lines" />
            <GitMerge v-if="model.streams.length > 1" />
            <span v-else class="effective-route-merge-dot" />
          </div>

          <div v-if="showPredictor" class="effective-route-predictor">
            <div class="effective-route-group-heading effective-route-predictor-heading">
              <h3><span aria-hidden="true">02</span>{{ t('routeDiagram.predictor') }}</h3>
              <small>{{ t('routeDiagram.output') }}</small>
            </div>
            <ol class="effective-route-stage-run effective-route-stage-run-predictor">
              <li
                v-for="stage in model.predictorStages"
                :key="stage.key"
                class="effective-route-stage"
                :class="[
                  `effective-route-stage-${stage.kind}`,
                  {
                    'effective-route-stage-muted': stage.muted,
                    'effective-route-stage-changed': changedStageKeys.has(stage.key),
                  },
                ]"
              >
                <div class="effective-route-stage-heading">
                  <span class="effective-route-stage-icon" aria-hidden="true">
                    <component :is="stageIcons[stage.kind]" />
                  </span>
                  <span>{{ stageLabel(stage.kind) }}</span>
                </div>
                <strong>{{ stage.component.label || stage.component.id }}</strong>
                <small v-if="stageDetail(stage)">{{ stageDetail(stage) }}</small>
              </li>
            </ol>
          </div>
        </div>

        <p v-else class="effective-route-fallback" aria-live="polite">
          {{ fallback || t('routeDiagram.fallback') }}
        </p>

        <dl v-if="model && showPredictor" class="effective-route-metadata">
          <div>
            <dt>{{ t('routeDiagram.routeId') }}</dt>
            <dd
              ><code>{{ model.routeId }}</code></dd
            >
          </div>
          <div>
            <dt>{{ t('routeDiagram.trainingMode') }}</dt>
            <dd
              ><code>{{ model.trainingMode }}</code></dd
            >
          </div>
          <div>
            <dt>{{ t('routeDiagram.schemaVersion') }}</dt>
            <dd
              ><code>v{{ model.schemaVersion }}</code></dd
            >
          </div>
        </dl>
      </div>
    </Transition>
  </section>
</template>

<style scoped>
.effective-route {
  --route-state-color: var(--color-action-cool);
  container-type: inline-size;
  min-width: 0;
  overflow: hidden;
  border: 1px solid var(--color-line);
  border-radius: 0.5rem;
  background: var(--surface-strong);
  box-shadow: var(--shadow-soft);
}

.effective-route-review,
.effective-route-actual {
  --route-state-color: var(--color-moss);
}

.effective-route-state-recommended {
  --route-state-color: var(--color-brass);
}

.effective-route-state-updating {
  --route-state-color: var(--color-action-cool);
}

.effective-route-header {
  display: flex;
  min-height: 5rem;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.9rem 1rem;
}

.effective-route-heading,
.effective-route-context,
.effective-route-state,
.effective-route-stage-heading {
  display: flex;
  align-items: center;
}

.effective-route-heading {
  min-width: 0;
  gap: 0.8rem;
}

.effective-route-heading-copy {
  min-width: 0;
}

.effective-route-brand-mark {
  display: grid;
  width: 2.35rem;
  height: 2.35rem;
  flex: 0 0 2.35rem;
  place-items: center;
  border: 1px solid color-mix(in srgb, var(--route-state-color) 35%, var(--color-line));
  border-radius: 0.45rem;
  background: color-mix(in srgb, var(--route-state-color) 7%, var(--surface-bg));
  color: var(--route-state-color);
}

.effective-route-brand-mark svg {
  width: 1rem;
  height: 1rem;
  stroke-width: 1.8;
}

.effective-route-context {
  min-width: 0;
  flex-wrap: wrap;
  gap: 0.45rem 0.7rem;
}

.effective-route-kicker {
  color: var(--color-muted);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

.effective-route-state {
  gap: 0.3rem;
  color: var(--route-state-color);
  font-size: 0.66rem;
  font-weight: 600;
}

.effective-route-state svg {
  width: 0.78rem;
  height: 0.78rem;
}

.effective-route-state-dot {
  width: 0.42rem;
  height: 0.42rem;
  border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 0 3px color-mix(in srgb, currentColor 12%, transparent);
}

.effective-route-spinner {
  animation: effective-route-spin 0.8s linear infinite;
}

.effective-route-title {
  margin: 0.28rem 0 0;
  overflow-wrap: anywhere;
  color: var(--color-ink);
  font-size: 0.92rem;
  font-weight: 600;
  letter-spacing: 0;
  line-height: 1.45;
}

.effective-route-toggle {
  display: grid;
  width: 2.5rem;
  height: 2.5rem;
  flex: 0 0 2.5rem;
  place-items: center;
  border: 1px solid var(--line-soft);
  border-radius: 0.45rem;
  color: var(--color-muted);
}

.effective-route-toggle:hover {
  border-color: color-mix(in srgb, var(--route-state-color) 45%, var(--color-line));
  background: color-mix(in srgb, var(--route-state-color) 6%, transparent);
  color: var(--route-state-color);
}

.effective-route-toggle svg {
  width: 1rem;
  height: 1rem;
  transition: transform 180ms ease;
}

.effective-route-toggle-open {
  transform: rotate(180deg);
}

.effective-route-content {
  border-top: 1px solid var(--line-soft);
}

.effective-route-canvas {
  --stream-gap: 1.5rem;
  display: grid;
  min-width: 0;
  grid-template-columns: minmax(0, 1fr) 2.5rem minmax(0, 2fr);
  grid-template-rows: auto auto;
  align-items: stretch;
  row-gap: 0.85rem;
  background: color-mix(in srgb, var(--surface-bg) 58%, transparent);
  padding: 1rem 1.1rem 1.25rem;
}

.effective-route-canvas-representation {
  grid-template-columns: minmax(0, 1fr);
}

.effective-route-representation-group,
.effective-route-predictor {
  display: contents;
}

.effective-route-group-heading {
  display: flex;
  min-width: 0;
  min-height: 2rem;
  grid-area: 1 / 1;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.effective-route-predictor-heading {
  grid-column: 3;
}

.effective-route-group-heading h3 {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 0.5rem;
  margin: 0;
  color: var(--color-ink);
  font-size: 0.78rem;
  font-weight: 650;
  line-height: 1.4;
}

.effective-route-group-heading h3 > span {
  color: var(--color-accent);
  font-size: 0.61rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.effective-route-group-heading > small {
  color: var(--color-muted);
  font-size: 0.65rem;
  line-height: 1.4;
}

.effective-route-streams {
  display: grid;
  min-width: 0;
  grid-area: 2 / 1;
  gap: var(--stream-gap);
}

.effective-route-streams-pair {
  grid-template-rows: repeat(2, minmax(0, 1fr));
}

.effective-route-stream {
  display: grid;
  min-width: 0;
  grid-template-columns: minmax(0, 1fr);
}

.effective-route-streams-pair .effective-route-stream {
  grid-template-columns: 3.25rem minmax(0, 1fr);
  gap: 0.6rem;
}

.effective-route-stream-role {
  min-width: 0;
  align-self: center;
  overflow-wrap: anywhere;
  color: var(--color-muted);
  font-size: 0.65rem;
  font-weight: 600;
  line-height: 1.35;
}

.effective-route-stage-run {
  --stage-gap: 1.15rem;
  display: grid;
  min-width: 0;
  margin: 0;
  padding: 0;
  gap: var(--stage-gap);
  list-style: none;
}

.effective-route-stage-run-stream {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.effective-route-stage-run-predictor {
  grid-area: 2 / 3;
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.effective-route-canvas-pair .effective-route-stage-run-predictor {
  align-self: center;
}

.effective-route-stage {
  --stage-color: var(--color-action-cool);
  position: relative;
  min-width: 0;
  min-height: 9rem;
  border: 1px solid var(--line-soft);
  border-top: 2px solid color-mix(in srgb, var(--stage-color) 58%, var(--color-line));
  border-radius: 0.45rem;
  background: var(--surface-strong);
  padding: 0.65rem 0.6rem 0.75rem;
}

.effective-route-stage-encoder {
  --stage-color: var(--color-moss);
}

.effective-route-stage-inputAdapter,
.effective-route-stage-readout {
  --stage-color: var(--color-brass);
}

.effective-route-stage-architecture,
.effective-route-stage-head {
  --stage-color: var(--color-accent);
}

.effective-route-stage:not(:last-child)::after {
  position: absolute;
  top: 50%;
  left: 100%;
  width: var(--stage-gap);
  border-top: 1px solid color-mix(in srgb, var(--color-muted) 55%, transparent);
  content: '';
}

.effective-route-stage:not(:last-child)::before {
  position: absolute;
  z-index: 1;
  top: calc(50% - 0.17rem);
  right: calc(var(--stage-gap) * -1 + 0.1rem);
  width: 0.35rem;
  height: 0.35rem;
  border-top: 1px solid var(--color-muted);
  border-right: 1px solid var(--color-muted);
  content: '';
  transform: rotate(45deg);
}

.effective-route-stage-heading {
  min-width: 0;
  min-height: 2.25rem;
  gap: 0.4rem;
  color: var(--color-muted);
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 0;
  line-height: 1.45;
}

.effective-route-stage-icon {
  display: grid;
  width: 1.35rem;
  height: 1.35rem;
  flex: 0 0 1.35rem;
  place-items: center;
  border: 1px solid color-mix(in srgb, var(--stage-color) 32%, var(--color-line));
  border-radius: 0.32rem;
  color: var(--stage-color);
}

.effective-route-stage-icon svg {
  width: 0.72rem;
  height: 0.72rem;
  stroke-width: 1.9;
}

.effective-route-stage strong {
  display: block;
  min-width: 0;
  min-height: 2.7em;
  margin-top: 0.5rem;
  overflow-wrap: anywhere;
  color: var(--color-ink);
  font-size: 0.78rem;
  font-weight: 600;
  line-height: 1.35;
}

.effective-route-stage > small {
  display: block;
  margin-top: 0.22rem;
  overflow-wrap: anywhere;
  color: var(--color-muted);
  font-size: 0.62rem;
  line-height: 1.45;
}

.effective-route-stage-muted {
  opacity: 0.58;
}

.effective-route-stage-changed {
  animation: effective-route-stage-change 650ms ease-out;
}

.effective-route-merge {
  position: relative;
  display: grid;
  grid-area: 2 / 2;
  min-height: 4rem;
  place-items: center;
  color: var(--route-state-color);
}

.effective-route-merge::after {
  position: absolute;
  top: 50%;
  right: 0;
  left: 0;
  border-top: 1px solid color-mix(in srgb, var(--color-muted) 55%, transparent);
  content: '';
}

.effective-route-merge-lines {
  position: absolute;
  top: calc((100% - var(--stream-gap)) / 4);
  bottom: calc((100% - var(--stream-gap)) / 4);
  left: 50%;
  border-left: 1px solid color-mix(in srgb, var(--color-muted) 55%, transparent);
}

.effective-route-merge-lines::before,
.effective-route-merge-lines::after {
  position: absolute;
  left: -1.25rem;
  width: 1.25rem;
  border-top: 1px solid color-mix(in srgb, var(--color-muted) 55%, transparent);
  content: '';
}

.effective-route-merge-lines::before {
  top: 0;
}

.effective-route-merge-lines::after {
  bottom: 0;
}

.effective-route-canvas-pair .effective-route-merge::after {
  left: 50%;
}

.effective-route-merge-dot {
  z-index: 1;
  width: 0.4rem;
  height: 0.4rem;
  border-radius: 50%;
  background: var(--surface-strong);
  box-shadow: inset 0 0 0 1.5px var(--route-state-color);
}

.effective-route-merge svg {
  position: relative;
  z-index: 2;
  width: 1.15rem;
  height: 1.15rem;
  padding: 0.18rem;
  background: var(--surface-bg);
}

.effective-route-metadata {
  display: flex;
  min-width: 0;
  flex-wrap: wrap;
  gap: 0.55rem 1.25rem;
  margin: 0;
  border-top: 1px solid var(--line-soft);
  padding: 0.65rem 1rem;
  color: var(--color-muted);
  font-size: 0.61rem;
}

.effective-route-metadata > div {
  display: flex;
  min-width: 0;
  align-items: baseline;
  gap: 0.35rem;
}

.effective-route-metadata dt {
  font-weight: 600;
}

.effective-route-metadata dd {
  min-width: 0;
  margin: 0;
}

.effective-route-metadata code {
  overflow-wrap: anywhere;
  color: var(--color-ink);
  font-size: 0.62rem;
}

.effective-route-fallback {
  margin: 0;
  background: color-mix(in srgb, var(--surface-bg) 58%, transparent);
  padding: 1rem;
  color: var(--color-muted);
  font-size: 0.78rem;
  line-height: 1.55;
}

.effective-route-reveal-enter-active,
.effective-route-reveal-leave-active {
  transition:
    opacity 180ms ease,
    transform 180ms ease;
}

.effective-route-reveal-enter-from,
.effective-route-reveal-leave-to {
  opacity: 0;
  transform: translateY(-0.25rem);
}

@keyframes effective-route-spin {
  to {
    transform: rotate(360deg);
  }
}

@keyframes effective-route-stage-change {
  0% {
    background: color-mix(in srgb, var(--stage-color) 13%, transparent);
    box-shadow: 0 -2px 0 var(--stage-color);
  }
  100% {
    background: transparent;
    box-shadow: none;
  }
}

@container (max-width: 54rem) {
  .effective-route-canvas {
    grid-template-columns: minmax(0, 1fr);
    grid-template-rows: repeat(5, auto);
    row-gap: 0.75rem;
  }

  .effective-route-canvas-representation {
    grid-template-rows: auto auto;
  }

  .effective-route-streams,
  .effective-route-stage-run-predictor,
  .effective-route-predictor-heading,
  .effective-route-merge {
    grid-column: 1;
  }

  .effective-route-predictor-heading {
    grid-row: 4;
  }

  .effective-route-stage-run-predictor {
    grid-row: 5;
  }

  .effective-route-merge {
    grid-row: 3;
    min-height: 1.75rem;
  }

  .effective-route-merge::after,
  .effective-route-canvas-pair .effective-route-merge::after {
    top: 0;
    bottom: 0;
    left: 50%;
    right: auto;
    border-top: 0;
    border-left: 1px solid color-mix(in srgb, var(--color-muted) 55%, transparent);
  }

  .effective-route-merge-lines {
    display: none;
  }
}

@container (max-width: 36rem) {
  .effective-route-header {
    align-items: flex-start;
  }

  .effective-route-canvas {
    padding: 0.9rem;
  }

  .effective-route-streams-pair {
    grid-template-rows: none;
  }

  .effective-route-streams-pair .effective-route-stream {
    grid-template-columns: minmax(0, 1fr);
    gap: 0.5rem;
  }

  .effective-route-stream-role {
    color: var(--color-ink);
  }

  .effective-route-stage-run {
    --stage-gap: 1.15rem;
    grid-template-columns: minmax(0, 1fr);
  }

  .effective-route-stage {
    min-height: 0;
    padding: 0.6rem 0.75rem 0.8rem;
  }

  .effective-route-stage-heading {
    min-height: 1.5rem;
  }

  .effective-route-stage strong {
    min-height: 0;
  }

  .effective-route-stage:not(:last-child)::after {
    top: 100%;
    left: 1.42rem;
    width: 0;
    height: var(--stage-gap);
    border-top: 0;
    border-left: 1px solid color-mix(in srgb, var(--color-muted) 55%, transparent);
  }

  .effective-route-stage:not(:last-child)::before {
    top: calc(100% + var(--stage-gap) - 0.45rem);
    right: auto;
    left: 1.27rem;
    transform: rotate(135deg);
  }

  .effective-route-metadata {
    display: grid;
    gap: 0.35rem;
  }
}

@container (max-width: 24rem) {
  .effective-route-header {
    padding: 0.8rem;
  }

  .effective-route-brand-mark {
    display: none;
  }

  .effective-route-title {
    font-size: 0.84rem;
  }
}

@media (prefers-reduced-motion: reduce) {
  .effective-route-spinner,
  .effective-route-stage-changed {
    animation: none;
  }

  .effective-route-toggle svg,
  .effective-route-reveal-enter-active,
  .effective-route-reveal-leave-active {
    transition: none;
  }
}
</style>
