import { normalizePairFields } from '../pairTerminology'
import type { EffectiveRoute, EffectiveRouteStream, RouteComponent } from '../types'

export type RouteDiagramVariant = 'representation' | 'draft' | 'review' | 'actual'
export type RouteDiagramState = 'recommended' | 'draft' | 'resolved' | 'actual'
export type RouteStageKind =
  'tokenizer' | 'encoder' | 'inputAdapter' | 'architecture' | 'readout' | 'head'

export interface RouteDiagramStage {
  key: string
  kind: RouteStageKind
  component: RouteComponent
  featureRoute?: string
  muted: boolean
}

export interface RouteDiagramStream {
  key: string
  role: string
  stages: RouteDiagramStage[]
}

export interface EffectiveRouteDiagramModel {
  summary: string
  representationSummary: string
  streams: RouteDiagramStream[]
  predictorStages: RouteDiagramStage[]
  routeId: string
  trainingMode: string
  schemaVersion: number
}

function componentLabel(component: RouteComponent) {
  return String(component.label || component.id || '').trim()
}

function componentIsMuted(component: RouteComponent) {
  return ['identity', 'none'].includes(
    String(component.id || '')
      .trim()
      .toLowerCase(),
  )
}

function stage(
  key: string,
  kind: RouteStageKind,
  component: RouteComponent,
  featureRoute?: string,
): RouteDiagramStage {
  return { key, kind, component, featureRoute, muted: componentIsMuted(component) }
}

function streamRepresentation(stream: EffectiveRouteStream, separator: string) {
  return [componentLabel(stream.tokenizer), componentLabel(stream.encoder)]
    .filter(Boolean)
    .join(separator)
}

function uniqueLabels(values: string[]) {
  return values.filter((value, index) => value && values.indexOf(value) === index)
}

export function isResolvedEffectiveRoute(route: unknown): route is EffectiveRoute {
  if (!route || typeof route !== 'object') return false
  const candidate = route as Partial<EffectiveRoute>
  return Boolean(
    Array.isArray(candidate.streams) &&
    candidate.streams.length > 0 &&
    candidate.predictor &&
    candidate.predictor.input_adapter &&
    candidate.predictor.architecture &&
    candidate.predictor.readout &&
    candidate.predictor.head,
  )
}

export function buildEffectiveRouteDiagramModel(route: EffectiveRoute): EffectiveRouteDiagramModel {
  route = normalizePairFields(route)
  const streams = route.streams.map((stream, index) => ({
    key: `${stream.role}-${index}`,
    role: stream.role,
    stages: [
      stage(`${stream.role}-${index}-tokenizer`, 'tokenizer', stream.tokenizer, stream.feature_route),
      stage(`${stream.role}-${index}-encoder`, 'encoder', stream.encoder, stream.feature_route),
    ],
  }))
  const representationSummary =
    route.streams.length === 1
      ? streamRepresentation(route.streams[0], ' → ')
      : route.streams.map((stream) => streamRepresentation(stream, ' / ')).join(' + ')
  const predictorLabels = uniqueLabels([
    route.predictor.architecture.id === 'identity'
      ? ''
      : componentLabel(route.predictor.architecture),
    componentLabel(route.predictor.head),
  ])

  return {
    summary: [...(representationSummary ? [representationSummary] : []), ...predictorLabels].join(
      ' → ',
    ),
    representationSummary,
    streams,
    predictorStages: [
      stage('predictor-input-adapter', 'inputAdapter', route.predictor.input_adapter),
      stage('predictor-architecture', 'architecture', route.predictor.architecture),
      stage('predictor-readout', 'readout', route.predictor.readout),
      stage('predictor-head', 'head', route.predictor.head),
    ],
    routeId: route.route_id,
    trainingMode: route.training_mode,
    schemaVersion: route.schema_version,
  }
}

export function routeStageFingerprint(stage: RouteDiagramStage) {
  return JSON.stringify({
    id: stage.component.id,
    label: stage.component.label,
    outputKind: stage.component.output_kind,
    transformation: stage.component.transformation,
    frozen: stage.component.frozen,
    learnable: stage.component.learnable,
    normalization: stage.component.normalization,
    outputDimension: stage.component.output_dimension,
    fusionMode: stage.component.fusion_mode,
    joinMode: stage.component.join_mode,
    hiddenDims: stage.component.hidden_dims,
    activation: stage.component.activation,
    dropout: stage.component.dropout,
  })
}
