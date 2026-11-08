<script setup lang="ts">
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { onBeforeUnmount, onMounted, ref } from 'vue'

import {
  configureStageRenderer,
  createStageEnvironment,
  glowTexture,
  STAGE_ENVIRONMENT_INTENSITY,
  watchStageVisibility,
} from './stageScene'
import { applyDrag, canOrbit, createOrbit, decayOrbit, restOrbit, type OrbitState } from './orbit'
import {
  anchorIndices,
  grooveHelices,
  grooveSheetMesh,
  peptideChain,
  type Vec3Point,
} from './hlaiGroove'

/**
 * Render the HLA-I illustration from `hlaiGroove.ts` and manage the camera,
 * meshes and WebGL lifecycle.
 */
const emit = defineEmits<{
  /** WebGL context lost: the caller should swap in the static schematic. */
  fallback: []
}>()

const STAGE = {
  background: 0x050f0e,
  helix: 0x0f766e,
  helixEmissive: 0x06322f,
  sheet: 0x0b4f49,
  peptide: 0x5eead4,
  anchor: 0xe8f3f1,
  glow: 0x0d9488,
} as const

/** Camera framing: oblique three-quarter view, slightly above the groove. */
const BASE_YAW = 0.52
const BASE_PITCH = 0.54
const CAMERA_DISTANCE = 11.5
const PITCH_MIN = 0.08
const PITCH_MAX = 1.15
/** Seconds of the enter animation (camera pulls back, structure settles). */
const INTRO_SECONDS = 1.2
const IDLE_SPIN = 0.11
const STILL_FRAME_SECONDS = 3

const canvasRef = ref<HTMLCanvasElement | null>(null)

let renderer: THREE.WebGLRenderer | null = null
let composer: EffectComposer | null = null
let scene: THREE.Scene | null = null
let camera: THREE.PerspectiveCamera | null = null
let group: THREE.Group | null = null
let resizeObserver: ResizeObserver | null = null
let stopWatchingStage: (() => void) | null = null
let motionQuery: MediaQueryList | null = null
let coarsePointerQuery: MediaQueryList | null = null
let introAmount = 0
let idleSpin = 0
let frame = 0
let running = false
let startedAt = 0
let stageVisible = false
let lastFrameAt = 0
let dragging = false
let pointerDownId: number | null = null
const downPoint = { x: 0, y: 0 }
const lastPoint = { x: 0, y: 0 }
let orbitState: OrbitState = createOrbit()
const disposables: { dispose: () => void }[] = []

function track<T extends { dispose: () => void }>(resource: T): T {
  disposables.push(resource)
  return resource
}

function toVector(point: Vec3Point) {
  return new THREE.Vector3(point.x, point.y, point.z)
}

function stageGlowTexture(): THREE.Texture {
  return track(
    glowTexture('rgba(94, 234, 212, 0.5)', 'rgba(13, 148, 136, 0.2)', 'rgba(5, 15, 14, 0)'),
  )
}

function buildScene() {
  const stage = new THREE.Scene()
  stage.background = new THREE.Color(STAGE.background)
  stage.fog = new THREE.FogExp2(STAGE.background, 0.062)

  const perspective = new THREE.PerspectiveCamera(36, 1, 0.1, 60)
  perspective.position.set(0, 4, 7)

  const root = new THREE.Group()
  stage.add(root)

  // Alpha1 / alpha2: one tube per helix.
  const helixMaterial = track(
    new THREE.MeshStandardMaterial({
      color: STAGE.helix,
      emissive: STAGE.helixEmissive,
      emissiveIntensity: 0.85,
      roughness: 0.34,
      metalness: 0.16,
    }),
  )
  for (const strand of grooveHelices()) {
    const curve = new THREE.CatmullRomCurve3(strand.map(toVector))
    const geometry = track(new THREE.TubeGeometry(curve, 260, 0.105, 10, false))
    root.add(new THREE.Mesh(geometry, helixMaterial))
  }

  // Beta sheet floor: the strands are stitched into one corrugated ribbon rather
  // than eight loose tubes, which is what makes it read as a floor.
  const sheetMaterial = track(
    new THREE.MeshStandardMaterial({
      color: STAGE.sheet,
      emissive: STAGE.helixEmissive,
      emissiveIntensity: 0.4,
      roughness: 0.45,
      metalness: 0.12,
      side: THREE.DoubleSide,
    }),
  )
  const sheet = grooveSheetMesh()
  const sheetGeometry = track(new THREE.BufferGeometry())
  sheetGeometry.setAttribute('position', new THREE.Float32BufferAttribute(sheet.positions, 3))
  sheetGeometry.setIndex(sheet.indices)
  sheetGeometry.computeVertexNormals()
  root.add(new THREE.Mesh(sheetGeometry, sheetMaterial))

  // Peptide: instanced residues plus two brighter anchors and a backbone line.
  const chain = peptideChain()
  const anchors = anchorIndices()
  const residueGeometry = track(new THREE.SphereGeometry(0.12, 18, 14))
  const residueMaterial = track(
    new THREE.MeshStandardMaterial({
      color: STAGE.peptide,
      emissive: 0x0b4f49,
      emissiveIntensity: 0.6,
      roughness: 0.28,
      metalness: 0.2,
    }),
  )
  const residues = new THREE.InstancedMesh(residueGeometry, residueMaterial, chain.length)
  const matrix = new THREE.Matrix4()
  chain.forEach((point, index) => {
    matrix.makeTranslation(point.x, point.y, point.z)
    residues.setMatrixAt(index, matrix)
  })
  residues.instanceMatrix.needsUpdate = true
  root.add(residues)

  const anchorGeometry = track(new THREE.SphereGeometry(0.17, 20, 16))
  const anchorMaterial = track(
    new THREE.MeshStandardMaterial({
      color: STAGE.anchor,
      emissive: STAGE.peptide,
      emissiveIntensity: 0.9,
      roughness: 0.22,
      metalness: 0.14,
    }),
  )
  for (const index of anchors) {
    const point = chain[index]
    if (!point) continue
    const anchor = new THREE.Mesh(anchorGeometry, anchorMaterial)
    anchor.position.set(point.x, point.y, point.z)
    root.add(anchor)
  }

  const backboneGeometry = track(new THREE.BufferGeometry().setFromPoints(chain.map(toVector)))
  const backboneMaterial = track(
    new THREE.LineBasicMaterial({
      color: STAGE.peptide,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  )
  root.add(new THREE.Line(backboneGeometry, backboneMaterial))

  // Ambient glow behind the structure.
  const glowMaterial = track(
    new THREE.SpriteMaterial({
      map: stageGlowTexture(),
      color: STAGE.glow,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      opacity: 0.34,
    }),
  )
  const glow = new THREE.Sprite(glowMaterial)
  glow.scale.set(8.5, 8.5, 1)
  glow.position.set(-0.6, 0.4, -3.4)
  stage.add(glow)

  stage.environment = track(createStageEnvironment(renderer!))
  stage.environmentIntensity = STAGE_ENVIRONMENT_INTENSITY

  const key = new THREE.DirectionalLight(0xd8fbf4, 2)
  key.position.set(3.5, 6, 5)
  stage.add(key)

  const rim = new THREE.DirectionalLight(0x5eead4, 1.4)
  rim.position.set(-5, 1.5, -5)
  stage.add(rim)

  stage.add(new THREE.AmbientLight(0x0f766e, 0.3))

  scene = stage
  camera = perspective
  group = root
}

function createRenderer() {
  const canvas = canvasRef.value
  if (!canvas) return
  const desktop = window.innerWidth >= 1024
  const quality = {
    pixelRatio: desktop ? Math.min(1.5, window.innerWidth / 1100) : 1,
    bloom: desktop,
    antialias: desktop,
  }
  const webgl = new THREE.WebGLRenderer({
    canvas,
    antialias: quality.antialias,
    alpha: false,
    powerPreference: 'high-performance',
  })
  configureStageRenderer(webgl, quality)
  renderer = webgl
  buildScene()

  if (quality.bloom && scene && camera) {
    const target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType })
    track(target)
    const effectComposer = new EffectComposer(webgl, target)
    effectComposer.addPass(new RenderPass(scene, camera))
    const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.25, 0.7, 0.85)
    track(bloom)
    effectComposer.addPass(bloom)
    effectComposer.addPass(new OutputPass())
    composer = effectComposer
  }
}

function resize() {
  const canvas = canvasRef.value
  if (!canvas || !renderer || !camera) return
  const rect = canvas.getBoundingClientRect()
  if (rect.width < 1 || rect.height < 1) return
  renderer.setSize(rect.width, rect.height, false)
  composer?.setSize(rect.width, rect.height)
  camera.aspect = rect.width / rect.height
  camera.updateProjectionMatrix()
}

function prefersReducedMotion() {
  return motionQuery?.matches ?? false
}

function easeOutCubic(value: number) {
  const clamped = Math.min(1, Math.max(0, value))
  return 1 - Math.pow(1 - clamped, 3)
}

function draw(seconds = (performance.now() - startedAt) / 1000) {
  if (!renderer || !scene || !camera) return
  const reduced = prefersReducedMotion()
  const now = performance.now()
  const delta = lastFrameAt > 0 ? Math.min(1, (now - lastFrameAt) / 1000) : 0
  lastFrameAt = now

  if (reduced) {
    introAmount = 1
  } else {
    introAmount = Math.min(1, introAmount + delta / INTRO_SECONDS)
    if (!dragging) {
      orbitState = decayOrbit(orbitState, delta, { pitchLimit: 0.6 })
      idleSpin += delta * IDLE_SPIN
    }
  }

  const eased = easeOutCubic(introAmount)
  const yaw = BASE_YAW + idleSpin + orbitState.yaw
  const pitch = Math.min(PITCH_MAX, Math.max(PITCH_MIN, BASE_PITCH + orbitState.pitch))
  const distance = CAMERA_DISTANCE * (1.14 - 0.14 * eased)
  camera.position.set(
    Math.sin(yaw) * Math.cos(pitch) * distance,
    Math.sin(pitch) * distance,
    Math.cos(yaw) * Math.cos(pitch) * distance,
  )
  camera.fov = 36
  camera.updateProjectionMatrix()
  camera.lookAt(0, -0.05, 0)

  if (group) {
    group.scale.setScalar(0.88 + 0.12 * eased)
    group.rotation.y = (1 - eased) * -0.35
    group.position.y = reduced ? 0 : Math.sin(seconds * 0.4) * 0.04
  }

  if (composer) composer.render()
  else renderer.render(scene, camera)
}

function loop() {
  if (!running) return
  draw()
  frame = requestAnimationFrame(loop)
}

function start() {
  if (running) return
  if (prefersReducedMotion()) {
    draw(STILL_FRAME_SECONDS)
    return
  }
  if (!stageVisible || document.hidden) return
  running = true
  frame = requestAnimationFrame(loop)
}

function stop() {
  running = false
  if (frame) cancelAnimationFrame(frame)
  frame = 0
}

function handlePointerDown(event: PointerEvent) {
  if (event.button !== 0) return
  pointerDownId = event.pointerId
  downPoint.x = event.clientX
  downPoint.y = event.clientY
  lastPoint.x = event.clientX
  lastPoint.y = event.clientY
  const canDrag = canOrbit({
    reducedMotion: prefersReducedMotion(),
    coarsePointer: coarsePointerQuery?.matches ?? false,
  })
  if (!canDrag) return
  dragging = true
  orbitState = restOrbit(orbitState)
  canvasRef.value?.setPointerCapture?.(event.pointerId)
  if (canvasRef.value) canvasRef.value.dataset.cursor = 'grabbing'
  event.preventDefault()
}

function handlePointerMove(event: PointerEvent) {
  if (!dragging || event.pointerId !== pointerDownId) return
  const deltaX = event.clientX - lastPoint.x
  const deltaY = event.clientY - lastPoint.y
  lastPoint.x = event.clientX
  lastPoint.y = event.clientY
  orbitState = applyDrag(orbitState, deltaX, deltaY, { pitchLimit: 0.6 })
}

function handlePointerUp(event: PointerEvent) {
  if (event.pointerId !== pointerDownId) return
  pointerDownId = null
  dragging = false
  canvasRef.value?.releasePointerCapture?.(event.pointerId)
  syncCursor()
}

function syncCursor() {
  const canvas = canvasRef.value
  if (!canvas) return
  const canDrag = canOrbit({
    reducedMotion: prefersReducedMotion(),
    coarsePointer: coarsePointerQuery?.matches ?? false,
  })
  if (canDrag) canvas.dataset.cursor = 'grab'
  else delete canvas.dataset.cursor
}

function handleMotionChange() {
  stop()
  if (prefersReducedMotion()) draw(STILL_FRAME_SECONDS)
  else start()
}

function handleContextLost(event: Event) {
  event.preventDefault()
  stop()
  emit('fallback')
}

onMounted(() => {
  createRenderer()
  if (!renderer) {
    emit('fallback')
    return
  }
  resize()

  motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
  coarsePointerQuery = window.matchMedia('(pointer: coarse)')
  motionQuery.addEventListener('change', handleMotionChange)
  coarsePointerQuery.addEventListener('change', handleMotionChange)
  syncCursor()

  if (typeof ResizeObserver !== 'undefined' && canvasRef.value) {
    resizeObserver = new ResizeObserver(() => {
      resize()
      if (prefersReducedMotion()) draw(STILL_FRAME_SECONDS)
    })
    resizeObserver.observe(canvasRef.value)
  }

  if (canvasRef.value) {
    stopWatchingStage = watchStageVisibility(canvasRef.value, {
      onEnter: () => {
        stageVisible = true
        startedAt = performance.now()
        start()
      },
      onLeave: () => {
        stageVisible = false
        stop()
      },
    })
    canvasRef.value.addEventListener('pointerdown', handlePointerDown)
    canvasRef.value.addEventListener('pointermove', handlePointerMove)
    canvasRef.value.addEventListener('pointerup', handlePointerUp)
    canvasRef.value.addEventListener('pointercancel', handlePointerUp)
    canvasRef.value.addEventListener('webglcontextlost', handleContextLost)
  }

  startedAt = performance.now()
  if (prefersReducedMotion()) draw(STILL_FRAME_SECONDS)
  else start()
})

onBeforeUnmount(() => {
  stop()
  resizeObserver?.disconnect()
  stopWatchingStage?.()
  motionQuery?.removeEventListener('change', handleMotionChange)
  coarsePointerQuery?.removeEventListener('change', handleMotionChange)
  const canvas = canvasRef.value
  canvas?.removeEventListener('pointerdown', handlePointerDown)
  canvas?.removeEventListener('pointermove', handlePointerMove)
  canvas?.removeEventListener('pointerup', handlePointerUp)
  canvas?.removeEventListener('pointercancel', handlePointerUp)
  canvas?.removeEventListener('webglcontextlost', handleContextLost)
  resizeObserver = null
  stopWatchingStage = null
  motionQuery = null
  coarsePointerQuery = null
  for (const resource of disposables.splice(0)) resource.dispose()
  composer = null
  group = null
  camera = null
  scene = null
  renderer?.dispose()
  renderer = null
})
</script>

<template>
  <canvas
    ref="canvasRef"
    class="landing-groove-canvas"
    data-quality="groove"
    aria-hidden="true"
    focusable="false"
  />
</template>
