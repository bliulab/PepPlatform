<script setup lang="ts">
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { INTRO_DURATION_MS, TASK_TRANSITION_MS, introProgress } from '../landingPlayback'
import {
  applyDrag,
  createOrbit,
  decayOrbit,
  isClick,
  resolveTouchGesture,
  restOrbit,
} from './orbit'
import { createSculpture, type Sculpture, type SculptureKind } from './landingSculptures'
import {
  nearestParticleArc,
  type ParticleFeedback,
  type PeptideParticles,
} from './peptideParticles'
import { configureStageRenderer, createStageEnvironment, watchStageVisibility } from './stageScene'

// Welcome owns this preference. System motion settings never disable manual input.
const props = withDefaults(
  defineProps<{ kind: SculptureKind; paused?: boolean; playIntro?: boolean }>(),
  { paused: false, playIntro: false },
)
const emit = defineEmits<{
  ready: []
  fallback: []
  interactionstart: []
  interactionend: []
  introcomplete: []
}>()
const canvasRef = ref<HTMLCanvasElement | null>(null)
let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene | null = null
let camera: THREE.PerspectiveCamera | null = null
let composer: EffectComposer | null = null
let bloom: UnrealBloomPass | null = null
let environment: THREE.Texture | null = null
let current: Sculpture | null = null
let outgoing: Sculpture | null = null
let stars: THREE.Points | null = null
let resizeObserver: ResizeObserver | null = null
let stopVisibility: (() => void) | null = null
let widthQuery: MediaQueryList | null = null
let mobile = false
let visible = true
let failed = false
let frame = 0
let previous = 0
let clock = 0
let autoTime = 0
let idleTime = 0
let releasedAt = -2
let introMs = INTRO_DURATION_MS
let entranceMs = 1400
let transitionMs = TASK_TRANSITION_MS
let completeEmitted = false
let orbit = createOrbit()
let activePointer: number | null = null
let dragging = false
let pointerType = ''
let pointerInside = false
let origin = { x: 0, y: 0 }
let last = { x: 0, y: 0 }
let interactionRemaining = 0
let pulseAt = -10
let pulseOrigin = 0.5
let targetDistance = 12
let cameraDistance = 0
let snapParticles = false
let renderCount = 0
let mountFrame = 0
const pointerNdc = new THREE.Vector2()
const raycaster = new THREE.Raycaster()
const plane = new THREE.Plane()
const direction = new THREE.Vector3()
const zero = new THREE.Vector3()
const localPointer = new THREE.Vector3()
const feedback: ParticleFeedback = {
  dt: 0,
  pointer: null,
  radius: 1.5,
  displacement: 0.5,
  wave: null,
  pointScale: 1000,
}

function setOpacity(sculpture: Sculpture, amount: number) {
  sculpture.root.traverse((object) => {
    const renderable = object as THREE.Mesh
    if (!renderable.material) return
    const materials = Array.isArray(renderable.material)
      ? renderable.material
      : [renderable.material]
    for (const material of materials) {
      material.userData.baseOpacity ??= material.opacity
      material.opacity = (material.userData.baseOpacity as number) * amount
      material.depthWrite = amount > 0.97 && !(material instanceof THREE.ShaderMaterial)
    }
  })
}
function removeSculpture(sculpture: Sculpture | null) {
  if (!sculpture) return
  scene?.remove(sculpture.root)
  sculpture.dispose()
}
function changeSculpture() {
  if (!scene || failed) return
  removeSculpture(outgoing)
  outgoing = current
  current = createSculpture(props.kind, mobile)
  scene.add(current.root)
  transitionMs = outgoing && !props.paused ? 0 : TASK_TRANSITION_MS
  entranceMs = props.paused ? 1400 : 0
  orbit = createOrbit()
  pulseAt = -10
  if (props.paused) {
    removeSculpture(outgoing)
    outgoing = null
  }
  resize()
  draw(0)
  start()
}
function resize() {
  const canvas = canvasRef.value
  if (!canvas || !renderer || !camera || !current) return
  const { width, height } = canvas.getBoundingClientRect()
  if (width < 1 || height < 1) return
  renderer.setPixelRatio(mobile ? 1 : Math.min(window.devicePixelRatio || 1, 1.5))
  renderer.setSize(width, height, false)
  composer?.setPixelRatio(renderer.getPixelRatio())
  composer?.setSize(width, height)
  camera.aspect = width / height
  const vertical = THREE.MathUtils.degToRad(camera.fov / 2)
  const horizontal = Math.atan(Math.tan(vertical) * camera.aspect)
  targetDistance = (current.radius / Math.sin(Math.min(vertical, horizontal))) * 1.06
  if (!cameraDistance || !outgoing) cameraDistance = targetDistance
  camera.updateProjectionMatrix()
  feedback.pointScale =
    ((height * renderer.getPixelRatio()) / (2 * Math.tan(vertical))) * (mobile ? 1.4 : 1)
  const unitsPerPixel = (2 * targetDistance * Math.tan(vertical)) / height
  feedback.radius = unitsPerPixel * 110
  feedback.displacement = Math.min(0.8, unitsPerPixel * 36)
  draw(0)
}
function updateLocalPointer() {
  if (!camera || !current) return false
  raycaster.setFromCamera(pointerNdc, camera)
  plane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(direction), zero)
  if (!raycaster.ray.intersectPlane(plane, localPointer)) return false
  current.root.updateWorldMatrix(true, false)
  current.root.worldToLocal(localPointer)
  return true
}
function draw(dt: number) {
  if (!renderer || !scene || !camera || !current || failed) return
  clock += dt
  if (!props.paused) {
    autoTime += dt
    if (!dragging) idleTime += dt * THREE.MathUtils.smoothstep(clock - releasedAt, 0, 1.2)
    introMs = Math.min(INTRO_DURATION_MS, introMs + dt * 1000)
    entranceMs = Math.min(1400, entranceMs + dt * 1000)
    transitionMs = Math.min(TASK_TRANSITION_MS, transitionMs + dt * 1000)
    if (!dragging) orbit = decayOrbit(orbit, dt, { pitchLimit: 0.48 })
  }
  const opening = introProgress(introMs)
  const p = THREE.MathUtils.smoothstep(opening, 0, 1)
  const blend = THREE.MathUtils.smoothstep(transitionMs / TASK_TRANSITION_MS, 0, 1)
  const entrance = props.kind === 'hero' ? opening : entranceMs / 1400
  cameraDistance = props.paused
    ? targetDistance
    : THREE.MathUtils.lerp(cameraDistance || targetDistance, targetDistance, 1 - Math.exp(-dt * 7))
  camera.position.set(0, 0, cameraDistance * (1 + (1 - p) * 0.22))
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld()

  function pose(sculpture: Sculpture, opacity: number, offset: number) {
    sculpture.root.rotation.y =
      (props.kind === 'hero' ? (idleTime * Math.PI) / 15 : Math.sin(idleTime * 0.22) * 0.28) +
      orbit.yaw
    sculpture.root.rotation.x = (props.kind === 'hero' ? 0.08 : 0.3) + orbit.pitch
    sculpture.root.position.set(offset, Math.sin(autoTime * 0.55) * 0.055, 0)
    setOpacity(sculpture, opacity)
  }
  pose(current, outgoing ? blend : 1, outgoing ? (1 - blend) * 0.45 : 0)
  feedback.dt = dt
  feedback.pointer = pointerInside && !dragging && updateLocalPointer() ? localPointer : null
  feedback.wave = clock - pulseAt < 1.25 ? { origin: pulseOrigin, age: clock - pulseAt } : null
  feedback.instant = snapParticles
  current.update(autoTime, entrance, feedback)
  snapParticles = false
  if (outgoing) {
    pose(outgoing, 1 - blend, -blend * 0.45)
    outgoing.update(autoTime, 1)
    if (blend >= 1) {
      removeSculpture(outgoing)
      outgoing = null
    }
  }
  if (stars) stars.rotation.y = autoTime * 0.012
  const canvas = canvasRef.value
  if (canvas) {
    canvas.dataset.intro = opening >= 1 ? 'complete' : 'playing'
    canvas.dataset.phase = props.paused
      ? interactionRemaining > 0 || dragging
        ? 'interacting'
        : 'paused'
      : 'playing'
    canvas.dataset.interaction = dragging
      ? 'dragging'
      : feedback.wave
        ? 'wave'
        : pointerInside
          ? 'repelling'
          : 'idle'
    canvas.dataset.particleCount = String(
      (current.root.userData.particleCloud as PeptideParticles | undefined)?.count ?? 0,
    )
    canvas.dataset.deformation = Number(current.root.userData.deformation ?? 0).toFixed(4)
    canvas.dataset.renders = String(++renderCount)
  }
  if (opening >= 1 && !completeEmitted) {
    completeEmitted = true
    emit('introcomplete')
  }
  if (composer && !mobile) composer.render()
  else renderer.render(scene, camera)
}
function tick(now: number) {
  frame = 0
  if (!visible || document.hidden || failed) return
  if (props.paused && !dragging && interactionRemaining <= 0) {
    draw(0)
    return
  }
  const dt = previous ? Math.min(0.25, (now - previous) / 1000) : 0
  previous = now
  interactionRemaining = Math.max(0, interactionRemaining - dt)
  draw(dt)
  frame = requestAnimationFrame(tick)
}
function start() {
  if (frame || !visible || document.hidden || failed || !renderer) return
  if (props.paused && !dragging && interactionRemaining <= 0) {
    draw(0)
    return
  }
  previous = 0
  frame = requestAnimationFrame(tick)
}
function stop() {
  if (frame) cancelAnimationFrame(frame)
  frame = 0
  previous = 0
  if (canvasRef.value) canvasRef.value.dataset.phase = 'suspended'
}
function finishIntro() {
  if (introMs >= INTRO_DURATION_MS) return
  introMs = INTRO_DURATION_MS
  snapParticles = true
  if (visible && !document.hidden) draw(0)
}
function replayIntro() {
  introMs = 0
  autoTime = 0
  idleTime = 0
  completeEmitted = false
  pulseAt = -10
  orbit = createOrbit()
  draw(0)
  start()
}
defineExpose({ finishIntro, replayIntro })
function locatePointer(event: PointerEvent) {
  const rect = canvasRef.value?.getBoundingClientRect()
  if (!rect) return
  pointerNdc.set(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    1 - ((event.clientY - rect.top) / rect.height) * 2,
  )
  pointerInside = true
}
function beginDrag(event: PointerEvent) {
  finishIntro()
  dragging = true
  orbit = restOrbit(orbit)
  canvasRef.value?.setPointerCapture(event.pointerId)
  emit('interactionstart')
  start()
}
function pointerDown(event: PointerEvent) {
  if (event.button !== 0) return
  locatePointer(event)
  activePointer = event.pointerId
  pointerType = event.pointerType
  origin = { x: event.clientX, y: event.clientY }
  last = { ...origin }
  if (pointerType !== 'touch') beginDrag(event)
}
function pointerMove(event: PointerEvent) {
  locatePointer(event)
  interactionRemaining = 1.1
  if (event.pointerId === activePointer) {
    if (!dragging && pointerType === 'touch') {
      const gesture = resolveTouchGesture({
        dx: event.clientX - origin.x,
        dy: event.clientY - origin.y,
      })
      if (gesture === 'scroll') {
        activePointer = null
        pointerInside = false
        return
      }
      if (gesture === 'drag') beginDrag(event)
    }
    if (dragging) {
      orbit = applyDrag(orbit, event.clientX - last.x, event.clientY - last.y, { pitchLimit: 0.48 })
      last = { x: event.clientX, y: event.clientY }
    }
  }
  start()
}
function endPointer(event?: PointerEvent) {
  if (event && event.pointerId !== activePointer) return
  const click =
    event?.type === 'pointerup' && isClick(origin, { x: event.clientX, y: event.clientY }, 6)
  if (activePointer !== null && canvasRef.value?.hasPointerCapture(activePointer))
    canvasRef.value.releasePointerCapture(activePointer)
  activePointer = null
  if (dragging) {
    dragging = false
    releasedAt = clock
    if (props.paused) orbit = restOrbit(orbit)
    emit('interactionend')
  }
  if (click && current?.root.userData.particleCloud && updateLocalPointer()) {
    finishIntro()
    pulseOrigin = nearestParticleArc(
      current.root.userData.particleCloud as PeptideParticles,
      localPointer,
    )
    pulseAt = clock
  }
  if (pointerType === 'touch') pointerInside = false
  interactionRemaining = visible ? 1.5 : 0
  start()
}
function pointerLeave() {
  pointerInside = false
  interactionRemaining = 1.2
  start()
}
function loseContext(event: Event) {
  event.preventDefault()
  failed = true
  stop()
  endPointer()
  emit('fallback')
}
function createPostprocessing() {
  if (!renderer || !scene || !camera || composer) return
  composer = new EffectComposer(
    renderer,
    new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType }),
  )
  composer.addPass(new RenderPass(scene, camera))
  bloom = new UnrealBloomPass(
    new THREE.Vector2(1, 1),
    props.kind === 'hero' ? 0.24 : 0.16,
    0.5,
    1.0,
  )
  composer.addPass(bloom)
  composer.addPass(new OutputPass())
}
function syncQuality() {
  const changed = mobile !== (widthQuery?.matches ?? false)
  mobile = widthQuery?.matches ?? false
  if (!mobile && !composer) createPostprocessing()
  if (bloom) bloom.enabled = !mobile
  if (changed && scene) changeSculpture()
  resize()
}
watch(() => props.kind, changeSculpture)
watch(
  () => props.paused,
  () => {
    stop()
    endPointer()
    interactionRemaining = 0
    pointerInside = false
    if (props.paused) {
      finishIntro()
      transitionMs = TASK_TRANSITION_MS
      entranceMs = 1400
      snapParticles = true
    }
    draw(0)
    start()
  },
)
function initializeStage() {
  const canvas = canvasRef.value
  if (!canvas) return
  widthQuery = window.matchMedia('(max-width: 767px)')
  mobile = widthQuery.matches
  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance',
    })
    configureStageRenderer(renderer, {
      pixelRatio: mobile ? 1 : 1.5,
      bloom: !mobile,
      antialias: true,
    })
    renderer.toneMappingExposure = 1.05
    scene = new THREE.Scene()
    camera = new THREE.PerspectiveCamera(34, 1, 0.1, 120)
    environment = createStageEnvironment(renderer)
    scene.environment = environment
    scene.environmentIntensity = 0.45
    const key = new THREE.DirectionalLight(0xd8fbf4, 2)
    key.position.set(-3, 6, 8)
    const rim = new THREE.DirectionalLight(0x5eead4, 1.2)
    rim.position.set(4, -1, -3)
    scene.add(key, rim, new THREE.AmbientLight(0x0f766e, 0.4))
    const positions = new Float32Array((mobile ? 35 : 100) * 3)
    for (let i = 0; i < positions.length; i++)
      positions[i] = Math.sin(i * 127.1 + 23.8) * (i % 3 === 2 ? 3 : 7)
    stars = new THREE.Points(
      new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(positions, 3)),
      new THREE.PointsMaterial({
        color: 0x5eead4,
        size: 0.02,
        transparent: true,
        opacity: 0.35,
        depthWrite: false,
      }),
    )
    scene.add(stars)
    if (!mobile) createPostprocessing()
    introMs = props.playIntro && !props.paused ? 0 : INTRO_DURATION_MS
    changeSculpture()
    resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(canvas)
    widthQuery.addEventListener('change', syncQuality)
    stopVisibility = watchStageVisibility(canvas, {
      onEnter: () => {
        visible = true
        start()
      },
      onLeave: () => {
        visible = false
        stop()
        endPointer()
        pointerInside = false
        interactionRemaining = 0
      },
    })
    emit('ready')
    start()
  } catch {
    failed = true
    stop()
    emit('fallback')
  }
}
// Let the brand and controls paint before compiling the first WebGL frame.
onMounted(() => {
  mountFrame = requestAnimationFrame(() => {
    mountFrame = requestAnimationFrame(() => {
      mountFrame = 0
      initializeStage()
    })
  })
})

onBeforeUnmount(() => {
  if (mountFrame) cancelAnimationFrame(mountFrame)
  visible = false
  stop()
  endPointer()
  stopVisibility?.()
  resizeObserver?.disconnect()
  widthQuery?.removeEventListener('change', syncQuality)
  removeSculpture(current)
  removeSculpture(outgoing)
  stars?.geometry.dispose()
  ;(stars?.material as THREE.Material | undefined)?.dispose()
  environment?.dispose()
  if (composer) {
    for (const pass of composer.passes) pass.dispose()
    composer.dispose()
  }
  renderer?.dispose()
  renderer = null
})
</script>

<template>
  <canvas
    ref="canvasRef"
    class="molecular-stage-canvas"
    :data-sculpture="kind"
    data-draggable="true"
    aria-hidden="true"
    @pointerdown="pointerDown"
    @pointermove="pointerMove"
    @pointerup="endPointer"
    @pointercancel="endPointer"
    @lostpointercapture="endPointer"
    @pointerleave="pointerLeave"
    @webglcontextlost="loseContext"
  />
</template>
