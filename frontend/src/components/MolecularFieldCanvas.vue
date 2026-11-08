<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'

import {
  createMolecularField,
  sampleMolecularField,
  type MolecularField,
  type PointerState,
} from './molecularField'

/**
 * Thin canvas painter for the landing hero field. All geometry lives in
 * `molecularField.ts`; this component only owns lifecycle concerns:
 * device-pixel scaling, resizes, tab visibility, theme recolouring, pointer
 * easing and the reduced-motion freeze. Colours come from the `--field-*`
 * custom properties so dark/light both resolve through the token layer.
 */
const FIELD_SEED = 20260214
const POINTER_EASE = 0.06
const FALLBACK_COLORS = {
  node: 'rgba(94, 234, 212, 0.78)',
  edge: 'rgba(94, 234, 212, 0.22)',
  accent: 'rgba(153, 246, 228, 0.95)',
}

const canvasRef = ref<HTMLCanvasElement | null>(null)

let context: CanvasRenderingContext2D | null = null
let field: MolecularField | null = null
let resizeObserver: ResizeObserver | null = null
let themeObserver: MutationObserver | null = null
let motionQuery: MediaQueryList | null = null
let hoverQuery: MediaQueryList | null = null
let frame = 0
let startedAt = 0
let running = false
let colors = { ...FALLBACK_COLORS }
const pointerTarget: PointerState = { x: 0, y: 0 }
const pointerCurrent: PointerState = { x: 0, y: 0 }

function readColors() {
  const canvas = canvasRef.value
  if (!canvas) return
  const styles = getComputedStyle(canvas)
  colors = {
    node: styles.getPropertyValue('--field-node').trim() || FALLBACK_COLORS.node,
    edge: styles.getPropertyValue('--field-edge').trim() || FALLBACK_COLORS.edge,
    accent: styles.getPropertyValue('--field-accent').trim() || FALLBACK_COLORS.accent,
  }
}

function prefersReducedMotion() {
  return motionQuery?.matches ?? false
}

function canTrackPointer() {
  return !(hoverQuery?.matches ?? false)
}

function resize() {
  const canvas = canvasRef.value
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  if (rect.width < 1 || rect.height < 1) return
  const ratio = Math.min(window.devicePixelRatio || 1, 2)
  canvas.width = Math.round(rect.width * ratio)
  canvas.height = Math.round(rect.height * ratio)
  context = canvas.getContext('2d')
  if (!context) return
  context.setTransform(ratio, 0, 0, ratio, 0, 0)
  readColors()
  field = createMolecularField({ seed: FIELD_SEED, width: rect.width, height: rect.height })
  if (prefersReducedMotion()) draw(0)
}

function draw(time: number) {
  const ctx = context
  if (!ctx || !field) return
  const sampled = sampleMolecularField(field, time, prefersReducedMotion() ? null : pointerCurrent)
  ctx.clearRect(0, 0, field.width, field.height)

  ctx.strokeStyle = colors.edge
  ctx.lineWidth = 1
  for (const edge of sampled.edges) {
    ctx.globalAlpha = edge.alpha
    ctx.beginPath()
    ctx.moveTo(edge.x1, edge.y1)
    ctx.lineTo(edge.x2, edge.y2)
    ctx.stroke()
  }

  ctx.fillStyle = colors.node
  for (const node of sampled.nodes) {
    if (node.accent) continue
    ctx.globalAlpha = node.alpha
    ctx.beginPath()
    ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2)
    ctx.fill()
  }

  for (const node of sampled.nodes) {
    if (!node.accent) continue
    const glow = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.radius * 7)
    glow.addColorStop(0, colors.accent)
    glow.addColorStop(1, 'rgba(94, 234, 212, 0)')
    ctx.globalAlpha = 0.5
    ctx.fillStyle = glow
    ctx.beginPath()
    ctx.arc(node.x, node.y, node.radius * 7, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = colors.accent
    ctx.globalAlpha = 0.95
    ctx.beginPath()
    ctx.arc(node.x, node.y, node.radius * 1.1, 0, Math.PI * 2)
    ctx.fill()
  }

  ctx.globalAlpha = 1
}

function loop(now: number) {
  if (!running) return
  if (prefersReducedMotion()) {
    running = false
    draw(0)
    return
  }
  pointerCurrent.x += (pointerTarget.x - pointerCurrent.x) * POINTER_EASE
  pointerCurrent.y += (pointerTarget.y - pointerCurrent.y) * POINTER_EASE
  draw((now - startedAt) / 1000)
  frame = requestAnimationFrame(loop)
}

function start() {
  if (running || prefersReducedMotion()) {
    draw(0)
    return
  }
  running = true
  frame = requestAnimationFrame(loop)
}

function stop() {
  running = false
  if (frame) cancelAnimationFrame(frame)
  frame = 0
}

function handlePointerMove(event: PointerEvent) {
  const canvas = canvasRef.value
  if (!canvas || !canTrackPointer()) return
  const rect = canvas.getBoundingClientRect()
  if (rect.width < 1 || rect.height < 1) return
  pointerTarget.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  pointerTarget.y = ((event.clientY - rect.top) / rect.height) * 2 - 1
}

function resetPointer() {
  pointerTarget.x = 0
  pointerTarget.y = 0
  if (prefersReducedMotion()) draw(0)
}

function handleVisibilityChange() {
  if (document.hidden) {
    stop()
    return
  }
  startedAt = performance.now()
  start()
}

function handleMotionChange() {
  stop()
  start()
}

onMounted(() => {
  const canvas = canvasRef.value
  if (!canvas) return
  motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
  hoverQuery = window.matchMedia('(hover: none)')
  motionQuery.addEventListener('change', handleMotionChange)

  resizeObserver = new ResizeObserver(() => {
    const reduced = prefersReducedMotion()
    resize()
    if (!reduced) start()
  })
  resizeObserver.observe(canvas)

  themeObserver = new MutationObserver(() => {
    readColors()
    if (prefersReducedMotion()) draw(0)
  })
  themeObserver.observe(document.documentElement, { attributeFilter: ['data-theme'] })

  window.addEventListener('pointermove', handlePointerMove, { passive: true })
  canvas.addEventListener('pointerleave', resetPointer)
  document.addEventListener('visibilitychange', handleVisibilityChange)

  startedAt = performance.now()
  resize()
  start()
})

onBeforeUnmount(() => {
  stop()
  resizeObserver?.disconnect()
  themeObserver?.disconnect()
  motionQuery?.removeEventListener('change', handleMotionChange)
  window.removeEventListener('pointermove', handlePointerMove)
  canvasRef.value?.removeEventListener('pointerleave', resetPointer)
  document.removeEventListener('visibilitychange', handleVisibilityChange)
  resizeObserver = null
  themeObserver = null
  motionQuery = null
  hoverQuery = null
  context = null
  field = null
})
</script>

<template>
  <canvas ref="canvasRef" class="landing-field" aria-hidden="true" focusable="false" />
</template>
