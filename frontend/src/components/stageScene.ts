/**
 * Shared three.js plumbing for the two landing stages (hero helix + HLA-I groove).
 *
 * Both stages want exactly the same renderer configuration, the same procedural
 * room environment (no external HDR asset) and the same visibility gating, so
 * the wiring lives here and the scene components only describe their geometry.
 */

import * as THREE from 'three'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'

export interface StageQuality {
  pixelRatio: number
  bloom: boolean
  antialias: boolean
}

/** Tone mapping shared by every stage, so the two canvases read as one family. */
export const STAGE_TONE_MAPPING_EXPOSURE = 1.15

/** Environment intensity applied to `scene.environmentIntensity`. */
export const STAGE_ENVIRONMENT_INTENSITY = 0.5

/**
 * Colour management and tone mapping are renderer-wide, not per scene: set them
 * once at creation so a stage cannot drift from the other one.
 */
export function configureStageRenderer(renderer: THREE.WebGLRenderer, quality: StageQuality) {
  renderer.setPixelRatio(quality.pixelRatio)
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = STAGE_TONE_MAPPING_EXPOSURE
}

/**
 * Procedural image-based lighting. `RoomEnvironment` is a throwaway scene, so it
 * is disposed as soon as the prefiltered cube map exists; the returned texture is
 * owned by the caller and must be disposed with the rest of the stage.
 */
export function createStageEnvironment(renderer: THREE.WebGLRenderer): THREE.Texture {
  const environment = new RoomEnvironment()
  const pmrem = new THREE.PMREMGenerator(renderer)
  const texture = pmrem.fromScene(environment, 0.04).texture
  pmrem.dispose()
  environment.dispose()
  return texture
}

/**
 * Radial additive sprite used in place of CSS glows, which an opaque canvas
 * would otherwise cover. Colours arrive as CSS rgb() strings so they stay
 * readable next to the stage token table.
 */
export function glowTexture(inner: string, middle: string, outer: string): THREE.CanvasTexture {
  const size = 256
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const context = canvas.getContext('2d')
  if (context) {
    const gradient = context.createRadialGradient(
      size / 2,
      size / 2,
      0,
      size / 2,
      size / 2,
      size / 2,
    )
    gradient.addColorStop(0, inner)
    gradient.addColorStop(0.45, middle)
    gradient.addColorStop(1, outer)
    context.fillStyle = gradient
    context.fillRect(0, 0, size, size)
  }
  return new THREE.CanvasTexture(canvas)
}

export interface StageVisibilityHandlers {
  /** The stage should run: canvas on screen and the document visible. */
  onEnter: () => void
  /** The stage must stop: canvas off screen or the document hidden. */
  onLeave: () => void
}

/**
 * One gate for both "scrolled out of view" and "tab in the background". Returns
 * the cleanup that detaches every listener it added.
 *
 * Without `IntersectionObserver` the stage is treated as permanently visible,
 * which matches the pre-existing fallback in the hero.
 */
export function watchStageVisibility(
  canvas: HTMLElement,
  handlers: StageVisibilityHandlers,
): () => void {
  let intersecting = true

  function sync() {
    if (intersecting && !document.hidden) handlers.onEnter()
    else handlers.onLeave()
  }

  const handleVisibilityChange = () => sync()
  document.addEventListener('visibilitychange', handleVisibilityChange)

  if (typeof IntersectionObserver === 'undefined') {
    sync()
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange)
  }

  const observer = new IntersectionObserver(
    (entries) => {
      intersecting = entries.some((entry) => entry.isIntersecting)
      sync()
    },
    { threshold: 0.01 },
  )
  observer.observe(canvas)

  return () => {
    observer.disconnect()
    document.removeEventListener('visibilitychange', handleVisibilityChange)
  }
}
