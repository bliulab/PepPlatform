import * as THREE from 'three'
import type { Sculpture } from './landingSculptures'
import {
  createPeptideParticles,
  stepPeptideParticles,
  type ParticleFeedback,
} from './peptideParticles'

export function createParticleSculpture(mobile: boolean): Sculpture {
  const cloud = createPeptideParticles(mobile ? 450 : 1800)
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(cloud.positions, 3).setUsage(THREE.DynamicDrawUsage),
  )
  geometry.setAttribute(
    'aGlow',
    new THREE.BufferAttribute(cloud.glow, 1).setUsage(THREE.DynamicDrawUsage),
  )
  geometry.setAttribute('aSize', new THREE.BufferAttribute(cloud.sizes, 1))
  const material = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: { uScale: { value: 1000 }, uOpacity: { value: 1 } },
    vertexShader: `
      attribute float aGlow;
      attribute float aSize;
      uniform float uScale;
      varying float vGlow;
      void main() {
        vec4 view = modelViewMatrix * vec4(position, 1.0);
        vGlow = aGlow;
        gl_Position = projectionMatrix * view;
        gl_PointSize = clamp(aSize * uScale / max(1.0, -view.z) * (1.0 + aGlow * 0.45), 1.5, 12.0);
      }`,
    fragmentShader: `
      uniform float uOpacity;
      varying float vGlow;
      void main() {
        float d = length(gl_PointCoord - vec2(0.5));
        if (d > 0.5) discard;
        float core = exp(-d * d * 36.0);
        float halo = smoothstep(0.5, 0.04, d);
        vec3 teal = vec3(0.16, 0.77, 0.64);
        vec3 light = vec3(0.71, 1.0, 0.93);
        vec3 color = mix(teal, light, min(1.0, vGlow * 0.75 + core * 0.35));
        gl_FragColor = vec4(color * (0.9 + vGlow * 0.6), halo * uOpacity * 0.85);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  })
  const points = new THREE.Points(geometry, material)
  points.frustumCulled = false
  const root = new THREE.Group()
  root.add(points)
  root.userData.particleCloud = cloud
  const idle: ParticleFeedback = {
    dt: 0,
    pointer: null,
    radius: 1.5,
    displacement: 0.5,
    wave: null,
    pointScale: 1000,
  }
  return {
    root,
    radius: 4.15,
    update(time, entrance, feedback = idle) {
      root.userData.deformation = stepPeptideParticles(cloud, time, entrance, feedback)
      geometry.getAttribute('position').needsUpdate = true
      geometry.getAttribute('aGlow').needsUpdate = true
      material.uniforms.uScale.value = feedback.pointScale
      material.uniforms.uOpacity.value = material.opacity
    },
    dispose() {
      geometry.dispose()
      material.dispose()
    },
  }
}
