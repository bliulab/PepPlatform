<script setup lang="ts">
import { onActivated, onBeforeUnmount, onDeactivated, onMounted, ref } from 'vue'

const props = defineProps<{ label: string; side: 'left' | 'right' }>()
const emit = defineEmits<{ close: [] }>()
const panel = ref<HTMLDialogElement | null>(null)
let viewport: MediaQueryList | undefined
let returnFocus: HTMLElement | null = null
let active = true

function present() {
  const dialog = panel.value
  if (!dialog || !active) return
  const focused = document.activeElement instanceof HTMLElement ? document.activeElement : null
  if (dialog.open) dialog.close()
  if (viewport?.matches) dialog.showModal()
  else dialog.show()
  if (focused && dialog.contains(focused)) focused.focus()
}
function handleBackdrop(event: MouseEvent) {
  const dialog = panel.value
  if (!dialog || event.target !== dialog) return
  const bounds = dialog.getBoundingClientRect()
  if (
    event.clientX < bounds.left ||
    event.clientX > bounds.right ||
    event.clientY < bounds.top ||
    event.clientY > bounds.bottom
  )
    emit('close')
}
function onKeydown(event: KeyboardEvent) {
  const dialog = panel.value
  if (event.key === 'Escape') {
    event.preventDefault()
    event.stopPropagation()
    emit('close')
    return
  }
  if (event.key !== 'Tab' || !dialog?.matches(':modal')) return
  const controls = Array.from(
    dialog.querySelectorAll<HTMLElement>(
      'button, a[href], input:not([type="hidden"]), select, textarea, summary, [tabindex]',
    ),
  ).filter(
    (element) =>
      element.tabIndex >= 0 &&
      !element.matches(':disabled') &&
      element.getClientRects().length > 0 &&
      getComputedStyle(element).visibility !== 'hidden',
  )
  const first = controls[0]
  const last = controls[controls.length - 1]
  if (!first || !last) {
    event.preventDefault()
    dialog.focus()
    return
  }
  const focused = document.activeElement
  if (event.shiftKey && (focused === first || !dialog.contains(focused))) {
    event.preventDefault()
    last.focus()
  } else if (!event.shiftKey && (focused === last || !dialog.contains(focused))) {
    event.preventDefault()
    first.focus()
  }
}

onMounted(() => {
  returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
  viewport = window.matchMedia('(max-width: 1099px)')
  viewport.addEventListener('change', present)
  present()
})
onActivated(() => {
  active = true
  present()
})
onDeactivated(() => {
  active = false
  panel.value?.close()
})
onBeforeUnmount(() => {
  active = false
  viewport?.removeEventListener('change', present)
  const focused = document.activeElement instanceof HTMLElement ? document.activeElement : null
  const shouldRestore = !focused || focused === document.body || panel.value?.contains(focused)
  panel.value?.close()
  const target = shouldRestore ? returnFocus : focused
  if (target?.isConnected) target.focus({ preventScroll: true })
})
</script>

<template>
  <dialog
    ref="panel"
    class="workspace-panel"
    :class="`workspace-panel--${props.side}`"
    :aria-label="label"
    @keydown="onKeydown"
    @cancel.prevent="emit('close')"
    @click="handleBackdrop"
  >
    <slot />
  </dialog>
</template>

<style scoped>
.workspace-panel {
  position: sticky;
  z-index: 4;
  top: 5.5rem;
  display: none;
  align-self: start;
  flex-direction: column;
  width: 100%;
  max-width: none;
  height: min(48rem, calc(100dvh - 7rem));
  max-height: none;
  min-width: 0;
  margin: 0;
  border: 0;
  border-radius: 0;
  background: var(--surface-strong);
  padding: 0;
  color: var(--color-ink);
  box-shadow: none;
}
.workspace-panel[open] {
  display: flex;
}
.workspace-panel--left {
  border-right: 1px solid var(--color-line);
  border-radius: 1rem 0 0 1rem;
}
.workspace-panel--right {
  border-left: 1px solid var(--color-line);
  border-radius: 0 1rem 1rem 0;
}
.workspace-panel::backdrop {
  background: var(--scrim);
}
@media (max-width: 1099px) {
  .workspace-panel {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    left: auto;
    z-index: 20;
    width: min(26rem, 92vw);
    height: 100dvh;
    max-height: 100dvh;
    border: 0;
    border-radius: 1rem 0 0 1rem;
    box-shadow: -15px 0 50px color-mix(in srgb, var(--color-ink) 18%, transparent);
  }
  .workspace-panel--left {
    right: auto;
    left: 0;
    border-radius: 0 1rem 1rem 0;
  }
}
</style>
