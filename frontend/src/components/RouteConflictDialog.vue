<script setup lang="ts">
import { AlertTriangle, X } from '@lucide/vue'
import { nextTick, ref, useId, watch } from 'vue'

const props = defineProps<{
  open: boolean
  title: string
  description: string
  confirmLabel: string
  cancelLabel: string
  closeLabel: string
}>()

const emit = defineEmits<{
  confirm: []
  cancel: []
}>()

const dialog = ref<HTMLDialogElement | null>(null)
const cancelButton = ref<HTMLButtonElement | null>(null)
const titleId = `route-conflict-title-${useId()}`
const descriptionId = `route-conflict-description-${useId()}`

watch(
  () => props.open,
  async (open) => {
    const element = dialog.value
    if (!element) return
    if (open && !element.open) {
      element.showModal()
      await nextTick()
      cancelButton.value?.focus()
    } else if (!open && element.open) {
      element.close()
    }
  },
  { immediate: true },
)

function handleBackdrop(event: MouseEvent) {
  if (event.target === dialog.value) emit('cancel')
}
</script>

<template>
  <dialog
    ref="dialog"
    class="route-conflict-dialog"
    :aria-labelledby="titleId"
    :aria-describedby="descriptionId"
    @cancel.prevent="emit('cancel')"
    @click="handleBackdrop"
  >
    <section class="route-conflict-panel">
      <header>
        <span class="route-conflict-icon"><AlertTriangle aria-hidden="true" /></span>
        <div>
          <h2 :id="titleId">{{ title }}</h2>
          <p :id="descriptionId">{{ description }}</p>
        </div>
        <button
          type="button"
          class="icon-action"
          :title="closeLabel"
          :aria-label="closeLabel"
          @click="emit('cancel')"
        >
          <X aria-hidden="true" />
        </button>
      </header>
      <footer>
        <button ref="cancelButton" type="button" class="secondary-action" @click="emit('cancel')">
          {{ cancelLabel }}
        </button>
        <button type="button" class="primary-action" @click="emit('confirm')">
          {{ confirmLabel }}
        </button>
      </footer>
    </section>
  </dialog>
</template>

<style scoped>
.route-conflict-dialog {
  width: min(31rem, calc(100vw - 2rem));
  max-width: none;
  border: 0;
  padding: 0;
  background: transparent;
  color: var(--color-ink);
}

.route-conflict-dialog::backdrop {
  background: rgb(20 18 16 / 48%);
  backdrop-filter: blur(2px);
}

.route-conflict-panel {
  overflow: hidden;
  border: 1px solid var(--color-line);
  border-radius: 0.5rem;
  background: var(--color-paper);
  box-shadow: 0 24px 70px rgb(20 18 16 / 24%);
}

.route-conflict-panel > header {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: start;
  gap: 0.85rem;
  padding: 1.25rem;
}

.route-conflict-panel h2 {
  font-size: 1rem;
  font-weight: 700;
}

.route-conflict-panel p {
  margin-top: 0.35rem;
  color: var(--color-muted);
  font-size: 0.8rem;
  line-height: 1.6;
}

.route-conflict-icon {
  display: grid;
  width: 2rem;
  height: 2rem;
  place-items: center;
  border-radius: 0.4rem;
  background: var(--color-warning-bg);
  color: var(--color-warning-text);
}

.route-conflict-icon svg {
  width: 1rem;
  height: 1rem;
}

.route-conflict-panel > footer {
  display: flex;
  justify-content: flex-end;
  gap: 0.65rem;
  border-top: 1px solid var(--line-soft);
  padding: 0.9rem 1.25rem;
  background: var(--surface-muted);
}

@media (prefers-reduced-motion: no-preference) {
  .route-conflict-dialog[open] .route-conflict-panel {
    animation: route-conflict-enter 160ms ease-out;
  }

  @keyframes route-conflict-enter {
    from {
      opacity: 0;
      transform: translateY(6px);
    }
  }
}
</style>
