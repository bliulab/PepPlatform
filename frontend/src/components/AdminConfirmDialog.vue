<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { AlertTriangle, X } from '@lucide/vue'
import { confirmationMatches } from '../admin'

const props = withDefaults(
  defineProps<{
    open: boolean
    title: string
    description: string
    confirmLabel: string
    cancelLabel: string
    requiredText?: string
    requiredLabel?: string
    danger?: boolean
  }>(),
  {
    requiredText: '',
    requiredLabel: '',
    danger: false,
  },
)

const emit = defineEmits<{
  confirm: []
  cancel: []
}>()

const dialog = ref<HTMLDialogElement | null>(null)
const confirmation = ref('')
const canConfirm = computed(
  () => !props.requiredText || confirmationMatches(confirmation.value, props.requiredText),
)

watch(
  () => props.open,
  async (open) => {
    const element = dialog.value
    if (!element) return
    if (open && !element.open) {
      confirmation.value = ''
      await nextTick()
      element.showModal()
    } else if (!open && element.open) {
      element.close()
    }
  },
  { immediate: true },
)

function cancel() {
  emit('cancel')
}

function confirm() {
  if (canConfirm.value) emit('confirm')
}

function handleBackdrop(event: MouseEvent) {
  if (event.target === dialog.value) cancel()
}
</script>

<template>
  <dialog
    ref="dialog"
    class="admin-confirm-dialog"
    @cancel.prevent="cancel"
    @click="handleBackdrop"
  >
    <section class="admin-confirm-dialog-panel" :class="{ 'admin-confirm-dialog-danger': danger }">
      <header>
        <span class="admin-confirm-dialog-icon"><AlertTriangle aria-hidden="true" /></span>
        <div>
          <h2>{{ title }}</h2>
          <p>{{ description }}</p>
        </div>
        <button
          type="button"
          class="icon-action"
          :title="cancelLabel"
          :aria-label="cancelLabel"
          @click="cancel"
        >
          <X aria-hidden="true" />
        </button>
      </header>

      <label v-if="requiredText" class="admin-confirm-input">
        <span>{{ requiredLabel }}</span>
        <strong>{{ requiredText }}</strong>
        <input v-model="confirmation" class="field-control" autocomplete="off" />
      </label>

      <footer>
        <button type="button" class="secondary-action" @click="cancel">{{ cancelLabel }}</button>
        <button
          type="button"
          :class="danger ? 'danger-action' : 'primary-action'"
          :disabled="!canConfirm"
          @click="confirm"
        >
          {{ confirmLabel }}
        </button>
      </footer>
    </section>
  </dialog>
</template>

<style scoped>
.admin-confirm-dialog {
  width: min(31rem, calc(100vw - 2rem));
  max-width: none;
  border: 0;
  padding: 0;
  background: transparent;
  color: var(--color-ink);
}

.admin-confirm-dialog::backdrop {
  background: var(--scrim-strong);
  backdrop-filter: blur(2px);
}

.admin-confirm-dialog-panel {
  overflow: hidden;
  border: 1px solid var(--color-line);
  border-radius: 0.5rem;
  background: var(--color-paper);
  box-shadow: var(--shadow-lift);
}

.admin-confirm-dialog-panel > header {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: start;
  gap: 0.85rem;
  padding: 1.25rem;
}

.admin-confirm-dialog-panel h2 {
  font-size: 1rem;
  font-weight: 700;
}
.admin-confirm-dialog-panel p {
  margin-top: 0.35rem;
  color: var(--color-muted);
  font-size: 0.8rem;
  line-height: 1.6;
}
.admin-confirm-dialog-icon {
  display: grid;
  width: 2rem;
  height: 2rem;
  place-items: center;
  border-radius: 0.4rem;
  background: var(--color-warning-bg);
  color: var(--color-warning-text);
}
.admin-confirm-dialog-icon svg {
  width: 1rem;
  height: 1rem;
}
.admin-confirm-dialog-danger .admin-confirm-dialog-icon {
  background: var(--color-danger-bg);
  color: var(--color-danger);
}

.admin-confirm-input {
  display: grid;
  gap: 0.45rem;
  border-top: 1px solid var(--line-soft);
  padding: 1rem 1.25rem;
  font-size: 0.75rem;
}
.admin-confirm-input span {
  color: var(--color-muted);
}
.admin-confirm-input strong {
  overflow-wrap: anywhere;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}

.admin-confirm-dialog-panel > footer {
  display: flex;
  justify-content: flex-end;
  gap: 0.65rem;
  border-top: 1px solid var(--line-soft);
  padding: 0.9rem 1.25rem;
  background: var(--surface-muted);
}

@media (prefers-reduced-motion: no-preference) {
  .admin-confirm-dialog[open] .admin-confirm-dialog-panel {
    animation: admin-dialog-enter 160ms ease-out;
  }
  @keyframes admin-dialog-enter {
    from {
      opacity: 0;
      transform: translateY(6px);
    }
  }
}
</style>
