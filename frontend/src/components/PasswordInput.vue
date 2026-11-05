<script setup lang="ts">
import { Eye, EyeOff } from '@lucide/vue'
import { nextTick, ref } from 'vue'
import { useI18n } from 'vue-i18n'

defineOptions({ inheritAttrs: false })
const model = defineModel<string>({ required: true })
const { t } = useI18n()
const visible = ref(false)
const input = ref<HTMLInputElement>()

async function toggle() {
  const start = input.value?.selectionStart
  const end = input.value?.selectionEnd
  visible.value = !visible.value
  await nextTick()
  input.value?.focus({ preventScroll: true })
  if (start != null && end != null) input.value?.setSelectionRange(start, end)
}
</script>

<template>
  <span class="password-control">
    <input ref="input" v-model="model" v-bind="$attrs" :type="visible ? 'text' : 'password'" />
    <button
      type="button"
      class="password-visibility"
      :aria-label="t(visible ? 'auth.hidePassword' : 'auth.showPassword')"
      :title="t(visible ? 'auth.hidePassword' : 'auth.showPassword')"
      :aria-pressed="visible"
      :disabled="$attrs.disabled !== undefined && $attrs.disabled !== false"
      @click="toggle"
    >
      <EyeOff v-if="visible" aria-hidden="true" />
      <Eye v-else aria-hidden="true" />
    </button>
  </span>
</template>

<style scoped>
.password-control {
  position: relative;
  display: block;
  min-width: 0;
}
.password-control input {
  width: 100%;
  padding-right: 3.5rem;
}
.password-visibility {
  position: absolute;
  top: 50%;
  right: 0.35rem;
  display: grid;
  width: 2.75rem;
  height: 2.75rem;
  place-items: center;
  transform: translateY(-50%);
  border: 0;
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--color-muted);
  cursor: pointer;
}
.password-visibility:hover {
  color: var(--color-accent);
  background: var(--surface-muted);
}
.password-visibility:focus-visible {
  outline: 2px solid var(--color-accent);
  outline-offset: -3px;
}
.password-visibility svg {
  width: 1.15rem;
  height: 1.15rem;
}
</style>
