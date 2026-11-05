<script setup lang="ts">
import { Check, ChevronDown } from '@lucide/vue'
import { computed, nextTick, onBeforeUnmount, ref, useId, watch } from 'vue'
import {
  boundaryEnabledOptionIndex,
  findTypeaheadOptionIndex,
  nextEnabledOptionIndex,
  resolveSelectMenuPlacement,
  selectedOptionIndex,
  type SelectOption,
  type SelectValue,
} from './selectControl'

defineOptions({ inheritAttrs: false })

const props = withDefaults(
  defineProps<{
    modelValue: SelectValue
    label: string
    options: SelectOption[]
    disabled?: boolean
    multiline?: boolean
    placeholder?: string
    id?: string
  }>(),
  {
    disabled: false,
    multiline: false,
    placeholder: '',
    id: '',
  },
)

const emit = defineEmits<{
  'update:modelValue': [value: SelectValue]
}>()

const generatedId = useId()
const controlId = computed(() => props.id || `select-control-${generatedId}`)
const labelId = computed(() => `${controlId.value}-label`)
const valueId = computed(() => `${controlId.value}-value`)
const listboxId = computed(() => `${controlId.value}-listbox`)
const root = ref<HTMLElement>()
const trigger = ref<HTMLButtonElement>()
const menu = ref<HTMLElement>()
const optionList = ref<HTMLElement>()
const isOpen = ref(false)
const activeIndex = ref(-1)
const placement = ref<'top' | 'bottom'>('bottom')
const menuTop = ref(0)
const menuLeft = ref(0)
const menuWidth = ref(0)
const menuMaxHeight = ref(288)
let typeaheadQuery = ''
let typeaheadTimer: number | undefined

const currentIndex = computed(() => selectedOptionIndex(props.options, props.modelValue))
const currentOption = computed(() => props.options[currentIndex.value])
const displayLabel = computed(() => {
  if (currentOption.value) return currentOption.value.label
  if (props.modelValue !== '') return String(props.modelValue)
  return props.placeholder
})
const isPlaceholder = computed(() => !currentOption.value && props.modelValue === '')
const activeOptionId = computed(() =>
  isOpen.value && activeIndex.value >= 0
    ? `${controlId.value}-option-${activeIndex.value}`
    : undefined,
)
const menuStyle = computed(() => ({
  top: `${menuTop.value}px`,
  left: `${menuLeft.value}px`,
  width: `${menuWidth.value}px`,
  '--select-menu-max-height': `${Math.max(44, menuMaxHeight.value - 8)}px`,
}))

function optionId(index: number) {
  return `${controlId.value}-option-${index}`
}

function syncActiveIndex(fallback: 'first' | 'last' = 'first') {
  const selected = currentIndex.value
  activeIndex.value =
    selected >= 0 && !props.options[selected]?.disabled
      ? selected
      : boundaryEnabledOptionIndex(props.options, fallback)
}

function setInitialPosition() {
  const rect = trigger.value?.getBoundingClientRect()
  if (!rect) return
  menuTop.value = rect.bottom + 6
  menuLeft.value = rect.left
  menuWidth.value = rect.width
  menuMaxHeight.value = 288
  placement.value = 'bottom'
}

function updateMenuPosition() {
  if (!isOpen.value || !trigger.value) return
  const rect = trigger.value.getBoundingClientRect()
  const measuredHeight = Math.min((optionList.value?.scrollHeight ?? 280) + 8, 288)
  const resolved = resolveSelectMenuPlacement({
    trigger: rect,
    menuHeight: measuredHeight,
    viewportWidth: window.innerWidth,
    viewportHeight: window.innerHeight,
  })
  placement.value = resolved.placement
  menuTop.value = resolved.top
  menuLeft.value = resolved.left
  menuWidth.value = resolved.width
  menuMaxHeight.value = resolved.maxHeight
}

function scrollActiveOptionIntoView() {
  if (activeIndex.value < 0) return
  const activeOption = optionList.value?.querySelector<HTMLElement>(
    `[data-option-index="${activeIndex.value}"]`,
  )
  activeOption?.scrollIntoView({ block: 'nearest' })
}

function addOpenListeners() {
  document.addEventListener('pointerdown', handleDocumentPointerDown, true)
  window.addEventListener('resize', updateMenuPosition)
  window.addEventListener('scroll', updateMenuPosition, true)
}

function removeOpenListeners() {
  document.removeEventListener('pointerdown', handleDocumentPointerDown, true)
  window.removeEventListener('resize', updateMenuPosition)
  window.removeEventListener('scroll', updateMenuPosition, true)
}

function openMenu(fallback: 'first' | 'last' = 'first') {
  if (props.disabled || !props.options.some((option) => !option.disabled) || isOpen.value) return
  syncActiveIndex(fallback)
  setInitialPosition()
  isOpen.value = true
  addOpenListeners()
  void nextTick(() => {
    updateMenuPosition()
    scrollActiveOptionIntoView()
  })
}

function closeMenu() {
  if (!isOpen.value) return
  isOpen.value = false
  removeOpenListeners()
  typeaheadQuery = ''
  if (typeaheadTimer !== undefined) {
    window.clearTimeout(typeaheadTimer)
    typeaheadTimer = undefined
  }
}

function toggleMenu() {
  if (isOpen.value) closeMenu()
  else openMenu()
}

function moveActiveOption(direction: 1 | -1) {
  if (!isOpen.value) {
    openMenu(direction === 1 ? 'first' : 'last')
    return
  }
  const nextIndex = nextEnabledOptionIndex(props.options, activeIndex.value, direction)
  if (nextIndex < 0) return
  activeIndex.value = nextIndex
  void nextTick(scrollActiveOptionIntoView)
}

function moveToBoundary(edge: 'first' | 'last') {
  const nextIndex = boundaryEnabledOptionIndex(props.options, edge)
  if (nextIndex < 0) return
  activeIndex.value = nextIndex
  void nextTick(scrollActiveOptionIntoView)
}

function commitOption(index: number) {
  const option = props.options[index]
  if (!option || option.disabled) return
  if (!Object.is(option.value, props.modelValue)) emit('update:modelValue', option.value)
  closeMenu()
}

function handleTypeahead(character: string) {
  typeaheadQuery += character.toLocaleLowerCase()
  if (typeaheadTimer !== undefined) window.clearTimeout(typeaheadTimer)
  typeaheadTimer = window.setTimeout(() => {
    typeaheadQuery = ''
    typeaheadTimer = undefined
  }, 600)
  if (!isOpen.value) openMenu()
  const match = findTypeaheadOptionIndex(props.options, typeaheadQuery, activeIndex.value)
  if (match >= 0) {
    activeIndex.value = match
    void nextTick(scrollActiveOptionIntoView)
  }
}

function handleTriggerKeydown(event: KeyboardEvent) {
  if (props.disabled || event.isComposing) return
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault()
    moveActiveOption(event.key === 'ArrowDown' ? 1 : -1)
    return
  }
  if (event.key === 'Home' || event.key === 'End') {
    if (!isOpen.value) return
    event.preventDefault()
    moveToBoundary(event.key === 'Home' ? 'first' : 'last')
    return
  }
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault()
    if (isOpen.value) commitOption(activeIndex.value)
    else openMenu()
    return
  }
  if (event.key === 'Escape' && isOpen.value) {
    event.preventDefault()
    closeMenu()
    return
  }
  if (event.key === 'Tab') {
    closeMenu()
    return
  }
  if (event.key.length === 1 && !event.altKey && !event.ctrlKey && !event.metaKey) {
    event.preventDefault()
    handleTypeahead(event.key)
  }
}

function handleDocumentPointerDown(event: PointerEvent) {
  const target = event.target
  if (!(target instanceof Node)) return
  if (root.value?.contains(target) || menu.value?.contains(target)) return
  closeMenu()
}

watch(
  () => props.options,
  () => {
    if (!isOpen.value) return
    syncActiveIndex()
    void nextTick(updateMenuPosition)
  },
  { deep: true },
)

watch(
  () => props.disabled,
  (disabled) => {
    if (disabled) closeMenu()
  },
)

onBeforeUnmount(() => {
  removeOpenListeners()
  if (typeaheadTimer !== undefined) window.clearTimeout(typeaheadTimer)
})
</script>

<template>
  <div
    ref="root"
    class="select-control"
    :class="{ 'select-control-disabled': disabled, 'select-control-multiline': multiline }"
    v-bind="$attrs"
  >
    <span :id="labelId" class="select-control-label">{{ label }}</span>
    <button
      :id="controlId"
      ref="trigger"
      type="button"
      class="select-control-trigger"
      :class="{ 'select-control-trigger-open': isOpen }"
      role="combobox"
      aria-autocomplete="none"
      aria-haspopup="listbox"
      :aria-controls="listboxId"
      :aria-expanded="isOpen"
      :aria-activedescendant="activeOptionId"
      :aria-labelledby="`${labelId} ${valueId}`"
      :disabled="disabled"
      @click="toggleMenu"
      @keydown="handleTriggerKeydown"
      @blur="closeMenu"
    >
      <span
        :id="valueId"
        class="select-control-value"
        :class="{ 'select-control-placeholder': isPlaceholder }"
      >
        {{ displayLabel }}
      </span>
      <span class="select-control-indicator" aria-hidden="true">
        <ChevronDown />
      </span>
    </button>
  </div>

  <Teleport to="body">
    <Transition name="select-menu">
      <div
        v-if="isOpen"
        :id="listboxId"
        ref="menu"
        class="select-control-menu"
        :data-placement="placement"
        :style="menuStyle"
        role="listbox"
        :aria-labelledby="labelId"
      >
        <div ref="optionList" class="select-control-options">
          <template v-for="(option, index) in options" :key="`${String(option.value)}-${index}`">
            <div
              v-if="option.group && options[index - 1]?.group !== option.group"
              class="select-control-group-label"
              role="presentation"
            >
              {{ option.group }}
            </div>
            <div
              :id="optionId(index)"
              class="select-control-option"
              :class="{
                'select-control-option-active': index === activeIndex,
                'select-control-option-selected': Object.is(option.value, modelValue),
                'select-control-option-disabled': option.disabled,
              }"
              role="option"
              :aria-label="option.group ? `${option.group}: ${option.label}` : option.label"
              :aria-selected="Object.is(option.value, modelValue)"
              :aria-disabled="option.disabled || undefined"
              :data-option-index="index"
              @pointerenter="option.disabled ? undefined : (activeIndex = index)"
              @pointerdown.prevent
              @click="commitOption(index)"
            >
              <span class="select-control-check" aria-hidden="true">
                <Check v-if="Object.is(option.value, modelValue)" />
              </span>
              <span class="select-control-option-label">{{ option.label }}</span>
            </div>
          </template>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.select-control {
  display: grid;
  min-width: 0;
  gap: 0.5rem;
  color: var(--color-ink);
  font-size: 0.875rem;
}

.select-control-label {
  min-width: 0;
  line-height: 1.4;
}

.select-control-trigger {
  position: relative;
  display: grid;
  width: 100%;
  min-width: 0;
  min-height: 2.5rem;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 0.75rem;
  overflow: hidden;
  border: 1px solid var(--color-line);
  border-radius: var(--radius-control);
  background: var(--surface-strong);
  padding: 0.5rem 0.6rem 0.5rem 0.85rem;
  color: var(--color-ink);
  font-size: 0.875rem;
  line-height: 1.4;
  text-align: left;
}

.select-control-trigger:hover:not(:disabled) {
  border-color: color-mix(in srgb, var(--color-accent) 40%, var(--color-line));
}

.select-control-trigger:focus-visible,
.select-control-trigger-open {
  outline: 0;
  border-color: var(--color-accent);
  box-shadow: 0 0 0 3px var(--focus-ring);
}

.select-control-trigger:disabled {
  cursor: not-allowed;
  background: var(--surface-muted);
  opacity: 0.68;
}

.select-control-value {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.select-control-placeholder {
  color: var(--color-muted);
}

.select-control-multiline .select-control-trigger {
  height: auto;
}

.select-control-multiline .select-control-value {
  overflow: visible;
  overflow-wrap: anywhere;
  white-space: normal;
}

.select-control-indicator {
  display: grid;
  width: 1.5rem;
  height: 1.5rem;
  place-items: center;
  color: var(--color-muted);
}

.select-control-indicator svg {
  width: 0.9rem;
  height: 0.9rem;
  stroke-width: 2;
  transition: transform 140ms ease;
}

.select-control-trigger-open .select-control-indicator svg {
  transform: rotate(180deg);
}

.select-control-menu {
  position: fixed;
  z-index: 1200;
  overflow: hidden;
  border: 1px solid var(--color-line);
  border-radius: var(--radius-control);
  background: var(--surface-strong);
  color: var(--color-ink);
  box-shadow: var(--shadow-lift);
}

.select-control-options {
  position: relative;
  z-index: 2;
  max-height: var(--select-menu-max-height);
  overflow-y: auto;
  overscroll-behavior: contain;
  padding: 0.4rem;
  scrollbar-color: var(--color-line) transparent;
  scrollbar-width: thin;
}

.select-control-option {
  display: grid;
  min-height: 2.35rem;
  grid-template-columns: 1.15rem minmax(0, 1fr);
  align-items: center;
  gap: 0.55rem;
  border-radius: 0.4rem;
  padding: 0.45rem 0.6rem;
  color: var(--color-ink);
  cursor: pointer;
  font-size: 0.875rem;
  line-height: 1.4;
}

.select-control-group-label {
  margin: 0.45rem 0.7rem 0.2rem;
  color: var(--color-muted);
  font-size: 0.6875rem;
  font-weight: 600;
  line-height: 1.35;
  text-transform: uppercase;
}

.select-control-group-label:first-child {
  margin-top: 0.2rem;
}

.select-control-option-active {
  background: color-mix(in srgb, var(--color-accent) 7%, var(--surface-strong));
}

.select-control-option-selected {
  background: var(--state-selected-bg);
  color: var(--state-selected-text);
  font-weight: 600;
}

.select-control-option-selected.select-control-option-active {
  background: var(--state-selected-bg);
}

.select-control-option-disabled {
  color: var(--color-muted);
  cursor: not-allowed;
  opacity: 0.52;
}

.select-control-check {
  display: grid;
  width: 1.15rem;
  height: 1.15rem;
  place-items: center;
  color: var(--color-accent-dark);
}

.select-control-check svg {
  width: 0.9rem;
  height: 0.9rem;
  stroke-width: 2.4;
}

.select-control-option-label {
  min-width: 0;
  overflow-wrap: anywhere;
}

.select-menu-enter-active,
.select-menu-leave-active {
  transition:
    opacity 140ms ease,
    transform 140ms ease;
  transform-origin: top center;
}

.select-control-menu[data-placement='top'].select-menu-enter-active,
.select-control-menu[data-placement='top'].select-menu-leave-active {
  transform-origin: bottom center;
}

.select-menu-enter-from,
.select-menu-leave-to {
  opacity: 0;
  transform: translateY(-4px) scale(0.985);
}

.select-control-menu[data-placement='top'].select-menu-enter-from,
.select-control-menu[data-placement='top'].select-menu-leave-to {
  transform: translateY(4px) scale(0.985);
}

@media (max-width: 640px) {
  .select-control-menu {
    box-shadow: 0 14px 32px color-mix(in srgb, var(--color-ink) 16%, transparent);
  }
}

@media (prefers-reduced-motion: reduce) {
  .select-control-trigger::before,
  .select-control-indicator svg,
  .select-menu-enter-active,
  .select-menu-leave-active {
    transition: none;
  }
}
</style>
