<script setup lang="ts">
import {
  ChevronDown,
  Eye,
  EyeOff,
  Plus,
  PlugZap,
  RefreshCw,
  Save,
  TestTube,
  Trash2,
  X,
} from '@lucide/vue'
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { api } from '../api/client'
import { localizedApiError } from '../errorLocalization'
import type { AiProviderConfig, AiProviderPayload, AiProviderPreset } from '../types'
import SelectControl from './SelectControl.vue'
import type { SelectOption, SelectValue } from './selectControl'

const { t, te } = useI18n()

const providers = ref<AiProviderConfig[]>([])
const loading = ref(false)
const saving = ref(false)
const testingId = ref('')
const deletingId = ref('')
const editingId = ref('')
const message = ref('')
const errorMessage = ref('')
const presets = ref<AiProviderPreset[]>([])
const applyingExistingProvider = ref(false)
const apiKeyVisible = ref(false)
const advancedOpen = ref(false)
const deleteCandidate = ref<AiProviderConfig | null>(null)
const deleteDialog = ref<HTMLDialogElement | null>(null)
const deleteCancelButton = ref<HTMLButtonElement | null>(null)

function defaultsForProvider(provider: string): AiProviderPreset | undefined {
  return presets.value.find((preset) => preset.provider === provider) ?? presets.value[0]
}

function blankForm(): AiProviderPayload {
  const defaults = defaultsForProvider('deepseek')
  return {
    provider: defaults?.provider ?? '',
    display_name: defaults?.display_name ?? '',
    base_url: defaults?.base_url ?? '',
    model: defaults?.model ?? '',
    api_key: '',
    capabilities: { ...(defaults?.capabilities ?? {}) },
    enabled: true,
  }
}

const form = ref<AiProviderPayload>(blankForm())
const isEditing = computed(() => Boolean(editingId.value))
const editingProvider = computed(() =>
  providers.value.find((provider) => provider.id === editingId.value),
)
const selectedPreset = computed(() => defaultsForProvider(form.value.provider || ''))
const enabledCount = computed(() => providers.value.filter((provider) => provider.enabled).length)
const providerOptions = computed<SelectOption[]>(() =>
  presets.value.map((preset) => ({
    value: preset.provider,
    label: preset.display_name,
  })),
)
const reasoningOptions = computed<SelectOption[]>(() => [
  { value: 'none', label: t('aiServices.reasoningNone') },
  { value: 'thinking_disabled', label: t('aiServices.reasoningThinkingDisabled') },
])
const canSave = computed(() =>
  Boolean(
    form.value.provider?.trim() &&
    form.value.display_name.trim() &&
    form.value.base_url.trim() &&
    form.value.model.trim() &&
    (isEditing.value || form.value.api_key?.trim()),
  ),
)

function errorText(error: unknown) {
  return localizedApiError(error, t, te, 'aiServices.genericError')
}

function clearFeedback() {
  message.value = ''
  errorMessage.value = ''
}

function resetEditor() {
  editingId.value = ''
  form.value = blankForm()
  apiKeyVisible.value = false
  advancedOpen.value = false
}

function startAdding() {
  clearFeedback()
  resetEditor()
}

function editProvider(provider: AiProviderConfig, clear = true) {
  if (clear) clearFeedback()
  editingId.value = provider.id
  applyingExistingProvider.value = true
  apiKeyVisible.value = false
  advancedOpen.value = false
  form.value = {
    provider: provider.provider,
    display_name: provider.display_name,
    base_url: provider.base_url,
    model: provider.model,
    api_key: '',
    capabilities: {
      ...(defaultsForProvider(provider.provider)?.capabilities ?? {}),
      ...(provider.capabilities ?? {}),
    },
    enabled: provider.enabled,
  }
  void nextTick(() => {
    applyingExistingProvider.value = false
  })
}

function restoreEditor() {
  if (editingProvider.value) editProvider(editingProvider.value)
  else startAdding()
}

function providerLabel(provider: AiProviderConfig) {
  return (
    presets.value.find((preset) => preset.provider === provider.provider)?.display_name ??
    provider.provider
  )
}

function providerMark(provider: AiProviderConfig) {
  return (providerLabel(provider).trim().charAt(0) || 'AI').toLocaleUpperCase()
}

function hostLabel(value: string) {
  try {
    return new URL(value).host
  } catch {
    return value
  }
}

function updateProvider(value: SelectValue) {
  form.value.provider = String(value)
}

function updateReasoningControl(value: SelectValue) {
  form.value.capabilities.reasoning_control = String(value)
}

function syncAdvancedOpen(event: Event) {
  advancedOpen.value = (event.currentTarget as HTMLDetailsElement).open
}

async function loadPresets() {
  try {
    presets.value = await api.aiProviderPresets()
    if (!form.value.provider && presets.value.length) form.value = blankForm()
  } catch (error) {
    errorMessage.value = errorText(error)
  }
}

async function loadProviders() {
  loading.value = true
  try {
    providers.value = await api.aiProviders()
    if (editingId.value && !providers.value.some((provider) => provider.id === editingId.value)) {
      resetEditor()
    }
  } catch (error) {
    errorMessage.value = errorText(error)
  } finally {
    loading.value = false
  }
}

async function refreshProviders() {
  clearFeedback()
  await loadProviders()
}

async function saveProvider() {
  clearFeedback()
  if (!canSave.value) return
  saving.value = true
  try {
    const payload: AiProviderPayload = {
      ...form.value,
      display_name: form.value.display_name.trim(),
      base_url: form.value.base_url.trim(),
      model: form.value.model.trim(),
      api_key: form.value.api_key?.trim(),
      capabilities: form.value.capabilities,
    }
    let saved: AiProviderConfig
    let successKey: string
    if (editingId.value) {
      if (!payload.api_key) delete payload.api_key
      saved = await api.updateAiProvider(editingId.value, payload)
      successKey = 'aiServices.updated'
    } else {
      saved = await api.createAiProvider(payload)
      successKey = 'aiServices.created'
    }
    await loadProviders()
    const refreshed = providers.value.find((provider) => provider.id === saved.id) ?? saved
    editProvider(refreshed, false)
    message.value = t(successKey)
  } catch (error) {
    errorMessage.value = errorText(error)
  } finally {
    saving.value = false
  }
}

async function testProvider(provider: AiProviderConfig) {
  clearFeedback()
  testingId.value = provider.id
  try {
    const response = await api.testAiProvider(provider.id)
    message.value = t('aiServices.tested', { model: response.model })
  } catch (error) {
    errorMessage.value = errorText(error)
  } finally {
    testingId.value = ''
  }
}

async function requestDelete(provider: AiProviderConfig) {
  clearFeedback()
  deleteCandidate.value = provider
  await nextTick()
  if (!deleteDialog.value?.open) deleteDialog.value?.showModal()
  deleteCancelButton.value?.focus()
}

function dismissDeleteDialog(force = false) {
  if (deletingId.value && !force) return
  if (deleteDialog.value?.open) deleteDialog.value.close()
  deleteCandidate.value = null
}

function handleDeleteBackdrop(event: MouseEvent) {
  if (event.target === deleteDialog.value) dismissDeleteDialog()
}

async function confirmDelete() {
  const provider = deleteCandidate.value
  if (!provider) return
  deletingId.value = provider.id
  try {
    await api.deleteAiProvider(provider.id)
    if (editingId.value === provider.id) resetEditor()
    await loadProviders()
    message.value = t('aiServices.deleted')
    dismissDeleteDialog(true)
  } catch (error) {
    errorMessage.value = errorText(error)
  } finally {
    deletingId.value = ''
  }
}

onMounted(async () => {
  await Promise.all([loadPresets(), loadProviders()])
})

watch(
  () => form.value.provider,
  (provider, previous) => {
    if (!provider || provider === previous) return
    if (applyingExistingProvider.value) {
      applyingExistingProvider.value = false
      return
    }
    const defaults = defaultsForProvider(provider)
    if (!defaults) return
    form.value.display_name = defaults.display_name
    form.value.base_url = defaults.base_url
    form.value.model = defaults.model
    form.value.capabilities = { ...defaults.capabilities }
  },
)
</script>

<template>
  <section class="ai-service-workspace" :aria-label="t('aiServices.section')">
    <aside class="ai-service-sidebar">
      <header class="ai-service-sidebar-head">
        <div>
          <p>{{ t('aiServices.connectedEyebrow') }}</p>
          <h2>{{ t('aiServices.connected') }}</h2>
          <span>{{
            t('aiServices.connectedSummary', { count: providers.length, enabled: enabledCount })
          }}</span>
        </div>
        <button
          type="button"
          class="ai-service-icon-button"
          :disabled="loading"
          :title="t('aiServices.refresh')"
          :aria-label="t('aiServices.refresh')"
          @click="refreshProviders"
        >
          <RefreshCw :class="{ 'ai-service-spin': loading }" aria-hidden="true" />
        </button>
      </header>

      <button
        type="button"
        class="ai-service-add"
        :class="{ 'ai-service-add-active': !isEditing }"
        :aria-pressed="!isEditing"
        @click="startAdding"
      >
        <Plus aria-hidden="true" />
        <span>{{ t('aiServices.connect') }}</span>
      </button>

      <div v-if="loading && !providers.length" class="ai-service-skeletons" aria-hidden="true">
        <span v-for="index in 3" :key="index" />
      </div>
      <div v-else-if="providers.length" class="ai-service-list" role="list">
        <button
          v-for="provider in providers"
          :key="provider.id"
          type="button"
          class="ai-service-list-item"
          :class="{ 'ai-service-list-item-active': editingId === provider.id }"
          :aria-pressed="editingId === provider.id"
          role="listitem"
          @click="editProvider(provider)"
        >
          <span class="ai-service-provider-mark" aria-hidden="true">{{
            providerMark(provider)
          }}</span>
          <span class="ai-service-list-copy">
            <strong>{{ provider.display_name }}</strong>
            <span>{{ providerLabel(provider) }} · {{ provider.model }}</span>
            <small
              >{{ hostLabel(provider.base_url) }} ·
              {{ t('aiServices.keyEnding', { suffix: provider.api_key_last4 || '----' }) }}</small
            >
          </span>
          <span
            class="ai-service-status-dot"
            :class="
              provider.enabled ? 'ai-service-status-dot-enabled' : 'ai-service-status-dot-disabled'
            "
            :title="provider.enabled ? t('aiServices.enabled') : t('aiServices.disabled')"
          />
        </button>
      </div>
      <div v-else class="ai-service-empty">
        <PlugZap aria-hidden="true" />
        <strong>{{ t('aiServices.emptyTitle') }}</strong>
        <span>{{ t('aiServices.emptyHint') }}</span>
      </div>
    </aside>

    <form class="ai-service-editor" :aria-busy="saving" @submit.prevent="saveProvider">
      <header class="ai-service-editor-head">
        <div class="ai-service-editor-title">
          <span class="ai-service-editor-mark" aria-hidden="true">
            <span v-if="editingProvider">{{ providerMark(editingProvider) }}</span>
            <PlugZap v-else />
          </span>
          <div>
            <p>{{ isEditing ? t('aiServices.editEyebrow') : t('aiServices.addEyebrow') }}</p>
            <h2>{{ isEditing ? form.display_name : t('aiServices.addTitle') }}</h2>
            <span>
              {{
                isEditing
                  ? `${selectedPreset?.display_name || form.provider} · ${form.model}`
                  : t('aiServices.addSubtitle')
              }}
            </span>
          </div>
        </div>
        <div v-if="editingProvider" class="ai-service-editor-actions">
          <button
            type="button"
            class="secondary-action compact-action"
            :disabled="testingId === editingProvider.id || saving"
            @click="testProvider(editingProvider)"
          >
            <TestTube aria-hidden="true" />
            {{
              testingId === editingProvider.id
                ? t('aiServices.testing')
                : t('aiServices.testConnection')
            }}
          </button>
          <button
            type="button"
            class="danger-action compact-action"
            :disabled="saving"
            @click="requestDelete(editingProvider)"
          >
            <Trash2 aria-hidden="true" />
            {{ t('common.delete') }}
          </button>
        </div>
      </header>

      <div v-if="message || errorMessage" class="ai-service-feedback" aria-live="polite">
        <p v-if="message" class="success-card" role="status">{{ message }}</p>
        <p v-if="errorMessage" class="error-card" role="alert">{{ errorMessage }}</p>
      </div>

      <div class="ai-service-editor-body">
        <section class="ai-service-form-section" :aria-labelledby="'ai-service-connection-title'">
          <header>
            <div>
              <h3 id="ai-service-connection-title">{{ t('aiServices.connectionTitle') }}</h3>
              <p>{{ t('aiServices.connectionHint') }}</p>
            </div>
            <span
              v-if="editingProvider"
              class="status-pill"
              :class="form.enabled ? 'status-pill-success' : 'status-pill-neutral'"
            >
              {{ form.enabled ? t('aiServices.enabled') : t('aiServices.disabled') }}
            </span>
          </header>

          <div class="ai-service-form-grid">
            <SelectControl
              :model-value="form.provider || ''"
              :label="t('aiServices.provider')"
              :options="providerOptions"
              :disabled="saving || !providerOptions.length"
              :placeholder="t('aiServices.loadingProviders')"
              @update:model-value="updateProvider"
            />

            <label class="ai-service-field">
              <span>{{ t('aiServices.configName') }}</span>
              <input
                v-model="form.display_name"
                class="ai-service-input"
                type="text"
                maxlength="80"
                required
              />
            </label>

            <label class="ai-service-field ai-service-field-wide">
              <span>{{ t('aiServices.baseUrl') }}</span>
              <input
                v-model="form.base_url"
                class="ai-service-input ai-service-monospace"
                type="url"
                required
              />
            </label>

            <label class="ai-service-field">
              <span>{{ t('aiServices.modelId') }}</span>
              <input
                v-model="form.model"
                class="ai-service-input ai-service-monospace"
                type="text"
                required
              />
            </label>

            <label class="ai-service-field">
              <span>{{ t('aiServices.apiKey') }}</span>
              <span class="ai-service-secret-control">
                <input
                  v-model="form.api_key"
                  class="ai-service-input ai-service-monospace"
                  :type="apiKeyVisible ? 'text' : 'password'"
                  autocomplete="new-password"
                  :placeholder="isEditing ? t('aiServices.apiKeyKeep') : ''"
                  :required="!isEditing"
                />
                <button
                  type="button"
                  :title="apiKeyVisible ? t('aiServices.hideKey') : t('aiServices.showKey')"
                  :aria-label="apiKeyVisible ? t('aiServices.hideKey') : t('aiServices.showKey')"
                  :aria-pressed="apiKeyVisible"
                  @click="apiKeyVisible = !apiKeyVisible"
                >
                  <EyeOff v-if="apiKeyVisible" aria-hidden="true" />
                  <Eye v-else aria-hidden="true" />
                </button>
              </span>
              <small v-if="editingProvider?.has_api_key">
                {{ t('aiServices.keyStored', { suffix: editingProvider.api_key_last4 || '----' }) }}
              </small>
            </label>
          </div>

          <label class="ai-service-switch-row">
            <span>
              <strong>{{ t('aiServices.enableService') }}</strong>
              <small>{{ t('aiServices.enableServiceHint') }}</small>
            </span>
            <span class="ai-service-switch">
              <input v-model="form.enabled" type="checkbox" role="switch" />
              <span aria-hidden="true" />
            </span>
          </label>
        </section>

        <details class="ai-service-advanced" :open="advancedOpen" @toggle="syncAdvancedOpen">
          <summary>
            <span>
              <strong>{{ t('aiServices.advancedTitle') }}</strong>
              <small>{{ t('aiServices.advancedHint') }}</small>
            </span>
            <ChevronDown aria-hidden="true" />
          </summary>

          <div class="ai-service-advanced-body">
            <label class="ai-service-switch-row">
              <span>
                <strong>{{ t('aiServices.supportsJsonObject') }}</strong>
                <small>{{ t('aiServices.supportsJsonObjectHint') }}</small>
              </span>
              <span class="ai-service-switch">
                <input
                  v-model="form.capabilities.supports_json_object"
                  type="checkbox"
                  role="switch"
                />
                <span aria-hidden="true" />
              </span>
            </label>

            <label class="ai-service-switch-row">
              <span>
                <strong>{{ t('aiServices.supportsReasoningToggle') }}</strong>
                <small>{{ t('aiServices.supportsReasoningToggleHint') }}</small>
              </span>
              <span class="ai-service-switch">
                <input
                  v-model="form.capabilities.supports_reasoning_toggle"
                  type="checkbox"
                  role="switch"
                />
                <span aria-hidden="true" />
              </span>
            </label>

            <div class="ai-service-advanced-fields">
              <SelectControl
                v-if="form.capabilities.supports_reasoning_toggle"
                :model-value="form.capabilities.reasoning_control || 'none'"
                :label="t('aiServices.reasoningControl')"
                :options="reasoningOptions"
                :disabled="saving"
                @update:model-value="updateReasoningControl"
              />

              <label class="ai-service-field">
                <span>{{ t('aiServices.defaultMaxTokens') }}</span>
                <input
                  v-model.number="form.capabilities.default_max_tokens"
                  class="ai-service-input"
                  type="number"
                  min="256"
                  max="6000"
                />
              </label>
            </div>
          </div>
        </details>
      </div>

      <footer class="ai-service-editor-foot">
        <button
          v-if="isEditing"
          type="button"
          class="tertiary-action"
          :disabled="saving"
          @click="restoreEditor"
        >
          <X aria-hidden="true" />
          {{ t('common.cancel') }}
        </button>
        <button type="submit" class="primary-action" :disabled="saving || !canSave">
          <Save aria-hidden="true" />
          {{ saving ? t('common.saving') : t('aiServices.save') }}
        </button>
      </footer>
    </form>
  </section>

  <dialog
    ref="deleteDialog"
    class="ai-service-delete-dialog"
    @cancel.prevent="dismissDeleteDialog()"
    @click="handleDeleteBackdrop"
  >
    <section v-if="deleteCandidate" class="ai-service-delete-panel">
      <header>
        <span><Trash2 aria-hidden="true" /></span>
        <div>
          <h2>{{ t('aiServices.deleteTitle') }}</h2>
          <p>{{ t('aiServices.deleteDescription', { name: deleteCandidate.display_name }) }}</p>
        </div>
        <button
          type="button"
          class="ai-service-icon-button"
          :disabled="Boolean(deletingId)"
          :title="t('common.cancel')"
          :aria-label="t('common.cancel')"
          @click="dismissDeleteDialog()"
        >
          <X aria-hidden="true" />
        </button>
      </header>
      <footer>
        <button
          ref="deleteCancelButton"
          type="button"
          class="secondary-action"
          :disabled="Boolean(deletingId)"
          @click="dismissDeleteDialog()"
        >
          {{ t('common.cancel') }}
        </button>
        <button
          type="button"
          class="danger-action"
          :disabled="Boolean(deletingId)"
          @click="confirmDelete"
        >
          <Trash2 aria-hidden="true" />
          {{ deletingId ? t('common.deleting') : t('aiServices.deleteAction') }}
        </button>
      </footer>
    </section>
  </dialog>
</template>

<style scoped>
.ai-service-workspace {
  display: grid;
  min-height: 40rem;
  grid-template-columns: minmax(17rem, 19rem) minmax(0, 1fr);
  overflow: hidden;
  border: 1px solid var(--color-line);
  border-radius: var(--radius-panel);
  background: var(--surface-strong);
  box-shadow: var(--shadow-soft);
}

.ai-service-sidebar {
  display: flex;
  min-width: 0;
  flex-direction: column;
  border-right: 1px solid var(--color-line);
  background: var(--surface-strong);
}

.ai-service-sidebar-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  border-bottom: 1px solid var(--line-soft);
  padding: 1.1rem 1rem 0.95rem;
}

.ai-service-sidebar-head p,
.ai-service-editor-title p {
  margin: 0;
  color: var(--color-accent-dark);
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0;
  line-height: 1.4;
  text-transform: uppercase;
}

.ai-service-sidebar-head h2 {
  margin-top: 0.2rem;
  font-size: 0.95rem;
  font-weight: 700;
  line-height: 1.35;
}

.ai-service-sidebar-head span {
  display: block;
  margin-top: 0.2rem;
  color: var(--color-muted);
  font-size: 0.72rem;
  line-height: 1.45;
}

.ai-service-icon-button {
  display: grid;
  width: 2.5rem;
  height: 2.5rem;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid var(--line-soft);
  border-radius: 0.375rem;
  background: var(--surface-strong);
  color: var(--color-muted);
}

.ai-service-icon-button:hover:not(:disabled) {
  border-color: var(--color-accent);
  color: var(--color-accent-dark);
}

.ai-service-icon-button:focus-visible,
.ai-service-list-item:focus-visible,
.ai-service-add:focus-visible,
.ai-service-secret-control button:focus-visible,
.ai-service-advanced summary:focus-visible {
  outline: 3px solid var(--focus-ring);
  outline-offset: 2px;
}

.ai-service-icon-button:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.ai-service-icon-button svg {
  width: 1rem;
  height: 1rem;
}

.ai-service-add {
  display: flex;
  min-height: 2.75rem;
  align-items: center;
  gap: 0.65rem;
  margin: 0.85rem 0.75rem 0;
  border: 1px solid var(--color-line);
  border-radius: 0.375rem;
  background: var(--surface-strong);
  padding: 0.65rem 0.8rem;
  color: var(--color-ink);
  font-size: 0.8rem;
  font-weight: 600;
  text-align: left;
}

.ai-service-add:hover,
.ai-service-add-active {
  border-color: color-mix(in srgb, var(--color-accent) 52%, var(--color-line));
  background: color-mix(in srgb, var(--color-accent) 6%, var(--surface-strong));
  color: var(--color-accent-dark);
}

.ai-service-add svg {
  width: 1rem;
  height: 1rem;
}

.ai-service-list {
  display: grid;
  min-height: 0;
  align-content: start;
  gap: 0.35rem;
  overflow-y: auto;
  padding: 0.75rem;
  scrollbar-color: var(--color-line) transparent;
  scrollbar-width: thin;
}

.ai-service-list-item {
  position: relative;
  display: grid;
  min-width: 0;
  grid-template-columns: 2rem minmax(0, 1fr) auto;
  align-items: start;
  gap: 0.65rem;
  border: 1px solid transparent;
  border-radius: 0.375rem;
  background: transparent;
  padding: 0.75rem 0.65rem;
  color: var(--color-ink);
  text-align: left;
}

.ai-service-list-item::before {
  position: absolute;
  top: 0.6rem;
  bottom: 0.6rem;
  left: 0;
  width: 2px;
  border-radius: 2px;
  background: transparent;
  content: '';
}

.ai-service-list-item:hover {
  border-color: var(--line-soft);
  background: var(--surface-strong);
}

.ai-service-list-item-active {
  border-color: color-mix(in srgb, var(--color-accent) 34%, var(--color-line));
  background: var(--surface-strong);
  box-shadow: 0 8px 20px color-mix(in srgb, var(--color-ink) 5%, transparent);
}

.ai-service-list-item-active::before {
  background: var(--color-accent);
}

.ai-service-provider-mark,
.ai-service-editor-mark {
  display: grid;
  place-items: center;
  border-radius: var(--radius-control);
  background: var(--color-accent-soft);
  color: var(--color-accent-dark);
  font-size: 0.72rem;
  font-weight: 700;
}

:root[data-theme='dark'] .ai-service-provider-mark,
:root[data-theme='dark'] .ai-service-editor-mark {
  color: var(--color-accent);
}

.ai-service-provider-mark {
  width: 2rem;
  height: 2rem;
}

.ai-service-list-copy {
  display: grid;
  min-width: 0;
  gap: 0.14rem;
}

.ai-service-list-copy strong,
.ai-service-list-copy span,
.ai-service-list-copy small {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ai-service-list-copy strong {
  font-size: 0.78rem;
  line-height: 1.4;
}

.ai-service-list-copy span {
  color: var(--color-muted);
  font-size: 0.68rem;
  line-height: 1.45;
}

.ai-service-list-copy small {
  color: color-mix(in srgb, var(--color-muted) 78%, transparent);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.6rem;
  line-height: 1.5;
}

.ai-service-status-dot {
  width: 0.55rem;
  height: 0.55rem;
  margin-top: 0.28rem;
  border: 2px solid var(--surface-muted);
  border-radius: 50%;
  box-shadow: 0 0 0 1px var(--color-line);
}

.ai-service-status-dot-enabled {
  background: var(--color-moss);
}
.ai-service-status-dot-disabled {
  background: var(--color-muted);
}

.ai-service-skeletons {
  display: grid;
  gap: 0.55rem;
  padding: 0.8rem;
}

.ai-service-skeletons span {
  height: 4.25rem;
  border-radius: 0.375rem;
  background: color-mix(in srgb, var(--color-line) 55%, transparent);
  animation: ai-service-pulse 1.2s ease-in-out infinite alternate;
}

.ai-service-empty {
  display: grid;
  place-items: center;
  gap: 0.45rem;
  padding: 3rem 1.5rem;
  color: var(--color-muted);
  text-align: center;
}

.ai-service-empty svg {
  width: 1.5rem;
  height: 1.5rem;
  color: var(--color-brass);
}

.ai-service-empty strong {
  color: var(--color-ink);
  font-size: 0.82rem;
}

.ai-service-empty span {
  max-width: 13rem;
  font-size: 0.72rem;
  line-height: 1.6;
}

.ai-service-editor {
  display: grid;
  min-width: 0;
  grid-template-rows: auto auto minmax(0, 1fr) auto;
  background: var(--surface-strong);
}

.ai-service-editor-head {
  display: flex;
  min-width: 0;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  border-bottom: 1px solid var(--line-soft);
  padding: 1rem 1.25rem;
}

.ai-service-editor-title {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 0.8rem;
}

.ai-service-editor-mark {
  width: 2.75rem;
  height: 2.75rem;
  flex: 0 0 auto;
  font-size: 0.9rem;
}

.ai-service-editor-mark svg {
  width: 1.2rem;
  height: 1.2rem;
}

.ai-service-editor-title > div {
  min-width: 0;
}

.ai-service-editor-title h2 {
  overflow: hidden;
  margin-top: 0.12rem;
  font-family: var(--font-display);
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: 0;
  line-height: 1.35;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ai-service-editor-title > div > span {
  display: block;
  overflow: hidden;
  margin-top: 0.1rem;
  color: var(--color-muted);
  font-size: 0.68rem;
  line-height: 1.45;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ai-service-editor-actions {
  display: flex;
  flex: 0 0 auto;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.45rem;
}

.ai-service-feedback {
  display: grid;
  gap: 0.5rem;
  padding: 0.75rem 1.25rem 0;
}

.ai-service-editor-body {
  min-width: 0;
  padding: 1.25rem;
}

.ai-service-form-section > header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1rem;
}

.ai-service-form-section h3,
.ai-service-advanced summary strong {
  font-size: 0.85rem;
  font-weight: 700;
  line-height: 1.4;
}

.ai-service-form-section header p,
.ai-service-advanced summary small {
  margin-top: 0.2rem;
  color: var(--color-muted);
  font-size: 0.7rem;
  line-height: 1.55;
}

.ai-service-form-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
}

.ai-service-field {
  display: grid;
  min-width: 0;
  align-content: start;
  gap: 0.5rem;
  color: var(--color-ink);
  font-size: 0.875rem;
}

.ai-service-field > span:first-child {
  line-height: 1.4;
}

.ai-service-field small {
  color: var(--color-muted);
  font-size: 0.68rem;
  line-height: 1.5;
}

.ai-service-field-wide {
  grid-column: 1 / -1;
}

.ai-service-input {
  width: 100%;
  min-width: 0;
  min-height: 3rem;
  border: 1px solid var(--color-line);
  border-radius: 0.5rem;
  background: var(--surface-strong);
  padding: 0.65rem 0.9rem;
  color: var(--color-ink);
  box-shadow: inset 0 1px 0 var(--action-inset-highlight);
  font-size: 0.875rem;
  line-height: 1.4;
  outline: 0;
}

.ai-service-input:hover {
  border-color: color-mix(in srgb, var(--color-accent) 42%, var(--color-line));
}

.ai-service-input:focus {
  border-color: var(--color-accent);
  box-shadow:
    0 0 0 4px var(--focus-ring),
    inset 0 1px 0 var(--action-inset-highlight);
}

.ai-service-input::placeholder {
  color: var(--color-muted);
}

.ai-service-monospace {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.8rem;
}

.ai-service-secret-control {
  position: relative;
  display: block;
}

.ai-service-secret-control .ai-service-input {
  padding-right: 3rem;
}

.ai-service-secret-control button {
  position: absolute;
  top: 50%;
  right: 0.4rem;
  display: grid;
  width: 2.2rem;
  height: 2.2rem;
  place-items: center;
  border: 0;
  border-radius: 0.375rem;
  background: transparent;
  color: var(--color-muted);
  transform: translateY(-50%);
}

.ai-service-secret-control button:hover {
  background: var(--action-tertiary-hover);
  color: var(--color-ink);
}

.ai-service-secret-control svg {
  width: 1rem;
  height: 1rem;
}

.ai-service-switch-row {
  display: flex;
  min-width: 0;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  border-top: 1px solid var(--line-soft);
  padding: 0.9rem 0;
  color: var(--color-ink);
}

.ai-service-form-section > .ai-service-switch-row {
  margin-top: 1rem;
}

.ai-service-switch-row > span:first-child {
  display: grid;
  min-width: 0;
  gap: 0.15rem;
}

.ai-service-switch-row strong {
  font-size: 0.78rem;
  line-height: 1.45;
}

.ai-service-switch-row small {
  color: var(--color-muted);
  font-size: 0.68rem;
  line-height: 1.5;
}

.ai-service-switch {
  position: relative;
  display: block;
  width: 2.65rem;
  height: 1.5rem;
  flex: 0 0 auto;
}

.ai-service-switch input {
  position: absolute;
  width: 1px;
  height: 1px;
  opacity: 0;
}

.ai-service-switch > span {
  position: absolute;
  inset: 0;
  border: 1px solid var(--color-line);
  border-radius: 999px;
  background: var(--surface-muted);
  cursor: pointer;
  transition:
    background-color 140ms ease,
    border-color 140ms ease;
}

.ai-service-switch > span::after {
  position: absolute;
  top: 0.2rem;
  left: 0.22rem;
  width: 1rem;
  height: 1rem;
  border-radius: 50%;
  background: var(--color-muted);
  box-shadow: var(--shadow-soft);
  content: '';
  transition:
    background-color 140ms ease,
    transform 140ms ease;
}

.ai-service-switch input:checked + span {
  border-color: var(--color-moss);
  background: color-mix(in srgb, var(--color-moss) 22%, var(--surface-strong));
}

.ai-service-switch input:checked + span::after {
  background: var(--color-moss);
  transform: translateX(1.12rem);
}

.ai-service-switch input:focus-visible + span {
  outline: 3px solid var(--focus-ring);
  outline-offset: 2px;
}

.ai-service-advanced {
  border-top: 1px solid var(--color-line);
  border-bottom: 1px solid var(--color-line);
}

.ai-service-advanced summary {
  display: flex;
  min-height: 4rem;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.85rem 0;
  cursor: pointer;
  list-style: none;
}

.ai-service-advanced summary::-webkit-details-marker {
  display: none;
}

.ai-service-advanced summary > span {
  display: grid;
  gap: 0.05rem;
}

.ai-service-advanced summary svg {
  width: 1rem;
  height: 1rem;
  flex: 0 0 auto;
  color: var(--color-muted);
  transition: transform 140ms ease;
}

.ai-service-advanced[open] summary svg {
  transform: rotate(180deg);
}

.ai-service-advanced-body {
  padding-bottom: 1rem;
}

.ai-service-advanced-fields {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
  border-top: 1px solid var(--line-soft);
  padding-top: 1rem;
}

.ai-service-editor-foot {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 0.65rem;
  border-top: 1px solid var(--color-line);
  background: var(--surface-strong);
  padding: 0.85rem 1.25rem;
}

.ai-service-delete-dialog {
  width: min(30rem, calc(100vw - 2rem));
  max-width: none;
  border: 0;
  padding: 0;
  background: transparent;
  color: var(--color-ink);
}

.ai-service-delete-dialog::backdrop {
  background: var(--scrim-strong);
  backdrop-filter: blur(2px);
}

.ai-service-delete-panel {
  overflow: hidden;
  border: 1px solid var(--color-line);
  border-radius: 0.5rem;
  background: var(--color-paper);
  box-shadow: var(--shadow-lift);
}

.ai-service-delete-panel > header {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: start;
  gap: 0.85rem;
  padding: 1.25rem;
}

.ai-service-delete-panel > header > span {
  display: grid;
  width: 2rem;
  height: 2rem;
  place-items: center;
  border-radius: 0.375rem;
  background: var(--color-danger-bg);
  color: var(--color-danger);
}

.ai-service-delete-panel > header > span svg {
  width: 1rem;
  height: 1rem;
}

.ai-service-delete-panel h2 {
  font-size: 1rem;
  font-weight: 700;
}

.ai-service-delete-panel p {
  margin-top: 0.35rem;
  color: var(--color-muted);
  font-size: 0.8rem;
  line-height: 1.6;
}

.ai-service-delete-panel > footer {
  display: flex;
  justify-content: flex-end;
  gap: 0.65rem;
  border-top: 1px solid var(--line-soft);
  background: var(--surface-muted);
  padding: 0.9rem 1.25rem;
}

@keyframes ai-service-pulse {
  from {
    opacity: 0.48;
  }
  to {
    opacity: 0.86;
  }
}

@keyframes ai-service-rotate {
  to {
    transform: rotate(360deg);
  }
}

.ai-service-spin {
  animation: ai-service-rotate 0.8s linear infinite;
}

@media (max-width: 900px) {
  .ai-service-workspace {
    min-height: 0;
    grid-template-columns: minmax(0, 1fr);
  }

  .ai-service-sidebar {
    border-right: 0;
    border-bottom: 1px solid var(--color-line);
  }

  .ai-service-list {
    max-height: 18rem;
  }
}

@media (max-width: 640px) {
  .ai-service-editor-head,
  .ai-service-form-section > header {
    align-items: stretch;
    flex-direction: column;
  }

  .ai-service-editor-actions {
    justify-content: flex-start;
  }

  .ai-service-editor-actions > button {
    flex: 1 1 auto;
  }

  .ai-service-editor-body {
    padding: 1rem;
  }

  .ai-service-form-grid,
  .ai-service-advanced-fields {
    grid-template-columns: minmax(0, 1fr);
  }

  .ai-service-field-wide {
    grid-column: auto;
  }

  .ai-service-editor-foot {
    padding: 0.85rem 1rem;
  }

  .ai-service-editor-foot .primary-action {
    flex: 1 1 auto;
  }

  .ai-service-delete-panel > footer {
    align-items: stretch;
    flex-direction: column-reverse;
  }
}

@media (prefers-reduced-motion: reduce) {
  .ai-service-skeletons span,
  .ai-service-spin {
    animation: none;
  }

  .ai-service-switch > span,
  .ai-service-switch > span::after,
  .ai-service-advanced summary svg {
    transition: none;
  }
}
</style>
