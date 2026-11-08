<script setup lang="ts">
import { computed, ref } from 'vue'
import { useQuery } from '@tanstack/vue-query'
import { useI18n } from 'vue-i18n'
import { request } from '../api/client'
import { shortVersion, type RunVersions } from '../versioning'

const props = defineProps<{ experimentId: string }>()
const { t, te } = useI18n()
const versions = useQuery({
  queryKey: computed(() => ['experiment-versions', props.experimentId]),
  queryFn: () =>
    request<RunVersions>(`/api/experiments/${encodeURIComponent(props.experimentId)}/versions`),
  refetchInterval: (query) =>
    ['preparing', 'running'].includes(query.state.data?.status ?? '') ? 4000 : false,
})
const primaryRoles = new Set([
  'config',
  'effective_config',
  'dataset',
  'split',
  'source',
  'model',
  'checkpoint',
  'environment',
])
const primary = computed(
  () => versions.data.value?.versions.filter((v) => primaryRoles.has(v.role)) ?? [],
)
const dependencies = computed(
  () => versions.data.value?.versions.filter((v) => !primaryRoles.has(v.role)) ?? [],
)
const selectedVersion = ref('')
const related = useQuery({
  queryKey: computed(() => ['version-runs', selectedVersion.value]),
  queryFn: () =>
    request<{ items: { id: string; name: string; status: string }[] }>(
      `/api/versions/${encodeURIComponent(selectedVersion.value)}/runs`,
    ),
  enabled: computed(() => Boolean(selectedVersion.value)),
})
const checked = computed(
  () =>
    versions.data.value?.details?.last_verification?.checked_at ??
    versions.data.value?.details?.checked_at,
)
const label = (key: string) => (te(`versions.${key}`) ? t(`versions.${key}`) : key)
</script>

<template>
  <section class="panel space-y-4" aria-labelledby="experiment-versions-title">
    <div class="flex flex-wrap items-start justify-between gap-3">
      <div>
        <h2 id="experiment-versions-title" class="font-display text-xl">{{
          t('versions.title')
        }}</h2>
        <p class="mt-1 text-sm text-slate-500">{{ t('versions.hint') }}</p>
      </div>
      <a
        v-if="versions.data.value && versions.data.value.status !== 'not_recorded'"
        class="text-sm font-medium text-teal-800 underline underline-offset-4 focus-visible:outline focus-visible:outline-2"
        :href="`/api/experiments/${encodeURIComponent(experimentId)}/versions?download=true`"
      >
        {{ t('versions.export') }}
      </a>
    </div>
    <p v-if="versions.isPending.value" class="text-sm" role="status">{{ t('versions.loading') }}</p>
    <div v-else-if="versions.isError.value" class="text-sm" role="alert">
      {{ t('versions.unavailable') }}
      <button class="ml-2 underline" @click="versions.refetch()">{{ t('versions.retry') }}</button>
    </div>
    <template v-else-if="versions.data.value">
      <p
        class="text-sm"
        :class="
          ['seal_failed', 'incomplete'].includes(versions.data.value.status)
            ? 'text-amber-800'
            : 'text-slate-600'
        "
        role="status"
      >
        {{ label(versions.data.value.status) }}
        <span v-if="checked" class="ml-3 text-xs"
          >{{ t('versions.checked') }}: {{ new Date(checked).toLocaleString() }}</span
        >
        <span v-if="versions.data.value.details?.last_verification" class="ml-2">{{
          label(versions.data.value.details.last_verification.status)
        }}</span>
      </p>
      <dl v-if="primary.length" class="grid gap-x-6 gap-y-3 sm:grid-cols-2 xl:grid-cols-4">
        <div
          v-for="version in primary"
          :key="version.role"
          class="border-l-2 border-teal-700/30 pl-3"
        >
          <dt class="text-xs text-slate-500">{{ label(version.role) }}</dt>
          <dd class="mt-1">
            <button
              class="break-all text-left font-mono text-xs text-teal-900 underline decoration-teal-900/30 underline-offset-4 focus-visible:outline focus-visible:outline-2"
              :title="`${t('versions.related')}: ${version.id}`"
              @click="selectedVersion = selectedVersion === version.id ? '' : version.id"
            >
              {{ shortVersion(version.id) }}
            </button>
          </dd>
        </div>
      </dl>
      <p
        v-if="versions.data.value.details?.source_revision"
        class="break-all font-mono text-xs text-slate-500"
      >
        Git {{ versions.data.value.details.source_revision.git_commit ?? t('versions.unknown') }}
        <span v-if="versions.data.value.details.source_revision.git_dirty">
          · {{ t('versions.dirty') }}</span
        >
      </p>
      <div v-if="selectedVersion" class="section-card space-y-2 text-sm">
        <h3 class="font-medium">{{ t('versions.related') }}</h3>
        <p v-if="related.isPending.value" role="status">{{ t('versions.loading') }}</p>
        <p v-else-if="related.isError.value" role="alert">{{ t('versions.unavailable') }}</p>
        <ul v-else class="space-y-1"
          ><li v-for="run in related.data.value?.items" :key="run.id"
            ><RouterLink class="underline" :to="`/experiments/${run.id}`">{{
              run.name
            }}</RouterLink></li
          ></ul
        >
      </div>
      <details v-if="versions.data.value.relations.length" class="text-sm">
        <summary class="cursor-pointer">{{ t('versions.lineage') }}</summary>
        <ul class="mt-2 space-y-1"
          ><li
            v-for="edge in versions.data.value.relations"
            :key="edge.relation + edge.target"
            class="break-all"
            >{{ label(edge.relation) }}: <code>{{ edge.target }}</code></li
          ></ul
        >
      </details>
      <details v-if="dependencies.length" class="text-sm">
        <summary class="cursor-pointer"
          >{{ t('versions.more') }} ({{ dependencies.length }})</summary
        >
        <ul class="mt-2 space-y-1 font-mono text-xs"
          ><li v-for="dependency in dependencies" :key="dependency.role" class="break-all"
            >{{ dependency.role }} · {{ dependency.id }}</li
          ></ul
        >
      </details>
      <details v-if="primary.length" class="text-sm">
        <summary class="cursor-pointer">{{ t('versions.inspect') }}</summary>
        <pre class="mt-3 max-h-96 overflow-auto whitespace-pre-wrap break-all text-xs">{{
          JSON.stringify(versions.data.value, null, 2)
        }}</pre>
      </details>
    </template>
  </section>
</template>
