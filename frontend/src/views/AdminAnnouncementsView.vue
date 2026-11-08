<script setup lang="ts">
import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import { Bell, Send } from '@lucide/vue'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { api } from '../api/client'
import { localizedApiError } from '../errorLocalization'
import { formatLocalizedDate } from '../localization'
import type { Announcement, AnnouncementPayload } from '../resourceTypes'
import AdminConfirmDialog from '../components/AdminConfirmDialog.vue'
import QuotaChangeDetails from '../components/QuotaChangeDetails.vue'
import { useDirtyRouteGuard } from '../admin'

const { t, te, locale } = useI18n()
const client = useQueryClient()
const query = useQuery({ queryKey: ['admin-announcements'], queryFn: api.adminAnnouncements })
const title = ref(''),
  body = ref(''),
  publishAt = ref(''),
  expiresAt = ref(''),
  editing = ref('')
const allUsers = ref(true),
  pinned = ref(false),
  search = ref(''),
  removeId = ref('')
const selected = ref<Record<string, string>>({})
const users = useQuery({
  queryKey: computed(() => ['admin-users', 'announcement', search.value]),
  queryFn: () => api.adminUsers({ search: search.value, limit: 20 }),
  enabled: computed(() => !allUsers.value),
})
const message = ref('')
const guard = useDirtyRouteGuard(computed(() => Boolean(title.value || body.value)))
function reset() {
  title.value = ''
  body.value = ''
  editing.value = ''
  publishAt.value = ''
  expiresAt.value = ''
  selected.value = {}
  allUsers.value = true
  pinned.value = false
}
function localDate(value: string) {
  const date = new Date(value)
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16)
}
function edit(row: Announcement) {
  editing.value = row.id
  title.value = row.title
  body.value = row.body
  pinned.value = row.pinned
  allUsers.value = row.audience === 'all'
  selected.value = Object.fromEntries(
    (row.audience === 'all' ? [] : row.audience).map((id) => [
      id,
      row.audience_labels?.[id] ?? t('resource.selectedUsers'),
    ]),
  )
  publishAt.value = localDate(row.publish_at)
  expiresAt.value = row.expires_at ? localDate(row.expires_at) : ''
}
const mutation = useMutation({
  mutationFn: async () => {
    const payload: AnnouncementPayload = {
      title: title.value,
      body: body.value,
      pinned: pinned.value,
      audience: allUsers.value ? 'all' : Object.keys(selected.value),
      publish_at: publishAt.value ? new Date(publishAt.value).toISOString() : undefined,
      expires_at: expiresAt.value ? new Date(expiresAt.value).toISOString() : null,
    }
    return editing.value
      ? api.updateAnnouncement(editing.value, payload)
      : api.publishAnnouncement(payload)
  },
  onSuccess: async () => {
    reset()
    message.value = t('resource.saved')
    await Promise.all([
      client.invalidateQueries({ queryKey: ['admin-announcements'] }),
      client.invalidateQueries({ queryKey: ['announcements'] }),
    ])
  },
})
function publish() {
  message.value = ''
  mutation.mutate()
}
const removal = useMutation({
  mutationFn: (id: string) => api.deleteAnnouncement(id),
  onSuccess: async () => {
    removeId.value = ''
    await client.invalidateQueries({ queryKey: ['admin-announcements'] })
    await client.invalidateQueries({ queryKey: ['announcements'] })
  },
})
function date(value: string) {
  return formatLocalizedDate(value, locale.value, { dateStyle: 'medium', timeStyle: 'short' })
}
</script>

<template>
  <section class="admin-page resource-page">
    <header class="admin-page-header"
      ><div
        ><p>{{ t('resource.announcements') }}</p
        ><h1>{{ t('resource.announcementTitle') }}</h1
        ><span>{{ t('resource.announcementSubtitle') }}</span></div
      ><Bell class="resource-heading-icon" aria-hidden="true"
    /></header>
    <form class="admin-settings-panel resource-form" @submit.prevent="publish">
      <label
        ><span>{{ t('resource.title') }}</span
        ><input v-model="title" class="field-control" maxlength="200" required
      /></label>
      <label
        ><span>{{ t('resource.body') }}</span
        ><textarea v-model="body" class="field-control" rows="6" maxlength="20000" required />
      </label>
      <div class="resource-mode"
        ><label class="resource-check"
          ><input v-model="allUsers" type="radio" :value="true" />{{
            t('resource.allUsers')
          }}</label
        ><label class="resource-check"
          ><input v-model="allUsers" type="radio" :value="false" />{{
            t('resource.selectedUsers')
          }}</label
        ><label class="resource-check"
          ><input v-model="pinned" type="checkbox" />{{ t('resource.pinned') }}</label
        ></div
      >
      <section v-if="!allUsers" class="resource-recipient-picker"
        ><label
          ><span>{{ t('resource.audienceHint') }}</span
          ><input v-model="search" type="search" class="field-control" /></label
        ><div class="resource-recipient-chips"
          ><button
            v-for="(email, id) in selected"
            :key="id"
            type="button"
            class="resource-state"
            @click="delete selected[id]"
            >{{ email }} ×</button
          ></div
        ><div class="resource-recipient-results"
          ><button
            v-for="user in users.data.value?.items ?? []"
            :key="user.id"
            type="button"
            class="secondary-action"
            :disabled="Boolean(selected[user.id])"
            @click="selected[user.id] = user.email"
            >{{ user.email }}</button
          ></div
        ><p v-if="users.error.value" class="error-copy">{{
          localizedApiError(users.error.value, t, te, 'resource.error')
        }}</p></section
      >
      <div class="admin-quota-grid"
        ><label
          ><span>{{ t('resource.publishAt') }}</span
          ><input v-model="publishAt" type="datetime-local" class="field-control" /></label
        ><label
          ><span>{{ t('resource.expiresAt') }}</span
          ><input v-model="expiresAt" type="datetime-local" class="field-control" /></label
      ></div>
      <div class="resource-actions"
        ><button
          class="primary-action"
          type="submit"
          :disabled="mutation.isPending.value || (!allUsers && !Object.keys(selected).length)"
          ><Send aria-hidden="true" />{{
            editing ? t('resource.save') : t('resource.publish')
          }}</button
        ><button v-if="editing" class="secondary-action" type="button" @click="reset">{{
          t('resource.cancel')
        }}</button></div
      >
      <p v-if="message" class="success-copy" aria-live="polite">{{ message }}</p>
      <p v-if="mutation.error.value" class="error-copy" role="alert">{{
        localizedApiError(mutation.error.value, t, te, 'resource.error')
      }}</p>
    </form>
    <section class="resource-announcement-list">
      <article
        v-for="row in query.data.value?.items ?? []"
        :key="row.id"
        class="admin-settings-panel resource-announcement"
      >
        <header
          ><div
            ><span class="resource-state"
              >{{
                t(
                  `resource.${row.kind === 'manual' ? 'manual' : row.kind === 'quota_change' ? 'quotaChangeTitle' : 'cleanupNotice'}`,
                )
              }}
              <span v-if="row.pinned">· {{ t('resource.pinned') }}</span></span
            ><h2>{{ row.kind === 'quota_change' ? t('resource.quotaChangeTitle') : row.title }}</h2
            ><p
              >{{ date(row.publish_at) }}{{ row.expires_at ? ' → ' + date(row.expires_at) : '' }}</p
            ></div
          ><div v-if="row.kind === 'manual'" class="resource-actions"
            ><button class="secondary-action" @click="edit(row)">{{ t('resource.edit') }}</button
            ><button class="secondary-action" @click="removeId = row.id">{{
              t('resource.remove')
            }}</button></div
          ></header
        >
        <QuotaChangeDetails v-if="row.quota_change" :changes="row.quota_change" />
        <p v-else class="resource-notice-body">{{ row.body }}</p>
      </article>
      <p v-if="!query.isLoading.value && !query.data.value?.items.length" class="resource-help">{{
        t('resource.noNotices')
      }}</p>
      <p v-if="query.error.value || removal.error.value" class="error-copy">{{
        localizedApiError(query.error.value || removal.error.value, t, te, 'resource.error')
      }}</p>
    </section>
    <AdminConfirmDialog
      :open="Boolean(removeId)"
      :title="t('resource.remove')"
      :description="query.data.value?.items.find((row) => row.id === removeId)?.title ?? ''"
      :confirm-label="t('resource.remove')"
      :cancel-label="t('resource.cancel')"
      :busy="removal.isPending.value"
      @confirm="removal.mutate(removeId)"
      @cancel="removeId = ''"
    />
    <AdminConfirmDialog
      :open="guard.discardDialogOpen.value"
      :title="t('admin.discardTitle')"
      :description="t('admin.discardDescription')"
      :confirm-label="t('admin.discardChanges')"
      :cancel-label="t('admin.keepEditing')"
      @confirm="guard.confirmDiscard"
      @cancel="guard.keepEditing"
    />
  </section>
</template>
