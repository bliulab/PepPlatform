import { createApp, watch } from 'vue'
import { createPinia } from 'pinia'
import { QueryClient, VueQueryPlugin } from '@tanstack/vue-query'
import App from './App.vue'
import { i18n } from './i18n'
import { normalizeLocale } from './localization'
import { router } from './router'
import { useAuthStore } from './stores/auth'
import { useAgentChatStore } from './stores/agentChat'
import { useExperimentDraftStore } from './stores/experimentDraft'
import { setUnauthorizedHandler } from './authSession'
import { loginLocation } from './authNavigation'
import { clearLoginDrafts, saveLoginDrafts } from './loginDraft'
import './styles.css'

async function bootstrap() {
  const pinia = createPinia()
  const auth = useAuthStore(pinia)
  const queryClient = new QueryClient()
  const chat = useAgentChatStore(pinia)
  const draft = useExperimentDraftStore(pinia)
  watch(
    () => auth.user?.id,
    (id, previousId) => {
      if (!previousId || id === previousId) return
      queryClient.clear()
      chat.reset()
      draft.$reset()
      if (id) clearLoginDrafts()
    },
    { flush: 'sync' },
  )
  setUnauthorizedHandler(() => {
    if (!auth.authEnabled || !auth.user) return
    const target = router.currentRoute.value.fullPath
    saveLoginDrafts(auth.user.id)
    auth.clearCurrentUser()
    void router.replace(loginLocation(target))
  })
  try {
    await auth.loadCurrentUser()
  } catch (error) {
    console.error('Failed to load the current account before startup.', error)
  }
  i18n.global.locale.value = normalizeLocale(auth.user?.preferred_locale || 'en')
  document.documentElement.lang = i18n.global.locale.value
  createApp(App).use(pinia).use(router).use(i18n).use(VueQueryPlugin, { queryClient }).mount('#app')
}

void bootstrap()
