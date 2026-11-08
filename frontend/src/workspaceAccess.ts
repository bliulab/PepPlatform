import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { loginLocation } from './authNavigation'
import { saveLoginDrafts } from './loginDraft'
import { useAuthStore } from './stores/auth'

export function useWorkspaceAccess() {
  const auth = useAuthStore()
  const route = useRoute()
  const router = useRouter()
  const isGuest = computed(() => auth.authEnabled && !auth.user)
  const canUseWorkspace = computed(() => !isGuest.value)
  function requireLogin() {
    if (!isGuest.value) return true
    saveLoginDrafts(null)
    void router.push(loginLocation(route.fullPath))
    return false
  }
  return { isGuest, canUseWorkspace, requireLogin }
}
