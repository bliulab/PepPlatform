import { onBeforeUnmount, onMounted, watch, type Ref } from 'vue'
import { useRoute } from 'vue-router'
import { onQuotaUsageChanged } from './accountQuota'
import { useAuthStore } from './stores/auth'
import { useCalendarClock } from './useCalendarClock'

export function useQuotaRefresh(enabled: Ref<boolean>) {
  const auth = useAuthStore()
  const route = useRoute()
  const now = useCalendarClock()
  let timer: ReturnType<typeof setInterval> | undefined
  let unsubscribe: (() => void) | undefined
  const refresh = () => {
    if (enabled.value && !document.hidden) void auth.refreshUsage()
  }
  watch([enabled, () => route.fullPath, () => now.value.toISOString().slice(0, 10)], refresh)
  onMounted(() => {
    unsubscribe = onQuotaUsageChanged(refresh)
    window.addEventListener('focus', refresh)
    document.addEventListener('visibilitychange', refresh)
    timer = setInterval(refresh, 60_000)
    refresh()
  })
  onBeforeUnmount(() => {
    unsubscribe?.()
    if (timer) clearInterval(timer)
    window.removeEventListener('focus', refresh)
    document.removeEventListener('visibilitychange', refresh)
  })
}
