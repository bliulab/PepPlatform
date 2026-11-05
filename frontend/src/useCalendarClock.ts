import { onBeforeUnmount, onMounted, ref } from 'vue'

/** Calendar buckets update at local/UTC midnight and after a suspended tab wakes. */
export function useCalendarClock() {
  const now = ref(new Date())
  let timer: ReturnType<typeof setTimeout> | undefined
  function update() {
    now.value = new Date()
    if (timer) clearTimeout(timer)
    const local = new Date(now.value.getFullYear(), now.value.getMonth(), now.value.getDate() + 1)
    const utc = new Date(
      Date.UTC(now.value.getUTCFullYear(), now.value.getUTCMonth(), now.value.getUTCDate() + 1),
    )
    timer = setTimeout(update, Math.min(local.getTime(), utc.getTime()) - now.value.getTime() + 30)
  }
  function visible() {
    if (!document.hidden) update()
  }
  onMounted(() => {
    update()
    window.addEventListener('focus', update)
    document.addEventListener('visibilitychange', visible)
  })
  onBeforeUnmount(() => {
    if (timer) clearTimeout(timer)
    window.removeEventListener('focus', update)
    document.removeEventListener('visibilitychange', visible)
  })
  return now
}
