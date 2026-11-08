<script setup lang="ts">
import {
  Bot,
  Boxes,
  Database,
  FlaskConical,
  House,
  LogIn,
  Menu as MenuIcon,
  ShieldCheck,
  Trophy,
  X,
} from '@lucide/vue'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { Component } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { normalizeLocale, type SupportedLocale } from './localization'
import { useAuthStore } from './stores/auth'
import { useAgentChatStore } from './stores/agentChat'
import GuestNotice from './components/GuestNotice.vue'
import PlatformNotices from './components/PlatformNotices.vue'
import { useQuotaRefresh } from './useQuotaRefresh'
import { clearLoginDrafts, saveLoginDrafts } from './loginDraft'
import { allowsGuestPreview, authRoutePaths, loginLocation } from './authNavigation'

const { locale, t } = useI18n()
const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const agentChat = useAgentChatStore()
const themeStorageKey = 'pepplatform-theme'
type Theme = 'light' | 'dark'

function isTheme(value: string | null): value is Theme {
  return value === 'light' || value === 'dark'
}

function readStoredTheme(): Theme {
  if (typeof window === 'undefined') return 'light'
  const storedTheme = window.localStorage.getItem(themeStorageKey)
  return isTheme(storedTheme) ? storedTheme : 'light'
}

const theme = ref<Theme>(readStoredTheme())
const localeSaving = ref(false)
const accountMenu = ref<HTMLElement | null>(null)
const accountMenuOpen = ref(false)
const mobileNavOpen = ref(false)
const themeToggleLabel = computed(() =>
  theme.value === 'dark' ? t('theme.switchToLight') : t('theme.switchToDark'),
)
const isWideRoute = computed(() => Boolean(route.meta.wide))
const isAgentChatRoute = computed(() => route.meta.agentTab === 'chat')
const isAuthRoute = computed(() => authRoutePaths.has(route.path))
const isLandingRoute = computed(() => Boolean(route.meta.stage))
const canRenderRoute = computed(
  () =>
    !auth.authEnabled ||
    Boolean(auth.user) ||
    Boolean(route.meta.public) ||
    allowsGuestPreview(route),
)
const showPrimaryNav = computed(() => !isAuthRoute.value && !isLandingRoute.value)
useQuotaRefresh(computed(() => showPrimaryNav.value && Boolean(auth.user)))
/** Theme colour of the browser chrome while the forced-dark stage is on screen. */
const STAGE_THEME_COLOR = '#050f0e'

function syncThemeColor() {
  if (typeof document === 'undefined') return
  const onStage = document.documentElement.dataset.stage === 'landing'
  const content = onStage ? STAGE_THEME_COLOR : theme.value === 'dark' ? '#0b0f17' : '#ffffff'
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', content)
}

/**
 * `/` and `/welcome` are always dark: the attribute drives the token
 * remap in `styles.css`. It is scoped to the document element and removed on
 * leave, so the stored light/dark preference is never written or shadowed.
 */
function applyStageChrome(onStage: boolean) {
  if (typeof document === 'undefined') return
  if (onStage) document.documentElement.dataset.stage = 'landing'
  else delete document.documentElement.dataset.stage
  syncThemeColor()
}
const mainClass = computed(() => {
  if (isLandingRoute.value) return 'app-main--landing'
  if (isAuthRoute.value) return 'app-main--auth'
  const width = isWideRoute.value ? 'max-w-[96rem]' : 'max-w-7xl'
  return `mx-auto px-4 pb-10 pt-6 sm:px-6 ${width}`
})
const navItems = computed<{ to: string; label: string; icon: Component }[]>(() => [
  { to: '/dashboard', label: t('nav.home'), icon: House },
  { to: '/datasets', label: t('nav.datasets'), icon: Database },
  { to: '/experiments', label: t('nav.experiments'), icon: FlaskConical },
  { to: '/models', label: t('nav.models'), icon: Boxes },
  { to: '/leaderboard', label: t('nav.leaderboard'), icon: Trophy },
  { to: '/agent', label: t('nav.agent'), icon: Bot },
  ...(isAdmin.value ? [{ to: '/admin', label: t('nav.admin'), icon: ShieldCheck }] : []),
])

function isNavActive(path: string) {
  return route.path === path || route.path.startsWith(`${path}/`)
}
const accountDisplayName = computed(() => {
  const name = auth.user?.display_name?.trim()
  if (name) return name
  const prefix = auth.user?.email?.split('@')[0]?.trim()
  return prefix || t('account.user')
})
const isAdmin = computed(() => auth.user?.role === 'admin')
const accountMeta = computed(() => {
  if (!auth.user) return ''
  const role = auth.user.role === 'admin' ? t('account.admin') : t('account.user')
  const verification = auth.user.email_verified ? t('account.verified') : t('account.unverified')
  return `${role} · ${verification}`
})

if (typeof document !== 'undefined') {
  document.documentElement.dataset.theme = theme.value
}

function applyTheme(nextTheme: Theme) {
  theme.value = nextTheme
  if (typeof document !== 'undefined') {
    document.documentElement.dataset.theme = nextTheme
  }
  syncThemeColor()
  if (typeof window !== 'undefined') {
    window.localStorage.setItem(themeStorageKey, nextTheme)
  }
}

function toggleTheme() {
  applyTheme(theme.value === 'dark' ? 'light' : 'dark')
}

function applyLocale(nextLocale: SupportedLocale) {
  locale.value = nextLocale
  if (typeof document !== 'undefined') {
    document.documentElement.lang = nextLocale
  }
}

async function toggleLocale() {
  if (localeSaving.value) return
  const previousLocale = normalizeLocale(locale.value)
  const nextLocale: SupportedLocale = previousLocale === 'zh' ? 'en' : 'zh'
  applyLocale(nextLocale)
  if (!auth.user) return
  localeSaving.value = true
  try {
    await auth.updatePreferredLocale(nextLocale)
  } catch {
    applyLocale(previousLocale)
    window.alert(t('language.saveFailed'))
  } finally {
    localeSaving.value = false
  }
}

function closeAccountMenu() {
  accountMenuOpen.value = false
}

function handleAccountFocusOut(event: FocusEvent) {
  if (!(event.relatedTarget instanceof Node) || !accountMenu.value?.contains(event.relatedTarget))
    closeAccountMenu()
}

async function toggleAccountMenu() {
  accountMenuOpen.value = !accountMenuOpen.value
  if (accountMenuOpen.value) {
    mobileNavOpen.value = false
    await nextTick()
    accountMenu.value?.querySelector<HTMLElement>('[role="menuitem"]')?.focus()
  }
}

async function handleAccountKeydown(event: KeyboardEvent) {
  const keys = ['ArrowDown', 'ArrowUp', 'Home', 'End', 'Escape']
  if (!keys.includes(event.key)) return
  event.preventDefault()
  event.stopPropagation()
  if (event.key === 'Escape') {
    closeAccountMenu()
    accountMenu.value?.querySelector<HTMLButtonElement>('.account-trigger')?.focus()
    return
  }
  accountMenuOpen.value = true
  await nextTick()
  const items = Array.from(
    accountMenu.value?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? [],
  )
  const current = items.indexOf(document.activeElement as HTMLElement)
  const index =
    event.key === 'Home'
      ? 0
      : event.key === 'End'
        ? items.length - 1
        : event.key === 'ArrowUp'
          ? current <= 0
            ? items.length - 1
            : current - 1
          : (current + 1) % items.length
  items[index]?.focus()
}

function signIn() {
  saveLoginDrafts(null)
  void router.push(loginLocation(isLandingRoute.value ? '/dashboard' : route.fullPath))
}

function toggleMobileNav() {
  mobileNavOpen.value = !mobileNavOpen.value
  if (mobileNavOpen.value) closeAccountMenu()
}

function closeMobileNav() {
  mobileNavOpen.value = false
}

function handleDocumentPointerDown(event: PointerEvent) {
  if (!accountMenuOpen.value) return
  const target = event.target
  if (target instanceof Node && accountMenu.value?.contains(target)) return
  closeAccountMenu()
}

function handleDocumentKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape') {
    closeAccountMenu()
    closeMobileNav()
  }
}

async function logout() {
  closeAccountMenu()
  closeMobileNav()
  clearLoginDrafts()
  await auth.logout()
  router.push('/')
}

onMounted(() => {
  applyTheme(theme.value)
  applyStageChrome(isLandingRoute.value)
  auth.ensureLoaded()
  document.addEventListener('pointerdown', handleDocumentPointerDown)
  document.addEventListener('keydown', handleDocumentKeydown)
})

onBeforeUnmount(() => {
  applyStageChrome(false)
  document.removeEventListener('pointerdown', handleDocumentPointerDown)
  document.removeEventListener('keydown', handleDocumentKeydown)
})

watch(isLandingRoute, (onStage) => applyStageChrome(onStage))

watch(
  () => route.fullPath,
  () => {
    closeAccountMenu()
    closeMobileNav()
  },
)

watch(
  () => auth.user?.id,
  (userId, previousUserId) => {
    if (!userId) closeAccountMenu()
    if (previousUserId && userId !== previousUserId) agentChat.reset()
  },
)

watch(
  () => auth.user?.preferred_locale,
  (preferredLocale) => {
    if (preferredLocale) {
      applyLocale(normalizeLocale(preferredLocale))
    } else if (auth.loaded) {
      applyLocale('en')
    }
  },
)
</script>

<template>
  <div class="app-frame min-h-screen" :class="{ 'app-frame-chat': isAgentChatRoute }">
    <header class="app-header">
      <div
        class="app-header-inner"
        :class="{
          'app-header-inner--wide': isWideRoute,
          'app-header-inner--authless': isAuthRoute,
        }"
      >
        <!--
          Auth pages carry their own brand rail: repeating the lockup here would
          put the logo on screen twice. The rail is the poster; the header keeps
          only the utility controls.
        -->
        <RouterLink v-if="!isAuthRoute" to="/" class="brand-lockup">
          <span class="brand-mark">
            <img src="/logo-mark.png" alt="PepPlatform" />
          </span>
          <span class="brand-wordmark">PepPlatform</span>
        </RouterLink>
        <nav v-if="showPrimaryNav" class="nav-shell" :aria-label="t('nav.primary')">
          <RouterLink
            v-for="item in navItems"
            :key="item.to"
            :to="item.to"
            class="nav-link"
            :class="{ 'nav-link--active': isNavActive(item.to) }"
            :aria-current="isNavActive(item.to) ? 'page' : undefined"
          >
            <component :is="item.icon" aria-hidden="true" />
            <span>{{ item.label }}</span>
          </RouterLink>
        </nav>
        <div class="app-actions">
          <div
            v-if="auth.authEnabled && auth.user"
            ref="accountMenu"
            class="account-menu"
            @keydown="handleAccountKeydown"
            @focusout="handleAccountFocusOut"
          >
            <button
              class="account-trigger"
              type="button"
              aria-haspopup="menu"
              :aria-expanded="accountMenuOpen"
              aria-controls="account-menu"
              @click="toggleAccountMenu"
            >
              <span>{{ t('account.menu') }}</span>
              <span class="account-chevron" aria-hidden="true" />
            </button>
            <div v-if="accountMenuOpen" id="account-menu" class="account-popover" role="menu">
              <div class="account-summary">
                <span class="account-summary-text">
                  <span class="account-name">{{ accountDisplayName }}</span>
                  <span class="account-email">{{ auth.user.email }}</span>
                  <span class="account-meta">{{ accountMeta }}</span>
                </span>
              </div>
              <RouterLink
                class="account-menu-action account-menu-link"
                to="/account/profile"
                role="menuitem"
              >
                {{ t('account.profile') }}
              </RouterLink>
              <RouterLink
                class="account-menu-action account-menu-link"
                to="/account/ai-services"
                role="menuitem"
              >
                {{ t('account.aiServices') }}
              </RouterLink>
              <RouterLink
                v-if="isAdmin"
                class="account-menu-action account-menu-link"
                to="/admin"
                role="menuitem"
              >
                {{ t('nav.admin') }}
              </RouterLink>
              <button class="account-menu-action" type="button" role="menuitem" @click="logout">
                {{ t('account.logout') }}
              </button>
            </div>
          </div>
          <button
            v-else-if="auth.authEnabled && !isAuthRoute"
            type="button"
            class="secondary-action compact-action"
            @click="signIn"
          >
            <LogIn aria-hidden="true" />
            {{ t('account.login') }}
          </button>
          <button
            class="language-toggle"
            type="button"
            :disabled="localeSaving"
            :aria-label="t('language.switchLanguage')"
            :title="t('language.switchLanguage')"
            @click="toggleLocale"
          >
            {{ locale === 'zh' ? 'EN' : '中文' }}
          </button>
          <button
            v-if="!isLandingRoute"
            class="theme-toggle"
            type="button"
            :aria-label="themeToggleLabel"
            :aria-pressed="theme === 'dark'"
            :title="themeToggleLabel"
            @click="toggleTheme"
          >
            <svg
              v-if="theme === 'dark'"
              aria-hidden="true"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <circle cx="12" cy="12" r="4" />
              <path
                d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"
              />
            </svg>
            <svg
              v-else
              aria-hidden="true"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M20.99 12.42A8.4 8.4 0 1 1 11.58 3a6.6 6.6 0 0 0 9.41 9.42Z" />
            </svg>
          </button>
          <button
            v-if="showPrimaryNav"
            class="mobile-nav-toggle"
            type="button"
            :aria-label="mobileNavOpen ? t('nav.closeMenu') : t('nav.openMenu')"
            :aria-expanded="mobileNavOpen"
            aria-controls="mobile-nav"
            :title="mobileNavOpen ? t('nav.closeMenu') : t('nav.openMenu')"
            @click="toggleMobileNav"
          >
            <X v-if="mobileNavOpen" aria-hidden="true" />
            <MenuIcon v-else aria-hidden="true" />
          </button>
        </div>
      </div>
      <nav
        v-if="showPrimaryNav && mobileNavOpen"
        id="mobile-nav"
        class="mobile-nav-panel"
        :aria-label="t('nav.primary')"
      >
        <RouterLink
          v-for="item in navItems"
          :key="item.to"
          :to="item.to"
          class="nav-link"
          :class="{ 'nav-link--active': isNavActive(item.to) }"
          :aria-current="isNavActive(item.to) ? 'page' : undefined"
        >
          <component :is="item.icon" aria-hidden="true" />
          <span>{{ item.label }}</span>
        </RouterLink>
      </nav>
    </header>
    <main class="app-main w-full" :class="mainClass">
      <GuestNotice v-if="route.meta.guestPreview" />
      <PlatformNotices v-if="auth.user && !isAuthRoute && !isLandingRoute" />
      <RouterView
        v-if="canRenderRoute"
        :key="auth.user?.id ?? (auth.authEnabled ? 'guest' : 'local')"
      />
    </main>
  </div>
</template>
