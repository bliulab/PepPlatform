import { defineStore } from 'pinia'
import { ApiError, api } from '../api/client'
import type { SupportedLocale } from '../localization'
import type { AdminResourceCounts, AuthResponse, AuthUser, QuotaUsage, UserQuota } from '../types'

let currentUserRequest: Promise<void> | null = null
let sessionRevision = 0
let currentUserRequestId = 0

export const useAuthStore = defineStore('auth', {
  state: () => ({
    loaded: false,
    loading: false,
    authEnabled: true,
    user: null as AuthUser | null,
    quota: null as UserQuota | null,
    usage: null as QuotaUsage | null,
    resourceCounts: null as AdminResourceCounts | null,
    usageUnavailable: false,
  }),
  actions: {
    clearCurrentUser() {
      sessionRevision += 1
      currentUserRequest = null
      this.loading = false
      this.loaded = true
      this.user = null
      this.quota = null
      this.usage = null
      this.resourceCounts = null
      this.usageUnavailable = false
    },
    applyAuthResponse(response: AuthResponse) {
      this.authEnabled = response.auth_enabled
      this.user = response.user
      this.quota = response.quota ?? null
      this.usage = response.usage ?? null
      this.resourceCounts = response.resource_counts ?? null
      this.usageUnavailable = false
    },
    async loadCurrentUser() {
      if (currentUserRequest) {
        await currentUserRequest
        return
      }
      this.loading = true
      const revision = sessionRevision
      const requestId = ++currentUserRequestId
      const request = (async () => {
        try {
          const response = await api.me()
          if (revision === sessionRevision) this.applyAuthResponse(response)
        } catch (error) {
          if (revision !== sessionRevision) return
          if (error instanceof ApiError && error.status === 401) {
            this.authEnabled = true
            this.clearCurrentUser()
          } else {
            throw error
          }
        } finally {
          if (revision === sessionRevision) {
            this.loaded = true
            this.loading = false
          }
          if (currentUserRequestId === requestId) currentUserRequest = null
        }
      })()
      currentUserRequest = request
      await request
    },
    async refreshUsage() {
      if (!this.authEnabled || !this.user) return
      const revision = sessionRevision
      try {
        await this.loadCurrentUser()
      } catch {
        if (revision === sessionRevision) this.usageUnavailable = true
      }
    },
    async ensureLoaded() {
      if (!this.loaded) {
        await this.loadCurrentUser()
      }
    },
    async login(email: string, password: string) {
      sessionRevision += 1
      currentUserRequest = null
      this.loading = false
      const response = await api.login({ email, password })
      this.applyAuthResponse(response)
      this.loaded = true
      return response
    },
    async verifyEmail(email: string, code: string) {
      sessionRevision += 1
      currentUserRequest = null
      this.loading = false
      const response = await api.verifyEmail({ email, code })
      this.applyAuthResponse(response)
      this.loaded = true
      return response
    },
    async updateProfile(displayName: string) {
      const response = await api.updateProfile({ display_name: displayName })
      this.applyAuthResponse(response)
      this.loaded = true
      return response
    },
    async updatePreferredLocale(preferredLocale: SupportedLocale) {
      const response = await api.updatePreferences({ preferred_locale: preferredLocale })
      this.applyAuthResponse(response)
      this.loaded = true
      return response
    },
    async logout() {
      await api.logout()
      this.clearCurrentUser()
      this.loaded = true
    },
  },
})
