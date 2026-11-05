import { createRouter, createWebHistory } from 'vue-router'
import RootView from './views/RootView.vue'
import HomeView from './views/HomeView.vue'
import DatasetsView from './views/DatasetsView.vue'
import ExperimentsView from './views/ExperimentsView.vue'
import ExperimentWizardView from './views/ExperimentWizardView.vue'
import ExperimentDetailView from './views/ExperimentDetailView.vue'
import ExperimentAnalysisView from './views/ExperimentAnalysisView.vue'
import ExperimentWorkspaceView from './views/ExperimentWorkspaceView.vue'
import ExperimentCompareView from './views/ExperimentCompareView.vue'
import LeaderboardView from './views/LeaderboardView.vue'
import ModelsView from './views/ModelsView.vue'
import EnsembleWizardView from './views/EnsembleWizardView.vue'
import AgentView from './views/AgentView.vue'
import AgentWorkspaceView from './views/AgentWorkspaceView.vue'
import AgentChatView from './views/AgentChatView.vue'
import AgentPlansView from './views/AgentPlansView.vue'
import AgentCampaignView from './views/AgentCampaignView.vue'
import LoginView from './views/LoginView.vue'
import RegisterView from './views/RegisterView.vue'
import VerifyEmailView from './views/VerifyEmailView.vue'
import PasswordResetView from './views/PasswordResetView.vue'
import ProfileView from './views/ProfileView.vue'
import AiServicesView from './views/AiServicesView.vue'
import AdminView from './views/AdminView.vue'
import AdminOverviewView from './views/AdminOverviewView.vue'
import AdminUsersView from './views/AdminUsersView.vue'
import AdminUserDetailView from './views/AdminUserDetailView.vue'
import AdminAuditView from './views/AdminAuditView.vue'
import { useAuthStore } from './stores/auth'

import {
  allowsGuestPreview,
  authRoutePaths,
  loginLocation,
  postAuthRedirect,
} from './authNavigation'
import { saveLoginDrafts } from './loginDraft'

export const router = createRouter({
  history: createWebHistory(),
  scrollBehavior(to) {
    if (to.hash) return { el: to.hash, top: 80, behavior: 'smooth' }
    return { top: 0 }
  },
  routes: [
    { path: '/', component: RootView, meta: { public: true, stage: true } },
    {
      path: '/welcome',
      component: () => import('./views/LandingView.vue'),
      meta: { public: true, stage: true },
    },
    { path: '/dashboard', component: HomeView, meta: { guestPreview: true } },
    { path: '/datasets', component: DatasetsView, meta: { guestPreview: true } },
    { path: '/experiments', component: ExperimentsView, meta: { guestPreview: true } },
    { path: '/experiments/new', component: ExperimentWizardView, meta: { guestPreview: true } },
    {
      path: '/experiments/compare',
      component: ExperimentCompareView,
      meta: { guestPreview: true },
    },
    {
      path: '/experiments/:id',
      component: ExperimentWorkspaceView,
      props: true,
      children: [
        {
          path: 'visualizations',
          name: 'experiment-visualizations',
          component: () => import('./views/ExperimentVisualizationsView.vue'),
          props: true,
          meta: { experimentTab: 'visualizations', wide: true },
        },
        {
          path: '',
          name: 'experiment-overview',
          component: ExperimentDetailView,
          props: true,
          meta: { experimentTab: 'overview' },
        },
        {
          path: 'analysis/:conversationId?',
          name: 'experiment-analysis',
          component: ExperimentAnalysisView,
          props: true,
          meta: { experimentTab: 'analysis', wide: true },
        },
      ],
    },
    { path: '/leaderboard', component: LeaderboardView, meta: { guestPreview: true } },
    { path: '/models', component: ModelsView, meta: { guestPreview: true } },
    { path: '/ensembles/new', component: EnsembleWizardView, meta: { guestPreview: true } },
    {
      path: '/agent',
      component: AgentWorkspaceView,
      meta: { wide: true },
      children: [
        {
          path: '',
          redirect: (to) => ({ name: 'agent-design', query: to.query }),
        },
        {
          path: 'design',
          name: 'agent-design',
          component: AgentView,
          meta: { agentTab: 'design', wide: true, guestPreview: true },
        },
        {
          path: 'chat/:conversationId?',
          name: 'agent-chat',
          component: AgentChatView,
          meta: { agentTab: 'chat', wide: true, guestPreview: true },
        },
        {
          path: 'plans',
          component: AgentPlansView,
          meta: { agentTab: 'design', wide: true, guestPreview: true },
        },
        {
          path: 'campaigns/:id',
          component: AgentCampaignView,
          meta: { agentTab: 'design', wide: true },
        },
      ],
    },
    {
      path: '/admin',
      component: AdminView,
      meta: { admin: true, wide: true },
      children: [
        { path: '', redirect: { name: 'admin-overview' } },
        {
          path: 'overview',
          name: 'admin-overview',
          component: AdminOverviewView,
          meta: { adminSection: 'overview' },
        },
        {
          path: 'users',
          name: 'admin-users',
          component: AdminUsersView,
          meta: { adminSection: 'users' },
        },
        {
          path: 'users/:id',
          redirect: (to) => ({
            path: `/admin/users/${String(to.params.id)}/account`,
            query: to.query,
          }),
        },
        ...['account', 'usage', 'security', 'activity'].map((tab) => ({
          path: `users/:id/${tab}`,
          name: `admin-user-${tab}`,
          component: AdminUserDetailView,
          meta: { adminSection: 'users', adminUserTab: tab },
        })),
        {
          path: 'audit',
          name: 'admin-audit',
          component: AdminAuditView,
          meta: { adminSection: 'audit' },
        },
      ],
    },
    { path: '/account/profile', component: ProfileView },
    { path: '/account/ai-services', name: 'ai-services', component: AiServicesView },
    { path: '/login', component: LoginView, meta: { public: true } },
    { path: '/register', component: RegisterView, meta: { public: true } },
    { path: '/verify-email', component: VerifyEmailView, meta: { public: true } },
    { path: '/password-reset', component: PasswordResetView, meta: { public: true } },
  ],
})

router.beforeEach(async (to) => {
  const auth = useAuthStore()
  await auth.ensureLoaded()
  if (!auth.authEnabled) return true
  if (to.meta.public) {
    if (auth.user && authRoutePaths.has(to.path)) {
      return postAuthRedirect(to.query.redirect)
    }
    return true
  }
  if (!auth.user) {
    if (allowsGuestPreview(to)) return true
    saveLoginDrafts(null)
    return loginLocation(to.fullPath)
  }
  if (to.meta.admin && auth.user.role !== 'admin') {
    return { path: '/dashboard' }
  }
  return true
})
