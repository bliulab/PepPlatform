import { createPinia, setActivePinia } from 'pinia'
import { effectScope } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  clearLoginDrafts,
  draftValues,
  saveLoginDrafts,
  takeLoginDraft,
  useLoginDraft,
} from './loginDraft'
import { useAuthStore } from './stores/auth'
import { useExperimentDraftStore } from './stores/experimentDraft'

vi.mock('vue-router', () => ({ useRoute: () => ({ fullPath: '/experiments/new' }) }))

const entries = new Map<string, string>()
beforeEach(() => {
  setActivePinia(createPinia())
  entries.clear()
  vi.stubGlobal('window', {
    sessionStorage: {
      getItem: (key: string) => entries.get(key) ?? null,
      setItem: (key: string, value: string) => entries.set(key, value),
      removeItem: (key: string) => entries.delete(key),
    },
  })
})
afterEach(() => {
  vi.unstubAllGlobals()
})

describe('tab-scoped login drafts', () => {
  it('keeps zero values and tokenizer options while excluding files and credentials', () => {
    expect(
      draftValues({
        count: 0,
        tokenizer: 'kmer',
        password: 'private',
        apiKey: 'private',
        files: ['private'],
        blob: new Blob(['private']),
        enabled: false,
      }),
    ).toEqual({ count: 0, tokenizer: 'kmer', enabled: false })
  })

  it('restores once, keeps an expired owner draft until sign-in, and rejects a different account', () => {
    const key = '/experiments/new:form'
    const scope = effectScope()
    scope.run(() =>
      useLoginDraft(
        'form',
        () => ({ name: 'research draft' }),
        () => {},
      ),
    )
    saveLoginDrafts('owner')
    expect(takeLoginDraft(key, null)).toBeUndefined()
    expect(takeLoginDraft(key, 'owner')).toEqual({ name: 'research draft' })
    expect(takeLoginDraft(key, 'owner')).toBeUndefined()
    saveLoginDrafts('owner')
    expect(takeLoginDraft(key, 'another-account')).toBeUndefined()
    saveLoginDrafts(null)
    clearLoginDrafts()
    expect(entries.size).toBe(0)
    scope.stop()
  })

  it('works without session storage and ignores malformed stored JSON', () => {
    entries.set('pepplatform:login-drafts:v1', '{broken')
    expect(takeLoginDraft('/experiments/new:form', null)).toBeUndefined()
    vi.stubGlobal('window', {
      get sessionStorage() {
        throw new Error('blocked')
      },
    })
    expect(() => saveLoginDrafts(null)).not.toThrow()
  })

  it.each(
    ['pep_function', 'pep_protein_interaction', 'pep_hlai', 'pep_hlaii'].flatMap((task) =>
      [false, true].map((semantic) => [task, semantic] as const),
    ),
  )(
    'preserves the serialized experiment configuration across sign-in: %s, semantic=%s',
    (task, semantic) => {
      const draft = useExperimentDraftStore()
      draft.resetForTask(task)
      draft.enableSemanticSimilarity = semantic
      draft.semanticLossType = 'jaccard_supcon'
      draft.semanticPositivePartners = 4
      draft.semanticZeroPartners = 0
      const options = { isPairTask: task !== 'pep_function', encoderFamily: 'sequence' }
      const before = structuredClone(draft.toConfig(options))
      const source = effectScope()
      source.run(() =>
        useLoginDraft(
          'experiment',
          () => ({ state: { ...draft.$state } }),
          () => {},
        ),
      )
      saveLoginDrafts(null)
      source.stop()
      draft.$reset()
      useAuthStore().user = { id: 'new-account' } as ReturnType<typeof useAuthStore>['user']
      const target = effectScope()
      target.run(() =>
        useLoginDraft(
          'experiment',
          () => ({ state: { ...draft.$state } }),
          (saved) => {
            draft.$patch((state) => Object.assign(state, saved.state))
          },
        ),
      )
      expect(draft.taskType).toBe(task)
      expect(draft.toConfig(options)).toEqual(before)
      expect(entries.size).toBe(0)
      target.stop()
    },
  )
})
