import { afterEach, describe, expect, it, vi } from 'vitest'
import { watchStageVisibility } from './stageScene'

afterEach(() => vi.unstubAllGlobals())

describe('stage visibility without IntersectionObserver', () => {
  it('still pauses in the background and removes the listener on teardown', () => {
    const document = Object.assign(new EventTarget(), { hidden: false })
    vi.stubGlobal('document', document)
    vi.stubGlobal('IntersectionObserver', undefined)
    const onEnter = vi.fn()
    const onLeave = vi.fn()
    const cleanup = watchStageVisibility({} as HTMLElement, { onEnter, onLeave })
    expect(onEnter).toHaveBeenCalledTimes(1)
    document.hidden = true
    document.dispatchEvent(new Event('visibilitychange'))
    expect(onLeave).toHaveBeenCalledTimes(1)
    document.hidden = false
    document.dispatchEvent(new Event('visibilitychange'))
    expect(onEnter).toHaveBeenCalledTimes(2)
    cleanup()
    document.hidden = true
    document.dispatchEvent(new Event('visibilitychange'))
    expect(onLeave).toHaveBeenCalledTimes(1)
  })
})
