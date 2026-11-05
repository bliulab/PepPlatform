/// <reference types="vite/client" />

declare module 'echarts-gl/charts' {
  import type { EChartsExtensionInstaller } from 'echarts/types/src/extension'

  export const Scatter3DChart: EChartsExtensionInstaller
}

declare module 'echarts-gl/components' {
  import type { EChartsExtensionInstaller } from 'echarts/types/src/extension'

  export const Grid3DComponent: EChartsExtensionInstaller
}
