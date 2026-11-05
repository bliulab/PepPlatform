import { request, requestBlob } from './client'
import type { HeatmapCapabilities, HeatmapJob, HeatmapParams, HeatmapResult } from '../heatmaps'

function base(id: string) {
  return `/api/experiments/${encodeURIComponent(id)}/visualizations`
}
export const heatmapApi = {
  capabilities: (id: string) => request<HeatmapCapabilities>(`${base(id)}/capabilities`),
  submit: (id: string, params: HeatmapParams) =>
    request<HeatmapJob>(`${base(id)}/jobs`, { method: 'POST', body: JSON.stringify(params) }),
  job: (id: string, job: string) => request<HeatmapJob>(`${base(id)}/jobs/${job}`),
  cancel: (id: string, job: string) =>
    request<HeatmapJob>(`${base(id)}/jobs/${job}/cancel`, { method: 'POST' }),
  result: (id: string, job: string, row: number, column: number, search: string) =>
    request<HeatmapResult>(
      `${base(id)}/jobs/${job}/result?${new URLSearchParams({ row_offset: String(row), column_offset: String(column), search })}`,
    ),
  export: (id: string, job: string, format: 'csv' | 'json') =>
    requestBlob(`${base(id)}/jobs/${job}/export?format=${format}`),
}
