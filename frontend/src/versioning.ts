export interface ContentVersion {
  id: string
  kind: string
  role: string
  identity: Record<string, unknown>
  created_at: string
}

export interface RunVersions {
  run_id: string
  status: string
  versions: ContentVersion[]
  relations: { relation: string; target: string }[]
  details?: {
    checked_at?: string
    source_revision?: { git_commit: string | null; git_dirty: boolean | null }
    last_verification?: { status: string; checked_at: string }
  }
}

export const versionMessages = {
  zh: {
    title: '实验版本',
    hint: '固定本次运行使用的输入与生成的模型。',
    export: '导出版本清单',
    loading: '正在读取版本…',
    unavailable: '暂时无法读取版本，请重试。',
    retry: '重试',
    checked: '最近校验',
    related: '使用此版本的实验',
    inspect: '查看清单',
    config: '配置',
    effective_config: '最终有效配置',
    dataset: '数据集',
    split: '数据划分',
    source: '运行源码',
    model: '模型',
    checkpoint: 'Checkpoint',
    environment: '环境',
    more: '输入与依赖',
    lineage: '来源关系',
    dirty: '包含未提交源码',
    unknown: '未知',
    not_recorded: '此实验未记录统一版本；不会从当前文件补认证历史状态。',
    preparing: '准备快照',
    running: '输入已固定',
    sealed: '产物已封存',
    failed: '训练失败，保留已采集的证据',
    canceled: '已取消，保留已采集的证据',
    seal_failed: '产物封存失败，模型不可用于推理',
    no_model: '未生成可登记模型',
    incomplete: '版本记录不完整',
    verified: '校验通过',
    invalid: '校验未通过',
    rerun_of: '重跑自',
    reuses_representation_from: '复用表示自',
    ensemble_member: '集成成员',
  },
  en: {
    title: 'Experiment versions',
    hint: 'Pinned inputs and retained model artifacts for this run.',
    export: 'Export version manifest',
    loading: 'Loading versions…',
    unavailable: 'Versions are unavailable. Please retry.',
    retry: 'Retry',
    checked: 'Last checked',
    related: 'Runs using this version',
    inspect: 'Inspect manifest',
    config: 'Configuration',
    effective_config: 'Effective configuration',
    dataset: 'Dataset',
    split: 'Split',
    source: 'Runtime source',
    model: 'Model',
    checkpoint: 'Checkpoint',
    environment: 'Environment',
    more: 'Inputs and dependencies',
    lineage: 'Lineage',
    dirty: 'Includes uncommitted source',
    unknown: 'Unknown',
    not_recorded:
      'This run has no unified version record. Current files cannot certify its historical state.',
    preparing: 'Preparing snapshots',
    running: 'Inputs pinned',
    sealed: 'Artifacts sealed',
    failed: 'Training failed; captured evidence retained',
    canceled: 'Canceled; captured evidence retained',
    seal_failed: 'Artifact sealing failed; inference disabled',
    no_model: 'No model artifact registered',
    incomplete: 'Incomplete version record',
    verified: 'Verified',
    invalid: 'Verification failed',
    rerun_of: 'Rerun of',
    reuses_representation_from: 'Reuses representation from',
    ensemble_member: 'Ensemble member',
  },
}

export function shortVersion(identifier: string): string {
  const [kind, digest] = identifier.split(':')
  return digest ? `${kind}:${digest.slice(0, 12)}` : identifier
}
