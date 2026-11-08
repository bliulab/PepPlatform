import type { QuotaKey, UserQuota } from './types'
import type { QuotaChanges } from './resourceTypes'

export const dailyQuotaFields: QuotaKey[] = [
  'daily_experiment_limit',
  'daily_inference_request_limit',
  'daily_inference_input_limit',
  'daily_training_seconds_limit',
]

export function pendingQuotaUpdates(quota: UserQuota): Partial<Record<QuotaKey, number | null>> {
  if (quota.enforcement === 'exempt_admin') return {}
  return Object.fromEntries(
    dailyQuotaFields.flatMap((key) => {
      const pending = quota.pending?.[key]
      if (!pending) return []
      const personal = quota.sources?.[key] === 'personal' || key in (quota.overrides ?? {})
      return [[key, personal ? pending.value : null]]
    }),
  )
}

export function pendingQuotaChanges(quota: UserQuota | null | undefined): QuotaChanges {
  if (!quota || quota.enforcement === 'exempt_admin') return {}
  return Object.fromEntries(
    Object.entries(quota.pending ?? {}).map(([key, change]) => [
      key,
      {
        from: quota[key as QuotaKey] ?? 0,
        to: change.value,
        effective_at: change.effective_at,
      },
    ]),
  )
}

export const quotaFields: { key: QuotaKey; divisor: number; unit?: string; minimum: number }[] = [
  { key: 'daily_experiment_limit', divisor: 1, minimum: 0 },
  { key: 'daily_concurrent_experiment_limit', divisor: 1, minimum: 0 },
  { key: 'daily_training_seconds_limit', divisor: 60, unit: 'min', minimum: 0 },
  { key: 'max_experiment_seconds', divisor: 60, unit: 'min', minimum: 0 },
  { key: 'daily_inference_request_limit', divisor: 1, minimum: 0 },
  { key: 'daily_inference_input_limit', divisor: 1, minimum: 0 },
  { key: 'max_model_file_bytes', divisor: 1024 ** 3, unit: 'GiB', minimum: 0 },
  { key: 'max_model_bundle_bytes', divisor: 1024 ** 3, unit: 'GiB', minimum: 0 },
  { key: 'storage_limit_bytes', divisor: 1024 ** 3, unit: 'GiB', minimum: 0 },
  { key: 'max_model_count', divisor: 1, minimum: 0 },
  { key: 'model_retention_days', divisor: 1, unit: 'days', minimum: 1 },
  { key: 'protected_model_limit', divisor: 1, minimum: 0 },
]

export function quotaValue(value: number, divisor: number, minimum = 0) {
  if (typeof value !== 'number') return null
  const encoded = Number(value) * divisor
  return Number.isFinite(encoded) &&
    Number.isSafeInteger(Math.round(encoded)) &&
    (divisor > 1 || Math.abs(encoded - Math.round(encoded)) < 0.00001) &&
    (encoded === 0 || Math.round(encoded) > 0) &&
    encoded >= minimum
    ? Math.round(encoded)
    : null
}

export function formatResourceBytes(value: number) {
  if (!Number.isFinite(value)) return '—'
  const unit = value >= 1024 ** 3 ? 'GiB' : value >= 1024 ** 2 ? 'MiB' : 'KiB'
  const divisor = unit === 'GiB' ? 1024 ** 3 : unit === 'MiB' ? 1024 ** 2 : 1024
  return `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value / divisor)} ${unit}`
}

export const resourceMessages = {
  zh: {
    unlimited: '不限',
    usedUnlimited: '已用 {used} · 不限',
    adminUsage: '管理员用量',
    adminExempt:
      '管理员豁免个人额度，继续记录实际用量；平台磁盘、队列与计算调度保护仍然生效。模型清理由独立策略管理。',
    dailyChangeHint:
      '每日次数及累计训练时长下调默认于下一次 UTC 重置生效，可选择强制立即生效；上调立即生效。已接收任务沿用接收时额度。',
    reductionTiming: '每日额度下调生效方式',
    nextResetMode: '下次 UTC 重置生效（默认）',
    forceImmediateMode: '强制立即生效（仅限新提交）',
    effectiveImmediately: '立即生效',
    applyPendingNow: '立即应用待生效额度',
    pendingEditHint: '请先保存或撤销当前编辑，再提前应用待生效额度。',
    pendingTitle: '下次重置后的额度',
    quotaChangeTitle: '额度调整通知',
    quotaChangeHint:
      '历史使用记录保留。已有模型和保护继续有效，新额度限制后续新增；已接收任务沿用接收时额度。',
    changeValues: '{before} → {after}',
    effectiveAt: '{time} 生效',
    impactSummary:
      '立即影响 {immediate} 人 · 次日影响 {deferred} 人 · 新增受限 {blocked} 人 · 管理员豁免 {exempt} 人',
    publishQuotaAnnouncement: '发布本次用户资源限制变更公告',
    quotaAnnouncementHint:
      '默认不发布。实验与推理次数、在途实验数、单次及每日累计训练时长、存储、模型数量、保留天数和保护名额等用户资源限制，均由此开关控制；配置、生效时间和后台变更记录照常保存。',
    platformAnnouncementHint:
      '队列容量、磁盘保护和邮件发送上限等平台级设置保存时不自动发布公告；清理公告由清理设置单独控制。',
    announcementCount: '本次将向 {count} 位用户发布公告',
    publishAutomaticCleanupAnnouncements: '为后台自动新建的清理计划发布公告',
    publishCleanupAnnouncements: '为本次新建的清理计划发布公告',
    cleanupAnnouncementHint:
      '为新清理计划发送预告，并按公告提前量执行；已有公告继续遵守原截止时间。手动清理可独立选择本次是否推送。',
    silentCleanupHint:
      '静默清理会直接处理达到闲置保留期限、且通过保护和依赖检查的资源，不额外等待。已有公告继续遵守原截止时间。',
    manualCleanupNoticeHint:
      '本次为新清理计划发送预告，并按资源策略中的公告提前量执行。已有静默计划可直接执行，已有公告继续遵守原截止时间。',
    policies: '资源策略',
    storage: '存储与清理',
    announcements: '公告',
    policyTitle: '把资源留给每一次有效实验',
    policySubtitle: '统一设置默认额度，按需调整个人配置，管理存储与计算资源。',
    defaults: '用户默认额度',
    global: '继承全局',
    personal: '个人设置',
    inherit: '使用全局默认',
    allowGlobal: '允许全局配置覆盖个人设置',
    allowGlobalHint: '批量同步时允许更新这个账号的个人额度；整机资源保护始终生效。',
    limits: '平台保护',
    defaultOnly: '仅保存全局默认',
    applyUsers: '同步已有个人配置',
    selectFields: '勾选需要修改或批量同步的额度；未选字段保持当前设置。',
    preview: '预览影响',
    save: '保存策略',
    saved: '设置已保存。',
    affected: '将更新 {count} 位用户的个人配置',
    skipped: '跳过 {count} 位受保护用户',
    previewRequired: '保存前先预览本次变更；修改内容后需重新预览。',
    protectedHint:
      '已保护的模型不会参与定期清理。其他模型按闲置保留期限处理，已有公告遵守原截止时间。',
    zeroHint: '配额为 0 表示禁用对应能力；保留天数至少为 1。时间不包含排队。',
    dailyRegistration: '每日公开注册上限',
    dailyVerificationEmails: '每日账号验证码邮件上限',
    emailLimitHint:
      '注册验证、重发、找回密码及管理员重发共用上限，发送失败也计入；0 表示暂停发送。每天 UTC 00:00（北京时间 08:00）重置。邮件额度耗尽后暂停新注册，与注册人数上限独立执行。',
    emailUsage: '当前已保存策略：今日已用 {used} / {limit} 次发送，剩余 {remaining} 次',
    emailReset: '下次重置：{time}',
    emailPaused: '管理员已暂停验证码邮件发送，调整上限后恢复。',
    refreshEmailUsage: '刷新邮件用量',
    emailCountingHint: '按本功能启用后的发送尝试统计，不包含启用前历史发信量。',
    queue: '排队任务上限',
    diskReserve: '磁盘安全余量（GiB）',
    diskReservePercent: '磁盘安全余量（%）',
    diskCritical: '磁盘停止写入余量（GiB）',
    diskCriticalPercent: '磁盘停止写入余量（%）',
    inferenceTimeout: '推理最长时间（秒）',
    allowedModels: '允许的预训练模型（每行一个标识或本地路径）',
    retention: '自动清理',
    enableRetention: '启用自动清理',
    noticeHours: '清理公告提前量（小时，至少 72）',
    idleHours: '无引用临时目录闲置期限（小时）',
    cleanupHint:
      '缓存清理与模型清理分别启用。静默模式下，符合条件的到期资源可直接清理；选择发送预告时按公告时间执行。',
    storageSubtitle: '查看整机空间、用户用量和定期清理结果。',
    gpuMemory: '设备显存占用',
    gpuUtilization: 'GPU 利用率',
    free: '磁盘可用',
    retained: '系统与保留文件',
    reclaimable: '可回收副本',
    queued: '排队',
    running: '运行',
    registrations: '今日注册',
    scan: '重新盘点',
    checkRetention: '清理到期资源',
    ready: '可接受任务',
    paused: '暂停新增',
    critical: '停止增长任务',
    name: '资源',
    kind: '类型',
    size: '逻辑大小',
    state: '状态',
    deleteAfter: '清理时间',
    protected: '受保护',
    available: '待评估',
    cleaning: '正在清理',
    scheduled: '待执行',
    immediateCleanup: '符合保留与保护条件即可清理',
    notified: '已通知',
    failed: '清理失败',
    deleted: '已清理',
    cleanup: '执行到期清理',
    empty: '暂无资源',
    recentCleanup: '最近清理',
    released: '观察到释放空间',
    userStorage: '我的存储',
    trainingTime: '今日训练用时',
    usedOf: '{used} / {limit}',
    refresh: '刷新',
    notices: '平台通知',
    read: '已读',
    markRead: '标为已读',
    announcementTitle: '公告管理',
    announcementSubtitle: '发布服务说明、资源政策和维护通知。',
    title: '标题',
    body: '正文',
    audience: '接收范围',
    allUsers: '所有用户',
    selectedUsers: '指定用户',
    pinned: '置顶',
    publishAt: '开始时间',
    expiresAt: '结束时间（可留空）',
    publish: '发布公告',
    remove: '删除',
    edit: '编辑',
    cancel: '取消',
    manual: '管理员公告',
    cleanupNotice: '清理通知',
    noNotices: '暂无公告',
    audienceHint: '从用户列表搜索并选择接收人。',
    download: '下载权重与配置',
    pin: '保护模型',
    unpin: '取消保护',
    modelRetained: '按保留规则管理',
    modelProtectionHint: '已保护模型不参与定期清理；普通账号受保护额度限制，管理员自有模型不限。',
    modelQuota: '模型额度',
    modelLimitHint: '可保护数量不得超过模型保留总数；模型额度不会每日重置。',
    modelLimitsInvalid: '可保护模型数不能大于模型保留总数上限，请一起调整这两项配置。',
    estimatedSize: '预计可回收逻辑大小',
    retainedModels: '保留模型',
    protectedModels: '保护模型',
    modelRemaining: '还可保存 {count} 个',
    protectionRemaining: '还可保护 {count} 个',
    modelReserved: '排队／运行任务占位 {count} 个',
    modelQuotaHint:
      '模型额度不会每日重置。删除或定期清理模型后释放名额；平台共享模型不占个人额度。',
    modelOverLimit:
      '当前上限低于已有模型与任务占位数量，暂不可新增。已有模型保留，释放足够名额后可继续新增。',
    protectionOverLimit:
      '当前保护上限低于已有保护数量，现有保护继续有效，释放足够名额后可新增保护。',
    unprotectedModel: '未保护',
    deleteModels: '允许定期清理未保护模型及其权重',
    modelCleanup: '模型清理预览',
    modelCleanupOff: '模型清理尚未启用。预览不会删除模型；请先在资源策略中设置并启用。',
    modelCleanupNotice:
      '静默模式下，到期且符合条件的模型可直接清理；已有清理公告遵守原截止时间。预览不删除文件，共享文件可能继续保留，实际释放空间以清理记录为准。',
    modelUnavailable: '模型不可用',
    selfHostHint: '具备计算和存储资源？欢迎自行部署 Web 服务，自主配置用户、额度与模型保留策略。',
    selfHostLink: '查看 Web 部署指南',
    error: '操作失败，请刷新后重试。',
    invalid: '请输入有效额度。',
    minute: '分钟',
    days: '天',
    errors: {
      disk_space_insufficient: '服务器磁盘安全余量不足，已暂停新增。请清理可回收副本或联系管理员。',
      inference_timeout: '推理已达到时间上限，计算进程已停止，请减少输入后重试。',
      inference_busy: '推理工作进程正在使用中，请稍后重试。',
      gpu_busy_training: 'GPU 正被其他任务使用，请等待当前任务结束后重试。',
      gpu_reserved_for_training: 'GPU 已为排队中的训练任务预留，请稍后重试。',
      inference_worker_stopped: '推理工作进程已停止，请重试或联系管理员。',
      pretrained_model_not_allowed: '管理员尚未允许使用这个预训练模型。',
      resource_policy_conflict: '配置或用户设置已经变化，请刷新并重新预览。',
      invalid_resource_policy: '配置无效，请检查数值、磁盘阈值和接收范围。',
      resource_is_retained_or_in_use: '资源需要保留或正在使用，已跳过清理。',
      resource_used_after_notice: '资源在清理计划建立后被使用，请重新评估保留期限。',
      cleanup_notice_pending: '尚未到清理公告中的执行时间。',
      resource_cleanup_disabled: '资源清理尚未启用，请先在资源策略中启用。',
      retention_not_due: '资源尚未达到闲置保留期限，已跳过清理。',
      cleanup_notice_is_managed: '清理通知由保留策略管理。',
      quota_daily_training_seconds_limit:
        '今日累计训练时长已达到上限，请等待每日重置或联系管理员。',
      quota_storage_limit_bytes: '个人存储额度不足，请检查存储用量或联系管理员调整额度。',
      quota_daily_registration_limit:
        '很抱歉，受当前服务器存储与计算资源限制，今日注册名额已用完。您可在 {resetTime} 后再试。感谢您的理解与支持。',
      quota_daily_registration_limit_disabled:
        '很抱歉，受当前服务器存储与计算资源限制，平台目前暂停新用户注册。欢迎稍后关注开放情况，感谢您的理解与支持。',
      quota_daily_verification_email_limit:
        '平台今日验证码邮件发送额度已用完，暂时无法注册、重发验证码或找回密码。请在 {resetTime} 后重试，或联系管理员。已收到的有效验证码仍可使用。',
      quota_daily_verification_email_limit_disabled:
        '管理员已暂停验证码邮件发送，暂时无法注册、重发验证码或找回密码，请联系管理员。已收到的有效验证码仍可使用。',
      quota_daily_inference_request_limit:
        '今日推理次数已用完（已用 {used} / {limit}），请在 {resetTime} 后再试。',
      quota_daily_inference_input_limit:
        '本次提交 {requested} 条输入，今日剩余 {remaining} 条。请减少输入，或在 {resetTime} 后再试。',
      quota_daily_inference_request_limit_disabled:
        '当前账号暂未开放推理次数额度，请联系管理员调整。',
      quota_daily_inference_input_limit_disabled:
        '当前账号暂未开放推理输入额度，请联系管理员调整。',
      quota_max_model_count:
        '模型名额已用完（已保存或占位 {used} / {limit}）。请先删除不再需要的模型或等待定期清理，再创建新任务。',
      model_removed_by_retention:
        '此模型已按保留规则清理，无法继续推理或下载权重。历史记录仍然保留。',
      model_artifact_removed: '此模型的权重已清理，无法下载或重新加载。',
      model_cleanup_in_progress: '此模型正在清理或等待清理重试，暂时无法使用。',
      model_cleanup_disabled: '尚未启用模型清理，请先设置并预览清理策略。',
      model_cleanup_verification_failed:
        '模型或保留输入未通过完整性检查，已跳过清理。请管理员查看清理记录。',
      quota_queue_capacity: '平台任务队列已满，请稍后提交。',
      quota_max_experiment_seconds: '管理员已暂停此账号训练，请联系管理员调整单次运行额度。',
      quota_max_model_file_bytes: '单文件额度不足，无法启动本次任务。',
      quota_max_model_bundle_bytes: '模型产物额度不足，无法启动本次任务。',
      quota_protected_model_limit: '保护模型数已达上限，请先取消其他模型的保护。',
    },
    reason: {
      original_dataset: '原始数据集',
      reproducibility_dependency: '系统保留文件',
      retention_not_due: '尚未到保留期限',
      model_in_use: '模型正在使用',
      system_model: '平台共享模型',
      shared_or_system_retained: '共享或系统保留文件',
      immutable_version: '不可变版本',
      retained_input_view: '训练保留输入',
      rebuildable_copy: '可从版本重建',
      pinned_model: '用户保护',
      unverified_dependency: '尚未验证可重建',
      active_publication: '正在写入版本',
      abandoned_temporary: '无引用临时目录',
      abandoned_workspace: '无引用工作目录',
      platform_or_unclassified: '平台或未分类文件',
    },
    fields: {
      daily_experiment_limit: '每日训练次数',
      daily_concurrent_experiment_limit: '在途实验数（排队＋运行）',
      daily_training_seconds_limit: '每日累计训练时长',
      max_experiment_seconds: '单次训练最长时间',
      daily_inference_request_limit: '每日推理请求',
      daily_inference_input_limit: '每日推理输入条数',
      max_model_file_bytes: '单权重文件上限',
      max_model_bundle_bytes: '单模型产物总上限',
      storage_limit_bytes: '个人存储上限',
      model_retention_days: '模型闲置保留天数',
      max_model_count: '模型保留总数上限',
      protected_model_limit: '可保护模型数',
    },
  },
  en: {
    unlimited: 'Unlimited',
    usedUnlimited: '{used} used · Unlimited',
    adminUsage: 'Administrator usage',
    adminExempt:
      'Administrators are exempt from personal quotas. Usage is still recorded; platform disk, queue and scheduling safeguards still apply. Model cleanup follows its separate policy.',
    dailyChangeHint:
      'Lower daily counts and training runtime apply at the next UTC reset by default, or immediately when selected. Increases apply immediately. Accepted work keeps its admission limits.',
    reductionTiming: 'When daily quota reductions take effect',
    nextResetMode: 'Next UTC reset (default)',
    forceImmediateMode: 'Force immediately (new submissions only)',
    effectiveImmediately: 'Effective immediately',
    applyPendingNow: 'Apply pending reductions now',
    pendingEditHint: 'Save or discard your current edits before applying pending reductions early.',
    pendingTitle: 'Allowances after the next reset',
    quotaChangeTitle: 'Quota updated',
    quotaChangeHint:
      'Usage history is preserved. Existing models and protection remain; new limits govern additional work. Accepted work keeps its admission limits.',
    changeValues: '{before} → {after}',
    effectiveAt: 'Effective {time}',
    impactSummary:
      'Immediate: {immediate} · Next reset: {deferred} · New work restricted: {blocked} · Administrators exempt: {exempt}',
    publishQuotaAnnouncement: 'Publish an announcement about this change to user resource limits',
    quotaAnnouncementHint:
      'Off by default. This controls announcements for training and inference counts, in-flight experiments, per-run and daily training time, storage, model counts, retention and protection allowances. Settings, effective times and administrative change records are saved either way.',
    platformAnnouncementHint:
      'Saving platform settings such as queue capacity, disk protection and email limits does not automatically publish announcements. Cleanup announcements have separate controls.',
    announcementCount: 'Publish announcements to {count} users',
    publishAutomaticCleanupAnnouncements: 'Announce new cleanup plans created by background checks',
    publishCleanupAnnouncements: 'Announce new cleanup plans created by this check',
    cleanupAnnouncementHint:
      'Announce new cleanup plans and apply their configured lead time. Existing announcements keep their original deadlines. Manual cleanup chooses independently.',
    silentCleanupHint:
      'Silent cleanup directly removes resources past their idle retention period that pass protection and dependency checks, without an extra delay. Existing announcements keep their deadlines.',
    manualCleanupNoticeHint:
      'Announce new plans using the lead time in resource policies. Existing silent plans can run immediately; existing announcements keep their original deadlines.',
    policies: 'Resource policies',
    storage: 'Storage & cleanup',
    announcements: 'Announcements',
    policyTitle: 'Make room for each experiment',
    policySubtitle: 'Manage default quotas, personal settings, storage and computing resources.',
    defaults: 'Default user quotas',
    global: 'Global default',
    personal: 'Personal setting',
    inherit: 'Use global default',
    allowGlobal: 'Allow global policies to overwrite personal settings',
    allowGlobalHint:
      'Bulk synchronization can update this account. Platform resource protection always applies.',
    limits: 'Platform protection',
    defaultOnly: 'Update defaults only',
    applyUsers: 'Synchronize personal settings',
    selectFields:
      'Select the quotas to update or synchronize. Other fields keep their current settings.',
    preview: 'Preview changes',
    save: 'Save policy',
    saved: 'Settings saved.',
    affected: 'Update personal settings for {count} users',
    skipped: 'Skip {count} protected users',
    previewRequired: 'Preview before saving. Editing the draft requires another preview.',
    protectedHint:
      'Protected models are excluded from scheduled cleanup. Other models follow idle retention rules and honor existing announcement deadlines.',
    zeroHint:
      'Zero disables the corresponding capability. Retention must be at least one day. Queuing time is excluded.',
    dailyRegistration: 'Daily public registrations',
    dailyVerificationEmails: 'Daily account verification emails',
    emailLimitHint:
      'Registration, resends, password resets and administrator resends share this limit. Failed attempts count; zero pauses sending. Resets at 00:00 UTC (08:00 Beijing time). Exhaustion pauses registration independently of the registration count limit.',
    emailUsage: 'Saved policy: {used} / {limit} sending attempts today; {remaining} remaining',
    emailReset: 'Next reset: {time}',
    emailPaused: 'An administrator paused verification emails. Increase the limit to resume.',
    refreshEmailUsage: 'Refresh email usage',
    emailCountingHint: 'Counts attempts since this feature was enabled, excluding earlier email history.',
    queue: 'Queued task capacity',
    diskReserve: 'Disk safety reserve (GiB)',
    diskReservePercent: 'Disk safety reserve (%)',
    diskCritical: 'Disk stop-writing reserve (GiB)',
    diskCriticalPercent: 'Disk stop-writing reserve (%)',
    inferenceTimeout: 'Inference timeout (seconds)',
    allowedModels: 'Allowed pretrained models (one identifier or local path per line)',
    retention: 'Automatic cleanup',
    enableRetention: 'Enable automatic cleanup',
    noticeHours: 'Cleanup announcement lead time (hours, at least 72)',
    idleHours: 'Unreferenced temporary directory idle limit (hours)',
    cleanupHint:
      'Cache and model cleanup are enabled separately. Eligible expired resources can be cleaned directly in silent mode; announced plans follow their published deadlines.',
    storageSubtitle: 'Inspect volume capacity, user storage and scheduled cleanup results.',
    gpuMemory: 'Device GPU memory',
    gpuUtilization: 'GPU utilization',
    free: 'Disk available',
    retained: 'System & retained files',
    reclaimable: 'Rebuildable copies',
    queued: 'Queued',
    running: 'Running',
    registrations: 'Registered today',
    scan: 'Rescan storage',
    checkRetention: 'Clean up expired resources',
    ready: 'Accepting tasks',
    paused: 'New work paused',
    critical: 'Stopping growing tasks',
    name: 'Resource',
    kind: 'Type',
    size: 'Logical size',
    state: 'Status',
    deleteAfter: 'Cleanup after',
    protected: 'Protected',
    available: 'Awaiting assessment',
    cleaning: 'Cleaning up',
    scheduled: 'Pending execution',
    immediateCleanup: 'Cleanup can run once retention and protection checks pass',
    notified: 'Notified',
    failed: 'Cleanup failed',
    deleted: 'Removed',
    cleanup: 'Run due cleanup',
    empty: 'No resources',
    recentCleanup: 'Recent cleanup',
    released: 'Observed space released',
    userStorage: 'My storage',
    trainingTime: 'Training time today',
    usedOf: '{used} / {limit}',
    refresh: 'Refresh',
    notices: 'Platform notices',
    read: 'Read',
    markRead: 'Mark as read',
    announcementTitle: 'Announcements',
    announcementSubtitle: 'Publish service updates, resource policies and maintenance notices.',
    title: 'Title',
    body: 'Message',
    audience: 'Audience',
    allUsers: 'All users',
    selectedUsers: 'Selected users',
    pinned: 'Pin',
    publishAt: 'Starts at',
    expiresAt: 'Ends at (optional)',
    publish: 'Publish',
    remove: 'Delete',
    edit: 'Edit',
    cancel: 'Cancel',
    manual: 'Admin announcement',
    cleanupNotice: 'Cleanup notice',
    noNotices: 'No announcements',
    audienceHint: 'Search the user list and choose recipients.',
    download: 'Download weights & configuration',
    pin: 'Protect model',
    unpin: 'Remove protection',
    modelRetained: 'Managed by retention policy',
    modelProtectionHint:
      'Protected models are excluded from scheduled cleanup. Ordinary accounts use a protection allowance; administrator-owned models are unlimited.',
    modelQuota: 'Model allowance',
    modelLimitHint:
      'The protected model limit cannot exceed the saved model limit. Model allowances do not reset daily.',
    modelLimitsInvalid:
      'The protected model limit exceeds the saved model limit. Adjust both settings together.',
    estimatedSize: 'Estimated reclaimable size',
    retainedModels: 'Saved models',
    protectedModels: 'Protected models',
    modelRemaining: '{count} more models can be saved',
    protectionRemaining: '{count} more models can be protected',
    modelReserved: '{count} slots reserved by queued or running tasks',
    modelQuotaHint:
      'Model allowances do not reset daily. Deletion or scheduled cleanup releases slots. Platform models do not use personal slots.',
    modelOverLimit:
      'The current allowance is below existing models and task reservations. Existing models are retained; free enough slots to add more.',
    protectionOverLimit:
      'The current protection allowance is below existing protection. That protection remains; free enough slots to protect another model.',
    unprotectedModel: 'Unprotected',
    deleteModels: 'Allow scheduled removal of unprotected models and their weights',
    modelCleanup: 'Preview model cleanup',
    modelCleanupOff:
      'Model cleanup is disabled. Preview does not delete models; configure and enable the policy first.',
    modelCleanupNotice:
      'Eligible expired models can be cleaned directly in silent mode. Existing announcements keep their deadlines. Preview does not delete files; shared files may remain, and cleanup records show actual released space.',
    modelUnavailable: 'Model unavailable',
    selfHostHint:
      'Have computing and storage resources? Deploy your own Web service to configure users, allowances and model retention.',
    selfHostLink: 'View the Web deployment guide',
    error: 'The operation failed. Refresh and try again.',
    invalid: 'Enter a valid quota.',
    minute: 'minutes',
    days: 'days',
    errors: {
      disk_space_insufficient:
        'The server disk safety reserve is low. New work is paused; reclaim copies or contact an administrator.',
      inference_timeout:
        'Inference reached its time limit and the worker was stopped. Reduce the input and retry.',
      inference_busy: 'The inference worker is busy. Try again shortly.',
      gpu_busy_training: 'The GPU is in use. Retry after the current task finishes.',
      gpu_reserved_for_training: 'The GPU is reserved for queued training. Try again later.',
      inference_worker_stopped: 'The inference worker stopped. Retry or contact an administrator.',
      pretrained_model_not_allowed:
        'This pretrained model has not been approved by an administrator.',
      resource_policy_conflict: 'Policies or user settings changed. Refresh and preview again.',
      invalid_resource_policy: 'Check the values, disk thresholds and recipients.',
      resource_is_retained_or_in_use:
        'This resource must be retained or is in use. Cleanup was skipped.',
      resource_used_after_notice:
        'This resource was used after cleanup was scheduled. Reassess its retention period.',
      cleanup_notice_pending: 'The cleanup announcement deadline has not been reached.',
      resource_cleanup_disabled:
        'Resource cleanup is disabled. Enable it in resource policies first.',
      retention_not_due:
        'This resource has not reached its idle retention limit. Cleanup was skipped.',
      cleanup_notice_is_managed: 'Cleanup notices are managed by the retention policy.',
      quota_daily_training_seconds_limit:
        'Your daily training runtime is exhausted. Wait for reset or contact an administrator.',
      quota_storage_limit_bytes:
        'Your storage allowance is insufficient. Review usage or ask an administrator to adjust it.',
      quota_daily_registration_limit:
        'We are sorry, but today’s registration capacity is full because this server has limited computing and storage resources. Please try again after {resetTime}. Thank you for your understanding and support.',
      quota_daily_registration_limit_disabled:
        'We are sorry, but registration is currently paused because this server has limited computing and storage resources. Please check back later. Thank you for your understanding and support.',
      quota_daily_verification_email_limit:
        'The platform’s daily verification email allowance is exhausted. Registration, code resends and password resets are paused. Retry after {resetTime} or contact an administrator. Previously received valid codes still work.',
      quota_daily_verification_email_limit_disabled:
        'An administrator paused verification emails. Registration, code resends and password resets are unavailable; contact an administrator. Previously received valid codes still work.',
      quota_daily_inference_request_limit:
        'Your daily inference allowance is used up ({used} / {limit}). Please try again after {resetTime}.',
      quota_daily_inference_input_limit:
        'This request contains {requested} inputs, but {remaining} remain today. Reduce the input or try again after {resetTime}.',
      quota_daily_inference_request_limit_disabled:
        'Inference requests are currently disabled for this account. Contact an administrator.',
      quota_daily_inference_input_limit_disabled:
        'Inference inputs are currently disabled for this account. Contact an administrator.',
      quota_max_model_count:
        'All model slots are used or reserved ({used} / {limit}). Delete an unneeded model or wait for scheduled cleanup before creating a task.',
      model_removed_by_retention:
        'This model was removed under the retention policy. Inference and weight downloads are unavailable; historical records remain.',
      model_artifact_removed:
        'These model weights were removed and cannot be downloaded or loaded.',
      model_cleanup_in_progress:
        'This model is being cleaned up or awaiting a cleanup retry and is temporarily unavailable.',
      model_cleanup_disabled: 'Model cleanup is disabled. Configure and preview the policy first.',
      model_cleanup_verification_failed:
        'The model or retained inputs failed integrity checks. Cleanup was skipped; an administrator can inspect the cleanup log.',
      quota_queue_capacity: 'The platform queue is full. Submit again later.',
      quota_max_experiment_seconds:
        'Training is paused for this account. Ask an administrator to adjust the run duration allowance.',
      quota_max_model_file_bytes: 'The file allowance is insufficient to start this task.',
      quota_max_model_bundle_bytes:
        'The model artifact allowance is insufficient to start this task.',
      quota_protected_model_limit:
        'The protected model limit is reached. Remove protection from another model first.',
    },
    reason: {
      original_dataset: 'Original dataset',
      reproducibility_dependency: 'System-retained file',
      retention_not_due: 'Retention period has not expired',
      model_in_use: 'Model in use',
      system_model: 'Platform model',
      shared_or_system_retained: 'Shared or system-retained files',
      immutable_version: 'Immutable version',
      retained_input_view: 'Retained training input',
      rebuildable_copy: 'Rebuildable from version',
      pinned_model: 'User protection',
      unverified_dependency: 'Reconstruction not verified',
      active_publication: 'Active publication',
      abandoned_temporary: 'Unreferenced temporary directory',
      abandoned_workspace: 'Unreferenced workspace',
      platform_or_unclassified: 'Platform or unclassified files',
    },
    fields: {
      daily_experiment_limit: 'Daily training runs',
      daily_concurrent_experiment_limit: 'In-flight runs (queued + running)',
      daily_training_seconds_limit: 'Daily training runtime',
      max_experiment_seconds: 'Maximum run duration',
      daily_inference_request_limit: 'Daily inference requests',
      daily_inference_input_limit: 'Daily inference inputs',
      max_model_file_bytes: 'Maximum weight file',
      max_model_bundle_bytes: 'Maximum model artifacts',
      storage_limit_bytes: 'Personal storage limit',
      max_model_count: 'Maximum saved models',
      model_retention_days: 'Idle model retention',
      protected_model_limit: 'Maximum protected models',
    },
  },
}
