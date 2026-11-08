# Administrative resource governance

The admin console provides **Resource policies**, **Storage & cleanup** and **Announcements**. Users see model allowances in the model library and account page, protect models from each model card, and download bundles. Administrators choose whether to publish quota and cleanup announcements; publication defaults to off.

## Self-hosting

Our public service has limited computing and storage resources. If you have suitable resources, we encourage you to [deploy your own Web service](deployment.md). You control users, quotas, model retention and maintenance schedules.

## Account-code email budget

The daily account verification email limit defaults to **20**; zero pauses sending. Registration, user resends, password resets and administrator resends share the budget, without an administrator exemption. It resets at 00:00 UTC (08:00 Beijing time), independently of registration count limits. This platform limit is not a guarantee of the mail provider's capacity.

Each SMTP attempt is reserved atomically in a short database transaction before connecting. Failed, timed-out or interrupted attempts are not refunded. Restarting or deleting accounts/codes cannot reset usage. Development codes without configured SMTP do not consume attempts. Accounting begins when the feature is enabled, without inferring historical sends; messages are not queued for later delivery.

Administrators edit the nonnegative integer `limits.daily_verification_email_limit` through resource policies; changes immediately affect subsequent requests without clearing usage. The administrator-only `GET /api/admin/email-delivery-usage` reports date, usage, limit, remaining budget and reset time. Exhaustion returns HTTP 429 with `quota_type=daily_verification_email_limit`, `reset_at` and `Retry-After`. Zero indicates an administrative pause, without promising recovery the next day. Rejected registrations leave no account or registration charge; rejected resends preserve previously valid codes. Existing codes and ordinary sign-in remain available.

The announcement checkbox now describes changes to user resource limits, including in-flight experiments and per-run/daily training time. It defaults to off. Saving platform settings such as queue, disk protection or email limits records audits without automatically publishing user announcements; cleanup controls are separate.

Migration `20261108_0017` adds only the independent email counter table and supports direct initialization. Existing policies without the new setting use 20. Email settings and counters never enter scientific configurations, training commands, admission snapshots or version identities.

## Model allowances

**Saved model limit** and **protected model limit** are separate settings. Fresh installations default to **5 saved models**, of which up to **3 can be protected**. Protected models count toward the saved total. The protected limit cannot exceed the saved limit; the server validates both personal settings and effective settings after a global change.

These allowances do not reset daily. Queued and running training or ensemble tasks reserve model slots. When no slot remains, new tasks are rejected before computing starts; successful publication converts the reservation into a saved model. Failure or cancellation releases the reservation. Deleting a model or completing scheduled removal frees its slot. Platform models without an owner do not use personal slots.

Lowering a limit does not delete existing models or revoke existing protection. The user sees the excess and cannot add models or protection until usage permits it. Removing protection releases a protection slot only; it does not remove the saved model. Download a backup before a model's removal deadline if you need its weights later.

## Defaults and individual settings

Administrators are exempt from personal counts, concurrency, runtime, storage, file/artifact sizes, and saved/protected model limits. Usage is still recorded and shown with an unlimited allowance. Platform disk, queue, GPU scheduling, inference timeout and approved-model rules still apply; idle model cleanup retains its separate policy. Operations on another user's resources use that owner's allowance.

Unset fields inherit global defaults. The personal quota API accepts `null` to restore inheritance; zero disables a quota-controlled capability, while retention days must be positive. Existing quota rows remain personal settings protected from global replacement.

Administrators can allow global replacement per user. Global edits either update defaults only or synchronize selected fields for eligible users. Preview tokens bind the policy revision, selected fields, reduction timing, synchronization mode, UTC date and user settings; stale previews are rejected. Changing timing requires a new preview. Restricted-user counts reflect usage and reservations at preview time. Synchronization preserves unselected fields.

Lower daily experiment counts, inference requests/inputs and training runtime take effect at the next UTC midnight by default. Administrators can instead choose **Force immediately (new submissions only)**. Increases apply immediately. The default mode preserves today's granted allowance; immediate mode replaces the selected fields' current allowances with their configured targets. Tomorrow uses the latest target in either mode. Personal edits, inheritance and synchronization follow these rules while retaining personal replacement protections. New accounts start with the configured target. Other quotas retain their existing behavior.

The personal quota PATCH, global preview POST and global policy PATCH accept optional `daily_reduction_mode`: `next_reset` (default) or `immediate`. Invalid values return HTTP 400. This controls only the daily fields included in the operation and is not persisted as a quota or scientific configuration. Pending reductions can be applied early without changing their values: the personal **Apply pending reductions now** action preserves each field's inheritance; global changes apply selected fields after preview. Unselected pending fields remain deferred.

Reductions never clear, truncate or refund usage or reservations. For example, forcing an allowance from 10 to 5 after 8 uses retains all 8 uses, shows zero remaining and restricts subsequent submissions. Accepted work keeps its original snapshots. Numeric quota fields describe current allowances; `configured` contains targets, `pending` describes scheduled values/times, and `enforcement` identifies administrator exemption. The UI shows local times. Actual changes, including early application of an unchanged pending target, always retain their timing and effective changes in administrative records.

The personal quota PATCH, global preview POST and global policy PATCH accept optional boolean `publish_announcements`, defaulting to `false`; invalid types return HTTP 400. Each save or early-application action chooses independently, and the UI resets the choice after success. Announcements target only users with actual quota changes; unchanged or repeated operations do not generate notices. Global previews report the announcement count and bind the choice into their token, so changing it requires another preview. This option does not affect allowances, timing, usage or accepted admission snapshots, and never enters scientific configurations. Audit logs retain the choice. When requested, announcements commit or roll back with the change. Existing announcements and read receipts remain; nothing is announced retroactively or withdrawn automatically.

Lowering a saved limit from five to two retains all five models. A new model can be admitted only when saved models plus reservations total fewer than two. Existing protection also remains after a reduction. Capacity changes restrict new work while preserving legitimate usage history; the UI explains the allowance and current usage without treating historical use as a violation.

Experiments, ensembles, Agent batches, resumed training and inference capture admission limits transactionally. Accepted queued/running work uses those limits during execution, writes and publication; subsequent submissions use current allowances. Platform safeguards remain active and interrupted work stays incomplete. Admission snapshots live outside scientific configurations and content identities.

Other fresh defaults are 10 runs per UTC day, one in-flight run including queued work, 7200 seconds of daily training, 3600 seconds per run, 200 inference requests / 10000 inputs per day, 1 GiB per output file, 2 GiB per model artifact directory, 10 GiB of personal logical storage and 14 idle days for models. Public registration defaults to 10 accounts per UTC day; queue capacity is 100 and inference timeout is 300 seconds. Upgrade preserves configured values, including existing protection limits; a new saved limit is raised if needed to accommodate those settings.

Successful submission consumes a daily training count; deletion does not refund it. Queuing time is excluded from runtime; preparation, execution and publication count. Failed or canceled runs retain consumed runtime, split across UTC midnight. Inference reserves allowance before execution and settles once a result is stored. Request allowance, input allowance, GPU contention and request rate limiting have distinct messages. Positive daily quota errors include the next UTC reset time, shown in the user's local timezone. Zero means disabled and does not promise a reset. Successful public account creation counts even before email verification; bootstrap admin creation does not.

## Execution and storage

One dispatcher per runtime directory processes a durable database queue with user rotation. Recovery runs only in the leader. GPU training and GPU inference share an interprocess lease. Pure CPU classical models and their ensembles use an independent inference process. Timeouts terminate the owned worker, and warm GPU inference workers are released before training.

Resource settings live outside experiment configurations. They do not rewrite commands, seeds, splits, epochs, batch size, precision, semantic options, selection or evaluation. Resource-limited runs are rejected, queued or marked incomplete rather than silently reduced or presented as completed results. The CLI and version identity algorithms retain their behavior.

PostgreSQL row locks and SQLite IMMEDIATE transactions keep admission, batch creation, accounting and durable submission atomic. Registration email delivery runs after commit. Individual storage accounts for owned logical file sizes and retained dependencies; the same content object is charged once per user. Physical free space is shown separately, including CoW effects.

Uploads, snapshots and materializations reserve space before writing. Completed writes remain charged until inventory catches up. By default, free space below `max(10 GiB, 10% of volume capacity)` pauses new work, while below `max(2 GiB, 2%)` stops growing tasks. Runtime checks cover artifacts, logs and personal storage. Use a capacity-limited data volume when a strict physical filesystem ceiling is required; application monitoring is not a filesystem quota. Remote pretrained identifiers require admin approval; existing local resources keep their original paths and formats.

## Cleanup and announcements

Automatic cleanup starts disabled. **Allow scheduled removal of unprotected models and their weights** is a separate opt-in; upgrading an instance that already enabled cache cleanup does not enable model deletion. Turning either cleanup switch off cancels pending model removal plans. Enabling it again rechecks expiry and protection, with administrators still choosing whether to announce new plans.

Cache cleanup assesses rebuildable inference copies and abandoned, unreferenced temporary/workspace directories. Model cleanup assesses owned, unprotected, sealed models after their idle retention period. **Silent cleanup has no extra waiting period**: eligible expired resources are removed during the current manual cleanup or background check. Original idle retention limits still apply, including when calling cleanup endpoints directly. Protecting or using a model cancels its pending plan and renews the idle period.

`retention.publish_announcements` defaults to `false` and controls announcements for new plans created by background checks. Manual **Clean up expired resources** actions choose independently through optional boolean `publish_announcements` in `POST /api/admin/resources/retention`, defaulting to `false` without changing the background setting. When announcing a new plan, `notice_hours` provides at least **72 hours** of announcement lead time; the UI shows this setting only when announcements are selected. Existing announced plans always keep their original deadlines even if announcements are later disabled. Read receipts are not required.

`scheduled_at` records the plan start independently of actual announcement time `notified_at`. New silent plans set `delete_after` to their start time and execute in the same check after persisting intent. Older silent plans lose their extra delay at the next check, subject to fresh eligibility checks; opting into announcements later does not announce them retroactively. Direct cleanup of individual eligible resources does not require a prior waiting plan. Audit records retain scheduling and adjustments to older silent deadlines. Failure recovery and slot accounting keep their existing behavior.

The model preview shows ownership, status, deadline, skip reasons and estimated reclaimable logical size without deleting files. Cleanup rechecks ownership, protection, recent use, active tasks, prediction/download leases, shared references and content integrity under locks. It verifies retained inputs before removing weights. External paths, explicitly excluded subtrees, datasets, source/environment records and pretrained resources are outside model deletion. Operators can set colon-separated absolute paths in `PEP_RETENTION_EXCLUDED_ROOTS` on Linux; Docker deployments must pass it into the backend through a persistent Compose override.

Removal hides the model from the active library, ends inference and weight downloads, and reclaims exclusive managed output/checkpoint/member copies, original artifacts and inference copies when safe. Shared objects remain while another live model or version binding needs them, so removing one model may release little space. Experiment parameters, metrics, historical version identities and bindings, inference history, and small reports remain. Retained version records do not imply that deleted weights can be downloaded or rerun; those requests return HTTP 410. Keep protected or separately backed-up models needed for future use.

A durable removal intent makes interruptions retryable. Failed attempts report zero released bytes and keep the model slot occupied until removal completes. Successful repeated cleanup reports zero additional space. Released space is the observed increase in physical volume free space and may differ from the logical estimate. Low free space never advances an announced deadline; if nothing safe can be reclaimed, new work pauses.

Manual announcements support all or selected users, pinning, scheduled visibility and private read receipts. Retention-managed notices cannot be changed through the manual announcement API.

## Upgrade

Alembic revision `20261107_0014` adds lifecycle/availability tables; `20261107_0015` adds daily entitlement transitions, change records and admission snapshots; `20261107_0016` separates cleanup scheduling from announcement times, backfilling historical plan starts without changing deadlines. Older policies default to announcements off, and direct database initialization also supports the upgrade. These migrations do not rewrite scientific records or identities. Application initialization backfills queued jobs lacking historical snapshots using upgrade-time allowances and labels the compatibility source; it does not infer historical values. Docker's entrypoint migrates automatically; bare-metal deployments can run `alembic upgrade head`. Back up the database and runtime, wait for running jobs to finish, and retain queued work when deploying. Policy changes persist without a restart.

Removing the extra delay for silent cleanup needs no further database migration. Existing silent plans are rechecked through normal cleanup entrypoints; application startup and database migrations do not delete files. Announced deadlines remain effective.
