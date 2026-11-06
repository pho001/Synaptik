# Real model training and live terminal CLI

Status: In progress — implementation, 58 tests and real CPU/native consumers verified; terminal/documentation gate continues with user-authorized TUI 0002.

## Authorization and dependency

The user requested real CLI tools after the autotuning audit, with live epochs/timing/speed,
actual Mbit/s, an ASCII Synaptik logo, a text loss plot and sensible refresh defaults. The user
forbids architecture-changing repairs. This task implements only an outer public-API caller.

- **Depends on:** [Tuning 0006](../../tuning/tasks/0006-candidate-coverage-and-correctness-audit.md) final integration and actual verification; delivered public NN/Training/Engine/Checkpoint contracts.
- **Conflicts with:** other CLI source/build writers; NN/Training public contract changes.
- **Parallel group:** `live-cli`; runner/data and terminal/progress scopes share the interface below.
- **Common base revision:** the committed Tuning 0006 documentation/identity closure; exact worktree receipts record its hash before writers start.
- **Integration order:** terminal/progress then runner/data/build; coordinator owns shared docs/notices/source inventory.
- **Integration validation:** actual CSV train/save/load/predict and native qualified consumers, PTY/nonTTY/resize/final/error/interrupt scenarios, focused behavior tests and affected architecture tests.
- **Shared-document integration owner:** main coordinator.
- **Frontier verification:** public-workflow/delivery research and Tuning prerequisite are complete. The user authorized this outer public-API caller; two disjoint writer scopes below may now start.

## Scope and architecture assessment

Read [AGENTS.md](../../../../../AGENTS.md), root “Core invariants” and “Dependency rules”,
[other tool scopes](../../../../architecture/contracts/backend-execution.md#other-tool-scopes),
and the [owned Model](../../../../architecture/contracts/extensions-training.md#extensionsnn)
and [Training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
headings. The root does not impose an Engine-only dependency whitelist on external consumers.
The CLI may add an inward public NN dependency while NN/Training/Engine keep their existing
owners, edges and lifetimes. New-edge and terminal-lifecycle risk require clean implementation
and independent review; that classification does not itself change architecture.

CLI owns file decoding, epoch traversal, counters and terminal presentation. Public NN Model
owns private parameters, its native parameter Arena, TrainingSession and cached execution.
Training owns optimizer math/state and atomic updates; Engine owns execution. Do not implement
optimizer/AD/kernel logic, a second executor, hard mid-step cancellation or private state access.

## Real command workflow

- `train`: numeric data and an output model path are required. Build a deterministic static
  FLOAT32 rank-two Linear Model, or acquire a supported saved one-input/one-output graph Model.
  Configure first-class mean-squared-error and public SGD/Adam once; train real batches and save
  current graph/weights only after successful completion. Default batch size 1 admits every
  positive dataset cardinality. For a loaded/static larger batch, require exact divisibility
  before the first update; never drop, repeat, pad or mask a suffix.
- `predict`: load the model's public source signature, accept matching feature rows and emit
  actual dense prediction values. No silent new model on a load failure.
- `inspect`: public descriptors/leaf metadata only; no Engine execution or claim of optimizer
  continuation. Graph/current values do not restore optimizer state or the dataset cursor.
- Default CPU composition; Metal and oneDNN use explicit existing public integration factories
  and absolute native-library options. Native preference does not prove actual native selection.

Numeric CSV is a documented finite FLOAT32 format without headers or quoted fields, not an
unrestricted CSV importer. Training uses the last output-width columns as targets. Scan and
validate the declared geometry before updates; keep data stable during the run. Refill separate
stored input/target buffers only after each synchronous call returns.

## Stable internal progress interface

The renderer owns package-private `TrainingProgress` in `io.github.pho001.synaptik.tools.cli`:

```java
// Runner supplies only actual successful training events; the monitor owns display clocks/history.
static TrainingProgress open(PrintStream error, boolean plain, long refreshMillis,
        int epochs, long stepsPerEpoch, long examplesPerEpoch, String backendLabel);
void beginEpoch(int oneBasedEpoch);
void step(double preUpdateLoss, long examples, long inputTargetBytes);
void endEpoch();
boolean cancellationRequested();
void finish(Outcome outcome, String detail); // COMPLETED, CANCELLED or FAILED
void close(); // Restore owned terminal state; never close a Model or Engine.
```

The factory signature is an internal interface specification, not a complete Java program.
The runner must not manufacture loss curves or completed steps. Actual loss comes from detached
`TrainingStep.objective()` and describes the forward pass before that step's update. An epoch
average describes a changing-parameter trajectory, not evaluation of the final epoch Model.
A nonfinite detached objective alone does not invalidate a committed step when its required
gradients and optimizer arithmetic are finite. Show `NaN`/`Infinity` honestly and plot a gap or
marker; do not invent a finite loss, omit successful counters or override Training's transaction.

## Presentation and measurement

- Refresh every **500 ms** by default. Render ASCII SYNAPTIK, epoch/progress, step rate, current
  and recent epoch timings/rates, decoded input+target Mbit/s and a bounded text loss history.
- Place the plot to the right when dimensions permit and below on narrow terminals. Query real
  dimensions, fit the frame, redraw only owned lines and restore the cursor.
- Count logical successful FLOAT32 payload, not file/network/device bytes. For batch B, F input
  features and O targets, payload is `4 * B * (F + O)` bytes. Use bytes and elapsed time from the
  same interval; zero intervals are unavailable rates, not infinity.
- Pin JLine terminal and FFM provider 3.30.6, explicit FFM provider and stderr selection. No
  fallback provider discovery. Real tty input/stderr, useful capabilities and dimensions are
  required; pipes, dumb/unsupported terminals and explicit plain mode use append-only logs.
- UI consumes detached statistics only. Interrupt requests stop future scheduling after the
  current synchronous step finishes; report cancellation, do not save a misleading successful
  model, and close normal owners in order. No rollback or forced cancellation claim.
- Always flush final success/error/cancellation even for runs shorter than the refresh period.
  Cleanup is best-effort on shutdown, not a guarantee after SIGKILL or VM failure.

## Delivery and acceptance

Use the standard Gradle application plugin and `SynaptikCli` main, application name `synaptik`,
JDK26 native-access arguments and `installDist` start scripts. The conservative ordinary workflow
does not need the optional incubator Vector module. Remove the obsolete marker
and migrate current references. This separate local CLI image does not extend or reissue the
sealed eighteen-product distribution and grants no public publishing permission. Carry existing
notices plus pinned JLine BSD and Apache-derived-component/Gradle-script licence material; do not
incidentally bundle a JDK, OpenBLAS, oneDNN or Metal native package.

Writers skip all builds/tests/lint/format/native runs; coordinator runs the final gate once.
Permanent tests cover actual training/prediction/signature/data errors and deterministic
counter/rate/loss boundaries, not source text, wording, wiring or ASCII snapshots. Actual PTY
proof must show periodic redraw, size change, final/error state and cooperative interrupt/cursor
restoration; plain output must contain no terminal escapes. Record backend/native facts and
scope limits. Architecture-dependent extensions receive analysis/proposals, not hidden workarounds.
