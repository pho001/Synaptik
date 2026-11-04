# Production release goal

## Purpose and status

Prepare Synaptik's first public production release for Java CPU inference and training, with
equally supported FLOAT32 (F32) and FLOAT64 (F64), trainable networks, reliable model save/load
and production-supported Metal. The current technical checkpoint completed framework-wide audit,
preserving repairs, independent review and source-matched local candidate qualification.

Status: the [audit result](public-release-audit-2026-11-04.md#result) accepts
`0.1.0-public-audit.20261104` within its explicit matrix. Earlier private/internal candidates
retain only their original bounded receipts. Public publication is **Blocked** by the owner's
decision to retain **no public redistribution grant for now**.
This goal is not an architecture contract, certification, Ready task or implementation grant.
Current behavior follows [architecture authority](../../ARCHITECTURE.md#authority-incorporation-and-precedence) and the
[existing lifecycle](../../ARCHITECTURE.md#core-lifecycle). Any later contract change needs its
coordinated approval.

The user's subsequent instruction authorizes continuous implementation through release acceptance,
using established framework practice for technical decisions. MAIN owns scoped Ready admission,
contract coordination, independent review and truthful verification. This execution authorization
does not waive the numerical, compatibility, lifecycle or release requirements below.

Public readiness supersedes the earlier private-only product assumption, not the owner's control
of distribution. Do not invent a public project licence, SPDX grant, POM licence, registry upload
or trusted public prebuilt. Public publication requires a separate licence/publisher decision.
Preserve and package the existing Cephes/Gradle third-party notices; they grant no rights to
Synaptik's own code. Public publication remains separate from the completed technical audit.

The [production support ledger](../backend-guide/production-support.md) separates current
capability, component evidence and exact historical candidate receipts. A previous documentation-
only artifact freeze does not extend acceptance to later numerical, lifecycle, persistence,
diagnostic or native repairs. The coordinator records final audit receipts after joined checks.

## Engineering acceptance

The user explicitly requires straightforward code that human programmers can read, maintain and
extend. Reuse existing constructors, semantic rules, validation owners, lowering and emitters
where their contracts apply; do not duplicate mathematics or introduce a second execution path.
Keep necessary independent boundary checks, but reuse their existing inference/validation owners
rather than copying algorithms. Add a helper or abstraction only for a concrete responsibility.
No special-case workaround, hidden fallback, suppressed error or weakened contract/test may stand
in for a complete implementation. Independent code review must assess these criteria explicitly.

## Java CPU target

F32 and F64 are equal production targets: the same declared operations, shapes, layouts,
forward/backward support, user workflows and applicable execution strategies. Equality means
capability parity, not equal speed or bit-identical floating results between dtypes or routes.
Support on another backend does not satisfy this Java CPU requirement.

The approved [CPU low-precision parity cutover](../architecture/contracts/backend-execution.md#cpu-low-precision-parity-cutover)
supersedes the temporary CPU0078 BFLOAT16 (BF16)/FLOAT16 (F16) arithmetic lock. BF16, F16,
FLOAT8_E4M3 and FLOAT8_E5M2 reuse ordinary CPU numerical, derivative, recurrent, symbolic and
training owners within their qualified family predicates. Low arithmetic uses F32 working values
and one final declared narrowing; raw movement, casts, guards and publication retain exact rules.
Parity is not universal geometry, strategy or performance qualification, and retained code alone
is not acceptance evidence.

Optional oneDNN is a cold-selected CPU route, not a separate backend or a condition of portable
CPU availability. Native F64 is excluded; F64 remains ordinary CPU/qualified OpenBLAS. Native
low-type support depends on the qualified API, geometry and target; no native FP8 on Arm or
oneDNN Graph FP8 execution is claimed. See the
[owning route limits](../architecture/contracts/backend-execution.md#cpu-low-precision-parity-cutover).
INT32, INT64 and BOOL remain necessary for indices, lengths, masks and control values.

## Metal target

Metal must be production-supported in the first release, not deferred or labelled experimental.
Its separate bounded domain includes qualified F32/BF16/F16 ordinary numerical, convolution/
pooling, fixed-cell first-order and native general Scan occurrences. CPU low parity grants Metal
no FP8 capability, general F64 numerical arithmetic or additional mixed-low permission. Exact/raw
F64 carriers do not imply F64 arithmetic. Follow the
[current Metal contract](../architecture/contracts/backend-execution.md#metal-backend) and
[support ledger](../backend-guide/production-support.md#metal-production-target), not old identity
numbers or a cross-backend parity assumption.

Release acceptance requires source-matched native packaging, installation and supported consumer
workflows verified on a real Metal device. Ordinary JVM success, compile-only CI or skipped
device tests are not that evidence. The [hosted GPU CI gate](backends/metal/tasks/0047-apple-silicon-metal-ci.md)
remains Blocked; this product decision does not select a runner or mark Metal release-ready.

## Required neural-network workflows

The release includes multilayer perceptrons (MLP), convolutional neural networks (CNN) and
recurrent neural networks (RNN). Each must support ordinary `compute`, `configureTraining`,
repeated `train`, save and load through the existing Model, Engine, Training and Checkpoint
owners. Loaded models must compute and train with new configuration on restored weights,
without promising exact optimizer/session continuation.

CNN training is mandatory. The user's explicit Conv3d completion request supersedes the earlier
1D/2D-only release choice. The [qualified convolution matrix](../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary)
includes Conv1d/2d/3d; static Conv3d input/weight/bias gradients and owned train/save/load/new-train
passed CPU FLOAT32/FLOAT64 and Metal-only FLOAT32 programs. Shared grouped convolution/window
and pooling adjoint owners also complete trainable Conv3d → Max/AveragePool3d → flatten → linear
with all-parameter finite differences, SGD and save/close/load/fresh training in those domains.

The practical RNN target uses runtime valid lengths L through the existing general Scan, which
repeats an ordinary tensor body while carrying state between steps. Reuse the
[owned Model workflow](../architecture/contracts/extensions-training.md#owned-tensor-models),
not a second RNN executor or dedicated replacement mathematical primitives. Keep ordinary
static-unrolled `.rnn` unchanged by this entry. NN0032's second-order compatibility gates remain;
first-order length support does not waive them or authorize that separate migration.

Metal has separate component evidence for bounded static F32/BF16/F16 runtime-length general
Scan and selected first-order fixed RNN/GRU/LSTM workflows. Those two recurrent paths retain
their distinct working/publication semantics; neither grants F64 arithmetic or higher-order AD.

## General Scan composition target

General Scan must compose all ordinary pure operations supported by the backend when bindings,
geometry and stable carry shape/dtype are legal. Pure means no hidden mutation or stateful effects.
Scan supplies traversal, bindings and activity, not a second mathematical implementation whitelist.
Reuse ordinary lowering and emitters, arranging scratch before run; no separate Scan mathematical
implementation, runtime interpreter, per-operation lookup, repeated plan/resource rebuilding or
avoidable per-step allocation. This does not prohibit direct calls to existing prepared kernel
helpers or their ordinary synchronous parallel-range dispatcher at each step, including necessary
invocation and worker joins, where current contracts qualify them. Existing bounded dispatch and
resource rules remain unchanged; no new scheduler is selected.

Automatic differentiation (AD) reuses ordinary rules and qualifies the actual generated
backward body. Forward-only operation support does not create a derivative rule. Individual body
values retain their ordinary dtypes; F32/F64 networks still need integral and Boolean values.
Legal exclusions must follow from binder, shape/layout, effect or activity constraints, or ordinary
unsupported AD, not a permanent separate list of Scan mathematical operations.

Runtime lengths do not prove row independence for all operations. Feature-wise
Softmax or per-sample convolution can preserve row independence; SUM or normalization across batch
axis B needs explicit active-row semantics. Random-number generation and mutation involve explicit
state/effects and are not automatically pure. Preserve the existing activity/skipping rules rather
than treating result masking as skipped work or inventing a model-wide loss mask.

Today's Scan execution is a bounded subset, not this complete target; source validity, cold
reconstruction, AD, execution and persistence are separate admissions. The
[declared qualification](../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification),
[per-value source rules](../architecture/contracts/recurrent-scan.md#accepted-per-value-typed-length-aware-source)
and [row-role rules](../architecture/contracts/recurrent-scan.md#bounded-row-role-analysis) describe
current boundaries, unchanged here; successors are not thereby implemented or Ready.

## Project order and current frontier

The earlier project order delivered practical runtime-length RNN, CPU F32/F64 and Scan closure,
Conv3d/Pool3d training, then bounded production Metal. Optional oneDNN and the approved CPU
low-parity cutover now have their own source and evidence; the old “oneDNN later” and CPU0078
lock decisions are not current implementation restrictions. Completed task rows and their
original failed commands remain historical records.

The current authorized frontier is the
[framework-wide audit repair packet](public-release-audit-2026-11-04.md#confirmed-finding-repair-frontiers),
including the Metal and diagnostic joins recorded there. It covers numerical and exact-count
boundaries, categorical AD, Scan traversal/admission, lifecycle/storage ownership, transactional
restore, source/import scaling and tuning diagnostics. This goal creates no additional task,
contract change or implementation grant.

After ordered integration and independent review, the coordinator runs joined verification and
qualifies a new exact-source/native/artifact candidate through clean external consumers. Earlier
operation-catalog execution totals and accepted local packages remain evidence for their own
snapshots, not proof that the repaired framework passed.

## Release acceptance checklist

Current audit candidate requirements are open until the coordinator records their actual results:

- [ ] Close confirmed release-critical findings with preserving repairs and independent review;
  update the support/disposition record without turning execution counts into numerical proof.
- [ ] Pass changed public paths, affected gates and the joined repository/native/Javadoc checkpoint,
  recording original failures, recovery, skips, target limitations and performance observations.
- [ ] Freeze matched repaired source, native package and versioned artifacts; verify exact-path
  provenance, checksums and dependency closure, then exercise clean external source/Maven/Gradle
  consumers and the required CPU/real-device Metal workflows.
- [ ] Obtain the owner's public licence/publisher decision before public publication. No grant is
  selected now; local hashes and ad-hoc signing are not an official publisher trust chain.

### Historical private/internal candidate checklist

The checked items below describe only the earlier bounded candidate recorded in the
[Pool3d/operation-audit receipt](../backend-guide/production-support.md#accepted-local-candidate-pool3d-and-operation-audit-closure).
They retain that checkpoint's scope and then-current CPU lock; they are not the current audit
checklist, a claim of unrun repair acceptance or public redistribution permission.

- [x] Publish a truthful operation × dtype × shape/layout × forward/backward × backend matrix,
  including applicable strategies and whole generated derivative closure. Java CPU F32/F64 have
  equal declared support; audit the temporary CPU low-arithmetic lock separately from retained
  implementation, shared types, storage/transfer/conversion and discrete semantics. Freeze CNN
  ranks through Conv3d under explicit dtype/geometry predicates, and qualify Metal's separate production support matrix.
- [x] Verify BF16/FP16 floating computation is locked on Java CPU without deleting its existing
  implementation or disabling retained storage/raw transfers/conversions and discrete semantics.
  No implicit low-precision CPU fallback or expansion of the locked capability is permitted.
- [x] Qualify exact/discrete results and permitted floating approximations under the existing
  [single Model numerical semantics](../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics).
  Use operation-specific test tolerances, special-value and saved-value checks, not new numerical
  profiles, runtime tolerance stores, certificates or wrapper layers.
- [x] Pass complete MLP, CNN and runtime-length RNN scenarios in both CPU dtypes: compute,
  configure training, repeated updates, save, load, compare restored inference and configure/train
  again. Cover changed inputs, relevant lengths/empty domains and actual backward numerics, not
  only construction, admission or forward smoke tests.
- [x] Pass real-device Metal conformance and supported consumer workflows against its frozen
  production matrix using a source-matched native package and reproducible installation.
  Record hardware/environment and limitations; compile-only or device-skipped checks are insufficient.
- [x] Verify invalid-input errors, failure ordering/cleanup, close and borrowed-resource lifetimes,
  repeated-run memory growth and concurrency under current contracts, with no stale inputs,
  accumulating resources or partial state publication.
- [x] Verify a clean installation and runnable documented API examples without developer-local paths;
  ship accurate support/limitations, API documentation, licences and third-party notices.
- [x] Freeze an RC matched to the validated source. Pass affected tests and one appropriate
  root/integration checkpoint, with environment, skips and limitations recorded. Historical failed
  commands remain failed; targeted recovery does not rename them PASS or imply a clean full suite.
- [x] Use short representative performance regressions and generated-versus-optimal-clean-Java
  structural/performance checks for actual hot-path changes. Global speed parity or a five-fork
  benchmark campaign is not a release gate; diagnosed avoidable overhead still needs disposition.

## Later scope and open choices

Full higher-order Scan, nested control flow/arbitrary while, all generic dynamic dimensions,
exact full training resume, broader autotuning and backends beyond required Java CPU and Metal
are not automatic release prerequisites. Existing compatibility gates remain in force. CNN,
F64 parity and ordinary-operation Scan composition do not move into optional later work; their
current bounded admissions and qualification remain distinct from the complete product target.

The [roadmap](roadmap.md#current-priority-and-next-design-decision) indexes current execution.
Public publication remains Blocked pending the owner's licence/publisher decision even if local
technical qualification succeeds. No historical success or failure is rewritten by this goal.
