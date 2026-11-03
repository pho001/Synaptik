# Production release goal

## Purpose and status

Deliver Synaptik's first production release for Java CPU inference and training with equally
supported FLOAT32 (F32) and FLOAT64 (F64), trainable networks and reliable model save/load.
Production-supported Metal is also required in this first release. The path is practical recurrent
workflow → capability audit and scope freeze → required CPU/Metal fixes → release acceptance.

Status: user-agreed first-production scope (2026-11-01), with accepted component implementations.
Final candidate readiness is determined by exact-source/artifact/install/native/root qualification
receipts, not by this goal's status.
This goal is not an architecture contract, certification, Ready task or implementation grant.
Current behavior follows [architecture authority](../../ARCHITECTURE.md#authority-incorporation-and-precedence) and the
[existing lifecycle](../../ARCHITECTURE.md#core-lifecycle). Any later contract change needs its
coordinated approval.

The user's subsequent instruction authorizes continuous implementation through release acceptance,
using established framework practice for technical decisions. MAIN owns scoped Ready admission,
contract coordination, independent review and truthful verification. This execution authorization
does not waive the numerical, compatibility, lifecycle or release requirements below.

The owner explicitly selected **private/internal distribution** for this release. Do not invent a
public project licence, SPDX grant or POM licence, upload to public registries, or advertise a
trusted public prebuilt. Preserve and package the existing Cephes/Gradle third-party notices.
The licensing decision is resolved for this internal scope; future public redistribution needs
separate owner authorization. This changes no technical acceptance or required backend/workflow.

The [reviewed support matrix](../backend-guide/production-support.md) freezes the first-release
scope and separates execution counts from independent numerical/workflow proof. Required code
repairs and bounded local source/runtime qualification are accepted. The separate documentation-
only artifact-freeze receipt binds final candidate acceptance to the actual archive checksums
and preservation of the qualified non-documentation bytes; it is not an open implementation gate.

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

Temporarily lock BFLOAT16 (BF16) and FLOAT16 (FP16) floating computation on Java CPU for this
production target. Retain the existing implementation without further capability expansion;
do not delete it. The implemented lock is qualified in the support ledger; unlocking requires
a later explicit decision. No implicit emulated low-precision CPU fallback belongs to the target.

Shared Model data types, host storage, raw transfers and conversion helpers remain available;
the floating-compute lock does not remove them. INT32, INT64 and BOOL semantics needed for
indices, lengths, masks and control values remain necessary inside F32/F64 networks; this is not
a ban on per-value internal types.

## Metal target

Metal must be production-supported in the first release, not deferred or labelled experimental.
Freeze its declared operation, dtype, shape/layout and forward/backward matrix against actual
hardware support. Java CPU F32/F64 parity does not imply Metal F64 capability or cross-backend
feature parity; document Metal's own supported domains and limitations explicitly.

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

Metal now independently qualifies the static FLOAT32 runtime-length recurrent workflow through
its native general Scan and existing ordinary kernels. This does not grant Metal FLOAT64
numerical parity, widen low domains, or add BPTT to the three fixed-recurrent signatures.

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

## Project order and release scope freeze

The user's project order is:

1. Finish the practical RNN length workflow: accepted Compiler0018 cold reconstruction →
   accepted Checkpoint0014 v5 → NN0033 compute/train/save/load using general Scan. Cold means
   reconstructing tensor expressions without execution. All three have independent reviews,
   affected gates and public numerical proof; the complete all-PURE composition target remains below.
2. Audit missing Java CPU capability, including F32/F64 parity, CNN training, all-pure-operation
   Scan composition and the temporary Java low-precision arithmetic lock, preserving its existing
   implementation without expansion. Classify findings as release blocker, documented exclusion
   or later enhancement, then freeze the capability scope. Exclusions require an explicit reason
   and cannot silently relax the mandatory targets above.
3. Fix required CPU release blockers. Add no further features without a stated link to v1 acceptance.
4. Complete Metal for production support in the first release, then work on a oneDNN backend,
   in the user's requested order.

The release gate flow is scope freeze → required CPU/Metal blocker fixes → a frozen release
candidate (RC) and the checklist below. oneDNN is not a prerequisite for the first release.
Metal's production role and qualified hardware-bounded scope are settled by the accepted support
matrix. No new oneDNN architecture, build layout, native binary interface or dependencies are selected.

Compiler0017 and CPU0081's complete all-valid/length typed selected-proof/activity/ordinary-unit
cutover is integrated and qualified. Pool3d training and the CPU/Metal operation audit additionally
passed independent numerical, joined-root and installed-artifact/source consumer gates. Historical
Ready/dirty-source launch records do not describe the current implementation state.

## Release acceptance checklist

The checked items below are accepted for the explicit bounded matrix and local candidate recorded
in the [support ledger](../backend-guide/production-support.md#accepted-local-candidate-pool3d-and-operation-audit-closure).

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
exact full training resume, autotuning and backends beyond required Java CPU and Metal are not
automatic v1 prerequisites. Recording this goal does not itself lock or retire any existing
supported feature. Retaining, not expanding and temporarily locking Java CPU low-precision
computation remains part of the target; unlocking requires a later explicit decision. CNN,
F64 parity and all-pure-operation Scan composition do not move into optional later work.
Existing compatibility gates remain in force.

The 3D CNN scope and Metal's required first-release production role are settled. The Pool3d
and CPU/Metal operation-audit candidate is locally accepted against its explicit support
matrix and exact artifact receipts, not as universal operation/geometry coverage or public
distribution approval. The [roadmap](roadmap.md#current-priority-and-next-design-decision)
remains the execution index; no successor tasks or implementation grants are created here.
