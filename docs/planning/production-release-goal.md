# Production [release](../glossary.md#term-release) goal

## Purpose and status

[Prepare](../glossary.md#prepare) Synaptik's first public production [release](../glossary.md#term-release) for Java [CPU](../glossary.md#term-cpu) [inference](../glossary.md#term-inference) and training, with
equally supported [FLOAT32](../glossary.md#term-float32) ([F32](../glossary.md#floating-point-precision-and-rounding)) and [FLOAT64](../glossary.md#term-float64) ([F64](../glossary.md#floating-point-precision-and-rounding)), trainable networks, reliable [model](../glossary.md#term-model) save/load
and production-supported [Metal](../glossary.md#term-metal). The current technical [checkpoint](../glossary.md#term-checkpoint) completed framework-wide audit,
preserving repairs, independent review and source-matched local candidate [qualification](../glossary.md#term-qualification).

Status: the [audit result](public-release-audit-2026-11-04.md#result) accepts
`0.1.0-public-audit.20261104` within its explicit [matrix](../glossary.md#term-matrix-meanings). Earlier private/internal candidates
retain only their original bounded receipts. Public [publication](../glossary.md#term-publication) is **Blocked** by the [owner](../glossary.md#term-owner)'s
decision to retain **no public redistribution grant for now**.
This goal is not an [architecture contract](../glossary.md#architecture-contract), certification, [Ready](../glossary.md#term-ready) task or implementation grant.
Current behavior follows [architecture authority](../../ARCHITECTURE.md#authority-incorporation-and-precedence) and the
[existing lifecycle](../../ARCHITECTURE.md#core-lifecycle). Any later contract change needs its
coordinated approval.

The user's subsequent instruction authorizes continuous implementation through release acceptance,
using established framework practice for technical decisions. MAIN owns scoped Ready [admission](../glossary.md#term-admission),
contract coordination, independent review and truthful verification. This execution authorization
does not waive the numerical, compatibility, [lifecycle](../glossary.md#lifecycle) or release requirements below.

Public readiness supersedes the earlier private-only product assumption, not the owner's control
of [distribution](../glossary.md#term-distribution). Do not invent a public project licence, SPDX grant, [POM](../glossary.md#term-pom) licence, [registry](../glossary.md#term-registry) upload
or trusted public prebuilt. Public publication requires a separate licence/publisher decision.
Preserve and package the existing Cephes/[Gradle](../glossary.md#term-gradle) third-party notices; they grant no rights to
Synaptik's own code. Public publication remains separate from the completed technical audit.

The [production support ledger](../backend-guide/production-support.md) separates current
capability, component evidence and exact historical candidate receipts. A previous documentation-
only [artifact](../glossary.md#term-artifact) freeze does not extend acceptance to later numerical, lifecycle, [persistence](../glossary.md#term-persistence),
diagnostic or native repairs. The coordinator records final audit receipts after joined checks.

## Engineering acceptance

The user explicitly requires straightforward code that human programmers can read, maintain and
extend. Reuse existing constructors, semantic rules, [validation](../glossary.md#term-validation) [owners](../glossary.md#term-owner), [lowering](../glossary.md#lowering) and emitters
where their contracts apply; do not duplicate mathematics or introduce a second execution path.
Keep necessary independent boundary checks, but reuse their existing [inference](../glossary.md#term-inference)/validation owners
rather than copying algorithms. Add a helper or abstraction only for a concrete responsibility.
No special-case workaround, hidden fallback, suppressed error or weakened contract/test may stand
in for a complete implementation. Independent code review must assess these criteria explicitly.

## Java [CPU](../glossary.md#term-cpu) [target](../glossary.md#term-target)

[F32](../glossary.md#floating-point-precision-and-rounding) and [F64](../glossary.md#floating-point-precision-and-rounding) are equal production targets: the same declared operations, [shapes](../glossary.md#shape), [layouts](../glossary.md#layout),
forward/backward support, user workflows and applicable execution strategies. Equality means
capability parity, not equal speed or bit-identical floating results between [dtypes](../glossary.md#data-type--datatype) or routes.
Support on another [backend](../glossary.md#backend) does not satisfy this Java CPU requirement.

The approved [CPU low-precision parity cutover](../architecture/contracts/backend-execution.md#cpu-low-precision-parity-cutover)<span data-glossary-navigation="true"> ([precision glossary](../glossary.md#term-precision))</span>
supersedes the temporary CPU0078 [BFLOAT16](../glossary.md#term-bfloat16) ([BF16](../glossary.md#floating-point-precision-and-rounding))/[FLOAT16](../glossary.md#term-float16) ([F16](../glossary.md#floating-point-precision-and-rounding)) arithmetic lock. BF16, F16,
FLOAT8_E4M3 and FLOAT8_E5M2 reuse ordinary CPU numerical, [derivative](../glossary.md#term-derivative), recurrent, symbolic and
training [owners](../glossary.md#term-owner) within their qualified family predicates. Low arithmetic uses F32 working values
and one final declared [narrowing](../glossary.md#term-narrowing); raw movement, casts, guards and [publication](../glossary.md#term-publication) retain exact rules.
Parity is not universal geometry, strategy or performance [qualification](../glossary.md#term-qualification), and retained code alone
is not acceptance evidence.

Optional [oneDNN](../glossary.md#term-onednn) is a cold-selected CPU route, not a separate backend or a condition of portable
CPU availability. Native F64 is excluded; F64 remains ordinary CPU/qualified [OpenBLAS](../glossary.md#term-openblas). Native
low-type support depends on the qualified [API](../glossary.md#term-application-programming-interface-api), geometry and target; no native [FP8](../glossary.md#term-fp8) on Arm or
oneDNN [Graph](../glossary.md#graph) FP8 execution is claimed. See the
[owning route limits](../architecture/contracts/backend-execution.md#cpu-low-precision-parity-cutover).
[INT32](../glossary.md#data-type--datatype), [INT64](../glossary.md#data-type--datatype) and [BOOL](../glossary.md#data-type--datatype) remain necessary for indices, lengths, [masks](../glossary.md#term-boolean-mask) and control values.

## [Metal](../glossary.md#term-metal) [target](../glossary.md#term-target)

Metal must be production-supported in the first [release](../glossary.md#term-release), not deferred or labelled experimental.
Its separate bounded domain includes qualified [F32](../glossary.md#floating-point-precision-and-rounding)/[BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding) ordinary numerical, [convolution](../glossary.md#term-channels-first-convolution-layers)/
[pooling](../glossary.md#term-pooling), fixed-cell first-order and native [general Scan](../glossary.md#general-differentiable-scan) occurrences. [CPU](../glossary.md#term-cpu) low parity grants Metal
no [FP8](../glossary.md#term-fp8) capability, general [F64](../glossary.md#floating-point-precision-and-rounding) numerical arithmetic or additional mixed-low permission. Exact/raw
F64 carriers do not imply F64 arithmetic. Follow the
[current Metal contract](../architecture/contracts/backend-execution.md#metal-backend) and
[support ledger](../backend-guide/production-support.md#metal-production-target), not old [identity](../glossary.md#term-identity-meanings)
numbers or a cross-[backend](../glossary.md#backend) parity assumption.

Release acceptance requires source-matched native packaging, installation and supported consumer
workflows verified on a real Metal [device](../glossary.md#term-device). Ordinary [JVM](../glossary.md#term-java-virtual-machine-jvm) success, [compile](../glossary.md#compile)-only [CI](../glossary.md#term-continuous-integration-ci) or skipped
device tests are not that evidence. The [hosted GPU CI gate](backends/metal/tasks/0047-apple-silicon-metal-ci.md)<span data-glossary-navigation="true"> ([GPU glossary](../glossary.md#term-gpu))</span>
remains Blocked; this product decision does not select a runner or mark Metal release-[ready](../glossary.md#term-ready).

## Required neural-network workflows

The [release](../glossary.md#term-release) includes multilayer perceptrons ([MLP](../glossary.md#term-multilayer-perceptron-mlp)), convolutional neural networks ([CNN](../glossary.md#term-convolutional-neural-network-cnn)) and
recurrent neural networks ([RNN](../glossary.md#term-recurrent-neural-network-rnn)). Each must support ordinary `compute`, `configureTraining`,
repeated `train`, save and load through the existing [Model](../glossary.md#term-model), [Engine](../glossary.md#term-engine), Training and [Checkpoint](../glossary.md#term-checkpoint)
[owners](../glossary.md#term-owner). Loaded models must compute and train with new configuration on restored [weights](../glossary.md#term-weight),
without promising exact [optimizer](../glossary.md#term-optimizer)/session continuation.

CNN training is mandatory. The user's explicit [Conv3d](../glossary.md#conv3d--ncdhw) completion request supersedes the earlier
1D/2D-only release choice. The [qualified convolution matrix](../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary)<span data-glossary-navigation="true"> ([matrix glossary](../glossary.md#term-matrix-meanings))</span>
includes Conv1d/2d/3d; static Conv3d input/weight/[bias](../glossary.md#term-bias) [gradients](../glossary.md#term-gradient) and owned train/save/load/new-train
passed [CPU](../glossary.md#term-cpu) [FLOAT32](../glossary.md#term-float32)/[FLOAT64](../glossary.md#term-float64) and [Metal](../glossary.md#term-metal)-only FLOAT32 programs. Shared grouped [convolution](../glossary.md#term-channels-first-convolution-layers)/window
and [pooling](../glossary.md#term-pooling) [adjoint](../glossary.md#term-adjoint) owners also complete trainable Conv3d → Max/AveragePool3d → flatten → [linear](../glossary.md#term-linear-meanings)
with all-[parameter](../glossary.md#term-parameter) [finite differences](../glossary.md#arithmetic-rearrangement-and-finite-difference-checks), [SGD](../glossary.md#term-stochastic-gradient-descent-sgd) and save/close/load/fresh training in those domains.

The practical RNN [target](../glossary.md#term-target) uses [runtime](../glossary.md#term-runtime) valid lengths L through the existing [general Scan](../glossary.md#general-differentiable-scan), which
repeats an ordinary [tensor](../glossary.md#tensor) body while carrying state between steps. Reuse the
[owned Model workflow](../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([owned Model glossary](../glossary.md#model-module-buffer-and-scan-distinctions))</span>,
not a second RNN executor or dedicated replacement mathematical primitives. Keep ordinary
static-[unrolled](../glossary.md#backpropagation-through-time-and-recurrent-state) `.rnn` unchanged by this entry. NN0032's second-order compatibility gates remain;
first-order length support does not waive them or authorize that separate migration.

Metal has separate component evidence for bounded static [F32](../glossary.md#floating-point-precision-and-rounding)/[BF16](../glossary.md#floating-point-precision-and-rounding)/[F16](../glossary.md#floating-point-precision-and-rounding) runtime-length general
Scan and selected first-order fixed RNN/[GRU](../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../glossary.md#term-long-short-term-memory-lstm) workflows. Those two recurrent paths retain
their distinct working/[publication](../glossary.md#term-publication) semantics; neither grants [F64](../glossary.md#floating-point-precision-and-rounding) arithmetic or higher-order [AD](../glossary.md#autograd).

## [General Scan](../glossary.md#general-differentiable-scan) composition [target](../glossary.md#term-target)

General Scan must compose all ordinary pure operations supported by the [backend](../glossary.md#backend) when [bindings](../glossary.md#term-binding),
geometry and stable [carry](../glossary.md#term-carry) [shape](../glossary.md#shape)/[dtype](../glossary.md#data-type--datatype) are legal. Pure means no hidden mutation or stateful effects.
[Scan](../glossary.md#term-cumulative-scan) supplies traversal, bindings and activity, not a second mathematical implementation whitelist.
Reuse ordinary [lowering](../glossary.md#lowering) and emitters, arranging scratch before [run](../glossary.md#term-run); no separate Scan mathematical
implementation, [runtime](../glossary.md#term-runtime) interpreter, per-[operation](../glossary.md#operation) [lookup](../glossary.md#term-lookup), repeated plan/resource rebuilding or
avoidable per-step allocation. This does not prohibit direct calls to existing prepared [kernel](../glossary.md#kernel)
helpers or their ordinary [synchronous](../glossary.md#term-synchronous) parallel-range dispatcher at each step, including necessary
invocation and worker joins, where current contracts qualify them. Existing bounded [dispatch](../glossary.md#term-dispatch) and
resource rules remain unchanged; no new scheduler is selected.

[Automatic differentiation](../glossary.md#autograd) ([AD](../glossary.md#autograd)) reuses ordinary rules and qualifies the actual generated
backward body. Forward-only operation support does not create a [derivative](../glossary.md#term-derivative) rule. Individual body
values retain their ordinary dtypes; [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) networks still need integral and [Boolean](../glossary.md#data-type--datatype) values.
Legal exclusions must follow from binder, shape/[layout](../glossary.md#layout), effect or activity constraints, or ordinary
unsupported AD, not a permanent separate list of Scan mathematical operations.

Runtime lengths do not prove row independence for all operations. [Feature](../glossary.md#term-feature)-wise
[Softmax](../glossary.md#softmax--log-softmax) or per-[sample](../glossary.md#term-sample-machine-learning-and-hyperparameters) [convolution](../glossary.md#term-channels-first-convolution-layers) can preserve row independence; SUM or [normalization](../glossary.md#term-normalization) across [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters)
[axis](../glossary.md#term-axis) B needs explicit active-row semantics. Random-number generation and mutation involve explicit
state/effects and are not automatically pure. Preserve the existing activity/skipping rules rather
than treating result masking as skipped work or inventing a [model](../glossary.md#term-model)-wide [loss](../glossary.md#loss) [mask](../glossary.md#term-boolean-mask).

Today's Scan execution is a bounded subset, not this complete target; source validity, [cold
reconstruction](../glossary.md#term-cold-reconstruction), AD, execution and [persistence](../glossary.md#term-persistence) are separate admissions. The
[declared qualification](../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification)<span data-glossary-navigation="true"> ([qualification glossary](../glossary.md#term-qualification))</span>,
[per-value source rules](../architecture/contracts/recurrent-scan.md#accepted-per-value-typed-length-aware-source)
and [row-role rules](../architecture/contracts/recurrent-scan.md#bounded-row-role-analysis) describe
current boundaries, unchanged here; successors are not thereby implemented or [Ready](../glossary.md#term-ready).

## Project order and current frontier

The earlier project order delivered practical [runtime](../glossary.md#term-runtime)-length [RNN](../glossary.md#term-recurrent-neural-network-rnn), [CPU](../glossary.md#term-cpu) [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) and [Scan](../glossary.md#term-cumulative-scan) closure,
[Conv3d](../glossary.md#conv3d--ncdhw)/[Pool3d](../glossary.md#pool3d--ncdhw-pooling) training, then bounded production [Metal](../glossary.md#term-metal). Optional [oneDNN](../glossary.md#term-onednn) and the approved CPU
low-parity cutover now have their own source and evidence; the old “oneDNN later” and CPU0078
lock decisions are not current implementation restrictions. Completed task rows and their
original failed commands remain historical records.

The current [authorized frontier](../glossary.md#term-authorized-frontier) is the
[framework-wide audit repair packet](public-release-audit-2026-11-04.md#confirmed-finding-repair-frontiers),
including the Metal and diagnostic joins recorded there. It covers numerical and exact-count
boundaries, categorical [AD](../glossary.md#autograd), Scan traversal/[admission](../glossary.md#term-admission), [lifecycle](../glossary.md#lifecycle)/storage [ownership](../glossary.md#term-ownership-meanings), transactional
restore, source/import scaling and tuning diagnostics. This goal creates no additional task,
contract change or implementation grant.

After ordered integration and independent review, the coordinator runs joined verification and
qualifies a new exact-source/native/[artifact](../glossary.md#term-artifact) candidate through clean external consumers. Earlier
[operation](../glossary.md#operation)-catalog execution totals and accepted local packages remain evidence for their own
snapshots, not proof that the repaired framework passed.

## [Release](../glossary.md#term-release) acceptance checklist

Current audit candidate requirements are open until the coordinator records their actual results:

- [ ] Close confirmed release-critical findings with preserving repairs and independent review;
  update the support/disposition record without turning execution counts into numerical proof.
- [ ] Pass changed public paths, affected gates and the joined repository/native/Javadoc checkpoint,
  recording original failures, recovery, skips, target limitations and performance observations.<span data-glossary-navigation="true"> ([Javadoc glossary](../glossary.md#term-javadoc); [checkpoint glossary](../glossary.md#term-checkpoint); [target glossary](../glossary.md#term-target))</span>
- [ ] Freeze matched repaired source, native package and versioned artifacts; verify exact-path
  provenance, checksums and dependency closure, then exercise clean external source/Maven/Gradle
  consumers and the required CPU/real-device Metal workflows.<span data-glossary-navigation="true"> ([artifacts glossary](../glossary.md#term-artifact); [provenance glossary](../glossary.md#provenance); [dependency glossary](../glossary.md#term-dependency); [Maven glossary](../glossary.md#term-maven); [Gradle glossary](../glossary.md#term-gradle); [CPU glossary](../glossary.md#term-cpu); [device glossary](../glossary.md#term-device); [Metal glossary](../glossary.md#term-metal))</span>
- [ ] Obtain the owner's public licence/publisher decision before public publication. No grant is
  selected now; local hashes and ad-hoc signing are not an official publisher trust chain.<span data-glossary-navigation="true"> ([hashes glossary](../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([owner glossary](../glossary.md#term-owner); [publication glossary](../glossary.md#term-publication); [ad-hoc signing glossary](../glossary.md#term-ad-hoc-signing))</span>

### Historical private/internal candidate checklist

The checked items below describe only the earlier bounded candidate recorded in the
[Pool3d/operation-audit receipt](../backend-guide/production-support.md#accepted-local-candidate-pool3d-and-operation-audit-closure)<span data-glossary-navigation="true"> ([Pool3d glossary](../glossary.md#pool3d--ncdhw-pooling); [operation glossary](../glossary.md#operation))</span>.
They retain that [checkpoint](../glossary.md#term-checkpoint)'s [scope](../glossary.md#term-scope) and then-current [CPU](../glossary.md#term-cpu) lock; they are not the current audit
checklist, a claim of unrun repair acceptance or public redistribution permission.

- [x] Publish a truthful operation × dtype × shape/layout × forward/backward × backend matrix,
  including applicable strategies and whole generated derivative closure. Java CPU F32/F64 have
  equal declared support; audit the temporary CPU low-arithmetic lock separately from retained
  implementation, shared types, storage/transfer/conversion and discrete semantics. Freeze CNN
  ranks through Conv3d under explicit dtype/geometry predicates, and qualify Metal's separate production support matrix.<span data-glossary-navigation="true"> ([support matrix glossary](../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([dtype glossary](../glossary.md#data-type--datatype); [shape glossary](../glossary.md#shape); [layout glossary](../glossary.md#layout); [backend glossary](../glossary.md#backend); [matrix glossary](../glossary.md#term-review-matrix); [derivative glossary](../glossary.md#term-derivative); [F32 glossary](../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../glossary.md#floating-point-precision-and-rounding); [CNN glossary](../glossary.md#term-convolutional-neural-network-cnn); [Conv3d glossary](../glossary.md#conv3d--ncdhw); [Metal glossary](../glossary.md#term-metal))</span>
- [x] Verify BF16/FP16 floating computation is locked on Java CPU without deleting its existing
  implementation or disabling retained storage/raw transfers/conversions and discrete semantics.
  No implicit low-precision CPU fallback or expansion of the locked capability is permitted.<span data-glossary-navigation="true"> ([BF16 glossary](../glossary.md#floating-point-precision-and-rounding); [FP16 glossary](../glossary.md#term-fp16); [precision glossary](../glossary.md#term-precision))</span>
- [x] Qualify exact/discrete results and permitted floating approximations under the existing
  [single Model numerical semantics](../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([Model glossary](../glossary.md#term-model))</span>.
  Use operation-specific test tolerances, special-value and saved-value checks, not new numerical
  profiles, runtime tolerance stores, certificates or wrapper layers.<span data-glossary-navigation="true"> ([runtime glossary](../glossary.md#term-runtime); [tolerance glossary](../glossary.md#term-tolerance))</span>
- [x] Pass complete MLP, CNN and runtime-length RNN scenarios in both CPU dtypes: compute,
  configure training, repeated updates, save, load, compare restored inference and configure/train
  again. Cover changed inputs, relevant lengths/empty domains and actual backward numerics, not
  only construction, admission or forward smoke tests.<span data-glossary-navigation="true"> ([MLP glossary](../glossary.md#term-multilayer-perceptron-mlp); [RNN glossary](../glossary.md#term-recurrent-neural-network-rnn); [inference glossary](../glossary.md#term-inference); [admission glossary](../glossary.md#term-admission); [smoke glossary](../glossary.md#term-smoke-test))</span>
- [x] Pass real-device Metal conformance and supported consumer workflows against its frozen
  production matrix using a source-matched native package and reproducible installation.
  Record hardware/environment and limitations; compile-only or device-skipped checks are insufficient.<span data-glossary-navigation="true"> ([device glossary](../glossary.md#term-device); [conformance glossary](../glossary.md#term-conformance); [compile glossary](../glossary.md#compile))</span>
- [x] Verify invalid-input errors, failure ordering/cleanup, close and borrowed-resource lifetimes,
  repeated-run memory growth and concurrency under current contracts, with no stale inputs,
  accumulating resources or partial state publication.<span data-glossary-navigation="true"> ([borrowed glossary](../glossary.md#ownership-borrowing-and-lease); [run glossary](../glossary.md#term-run); [concurrency glossary](../glossary.md#term-concurrency); [publication glossary](../glossary.md#term-publication))</span>
- [x] Verify a clean installation and runnable documented API examples without developer-local paths;
  ship accurate support/limitations, API documentation, licences and third-party notices.<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span>
- [x] Freeze an RC matched to the validated source. Pass affected tests and one appropriate
  root/integration checkpoint, with environment, skips and limitations recorded. Historical failed
  commands remain failed; targeted recovery does not rename them PASS or imply a clean full suite.
- [x] Use short representative performance regressions and generated-versus-optimal-clean-Java
  structural/performance checks for actual hot-path changes. Global speed parity or a five-fork
  benchmark campaign is not a release gate; diagnosed avoidable overhead still needs disposition.<span data-glossary-navigation="true"> ([fork glossary](../glossary.md#term-fork-meanings); [release glossary](../glossary.md#term-release))</span>

## Later [scope](../glossary.md#term-scope) and open choices

Full higher-order [Scan](../glossary.md#term-cumulative-scan), nested control flow/arbitrary while, all generic dynamic [dimensions](../glossary.md#dimension),
exact full training [resume](../glossary.md#schema-serialization-and-checkpoint), broader [autotuning](../glossary.md#autotuning--model-autotuning) and [backends](../glossary.md#backend) beyond required Java [CPU](../glossary.md#term-cpu) and [Metal](../glossary.md#term-metal)
are not automatic [release](../glossary.md#term-release) prerequisites. Existing compatibility gates remain in force. [CNN](../glossary.md#term-convolutional-neural-network-cnn),
[F64](../glossary.md#floating-point-precision-and-rounding) parity and ordinary-[operation](../glossary.md#operation) Scan composition do not move into optional later work; their
current bounded admissions and [qualification](../glossary.md#term-qualification) remain distinct from the complete product [target](../glossary.md#term-target).

The [roadmap](roadmap.md#current-priority-and-next-design-decision) indexes current execution.
Public [publication](../glossary.md#term-publication) remains Blocked pending the [owner](../glossary.md#term-owner)'s licence/publisher decision even if local
technical qualification succeeds. No historical success or failure is rewritten by this goal.
