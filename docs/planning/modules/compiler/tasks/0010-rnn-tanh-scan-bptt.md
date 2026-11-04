# Task 0010: First RNN_TANH [scan](../../../../glossary.md#term-fixed-recurrent-scan) [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) proposal

> **Editorial reading key — not part of the historical record.**
>
> This is explicitly a superseded proposal. Its fixed public adjoint, saved-history strategy and
> future gates are preserved design history, not current implementation instructions or
> authorization. Read BPTT as reverse propagation through time-dependent recurrence, and
> distinguish constructing such a graph from retaining physical hidden-state values during a run.
> The selected general-Scan approach constructs ordinary derivative expressions with explicit
> saved-primal bindings instead; current support must be read from that focused explanation.
>
> Current reading: [scoped Scan capture, roles and first-order differentiation](../../../../reference/compiler/verification-scan.md#current-scoped-scan-capture-and-first-order-ad); [pre-capture differentiation and functional requests](../../../../reference/compiler/autograd.md#current-package-private-pre-capture-autograd).

## Status and goal

Superseded — user selected [general differentiable scan](../../../../glossary.md#general-differentiable-scan) on 2026-10-21. Replaced by
[Model0034 decision](../../model/tasks/0034-general-differentiable-scan-decision.md) and the
[general scan program](../../../general-differentiable-scan.md)<span data-glossary-navigation="true"> ([general scan glossary](../../../../glossary.md#general-differentiable-scan))</span>. No fixed public [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) [adjoint](../../../../glossary.md#term-adjoint) is
selected. The content and Wegener's prior Draft-review evidence below are historical proposals,
not an [authorized frontier](../../../../glossary.md#term-authorized-frontier) or enacted policy. Implementation frontiers remain None.

## Historical proposal (not live gates)

Every approval, [dependency](../../../../glossary.md#term-dependency), frontier and future-tense requirement in the remaining [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) is
historical and inactive. Current decisions/frontiers are owned only by Model0034 and its program.
Historical proposed goal (retained unchanged):
Enable [backpropagation through time](../../../../glossary.md#term-backpropagation-through-time-bptt) ([BPTT](../../../../glossary.md#backpropagation-through-time-and-recurrent-state): reverse propagation through executed recurrent steps)
for compact `rnnScan`, not a new Training framework. Proposed flow: retained forward hidden values
→ one fixed [adjoint](../../../../glossary.md#term-adjoint) [occurrence](../../../../glossary.md#term-occurrence) → existing [Compiler](../../../../glossary.md#term-compiler) accumulation → existing [TrainingSession](../../../../glossary.md#training-session--trainingsession).

## Change class and [scope](../../../../glossary.md#term-scope)

Proposed implementation is Class C: new public [Model](../../../../glossary.md#term-model) semantics, saved-value [lifetime](../../../../glossary.md#term-lifetime), hot-loop
numerics and an [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [lifecycle](../../../../glossary.md#lifecycle) cutover. This change edits [planning](../../../../glossary.md#term-planning) only. First [slice](../../../../glossary.md#slice): first-order
RNN_TANH, [homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT32](../../../../glossary.md#term-float32), [CPU](../../../../glossary.md#term-cpu) direct [scalar](../../../../glossary.md#term-scalar) Java, biased/unbiased, FORWARD/REVERSE,
static `[T,B,I]`, `[B,H]`, `[H,I]`, `[H,H]`, optional `[H]`; I/H positive, T/B nonnegative.
[Runtime](../../../../glossary.md#term-runtime) non-[gradient](../../../../glossary.md#term-gradient) [INT64](../../../../glossary.md#data-type--datatype) `[B]` lengths vary without recompilation. Differentiate input, initial
hidden, [weights](../../../../glossary.md#term-weight) and present [bias](../../../../glossary.md#term-bias); never lengths/direction. Require [canonical](../../../../glossary.md#term-canonical-form) zero-offset dense
[adjoint](../../../../glossary.md#term-adjoint) boundaries after logical [layout](../../../../glossary.md#layout) closure (raw produced layouts may be unresolved); no copies.
Forward CPU0029 [dtype](../../../../glossary.md#data-type--datatype)/layout support remains unchanged and does not confer backward support.

Non-goals: [GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) adjoints, [FLOAT64](../../../../glossary.md#term-float64)/low types, noncanonical backward reads, dynamic [Shapes](../../../../glossary.md#shape),
higher order, [Metal](../../../../glossary.md#term-metal)/native/[vector](../../../../glossary.md#term-vector)/parallel adjoints, automatic `.rnn()` migration, new [optimizer](../../../../glossary.md#term-optimizer),
tape/session/regions/[registry](../../../../glossary.md#term-registry), second algebra, [persistence](../../../../glossary.md#term-persistence) wires or serialized backward [graphs](../../../../glossary.md#graph).
Retain existing forward APIs/[nodes](../../../../glossary.md#node), raw state behavior, and [O(1)](../../../../glossary.md#memory-management-atomic-counters-and-complexity)-in-T graph/preparation structure.

## Contracts and approval gates

Read [ARCHITECTURE.md](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), headings
[Scope](../../../../glossary.md#term-scope)-indexed [normative](../../../../glossary.md#term-normative) contracts, Core invariants, [Dependency](../../../../glossary.md#term-dependency) rules and Explicit non-goals.
Scoped reading: [recurrent-scan](../../../../architecture/contracts/recurrent-scan.md)<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-fixed-recurrent-scan))</span>, all headings,
especially Purity and [lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings) / Performance and migration invariants;
[Compiler contract](../../../../architecture/contracts/compiler-autograd.md#modulescompiler), headings
`modules/compiler`, [Training graph](../../../../glossary.md#training-graph) [model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler)-owned [automatic differentiation](../../../../glossary.md#autograd), [Compile](../../../../glossary.md#compile) lifecycle;
[Foundational contract](../../../../architecture/contracts/foundational-modules.md#modulesmodel),
`modules/model` / [Profile-free numerical semantics](../../../../glossary.md#profile-free-numerical-semantics);
[Backend contract](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules),
[Concrete backend](../../../../glossary.md#concrete-backend) [modules](../../../../glossary.md#term-module) / [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) routes;
[Extensions contract](../../../../architecture/contracts/extensions-training.md#owned-tensor-models),
Owned [tensor](../../../../glossary.md#tensor) Models / `modules/training`.

Current authority selects NO backward [operation](../../../../glossary.md#operation)/saved policy. Before [Ready](../../../../glossary.md#term-ready):

1. Explicit USER approval of the public primitive, saved-output semantics,
   numerical sites and first-order domain below (including always-present bias-gradient port).<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [gradient glossary](../../../../glossary.md#term-gradient))</span>
2. Main coordinates root/index plus recurrent-scan as sole owner of fixed-adjoint semantics;
   Compiler's owning contract changes its admission/rule policy, not Model derivative-rule ownership.
   Separately approve root + owning NN inference-only rejection cutover before NN work. Update
   relevant architecture explanations and a significant-decision ADR in that future change;
   establish unambiguous scoped ownership first, never duplicate semantic authority.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [adjoint glossary](../../../../glossary.md#term-adjoint); [admission glossary](../../../../glossary.md#term-admission); [derivative glossary](../../../../glossary.md#term-derivative); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [inference glossary](../../../../glossary.md#term-inference); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
3. Finalize signatures, flags, liveness and CPU resources against approved authority; stop on gaps.
   Prepare only the next selected module brief and verify its base/dependencies before authorization.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [liveness glossary](../../../../glossary.md#term-liveness); [Prepare glossary](../../../../glossary.md#prepare))</span>

## Dependencies and integration

- Depends on: Model0025E/0025F, Compiler0006A/0006D/0009, CPU0028/CPU0029, NN0021B/0022B,
  Engine0021 and Training0008 (Complete at base); gates above remain unsatisfied.
- Conflicts with: recurrent/adjoint API, autograd admission/accumulation, saved-value semantics,
  CPU capability/lowering/resources/ledger, NN training admission and shared authority/docs; serialize.<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [autograd glossary](../../../../glossary.md#autograd); [admission glossary](../../../../glossary.md#term-admission); [CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Parallel group: None; sole [planning](../../../../glossary.md#term-planning) writer, no worktree or implementation writer launched.
- Common base revision: main = origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`.
- Integration order: proposal review (Complete) → USER decision → coordinated authority → Model
  primitive → Compiler capability → CPU executable capability → explicit NN admission → main join.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Integration validation: affected-module gates plus focused conformance/integration below;
  main runs one source-matched native-enabled root <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance))</span>`test javadoc` at the completed program join.
- Shared-document integration owner: main; serial shared authority/explanations/planning integration.
  Preserve preexisting dirty THIRD_PARTY_NOTICES.md and CpuOrdinaryPointwiseCastCleanJavaOracle.java.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

## Current blockers and source packet

Paths are repository-relative; new types are listed separately as Proposed.
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/recurrent/RecurrentScanKind.java`
  and `model/tensor/RecurrentScan.java` under that source root: fixed forward transitions only.
  `model/operation/scan/CumulativeScanKind.java` offers SUM/PROD, not state-dependent matrix
  recurrence. Current algebra expresses local terms, but requires O(T) adjoint unrolling.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [adjoint glossary](../../../../glossary.md#term-adjoint); [unrolling glossary](../../../../glossary.md#term-unrolling))</span>
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/`: `AutogradPreflight`
  `rejectForwardOnlyOccurrences` rejects recurrence from the COMPLETE original inventory before
  seed/formula allocation, even on unrelated branches. <span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed))</span>`FirstOrderAutograd.expand` groups producers
  but applies each selected output separately. Add an approved joint-output rule, not two loops.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span>
  `ElementwiseGradientRules` [TANH](../../../../glossary.md#term-tanh) already uses stored output in `g * (1 - h*h)`.
  `RecurrentScanInference`, `CapturedGraphInference`, `FirstOrderGradientCoverage`,
  `StaticResultLogicalLayoutClosure` / `ForwardCommonSubexpressionElimination` own affected checks.
  Matching tests: `RecurrentScanCompilerTest` (siblings/[identity](../../../../glossary.md#term-identity-meanings)/[DCE](../../../../glossary.md#term-dead-code-elimination-dce)/empty [axes](../../../../glossary.md#term-axis)/preallocation rejection),
  `AutogradPreflightTest` (saved/sibling/higher-order rejection).
- `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`: `CpuCapabilityProvider`,
  `internal/lowering/CpuRecurrentScanLowering`, `internal/executable/CpuRecurrentScanPreparedExecutable`,
  `internal/route/portable/CpuRecurrentScanKernel`, `internal/prepare/CpuPartitionPreparer` /
  `CpuPartitionFinalizer`: forward lowering, pre-write checks, run scratch. Kernel stores the same
  FLOAT32 h to outputs/state; no backward route exists.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [run glossary](../../../../glossary.md#term-run); [Kernel glossary](../../../../glossary.md#kernel); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModelExecutionState.java`
  `rejectScanTraining` rejects all scans before setup effects, independently of [Compiler](../../../../glossary.md#term-compiler) support.
  `RuntimeLengthScanModelTest` checks rejection before/after compute/load and preserved setup state.
  `ModelBuilderExecutionTest.unroll` / `validMultiStepRnnTrainingMatchesDirectUnrolledGradientsAndSgd`
  is the training oracle; extend test equations for lengths/directions.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>

## Proposed primitive and saved values

Recommend ONE public fixed semantic primitive, explicitly requiring [API](../../../../glossary.md#term-application-programming-interface-api) approval:
`RecurrentScan.rnnTanhAdjoint(x, lengths, h0, W, U, savedOutputs, dOutputs, dFinalHidden, direction)`.
Eight [Tensor](../../../../glossary.md#tensor) inputs plus RecurrentDirection; one flat [identity](../../../../glossary.md#term-identity-meanings)-distinct [producer](../../../../glossary.md#term-producer).
`savedOutputs` is dense `[T,B,H]`; cotangents (incoming sensitivities) are `[T,B,H]` and `[B,H]`.
Return `[dInput, dInitialHidden, dInputWeight, dHiddenWeight, dBias]` with matching [Shapes](../../../../glossary.md#shape).
Always return `[H]` [bias](../../../../glossary.md#term-bias) sum, ignored when unbiased; bias/finalHidden VALUES are unnecessary.
This public numeric [operation](../../../../glossary.md#operation) is not a [Model](../../../../glossary.md#term-model)-owned [autograd](../../../../glossary.md#autograd) rule or hidden [Compiler](../../../../glossary.md#term-compiler) helper.
Construction is [descriptor](../../../../glossary.md#term-descriptor)-only; Compiler supplies [canonical](../../../../glossary.md#term-canonical-form) saved h. Proposed flags are floating-
input OR, not false to suppress differentiation; exact [signature](../../../../glossary.md#term-signature)/flag approval remains required.

New production types proposed (no new packages; prefix `io.github.pho001.synaptik.`):
- Model: `model.operation.recurrent.RnnTanhScanAdjointKind` (one fixed 8-input/5-output kind);
  `model.tensor.RnnTanhScanAdjointResult` (five canonical wrappers). Extend existing RecurrentScan
  for construction/descriptor checks; no new namespace or generic expression abstraction.
- Compiler: package-private `compiler.RnnTanhScanAdjointInference` (independent descriptor checks)
  and `compiler.RnnTanhScanGradientRules` (joint port cotangents → public primitive).
- [CPU](../../../../glossary.md#term-cpu): internal `backend.cpu.internal.lowering.CpuRnnTanhScanAdjointLowering` with nested
  `Geometry`/`Facts` (checked descriptors/positional [bindings](../../../../glossary.md#term-binding)); `internal.executable.` and
  `internal.route.portable.` under `backend.cpu` own `CpuRnnTanhScanAdjointPreparedExecutable`
  (nested invocation-local `Invocation`) and `CpuRnnTanhScanAdjointKernel` respectively.
  Existing preparation-plan/resource owners carry facts; no registry/loop IR.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- NN/Training/Engine/Prepare/Runtime: no new production type/package proposed. Any additional
  production concept discovered during design is a Draft replan gate, not implied permission.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

Retain dense forward h through ordinary [graph](../../../../glossary.md#graph) edges until [adjoint](../../../../glossary.md#term-adjoint) completion, even for final-only
objectives. [TANH](../../../../glossary.md#term-tanh) needs stored h, not gates/preactivation; predecessor h is a valid output or h0.
Keep x/h0/W/U/lengths live as ordinary reads; no tape/[cache](../../../../glossary.md#term-cache)/hidden state/serialized [run](../../../../glossary.md#term-run) values.
Bounded recomputation repeats forward into O([TBH](../../../../glossary.md#term-tbh)) scratch and may change allowed rounded h;
prefix replay can cost O(T²). Recommend retaining already-produced O(TBH) outputs instead:
longer [liveness](../../../../glossary.md#term-liveness), no duplicate computation. Exact bits, saved-boundary protection and [materialization](../../../../glossary.md#materialization)/
liveness tests remain approval gates.

## Proposed [adjoint](../../../../glossary.md#term-adjoint) semantics and acceptance

For each row, initialize [carry](../../../../glossary.md#term-carry) from dFinalHidden; visit executed steps in opposite forward order.
At original [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) t: `q = dOutputs[t] + carry`; `delta = q * (1 - h[t]*h[t])` with separate
[FLOAT32](../../../../glossary.md#term-float32) MUL, SUB, MUL sites as the existing [TANH](../../../../glossary.md#term-tanh) rule; then `dX[t] = delta @ W`,
`dW += outer(delta,x[t])`, `dU += outer(delta,hPrevious)`, `db += delta`, `carry = delta @ U`.
Every contributor participates once under [Model](../../../../glossary.md#term-model) arithmetic rules; no active zero/[saturation](../../../../glossary.md#term-saturation) shortcut,
hidden [widening](../../../../glossary.md#term-widening) or new [tolerance](../../../../glossary.md#term-tolerance) policy. Finish dH0 from carry. Shared [weights](../../../../glossary.md#term-weight) sum over valid time/rows
and occurrences through ordinary [Tensor](../../../../glossary.md#tensor).add. Tied Tensor roles receive EACH positional contribution
before [identity](../../../../glossary.md#term-identity-meanings) accumulation (including W==U when I==H); never deduplicate terms.
Accumulate incoming contributions per output first, then call the adjoint once per forward [occurrence](../../../../glossary.md#term-occurrence);
missing cotangents must be [canonical](../../../../glossary.md#term-canonical-form) after closure without copies. `DerivativeConstants.zeroLike`
constructs an EXPAND [view](../../../../glossary.md#term-view-meanings): do not blindly reuse it. Future [Compiler](../../../../glossary.md#term-compiler) gate/test: canonical typed zeros,
e.g. [shape](../../../../glossary.md#shape)-sized registered logical zero splats, including missing final-state [sensitivity](../../../../glossary.md#term-sensitivity).

- FORWARD's adjoint visits L-1..0; REVERSE's visits 0..L-1, using h[t+1] as predecessor except
  at L-1, which uses h0. Never reverse the padded suffix or use output[T-1] as semantic final state.
- Padded dX and empty parameter sums are positive zero. Branch before reading padded x/h/dOutputs:
  NaN/Inf padding cannot enter arithmetic through <span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [padding glossary](../../../../glossary.md#padding))</span>`0 * NaN`. Active NaN/Inf follows named formulas.
  L=0: raw-copy dFinalHidden to dH0, no transition or padded cotangent read; T=0 is the same for all
  rows. B=0: empty dX/dH0 and zero parameter sums. I/H remain positive. Test signed zeros/raw identity.<span data-glossary-navigation="true"> ([cotangent glossary](../../../../glossary.md#cotangent))</span>
- CPU design gate: bounded row scratch (candidate: carry/delta H-vectors, omitted for empty T/B);
  declare exact checked resources and five dense outputs before slot assignment. Preserve positional
  roles/read aliases; reject write/read, write/write and scratch overlap. Check spans/counts/bytes,
  capacity/alignment/access/writability and ALL lengths after producers BEFORE any result write.
  Test isolated runs, invalid-length retry, produced lengths and heap/native/mixed bindings.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [vectors glossary](../../../../glossary.md#term-vector); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [alignment glossary](../../../../glossary.md#term-alignment); [heap glossary](../../../../glossary.md#term-heap); [bindings glossary](../../../../glossary.md#term-binding))</span>
- First-order only: preflight complete inventory before allocations, admit only the approved RNN
  domain, retain GRU/LSTM/other-type rejection even on unrelated branches. Reject two-stage/createGraph
  requests involving recurrence before stage-one allocation, and differentiation of the adjoint
  itself; connected siblings/DisconnectedPolicy.ZERO must not hide unsupported required routes.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- Compare all five gradients to independent masked/directional unrolling with both cotangents,
  outputs-only, final-only, tied roles, shared parameters, mixed zero/full lengths, empty axes and
  saturated/special values. Finite differences supplement, not replace, ordinary-algebra oracles.<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [unrolling glossary](../../../../glossary.md#term-unrolling); [axes glossary](../../../../glossary.md#term-axis); [Finite differences glossary](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks))</span>

## Ordered [module](../../../../glossary.md#term-module) capabilities and [validation](../../../../glossary.md#term-validation) (future, not launched)

After approval, each serial [owner](../../../../glossary.md#term-owner) [slice](../../../../glossary.md#slice) includes tests/[Javadoc](../../../../glossary.md#term-javadoc)/docs and independent Class C review:
[Model](../../../../glossary.md#term-model) primitive → [Compiler](../../../../glossary.md#term-compiler) joint-output construction/[inference](../../../../glossary.md#term-inference)/[liveness](../../../../glossary.md#term-liveness) → [CPU](../../../../glossary.md#term-cpu) executable/[conformance](../../../../glossary.md#term-conformance)
→ [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [admission](../../../../glossary.md#term-admission). NN needs explicit root/owning-NN approval: positive T/B [canonical](../../../../glossary.md#term-canonical-form) [F32](../../../../glossary.md#floating-point-precision-and-rounding) [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) [scan](../../../../glossary.md#term-fixed-recurrent-scan)
sources with supported ordinary heads; retain unsupported-scan early rejection/[lifecycle](../../../../glossary.md#lifecycle) precedence.
Empty-[axis](../../../../glossary.md#term-axis) [gradients](../../../../glossary.md#term-gradient) are low-level, not new Training empty-input admission. Test `.rnnScan` dense/
finalHidden training, [graph](../../../../glossary.md#graph)/load, fresh lengths and each [SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd) update against [unrolling](../../../../glossary.md#term-unrolling). Compiler alone
does NOT unlock model.train; no Training [optimizer](../../../../glossary.md#term-optimizer) change or arbitrary NN-chain promise.

Future workers [run](../../../../glossary.md#term-run) focused new [adjoint](../../../../glossary.md#term-adjoint) cases plus once each owned module's `test javadoc`:
`./gradlew :modules:model:test :modules:model:javadoc`;
`./gradlew :modules:compiler:test :modules:compiler:javadoc`;
`./gradlew :backends:cpu:test :backends:cpu:javadoc`;
`./gradlew :extensions:nn:test :extensions:nn:javadoc`.
Main's focused join extends `CpuRecurrentScanConformanceTest`, `EngineCpuRecurrentScanIntegrationTest`
and `RuntimeLengthScanModelIntegrationTest` in `:testing:backend-conformance:test` /
`:testing:integration-tests:test`; include current capability-ledger negatives/positive new [occurrence](../../../../glossary.md#term-occurrence),
architecture validation for the authority cutover and ONE native-enabled root `test javadoc` join.
No benchmark campaign or duplicate review suites. CPU0029's eight diagnostic
ratios 0.88–5.84 have unequal invocation boundaries and unresolved [warm-up](../../../../glossary.md#benchmark-measurement-vocabulary)/[JIT](../../../../glossary.md#term-just-in-time-compilation-jit); speed parity is unproven.

## Documentation, review and [planning](../../../../glossary.md#term-planning) result

Wegener: APPROVE as Draft proposal, no blocking findings; USER/authority gates remain.
Planning checks: delta links/anchors, size, `git diff --check`, protected [hashes](../../../../glossary.md#term-hash) PASS. Planning/general
profiles and targeted glossary review complete; no glossary/authority edit. No [Gradle](../../../../glossary.md#term-gradle)/timing [run](../../../../glossary.md#term-run).
Implementation not started; Draft/None frontiers unchanged. Future [Javadoc](../../../../glossary.md#term-javadoc)/docs/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) review required.
