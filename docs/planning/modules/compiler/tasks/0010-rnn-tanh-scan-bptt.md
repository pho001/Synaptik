# Task 0010: First RNN_TANH scan BPTT proposal

## Status and goal

Superseded — user selected general differentiable scan on 2026-10-21. Replaced by
[Model0034 decision](../../model/tasks/0034-general-differentiable-scan-decision.md) and the
[general scan program](../../../general-differentiable-scan.md). No fixed public RNN adjoint is
selected. The content and Wegener's prior Draft-review evidence below are historical proposals,
not an authorized frontier or enacted policy. Implementation frontiers remain None.

## Historical proposal (not live gates)

Every approval, dependency, frontier and future-tense requirement in the remaining payload is
historical and inactive. Current decisions/frontiers are owned only by Model0034 and its program.
Historical proposed goal (retained unchanged):
Enable backpropagation through time (BPTT: reverse propagation through executed recurrent steps)
for compact `rnnScan`, not a new Training framework. Proposed flow: retained forward hidden values
→ one fixed adjoint occurrence → existing Compiler accumulation → existing TrainingSession.

## Change class and scope

Proposed implementation is Class C: new public Model semantics, saved-value lifetime, hot-loop
numerics and an NN lifecycle cutover. This change edits planning only. First slice: first-order
RNN_TANH, homogeneous FLOAT32, CPU direct scalar Java, biased/unbiased, FORWARD/REVERSE,
static `[T,B,I]`, `[B,H]`, `[H,I]`, `[H,H]`, optional `[H]`; I/H positive, T/B nonnegative.
Runtime non-gradient INT64 `[B]` lengths vary without recompilation. Differentiate input, initial
hidden, weights and present bias; never lengths/direction. Require canonical zero-offset dense
adjoint boundaries after logical layout closure (raw produced layouts may be unresolved); no copies.
Forward CPU0029 dtype/layout support remains unchanged and does not confer backward support.

Non-goals: GRU/LSTM adjoints, FLOAT64/low types, noncanonical backward reads, dynamic Shapes,
higher order, Metal/native/vector/parallel adjoints, automatic `.rnn()` migration, new optimizer,
tape/session/regions/registry, second algebra, persistence wires or serialized backward graphs.
Retain existing forward APIs/nodes, raw state behavior, and O(1)-in-T graph/preparation structure.

## Contracts and approval gates

Read [ARCHITECTURE.md](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), headings
Scope-indexed normative contracts, Core invariants, Dependency rules and Explicit non-goals.
Scoped reading: [recurrent-scan](../../../../architecture/contracts/recurrent-scan.md), all headings,
especially Purity and lifecycle ownership / Performance and migration invariants;
[Compiler contract](../../../../architecture/contracts/compiler-autograd.md#modulescompiler), headings
`modules/compiler`, Training graph model, Compiler-owned automatic differentiation, Compile lifecycle;
[Foundational contract](../../../../architecture/contracts/foundational-modules.md#modulesmodel),
`modules/model` / Profile-free numerical semantics;
[Backend contract](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules),
Concrete backend modules / CPU backend routes;
[Extensions contract](../../../../architecture/contracts/extensions-training.md#owned-tensor-models),
Owned tensor Models / `modules/training`.

Current authority selects NO backward operation/saved policy. Before Ready:

1. Explicit USER approval of the public primitive, saved-output semantics,
   numerical sites and first-order domain below (including always-present bias-gradient port).
2. Main coordinates root/index plus recurrent-scan as sole owner of fixed-adjoint semantics;
   Compiler's owning contract changes its admission/rule policy, not Model derivative-rule ownership.
   Separately approve root + owning NN inference-only rejection cutover before NN work. Update
   relevant architecture explanations and a significant-decision ADR in that future change;
   establish unambiguous scoped ownership first, never duplicate semantic authority.
3. Finalize signatures, flags, liveness and CPU resources against approved authority; stop on gaps.
   Prepare only the next selected module brief and verify its base/dependencies before authorization.

## Dependencies and integration

- Depends on: Model0025E/0025F, Compiler0006A/0006D/0009, CPU0028/CPU0029, NN0021B/0022B,
  Engine0021 and Training0008 (Complete at base); gates above remain unsatisfied.
- Conflicts with: recurrent/adjoint API, autograd admission/accumulation, saved-value semantics,
  CPU capability/lowering/resources/ledger, NN training admission and shared authority/docs; serialize.
- Parallel group: None; sole planning writer, no worktree or implementation writer launched.
- Common base revision: main = origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`.
- Integration order: proposal review (Complete) → USER decision → coordinated authority → Model
  primitive → Compiler capability → CPU executable capability → explicit NN admission → main join.
- Integration validation: affected-module gates plus focused conformance/integration below;
  main runs one source-matched native-enabled root `test javadoc` at the completed program join.
- Shared-document integration owner: main; serial shared authority/explanations/planning integration.
  Preserve preexisting dirty THIRD_PARTY_NOTICES.md and CpuOrdinaryPointwiseCastCleanJavaOracle.java.

## Current blockers and source packet

Paths are repository-relative; new types are listed separately as Proposed.
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/recurrent/RecurrentScanKind.java`
  and `model/tensor/RecurrentScan.java` under that source root: fixed forward transitions only.
  `model/operation/scan/CumulativeScanKind.java` offers SUM/PROD, not state-dependent matrix
  recurrence. Current algebra expresses local terms, but requires O(T) adjoint unrolling.
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/`: `AutogradPreflight`
  `rejectForwardOnlyOccurrences` rejects recurrence from the COMPLETE original inventory before
  seed/formula allocation, even on unrelated branches. `FirstOrderAutograd.expand` groups producers
  but applies each selected output separately. Add an approved joint-output rule, not two loops.
  `ElementwiseGradientRules` TANH already uses stored output in `g * (1 - h*h)`.
  `RecurrentScanInference`, `CapturedGraphInference`, `FirstOrderGradientCoverage`,
  `StaticResultLogicalLayoutClosure` / `ForwardCommonSubexpressionElimination` own affected checks.
  Matching tests: `RecurrentScanCompilerTest` (siblings/identity/DCE/empty axes/preallocation rejection),
  `AutogradPreflightTest` (saved/sibling/higher-order rejection).
- `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`: `CpuCapabilityProvider`,
  `internal/lowering/CpuRecurrentScanLowering`, `internal/executable/CpuRecurrentScanPreparedExecutable`,
  `internal/route/portable/CpuRecurrentScanKernel`, `internal/prepare/CpuPartitionPreparer` /
  `CpuPartitionFinalizer`: forward lowering, pre-write checks, run scratch. Kernel stores the same
  FLOAT32 h to outputs/state; no backward route exists.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModelExecutionState.java`
  `rejectScanTraining` rejects all scans before setup effects, independently of Compiler support.
  `RuntimeLengthScanModelTest` checks rejection before/after compute/load and preserved setup state.
  `ModelBuilderExecutionTest.unroll` / `validMultiStepRnnTrainingMatchesDirectUnrolledGradientsAndSgd`
  is the training oracle; extend test equations for lengths/directions.

## Proposed primitive and saved values

Recommend ONE public fixed semantic primitive, explicitly requiring API approval:
`RecurrentScan.rnnTanhAdjoint(x, lengths, h0, W, U, savedOutputs, dOutputs, dFinalHidden, direction)`.
Eight Tensor inputs plus RecurrentDirection; one flat identity-distinct producer.
`savedOutputs` is dense `[T,B,H]`; cotangents (incoming sensitivities) are `[T,B,H]` and `[B,H]`.
Return `[dInput, dInitialHidden, dInputWeight, dHiddenWeight, dBias]` with matching Shapes.
Always return `[H]` bias sum, ignored when unbiased; bias/finalHidden VALUES are unnecessary.
This public numeric operation is not a Model-owned autograd rule or hidden Compiler helper.
Construction is descriptor-only; Compiler supplies canonical saved h. Proposed flags are floating-
input OR, not false to suppress differentiation; exact signature/flag approval remains required.

New production types proposed (no new packages; prefix `io.github.pho001.synaptik.`):
- Model: `model.operation.recurrent.RnnTanhScanAdjointKind` (one fixed 8-input/5-output kind);
  `model.tensor.RnnTanhScanAdjointResult` (five canonical wrappers). Extend existing RecurrentScan
  for construction/descriptor checks; no new namespace or generic expression abstraction.
- Compiler: package-private `compiler.RnnTanhScanAdjointInference` (independent descriptor checks)
  and `compiler.RnnTanhScanGradientRules` (joint port cotangents → public primitive).
- CPU: internal `backend.cpu.internal.lowering.CpuRnnTanhScanAdjointLowering` with nested
  `Geometry`/`Facts` (checked descriptors/positional bindings); `internal.executable.` and
  `internal.route.portable.` under `backend.cpu` own `CpuRnnTanhScanAdjointPreparedExecutable`
  (nested invocation-local `Invocation`) and `CpuRnnTanhScanAdjointKernel` respectively.
  Existing preparation-plan/resource owners carry facts; no registry/loop IR.
- NN/Training/Engine/Prepare/Runtime: no new production type/package proposed. Any additional
  production concept discovered during design is a Draft replan gate, not implied permission.

Retain dense forward h through ordinary graph edges until adjoint completion, even for final-only
objectives. TANH needs stored h, not gates/preactivation; predecessor h is a valid output or h0.
Keep x/h0/W/U/lengths live as ordinary reads; no tape/cache/hidden state/serialized run values.
Bounded recomputation repeats forward into O(TBH) scratch and may change allowed rounded h;
prefix replay can cost O(T²). Recommend retaining already-produced O(TBH) outputs instead:
longer liveness, no duplicate computation. Exact bits, saved-boundary protection and materialization/
liveness tests remain approval gates.

## Proposed adjoint semantics and acceptance

For each row, initialize carry from dFinalHidden; visit executed steps in opposite forward order.
At original coordinate t: `q = dOutputs[t] + carry`; `delta = q * (1 - h[t]*h[t])` with separate
FLOAT32 MUL, SUB, MUL sites as the existing TANH rule; then `dX[t] = delta @ W`,
`dW += outer(delta,x[t])`, `dU += outer(delta,hPrevious)`, `db += delta`, `carry = delta @ U`.
Every contributor participates once under Model arithmetic rules; no active zero/saturation shortcut,
hidden widening or new tolerance policy. Finish dH0 from carry. Shared weights sum over valid time/rows
and occurrences through ordinary Tensor.add. Tied Tensor roles receive EACH positional contribution
before identity accumulation (including W==U when I==H); never deduplicate terms.
Accumulate incoming contributions per output first, then call the adjoint once per forward occurrence;
missing cotangents must be canonical after closure without copies. `DerivativeConstants.zeroLike`
constructs an EXPAND view: do not blindly reuse it. Future Compiler gate/test: canonical typed zeros,
e.g. shape-sized registered logical zero splats, including missing final-state sensitivity.

- FORWARD's adjoint visits L-1..0; REVERSE's visits 0..L-1, using h[t+1] as predecessor except
  at L-1, which uses h0. Never reverse the padded suffix or use output[T-1] as semantic final state.
- Padded dX and empty parameter sums are positive zero. Branch before reading padded x/h/dOutputs:
  NaN/Inf padding cannot enter arithmetic through `0 * NaN`. Active NaN/Inf follows named formulas.
  L=0: raw-copy dFinalHidden to dH0, no transition or padded cotangent read; T=0 is the same for all
  rows. B=0: empty dX/dH0 and zero parameter sums. I/H remain positive. Test signed zeros/raw identity.
- CPU design gate: bounded row scratch (candidate: carry/delta H-vectors, omitted for empty T/B);
  declare exact checked resources and five dense outputs before slot assignment. Preserve positional
  roles/read aliases; reject write/read, write/write and scratch overlap. Check spans/counts/bytes,
  capacity/alignment/access/writability and ALL lengths after producers BEFORE any result write.
  Test isolated runs, invalid-length retry, produced lengths and heap/native/mixed bindings.
- First-order only: preflight complete inventory before allocations, admit only the approved RNN
  domain, retain GRU/LSTM/other-type rejection even on unrelated branches. Reject two-stage/createGraph
  requests involving recurrence before stage-one allocation, and differentiation of the adjoint
  itself; connected siblings/DisconnectedPolicy.ZERO must not hide unsupported required routes.
- Compare all five gradients to independent masked/directional unrolling with both cotangents,
  outputs-only, final-only, tied roles, shared parameters, mixed zero/full lengths, empty axes and
  saturated/special values. Finite differences supplement, not replace, ordinary-algebra oracles.

## Ordered module capabilities and validation (future, not launched)

After approval, each serial owner slice includes tests/Javadoc/docs and independent Class C review:
Model primitive → Compiler joint-output construction/inference/liveness → CPU executable/conformance
→ NN admission. NN needs explicit root/owning-NN approval: positive T/B canonical F32 RNN scan
sources with supported ordinary heads; retain unsupported-scan early rejection/lifecycle precedence.
Empty-axis gradients are low-level, not new Training empty-input admission. Test `.rnnScan` dense/
finalHidden training, graph/load, fresh lengths and each SGD update against unrolling. Compiler alone
does NOT unlock model.train; no Training optimizer change or arbitrary NN-chain promise.

Future workers run focused new adjoint cases plus once each owned module's `test javadoc`:
`./gradlew :modules:model:test :modules:model:javadoc`;
`./gradlew :modules:compiler:test :modules:compiler:javadoc`;
`./gradlew :backends:cpu:test :backends:cpu:javadoc`;
`./gradlew :extensions:nn:test :extensions:nn:javadoc`.
Main's focused join extends `CpuRecurrentScanConformanceTest`, `EngineCpuRecurrentScanIntegrationTest`
and `RuntimeLengthScanModelIntegrationTest` in `:testing:backend-conformance:test` /
`:testing:integration-tests:test`; include current capability-ledger negatives/positive new occurrence,
architecture validation for the authority cutover and ONE native-enabled root `test javadoc` join.
No benchmark campaign or duplicate review suites. CPU0029's eight diagnostic
ratios 0.88–5.84 have unequal invocation boundaries and unresolved warm-up/JIT; speed parity is unproven.

## Documentation, review and planning result

Wegener: APPROVE as Draft proposal, no blocking findings; USER/authority gates remain.
Planning checks: delta links/anchors, size, `git diff --check`, protected hashes PASS. Planning/general
profiles and targeted glossary review complete; no glossary/authority edit. No Gradle/timing run.
Implementation not started; Draft/None frontiers unchanged. Future Javadoc/docs/ADR review required.
