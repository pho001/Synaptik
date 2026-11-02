# General differentiable scan: accepted decision and implementation program

<a id="general-differentiable-scan-proposed-decision-and-program"></a>

## Purpose and status

Coordinate fixed-count scan over ordinary Tensor algebra, with Compiler-owned automatic differentiation
(AD): source body → scoped capture → one-owner prepared loop, plus a sibling reverse scan for AD.
Generic scan is selected (2026-10-21), NOT general while. Revised planning review: APPROVE.
After the concrete ScanBody/reverse-scan/canonical-residual proposal and source-checked framework
comparison, the user said “ok, tak tedy pokracuj”. Main explicitly records this as approval of
that proposal and its authority cutover, not arbitrary control flow or all implementation slices.
The [decision brief](modules/model/tasks/0034-general-differentiable-scan-decision.md) is **Complete
for authority decision/cutover only**, accepted by main on 2026-10-21 after Ramanujan the 2nd's final
independent Class C APPROVE, zero blocking findings. Model0035 source/IR foundation is **Complete**,
accepted by main 2026-10-22 after pre-code authority approval, final independent code/docs APPROVE
and the named compatibility checkpoint.
[Planning0008](modules/planning/tasks/0008-general-scan-body-capability-query.md) is **Complete**,
accepted by main 2026-10-22 after final independent Feynman APPROVE and exact compatibility gates,
for tested body-bearing capability transport only.
[Compiler0011](modules/compiler/tasks/0011-general-scan-capture-and-first-order-ad.md) is Complete:
scoped capture/inference, joint first-order AD and final body queries/demand, accepted by main
2026-10-22 after independent Class C code/docs APPROVE and compatibility checkpoint.
[Prepare0010](modules/prepare/tasks/0010-general-scan-projection-and-resource-handoff.md) is Complete:
exact scoped body/roles/demand validation and resource handoff, accepted by main 2026-10-22 after
Huygens final independent Class C code/docs APPROVE and the exact compatibility checkpoint.
CPU frontier: None. [CPU0037](backends/cpu/tasks/0037-general-scan-dense-read-views.md#result) is Complete,
accepted by main 2026-10-24 after independent review and final gates; CPU0030–0033 and low-level join Complete;
At that all-valid checkpoint, Prepare/Planning/Model/Compiler/NN frontiers were None; the current
length-aware authority/Compiler status and nearest Prepare planning gate are recorded below.
Compiler0012 is Complete as recorded below.
Bounded CPU F32 pointwise forward + actual
automatic reverse [CPU0030](backends/cpu/tasks/0030-general-scan-f32-pointwise-vertical.md) is Complete,
accepted by main 2026-10-22 after independent review and focused checkpoint PASS.
[CPU0031](backends/cpu/tasks/0031-general-scan-f32-broadcast-sum-layout-closure.md#result) is Complete
for bounded F32 broadcast/SUM/layout forward and actual reverse, accepted by main 2026-10-22,
14:38+ UTC after final independent code/docs APPROVE (0 P1/P2), module and named checkpoint PASS.
[CPU0032](backends/cpu/tasks/0032-general-scan-f32-matmul-and-reverse.md#result) is Complete:
rank-two F32 MATMUL and actual Compiler reverse accepted by main 2026-10-22, 17:09+ UTC after
independent code/docs and test-addendum APPROVE, CPU module/Javadoc and named checkpoint PASS.
[CPU0033 full F64 domain closure](backends/cpu/tasks/0033-general-scan-f64-domain-closure.md#result) is Complete,
accepted by main after independent code/docs and identity-test addendum approval, final CPU/Javadoc
and source-matched native root PASS; low-level join Complete, no successor launch. The brief owns
counts, initial root failure/test-only fix and unexplained timing limitation; no parity/JIT-cause claim.
No new production type. Padding/low/vector/parallel/OpenBLAS remain separately gated Draft;
CP0013 persistence is now Complete below.
[NN0030 owned Model decision](extensions/nn/tasks/0030-general-scan-owned-model-admission-decision.md)
is Complete for the explicitly approved limited direction/planning only. It selects existing
graph/defineTensor and compute/configureTraining/train owners,
with initially tested all-valid canonical static-T F32/F64 CPU execution, not padding or body wire.
Model0035 PURE-body source validity and applicable dtypes remain independent of execution support;
cold copy/rebind needs its own exact constructor-backed admission, never a CPU capability predicate.
[Compiler0012 Result](modules/compiler/tasks/0012-general-scan-cold-source-reconstruction.md#result) records
Complete bounded cold reconstruction, independent code/docs APPROVE and main's final gates (2026-10-23);
historical pre-code/Ready verification is retained.
[CP0012](extensions/checkpoint/tasks/0012-general-scan-in-memory-copy-and-rebind.md#result) is Complete
for attachment-preserving in-memory copy/rebind after independent review and scoped gates, with
body wire closed and the unrelated missing-native-environment root exception recorded, not root green.
The separately user-approved [NN0031 Result](extensions/nn/tasks/0031-general-scan-owned-model-workflow.md#result)
records Complete owned workflow qualification, independent review, final NN module/Javadoc and main's
single configured native-enabled root PASS (2026-10-23). Required Metal TrainingSession/native cases
executed unskipped/pass with the existing library; CP0012's historical environment exception required
explicit configuration, not native/gating code changes. At that checkpoint NN/Compiler/Checkpoint/other
frontiers were None; CPU0037 is separately Complete above. CP0013 body wire is Complete below; padding,
fixed-scan BPTT and low-type execution stay gated.
[Compiler0010](modules/compiler/tasks/0010-rnn-tanh-scan-bptt.md)
is Superseded, no fixed public adjoint selected or existing kernel/API retired.

The brief names [source evidence](modules/model/tasks/0034-general-differentiable-scan-decision.md#files-and-symbols)
and [authority headings](modules/model/tasks/0034-general-differentiable-scan-decision.md#contracts).
The enacted [scan semantic owner](../architecture/contracts/recurrent-scan.md#general-differentiable-scan)
and [ADR 0045](../design/decisions/0045-general-differentiable-scan.md) record the accepted bounded
exception, with no new contract/module/dependency edge. This program is non-authoritative
coordination. Sagan the 2nd's earlier planning APPROVE is distinct from completed authority review
and main acceptance. The [Model0035 brief](modules/model/tasks/0035-general-scan-source-and-compiled-body.md)
owns the executable scope, seven integration fields and serial clean-context gates.

<a id="proposed-source-body-and-minimum-apiir"></a>

## Accepted source body and minimum API/IR

The six Model types and source/compiled attachments below are implemented by Model0035;
package prefix `io.github.pho001.synaptik.`. No additional layer/type is selected by this closure.

- `model.tensor.ScanBody`: immutable ordered ordinary Tensor formals `(carry, step, invariants)` and
  roots `(nextCarry, emissions)`, plus a derived ordered retained-value inventory. This is source
  expression containment, NOT a second DAG, captured graph or runnable closure.
- `model.tensor.Scan` static `apply(initialCarry, sequences, invariants, body, reverse)` and
  `model.tensor.ScanResult(finalCarry, outputs, residuals)`: explicit lists of canonical Tensors.
  Construct formals once with TensorFactory; retain no callback.
- `model.operation.scan.ScanKind` / `ScanAttrs`: fixed-count meaning and scalar header only:
  inferred static time count T, carry/sequence/invariant/emission/residual counts and reverse flag.
  Axis is initially zero. No Tensor, graph, registry key, executable or saved runtime values in attrs.
- Extend existing TensorProducer with `Optional<ScanBody>` (empty for ordinary/fixed operations),
  validated before canonical output allocation; extend CompiledNode with
  `Optional<model.graph.CompiledScanBody>`. Keep old constructor forms for empty attachments.
- `model.graph.CompiledScanBody`: existing CompiledGraphModel, ordered role-result ValueId
  bindings and ordered demanded outer-port indices. It is the ONE captured body representation.
  Graph inputs are formals in declared order, including unused formals. Graph outputs are unique;
  role bindings may repeat (e.g. emit the same h as next carry) without duplicate graph values.

Formals are identity-distinct, provenance/storage-free, static, canonical and disjoint from actual
outer bindings. Remove axis zero from each sequence descriptor for
the step formal; carry/invariant formals match actual shape/dtype/gradient eligibility, with canonical
step storage geometry. Actual bindings may repeat exact identities across roles; never deduplicate
positions. Body leaves must equal the declared formal set by identity, except unused formals are
allowed; hidden scalar Tensor constants must become explicit invariants (scalar operation attrs remain
ordinary algebra). Reject free leaves, host-backed constants, RNG, mutation and nested/recursive bodies.
The same immutable ScanBody template and its producers may serve multiple scan occurrences, each
captured independently. Reject producer escape into the outer graph or sharing across DIFFERENT
lexical templates, not reuse of one template.
No outer ValueId appears inside a body. Capture assigns IDs once per scope; numeric IDs may coincide
across scopes and only the containing CompiledScanBody qualifies them. Ordinary TensorIds remain global.

Required first completed CPU domain (Complete through CPU0033): homogeneous FLOAT32 AND FLOAT64, canonical boundaries, static T≥0,
at least one carry/sequence, zero or more invariants/emissions, shared T and fixed body shapes/dtypes.
Start with pure
ADD/SUB/MUL/DIV (tensor and existing scalar forms), NEG/TANH/SIGMOID, rank-two MATMUL and rank-two
`[1,0]` PERMUTE under existing guards. Reuse ElementwiseGradientRules' ordinary DIV derivatives,
not new masks/numerics. Inventory the WHOLE generated closure: broadcast reductions/sumToShape,
expands, views/contiguous materialization, constants and any required casts. Qualify NaN/Inf,
signed zero, zero divisors, overflow/underflow and empty reductions under current family rules;
unsupported occurrences reject, not blanket all-op admission. F32 was the initial working case.
BFLOAT16/FLOAT16 remain explicit follow-up qualification under existing working/rounding/saved-state
semantics, not abandoned and not whole-model conversion. Emitter availability grants no dtype support.

User-approved clarification (2026-10-22), enacted and independently accepted before Model0035 code: Model construction
admits existing valid PURE Tensor operations/applicable Model dtypes, not just this CPU domain (e.g.
FLOAT16 TANH or F32 RELU); unknown/custom semantics and invalid/effectful bodies still reject.
Let G be OR of actual carry/sequence/invariant gradient flags: every differentiable outer final-carry,
emission and residual descriptor gets G, nondifferentiable outputs false, even with unused inputs/T=0.
Formal flags still match actuals. This preserves cross-time eligibility, not an AD/backend guarantee.
Model0035 pins `h0=false`, `x=true`, `next=h+xStep`, `emit=h` as the temporal-eligibility regression;
the six-type Model foundation does not implement downstream AD, capture or execution.

Conceptual example (not API): F32 `h0=0`, `x=[1,2]`, invariant `w=2`, formals `(h,xStep,wStep)`,
next/emit `h+xStep*wStep` give `[2,6]`, final `6`. Undeclared w rejects. Reverse emits at original indices.

<a id="proposed-saved-output-bridge-and-ad"></a>

## Accepted saved-output bridge and AD

Select **construction-time canonical residual ports**, a new Model semantic protocol plus a
Compiler primal-binding rule. Residual ports name body values available for saving to reverse formulas.
This is Synaptik's immutable-producer design choice, not an industry requirement. The subsequent
user approval includes this output/API policy with demand-only history allocation. Metadata is not
allocated history; ordinary users write body algebra and do not manually select saved tensors.

1. ScanBody derives retainedValues by deterministic producer postorder/ascending canonical output
   slot, identity-unique, including unused siblings and each carry formal (previous state).
   Step/invariant formals use explicit outer inputs instead of duplicate saved sequences.
   Every other reachable body value receives a latent slot; Model selects no derivative rule.
2. Scan.apply reserves canonical residual descriptors/wrappers at original construction after
   finalCarry/emissions, NOT stacked storage. Only publication/downstream consumers demand history; internal uses need
   step temporaries, not history. Propagate ordinary-use demand into query/Prepare and revalidate
   after every rewrite. Live histories use original time indices (prior carry BEFORE the step).
   Aliased roles share body value bindings; outer wrappers remain distinct. Only live saved sites
   restrict fusion/recompute. Dead ports allocate/store ZERO history and retain fusion freedom,
   INCLUDING the generated reverse scan's own residuals. This is new liveness/Prepare work, NOT
   existing sibling-DCE or an opaque saved-workspace handshake.
3. Accepted package-private `compiler.ScanGradientRules` inventories the body, combines cotangents
   for all selected finalCarry/output ports ONCE per scan, then invokes existing Compiler rules.
   Extend selected rule entry points with an explicit compile-local primal-operand binding function:
   exact original body Tensor → reverse-body formal backed by its residual/step/invariant input.
   Existing TANH's saved output, for example, resolves to the residual formal, NOT recomputed tanh.
   The identity contribution maps still address original body Tensors; every formula operand that
   denotes a primal must resolve through the binding. This is a new compiler rule protocol, not
   today's rules unchanged. No ValueId→Tensor AD map, captured-graph differentiation or second algebra.
   Close derivative constants (including TANH/SIGMOID oneLike scalar leaves) as explicit non-gradient
   invariant formals bound to outer Compiler-created splats through existing constant ingress;
   expand remains ordinary body algebra. Zero seeds/accumulator initializers use registered outer
   splats/expressions: bind carry initializers through carry formals and body constants through
   invariant formals, NEVER undeclared free leaves.
4. Construct an ordinary sibling Scan.apply in opposite traversal, with cotangent carry initialized
   from finalCarry seeds, per-step output seeds and residual sequences as explicit inputs, and
   shared parameters as invariants. Its body is existing Tensor derivative algebra. Carry parameter
   sums initialized to exact typed zero, add each positional contribution once in traversal order;
   return dSequence emissions, final dInitialCarry and parameter sums. Sum tied actual-role and
   cross-occurrence contributions afterward with ordinary Tensor.add, never identity-deduplicate terms.
5. One combined capture includes ORIGINAL forward wrappers and generated reverse scan, each body
   captured once in its own scope. Public TensorId/publication roles are unchanged; residual edges
   enforce forward-before-reverse liveness. Exact stored residual sites survive layout closure,
   reindexing, DCE/CSE and fusion; no merging scan occurrences or crossing saved boundaries.

Accepted `compiler.ScanBodyInference` independently validates scoped descriptors/roles. Residual
ports preserve true gradient eligibility; reject differentiation rooted at residual ports, unsupported
body rules and every scan higher-order/createGraph request before derivative allocations, including
unsupported occurrences on otherwise unrelated original branches. Initial first-order support does
not admit fixed RNN/GRU/LSTM to BPTT. Missing output seeds use explicit typed zeros, both carry and
sequence seeds may coexist. T=0 executes no body: empty emissions/residuals, exact initial final
carry, dInitialCarry=dFinalCarry, zero invariant sums. Empty dimensions follow constituent semantics;
an empty dimension does not justify skipping a nonempty reduction or unsupported operation.

Rejected: postcapture augmentation/replacement, replay (inexact saved sites), manual AD recipes,
fixed RNN adjoint, attrs/registry body or second executor. Demanded conservative residuals cost
O(T × retained elements) memory and inhibit fusion. More selective saving/recompute requires review.

<a id="proposed-capability-preparation-and-padding-gates"></a>

## Accepted capability, preparation and padding gates

Extend existing Planning-owned OperationCapabilityQuery with `Optional<CompiledScanBody>`; retain
three-argument empty-body construction. Compiler supplies validated complete scoped body/ports,
not source Tensors. Provider boolean supports(query) must verify the header AND every body occurrence,
descriptor, residual/binding requirement and generated topology. Ordinary true child queries are
necessary but insufficient if the loop binder cannot execute them. Planning forwards semantic facts
and selects ONE owner for a complete scan body; no routes, mixed-owner body partitioning or backend AD.

PrepareContext/PartitionDag preserve the attachment and validate static local descriptors separately
from outer IDs. CPU analysis prepares the body once and declares exact resources; only finalization
AFTER assigned slots emits an OUTER TIME LOOP + existing typed body emitters at generation time.
No runtime graph interpreter, per-op lookup, per-step callback dispatch/object allocation,
or T-fold source/captured graph/bytecode growth.

Current evidence: CpuLoopEmitter.emit(List<CpuAccessPlan>, Consumer<State>) composes at generation
time; CpuScalarEmitter.emit(ir,instruction,locals) writes typed locals; CpuClassFileKernelGenerator's
scalarBody (around line 201) combines carrier loads, instructions and stores. Reuse scalar/vector/
carrier arithmetic only where admitted; no vector/parallel promise. CpuKernelIr's instruction list
is pointwise, CpuMatmulEmitter reads fixed entry slots/geometry, and CpuCarrierEmitter's
firstGeneratedLocal/layout block assumes the entry application binary interface (ABI). They are NOT
all plug-and-play: specify invocation-vs-step setup and carrier/local/base/range bindings explicitly.
Hoist invariant setup outside time; prefer direct time-dependent offsets/addresses into sequence,
emission and residual storage. Select/materialize only when necessary; justify every staging copy,
carry bank and body temporary. Emission removes neither live saved histories nor all MATMUL temporaries.
Declare checked staging/carry/temporary/history/output bytes BEFORE slot assignment. One RunState
owns mutable memory, changing addresses/carry/geometry arrays; none live in shared immutable prepared
state. No new Engine/Runtime owner or runtime tape.
Future CPU brief names every added type; no speculative generic control-flow infrastructure.
Direct static helpers are allowed when technically needed with clean Java oracle rationale, not a
one-giant-method mandate or hidden fallback overhead. Generic no-copy/JIT/performance parity is unproved.
CpuKernelIr identity excludes extents/slots/run bindings: distinguish baked/runtime scan facts and
key every baked fact (including body/roles/saved-port demand); reject stale format/meaning artifacts.
CPU0030 implements the bounded F32 pointwise family at global/family identity 71; no broader type support follows.

Current symbolic implementation metadata: [CPU0039](backends/cpu/tasks/0039-symbolic-scan-fixed-geometry-specialization.md)
specializes coefficient-proved fixed extents, byte strides and frame/offset facts in the existing
CPU owners while actual B remains dynamic. Initial scalar and evolving dense carry mappings stay
distinct; affine offset sharing is generation-only. Symbolic int K uses the existing checked
extent admission, with long byte addressing. The sole baked payload advances to
`|scan-symbolic-batch-v2=` in both specialization streams, invalidating symbolic baked v1
artifacts. Canonical `:symbolic-batch-v1`,
schema/family 71 and static shaped/MATMUL v4, pointwise/fixed/ordinary keys and generated bytes
remain unchanged. The [CPU guide](../backend-guide/cpu-backend.md#symbolic-scan-cpu-execution)
explains these implementation facts; they add no pipeline or performance guarantee.

First slice has NO per-row runtime lengths; old four-dtype/layout fixed routes retain padding/reverse.
Padding-aware NN migration needs a separate row-skipping design for zero/full/mixed lengths,
valid-prefix reverse, dense positive-zero padding and exact initial state for empty rows. It must
branch BEFORE any padded arithmetic/read of padded primal or cotangent values (NaN/Inf included).
Result masking is not skipping. Whole-batch generic bodies may mix rows, so adding lengths is a real
semantic decision, not merely another input. Do not imply legacy padded scans gain gradients now.

Current next planning priority is [true per-run symbolic batch](dynamic-batch-execution.md) and
[Prepare0012 DESIGN proposal](modules/prepare/dynamic-batch-geometry-contract-proposal.md),
Complete as reviewed DESIGN delivery: independent Class C APPROVE, zero open P1/P2; MAIN delivery
accepted 2026-10-24. User architecture acceptance remains pending; separately authorized authority
cutover is next. `DesignAuthorized=true; ImplementationAuthorized=false`; no Runnable/Ready
implementation frontier or successor brief.
This changes no completed scan result or static contract: ordinary dynamic-B CPU forward/backward
precedes general Scan extension; static length skipping remains an independent, separately gated
capability. No feature execution or conflicting CPU binder/resource ABI change is authorized here.

## Immediate length-aware authority and source foundation

Model0036 remains Complete for reviewed proposal delivery. The subsequent user approval selects
the [length-aware design](length-aware-general-scan.md) and immediate authority plus first Model
foundation. [Model0037](modules/model/tasks/0037-length-aware-general-scan-authority.md) was verified
Ready for this authority-only execution and is now Complete: Hume's final overall independent
Class C APPROVE, zero open P1/P2, and MAIN acceptance on 2026-10-24.
[Model0038](modules/model/tasks/0038-length-aware-scan-source-foundation.md)
is Complete, MAIN-accepted 2026-10-24 after independent Class C/API/docs APPROVE, zero open P1/P2,
and combined compile/architecture/root retry PASS after a test-only ledger repair. Model frontier None;
Compiler0013 below is Complete, MAIN accepted 2026-10-24. The
[sole scan owner](../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan) defines
accepted future semantics; [ADR0049](../design/decisions/0049-length-aware-general-scan.md) records
the decision. Model0038 implements source header/API/roles/attachments and explicit old-owner
rejection only, not length-mode compile/AD/runtime/cold/wire capability. Beyond the immediate
Compiler packet below, owner phases remain separately gated coarse dependencies, not a task cascade.
The immediate successor is [Compiler0013](modules/compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md),
launched after Popper final APPROVE (0 P1/P2) and MAIN dependency/Ready acceptance 2026-10-24. Complete
Model0037/0038 + Compiler0011 + Planning0008 feed this existing-owner inference/first-order
AD/query/demand scope. No new type/layer; user “ok, pokracuj dal” authorizes this implementation.
MAIN recorded admission before executor launch: ImplementationAuthorized=true. Implementation is
now frozen; independent code/docs APPROVE (0 P1/P2), worker module, MAIN compatibility and final
docs/Javadoc gates PASS. MAIN accepted Complete 2026-10-24. Cold capture/
reconstruction, CPU and codec remain closed; Prepare0011's separate static projection scope is
below. MAIN owns serial integration; Compiler Ready frontier None, no automatic successor launch.

The Prepare0011 launch admission and review state are recorded:
[Prepare0011](modules/prepare/tasks/0011-length-aware-scan-projection-and-resource-handoff.md) was Ready
after independent Tesla planning APPROVE, zero open P1/P2; MAIN metadata accepted
2026-10-24. Complete Model0037/0038 +
Compiler0013 + Planning0008 + Prepare0010 feed static lengths/formal separation, mixed reverse
role/demand projection and existing staged resource handoff. No new type/API/dependency or
row-algebra proof owner. At Prepare0011 completion, Prepare Ready frontier was None;
ImplementationAuthorized=true records completed
Prepare0011 only, authorized by the user's latest “pust se do prace” request.
MAIN verified admission 2026-10-24 12:25 UTC; clean sole writer started 12:31 UTC, now Complete.
Worker 71-focused/92-full tests and Javadoc PASS, zero failures/errors/skips; MAIN SPI-consumer
compile/four retained guards PASS. Independent code/docs APPROVE, zero open P1/P2; MAIN accepted 2026-10-24 13:00:15 UTC.
No new launch, downstream tasks, commit or push is authorized.
CPU mixed-boundary/preflight/skipping, cold/v5 and NN remain separate coarse gates, no detailed cascade.

No `.rnn`/`.rnnScan` migration, higher-order waiver or model/loss-wide padding mask follows.
[NN0032](extensions/nn/tasks/0032-fluent-rnn-general-scan-migration.md) retains the confirmed advanced
second-order CPU compile regression, singleton-CONCAT copy and unproved mixed-engine compatibility.
MAIN owns serial shared integration, Parallel group None; authority and Model0038 acceptance are
Complete. Combined named compile/architecture and existing-native root test/Javadoc passed on
retry after the sole test-serialization failure; both performance flags false, no rebuild/campaign.

## Serial DAG and acceptance gates

Current persistence checkpoint: [CP0013 Complete](extensions/checkpoint/tasks/0013-general-scan-graph-persistence-decision.md#result),
user-approved selective v4/existing cold inventory. Main accepted final Cicero APPROVE and final
module/Javadoc, consumer, architecture and one configured root PASS on 2026-10-23. The same cohesive
capability delivers wire/save-load/tests/docs and loaded consumer proof; no new type/API or cascade.
[NN0032 fluent RNN migration](extensions/nn/tasks/0032-fluent-rnn-general-scan-migration.md) stays Draft;
`.rnn` is unchanged. Audit finds CPU Conv1d-prefix and singleton-CONCAT reverse-seed dense-view
rejections, not a proved Metal-only regression; mixed execution untested. CPU0037 now closes the
narrow read gap, accepted by main 2026-10-24 after Newton final APPROVE and all gates; no new
performance claim. Remaining order: cold output descriptor/remaining compatibility → fluent authority →
separately Ready NN migration. Fluent/CPU source frontiers None; length-aware authority gate above
is distinct. Parallel group None, MAIN owns shared integration.
CP0013 retains its historical verified 292-path launch base and empty index in the brief.
GRU/LSTM migration remains later Draft follow-up without a speculative detailed queue.

Main owns integration/docs at main=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
plus all accepted uncommitted authority/Model0034/0035, Planning0008, Compiler0011, Prepare0010 and CPU0030–0033 work;
Parallel group None. Authority cutover,
All-valid Model source/IR and Planning query transport are Complete; Planning frontier None.
Current Model0037/0038 authority/foundation status is above.
Their briefs own accepted evidence. Compiler0011 is Complete, accepted by main 2026-10-22 after
independent code/docs APPROVE and compatibility checkpoint; Compiler0012 Complete, frontier then None.

| Slice | Depends on | State / bounded result |
|---|---|---|
| Model0034 decision/cutover | Complete declared baseline + user-approved concrete proposal | Complete: authority/explanations/ADR, final independent Class C APPROVE and main acceptance on 2026-10-21; no Java. |
| [Model0035 source/IR](modules/model/tasks/0035-general-scan-source-and-compiled-body.md) | Model0034 Complete; Ready verification 2026-10-22 | Complete: bounded source/IR foundation accepted by main 2026-10-22 after independent authority/code/docs approval and compatibility checkpoint; no downstream execution. |
| [Planning0008 query](modules/planning/tasks/0008-general-scan-body-capability-query.md) | Planning0001–0007 + Model0034/0035 Complete | Complete: optional compiled-body transport, old-query compatibility and fail-closed contract tests; final independent APPROVE and main compatibility gates accepted 2026-10-22; no real backend support/execution. |
| [Compiler0011 AD/capture](modules/compiler/tasks/0011-general-scan-capture-and-first-order-ad.md) | Model0034/0035 + Planning0008 Complete; existing Compiler baseline | Complete: scoped capture/inference, joint first-order AD and final body queries/demand; main acceptance 2026-10-22 after independent code/docs APPROVE and compatibility checkpoint. |
| [Prepare0010 projection](modules/prepare/tasks/0010-general-scan-projection-and-resource-handoff.md) | Prepare0001–0009/0003A + Model0034/0035 + Planning0008 + Compiler0011 Complete | Complete: exact scoped body/roles/demand validation and existing resource handoff; Huygens final Class C code/docs APPROVE and main exact compatibility checkpoint accepted 2026-10-22; no new type or real execution. |
| [CPU0030 vertical milestone](backends/cpu/tasks/0030-general-scan-f32-pointwise-vertical.md) | Model0034/0035 + Planning0008 + Compiler0011 + Prepare0010; CPU0022/0027/0028/0029 Complete | Complete: bounded F32 pointwise forward/actual automatic reverse, accepted by main 2026-10-22 after independent review and focused checkpoint PASS. |
| [CPU0031 F32 broadcast/SUM/layout closure](backends/cpu/tasks/0031-general-scan-f32-broadcast-sum-layout-closure.md#result) | CPU0030 and accepted predecessor baseline Complete | Complete: main accepted 2026-10-22, 14:38+ UTC after final independent code/docs APPROVE (0 P1/P2), module and named checkpoint PASS. |
| [CPU0032 F32 MATMUL + actual reverse](backends/cpu/tasks/0032-general-scan-f32-matmul-and-reverse.md#result) | CPU0031 and accepted foundation baseline Complete | Complete: main accepted 2026-10-22, 17:09+ UTC after independent code/docs and test-addendum APPROVE, CPU module/Javadoc and named checkpoint PASS. |
| [CPU0033 F64 domain closure](backends/cpu/tasks/0033-general-scan-f64-domain-closure.md#result) | Model0034/0035 + Planning0008 + Compiler0011 + Prepare0010 + CPU0030–0032 Complete | Complete: full homogeneous F64 algebra/actual reverse accepted by main after independent reviews and final gates; unexplained timing limitation retained. |
| Low-level program join | All low-level slices reviewed including domain completion | Complete: main's source-matched native root test/Javadoc PASS after reviewed stale identity-test correction; no successor launch. |
| [NN0030 owned Model decision](extensions/nn/tasks/0030-general-scan-owned-model-admission-decision.md) | Complete low-level join + existing owned Model/source/Training baseline | Complete for approved direction/planning only; initial F32/F64 all-valid compute/train target, not implemented. |
| [Compiler0012 cold reconstruction](modules/compiler/tasks/0012-general-scan-cold-source-reconstruction.md#result) | Approved NN0030 + Complete Compiler0011/foundation | Complete: bounded cold body reconstruction, independent code/docs APPROVE and main final gates accepted 2026-10-23; Checkpoint/NN still fail-closed. |
| [CP0012 in-memory copy/rebind](extensions/checkpoint/tasks/0012-general-scan-in-memory-copy-and-rebind.md#result) | NN0030; Compiler0012; accepted cold/low-level baseline Complete | Complete: attachments and independent body IDs preserved, wires closed, independent review/scoped gates accepted; unrestricted root not green. |
| [NN0031 owned workflow](extensions/nn/tasks/0031-general-scan-owned-model-workflow.md#result) | NN0030; Compiler0012; CP0012; accepted CPU domain/low-level join and owned Model/Engine/Training baseline Complete | Complete: bounded owned graph/defineTensor compute/first-order train, clean implementation, independent Class C review and final module/Javadoc passed. |
| Owned-workflow program join | NN0031 reviewed/documented and final NN module gate | Complete: main's single configured native-enabled root PASS, both scan performance flags false, required Metal TrainingSession/native cases unskipped/pass; no native builds/gating fixes. Evidence in NN0031 Result. |
| [CP0013 general-body persistence](extensions/checkpoint/tasks/0013-general-scan-graph-persistence-decision.md#result) | Compiler0012; CP0010/0011/0012; NN0031/CPU0030–0036/joins Complete; proposal/authority accepted | Complete: selective v4/scoped cold bodies and loaded consumer proof, final Cicero APPROVE and main module/Javadoc/consumer/architecture/configured-root PASS. No successor. |
| [CPU0037 dense actual read views](backends/cpu/tasks/0037-general-scan-dense-read-views.md#result) | CPU0036 and accepted foundation/low-level join Complete | Complete; main accepted Newton final APPROVE and CPU/Javadoc, named and configured-root gates on 2026-10-24. CPU0036 limits retained; no NN migration. |
| [Model0037 length-aware authority](modules/model/tasks/0037-length-aware-general-scan-authority.md) | Model0036 Complete + subsequent user design approval | Complete authority only; Hume final overall Class C APPROVE, zero open P1/P2; MAIN accepted 2026-10-24. |
| [Model0038 source/IR foundation](modules/model/tasks/0038-length-aware-scan-source-foundation.md) | Model0037 independent approval/MAIN acceptance Complete | Complete; MAIN accepted 2026-10-24 after independent APPROVE and combined compile/architecture/root retry PASS; source/IR and fail-closed guards only. |
| [Compiler0013 length-aware capture/first-order AD](modules/compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md) | Model0037/0038, Compiler0011 and Planning0008 Complete; cold baseline preserved | Complete; MAIN accepted 2026-10-24 after independent code/docs APPROVE, worker module, compatibility and final docs/Javadoc PASS; [evidence](modules/compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md#result). No runtime/cold/wire support. |
| [Prepare0011 length-aware projection/resource handoff](modules/prepare/tasks/0011-length-aware-scan-projection-and-resource-handoff.md#result) | Model0037/0038 + Compiler0013 + Planning0008 + Prepare0010 Complete | Complete, MAIN accepted 2026-10-24; static metadata implemented, focused 71/full Prepare 92 tests, zero failures/errors/skips, Javadoc and MAIN SPI/four-guard checkpoint PASS. Independent code/docs APPROVE, zero open P1/P2; at Prepare0011 completion, Prepare Ready frontier was None. CPU execution/cold/v5/NN stay held. |
| Padding-aware NN integration | Low-level join + separately approved padding packet | Draft, independently gated; no lengths/row-skipping grant. |

Approved decision: specific ScanBody/producer+node attachment, construction-time residual-output API
with demand-only history and exact primal-binding protocol. Do not ask for the same approval again.
Model0034 was the sole authorized authority frontier, verified on 2026-10-21 at the base above with
Complete declared dependencies and inactive serialized conflicts. Its authority execution, review
and main acceptance are complete. Model0035 and Planning0008 are also Complete; Planning frontier
None. Compiler0011, Prepare0010 and CPU0030–0033 bounded capabilities and low-level join are Complete;
At that low-level checkpoint CPU/Compiler/NN/Checkpoint frontiers were None; Compiler0012 and
CP0012, NN0031 and the configured native owned-workflow join, then CP0013 persistence and its
configured root checkpoint are Complete. CPU0037 is also Complete; those all-valid frontiers remain
closed. Current length-aware Model0037/0038 status is above.
Prepare0010's brief retains historical authorization/Ready admission, all seven integration fields
and final independent approval/main checkpoint evidence. Serial, Parallel group None; accepted dirty
changes retained, shared-document owner main. The bounded F32 pointwise forward/actual-reverse
[CPU0030 result](backends/cpu/tasks/0030-general-scan-f32-pointwise-vertical.md#result) owns completion evidence.
CPU0031's brief owns the final Result, seven integration fields and historical main launch verification.
CPU0032's brief owns final evidence, seven integration fields and historical main launch verification.
Its only added production type is generation-only `CpuGeneralScanMatmulEmitter` in the existing
CPU emit package; stages gain both operand dependencies and step bindings. Main completed shared
documentation integration and the named conformance/ledger/Engine plus targeted architecture checkpoint.
No other area master changes. Source-grounded split: CPU0030 used one common N and pointwise locals;
shape-changing reverse introduces SUM,
views and differently shaped parameter carries. Empty `[0,H]` steps still require nonempty `[H]`
SUM/parameter updates, so no global N=0 return. Ordinary aggregate and MATMUL emitters have fixed
entry ABIs, not plug-and-play step bindings. Completed CPU0031 supplies explicit per-stage binding/liveness
without another executor; CPU0032 MATMUL/actual F32 reverse is Complete, with full homogeneous F64
qualification Complete in [CPU0033](backends/cpu/tasks/0033-general-scan-f64-domain-closure.md#result).
The package map reuses existing CPU typed seams, with no duplicate F64 executor; CPU0037 uses the
existing lowering read predicate only. CPU0032 preserves whole-body rejection even T=0/dead, old/new carry versions,
exact demanded history and liveness-based reusable MATMUL cuts; no blanket no-materialization claim.
F64 is not a carrier toggle: double locals/eight-byte resources and no-DAZ/FTZ conformance have
independent review and completed join evidence. Existing Compiler rules suffice; no new architecture or AD is proposed.
Replacing the common-N restriction does not retire fixed routes or require automatic type deletion.
Root/native low-level join and NN0031's separate all-valid owned-workflow qualification/join are Complete.
Padding and low-type execution remain independently gated Draft; CP0013 persistence is now Complete.
If/while/general control-flow and associative/parallel scan are separate future decisions, NOT
prerequisites. Add only reuse seams needed here; no new AD algebra or architecture file.

Acceptance: template reuse/escape rejection, closed derivative constants, tied roles/roots, two carries,
both seed classes, shared parameters, saved rounding/special values, T=0/empties and higher-order rejection.
Completed Compiler0011 acceptance compares at least three steps with unrolled gradients using
non-gradient zero h0 and trainable w/u: INTERNAL carry cotangents must still propagate. Do not copy
current public preflight's non-gradient input rejection unchanged into local carry AD. This tests
the accepted reverse semantics, not a new AD system or numerical policy.
Vary T: source/captured/prepared object counts O(body), checked live saved bytes O(T); forward-only
undemanded history bytes/stores ZERO. Requested T-step emissions still require T-dependent output
storage: zero dead history does not imply O(1) total memory. Test overflow/overlap/cleanup/retry/concurrency, unsupported
interior/generated occurrence, missing body, supported children but unsupported binder, projection
and no runtime graph access. Fixed-route regression coverage remains.

CPU performance gate: use a specialized optimal clean Java algorithm/hot-loop/dataflow oracle;
inspect/decompile Class-File for accidental helpers/dispatch/allocation/copies. Run short targeted
warmed generated-vs-direct measurements with same dtype/carrier/layout/numerics/work and comparable
validation. Separate generation/class-load/JIT cold cost, entry/preflight end-to-end overhead and
kernel/step cost. No guaranteed identical assembly/zero penalty or inherited CPU0029 timing proof.
Fix concrete diagnosed avoidable overhead or explicitly justify it; do not relabel it acceptable fallback.
Completed low-level gates: worker focused tests and affected-module `test javadoc`; main's named
conformance/Engine checkpoints, `:testing:architecture-tests:test` for boundary risks and native-enabled
root `test javadoc`, including the reviewed test-only retry. CPU0033 Result owns exact reused/executed
evidence. No extra timing campaign, broad/five-fork campaign or CPU0029 parity claim; later slices own docs/Javadoc.

Size exception: this single cross-boundary coordination protocol exceeds 200 lines/15 KB to retain residual/scoping,
AD binding, resource and qualification semantics together. Executable module briefs will be smaller;
splitting this record would separate the accepted protocol and its serial acceptance gates.
