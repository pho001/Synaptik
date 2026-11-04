# [General differentiable scan](../glossary.md#general-differentiable-scan): accepted decision and implementation program

<a id="general-differentiable-scan-proposed-decision-and-program"></a>

## Reading context: a sequence of state updates

Here **Scan** means an application-defined [recurrent body](../reference/tensor/general-scan.md),
not the separate cumulative-sum operation. A simple body computes
$h_{t+1}=f(h_t,x_t)$: $t\in\{0,\ldots,T-1\}$ is a step index, $T$ the fixed step count,
$h_t$ the state entering that step, $x_t$ its input, and $f$ the Tensor expression supplied
by the application. Actual bodies may have several state and output ports.

The application describes the body; Compiler captures and differentiates it, Prepare hands
off its resources, and the selected backend executes the prepared loop. See
[who calls these stages](../explanation/framework.md),
[why reverse-mode differentiation follows the state dependency](../architecture/general-differentiable-scan.md#conceptual-forward-and-reverse-example),
and [the current Scan contract](../architecture/contracts/recurrent-scan.md#general-differentiable-scan).
The chronology below distinguishes an accepted design from each separately delivered capability.

## Purpose and status

Coordinate fixed-count [scan](../glossary.md#general-differentiable-scan) over ordinary [Tensor](../glossary.md#tensor) algebra, with [Compiler](../glossary.md#term-compiler)-owned [automatic differentiation](../glossary.md#autograd)
([AD](../glossary.md#autograd)): source body → scoped [capture](../glossary.md#term-capture-meanings) → one-[owner](../glossary.md#term-owner) prepared loop, plus a sibling reverse scan for AD.
Generic scan is selected (2026-10-21), NOT general while. Revised [planning](../glossary.md#term-planning) review: APPROVE.
After the concrete ScanBody/reverse-scan/[canonical](../glossary.md#term-canonical-form)-[residual](../glossary.md#scan-residual-port-and-history-demand) proposal and source-checked framework
comparison, the user said “ok, tak tedy pokracuj”. Main explicitly records this as approval of
that proposal and its authority cutover, not arbitrary control flow or all implementation slices.
The [decision brief](modules/model/tasks/0034-general-differentiable-scan-decision.md) is **Complete
for authority decision/cutover only**, accepted by main on 2026-10-21 after Ramanujan the 2nd's final
independent Class C APPROVE, zero blocking findings. Model0035 source/[IR](../glossary.md#term-intermediate-representation-ir) foundation is **Complete**,
accepted by main 2026-10-22 after pre-code authority approval, final independent code/docs APPROVE
and the named compatibility [checkpoint](../glossary.md#term-checkpoint).
[Planning0008](modules/planning/tasks/0008-general-scan-body-capability-query.md) is **Complete**,
accepted by main 2026-10-22 after final independent Feynman APPROVE and exact compatibility gates,
for tested body-bearing capability transport only.
[Compiler0011](modules/compiler/tasks/0011-general-scan-capture-and-first-order-ad.md) is Complete:
scoped capture/[inference](../glossary.md#term-inference), joint first-order AD and final body queries/demand, accepted by main
2026-10-22 after independent Class C code/docs APPROVE and compatibility checkpoint.
[Prepare0010](modules/prepare/tasks/0010-general-scan-projection-and-resource-handoff.md) is Complete:
exact scoped body/roles/demand [validation](../glossary.md#term-validation) and resource [handoff](../glossary.md#term-handoff), accepted by main 2026-10-22 after
Huygens final independent Class C code/docs APPROVE and the exact compatibility checkpoint.
[CPU](../glossary.md#term-cpu) frontier: None. [CPU0037](backends/cpu/tasks/0037-general-scan-dense-read-views.md#result) is Complete,
accepted by main 2026-10-24 after independent review and final gates; CPU0030–0033 and low-level join Complete;
At that all-valid checkpoint, [Prepare](../glossary.md#prepare)/Planning/[Model](../glossary.md#term-model)/Compiler/[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) frontiers were None; the current
length-aware authority/Compiler status and nearest Prepare planning gate are recorded below.
Compiler0012 is Complete as recorded below.
Bounded CPU [F32](../glossary.md#floating-point-precision-and-rounding) [pointwise](../glossary.md#term-pointwise) forward + actual
automatic reverse [CPU0030](backends/cpu/tasks/0030-general-scan-f32-pointwise-vertical.md) is Complete,
accepted by main 2026-10-22 after independent review and focused checkpoint PASS.
[CPU0031](backends/cpu/tasks/0031-general-scan-f32-broadcast-sum-layout-closure.md#result) is Complete
for bounded F32 broadcast/SUM/[layout](../glossary.md#layout) forward and actual reverse, accepted by main 2026-10-22,
14:38+ UTC after final independent code/docs APPROVE (0 P1/P2), [module](../glossary.md#term-module) and named checkpoint PASS.
[CPU0032](backends/cpu/tasks/0032-general-scan-f32-matmul-and-reverse.md#result) is Complete:
[rank-two](../glossary.md#axis-rank-scalar-vector-and-matrix) F32 [MATMUL](../glossary.md#matrix-multiplication--matmul) and actual Compiler reverse accepted by main 2026-10-22, 17:09+ UTC after
independent code/docs and test-addendum APPROVE, CPU module/[Javadoc](../glossary.md#term-javadoc) and named checkpoint PASS.
[CPU0033 full F64 domain closure](backends/cpu/tasks/0033-general-scan-f64-domain-closure.md#result) is Complete,
accepted by main after independent code/docs and [identity](../glossary.md#term-identity-meanings)-test addendum approval, final CPU/Javadoc
and source-matched native root PASS; low-level join Complete, no successor launch. The brief owns
counts, initial root failure/test-only fix and unexplained timing limitation; no parity/[JIT](../glossary.md#term-just-in-time-compilation-jit)-cause claim.
No new production type. [Padding](../glossary.md#padding)/low/[vector](../glossary.md#term-vector)/parallel/[OpenBLAS](../glossary.md#term-openblas) remain separately gated Draft;
CP0013 [persistence](../glossary.md#term-persistence) is now Complete below.
[NN0030 owned Model decision](extensions/nn/tasks/0030-general-scan-owned-model-admission-decision.md)<span data-glossary-navigation="true"> ([owned Model glossary](../glossary.md#model-module-buffer-and-scan-distinctions))</span>
is Complete for the explicitly approved limited direction/planning only. It selects existing
[graph](../glossary.md#graph)/defineTensor and compute/configureTraining/train owners,
with initially tested all-valid canonical static-T F32/[F64](../glossary.md#floating-point-precision-and-rounding) CPU execution, not padding or body wire.
Model0035 [PURE](../glossary.md#term-pure)-body source validity and applicable [dtypes](../glossary.md#data-type--datatype) remain independent of execution support;
cold copy/[rebind](../glossary.md#capture-graph-ports-and-reconstruction) needs its own exact constructor-backed [admission](../glossary.md#term-admission), never a CPU capability predicate.
[Compiler0012 Result](modules/compiler/tasks/0012-general-scan-cold-source-reconstruction.md#result) records
Complete bounded [cold reconstruction](../glossary.md#term-cold-reconstruction), independent code/docs APPROVE and main's final gates (2026-10-23);
historical pre-code/[Ready](../glossary.md#term-ready) verification is retained.
[CP0012](extensions/checkpoint/tasks/0012-general-scan-in-memory-copy-and-rebind.md#result) is Complete
for attachment-preserving in-memory copy/rebind after independent review and scoped gates, with
body wire closed and the unrelated missing-native-environment root exception recorded, not root green.
The separately user-approved [NN0031 Result](extensions/nn/tasks/0031-general-scan-owned-model-workflow.md#result)
records Complete owned workflow [qualification](../glossary.md#term-qualification), independent review, final NN module/Javadoc and main's
single configured native-enabled root PASS (2026-10-23). Required [Metal](../glossary.md#term-metal) [TrainingSession](../glossary.md#training-session--trainingsession)/native cases
executed unskipped/pass with the existing library; CP0012's historical environment exception required
explicit configuration, not native/gating code changes. At that checkpoint NN/Compiler/Checkpoint/other
frontiers were None; CPU0037 is separately Complete above. CP0013 body wire is Complete below; padding,
fixed-scan [BPTT](../glossary.md#term-backpropagation-through-time-bptt) and low-type execution stay gated.
[Compiler0010](modules/compiler/tasks/0010-rnn-tanh-scan-bptt.md)
is Superseded, no fixed public [adjoint](../glossary.md#term-adjoint) selected or existing [kernel](../glossary.md#kernel)/[API](../glossary.md#term-application-programming-interface-api) retired.

The brief names [source evidence](modules/model/tasks/0034-general-differentiable-scan-decision.md#files-and-symbols)
and [authority headings](modules/model/tasks/0034-general-differentiable-scan-decision.md#contracts).
The enacted [scan semantic owner](../architecture/contracts/recurrent-scan.md#general-differentiable-scan)
and [ADR 0045](../design/decisions/0045-general-differentiable-scan.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span> record the accepted bounded
exception, with no new contract/module/[dependency edge](../glossary.md#term-dependency-edge). This program is non-authoritative
coordination. Sagan the 2nd's earlier planning APPROVE is distinct from completed authority review
and main acceptance. The [Model0035 brief](modules/model/tasks/0035-general-scan-source-and-compiled-body.md)
owns the executable [scope](../glossary.md#term-scope), seven integration fields and serial clean-context gates.

<a id="proposed-source-body-and-minimum-apiir"></a>

## Accepted source body and minimum [API](../glossary.md#term-application-programming-interface-api)/[IR](../glossary.md#term-intermediate-representation-ir)

The six [Model](../glossary.md#term-model) types and source/compiled attachments below are implemented by Model0035;
package prefix `io.github.pho001.synaptik.`. No additional layer/type is selected by this closure.

- `model.tensor.ScanBody`: [immutable](../glossary.md#term-immutable) ordered ordinary [Tensor](../glossary.md#tensor) formals `(carry, step, invariants)` and
  roots `(nextCarry, emissions)`, plus a derived ordered retained-value inventory. This is source
  expression containment, NOT a second DAG, captured graph or runnable closure.<span data-glossary-navigation="true"> ([DAG glossary](../glossary.md#term-directed-acyclic-graph-dag))</span>
- `model.tensor.Scan` static `apply(initialCarry, sequences, invariants, body, reverse)` and
  `model.tensor.ScanResult(finalCarry, outputs, residuals)`: explicit lists of canonical Tensors.
  Construct formals once with TensorFactory; retain no callback.<span data-glossary-navigation="true"> ([callback glossary](../glossary.md#term-callback))</span>
- `model.operation.scan.ScanKind` / `ScanAttrs`: fixed-count meaning and scalar header only:
  inferred static time count T, carry/sequence/invariant/emission/residual counts and reverse flag.
  Axis is initially zero. No Tensor, graph, registry key, executable or saved runtime values in attrs.<span data-glossary-navigation="true"> ([registry glossary](../glossary.md#term-registry); [runtime glossary](../glossary.md#term-runtime))</span>
- Extend existing [TensorProducer](../glossary.md#producer--tensorproducer) with `Optional<ScanBody>` (empty for ordinary/fixed operations),
  validated before canonical output allocation; extend CompiledNode with
  `Optional<model.graph.CompiledScanBody>`. Keep old constructor forms for empty attachments.
- `model.graph.CompiledScanBody`: existing CompiledGraphModel, ordered role-result ValueId
  bindings and ordered demanded outer-port indices. It is the ONE captured body representation.
  Graph inputs are formals in declared order, including unused formals. Graph outputs are unique;
  role bindings may repeat (e.g. emit the same h as next carry) without duplicate graph values.<span data-glossary-navigation="true"> ([CompiledGraphModel glossary](../glossary.md#compiled-graph--compiledgraphmodel))</span>

Formals are [identity](../glossary.md#term-identity-meanings)-distinct, [provenance](../glossary.md#provenance)/storage-free, static, [canonical](../glossary.md#term-canonical-form) and disjoint from actual
outer [bindings](../glossary.md#term-binding). Remove [axis](../glossary.md#term-axis) zero from each sequence [descriptor](../glossary.md#term-descriptor) for
the step formal; [carry](../glossary.md#term-carry)/[invariant](../glossary.md#term-invariant) formals match actual [shape](../glossary.md#shape)/[dtype](../glossary.md#data-type--datatype)/[gradient](../glossary.md#term-gradient) eligibility, with canonical
step storage geometry. Actual bindings may repeat exact identities across roles; never deduplicate
positions. Body [leaves](../glossary.md#term-leaf) must equal the declared formal set by identity, except unused formals are
allowed; hidden [scalar](../glossary.md#term-scalar) Tensor [constants](../glossary.md#term-constant) must become explicit invariants (scalar [operation](../glossary.md#operation) attrs remain
ordinary algebra). Reject free leaves, [host](../glossary.md#term-host)-backed constants, [RNG](../glossary.md#graph-rng-state--graphrngstate), mutation and nested/recursive bodies.
The same immutable ScanBody template and its [producers](../glossary.md#term-producer) may serve multiple [scan](../glossary.md#term-cumulative-scan) occurrences, each
captured independently. Reject [producer](../glossary.md#term-producer) escape into the outer [graph](../glossary.md#graph) or sharing across DIFFERENT
lexical templates, not reuse of one template.
No outer [ValueId](../glossary.md#logical-valueid-versus-physical-memory-slot) appears inside a body. [Capture](../glossary.md#term-capture-meanings) assigns IDs once per [scope](../glossary.md#term-scope); numeric IDs may coincide
across scopes and only the containing CompiledScanBody qualifies them. Ordinary TensorIds remain global.

Required first completed [CPU](../glossary.md#term-cpu) domain (Complete through CPU0033): [homogeneous](../glossary.md#term-homogeneous) [FLOAT32](../glossary.md#term-float32) AND [FLOAT64](../glossary.md#term-float64), canonical boundaries, static T≥0,
at least one carry/sequence, zero or more invariants/emissions, shared T and fixed body shapes/dtypes.
Start with pure
ADD/SUB/MUL/DIV (tensor and existing scalar forms), NEG/[TANH](../glossary.md#term-tanh)/[SIGMOID](../glossary.md#term-sigmoid), [rank-two](../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../glossary.md#matrix-multiplication--matmul) and rank-two
`[1,0]` PERMUTE under existing guards. Reuse ElementwiseGradientRules' ordinary DIV [derivatives](../glossary.md#term-derivative),
not new [masks](../glossary.md#term-boolean-mask)/numerics. Inventory the WHOLE generated closure: broadcast reductions/sumToShape,
expands, [views](../glossary.md#term-view-meanings)/[contiguous](../glossary.md#term-contiguous) [materialization](../glossary.md#materialization), constants and any required casts. Qualify [NaN](../glossary.md#term-nan)/Inf,
[signed zero](../glossary.md#term-signed-zero), zero divisors, [overflow](../glossary.md#term-overflow)/[underflow](../glossary.md#term-underflow) and empty reductions under current family rules;
unsupported occurrences reject, not blanket all-op [admission](../glossary.md#term-admission). [F32](../glossary.md#floating-point-precision-and-rounding) was the initial working case.
[BFLOAT16](../glossary.md#term-bfloat16)/[FLOAT16](../glossary.md#term-float16) remain explicit follow-up [qualification](../glossary.md#term-qualification) under existing working/[rounding](../glossary.md#term-rounding)/saved-state
semantics, not abandoned and not whole-model conversion. Emitter availability grants no dtype support.

User-approved clarification (2026-10-22), enacted and independently accepted before Model0035 code: Model construction
admits existing valid [PURE](../glossary.md#term-pure) Tensor operations/applicable Model dtypes, not just this CPU domain (e.g.
FLOAT16 TANH or F32 RELU); unknown/custom semantics and invalid/effectful bodies still reject.
Let G be OR of actual carry/sequence/invariant gradient flags: every [differentiable](../glossary.md#derivative-and-chain-rule) outer final-carry,
emission and [residual](../glossary.md#scan-residual-port-and-history-demand) descriptor gets G, nondifferentiable outputs false, even with unused inputs/T=0.
Formal flags still match actuals. This preserves cross-time eligibility, not an [AD](../glossary.md#autograd)/[backend](../glossary.md#backend) guarantee.
Model0035 pins `h0=false`, `x=true`, `next=h+xStep`, `emit=h` as the temporal-eligibility [regression](../glossary.md#term-regression);
the six-type Model foundation does not implement downstream AD, capture or execution.

Conceptual example (not API): F32 `h0=0`, `x=[1,2]`, invariant `w=2`, formals `(h,xStep,wStep)`,
next/emit `h+xStep*wStep` give `[2,6]`, final `6`. Undeclared w rejects. Reverse emits at original indices.

<a id="proposed-saved-output-bridge-and-ad"></a>

## Accepted saved-output bridge and [AD](../glossary.md#autograd)

Select **construction-time [canonical](../glossary.md#term-canonical-form) [residual](../glossary.md#scan-residual-port-and-history-demand) ports**, a new [Model](../glossary.md#term-model) semantic protocol plus a
[Compiler](../glossary.md#term-compiler) [primal](../glossary.md#term-primal)-[binding](../glossary.md#term-binding) rule. Residual ports name body values available for saving to reverse formulas.
This is Synaptik's [immutable](../glossary.md#term-immutable)-[producer](../glossary.md#term-producer) design choice, not an industry requirement. The subsequent
user approval includes this output/[API](../glossary.md#term-application-programming-interface-api) policy with demand-only history allocation. [Metadata](../glossary.md#term-metadata) is not
allocated history; ordinary users write body algebra and do not manually select saved [tensors](../glossary.md#tensor).

1. ScanBody derives retainedValues by deterministic producer postorder/ascending canonical output
   slot, identity-unique, including unused siblings and each carry formal (previous state).
   Step/invariant formals use explicit outer inputs instead of duplicate saved sequences.
   Every other reachable body value receives a latent slot; Model selects no derivative rule.<span data-glossary-navigation="true"> ([identity glossary](../glossary.md#term-identity-meanings))</span>
2. Scan.apply reserves canonical residual descriptors/wrappers at original construction after
   finalCarry/emissions, NOT stacked storage. Only publication/downstream consumers demand history; internal uses need
   step temporaries, not history. Propagate ordinary-use demand into query/Prepare and revalidate
   after every rewrite. Live histories use original time indices (prior carry BEFORE the step).
   Aliased roles share body value bindings; outer wrappers remain distinct. Only live saved sites
   restrict fusion/recompute. Dead ports allocate/store ZERO history and retain fusion freedom,
   INCLUDING the generated reverse scan's own residuals. This is new liveness/Prepare work, NOT
   existing sibling-DCE or an opaque saved-workspace handshake.<span data-glossary-navigation="true"> ([publication glossary](../glossary.md#term-publication); [Prepare glossary](../glossary.md#prepare); [liveness glossary](../glossary.md#term-liveness); [DCE glossary](../glossary.md#term-dead-code-elimination-dce); [opaque glossary](../glossary.md#term-opaque))</span>
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
   invariant formals, NEVER undeclared free leaves.<span data-glossary-navigation="true"> ([leaves glossary](../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([compile glossary](../glossary.md#compile); [invariant input glossary](../glossary.md#term-invariant-input); [TANH glossary](../glossary.md#term-tanh); [ValueId glossary](../glossary.md#logical-valueid-versus-physical-memory-slot); [graph glossary](../glossary.md#graph); [constants glossary](../glossary.md#term-constant); [SIGMOID glossary](../glossary.md#term-sigmoid); [scalar glossary](../glossary.md#term-scalar); [accumulator glossary](../glossary.md#accumulator-type))</span>
4. Construct an ordinary sibling Scan.apply in opposite traversal, with cotangent carry initialized
   from finalCarry seeds, per-step output seeds and residual sequences as explicit inputs, and
   shared parameters as invariants. Its body is existing Tensor derivative algebra. Carry parameter
   sums initialized to exact typed zero, add each positional contribution once in traversal order;
   return dSequence emissions, final dInitialCarry and parameter sums. Sum tied actual-role and
   cross-occurrence contributions afterward with ordinary Tensor.add, never identity-deduplicate terms.<span data-glossary-navigation="true"> ([cotangent glossary](../glossary.md#cotangent); [parameters glossary](../glossary.md#term-parameter); [occurrence glossary](../glossary.md#term-occurrence))</span>
5. One combined capture includes ORIGINAL forward wrappers and generated reverse scan, each body
   captured once in its own scope. Public TensorId/publication roles are unchanged; residual edges
   enforce forward-before-reverse liveness. Exact stored residual sites survive layout closure,
   reindexing, DCE/CSE and fusion; no merging scan occurrences or crossing saved boundaries.<span data-glossary-navigation="true"> ([capture glossary](../glossary.md#term-capture-meanings); [scope glossary](../glossary.md#term-scope); [TensorId glossary](../glossary.md#tensorid); [layout glossary](../glossary.md#layout); [CSE glossary](../glossary.md#term-common-subexpression-elimination-cse))</span>

Accepted `compiler.ScanBodyInference` independently validates scoped [descriptors](../glossary.md#term-descriptor)/roles. Residual
ports preserve true [gradient](../glossary.md#term-gradient) eligibility; reject differentiation rooted at residual ports, unsupported
body rules and every [scan](../glossary.md#term-cumulative-scan) higher-order/createGraph request before [derivative](../glossary.md#term-derivative) allocations, including
unsupported occurrences on otherwise unrelated original branches. Initial first-order support does
not admit fixed [RNN](../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../glossary.md#term-long-short-term-memory-lstm) to [BPTT](../glossary.md#term-backpropagation-through-time-bptt). Missing output [seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings) use explicit typed zeros, both [carry](../glossary.md#term-carry) and
sequence seeds may coexist. T=0 executes no body: empty emissions/residuals, exact initial final
carry, dInitialCarry=dFinalCarry, zero [invariant](../glossary.md#term-invariant) sums. Empty [dimensions](../glossary.md#dimension) follow constituent semantics;
an empty dimension does not justify skipping a nonempty reduction or unsupported [operation](../glossary.md#operation).

Rejected: postcapture augmentation/replacement, replay (inexact saved sites), manual AD recipes,
fixed RNN [adjoint](../glossary.md#term-adjoint), attrs/[registry](../glossary.md#term-registry) body or second executor. Demanded conservative residuals cost
O(T × retained [elements](../glossary.md#term-element)) memory and inhibit [fusion](../glossary.md#term-fusion). More selective saving/recompute requires review.

<a id="proposed-capability-preparation-and-padding-gates"></a>

## Accepted capability, preparation and [padding](../glossary.md#padding) gates

Extend existing [Planning](../glossary.md#term-planning)-owned [OperationCapabilityQuery](../glossary.md#operation-capability-query--operationcapabilityquery) with `Optional<CompiledScanBody>`; retain
three-[argument](../glossary.md#term-argument) empty-body construction. [Compiler](../glossary.md#term-compiler) supplies validated complete scoped body/ports,
not source [Tensors](../glossary.md#tensor). Provider [boolean](../glossary.md#data-type--datatype) supports(query) must verify the header AND every body [occurrence](../glossary.md#term-occurrence),
[descriptor](../glossary.md#term-descriptor), [residual](../glossary.md#scan-residual-port-and-history-demand)/[binding](../glossary.md#term-binding) requirement and generated [topology](../glossary.md#term-topology). Ordinary true child queries are
necessary but insufficient if the loop binder cannot execute them. Planning forwards semantic facts
and selects ONE [owner](../glossary.md#term-owner) for a complete [scan body](../glossary.md#backpropagation-through-time-and-recurrent-state); no routes, mixed-owner body partitioning or [backend](../glossary.md#backend) [AD](../glossary.md#autograd).

[PrepareContext](../glossary.md#prepare-context-preparecontext)/[PartitionDag](../glossary.md#partition-dag-partitiondag) preserve the attachment and validate static local descriptors separately
from outer IDs. [CPU](../glossary.md#term-cpu) analysis prepares the body once and declares exact resources; only finalization
AFTER assigned slots emits an OUTER TIME LOOP + existing typed body emitters at generation time.
No [runtime](../glossary.md#term-runtime) [graph](../glossary.md#graph) interpreter, per-op [lookup](../glossary.md#term-lookup), per-step [callback](../glossary.md#term-callback) [dispatch](../glossary.md#term-dispatch)/object allocation,
or T-fold source/captured graph/[bytecode](../glossary.md#term-bytecode) growth.

Current evidence: CpuLoopEmitter.emit(List<CpuAccessPlan>, Consumer<State>) composes at generation
time; CpuScalarEmitter.emit(ir,instruction,locals) writes typed locals; CpuClassFileKernelGenerator's
scalarBody (around line 201) combines [carrier](../glossary.md#term-storage-carrier) loads, instructions and stores. Reuse [scalar](../glossary.md#term-scalar)/[vector](../glossary.md#term-vector)/
carrier arithmetic only where admitted; no vector/parallel promise. CpuKernelIr's instruction list
is [pointwise](../glossary.md#term-pointwise), CpuMatmulEmitter reads fixed entry slots/geometry, and CpuCarrierEmitter's
firstGeneratedLocal/[layout](../glossary.md#layout) block assumes the entry [application binary interface](../glossary.md#term-application-binary-interface-abi) ([ABI](../glossary.md#application-binary-interface-and-native-bridge)). They are NOT
all plug-and-play: specify invocation-vs-step setup and carrier/local/base/range bindings explicitly.
Hoist [invariant](../glossary.md#term-invariant) setup outside time; prefer direct time-dependent offsets/addresses into sequence,
emission and residual storage. Select/materialize only when necessary; justify every staging copy,
[carry](../glossary.md#term-carry) bank and body temporary. Emission removes neither live saved histories nor all [MATMUL](../glossary.md#matrix-multiplication--matmul) temporaries.
Declare checked staging/carry/temporary/history/output bytes BEFORE slot assignment. One [RunState](../glossary.md#run-state--runstate)
owns [mutable](../glossary.md#term-mutable) memory, changing addresses/carry/geometry arrays; none live in shared [immutable](../glossary.md#term-immutable) prepared
state. No new [Engine](../glossary.md#term-engine)/Runtime owner or runtime tape.
Future CPU brief names every added type; no speculative generic control-flow infrastructure.
Direct static helpers are allowed when technically needed with clean Java [oracle](../glossary.md#term-test-oracle) rationale, not a
one-giant-method mandate or hidden fallback overhead. Generic no-copy/[JIT](../glossary.md#term-just-in-time-compilation-jit)/performance parity is unproved.
CpuKernelIr [identity](../glossary.md#term-identity-meanings) excludes extents/slots/[run](../glossary.md#term-run) bindings: distinguish baked/runtime [scan](../glossary.md#term-cumulative-scan) facts and
key every baked fact (including body/roles/saved-port demand); reject stale format/meaning [artifacts](../glossary.md#term-artifact).
CPU0030 implements the bounded [F32](../glossary.md#floating-point-precision-and-rounding) pointwise family at global/family identity 71; no broader type support follows.

Current symbolic implementation [metadata](../glossary.md#term-metadata): [CPU0039](backends/cpu/tasks/0039-symbolic-scan-fixed-geometry-specialization.md)
specializes coefficient-proved fixed extents, byte [strides](../glossary.md#term-stride) and frame/offset facts in the existing
CPU owners while actual B remains dynamic. Initial scalar and evolving dense carry mappings stay
distinct; affine offset sharing is generation-only. Symbolic int K uses the existing checked
[extent](../glossary.md#term-extent) [admission](../glossary.md#term-admission), with long byte addressing. The sole baked [payload](../glossary.md#artifact-envelope-and-atomic-publication) advances to
`|scan-symbolic-batch-v2=` in both [specialization](../glossary.md#term-specialization) streams, invalidating symbolic baked v1
artifacts. [Canonical](../glossary.md#term-canonical-form) `:symbolic-batch-v1`,
[schema](../glossary.md#term-schema)/family 71 and static shaped/MATMUL v4, pointwise/fixed/ordinary keys and generated bytes
remain unchanged. The [CPU guide](../backend-guide/cpu-backend.md#symbolic-scan-cpu-execution)
explains these implementation facts; they add no pipeline or performance guarantee.

First [slice](../glossary.md#slice) has NO per-row runtime lengths; old four-[dtype](../glossary.md#data-type--datatype)/layout fixed routes retain padding/reverse.
Padding-aware [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) migration needs a separate row-skipping design for zero/full/mixed lengths,
valid-prefix reverse, dense positive-zero padding and exact initial state for empty rows. It must
branch BEFORE any padded arithmetic/read of padded [primal](../glossary.md#term-primal) or [cotangent](../glossary.md#cotangent) values ([NaN](../glossary.md#term-nan)/Inf included).
Result masking is not skipping. Whole-[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) generic bodies may mix rows, so adding lengths is a real
semantic decision, not merely another input. Do not imply legacy padded scans gain [gradients](../glossary.md#term-gradient) now.

Current next planning priority is [true per-run symbolic batch](dynamic-batch-execution.md) and
[Prepare0012 DESIGN proposal](modules/prepare/dynamic-batch-geometry-contract-proposal.md),
Complete as reviewed DESIGN delivery: independent Class C APPROVE, zero open P1/P2; MAIN delivery
accepted 2026-10-24. User architecture acceptance remains pending; separately authorized authority
cutover is next. `DesignAuthorized=true; ImplementationAuthorized=false`; no Runnable/[Ready](../glossary.md#term-ready)
implementation frontier or successor brief.
This changes no completed scan result or static contract: ordinary dynamic-B CPU forward/backward
precedes [general Scan](../glossary.md#general-differentiable-scan) extension; static length skipping remains an independent, separately gated
capability. No [feature](../glossary.md#term-software-feature) execution or conflicting CPU binder/resource ABI change is authorized here.

## Immediate length-aware authority and source foundation

Model0036 remains Complete for reviewed proposal delivery. The subsequent user approval selects
the [length-aware design](length-aware-general-scan.md) and immediate authority plus first [Model](../glossary.md#term-model)
foundation. [Model0037](modules/model/tasks/0037-length-aware-general-scan-authority.md) was verified
[Ready](../glossary.md#term-ready) for this authority-only execution and is now Complete: Hume's final overall independent
Class C APPROVE, zero open P1/P2, and MAIN acceptance on 2026-10-24.
[Model0038](modules/model/tasks/0038-length-aware-scan-source-foundation.md)
is Complete, MAIN-accepted 2026-10-24 after independent Class C/[API](../glossary.md#term-application-programming-interface-api)/docs APPROVE, zero open P1/P2,
and combined [compile](../glossary.md#compile)/architecture/root retry PASS after a test-only ledger repair. Model frontier None;
Compiler0013 below is Complete, MAIN accepted 2026-10-24. The
[sole scan owner](../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan)<span data-glossary-navigation="true"> ([scan glossary](../glossary.md#term-cumulative-scan))</span> defines
accepted future semantics; [ADR0049](../design/decisions/0049-length-aware-general-scan.md) records
the decision. Model0038 implements source header/API/roles/attachments and explicit old-[owner](../glossary.md#term-owner)
rejection only, not length-mode compile/[AD](../glossary.md#autograd)/[runtime](../glossary.md#term-runtime)/cold/wire capability. Beyond the immediate
[Compiler](../glossary.md#term-compiler) packet below, owner phases remain separately gated coarse dependencies, not a task cascade.
The immediate successor is [Compiler0013](modules/compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md),
launched after Popper final APPROVE (0 P1/P2) and MAIN [dependency](../glossary.md#term-dependency)/Ready acceptance 2026-10-24. Complete
Model0037/0038 + Compiler0011 + Planning0008 feed this existing-owner [inference](../glossary.md#term-inference)/first-order
AD/query/demand [scope](../glossary.md#term-scope). No new type/layer; user “ok, pokracuj dal” authorizes this implementation.
MAIN recorded [admission](../glossary.md#term-admission) before executor launch: ImplementationAuthorized=true. Implementation is
now frozen; independent code/docs APPROVE (0 P1/P2), worker [module](../glossary.md#term-module), MAIN compatibility and final
docs/[Javadoc](../glossary.md#term-javadoc) gates PASS. MAIN accepted Complete 2026-10-24. Cold [capture](../glossary.md#term-capture-meanings)/
reconstruction, [CPU](../glossary.md#term-cpu) and [codec](../glossary.md#term-codec) remain closed; Prepare0011's separate static projection scope is
below. MAIN owns serial integration; Compiler Ready frontier None, no automatic successor launch.

The Prepare0011 launch admission and review state are recorded:
[Prepare0011](modules/prepare/tasks/0011-length-aware-scan-projection-and-resource-handoff.md) was Ready
after independent Tesla [planning](../glossary.md#term-planning) APPROVE, zero open P1/P2; MAIN [metadata](../glossary.md#term-metadata) accepted
2026-10-24. Complete Model0037/0038 +
Compiler0013 + Planning0008 + Prepare0010 feed static lengths/formal separation, mixed reverse
role/demand projection and existing staged resource [handoff](../glossary.md#term-handoff). No new type/API/dependency or
row-algebra proof owner. At Prepare0011 completion, [Prepare](../glossary.md#prepare) Ready frontier was None;
ImplementationAuthorized=true records completed
Prepare0011 only, authorized by the user's latest “pust se do prace” request.
MAIN verified admission 2026-10-24 12:25 UTC; clean sole writer started 12:31 UTC, now Complete.
Worker 71-focused/92-full tests and Javadoc PASS, zero failures/errors/skips; MAIN [SPI](../glossary.md#term-service-provider-interface-spi)-consumer
compile/four retained guards PASS. Independent code/docs APPROVE, zero open P1/P2; MAIN accepted 2026-10-24 13:00:15 UTC.
No new launch, downstream tasks, [commit](../glossary.md#term-commit) or push is authorized.
CPU mixed-boundary/[preflight](../glossary.md#term-preflight)/skipping, cold/v5 and [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) remain separate coarse gates, no detailed cascade.

No `.rnn`/`.rnnScan` migration, higher-order waiver or model/[loss](../glossary.md#loss)-wide [padding mask](../glossary.md#masks-padding-and-eager-evaluation) follows.
[NN0032](extensions/nn/tasks/0032-fluent-rnn-general-scan-migration.md) retains the confirmed advanced
second-order CPU compile [regression](../glossary.md#term-regression), [singleton](../glossary.md#term-singleton)-CONCAT copy and unproved mixed-[engine](../glossary.md#term-engine) compatibility.
MAIN owns serial shared integration, Parallel group None; authority and Model0038 acceptance are
Complete. Combined named compile/architecture and existing-native root test/Javadoc passed on
retry after the sole test-[serialization](../glossary.md#term-serialization) failure; both performance flags false, no rebuild/campaign.

## Serial [DAG](../glossary.md#term-directed-acyclic-graph-dag) and acceptance gates

Current [persistence](../glossary.md#term-persistence) [checkpoint](../glossary.md#term-checkpoint): [CP0013 Complete](extensions/checkpoint/tasks/0013-general-scan-graph-persistence-decision.md#result),
user-approved selective v4/existing cold inventory. Main accepted final Cicero APPROVE and final
[module](../glossary.md#term-module)/[Javadoc](../glossary.md#term-javadoc), consumer, architecture and one configured root PASS on 2026-10-23. The same cohesive
capability delivers wire/save-load/tests/docs and loaded consumer proof; no new type/[API](../glossary.md#term-application-programming-interface-api) or cascade.
[NN0032 fluent RNN migration](extensions/nn/tasks/0032-fluent-rnn-general-scan-migration.md)<span data-glossary-navigation="true"> ([RNN glossary](../glossary.md#term-recurrent-neural-network-rnn))</span> stays Draft;
`.rnn` is unchanged. Audit finds [CPU](../glossary.md#term-cpu) Conv1d-prefix and [singleton](../glossary.md#term-singleton)-CONCAT reverse-[seed](../glossary.md#term-seed) dense-[view](../glossary.md#term-view-meanings)
rejections, not a proved [Metal](../glossary.md#term-metal)-only [regression](../glossary.md#term-regression); mixed execution untested. CPU0037 now closes the
narrow read gap, accepted by main 2026-10-24 after Newton final APPROVE and all gates; no new
performance claim. Remaining order: cold output [descriptor](../glossary.md#term-descriptor)/remaining compatibility → fluent authority →
separately [Ready](../glossary.md#term-ready) [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) migration. Fluent/CPU source frontiers None; length-aware authority gate above
is distinct. Parallel group None, MAIN owns shared integration.
CP0013 retains its historical verified 292-path launch base and empty index in the brief.
[GRU](../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../glossary.md#term-long-short-term-memory-lstm) migration remains later Draft follow-up without a speculative detailed queue.

Main owns integration/docs at main=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
plus all accepted uncommitted authority/Model0034/0035, Planning0008, Compiler0011, Prepare0010 and CPU0030–0033 work;
Parallel group None. Authority cutover,
All-valid [Model](../glossary.md#term-model) source/[IR](../glossary.md#term-intermediate-representation-ir) and [Planning](../glossary.md#term-planning) query transport are Complete; Planning frontier None.
Current Model0037/0038 authority/foundation status is above.
Their briefs own accepted evidence. Compiler0011 is Complete, accepted by main 2026-10-22 after
independent code/docs APPROVE and compatibility checkpoint; Compiler0012 Complete, frontier then None.

| [Slice](../glossary.md#slice) | Depends on | State / bounded result |
|---|---|---|
| Model0034 decision/cutover | Complete declared baseline + user-approved concrete proposal | Complete: authority/explanations/[ADR](../glossary.md#term-architecture-decision-record-adr), final independent Class C APPROVE and main acceptance on 2026-10-21; no Java. |
| [Model0035 source/IR](modules/model/tasks/0035-general-scan-source-and-compiled-body.md) | Model0034 Complete; Ready verification 2026-10-22 | Complete: bounded source/IR foundation accepted by main 2026-10-22 after independent authority/code/docs approval and compatibility checkpoint; no downstream execution. |
| [Planning0008 query](modules/planning/tasks/0008-general-scan-body-capability-query.md) | Planning0001–0007 + Model0034/0035 Complete | Complete: optional compiled-body transport, old-query compatibility and [fail-closed](../glossary.md#term-fail-closed) contract tests; final independent APPROVE and main compatibility gates accepted 2026-10-22; no real [backend](../glossary.md#backend) support/execution. |
| [Compiler0011 AD/capture](modules/compiler/tasks/0011-general-scan-capture-and-first-order-ad.md) | Model0034/0035 + Planning0008 Complete; existing [Compiler](../glossary.md#term-compiler) baseline | Complete: scoped [capture](../glossary.md#term-capture-meanings)/[inference](../glossary.md#term-inference), joint first-order [AD](../glossary.md#autograd) and final body queries/demand; main acceptance 2026-10-22 after independent code/docs APPROVE and compatibility checkpoint. |
| [Prepare0010 projection](modules/prepare/tasks/0010-general-scan-projection-and-resource-handoff.md) | Prepare0001–0009/0003A + Model0034/0035 + Planning0008 + Compiler0011 Complete | Complete: exact scoped body/roles/demand [validation](../glossary.md#term-validation) and existing resource [handoff](../glossary.md#term-handoff); Huygens final Class C code/docs APPROVE and main exact compatibility checkpoint accepted 2026-10-22; no new type or real execution. |
| [CPU0030 vertical milestone](backends/cpu/tasks/0030-general-scan-f32-pointwise-vertical.md) | Model0034/0035 + Planning0008 + Compiler0011 + Prepare0010; CPU0022/0027/0028/0029 Complete | Complete: bounded [F32](../glossary.md#floating-point-precision-and-rounding) [pointwise](../glossary.md#term-pointwise) forward/actual automatic reverse, accepted by main 2026-10-22 after independent review and focused checkpoint PASS. |
| [CPU0031 F32 broadcast/SUM/layout closure](backends/cpu/tasks/0031-general-scan-f32-broadcast-sum-layout-closure.md#result) | CPU0030 and accepted predecessor baseline Complete | Complete: main accepted 2026-10-22, 14:38+ UTC after final independent code/docs APPROVE (0 P1/P2), module and named checkpoint PASS. |
| [CPU0032 F32 MATMUL + actual reverse](backends/cpu/tasks/0032-general-scan-f32-matmul-and-reverse.md#result) | CPU0031 and accepted foundation baseline Complete | Complete: main accepted 2026-10-22, 17:09+ UTC after independent code/docs and test-addendum APPROVE, CPU module/Javadoc and named checkpoint PASS. |
| [CPU0033 F64 domain closure](backends/cpu/tasks/0033-general-scan-f64-domain-closure.md#result) | Model0034/0035 + Planning0008 + Compiler0011 + Prepare0010 + CPU0030–0032 Complete | Complete: full [homogeneous](../glossary.md#term-homogeneous) [F64](../glossary.md#floating-point-precision-and-rounding) algebra/actual reverse accepted by main after independent reviews and final gates; unexplained timing limitation retained. |
| Low-level program join | All low-level slices reviewed including domain completion | Complete: main's source-matched native root test/Javadoc PASS after reviewed stale [identity](../glossary.md#term-identity-meanings)-test correction; no successor launch. |
| [NN0030 owned Model decision](extensions/nn/tasks/0030-general-scan-owned-model-admission-decision.md) | Complete low-level join + existing [owned Model](../glossary.md#model-module-buffer-and-scan-distinctions)/source/Training baseline | Complete for approved direction/planning only; initial F32/F64 all-valid compute/train [target](../glossary.md#term-target), not implemented. |
| [Compiler0012 cold reconstruction](modules/compiler/tasks/0012-general-scan-cold-source-reconstruction.md#result)<span data-glossary-navigation="true"> ([cold reconstruction glossary](../glossary.md#term-cold-reconstruction))</span> | Approved NN0030 + Complete Compiler0011/foundation | Complete: bounded cold body reconstruction, independent code/docs APPROVE and main final gates accepted 2026-10-23; Checkpoint/NN still fail-closed. |
| [CP0012 in-memory copy/rebind](extensions/checkpoint/tasks/0012-general-scan-in-memory-copy-and-rebind.md#result)<span data-glossary-navigation="true"> ([rebind glossary](../glossary.md#capture-graph-ports-and-reconstruction))</span> | NN0030; Compiler0012; accepted cold/low-level baseline Complete | Complete: attachments and independent body IDs preserved, wires closed, independent review/scoped gates accepted; unrestricted root not green. |
| [NN0031 owned workflow](extensions/nn/tasks/0031-general-scan-owned-model-workflow.md#result) | NN0030; Compiler0012; CP0012; accepted CPU domain/low-level join and owned Model/[Engine](../glossary.md#term-engine)/Training baseline Complete | Complete: bounded owned [graph](../glossary.md#graph)/defineTensor compute/first-order train, clean implementation, independent Class C review and final module/Javadoc passed. |
| Owned-workflow program join | NN0031 reviewed/documented and final NN module gate | Complete: main's single configured native-enabled root PASS, both [scan](../glossary.md#term-cumulative-scan) performance flags false, required Metal [TrainingSession](../glossary.md#training-session--trainingsession)/native cases unskipped/pass; no native builds/gating fixes. Evidence in NN0031 Result. |
| [CP0013 general-body persistence](extensions/checkpoint/tasks/0013-general-scan-graph-persistence-decision.md#result) | Compiler0012; CP0010/0011/0012; NN0031/CPU0030–0036/joins Complete; proposal/authority accepted | Complete: selective v4/scoped cold bodies and loaded consumer proof, final Cicero APPROVE and main module/Javadoc/consumer/architecture/configured-root PASS. No successor. |
| [CPU0037 dense actual read views](backends/cpu/tasks/0037-general-scan-dense-read-views.md#result) | CPU0036 and accepted foundation/low-level join Complete | Complete; main accepted Newton final APPROVE and CPU/Javadoc, named and configured-root gates on 2026-10-24. CPU0036 limits retained; no NN migration. |
| [Model0037 length-aware authority](modules/model/tasks/0037-length-aware-general-scan-authority.md) | Model0036 Complete + subsequent user design approval | Complete authority only; Hume final overall Class C APPROVE, zero open P1/P2; MAIN accepted 2026-10-24. |
| [Model0038 source/IR foundation](modules/model/tasks/0038-length-aware-scan-source-foundation.md) | Model0037 independent approval/MAIN acceptance Complete | Complete; MAIN accepted 2026-10-24 after independent APPROVE and combined [compile](../glossary.md#compile)/architecture/root retry PASS; source/IR and fail-closed guards only. |
| [Compiler0013 length-aware capture/first-order AD](modules/compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md) | Model0037/0038, Compiler0011 and Planning0008 Complete; cold baseline preserved | Complete; MAIN accepted 2026-10-24 after independent code/docs APPROVE, worker module, compatibility and final docs/Javadoc PASS; [evidence](modules/compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md#result). No [runtime](../glossary.md#term-runtime)/cold/wire support. |
| [Prepare0011 length-aware projection/resource handoff](modules/prepare/tasks/0011-length-aware-scan-projection-and-resource-handoff.md#result) | Model0037/0038 + Compiler0013 + Planning0008 + Prepare0010 Complete | Complete, MAIN accepted 2026-10-24; static [metadata](../glossary.md#term-metadata) implemented, focused 71/full [Prepare](../glossary.md#prepare) 92 tests, zero failures/errors/skips, Javadoc and MAIN [SPI](../glossary.md#term-service-provider-interface-spi)/four-[guard](../glossary.md#term-guard) checkpoint PASS. Independent code/docs APPROVE, zero open P1/P2; at Prepare0011 completion, Prepare Ready frontier was None. CPU execution/cold/v5/NN stay held. |
| [Padding](../glossary.md#padding)-aware NN integration | Low-level join + separately approved padding packet | Draft, independently gated; no lengths/row-skipping grant. |

Approved decision: specific ScanBody/[producer](../glossary.md#term-producer)+[node](../glossary.md#node) attachment, construction-time [residual](../glossary.md#scan-residual-port-and-history-demand)-output API
with demand-only history and exact [primal](../glossary.md#term-primal)-[binding](../glossary.md#term-binding) protocol. Do not ask for the same approval again.
Model0034 was the sole authorized authority frontier, verified on 2026-10-21 at the base above with
Complete declared dependencies and inactive serialized conflicts. Its authority execution, review
and main acceptance are complete. Model0035 and Planning0008 are also Complete; Planning frontier
None. Compiler0011, Prepare0010 and CPU0030–0033 bounded capabilities and low-level join are Complete;
At that low-level checkpoint CPU/Compiler/NN/Checkpoint frontiers were None; Compiler0012 and
CP0012, NN0031 and the configured native owned-workflow join, then CP0013 persistence and its
configured root checkpoint are Complete. CPU0037 is also Complete; those all-valid frontiers remain
closed. Current length-aware Model0037/0038 status is above.
Prepare0010's brief retains historical authorization/Ready [admission](../glossary.md#term-admission), all seven integration fields
and final independent approval/main checkpoint evidence. Serial, Parallel group None; accepted dirty
changes retained, shared-document [owner](../glossary.md#term-owner) main. The bounded F32 pointwise forward/actual-reverse
[CPU0030 result](backends/cpu/tasks/0030-general-scan-f32-pointwise-vertical.md#result) owns completion evidence.
CPU0031's brief owns the final Result, seven integration fields and historical main launch verification.
CPU0032's brief owns final evidence, seven integration fields and historical main launch verification.
Its only added production type is generation-only `CpuGeneralScanMatmulEmitter` in the existing
CPU emit package; stages gain both operand dependencies and step bindings. Main completed shared
documentation integration and the named [conformance](../glossary.md#term-conformance)/ledger/Engine plus targeted architecture checkpoint.
No other area master changes. Source-grounded split: CPU0030 used one common N and pointwise locals;
[shape](../glossary.md#shape)-changing reverse introduces SUM,
views and differently shaped [parameter](../glossary.md#term-parameter) carries. Empty `[0,H]` steps still require nonempty `[H]`
SUM/parameter updates, so no global N=0 return. Ordinary aggregate and [MATMUL](../glossary.md#matrix-multiplication--matmul) emitters have fixed
entry ABIs, not plug-and-play step bindings. Completed CPU0031 supplies explicit per-stage binding/[liveness](../glossary.md#term-liveness)
without another executor; CPU0032 MATMUL/actual F32 reverse is Complete, with full homogeneous F64
[qualification](../glossary.md#term-qualification) Complete in [CPU0033](backends/cpu/tasks/0033-general-scan-f64-domain-closure.md#result).
The package map reuses existing CPU typed seams, with no duplicate F64 executor; CPU0037 uses the
existing [lowering](../glossary.md#lowering) read predicate only. CPU0032 preserves whole-body rejection even T=0/dead, old/new [carry](../glossary.md#term-carry) versions,
exact demanded history and liveness-based reusable MATMUL cuts; no blanket no-[materialization](../glossary.md#materialization) claim.
F64 is not a [carrier](../glossary.md#term-storage-carrier) toggle: double locals/eight-byte resources and no-[DAZ](../glossary.md#term-daz)/[FTZ](../glossary.md#term-ftz) conformance have
independent review and completed join evidence. Existing Compiler rules suffice; no new architecture or AD is proposed.
Replacing the common-N restriction does not retire fixed routes or require automatic type deletion.
Root/native low-level join and NN0031's separate all-valid owned-workflow qualification/join are Complete.
Padding and low-type execution remain independently gated Draft; CP0013 persistence is now Complete.
If/while/general control-flow and associative/parallel scan are separate future decisions, NOT
prerequisites. Add only reuse seams needed here; no new AD algebra or architecture file.

Acceptance: template reuse/escape rejection, closed [derivative](../glossary.md#term-derivative) [constants](../glossary.md#term-constant), tied roles/roots, two carries,
both seed classes, shared parameters, saved [rounding](../glossary.md#term-rounding)/special values, T=0/empties and higher-order rejection.
Completed Compiler0011 acceptance compares at least three steps with [unrolled](../glossary.md#backpropagation-through-time-and-recurrent-state) [gradients](../glossary.md#term-gradient) using
non-gradient zero h0 and trainable w/u: INTERNAL carry cotangents must still propagate. Do not copy
current public [preflight](../glossary.md#term-preflight)'s non-gradient input rejection unchanged into local carry AD. This tests
the accepted reverse semantics, not a new AD system or numerical policy.
Vary T: source/captured/prepared object counts O(body), checked live saved bytes O(T); forward-only
undemanded history bytes/stores ZERO. Requested T-step emissions still require T-dependent output
storage: zero dead history does not imply [O(1)](../glossary.md#memory-management-atomic-counters-and-complexity) total memory. Test [overflow](../glossary.md#term-overflow)/overlap/cleanup/retry/[concurrency](../glossary.md#term-concurrency), unsupported
interior/generated [occurrence](../glossary.md#term-occurrence), missing body, supported children but unsupported binder, projection
and no runtime graph access. Fixed-route regression coverage remains.

CPU performance gate: use a specialized optimal clean Java algorithm/hot-loop/[dataflow](../glossary.md#term-dataflow) [oracle](../glossary.md#term-test-oracle);
inspect/decompile Class-File for accidental helpers/[dispatch](../glossary.md#term-dispatch)/allocation/copies. [Run](../glossary.md#term-run) short targeted
warmed generated-vs-direct measurements with same [dtype](../glossary.md#data-type--datatype)/carrier/[layout](../glossary.md#layout)/numerics/work and comparable
validation. Separate generation/class-load/[JIT](../glossary.md#term-just-in-time-compilation-jit) cold cost, entry/preflight end-to-end overhead and
[kernel](../glossary.md#kernel)/step cost. No guaranteed identical assembly/zero penalty or inherited CPU0029 timing proof.
Fix concrete diagnosed avoidable overhead or explicitly justify it; do not relabel it acceptable fallback.
Completed low-level gates: worker focused tests and affected-module `test javadoc`; main's named
conformance/Engine checkpoints, `:testing:architecture-tests:test` for boundary risks and native-enabled
root `test javadoc`, including the reviewed test-only retry. CPU0033 Result owns exact reused/executed
evidence. No extra timing campaign, broad/five-[fork](../glossary.md#term-fork-meanings) campaign or CPU0029 parity claim; later slices own docs/Javadoc.

Size exception: this single cross-boundary coordination protocol exceeds 200 lines/15 KB to retain residual/scoping,
AD binding, resource and qualification semantics together. Executable module briefs will be smaller;
splitting this record would separate the accepted protocol and its serial acceptance gates.
