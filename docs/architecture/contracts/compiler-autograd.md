# Compiler and automatic-differentiation contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.

## Scope

This contract owns the detailed partition-scoring inputs and output boundary, Compiler
responsibilities, compile artifacts, training-graph construction, compiler-owned automatic
differentiation, and the compile lifecycle. General Planning module responsibilities remain with
the foundational contract. This file does not own physical resources, backend lowering, prepared
execution, or optimizer orchestration.

## Partition scoring

Planning includes backend-neutral partition scoring.

Partition scoring may use compile-time information only, such as:

- graph metadata
- op kind
- data type
- shape
- estimated element count
- estimated byte size
- backend capabilities
- backend intent
- graph phase
- producer/consumer ownership candidates
- logical materialization estimates
- transfer estimates
- boundary penalties
- accelerator bonuses
- small-region penalties
- backend-neutral planning cost profiles

Partition scoring must not use:

- current runtime residency
- current device buffers
- concrete kernel classes
- concrete MPSGraph executables
- concrete CUDA kernels
- concrete OpenBLAS calls
- current `RunState`
- physical buffer addresses
- prepared executables
- backend-local workload-cache entries, route configurations, or other model-autotuning values

Partition scoring decides backend ownership at node or segment level before maximal same-owner partitioning.

The output of scoring is ownership, not executable implementation.

Backend-specific lowering and kernel selection belong to backend prepare.

A planning cost model is not a model-autotuning parameter set. It may estimate ownership
cost from backend-neutral graph and transfer facts, but it must not interpret vector, thread,
tile, route, kernel, or other backend implementation vocabulary.

## `modules/compiler`

`modules/compiler` owns graph compilation.

Allowed:

- graph capture
- bounded cold raw source capture with a borrowed, short-lived ordered Tensor-leaf sidecar, and
  reconstruction through existing public Tensor family constructors; see Compile lifecycle below
- fail-closed autograd preflight over Tensor expression occurrences
- compiler-owned gradient-rule dispatch expressed through public Tensor operations
- per-compile identity-based gradient-contribution accumulation
- topological sorting
- producer/use indexing
- canonicalization
- shape inference
- data type inference
- validation
- dead-code elimination
- common subexpression elimination
- constant folding
- algebraic simplification
- autograd expansion
- backward graph construction
- combined forward/backward graph optimization
- publication binding
- partition planning orchestration
- logical memory planning orchestration
- compile diagnostics
- `CompileArtifacts`

Forbidden:

- physical buffers
- `PreparedSchedule`
- `PreparedExecution`
- backend executables
- concrete kernel selection
- backend-specific lowering
- runtime workspace state
- runtime residency
- concrete backend dependencies

The compiler produces immutable compile-time artifacts.

It must not construct runtime execution units.

For the accepted general scan, Compiler implements independent lexical capture and validation through
package-private `compiler.ScanBodyInference`, and first-order rule orchestration through
package-private `compiler.ScanGradientRules`. These existing owners retain source/selected
proof and reverse orchestration; the coordinated continuation below selects their extension,
not another derivative framework. Prepare0010 implements scoped projection and static binding/demand validation. CPU0030
implements bounded FLOAT32 same-shape pointwise execution for the forward and independently
qualified actual reverse bodies. CPU0031 additionally qualifies homogeneous FLOAT32 broadcast,
SUM/sumToShape and bounded shape/layout closure, including actual reverse bodies. CPU0032 adds
static homogeneous FLOAT32 rank-two MATMUL, including independently qualified actual reverse
bodies, without new Compiler rules. CPU0033 implements homogeneous FLOAT64 across the complete
bounded declared algebra and actual first-order reverse closure. Whole-occurrence qualification
and resource/binder limits still apply; execution is not inferred from Compiler AD admission. The complete
source/residual/primal-binding semantics are owned by
[the scan contract](recurrent-scan.md#general-differentiable-scan).

## Compile artifacts

`CompileArtifacts` is one immutable compile-time recipe containing exactly these eight semantic
components, in current record order:

1. the exact graph-scope `CompileMode`;
2. the exact final `CompiledGraphModel`;
3. immutable graph-order `List<PlannedPartition>` membership for the maximal partitions;
4. the `LogicalMemoryPlan` derived from that graph and partition list;
5. the `PublicationPlan` for the exact final graph;
6. the complete graph-input source classification in `CompileConstantPlan`;
7. successful-compile deferred diagnostics in `CompileDiagnostics`; and
8. the exact-graph derivative metadata in `DerivativeGraphMetadata`, including derivative orders
   and the separately qualified optional containing-Scan activity plans below.

These components are logical compile output, not physical, prepared, backend-executable, or
runtime state.

`CompileArtifacts` must not contain:

- physical buffers
- prepared executables
- backend executable objects
- concrete kernel routes
- runtime workspaces
- runtime residency state
- mutable run state
- Tensor, producer, provenance, or source-capture sidecars; `CompiledGraphModel` also remains
  Tensor-free

### Controlled Scan derivative metadata

Implemented coordinated Class C cutover, accepted with Compiler0017/CPU0081 after independent
authority/interface and exact-source implementation reviews, actual selected-backward consumers
and the unfiltered affected test/Javadoc gate. This is component acceptance, not a whole-release
or wider cold/wire qualification. The
[Scan owner](recurrent-scan.md#general-ordinary-unit-scan-cutover) alone defines selected finite
routes, structural presence, optional merges, temporal ADDs and generated history semantics.

Extend existing AutogradPreflight/ScanGradientRules/FirstOrderAutograd and checked
OrdinaryConstruction owners, not a parallel derivative engine. Keep whole-original source checks
separate from selected finite root/input-pair qualification; both route passes share the same
occurrence source facts. StagePlan retains selected BackwardPlans before any derivative IDs and
joint expansion consumes their selected seeds/positions. Metadata and real construction share
the central contribution engine and all enclosing normalization/cast/merge operations, as well
as the same eleven ordinary family bodies. Unobserved ordinary AD retains direct Tensor values;
no per-contribution wrappers, scope maps or observer allocations are imposed on that path.

Ephemeral selected-pair facts distinguish uniformly impossible complete routes from retained
materialized channels under the Scan owner's local-boundary semantics. StagePlan/BackwardPlans
and real expansion consume the same positional pair selection. Retained Tensor seeds are present
to metadata and real construction over the consuming occurrence's legal prefixes; qualify both
FC and aggregate ND-only obligations, not only global source reachability. Correlated source
prefixes require proved row correspondence through existing semantic owners. No external
activity sideband, numerical-zero inference or global sparse AD follows.
Observed Scan construction also applies the scoped four-argument Model ordinary guard with the
exact optional declared symbol to every created canonical operation, including eager unused
temporaries, before derivative IDs. Reuse its factored symbolic/deferred proof; the unobserved
ordinary three-argument deferred-inference and direct Tensor paths remain unchanged.

Factor existing computed logical-layout closure into Model's
`ScanBody.closeOrdinaryLayouts(Operation, List<TensorDescriptor>, List<TensorDescriptor>)`;
StaticResultLogicalLayoutClosure keeps graph/splat/sidecar orchestration, not another family
classification or affine/reshape decision. Convolution closure delegates the same specialized
completion while retaining any necessary Conv-to-SQUEEZE orchestration. Preserve final ordinary
descriptors, graph identities/order/phases, constraints, constants and activity associations.
Both StaticResultLogicalLayoutClosure and BatchBindingPlan migrate to Model's shared
`hasCanonicalContiguousResults(Operation)` fact with the exact existing generic set; remove the
obsolete Compiler classifier/alias. Do not probe a concrete B or widen symbolic eligibility.
LayoutInference retains independent type/Shape/FitsWithin constraints; move only its existing
positive-static crop layout recipe to the existing Model family helper and delegate that recipe.
Ordinary Tensor construction, deferred inference and unproved view behavior stay unchanged.

Finite-time simplification may leave a selected gradient equal to a caller seed with no
producer. Preserve its real source binding and per-occurrence publication metadata; complete
Prepare contribution, not an artificial derivative node, supplies its physical assignment.
The reviewed producerless-value cutover adds no CompileArtifacts component, materialized
Tensor, capture identity, sparse-AD transport or change to ordinary derivative mathematics.

Component eight remains DerivativeGraphMetadata; no ninth CompileArtifacts component. Its
three-argument constructor adds `Map<NodeId, ScanActivityPlan> scanActivityByNode`, exposed through
`scanActivityByNode()`. The existing two-argument constructor intentionally creates genuinely
unannotated metadata. Each key names a containing Scan occurrence in the exact outer graph and
each plan refers by identity to that occurrence's exact local body graph. Body-local phases
remain independent of containing derivative order.

Existing capture may optionally observe formal/source-producer to local ValueId/NodeId assignment
synchronously. Expansion activity drafts are ephemeral and keyed by the generated containing
Scan producer; finish one independently scoped plan per captured occurrence, even for reused
immutable templates. No Tensor/producer/capture correspondence escapes into compiled state.
Every captured generated node has a scope, including primal-only work; eager unreachable
temporaries are checked pre-ID but need no runtime node.

Canonicalization, exact rewriting, constant folding, DCE/CSE, demand refresh and all layout/
constant-descriptor/same-ID rebases retain or remap plans with the exact local body. Annotated
metadata must not be reconstructed through the empty-plan constructor. Ordinary optimization
cannot alter optional merge, temporal ADD or availability semantics. Query emission uses the
final containing-node map entry through the Planning neutral optional fifth component; all
body-bearing callers migrate, rather than retaining a four-argument compatibility bypass.
Providers and Prepare independently validate their boundaries; Compiler proof grants neither
backend support nor broader cold reconstruction. Derivative plans are compile output, not
portable source/wire state.


### Accepted symbolic-batch metadata binding

Accepted on 2026-10-24 and implemented by
[Prepare0013](../../planning/modules/prepare/tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result)
for the bounded ordinary homogeneous FLOAT32 CPU path; symbolic cold-source reconstruction
remains unsupported. Final independent reviews and MAIN's native-enabled integration gate passed;
MAIN accepted the capability on 2026-10-24.
The eight CompileArtifacts components and their order remain unchanged. Compiler adds exactly
public final `io.github.pho001.synaptik.compiler.BatchBindingPlan` and its public static final
nested `Binding`, both privately constructed. Declaration excerpts (not runnable examples):

```java
public static BatchBindingPlan prepare(CompileArtifacts artifacts);
public BatchBindingPlan.Binding bind(List<TensorDescriptor> actualInputs);
public static final class Binding { // private constructor
    public long batchSize();
    public int valueCount();
    public TensorDescriptor descriptor(int valueIndex);
}
```

Preparation once preindexes final descriptor templates, input/value/publication positions and
exact private typed obligations from same-package CompileDiagnostics.constraints(). Typed
predicates/GraphPredicateProof gain bound evaluation inside Compiler only. Diagnostics remain
diagnostics, not executable strings, a public predicate language or Tensor/storage sidecars.
Engine retains this prepared semantic metadata; Runtime receives none of it.

`bind` snapshots non-null descriptors in final compiled input order. Derive B>=0 from the first
declared named-axis occurrence, not storage bytes; every occurrence, including unused inputs,
must agree. Match rank/fixed axes/dtype/requiresGrad and admitted layout exactly; fixed parameters
and seeds stay fixed. Only B/constants may determine expressions; unsupported expressions,
output-only symbols and anonymous unknowns reject at prepare. Unsatisfied predicates or negative
bound extents reject before borrowing. No per-run graph traversal, operation inference, recapture,
optimization or backend selection. Actual layout/storage policy belongs to
[Engine admission](runtime-prepare-engine.md#public-reusable-prepared-execution).

`valueCount()` is graph value-list size, not maximum ValueId. Resolve descriptors only on requested
value-list indices; Engine requests publications, not eager intermediate Shapes/descriptors.
Cold cost is O(inputs + requirements + obligations + publications), never per element or DAG rebuild.
References/results are non-null; lists snapshot/reject null members in encounter order. Null fails
with NullPointerException, invalid metadata with IllegalArgumentException, checked overflow with
ArithmeticException and invalid dense index with IndexOutOfBoundsException. Recipes are immutable,
thread-safe and storage-free; per-run binding belongs to its invocation.

For the accepted
[symbolic Scan domain](recurrent-scan.md#accepted-symbolic-batch-scan), extend existing scoped
inference and binding owners rather than adding a graph representation or public signature.
`ScanBodyInference` must independently validate symbolic actual/formal/result obligations and
every local declaration in its exact body scope. Preindex private bound obligations by containing
occurrence/scope, never bare coincident local ValueIds.
Every invocation must evaluate those scoped private obligations during its complete fresh bind;
numeric Engine memo coverage cannot stand in for Compiler's proof. Preindex all-port canonical actual-layout
recipes for the exact body-bearing SCAN case: every final-carry, emission and residual port,
including undemanded siblings. Unknown symbolic multi-output kinds still reject. Dynamic templates
remain numerically unresolved; `StaticResultLogicalLayoutClosure` must not fabricate static
layouts for them. Preserve role multiplicity, scope/demand after rewrites and exact saved sites.

The accepted future Engine metadata memo may not skip or cache Compiler admission. Each call
performs the complete `BatchBindingPlan.bind`, all input occurrences and private typed obligations,
and obtains a fresh `Binding` retaining that call's supplied descriptors. Input-backed publication
descriptors come from this fresh binding, including current admitted `isView`, never a previous
caller. Only template-derived non-input publication facts may be reused under the
[Engine memo owner](runtime-prepare-engine.md#accepted-engine-geometry-memo). The public signatures
above and lazy value-list indexing remain unchanged; any future Compiler refactoring or protocol
change requires its own exact approval. Compiler0014 provides scoped symbolic Scan inference,
actual FLOAT32/FLOAT64 first-order AD and fresh all-port binding, independently reviewed and
MAIN accepted 2026-10-25 after the scoped-binding compatibility checkpoint. Unknown symbolic
multi-output kinds still reject. Compiler0014 does not itself grant backend execution or geometry
memoization. Prepare0014 separately implements bounded FLOAT32 symbolic Scan execution through
Prepare/CPU/Engine, with final qualification tracked in its
[task](../../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md); geometry memoization remains future.

The [symbolic-length qualification](recurrent-scan.md#runtime-preflight-and-implementation-gates)
is implemented by Compiler0015 for fixed-T/features one-B FLOAT32/FLOAT64 inference,
actual first-order sibling AD and complete fresh scoped all-port metadata binding. Existing
BatchBindingPlan/Binding implementations, signatures and lazy indexing remain unchanged.
Fresh binding covers every actual, scoped declaration/obligation and forward/reverse canonical port,
including undemanded siblings and gradient ports. It sees descriptors only, never input/length
storage or payloads. The scan owner retains all row, length-actual/no-formal and preflight rules.
Compiler0015 itself grants no execution. Prepare0015 separately implements bounded combined
Prepare/CPU/Engine execution and outer typed/control closure under the linked scan gates. Owning
validation and independent code/structural review passed; root validation and the sole bounded probe completed
(evidence and limits:
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending. No Compiler AD/binder
algorithm or API changes here, and cold/copy/wire, NN/Training and memo support remain closed.

## Training graph model

In backward-capable compile modes, the compiler constructs backward Tensor expressions before
graph capture. It then captures the forward outputs and requested gradient roots together exactly
once into a combined forward + backward graph.

The combined graph is immutable compile-time graph state. Public Tensors remain expression model
state and never become graph nodes or values.

Runtime may expose separate forward and backward schedules, or a single training-step schedule, depending on prepare-time decisions.

Backends must not implement global autograd.

Backends execute prepared regions only.

The accepted generic-body path constructs an ordinary reverse sibling scan before that combined
capture; it does not differentiate captured IR or retrofit a captured forward producer. Scoped
body capture is part of the one combined capture, not a second outer capture. Fixed recurrent
operations remain forward-only and do not acquire BPTT from the general-scan exception.

Optimizer updates are either:

- backend-agnostic runtime/training steps, or
- graph operations generated by the training extension and lowered by backend prepare

Training must not depend on concrete backend modules.

## Compiler-owned automatic differentiation

The compiler owns reverse-mode automatic differentiation.

Symbolic-B qualification must cover the entire actual generated first-order closure,
including MATMUL, SUM/MEAN/count, arithmetic/splats, expand/divide, rank edits,
transpose/contiguous/sumToShape and batch-contracting K=B adjoints. Preserve existing count algebra
(`oneLike(input).sum(...)` for MEAN), not a prototype-batch denominator. Parameter gradients remain
parameter-shaped, never B×parameter banks. Forward/provider support implies no backward support.
For the selected FLOAT32 X `[B,4]`, W `[4,3]`, L=mean(X@W) gate with all-one inputs, B=0 when
predicates allow zero gives empty Y/dX, scalar NaN L from empty MEAN division, and nonempty
typed-zero `[4,3]` dW empty contraction. Preserve constituent static empty/special-value behavior;
do not skip the whole run or read nonexistent dY. Prepare0013 implements and tests this bounded
ordinary CPU forward/actual first-order closure; its linked Result records evidence and MAIN's
2026-10-24 acceptance after final reviews and integration. Dynamic higher-order and Scan qualification remain separate.

The required compile-time flow is:

```text
original forward Tensor expression DAG
  -> fail-closed backward-reachable operation/attribute/policy preflight
  -> compiler-owned reverse traversal and Tensor gradient-expression construction
  -> combined forward + gradient Tensor expression DAG
  -> one phase-aware graph capture
  -> immutable combined compiler graph
  -> inference and validation
  -> canonicalization and exact combined-graph optimization
  -> final validation
  -> publication and planning
```

`FORWARD_ONLY` skips autograd. `FORWARD_AND_BACKWARD` and `TRAINING_STEP` construct the combined
Tensor expression DAG before capture. `TRAINING_STEP` does not add optimizer updates until a later
architecture decision and implementation task explicitly introduces them.

Gradient-rule dispatch belongs to named compiler components such as
`ElementwiseGradientRules`. A rule constructs formulas only by calling existing public Tensor
operations such as `mul`, `add`, `sumToShape`, and `transpose`. The compiler must not add
model-owned derivative rules, a second low-level algebra language, direct generated graph-node
formula construction, a public gradient registry or facade, or Tensor gradient/backward state.

For one compile request, the compiler may use identity-based maps from exact Tensor objects to
ordered gradient contributions and accumulated gradients. This is ephemeral reverse-accumulation
bookkeeping, not public Tensor state and not another graph representation. Multiple contributions
are accumulated with ordinary `Tensor.add`.

Generated operations use the same single Model family semantics as forward operations. Their
formulas inherit arithmetic-site DAZ/FTZ permission only at named FLOAT32/BFLOAT16/FLOAT16
floating primitives, never at FLOAT64 arithmetic or exact/discrete/stored-value sites. The
compiler creates no gradient-only tolerance or alternate numerical policy. Saved outputs, masks,
indices, statistics, and state remain the exact stored values named by the derivative formula. A
backend is eligible only if it supports the entire generated topology as queried.

The implemented Conv3d training extension uses the existing convolution formula owner for
both grouped NCHW and NCDHW cross-correlation. Flatten only the kernel and output-position axes;
retain the group axis in every contraction. Input cotangents use weight-transpose MATMUL followed
by the matching public overlap-add fold; weight cotangents use the matching public unfold,
MATMUL and batch reduction; bias cotangents reduce batch and every spatial axis. Restore each
selected target's exact Shape and data type through the existing normalization. No new operation
kind, backward-only kernel, duplicated 2D formula, or backend-owned derivative is introduced.
Checked metadata construction, ordinary Tensor construction, first-order coverage, higher-order
composition and selected-target pruning must agree on this common formula.
First-order coverage is 39 families, 112 constants and 134 signatures; the forward inventory stays
40/115/137. Only the three fixed RNN/GRU/LSTM signatures remain deferred. Symbolic dimensions in
the formula do not widen ordinary public CPU named-B convolution admission. General Scan uses
this same ordinary formula and independently qualifies every generated body occurrence; Conv3d
is not a separate Scan backward exclusion.

Before constructing any backward expression, the compiler must inventory every
backward-reachable operation occurrence and its exact attributes and derivative policies. Any
unsupported or ambiguous occurrence fails closed. Full inference and validation still occur only
after the one combined capture, so a later construction, capture, inference, validation, or
optimization failure may consume temporary model-level `TensorId` values. Tensor IDs are opaque
and are never rolled back or reused.

Seeds and derivative constants are storage-free Tensor leaves or expressions explicitly
registered as compile-time constant splats. Tensor storage, labels, descriptor shape, factory
history, or provenance absence must never silently imply a compile-time constant.

Phase-aware capture receives the ordered forward outputs, requested gradient roots and their
target roles, the identity set of original forward producers, and explicit constant facts. It
assigns every `NodeId` and `ValueId` exactly once. Nodes whose producer identity belongs to the
original forward set have phase `FORWARD`; generated derivative producers have phase `BACKWARD`.
Per-node `GraphPhase` remains authoritative and must not be replaced by only a positional
backward-start index.

Distinct differentiation targets may legitimately resolve to the same captured gradient
`ValueId`. Gradient result roles map each target independently, while the graph's public output
boundary contains each distinct gradient value once. The compiler must not create identity nodes
solely to make those result values distinct.

Optimization operates on the immutable combined graph. The guarded seven-rule exact arithmetic
scan, constant folding, dead-code elimination, and common-subexpression elimination established by
compiler tasks 0003, 0003A, and 0003B must be reassessed for both phases and applied only where
their existing semantic guards remain valid. Common-subexpression elimination is phase-local
unless a later architecture update and proof establishes a broader safe rule. Every changed
candidate is revalidated through the compiler's inference-and-validation boundary. Rewrites must
retain phase, descriptor, output, saved-value, gradient, special-value, and rounding guards.
Finite tolerance or a provider observation cannot license `x * 0`, `x - x`, reassociation,
contraction, or fusion across an observable boundary. Unproved legality keeps the old graph path.
This contract does not authorize new algebraic rewrites.

Generated gradient formulas are ordinary differentiable Tensor expressions. A functional request
contains exactly one or two ordered reverse-mode stages. Every stage has non-empty ordered output
references, an output-aligned list of optional cotangent seeds, and a non-empty ordered target
list whose exact Tensor references are unique and belong to the complete original forward
inventory. Stage one selects exact Tensors from the requested forward-output boundary through
forward references. Stage two, when present, selects only generated first-stage gradients through
a stage-one target index.

An absent seed means an exact typed positive one only for a scalar, floating,
gradient-eligible output. A present seed must have the output's exact Shape and floating data type
and must not itself request gradients. `DisconnectedPolicy.ERROR` rejects a valid target that has
no differentiable route from the selected outputs. `DisconnectedPolicy.ZERO` instead returns an
ordinary exact typed zero expression; several target roles may share that same final graph value.

A one-stage request requires `createGraph == false`. A two-stage request requires
`createGraph == true` for stage one and `false` for stage two. `createGraph` retains first-stage
formulas only for the immediate second stage inside the same compile. It does not establish an
arbitrary nested or persistent derivative chain. A third stage and derivative order greater than
two are unsupported.

The compiler captures the original forward outputs and all requested stage-one and stage-two
gradient roots together once. `DerivativeGraphMetadata` augments, but does not replace,
`GraphPhase`: original forward producers have derivative order zero, producers first owned by the
first reverse stage have order one, and producers first owned by the second reverse stage have
order two. Both derivative orders one and two remain `BACKWARD` graph phase.

Gradient publication bindings exist only for derivative orders one and two. They are ordered by
derivative order and then stage-local target index, retain the target identity and final gradient
value, and remain target-distinct when several targets share one captured value. The graph output
boundary starts with the ordered forward values and then lists each previously unseen gradient
value once in binding order.

The bounded lifecycle creates no persistent or runtime tape, mutable Tensor gradient state,
`Tensor.backward()`, model-owned derivative rules, backend-owned global autograd, or second
gradient algebra. The ordinary `Engine.backward(...)` convenience is narrower than the complete
functional request boundary: it always requests one first-order stage for one scalar objective,
uses an absent seed, sets `createGraph == false`, and uses `DisconnectedPolicy.ERROR`. The full
bounded request surface is available through the advanced compile boundary.

General scan is the narrow first-order exception specified by the
[scan owner's reverse protocol](recurrent-scan.md#reverse-scan-and-exact-primal-bindings).
Its accepted [symbolic extension](recurrent-scan.md#accepted-symbolic-batch-scan) keeps
Model PURE/dtype validity separate from bounded FLOAT32/FLOAT64 Compiler AD and initial FLOAT32
CPU qualification. Compiler must prove the entire actual generated closure, including bounded
rank-three outer seed/constant/layout plumbing and scoped obligations; ordinary rank-two binding
does not imply that support. Count, batch-contracting adjoints, shared empty outputs and separate
fixed-T zero plans follow the Scan semantic owner. No forward-only or handwritten-reverse test
substitutes for actual first-order generation. Compiler0014 covers these Compiler obligations,
independently reviewed and MAIN accepted 2026-10-25; execution and numeric resource qualification
remain separate.
ScanGradientRules inventories source algebra, combines joint output seeds once per occurrence and
calls existing rules with a new explicit compile-local primal-operand binding function. Every
primal operand, including a saved activation output, must resolve through that function; original
Tensor contribution identities remain intact. This is a rule protocol extension, not a claim
that today's rules work unchanged. Derivative constants close through explicit invariant bindings
and registered outer splats/expressions; no undeclared leaves or ValueId-to-Tensor AD map is allowed.

[Compiler0017A](../../planning/modules/compiler/tasks/0017a-ordinary-lexical-primal-binding.md#result)
completes this existing resolver protocol through all ordinary PURE derivative owners, including
saved statistics/indices and nondifferentiable value operands, with independent source review and
actual lexical VJP/ordinary regression evidence. It adds no mathematical rule or new Scan source,
inference, AD, execution, cold/wire or higher-order admission. Wider typed/demand/coordinate proof
remains separately required.

Preflight enforces residual-root, unsupported-body and scan higher-order/createGraph rejection
before derivative allocation, including unsupported scans on unrelated original branches. Capture
retains original forward wrappers and independently captures each forward/reverse occurrence's
body, even for reused immutable templates. ScanBodyInference validates all scoped descriptors and
roles without outer ValueIds leaking inward. Layout closure, reindexing, DCE/CSE and fusion preserve
live saved sites and revalidate demand; neither scan occurrences nor scope identities may merge.
The accepted [shared coordinate contract](recurrent-scan.md#bounded-row-role-analysis) requires
raw scoped inference and derivative role discovery to recompute all canonical output facts through
the Model semantic owner after independent ordinary inference. It does not replace that inference,
establish selected VJP closure or widen the old cold/AD/physical domains. The Model0048 cutover
must preserve those independently owned effective admission predicates.
Compiler0011 implements these compile-time obligations for its bounded FLOAT32/FLOAT64 first-order
domain, without body optimization or fusion. Prepare0010 implements scoped projection and static
binding/demand validation. CPU0030 executes the independently qualified bounded FLOAT32 same-shape
pointwise forward/actual-reverse slice. CPU0031 extends execution to qualified homogeneous FLOAT32
broadcast, SUM/sumToShape and bounded shape/layout closure; actual reverse bodies must independently
qualify. CPU0032 adds static homogeneous FLOAT32 rank-two MATMUL under that same qualification,
using the existing Compiler derivative algebra. CPU0033 implements homogeneous FLOAT64 execution
across the complete bounded declared algebra and actual first-order reverse closure using those
same rules; it changes no Compiler ownership, saved-value semantics or higher-order admission.


Compiler0013 implements length-mode inference and first-order reverse construction under the
[scan owner's first-order restriction](recurrent-scan.md#first-order-length-mode-reverse).
Whole-original preflight checks every carry and sequence, including unused/non-gradient positions
and unrelated branches, before derivative IDs. Each incoming cotangent for an original SHARED
value is normalized before append/accumulation, including shared intermediates at B=1. Exact
typed row-zero/literal seeds have explicit sequence roles; no SHARED expansion invents ROW.
Bounded SUM restoration uses row-leading layout edits for checked ROW(1) feature adjoints, not
new arithmetic or saved-primal loads. Capture, canonicalization, guarded outer rewrites and final
queries preserve roles, saved sites, demand and the lengths-actual/no-formal distinction.
Old all-valid AD and every higher-order rejection remain unchanged. This compile-time support
does not enable cold source copying, Prepare, CPU execution, encoding or owned NN workflows.

Compiler0015 combines that existing length-mode protocol
with one named B for fixed T/features and homogeneous FLOAT32/FLOAT64. The
[sole Scan owner](recurrent-scan.md#runtime-preflight-and-implementation-gates) defines its support
gate, unchanged original-source ROW(0) restriction and generated mixed roles. Independent inference
and the actual generated closure must prove Dimension provenance, including ROW(1) constant sequences,
SHARED histories and parameter-shaped accumulators, without sampling B or changing derivative algebra.
Numerical qualification uses existing test-local evaluation of the actual generated graph against
independent nonuniform active-prefix unrolling; only the oracle reads length values. This is not
a production runner or new test framework. Whole-original preflight rejection before derivative IDs,
exact primal bindings and complete fresh scoped all-port admission remain mandatory. Execution,
cold/wire and NN/Training gates stay closed. Compiler0015's completed acceptance is recorded in its
[task result](../../planning/modules/compiler/tasks/0015-symbolic-length-scan-inference-ad-and-binding.md#result).

## Compile lifecycle

Cold captureSource/reconstructSource may additionally admit general `ScanKind.SCAN` occurrences
with a complete independently scoped body under the bounded constructor inventory below. This
approved exception is not implemented by the authority edit itself and grants no persistence,
NN acquisition, AD or backend capability. Bounded Compiler cold reconstruction and Checkpoint
attachment-preserving in-memory copy/rebind and NN's bounded owned workflow are implemented.
The [owning extension contract](extensions-training.md#generic-graph-source-and-values) records
the separate Checkpoint/NN gates. Implemented
[graph-model v4 persistence](extensions-training.md#canonical-graph-model-version-4)
changes none of this cold reconstruction inventory or protocol.

The supported `GraphCompilationPort` also owns a cold source boundary independent of ordinary
compilation: `captureSource` captures once with empty constant ingress and collects exact leaves
at first input-ID assignment; `reconstructSource` substitutes descriptor-compatible Tensors by
graph-input position. The source is the raw graph, not optimized `CompileArtifacts.graph()`.
Both operations admit only fully static FORWARD sources. The unchanged **flat outer-source**
inventory is the following closed slice, plus the explicit general-SCAN exception below. The
original eight kinds remain ADD/MUL/NEG/MATMUL/RELU/TANH/SIGMOID with
`NoOperationAttrs.INSTANCE` and rank-two PERMUTE with exactly `[1,0]`. Activations retain Model's
floating-only guards and reconstruct through `Tensor.relu/tanh/sigmoid`, without decomposition.

The additional kinds have these exact source signatures. Model guards and family inference still
apply; an operation outside these attributes or arities is unsupported, even if executable.

| Kind | Inputs and attributes | Public family reconstruction |
|---|---|---|
| SUB | Two floating operands; `NoOperationAttrs.INSTANCE`; one output. | `a.sub(b)` |
| RESHAPE | One input; `TargetShapeAttrs` with positive fully static target extents and equal checked element count; one output. | `a.reshape(shape)` |
| EXPAND_DIMS / SQUEEZE | One input; `AxisTransformAttrs`, axis 2 only; SQUEEZE requires that axis to be singleton; one output. | `a.expandDims(2)` / `a.squeeze(2)` |
| SELECT | One rank-three input; `SelectAttrs`, axis 0 and an in-bounds static index; one output. | `a.select(axis,index)` |
| GATHER | Rank-two data and rank-one non-gradient INT64 indices; `IndexAxisAttrs`, axis 0; one output. | `data.gather(indices,axis)` |
| SLICE | One rank-two input; `SliceAttrs` with exactly one nonempty last-axis entry, normalized start/length and step 1; one output. | `a.sliceByLength(starts,lengths,axes,steps)`; never reverse-normalize ends. |
| CONCAT | Positive input count, rank-three inputs; `CompositionAxisAttrs`, axis 0; one output. | `Tensor.concat(axis,inputs)` |
| CONV2D / CONV3D | Exactly three inputs in input/weight/bias order; `Conv2dAttrs.defaults()` / `Conv3dAttrs.defaults()` geometry only; one output. | `input.conv2d(weight,bias,attrs)` / `input.conv3d(weight,bias,attrs)` |
| MAX_POOL2D | One input; `MaxPool2dAttrs` with positive kernels, stride=kernel, padding 0, dilation 1, floor mode; one output, no index port. | `a.maxPool2d(attrs)` |
| MAX_POOL3D / AVERAGE_POOL3D | One input; exact MaxPool3dAttrs / AveragePool3dAttrs, positive kernel/stride/dilation, nonnegative symmetric padding and literal floor/ceil geometry under existing Model guards; one output. | `a.maxPool3d(attrs)` / `a.averagePool3d(attrs)` |
| RNN_TANH / GRU_RESET_AFTER / LSTM | Exactly the six biased/unbiased fixed signatures and two directions in the [recurrent contract](recurrent-scan.md#fixed-family-and-current-model-surface); two outputs for RNN/GRU, three for LSTM. | Call `RecurrentScan.rnn/gru/lstm` once, direction last; retain every returned canonical component. |

CONV1D is not a new kind: the selected composition expands input/weight at axis 2, uses biased
default CONV2D and squeezes axis 2. Unbiased/nondefault convolutions and MAX_POOL2D geometry
outside the row above remain unsupported. Pool3d alone admits the complete typed geometry above.
Fixed scans remain one flat producer, never static
unrolling, a body, callback or region. Their static time and batch may be zero; feature/hidden
sizes remain positive. Cold admission reads no GATHER indices or scan valid-length values;
execution owns their value bounds. Cold admission itself grants no execution or gradients;
those require independent Compiler AD and complete backend-closure qualification.

### Bounded general-scan cold reconstruction

The implemented cold inventory includes the all-valid constructors below and the fully static
floating length-mode intersection specified here.
[Compiler0018](../../planning/modules/compiler/tasks/0018-static-length-scan-cold-reconstruction.md#result)
records the original homogeneous F32/F64 implementation and its qualification. The per-role
floating cutover below extends source admission; it does not claim new backend/runtime qualification.
Preserve the [scan header and roles](recurrent-scan.md#header-bindings-and-source-admission), with
complete preflight of late/dead/zero-time scans before any formal/producer IDs or host copy/import.
Compiler0013's ordinary length-mode inference/first-order reverse construction is separate:
inference success is necessary but is not the independent cold admission predicate.

General SCAN uses existing `ScanAttrs`, `CompiledScanBody`, `ScanBody` and `Scan.apply`; no new
graph, API, type, session or executor is selected. Model PURE-body validity is broader than cold
constructor support, which is independent of dtype-specific execution/AD and wire eligibility.
Preserve every applicable Model dtype of an admitted signature, including low TANH and F32 RELU
without a CPU route; never use a provider query as cold admission.

The **body-only** inventory reuses the flat constructors above, except fixed scans (nested scans
remain forbidden), with these explicit overrides/additions. All have one output and exact raw
family inference; attributes/arities not listed remain unsupported. Existing MATMUL constructor
semantics, including vector/batched/promotion forms, stay intact; rank two bounds initial execution,
not cold copying. No arbitrary PURE-body copier is authorized.

| Body signature | Exact attributes and public reconstruction |
|---|---|
| Tensor ADD/SUB/MUL/DIV, two inputs | `NoOperationAttrs.INSTANCE`; `a.add/sub/mul/div(b)`. ADD/SUB/MUL use Model numeric category/promotion/broadcast guards; DIV floating only. Body SUB includes integral forms, unlike outer SUB. |
| Scalar ADD/SUB/MUL/DIV, one input | `ScalarValueAttrs`; `a.add/sub/mul/div(attrs.value())`, preserving exact typed ScalarValue. Numeric ADD/SUB/MUL; floating DIV; scalar dtype equals input. |
| NEG/RELU/TANH/SIGMOID, one input | `NoOperationAttrs.INSTANCE`; existing floating family constructor, not decomposition. |
| SUM, one input | `NoOperationAttrs` → `sum()`; `AxisReductionAttrs` → `sum(axis,keepDimensions)`; `MultiAxisReductionAttrs` → `sum(axes,keepDimensions)`; `SumToShapeAttrs` → `sumToShape(targetShape)`. Numeric Model guards; preserve axis order, empty axes and keep-dimensions identity. No masked SUM. |
| EXPAND / RESHAPE, one input | `TargetShapeAttrs`; `expand(shape)` / `reshape(shape)`. Fully static normalized target; exact broadcast / equal checked element count. Body reshape includes scalar/zero extents accepted by the Shape overload, not raw `-1` requests. |
| EXPAND_DIMS / SQUEEZE, one input | `AxisTransformAttrs`; `expandDims(axis)` / `squeeze(axis)` at any normalized valid axis; squeeze singleton only. |
| PERMUTE, one input | Existing rank-two `PermutationAttrs [1,0]` only; `permute(1,0)`. |
| CONTIGUOUS, one input | `NoOperationAttrs.INSTANCE`; `contiguous()`; retain exact canonical result geometry without reading/copying storage. |
| CAST, one input | `CastAttrs`; `cast(targetDataType)` between any two floating dtypes, or identity for nonfloating dtypes. Preserve the explicit target, exact inferred result dtype/gradient/layout and fresh producer; never replace casts with aliases or normalize stored descriptors to F32. Nonfloating cross-dtype casts remain outside this cold closure. |

Constants are explicit outer input substitutions bound to invariant formals, or scalar operation
attributes; no free/host-backed body Tensor, hidden literal producer or constant sidecar is added.
Other existing flat mappings retain their exact guards inside bodies; new body mappings must not
leak into flat outer admission. Unsupported PURE operations remain Model-valid but cold-unsupported.

Length-mode cold admission intersects that same constructor inventory with the existing
ROW(0), rank-two ROW(1), SHARED and terminal-SHARED length rules. Every carry/sequence/invariant
actual, formal and body value must be floating (FLOAT32, FLOAT64, BFLOAT16 or FLOAT16), including
unused declarations, dead interiors and intermediates hidden by casts. Each positional actual
matches its own formal, and each next carry matches its own initial carry dtype/shape; roles need
not share one dtype. Ordinary family inference owns promotion and exact result descriptors.
Explicit low↔F32 (or other floating) CAST boundaries retain both original-storage and working-site
metadata; there is no implicit compute-dtype rewrite. Nonfloating length-body values remain
unsupported, including otherwise valid identity casts. The separate final non-gradient INT64[B]
lengths actual has no body formal; no length value is read. Reject symbolic geometry and deferred
obligations here, without narrowing the existing all-valid cold inventory or querying a backend.
The length constructor whitelist additionally admits only MAX_POOL3D and AVERAGE_POOL3D with
their exact typed attributes. This does not relax static geometry, ROW/SHARED, terminal-SHARED,
role/axis or no-deferred-obligation restrictions, nor admit other CNN constructors.

Existing graph-model V4/V5/V6 descriptor/scalar dtype words and body CAST opcode 32 already
represent these original floating types and explicit targets; length opcode 33 retains its role
fields. This cutover requires no schema bump. Checkpoint copy/rebind/save/load must preserve raw
low named-state bytes, aliases and canonical ports, and validate complete source metadata before
candidate identities or state import. Cold support does not authorize CPU low arithmetic; the CPU
BF16/F16 lock remains. Training objectives remain scalar gradient-eligible F32/F64, so low outputs
require an explicit ordinary cast when used as an objective.

Before any formal or producer allocation, validate positional substitutions against each
length occurrence's outer ROW/SHARED roles. Newly tied actuals must denote the same outer row
axis or both be SHARED; step axis r maps to outer axis r+1. Descriptor equality alone is not
sufficient to admit a newly aliased ROW/SHARED or differently mapped pair. Preserve compatible
repeated substitutions and opaque upstream provenance. Reconstruct exactly c+s+i formals,
slice invariants at c+s+i, and supply the final lengths actual separately to the length overload,
preserving rowCarryCount, every ordered optional sequence axis, traversal and all canonical ports.

Preflight the complete outer graph, every contained node/value (including dead outer scans and
T=0 bodies), all formal/role/residual bindings, phases, exact descriptors, static element counts
and descriptor-layout arithmetic before creating **any** formal or producer. Reuse raw inference
and canonical residual/formal-geometry validation; reject deferred obligations, nested scans and
unknown families. Do not add physical byte/resource/alignment admission or a canonical outer-layout
filter: those remain Checkpoint/backend concerns. Preserve existing raw unresolved MATMUL/CONCAT
layout cases. Structural constructors still own missing attachments and malformed topology. Scope
failure in a later body must precede even an earlier valid outer producer's allocation.

Cold admission additionally requires empty `demandedOutputIndices` and `body.graph().outputs()`
exactly equal to the first-occurrence-unique concatenation of `nextCarry`, `emissions` and
`residuals`, in that encounter order. Reject nonempty demand or extra, reordered or missing role
outputs before **any** formal/producer allocation, including late/dead outer scans and T=0.
These are raw cold-source checks only; ordinary compilation may retain nonempty demand and is
not narrowed by this reconstruction boundary.

After preflight, use separate call-local scope tables only. Create fresh distinct canonical
storage-free formals in declaration order, including unused positions; reconstruct each body
node once, one ScanBody and one Scan.apply per occurrence, never T-unroll. Bind every ordered
canonical final-carry/emission/latent-residual port exactly once, including unrequested ports and
aliased role bindings. Both traversal flags, repeated actual positions and distinct occurrences
survive; reuse of one template does not merge scopes. Body IDs never resolve in an outer table.
Substitute only outer inputs; produced/repeated descriptor-compatible replacements retain opaque
upstream provenance. Discard all reconstruction tables on return/failure; retain no original
body/formals/template and introduce no autograd conversion map or persistent captured graph owner.

All stored nodes, including dead nodes, and all values
must pass static checked geometry, family inference, exact descriptor validation, and absence of
deferred obligations before reconstruction creates any producer. Model constructors continue to
own structural closure, attributes/signatures, and layout geometry.

`GraphCompilationPort.validateSource(CompiledGraphModel)` exposes this same complete metadata
preflight without creating any Tensor or consuming Tensor identities, reading/copying host storage,
or allocating tensor storage. It validates all stored nodes and contained bodies before consumers
create candidate leaves; it does not validate leaf identity/substitution associations or grant
execution/persistence capability. Checkpoint decode must call it for every wire version before
creating any candidate Tensor, then retain ordinary acquisition/reconstruction association checks.

Use raw `CapturedGraphInference`, not `StaticResultLogicalLayoutClosure` or an optimized graph.
SELECT and SLICE derive views only from present input layouts; CONCAT and fixed-scan outputs remain
unresolved. RESHAPE derives a view only from present contiguous geometry, otherwise remaining
unresolved. Raw unresolved MATMUL→SLICE/RESHAPE and CONCAT→SELECT/RESHAPE chains are valid when
their exact descriptors match raw family inference and no deferred obligation remains. Do not
force copies, resolve missing layouts or read storage to make a source fit admission.

`SourceCapture` snapshots unique provenance-free exact leaf references in graph-input order,
with full descriptor equality, even for direct record construction. It borrows those Tensors
without owning storage, a close lifecycle, or a cache. Reconstruction snapshots its supplied list
and checks count, nulls, and full descriptor equality, but permits ordinary produced expressions
and repeated references as explicit substitutions, subject to the length-role alias checks above.
Their upstream provenance is opaque to source admission and remains attached for later ordinary
compile/autograd. Position is the association contract; same-descriptor swaps cannot be detected.

Only after complete preflight may reconstruction allocate call-local `ValueId`-to-Tensor
tables, one per scope. It constructs each stored node once in topological order through its public family
constructor, mapping every ordered output slot to its canonical wrapper, retaining sharing,
repeated port uses and output order, never
merging occurrences or fabricating intermediate placeholders. The table is discarded on return
or failure; returned outputs retain only ordinary provenance. Zero-node pass-through and unused
inputs are valid source structure, not an Engine executability promise. Neither cold operation
optimizes, closes descriptors, accesses or allocates storage, selects backends, executes, changes
ordinary compilation/autograd, or permits an autograd conversion map or another gradient algebra.

Compile lifecycle:

```text
Tensor forward outputs
  -> if backward is requested:
     - fail-closed autograd preflight
     - reverse accumulation through public Tensor operations
     - combined forward + gradient Tensor expression DAG
  -> one phase-aware GraphCapture
  -> topological sort
  -> producer/use index
  -> shape and data type inference
  -> validation
  -> canonicalization
  -> combined-graph optimization
     - DCE
     - phase-local CSE
     - constant folding
     - algebraic simplification
  -> final validation
  -> publication binding
  -> backend intent propagation
  -> capability analysis
  -> partition scoring
  -> ownership decision
  -> maximal same-owner partitioning
  -> logical memory/materialization requirements
  -> CompileArtifacts
```

Compile must not create:

- prepared schedules
- prepared units
- prepared executions
- backend executables
- physical buffers
- kernel routes
- runtime workspaces
- backend-specific DAGs

## Profile-free compile identity

<a id="numerical-profile-compile-identity"></a>

The compile port, capability queries, and
`CompileArtifacts` carry no numerical-profile selector or replacement identity. Compiler retains
its semantic guards, graph-phase and derivative metadata, and complete occurrence facts. It does
not convert test-only arithmetic tolerances into optimization permissions or provider capability.

## Accepted ordinary materializing RESHAPE final layout

[ADR0062](../../design/decisions/0062-ordinary-materializing-reshape-closure.md) and
[Compiler0019](../../planning/modules/compiler/tasks/0019-static-materializing-reshape-logical-layout-closure.md)
authorize a bounded final-layout extension, not current execution qualification. The second
topological StaticResultLogicalLayoutClosure phase prefers existing exact family-inferred views.
For validated fully static RESHAPE with present noncontiguous input geometry and an unresolved
inferred result, it may select canonical non-view logical output through existing closeValue.
Preserve inferred dtype/Shape agreement, captured gradient eligibility, constraints, identities,
topology and sidecars. Do not broaden the first-phase family classifier or fabricate dynamic or
missing-input geometry. Static scalar/empty geometry follows ordinary checked arithmetic.

Raw Model/LayoutInference/CapturedGraphInference and source/cold reconstruction remain unchanged;
the raw unresolved descriptor is still valid source and is not rewritten for persistence.
No VJP formula, consumer CONTIGUOUS, physical buffer, backend route or codec change follows.
CPU0079 and Metal0081 must independently qualify their materializing consumers. Existing exact
offset views remain exact; selecting materialization is not proof that every other alias is
mathematically impossible.
