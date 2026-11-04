# [Compiler](../../glossary.md#term-compiler) and automatic-differentiation contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.<span data-glossary-navigation="true"> ([normative glossary](../../glossary.md#term-normative); [scope glossary](../../glossary.md#term-scope); [dependency glossary](../../glossary.md#term-dependency))</span>

## [Scope](../../glossary.md#term-scope)

This contract owns the detailed [partition](../../glossary.md#partition)-scoring inputs and output boundary, [Compiler](../../glossary.md#term-compiler)
responsibilities, [compile artifacts](../../glossary.md#compile-artifacts), training-[graph](../../glossary.md#graph) construction, compiler-owned [automatic
differentiation](../../glossary.md#autograd), and the [compile](../../glossary.md#compile) [lifecycle](../../glossary.md#lifecycle). General [Planning](../../glossary.md#term-planning) [module](../../glossary.md#term-module) responsibilities remain with
the foundational contract. This file does not own physical resources, [backend](../../glossary.md#backend) [lowering](../../glossary.md#lowering), [prepared
execution](../../glossary.md#prepared-execution--preparedexecution), or [optimizer](../../glossary.md#term-optimizer) orchestration.

## [Partition](../../glossary.md#partition) scoring

[Planning](../../glossary.md#term-planning) includes [backend](../../glossary.md#backend)-neutral partition scoring.

Partition scoring may use [compile](../../glossary.md#compile)-time information only, such as:

- [graph](../../glossary.md#graph) [metadata](../../glossary.md#term-metadata)
- op kind
- [data type](../../glossary.md#data-type--datatype)
- [shape](../../glossary.md#shape)
- estimated [element](../../glossary.md#term-element) count
- estimated byte size
- backend capabilities
- [backend intent](../../glossary.md#backend-intent--backendintent)
- [graph phase](../../glossary.md#graph-phase)
- [producer](../../glossary.md#term-producer)/consumer [ownership](../../glossary.md#term-ownership-meanings) candidates
- logical [materialization](../../glossary.md#materialization) estimates
- transfer estimates
- boundary penalties
- [accelerator](../../glossary.md#term-accelerator) bonuses
- small-region penalties
- backend-neutral planning cost profiles

Partition scoring must not use:

- current [runtime](../../glossary.md#term-runtime) [residency](../../glossary.md#residency)
- current [device](../../glossary.md#term-device) [buffers](../../glossary.md#term-buffer)
- concrete [kernel](../../glossary.md#kernel) classes
- concrete [MPSGraph](../../glossary.md#term-mpsgraph) executables
- concrete [CUDA](../../glossary.md#term-cuda) [kernels](../../glossary.md#kernel)
- concrete [OpenBLAS](../../glossary.md#term-openblas) calls
- current [`RunState`](../../glossary.md#run-state--runstate)
- physical buffer addresses
- prepared executables
- backend-local workload-[cache](../../glossary.md#term-cache) entries, route configurations, or other [model](../../glossary.md#term-model)-[autotuning](../../glossary.md#autotuning--model-autotuning) values

Partition scoring decides [backend ownership](../../glossary.md#backend-ownership) at [node](../../glossary.md#node) or segment level before maximal same-[owner](../../glossary.md#term-owner) partitioning.

The output of scoring is ownership, not executable implementation.

Backend-specific [lowering](../../glossary.md#lowering) and kernel selection belong to backend [prepare](../../glossary.md#prepare).

A [planning cost model](../../glossary.md#term-planning-cost-model) is not a model-autotuning [parameter](../../glossary.md#term-parameter) set. It may estimate ownership
cost from backend-neutral graph and transfer facts, but it must not interpret [vector](../../glossary.md#term-vector), thread,
tile, route, kernel, or other backend implementation vocabulary.

## `modules/compiler`

`modules/compiler` owns [graph](../../glossary.md#graph) [compilation](../../glossary.md#compiler-engine-and-execution-stage-names).

Allowed:

- [graph capture](../../glossary.md#capture-graph-ports-and-reconstruction)
- bounded cold raw source capture with a borrowed, short-lived ordered Tensor-leaf sidecar, and
  reconstruction through existing public Tensor family constructors; see Compile lifecycle below<span data-glossary-navigation="true"> ([capture glossary](../../glossary.md#term-structural-capture))</span><span data-glossary-navigation="true"> ([borrowed glossary](../../glossary.md#ownership-borrowing-and-lease); [leaf glossary](../../glossary.md#term-leaf); [lifecycle glossary](../../glossary.md#lifecycle))</span>
- [fail-closed](../../glossary.md#term-fail-closed) [autograd](../../glossary.md#autograd) [preflight](../../glossary.md#term-preflight) over [Tensor](../../glossary.md#tensor) expression occurrences
- [compiler](../../glossary.md#term-compiler)-owned [gradient](../../glossary.md#term-gradient)-rule [dispatch](../../glossary.md#term-dispatch) expressed through public Tensor operations
- per-[compile](../../glossary.md#compile) [identity](../../glossary.md#term-identity-meanings)-based gradient-contribution accumulation
- [topological](../../glossary.md#term-topological-order) sorting
- [producer](../../glossary.md#term-producer)/use indexing
- [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form)
- [shape](../../glossary.md#shape) [inference](../../glossary.md#term-inference-meanings)
- [data type](../../glossary.md#data-type--datatype) inference
- [validation](../../glossary.md#term-validation)
- [dead-code elimination](../../glossary.md#term-dead-code-elimination-dce)
- common subexpression elimination
- [constant folding](../../glossary.md#term-constant-folding)
- algebraic simplification
- autograd expansion
- [backward graph](../../glossary.md#backward-graph) construction
- combined forward/backward graph optimization
- [publication](../../glossary.md#term-publication) [binding](../../glossary.md#term-binding)
- [partition](../../glossary.md#partition) [planning](../../glossary.md#term-planning) orchestration
- logical memory planning orchestration
- [compile diagnostics](../../glossary.md#compile-diagnostics)
- `CompileArtifacts`

Forbidden:

- physical [buffers](../../glossary.md#term-buffer)
- [`PreparedSchedule`](../../glossary.md#prepared-schedule--preparedschedule)
- [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution)
- [backend](../../glossary.md#backend) executables
- concrete [kernel](../../glossary.md#kernel) selection
- backend-specific [lowering](../../glossary.md#lowering)
- [runtime](../../glossary.md#term-runtime) workspace state
- runtime [residency](../../glossary.md#residency)
- [concrete backend](../../glossary.md#concrete-backend) dependencies

The compiler produces [immutable](../../glossary.md#term-immutable) compile-time [artifacts](../../glossary.md#term-artifact).

It must not construct runtime execution units.

For the accepted [general scan](../../glossary.md#general-differentiable-scan), Compiler implements independent lexical [capture](../../glossary.md#term-capture-meanings) and validation through
package-private `compiler.ScanBodyInference`, and first-order rule orchestration through
package-private `compiler.ScanGradientRules`. These existing [owners](../../glossary.md#term-owner) retain source/selected
proof and reverse orchestration; the coordinated continuation below selects their extension,
not another [derivative](../../glossary.md#term-derivative) framework. Prepare0010 implements scoped projection and static binding/demand validation. CPU0030
implements bounded [FLOAT32](../../glossary.md#term-float32) same-shape [pointwise](../../glossary.md#term-pointwise) execution for the forward and independently
qualified actual reverse bodies. CPU0031 additionally qualifies [homogeneous](../../glossary.md#term-homogeneous) FLOAT32 broadcast,
SUM/sumToShape and bounded shape/[layout](../../glossary.md#layout) closure, including actual reverse bodies. CPU0032 adds
static homogeneous FLOAT32 [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../glossary.md#matrix-multiplication--matmul), including independently qualified actual reverse
bodies, without new Compiler rules. CPU0033 implements homogeneous [FLOAT64](../../glossary.md#term-float64) across the complete
bounded declared algebra and actual first-order reverse closure. Whole-[occurrence](../../glossary.md#term-occurrence) [qualification](../../glossary.md#term-qualification)
and resource/binder limits still apply; execution is not inferred from Compiler [AD](../../glossary.md#autograd) [admission](../../glossary.md#term-admission). The complete
source/[residual](../../glossary.md#scan-residual-port-and-history-demand)/[primal](../../glossary.md#term-primal)-binding semantics are owned by
[the scan contract](recurrent-scan.md#general-differentiable-scan).

## [Compile artifacts](../../glossary.md#compile-artifacts)

`CompileArtifacts` is one [immutable](../../glossary.md#term-immutable) [compile](../../glossary.md#compile)-time recipe containing exactly these eight semantic
components, in current record order:

1. the exact [graph](../../glossary.md#graph)-[scope](../../glossary.md#term-scope) [`CompileMode`](../../glossary.md#compile-mode--compilemode);
2. the exact final [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel);
3. immutable graph-order `List<PlannedPartition>` membership for the maximal partitions;
4. the `LogicalMemoryPlan` derived from that graph and [partition](../../glossary.md#partition) list;
5. the `PublicationPlan` for the exact final graph;
6. the complete graph-input source classification in `CompileConstantPlan`;
7. successful-compile deferred diagnostics in `CompileDiagnostics`; and
8. the exact-graph [derivative](../../glossary.md#term-derivative) [metadata](../../glossary.md#term-metadata) in `DerivativeGraphMetadata`, including derivative orders
   and the separately qualified optional containing-Scan activity plans below.

These components are logical compile output, not physical, prepared, [backend](../../glossary.md#backend)-executable, or
[runtime](../../glossary.md#term-runtime) state.

`CompileArtifacts` must not contain:

- physical [buffers](../../glossary.md#term-buffer)
- prepared executables
- backend executable objects
- concrete [kernel](../../glossary.md#kernel) routes
- runtime workspaces
- runtime [residency](../../glossary.md#residency) state
- [mutable](../../glossary.md#term-mutable) [run state](../../glossary.md#run-state--runstate)
- [Tensor](../../glossary.md#tensor), [producer](../../glossary.md#term-producer), [provenance](../../glossary.md#provenance), or source-[capture](../../glossary.md#term-capture-meanings) sidecars; `CompiledGraphModel` also remains
  Tensor-free

### Controlled Scan [derivative](../../glossary.md#term-derivative) [metadata](../../glossary.md#term-metadata)

Implemented coordinated Class C cutover, accepted with Compiler0017/CPU0081 after independent
authority/interface and exact-source implementation reviews, actual selected-backward consumers
and the unfiltered affected test/[Javadoc](../../glossary.md#term-javadoc) gate. This is component acceptance, not a whole-[release](../../glossary.md#term-release)
or wider cold/wire [qualification](../../glossary.md#term-qualification). The
[Scan owner](recurrent-scan.md#general-ordinary-unit-scan-cutover) alone defines selected [finite](../../glossary.md#term-finite)
routes, structural presence, optional merges, temporal ADDs and generated history semantics.

Extend existing AutogradPreflight/ScanGradientRules/FirstOrderAutograd and checked
OrdinaryConstruction [owners](../../glossary.md#term-owner), not a parallel derivative [engine](../../glossary.md#term-engine). Keep whole-original source checks
separate from selected finite root/input-pair qualification; both route passes share the same
[occurrence](../../glossary.md#term-occurrence) source facts. StagePlan retains selected BackwardPlans before any derivative IDs and
joint expansion consumes their selected [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings)/positions. Metadata and real construction share
the central contribution engine and all enclosing [normalization](../../glossary.md#term-normalization)/cast/merge operations, as well
as the same eleven ordinary family bodies. Unobserved ordinary [AD](../../glossary.md#autograd) retains direct [Tensor](../../glossary.md#tensor) values;
no per-contribution wrappers, [scope](../../glossary.md#term-scope) maps or [observer](../../glossary.md#term-observer) allocations are imposed on that path.

Ephemeral selected-pair facts distinguish uniformly impossible complete routes from retained
materialized channels under the Scan [owner](../../glossary.md#term-owner)'s local-boundary semantics. StagePlan/BackwardPlans
and real expansion consume the same positional pair selection. Retained Tensor seeds are present
to metadata and real construction over the consuming occurrence's legal prefixes; qualify both
FC and aggregate [ND](../../glossary.md#term-nd)-only obligations, not only global source reachability. Correlated source
prefixes require proved row correspondence through existing semantic owners. No external
activity sideband, numerical-zero [inference](../../glossary.md#term-inference-meanings) or global sparse AD follows.
Observed Scan construction also applies the scoped four-[argument](../../glossary.md#term-argument) [Model](../../glossary.md#term-model) ordinary [guard](../../glossary.md#term-guard) with the
exact optional declared [symbol](../../glossary.md#term-symbol) to every created [canonical](../../glossary.md#term-canonical-form) [operation](../../glossary.md#operation), including [eager](../../glossary.md#term-eager-evaluation) unused
temporaries, before derivative IDs. Reuse its factored symbolic/deferred proof; the unobserved
ordinary three-argument deferred-inference and direct Tensor paths remain unchanged.

Factor existing computed logical-[layout](../../glossary.md#layout) closure into Model's
`ScanBody.closeOrdinaryLayouts(Operation, List<TensorDescriptor>, List<TensorDescriptor>)`;
StaticResultLogicalLayoutClosure keeps [graph](../../glossary.md#graph)/splat/sidecar orchestration, not another family
classification or affine/reshape decision. [Convolution](../../glossary.md#term-channels-first-convolution-layers) closure delegates the same specialized
completion while retaining any necessary Conv-to-SQUEEZE orchestration. Preserve final ordinary
[descriptors](../../glossary.md#term-descriptor), graph [identities](../../glossary.md#term-identity-meanings)/order/phases, constraints, [constants](../../glossary.md#term-constant) and activity associations.
Both StaticResultLogicalLayoutClosure and BatchBindingPlan migrate to Model's shared
`hasCanonicalContiguousResults(Operation)` fact with the exact existing generic set; remove the
obsolete [Compiler](../../glossary.md#term-compiler) classifier/[alias](../../glossary.md#aliasing-and-arena). Do not probe a concrete B or widen symbolic eligibility.
LayoutInference retains independent type/[Shape](../../glossary.md#shape)/FitsWithin constraints; move only its existing
positive-static crop layout recipe to the existing Model family helper and delegate that recipe.
Ordinary Tensor construction, deferred inference and unproved [view](../../glossary.md#term-view-meanings) behavior stay unchanged.

Finite-time simplification may leave a selected [gradient](../../glossary.md#term-gradient) equal to a caller [seed](../../glossary.md#term-seed) with no
[producer](../../glossary.md#term-producer). Preserve its real source [binding](../../glossary.md#term-binding) and per-occurrence [publication](../../glossary.md#term-publication) metadata; complete
[Prepare](../../glossary.md#prepare) contribution, not an artificial derivative [node](../../glossary.md#node), supplies its physical assignment.
The reviewed producerless-value cutover adds no CompileArtifacts component, materialized
Tensor, [capture](../../glossary.md#term-capture-meanings) [identity](../../glossary.md#term-identity-meanings), sparse-AD transport or change to ordinary derivative mathematics.

Component eight remains DerivativeGraphMetadata; no ninth CompileArtifacts component. Its
three-argument constructor adds `Map<NodeId, ScanActivityPlan> scanActivityByNode`, exposed through
`scanActivityByNode()`. The existing two-argument constructor intentionally creates genuinely
unannotated metadata. Each key names a containing Scan occurrence in the exact outer graph and
each plan refers by identity to that occurrence's exact local body graph. Body-local phases
remain independent of containing derivative order.

Existing capture may optionally observe formal/source-producer to local [ValueId](../../glossary.md#logical-valueid-versus-physical-memory-slot)/[NodeId](../../glossary.md#nodeid) assignment
synchronously. Expansion activity drafts are ephemeral and keyed by the generated containing
Scan producer; finish one independently scoped plan per captured occurrence, even for reused
[immutable](../../glossary.md#term-immutable) templates. No Tensor/producer/capture correspondence escapes into compiled state.
Every captured generated node has a scope, including [primal](../../glossary.md#term-primal)-only work; eager unreachable
temporaries are checked pre-ID but need no [runtime](../../glossary.md#term-runtime) node.

[Canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), exact rewriting, [constant folding](../../glossary.md#term-constant-folding), [DCE](../../glossary.md#term-dead-code-elimination-dce)/[CSE](../../glossary.md#term-common-subexpression-elimination-cse), demand refresh and all layout/
constant-descriptor/same-ID rebases retain or remap plans with the exact local body. Annotated
metadata must not be reconstructed through the empty-plan constructor. Ordinary optimization
cannot alter optional merge, temporal ADD or availability semantics. Query emission uses the
final containing-node map entry through the [Planning](../../glossary.md#term-planning) neutral optional fifth component; all
body-bearing callers migrate, rather than retaining a four-argument compatibility bypass.
Providers and Prepare independently validate their boundaries; Compiler proof grants neither
[backend](../../glossary.md#backend) support nor broader [cold reconstruction](../../glossary.md#term-cold-reconstruction). Derivative plans are [compile](../../glossary.md#compile) output, not
portable source/wire state.


### Accepted symbolic-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [metadata](../../glossary.md#term-metadata) [binding](../../glossary.md#term-binding)

Accepted on 2026-10-24 and implemented by
[Prepare0013](../../planning/modules/prepare/tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result)
for the bounded ordinary [homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32) [CPU](../../glossary.md#term-cpu) path; symbolic cold-source reconstruction
remains unsupported. Final independent reviews and MAIN's native-enabled integration gate passed;
MAIN accepted the capability on 2026-10-24.
The eight CompileArtifacts components and their order remain unchanged. [Compiler](../../glossary.md#term-compiler) adds exactly
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

Preparation once preindexes final [descriptor](../../glossary.md#term-descriptor) templates, input/value/[publication](../../glossary.md#term-publication) positions and
exact private typed obligations from same-package CompileDiagnostics.constraints(). Typed
predicates/GraphPredicateProof gain bound [evaluation](../../glossary.md#term-evaluation) inside Compiler only. Diagnostics remain
diagnostics, not executable strings, a public predicate language or [Tensor](../../glossary.md#tensor)/storage sidecars.
[Engine](../../glossary.md#term-engine) retains this prepared semantic metadata; [Runtime](../../glossary.md#term-runtime) receives none of it.

`bind` snapshots non-null descriptors in final compiled input order. Derive B>=0 from the first
declared named-[axis](../../glossary.md#term-axis) [occurrence](../../glossary.md#term-occurrence), not storage bytes; every occurrence, including unused inputs,
must agree. Match [rank](../../glossary.md#term-rank)/fixed axes/[dtype](../../glossary.md#data-type--datatype)/requiresGrad and admitted [layout](../../glossary.md#layout) exactly; fixed [parameters](../../glossary.md#term-parameter)
and [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings) stay fixed. Only B/[constants](../../glossary.md#term-constant) may determine expressions; unsupported expressions,
output-only symbols and anonymous unknowns reject at [prepare](../../glossary.md#prepare). Unsatisfied predicates or negative
bound extents reject before [borrowing](../../glossary.md#term-borrowing). No per-[run](../../glossary.md#term-run) [graph](../../glossary.md#graph) traversal, [operation](../../glossary.md#operation) [inference](../../glossary.md#term-inference-meanings), recapture,
optimization or [backend](../../glossary.md#backend) selection. Actual layout/storage policy belongs to
[Engine admission](runtime-prepare-engine.md#public-reusable-prepared-execution).

`valueCount()` is [graph value](../../glossary.md#graph-value)-list size, not maximum [ValueId](../../glossary.md#logical-valueid-versus-physical-memory-slot). Resolve descriptors only on requested
value-list indices; Engine requests publications, not [eager](../../glossary.md#term-eager-evaluation) intermediate [Shapes](../../glossary.md#shape)/descriptors.
Cold cost is O(inputs + requirements + obligations + publications), never per [element](../../glossary.md#term-element) or [DAG](../../glossary.md#term-directed-acyclic-graph-dag) rebuild.
References/results are non-null; lists [snapshot](../../glossary.md#term-snapshot)/reject null members in encounter order. Null fails
with NullPointerException, invalid metadata with IllegalArgumentException, checked [overflow](../../glossary.md#term-overflow) with
ArithmeticException and invalid dense index with IndexOutOfBoundsException. Recipes are [immutable](../../glossary.md#term-immutable),
[thread-safe](../../glossary.md#term-thread-safe) and storage-free; per-run binding belongs to its invocation.

For the accepted
[symbolic Scan domain](recurrent-scan.md#accepted-symbolic-batch-scan), extend existing scoped
inference and binding [owners](../../glossary.md#term-owner) rather than adding a graph representation or public [signature](../../glossary.md#term-signature).
`ScanBodyInference` must independently validate symbolic actual/formal/result obligations and
every local declaration in its exact body [scope](../../glossary.md#term-scope). Preindex private bound obligations by containing
occurrence/scope, never bare coincident local ValueIds.
Every invocation must evaluate those scoped private obligations during its complete fresh bind;
numeric Engine [memo](../../glossary.md#cache-and-memoization) coverage cannot stand in for Compiler's proof. Preindex all-port [canonical](../../glossary.md#term-canonical-form) actual-layout
recipes for the exact body-bearing SCAN case: every final-[carry](../../glossary.md#term-carry), emission and [residual port](../../glossary.md#scan-residual-port-and-history-demand),
including undemanded siblings. Unknown symbolic multi-output kinds still reject. Dynamic templates
remain numerically unresolved; `StaticResultLogicalLayoutClosure` must not fabricate static
layouts for them. Preserve role multiplicity, scope/demand after rewrites and exact saved sites.

The accepted future Engine metadata memo may not skip or [cache](../../glossary.md#term-cache) Compiler [admission](../../glossary.md#term-admission). Each call
performs the complete `BatchBindingPlan.bind`, all input occurrences and private typed obligations,
and obtains a fresh `Binding` retaining that call's supplied descriptors. Input-backed publication
descriptors come from this fresh binding, including current admitted `isView`, never a previous
caller. Only template-derived non-input publication facts may be reused under the
[Engine memo owner](runtime-prepare-engine.md#accepted-engine-geometry-memo). The public signatures
above and [lazy](../../glossary.md#term-lazy-evaluation) value-list indexing remain unchanged; any future Compiler refactoring or protocol
change requires its own exact approval. Compiler0014 provides scoped [symbolic Scan](../../glossary.md#symbolic-scan) inference,
actual FLOAT32/[FLOAT64](../../glossary.md#term-float64) first-order [AD](../../glossary.md#autograd) and fresh all-port binding, independently reviewed and
MAIN accepted 2026-10-25 after the scoped-binding compatibility [checkpoint](../../glossary.md#term-checkpoint). Unknown symbolic
multi-output kinds still reject. Compiler0014 does not itself grant backend execution or geometry
[memoization](../../glossary.md#term-memoization). Prepare0014 separately implements bounded FLOAT32 symbolic Scan execution through
Prepare/CPU/Engine, with final [qualification](../../glossary.md#term-qualification) tracked in its
[task](../../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md); geometry memoization remains future.

The [symbolic-length qualification](recurrent-scan.md#runtime-preflight-and-implementation-gates)
is implemented by Compiler0015 for fixed-T/[features](../../glossary.md#term-feature) one-B FLOAT32/FLOAT64 inference,
actual first-order sibling AD and complete fresh scoped all-port metadata binding. Existing
BatchBindingPlan/Binding implementations, signatures and lazy indexing remain unchanged.
Fresh binding covers every actual, scoped declaration/obligation and forward/reverse canonical port,
including undemanded siblings and [gradient](../../glossary.md#term-gradient) ports. It sees descriptors only, never input/length
storage or payloads. The scan [owner](../../glossary.md#term-owner) retains all row, length-actual/no-formal and [preflight](../../glossary.md#term-preflight) rules.
Compiler0015 itself grants no execution. Prepare0015 separately implements bounded combined
Prepare/CPU/Engine execution and outer typed/control closure under the linked scan gates. Owning
[validation](../../glossary.md#term-validation) and independent code/structural review passed; root validation and the sole bounded probe completed
(evidence and limits:
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending. No Compiler AD/binder
algorithm or [API](../../glossary.md#term-application-programming-interface-api) changes here, and cold/copy/wire, [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training and memo support remain closed.

## [Training graph](../../glossary.md#training-graph) [model](../../glossary.md#term-model)

In backward-capable [compile](../../glossary.md#compile) modes, the [compiler](../../glossary.md#term-compiler) constructs backward [Tensor](../../glossary.md#tensor) expressions before
[graph capture](../../glossary.md#capture-graph-ports-and-reconstruction). It then [captures](../../glossary.md#term-capture-meanings) the forward outputs and requested [gradient](../../glossary.md#term-gradient) roots together exactly
once into a combined forward + [backward graph](../../glossary.md#backward-graph).

The combined [graph](../../glossary.md#graph) is [immutable](../../glossary.md#term-immutable) compile-time graph state. Public Tensors remain expression model
state and never become graph [nodes](../../glossary.md#node) or values.

[Runtime](../../glossary.md#term-runtime) may expose separate forward and backward schedules, or a single training-step schedule, depending on [prepare](../../glossary.md#prepare)-time decisions.

[Backends](../../glossary.md#backend) must not implement global [autograd](../../glossary.md#autograd).

Backends execute prepared regions only.

The accepted generic-body path constructs an ordinary reverse sibling scan before that combined
[capture](../../glossary.md#term-capture-meanings); it does not differentiate captured [IR](../../glossary.md#term-intermediate-representation-ir) or retrofit a captured forward [producer](../../glossary.md#term-producer). Scoped
body capture is part of the one combined capture, not a second outer capture. Fixed recurrent
operations remain forward-only and do not acquire [BPTT](../../glossary.md#term-backpropagation-through-time-bptt) from the general-scan exception.

[Optimizer](../../glossary.md#term-optimizer) updates are either:

- backend-agnostic runtime/training steps, or
- graph operations generated by the training extension and lowered by backend prepare

Training must not depend on [concrete backend](../../glossary.md#concrete-backend) [modules](../../glossary.md#term-module).

## [Compiler](../../glossary.md#term-compiler)-owned [automatic differentiation](../../glossary.md#autograd)

The compiler owns [reverse-mode](../../glossary.md#primal-adjoint-and-differentiation-seed) automatic differentiation.

Symbolic-B [qualification](../../glossary.md#term-qualification) must cover the entire actual generated first-order closure,
including [MATMUL](../../glossary.md#matrix-multiplication--matmul), SUM/[MEAN](../../glossary.md#term-mean)/count, arithmetic/splats, expand/divide, [rank](../../glossary.md#term-rank) edits,
[transpose](../../glossary.md#term-transpose)/[contiguous](../../glossary.md#term-contiguous)/sumToShape and [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-contracting K=B adjoints. Preserve existing count algebra
(`oneLike(input).sum(...)` for MEAN), not a [prototype](../../glossary.md#term-prototype)-batch denominator. [Parameter](../../glossary.md#term-parameter) [gradients](../../glossary.md#term-gradient) remain
parameter-shaped, never B×parameter banks. Forward/provider support implies no backward support.
For the selected [FLOAT32](../../glossary.md#term-float32) X `[B,4]`, W `[4,3]`, L=mean(X@W) gate with all-one inputs, B=0 when
predicates allow zero gives empty Y/dX, [scalar](../../glossary.md#term-scalar) [NaN](../../glossary.md#term-nan) L from empty MEAN division, and nonempty
typed-zero `[4,3]` dW empty [contraction](../../glossary.md#contraction-dimension). Preserve constituent static empty/special-value behavior;
do not skip the whole [run](../../glossary.md#term-run) or read nonexistent dY. Prepare0013 implements and tests this bounded
ordinary [CPU](../../glossary.md#term-cpu) forward/actual first-order closure; its linked Result records evidence and MAIN's
2026-10-24 acceptance after final reviews and integration. Dynamic higher-order and Scan qualification remain separate.

The required [compile](../../glossary.md#compile)-time flow is:

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

`FORWARD_ONLY` skips [autograd](../../glossary.md#autograd). `FORWARD_AND_BACKWARD` and `TRAINING_STEP` construct the combined
[Tensor](../../glossary.md#tensor) expression [DAG](../../glossary.md#term-directed-acyclic-graph-dag) before [capture](../../glossary.md#term-capture-meanings). `TRAINING_STEP` does not add [optimizer](../../glossary.md#term-optimizer) updates until a later
architecture decision and implementation task explicitly introduces them.

Gradient-rule [dispatch](../../glossary.md#term-dispatch) belongs to named compiler components such as
`ElementwiseGradientRules`. A rule constructs formulas only by calling existing public Tensor
operations such as `mul`, `add`, `sumToShape`, and `transpose`. The compiler must not add
[model](../../glossary.md#term-model)-owned [derivative](../../glossary.md#term-derivative) rules, a second low-level algebra language, direct generated [graph](../../glossary.md#graph)-[node](../../glossary.md#node)
formula construction, a public gradient [registry](../../glossary.md#term-registry) or [facade](../../glossary.md#term-facade), or Tensor gradient/backward state.

For one compile request, the compiler may use [identity](../../glossary.md#term-identity-meanings)-based maps from exact Tensor objects to
ordered gradient contributions and accumulated gradients. This is ephemeral reverse-accumulation
bookkeeping, not public Tensor state and not another graph representation. Multiple contributions
are accumulated with ordinary `Tensor.add`.

Generated operations use the same single Model family semantics as forward operations. Their
formulas inherit arithmetic-site [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz) permission only at named FLOAT32/[BFLOAT16](../../glossary.md#term-bfloat16)/[FLOAT16](../../glossary.md#term-float16)
floating primitives, never at [FLOAT64](../../glossary.md#term-float64) arithmetic or exact/discrete/stored-value sites. The
compiler creates no gradient-only [tolerance](../../glossary.md#term-tolerance) or alternate numerical policy. Saved outputs, [masks](../../glossary.md#term-boolean-mask),
indices, statistics, and state remain the exact stored values named by the derivative formula. A
[backend](../../glossary.md#backend) is eligible only if it supports the entire generated [topology](../../glossary.md#term-topology) as queried.

The implemented [Conv3d](../../glossary.md#conv3d--ncdhw) training extension uses the existing [convolution](../../glossary.md#term-channels-first-convolution-layers) formula [owner](../../glossary.md#term-owner) for
both grouped [NCHW](../../glossary.md#term-nchw) and [NCDHW](../../glossary.md#conv3d--ncdhw) cross-correlation. Flatten only the [kernel](../../glossary.md#kernel) and output-position [axes](../../glossary.md#term-axis);
retain the group axis in every contraction. Input cotangents use [weight](../../glossary.md#term-weight)-transpose MATMUL followed
by the matching public overlap-add fold; weight cotangents use the matching public unfold,
MATMUL and batch reduction; [bias](../../glossary.md#term-bias) cotangents reduce batch and every [spatial axis](../../glossary.md#spatial-axes-channels-and-window-sampling). Restore each
selected [target](../../glossary.md#term-target)'s exact [Shape](../../glossary.md#shape) and [data type](../../glossary.md#data-type--datatype) through the existing [normalization](../../glossary.md#term-normalization). No new [operation](../../glossary.md#operation)
kind, backward-only kernel, duplicated 2D formula, or backend-owned derivative is introduced.
Checked [metadata](../../glossary.md#term-metadata) construction, ordinary Tensor construction, first-order coverage, higher-order
composition and selected-target pruning must agree on this common formula.
First-order coverage is 39 families, 112 [constants](../../glossary.md#term-constant) and 134 [signatures](../../glossary.md#term-signature); the forward inventory stays
40/115/137. Only the three fixed [RNN](../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../glossary.md#term-long-short-term-memory-lstm) signatures remain deferred. Symbolic [dimensions](../../glossary.md#dimension) in
the formula do not widen ordinary public CPU named-B convolution [admission](../../glossary.md#term-admission). [General Scan](../../glossary.md#general-differentiable-scan) uses
this same ordinary formula and independently qualifies every generated body [occurrence](../../glossary.md#term-occurrence); Conv3d
is not a separate Scan backward exclusion.

Before constructing any backward expression, the compiler must inventory every
backward-reachable operation occurrence and its exact attributes and derivative policies. Any
unsupported or ambiguous occurrence fails closed. Full [inference](../../glossary.md#term-inference-meanings) and [validation](../../glossary.md#term-validation) still occur only
after the one combined capture, so a later construction, capture, inference, validation, or
optimization failure may consume temporary model-level [`TensorId`](../../glossary.md#tensorid) values. Tensor IDs are [opaque](../../glossary.md#term-opaque)
and are never rolled back or reused.

[Seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings) and derivative constants are storage-free Tensor [leaves](../../glossary.md#term-leaf) or expressions explicitly
registered as compile-time constant splats. Tensor storage, labels, [descriptor](../../glossary.md#term-descriptor) shape, factory
history, or [provenance](../../glossary.md#provenance) absence must never silently imply a compile-time constant.

Phase-aware capture receives the ordered forward outputs, requested gradient roots and their
target roles, the identity set of original forward [producers](../../glossary.md#term-producer), and explicit constant facts. It
assigns every [`NodeId`](../../glossary.md#nodeid) and [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) exactly once. Nodes whose [producer](../../glossary.md#term-producer) identity belongs to the
original forward set have phase `FORWARD`; generated derivative producers have phase `BACKWARD`.
Per-node `GraphPhase` remains authoritative and must not be replaced by only a positional
backward-start index.

Distinct differentiation targets may legitimately resolve to the same captured gradient
`ValueId`. Gradient result roles map each target independently, while the graph's public output
boundary contains each distinct gradient value once. The compiler must not create identity nodes
solely to make those result values distinct.

Optimization operates on the [immutable](../../glossary.md#term-immutable) combined graph. The guarded seven-rule exact arithmetic
scan, [constant folding](../../glossary.md#term-constant-folding), [dead-code elimination](../../glossary.md#term-dead-code-elimination-dce), and [common-subexpression elimination](../../glossary.md#term-common-subexpression-elimination-cse) established by
compiler tasks 0003, 0003A, and 0003B must be reassessed for both phases and applied only where
their existing semantic guards remain valid. Common-subexpression elimination is phase-local
unless a later architecture update and proof establishes a broader safe rule. Every changed
candidate is revalidated through the compiler's inference-and-validation boundary. Rewrites must
retain phase, descriptor, output, saved-value, gradient, special-value, and [rounding](../../glossary.md#term-rounding) guards.
[Finite](../../glossary.md#term-finite) tolerance or a provider observation cannot license `x * 0`, `x - x`, [reassociation](../../glossary.md#term-reassociation),
contraction, or [fusion](../../glossary.md#term-fusion) across an observable boundary. Unproved legality keeps the old graph path.
This contract does not authorize new algebraic rewrites.

Generated gradient formulas are ordinary [differentiable](../../glossary.md#derivative-and-chain-rule) Tensor expressions. A functional request
contains exactly one or two ordered reverse-mode stages. Every stage has non-empty ordered output
references, an output-aligned list of optional [cotangent](../../glossary.md#cotangent) seeds, and a non-empty ordered target
list whose exact Tensor references are unique and belong to the complete original forward
inventory. Stage one selects exact Tensors from the requested forward-output boundary through
forward references. Stage two, when present, selects only generated first-stage gradients through
a stage-one target index.

An absent [seed](../../glossary.md#term-seed) means an exact typed positive one only for a scalar, floating,
gradient-eligible output. A present seed must have the output's exact Shape and floating data type
and must not itself request gradients. `DisconnectedPolicy.ERROR` rejects a valid target that has
no differentiable route from the selected outputs. `DisconnectedPolicy.ZERO` instead returns an
ordinary exact typed zero expression; several target roles may share that same final [graph value](../../glossary.md#graph-value).

A one-stage request requires `createGraph == false`. A two-stage request requires
`createGraph == true` for stage one and `false` for stage two. `createGraph` retains first-stage
formulas only for the immediate second stage inside the same compile. It does not establish an
arbitrary nested or persistent derivative chain. A third stage and derivative order greater than
two are unsupported.

The compiler captures the original forward outputs and all requested stage-one and stage-two
gradient roots together once. `DerivativeGraphMetadata` augments, but does not replace,
`GraphPhase`: original forward producers have derivative order zero, producers first owned by the
first reverse stage have order one, and producers first owned by the second reverse stage have
order two. Both derivative orders one and two remain `BACKWARD` [graph phase](../../glossary.md#graph-phase).

Gradient [publication](../../glossary.md#term-publication) [bindings](../../glossary.md#term-binding) exist only for derivative orders one and two. They are ordered by
derivative order and then stage-local target index, retain the target identity and final gradient
value, and remain target-distinct when several targets share one captured value. The graph output
boundary starts with the ordered forward values and then lists each previously unseen gradient
value once in binding order.

The bounded [lifecycle](../../glossary.md#lifecycle) creates no persistent or [runtime](../../glossary.md#term-runtime) tape, [mutable](../../glossary.md#term-mutable) Tensor gradient state,
`Tensor.backward()`, model-owned derivative rules, backend-owned global autograd, or second
gradient algebra. The ordinary `Engine.backward(...)` convenience is narrower than the complete
functional request boundary: it always requests one first-order stage for one [scalar objective](../../glossary.md#loss),
uses an absent seed, sets `createGraph == false`, and uses `DisconnectedPolicy.ERROR`. The full
bounded request surface is available through the advanced compile boundary.

General scan is the narrow first-order exception specified by the
[scan owner's reverse protocol](recurrent-scan.md#reverse-scan-and-exact-primal-bindings).
Its accepted [symbolic extension](recurrent-scan.md#accepted-symbolic-batch-scan) keeps
Model [PURE](../../glossary.md#term-pure)/[dtype](../../glossary.md#data-type--datatype) validity separate from bounded FLOAT32/FLOAT64 Compiler [AD](../../glossary.md#autograd) and initial FLOAT32
CPU qualification. Compiler must prove the entire actual generated closure, including bounded
rank-three outer seed/constant/[layout](../../glossary.md#layout) plumbing and scoped obligations; ordinary [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) binding
does not imply that support. Count, batch-contracting adjoints, shared empty outputs and separate
fixed-T zero plans follow the Scan semantic owner. No forward-only or handwritten-reverse test
substitutes for actual first-order generation. Compiler0014 covers these Compiler obligations,
independently reviewed and MAIN accepted 2026-10-25; execution and numeric resource qualification
remain separate.
ScanGradientRules inventories source algebra, combines joint output seeds once per occurrence and
calls existing rules with a new explicit compile-local [primal](../../glossary.md#term-primal)-operand binding function. Every
primal operand, including a saved [activation](../../glossary.md#term-activation) output, must resolve through that function; original
Tensor contribution identities remain intact. This is a rule protocol extension, not a claim
that today's rules work unchanged. Derivative constants close through explicit [invariant](../../glossary.md#term-invariant) bindings
and registered outer splats/expressions; no undeclared leaves or ValueId-to-Tensor AD map is allowed.

[Compiler0017A](../../planning/modules/compiler/tasks/0017a-ordinary-lexical-primal-binding.md#result)
completes this existing resolver protocol through all ordinary PURE derivative owners, including
saved statistics/indices and nondifferentiable value operands, with independent source review and
actual lexical [VJP](../../glossary.md#term-vector-jacobian-product-vjp)/ordinary [regression](../../glossary.md#term-regression) evidence. It adds no mathematical rule or new Scan source,
inference, AD, execution, cold/wire or higher-order admission. Wider typed/demand/[coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) proof
remains separately required.

[Preflight](../../glossary.md#term-preflight) enforces [residual](../../glossary.md#scan-residual-port-and-history-demand)-root, unsupported-body and scan higher-order/createGraph rejection
before derivative allocation, including unsupported scans on unrelated original branches. Capture
retains original forward wrappers and independently captures each forward/reverse occurrence's
body, even for reused immutable templates. ScanBodyInference validates all scoped descriptors and
roles without outer ValueIds leaking inward. Layout closure, reindexing, [DCE](../../glossary.md#compiler-transformations-and-dominance)/[CSE](../../glossary.md#compiler-transformations-and-dominance) and fusion preserve
live saved sites and revalidate demand; neither scan occurrences nor [scope](../../glossary.md#term-scope) identities may merge.
The accepted [shared coordinate contract](recurrent-scan.md#bounded-row-role-analysis) requires
raw scoped inference and derivative role discovery to recompute all [canonical](../../glossary.md#term-canonical-form) output facts through
the Model semantic owner after independent ordinary inference. It does not replace that inference,
establish selected VJP closure or widen the old cold/AD/physical domains. The Model0048 cutover
must preserve those independently owned effective admission predicates.
Compiler0011 implements these compile-time obligations for its bounded FLOAT32/FLOAT64 first-order
domain, without body optimization or fusion. Prepare0010 implements scoped projection and static
binding/demand validation. CPU0030 executes the independently qualified bounded FLOAT32 same-shape
[pointwise](../../glossary.md#term-pointwise) forward/actual-reverse [slice](../../glossary.md#slice). CPU0031 extends execution to qualified [homogeneous](../../glossary.md#term-homogeneous) FLOAT32
broadcast, SUM/sumToShape and bounded shape/layout closure; actual reverse bodies must independently
qualify. CPU0032 adds static homogeneous FLOAT32 rank-two MATMUL under that same qualification,
using the existing Compiler derivative algebra. CPU0033 implements homogeneous FLOAT64 execution
across the complete bounded declared algebra and actual first-order reverse closure using those
same rules; it changes no Compiler [ownership](../../glossary.md#term-ownership-meanings), saved-value semantics or higher-order admission.


Compiler0013 implements length-mode inference and first-order reverse construction under the
[scan owner's first-order restriction](recurrent-scan.md#first-order-length-mode-reverse).
Whole-original preflight checks every [carry](../../glossary.md#term-carry) and sequence, including unused/non-gradient positions
and unrelated branches, before derivative IDs. Each incoming cotangent for an original SHARED
value is normalized before append/accumulation, including shared intermediates at B=1. Exact
typed row-zero/literal seeds have explicit sequence roles; no SHARED expansion invents ROW.
Bounded SUM restoration uses row-leading layout edits for checked ROW(1) [feature](../../glossary.md#term-feature) adjoints, not
new arithmetic or saved-primal loads. Capture, [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), guarded outer rewrites and final
queries preserve roles, saved sites, demand and the lengths-actual/no-formal distinction.
Old all-valid AD and every higher-order rejection remain unchanged. This compile-time support
does not enable cold source copying, [Prepare](../../glossary.md#prepare), CPU execution, encoding or owned [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) workflows.

Compiler0015 combines that existing length-mode protocol
with one named B for fixed T/features and homogeneous FLOAT32/FLOAT64. The
[sole Scan owner](recurrent-scan.md#runtime-preflight-and-implementation-gates) defines its support
gate, unchanged original-source ROW(0) restriction and generated mixed roles. Independent inference
and the actual generated closure must prove Dimension provenance, including ROW(1) constant sequences,
SHARED histories and parameter-shaped accumulators, without [sampling](../../glossary.md#term-sampling-machine-learning-and-hyperparameters) B or changing derivative algebra.
Numerical qualification uses existing test-local [evaluation](../../glossary.md#term-evaluation) of the actual generated graph against
independent nonuniform active-prefix [unrolling](../../glossary.md#term-unrolling); only the [oracle](../../glossary.md#term-test-oracle) reads length values. This is not
a production runner or new test framework. Whole-original preflight rejection before derivative IDs,
exact primal bindings and complete fresh scoped all-port admission remain mandatory. Execution,
cold/wire and NN/Training gates stay closed. Compiler0015's completed acceptance is recorded in its
[task result](../../planning/modules/compiler/tasks/0015-symbolic-length-scan-inference-ad-and-binding.md#result).

## [Compile](../../glossary.md#compile) [lifecycle](../../glossary.md#lifecycle)

Cold captureSource/reconstructSource may additionally admit general `ScanKind.SCAN` occurrences
with a complete independently scoped body under the bounded constructor inventory below. This
approved exception is not implemented by the authority edit itself and grants no [persistence](../../glossary.md#term-persistence),
[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) acquisition, [AD](../../glossary.md#autograd) or [backend capability](../../glossary.md#backend-capability). Bounded [Compiler](../../glossary.md#term-compiler) [cold reconstruction](../../glossary.md#term-cold-reconstruction) and [Checkpoint](../../glossary.md#term-checkpoint)
attachment-preserving in-memory copy/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction) and NN's bounded owned workflow are implemented.
The [owning extension contract](extensions-training.md#generic-graph-source-and-values) records
the separate Checkpoint/NN gates. Implemented
[graph-model v4 persistence](extensions-training.md#canonical-graph-model-version-4)
changes none of this cold reconstruction inventory or protocol.

The supported `GraphCompilationPort` also owns a cold source boundary independent of ordinary
[compilation](../../glossary.md#compiler-engine-and-execution-stage-names): `captureSource` [captures](../../glossary.md#term-structural-capture) once with empty [constant](../../glossary.md#term-constant) ingress and collects exact [leaves](../../glossary.md#term-leaf)
at first input-ID assignment; `reconstructSource` substitutes [descriptor](../../glossary.md#term-descriptor)-compatible [Tensors](../../glossary.md#tensor) by
[graph](../../glossary.md#graph)-input position. The source is the raw graph, not optimized `CompileArtifacts.graph()`.
Both operations admit only fully static FORWARD sources. The unchanged **flat outer-source**
inventory is the following closed [slice](../../glossary.md#slice), plus the explicit general-SCAN exception below. The
original eight kinds remain ADD/MUL/NEG/[MATMUL](../../glossary.md#matrix-multiplication--matmul)/RELU/[TANH](../../glossary.md#term-tanh)/[SIGMOID](../../glossary.md#term-sigmoid) with
`NoOperationAttrs.INSTANCE` and [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) PERMUTE with exactly `[1,0]`. [Activations](../../glossary.md#term-activation) retain [Model](../../glossary.md#term-model)'s
floating-only guards and reconstruct through `Tensor.relu/tanh/sigmoid`, without [decomposition](../../glossary.md#term-decomposition).

The additional kinds have these exact source [signatures](../../glossary.md#term-signature). Model guards and family [inference](../../glossary.md#term-inference-meanings) still
apply; an [operation](../../glossary.md#operation) outside these attributes or arities is unsupported, even if executable.

| Kind | Inputs and attributes | Public family reconstruction |
|---|---|---|
| SUB | Two floating operands; `NoOperationAttrs.INSTANCE`; one output. | `a.sub(b)` |
| RESHAPE | One input; `TargetShapeAttrs` with positive fully static [target](../../glossary.md#term-target) extents and equal checked [element](../../glossary.md#term-element) count; one output. | `a.reshape(shape)` |
| EXPAND_DIMS / SQUEEZE | One input; `AxisTransformAttrs`, [axis](../../glossary.md#term-axis) 2 only; SQUEEZE requires that axis to be [singleton](../../glossary.md#term-singleton); one output. | `a.expandDims(2)` / `a.squeeze(2)` |
| SELECT | One [rank](../../glossary.md#term-rank)-three input; `SelectAttrs`, axis 0 and an in-bounds static index; one output. | `a.select(axis,index)` |
| [GATHER](../../glossary.md#gather) | Rank-two data and [rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) non-[gradient](../../glossary.md#term-gradient) [INT64](../../glossary.md#data-type--datatype) indices; `IndexAxisAttrs`, axis 0; one output. | `data.gather(indices,axis)` |
| SLICE | One rank-two input; `SliceAttrs` with exactly one nonempty last-axis entry, normalized start/length and step 1; one output. | `a.sliceByLength(starts,lengths,axes,steps)`; never reverse-normalize ends. |
| CONCAT | Positive input count, rank-three inputs; `CompositionAxisAttrs`, axis 0; one output. | `Tensor.concat(axis,inputs)` |
| [CONV2D](../../glossary.md#convolution--conv2d) / [CONV3D](../../glossary.md#conv3d--ncdhw) | Exactly three inputs in input/[weight](../../glossary.md#term-weight)/[bias](../../glossary.md#term-bias) order; `Conv2dAttrs.defaults()` / `Conv3dAttrs.defaults()` geometry only; one output. | `input.conv2d(weight,bias,attrs)` / `input.conv3d(weight,bias,attrs)` |
| MAX_POOL2D | One input; `MaxPool2dAttrs` with positive [kernels](../../glossary.md#kernel), [stride](../../glossary.md#term-stride)=[kernel](../../glossary.md#kernel), [padding](../../glossary.md#padding) 0, [dilation](../../glossary.md#term-dilation) 1, floor mode; one output, no index port. | `a.maxPool2d(attrs)` |
| MAX_POOL3D / AVERAGE_POOL3D | One input; exact MaxPool3dAttrs / AveragePool3dAttrs, positive kernel/stride/dilation, nonnegative symmetric padding and literal floor/ceil geometry under existing Model guards; one output. | `a.maxPool3d(attrs)` / `a.averagePool3d(attrs)` |
| RNN_TANH / GRU_RESET_AFTER / [LSTM](../../glossary.md#term-long-short-term-memory-lstm) | Exactly the six biased/unbiased fixed signatures and two directions in the [recurrent contract](recurrent-scan.md#fixed-family-and-current-model-surface); two outputs for [RNN](../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../glossary.md#term-gated-recurrent-unit-gru), three for LSTM. | Call `RecurrentScan.rnn/gru/lstm` once, direction last; retain every returned [canonical](../../glossary.md#term-canonical-form) component. |

CONV1D is not a new kind: the selected composition expands input/weight at axis 2, uses biased
default CONV2D and squeezes axis 2. Unbiased/nondefault convolutions and MAX_POOL2D geometry
outside the row above remain unsupported. [Pool3d](../../glossary.md#pool3d--ncdhw-pooling) alone admits the complete typed geometry above.
Fixed scans remain one flat [producer](../../glossary.md#term-producer), never static
[unrolling](../../glossary.md#term-unrolling), a body, [callback](../../glossary.md#term-callback) or region. Their static time and [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) may be zero; [feature](../../glossary.md#term-feature)/hidden
sizes remain positive. Cold [admission](../../glossary.md#term-admission) reads no GATHER indices or scan valid-length values;
execution owns their value bounds. Cold admission itself grants no execution or gradients;
those require independent Compiler AD and complete [backend](../../glossary.md#backend)-closure [qualification](../../glossary.md#term-qualification).

### Bounded general-scan [cold reconstruction](../../glossary.md#term-cold-reconstruction)

The implemented cold inventory includes the all-valid constructors below and the fully static
floating length-mode intersection specified here.
[Compiler0018](../../planning/modules/compiler/tasks/0018-static-length-scan-cold-reconstruction.md#result)
records the original [homogeneous](../../glossary.md#term-homogeneous) [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) implementation and its [qualification](../../glossary.md#term-qualification). The per-role
floating cutover below extends source [admission](../../glossary.md#term-admission); it does not claim new [backend](../../glossary.md#backend)/[runtime](../../glossary.md#term-runtime) qualification.
Preserve the [scan header and roles](recurrent-scan.md#header-bindings-and-source-admission), with
complete [preflight](../../glossary.md#term-preflight) of late/dead/zero-time scans before any formal/[producer](../../glossary.md#term-producer) IDs or [host](../../glossary.md#term-host) copy/import.
Compiler0013's ordinary length-mode [inference](../../glossary.md#term-inference-meanings)/first-order reverse construction is separate:
inference success is necessary but is not the independent cold admission predicate.

[General SCAN](../../glossary.md#general-differentiable-scan) uses existing `ScanAttrs`, `CompiledScanBody`, `ScanBody` and `Scan.apply`; no new
[graph](../../glossary.md#graph), [API](../../glossary.md#term-application-programming-interface-api), type, session or executor is selected. [Model](../../glossary.md#term-model) [PURE](../../glossary.md#term-pure)-body validity is broader than cold
constructor support, which is independent of [dtype](../../glossary.md#data-type--datatype)-specific execution/[AD](../../glossary.md#autograd) and wire eligibility.
Preserve every applicable Model dtype of an admitted [signature](../../glossary.md#term-signature), including low [TANH](../../glossary.md#term-tanh) and F32 RELU
without a [CPU](../../glossary.md#term-cpu) route; never use a provider query as cold admission.

The **body-only** inventory reuses the flat constructors above, except fixed scans (nested scans
remain forbidden), with these explicit overrides/additions. All have one output and exact raw
family inference; attributes/arities not listed remain unsupported. Existing [MATMUL](../../glossary.md#matrix-multiplication--matmul) constructor
semantics, including [vector](../../glossary.md#term-vector)/batched/promotion forms, stay intact; [rank](../../glossary.md#term-rank) two bounds initial execution,
not cold copying. No arbitrary PURE-body copier is authorized.

| Body signature | Exact attributes and public reconstruction |
|---|---|
| [Tensor](../../glossary.md#tensor) ADD/SUB/MUL/DIV, two inputs | `NoOperationAttrs.INSTANCE`; `a.add/sub/mul/div(b)`. ADD/SUB/MUL use Model numeric category/promotion/broadcast guards; DIV floating only. Body SUB includes integral forms, unlike outer SUB. |
| [Scalar](../../glossary.md#term-scalar) ADD/SUB/MUL/DIV, one input | `ScalarValueAttrs`; `a.add/sub/mul/div(attrs.value())`, preserving exact typed [ScalarValue](../../glossary.md#typed-scalar-value--scalarvalue). Numeric ADD/SUB/MUL; floating DIV; scalar dtype equals input. |
| NEG/RELU/TANH/[SIGMOID](../../glossary.md#term-sigmoid), one input | `NoOperationAttrs.INSTANCE`; existing floating family constructor, not [decomposition](../../glossary.md#term-decomposition). |
| SUM, one input | [`NoOperationAttrs`](../../glossary.md#nooperationattrs) → `sum()`; `AxisReductionAttrs` → `sum(axis,keepDimensions)`; `MultiAxisReductionAttrs` → `sum(axes,keepDimensions)`; `SumToShapeAttrs` → `sumToShape(targetShape)`. Numeric Model guards; preserve [axis](../../glossary.md#term-axis) order, empty axes and keep-[dimensions](../../glossary.md#dimension) [identity](../../glossary.md#term-identity-meanings). No masked SUM. |
| EXPAND / RESHAPE, one input | `TargetShapeAttrs`; `expand(shape)` / `reshape(shape)`. Fully static normalized [target](../../glossary.md#term-target); exact broadcast / equal checked [element](../../glossary.md#term-element) count. Body reshape includes scalar/zero extents accepted by the [Shape](../../glossary.md#shape) overload, not raw `-1` requests. |
| EXPAND_DIMS / SQUEEZE, one input | `AxisTransformAttrs`; `expandDims(axis)` / `squeeze(axis)` at any normalized valid axis; squeeze [singleton](../../glossary.md#term-singleton) only. |
| PERMUTE, one input | Existing [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) `PermutationAttrs [1,0]` only; `permute(1,0)`. |
| [CONTIGUOUS](../../glossary.md#term-contiguous), one input | `NoOperationAttrs.INSTANCE`; `contiguous()`; retain exact [canonical](../../glossary.md#term-canonical-form) result geometry without reading/copying storage. |
| CAST, one input | `CastAttrs`; `cast(targetDataType)` between any two floating dtypes, or identity for nonfloating dtypes. Preserve the explicit target, exact inferred result dtype/[gradient](../../glossary.md#term-gradient)/[layout](../../glossary.md#layout) and fresh producer; never replace casts with [aliases](../../glossary.md#aliasing-and-arena) or normalize stored [descriptors](../../glossary.md#term-descriptor) to F32. Nonfloating cross-dtype casts remain outside this cold closure. |

[Constants](../../glossary.md#term-constant) are explicit outer input substitutions bound to [invariant](../../glossary.md#term-invariant) formals, or scalar [operation](../../glossary.md#operation)
attributes; no free/host-backed body Tensor, hidden literal producer or constant sidecar is added.
Other existing flat mappings retain their exact guards inside bodies; new body mappings must not
leak into flat outer admission. Unsupported PURE operations remain Model-valid but cold-unsupported.

Length-mode cold admission intersects that same constructor inventory with the existing
ROW(0), rank-two ROW(1), SHARED and terminal-SHARED length rules. Every [carry](../../glossary.md#term-carry)/sequence/invariant
actual, formal and body value must be floating ([FLOAT32](../../glossary.md#term-float32), [FLOAT64](../../glossary.md#term-float64), [BFLOAT16](../../glossary.md#term-bfloat16), [FLOAT16](../../glossary.md#term-float16), FLOAT8_E4M3 or FLOAT8_E5M2), including
unused declarations, dead interiors and intermediates hidden by casts. Each positional actual
matches its own formal, and each next carry matches its own initial carry dtype/shape; roles need
not share one dtype. Ordinary family inference owns promotion and exact result descriptors.
Explicit low↔F32 (or other floating) CAST boundaries retain both original-storage and working-site
[metadata](../../glossary.md#term-metadata); there is no implicit compute-dtype rewrite. Nonfloating length-body values remain
unsupported, including otherwise valid identity casts. The separate final non-gradient [INT64](../../glossary.md#data-type--datatype)[B]
lengths actual has no body formal; no length value is read. Reject symbolic geometry and deferred
obligations here, without [narrowing](../../glossary.md#term-narrowing) the existing all-valid cold inventory or querying a backend.
The length constructor whitelist additionally admits only MAX_POOL3D and AVERAGE_POOL3D with
their exact typed attributes. This does not relax static geometry, ROW/SHARED, terminal-SHARED,
role/axis or no-deferred-obligation restrictions, nor admit other [CNN](../../glossary.md#term-convolutional-neural-network-cnn) constructors.

Existing graph-model V4/V5/V6 descriptor/scalar dtype words and body CAST [opcode](../../glossary.md#term-opcode) 32 already
represent the original four floating types and explicit targets; length opcode 33 retains its
role fields. [FP8](../../glossary.md#term-fp8) uses the append-only graph-model v7 dtype namespace; old-only sources retain
their smallest sufficient old version and bytes. Copy/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction)/save/load preserve raw low named-state
bytes, aliases and canonical ports, and validate complete source metadata before
candidate identities or state import. Cold support alone does not authorize numerical execution;
CPU admission follows its low-[precision](../../glossary.md#term-precision) parity cutover. Training objectives remain scalar gradient-eligible F32/F64, so low outputs
require an explicit ordinary cast when used as an [objective](../../glossary.md#term-objective).

Before any formal or producer allocation, validate positional substitutions against each
length [occurrence](../../glossary.md#term-occurrence)'s outer ROW/SHARED roles. Newly tied actuals must denote the same outer row
axis or both be SHARED; step axis r maps to outer axis r+1. Descriptor equality alone is not
sufficient to admit a newly aliased ROW/SHARED or differently mapped pair. Preserve compatible
repeated substitutions and [opaque](../../glossary.md#term-opaque) upstream [provenance](../../glossary.md#provenance). Reconstruct exactly c+s+i formals,
[slice](../../glossary.md#slice) invariants at c+s+i, and supply the final lengths actual separately to the length overload,
preserving rowCarryCount, every ordered optional sequence axis, traversal and all canonical ports.

Preflight the complete outer graph, every contained [node](../../glossary.md#node)/value (including dead outer scans and
T=0 bodies), all formal/role/[residual](../../glossary.md#scan-residual-port-and-history-demand) [bindings](../../glossary.md#term-binding), phases, exact descriptors, static element counts
and descriptor-layout arithmetic before creating **any** formal or producer. Reuse raw inference
and canonical residual/formal-geometry [validation](../../glossary.md#term-validation); reject deferred obligations, nested scans and
unknown families. Do not add physical byte/resource/[alignment](../../glossary.md#term-alignment) admission or a canonical outer-layout
filter: those remain [Checkpoint](../../glossary.md#term-checkpoint)/backend concerns. Preserve existing raw unresolved MATMUL/CONCAT
layout cases. Structural constructors still own missing attachments and malformed [topology](../../glossary.md#term-topology). [Scope](../../glossary.md#term-scope)
failure in a later body must precede even an earlier valid outer producer's allocation.

Cold admission additionally requires empty `demandedOutputIndices` and `body.graph().outputs()`
exactly equal to the first-occurrence-unique concatenation of `nextCarry`, `emissions` and
`residuals`, in that encounter order. Reject nonempty demand or extra, reordered or missing role
outputs before **any** formal/producer allocation, including late/dead outer scans and T=0.
These are raw cold-source checks only; ordinary [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) may retain nonempty demand and is
not narrowed by this reconstruction boundary.

After preflight, use separate call-local scope tables only. Create fresh distinct canonical
storage-free formals in declaration order, including unused positions; reconstruct each body
node once, one ScanBody and one Scan.apply per occurrence, never T-unroll. Bind every ordered
canonical final-carry/emission/latent-[residual port](../../glossary.md#scan-residual-port-and-history-demand) exactly once, including unrequested ports and
aliased role bindings. Both traversal flags, repeated actual positions and distinct occurrences
survive; reuse of one template does not merge scopes. Body IDs never resolve in an outer table.
Substitute only outer inputs; produced/repeated descriptor-compatible replacements retain opaque
upstream provenance. Discard all reconstruction tables on return/failure; retain no original
body/formals/template and introduce no [autograd](../../glossary.md#autograd) conversion map or persistent captured graph [owner](../../glossary.md#term-owner).

All stored nodes, including dead nodes, and all values
must pass static checked geometry, family inference, exact descriptor validation, and absence of
deferred obligations before reconstruction creates any producer. Model constructors continue to
own structural closure, attributes/signatures, and layout geometry.

`GraphCompilationPort.validateSource(CompiledGraphModel)` exposes this same complete metadata
preflight without creating any Tensor or consuming Tensor identities, reading/copying [host storage](../../glossary.md#host-storage),
or allocating tensor storage. It validates all stored nodes and contained bodies before consumers
create candidate [leaves](../../glossary.md#term-leaf); it does not validate leaf identity/substitution associations or grant
execution/[persistence](../../glossary.md#term-persistence) capability. Checkpoint decode must call it for every wire version before
creating any candidate Tensor, then retain ordinary acquisition/reconstruction association checks.

Use raw `CapturedGraphInference`, not `StaticResultLogicalLayoutClosure` or an optimized graph.
SELECT and SLICE derive [views](../../glossary.md#term-view-meanings) only from present input layouts; CONCAT and fixed-scan outputs remain
unresolved. RESHAPE derives a view only from present contiguous geometry, otherwise remaining
unresolved. Raw unresolved MATMUL→SLICE/RESHAPE and CONCAT→SELECT/RESHAPE chains are valid when
their exact descriptors match raw family inference and no [deferred obligation](../../glossary.md#term-deferred-obligation) remains. Do not
force copies, resolve missing layouts or read storage to make a source fit admission.

`SourceCapture` snapshots unique provenance-free exact leaf references in graph-input order,
with full descriptor equality, even for direct record construction. It borrows those Tensors
without owning storage, a [close lifecycle](../../glossary.md#ownership-borrowing-and-lease), or a [cache](../../glossary.md#term-cache). Reconstruction snapshots its supplied list
and checks count, nulls, and full descriptor equality, but permits ordinary produced expressions
and repeated references as explicit substitutions, subject to the length-role alias checks above.
Their upstream provenance is opaque to source admission and remains attached for later ordinary
[compile](../../glossary.md#compile)/autograd. Position is the association contract; same-descriptor swaps cannot be detected.

Only after complete preflight may reconstruction allocate call-local [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot)-to-Tensor
tables, one per scope. It constructs each stored node once in [topological order](../../glossary.md#term-topological-order) through its public family
constructor, mapping every ordered output slot to its canonical wrapper, retaining sharing,
repeated port uses and output order, never
merging occurrences or fabricating intermediate placeholders. The table is discarded on return
or failure; returned outputs retain only ordinary provenance. Zero-node pass-through and unused
inputs are valid source structure, not an [Engine](../../glossary.md#term-engine) executability promise. Neither cold operation
optimizes, closes descriptors, accesses or allocates storage, selects backends, executes, changes
ordinary compilation/autograd, or permits an autograd conversion map or another gradient algebra.

Compile [lifecycle](../../glossary.md#lifecycle):

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
- physical [buffers](../../glossary.md#term-buffer)
- [kernel](../../glossary.md#kernel) routes
- runtime workspaces
- backend-specific DAGs

## Profile-free [compile](../../glossary.md#compile) [identity](../../glossary.md#term-identity-meanings)

<a id="numerical-profile-compile-identity"></a>

The compile port, capability queries, and
`CompileArtifacts` carry no numerical-profile selector or replacement identity. [Compiler](../../glossary.md#term-compiler) retains
its semantic guards, [graph](../../glossary.md#graph)-phase and [derivative](../../glossary.md#term-derivative) [metadata](../../glossary.md#term-metadata), and complete [occurrence](../../glossary.md#term-occurrence) facts. It does
not convert test-only arithmetic tolerances into optimization permissions or provider capability.

## Accepted ordinary materializing RESHAPE final [layout](../../glossary.md#layout)

[ADR0062](../../design/decisions/0062-ordinary-materializing-reshape-closure.md) and
[Compiler0019](../../planning/modules/compiler/tasks/0019-static-materializing-reshape-logical-layout-closure.md)
authorize a bounded final-layout extension, not current execution [qualification](../../glossary.md#term-qualification). The second
[topological](../../glossary.md#term-topological-order) StaticResultLogicalLayoutClosure phase prefers existing exact family-inferred [views](../../glossary.md#term-view-meanings).
For validated fully static RESHAPE with present noncontiguous input geometry and an unresolved
inferred result, it may select [canonical](../../glossary.md#term-canonical-form) non-view logical output through existing closeValue.
Preserve inferred [dtype](../../glossary.md#data-type--datatype)/[Shape](../../glossary.md#shape) agreement, captured [gradient](../../glossary.md#term-gradient) eligibility, constraints, [identities](../../glossary.md#term-identity-meanings),
[topology](../../glossary.md#term-topology) and sidecars. Do not broaden the first-phase family classifier or fabricate dynamic or
missing-input geometry. Static [scalar](../../glossary.md#term-scalar)/empty geometry follows ordinary checked arithmetic.

Raw [Model](../../glossary.md#term-model)/LayoutInference/CapturedGraphInference and source/[cold reconstruction](../../glossary.md#term-cold-reconstruction) remain unchanged;
the raw unresolved [descriptor](../../glossary.md#term-descriptor) is still valid source and is not rewritten for [persistence](../../glossary.md#term-persistence).
No [VJP](../../glossary.md#term-vector-jacobian-product-vjp) formula, consumer [CONTIGUOUS](../../glossary.md#term-contiguous), physical [buffer](../../glossary.md#term-buffer), [backend route](../../glossary.md#backend-route) or [codec](../../glossary.md#term-codec) change follows.
CPU0079 and Metal0081 must independently qualify their materializing consumers. Existing exact
offset views remain exact; selecting [materialization](../../glossary.md#materialization) is not proof that every other [alias](../../glossary.md#aliasing-and-arena) is
mathematically impossible.
