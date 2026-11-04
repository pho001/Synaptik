# Task 0005: [Publication](../../../../glossary.md#term-publication), [Planning](../../../../glossary.md#term-planning) Orchestration, and [Compile Artifacts](../../../../glossary.md#compile-artifacts)

## Status

Complete

## Goal

Complete the current package-private [compile](../../../../glossary.md#compile) [lifecycle](../../../../glossary.md#lifecycle) after `GraphCompilation` by deriving
[publication](../../../../glossary.md#term-publication) roles, orchestrating the completed [backend](../../../../glossary.md#backend)-neutral [planning](../../../../glossary.md#term-planning) operations, and returning
one [immutable](../../../../glossary.md#term-immutable) public `CompileArtifacts` recipe.

The completed flow is:

```text
final GraphCompilation
  -> validate and bind ordered forward-output and gradient-result publication roles
  -> query capability and select one BackendId owner for each node in graph order
  -> assemble the complete node-to-owner map
  -> derive maximal consecutive same-owner partitions
  -> derive logical memory requirements
  -> transport compile-time constants, bindable inputs, and deferred diagnostics
  -> immutable CompileArtifacts
```

[Compiler](../../../../glossary.md#term-compiler) remains the [graph](../../../../glossary.md#graph)-wide orchestrator. Planning continues to own the backend-neutral
eligibility, baseline [owner](../../../../glossary.md#term-owner)-selection, [partition](../../../../glossary.md#partition)-generation, and logical-memory semantics. This
task opens only the three package-cohesive stateless Planning operations required by that concrete
consumer: one owner-selection collaboration that hides the eligibility intermediate, plus the
already-audited maximal-partition and logical-memory operations. It does not add a public planning
workflow, [registry](../../../../glossary.md#term-registry), service, [cost model](../../../../glossary.md#compiler-transformations-and-dominance), [device](../../../../glossary.md#term-device) selection, or orchestration policy.

The result is compile-time state only. It contains no live backend provider, availability
[snapshot](../../../../glossary.md#term-snapshot), device selection, route, [kernel](../../../../glossary.md#kernel), physical [buffer](../../../../glossary.md#term-buffer), byte count, [lifetime](../../../../glossary.md#term-lifetime), slot, transfer,
[prepared schedule](../../../../glossary.md#prepared-schedule--preparedschedule), executable, [runtime](../../../../glossary.md#term-runtime) [residency](../../../../glossary.md#residency), [trace](../../../../glossary.md#trace) event, or [mutable](../../../../glossary.md#term-mutable) [run state](../../../../glossary.md#run-state--runstate).

## Current gap

[Compiler](../../../../glossary.md#term-compiler) 0001–0004B currently stop at package-private `GraphCompilation`. That result already
contains the final validated [graph](../../../../glossary.md#graph), exact [compile mode](../../../../glossary.md#compile-mode--compilemode), [forward graph](../../../../glossary.md#forward-graph)-output prefix, ordered
[target](../../../../glossary.md#term-target)-to-[gradient](../../../../glossary.md#term-gradient) roles, explicit logical-splat sidecar, bindable-input classification, deferred
constraints, and per-[node](../../../../glossary.md#node) phases. It does not:

- associate forward and gradient result roles with standalone [model](../../../../glossary.md#term-model) `PublicationBinding` values;
- prove that requested [publication](../../../../glossary.md#term-publication) values belong to the final graph output boundary;
- construct one capability query per final graph node;
- consume Planning's package-private eligibility, owner-selection, partitioning, or logical-memory
  operations across their sibling Java packages;
- assemble the complete node-to-[owner](../../../../glossary.md#term-owner) map;
- transport logical [constants](../../../../glossary.md#term-constant) into a cross-package [compile](../../../../glossary.md#compile) [artifact](../../../../glossary.md#term-artifact);
- expose deferred [compile diagnostics](../../../../glossary.md#compile-diagnostics) without exposing the internal predicate implementation; or
- produce `CompileArtifacts`.

[Planning](../../../../glossary.md#term-planning) 0001–0006 is complete with a `CLOSED` audit verdict. Its four evaluator/generator
operations are package-private because no external consumer previously existed. Compiler 0005 is
the concrete consumer that now justifies one narrow callable seam while preserving Planning's
existing semantics and Compiler's graph-wide orchestration [ownership](../../../../glossary.md#term-ownership-meanings).

## Terminology

- **Forward [publication](../../../../glossary.md#term-publication)** — an ordered `PublicationBinding` from the exact requested forward
  Tensor's [`TensorId`](../../../../glossary.md#tensorid) to its final [forward graph](../../../../glossary.md#forward-graph)-output [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot).
- **[Gradient](../../../../glossary.md#term-gradient) result publication** — an ordered `PublicationBinding` whose `TensorId` identifies
  the exact requested differentiation target and whose <span data-glossary-navigation="true"> ([differentiation target glossary](../../../../glossary.md#functional-gradient-request))</span>`ValueId` identifies that target's final
  gradient result. It does not add gradient state to <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>[`Tensor`](../../../../glossary.md#tensor).
- **[Publication plan](../../../../glossary.md#publication-plan)** — compiler-owned graph context plus separate ordered forward and gradient
  binding lists, validated against the final graph boundary.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [binding glossary](../../../../glossary.md#term-binding))</span>
- **[Compile constant plan](../../../../glossary.md#compile-constant-plan)** — the cross-package immutable form of Compiler 0003B's exact logical
  splats and bindable-input classification. It is logical data, not physical materialization.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- **[Compile diagnostics](../../../../glossary.md#compile-diagnostics)** — immutable successful-compile diagnostics derived from the final
  deferred constraints. The exact internal predicates remain available inside compiler without
  becoming a public binding or trace schema.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [schema glossary](../../../../glossary.md#term-schema))</span>
- **[Compile artifacts](../../../../glossary.md#compile-artifacts)** — the [immutable](../../../../glossary.md#term-immutable) non-executable aggregate handed to later [prepare](../../../../glossary.md#prepare) work.
- **[Planning](../../../../glossary.md#term-planning) callable seam** — the three public stateless operations used by Compiler across the
  Java package boundary: owner selection in <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>`planning.capability`, maximal partition generation
  in <span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>`planning.partition`, and logical-memory derivation in `planning.memory`. It is not a
  planner service, registry, facade, or graph-wide workflow.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [facade glossary](../../../../glossary.md#term-facade))</span>

## [Scope](../../../../glossary.md#term-scope)

### Package-private [compile](../../../../glossary.md#compile)-[artifact](../../../../glossary.md#term-artifact) entry

Keep the existing package-private five-[argument](../../../../glossary.md#term-argument) `GraphCompiler.compile(...)` entry and
`GraphCompilation` result unchanged. Add one package-private overload with this exact [shape](../../../../glossary.md#shape):

```java
static CompileArtifacts compile(
        CompileMode mode,
        List<Tensor> forwardOutputs,
        Optional<AutogradPreflight.FirstOrderRequest> firstOrderRequest,
        CompileTimeConstantGraph.Ingress forwardConstants,
        GraphOptimizationConfig optimizationConfig,
        BackendIntent backendIntent,
        PartitionScoringConfig partitionScoringConfig,
        List<BackendCapabilityProvider> capabilityProviders,
        List<BackendAvailabilitySnapshot> availabilitySnapshots)
```

The overload is the sole current complete compile-artifact entry. It is package-private, direct,
and stateless. Do not add a request aggregate, builder, service, public compile method,
`CompileConfig`, `CompiledGraph`, [engine](../../../../glossary.md#term-engine) [facade](../../../../glossary.md#term-facade), or alternate artifact [compiler](../../../../glossary.md#term-compiler).

Validate all nine top-level arguments in declaration order before [graph](../../../../glossary.md#graph) construction or any
[derivative](../../../../glossary.md#term-derivative) [Tensor](../../../../glossary.md#tensor) allocation. `forwardOutputs` retains the existing [element](../../../../glossary.md#term-element) and [identity](../../../../glossary.md#term-identity-meanings)
[validation](../../../../glossary.md#term-validation). Provider and [snapshot](../../../../glossary.md#term-snapshot) list elements are validated by the [Planning](../../../../glossary.md#term-planning) collaboration when
the first [node](../../../../glossary.md#node) is planned; a valid zero-node pass-through graph performs no capability query and
therefore does not inspect unused list elements. The caller collections are never retained or
mutated.

After top-level validation, invoke the existing five-argument graph-stage [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) exactly
once. Do not duplicate [capture](../../../../glossary.md#term-capture-meanings), [autograd](../../../../glossary.md#autograd), [inference](../../../../glossary.md#term-inference), [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), rewriting, folding, [DCE](../../../../glossary.md#term-dead-code-elimination-dce),
[CSE](../../../../glossary.md#term-common-subexpression-elimination-cse), or final validation.

### [Publication plan](../../../../glossary.md#publication-plan)

Add public final `io.github.pho001.synaptik.compiler.PublicationPlan` as an output-only [immutable](../../../../glossary.md#term-immutable)
type with a package-private constructor and exactly these public accessors:

```java
public CompiledGraphModel graph()
public List<PublicationBinding> forwardOutputs()
public List<PublicationBinding> gradientResults()
```

The plan retains the exact final [graph](../../../../glossary.md#graph) reference. It snapshots list membership while retaining the
exact `PublicationBinding` references.

Construct forward [bindings](../../../../glossary.md#term-binding) in original `forwardOutputs` order by pairing:

```text
forwardOutputs[i].id() -> GraphCompilation.forwardOutputs()[i]
```

Construct [gradient](../../../../glossary.md#term-gradient) bindings in `GraphCompilation.gradientResults()` order by pairing:

```text
GradientResultRole.target() -> GradientResultRole.gradient()
```

The two lists have distinct meanings:

- forward binding `tensorId` identifies the requested forward [Tensor](../../../../glossary.md#tensor);
- gradient binding `tensorId` identifies the [differentiation target](../../../../glossary.md#functional-gradient-request) whose result is published;
- a Tensor ID may occur once in each list because a forward result may also be a differentiation
  target;
- Tensor IDs are unique within each list;
- forward value IDs are unique and equal the graph-output prefix;
- gradient value IDs may repeat and may equal a forward value ID; and
- the final graph output boundary is exactly the forward value prefix followed by each gradient
  value not already present, in first gradient-role occurrence order.

Construction validates `graph`, `forwardOutputs`, and `gradientResults` in declaration order,
then list [elements](../../../../glossary.md#term-element) in encounter order. It rejects a non-output value, duplicate Tensor ID within
one role list, duplicate forward value, incorrect forward prefix, or graph boundary inconsistent
with stable first-[occurrence](../../../../glossary.md#term-occurrence) gradient de-duplication. Use indexed messages beginning with
`forwardOutputs[index]` or `gradientResults[index]`; boundary mismatch uses
`graph output boundary does not match publication roles`.

`PublicationPlan` is the owning graph context missing from standalone [model](../../../../glossary.md#term-model)
`PublicationBinding`. It neither changes that record nor adds policy, storage, [alias](../../../../glossary.md#aliasing-and-arena)/copy choice,
delivery, or Tensor gradient state.

### [Compile constant plan](../../../../glossary.md#compile-constant-plan)

Add public final `io.github.pho001.synaptik.compiler.CompileConstantPlan` with a package-private
constructor, exactly these public accessors, and one nested public [immutable](../../../../glossary.md#term-immutable) value:

```java
public List<ValueId> bindableInputs()
public List<ConstantSource> constantSources()

public record ConstantSource(ValueId valueId, ScalarValue value) {}
```

Derive both lists from the final `ValidatedGraph.constantGraph()` in exact [graph](../../../../glossary.md#graph)-input order:

- `bindableInputs` contains each input without a logical-splat fact;
- `constantSources` contains one `ConstantSource` for each input with a fact;
- each source retains the exact final [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) and exact immutable [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) reference; and
- together the two role lists classify every graph input exactly once.

Both lists are immutable membership snapshots. IDs are unique within and across the two lists.
The plan contains no [`Tensor`](../../../../glossary.md#tensor), [`Shape`](../../../../glossary.md#shape), [descriptor](../../../../glossary.md#term-descriptor) copy, dense [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), storage, [buffer](../../../../glossary.md#term-buffer), [backend](../../../../glossary.md#backend)
value, [materialization](../../../../glossary.md#materialization) instruction, or physical allocation.

The nested record validates `valueId` then `value`. The plan validates top-level lists in
declaration order, their [elements](../../../../glossary.md#term-element) in encounter order, duplicate IDs, and cross-list overlap.
`CompileArtifacts` performs final owning-graph coverage, order, descriptor-type, and
[gradient](../../../../glossary.md#term-gradient)-eligibility checks.

This type is the named cross-package transport with semantics equivalent to [Compiler](../../../../glossary.md#term-compiler) 0003B's
package-private sidecar. It does not make [constant](../../../../glossary.md#term-constant) ingress public and does not fold additional
operations.

### [Compile diagnostics](../../../../glossary.md#compile-diagnostics)

Add public final `io.github.pho001.synaptik.compiler.CompileDiagnostics` as an output-only
[immutable](../../../../glossary.md#term-immutable) type with a package-private constructor and this public accessor:

```java
public List<DeferredConstraintDiagnostic> deferredConstraints()
```

Add exactly one nested public record:

```java
public record DeferredConstraintDiagnostic(
        NodeId nodeId,
        String subject,
        String predicate) {}
```

The outer type snapshots and privately preserves the final ordered
`DeferredGraphConstraint` instances for later [compiler](../../../../glossary.md#term-compiler)-owned [binding](../../../../glossary.md#term-binding) [validation](../../../../glossary.md#term-validation), while the public
list exposes only immutable diagnostic projections in the same order. It may have a
package-private accessor for the exact internal constraint [snapshot](../../../../glossary.md#term-snapshot); no internal predicate type
appears in a public or protected [signature](../../../../glossary.md#term-signature).

Each diagnostic retains the exact [`NodeId`](../../../../glossary.md#nodeid), exact nonblank subject, and a nonblank deterministic
diagnostic rendering of the immutable predicate. The rendering is not [serialization](../../../../glossary.md#term-serialization), a public
predicate language, a [trace payload](../../../../glossary.md#trace-payload--tracepayload), or a concrete-[dimension](../../../../glossary.md#dimension) binding [API](../../../../glossary.md#term-application-programming-interface-api).

Successful [artifacts](../../../../glossary.md#term-artifact) contain only deferred-constraint diagnostics. Rejected [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) returns
no partial artifact and does not record caught failures as successful diagnostics. This task adds
no warning taxonomy, severity, [trace](../../../../glossary.md#trace) event, sink, logger, exception hierarchy, or serialization.

### [Immutable](../../../../glossary.md#term-immutable) [compile artifacts](../../../../glossary.md#compile-artifacts)

Add this exact public aggregate [shape](../../../../glossary.md#shape):

```java
public record CompileArtifacts(
        CompileMode mode,
        CompiledGraphModel graph,
        List<PlannedPartition> partitions,
        LogicalMemoryPlan memory,
        PublicationPlan publication,
        CompileConstantPlan constants,
        CompileDiagnostics diagnostics) {}
```

The additional `mode` and [`constants`](../../../../glossary.md#term-constant) components preserve already-implemented [compiler](../../../../glossary.md#term-compiler) state that
the architecture's illustrative five-component shape predates. They do not change architecture:
[compile mode](../../../../glossary.md#compile-mode--compilemode) remains declarative [graph](../../../../glossary.md#graph) [scope](../../../../glossary.md#term-scope), and Compiler 0003B explicitly requires exact
constant facts and bindable inputs to survive into compile artifacts.

The [canonical](../../../../glossary.md#term-canonical-form) constructor validates components in declaration order, snapshots [partition](../../../../glossary.md#partition) list
membership, and verifies:

- `graph` is the exact graph reference retained by [`publication`](../../../../glossary.md#term-publication);
- `publication` satisfies its complete forward/[gradient](../../../../glossary.md#term-gradient) boundary contract;
- `FORWARD_ONLY` has no gradient results and no `BACKWARD` [node](../../../../glossary.md#node);
- both backward-capable modes have at least one gradient result;
- each partition is non-null and the ordered list is the exact maximal graph-order partitioning;
- `memory` equals the logical-memory plan derived from that exact graph and partition list;
- bindable and constant roles classify every graph input exactly once in graph-input order;
- every constant source value type equals its graph input [descriptor](../../../../glossary.md#term-descriptor) [data type](../../../../glossary.md#data-type--datatype);
- no constant source fixes a gradient-eligible input;
- every deferred diagnostic names a node in the graph; and
- no live provider, [snapshot](../../../../glossary.md#term-snapshot), [device](../../../../glossary.md#term-device), or [mutable](../../../../glossary.md#term-mutable) request object is retained.

The aggregate retains exact immutable graph, plan, and [element](../../../../glossary.md#term-element) references except where list
membership is explicitly snapshotted. It does not retain `GraphCompilation` or duplicate graph
state. Graph phases remain in the exact [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel); forward and gradient roles remain
in `PublicationPlan`; constants and input roles remain in `CompileConstantPlan`; deferred
constraints remain in `CompileDiagnostics`.

### Narrow [Planning](../../../../glossary.md#term-planning) callable seam

Add one public final stateless [owner](../../../../glossary.md#term-owner)-selection collaboration in the existing capability package:

```text
io.github.pho001.synaptik.planning.capability.BackendOwnerPlanning
```

It has a private constructor and exactly this one public static [operation](../../../../glossary.md#operation):

```java
public static BackendId selectOwner(
        OperationCapabilityQuery query,
        BackendIntent intent,
        List<BackendCapabilityProvider> providers,
        List<BackendAvailabilitySnapshot> availabilitySnapshots,
        PartitionScoringConfig scoringConfig)
```

`selectOwner` validates the five top-level references in declaration order, then delegates exactly
once to package-private `BackendEligibility.evaluate(...)` and exactly once to package-private
`BackendOwnerSelection.select(...)`. The intermediate eligibility value does not escape.
Filtering, provider calls, exact requirement matching, optional preferred-class comparison,
reference retention, ordering, and failure messages remain those already tested by Planning
0002–0003. `BackendEligibility` and `BackendOwnerSelection` themselves remain package-private.

Widen the already-audited operations in their owning packages without changing their names,
[signatures](../../../../glossary.md#term-signature), constructors, implementations, or semantics:

```java
public final class MaximalSameOwnerPartitioning {
    public static List<PlannedPartition> partition(
            CompiledGraphModel graph,
            Map<NodeId, BackendId> ownershipByNodeId)
}

public final class LogicalMemoryPlanning {
    public static LogicalMemoryPlan plan(
            CompiledGraphModel graph,
            List<PlannedPartition> partitions)
}
```

Both classes retain private zero-[argument](../../../../glossary.md#term-argument) constructors and no fields. Their public static methods
remain direct stateless operations over their architecture-owned inputs and preserve the completed
[validation](../../../../glossary.md#term-validation), [identity](../../../../glossary.md#term-identity-meanings), reference, ordering, zero-[node](../../../../glossary.md#node), and failure contracts without a wrapper or
algorithm copy.

No Planning callable type contains a service instance, [registry](../../../../glossary.md#term-registry), discovery, [callback](../../../../glossary.md#term-callback), [plugin](../../../../glossary.md#term-plugin)
mechanism, [cache](../../../../glossary.md#term-cache), policy object, [graph](../../../../glossary.md#graph)-wide `plan(...)` workflow, owner-map assembly, diagnostics
translation, or [artifact](../../../../glossary.md#term-artifact) construction. None depends on [compiler](../../../../glossary.md#term-compiler) or imports a compiler type.
Keeping each operation in its existing cohesive package avoids a fourth `planning.compiler`
package whose [facade](../../../../glossary.md#term-facade) still could not legally invoke package-private operations in sibling Java
packages.

Compiler constructs every [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery), invokes
`BackendOwnerPlanning.selectOwner(...)` in final graph-node order, and inserts the exact selected
owner reference into one deterministic complete `LinkedHashMap<NodeId, BackendId>` or equivalently
ordered construction-local map. Map iteration order is not later used for [partition](../../../../glossary.md#partition) order.
Compiler then invokes `MaximalSameOwnerPartitioning.partition(...)` and
`LogicalMemoryPlanning.plan(...)` directly. Planning never receives [Tensors](../../../../glossary.md#tensor), [publication](../../../../glossary.md#term-publication)
[bindings](../../../../glossary.md#term-binding), [compile](../../../../glossary.md#compile) [constants](../../../../glossary.md#term-constant), [gradient](../../../../glossary.md#term-gradient) roles, or diagnostics.

### Capability-query orchestration

For each final `CompiledNode` in `CompiledGraphModel.nodes()` order:

1. resolve each input [descriptor](../../../../glossary.md#term-descriptor) in exact [node](../../../../glossary.md#node) input-position order;
2. resolve each output descriptor in exact node output-position order;
3. construct one [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) retaining the exact node [`Operation`](../../../../glossary.md#operation) and exact
   descriptor references;
4. call `BackendOwnerPlanning.selectOwner(...)` once; and
5. associate the exact [graph](../../../../glossary.md#graph) [`NodeId`](../../../../glossary.md#nodeid) reference with the exact returned [`BackendId`](../../../../glossary.md#backend-identity--backendid) reference.

The final validated graph, including generated `BACKWARD` nodes, is the sole [planning](../../../../glossary.md#term-planning) unit.
Capability providers see ordinary operation occurrences and descriptors; [graph phase](../../../../glossary.md#graph-phase) does not
change the query [shape](../../../../glossary.md#shape) or capability contract. No output role, hidden output, [constant](../../../../glossary.md#term-constant) source, or
[graph value](../../../../glossary.md#graph-value) is removed for planning.

A zero-node pass-through graph creates no query, makes no provider call, produces an empty
[partition](../../../../glossary.md#partition) list, and still receives one logical-memory requirement per graph value.

### Failure [ownership](../../../../glossary.md#term-ownership-meanings) and order

The complete overload follows this order:

1. validate all nine top-level [arguments](../../../../glossary.md#term-argument) in declaration order without allocating a [Tensor](../../../../glossary.md#tensor);
2. execute the existing [graph](../../../../glossary.md#graph)-stage [compile](../../../../glossary.md#compile) once, preserving all established failures;
3. build and validate [publication](../../../../glossary.md#term-publication) roles against the final graph;
4. derive [constant](../../../../glossary.md#term-constant) and deferred-diagnostic snapshots from the final validated graph;
5. visit [nodes](../../../../glossary.md#node) in stored order and perform one [planning](../../../../glossary.md#term-planning) selection per node;
6. derive partitions;
7. derive logical memory;
8. construct and cross-validate `CompileArtifacts`; and
9. return only the complete [immutable](../../../../glossary.md#term-immutable) result.

Existing graph-stage `NullPointerException` and `IllegalArgumentException` contracts remain
unchanged. Planning request-composition errors and provider-thrown [runtime](../../../../glossary.md#term-runtime) exceptions propagate
unchanged. The internal no-hard-eligible `IllegalStateException` is the only planning failure
translated by [Compiler](../../../../glossary.md#term-compiler): preserve it as the cause of a new `IllegalStateException` with exact
message [shape](../../../../glossary.md#shape):

```text
nodes[<index>] NodeId[value=<id>] <kind-class-name>.<kind.name()>:
no hard-eligible backend is available for ownership selection
```

Use one physical line in the actual message. This adds graph [occurrence](../../../../glossary.md#term-occurrence) context while preserving
Planning's terminal meaning and hard requirement. Selection stops immediately; later nodes and
providers are not queried.

[Artifact](../../../../glossary.md#term-artifact) constructor failures indicate invalid direct assembly or an implementation defect and
use `NullPointerException`/`IllegalArgumentException` with indexed component messages. This task
does not introduce a public compile exception taxonomy because no public compile entry exists.
[Trace](../../../../glossary.md#trace) 0004 remains the [owner](../../../../glossary.md#term-owner) of future typed compile payloads; Compiler remains the owner of
graph-wide failure context and successful [compile diagnostics](../../../../glossary.md#compile-diagnostics).

## Out of [scope](../../../../glossary.md#term-scope)

- changing `ARCHITECTURE.md`, an [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), or [dependency direction](../../../../glossary.md#java-platform-and-build-tools)
- a public [compile](../../../../glossary.md#compile) entry point, `CompiledGraph`, [engine](../../../../glossary.md#term-engine) [facade](../../../../glossary.md#term-facade), `CompileConfig`, request builder,
  service, registry, or lifecycle orchestration outside compiler<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- changing the existing five-[argument](../../../../glossary.md#term-argument) `GraphCompiler.compile(...)`, `GraphCompilation`,
  `AutogradPreflight.FirstOrderRequest`, objective/target/seed policy, rule matrix, graph capture,
  inference, validation, canonicalization, rewriting, folding, DCE, CSE, or pass order<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [target glossary](../../../../glossary.md#term-target); [seed glossary](../../../../glossary.md#term-seed); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- Compiler 0005A–0005E current-inventory first-order gradient completion, derivative-policy
  selection, or closure checkpoint; Compiler 0006 functional requests, explicit seeds,
  create-graph, derivative order, or higher derivatives<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [derivative glossary](../../../../glossary.md#term-derivative); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- public Planning eligibility result, owner map, capability matrix, graph-wide planning workflow,
  cost score, workload/profile classification, Config 0004, tuning candidate, or model autotuning<span data-glossary-navigation="true"> ([capability matrix glossary](../../../../glossary.md#review-tables-and-support-matrices))</span><span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [owner glossary](../../../../glossary.md#term-owner); [tuning candidate glossary](../../../../glossary.md#candidate-batch); [model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- live [backend](../../../../glossary.md#backend) instances beyond the explicitly supplied capability-provider collaboration
- retaining capability providers or availability snapshots in [compile artifacts](../../../../glossary.md#compile-artifacts)
- selecting or retaining a concrete device, route, kernel, executable, backend DAG, transfer,
  copy, alias, layout realization, physical bytes, lifetime, slot, buffer, allocation, workspace,
  schedule, residency, or run state<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [kernel glossary](../../../../glossary.md#kernel); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [alias glossary](../../../../glossary.md#aliasing-and-arena); [layout glossary](../../../../glossary.md#layout); [lifetime glossary](../../../../glossary.md#term-lifetime); [buffer glossary](../../../../glossary.md#term-buffer); [residency glossary](../../../../glossary.md#residency); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- [prepare](../../../../glossary.md#prepare), [runtime](../../../../glossary.md#term-runtime), backend, engine, training, [optimizer](../../../../glossary.md#term-optimizer), or [publication](../../../../glossary.md#term-publication)-delivery behavior
- physical logical-splat materialization, payload expansion, host-storage reads, new constant
  folding, public constant ingress, or runtime binding<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [host glossary](../../../../glossary.md#term-host); [constant folding glossary](../../../../glossary.md#term-constant-folding); [constant glossary](../../../../glossary.md#term-constant); [binding glossary](../../../../glossary.md#term-binding))</span>
- binding or solving dynamic Dimensions, exposing the internal predicate vocabulary, or deciding
  value-dependent graph validity<span data-glossary-navigation="true"> ([Dimensions glossary](../../../../glossary.md#dimension))</span>
- trace payloads, events, correlation allocation, emission, sinks, logging, serialization, or a
  rejection taxonomy<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- changing [model](../../../../glossary.md#term-model) [graph](../../../../glossary.md#graph), [Tensor](../../../../glossary.md#tensor), [operation](../../../../glossary.md#operation), [descriptor](../../../../glossary.md#term-descriptor), identifier, or `PublicationBinding`
  executable behavior
- backend [conformance](../../../../glossary.md#term-conformance) or end-to-end integration behavior
- creating a detailed [Compiler](../../../../glossary.md#term-compiler) 0005A–0005E, Compiler 0006, or any later task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Partition scoring](../../../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../../../../glossary.md#partition))</span>
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Tracing](../../../../architecture/tracing.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Backend guide style](../../../../developer-guide/documentation/backend-guide-style.md)
- [User guide style](../../../../developer-guide/documentation/user-guide-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- [Planning master plan](../../planning/master-plan.md)
- [Planning closure audit](../../planning/planning-contract-closure-audit.md)
- [Compiler 0003B](0003b-compile-time-constants-and-constant-folding.md)
- [Compiler 0004](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
- [Compiler 0004A](0004a-exact-composition-gradient-rule-extensions.md)
- [Compiler 0004B](0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md)
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Public API](../../../../api/public-api.md)
- [Runtime API](../../../../api/runtime-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `ARCHITECTURE.md` is authoritative. This task implements its existing final compile stages and
  immutable artifact boundary; it changes no ownership or dependency rule.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Compiler owns graph-wide sequencing, query construction, owner-map assembly, publication,
  constants transport, diagnostics, failure context, and <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [publication glossary](../../../../glossary.md#term-publication); [constants glossary](../../../../glossary.md#term-constant))</span>`CompileArtifacts`.
- Planning owns backend-neutral hard eligibility, baseline owner selection, maximal same-owner
  partitioning, and logical-memory derivation. It exposes only the three package-cohesive stateless
  operations required by Compiler and does not gain public graph-wide orchestration.
- [Model](../../../../glossary.md#term-model) owns [immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph) data and standalone `PublicationBinding`; the binding alone remains
  unable to prove owning-graph membership.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- [Compile](../../../../glossary.md#compile)-time [ownership](../../../../glossary.md#term-ownership-meanings) contains [`BackendId`](../../../../glossary.md#backend-identity--backendid), never a provider, live backend, selected device,
  route, or executable.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- The final graph is not rebuilt after publication or planning. Exact graph IDs, operation and
  descriptor references, graph phases, graph-output order, constant roles, and deferred
  constraint order remain stable.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- `CompileArtifacts` is an immutable recipe. It is distinct from package-private
  `GraphCompilation`, future [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), and [engine](../../../../glossary.md#term-engine) `CompiledGraph`.
- [Compiler](../../../../glossary.md#term-compiler) does not allocate physical [buffers](../../../../glossary.md#term-buffer) or construct prepared/[runtime](../../../../glossary.md#term-runtime) state.
- [Planning](../../../../glossary.md#term-planning) does not choose [kernels](../../../../glossary.md#kernel) or interpret [backend](../../../../glossary.md#backend) implementation vocabulary.
- [Trace](../../../../glossary.md#trace) remains a [DTO](../../../../glossary.md#data-transfer-object--dto)-only [leaf](../../../../glossary.md#term-leaf). This task creates no trace [dependency](../../../../glossary.md#term-dependency) reversal or [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) [schema](../../../../glossary.md#term-schema).
- Existing compiler dependencies on model, config, planning, backend-contract, and trace are
  already present. No Gradle or architecture-test change is authorized.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>
- If implementation requires another module edge, a public compiler facade, a graph-wide Planning
  workflow, a device/route decision, a changed graph-stage contract, or a path outside the revised
  maximum scope, stop and report the conflict.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [facade glossary](../../../../glossary.md#term-facade); [scope glossary](../../../../glossary.md#term-scope))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.compiler` — owns the package-private complete compile entry and public
  immutable output-only artifact types.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [immutable glossary](../../../../glossary.md#term-immutable); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- `io.github.pho001.synaptik.planning.capability` — retains public query/provider contracts,
  package-private eligibility/selection mechanics, and adds one public stateless collaboration
  that composes those internal mechanics without exposing their intermediate.
- `io.github.pho001.synaptik.planning.partition` — retains public partition recipes and widens the
  already-audited maximal same-owner generation operation for the concrete compiler consumer.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [owner glossary](../../../../glossary.md#term-owner); [operation glossary](../../../../glossary.md#operation))</span>
- `io.github.pho001.synaptik.planning.memory` — retains public logical-memory recipes and widens
  the already-audited derivation operation for the concrete compiler consumer.

No package is added.

Type placement:

- `io.github.pho001.synaptik.compiler.CompileArtifacts` — public immutable compile recipe for
  future prepare consumption.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>
- `io.github.pho001.synaptik.compiler.PublicationPlan` — public immutable output-only graph context
  for ordered forward and gradient result bindings.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [bindings glossary](../../../../glossary.md#term-binding))</span>
- `io.github.pho001.synaptik.compiler.CompileConstantPlan` — public immutable output-only logical
  constant and bindable-input transport.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- `io.github.pho001.synaptik.compiler.CompileDiagnostics` — public immutable output-only successful
  diagnostic bundle that privately preserves exact deferred constraints.
- `io.github.pho001.synaptik.compiler.GraphCompiler` — remains package-private and owns the direct
  artifact-producing overload plus graph-wide orchestration.
- `io.github.pho001.synaptik.planning.capability.BackendOwnerPlanning` — public stateless
  owner-selection collaboration that composes the package-private eligibility and selector.
- `io.github.pho001.synaptik.planning.partition.MaximalSameOwnerPartitioning` — existing stateless
  generator widened to public for direct Compiler invocation.
- `io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanning` — existing stateless derivation
  widened to public for direct Compiler invocation.

Tests mirror production packages. No root [planning](../../../../glossary.md#term-planning) [facade](../../../../glossary.md#term-facade), [ownership](../../../../glossary.md#term-ownership-meanings) package, [compiler](../../../../glossary.md#term-compiler) service,
generic utility package, or public [graph](../../../../glossary.md#graph)-stage type is added.

## Affected files

Expected production and [Javadoc](../../../../glossary.md#term-javadoc) paths:

- add
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/BackendOwnerPlanning.java`
- update
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/BackendCapabilityProvider.java`
  only for current compiler-consumer Javadoc<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>
- update
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/package-info.java`
  only for the new owner-selection collaboration/current integration boundary<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- update
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/partition/MaximalSameOwnerPartitioning.java`
  only to widen the existing final type and static operation plus finalize affected Javadoc<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- update
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/partition/package-info.java`
  only for the public current compiler-consumer operation
- update
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/memory/LogicalMemoryPlanning.java`
  only to widen the existing final type and static operation plus finalize affected Javadoc
- update
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/memory/package-info.java`
  only for the public current compiler-consumer operation
- update `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompiler.java`
- add `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java`
- add `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/PublicationPlan.java`
- add `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileConstantPlan.java`
- add `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileDiagnostics.java`
- update
  `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/PublicationBinding.java`
  only to replace later-plan wording with the current compiler-owned plan boundary
- update
  `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/CompileMode.java`
  only for the current package-private compiler consumer
- update
  `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/BackendIntent.java`
  only for the current package-private compiler/planning consumer
- update
  `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/PartitionScoringConfig.java`
  only for the current package-private compiler/planning consumer
- update
  `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/package-info.java`
  only for current consumption status

Expected test paths:

- add
  `modules/planning/src/test/java/io/github/pho001/synaptik/planning/capability/BackendOwnerPlanningTest.java`
- update
  `modules/planning/src/test/java/io/github/pho001/synaptik/planning/partition/MaximalSameOwnerPartitioningTest.java`
- update
  `modules/planning/src/test/java/io/github/pho001/synaptik/planning/memory/LogicalMemoryPlanningTest.java`
- update
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`
- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/CompileArtifactsTest.java`
- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/PublicationPlanTest.java`
- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/CompileConstantPlanTest.java`
- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/CompileDiagnosticsTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) paths:

- update [Partition scoring](../../../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../../../../glossary.md#partition))</span>
- update [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- update [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- update [Public API](../../../../api/public-api.md)
- update [Capability-provider guide](../../../../backend-guide/capability-provider.md)
- update [Compiling graphs guide](../../../../user-guide/compiling-graphs.md)<span data-glossary-navigation="true"> ([graphs glossary](../../../../glossary.md#graph))</span>
- update [Glossary](../../../../glossary.md)
- create and finalize this task
- update [Compiler master plan](../master-plan.md)
- update [Planning master plan](../../planning/master-plan.md)
- update [Config master plan](../../config/master-plan.md)
- update [Roadmap](../../../roadmap.md)

Review without modification unless a documented conflict requires stopping: `ARCHITECTURE.md`,
other focused architecture documents, ADRs, the historical Planning closure audit and existing
completed task specifications, [model](../../../../glossary.md#term-model)/config/planning behavior and tests outside the paths above,
[trace](../../../../glossary.md#trace) source and master plan, [runtime](../../../../glossary.md#term-runtime), [prepare](../../../../glossary.md#prepare), [engine](../../../../glossary.md#term-engine), training, concrete [backends](../../../../glossary.md#backend), [Gradle](../../../../glossary.md#term-gradle)/build
files, architecture tests, backend-[conformance](../../../../glossary.md#term-conformance) tests, integration tests, Runtime [API](../../../../glossary.md#term-application-programming-interface-api), Training
API, memory-planning design note, backend-selection guide, and all other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the exact 37 paths listed under
[Affected files](#affected-files): 17 production/[Javadoc](../../../../glossary.md#term-javadoc) paths, eight test paths, and 12
documentation/[planning](../../../../glossary.md#term-planning) paths.

The larger-than-normal count is justified by one indivisible cross-[module](../../../../glossary.md#term-module) [handoff](../../../../glossary.md#term-handoff):
`GraphCompilation` source roles and output roles must be converted once from final [graph](../../../../glossary.md#graph) IDs,
every final [node](../../../../glossary.md#node) must receive one [owner](../../../../glossary.md#term-owner) before [partition](../../../../glossary.md#partition) and logical-memory derivation, and each
audited Planning [operation](../../../../glossary.md#operation) must be callable from [Compiler](../../../../glossary.md#term-compiler) through Java visibility that matches its
owning package. The new owner-selection collaboration, the two widened operations, their existing
surface-locking tests and package Javadocs, and the resulting public [artifact](../../../../glossary.md#term-artifact) contracts must agree
atomically. Executable behavior changes only in Planning and Compiler; [model](../../../../glossary.md#term-model) and config Java
changes are Javadoc-only.

No [Gradle](../../../../glossary.md#term-gradle), [dependency](../../../../glossary.md#term-dependency), architecture, [trace](../../../../glossary.md#trace), [runtime](../../../../glossary.md#term-runtime), [prepare](../../../../glossary.md#prepare), [backend](../../../../glossary.md#backend), [engine](../../../../glossary.md#term-engine), [conformance](../../../../glossary.md#term-conformance), or
integration path is authorized. If a thirty-eighth path, another public type, another test class,
or another documentation page is required, stop and propose a task revision or follow-up instead
of expanding scope silently.

## Acceptance criteria

### Public and package-private surface

- The existing five-[argument](../../../../glossary.md#term-argument) `GraphCompiler.compile(...)` and package-private
  `GraphCompilation` [shape](../../../../glossary.md#shape) remain unchanged.
- The only new complete entry is the exact package-private nine-argument overload.
- No public compiler entry, compile request aggregate, engine facade, planning workflow, provider
  registry, or service object is added.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [compile glossary](../../../../glossary.md#compile); [engine glossary](../../../../glossary.md#term-engine); [facade glossary](../../../../glossary.md#term-facade); [planning glossary](../../../../glossary.md#term-planning); [registry glossary](../../../../glossary.md#term-registry))</span>
- `CompileArtifacts` has exactly the seven public record components specified above.
- `PublicationPlan`, `CompileConstantPlan`, and `CompileDiagnostics` are public final output-only
  immutable types with package-private construction and only the specified public accessors and
  nested public records.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- `BackendOwnerPlanning` is public, final, stateless, has a private constructor, and exposes
  exactly the specified public static `selectOwner(...)` [operation](../../../../glossary.md#operation).
- `BackendEligibility` and `BackendOwnerSelection`, including their operations and intermediate
  result, remain package-private.
- `MaximalSameOwnerPartitioning` and `LogicalMemoryPlanning` are public final stateless types with
  private constructors and exactly their existing public static operations; no overload or state
  is added.

### [Publication](../../../../glossary.md#term-publication) and output roles

- Forward [bindings](../../../../glossary.md#term-binding) pair exact requested [Tensor](../../../../glossary.md#tensor) IDs with final forward values in request order.
- [Gradient](../../../../glossary.md#term-gradient) bindings pair exact [target](../../../../glossary.md#term-target) Tensor IDs with final gradient values in target order.
- Forward and gradient roles remain separate; no Tensor gradient field or result Tensor is added.
- Same Tensor ID across the two lists is valid; duplicate Tensor IDs within one list fail.
- Gradient values may be shared by multiple targets or equal a forward value without an identity
  node or duplicated graph output.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [node glossary](../../../../glossary.md#node); [graph glossary](../../../../glossary.md#graph))</span>
- `PublicationPlan` validates every binding against its exact final graph and enforces the exact
  stable graph-output boundary.

### [Constants](../../../../glossary.md#term-constant), diagnostics, and immutability

- Every final [graph](../../../../glossary.md#graph) input is classified exactly once as bindable or a constant source.
- Constant sources retain exact [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) bits and types without dense expansion, storage
  reads, or physical state.
- Compile artifacts preserve the exact final graph reference, node phases, graph-output order,
  operations, descriptors, partition elements, logical requirements, output roles, constant
  roles, and deferred-constraint order.<span data-glossary-navigation="true"> ([Compile artifacts glossary](../../../../glossary.md#compile-artifacts); [node glossary](../../../../glossary.md#node); [descriptors glossary](../../../../glossary.md#term-descriptor); [partition glossary](../../../../glossary.md#partition))</span>
- Public deferred diagnostics contain exact node IDs and deterministic nonblank subject/predicate
  text; the internal predicate objects remain package-private and privately preserved.
- All exposed collections are [immutable](../../../../glossary.md#term-immutable) snapshots with documented [element](../../../../glossary.md#term-element)-reference retention.
- No provider, snapshot, device, route, kernel, physical memory, prepare/runtime state, or mutable
  request object is retained.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [device glossary](../../../../glossary.md#term-device); [kernel glossary](../../../../glossary.md#kernel); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [mutable glossary](../../../../glossary.md#term-mutable))</span>

### [Planning](../../../../glossary.md#term-planning) orchestration

- [Compiler](../../../../glossary.md#term-compiler) constructs one exact [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) per final [graph](../../../../glossary.md#graph) [node](../../../../glossary.md#node) in stored order.
- Query input and output descriptors follow exact node positions and retain graph descriptor
  references.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Each node invokes eligibility and baseline selection exactly once; the selected exact
  [`BackendId`](../../../../glossary.md#backend-identity--backendid) is associated with the exact graph [`NodeId`](../../../../glossary.md#nodeid).
- Compiler, not Planning, assembles the complete [owner](../../../../glossary.md#term-owner) map and controls graph-wide order.
- `BackendOwnerPlanning.selectOwner(...)` delegates exactly once to each of the two existing
  package-private capability operations without exposing the intermediate eligibility result.
- Compiler invokes the widened existing partition and logical-memory operations directly; no
  fourth-package facade, reflection, wrapper delegation, or algorithm duplication is used.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- [Partition](../../../../glossary.md#partition) and logical-memory results equal the completed Planning 0004–0005 contracts.
- A zero-node pass-through graph makes no provider call, has empty partitions, and receives
  graph-value-order logical requirements.

### [Validation](../../../../glossary.md#term-validation), failures, and boundaries

- Top-level validation, graph-stage compilation, publication, constant/diagnostic derivation,
  per-node planning, partitioning, memory planning, and artifact construction occur in the exact
  specified order.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [publication glossary](../../../../glossary.md#term-publication); [constant glossary](../../../../glossary.md#term-constant); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Existing [graph](../../../../glossary.md#graph)-stage and [Planning](../../../../glossary.md#term-planning) composition/provider failures retain their types and messages.
- A no-hard-eligible [node](../../../../glossary.md#node) receives the exact [compiler](../../../../glossary.md#term-compiler)-owned contextual failure message and cause.
- No partial artifact is returned after any failure, and later nodes are not queried after the
  first planning failure.
- `FORWARD_ONLY`, `FORWARD_AND_BACKWARD`, and `TRAINING_STEP` mode/phase/result invariants are
  enforced without adding optimizer work.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- No architecture, dependency, Gradle, trace, runtime, prepare, backend, engine, conformance, or
  integration behavior changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [trace glossary](../../../../glossary.md#trace); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Compiler 0005A–0005E and Compiler 0006 remain Draft and have no detailed specifications.

### Documentation and completion

- Production Javadocs fully document ownership, visibility, nullability, validation/failure
  order, exact references, immutability, output roles, constants, diagnostics, and exclusions.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [nullability glossary](../../../../glossary.md#term-nullability); [validation glossary](../../../../glossary.md#term-validation); [constants glossary](../../../../glossary.md#term-constant))</span>
- Compile, Tensor, Public API, provider, compiling-graphs, partition-scoring, glossary, and
  planning documents distinguish current internal compilation/public artifacts from the still
  absent public compile facade, preparation, execution, trace payloads, and higher derivatives.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graphs glossary](../../../../glossary.md#graph); [partition glossary](../../../../glossary.md#partition); [planning glossary](../../../../glossary.md#term-planning); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [artifacts glossary](../../../../glossary.md#term-artifact); [facade glossary](../../../../glossary.md#term-facade); [trace glossary](../../../../glossary.md#trace); [derivatives glossary](../../../../glossary.md#term-derivative))</span>
- The glossary defines or updates publication plan, compile artifacts, compile constant plan, and
  compile diagnostics without duplicating architecture rules.<span data-glossary-navigation="true"> ([publication plan glossary](../../../../glossary.md#publication-plan); [compile artifacts glossary](../../../../glossary.md#compile-artifacts); [compile constant plan glossary](../../../../glossary.md#compile-constant-plan); [compile diagnostics glossary](../../../../glossary.md#compile-diagnostics))</span>
- Documentation records reasoned no-change conclusions for Runtime and Training APIs, other
  architecture pages/ADRs, trace, prepare, runtime, engine, training, backends, Gradle,
  architecture/conformance/integration tests, and unrelated model/config/planning contracts.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [model glossary](../../../../glossary.md#term-model))</span>
- A separate clean-context documentation-focused agent finalizes affected Javadocs,
  documentation, glossary impact, examples, links, planning evidence, and no-change conclusions
  in the same overall change without repeating successful Java tests unless executable behavior
  changes or a concrete risk is recorded.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused tests:

```bash
./gradlew :modules:planning:test \
  --tests io.github.pho001.synaptik.planning.capability.BackendOwnerPlanningTest \
  --tests io.github.pho001.synaptik.planning.partition.MaximalSameOwnerPartitioningTest \
  --tests io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanningTest
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.GraphCompilerTest \
  --tests io.github.pho001.synaptik.compiler.CompileArtifactsTest \
  --tests io.github.pho001.synaptik.compiler.PublicationPlanTest \
  --tests io.github.pho001.synaptik.compiler.CompileConstantPlanTest \
  --tests io.github.pho001.synaptik.compiler.CompileDiagnosticsTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) one final affected-[module](../../../../glossary.md#term-module) command:

```bash
./gradlew :modules:planning:test :modules:compiler:test
```

This task changes public cross-module contracts and executable behavior in two modules. Run one
final repository/architecture [checkpoint](../../../../glossary.md#term-checkpoint):

```bash
./gradlew test :testing:architecture-tests:test
```

The documentation-focused pass reuses those successful Java results unless it changes executable
behavior. After final Javadocs and documentation:

```bash
./gradlew :modules:model:javadoc :modules:config:javadoc \
  :modules:planning:javadoc :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

If `/tmp/validate_synaptik_markdown.py` is absent, the documentation agent must create an
equivalent temporary validator outside the repository and record the exact command. It must check
all repository-local Markdown [targets](../../../../glossary.md#term-target) and heading anchors, balanced fences, final newlines, and
trailing whitespace.

Required manual/source checks:

- exact 37-path maximum and no unlisted path;
- public/package-private declaration and generated-[Javadoc](../../../../glossary.md#term-javadoc) indexes;
- exact `CompileArtifacts` components; exact one-method `BackendOwnerPlanning`; exact public
  `MaximalSameOwnerPartitioning.partition(...)` and `LogicalMemoryPlanning.plan(...)`; and
  package-private `BackendEligibility`/`BackendOwnerSelection`;
- unchanged five-[argument](../../../../glossary.md#term-argument) `GraphCompiler` and `GraphCompilation` [shapes](../../../../glossary.md#shape);
- no detailed [Compiler](../../../../glossary.md#term-compiler) 0005A–0005E or Compiler 0006 specification;
- synchronized [Ready](../../../../glossary.md#term-ready)/Complete/Draft task statuses;
- no provider/[snapshot](../../../../glossary.md#term-snapshot) retention or forbidden route/[device](../../../../glossary.md#term-device)/[runtime](../../../../glossary.md#term-runtime) vocabulary in [artifact](../../../../glossary.md#term-artifact) fields;
- no imports from runtime, prepare, engine, concrete backends, or tools in Planning/Compiler
  production changes; and<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend); [Planning glossary](../../../../glossary.md#term-planning))</span>
- final newlines, no trailing whitespace, and `git diff --check`.

## Dependencies

- [Compiler](../../../../glossary.md#term-compiler) 0001–0004B — Complete.
- [Model](../../../../glossary.md#term-model) 0025 and the model [graph](../../../../glossary.md#graph)/[publication](../../../../glossary.md#term-publication) contracts — Complete.
- Config 0001–0003 — Complete.
- [Planning](../../../../glossary.md#term-planning) 0001–0006 and its `CLOSED` audit — Complete.
- [Backend](../../../../glossary.md#backend)-contract 0001–0004 — Complete.
- Trace 0001–0002 — Complete; later payload work is not required because this task emits no
  trace event.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Existing Gradle dependencies from Compiler to model, config, planning, backend-contract, and
  trace — already present and unchanged.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>

Config 0004+, [Trace](../../../../glossary.md#trace) 0003+, [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), backends, [Engine](../../../../glossary.md#term-engine), and training are not prerequisites
for this bounded [compile](../../../../glossary.md#compile) [artifact](../../../../glossary.md#term-artifact). Their missing behavior must not be implemented here.

## Follow-up tasks

- Compiler 0005A–0005D — complete the current model inventory's first-order differentiation
  coverage in dependency order across elementwise/activation; reduction/scan/softmax/statistics/
  normalization; layout/window/indexing/scatter/ordering/stochastic; and structured attention/
  convolution/pooling/loss families. Each remains a concise Draft master-plan row with no
  detailed task specification.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [model glossary](../../../../glossary.md#term-model); [dependency glossary](../../../../glossary.md#term-dependency); [elementwise glossary](../../../../glossary.md#term-elementwise); [activation glossary](../../../../glossary.md#term-activation); [scan glossary](../../../../glossary.md#term-cumulative-scan); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [layout glossary](../../../../glossary.md#layout); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [loss glossary](../../../../glossary.md#loss))</span>
- Compiler 0005E — audit complete first-order role coverage, fail-closed inventory, auxiliary
  output handling, dynamic/binding-dependent rules, explicit derivative-policy decisions, and
  transitive formula-operation differentiability, then run the closure checkpoint. It remains
  Draft with no detailed task specification.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [binding glossary](../../../../glossary.md#term-binding); [derivative glossary](../../../../glossary.md#term-derivative); [operation glossary](../../../../glossary.md#operation); [run glossary](../../../../glossary.md#term-run); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Compiler 0006 — define explicit functional gradient requests and higher-order differentiation
  only after Compiler 0005E closes the first-order milestone and this artifact/publication
  boundary is stable. It remains Draft with no detailed task specification.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [artifact glossary](../../../../glossary.md#term-artifact); [publication glossary](../../../../glossary.md#term-publication))</span>
- Prepare planning may begin only through a later separate roadmap reassessment. It does not
  reorder Compiler 0005A–0005E or remove 0005E as Compiler 0006's dependency gate.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [planning glossary](../../../../glossary.md#term-planning))</span>
- Trace 0004 may later define typed compile payloads from stable producer facts. This task does
  not create that specification.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [compile glossary](../../../../glossary.md#compile); [producer glossary](../../../../glossary.md#term-producer))</span>
- Config 0005 may later aggregate existing standalone compile inputs. It is not required for the
  package-private direct compile overload.
- Future concrete binding validation may use the exact privately retained deferred constraints.
  This task deliberately exposes diagnostics, not a public predicate or binding language.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

Do not create any follow-up detailed specification during this task.

## Architecture impact

Expected impact: None.

The architecture already assigns [publication](../../../../glossary.md#term-publication) [binding](../../../../glossary.md#term-binding), [planning](../../../../glossary.md#term-planning) orchestration, logical-memory
orchestration, diagnostics, and [immutable](../../../../glossary.md#term-immutable) `CompileArtifacts` to [Compiler](../../../../glossary.md#term-compiler). It already assigns the
delegated [backend](../../../../glossary.md#backend)-neutral operations to Planning and permits Compiler to depend on Planning. The
new collaboration is the concrete narrow surface anticipated by the Planning closure audit.

If implementation requires Planning to own the [graph](../../../../glossary.md#graph)-wide loop, a new [dependency](../../../../glossary.md#term-dependency), a live backend
or [device](../../../../glossary.md#term-device) in [artifacts](../../../../glossary.md#term-artifact), public [compile](../../../../glossary.md#compile) orchestration, prepared/[runtime](../../../../glossary.md#term-runtime) state, or a changed
architecture rule, stop and report the exact conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the compiler and planning master
plans, the planning closure audit, completed Compiler 0003B and 0004–0004B tasks, and
docs/planning/modules/compiler/tasks/0005-publication-planning-orchestration-and-compile-artifacts.md.
Read the directly affected source, tests, APIs, guides, glossary, config/backend-contract/trace
status, and Gradle files needed to verify the task.

Implement Compiler 0005 exactly as specified within its 37 authorized paths. Preserve the existing
five-argument GraphCompiler and GraphCompilation contracts. Keep Compiler as graph-wide
orchestrator. Add only `BackendOwnerPlanning.selectOwner(...)` in `planning.capability`, retain
`BackendEligibility` and `BackendOwnerSelection` as package-private, and widen only the existing
`MaximalSameOwnerPartitioning.partition(...)` and `LogicalMemoryPlanning.plan(...)` operations for
direct Compiler use.
Do not implement Compiler 0005A–0005E, Compiler 0006, a public compile facade or request
aggregate, cost scoring, trace emission, physical memory, prepare/runtime/backend/engine
behavior, or any out-of-scope work. Stop on an architecture, package, or scope conflict.

After executable implementation and final affected-module/checkpoint evidence, hand the actual
diff and evidence to a separate documentation-focused agent or thread with clean context. That
targeted pass must follow docs/developer-guide/documentation-rules.md, apply the General,
API/Javadoc, Backend Guide, User Guide, Example, and Planning profiles as relevant, independently
finalize affected Javadocs, APIs, guides, glossary, planning status/evidence, links, examples, and
no-change conclusions in the same overall change, and not repeat successful Java tests unless it
changes executable behavior or records a concrete reason.

Update this task with local decisions, exact validation evidence, implementation notes,
completion summary, and final status. Do not mark it Complete before the documentation pass and
every acceptance criterion finish.
```

## Local decisions

- Planning visibility correction, 2026-07-27: the first clean Compiler 0005 implementation
  context stopped before editing because the planned <span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler))</span>`planning.compiler.CompilerPlanning` type
  could not invoke package-private top-level operations in sibling `planning.capability`,
  `planning.partition`, and `planning.memory` Java packages. The stopped context made no source,
  test, or documentation implementation change.
- Selected seam: add one public `planning.capability.BackendOwnerPlanning.selectOwner(...)`
  collaboration that alone can compose package-private `BackendEligibility` and
  `BackendOwnerSelection`; widen the already-audited `MaximalSameOwnerPartitioning.partition(...)`
  and `LogicalMemoryPlanning.plan(...)` operations in their owning packages for direct Compiler
  invocation.
- Rationale: a fourth-package facade cannot gain Java package access to three sibling packages.
  Making the two completed stage operations directly callable avoids redundant facades while
  preserving Compiler's owner-map assembly and graph-wide sequence. Exposing eligibility or its
  selector, moving algorithms, reflection, and duplicated logic were rejected.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- Scope correction: the exact ceiling increases from 32 to 37 paths to include the two widened
  production types, both affected package Javadocs, and their two existing visibility-locking
  tests. Architecture, semantic behavior, artifact design, failure ordering, and task status do
  not change.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- The existing five-[argument](../../../../glossary.md#term-argument) [graph](../../../../glossary.md#graph)-stage `GraphCompiler.compile(...)` and package-private
  `GraphCompilation` remain unchanged. The complete artifact lifecycle is a separate
  package-private nine-argument overload so the established capture, autograd, validation, and
  exact-optimization path still executes exactly once.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [validation glossary](../../../../glossary.md#term-validation))</span>
- The artifact boundary retains only the final graph, compile mode, logical partitions and
  memory, ordered publication roles, logical constants/bindable inputs, and deferred diagnostic
  projections. Compiler constructs one operation-capability query per final graph node and
  retains only the selected <span data-glossary-navigation="true"> ([compile mode glossary](../../../../glossary.md#compile-mode--compilemode); [publication glossary](../../../../glossary.md#term-publication); [constants glossary](../../../../glossary.md#term-constant); [operation glossary](../../../../glossary.md#operation); [node glossary](../../../../glossary.md#node))</span>[`BackendId`](../../../../glossary.md#backend-identity--backendid) in the complete [owner](../../../../glossary.md#term-owner) map.

## Known limitations

- The complete [compile](../../../../glossary.md#compile) entry remains package-private and uses direct [parameters](../../../../glossary.md#term-parameter) because
  `CompileConfig` and the [engine](../../../../glossary.md#term-engine) [facade](../../../../glossary.md#term-facade) remain Draft.
- The current ownership comparison is the completed preferred-class/provider-order cost-free
  baseline. No numeric or profile-driven scoring is present.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Capability remains backend-level even when availability proves an exact device or device class;
  artifacts retain only the selected backend identity.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [device glossary](../../../../glossary.md#term-device); [device class glossary](../../../../glossary.md#device-class--deviceclass); [artifacts glossary](../../../../glossary.md#term-artifact); [backend identity glossary](../../../../glossary.md#backend-identity--backendid))</span>
- `PublicationPlan` represents ordered forward and gradient result roles, not publication policy,
  delivery targets, aliases, copies, or Tensor gradient storage.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [targets glossary](../../../../glossary.md#term-target); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [Tensor glossary](../../../../glossary.md#tensor))</span>
- `CompileConstantPlan` supports only exact logical splats established by [Compiler](../../../../glossary.md#term-compiler) 0003B.
- Public diagnostics expose deterministic deferred-constraint descriptions, not the internal
  predicate vocabulary, binding, trace, serialization, or failure taxonomy.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [trace glossary](../../../../glossary.md#trace); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- A valid zero-node graph does not inspect provider or snapshot list elements because it asks no
  capability question.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [graph glossary](../../../../glossary.md#graph); [snapshot glossary](../../../../glossary.md#term-snapshot); [elements glossary](../../../../glossary.md#term-element))</span>
- No public caller can currently invoke compilation or supply an explicit publication/constant
  request through a released facade.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [constant glossary](../../../../glossary.md#term-constant))</span>
- The implemented first-order matrix remains the closed Compiler 0004–0004B subset. Draft tasks
  0005A–0005E, not this artifact task, own complete current-inventory formula coverage, explicit
  derivative-boundary decisions, auxiliary-output adoption, dynamic/binding-dependent rules, and
  the closure checkpoint before Compiler 0006.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [derivative glossary](../../../../glossary.md#term-derivative); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation evidence:

- The focused Planning command passed the three selected owner-planning, partitioning, and
  logical-memory suites.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [owner glossary](../../../../glossary.md#term-owner))</span>
- The focused Compiler command passed the five selected graph-compiler and compile-artifact
  suites.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
- `./gradlew :modules:planning:test :modules:compiler:test` passed: Planning ran 9 suites with
  68 tests and Compiler ran 22 suites with 150 tests, with no skipped tests, failures, or errors.
- `./gradlew test :testing:architecture-tests:test` passed: the repository ran 172 suites with
  1,294 tests, including 3 architecture suites with 3 tests, with no skipped tests, failures, or
  errors.
- No executable Java changed after those successful commands. The independent documentation
  context `/root/implement_compiler_0005/compiler_0005_docs` therefore reused the Java evidence as
  required by the planning guide.

Documentation and final-gate evidence:

- The independent pass applied the General, API/Javadoc, Backend Guide, User Guide, Example, and
  Planning profiles. It finalized all affected production Javadocs, Compile/Tensor/Public APIs,
  partition-scoring architecture explanation, capability-provider backend guide, compiling-graphs
  user guide, glossary, task, three module master plans, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Backend glossary](../../../../glossary.md#backend); [Tensor glossary](../../../../glossary.md#tensor); [partition glossary](../../../../glossary.md#partition); [module glossary](../../../../glossary.md#term-module))</span>
- Runtime API and Training API need no change because this task adds no public run, prepared,
  optimizer, training-request, or execution contract. Architecture/ADR and architecture-test
  sources need no change because the existing dependency direction and compiler/planning ownership
  contract are unchanged. Trace, prepare, runtime, engine, training, backend, conformance,
  integration, and Gradle files need no change because no trace schema, lifecycle/backend behavior,
  end-to-end behavior, module boundary, dependency, or build configuration changed.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run); [optimizer glossary](../../../../glossary.md#term-optimizer); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Trace glossary](../../../../glossary.md#trace); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [schema glossary](../../../../glossary.md#term-schema); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- `./gradlew :modules:model:javadoc :modules:config:javadoc :modules:planning:javadoc
  :modules:compiler:javadoc` passed.
- `python3 /tmp/validate_synaptik_markdown.py` validated all 12 changed Markdown files and 677
  repository-local links, including targets, heading anchors, balanced fences, final newlines, and
  trailing whitespace.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- `git diff --check` passed with no output.
- The final inventory contains exactly 37 authorized paths: 17 production/Javadoc paths, eight
  tests, and 12 documentation/planning paths, with no other path.
- `javap -p` and generated [Javadoc](../../../../glossary.md#term-javadoc) indexes confirm the exact seven `CompileArtifacts` components,
  one public `BackendOwnerPlanning.selectOwner(...)` method, public partition and logical-memory
  operations, package-private eligibility and selector types, and the unchanged five-argument<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
  `GraphCompiler`/`GraphCompilation` [graph](../../../../glossary.md#graph)-stage [shapes](../../../../glossary.md#shape).
- Status, later-spec, production-import, and [artifact](../../../../glossary.md#term-artifact)-field scans passed: [Compiler](../../../../glossary.md#term-compiler) 0005 is
  `Complete`; Compiler 0005A–0005E and 0006 remain `Draft` without detailed specifications; no
  forbidden runtime/prepare/engine/backend/tools import or provider/snapshot/device/route/runtime
  artifact field is present.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [device glossary](../../../../glossary.md#term-device))</span>

## Implementation notes

- Added the narrow public Planning owner-selection collaboration while keeping eligibility and
  baseline selection internal, and widened only the existing partition and logical-memory
  stateless operations.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [owner glossary](../../../../glossary.md#term-owner); [partition glossary](../../../../glossary.md#partition))</span>
- Added the package-private complete compile overload, immutable publication/constant/diagnostic
  outputs, and exact seven-component <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [immutable glossary](../../../../glossary.md#term-immutable); [publication glossary](../../../../glossary.md#term-publication); [constant glossary](../../../../glossary.md#term-constant))</span>`CompileArtifacts` aggregate with cross-[artifact](../../../../glossary.md#term-artifact) [validation](../../../../glossary.md#term-validation).
- Added focused tests for orchestration, immutable artifact contracts, validation order,
  zero-node behavior, visibility, and unchanged graph-stage compilation.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [graph glossary](../../../../glossary.md#graph); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Finalized the authorized Javadocs and explanatory/planning documentation without changing
  executable behavior during the documentation pass.

## Completion summary

- Completed the exact authorized 37-path change: 17 production/Javadoc paths, eight test paths,
  and 12 documentation/planning paths.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>
- Completed focused and final affected-module tests plus the repository/architecture checkpoint
  with no skipped tests, failures, or errors.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Completed the independent documentation pass and synchronized [Compiler](../../../../glossary.md#term-compiler) 0005 to `Complete`;
  Compiler 0005A–0005E and 0006 remain `Draft` without detailed specifications.
- No unresolved issue or required follow-up remains for this task. The next task may be promoted
  only through the normal progressive-planning workflow.

Status: Complete
