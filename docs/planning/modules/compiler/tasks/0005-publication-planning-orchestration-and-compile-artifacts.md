# Task 0005: Publication, Planning Orchestration, and Compile Artifacts

## Status

Complete

## Goal

Complete the current package-private compile lifecycle after `GraphCompilation` by deriving
publication roles, orchestrating the completed backend-neutral planning operations, and returning
one immutable public `CompileArtifacts` recipe.

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

Compiler remains the graph-wide orchestrator. Planning continues to own the backend-neutral
eligibility, baseline owner-selection, partition-generation, and logical-memory semantics. This
task opens only the three package-cohesive stateless Planning operations required by that concrete
consumer: one owner-selection collaboration that hides the eligibility intermediate, plus the
already-audited maximal-partition and logical-memory operations. It does not add a public planning
workflow, registry, service, cost model, device selection, or orchestration policy.

The result is compile-time state only. It contains no live backend provider, availability
snapshot, device selection, route, kernel, physical buffer, byte count, lifetime, slot, transfer,
prepared schedule, executable, runtime residency, trace event, or mutable run state.

## Current gap

Compiler 0001–0004B currently stop at package-private `GraphCompilation`. That result already
contains the final validated graph, exact compile mode, forward graph-output prefix, ordered
target-to-gradient roles, explicit logical-splat sidecar, bindable-input classification, deferred
constraints, and per-node phases. It does not:

- associate forward and gradient result roles with standalone model `PublicationBinding` values;
- prove that requested publication values belong to the final graph output boundary;
- construct one capability query per final graph node;
- consume Planning's package-private eligibility, owner-selection, partitioning, or logical-memory
  operations across their sibling Java packages;
- assemble the complete node-to-owner map;
- transport logical constants into a cross-package compile artifact;
- expose deferred compile diagnostics without exposing the internal predicate implementation; or
- produce `CompileArtifacts`.

Planning 0001–0006 is complete with a `CLOSED` audit verdict. Its four evaluator/generator
operations are package-private because no external consumer previously existed. Compiler 0005 is
the concrete consumer that now justifies one narrow callable seam while preserving Planning's
existing semantics and Compiler's graph-wide orchestration ownership.

## Terminology

- **Forward publication** — an ordered `PublicationBinding` from the exact requested forward
  Tensor's `TensorId` to its final forward graph-output `ValueId`.
- **Gradient result publication** — an ordered `PublicationBinding` whose `TensorId` identifies
  the exact requested differentiation target and whose `ValueId` identifies that target's final
  gradient result. It does not add gradient state to `Tensor`.
- **Publication plan** — compiler-owned graph context plus separate ordered forward and gradient
  binding lists, validated against the final graph boundary.
- **Compile constant plan** — the cross-package immutable form of Compiler 0003B's exact logical
  splats and bindable-input classification. It is logical data, not physical materialization.
- **Compile diagnostics** — immutable successful-compile diagnostics derived from the final
  deferred constraints. The exact internal predicates remain available inside compiler without
  becoming a public binding or trace schema.
- **Compile artifacts** — the immutable non-executable aggregate handed to later prepare work.
- **Planning callable seam** — the three public stateless operations used by Compiler across the
  Java package boundary: owner selection in `planning.capability`, maximal partition generation
  in `planning.partition`, and logical-memory derivation in `planning.memory`. It is not a
  planner service, registry, facade, or graph-wide workflow.

## Scope

### Package-private compile-artifact entry

Keep the existing package-private five-argument `GraphCompiler.compile(...)` entry and
`GraphCompilation` result unchanged. Add one package-private overload with this exact shape:

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
`CompileConfig`, `CompiledGraph`, engine facade, or alternate artifact compiler.

Validate all nine top-level arguments in declaration order before graph construction or any
derivative Tensor allocation. `forwardOutputs` retains the existing element and identity
validation. Provider and snapshot list elements are validated by the Planning collaboration when
the first node is planned; a valid zero-node pass-through graph performs no capability query and
therefore does not inspect unused list elements. The caller collections are never retained or
mutated.

After top-level validation, invoke the existing five-argument graph-stage compilation exactly
once. Do not duplicate capture, autograd, inference, canonicalization, rewriting, folding, DCE,
CSE, or final validation.

### Publication plan

Add public final `io.github.pho001.synaptik.compiler.PublicationPlan` as an output-only immutable
type with a package-private constructor and exactly these public accessors:

```java
public CompiledGraphModel graph()
public List<PublicationBinding> forwardOutputs()
public List<PublicationBinding> gradientResults()
```

The plan retains the exact final graph reference. It snapshots list membership while retaining the
exact `PublicationBinding` references.

Construct forward bindings in original `forwardOutputs` order by pairing:

```text
forwardOutputs[i].id() -> GraphCompilation.forwardOutputs()[i]
```

Construct gradient bindings in `GraphCompilation.gradientResults()` order by pairing:

```text
GradientResultRole.target() -> GradientResultRole.gradient()
```

The two lists have distinct meanings:

- forward binding `tensorId` identifies the requested forward Tensor;
- gradient binding `tensorId` identifies the differentiation target whose result is published;
- a Tensor ID may occur once in each list because a forward result may also be a differentiation
  target;
- Tensor IDs are unique within each list;
- forward value IDs are unique and equal the graph-output prefix;
- gradient value IDs may repeat and may equal a forward value ID; and
- the final graph output boundary is exactly the forward value prefix followed by each gradient
  value not already present, in first gradient-role occurrence order.

Construction validates `graph`, `forwardOutputs`, and `gradientResults` in declaration order,
then list elements in encounter order. It rejects a non-output value, duplicate Tensor ID within
one role list, duplicate forward value, incorrect forward prefix, or graph boundary inconsistent
with stable first-occurrence gradient de-duplication. Use indexed messages beginning with
`forwardOutputs[index]` or `gradientResults[index]`; boundary mismatch uses
`graph output boundary does not match publication roles`.

`PublicationPlan` is the owning graph context missing from standalone model
`PublicationBinding`. It neither changes that record nor adds policy, storage, alias/copy choice,
delivery, or Tensor gradient state.

### Compile constant plan

Add public final `io.github.pho001.synaptik.compiler.CompileConstantPlan` with a package-private
constructor, exactly these public accessors, and one nested public immutable value:

```java
public List<ValueId> bindableInputs()
public List<ConstantSource> constantSources()

public record ConstantSource(ValueId valueId, ScalarValue value) {}
```

Derive both lists from the final `ValidatedGraph.constantGraph()` in exact graph-input order:

- `bindableInputs` contains each input without a logical-splat fact;
- `constantSources` contains one `ConstantSource` for each input with a fact;
- each source retains the exact final `ValueId` and exact immutable `ScalarValue` reference; and
- together the two role lists classify every graph input exactly once.

Both lists are immutable membership snapshots. IDs are unique within and across the two lists.
The plan contains no `Tensor`, `Shape`, descriptor copy, dense payload, storage, buffer, backend
value, materialization instruction, or physical allocation.

The nested record validates `valueId` then `value`. The plan validates top-level lists in
declaration order, their elements in encounter order, duplicate IDs, and cross-list overlap.
`CompileArtifacts` performs final owning-graph coverage, order, descriptor-type, and
gradient-eligibility checks.

This type is the named cross-package transport with semantics equivalent to Compiler 0003B's
package-private sidecar. It does not make constant ingress public and does not fold additional
operations.

### Compile diagnostics

Add public final `io.github.pho001.synaptik.compiler.CompileDiagnostics` as an output-only
immutable type with a package-private constructor and this public accessor:

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
`DeferredGraphConstraint` instances for later compiler-owned binding validation, while the public
list exposes only immutable diagnostic projections in the same order. It may have a
package-private accessor for the exact internal constraint snapshot; no internal predicate type
appears in a public or protected signature.

Each diagnostic retains the exact `NodeId`, exact nonblank subject, and a nonblank deterministic
diagnostic rendering of the immutable predicate. The rendering is not serialization, a public
predicate language, a trace payload, or a concrete-dimension binding API.

Successful artifacts contain only deferred-constraint diagnostics. Rejected compilation returns
no partial artifact and does not record caught failures as successful diagnostics. This task adds
no warning taxonomy, severity, trace event, sink, logger, exception hierarchy, or serialization.

### Immutable compile artifacts

Add this exact public aggregate shape:

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

The additional `mode` and `constants` components preserve already-implemented compiler state that
the architecture's illustrative five-component shape predates. They do not change architecture:
compile mode remains declarative graph scope, and Compiler 0003B explicitly requires exact
constant facts and bindable inputs to survive into compile artifacts.

The canonical constructor validates components in declaration order, snapshots partition list
membership, and verifies:

- `graph` is the exact graph reference retained by `publication`;
- `publication` satisfies its complete forward/gradient boundary contract;
- `FORWARD_ONLY` has no gradient results and no `BACKWARD` node;
- both backward-capable modes have at least one gradient result;
- each partition is non-null and the ordered list is the exact maximal graph-order partitioning;
- `memory` equals the logical-memory plan derived from that exact graph and partition list;
- bindable and constant roles classify every graph input exactly once in graph-input order;
- every constant source value type equals its graph input descriptor data type;
- no constant source fixes a gradient-eligible input;
- every deferred diagnostic names a node in the graph; and
- no live provider, snapshot, device, or mutable request object is retained.

The aggregate retains exact immutable graph, plan, and element references except where list
membership is explicitly snapshotted. It does not retain `GraphCompilation` or duplicate graph
state. Graph phases remain in the exact `CompiledGraphModel`; forward and gradient roles remain
in `PublicationPlan`; constants and input roles remain in `CompileConstantPlan`; deferred
constraints remain in `CompileDiagnostics`.

### Narrow Planning callable seam

Add one public final stateless owner-selection collaboration in the existing capability package:

```text
io.github.pho001.synaptik.planning.capability.BackendOwnerPlanning
```

It has a private constructor and exactly this one public static operation:

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
signatures, constructors, implementations, or semantics:

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

Both classes retain private zero-argument constructors and no fields. Their public static methods
remain direct stateless operations over their architecture-owned inputs and preserve the completed
validation, identity, reference, ordering, zero-node, and failure contracts without a wrapper or
algorithm copy.

No Planning callable type contains a service instance, registry, discovery, callback, plugin
mechanism, cache, policy object, graph-wide `plan(...)` workflow, owner-map assembly, diagnostics
translation, or artifact construction. None depends on compiler or imports a compiler type.
Keeping each operation in its existing cohesive package avoids a fourth `planning.compiler`
package whose facade still could not legally invoke package-private operations in sibling Java
packages.

Compiler constructs every `OperationCapabilityQuery`, invokes
`BackendOwnerPlanning.selectOwner(...)` in final graph-node order, and inserts the exact selected
owner reference into one deterministic complete `LinkedHashMap<NodeId, BackendId>` or equivalently
ordered construction-local map. Map iteration order is not later used for partition order.
Compiler then invokes `MaximalSameOwnerPartitioning.partition(...)` and
`LogicalMemoryPlanning.plan(...)` directly. Planning never receives Tensors, publication
bindings, compile constants, gradient roles, or diagnostics.

### Capability-query orchestration

For each final `CompiledNode` in `CompiledGraphModel.nodes()` order:

1. resolve each input descriptor in exact node input-position order;
2. resolve each output descriptor in exact node output-position order;
3. construct one `OperationCapabilityQuery` retaining the exact node `Operation` and exact
   descriptor references;
4. call `BackendOwnerPlanning.selectOwner(...)` once; and
5. associate the exact graph `NodeId` reference with the exact returned `BackendId` reference.

The final validated graph, including generated `BACKWARD` nodes, is the sole planning unit.
Capability providers see ordinary operation occurrences and descriptors; graph phase does not
change the query shape or capability contract. No output role, hidden output, constant source, or
graph value is removed for planning.

A zero-node pass-through graph creates no query, makes no provider call, produces an empty
partition list, and still receives one logical-memory requirement per graph value.

### Failure ownership and order

The complete overload follows this order:

1. validate all nine top-level arguments in declaration order without allocating a Tensor;
2. execute the existing graph-stage compile once, preserving all established failures;
3. build and validate publication roles against the final graph;
4. derive constant and deferred-diagnostic snapshots from the final validated graph;
5. visit nodes in stored order and perform one planning selection per node;
6. derive partitions;
7. derive logical memory;
8. construct and cross-validate `CompileArtifacts`; and
9. return only the complete immutable result.

Existing graph-stage `NullPointerException` and `IllegalArgumentException` contracts remain
unchanged. Planning request-composition errors and provider-thrown runtime exceptions propagate
unchanged. The internal no-hard-eligible `IllegalStateException` is the only planning failure
translated by Compiler: preserve it as the cause of a new `IllegalStateException` with exact
message shape:

```text
nodes[<index>] NodeId[value=<id>] <kind-class-name>.<kind.name()>:
no hard-eligible backend is available for ownership selection
```

Use one physical line in the actual message. This adds graph occurrence context while preserving
Planning's terminal meaning and hard requirement. Selection stops immediately; later nodes and
providers are not queried.

Artifact constructor failures indicate invalid direct assembly or an implementation defect and
use `NullPointerException`/`IllegalArgumentException` with indexed component messages. This task
does not introduce a public compile exception taxonomy because no public compile entry exists.
Trace 0004 remains the owner of future typed compile payloads; Compiler remains the owner of
graph-wide failure context and successful compile diagnostics.

## Out of scope

- changing `ARCHITECTURE.md`, an ADR, module ownership, or dependency direction
- a public compile entry point, `CompiledGraph`, engine facade, `CompileConfig`, request builder,
  service, registry, or lifecycle orchestration outside compiler
- changing the existing five-argument `GraphCompiler.compile(...)`, `GraphCompilation`,
  `AutogradPreflight.FirstOrderRequest`, objective/target/seed policy, rule matrix, graph capture,
  inference, validation, canonicalization, rewriting, folding, DCE, CSE, or pass order
- Compiler 0005A–0005E current-inventory first-order gradient completion, derivative-policy
  selection, or closure checkpoint; Compiler 0006 functional requests, explicit seeds,
  create-graph, derivative order, or higher derivatives
- public Planning eligibility result, owner map, capability matrix, graph-wide planning workflow,
  cost score, workload/profile classification, Config 0004, tuning candidate, or model autotuning
- live backend instances beyond the explicitly supplied capability-provider collaboration
- retaining capability providers or availability snapshots in compile artifacts
- selecting or retaining a concrete device, route, kernel, executable, backend DAG, transfer,
  copy, alias, layout realization, physical bytes, lifetime, slot, buffer, allocation, workspace,
  schedule, residency, or run state
- prepare, runtime, backend, engine, training, optimizer, or publication-delivery behavior
- physical logical-splat materialization, payload expansion, host-storage reads, new constant
  folding, public constant ingress, or runtime binding
- binding or solving dynamic Dimensions, exposing the internal predicate vocabulary, or deciding
  value-dependent graph validity
- trace payloads, events, correlation allocation, emission, sinks, logging, serialization, or a
  rejection taxonomy
- changing model graph, Tensor, operation, descriptor, identifier, or `PublicationBinding`
  executable behavior
- backend conformance or end-to-end integration behavior
- creating a detailed Compiler 0005A–0005E, Compiler 0006, or any later task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Partition scoring](../../../../architecture/partition-scoring.md)
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Tracing](../../../../architecture/tracing.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Backend guide style](../../../../developer-guide/documentation/backend-guide-style.md)
- [User guide style](../../../../developer-guide/documentation/user-guide-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Planning master plan](../../planning/master-plan.md)
- [Planning closure audit](../../planning/planning-contract-closure-audit.md)
- [Compiler 0003B](0003b-compile-time-constants-and-constant-folding.md)
- [Compiler 0004](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
- [Compiler 0004A](0004a-exact-composition-gradient-rule-extensions.md)
- [Compiler 0004B](0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md)
- [Compile API](../../../../api/compile-api.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Public API](../../../../api/public-api.md)
- [Runtime API](../../../../api/runtime-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `ARCHITECTURE.md` is authoritative. This task implements its existing final compile stages and
  immutable artifact boundary; it changes no ownership or dependency rule.
- Compiler owns graph-wide sequencing, query construction, owner-map assembly, publication,
  constants transport, diagnostics, failure context, and `CompileArtifacts`.
- Planning owns backend-neutral hard eligibility, baseline owner selection, maximal same-owner
  partitioning, and logical-memory derivation. It exposes only the three package-cohesive stateless
  operations required by Compiler and does not gain public graph-wide orchestration.
- Model owns immutable graph data and standalone `PublicationBinding`; the binding alone remains
  unable to prove owning-graph membership.
- Compile-time ownership contains `BackendId`, never a provider, live backend, selected device,
  route, or executable.
- The final graph is not rebuilt after publication or planning. Exact graph IDs, operation and
  descriptor references, graph phases, graph-output order, constant roles, and deferred
  constraint order remain stable.
- `CompileArtifacts` is an immutable recipe. It is distinct from package-private
  `GraphCompilation`, future `PreparedExecution`, and engine `CompiledGraph`.
- Compiler does not allocate physical buffers or construct prepared/runtime state.
- Planning does not choose kernels or interpret backend implementation vocabulary.
- Trace remains a DTO-only leaf. This task creates no trace dependency reversal or payload schema.
- Existing compiler dependencies on model, config, planning, backend-contract, and trace are
  already present. No Gradle or architecture-test change is authorized.
- If implementation requires another module edge, a public compiler facade, a graph-wide Planning
  workflow, a device/route decision, a changed graph-stage contract, or a path outside the revised
  maximum scope, stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.compiler` — owns the package-private complete compile entry and public
  immutable output-only artifact types.
- `io.github.pho001.synaptik.planning.capability` — retains public query/provider contracts,
  package-private eligibility/selection mechanics, and adds one public stateless collaboration
  that composes those internal mechanics without exposing their intermediate.
- `io.github.pho001.synaptik.planning.partition` — retains public partition recipes and widens the
  already-audited maximal same-owner generation operation for the concrete compiler consumer.
- `io.github.pho001.synaptik.planning.memory` — retains public logical-memory recipes and widens
  the already-audited derivation operation for the concrete compiler consumer.

No package is added.

Type placement:

- `io.github.pho001.synaptik.compiler.CompileArtifacts` — public immutable compile recipe for
  future prepare consumption.
- `io.github.pho001.synaptik.compiler.PublicationPlan` — public immutable output-only graph context
  for ordered forward and gradient result bindings.
- `io.github.pho001.synaptik.compiler.CompileConstantPlan` — public immutable output-only logical
  constant and bindable-input transport.
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

Tests mirror production packages. No root planning facade, ownership package, compiler service,
generic utility package, or public graph-stage type is added.

## Affected files

Expected production and Javadoc paths:

- add
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/BackendOwnerPlanning.java`
- update
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/BackendCapabilityProvider.java`
  only for current compiler-consumer Javadoc
- update
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/package-info.java`
  only for the new owner-selection collaboration/current integration boundary
- update
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/partition/MaximalSameOwnerPartitioning.java`
  only to widen the existing final type and static operation plus finalize affected Javadoc
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

Expected documentation and planning paths:

- update [Partition scoring](../../../../architecture/partition-scoring.md)
- update [Compile API](../../../../api/compile-api.md)
- update [Tensor API](../../../../api/tensor-api.md)
- update [Public API](../../../../api/public-api.md)
- update [Capability-provider guide](../../../../backend-guide/capability-provider.md)
- update [Compiling graphs guide](../../../../user-guide/compiling-graphs.md)
- update [Glossary](../../../../glossary.md)
- create and finalize this task
- update [Compiler master plan](../master-plan.md)
- update [Planning master plan](../../planning/master-plan.md)
- update [Config master plan](../../config/master-plan.md)
- update [Roadmap](../../../roadmap.md)

Review without modification unless a documented conflict requires stopping: `ARCHITECTURE.md`,
other focused architecture documents, ADRs, the historical Planning closure audit and existing
completed task specifications, model/config/planning behavior and tests outside the paths above,
trace source and master plan, runtime, prepare, engine, training, concrete backends, Gradle/build
files, architecture tests, backend-conformance tests, integration tests, Runtime API, Training
API, memory-planning design note, backend-selection guide, and all other modules.

## Maximum scope

This task may create or modify at most the exact 37 paths listed under
[Affected files](#affected-files): 17 production/Javadoc paths, eight test paths, and 12
documentation/planning paths.

The larger-than-normal count is justified by one indivisible cross-module handoff:
`GraphCompilation` source roles and output roles must be converted once from final graph IDs,
every final node must receive one owner before partition and logical-memory derivation, and each
audited Planning operation must be callable from Compiler through Java visibility that matches its
owning package. The new owner-selection collaboration, the two widened operations, their existing
surface-locking tests and package Javadocs, and the resulting public artifact contracts must agree
atomically. Executable behavior changes only in Planning and Compiler; model and config Java
changes are Javadoc-only.

No Gradle, dependency, architecture, trace, runtime, prepare, backend, engine, conformance, or
integration path is authorized. If a thirty-eighth path, another public type, another test class,
or another documentation page is required, stop and propose a task revision or follow-up instead
of expanding scope silently.

## Acceptance criteria

### Public and package-private surface

- The existing five-argument `GraphCompiler.compile(...)` and package-private
  `GraphCompilation` shape remain unchanged.
- The only new complete entry is the exact package-private nine-argument overload.
- No public compiler entry, compile request aggregate, engine facade, planning workflow, provider
  registry, or service object is added.
- `CompileArtifacts` has exactly the seven public record components specified above.
- `PublicationPlan`, `CompileConstantPlan`, and `CompileDiagnostics` are public final output-only
  immutable types with package-private construction and only the specified public accessors and
  nested public records.
- `BackendOwnerPlanning` is public, final, stateless, has a private constructor, and exposes
  exactly the specified public static `selectOwner(...)` operation.
- `BackendEligibility` and `BackendOwnerSelection`, including their operations and intermediate
  result, remain package-private.
- `MaximalSameOwnerPartitioning` and `LogicalMemoryPlanning` are public final stateless types with
  private constructors and exactly their existing public static operations; no overload or state
  is added.

### Publication and output roles

- Forward bindings pair exact requested Tensor IDs with final forward values in request order.
- Gradient bindings pair exact target Tensor IDs with final gradient values in target order.
- Forward and gradient roles remain separate; no Tensor gradient field or result Tensor is added.
- Same Tensor ID across the two lists is valid; duplicate Tensor IDs within one list fail.
- Gradient values may be shared by multiple targets or equal a forward value without an identity
  node or duplicated graph output.
- `PublicationPlan` validates every binding against its exact final graph and enforces the exact
  stable graph-output boundary.

### Constants, diagnostics, and immutability

- Every final graph input is classified exactly once as bindable or a constant source.
- Constant sources retain exact `ScalarValue` bits and types without dense expansion, storage
  reads, or physical state.
- Compile artifacts preserve the exact final graph reference, node phases, graph-output order,
  operations, descriptors, partition elements, logical requirements, output roles, constant
  roles, and deferred-constraint order.
- Public deferred diagnostics contain exact node IDs and deterministic nonblank subject/predicate
  text; the internal predicate objects remain package-private and privately preserved.
- All exposed collections are immutable snapshots with documented element-reference retention.
- No provider, snapshot, device, route, kernel, physical memory, prepare/runtime state, or mutable
  request object is retained.

### Planning orchestration

- Compiler constructs one exact `OperationCapabilityQuery` per final graph node in stored order.
- Query input and output descriptors follow exact node positions and retain graph descriptor
  references.
- Each node invokes eligibility and baseline selection exactly once; the selected exact
  `BackendId` is associated with the exact graph `NodeId`.
- Compiler, not Planning, assembles the complete owner map and controls graph-wide order.
- `BackendOwnerPlanning.selectOwner(...)` delegates exactly once to each of the two existing
  package-private capability operations without exposing the intermediate eligibility result.
- Compiler invokes the widened existing partition and logical-memory operations directly; no
  fourth-package facade, reflection, wrapper delegation, or algorithm duplication is used.
- Partition and logical-memory results equal the completed Planning 0004–0005 contracts.
- A zero-node pass-through graph makes no provider call, has empty partitions, and receives
  graph-value-order logical requirements.

### Validation, failures, and boundaries

- Top-level validation, graph-stage compilation, publication, constant/diagnostic derivation,
  per-node planning, partitioning, memory planning, and artifact construction occur in the exact
  specified order.
- Existing graph-stage and Planning composition/provider failures retain their types and messages.
- A no-hard-eligible node receives the exact compiler-owned contextual failure message and cause.
- No partial artifact is returned after any failure, and later nodes are not queried after the
  first planning failure.
- `FORWARD_ONLY`, `FORWARD_AND_BACKWARD`, and `TRAINING_STEP` mode/phase/result invariants are
  enforced without adding optimizer work.
- No architecture, dependency, Gradle, trace, runtime, prepare, backend, engine, conformance, or
  integration behavior changes.
- Compiler 0005A–0005E and Compiler 0006 remain Draft and have no detailed specifications.

### Documentation and completion

- Production Javadocs fully document ownership, visibility, nullability, validation/failure
  order, exact references, immutability, output roles, constants, diagnostics, and exclusions.
- Compile, Tensor, Public API, provider, compiling-graphs, partition-scoring, glossary, and
  planning documents distinguish current internal compilation/public artifacts from the still
  absent public compile facade, preparation, execution, trace payloads, and higher derivatives.
- The glossary defines or updates publication plan, compile artifacts, compile constant plan, and
  compile diagnostics without duplicating architecture rules.
- Documentation records reasoned no-change conclusions for Runtime and Training APIs, other
  architecture pages/ADRs, trace, prepare, runtime, engine, training, backends, Gradle,
  architecture/conformance/integration tests, and unrelated model/config/planning contracts.
- A separate clean-context documentation-focused agent finalizes affected Javadocs,
  documentation, glossary impact, examples, links, planning evidence, and no-change conclusions
  in the same overall change without repeating successful Java tests unless executable behavior
  changes or a concrete risk is recorded.

## Tests / validation

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

After executable Java stabilizes, run one final affected-module command:

```bash
./gradlew :modules:planning:test :modules:compiler:test
```

This task changes public cross-module contracts and executable behavior in two modules. Run one
final repository/architecture checkpoint:

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
all repository-local Markdown targets and heading anchors, balanced fences, final newlines, and
trailing whitespace.

Required manual/source checks:

- exact 37-path maximum and no unlisted path;
- public/package-private declaration and generated-Javadoc indexes;
- exact `CompileArtifacts` components; exact one-method `BackendOwnerPlanning`; exact public
  `MaximalSameOwnerPartitioning.partition(...)` and `LogicalMemoryPlanning.plan(...)`; and
  package-private `BackendEligibility`/`BackendOwnerSelection`;
- unchanged five-argument `GraphCompiler` and `GraphCompilation` shapes;
- no detailed Compiler 0005A–0005E or Compiler 0006 specification;
- synchronized Ready/Complete/Draft task statuses;
- no provider/snapshot retention or forbidden route/device/runtime vocabulary in artifact fields;
- no imports from runtime, prepare, engine, concrete backends, or tools in Planning/Compiler
  production changes; and
- final newlines, no trailing whitespace, and `git diff --check`.

## Dependencies

- Compiler 0001–0004B — Complete.
- Model 0025 and the model graph/publication contracts — Complete.
- Config 0001–0003 — Complete.
- Planning 0001–0006 and its `CLOSED` audit — Complete.
- Backend-contract 0001–0004 — Complete.
- Trace 0001–0002 — Complete; later payload work is not required because this task emits no
  trace event.
- Existing Gradle dependencies from Compiler to model, config, planning, backend-contract, and
  trace — already present and unchanged.

Config 0004+, Trace 0003+, Runtime, Prepare, backends, Engine, and training are not prerequisites
for this bounded compile artifact. Their missing behavior must not be implemented here.

## Follow-up tasks

- Compiler 0005A–0005D — complete the current model inventory's first-order differentiation
  coverage in dependency order across elementwise/activation; reduction/scan/softmax/statistics/
  normalization; layout/window/indexing/scatter/ordering/stochastic; and structured attention/
  convolution/pooling/loss families. Each remains a concise Draft master-plan row with no
  detailed task specification.
- Compiler 0005E — audit complete first-order role coverage, fail-closed inventory, auxiliary
  output handling, dynamic/binding-dependent rules, explicit derivative-policy decisions, and
  transitive formula-operation differentiability, then run the closure checkpoint. It remains
  Draft with no detailed task specification.
- Compiler 0006 — define explicit functional gradient requests and higher-order differentiation
  only after Compiler 0005E closes the first-order milestone and this artifact/publication
  boundary is stable. It remains Draft with no detailed task specification.
- Prepare planning may begin only through a later separate roadmap reassessment. It does not
  reorder Compiler 0005A–0005E or remove 0005E as Compiler 0006's dependency gate.
- Trace 0004 may later define typed compile payloads from stable producer facts. This task does
  not create that specification.
- Config 0005 may later aggregate existing standalone compile inputs. It is not required for the
  package-private direct compile overload.
- Future concrete binding validation may use the exact privately retained deferred constraints.
  This task deliberately exposes diagnostics, not a public predicate or binding language.

Do not create any follow-up detailed specification during this task.

## Architecture impact

Expected impact: None.

The architecture already assigns publication binding, planning orchestration, logical-memory
orchestration, diagnostics, and immutable `CompileArtifacts` to Compiler. It already assigns the
delegated backend-neutral operations to Planning and permits Compiler to depend on Planning. The
new collaboration is the concrete narrow surface anticipated by the Planning closure audit.

If implementation requires Planning to own the graph-wide loop, a new dependency, a live backend
or device in artifacts, public compile orchestration, prepared/runtime state, or a changed
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
  context stopped before editing because the planned `planning.compiler.CompilerPlanning` type
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
  selector, moving algorithms, reflection, and duplicated logic were rejected.
- Scope correction: the exact ceiling increases from 32 to 37 paths to include the two widened
  production types, both affected package Javadocs, and their two existing visibility-locking
  tests. Architecture, semantic behavior, artifact design, failure ordering, and task status do
  not change.
- The existing five-argument graph-stage `GraphCompiler.compile(...)` and package-private
  `GraphCompilation` remain unchanged. The complete artifact lifecycle is a separate
  package-private nine-argument overload so the established capture, autograd, validation, and
  exact-optimization path still executes exactly once.
- The artifact boundary retains only the final graph, compile mode, logical partitions and
  memory, ordered publication roles, logical constants/bindable inputs, and deferred diagnostic
  projections. Compiler constructs one operation-capability query per final graph node and
  retains only the selected `BackendId` in the complete owner map.

## Known limitations

- The complete compile entry remains package-private and uses direct parameters because
  `CompileConfig` and the engine facade remain Draft.
- The current ownership comparison is the completed preferred-class/provider-order cost-free
  baseline. No numeric or profile-driven scoring is present.
- Capability remains backend-level even when availability proves an exact device or device class;
  artifacts retain only the selected backend identity.
- `PublicationPlan` represents ordered forward and gradient result roles, not publication policy,
  delivery targets, aliases, copies, or Tensor gradient storage.
- `CompileConstantPlan` supports only exact logical splats established by Compiler 0003B.
- Public diagnostics expose deterministic deferred-constraint descriptions, not the internal
  predicate vocabulary, binding, trace, serialization, or failure taxonomy.
- A valid zero-node graph does not inspect provider or snapshot list elements because it asks no
  capability question.
- No public caller can currently invoke compilation or supply an explicit publication/constant
  request through a released facade.
- The implemented first-order matrix remains the closed Compiler 0004–0004B subset. Draft tasks
  0005A–0005E, not this artifact task, own complete current-inventory formula coverage, explicit
  derivative-boundary decisions, auxiliary-output adoption, dynamic/binding-dependent rules, and
  the closure checkpoint before Compiler 0006.

## Validation evidence

Implementation evidence:

- The focused Planning command passed the three selected owner-planning, partitioning, and
  logical-memory suites.
- The focused Compiler command passed the five selected graph-compiler and compile-artifact
  suites.
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
  user guide, glossary, task, three module master plans, and roadmap.
- Runtime API and Training API need no change because this task adds no public run, prepared,
  optimizer, training-request, or execution contract. Architecture/ADR and architecture-test
  sources need no change because the existing dependency direction and compiler/planning ownership
  contract are unchanged. Trace, prepare, runtime, engine, training, backend, conformance,
  integration, and Gradle files need no change because no trace schema, lifecycle/backend behavior,
  end-to-end behavior, module boundary, dependency, or build configuration changed.
- `./gradlew :modules:model:javadoc :modules:config:javadoc :modules:planning:javadoc
  :modules:compiler:javadoc` passed.
- `python3 /tmp/validate_synaptik_markdown.py` validated all 12 changed Markdown files and 677
  repository-local links, including targets, heading anchors, balanced fences, final newlines, and
  trailing whitespace.
- `git diff --check` passed with no output.
- The final inventory contains exactly 37 authorized paths: 17 production/Javadoc paths, eight
  tests, and 12 documentation/planning paths, with no other path.
- `javap -p` and generated Javadoc indexes confirm the exact seven `CompileArtifacts` components,
  one public `BackendOwnerPlanning.selectOwner(...)` method, public partition and logical-memory
  operations, package-private eligibility and selector types, and the unchanged five-argument
  `GraphCompiler`/`GraphCompilation` graph-stage shapes.
- Status, later-spec, production-import, and artifact-field scans passed: Compiler 0005 is
  `Complete`; Compiler 0005A–0005E and 0006 remain `Draft` without detailed specifications; no
  forbidden runtime/prepare/engine/backend/tools import or provider/snapshot/device/route/runtime
  artifact field is present.

## Implementation notes

- Added the narrow public Planning owner-selection collaboration while keeping eligibility and
  baseline selection internal, and widened only the existing partition and logical-memory
  stateless operations.
- Added the package-private complete compile overload, immutable publication/constant/diagnostic
  outputs, and exact seven-component `CompileArtifacts` aggregate with cross-artifact validation.
- Added focused tests for orchestration, immutable artifact contracts, validation order,
  zero-node behavior, visibility, and unchanged graph-stage compilation.
- Finalized the authorized Javadocs and explanatory/planning documentation without changing
  executable behavior during the documentation pass.

## Completion summary

- Completed the exact authorized 37-path change: 17 production/Javadoc paths, eight test paths,
  and 12 documentation/planning paths.
- Completed focused and final affected-module tests plus the repository/architecture checkpoint
  with no skipped tests, failures, or errors.
- Completed the independent documentation pass and synchronized Compiler 0005 to `Complete`;
  Compiler 0005A–0005E and 0006 remain `Draft` without detailed specifications.
- No unresolved issue or required follow-up remains for this task. The next task may be promoted
  only through the normal progressive-planning workflow.

Status: Complete
