# Complete artifact compilation and integration port

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="current-complete-artifact-compilation-and-integration-port"></a>
### Current complete artifact compilation and integration port

The complete `GraphCompiler.compile(...)` overload has these nine direct inputs:

```text
CompileMode
ordered forward Tensors
optional public FunctionalGradientRequest
explicit logical-splat ingress
GraphOptimizationConfig
BackendIntent
PartitionScoringConfig
ordered BackendCapabilityProvider values
BackendAvailabilitySnapshot values
```

Public `GraphCompilationPort.compile(...)` is the cross-module SPI for the same complete pipeline.
It accepts the same inputs except explicit logical-splat ingress, supplies
`CompileTimeConstantGraph.Ingress.empty()` internally, and returns the resulting
`CompileArtifacts` directly. This compile method performs no graph or planning work of its own. Its public
visibility lets current Engine composition cross the Java package boundary; it is not an
ordinary-user facade, and both `GraphCompiler` entries remain package-private.

For a current runnable boundary case, a provenance-free Tensor has no graph node and therefore
needs no capability provider:

```java
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.List;
import java.util.Optional;

Tensor output = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT32, Shape.of(2), Optional.empty(), false));

CompileArtifacts artifacts = GraphCompilationPort.compile(
        CompileMode.FORWARD_ONLY,
        List.of(output),
        Optional.empty(),
        GraphOptimizationConfig.disabled(),
        BackendIntent.unconstrained(),
        PartitionScoringConfig.neutral(),
        List.of(),
        List.of());
```

The input is a caller-bindable rank-one floating leaf. The result contains
`BindableInput(output.id(), artifacts.graph().inputs().getFirst())` in
`artifacts.constants().bindableInputBindings()`. Its compatibility projection
`artifacts.constants().bindableInputs()` contains the same final `ValueId`, and there is no
planned partition because there is no operation node. This proves only public SPI accessibility,
constant-free input classification, and stable logical identity association; it does not prove
backend support, preparation, execution, or an Engine lifecycle. A graph with operation nodes
requires explicitly supplied matching capability and availability collaborators.

It validates the nine top-level references in declaration order before graph construction, then
invokes the unchanged five-argument graph-stage entry exactly once. Provider and snapshot list
elements are inspected only when a final graph node is planned. A valid zero-node pass-through
graph therefore asks no capability question, accepts unused list elements without inspecting
them, produces no partitions, and still receives one logical-memory requirement per graph value.
Caller collections are neither retained nor mutated.

The end of graph-stage compilation and subsequent complete-artifact work follow this exact
sequence:

```text
optimized and validated graph
  -> published compile-time constant descriptor closure
  -> final convolution logical-layout closure
  -> final static-result logical-layout closure
  -> final GraphCompilation
  -> PublicationPlan
  -> CompileConstantPlan + CompileDiagnostics
  -> one OperationCapabilityQuery and owner selection per final node
  -> complete Map<NodeId, BackendId>
  -> MaximalSameOwnerPartitioning.partition(...)
  -> LogicalMemoryPlanning.plan(...)
  -> CompileArtifacts
```

`PublicationPlan` is a public final output-only type with package-private construction. It retains
the exact final `CompiledGraphModel` reference and immutable membership snapshots exposed by
`forwardBindings()` and `gradientBindings()`. Forward `ForwardPublicationBinding` values pair
requested Tensor IDs with the final forward graph-output prefix in request order. Gradient
`GradientPublicationBinding` values retain derivative order, stage-local target index, requested
target Tensor ID, and final gradient `valueId()` in target order. Target IDs are unique within
each derivative order. Gradient values may repeat or equal a forward value; the graph boundary
contains the forward prefix followed by each previously unseen gradient value in binding order.

`CompileConstantPlan` is a public final output-only source-role classification with
package-private construction. Its immutable `bindableInputBindings()` list contains one
`BindableInput(TensorId, ValueId)` for each caller-bindable input in final graph-input order.
Repeated use of one exact leaf produces one entry, while distinct leaves with equal descriptors
remain distinct. The existing immutable `bindableInputs()` list is the ordered `ValueId`
projection of those entries, so existing Prepare consumers retain the same contract.
`constantSources()` contains the disjoint fixed inputs in graph-input order. A
`ConstantSource(ValueId, ScalarValue)` retains the exact input identity and exact typed scalar
whose bits repeat at every logical coordinate. Neither entry retains a Tensor, descriptor,
storage object, dense payload, materialization instruction, or physical allocation. A standalone
`BindableInput` validates its components but proves final graph membership and ordering only when
the owning `CompileArtifacts` validates the complete source plan.

`CompileDiagnostics` is a public final output-only successful-compile diagnostic bundle with
package-private construction. It exposes ordered immutable
`DeferredConstraintDiagnostic(NodeId, subject, predicate)` projections. Subject and predicate
text are nonblank and deterministic for diagnostics, but the predicate text is not a public
binding language, trace schema, or serialization format. The exact immutable internal constraints
remain privately retained for later compiler-owned binding validation. A rejected compile returns
no partial artifact and is not converted into a successful diagnostic.

`CompileArtifacts` is the exact eight-component public immutable record:

```java
public record CompileArtifacts(
        CompileMode mode,
        CompiledGraphModel graph,
        List<PlannedPartition> partitions,
        LogicalMemoryPlan memory,
        PublicationPlan publication,
        CompileConstantPlan constants,
        CompileDiagnostics diagnostics,
        DerivativeGraphMetadata derivatives) {}
```

Its canonical constructor validates components in declaration order, snapshots partition-list
membership, retains exact immutable element and plan references, and cross-checks graph identity,
mode/phase/result roles, maximal graph-order partitions, derived logical memory, complete
graph-input source classification, constant type and gradient eligibility, and diagnostic node
membership. The aggregate contains no provider, availability snapshot, selected device, route,
kernel, physical buffer, transfer, schedule, executable, runtime residency, or mutable run state.
It is an immutable recipe for later prepare work, not `GraphCompilation`, a public compile facade,
engine `CompiledGraph`, or `PreparedExecution`.

For each final `CompiledNode`, Compiler resolves ordered input and output descriptors, constructs
one `OperationCapabilityQuery` retaining the exact operation and descriptor references, and calls
`BackendOwnerPlanning.selectOwner(...)` once. Planning validates and evaluates hard eligibility
once, then applies its preferred-class/provider-order baseline once. Compiler retains only the
selected exact `BackendId` in its construction-local owner map; providers and snapshots do not
enter the artifacts. The first no-hard-eligible node fails immediately with graph occurrence
context and retains Planning's terminal failure as its cause. Provider-thrown runtime exceptions
and other Planning composition failures propagate unchanged.

This current boundary adds no ordinary-user lifecycle call, cost-bearing scoring,
trace event, concrete-dimension binding, prepare/runtime/backend/engine behavior, physical memory,
or more than two reverse-mode stages.


---

[Previous topic](autograd.md) · [Next topic](operation-qualification.md)
