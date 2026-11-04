# Complete [artifact](../../glossary.md#term-artifact) [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) and [integration port](../../glossary.md#term-integration-port)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

Complete compilation packages a verified [graph](../../glossary.md#graph) and its logical plans into one [immutable](../../glossary.md#term-immutable) result.
The integration **port** is a restricted doorway for other [modules](../../glossary.md#term-module) to request that work. The
result describes what later preparation must arrange; it contains neither physical [buffers](../../glossary.md#term-buffer) nor
an executable [ready](../../glossary.md#term-ready) to [run](../../glossary.md#term-run).

## Read the artifact as instructions for another owner

Engine is the caller of the public compilation port; Compiler invokes Planning to select backend
identities and returns `CompileArtifacts`. Prepare is the later consumer: it projects partitions
to backends, assigns resource slots and constructs a runnable recipe. The immutable lists copy
membership, not tensor payloads. None of these compile outputs owns input storage, native buffers
or a live device. That distinction is why a valid zero-node artifact in the example can still
fail preparation. Follow [Prepare's resource handoff](../runtime/prepare.md) to see where persistent
resource ownership first transfers, and [run-state ownership](../runtime/representations.md) for
per-invocation allocations.

<a id="current-complete-artifact-compilation-and-integration-port"></a>
### Current complete [artifact](../../glossary.md#term-artifact) [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) and [integration port](../../glossary.md#term-integration-port)

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

Public `GraphCompilationPort.compile(...)` is the cross-[module](../../glossary.md#term-module) [SPI](../../glossary.md#term-service-provider-interface-spi) for the same complete pipeline.
It accepts the same inputs except explicit logical-splat ingress, supplies
`CompileTimeConstantGraph.Ingress.empty()` internally, and returns the resulting
`CompileArtifacts` directly. This [compile](../../glossary.md#compile) method performs no [graph](../../glossary.md#graph) or [planning](../../glossary.md#term-planning) work of its own. Its public
visibility lets current [Engine](../../glossary.md#term-engine) composition cross the Java package boundary; it is not an
ordinary-user [facade](../../glossary.md#term-facade), and both `GraphCompiler` entries remain package-private.

For a current runnable boundary case, a [provenance](../../glossary.md#provenance)-free [Tensor](../../glossary.md#tensor) has no graph [node](../../glossary.md#node) and therefore
needs no capability provider:

```java
// This leaf has no producer, so the artifact needs no operation owner; preparation is still separate.
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

The input is a caller-bindable [rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) floating [leaf](../../glossary.md#term-leaf). The result contains
`BindableInput(output.id(), artifacts.graph().inputs().getFirst())` in
`artifacts.constants().bindableInputBindings()`. Its compatibility projection
`artifacts.constants().bindableInputs()` contains the same final [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot), and there is no
planned [partition](../../glossary.md#partition) because there is no [operation](../../glossary.md#operation) node. This proves only public SPI accessibility,
[constant](../../glossary.md#term-constant)-free input classification, and stable logical [identity](../../glossary.md#term-identity-meanings) association; it does not prove
[backend](../../glossary.md#backend) support, preparation, execution, or an Engine [lifecycle](../../glossary.md#lifecycle). A graph with operation nodes
requires explicitly supplied matching capability and availability collaborators.

It validates the nine top-level references in declaration order before graph construction, then
invokes the unchanged five-[argument](../../glossary.md#term-argument) graph-stage entry exactly once. Provider and [snapshot](../../glossary.md#term-snapshot) list
[elements](../../glossary.md#term-element) are inspected only when a final graph node is planned. A valid zero-node pass-through
graph therefore asks no capability question, accepts unused list elements without inspecting
them, produces no partitions, and still receives one logical-memory requirement per [graph value](../../glossary.md#graph-value).
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
the exact final [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel) reference and [immutable](../../glossary.md#term-immutable) membership snapshots exposed by
`forwardBindings()` and `gradientBindings()`. Forward `ForwardPublicationBinding` values pair
requested Tensor IDs with the final [forward graph](../../glossary.md#forward-graph)-output prefix in request order. [Gradient](../../glossary.md#term-gradient)
`GradientPublicationBinding` values retain [derivative](../../glossary.md#term-derivative) order, stage-local [target](../../glossary.md#term-target) index, requested
target Tensor ID, and final gradient `valueId()` in target order. Target IDs are unique within
each derivative order. Gradient values may repeat or equal a forward value; the graph boundary
contains the forward prefix followed by each previously unseen gradient value in [binding](../../glossary.md#term-binding) order.

`CompileConstantPlan` is a public final output-only source-role classification with
package-private construction. Its immutable `bindableInputBindings()` list contains one
`BindableInput(TensorId, ValueId)` for each caller-[bindable input](../../glossary.md#bindable-input) in final graph-input order.
Repeated use of one exact leaf produces one entry, while distinct leaves with equal [descriptors](../../glossary.md#term-descriptor)
remain distinct. The existing immutable `bindableInputs()` list is the ordered `ValueId`
projection of those entries, so existing [Prepare](../../glossary.md#prepare) consumers retain the same contract.
`constantSources()` contains the disjoint fixed inputs in graph-input order. A
`ConstantSource(ValueId, ScalarValue)` retains the exact input identity and exact typed [scalar](../../glossary.md#term-scalar)
whose bits repeat at every logical [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix). Neither entry retains a Tensor, descriptor,
storage object, dense [payload](../../glossary.md#artifact-envelope-and-atomic-publication), [materialization](../../glossary.md#materialization) instruction, or physical allocation. A standalone
`BindableInput` validates its components but proves final graph membership and ordering only when
the owning `CompileArtifacts` validates the complete source plan.

`CompileDiagnostics` is a public final output-only successful-compile diagnostic bundle with
package-private construction. It exposes ordered immutable
`DeferredConstraintDiagnostic(NodeId, subject, predicate)` projections. Subject and predicate
text are nonblank and deterministic for diagnostics, but the predicate text is not a public
binding language, [trace](../../glossary.md#trace) [schema](../../glossary.md#term-schema), or [serialization](../../glossary.md#term-serialization) format. The exact immutable internal constraints
remain privately retained for later [compiler](../../glossary.md#term-compiler)-owned binding [validation](../../glossary.md#term-validation). A rejected compile returns
no partial artifact and is not converted into a successful diagnostic.

`CompileArtifacts` is the exact eight-component public immutable record:

```java
// The record describes logical plans; none of its components is a per-run physical buffer.
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

Its [canonical](../../glossary.md#term-canonical-form) constructor validates components in declaration order, snapshots partition-list
membership, retains exact immutable element and plan references, and cross-checks graph identity,
mode/phase/result roles, maximal graph-order partitions, derived logical memory, complete
graph-input source classification, constant type and gradient eligibility, and diagnostic node
membership. The aggregate contains no provider, availability snapshot, selected [device](../../glossary.md#term-device), route,
[kernel](../../glossary.md#kernel), physical [buffer](../../glossary.md#term-buffer), transfer, schedule, executable, [runtime](../../glossary.md#term-runtime) [residency](../../glossary.md#residency), or [mutable](../../glossary.md#term-mutable) [run state](../../glossary.md#run-state--runstate).
It is an immutable recipe for later prepare work, not `GraphCompilation`, a public compile facade,
engine `CompiledGraph`, or [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution).

For each final `CompiledNode`, Compiler resolves ordered input and output descriptors, constructs
one [`OperationCapabilityQuery`](../../glossary.md#operation-capability-query--operationcapabilityquery) retaining the exact operation and descriptor references, and calls
`BackendOwnerPlanning.selectOwner(...)` once. Planning validates and evaluates hard eligibility
once, then applies its preferred-class/provider-order baseline once. Compiler retains only the
selected exact [`BackendId`](../../glossary.md#backend-identity--backendid) in its construction-local [owner](../../glossary.md#term-owner) map; providers and snapshots do not
enter the artifacts. The first no-hard-eligible node fails immediately with graph [occurrence](../../glossary.md#term-occurrence)
context and retains Planning's terminal failure as its cause. Provider-thrown runtime exceptions
and other Planning composition failures propagate unchanged.

This current boundary adds no ordinary-user lifecycle call, cost-bearing scoring,
trace event, concrete-[dimension](../../glossary.md#dimension) binding, prepare/runtime/backend/engine behavior, physical memory,
or more than two [reverse-mode](../../glossary.md#primal-adjoint-and-differentiation-seed) stages.


---

[Previous topic](autograd.md)
