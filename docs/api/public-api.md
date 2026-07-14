# Public API status

## Purpose and status

This page identifies which public contracts a caller can use today and which names are architecture-level plans. It prevents conceptual lifecycle examples from being mistaken for released Java APIs.

Synaptik has no published compatibility guarantee yet. The current implementation contains the
selected public model foundation, tensor-expression metadata surface, common trace-event envelope
plus model-correlation identifiers, and the first backend-neutral planning capability contracts.
It also contains backend and backend-scoped device identity values plus a coarse
CPU-versus-accelerator device classification. Compiler orchestration, capability matrices,
ownership planning, partitioning, prepare, runtime, concrete backend integration, and engine APIs
remain planned. The backend
contract also contains an immutable caller-supplied availability snapshot; it is data, not a
discovery or liveness API. A sealed requirement family can now name one hard eligibility target.
The current config module can record that hard-target optionality, requested graph scope,
permission for optional semantics-preserving compiler optimization, and one optional soft coarse
device-class preference. No current compile aggregate, compiler, capability matrix, scoring
evaluator, or ownership planner consumes those values. APIs may change through the ordered
planning process.
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) defines module boundaries, not source or binary
compatibility.

## Current public contracts

The implemented `modules:model` surface contains:

- data type metadata, typed scalar values, and numeric promotion;
- BFLOAT16 scalar bit conversion;
- static, named dynamic, and expression dimensions, immutable Shapes, and local broadcasting;
- resolved static layout geometry and host-storage contracts;
- public mutable `Tensor`, eager leaf factories, explicit-source random construction, and
  backend-independent expression metadata;
- typed operation attributes and occurrence signatures, shared multi-output producer provenance,
  and operation-specific result carriers; and
- immutable graph values, nodes, compiled graph-model data, publication bindings, and distinct
  tensor/node/value identifiers.

The [Tensor API reference](tensor-api.md) documents these current contracts, inputs, results,
failures, and examples. The current model surface records meaning and metadata; it does not imply
compiler capture, backend support, kernels, prepared execution, runtime residency, or numerical
execution.

The implemented `modules:trace` surface contains:

- producer-assigned non-negative `TraceEventId` values;
- `TracePhase` lifecycle classification for `COMPILE`, `PREPARE`, and `RUN`;
- `TraceLevel` detail and severity classification;
- the open method-free `TracePayload` marker; and
- the generic `TraceEvent<T extends TracePayload>` envelope with a producer-supplied monotonic
  nanosecond reading; and
- the nominal `TraceNodeId`, `TraceValueId`, and `TraceTensorId` records for trace-local
  correlation with producer-owned model identities.

The [tracing explanation](../architecture/tracing.md) documents the envelope, correlation, and
ownership boundaries. The producer owns correlation-value allocation, uniqueness, lifetime, and
mapping; a trace-local numeric value need not equal the corresponding model ID. Concrete payload
families, partition/backend/device/unit/run and other later correlation domains, typed backend
attributes, serialization, sinks, and emission remain planned. Backend is a payload family and
producer role, not another lifecycle phase.

The implemented `modules:backend-contract` surface contains:

- `BackendId`, an immutable open-string identity for a backend ownership domain; and
- `BackendDeviceId`, an immutable composite identity for one opaque device token scoped by its
  owning `BackendId`; and
- `DeviceClass`, a coarse declarative category with exactly `CPU` and `ACCELERATOR`, in that
  declaration order; and
- `BackendAvailabilitySnapshot`, an immutable point-in-time association from one backend's
  currently reported available device identities to their classes; and
- the sealed, method-free `BackendRequirement` family, with one-component
  `BackendIdRequirement`, `BackendDeviceIdRequirement`, and `DeviceClassRequirement` records.

The current concepts have separate roles:

```text
BackendId       = backend ownership domain
BackendDeviceId = one exact device identity inside that domain
DeviceClass     = coarse CPU or accelerator category
BackendAvailabilitySnapshot
                = one backend's supplied device-to-class availability fact
BackendRequirement
                = one hard target; no matching or preference behavior
```

The two identity records reject null components and blank string values. Every other identity
component is retained by the exact caller-supplied reference, and string content keeps its case
and surrounding whitespace; the records do not trim, normalize, intern, or resolve aliases.
Ordinary record equality therefore compares the exact stored content, and the backend component
prevents equal device tokens from different backends from colliding.
`DeviceClass` is not stored in `BackendDeviceId`; the snapshot supplies the association without
changing either identity or category. The enum declaration order supports stable identity and
diagnostics, not preference, score, priority, capability, or fallback policy.

For example, the following inputs report one accelerator device for the `"cuda"` backend:

```java
import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import java.util.Map;

BackendId cuda = new BackendId("cuda");
BackendDeviceId cudaZero = new BackendDeviceId(cuda, "0");
BackendAvailabilitySnapshot availability =
        new BackendAvailabilitySnapshot(
                cuda,
                Map.of(cudaZero, DeviceClass.ACCELERATOR));
```

The snapshot requires every device identity to have a `BackendId` equal to `cuda`. It retains the
exact backend, device, and class references and uses an immutable structural copy of the map, so
later changes to a mutable source map cannot change the snapshot. Map iteration order is
unspecified. An empty map means that the supplying context reports no currently available device
for that backend; the backend identity remains part of the snapshot.

The snapshot does not discover devices, register a backend, monitor liveness, refresh itself,
evaluate capability, choose ownership, or guarantee preparation or execution.

The requirement family records exactly one hard target. These inputs construct all three current
variants:

```java
import io.github.pho001.synaptik.backend.contract.BackendDeviceIdRequirement;
import io.github.pho001.synaptik.backend.contract.BackendIdRequirement;
import io.github.pho001.synaptik.backend.contract.BackendRequirement;
import io.github.pho001.synaptik.backend.contract.DeviceClassRequirement;

BackendRequirement exactBackend = new BackendIdRequirement(cuda);
BackendRequirement exactDevice = new BackendDeviceIdRequirement(cudaZero);
BackendRequirement acceleratorClass =
        new DeviceClassRequirement(DeviceClass.ACCELERATOR);
```

`exactBackend` targets later ownership by a `BackendId` equal to `cuda`. `exactDevice` targets a
`BackendDeviceId` equal to `cudaZero`, which also fixes the owning backend. `acceleratorClass`
allows any later eligible device whose class is `ACCELERATOR`; it does not identify a particular
backend or device. Each record rejects a null component with a `NullPointerException` whose
message is that component's name and returns the exact supplied reference from its accessor.

These values neither inspect `availability` nor prove that a target is registered, available,
capable, or preparable. They contain no sentinel for absence, preference, fallback, combination,
matcher, or score. Later configuration owns whether a requirement is present and how intent is
expressed; later planning owns evaluation with availability and capability facts and failure when
no eligible target remains. The no-match exception type and message are not yet defined.
Capability reporting, concrete backends, registration, planning, preparation, and execution
remain planned.

The implemented `modules:config` surface contains four standalone compile-configuration values:

- `BackendIntent` records whether later planning has one hard backend eligibility target;
- `CompileMode` records the requested compile-time graph scope;
- `GraphOptimizationConfig` permits or suppresses optional semantics-preserving compiler work; and
- `PartitionScoringConfig` records an optional soft `DeviceClass` preference for later ranking of
  already eligible ownership candidates.

They are immutable requests, not a runnable compiler configuration aggregate. For example:

```java
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.BackendIdRequirement;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;

BackendId cuda = new BackendId("cuda");
BackendIntent unconstrained = BackendIntent.unconstrained();
BackendIntent requireCuda =
        BackendIntent.requiring(new BackendIdRequirement(cuda));
CompileMode graphScope = CompileMode.FORWARD_AND_BACKWARD;
GraphOptimizationConfig optimization = GraphOptimizationConfig.standard();
PartitionScoringConfig neutralRanking = PartitionScoringConfig.neutral();
PartitionScoringConfig preferAccelerator =
        PartitionScoringConfig.preferring(DeviceClass.ACCELERATOR);
```

`unconstrained.hardRequirement()` is empty. That absence means only that no hard eligibility
target constrains later planning; it does not select a default backend or promise discovery,
fallback, availability, capability, or successful ownership. `requireCuda.hardRequirement()`
contains the exact requirement reference supplied to `requiring`. Direct construction with an
`Optional<BackendRequirement>` retains that exact optional reference.

The canonical constructor rejects a null optional with message `hardRequirement`, and
`requiring(null)` rejects null with message `requirement`. Each factory returns a fresh record.
The record evaluates no requirement and contains no preference, scoring, profile, service,
preparation, run, publication, or execution behavior.

`graphScope` requests later compiler autograd expansion and combined forward/backward compile-time
graph work. The other exact values are `FORWARD_ONLY` and `TRAINING_STEP`. The latter records the
architecture's training-step direction but does not add an optimizer, optimizer-update graph,
training session, schedule, or execution behavior.

`optimization.optionalOptimizationsEnabled()` is `true`, so a later compiler may apply its
standard optional semantics-preserving pipeline. `GraphOptimizationConfig.disabled()` returns a
fresh false value that requests skipping only optional optimization. It cannot disable capture,
ordering, inference, validation, mandatory canonical representation, mode-required autograd,
publication binding, planning, preparation, or execution. Neither value exposes a pass list,
pass order, numerical relaxation, backend fusion switch, or execution policy.

`neutralRanking.preferredDeviceClass()` is empty. That means only that this value supplies no
explicit coarse device-class preference; it does not choose a default, promise fallback, or imply
equal candidate scores. `preferAccelerator` contains the exact `DeviceClass.ACCELERATOR` reference.
The preference is soft and applies only after hard eligibility, so it neither makes an eligible
CPU candidate ineligible nor weakens a conflicting hard requirement or guarantees accelerator
ownership. Direct construction retains the exact non-null `Optional<DeviceClass>` reference and
rejects null with message `preferredDeviceClass`; `preferring(null)` rejects null with message
`deviceClass`. Both factories return fresh values.

`PartitionScoringConfig` does not enumerate or evaluate candidates, calculate or compare scores,
contain profile measurements, choose ownership or a device, select a route or kernel, or perform
compiler, prepare, runtime, or execution work. `CompileConfig`, immutable profiles, planning
interpretation, compiler consumption, and every lifecycle consumer remain planned.

The implemented `modules:planning` surface contains two backend-neutral compile-time contracts:

- `OperationCapabilityQuery`, an immutable operation occurrence consisting of one exact
  backend-independent `Operation` reference plus ordered immutable membership snapshots of input
  and output `TensorDescriptor` references; and
- `BackendCapabilityProvider`, an explicitly supplied collaboration with a stable non-null
  `BackendId` and a deterministic boolean capability answer.

The query validates only non-null references and the input/output occurrence counts declared by
the operation signature. It does not validate operand data types, Shapes, layouts, graph closure,
availability, hard requirements, scoring, or execution. Mutable source lists cannot change a
constructed query, while the exact operation and descriptor element references are retained.

This current example asks an illustrative local provider about one unary `ABS` occurrence:

```java
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

TensorDescriptor vector =
        new TensorDescriptor(DataType.FLOAT32, Shape.of(4), Optional.empty(), false);
Operation abs = new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE);
OperationCapabilityQuery query =
        new OperationCapabilityQuery(abs, List.of(vector), List.of(vector));

BackendId cpu = new BackendId("cpu");
BackendCapabilityProvider illustrativeCpu = new BackendCapabilityProvider() {
    @Override
    public BackendId backendId() {
        return cpu;
    }

    @Override
    public boolean supports(OperationCapabilityQuery candidate) {
        Objects.requireNonNull(candidate, "query");
        return candidate.operation().kind() == UnaryElementwiseKind.ABS;
    }
};

boolean semanticOwnershipSupported = illustrativeCpu.supports(query);
```

The concrete inputs are one FLOAT32 descriptor with Shape `[4]`, the `ABS` operation, and the
backend identity `"cpu"`. The result is `true` because this illustrative provider recognizes that
operation kind. It proves only semantic ownership support for this immutable occurrence; it does
not prove CPU registration or availability, evaluate `BackendIntent`, choose a device or CPU
route, prepare work, or execute values. The repository supplies no production provider
implementation and no current compiler or planning consumer.

Provider implementations must reject a null query with `NullPointerException("query")`. A false
answer carries no diagnostic reason. Capability matrices, hard-eligibility evaluation,
device-level queries, provider composition, ownership scoring, partitions, and diagnostics remain
planned.

## Planned public lifecycle

The architecture uses this conceptual shape:

```java
// Conceptual API: these lifecycle types and methods are not implemented yet.
CompiledGraph graph = CompiledGraph.compile(output, CompileConfig.auto());
PreparedExecution execution = graph.prepare(PrepareConfig.defaults());
RunResult result = execution.run(inputs, RunOptions.defaults());
```

Compile will create immutable graph and ownership artifacts. Prepare will ask explicitly registered concrete backends to lower their assigned partitions. Run will execute the prepared schedule with per-invocation state. See the [compile](compile-api.md) and [runtime](runtime-api.md) reference pages for the planned boundaries.

## Compatibility expectations during development

- Treat Javadoc and implemented tests as the contract for code that exists.
- Treat architecture snippets as conceptual unless a reference page explicitly marks them current.
- Do not depend on draft planning types or package names in external code.
- Check the [roadmap](../planning/roadmap.md) before assuming a planned module is available.

## Related documentation

- [Getting started](../getting-started.md)
- [Architecture overview](../architecture/overview.md)
- [Implementation roadmap](../planning/roadmap.md)
