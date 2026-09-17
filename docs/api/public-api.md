# Public API status

## Purpose and status

This page identifies which public contracts a caller can use today and which names are architecture-level plans. It prevents conceptual lifecycle examples from being mistaken for released Java APIs.

Synaptik has no published compatibility guarantee yet. The current implementation contains the
selected public model foundation, tensor-expression metadata surface, common trace-event envelope
plus model-correlation identifiers, and the first backend-neutral planning capability contracts.
It also contains backend and backend-scoped device identity values plus a coarse
CPU-versus-accelerator device classification. Planning also contains internal per-query hard-
eligibility evaluation and cost-free baseline owner comparison. Those two steps remain internal;
public `BackendOwnerPlanning` composes them for one operation occurrence. Public stateless
`MaximalSameOwnerPartitioning` and `LogicalMemoryPlanning` generate the current immutable
partition and logical-memory recipes from compiler-owned graph-wide inputs. Package-private
Compiler orchestration now consumes those three operations and uses the public immutable
`FunctionalGradientRequest`, `GradientPublicationBinding`, `DerivativeGraphMetadata`,
`CompileArtifacts`, `PublicationPlan`, `CompileConstantPlan`, and `CompileDiagnostics` contracts.
Public `GraphCompilationPort` exposes that complete constant-free pipeline as a narrow
cross-module integration service-provider interface (SPI). The CPU backend now also exposes the
supported cross-module `CpuBackendIntegration` SPI described below, including a bounded canonical
host-byte copy for a CPU publication representation borrowed from an open Runtime result. The
advanced Engine composition surface connects these contracts for one CPU-only,
representation-level compile,
prepare, and run lifecycle. The ordinary `Engine.standard()` surface constructs and owns one
fresh CPU-only composition and now exposes owner-bound compile and prepared handles,
`TensorId`-matched host-input binding, synchronous run, forward and gradient publication
occurrences, and explicit bounded materialization of one occurrence into a detached immutable
host value. Its four one-shot `compute(...)` overloads automatically discover reachable
provenance-free input leaves, then compose a fresh forward-only compile, prepare, run, complete
publication and aggregate-byte preflight, ordered materialization, and cleanup under one Engine
admission. Its one-shot `backward(objective, targets, maximumTotalBytes)` call applies automatic
input discovery to one scalar floating gradient-eligible objective, fixes Compiler's absent
positive-one scalar seed and disconnected-target `ERROR` policy, and returns a detached objective
plus immutable target-aligned gradients. Reusable or public capability
matrices and a public graph-wide Planning workflow also remain planned.
Prepare analysis, finalization, and complete graph-preparation contracts plus the initial Runtime
geometry, prepared representation creation, per-run resource/validity, executable and transfer
cold-binding, prepared publication and result leasing, and the ordered schedule contracts are
current and now form that deliberately low-level lifecycle through the CPU integration. The backend
contract also contains an immutable caller-supplied availability snapshot; it is data, not a
discovery or liveness API. A sealed requirement family can now name one hard eligibility target.
The current config module can record that hard-target optionality, requested graph scope,
permission for optional semantics-preserving compiler optimization, and one optional soft coarse
device-class preference. It can also hold one explicit immutable model-autotuning request as
declarative data. The internal baseline consumes the compile preference after hard eligibility.
The package-private complete Compiler entry consumes all four compile-config leaves directly, and
`GraphCompilationPort` supplies them through its public integration call. No current
`CompileConfig`, public capability-matrix or eligibility surface, or numeric scoring evaluator is
callable. The ordinary Engine supplies fixed compile settings rather than those configurable
surfaces. The model-autotuning
request likewise has no current Engine or tuning integration. APIs may change through the ordered
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
- immutable graph values, nodes, compiled graph-model data, forward publication bindings, and
  distinct tensor/node/value identifiers.

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
matcher, or score. Configuration owns whether a requirement is present. Current internal planning
evaluates it with availability and capability facts; an empty hard-eligible result then fails in
the internal selector with
`IllegalStateException("no hard-eligible backend is available for ownership selection")`.
That failure is not a public compile contract. Capability-provider implementations, concrete
backends, registration, public planning, preparation, and execution remain planned.

The implemented `modules:config` surface contains four standalone compile-configuration values:

- `BackendIntent` records whether planning has one hard backend eligibility target;
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
target constrains planning; it does not select a default backend or promise discovery,
fallback, availability, capability, or successful ownership. `requireCuda.hardRequirement()`
contains the exact requirement reference supplied to `requiring`. Direct construction with an
`Optional<BackendRequirement>` retains that exact optional reference.

The canonical constructor rejects a null optional with message `hardRequirement`, and
`requiring(null)` rejects null with message `requirement`. Each factory returns a fresh record.
The record evaluates no requirement and contains no preference, scoring, profile, service,
preparation, run, publication, or execution behavior.

`graphScope` requests current internal compiler autograd expansion and combined
forward/backward compile-time graph work. The other exact values are `FORWARD_ONLY` and
`TRAINING_STEP`. The latter records the
architecture's training-step direction but does not add an optimizer, optimizer-update graph,
training session, schedule, or execution behavior.

`optimization.optionalOptimizationsEnabled()` is `true`, so the current internal compiler may apply its
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

`PartitionScoringConfig` itself does not enumerate or evaluate candidates, calculate or compare
scores, contain profile measurements, choose ownership or a device, select a route or kernel, or
perform compiler, prepare, runtime, or execution work. Current Planning interprets only its
optional class preference through a cost-free provider-order baseline, and package-private
Compiler supplies that value per final graph node. `CompileConfig`, immutable cost profiles, and
public graph-wide planning remain planned; the current advanced Engine consumes the four
standalone compile inputs directly.

The implemented `config.tuning` package contains one separate declarative facade,
`ModelAutotuningConfig`. Possessing this value means that model autotuning was requested; there is
no `enabled` component, disabled sentinel, implicit default, or default cache location. Its five
inputs are:

- the sole current objective, `MIN_MEDIAN_ELAPSED_NANOS`;
- positive cache-miss and per-miss candidate maxima plus non-negative warmup and positive odd
  timed-sample counts;
- one caller-defined, schema-versioned representative-profile identity whose opaque bytes are
  snapshotted on construction and access;
- either `REQUIRE_TUNED_RESULT` or `ALLOW_SAFE_HEURISTIC`; and
- one explicit workload-cache `Path`, retained exactly without normalization or I/O.

For example, this current code constructs request data only:

```java
import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import java.nio.file.Path;

ModelAutotuningConfig request =
        new ModelAutotuningConfig(
                ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                new ModelAutotuningConfig.Budget(8, 16, 2, 5),
                new ModelAutotuningConfig.RepresentativeProfileIdentity(
                        1, new byte[] {0x2a, 0x11}),
                ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT,
                Path.of("cache", "workloads.bin"));
```

The concrete inputs permit at most eight distinct workload-cache misses, at most sixteen complete
candidates for each miss, two untimed warmup executions, and five timed executions per candidate.
The profile bytes identify the caller's representative profile; they are not representative input
values. Construction validates and retains policy data but does not inspect the path, run tuning,
read or write a cache, enumerate candidates, prepare an executable, or perform Engine or Runtime
work.

Later outer composition may depend on Config and `tools/tuning` and map the objective and four
budget values one-for-one, construct the tool-local profile fingerprint from the Config snapshot,
and pass the requested cache path. That composition must separately supply the actual model
fingerprint, tunable occurrences, representative input values and execution, and exact/default
eligible backend candidates. It also interprets the fallback policy: strict mode reports the
failed or unavailable complete tuning result, while safe-heuristic mode may continue through
ordinary safe heuristic preparation. The latter grants no candidate eligibility, numerical
relaxation, cache compatibility, partial-result acceptance, or suppression of an unrelated
preparation failure. No such translation, fallback control flow, or Engine tuning integration is
implemented yet; bounded graph/plan tuning also remains planned.

The public `modules:planning` surface contains eight backend-neutral compile-time declarations:

- `OperationCapabilityQuery`, an immutable operation occurrence consisting of one exact
  backend-independent `Operation` reference plus ordered immutable membership snapshots of input
  and output `TensorDescriptor` references; and
- `BackendCapabilityProvider`, an explicitly supplied collaboration with a stable non-null
  `BackendId` and a deterministic boolean capability answer; and
- `BackendOwnerPlanning`, a stateless one-method collaboration that composes internal hard
  eligibility and baseline owner comparison for one occurrence;
- `PlannedPartition`, an immutable owner-plus-node-ID recipe for one non-empty consecutive region
  of an owning compiled graph;
- `MaximalSameOwnerPartitioning`, the stateless generator over one complete compiler-assembled
  node-to-owner map;
- `LogicalMemoryRequirement`, one immutable graph-value recipe retaining logical descriptor,
  optional producer partition, distinct consumer partitions, and graph-output obligation; and
- `LogicalMemoryPlan`, an immutable ordered snapshot of distinct per-value requirements; and
- `LogicalMemoryPlanning`, the stateless derivation over one closed graph and its ordered complete
  partitions.

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
implementation. Package-private Compiler is the current consumer, while no public graph-wide
planning consumer exists.

Provider implementations must reject a null query with `NullPointerException("query")`. A false
answer carries no diagnostic reason. Inside the same package, current implementation code
validates complete provider/snapshot associations and combines backend-level support, non-empty
availability, and an optional exact hard requirement into an immutable provider-ordered
`BackendId` list. A second package-private step validates equal-ID snapshot associations and uses
that list directly as the complete candidate set. It returns the first preferred-class match, or
the first eligible identity when no preference or match exists; provider order resolves ties. It
returns the exact eligibility reference, allows extra unique snapshots, treats an empty matching
snapshot as a preference nonmatch, and selects no device. Empty eligibility fails internally
before snapshot elements are read.

The eligibility result and selector remain package-private. Public
`BackendOwnerPlanning.selectOwner(...)` composes them once for a non-null query, intent, provider
list, snapshot list, and scoring config. The intermediate does not escape. Public
`MaximalSameOwnerPartitioning.partition(...)` consumes one `CompiledGraphModel` and a complete
`Map<NodeId, BackendId>`. It validates exact owner-map coverage, then groups consecutive nodes in
the graph's stored topological order while equal `BackendId` values continue.

The following current code constructs the public recipe directly:

```java
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.List;

BackendId cpu = new BackendId("cpu");
NodeId first = new NodeId(10);
NodeId second = new NodeId(11);
PlannedPartition partition =
        new PlannedPartition(cpu, List.of(first, second));
```

`partition.owner()` returns the exact `cpu` reference. `partition.nodeIds()` is an immutable,
non-empty ordered snapshot that contains the exact `first` and `second` references. Construction
rejects a null owner or list, an empty list, the first null element, and the first later duplicate.
This direct example proves only the DTO contract; it does not prove that those node IDs belong to
one graph or share an owner. `MaximalSameOwnerPartitioning` establishes those graph-relative
facts.

The generator defines adjacency only by consecutive positions in `CompiledGraphModel.nodes()`.
Graph edges, phases, fan-out, merges, repeated inputs, graph input/output values, and multiple
outputs from one producer do not independently split an equal-owner run. A zero-node graph yields
no partitions. Each generated partition retains the exact graph `NodeId` references and the exact
owner reference associated with its first node, and both the result list and membership lists are
immutable.

Public `LogicalMemoryPlanning.plan(...)` accepts the closed graph and ordered complete partition
recipes. It validates partition nulls, membership, exact graph-order coverage, and adjacent-owner
maximality before deriving one requirement per graph value in `CompiledGraphModel.values()`
order. Generated producer and consumer entries retain exact supplied partition references;
consumers are distinct and follow partition order. `graph.outputs()` alone supplies the generated
graph-output flag.

The two logical-memory records are public data and can also be constructed directly. A standalone
`LogicalMemoryPlan` may be empty and validates only that its requirements are non-null and have
distinct `ValueId` values. A standalone requirement validates and snapshots its components but
has no owning graph against which to prove producer, consumer, or output relationships. The
public generator establishes those graph-relative facts.

These records retain `TensorDescriptor` rather than calculating element or byte counts, so
dynamic and expression dimensions remain representable. They carry no
`ForwardPublicationBinding`, `GradientPublicationBinding`, `TensorId`, lifetime, slot,
allocation, transfer, selected device, route, kernel, executable, prepared schedule, or runtime
residency. `graphOutput` is a logical preservation obligation, not a publication target or
policy.

Reusable or public capability matrices, public eligibility evaluation, a public graph-wide
Planning workflow, numeric or cost scoring, profiles, physical memory, preparation, runtime, and
execution remain planned. Compiler currently owns owner-map assembly and immutable artifact
construction.

The public `modules:compiler` contract surface now contains:

- `GraphCompilationPort`, the narrow constant-free cross-module compile SPI used by Engine
  composition, not an ordinary-user lifecycle facade;
- `FunctionalGradientRequest`, the immutable one/two-stage reverse-mode input with aligned
  cotangent seeds and an explicit disconnected-target policy;
- `GradientPublicationBinding`, the derivative-order and stage-local
  target-to-gradient-publication association;
- `DerivativeGraphMetadata`, the exact graph-adjacent per-node derivative-order sidecar;
- `CompileArtifacts`, the exact eight-component immutable recipe containing mode, final graph,
  maximal partitions, logical memory, publication, constants, diagnostics, and derivative
  metadata;
- `PublicationPlan`, an output-only exact-graph context exposing separate immutable ordered
  `forwardBindings()` of `ForwardPublicationBinding` values and `gradientBindings()` of
  `GradientPublicationBinding` values;
- `CompileConstantPlan`, an output-only immutable graph-input classification with nested
  `BindableInput(TensorId, ValueId)` associations and `ConstantSource(ValueId, ScalarValue)`
  values; and
- `CompileDiagnostics`, an output-only immutable successful-compile projection with nested
  `DeferredConstraintDiagnostic(NodeId, subject, predicate)` values.

The three output-only plan classes have package-private constructors. `CompileArtifacts` is a public
record whose canonical constructor cross-validates the supplied components and snapshots
partition membership. These types are public so later prepare and engine work can consume stable
compile-time data. `GraphCompilationPort.compile(...)` makes complete constant-free compilation
callable for cross-module integration while both `GraphCompiler` entries and `GraphCompilation`
remain package-private. The port always treats reachable provenance-free forward leaves as
caller-bindable, retains no provider or availability snapshot, and neither prepares nor executes
its result.

The artifact result retains no provider, availability snapshot, selected device, route, kernel,
physical buffer, transfer, prepared schedule, executable, residency, or mutable run state.
`PublicationPlan` validates graph membership and boundary order but adds no delivery policy.
`CompileConstantPlan.bindableInputBindings()` associates every final caller-bindable graph input
with its originating immutable Tensor identity in graph-input order. Its existing
`bindableInputs()` method remains the ordered `ValueId` projection for compatibility, and its
constant entries carry logical splats rather than dense data or storage. No source entry retains a
live Tensor, descriptor copy, provenance, or host storage.
`CompileDiagnostics` carries deterministic text projections rather than a public predicate
language, trace schema, or serialization.

The public `backends:cpu` integration surface contains two supported types:

- `CpuCapabilityProvider`, the stateless fail-closed CPU capability provider; and
- `CpuBackendIntegration`, the closeable lifecycle SPI intended for Engine composition rather
  than ordinary application code.

One `CpuBackendIntegration.open()` call creates the fixed exact/default CPU composition. It
reports the immutable `cpu/host` availability fact, exposes the retained capability provider,
builds the positional preparation and schedule-assembler collaborations for exactly one non-empty
maximal CPU-owned partition, and exposes `prepare(CompileArtifacts)` as the complete CPU-owned
composition entry. That entry derives exact physical declarations for eligible fully static,
canonical source-only published splat constants and passes them to shared Prepare; Engine delegates
without interpreting constant roles or CPU geometry. The integration also borrows intrinsically
compatible `HostTensorStorage` as a non-owning Runtime buffer representation. Its
`copyToCanonicalHostBytes(representation, descriptor, maximumBytes)` operation copies one exact
current CPU publication representation into a fresh caller-owned mutable `byte[]`. This is
Engine-facing SPI, not an ordinary application result API; the caller pairs the representation
with the final descriptor from the same publication occurrence while the Runtime result lease is
open. The adapter also attempts bounded automatic OpenBLAS discovery and qualification. That
attempt may add an eligible native candidate; the portable route remains available when discovery
or qualification fails, and OpenBLAS is neither guaranteed to be present nor guaranteed to be
selected.

The copy requires a fully static Shape, a present resolved layout, a non-negative byte limit, and
one of the two exact current CPU representation implementations. It traverses logical coordinates
in canonical row-major order, with the final axis changing fastest, through the descriptor's
non-negative element offset and positive or zero strides. A rank-zero descriptor reads its offset
element. A Shape with any zero extent returns a fresh empty array without source-element access.
Repeated physical addresses from a zero stride are valid and produce repeated logical values.

Segment-backed source elements are interpreted using the CPU's native-order unaligned primitive
access, but the detached output format is always big-endian. `FLOAT64` and `FLOAT32` preserve raw
NaN payloads and signed zeros; `BFLOAT16` preserves its exact stored 16 bits; `INT64` and `INT32`
preserve their two's-complement bit patterns; and `BOOL` accepts only the exact stored byte `0` or
`1`. No data-type conversion or BOOL normalization occurs. The checked logical byte count must
not exceed either `maximumBytes` or `Integer.MAX_VALUE`; arithmetic overflow and a genuine JVM
allocation failure remain distinct failures.

The accepted graph shape is deliberately exact. A pure zero-node constant or pass-through graph
remains rejected because the CPU integration requires one non-empty maximal CPU partition. A
mixed-owner or multi-partition artifact is rejected before CPU analysis or schedule assembly
because combining multiple backend contributions remains later Engine/Prepare work. For an
accepted artifact, the assembler describes creation of fresh run-owned CPU buffers and workspaces,
initialization of Compiler constants, one prepared CPU execution occurrence, and publications in
Compiler order. A source-only constant gains no executable schedule step. Assembly performs none
of that physical work: `PreparedExecution` retains immutable recipes only, and creation of every
fresh `RunState` invokes each initialized-buffer recipe exactly once. Sequential or concurrent
runs therefore own distinct initialized representations.

The adapter owns any qualified OpenBLAS coordinator until `close()`. Callers must keep it open
while preparing or running recipes obtained from it and must coordinate closure with active runs.
Borrowing transfers no storage ownership, and closing a borrowed wrapper does not close caller
storage. The borrow operation validates only intrinsic storage consistency and current access; it
cannot validate an expected logical input type, required span, or write role because those target
facts are absent. A successful host-byte copy likewise does not close, retain, mutate, transfer,
or change validity of its source. It is synchronous: the caller must keep the adapter, Runtime
result lease, and representation open, keep storage accessible to the calling thread, and prevent
source mutation or closure until return. Independent valid calls may run concurrently, but racing
mutation has no atomic-snapshot guarantee. Typed logical input binding, a simpler end-user execute
surface, and ordinary typed or host-materialized results remain Engine work.

The adapter does not expose CPU internals, discover backends for Runtime, perform tuning, or turn
OpenBLAS into another backend identity. CPU directly depends on Compiler only because this SPI
consumes public `CompileArtifacts`; CPU remains independent of Engine.

The public `modules:runtime` surface now contains five focused packages:

- `runtime.memory` defines `BufferSlot`, `WorkspaceSlot`, and immutable ordered
  `PreparedMemoryPlan` byte geometry;
- `runtime.resource` defines the nominal backend-implemented `BufferRepresentation` and
  `WorkspaceRepresentation` cleanup roles plus immutable `PreparedRepresentationPlan` origins,
  caller-input occurrences, ordinary and initialized buffer origins, and typed buffer/workspace
  creators;
- `runtime.run` defines borrowed/run-owned buffer bindings, the array-backed one-run `RunState`,
  dense-coordinate `PreparedPublication`, per-run `BoundPublication`, the whole-state
  `RunResult` lease, and stateless `PreparedExecutionRunner`; and
- `runtime.execution` defines the immutable two-component `PreparedExecution` root, reusable
  `PreparedExecutable` recipes, their dense buffer/representation and workspace selections,
  aligned buffer-access declarations, and
  per-run `BoundInvocation` objects, plus reusable `PreparedBufferTransfer` recipes and per-run
  `BoundBufferTransfer` actions; and
- `runtime.schedule` defines immutable `PreparedSchedule` recipes, their sealed plan-associated
  `Step` family, the optional first-only `RepresentationCreationStep`, `ExecutionStep`,
  `BufferTransferStep`, and the dense final `PublicationStep` suffix.

`PreparedRepresentationPlan` snapshots dense buffer origins and workspace creators against one
exact memory plan. Concrete backends implement immutable thread-safe creator callbacks that return
fresh physical results for each run. Package-private cold setup validates every borrowed caller
input before callbacks, creates buffers then workspaces, and rolls back successful created results
in reverse order if setup fails. The public runner is the narrow composition seam for that setup;
it is not an Engine or value facade.

Every representation bound into an open `RunState` is structurally resident until closure. Each
buffer copy has one independent explicit validity bit: borrowed inputs start valid, ordinary
newly created run-owned buffers start invalid, and initialized run-owned buffers start valid.
The initialized creator, not Runtime, materializes the logical value. Workspaces are run-owned
scratch without logical validity. Query and mutation are constant-time dense-array operations and
perform no physical copy, backend call, ownership change, or implicit coherence.

`PreparedExecutable.bind` requires the exact same `PreparedMemoryPlan` reference as the open
`RunState`. It resolves selections in supplied order and delegates explicit checked
representation compatibility to the concrete backend. The backend then creates a
`BoundInvocation` with direct concrete typed fields and the exact state association. The hot
`execute()` call checks only that the retained state is open before delegating to the backend;
execution after state closure is rejected.

Each executable buffer selection declares `READ_ONLY`, `WRITE_ONLY`, or `READ_WRITE` access. The
runner validates reads before invalidating every copy of each output buffer, then validates only
the exact declared writes after successful backend return. A failed action leaves all copies of
its output buffers invalid.

`PreparedBufferTransfer.bind` similarly requires an exact matching open state but selects two
distinct, already-created representations of one logical buffer. Concrete compatibility checks
remain cold, and the resulting `BoundBufferTransfer` stores direct concrete typed fields. An
already-valid destination is a no-op. Otherwise a valid source permits one backend action, and
only successful return marks the destination valid. Backend failure leaves Runtime validity
unchanged. Materialization is this same explicit transfer to an equivalent already-created
destination, not another operation, allocation path, or route search.

One schedule retains an exact `PreparedMemoryPlan` and an immutable ordered snapshot of creation,
executable, buffer-transfer, and publication occurrences. Every step must report that same plan by reference
identity. The
creation occurrence is optional for compatibility but, when present, is sole and first. Empty and
executable-only or transfer-only schedules and repeated executable or transfer occurrences remain
valid. Publication occurrences, when present, form a dense `0..N-1` final suffix; distinct result
positions may alias one exact representation. Construction performs
no callback, binding, execution, allocation, resource action, or ownership transfer. No
`PreparedUnit` is needed because list position is the occurrence and each current step supplies
its recipe and plan association.

`PreparedPublication` cold-binds one exact already-created buffer representation by dense buffer
and representation positions. `BoundPublication.publish()` requires that selected copy to be
valid at the publication moment and changes only a one-shot local flag. It performs no lookup,
fallback, transfer, conversion, or backend work. A complete ordered set of published occurrences
constructs `RunResult`, which privately preserves result aliases and leases cleanup of the entire
`RunState`. Empty results are valid; partial publication transfers no cleanup responsibility.
While that exact Runtime lease is open, `RunResult.publicationRepresentation(resultIndex)` returns
the borrowed exact `BufferRepresentation` selected for one dense occurrence. Repeated access and
aliased occurrences preserve exact reference identity. The caller must not close, transfer, or
retain the reference beyond result closure and must externally prevent access from racing result
or state activity because Runtime `RunResult` is not thread-safe. The method performs no copy,
wrapper construction, cast, validity query, transfer, or physical access; closed-state failure
precedes index validation.

This method is an inward Runtime service-provider interface (SPI), not an ordinary result-value
API. A `BufferRepresentation` is only a nominal backend-owned lifecycle role: it is not host
storage, a Tensor value, or guaranteed to be host-accessible. The ordinary Engine `RunResult`
continues to expose metadata only, and `AdvancedRunResult` continues to expose only result count
and lifecycle. Neither Engine result exposes the inward representation.

One `PreparedExecution(memoryPlan, schedule)` now supplies the current reusable Runtime root. It
retains both exact references and requires `schedule.memoryPlan() == memoryPlan`. It owns no
resource, has no close or run method, and creates no per-run state. Its immutable recipe may be
shared by concurrent readers, while each later invocation must use a distinct mutable
`RunState`.

`PreparedExecutionRunner.run(execution, callerInputs)` creates one isolated state, cold-binds
every non-creation occurrence before the first action, traverses direct bound references in
schedule order, and either returns the whole-state result lease or closes the state after failure.
The runner is stateless and may serve concurrent calls; each call remains synchronous and owns
distinct mutable state.

These contracts contain no concrete backend implementation and do not provide physical access,
host/Tensor value access, or Engine behavior. Creator callbacks leave physical allocation mechanics to
the implementing backend, and binding owns no auxiliary closeable resource or ownership change.
See the [Runtime API](runtime-api.md) for creation, validity, scheduling, selection, lifecycle,
failure, and thread-safety details.

The public `modules:prepare` surface now spans the existing `prepare.analysis` package, four
root-package finalization contracts, and five complete-graph orchestration declarations.
Analysis receives one immutable `PartitionDag` for exactly one planned partition and retains one
typed opaque selected plan plus exact buffer/workspace declarations. The DAG keeps exact nodes in
stable topological order and exposes immutable producer/output-port, consumer/input-port, edge,
external-input-occurrence, and local-sink facts. Repeated input ports and every output of a
multi-output node remain separate occurrences in node-and-port order. External inputs and local
sinks are topology-only: they imply no cross-partition owner, publication, transfer,
materialization, memory, fusion, route, or schedule policy.

`PrepareContext` has five canonical record components: `partitionDag`, `values`,
`memoryRequirements`, `constants`, and `backendInputs`. Its derived `partition()` and `nodes()`
views delegate to the DAG. A public six-argument constructor accepting the previous partition and
node-list call shape remains source-compatible by constructing one DAG; no binary compatibility
is promised for the changed record descriptor, component reflection, equality, hash code, or text
form. The context exposes no complete `CompiledGraphModel`, graph region, callback, or
Compiler-owned aggregate.

`PreparationResourceAssignment` associates each exact declaration
with an assigned Runtime slot and dense plan index. `BackendPartitionFinalization` carries one
typed analysis, the exact shared `PreparedMemoryPlan`, and assignments in declaration order to a
`BackendPartitionFinalizer`. `PreparedPartition` retains the exact planned partition and returned
`PreparedExecutable`. `PartitionPreparation` keeps each backend's typed inputs, preparer, and
finalizer positionally associated. `PreparedBufferAssignment` translates one graph `ValueId` to
its exact `BufferSlot` and dense plan index. `PreparedScheduleContext` exposes the complete
immutable finalized facts to one explicit `PreparedScheduleAssembler`, and
`GraphPreparation.prepare(...)` returns the exact validated `PreparedExecution`.

The four-argument preparation form additionally accepts a complete list of
`ProducerlessPublishedConstantResource` contributions. One such value carries the exact graph and
logical-requirement references plus externally supplied physical byte size and alignment for a
fully static source-only published compile-time constant. Shared Prepare validates the exact role,
canonicalizes contributions by final graph-value encounter order, and appends their buffer slots
after ordinary declarations. The value remains outside partition-local analysis and receives no
partition-finalizer assignment. The existing three-argument form contributes an empty list.

The complete-set operation that validates coverage and source identity, assigns slots, constructs
the shared memory plan, and invokes finalizers remains package-private behind the public graph
operation. Its current conservative
policy shares one buffer slot across declarations of the same `ValueId` using maximum geometry
and gives every workspace declaration a distinct slot. It performs no physical allocation,
closeable prepared-resource acquisition, run-state creation, invocation binding/execution,
transfer, residency, constant initialization or materialization, or publication. Producerless
contributions still require at least one non-empty planned partition and therefore do not make a
zero-node graph executable. The supplied assembler constructs an
immutable schedule recipe once; Prepare validates exact source, execution, coordinate, and
publication coverage before constructing the prepared root. The advanced Engine now composes this
operation with the exact CPU integration that it owns.

## Current ordinary and advanced CPU lifecycle

`Engine.standard()` is the current ordinary entry point. Every invocation directly opens one
fresh CPU integration, transfers its ownership into a private advanced lifecycle owner, and
returns a distinct `Engine`. The fixed standard inventory is exactly CPU; Metal and CUDA have no
current lifecycle adapters. There is no discovery, service lookup, caller-supplied backend set,
or process-global Engine. `isClosed()` observes when delegated closure begins, and `close()` uses
the advanced owner's thread-safe, idempotent, failure-retaining cleanup protocol.

The ordinary lifecycle is:

```text
ordered Tensor output expressions -> CompiledGraph
CompiledGraph                     -> PreparedExecution
PreparedExecution + caller Tensors in any order -> RunResult
```

`compile(List<Tensor>)` creates a forward-only handle. Its three-list overload accepts ordered
forward outputs, one explicit cotangent seed per output, and identity-unique ordered gradient
targets for one first-order reverse-mode request. It does not infer a seed or targets and does not
provide no-argument backward. `CompiledGraph.inputs()` reports the final caller-bindable
`TensorId` and descriptor pairs in Compiler binding order. Compilation and preparation retain no
caller Tensor or host-storage reference and do not read current host associations.

`prepare(...)` accepts only a compile handle from the same exact Engine and returns a fresh,
immutable, reusable owner-bound handle. `run(...)` accepts every required logical Tensor exactly
once in arbitrary order, matches by `TensorId`, validates the complete descriptor, and then
snapshots each Tensor's current `HostTensorStorage` association once in final binding order. Each
run borrows fresh non-owning CPU wrappers and owns an isolated Runtime state. The caller retains
the storage and its arena and must keep its scope alive, accessible where used, and free from
conflicting mutation until the returned result closes, including after synchronous `run(...)`
returns. Replacing a Tensor's association after the snapshot cannot redirect that run.

`RunResult.publications()` returns the same immutable forward-then-gradient occurrence list on
every call. Each occurrence has a dense result index, final descriptor, role, and logical Tensor
identity. A forward occurrence uses the requested output identity. A gradient occurrence uses the
requested target identity, derivative order one, and the target's explicit list index; it does
not claim a generated gradient Tensor identity. Repeated gradient values and forward/gradient
aliases remain distinct occurrence objects even when Runtime selects one physical
representation. The metadata does not reveal physical aliasing, storage, or numerical values.
While the result and Engine remain open, `materialize(publication, maximumBytes)` accepts only an
exact occurrence object from that result and returns a fresh detached `HostTensorValue`. Repeated
calls and aliased occurrences are copied independently; there is no cache or deduplication.

Closing a result releases its Runtime lease and Engine-created wrappers but never caller storage.
Closing the Engine closes still-open results in reverse successful-run order before the CPU
integration. Result metadata remains readable after either close; compiled and prepared metadata
also remains readable, and a completed `HostTensorValue` remains readable after either close, but
closed-Engine handles cannot start new work. Materialization is current only for the fixed CPU
composition and fully static resolved publication descriptors. It is not an implicit transfer,
cross-backend format promise, Tensor, storage association, typed array, or persistence format.

The current one-shot forms are exactly:

```java
HostTensorValue value = engine.compute(output);
HostTensorValue boundedValue = engine.compute(output, maximumTotalBytes);
List<HostTensorValue> values = engine.compute(outputs);
List<HostTensorValue> boundedValues = engine.compute(outputs, maximumTotalBytes);
```

Each call transiently and iteratively inventories provenance-free leaves reachable through Model
Tensor-expression provenance by exact object identity. After compilation,
`CompiledGraph.inputs()` alone selects membership and order by `TensorId`; Engine does not traverse
Compiler intermediate representation (IR), reconstruct liveness, or retain Tensor/provenance
state. Parameters and buffers need no Engine-specific category: their current Tensor bindings are
discovered like any other reachable provenance-free leaf. Each call freshly compiles with the
ordinary fixed forward-only settings, prepares, runs once, verifies the complete forward-publication set,
preflights the sum of canonical returned payload lengths before copying, materializes every
occurrence in requested order, and closes its temporary result before returning. The ordered form
returns an immutable list of immutable detached values; distinct occurrences are copied
independently even if they select one inward representation. The limit covers returned canonical
payload bytes only, not inputs, recipes, Runtime buffers or workspaces, object overhead, defensive
copies, peak memory, or other allocation.

One Engine admission covers that complete sequence. Engine closure waits for an admitted call,
including its result cleanup; a call that loses admission observes the closed-Engine failure
before argument validation. A failure returns no partial result. Temporary cleanup still runs,
with a distinct cleanup failure suppressed on the primary failure. Selected leaf storage remains
caller-owned and must stay live, accessible, and free from conflicting mutation until the
synchronous call completes; unselected discovered storage is not inspected. The `compute(...)`
forms create no compile/prepared/result/value/Tensor-inventory cache or reuse and add no backward,
implicit target/seed behavior, tuning, cross-backend transfer, or Tensor execution method.

The current one-shot backward form is:

```java
ScalarObjectiveBackwardResult result =
        engine.backward(objective, targets, maximumTotalBytes);
```

`objective` must be a scalar floating gradient-eligible Tensor expression. `targets` is an
explicit ordered non-empty list whose elements are unique by exact object identity; its order
defines `result.gradients()`. Targets may be leaves, intermediates, or the objective itself when
Compiler accepts their gradient semantics and differentiable connection. Engine inventories
reachable provenance-free leaves before compilation, then lets final `CompiledGraph.inputs()`
select authoritative input membership and order. It accepts no explicit input list, infers
neither targets nor liveness, mutates no Tensor, and creates no eager tape.

Every call freshly compiles one stage with an absent scalar cotangent seed and
`DisconnectedPolicy.ERROR`, prepares, runs, validates the objective-first then target-ordered
publications, and preflights every static/resolved per-value byte count plus the exact aggregate
before the first copy. Each occurrence is copied independently, including inward aliases. The
limit covers only the objective and gradient canonical payloads. Cleanup completes before return;
Engine close waits through an admitted call, concurrent calls use isolated run state, and cleanup
failures follow the ordinary primary/suppressed failure rules. The returned
`ScalarObjectiveBackwardResult`, its objective, and all gradients are detached and remain readable
after Engine and caller storage close.

The fixed seed and policy keep this convenience narrow. Use ordinary explicit-seed
`compile -> prepare -> run -> materialize` for reusable execution, explicit cotangent seeds, or
multiple forward outputs. Use `AdvancedEngine.compile(...)` with a
`FunctionalGradientRequest` for `ZERO`, two stages, higher-order construction, or the full policy
surface.

This complete supported example uses only the current CPU static/resolved `CONTIGUOUS` path:

```java
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;

static Tensor input(Arena arena, float... values) {
    Shape shape = Shape.of(values.length);
    TensorDescriptor descriptor = new TensorDescriptor(
            DataType.FLOAT32,
            shape,
            Optional.of(LayoutDescriptor.contiguous(shape)),
            false);
    MemorySegment source = MemorySegment.ofArray(values);
    MemorySegment storage = arena.allocate(source.byteSize(), Float.BYTES);
    MemorySegment.copy(source, 0, storage, 0, source.byteSize());
    return TensorFactory.create(
            descriptor,
            Optional.empty(),
            Optional.of(new MemorySegmentStorage(
                    DataType.FLOAT32, values.length, storage)));
}

List<HostTensorValue> retained;
try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
    Tensor left = input(arena, 1.5f, -2.25f);
    Tensor right = input(arena, 3.25f, 4.5f);
    retained = engine.compute(
            List.of(left.contiguous(), right.contiguous()),
            4L * Float.BYTES);
    assert retained.size() == 2;
    assert retained.get(0).bytes().getFloat(0) == 1.5f;
    assert retained.get(1).bytes().getFloat(0) == 3.25f;
}
ByteBuffer first = retained.get(0).bytes();
assert first.isReadOnly();
assert first.remaining() == 2 * Float.BYTES;
```

The expression leaves are discovered automatically, while the returned positions still follow
the output request. Both values remain readable after Engine and caller arena closure, while the
arena itself stayed caller-owned. Every call compiles and prepares afresh. Use the lower-level
`compile -> prepare -> run -> materialize` lifecycle when a prepared recipe should be reused or
only selected publications should be copied.

This complete one-shot backward example uses the currently supported resolved scalar `FLOAT32`
CPU path. Its input storage remains caller-owned, while the two returned values remain readable
after both the Engine and storage arena close:

```java
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.ScalarObjectiveBackwardResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;

ScalarObjectiveBackwardResult retainedBackward;
try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
    Shape shape = Shape.scalar();
    TensorDescriptor descriptor = new TensorDescriptor(
            DataType.FLOAT32,
            shape,
            Optional.of(LayoutDescriptor.contiguous(shape)),
            true);
    MemorySegment source = MemorySegment.ofArray(new float[] {2.5f});
    MemorySegment storage = arena.allocate(source.byteSize(), Float.BYTES);
    MemorySegment.copy(source, 0, storage, 0, source.byteSize());
    Tensor input = TensorFactory.create(
            descriptor,
            Optional.empty(),
            Optional.of(new MemorySegmentStorage(DataType.FLOAT32, 1, storage)));
    Tensor objective = input.contiguous();

    retainedBackward = engine.backward(
            objective,
            List.of(input),
            2L * Float.BYTES);
    assert retainedBackward.objective().bytes().getFloat(0) == 2.5f;
    assert retainedBackward.gradients().getFirst().bytes().getFloat(0) == 1.0f;
}
ByteBuffer objectiveBytes = retainedBackward.objective().bytes();
ByteBuffer gradientBytes = retainedBackward.gradients().getFirst().bytes();
assert objectiveBytes.getFloat(0) == 2.5f;
assert gradientBytes.getFloat(0) == 1.0f;
```

`contiguous()` supplies the non-empty supported CPU operation. Compiler supplies the absent
positive-one seed, which the completed Compiler 0006B5 -> Prepare 0005 -> CPU 0010H chain carries
as a source-only publication beside that partition. This proves the scalar objective and its
positive-one gradient on the current path; it does not promise pure zero-node or mixed-backend
composition, broader operation support, optimizer behavior, or a training session.

For a single already-constructed Tensor expression, the complete one-shot execution shape is:

```java
try (Engine engine = Engine.standard()) {
    HostTensorValue result = engine.compute(output);
}
```

Here `output` already retains the Model expression provenance leading to its leaves. The returned
value is detached; closing the Engine does not invalidate it.

An NN `Model` follows the same boundary. This conceptual example intentionally assumes only the
existing `model`, `input`, and `Tensor` contracts and invents no construction or initialization
API:

```java
Tensor output = model.forward(input);
try (Engine engine = Engine.standard()) {
    HostTensorValue result = engine.compute(output);
}
```

`model.forward(input)` constructs the Tensor expression directed acyclic graph (DAG); it does not
execute it. `engine.compute(output)` performs execution and host materialization. Repeated model
execution should retain the expression and use `compile -> prepare -> run(prepared, explicit
inputs)` rather than paying the one-shot compile and prepare cost on every call.

`AdvancedEngine` is the current low-level composition root. Construction accepts the exact
`CpuBackendIntegration` returned by `CpuBackendIntegration.open()` and takes ownership of it.
The lifecycle is deliberately explicit:

```text
Tensor outputs + compile leaves -> AdvancedCompiledGraph
AdvancedCompiledGraph           -> AdvancedPreparedExecution
prepared execution + borrowed BufferRepresentation inputs -> AdvancedRunResult
```

`compile(...)` requires a non-empty output list and accepts the four standalone compile inputs
plus an optional `FunctionalGradientRequest`. The returned `AdvancedCompiledGraph` is an opaque,
immutable handle bound to the exact open Engine that created it. `prepare(...)` accepts only such
a handle and currently succeeds only when CPU owns one non-empty maximal partition. A zero-node,
mixed-owner, or multiple-partition artifact therefore fails during this CPU-only preparation
step; it is not silently repartitioned or routed elsewhere. The returned
`AdvancedPreparedExecution` is another immutable owner-bound handle. It may be shared by
concurrent callers because every `run(...)` creates isolated mutable Runtime state.

Inputs to `run(...)` are already-created Runtime `BufferRepresentation` values in graph-input
order. `borrow(...)` wraps one caller-owned `MemorySegmentStorage` for this purpose. Neither the
wrapper nor the Engine owns or closes the storage or its backing arena. The wrapper must remain
open, and its storage must remain valid, through advanced-result closure, including after the
synchronous run returns. Closing the wrapper ends only that borrow. Compile and run do not retain
or mutate their caller lists.

`AdvancedRunResult` owns the completed run state until it or the Engine closes it. It currently
exposes only `resultCount()`, not published values. Its idempotent `close()` releases that state;
if a close is already in progress, another close waits uninterruptibly and restores interruption
before returning. Closing the Engine rejects new work with
`IllegalStateException("advanced engine is closed")` before argument, owner, or inward validation,
waits for admitted work, closes still-open results in reverse run order, and then closes its owned
CPU integration. Compiled and prepared handles have no independent resource lifecycle, but become
unusable when their Engine closes.

This advanced surface does not provide mixed-backend composition, backend discovery, typed
logical input binding, typed publication metadata, host materialization, public one-shot
execution, backward convenience, tuning integration, or ordinary Engine exception translation. Supplying a
gradient request to `compile(...)` invokes the existing compiler mode/request contract; it does
not add those later conveniences or guarantee that the current CPU-only prepare step can lower
the resulting artifact.

## Current CPU limitations and planned conveniences

The current CPU composition prepares exactly one non-empty maximal CPU partition. Zero-node
pass-through graphs, mixed-owner graphs, and multiple partitions are rejected. Inputs and every
CPU operation occurrence require fully static Shapes and resolved compatible layouts; adding
`contiguous()` after an operation does not retroactively resolve that operation's inputs or
output. A compile success therefore does not guarantee CPU preparation or execution success.

The focused supported examples use `CONTIGUOUS` directly over resolved `FLOAT32` leaves. The
seeded example reuses `seedLeaf.contiguous()` as the explicit seed for two outputs. It does not use
`ADD`, whose current public expression result has unresolved layout. The current CPU integration
does support a fully static canonical source-only published splat when it accompanies the required
non-empty CPU partition: CPU supplies its physical declaration, shared Prepare assigns it, and
each run initializes a distinct run-owned representation before binding and schedule traversal.
Pure constant graphs remain unsupported because they have no non-empty partition. See the
[Runtime API ordinary examples](runtime-api.md#current-ordinary-engine-boundary) for the complete
setup.

Ordinary Engine host materialization is current through the CPU integration's canonical copy.
Engine selects and authenticates the exact publication occurrence, serializes the synchronous
copy against result closure, and returns detached immutable canonical bytes. The CPU operation
itself still neither selects a publication nor obtains a representation, performs a transfer,
synchronizes result closure, constructs a Tensor or `HostTensorStorage`, streams a payload, or
generalizes the format across backends. Engine-owned one-shot forward execution is current through
the four ordinary `Engine.compute(...)` overloads described above; it does not add an
`output.execute()` Tensor method. Engine-owned one-shot scalar-objective backward execution is
current through the explicit-target `Engine.backward(...)` method described above. It adds no
no-argument backward contract, implicit targets, or Tensor mutation.

## Compatibility expectations during development

- Treat Javadoc and implemented tests as the contract for code that exists.
- Treat architecture snippets as conceptual unless a reference page explicitly marks them current.
- Do not depend on draft planning types or package names in external code.
- Check the [roadmap](../planning/roadmap.md) before assuming a planned module is available.

## Related documentation

- [Getting started](../getting-started.md)
- [Architecture overview](../architecture/overview.md)
- [Implementation roadmap](../planning/roadmap.md)
