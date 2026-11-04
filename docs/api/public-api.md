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
cross-module integration service-provider interface (SPI). The CPU backend exposes the supported
`CpuBackendIntegration` lifecycle SPI, including bounded canonical host-byte materialization. The Metal backend exposes
`MetalBackendConfiguration`, `MetalBackendIntegration`, `MetalTraceObserver`, and a closed
occurrence-qualified domain. The [current production support matrix](../backend-guide/production-support.md)
owns the detailed operation/type/layout/derivative inventory, including attention, normalization
and Scan. Metal retains seven storage carriers and its own predicate domain; it has no FP8
support. CPU low-precision parity does not widen Metal admission.
Direct BFLOAT16/FLOAT16 mixed-low execution remains outside the documented Metal authority;
use explicit FLOAT32 casts where required by that matrix.

Low-containing Metal partitions retain their qualified custom-program route rather than an
alternate CPU route. Explicit casts are graph operations, not implicit conversion or fallback.
Backend ABI, program schema and cache identities are implementation compatibility boundaries;
use the backend guide and authenticated current artifacts rather than pinning copies of those
version strings in application guidance.

Canonical typed host ingress/publication and CPU/Metal transfer support Metal's seven carriers.
Cross-owner values require the documented static rank-0..16 canonical or approved positive-stride
non-overlapping layouts and checked physical spans. Transfer preserves carrier values and
performs no conversion or fallback.
`Engine.builder()` is the public explicit composition root for
opened CPU and Metal integrations. It freezes their Planning inputs in registration order and
supports a complete non-empty plan only when every partition has one exact registered owner. The
`Engine.standard()` convenience still constructs one fresh CPU-only composition through that same
path. Both forms expose owner-bound compiled graphs and one reusable closeable PreparedExecution.
Engine.run binds exact Tensor IDs in arbitrary order; Engine.runValues binds fresh fully static
values in compiled order (exact descriptors for static slots, qualified binding for named B).
Both use synchronous execution, independent publication
leases and explicit detached host materialization.
The four one-shot `compute(...)` overloads automatically discover reachable provenance-free input
leaves, then compose a fresh forward-only compile, prepare, run, complete publication and aggregate-
byte preflight, ordered materialization, and cleanup under one Engine admission. The one-shot
`backward(objective, targets, maximumTotalBytes)` call applies automatic input discovery to one
scalar floating gradient-eligible objective, fixes Compiler's absent positive-one scalar seed and
disconnected-target `ERROR` policy, and returns a detached objective plus immutable target-aligned
gradients. Public capability matrices and a public graph-wide Planning workflow remain planned.
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
surfaces. The model-autotuning request is current through `Engine.prepareTuned(...)` for the
bounded single-owner CPU workflow and exact singleton-NEG Metal workflow described below. Generic
graph/plan tuning remains planned. APIs may change through the ordered planning process.
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) defines module boundaries, not source or binary
compatibility.

## Current public contracts

The implemented `extensions:data` surface pairs existing Tensor values with immutable host
`ValidSequenceLengths`, logical `SequenceAxisOrder` and explicit `PaddingSide` in a
`PaddedSequenceBatch` record, replacing the unpublished right-only type without an alias.
Its values builder requires explicit order, side and counts, copies counts at the setter, and
validates geometry without inspecting Tensor contents. Direct record construction remains available.
`LongSequenceBatcher` instead allocates one fresh INT64 heap Tensor and pads ordered long-array
rows to their maximum length, on either side and in either axis order, without truncation.
Neither path binds runtime length inputs or changes recurrent-scan semantics; see the
[Data sequence batching API](data-api.md) for construction, ownership and failures.

The implemented `extensions:text` surface contains final `WhitespaceTokenizer` and
`TextSequenceBatcher` in one flat package. The tokenizer freezes ordered ordinary IDs from 2,
reserves padding (PAD)=0 and unknown (UNK)=1, and splits exact strings using
`Character.isWhitespace(int)`. The adapter requires explicit padding side and axis order,
encodes each row once and delegates all numeric batching to Data. Text exports only Data,
with Model transitive; NN/Engine composition belongs to the caller. PAD selects an ordinary
Embedding row, not a mask or a frozen zero row. See the [Text API](text-api.md) for a complete example.

The implemented `extensions:nn` declarative boundary adds `ModelSpecification` and
`DeclarativeModel` in `nn.module`, plus closed `Activation` in `nn.layers`. Describe a
bound supported Sequential, reconstruct an evaluation-mode descriptor-only skeleton, then
use inherited strict state loading to supply values. The specification exposes immutable
typed steps, canonical current Tensor-identity groups, ordered `stateDescriptors()` and
input-specific `outputDescriptor(...)`; it captures neither arbitrary Model callbacks nor
bytes. Six parameterized layer families and Activation expose typed `specification()`
snapshots; `Linear.fromState(Tensor, Optional<Tensor>, boolean)` binds exact supplied state
while retaining supplied promotion or automatic input guards. Existing APIs stay supported.
`Module.trainingParameters()` now exposes Training's generic live admission boundary. Its private
adapter retains and re-traverses the root's exact ordered paths, wrappers and Tensors, while
Training owns storage/value validation. Open/copy callers pass this adapter instead of Module;
the same source survives explicit native copying. Training is physically `modules/training`
with unchanged `:extensions:training`/Java identities and exactly public Model/Engine edges;
NN exports Model/Training/Checkpoint/Engine. Owned Models provide high-level execution and
`withEngine`; the fluent builder composes existing layers without changing those owners.
See the [generic Training example](training-api.md#generic-live-parameter-admission) and
the [NN composition reference](training-api.md#current-nn-declarative-composition-contract)
and [complete reconstruction example](../user-guide/declarative-models.md). Whole-model files use
the separate ModelCheckpoint workflow; automatic Training-owned resume is not added by the builder.

The implemented `modules:model` surface contains:

- data type metadata, typed scalar values, and numeric promotion;
- BFLOAT16, FLOAT16 and both OFP8 scalar bit conversions (see [low floating types](../user-guide/tensors.md#low-floating-types-and-raw-bytes));
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
- the open method-free `TracePayload` marker;
- the generic `TraceEvent<T extends TracePayload>` envelope with a producer-supplied monotonic
  nanosecond reading;
- nominal non-negative `TraceNodeId`, `TraceValueId`, and `TraceTensorId` records for trace-local
  correlation with producer-owned model identities;
- nominal non-negative `TraceBackendId`, `TraceDeviceId`, `TracePreparedUnitId`, and
  `TraceInvocationId` records for producer-stream backend execution correlations;
- the closed `TraceOutcomeStatus`, `TraceRouteKind`, `TraceCacheStatus`,
  and `TraceNativeStatusKind` vocabulary plus exact-code `TraceNativeStatus`; and
- immutable `BackendPreparationOutcome` and `BackendInvocationOutcome` payloads.

The outcome payloads carry only trace-owned IDs and closed neutral facts. Success requires present
native success; failure permits no native status or a present non-success status. `NOT_QUERIED`
records no cache lookup and is not a miss. The [tracing explanation](../architecture/tracing.md)
documents exact component roles, correlation, and ownership boundaries. The producer owns
correlation allocation, uniqueness, lifetime, mapping, timestamps, and emission. Broader payload
families, partition/schedule/run IDs, typed backend attributes, serialization, sinks, and shared or
generic emission facilities remain planned. Backend is a producer role, not another lifecycle
phase.

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
That internal message is not the ordinary public compile failure contract. `Engine.standard()`
supplies one CPU provider; `Engine.builder()` compiles against the frozen providers of explicitly
registered CPU and/or Metal integrations. Generic plugin registration, other executable backends,
and a public graph-wide Planning surface remain planned.

The implemented `modules:config` surface contains four standalone compile-configuration values:

- `BackendIntent` records whether planning has one hard backend eligibility target;
- `CompileMode` records the requested compile-time graph scope;
- `GraphOptimizationConfig` permits or suppresses optional semantics-preserving compiler work;
- `PartitionScoringConfig` records an optional soft `DeviceClass` preference for later ranking of
  already eligible ownership candidates.

Model owns one family/dtype numerical contract; these configuration values contain no numerical
selector. Metal uses occurrence-specific provider capability and its fixed custom program for
admitted BFLOAT16/FLOAT16 partitions. Finite arithmetic tolerances are test-only route
qualification, not Engine or Runtime policy.

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
The record evaluates no requirement and contains no preference, scoring, numerical policy, service,
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
Compiler supplies that value per final graph node. There is no graph-wide numerical selection.
`CompileConfig` and public graph-wide planning remain planned; the current Engine passes the four
operational standalone compile inputs without a numerical selector.

The implemented `config.tuning` package contains one separate declarative facade,
`ModelAutotuningConfig`. Possessing this value means that model autotuning was requested; there is
no `enabled` component, disabled sentinel, implicit default, or default cache location. Its seven
ordered inputs are:

- the sole current objective, `MIN_MEDIAN_ELAPSED_NANOS`;
- Phase-1 positive cache-miss and per-miss candidate maxima plus non-negative warmup and positive
  odd timed-sample counts;
- one caller-defined, schema-versioned representative-profile identity whose opaque bytes are
  snapshotted on construction and access;
- either `REQUIRE_TUNED_RESULT` or `ALLOW_SAFE_HEURISTIC`; and
- one explicit workload-cache `Path`, retained exactly without normalization or I/O;
- one independent Phase-2 `CompletePlanBudget` containing a positive complete-plan candidate
  maximum, non-negative warmup count, positive odd timed-sample count, positive whole-transaction
  execution ceiling, and non-negative aggregate correctness ceiling in bytes; and
- one explicit non-empty-string model-plan-cache `Path`, retained exactly without normalization or
  I/O.

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
                Path.of("cache", "workloads.bin"),
                new ModelAutotuningConfig.CompletePlanBudget(
                        4, 1, 3, 20L, 1_048_576L),
                Path.of("cache", "model-plan.bin"));
```

The concrete inputs permit at most eight distinct workload-cache misses, at most sixteen complete
candidates for each Phase-1 miss, two Phase-1 untimed warmup executions, and five Phase-1 timed
executions per candidate. Independently, Phase 2 permits at most four actual complete-plan
candidates, one warmup and three timed samples per candidate, at most twenty total executions, and
at most 1,048,576 aggregate canonical-publication correctness bytes. Later Phase-2 preflight uses
the actual candidate count `N` and checks `N * (1 + W + S)` against the total-execution ceiling.
The profile bytes identify the caller's representative profile; they are not representative input
values. Construction validates and retains policy data but does not inspect either path, run
tuning, read or write a cache, enumerate candidates, perform correctness execution, prepare an
executable, or perform Engine or Runtime work.

Current `Engine.prepareTuned(...)` maps both budgets and the objective one-for-one, supplies the
request's caller-defined model and representative-profile identities, binds its live
representative inputs, and passes both explicit cache paths to `tools/tuning`. Phase 1 exposes
exactly one backend-local occurrence at occurrence index 0, partition index 0, and weight 1. CPU
retains its current local and complete-plan alternatives. Metal accepts only an exact singleton
NEG and exposes exactly the ordered local candidates custom kernel then MPSGraph. Its authenticated
selected decision then passes unchanged into a one-candidate complete-plan batch fixed to that
route. Phase 2 checks every candidate by exact canonical publication bytes before any warmup or
timed execution, measures the bounded candidates, authenticates the winner, cleans the
representative session, and prepares exactly one fresh selected production recipe.

Current CPU and Metal complete-plan reuse is `SESSION`, so Phase 2 retains but does not inspect,
create, or publish the supplied model-plan path. Metal local compatibility is also `SESSION`, so
the supplied workload-cache path can neither provide a Metal hit nor receive a Metal publication.
The same generic translation remains valid for a future producer that can honestly declare
`PERSISTENT`. Strict mode reports a failed or unavailable complete tuning result, while
safe-heuristic mode may return one fresh ordinary safe preparation through the already-selected
owner with no evidence. Fallback grants no candidate eligibility, numerical relaxation, cache
compatibility, partial-result acceptance, owner substitution, or suppression of an unrelated
preparation failure. Multiple-occurrence extraction, multiple partitions, mixed ownership, and
broader Compiler/Planning plan alternatives remain planned.

The public `modules:planning` surface contains these backend-neutral compile-time declarations:

- `OperationCapabilityQuery`, an immutable operation occurrence consisting of one exact
  backend-independent `Operation` reference, ordered immutable membership snapshots of input and
  output `TensorDescriptor` references, an optional independently scoped `CompiledScanBody`,
  and an optional `ScanActivityPlan`;
- `ScanActivityPlan`, immutable structural presence and ordered-merge facts for one exact local
  Scan body, with distinct carry/seed roots and temporal accumulators;
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

The query validates non-null references, signature occurrence counts and attachment applicability.
A present body requires `ScanKind.SCAN`; a present activity plan requires that exact attached
body graph and applicable header positions, next-presence bindings and ordered temporal ADDs.
Its canonical constructor has five arguments, including both optionals. The genuine
three-argument ordinary/fixed-operation constructor supplies empty body and activity attachments;
there is no four-argument body-bearing constructor. Missing-body Scan questions remain
representable for provider rejection. Attachment checks do not infer ordinary descriptors or
establish backend support. Mutable source lists cannot change a constructed query, while the
exact operation, descriptor elements and optional body/activity references are retained.
For scans, provider true requires the whole body and loop binder, not merely supported children.
The [complete capability contract](compile-api.md#current-operation-capability-contracts) separates
implemented transport, Compiler query/demand emission and Prepare scoped projection/static
binding/demand validation from actual backend support. CPU0030 now qualifies bounded FLOAT32
same-shape pointwise scans, including automatic reverse only when its actual generated body also
qualifies. CPU0031 adds qualified homogeneous FLOAT32 broadcast, SUM/sumToShape and bounded
shape/layout closure, including independently qualified actual reverse bodies. CPU0032 adds static
homogeneous FLOAT32 rank-two MATMUL under that same whole-body/binder qualification. CPU0033
implements homogeneous FLOAT64 across the complete bounded declared algebra and actual
first-order reverse closure. Those task inventories are historical qualification layers.
Current general Scan composes every ordinary CPU-supported PURE unit under its ordinary guards,
not a finite Scan-kind whitelist. All-valid WholeTensor differs from proved length-aware
ROW(axis) and SHARED roles; SHARED tensors are not restricted to scalars. Original length-AD
carries/sequences and emissions stay ROW(0), while intermediate ROW axes follow the proof.
Row-mixing requires materialized same-length support, not Tensor identity.
Selected ordinary VJPs retain ND cuts and finite-path guards; absence is not a present zero seed.
CPU BFLOAT16, FLOAT16 and both OFP8 types use these same ordinary and shared Compiler AD owners,
with FLOAT32 working values and final logical narrowing under existing family/geometry guards.
This adds neither Metal FP8 nor Metal general Scan support. Unlike low operands promote to
FLOAT32 and FLOAT64 dominates; Metal retains its independent mixed-low restrictions.
Cold/wire and NN workflow inventories remain separately bounded. The
[CPU guide](../backend-guide/cpu-backend.md#bounded-float32-general-scan) states the execution scope.

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
route, prepare work, or execute values. Production CPU and Metal providers exist for their
qualified occurrence domains; this example is not either provider and grants no general-scan
support. Package-private Compiler is the current consumer, while no public graph-wide
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
- `DerivativeGraphMetadata`, the exact graph-adjacent per-node derivative-order sidecar, including
  `scanActivityByNode()` associations from containing Scan IDs to exact local activity plans;
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

The public `backends:cpu` integration surface contains four supported types:

- `CpuCapabilityProvider`, the stateless fail-closed CPU capability provider; and
- `CpuBackendIntegration`, the closeable lifecycle SPI intended for Engine composition rather
  than ordinary application code;
- `CpuLocalWorkloadTuning`, the artifact-free Phase-1 route-candidate collaboration; and
- `CpuCompletePlanTuning`, the artifact-free Phase-2 retained-plan collaboration.

One `CpuBackendIntegration.open()` call creates the fixed profile-free CPU composition. It
reports the immutable `cpu/host` availability fact, exposes occurrence-specific capability,
one artifact-free positional `PartitionPreparation` and
one retained schedule assembler for exactly one non-empty maximal CPU-owned partition, and provides
artifact-free local and complete-plan tuning collaborations. CPU retains qualified portable,
generated, OpenBLAS and explicitly configured oneDNN routes with non-profile plan,
generated-artifact, tuning and cache identity. Engine owns the `CompileArtifacts`, obtains the exact
partition projection from `GraphPreparation`, and passes every ordinary, tuned, selected, or
fallback preparation back through `GraphPreparation`. During complete preparation, shared Prepare
identifies eligible fully static source-only published splat constants and asks the CPU assembler
for their physical geometry. The integration also borrows intrinsically compatible
`HostTensorStorage` as a non-owning Runtime buffer representation. Its
`copyToCanonicalHostBytes(representation, descriptor, maximumBytes)` operation copies one exact
current CPU publication representation into a fresh caller-owned mutable `byte[]`. This is
Engine-facing SPI, not an ordinary application result API; the caller pairs the representation
with the final descriptor from the same publication occurrence while the Runtime result lease is
open. The adapter also attempts bounded automatic OpenBLAS discovery and qualification. That
attempt may add an eligible native candidate; the portable route remains available when discovery
or qualification fails, and OpenBLAS is neither guaranteed to be present nor guaranteed to be
selected.

`CpuBackendIntegration.openOneDnn(absolutePath)` instead adds a required exact external
**oneDNN** native kernel library, defaulting to the nested `OneDnnApi.AUTO` preference. The
overload accepts `AUTO`, `PRIMITIVES` or `GRAPH`. The primitive and Graph APIs both consume
common CPU lowering; Graph does not mean automatic differentiation and node count does not
force it. A preference limits eligible native APIs, not provider selection: the cost comparison
may retain portable or OpenBLAS units. No provider priority or Runtime fallback chain is added.
The CPU backend identity and `cpu/host` availability remain the same.

Explicit opening requires an absolute path to oneDNN 3.14.0, little-endian LP64, sequential CPU
runtime (SEQ, code 1), with Graph support; the JDK Foreign Function and Memory (FFM) API needs
native-access permission. Loading/version/runtime/symbol failure throws rather than becoming
optional discovery. No oneDNN binary is bundled, and default `open()`/`Engine.standard()` behavior
is unchanged. The [CPU guide's complete configuration example](../backend-guide/cpu-backend.md#complete-public-configuration-example)
shows `Engine.builder().takeOwnership(configuredCpu).build()`, the JVM flag, exact external build
pin and resource closure.

The native F32 surface now includes qualified common arithmetic, predicates, aggregates,
coordinate-proved movement and literal MSE/BatchNorm/padded-pooling realizations, alongside bare
MATMUL with K > 1. Each API/implementation/geometry is independently admitted; the
[CPU qualification table](../backend-guide/cpu-backend.md#numerical-qualification-and-training)
defines the exact boundary. Unsupported occurrences, symbolic/Scan orchestration and opaque
convolution or fused activation replacements retain preplanned CPU execution. Literal native
pipelines preserve the existing common unit's stores and arithmetic sites; they are neither a
second graph nor a second AD system. Saved/public stores and current-weight visibility remain.

The oneDNN comparison uses conservative dimensionless whole-plan heuristics, not measured
latencies or promised speedups. Decisions account for arithmetic/remainder, control/dispatch,
preparation, memory, ingress and scratch. OpenBLAS Phase-1 decisions and selected Phase-2
topologies remain frozen; Phase-2 schema 4 includes the oneDNN policy/load-session identity and
rejects stale decisions. Selected plans are private; public timings alone cannot prove provider
selection. See the [guide's qualification, lease and benchmark details](../backend-guide/cpu-backend.md#explicit-onednn-composition).

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

The backend-local CPU complete-assembler entry remains deliberately exact. A pure zero-node
constant or pass-through graph is rejected because it requires one non-empty maximal CPU
partition; direct use also rejects mixed-owner or multiple-partition artifacts. Ordinary Engine
mixed-owner preparation instead invokes CPU through its partition finalizer and physical
contributor, then composes all owners centrally. For a backend-local accepted projection, the CPU
assembler describes creation of fresh run-owned buffers and workspaces, initialization of
projected constants, one prepared CPU execution occurrence, and publications in Prepare-supplied
order. A source-only constant gains no executable schedule step. Assembly performs none
of that physical work: `PreparedExecution` retains immutable recipes only, and creation of every
fresh `RunState` invokes each initialized-buffer recipe exactly once. Sequential or concurrent
runs therefore own distinct initialized representations.

The adapter owns any qualified OpenBLAS coordinator and explicitly loaded oneDNN library.
Callers must keep it open for preparation and new run admission. oneDNN prepared-resource leases
retain persistent kernels; each issued run workspace retains the needed kernels and library
until its own close. Stream, memory/tensor handles, ABI tables and user scratch are created by the
run workspace factory before binding, not in hot execution. Data/padding/scratch requirements are
declared before assignment; ingress copies refresh current inputs/weights once per shared source
span on every run. SEQ execution reserves one shared CPU-budget permit and waits synchronously.
Closure rejects new admission without invalidating already-issued native workspaces.
Borrowing transfers no storage ownership, and closing a borrowed wrapper does not close caller
storage. The borrow operation validates only intrinsic storage consistency and current access; it
cannot validate an expected logical input type, required span, or write role because those target
facts are absent. A successful host-byte copy likewise does not close, retain, mutate, transfer,
or change validity of its source. It is synchronous: the caller must keep the adapter, Runtime
result lease, and representation open, keep storage accessible to the calling thread, and prevent
source mutation or closure until return. Independent valid calls may run concurrently, but racing
mutation has no atomic-snapshot guarantee. Typed logical input binding, a simpler end-user execute
surface, and ordinary typed or host-materialized results remain Engine work.

The adapter does not expose CPU internals, discover backends for Runtime, run tuning policy, or
turn OpenBLAS or oneDNN into another backend identity. CPU has no production Compiler or Engine
dependency; its compiler-backed fixtures are tests only. Metal retains its independently
qualified restrictions and has no FP8 capability.

The public `modules:runtime` surface now contains five focused packages:

- `runtime.memory` defines `BufferSlot`, `WorkspaceSlot`, and immutable ordered
  `PreparedMemoryPlan` byte geometry;
- `runtime.resource` defines the nominal backend-implemented `BufferRepresentation` and
  `WorkspaceRepresentation` cleanup roles, the persistent `PreparedResource` cleanup role, plus
  immutable `PreparedRepresentationPlan` origins, caller-input occurrences, ordinary and
  initialized buffer origins, and typed buffer/workspace creators;
- `runtime.run` defines borrowed/run-owned buffer bindings, the array-backed one-run `RunState`,
  dense-coordinate `PreparedPublication`, per-run `BoundPublication`, the whole-state
  `RunResult` lease, and stateless `PreparedExecutionRunner`; and
- `runtime.execution` defines the immutable-recipe, explicitly closeable `PreparedExecution`
  owner, reusable
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

Completion and lifetime are separate: a successful synchronous run has finished its work before
returning, while the returned Engine/Runtime result may keep completed output resources and
publication state alive until close. Engine serializes materialization against outward result
close; Runtime `RunResult` and `RunState` remain non-thread-safe.

One `PreparedExecution` now supplies the current reusable Runtime root and unique persistent-
resource owner. It retains the exact memory-plan and same-plan schedule references. The
two-argument constructor remains resource-free; the three-argument constructor privately
snapshots non-null resources that are unique by exact identity. The final class intentionally
uses `Object` identity rather than the former record's structural equality and exposes no
resource aggregate or lookup.

Close is explicit, idempotent, and non-waiting. It rejects new run leases immediately and closes
resources in reverse acquisition order when no admitted run remains; the last lease performs
deferred cleanup after a close/run race. Cleanup attempts all resources, runs outside lifecycle
synchronization, and preserves the first unchecked failure with later distinct failures
suppressed. Immutable recipe sharing remains safe, while each invocation still uses a distinct
mutable `RunState`.

`PreparedExecutionRunner.run(execution, callerInputs)` first acquires the opaque execution lease,
then creates one isolated state, cold-binds
every non-creation occurrence before the first action, traverses direct bound references in
schedule order, and either returns the whole-state result lease or closes the state after failure.
The runner is stateless and may serve concurrent calls; each call remains synchronous and owns
distinct mutable state. Existing run failure remains primary over lease-cleanup failure. A
cleanup failure after successful execution closes the new result and prevents its return.

Successful public or Runtime run return is a synchronous completion barrier: all actions and native
work for that invocation have completed, or no result is returned. `RunResult` is therefore a
completed-state resource/publication lease, not a future, event, fence, or in-flight completion
handle. Concurrent callers receive isolated mutable state but gain no promise of device overlap,
cross-run order, fairness, queue topology, or throughput. The current APIs expose no asynchronous
run, cancellation, timeout, polling, callback, or completion-stage surface.

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

`PrepareContext` has six canonical record components: `partitionDag`, `values`,
`memoryRequirements`, `constants`, `backendInputs`, and `Map<NodeId, ScanActivityPlan>
scanActivityPlans`. Activity keys name containing Scan nodes in this projection and each plan
retains that occurrence's exact local body graph. Shared `GraphPreparation` projects these
associations from `artifacts.derivatives().scanActivityByNode()`; copying or subsetting contexts
preserves or filters applicable associations explicitly. Ordinary/synthetic constructors supply
an empty map, not a way to discard an annotated Scan's metadata. Its derived `partition()` and
`nodes()` views delegate to the DAG. No binary compatibility is promised for a changed record
descriptor, component reflection, equality, hash code or text form. The context exposes no
complete containing `CompiledGraphModel`, graph region, callback or Compiler-owned aggregate.

`PreparationResourceAssignment` associates each exact declaration
with an assigned Runtime slot and dense plan index. `BackendPartitionFinalization` carries one
typed analysis, the exact shared `PreparedMemoryPlan`, and assignments in declaration order to a
`BackendPartitionFinalizer`. The finalizer returns one
`BackendPartitionFinalizationResult`, which snapshots its exact-plan executable and persistent
resources in physical acquisition order. `PreparedPartition` retains only the exact planned
partition and returned `PreparedExecutable`; resource ownership is never inferred from that
association or from schedule occurrences. `PartitionPreparation` keeps each backend's typed
inputs, preparer, and finalizer positionally associated. `PreparedBufferAssignment` translates one
graph `ValueId` to its exact `BufferSlot` and dense plan index. `PreparedScheduleContext` exposes
only validated stable Model, Planning, Prepare, and Runtime facts to one explicit
`PreparedScheduleAssembler`: planned partitions, graph values, bindable source IDs, constants,
ordered publication IDs, the memory plan, prepared partitions, and buffer assignments. It
contains no Compiler aggregate.
`GraphPreparation.prepare(...)` rejects repeated resource identities, rolls successful results
back after any later preparation failure, and transfers the ordered unique resources only by
returning the exact validated `PreparedExecution`. Shared Prepare invokes cleanup but neither
looks up a resource nor performs its physical allocation.

The ordinary three-argument preparation form identifies every producerless, consumerless
published graph input, classified by the existing caller or constant source plan. It asks the
supplied assembler for a `ProducerlessPublishedValueResource`, retaining five components:
the exact stable value and logical requirement, backend owner, existing Runtime `ByteSize size`,
and byte alignment. The default geometry hook is
`producerlessPublishedValue(GraphValue, LogicalMemoryRequirement, Optional<ScalarValue>)`;
the scalar is present only for an authenticated constant, empty only for a caller. No payload
is read or representation allocated by this contribution. The four-argument form accepts the
complete explicit contribution list and uses the same exact-reference, role and coverage checks.

Source-only callers receive real slots in Compiler bindable occurrence order, before remaining
ordinary declarations; source-only constants remain last in final graph-value order. Neither
source receives a partition-local analysis or finalizer assignment. Caller payloads remain borrowed;
constants retain existing fresh per-run initialization. Static affine layouts use Model referenced
span; named templates use checked existing binding recipes, including structurally symbolic
zero-byte geometry. Complete source facts participate in explicit batch mode. A genuinely fixed
CPU unit within that complete batch plan authenticates its fixed selected geometry without changing
its lowering or relaxing Runtime defaults.

`PrepareContext.validateValueShapes(List<GraphValue>)` shares the existing fixed-or-sole-named-B
shape proof and returns the common symbol, if any. It does not establish caller-axis authority,
ROW provenance or backend admission.

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
publication coverage before constructing the prepared root. Ordinary Engine composes this
operation from all exact partition owners, backend physical contributions, and explicit transfer
recipes captured during cold preparation.

## Current ordinary, explicit-composition, and advanced lifecycle

`Engine.standard()` remains the ordinary CPU-only convenience. Every invocation opens one fresh
CPU integration, transfers ownership through `Engine.Builder`, and returns a distinct `Engine`.
Callers that need explicit composition use `Engine.builder()` and transfer opened CPU or Metal
integrations. There is no discovery, service lookup, process-global Engine, implicit fallback, or
runtime owner substitution. `isClosed()` observes when delegated closure begins, and `close()`
uses the advanced owner's thread-safe, idempotent, failure-retaining cleanup protocol.

The ordinary reusable lifecycle is:

```text
ordered Tensor output expressions -> CompiledGraph -> engine.prepare -> PreparedExecution
prepared + exact-ID Tensors in any order -> engine.run       -> RunResult
prepared + fresh values in compiled order -> engine.runValues -> RunResult
exact publication occurrence -> materialize -> detached HostTensorValue
```

`compile(List<Tensor>)` creates a forward-only handle. The three-list overload accepts ordered
outputs, explicit output-aligned cotangent seeds and identity-unique ordered gradient targets.
It infers neither seeds nor targets. `CompiledGraph.inputs()` is the final input membership and
Compiler occurrence-order authority, retaining descriptors and IDs, not caller storage.

`prepare(graph)` accepts a graph from that exact Engine, performs one preparation, and registers
one closeable outward `PreparedExecution` owner. Its `compiledGraph()` returns the exact graph.
The handle borrows the Engine's composition lifetime; Runtime owns persistent resources and the
unique run-lease protocol. Failed preparation, wrapper construction or publication rolls back
acquired resources while preserving the primary failure and suppressed cleanup failures.

`run(prepared, inputs)` matches every compiled TensorId exactly once in arbitrary order and checks
the complete descriptor. `runValues(prepared, values)` instead binds by final compiled input order,
with no ID requirement. It requires exact count and fully static exact descriptors: dtype, shape,
layout presence/geometry and requiresGrad. One fresh Tensor may fill distinct matching slots.
There is no shape specialization: [64,784] rejects a compiled [32,784] slot.

The qualified CPU-only exception is an explicitly symbolic ordinary or bounded all-valid Scan graph, not a static graph
silently specialized at run time. Compile/prepare once with FLOAT32 X `[B,F]` and fixed W `[F,H]`;
`compile(List.of(X.matmul(W).mean()), List.of(scalarOne), List.of(X,W))` supports actual first-order
gradients at changing B through existing `runValues`. Actual inputs retain type/gradient flags and
fixed axes; symbolic inputs require present zero-offset canonical dense layouts (dense read views
qualify). Compiler binds B and predicates; Engine checks logical counts/byte prefixes/storage
before borrowing. Runtime receives same-plan numeric geometry and actual publication descriptors
follow generated values, including dX `[B,F]` and fixed dW `[F,H]`.
At B=0, empty MEAN is NaN, dX is empty and nonempty dW is positive zero; the whole run is not skipped.
Slots/routes/code do not change, and different-B runs own independent writable state.
See the [complete public example](../user-guide/running-models.md#ordinary-cpu-execution-with-changing-batch-size).
Current symbolic Scan composes ordinary CPU-supported PURE units with fixed time/features/state
widths and the same ordinary dtype, layout and attribute guards. Each complete forward and
selected ordinary-VJP reverse body qualifies independently; source validity alone grants no
CPU or derivative support. BFLOAT16, FLOAT16 and both FP8 types reuse these same CPU owners
with FLOAT32 working arithmetic and declared logical narrowing. The same public lifecycle supplies actual
canonical final/emission/residual descriptors and fixed parameter-shaped gradients; see the
[Scan changing-B example](../user-guide/running-models.md#symbolic-scan-with-changing-batch-size).
Final qualification is tracked in [Prepare0014](../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md).
This does not enable NN Model/Training reuse, symbolic cold loading or wire formats, native,
mixed-owner, tuning, vector or parallel dynamic execution.

Both calls snapshot each selected host-storage association once and validate all inputs before
any borrow: presence, storage dtype, capacity, liveness and current-thread accessibility.
runValues uses checked referenced element/byte spans for resolved layouts and checked canonical
static geometry for absent layouts, rejecting unknown/overflow geometry. That extra absent-layout
check does not change identity-based run. The snapshot neither copies nor pins bytes and is not
atomic across inputs. The caller must keep storage usable and free from conflicting mutation until
result close; replacing a Tensor association later cannot redirect an existing run.

Both methods first admit the Engine, then validate prepared nullness/exact ownership, then
retrieve/check its exact inward delegate, then inspect inputs. This deliberately changes former
direct-run diagnostic precedence: closed preparation wins null/malformed/dead inputs with
`prepared execution is closed` and zero borrows. Null/foreign prepared handles also precede
input-list errors. Closed Engine still reports `advanced engine is closed` before all arguments.
A later close race is decided only by Runtime's unique nonwaiting lease; no outward monitor is
held through execution.

A successful run is a synchronous completion barrier, not a future or in-flight command.
`RunResult` owns completed run-state resources and the borrowed input wrappers. Its immutable
publication list is dense forward-then-gradient order. Gradient metadata identifies the requested
target, derivative order and target index; aliased values remain distinct occurrence objects.
`materialize` authenticates the exact occurrence from that result and returns a fresh detached
`HostTensorValue`, without caching or deduplication.

Repeated/concurrent runs reuse immutable recipes, precomputed geometry and direct input/publication
adapters, with isolated mutable Runtime state and results. They do not mutate prototypes, construct
synthetic Tensors, recapture, compile, prepare, query providers/registries, or add per-element binding
work. There is no device-overlap, throughput, fairness, asynchronous or cancellation promise.

Prepared close rejects new runs but does not close returned results. A leased run may finish;
Runtime defers prepared-resource cleanup to its last lease release without making close wait.
Result materialization remains valid after prepared close while result and Engine are open.
Engine shutdown rejects new work, waits for admitted operations, then closes results in reverse
run-publication order, ordinary/advanced preparations in reverse preparation-publication order,
and integrations in reverse registration order. Cleanup is attempt-all and failure-retaining.
Metadata stays readable after closure; detached values outlive result, preparation, Engine and
caller storage.

`InferenceSession` and `Engine.session` are removed without a compatibility shim. Migrate
`engine.session(graph)` to `engine.prepare(graph)` and `session.run(inputs)` to
`engine.run(prepared, inputs)`, then recompile. Fresh values explicitly opt into runValues.
`TrainingSession` and the advanced lifecycle remain; this is not a Training API redesign.
See the [fresh-value example](../user-guide/running-models.md#run-fresh-values-in-compiled-order).

Optional `prepareTuned(...)` is a cache-first preparation path for eligible single-owner CPU plans
and exact singleton-NEG Metal plans. This CPU example uses a `CONTIGUOUS` expression, which has no
eligible local tuning handoff in the current standard composition, so the allowed policy
deterministically demonstrates explicit safe fallback:

```java
import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import io.github.pho001.synaptik.engine.CompiledGraph;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.ModelAutotuningPreparation;
import io.github.pho001.synaptik.engine.ModelAutotuningRequest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
    Shape shape = Shape.of(2);
    TensorDescriptor descriptor = new TensorDescriptor(
            DataType.FLOAT32, shape,
            Optional.of(LayoutDescriptor.contiguous(shape)), false);
    MemorySegment storage = arena.allocate(2L * Float.BYTES, Float.BYTES);
    storage.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0, 1.0f);
    storage.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 1, 2.0f);
    Tensor input = TensorFactory.create(descriptor, Optional.empty(), Optional.of(
            new MemorySegmentStorage(DataType.FLOAT32, 2, storage)));
    CompiledGraph graph = engine.compile(List.of(input.contiguous()));

    ModelAutotuningConfig config = new ModelAutotuningConfig(
            ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS,
            new ModelAutotuningConfig.Budget(1, 2, 1, 3),
            new ModelAutotuningConfig.RepresentativeProfileIdentity(
                    1, new byte[] {0x01}),
            ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC,
            Path.of("build", "example-workload-cache.bin"),
            new ModelAutotuningConfig.CompletePlanBudget(1, 0, 1, 1L, 0L),
            Path.of("build", "unused-model-plan-cache.bin"));
    ModelAutotuningRequest request = new ModelAutotuningRequest(
            config,
            new ModelAutotuningRequest.ModelIdentity(1, new byte[] {0x42}),
            List.of(input));

    ModelAutotuningPreparation preparation = engine.prepareTuned(graph, request);
    assert preparation.outcome()
            == ModelAutotuningPreparation.Outcome.SAFE_HEURISTIC_FALLBACK;
    assert preparation.evidence().isEmpty();
    try (var prepared = preparation.preparedExecution();
            var result = engine.run(prepared, List.of(input))) {
        assert result.publications().size() == 1;
    }
}
```

The request snapshots the list and identity bytes but retains the live caller-owned Tensor and
storage. Phase 1 reuses or measures the sole eligible local workload. Its authenticated winner is
then fixed while Phase 2 considers the backend-owned complete candidates. For CPU those are the
retained complete topology and representation alternatives; for Metal there is one complete
candidate fixed to the Phase-1-selected route. Phase 2 performs every exact correctness action
before timing; correctness, warmup, and timed executions each use a fresh preparation and fresh
Runtime state. After winner authentication and representative cleanup, the returned production
handle is prepared afresh. Every trial closes its result before its temporary preparation on every
path. The `ModelAutotuningPreparation` metadata carrier is not another closeable owner: its
contained `PreparedExecution` alone owns the production preparation.

A tuned result has outcome `TUNED`. Its immutable evidence contains the Phase-1 workload rows and
required `completePlan` evidence: compatibility and reuse scope, the exact Phase-2 budget, source,
winner identity and summary, plus one correctness-and-sample row per measured candidate. A future
persistent hit has no candidate rows because compact reusable cache records do not contain rich
per-call correctness actions or raw samples. Current CPU and Metal complete-plan evidence is
always `SESSION` and measured, and its explicit model-plan path receives no filesystem access.
Metal workload evidence is also always `SESSION` and measured. Model and representative-profile
identities are caller-defined tuning-evidence labels, not a numerical selector or workload-cache
key. Current scope excludes multiple
workloads or occurrences, Compiler graph alternatives, Planning owner or partition alternatives,
multiple partitions, and mixed backends.

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
occurrence in requested order, and closes its temporary result before its temporary preparation
before the call returns. The ordered form
returns an immutable list of immutable detached values; distinct occurrences are copied
independently even if they select one inward representation. The limit covers returned canonical
payload bytes only, not inputs, recipes, Runtime buffers or workspaces, object overhead, defensive
copies, peak memory, or other allocation.

One Engine admission covers that complete sequence. Engine closure waits for an admitted call,
including its result cleanup; a call that loses admission observes the closed-Engine failure
before argument validation. A failure returns no partial result. Temporary result-then-preparation
cleanup still runs, with distinct cleanup failures suppressed on the primary failure in cleanup
order. Selected leaf storage remains
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
`compile -> prepare -> engine.run -> materialize` for reusable execution, explicit cotangent
seeds, or multiple forward outputs. Direct `prepare -> run` remains available when the prepared
handle itself must be owned. Use `AdvancedEngine.compile(...)` with a
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
arena itself stayed caller-owned. Every call compiles and prepares afresh. Use
`compile -> prepare -> engine.run -> materialize` when a prepared recipe should be reused or only
selected publications should be copied. The direct prepared-handle lifecycle remains available
when its lower-level ownership is specifically required.

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

An old generic/custom/declarative NN `Model` follows the same boundary. This conceptual example intentionally assumes only the
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
execution should retain the expression and use
`compile -> prepare -> engine.run(explicit inputs)` rather than paying the one-shot compile and
prepare cost on every call.

### Owned tensor Model capability

`Model<I,O>` remains generic and now implements `AutoCloseable`. Its fixed tensor methods are
independent of I/O: compute returns `HostTensorValue` or `List<HostTensorValue>`, and train returns
`TrainingStep`, never generic O. The fluent builder's nonempty-chain `build()`,
`defineTensor(signature, definition)` and `load(path[, limits])`
return `Model<Tensor,Tensor>`; builder `buildWithLengths()`, `graph(inputs, outputs, namedLeaves)` and `loadGraph(path[, limits])`
return `Model<List<Tensor>,List<Tensor>>`. Definition/graph/load factories have explicit borrowed-Engine
overloads (load forms take path, limits, Engine); built Models select it through `withEngine`.
Old generic factories and flexible v1 readers remain separate.

`withEngine(engine)` returns the same receiver and must precede the first valid compute/configure
setup attempt, including failed setup. Without a selection, execution lazily opens a model-owned
CPU-only standard Engine. Construction/load/save/cold forward open none. List compute additionally
accepts `maximumTotalBytes`; all publications are checked before materialization, and RunResult
closes before detached values return. Configuration takes a target descriptor and scalar objective
builder, or target prototypes and a list builder, plus an Optimizer. It succeeds once; train delegates
`RESET_AND_STEP`. Every declared input/target requires the exact static descriptor even if unused.

`compute(firstInput, secondInput)` returns the sole detached output of any owned Model with exactly
two declared inputs and one output, including ordinary graphs and loaded Models. It binds positions,
not inferred dtype or length roles, and uses the same cached execution as list compute with
`Long.MAX_VALUE`. Capability/lifecycle admission precedes arity, then arguments. Cold `forward`
is optional, never a prerequisite. Fixed and general recurrent sources use the existing configured
TrainingSession and Compiler first-order AD when their actual forward/reverse bodies qualify.
Mixed fixed/general sources share one configured session. Objectives still require a scalar,
gradient-eligible FLOAT32/FLOAT64 result; low-output objectives explicitly cast to FLOAT32/FLOAT64.
CPU low execution retains the existing family/geometry and actual derivative guards and setup/lifecycle ordering.

`Model.graph` and `defineTensor` also acquire bounded all-valid general-scan bodies through Checkpoint and
Compiler's [cold constructor inventory](../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction).
This does not add a ModelBuilder recipe. Cold admission preserves applicable Model dtypes, including
low TANH bodies; it is not arbitrary PURE-body support. Qualified static floating occurrences,
including BFLOAT16, FLOAT16 and both FP8 types, use the same compute/first-order Training owners
when their actual forward/reverse bodies and binders qualify.
All-valid means every static time step participates, with no lengths or padding mask.
Acquire private state once, configure once, and pass fresh exact-descriptor batches to the same
prepared/session owners. Connected nonempty parameter admission still applies at T=0; neither
fixed nor general recurrence bypasses the selected backend's actual derivative qualification.
Non-FP8 bounded all-valid general-body save/load uses
[v4](../architecture/contracts/extensions-training.md#canonical-graph-model-version-4);
the static qualified per-role floating length intersection uses
[v5](../architecture/contracts/extensions-training.md#canonical-graph-model-version-5),
with Pool3d selecting v6 as needed. FP8 anywhere in the encoded source selects v7 through the
same Model/GraphCheckpoint owners. Old-only sources retain their smallest sufficient exact
format and bytes. V1/v2/v3 remain body-free. Unsupported saves reject
before file effects without freezing Engine selection or disabling compute/train. Load restores
source/current raw values, not objective/optimizer history or execution plans.
See the
[complete general-scan workflow](../user-guide/running-models.md#train-an-all-valid-general-scan).

Model's separate [length-aware Scan source API](tensor-api.md#length-aware-source-only-construction)
constructs bounded source metadata and structural IR only. Its final INT64 lengths actual has no
body formal. Compiler independently supports length-mode inference, bounded first-order reverse
construction and final role/demand queries; original-source AD requires every carry and every
original sequence to be ROW(0), including unused/non-gradient positions. Generated reverse inputs
may retain explicit mixed roles. Prepare0011 transports those exact bindings and demand through
the existing staged resource handoff after static boundary validation; it reads no length values.
This does not enable owned-Model acquisition, compute/train or save/load: cold capture/
reconstruction and encoding still explicitly reject length mode. CPU0042 separately qualifies
direct fixed-geometry FLOAT32/FLOAT64 Scan execution through Engine, not owned-Model acquisition
or migration.

Unsupported generic/custom/declarative tensor methods throw `UnsupportedOperationException` before
file/Engine work; their close is a no-op. Owned operations reject busy before inspecting arguments.
Source mutation invalidates the capability without recapture; close waits and attempts all owner
cleanup with retained failure replay, never closing a borrowed Engine. See the complete
[owned model example](../user-guide/running-models.md#owned-static-model-workflow) and
[owning contract](../architecture/contracts/extensions-training.md#owned-tensor-models).

Graph-model v3 uses these same methods and no new public type. The
[exact source slice](../architecture/contracts/compiler-autograd.md#compile-lifecycle) admits
selected narrow transforms/indexing, biased default convolutions, builder pooling and fixed
RNN_TANH/GRU_RESET_AFTER/LSTM scans. It is not all-family serialization. Advanced `graph`/`loadGraph`
may declare sequence, non-gradient INT64 runtime lengths and initial states as ordered external
inputs for cold persistence/rebind. Prototypes are descriptor-only; their bytes are neither read
nor saved. A scan remains one multi-output producer with canonical sibling identities, even when
only a final-state port is a root. Cold persistence executes nothing; owned fixed-scan compute separately
uses public Engine's qualified canonical FLOAT32 CPU forward route. Fixed-scan training and
backpropagation through time (BPTT) remain unsupported. Persistence adds no execution capability
or optimizer resume. Existing default biased Conv3d can train after reload in its
[qualified backend domain](../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary)
with a fresh training configuration. Single `load` still rejects multiple boundaries.

### NN runtime-length recurrent composition

`RnnSequence` and `GruSequence` add
`forwardScan(input, validLengths, initialHidden, direction)` returning Model's
`RecurrentScanResult`. `LstmSequence` adds
`forwardScan(input, validLengths, initialHidden, initialCell, direction)` returning
`LstmRecurrentScanResult`. These are expression constructors using the same owned cell and
`cell.*` parameter paths, not execution methods or replacement static APIs.

Input is fully static `[T,B,I]`, lengths are non-gradient `INT64[B]`, states are `[B,H]`, and
all floating roles match the configured cell type exactly. `I/H` are positive; `T/B` may be zero.
Direction selects forward or reverse traversal of each valid prefix. The one canonical producer
returns dense original-time-aligned `[T,B,H]` outputs, final hidden and, for LSTM, final cell.
Padding is positive zero; zero-length rows preserve initial state values semantically. No length
values are read during construction, and no state is retained by the module.

Complete descriptor checks precede automatic initialization. Even empty scans bind the existing
reserved parameter group through the same cell-local gate; already-bound/strict-loaded cells do
not reinitialize. Provenance and gradient flags remain intact in train and eval. Callers coordinate
replacement; old occurrences keep old bindings. Existing Java `long[]` `forward` overloads,
compact result carriers, static training and builder recurrence remain unchanged.

Execute selected scan outputs separately through public Engine compile/prepare/runValues with
fresh exact descriptors in compiled input order. Fixed CPU scalar forward supports its declared
homogeneous floating types, including all four low types, and resolved nonnegative read domain.
Selected first-order AD uses the shared Compiler closure and qualifies actual reverse bodies independently.
Owned Models can acquire fixed expressions through `graph` or retained fixed builder recipes,
while unsupported occurrences still reject independently. No implicit state/default-length
overload or bidirectional migration is added. See the
[NN recurrent contract](training-api.md#current-nn-recurrent-composition-contract) and
[scan construction example](../user-guide/running-models.md#build-a-fluent-model).
BFLOAT16, FLOAT16 and both FP8 fixed implementations execute through the existing CPU owner,
with FLOAT32 working sites and declared state narrowing under the same geometry/derivative guards.

`RnnSequence.forwardGeneralScan(input, validLengths, initialHidden, direction)` separately composes
ordinary general Scan for homogeneous configured floating types, including both FP8 types. It reuses the same cell
initialization and equation, including bias order, with static T/B >= 0 and positive I/H.
Complete actual/formal/emission/latent geometry checks precede parameter publication or body IDs.
Canonical general outputs are emission port 1 and final hidden carry port 0; the result carrier's
component names are semantic, not universal producer indices. Fixed `forwardScan` is unchanged.
Qualified CPU reads use dense zero-offset geometry or existing proved scalar-broadcast exceptions,
not every fixed-scan read layout. No copy, conversion or fallback repairs unsupported actuals.
All four low types preserve ordinary low rounding at each projection, bias/ADD and TANH result.
That is a different represented program from fixed low RNN's FLOAT32 working sites followed
by state narrowing; low builder `rnnScan` therefore deliberately keeps its fixed semantics.
Low source acquisition/rebind/save/load does not execute CPU low arithmetic. Metal execution
and actual first-order closure remain backend-qualified; the scalar Training objective still
requires an explicit FLOAT32/FLOAT64 cast when derived from low outputs.
NN0033 qualifies first-order compute/train and v5 save/close/load/new training, with fresh lengths;
see the [complete RNN journey](../user-guide/running-models.md#train-a-runtime-length-rnn).


### Fluent Model builder

`Model.builder()` returns a fresh public final `ModelBuilder`, with no public constructor or
callback/module extension point. Set `input(DataType,long...)` once before steps. Configuration
returns the same caller-coordinated builder; `build()` returns a new closeable `Model<Tensor,Tensor>`
owning a frozen Sequential under `layers`. Every build owns fresh automatic state and privately
copies supplied leaves once by identity before expressions, retaining aliases but no caller storage
or recipes. Build requires a floating positive-static rank-at-least-one input and a nonempty chain.
The input prototype is storage-free, canonical contiguous, offset-zero, non-view, requiresGrad=false;
compute/cold forward match its complete descriptor, with no implicit dtype or batch-shape conversion.

Supported steps are `linear(outFeatures[,bias])`, `linear(weight[,bias])`, `relu/tanh/sigmoid`,
`conv1d/conv2d/conv3d`, `maxPool2d`, `flatten`, static `rnn/gru/lstm`, `lastTimeStep`,
runtime-length `rnnScan/gruScan/lstmScan(long hiddenSize)` and `finalHidden()`.
Automatic layers infer input widths, use Glorot uniform, and enable exact zero bias by default.
Default seed is 0; the last `seed(long)` applies to the whole build. Appended step i, including
stateless steps, derives `seed + 0x9E3779B97F4A7C15L * i` with Java long wraparound and a fresh
L64X128MixRandom generator per initialized layer. Supplied Linear leaves must be stored,
provenance-free, floating, gradient-eligible and match input dtype and existing Linear shape guards.
Build reads but never mutates caller state; unavailable or produced supplied state rejects.

Convolutions use NCW/NCHW/NCDHW, stride/dilation/groups 1, padding 0 and bias true. Pool stride
equals kernel, padding 0, dilation 1, floor geometry. Flatten preserves axis zero and checks the
remaining product: CNN `[B,C,H,W]` becomes `[B,C*H*W]`, TBF `[T,B,H]` becomes `[T,B*H]`.
Existing static `rnn/gru/lstm` steps use fixed all-valid TBF, start from zero h (and c for LSTM),
and return every hidden time step. Static stacks preserve TBF. Explicit `lastTimeStep()` selects
`[B,H]`; use it before flatten when batch rows must remain separate. Cached runs carry no previous
final state.

This compilable example constructs an MLP, CNN and recurrent stack; inputs/results are noted inline:

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.module.Model;

public final class FluentBuilderShapesExample {
    public static Model<Tensor, Tensor> mlp() {
        return Model.builder().input(DataType.FLOAT32, 32, 784).seed(42)
                .linear(128).relu().linear(10, false).build(); // [32,10]
    }
    public static Model<Tensor, Tensor> cnn() {
        return Model.builder().input(DataType.FLOAT32, 32, 1, 28, 28).seed(42)
                .conv2d(32, 3, 3).relu().maxPool2d(2, 2)
                .conv2d(64, 3, 3).relu().maxPool2d(2, 2)
                .flatten().linear(128).relu().linear(10).build(); // [32,10]
    }
    public static Model<Tensor, Tensor> recurrent() {
        return Model.builder().input(DataType.FLOAT32, 20, 32, 8).seed(42)
                .lstm(128).lstm(64).lastTimeStep().linear(10).build(); // [32,10]
    }
}
```

Callers close each returned Model. CNN spatial extents are 28→26→13→11→5, so flatten width is
`64*5*5=1600`; the LSTMs produce `[20,32,128]`, then `[20,32,64]` before explicit selection.
See the complete [compute/train/save example](../user-guide/running-models.md#build-a-fluent-model).

Construction and cold forward create no Engine. Compute/configured training consume fixed outputs
and cached bindings, never recipes per batch. They are independent of serialization admission.
Graph-model v3 enables saves for the selected default CNN/static recurrent DAGs;
other unsupported saves still fail before file work without freezing Engine selection or
invalidating execution. Only sources wholly within the original five/eight-kind slices retain
byte-identical graph-model.v1/v2 saves; any selected added body-free kind chooses v3, including CNN+RELU.
See the [single owning wire contract](../architecture/contracts/extensions-training.md#canonical-graph-model-version-3).
Execution still requires actual Engine capability;
static unrolling adds no scan BPTT, runtime lengths or streaming state. Conv3d separately uses
the shared grouped convolution adjoints within its qualified backend domain.

For runtime lengths, append `rnnScan`, `gruScan` or `lstmScan` with a positive hidden width and end
with `buildWithLengths()`, returning `Model<List<Tensor>,List<Tensor>>`. The two inputs are
`[values, validLengths]`, with one chain result. Each recipe uses a biased FORWARD scan, existing
initialization/seeds, independent parameters and zero hidden state (also zero cell for LSTM).
There is no hidden carry across batches. The first scan requires positive static TBF `[T,B,F]`
and establishes shared canonical, non-gradient `INT64[B]` lengths; later scans preserve exact T/B
and infer F. An ordinary prefix establishes this interpretation, not necessarily the original input
axes. Runtime length values must lie in `[0,T]`, including zero. Advanced graph/list workflows retain
explicit states, finalCell, reverse/bias-free variants and zero T/B; the builder remains positive-only.

`finalHidden()` must immediately follow a new scan recipe, with no intervening transform, static
recurrence or repeated selection. It selects the existing `[B,H]` port after each row's actual last
transition; length zero returns initial hidden state (zero here), not `select(length-1)`. It counts
as an appended seed step. Append an ordinary classifier head afterwards.
`build()` rejects scans and `buildWithLengths()` rejects no-scan chains before acquisition effects;
both snapshot recipes and create fresh private state.

FLOAT32/FLOAT64 `rnnScan` uses the general composition above, supporting qualified first-order
training and v5 persistence with canonical source outputs. Retained low RNN, GRU/LSTM and direct
fixed expressions remain fixed recipes with their own working/publication semantics. Qualified
first-order configured training now uses their existing Compiler AD without rewriting them as
general Scan. Old v3 fixed files remain fixed after load; fresh training does not restore optimizer
history. Low execution still requires a supporting backend and explicit FLOAT32/FLOAT64 objective.


There is no model-wide mask: bias and sigmoid may change padded zeros; convolution/pooling keep
dense channels-first semantics, flatten preserves axis zero, static recurrence ignores lengths, and
`lastTimeStep()` still selects `T-1`, not final hidden. Each scan ignores its own suffix, not arbitrary
ordinary results. Shape-valid construction or save does not guarantee backend execution. General
F32/F64 RNN uses its narrower dense zero-offset CPU read qualification; fixed recipes retain
their separately declared read domain. No descriptor is erased and no layout copy is inserted.
See the [runtime-length inference example](../user-guide/running-models.md#run-a-runtime-length-scan-model)
and [training journey](../user-guide/running-models.md#train-a-runtime-length-rnn).

### Advanced Engine lifecycle

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
`AdvancedPreparedExecution` is another immutable owner-bound, closeable handle. It owns exactly
one inward Runtime preparation and may be shared by concurrent callers because every `run(...)`
creates isolated mutable Runtime state. Callers should close it when reuse ends; closing it does
not close an already returned `AdvancedRunResult`.

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
and waits for admitted work. Engine then closes still-open results in reverse run-publication
order, retained prepared handles in reverse prepare-publication order, and its owned CPU
integration. Prepared handle close uses the same outward synchronization and inward Runtime lease
arbitration described for ordinary handles; it never holds the wrapper monitor over a complete
run. Concurrent/repeated close callers replay the first immediate wrapper-cleanup failure by
identity; a deferred resource failure belongs to the lease-releasing run path.

This advanced surface does not provide mixed-backend composition, backend discovery, typed
logical input binding, typed publication metadata, host materialization, public one-shot
execution, backward convenience, tuning integration, or ordinary Engine exception translation. Supplying a
gradient request to `compile(...)` invokes the existing compiler mode/request contract; it does
not add those later conveniences or guarantee that the current CPU-only prepare step can lower
the resulting artifact.

## Current explicit CPU and Metal composition

The public composition shape is:

```java
MetalBackendConfiguration metalConfiguration =
        new MetalBackendConfiguration(nativeLibrary.toAbsolutePath());
try (Engine.Builder builder = Engine.builder()) {
    builder.takeOwnership(CpuBackendIntegration.open());
    builder.takeOwnership(MetalBackendIntegration.open(metalConfiguration));
    try (Engine engine = builder.build()) {
        CompiledGraph graph = engine.compile(List.of(output));
        try (PreparedExecution prepared = engine.prepare(graph);
                RunResult result = engine.run(prepared, List.of(input))) {
            HostTensorValue value =
                    result.materialize(result.publications().getFirst(), maximumBytes);
        }
    }
}
```

`Engine.Builder` is a public single-use `AutoCloseable` construction owner. Its concrete
`takeOwnership(CpuBackendIntegration)` and
`takeOwnership(MetalBackendIntegration)` overloads transfer a non-null integration at method
entry, including when later registration validation fails. Callers must not close or reuse an
integration after invoking the overload. A successful `build()` atomically transfers the ordered
backend set to the returned Engine. Builder close and failed build release accepted integrations
in reverse registration order; failed build is terminal.

Because the overload parameters are backend-owned public types, Engine publishes CPU and Metal as
API-visible dependencies. Backend types remain confined to explicit construction and do not appear
in compile, prepare, run, handle, or result signatures.

Registration captures one `BackendId`, stable capability provider, and immutable point-in-time
availability snapshot. Their backend IDs must be equal, and an equal duplicate ID is rejected
without replacing the first entry. The snapshot remains fixed for that Engine's lifetime.
Registration order supplies deterministic compile-time Planning input; it is not a runtime
fallback order.

Compilation chooses owner identities from that fixed inventory. Cold preparation requires a
non-empty plan, resolves every distinct partition owner, and validates every required directed
cross-owner transfer before backend analysis. A missing owner, zero-partition plan, or unsupported
transfer fails without implicit CPU fallback, retry, owner substitution, or conversion.

Shared Prepare analyzes and finalizes partitions in plan order, assigns one logical slot and one
deterministic representation position per participating owner, and obtains exact backend physical
creators. Engine assembles execution occurrences in partition order, inserting one immutable
transfer per logical value and distinct destination owner immediately before that owner's first
consumer, then appends the dense publication suffix. Current CPU/Metal transfer is direct native
host-staged in either direction for all seven carriers over exact rank-0..16 static canonical or
positive-stride non-overlapping layouts with checked physical spans.

One outward `PreparedExecution` retains direct non-owning adapters in caller-input and publication occurrence order
beside the inward Runtime execution. Run uses each captured input adapter for caller-host ingress,
and the result uses each publication's captured adapter for backend-owned host materialization.
These outer calls do not re-query the registry. Engine shutdown closes retained results first,
then ordinary and advanced prepared handles in reverse preparation-publication
order, then integrations. Closing a prepared owner rejects later runs without closing an
already returned result; an already leased Runtime run may finish, and its final lease release
performs deferred prepared-resource cleanup.

`MetalBackendConfiguration` and `MetalBackendIntegration` are public Metal-owned types. Metal
validates and snapshots the explicit absolute native-library path and acquires one system-default-
device native context and command queue before the integration is transferred. The configuration
has no physical-device selector. Engine neither parses the native-library path nor owns a duplicate
Metal configuration. The integration supplies partition preparation, retained
`localWorkloadTuning()` and `completePlanTuning()` collaborations, physical creation contribution,
exact transfer endpoints, host ingress, materialization, execution, and close for that exact
context in addition to capability.

One Engine can own at most one Metal integration because its registry rejects a duplicate Metal
`BackendId`. `BackendDeviceId(metal, "default")` is the sole abstract availability slot, not a
stable physical-device fingerprint or selector; Planning proves eligibility and retains the
backend owner rather than a selected Metal device. Separate Engines may open separate default
contexts, but the API provides no enumeration, explicit selection, multi-device scheduling,
cross-device migration/coherence, failover, or hot-plug behavior.

`MetalTraceObserver` is a public functional interface whose sole abstract method receives
`TraceEvent<? extends TracePayload>`. Passing a non-null caller-owned thread-safe observer to
`MetalBackendIntegration.open(configuration, observer)` enables one integration-local trace
stream. Metal retains but never closes the observer; callbacks may be concurrent. The ordinary
one-argument open overload remains the no-trace path and performs no trace allocation, ID, clock,
or callback work.

The traced path reports one final `PREPARE` outcome per enabled prepared unit and one `RUN` outcome
per enabled native invocation. Stream-local event, prepared-unit, and invocation IDs use
independent non-negative sequences; backend/device correlations are both zero. Preparation reports
the neutral custom-kernel or graph-executable route and `NOT_QUERIED`; invocation
reports the same immutable facts and exact mapped native status when one exists. An observer
`RuntimeException`, ID exhaustion, or trace-object construction failure disables later tracing
without changing backend behavior or outward exceptions. An observer `Error` from success
reporting propagates normally; during failure reporting, the existing finalization or run failure
remains primary and receives a distinct acyclic observer `Error` as a suppressed failure without
losing earlier rollback suppression. Events contain only the bounded trace DTO fields and expose
no native path, handle, Tensor value, shape, exception, free-form string, or generic map.

For a low-precision prepared unit, `LowPrecisionTraceMetadata` identifies the selected custom
route and ordered feed-then-target logical dtype tuple. It has no numerical-profile, accumulator
or working-type field; the Model's FLOAT32 low-arithmetic rule is a separate semantic guarantee.
The DTO contains no hypothetical candidate, certificate, accuracy, determinism, or environment
state. Metal exposes no generic trace viewer, serializer, or exporter, so callers consume these
typed DTOs directly through `MetalTraceObserver`.

Metal retains a custom route for an eligible singleton `NEG`. A supported
partition containing any Task-0052 node, exact custom movement/replacement/fold/aggregate node,
non-direct MATMUL, ordering/top-K/arg-extrema node, or Task-0064 convolution/pooling node uses the
fixed shared custom whole-program resource with declared run-owned buffers and one synchronous
Java/native invocation. Existing MPSGraph-compatible nodes may remain nested in that recipe; there
is no per-node fallback. Other supported partitions use one typed whole-partition MPSGraph
executable. The general qualified FLOAT32 MATMUL domain also executes Compiler-generated
explicitly seeded gradients for both operands through authenticated local last-two-axis
transposes. Primitive-closed static FLOAT32 Conv2d/Conv3d graphs execute selected or joint
input, weight and bias cotangents through the shared convolution rule. FOLD2D/FOLD3D
overlap-add is FLOAT32-only, deterministic and output-owned, with no atomics, scratch or host
fallback/repair; FLOAT64/BFLOAT16/FLOAT16 image folds and FOLD_AXIS retain non-overlap.
Pooling gradients close only where every generated occurrence is admitted; their low image
folds remain non-overlapping. Attention and convolution transpose still reject before preparation.
See the [production support matrix](../backend-guide/production-support.md#metal-production-target)
for actual GPU Conv3d all-role and NN training/persistence evidence. No implicit seeding,
Metal Scan support, software FLOAT64, or unrestricted training support follows.

`MetalLocalWorkloadTuning` publicly exposes opaque candidates, session compatibility, identities,
decision encode/decode, and trial/selected `PartitionPreparation` for the exact two-route
singleton-NEG batch. `MetalCompletePlanTuning` authenticates that Phase-1 decision and exposes the
same opaque operations for exactly one fixed-route complete-plan candidate. Both are retained
integration collaborations, reject foreign/stale/closed use, and perform no execution,
measurement, selection, fallback, or cache input/output themselves. Public `prepareTuned(...)`
adapts them only for one Metal partition; MPSGraph-only, multiple-partition, and mixed-owner plans
remain ineligible. An allowed safe fallback stays with the selected adapter and never changes
owner.

`Engine.standard()` remains the CPU-only convenience and uses this builder ownership path
internally. `AdvancedEngine.takeOwnership(CpuBackendIntegration)` remains CPU-only; explicit
multi-backend construction belongs to ordinary `Engine.Builder`.

See the
[authoritative Engine composition contract](../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition),
[ADR 0015](../design/decisions/0015-explicit-engine-backend-composition.md), and
[ADR 0016](../design/decisions/0016-cpu-metal-mixed-owner-schedule.md).

## Current CPU limitations and planned conveniences

The fixed CPU-only and advanced CPU compositions prepare exactly one non-empty maximal CPU
partition. Zero-node pass-through and direct multi-partition artifacts are rejected. Inputs and
every CPU operation occurrence require fully static Shapes and resolved compatible layouts;
adding `contiguous()` after an operation does not retroactively resolve that operation's inputs or
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
