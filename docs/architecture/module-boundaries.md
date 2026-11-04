# Module Boundaries

This document explains the module responsibilities established by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract is authoritative when a summary here is incomplete.

The boundaries apply to both implemented and planned modules. Model, Backend Contract, Planning,
Compiler, Runtime, Prepare, Engine, CPU, and the occurrence-qualified Metal execution domain have
substantive implementations; Config and Trace are partial.
The current public execution path supports fixed CPU ownership and explicit CPU/Metal composition,
including mixed-owner schedules with bounded bidirectional all-seven-carrier transfer over exact
rank-0..16 static canonical or positive-stride non-overlapping layouts. Other concrete backends,
conversion-based transfer, and most extensions remain planned or incomplete. The
[roadmap](../planning/roadmap.md) records exact delivery status.

## Accepted symbolic-batch ownership

The implemented bounded ordinary [dynamic-batch handoff](runtime-prepare-backend-boundary.md#accepted-symbolic-batch-handoff)
and Prepare0014's bounded FLOAT32 symbolic Scan extension use existing modules and unchanged
dependency edges. Final Scan qualification is tracked in its
[task](../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md); static paths are retained. The
[root index](../../ARCHITECTURE.md#scope-indexed-normative-contracts) routes each concern to its sole
normative owner; [ADR 0050](../design/decisions/0050-dynamic-batch-geometry.md) records the decision.

| Stage/owner | Responsibility | Remains outside that owner |
|---|---|---|
| Compiler | Prepare BatchBindingPlan from final metadata/private typed obligations; bind B and requested descriptors. | Runtime state, storage or public diagnostic-string predicates. |
| Prepare | Complete fixed/run-size declarations, explicit batch-required mode, stable slots/positions and transaction. | CPU geometry evaluation, physical allocation or late resource discovery. |
| CPU | Numeric byte/access/MATMUL/aggregate recipes and one-time actual-byte/typed-carrier binding. | Semantic graph interpretation per invocation or per-element recipe dispatch. |
| Runtime | Authenticate exact-plan numeric geometry, create actual resources and own one isolated RunState. | Model/Compiler dependency, symbolic expressions, code/cache/route selection. |
| Engine | Admit actual static input descriptors/storage, exact-prefix borrowing, actual publication/materialization. | Another runner, lease protocol or per-B source rebuild. |

Read the table from immutable compile/prepare decisions toward one invocation's numeric resources.
For X `[B,4]` @ W `[4,3]`, the same plan serves B=64→40→1→64 while Y/dX bytes vary
and dW stays `[4,3]`; no shared currentB changes. A creator closed over prototype B or a Runtime
Shape interpreter would give incorrect spans or cross ownership. Complete recipes plus checked
pre-borrow admission avoid that failure without changing code/routes/slot membership.

The ordinary and Scan slices qualify genuine symbolic capability and the entire actual first-order
derivative closure independently, including empty/overflow/failure/concurrency and retained static
paths. Generated loops are compared with specialized clean Java and bounded matched-work evidence;
no timing parity follows. Engine geometry memoization, NN/Training one-state reuse,
symbolic Compiler/Checkpoint cold acquisition, symbolic wire, builder and
native/mixed/tuning remain separate gates. None grants a new dependency or execution owner.

## Shared modules

### `modules/trace`

Owns typed diagnostic DTOs: the event envelope, compile/prepare/run/backend payloads, trace-local identifiers, and typed trace attributes. It contains no graph traversal, execution, business logic, or mutable runtime state and must not import higher layers. This keeps trace data usable from every producer without creating reverse dependencies.

See [Tracing](tracing.md).

### `modules/backend-contract`

Owns the minimal backend vocabulary shared across layers: `BackendId`, `BackendDeviceId`, availability snapshots, declarative requirements, and device classes. Compile-time plans refer to backend identities rather than live backend services.

It does not own capabilities implementation, operation support logic, kernel registries, prepare services, executables, storage, physical buffers, or cost-model implementation.

### `modules/model`

Owns the public tensor model, operation semantics, shape/data type/layout types, host storage
abstractions, and immutable graph model. A `Tensor` has immutable identity, descriptor, and
expression provenance, plus its existing mutable borrowed host-storage association; it has no
gradient/backward lifecycle state. Each derived `TensorProducer` retains the canonical exact
wrapper for every output position so compiler-owned pre-capture formulas can use hidden auxiliary
outputs without reconstructing wrappers. `CompiledGraphModel`, `CompiledNode`, and graph values
remain distinct compile-time state.

The model does not know backend support, device residency, kernel selection, backend-specific storage, prepared execution, or runtime state. `Operation` expresses semantics and never exposes `supportedBackends()`. Runtime device storage belongs outside this module.

Model is the sole owner of one profile-free family/dtype numerical contract. Its
[normative numerical semantics](contracts/foundational-modules.md#profile-free-numerical-semantics)
keep stored comparisons/extrema, discrete choices, casts, and raw movement exact. Named FLOAT32,
BFLOAT16, and FLOAT16 arithmetic primitive sites may use denormals-are-zero (DAZ) at inputs and
flush-to-zero (FTZ) at results; FLOAT64 sites may not. Composites and generated gradients inherit
only their named arithmetic sites' permissions. Homogeneous low arithmetic uses FLOAT32 work and
accumulators followed by one final ties-to-even narrowing. Finite tolerances are test-only route
qualification, not public envelopes or Compiler rewrite authority. Provider answers remain
separate from semantic reachability, routes, backward ownership, and runtime/device facts.

Metal supports exact bounded seven-carrier movement, indexing, replacement, fold, aggregate,
ordering/top-K/numeric arg-extrema, and no-gradient promoted INT32/INT64 MATMUL domains. It also
supports qualified homogeneous
FLOAT32/BFLOAT16/FLOAT16 arithmetic, reduction, scan, MATMUL, MSE, convolution, pooling, dropout,
L1, ScatterAdd, and singleton-variance occurrence. Every partition containing BFLOAT16 or FLOAT16
values is custom-program-only, including exact no-gradient raw-preserving
RESHAPE/PERMUTE/CONTIGUOUS/SLICE/CONCAT/TILE. Direct BFLOAT16/FLOAT16 mixing and every unsupported
occurrence fail closed.
Rank-zero Metal support is local to produced reduction,
arg-extrema, or vector/vector MATMUL targets plus explicit scalar-seed ingress for vector/vector
gradients; it does not widen mixed-owner CPU/Metal transfer. The local explicitly seeded general
FLOAT32 MATMUL gradient path and ordinary Compiler-generated backward occurrences remain backend
execution of Compiler graphs, not implicit seeding, training ownership, or a complete Metal
backward claim.

The current Model fixed recurrent scan follows this same flat boundary. Model owns the fixed
`RNN_TANH`, `GRU_RESET_AFTER`, and `LSTM` meanings, one `FORWARD` or `REVERSE` attribute, ordered
ordinary Tensor inputs, fully static descriptor rules, canonical dense output and final-state
wrappers, and exactly six public static biased or bias-free constructors on the final field-free
`model.tensor.RecurrentScan` namespace. The time-major input is an explicit first argument;
`Tensor` has no recurrent receiver alias. This advanced low-level namespace is not an NN layer or
module, execution service, registry, or general scan-body abstraction. It owns no callback body,
nested graph, region, captured free variable, runtime length inspection, hidden state, or
execution loop. Current Compiler forward-only inference adopts the family, but every executable
route is qualified independently: CPU now implements the fixed four-floating-dtype/read-layout
domain described below. Current NN static `RnnSequence`, `GruSequence`, and `LstmSequence` routes
retain construction-time Java `long[]` lengths and static unrolling; additive `forwardScan` routes
delegate explicitly, without redirecting those static APIs.

Cold semantic persistence is independent: the
[Checkpoint v3 contract](contracts/extensions-training.md#canonical-graph-model-version-3) retains
the same fixed flat occurrence and all canonical ports. This persistence adds no codec/body/runtime
artifact ownership to Model; result carriers are not serialized.

The accepted [general-scan exception](contracts/recurrent-scan.md#general-differentiable-scan)
separately permits immutable source Tensor algebra on a producer and one scoped compiled body on
a node. Both attachments are implemented. Model constructs canonical latent residual
ports without choosing derivatives or allocating history; this preserves original immutable
producer identity for Compiler's exact saved-value references. Users write Tensor algebra, not
saved-tensor recipes. Operation attrs still contain no Tensor/graph. The
[focused explanation](general-differentiable-scan.md) distinguishes this decision from implemented
fixed scans and from arbitrary control flow.

### `modules/config`

Owns immutable declarative compile, prepare, run, publication, planning-cost, and model-autotuning
inputs after their consumers are stable. It may describe backend intent and scoring
policy, but it contains no benchmark runner, search algorithm, live discovery, mutable evidence,
live service, concrete backend class, executable unit, runtime state, or kernel class reference.
A concrete backend interprets its backend-specific prepare inputs inside that backend.

Config owns no numerical-profile selector. Its remaining immutable values describe compile,
prepare, run, scoring, and tuning inputs without defining Model numerical meaning.

### `modules/planning`

Owns backend-neutral compile-time planning: intent propagation, capability query contracts and matrices, ownership scoring, node or segment ownership, maximal same-owner partitions, and logical memory/materialization requirements.

Planning answers where work should run. It does not implement fusion or specialization, select a concrete kernel or backend route, allocate physical memory, construct backend DAGs, or create prepared schedules and runtime units.

Planning may interpret a backend-neutral cost model to choose `BackendId` ownership. It never
interprets route names, vector species or lanes, unroll factors, thread counts, chunks, tiles, or
other backend parameter vocabulary.

Planning asks whether a backend supports a complete operation occurrence through
`OperationCapabilityQuery`, with no numerical selector. Concrete backend preparation, not
Planning, qualifies and realizes a route under Model semantics.

The executable fixed recurrent scan enters Planning through the unchanged
ordinary operation capability query. Planning selects an owner but does not interpret cell
variant, direction, valid-length values, active rows, loop bounds, or a recurrent execution
route. CPU advertises only its qualified fixed domain.

General Scan transports optional CompiledScanBody and ScanActivityPlan query metadata, including
role bindings, residual demand, full source shapes and structural activity. These are not an
executable Runtime graph. Planning selects one complete-body owner; body and binder qualify
together, not mixed-owner body partitions. Compiler owns contextual inference and selected actual
FLOAT32/FLOAT64 first-order AD; Prepare projects exact body/activity metadata.
CPU composes every ordinary CPU-supported PURE unit under ordinary dtype/layout/value/derivative
guards, reusing ordinary lowering/emitters under the existing loop, direct same-class helpers and
run-owned staging. Model remains the logical-layout owner. This is no universal no-copy or
zero-allocation promise. Families without backward retain that limit; non-gradient primal paths
may contain ordinary nondifferentiable units. All-valid bodies use WholeTensor. Length bodies
may contain proved ROW(axis) values and arbitrary-shape SHARED tensors. The separate final
non-gradient INT64 `[B]` lengths control has no body formal; it is not a scalar alternative.
Original AD carries/sequences and emission boundaries remain ROW(0), while intermediate/history
ROW axes and ranks follow their proofs. Supported row-mixing materialization requires exactly
the same length values, not Tensor identity. Structural absence differs from a present all-zero
seed; cross-Scan materialization is explicit. CPU0030–0033 retain historical bounded qualification,
not a current finite operator whitelist. The bounded [owned NN workflow](#owned-tensor-model-composition)
and narrow [body persistence](contracts/extensions-training.md#canonical-graph-model-version-4)
remain separate admission; no Metal general Scan, nested control flow or fixed-family BPTT follows.

See [Partition Scoring](partition-scoring.md).

### `modules/compiler`

Owns graph capture and compilation: fail-closed pre-capture autograd inventory, named
gradient-rule dispatch, per-compile identity-based contribution accumulation, one phase-aware
combined capture, indexing and ordering, inference and validation, canonicalization and combined
optimization, publication binding, planning orchestration, compile diagnostics, and
`CompileArtifacts`.

Its output is immutable compile-time state. It does not create physical buffers, backend executables, runtime workspaces, prepared schedules, or `PreparedExecution`, and it has no concrete backend dependencies.

Compiler captures each current fixed Model scan producer as exactly one ordinary flat node and current
forward-only inference independently revalidates its fully static descriptors. Both backward-
capable modes reject any complete forward inventory containing fixed recurrence before any derivative
Tensor is allocated. Fixed BPTT remains unsupported. Compiler must not unroll fixed recurrence into
`time` nodes or attach a body graph to that family.

For accepted general scan, Compiler instead builds an ordinary reverse sibling through existing
Tensor rules before one combined capture. Every original primal formula operand binds explicitly
to its residual/step/invariant reverse formal; a saved TANH output is not replayed. Constant leaves
close through explicit outer splats and formals. Reused templates capture in independent scopes
without outer ValueIds, escaping producers or cross-template sharing. The
[Compiler boundary](contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
and [sole semantic protocol](contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings)
describe this implemented compile-time path; no captured-IR AD, second algebra, fixed-family adjoint
or general-scan execution follows.

Compiler0013 extends this boundary to bounded length-mode inference, first-order reverse
construction and complete final queries/demand. The extra INT64 lengths actual has no body
formal or cotangent. Every original carry and sequence must be ROW(0) for AD, including unused
and non-gradient positions; generated reverse histories/seeds instead retain their checked mixed
roles. Per-incoming SHARED normalization, explicit ROW literal/zero bindings and protected actual
binding sites preserve those roles without a new derivative engine or public API. For example,
a shared transposed weight history stays `[T,F,H]`, even at F=B, while a transposed row history
retains ROW(1). Prepare0011 now preserves those exact roles, scoped bodies and saved ports while
checking static boundaries and complete-graph demand before analysis. Its separate INT64 actual
has no local formal; backend analysis still owns resource geometry. Prepare does not duplicate
Model/Compiler row-algebra proof or impose original-source AD admission on generated reverse
bindings. CPU0042 separately implements fixed-geometry FLOAT32/FLOAT64 forward and actual reverse
through existing CPU owners, without changing this handoff. Cold capture/reconstruction and codec
still reject length mode; no symbolic-length, v5 or owned NN workflow follows.

The [cold source lifecycle](contracts/compiler-autograd.md#compile-lifecycle) also owns narrow
CNN/static recurrent admission and fixed-scan reconstruction. It preflights every stored
descriptor before constructing a producer, then calls each public family constructor once and
maps all canonical output slots in a call-local table. It never reads GATHER indices or scan
length values, closes raw layouts, imports host values or grants backend/gradient capability.

### `modules/runtime`

Owns prepared execution contracts and dynamic execution state, including `PreparedExecution`,
`PreparedExecutable`, `PreparedSchedule`, `PreparedMemoryPlan`, slots, `RunState`, resources,
transfers, residency, publication, and the stateless `PreparedExecutionRunner`. Ordered
`PreparedSchedule.Step` positions are the work occurrences; no additional Runtime aggregate sits
between an occurrence and its recipe.

One Runtime `PreparedExecution` retains one exact memory plan and same-plan schedule and uniquely
owns its persistent prepared resources. The schedule orders an optional first representation-
creation occurrence, executable and buffer-transfer occurrences, and a dense final publication
suffix. Repeated executable or transfer occurrences mean repeated work and do not duplicate
resource ownership. Prepared recipes are immutable and reusable. Each active complete logical run
has one isolated mutable `RunState`, even when its schedule spans multiple backends. Runtime owns logical slot
state, ownership transitions, validity/residency, cleanup orchestration, and run isolation;
concrete backends own physical buffer/workspace representation implementations and the actual
allocation, release, transfer, and access mechanics. Checked heterogeneous binding occurs once
before the hot path and produces backend-owned typed direct-reference invocation objects.

Runtime executes already-prepared schedules. It does not optimize graphs, construct autograd,
discover backends, look up backend services, lower partitions, select kernels, autotune, inspect
the graph for tuning, or access or mutate tuning caches. Runtime profiling is passive observation
translated into typed trace data; it cannot change settings. The hot path does not use
`Operation` or `CompiledNode`.

For each synchronous call, `PreparedExecutionRunner` acquires the prepared-execution run lease,
creates one `RunState`, cold-binds every executable, transfer, and publication occurrence before
the first action, traverses them in schedule order, and returns the complete state through a
`RunResult` lease. Separate calls may share the immutable recipe but never runner-created mutable
state.

For the executable fixed recurrent scan, Runtime receives only ordinary caller-input
representations, one reusable prepared executable recipe, cold-bound direct references, and an
isolated `RunState`. It does not inspect the valid-length value, interpret direction or
recurrence, choose loop bounds, or compact active rows.

### `modules/prepare`

Owns shared preparation orchestration and validation: `PrepareContext`,
`BackendPartitionPreparer`, `BackendPartitionAnalysis`, exact backend-neutral buffer/workspace
declarations, stable slot assignment, `PreparedPartition`, partition coverage validation,
prepared memory validation, and prepared schedule validation.

Prepare bridges compile artifacts to runtime contracts, but shared prepare code does not contain concrete CPU, Metal, or CUDA lowering, kernel selection, executable implementations, or backend-specific storage. Those belong to concrete backend modules.

The executable fixed recurrent scan needs no shared loop-body contract. Prepare projects one
ordinary static-Shape partition occurrence, accepts the backend's opaque one-time recurrent-loop
analysis and exact resource declarations, assigns slots, and supplies those assignments for
finalization through the existing staged lifecycle.

The accepted general scan requires new validated projection of its body, disjoint local/outer
descriptor scopes, role bindings and ordinary residual edges/demand. One owner prepares the body
once and declares exact staging/carry/temporary/live-history resources before slots. CPU finalization
then emits the time loop and typed body code; changing addresses/carry remain in the single RunState.
There is no Runtime interpreter, nested execution owner, per-step callback/allocation or opaque
saved-workspace protocol. See the [general handoff](runtime-prepare-backend-boundary.md#general-differentiable-scan-handoff).

### `modules/engine`

Owns the public lifecycle facade and explicit composition root. Current `Engine.standard()`
constructs and owns one fresh fixed CPU composition. It exposes owner-bound compile handles,
`PreparedExecution` as the sole ordinary reusable owner, exact-ID arbitrary-order `Engine.run`,
fresh exact-static-descriptor positional `Engine.runValues`, independent publication leases,
detached host materialization and fresh one-shot compute/backward conveniences. Runtime remains
the sole prepared-resource and run-lease authority; Engine retains direct outward adapters. `AdvancedEngine.takeOwnership(...)` instead takes explicit cleanup
ownership of one supported CPU integration and exposes the representation lifecycle.

The public multi-backend construction boundary is `Engine.Builder` with concrete CPU and Metal
`takeOwnership(...)` overloads. Engine owns the builder, ordered private registry, backend
adapters, compile-time provider/availability inventory, complete owner/transfer preflight, shared
Prepare routing, mixed-owner schedule composition, direct per-input and per-publication adapter
references in outward handles, construction rollback, and outer close order. It rejects duplicate
backend IDs, missing owners, and unsupported directed transfers before backend analysis. The
registry is neither visible to inward modules nor consulted by Runtime; Engine uses only captured
direct references for caller-storage ingress, prepared work, and host materialization.

Engine does not own kernels, backend internals, native backend configuration, graph optimization
passes, a runtime service locator, reflective plugin discovery, or a process-global registry.
Concrete backends never depend on Engine.

The fixed standard factory and public builder are explicit composition, not generic registration
or discovery. The builder prepares non-empty single-owner or mixed CPU/Metal plans through one
shared transaction. Mixed values use one deterministic representation per participating owner and
exact rank-0..16 static transfer for all seven carriers over canonical or positive-stride
non-overlapping layouts with checked physical spans; this does not imply a general conversion or
fallback system.

The builder adapts current CPU model autotuning and the exact singleton-NEG Metal workflow without
moving candidate meaning into Engine. Metal's public backend collaboration owns its two local
routes, one fixed complete-plan candidate, session compatibility, and decision codec. Mixed-owner,
multiple-partition, and MPSGraph-only Metal plans remain ineligible. Any allowed fallback stays
with the already-selected adapter.

For a runnable fixed recurrent scan, Engine owns the checked typed mapping from the logical input
Tensors, including `INT64[batch]` valid lengths, to ordered Runtime caller-input representations
and from publication positions to typed outputs. A length value never causes graph capture,
specialization, or preparation to repeat.

## Concrete backend modules

`modules/backend/cpu`, `modules/backend/metal`, and `modules/backend/cuda` are vertical implementations. Each backend
owns its capability provider, deterministic partition analysis, lowering, fusion, specialization,
route selection, opaque analysis plan, executable finalization against assigned slots, executable
units, physical buffer and workspace representations, their allocation/release/transfer/access
mechanics, typed cold-bound invocation objects, trace contributions, and native integration.

Concrete backend logic belongs in the backend that implements it:

- CPU scalar, Vector API, generated JVM-bytecode CPU computation kernels, and OpenBLAS are routes
  inside `modules/backend/cpu`, not separate backends. CPU owns generation and compatible generated-
  artifact caching during preparation; shared Prepare and Runtime do not interpret or select
  those kernels.
- MPSGraph and custom Metal kernels, Metal storage, and native bridges belong to `modules/backend/metal`.
- CUDA lowering, kernels, storage, and native integration belong to `modules/backend/cuda`.

Metal also owns its immutable public configuration, native-library path validation, native context
construction, and partial-open rollback. Engine may own an already-opened Metal integration, but
it neither duplicates that configuration nor interprets Metal options.

A backend that implements fixed recurrent scan advertises only the exact variant, floating type,
fully static Shape, and direction combinations it executes. Analysis lowers one ordinary node to
one bounded recurrent-loop plan and declares all resources before slot assignment; finalization
constructs the reusable executable. Before output mutation, execution validates every length in
`[0, time]`. Invalid coordinates perform no recurrent dot products, gates, or state update, but
the first capability makes no active-row-compaction or packed-memory promise.

Concrete backends also own typed, version-controlled, tested candidate generators beside their
routes, compatible workload-cache lookup during preparation, and safe heuristic fallback. Shared
prepare and tuning orchestration sees candidates opaquely; backend-specific parameter vocabulary
does not leak into planning or shared parameter bags.

Backends may consume model, config, planning, runtime, prepare, backend-contract, and trace contracts. They must not own public tensor APIs or global compiler logic, and must not depend on engine.

`modules/backend/openblas-provider` is a lower-level leaf used by the CPU backend. It owns loading, symbol binding, GEMM calls, and thread control only. It does not interpret config, decide fallback or ownership, participate in planning, manage residency, or expose the Tensor API. Dependency direction is `modules/backend/cpu -> modules/backend/openblas-provider`.

## Extensions

### `extensions/data`

The [Data contract](contracts/extensions-training.md#extensionsdata) authorizes a host numeric
sample/sequence preparation boundary with only a Model API dependency. Its initial foundation
added no production types or batching API; build registration and enforcement belong to
[Data0001](../planning/extensions/data/tasks/0001-data-extension-boundary.md).

The implemented [Data sequence batching API](../reference/data/sequences.md) provides copied host
`ValidSequenceLengths`, logical `SequenceAxisOrder`, explicit `PaddingSide` and a
`PaddedSequenceBatch` record borrowing an existing Tensor. The record replaces the unpublished
right-only type without a compatibility alias. For caller-supplied values of Shape `[4,3]`,
the fluent form is:

```java
var batch = PaddedSequenceBatch.builder(values)
        .timeMajor().rightPadded().validLengths(3, 2, 1).build();
```

The lengths describe valid prefixes in the three original rows and derive padding lengths
`[1,2,3]`. `.batchMajor()` instead requires values of Shape `[3,4]`; neither choice transposes or
copies data. Axis order, side and counts must be selected explicitly. `.leftPadded()` instead
declares valid suffixes, without moving values. The reusable builder
snapshots lengths at the setter and delegates geometry checks to the record constructor; it
introduces no storage owner or pipeline. The direct constructor and immutable lengths remain
available to advanced callers. Zeros are ordinary data: declaring lengths does not inspect or
prove padding contents.

`LongSequenceBatcher` requires an explicit padding value, side and order, then copies host long-array
rows into one fresh contiguous INT64 heap Tensor through public Model allocation. It pads to the
batch maximum without truncation, retains no caller arrays and introduces no close protocol.
Incremental geometry checks reject oversized input before batch-sized metadata or output allocation;
the cap is `Integer.MAX_VALUE` elements, not bytes. For input rows `[2,3]` and `[1]`, left/batch-major
padding with zero produces `[[2,3],[0,1]]` and counts `[2,1]`. Data owns numeric placement, not token
meaning, model execution or runtime length binding. Floating/sample batching remains deferred.

Neither metadata construction path reads host storage or promises its liveness. The implemented
`PaddedSequenceBatch.toScanLengthsTensor()` method is a separate opt-in materialization step,
used by the [numeric inference workflow](../reference/data/sequences.md#complete-numeric-inference-example).
For a RIGHT/TIME_MAJOR batch with counts `[3,1,0]`, it creates a fresh canonical non-gradient
INT64 `[3]` heap Tensor holding those counts. The application passes `batch.values()` and that
Tensor to the existing `model.compute(values, lengths)` boundary; Data does not bind or execute.
LEFT is rejected before BATCH_MAJOR, and both checks precede output allocation. The method never
reads or repacks values, so metadata-only values can export counts but still cannot execute
without live supported storage. Mutating an export does not change counts or another export.

This separates immutable host declarations from mutable execution inputs without making NN depend
on Data or adding another prepared/resource owner. The first proof uses canonical FLOAT32
`[T,B,F]` numeric values and existing scan Models; Text IDs → Embedding → scan requires a separate
integer-input/Embedding builder decision. [ADR 0044](../design/decisions/0044-opt-in-scan-length-tensor-export.md)
explains the choice and its explicit O(B) count-copy cost. Model retains Tensor/storage semantics;
NN, Training and Engine retain their existing roles. [ADR 0042](../design/decisions/0042-data-extension-boundary.md) explains the empty foundation
and why exporting Model does not require a placeholder Data type.

The shared left/right design preserves descriptor-only metadata checks independently of allocation.
[ADR 0043](../design/decisions/0043-whitespace-text-and-padded-integral-batches.md) explains this
migration and the implemented Text consumer. LEFT preparation does not
widen the scan's right-padded valid-prefix contract.

### `extensions/text`

The implemented [Text API](../reference/text/tokenization.md) follows the
[Text contract](contracts/extensions-training.md#extensionstext): a concrete frozen
whitespace tokenizer and Data-delegating adapter. Text exports
only Data; Model is available transitively. Text owns token-ID meaning, while Data owns counts,
padding side, axes and numeric preparation. Neither Text nor Data runs a model.

For vocabulary `["Hello", "world"]`, left-padded rows for `"Hello world"` and `"missing"`
are `[[2,3],[0,1]]`, with counts `[2,1]`. This preserves case and uses UNK=1 for an unknown run.
An application can compose the resulting Tensor with unchanged Embedding and public Engine;
the external integration project verifies actual numerical values, including an ordinary
nonzero PAD row. Adding NN/Engine to Text to perform this work would reverse the boundary.
LEFT preparation does not change the scan's right-prefix semantics or bind host counts at runtime.
See [ADR 0043](../design/decisions/0043-whitespace-text-and-padded-integral-batches.md) for migration
and the [Text API example](../reference/text/tokenization.md#complete-example) for caller-owned composition.
Vision remains separately gated.

### `extensions/nn`

Owns stateful neural-network composition: `Module`, `Parameter`, `Buffer`, module-tree traversal,
train/eval mode, forward context, layers, blocks, and neural-network functional conveniences.
A parameter is a module-owned trainable value; a buffer is persistent module state that an
optimizer does not update. `train()` and `eval()` are module forward-behavior modes, so their
propagation belongs here rather than in an optimizer.

`extensions/nn` composes the generic tensor semantics supplied by `modules/model`. It does not
own autograd construction, optimizer algorithms, training-step orchestration, backend storage,
kernel selection, or concrete backend dependencies.

The implemented [closed declarative boundary](contracts/extensions-training.md#closed-declarative-models)
also places semantic description and reconstruction in NN. Model supplies semantics; the separate
public Training edge supplies generic live admission and sessions. NN0026 is independently reviewed
and coordinator-validated. For an ordinary nested Sequential MLP, description records
typed configuration, ordered numeric paths and current Tensor-identity groups; reconstruction
returns the same paths and shared descriptor-only references, then existing strict state load supplies
values. This avoids random throwaway weights and a persistence-owned layer interpreter. Description
after a legal binding replacement observes the new groups under caller quiescence, not an old cache.
The [selected ADR scope](../design/decisions/0038-declarative-model-and-training-resume.md#nn-decision-selected-for-review)
explains why custom callbacks, physical-overlap reconstruction and backend execution stay outside it.
File formats and Training-owned run proposals still require their own decisions; no new dependency
or runner is introduced by this NN capability.

The existing static RNN, GRU, LSTM, and bidirectional containers retain their snapshotted Java
`long[]`, static unroll, compact per-step output, and final-state contracts. NN does not own the
fixed scan operation or its prepared loop, and no existing container is silently redirected to
it. The additive one-directional `forwardScan` methods delegate to Model's matching fixed scan
constructor using that same owned cell's current parameters, explicit runtime-length Tensor,
states and direction. For example, LSTM returns dense outputs and explicit final hidden/cell
Tensors; NN retains none as implicit next-call state. Automatic cells validate descriptors and
bind even for empty scans through their existing cell-local publication gate, not a dummy
forward. This keeps parameter initialization in NN and fixed transition meaning in Model.
The public Engine owns separate compile/prepare/runValues execution; CPU qualification remains
fully static homogeneous FLOAT32/FLOAT64/BFLOAT16/FLOAT16 scalar scans with resolved nonnegative
read layouts and canonical outputs, independent of adjacent-operation execution support.
These expressions add no runner or BPTT. The approved NN0022B cutover below permits owned scan
inference; existing static builder recipes and bidirectional containers remain unchanged.
Data integration is separate future work.

### Owned tensor Model composition

The implemented all-valid general-scan workflow uses existing Model.graph/defineTensor and cached
compute/configureTraining/train owners. Acquisition delegates to Compiler's bounded cold inventory
through Checkpoint's attachment-preserving in-memory copy/rebind, independently of CPU support.
Canonical static homogeneous FLOAT32/FLOAT64 occurrences may compute and train where their actual
forward/reverse bodies qualify. All-valid means all static time steps participate, not arbitrary
body support. Fixed scan-training rejection and padding exclusion remain; v4 body files use the
same bounded cold inventory independently of execution support. See the
[extension contract](contracts/extensions-training.md#owned-tensor-models) and
[ADR0046](../design/decisions/0046-general-scan-cold-source-and-owned-model-direction.md).

The [owning NN contract](contracts/extensions-training.md#owned-tensor-models) places high-level
owned state and exclusion in NN, without moving execution or optimizer algorithms there.
`Model.defineTensor` acquires a source after one eval prototype forward; `Model.graph` acquires an
explicit boundary. Both copy each external state identity once through GraphCheckpoint and keep
one working set for compute, configured train and save. Load adopts final heap values directly.

```text
NN owned source -> Engine compile/prepare/runValues -> detached values
               -> TrainingSession.runValues       -> same weights + detached step
               -> GraphCheckpoint.save            -> source + current raw values
```

Each arrow delegates to the existing public owner; save additionally requires wire admission.
General bodies use the bounded v4 inventory below, not arbitrary PURE operations.
For a supported source without fixed scans, configuring training after compute
closes inference preparation before one native parameter admission, preserving Tensor identity.
Rebuilding a source or replacing a weight per batch would invalidate this lifecycle, not refresh it.
Mode, wrappers, paths and storage associations are guarded; inherited mutators remain caller-excluded.
Close waits for active work, then releases sessions/preparations, native memory and only a lazily
owned Engine. A selected Engine is borrowed. See the [owned workflow](../user-guide/running-models.md#owned-static-model-workflow).

Selected default CNNs and static recurrent builders use the same source save path, not per-network
serializers. Their zero-state and row-index constants remain named CONSTANT leaves outside the
Module state dictionary. Advanced list Models may instead declare runtime INT64 lengths as inputs
for cold scan copy/save/load/rebind. The inference-only scan builder uses that same boundary
and execution path, not a new runner.

For example, the builder workflow takes FLOAT32 `[4,3,2]` values
and INT64 `[3]` lengths `[4,2,0]`, composes `gruScan(8).lstmScan(6).finalHidden().linear(5)`,
and ends with `buildWithLengths()`. Each scan starts from its own zero state and consumes the
shared lengths. The final hidden port has shape `[3,6]`; the head produces `[3,5]` detached scores
through `compute(values, validLengths)`, also after save/loadGraph. The zero-length hidden row is
zero, but the ordinary head's bias may make its scores nonzero. `lastTimeStep()` would instead
select dense time index 3, not the semantic final hidden state for short rows.
Only scans skip padded recurrent transitions; ordinary layers retain dense semantics, without a
global mask. Shape-valid composition/persistence is separate from backend execution admission:
CPU scalar scans support four homogeneous floating types and resolved nonnegative read layouts;
this does not qualify every ordinary prefix operation or composition. See the
[complete runtime-length example](../user-guide/running-models.md#run-a-runtime-length-scan-model).

Cold list forward recomposes instantiated owned steps only when explicitly requested. Inference
uses fixed outputs and cached Engine preparations; it never reconstructs recipes per batch or
requires a prior forward call. The two-input compute convenience applies equally to an ordinary
two-input/one-output graph and follows declared positions, not inferred length roles. Existing
graph-model v3 stores those ordered input descriptors, without new role metadata or format changes.

Fixed-scan-containing Models must reject configure/train before the training-transition arrow above:
after lifecycle/live-source checks, but before freezing binding, invoking an objective callback,
creating an Engine, closing inference preparation or copying parameters to native storage. Waiting
for Compiler's BPTT error would perform those side effects first. Static unrolled Models still
train normally; fixed scans remain inference-only even in a mixed fixed/general source.
The [owning contract](contracts/extensions-training.md#owned-tensor-models)
and [ADR 0041](../design/decisions/0041-runtime-length-scan-model-inference.md) define and explain
this limited cutover; backend support, recurrence numerics and execution ownership do not change.

### `modules/training` (Gradle `:extensions:training`)

Owns public backend-neutral optimizer algorithms, reusable training sessions, training steps, and
detached in-memory optimizer/session state. The implemented lifecycle provides `Optimizer`,
`Sgd`, `Adam`, `AdamW`, `ParameterGroup`, `GradientMode`, `TrainingSession`, `TrainingStep`, and
`TrainingState` over generic live `TrainingParameters` sources. NN's `Module.trainingParameters()`
retains its live tree and exact ordered paths/wrappers/Tensors; no frozen-map validation bypass
is allowed. Storage associations and bytes are checked by Training, not frozen by membership.
Ordered groups may assign different
algorithms to parameters within one session and one global optimizer step. Training does not own
`Parameter`, `Buffer`, layer behavior, or train/eval mode.

Training has exactly public Model/Engine edges and no NN dependency; NN has exactly public
Model/Training/Checkpoint/Engine edges. A Training-only caller can supply an ordinary Tensor owner without a Module.
Training uses one Engine compile and one Engine prepared execution; it
must not depend on Runtime, Prepare, tuning, or concrete backend modules or implement another
execution path. A fused optimizer route on Metal, for example, remains a Metal backend
prepare/kernel concern rather than a `MetalOptimizerBridge` in training. See [Training
Graph](training-graph.md).

Training does not own fixed recurrent execution, valid-length handling, or backpropagation through
time. Compiler owns the later BPTT decision and gradient construction; Training remains the
consumer of published gradients and owner of optimizer orchestration.

<a id="extensionscheckpoint"></a>

### `modules/checkpoint` (Gradle `:extensions:checkpoint`)

Generic Checkpoint owns cold graph/value copy/save/load/rebind through GraphCheckpoint and the
opaque artifact package. Its public Model edge and implementation-only Compiler edge expose no
NN/Training/Engine dependency. A static input prototype contributes only its descriptor; named
host leaves contribute raw values by exact identity. Load returns ordinary symbolic Tensors,
without execution or a second ownership copy. See the [graph API](../reference/checkpoint/graph.md).

General-body in-memory copy/rebind now retains each immutable scanBody attachment unchanged while
normalizing only outer node/value IDs. Coincident numeric IDs do not cross body scopes; Compiler
alone preflights and reconstructs fresh body formals and canonical ports per occurrence. For
example, two uses of one source template still become two independent reconstructed bodies.
The exact [cold constructor inventory](contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)
is independent of CPU dtype support, not arbitrary PURE-body copying. V1/v2/v3 remain independently
closed to bodies. Implemented [v4 persistence](contracts/extensions-training.md#canonical-graph-model-version-4)
uses that same cold inventory; unsupported bodies reject before host copy/file effects.
NN uses it for graph/definition acquisition and save/load, while execution
still requires its independently qualified body and binder domain.

For example, a three-step FLOAT32 `tanh(carry + step * weight)` source uses v4 with one inline
body and one raw named weight payload, not three copies of the body or saved activations. Loading
must delegate fresh scoped construction to Compiler; a Checkpoint interpreter or constructor table
would duplicate family ownership and risk scope aliasing. Focused tests cover cold roundtrips and
loaded FLOAT32/FLOAT64 execution/training; main accepted CP0013's final gates.
The [persistence decision](../design/decisions/0047-bounded-general-scan-graph-persistence.md) records
the compatibility tradeoff and pre-code gate. No new module edge or execution owner is involved.

Checkpoint owns [v3 wire and normalization](contracts/extensions-training.md#canonical-graph-model-version-3)
and [v4 body records](contracts/extensions-training.md#canonical-graph-model-version-4),
not operation semantics or Tensor reconstruction. All ordered producer ports receive distinct
implicit value positions; iterative producer-aware topology retains unused sibling outputs while
rejecting dead producers. For example, a finalCell-only LSTM root still retains one three-output
producer. Descriptor-only source preflight precedes named host-state import, and recapture precedes
return. No optimized CompileArtifacts, prepared/native/backend/runtime state or input prototype
bytes are persisted. Published old graph bytes/readers remain compatible; execution and optimizer
resume are separate owner decisions.

### Retained NN checkpoint glue

The old nine model persistence classes now belong to `extensions/nn`, package `nn.checkpoint`.
External old-model users need an explicit NN dependency and migrated imports; no compatibility
forwarders exist. The following old-format behavior is unchanged.
Its artifact API saves and reads one bounded file of opaque named byte payloads.
The file is forced and fully validated before one atomic move; success gives atomic live
visibility, not portable power-loss persistence of the rename. It does not interpret a payload
as a Tensor or model state. The public model facade adds that interpretation: it captures one
complete NN snapshot, materializes through bounded public Engine compute, and exposes detached
typed inspection. Load checks the caller fingerprints and complete file, then reconstructs host
Tensors and asks NN to install the entire state dictionary. A fully bound target defines v1
alias groups; unbound/lazy targets fail. NN paths stay in metadata while fixed ordinal artifact
keys carry bytes. For example, a buffer named `重み` is stored as metadata path `重み` while its
bytes reside at `state/00000000`. See the [model checkpoint API](../reference/checkpoint/models.md),
[artifact API](../reference/checkpoint/artifacts.md),
[envelope decision](../design/decisions/0032-checkpoint-artifact-envelope.md), and
[model schema decision](../design/decisions/0033-model-candidate-schema.md). The
[public model checkpoint decision](../design/decisions/0034-public-model-checkpoint.md)
explains typed inspection, the direct Model dependency, and target-defined aliases.

The model-envelope API extends this same facade to capture, inspect,
read canonical state bytes, and load the existing `ArtifactEnvelope` without file I/O. The
[owning NN contract](contracts/extensions-training.md#retained-nn-checkpoint-apis) and
[ADR 0036](../design/decisions/0036-model-envelope-composition.md) define that boundary.
The internal candidate and codec remain package-private; file calls keep their reread and publication
policies. Reusing the decoder prevents a standalone envelope from bypassing limits or BOOL
validation merely because it did not pass through the file reader.

The [accepted whole-model contract](contracts/extensions-training.md#whole-model-definition-and-composition)
adds semantic definition bytes downstream of NN without changing edges. This CP0005B capability
is implemented and reviewed (Complete, 2026-10-15). It keeps existing ModelCheckpoint, accepts
ordinary supported Sequential or DeclarativeModel roots, and reconstructs only NN's closed matrix. Values-only APIs remain the
advanced foundation; they do not acquire automatic reconstruction of arbitrary Model callbacks.
The separate whole-model kind binds canonical definition bytes, complete values and exact aliases.
NN now owns old model bytes/versions/preflight as well as meaning and strict installation;
Checkpoint retains only opaque artifacts and the separate generic graph format. Reusing public
component validation avoids inventing a trusted-image credential or private parsing interface.

Whole-model scenario: a quiescent Sequential contains two FLOAT32 Linear layers whose
weight paths `0.weight` and `1.weight` share the same Tensor. Whole-model save retains an explicit
identity group as well as both per-path values; load first compares schema/configuration and group
bytes, then returns an alias-preserving heap-backed eval DeclarativeModel. Equal independent
weights remain independent. A corrupted group with unequal bytes fails before returning a model.
Saving a custom Model.define callback instead rejects before Engine/file work: shapes and weights
cannot establish that Java code's meaning. No Training state or native run lifetime is implied.

Conceptual adapter scenario (requires a separately approved adapter): a model envelope contains
parameter `layer.weight`, FLOAT32 Shape.of(2), with canonical bytes `3f800000 80000000`.
Inspection supplies its descriptor; canonical-byte access supplies those exact bytes by NN path.
An optional adapter compares the complete parameter set, descriptors, and bytes with TrainingState.
A TrainingState value ending in `00000000` instead differs: positive and negative zero are not
interchangeable stored values. Rejection precedes publication of the private replacement.
Matching bytes alone do not establish optimizer/session compatibility; Training still validates
restore, and NN still owns installation. The model operations are available; the adapter scenario
is not a runnable current API. The [file-free API example](../reference/checkpoint/values-examples.md#file-free-example)
demonstrates public descriptor and raw-byte access without implementing that adapter.

Exact training continuation is a different problem: it also requires Training-owned state and
caller-owned progress or random state. An optional downstream Training Checkpoint adapter is
reserved for that coordination after a separate gate; it is not included or implemented today.
Putting that adapter's state into the model-only project would couple inference persistence to
Training restore rules and blur which owner validates restored values. NN's narrow admission edge
does not grant Checkpoint that ownership. See the [checkpoint boundary
decision](../design/decisions/0031-model-checkpoint-extension-boundary.md).
The selected first adapter persists only Model + TrainingState and restores saved state, not
automatic exact continuation. Modes, effective contexts, input cursor, external RNG, scheduler,
and scaler stay application-owned. The adapter remains Draft, absent from the build, and
unimplemented; 0005A does not authorize its codec or inclusion.

### `extensions/onnx`

Owns ONNX import, export, and mapping to and from the model. It remains outside the runtime hot path and does not own backend lowering, kernel selection, execution, or residency.

## Tooling

`tools/benchmarks` owns fixed reproducible workloads and observational `BenchmarkReport`
evidence. It compares commits, models, and environments without selecting or mutating production
settings.

`tools/tuning` coordinates one explicit model-autotuning workflow. It reuses or measures canonical
local workload signatures, then measures a bounded set of complete valid graph and plan
candidates end to end. Compiler, planning, prepare, and concrete backends generate candidates for
their own decisions. Tuning owns measurement, explicit cache coordination, and selection, not
semantics, ownership policy, lowering, or route vocabulary.

Future results use an explicit file-backed reusable workload cache plus a model-specific plan
cache or prepared-plan record. A representative model corpus may pre-seed the same workload cache;
there is no separate platform-calibration subsystem or profile.

## Boundary summary

```text
model      = computation semantics and immutable graph model
planning   = backend-neutral ownership and logical requirements
compiler   = graph transformation, autograd, and compile artifacts
prepare    = shared transition contracts and validation
backend    = concrete lowering, implementation choice, and storage
runtime    = prepared execution and dynamic run state
engine     = explicit composition and public lifecycle
trace      = typed diagnostic leaf
nn         = module composition, parameters, buffers, and forward mode
training   = optimizer algorithms and training orchestration
checkpoint = generic cold graph/value persistence and opaque artifacts; old model glue is NN
```

The dependency implications of these responsibilities are detailed in [Dependency Rules](dependency-rules.md).
