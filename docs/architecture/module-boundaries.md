# [Module](../glossary.md#term-module) Boundaries

This document explains the module responsibilities established by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract is authoritative when a summary here is incomplete.

The boundaries apply to both implemented and planned modules. [Model](../glossary.md#term-model), [Backend](../glossary.md#backend) Contract, [Planning](../glossary.md#term-planning),
[Compiler](../glossary.md#term-compiler), [Runtime](../glossary.md#term-runtime), [Prepare](../glossary.md#prepare), [Engine](../glossary.md#term-engine), [CPU](../glossary.md#term-cpu), and the [occurrence](../glossary.md#term-occurrence)-qualified [Metal](../glossary.md#term-metal) execution domain have
substantive implementations; Config and [Trace](../glossary.md#trace) are partial.
The current public execution path supports fixed CPU [ownership](../glossary.md#term-ownership-meanings) and explicit CPU/Metal composition,
including mixed-[owner](../glossary.md#term-owner) schedules with bounded bidirectional all-seven-[carrier](../glossary.md#term-storage-carrier) transfer over exact
[rank](../glossary.md#term-rank)-0..16 static [canonical](../glossary.md#term-canonical-form) or positive-[stride](../glossary.md#term-stride) non-overlapping [layouts](../glossary.md#layout). Other concrete backends,
conversion-based transfer, and most extensions remain planned or incomplete. The
[roadmap](../planning/roadmap.md) records exact delivery status.

Read **owns** in two distinct senses here. A module owns a *decision* when it is responsible for
that part of the design, such as Compiler constructing [derivatives](../glossary.md#term-derivative). A closeable object owns a
*resource* when it is responsible for releasing it. “[Backend ownership](../glossary.md#backend-ownership)” assigns computation to an
execution domain; it does not give that backend ownership of the caller's input memory.
The [execution atlas comparison](../explanation/packages/execution.md#the-distinctions-to-keep-while-reading)
separates logical values, physical representations, recipes and [leases](../glossary.md#term-lease).

### Read responsibilities separately from delivery checkpoints

The module sections below include staged implementation notes, such as initial fixed-scan
backward rejection and earlier cold-format limits. Read those task-numbered notes as checkpoints,
not as overrides of later qualified routes. The
[current support/evidence guide](../backend-guide/production-support.md#status-and-interpretation)
and [Scan explanation](general-differentiable-scan.md#purpose-and-mental-model) distinguish
current admission from the retained development chronology.

For one concrete collaboration, the application calls `Model.compute`; NN calls Engine;
Engine calls Compiler with expressions, shared Prepare with immutable recipes, and Runtime with
prepared work plus current values. Prepare calls backend analysis and finalization; Runtime
calls already-bound executables. No reverse module dependency is needed for these returned
values. The [call-by-call ownership table](../explanation/ownership.md#which-call-borrows-or-copies)
explains why this call chain does not transfer the application's input Arena to every callee.


## Accepted symbolic-[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) [ownership](../glossary.md#term-ownership-meanings)

In this section, B is a named [batch dimension](../glossary.md#batch-dimension) whose numeric size is supplied for each [run](../glossary.md#term-run).
A **[binding](../glossary.md#term-binding)** checks actual input descriptions against that declaration. A **recipe** retains
how to calculate a size or access pattern without retaining the current run's [buffers](../glossary.md#term-buffer).

The implemented bounded ordinary [dynamic-batch handoff](runtime-prepare-backend-boundary.md#accepted-symbolic-batch-handoff)<span data-glossary-navigation="true"> ([handoff glossary](../glossary.md#term-handoff))</span>
and Prepare0014's bounded [FLOAT32](../glossary.md#term-float32) [symbolic Scan](../glossary.md#symbolic-scan) extension use existing [modules](../glossary.md#term-module) and unchanged
[dependency](../glossary.md#term-dependency) edges. Final [Scan](../glossary.md#general-differentiable-scan) [qualification](../glossary.md#term-qualification) is tracked in its
[task](../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md); static paths are retained. The
[root index](../../ARCHITECTURE.md#scope-indexed-normative-contracts) routes each concern to its sole
[normative](../glossary.md#term-normative) [owner](../glossary.md#term-owner); [ADR 0050](../design/decisions/0050-dynamic-batch-geometry.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span> records the decision.

| Stage/owner | Responsibility | Remains outside that owner |
|---|---|---|
| [Compiler](../glossary.md#term-compiler) | [Prepare](../glossary.md#prepare) BatchBindingPlan from final [metadata](../glossary.md#term-metadata)/private typed obligations; bind B and requested [descriptors](../glossary.md#term-descriptor). | [Runtime](../glossary.md#term-runtime) state, storage or public diagnostic-string predicates. |
| Prepare | Complete fixed/run-size declarations, explicit batch-required mode, stable slots/positions and [transaction](../glossary.md#term-transaction). | [CPU](../glossary.md#term-cpu) geometry [evaluation](../glossary.md#term-evaluation), physical allocation or late resource discovery. |
| CPU | Numeric byte/access/[MATMUL](../glossary.md#matrix-multiplication--matmul)/aggregate recipes and one-time actual-byte/typed-[carrier](../glossary.md#term-storage-carrier) binding. | Semantic [graph](../glossary.md#graph) interpretation per invocation or per-[element](../glossary.md#term-element) recipe [dispatch](../glossary.md#term-dispatch). |
| Runtime | Authenticate exact-plan numeric geometry, create actual resources and own one isolated [RunState](../glossary.md#run-state--runstate). | [Model](../glossary.md#term-model)/Compiler dependency, symbolic expressions, code/[cache](../glossary.md#term-cache)/route selection. |
| [Engine](../glossary.md#term-engine) | Admit actual static input descriptors/storage, exact-prefix [borrowing](../glossary.md#term-borrowing), actual [publication](../glossary.md#term-publication)/[materialization](../glossary.md#materialization). | Another runner, [lease](../glossary.md#term-lease) protocol or per-B source rebuild. |

Read the table from [immutable](../glossary.md#term-immutable) [compile](../glossary.md#compile)/prepare decisions toward one invocation's numeric resources.
For X `[B,4]` @ W `[4,3]`, the same plan serves B=64→40→1→64 while Y/dX bytes vary
and dW stays `[4,3]`; no shared currentB changes. A creator closed over [prototype](../glossary.md#term-prototype) B or a Runtime
[Shape](../glossary.md#shape) interpreter would give incorrect spans or cross ownership. Complete recipes plus checked
pre-borrow [admission](../glossary.md#term-admission) avoid that failure without changing code/routes/slot membership.

The ordinary and Scan slices qualify genuine symbolic capability and the entire actual first-order
[derivative](../glossary.md#term-derivative) closure independently, including empty/[overflow](../glossary.md#term-overflow)/failure/[concurrency](../glossary.md#term-concurrency) and retained static
paths. Generated loops are compared with specialized clean Java and bounded matched-work evidence;
no timing parity follows. Engine geometry [memoization](../glossary.md#term-memoization), [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training one-state reuse,
symbolic Compiler/[Checkpoint](../glossary.md#term-checkpoint) cold acquisition, symbolic wire, builder and
native/mixed/tuning remain separate gates. None grants a new dependency or execution owner.

## Shared [modules](../glossary.md#term-module)

### `modules/trace`

Owns typed diagnostic DTOs: the event envelope, [compile](../glossary.md#compile)/[prepare](../glossary.md#prepare)/[run](../glossary.md#term-run)/[backend](../glossary.md#backend) payloads, [trace](../glossary.md#trace)-local identifiers, and typed trace attributes. It contains no [graph](../glossary.md#graph) traversal, execution, business logic, or [mutable](../glossary.md#term-mutable) [runtime](../glossary.md#term-runtime) state and must not import higher layers. This keeps trace data usable from every [producer](../glossary.md#term-producer) without creating reverse dependencies.

See [Tracing](tracing.md).

### `modules/backend-contract`

Owns the minimal [backend](../glossary.md#backend) vocabulary shared across layers: [`BackendId`](../glossary.md#backend-identity--backendid), [`BackendDeviceId`](../glossary.md#backend-device-identity--backenddeviceid), availability snapshots, declarative requirements, and [device](../glossary.md#term-device) classes. [Compile](../glossary.md#compile)-time plans refer to backend [identities](../glossary.md#term-identity-meanings) rather than live backend services.

It does not own capabilities implementation, [operation](../glossary.md#operation) support logic, [kernel](../glossary.md#kernel) registries, [prepare](../glossary.md#prepare) services, executables, storage, physical [buffers](../glossary.md#term-buffer), or cost-[model](../glossary.md#term-model) implementation.

### `modules/model`

Owns the public [tensor](../glossary.md#tensor) [model](../glossary.md#term-model), [operation](../glossary.md#operation) semantics, [shape](../glossary.md#shape)/[data type](../glossary.md#data-type--datatype)/[layout](../glossary.md#layout) types, [host storage](../glossary.md#host-storage)
abstractions, and [immutable](../glossary.md#term-immutable) [graph](../glossary.md#graph) model. A `Tensor` has immutable [identity](../glossary.md#term-identity-meanings), [descriptor](../glossary.md#term-descriptor), and
expression [provenance](../glossary.md#provenance), plus its existing [mutable](../glossary.md#term-mutable) [borrowed](../glossary.md#ownership-borrowing-and-lease) [host](../glossary.md#term-host)-storage association; it has no
[gradient](../glossary.md#term-gradient)/backward [lifecycle](../glossary.md#lifecycle) state. Each derived [`TensorProducer`](../glossary.md#producer--tensorproducer) retains the [canonical](../glossary.md#term-canonical-form) exact
wrapper for every output position so [compiler](../glossary.md#term-compiler)-owned pre-[capture](../glossary.md#term-capture-meanings) formulas can use hidden auxiliary
outputs without reconstructing wrappers. [`CompiledGraphModel`](../glossary.md#compiled-graph--compiledgraphmodel), `CompiledNode`, and graph values
remain distinct [compile](../glossary.md#compile)-time state.

The model does not know [backend](../glossary.md#backend) support, [device](../glossary.md#term-device) [residency](../glossary.md#residency), [kernel](../glossary.md#kernel) selection, backend-specific storage, [prepared execution](../glossary.md#prepared-execution--preparedexecution), or [runtime](../glossary.md#term-runtime) state. `Operation` expresses semantics and never exposes `supportedBackends()`. Runtime device storage belongs outside this [module](../glossary.md#term-module).

Model is the sole [owner](../glossary.md#term-owner) of one profile-free family/[dtype](../glossary.md#data-type--datatype) numerical contract. Its
[normative numerical semantics](contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([normative glossary](../glossary.md#term-normative))</span>
keep stored comparisons/[extrema](../glossary.md#saturation-pole-and-extremum), discrete choices, casts, and raw movement exact. Named [FLOAT32](../glossary.md#term-float32),
[BFLOAT16](../glossary.md#term-bfloat16), and [FLOAT16](../glossary.md#term-float16) arithmetic primitive sites may use [denormals-are-zero](../glossary.md#subnormal-daz-ftz-overflow-and-underflow) ([DAZ](../glossary.md#term-daz)) at inputs and
[flush-to-zero](../glossary.md#subnormal-daz-ftz-overflow-and-underflow) ([FTZ](../glossary.md#term-ftz)) at results; [FLOAT64](../glossary.md#term-float64) sites may not. Composites and generated gradients inherit
only their named arithmetic sites' permissions. [Homogeneous](../glossary.md#term-homogeneous) low arithmetic uses FLOAT32 work and
accumulators followed by one final [ties-to-even](../glossary.md#term-ties-to-even) [narrowing](../glossary.md#term-narrowing). [Finite](../glossary.md#term-finite) tolerances are test-only route
[qualification](../glossary.md#term-qualification), not public envelopes or Compiler rewrite authority. Provider answers remain
separate from semantic reachability, routes, backward [ownership](../glossary.md#term-ownership-meanings), and runtime/device facts.

[Metal](../glossary.md#term-metal) supports exact bounded seven-[carrier](../glossary.md#term-storage-carrier) movement, indexing, replacement, fold, aggregate,
ordering/top-K/numeric arg-extrema, and no-gradient promoted [INT32](../glossary.md#data-type--datatype)/[INT64](../glossary.md#data-type--datatype) [MATMUL](../glossary.md#matrix-multiplication--matmul) domains. It also
supports qualified homogeneous
FLOAT32/BFLOAT16/FLOAT16 arithmetic, reduction, [scan](../glossary.md#term-cumulative-scan), MATMUL, [MSE](../glossary.md#mean-squared-error--mse), [convolution](../glossary.md#term-channels-first-convolution-layers), [pooling](../glossary.md#term-pooling), [dropout](../glossary.md#dropout--inverted-dropout),
[L1](../glossary.md#term-l1), ScatterAdd, and [singleton](../glossary.md#term-singleton)-[variance](../glossary.md#term-variance) [occurrence](../glossary.md#term-occurrence). Every [partition](../glossary.md#partition) containing BFLOAT16 or FLOAT16
values is custom-program-only, including exact no-gradient raw-preserving
RESHAPE/PERMUTE/[CONTIGUOUS](../glossary.md#term-contiguous)/[SLICE](../glossary.md#slice)/CONCAT/TILE. Direct BFLOAT16/FLOAT16 mixing and every unsupported
occurrence [fail closed](../glossary.md#admission-preflight-and-fail-closed-validation).
[Rank-zero](../glossary.md#axis-rank-scalar-vector-and-matrix) Metal support is local to produced reduction,
arg-extrema, or [vector](../glossary.md#term-vector)/vector MATMUL [targets](../glossary.md#term-target) plus explicit [scalar](../glossary.md#term-scalar)-[seed](../glossary.md#term-seed) ingress for vector/vector
gradients; it does not widen mixed-owner [CPU](../glossary.md#term-cpu)/Metal transfer. The local explicitly seeded general
FLOAT32 MATMUL gradient path and ordinary Compiler-generated backward occurrences remain backend
execution of Compiler graphs, not implicit seeding, training ownership, or a complete Metal
backward claim.

The current Model fixed recurrent scan follows this same flat boundary. Model owns the fixed
`RNN_TANH`, `GRU_RESET_AFTER`, and [`LSTM`](../glossary.md#term-long-short-term-memory-lstm) meanings, one `FORWARD` or `REVERSE` attribute, ordered
ordinary Tensor inputs, fully static descriptor rules, canonical dense output and final-state
wrappers, and exactly six public static biased or [bias](../glossary.md#term-bias)-free constructors on the final field-free
`model.tensor.RecurrentScan` namespace. The [time-major](../glossary.md#term-time-major) input is an explicit first [argument](../glossary.md#term-argument);
`Tensor` has no recurrent receiver [alias](../glossary.md#aliasing-and-arena). This advanced low-level namespace is not an [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) layer or
module, execution service, [registry](../glossary.md#term-registry), or [general scan](../glossary.md#general-differentiable-scan)-body abstraction. It owns no [callback](../glossary.md#term-callback) body,
nested graph, region, captured free variable, runtime length inspection, hidden state, or
execution loop. Current Compiler forward-only [inference](../glossary.md#term-inference-meanings) adopts the family, but every executable
route is qualified independently: CPU now implements the fixed four-floating-dtype/read-layout
domain described below. Current NN static `RnnSequence`, `GruSequence`, and `LstmSequence` routes
retain construction-time Java `long[]` lengths and static [unrolling](../glossary.md#term-unrolling); additive `forwardScan` routes
delegate explicitly, without redirecting those static APIs.

Cold semantic [persistence](../glossary.md#term-persistence) is independent: the
[Checkpoint v3 contract](contracts/extensions-training.md#canonical-graph-model-version-3)<span data-glossary-navigation="true"> ([Checkpoint glossary](../glossary.md#term-checkpoint))</span> retains
the same fixed flat occurrence and all canonical ports. This persistence adds no [codec](../glossary.md#term-codec)/body/runtime
[artifact](../glossary.md#term-artifact) ownership to Model; result carriers are not serialized.

The accepted [general-scan exception](contracts/recurrent-scan.md#general-differentiable-scan)
separately permits immutable source Tensor algebra on a [producer](../glossary.md#term-producer) and one scoped compiled body on
a [node](../glossary.md#node). Both attachments are implemented. Model constructs canonical latent [residual](../glossary.md#scan-residual-port-and-history-demand)
ports without choosing [derivatives](../glossary.md#term-derivative) or allocating history; this preserves original immutable
producer identity for Compiler's exact saved-value references. Users write Tensor algebra, not
saved-tensor recipes. Operation attrs still contain no Tensor/graph. The
[focused explanation](general-differentiable-scan.md) distinguishes this decision from implemented
fixed scans and from arbitrary control flow.

### `modules/config`

Owns [immutable](../glossary.md#term-immutable) declarative [compile](../glossary.md#compile), [prepare](../glossary.md#prepare), [run](../glossary.md#term-run), [publication](../glossary.md#term-publication), [planning](../glossary.md#term-planning)-cost, and [model](../glossary.md#term-model)-[autotuning](../glossary.md#autotuning--model-autotuning)
inputs after their consumers are stable. It may describe [backend intent](../glossary.md#backend-intent--backendintent) and scoring
policy, but it contains no benchmark runner, search algorithm, live discovery, [mutable](../glossary.md#term-mutable) evidence,
live service, [concrete backend](../glossary.md#concrete-backend) class, executable unit, [runtime](../glossary.md#term-runtime) state, or [kernel](../glossary.md#kernel) class reference.
A concrete backend interprets its [backend](../glossary.md#backend)-specific prepare inputs inside that backend.

Config owns no numerical-profile selector. Its remaining immutable values describe compile,
prepare, run, scoring, and tuning inputs without defining Model numerical meaning.

### `modules/planning`

Owns [backend](../glossary.md#backend)-neutral [compile](../glossary.md#compile)-time [planning](../glossary.md#term-planning): intent propagation, capability query contracts and [matrices](../glossary.md#term-matrix-meanings), [ownership](../glossary.md#term-ownership-meanings) scoring, [node](../glossary.md#node) or segment ownership, maximal same-[owner](../glossary.md#term-owner) partitions, and logical memory/[materialization](../glossary.md#materialization) requirements.

Planning answers where work should [run](../glossary.md#term-run). It does not implement [fusion](../glossary.md#term-fusion) or [specialization](../glossary.md#term-specialization), select a concrete [kernel](../glossary.md#kernel) or [backend route](../glossary.md#backend-route), allocate physical memory, construct backend DAGs, or create prepared schedules and [runtime](../glossary.md#term-runtime) units.

Planning may interpret a backend-neutral [cost model](../glossary.md#term-planning-cost-model) to choose [`BackendId`](../glossary.md#backend-identity--backendid) ownership. It never
interprets route names, [vector species](../glossary.md#term-vector-species) or [lanes](../glossary.md#term-lane), unroll factors, thread counts, chunks, tiles, or
other backend [parameter](../glossary.md#term-parameter) vocabulary.

Planning asks whether a backend supports a complete [operation](../glossary.md#operation) [occurrence](../glossary.md#term-occurrence) through
[`OperationCapabilityQuery`](../glossary.md#operation-capability-query--operationcapabilityquery), with no numerical selector. [Concrete backend](../glossary.md#concrete-backend) preparation, not
Planning, qualifies and realizes a route under [Model](../glossary.md#term-model) semantics.

The executable fixed recurrent [scan](../glossary.md#term-fixed-recurrent-scan) enters Planning through the unchanged
ordinary [operation capability query](../glossary.md#operation-capability-query--operationcapabilityquery). Planning selects an owner but does not interpret cell
variant, direction, valid-length values, active rows, loop bounds, or a recurrent execution
route. [CPU](../glossary.md#term-cpu) advertises only its qualified fixed domain.

[General Scan](../glossary.md#general-differentiable-scan) transports optional CompiledScanBody and ScanActivityPlan query [metadata](../glossary.md#term-metadata), including
role [bindings](../glossary.md#term-binding), [residual](../glossary.md#scan-residual-port-and-history-demand) demand, full source [shapes](../glossary.md#shape) and structural activity. These are not an
executable Runtime [graph](../glossary.md#graph). Planning selects one complete-body owner; body and binder qualify
together, not mixed-owner body partitions. [Compiler](../glossary.md#term-compiler) owns contextual [inference](../glossary.md#term-inference-meanings) and selected actual
[FLOAT32](../glossary.md#term-float32)/[FLOAT64](../glossary.md#term-float64) first-order [AD](../glossary.md#autograd); [Prepare](../glossary.md#prepare) projects exact body/activity metadata.
CPU composes every ordinary CPU-supported [PURE](../glossary.md#term-pure) unit under ordinary [dtype](../glossary.md#data-type--datatype)/[layout](../glossary.md#layout)/value/[derivative](../glossary.md#term-derivative)
guards, reusing ordinary [lowering](../glossary.md#lowering)/emitters under the existing loop, direct same-class helpers and
run-owned staging. Model remains the logical-layout owner. This is no universal no-copy or
zero-allocation promise. Families without backward retain that limit; non-[gradient](../glossary.md#term-gradient) [primal](../glossary.md#term-primal) paths
may contain ordinary nondifferentiable units. All-valid bodies use WholeTensor. Length bodies
may contain proved ROW([axis](../glossary.md#term-axis)) values and arbitrary-shape SHARED [tensors](../glossary.md#tensor). The separate final
non-gradient [INT64](../glossary.md#data-type--datatype) `[B]` lengths control has no body formal; it is not a [scalar](../glossary.md#term-scalar) alternative.
Original AD carries/sequences and emission boundaries remain ROW(0), while intermediate/history
ROW axes and ranks follow their proofs. Supported row-mixing materialization requires exactly
the same length values, not Tensor [identity](../glossary.md#term-identity-meanings). Structural absence differs from a present all-zero
[seed](../glossary.md#term-seed); cross-Scan materialization is explicit. CPU0030–0033 retain historical bounded [qualification](../glossary.md#term-qualification),
not a current [finite](../glossary.md#term-finite) operator whitelist. The bounded [owned NN workflow](#owned-tensor-model-composition)<span data-glossary-navigation="true"> ([NN glossary](../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
and narrow [body persistence](contracts/extensions-training.md#canonical-graph-model-version-4)<span data-glossary-navigation="true"> ([persistence glossary](../glossary.md#term-persistence))</span>
remain separate [admission](../glossary.md#term-admission); no [Metal](../glossary.md#term-metal) general Scan, nested control flow or fixed-family [BPTT](../glossary.md#term-backpropagation-through-time-bptt) follows.

See [Partition Scoring](partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../glossary.md#partition))</span>.

### `modules/compiler`

Owns [graph capture](../glossary.md#capture-graph-ports-and-reconstruction) and [compilation](../glossary.md#compiler-engine-and-execution-stage-names): [fail-closed](../glossary.md#term-fail-closed) pre-[capture](../glossary.md#term-capture-meanings) [autograd](../glossary.md#autograd) inventory, named
[gradient](../glossary.md#term-gradient)-rule [dispatch](../glossary.md#term-dispatch), per-[compile](../glossary.md#compile) [identity](../glossary.md#term-identity-meanings)-based contribution accumulation, one phase-aware
combined capture, indexing and ordering, [inference](../glossary.md#term-inference-meanings) and [validation](../glossary.md#term-validation), [canonicalization](../glossary.md#occurrence-identity-and-canonical-form) and combined
optimization, [publication](../glossary.md#term-publication) [binding](../glossary.md#term-binding), [planning](../glossary.md#term-planning) orchestration, [compile diagnostics](../glossary.md#compile-diagnostics), and
`CompileArtifacts`.

Its output is [immutable](../glossary.md#term-immutable) compile-time state. It does not create physical [buffers](../glossary.md#term-buffer), [backend](../glossary.md#backend) executables, [runtime](../glossary.md#term-runtime) workspaces, prepared schedules, or [`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution), and it has no [concrete backend](../glossary.md#concrete-backend) dependencies.

[Compiler](../glossary.md#term-compiler) captures each current fixed [Model](../glossary.md#term-model) [scan](../glossary.md#term-fixed-recurrent-scan) [producer](../glossary.md#term-producer) as exactly one ordinary flat [node](../glossary.md#node) and current
forward-only inference independently revalidates its fully static [descriptors](../glossary.md#term-descriptor). Both backward-
capable modes reject any complete forward inventory containing fixed recurrence before any [derivative](../glossary.md#term-derivative)
[Tensor](../glossary.md#tensor) is allocated. Fixed [BPTT](../glossary.md#term-backpropagation-through-time-bptt) remains unsupported. Compiler must not unroll fixed recurrence into
`time` nodes or attach a body [graph](../glossary.md#graph) to that family.

For accepted [general scan](../glossary.md#general-differentiable-scan), Compiler instead builds an ordinary reverse sibling through existing
Tensor rules before one combined capture. Every original [primal](../glossary.md#term-primal) formula operand binds explicitly
to its [residual](../glossary.md#scan-residual-port-and-history-demand)/step/[invariant](../glossary.md#term-invariant) reverse formal; a saved [TANH](../glossary.md#term-tanh) output is not replayed. [Constant](../glossary.md#term-constant) [leaves](../glossary.md#term-leaf)
close through explicit outer splats and formals. Reused templates capture in independent scopes
without outer ValueIds, escaping producers or cross-template sharing. The
[Compiler boundary](contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
and [sole semantic protocol](contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings)
describe this implemented compile-time path; no captured-[IR](../glossary.md#term-intermediate-representation-ir) [AD](../glossary.md#autograd), second algebra, fixed-family [adjoint](../glossary.md#term-adjoint)
or general-scan execution follows.

Compiler0013 extends this boundary to bounded length-mode inference, first-order reverse
construction and complete final queries/demand. The extra [INT64](../glossary.md#data-type--datatype) lengths actual has no body
formal or [cotangent](../glossary.md#cotangent). Every original [carry](../glossary.md#term-carry) and sequence must be ROW(0) for AD, including unused
and non-gradient positions; generated reverse histories/[seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings) instead retain their checked mixed
roles. Per-incoming SHARED [normalization](../glossary.md#term-normalization), explicit ROW literal/zero bindings and protected actual
binding sites preserve those roles without a new derivative [engine](../glossary.md#term-engine) or public [API](../glossary.md#term-application-programming-interface-api). For example,
a shared transposed [weight](../glossary.md#term-weight) history stays `[T,F,H]`, even at F=B, while a transposed row history
retains ROW(1). Prepare0011 now preserves those exact roles, scoped bodies and saved ports while
checking static boundaries and complete-graph demand before analysis. Its separate INT64 actual
has no local formal; backend analysis still owns resource geometry. [Prepare](../glossary.md#prepare) does not duplicate
Model/Compiler row-algebra proof or impose original-source AD [admission](../glossary.md#term-admission) on generated reverse
bindings. CPU0042 separately implements fixed-geometry [FLOAT32](../glossary.md#term-float32)/[FLOAT64](../glossary.md#term-float64) forward and actual reverse
through existing [CPU](../glossary.md#term-cpu) [owners](../glossary.md#term-owner), without changing this [handoff](../glossary.md#term-handoff). Cold capture/reconstruction and [codec](../glossary.md#term-codec)
still reject length mode; no symbolic-length, v5 or owned [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) workflow follows.

The [cold source lifecycle](contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../glossary.md#lifecycle))</span> also owns narrow
[CNN](../glossary.md#term-convolutional-neural-network-cnn)/static recurrent admission and fixed-scan reconstruction. It preflights every stored
descriptor before constructing a producer, then calls each public family constructor once and
maps all [canonical](../glossary.md#term-canonical-form) output slots in a call-local table. It never reads [GATHER](../glossary.md#gather) indices or scan
length values, closes raw [layouts](../glossary.md#layout), imports [host](../glossary.md#term-host) values or grants backend/gradient capability.

### `modules/runtime`

Owns [prepared execution](../glossary.md#prepared-execution--preparedexecution) contracts and dynamic execution state, including [`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution),
[`PreparedExecutable`](../glossary.md#prepared-executable--preparedexecutable), [`PreparedSchedule`](../glossary.md#prepared-schedule--preparedschedule), [`PreparedMemoryPlan`](../glossary.md#prepared-memory-plan--preparedmemoryplan), slots, [`RunState`](../glossary.md#run-state--runstate), resources,
transfers, [residency](../glossary.md#residency), [publication](../glossary.md#term-publication), and the stateless [`PreparedExecutionRunner`](../glossary.md#prepared-execution-runner--preparedexecutionrunner). Ordered
`PreparedSchedule.Step` positions are the work occurrences; no additional [Runtime](../glossary.md#term-runtime) aggregate sits
between an [occurrence](../glossary.md#term-occurrence) and its recipe.

One Runtime `PreparedExecution` retains one exact memory plan and same-plan schedule and uniquely
owns its persistent prepared resources. The schedule orders an optional first representation-
creation occurrence, executable and [buffer](../glossary.md#term-buffer)-transfer occurrences, and a dense final publication
suffix. Repeated executable or transfer occurrences mean repeated work and do not duplicate
resource [ownership](../glossary.md#term-ownership). Prepared recipes are [immutable](../glossary.md#term-immutable) and reusable. Each active complete logical [run](../glossary.md#term-run)
has one isolated [mutable](../glossary.md#term-mutable) `RunState`, even when its schedule spans multiple [backends](../glossary.md#backend). Runtime owns logical slot
state, [ownership](../glossary.md#term-ownership-meanings) transitions, validity/residency, cleanup orchestration, and run isolation;
concrete backends own physical buffer/workspace representation implementations and the actual
allocation, [release](../glossary.md#term-release), transfer, and access mechanics. Checked [heterogeneous](../glossary.md#term-heterogeneous) [binding](../glossary.md#term-binding) occurs once
before the [hot path](../glossary.md#term-hot-path) and produces backend-owned typed direct-reference invocation objects.

Runtime executes already-prepared schedules. It does not optimize [graphs](../glossary.md#graph), construct [autograd](../glossary.md#autograd),
discover backends, look up backend services, lower partitions, select [kernels](../glossary.md#kernel), autotune, inspect
the graph for tuning, or access or mutate tuning caches. [Runtime profiling](../glossary.md#runtime-profiling) is passive observation
translated into typed [trace](../glossary.md#trace) data; it cannot change settings. The hot path does not use
[`Operation`](../glossary.md#operation) or `CompiledNode`.

For each [synchronous](../glossary.md#term-synchronous) call, `PreparedExecutionRunner` acquires the prepared-execution run [lease](../glossary.md#term-lease),
creates one `RunState`, cold-binds every executable, transfer, and [publication occurrence](../glossary.md#publication-occurrence) before
the first action, traverses them in schedule order, and returns the complete state through a
[`RunResult`](../glossary.md#run-result--runresult) lease. Separate calls may share the immutable recipe but never runner-created mutable
state.

For the executable fixed recurrent [scan](../glossary.md#term-fixed-recurrent-scan), Runtime receives only ordinary caller-input
representations, one reusable [prepared executable](../glossary.md#prepared-executable--preparedexecutable) recipe, cold-bound direct references, and an
isolated `RunState`. It does not inspect the valid-length value, interpret direction or
recurrence, choose loop bounds, or compact active rows.

### `modules/prepare`

Owns shared preparation orchestration and [validation](../glossary.md#term-validation): [`PrepareContext`](../glossary.md#prepare-context-preparecontext),
[`BackendPartitionPreparer`](../glossary.md#backend-partition-preparer--backendpartitionpreparer), [`BackendPartitionAnalysis`](../glossary.md#backend-partition-analysis-backendpartitionanalysis), exact [backend](../glossary.md#backend)-neutral [buffer](../glossary.md#term-buffer)/workspace
declarations, stable slot assignment, [`PreparedPartition`](../glossary.md#prepared-partition--preparedpartition), [partition](../glossary.md#partition) coverage validation,
prepared memory validation, and [prepared schedule](../glossary.md#prepared-schedule--preparedschedule) validation.

[Prepare](../glossary.md#prepare) bridges [compile artifacts](../glossary.md#compile-artifacts) to [runtime](../glossary.md#term-runtime) contracts, but shared prepare code does not contain concrete [CPU](../glossary.md#term-cpu), [Metal](../glossary.md#term-metal), or [CUDA](../glossary.md#term-cuda) [lowering](../glossary.md#lowering), [kernel](../glossary.md#kernel) selection, executable implementations, or backend-specific storage. Those belong to [concrete backend](../glossary.md#concrete-backend) [modules](../glossary.md#term-module).

The executable fixed recurrent [scan](../glossary.md#term-fixed-recurrent-scan) needs no shared loop-body contract. Prepare projects one
ordinary static-[Shape](../glossary.md#shape) partition [occurrence](../glossary.md#term-occurrence), accepts the backend's [opaque](../glossary.md#term-opaque) one-time recurrent-loop
analysis and exact resource declarations, assigns slots, and supplies those assignments for
finalization through the existing staged [lifecycle](../glossary.md#lifecycle).

The accepted [general scan](../glossary.md#general-differentiable-scan) requires new validated projection of its body, disjoint local/outer
[descriptor](../glossary.md#term-descriptor) scopes, role [bindings](../glossary.md#term-binding) and ordinary [residual](../glossary.md#scan-residual-port-and-history-demand) edges/demand. One [owner](../glossary.md#term-owner) prepares the body
once and declares exact staging/[carry](../glossary.md#term-carry)/temporary/live-history resources before slots. CPU finalization
then emits the time loop and typed body code; changing addresses/carry remain in the single [RunState](../glossary.md#run-state--runstate).
There is no Runtime interpreter, nested execution owner, per-step [callback](../glossary.md#term-callback)/allocation or opaque
saved-workspace protocol. See the [general handoff](runtime-prepare-backend-boundary.md#general-differentiable-scan-handoff)<span data-glossary-navigation="true"> ([handoff glossary](../glossary.md#term-handoff))</span>.

### `modules/engine`

Owns the public [lifecycle](../glossary.md#lifecycle) [facade](../glossary.md#term-facade) and explicit [composition root](../glossary.md#term-composition-root). Current `Engine.standard()`
constructs and owns one fresh fixed [CPU](../glossary.md#term-cpu) composition. It exposes [owner](../glossary.md#term-owner)-bound [compile](../glossary.md#compile) handles,
[`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution) as the sole ordinary reusable owner, exact-ID arbitrary-order `Engine.run`,
fresh exact-static-[descriptor](../glossary.md#term-descriptor) positional [`Engine.runValues`](../glossary.md#fresh-value-binding--enginerunvalues), independent [publication](../glossary.md#term-publication) [leases](../glossary.md#term-lease),
[detached](../glossary.md#detached-model-results) [host](../glossary.md#term-host) [materialization](../glossary.md#materialization) and fresh one-shot compute/backward conveniences. [Runtime](../glossary.md#term-runtime) remains
the sole prepared-resource and [run](../glossary.md#term-run)-lease authority; [Engine](../glossary.md#term-engine) retains direct outward adapters. `AdvancedEngine.takeOwnership(...)` instead takes explicit cleanup
[ownership](../glossary.md#term-ownership-meanings) of one supported CPU integration and exposes the representation lifecycle.

The public multi-[backend](../glossary.md#backend) construction boundary is `Engine.Builder` with concrete CPU and [Metal](../glossary.md#term-metal)
`takeOwnership(...)` overloads. Engine owns the builder, ordered private [registry](../glossary.md#term-registry), backend
adapters, compile-time provider/availability inventory, complete owner/transfer [preflight](../glossary.md#term-preflight), shared
[Prepare](../glossary.md#prepare) routing, mixed-owner schedule composition, direct per-input and per-publication adapter
references in outward handles, construction [rollback](../glossary.md#term-rollback), and outer close order. It rejects duplicate
backend IDs, missing owners, and unsupported directed transfers before backend analysis. The
registry is neither visible to inward [modules](../glossary.md#term-module) nor consulted by Runtime; Engine uses only captured
direct references for caller-storage ingress, prepared work, and host materialization.

Engine does not own [kernels](../glossary.md#kernel), backend internals, native backend configuration, [graph optimization](../glossary.md#forward-graph-optimization)
passes, a runtime [service locator](../glossary.md#term-service-locator), reflective [plugin](../glossary.md#term-plugin) discovery, or a process-global registry.
Concrete backends never depend on Engine.

The fixed standard factory and public builder are explicit composition, not generic registration
or discovery. The builder prepares non-empty single-owner or mixed CPU/Metal plans through one
shared [transaction](../glossary.md#term-transaction). Mixed values use one deterministic representation per participating owner and
exact [rank](../glossary.md#term-rank)-0..16 static transfer for all seven carriers over [canonical](../glossary.md#term-canonical-form) or positive-[stride](../glossary.md#term-stride)
non-overlapping [layouts](../glossary.md#layout) with checked physical spans; this does not imply a general conversion or
fallback system.

The builder adapts current CPU [model autotuning](../glossary.md#autotuning--model-autotuning) and the exact [singleton](../glossary.md#term-singleton)-NEG Metal workflow without
moving candidate meaning into Engine. Metal's public backend collaboration owns its two local
routes, one fixed [complete-plan candidate](../glossary.md#complete-plan-candidate), session compatibility, and decision [codec](../glossary.md#term-codec). Mixed-owner,
multiple-[partition](../glossary.md#partition), and [MPSGraph](../glossary.md#term-mpsgraph)-only Metal plans remain ineligible. Any allowed fallback stays
with the already-selected adapter.

For a runnable fixed recurrent [scan](../glossary.md#term-fixed-recurrent-scan), Engine owns the checked typed mapping from the logical input
[Tensors](../glossary.md#tensor), including `INT64[batch]` valid lengths, to ordered Runtime caller-input representations
and from publication positions to typed outputs. A length value never causes [graph capture](../glossary.md#capture-graph-ports-and-reconstruction),
[specialization](../glossary.md#term-specialization), or preparation to repeat.

## [Concrete backend](../glossary.md#concrete-backend) [modules](../glossary.md#term-module)

`modules/backend/cpu`, `modules/backend/metal`, and `modules/backend/cuda` are vertical implementations. Each [backend](../glossary.md#backend)
owns its capability provider, deterministic [partition](../glossary.md#partition) analysis, [lowering](../glossary.md#lowering), [fusion](../glossary.md#term-fusion), [specialization](../glossary.md#term-specialization),
route selection, [opaque](../glossary.md#term-opaque) analysis plan, executable finalization against assigned slots, executable
units, physical [buffer](../glossary.md#term-buffer) and workspace representations, their allocation/[release](../glossary.md#term-release)/transfer/access
mechanics, typed cold-[bound invocation](../glossary.md#bound-invocation--boundinvocation) objects, [trace](../glossary.md#trace) contributions, and native integration.

Concrete backend logic belongs in the backend that implements it:

- CPU scalar, Vector API, generated JVM-bytecode CPU computation kernels, and OpenBLAS are routes
  inside <span data-glossary-navigation="true"> ([scalar glossary](../glossary.md#term-scalar); [Vector API glossary](../glossary.md#term-vector-api); [JVM glossary](../glossary.md#term-java-virtual-machine-jvm); [bytecode glossary](../glossary.md#term-bytecode); [OpenBLAS glossary](../glossary.md#term-openblas))</span>`modules/backend/cpu`, not separate backends. CPU owns generation and compatible generated-
  artifact caching during preparation; shared Prepare and Runtime do not interpret or select
  those kernels.<span data-glossary-navigation="true"> ([artifact glossary](../glossary.md#term-artifact))</span>
- [MPSGraph](../glossary.md#term-mpsgraph) and custom [Metal](../glossary.md#term-metal) [kernels](../glossary.md#kernel), Metal storage, and native bridges belong to `modules/backend/metal`.
- [CUDA](../glossary.md#term-cuda) lowering, kernels, storage, and native integration belong to `modules/backend/cuda`.

Metal also owns its [immutable](../glossary.md#term-immutable) public configuration, native-library path [validation](../glossary.md#term-validation), native context
construction, and partial-open [rollback](../glossary.md#term-rollback). [Engine](../glossary.md#term-engine) may own an already-opened Metal integration, but
it neither duplicates that configuration nor interprets Metal options.

A backend that implements fixed recurrent [scan](../glossary.md#term-fixed-recurrent-scan) advertises only the exact variant, floating type,
fully static [Shape](../glossary.md#shape), and direction combinations it executes. Analysis lowers one ordinary [node](../glossary.md#node) to
one bounded recurrent-loop plan and declares all resources before slot assignment; finalization
constructs the reusable executable. Before output mutation, execution validates every length in
`[0, time]`. Invalid [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix) perform no recurrent [dot products](../glossary.md#linear-algebra-building-blocks), gates, or state update, but
the first capability makes no active-row-compaction or packed-memory promise.

Concrete backends also own typed, version-controlled, tested candidate generators beside their
routes, compatible workload-[cache](../glossary.md#term-cache) [lookup](../glossary.md#term-lookup) during preparation, and safe [heuristic](../glossary.md#term-heuristic) fallback. Shared
[prepare](../glossary.md#prepare) and tuning orchestration sees candidates opaquely; backend-specific [parameter](../glossary.md#term-parameter) vocabulary
does not leak into [planning](../glossary.md#term-planning) or shared parameter bags.

Backends may consume [model](../glossary.md#term-model), config, planning, [runtime](../glossary.md#term-runtime), prepare, backend-contract, and trace contracts. They must not own public [tensor](../glossary.md#tensor) APIs or global [compiler](../glossary.md#term-compiler) logic, and must not depend on engine.

`modules/backend/openblas-provider` is a lower-level [leaf](../glossary.md#term-dependency-leaf) used by the [CPU](../glossary.md#term-cpu) backend. It owns loading, [symbol](../glossary.md#term-symbol) [binding](../glossary.md#term-binding), [GEMM](../glossary.md#general-matrix-multiplication--gemm) calls, and thread control only. It does not interpret config, decide fallback or [ownership](../glossary.md#term-ownership-meanings), participate in planning, manage [residency](../glossary.md#residency), or expose the Tensor [API](../glossary.md#term-application-programming-interface-api). [Dependency direction](../glossary.md#java-platform-and-build-tools) is `modules/backend/cpu -> modules/backend/openblas-provider`.

## Extensions

### `extensions/data`

The [Data contract](contracts/extensions-training.md#extensionsdata) authorizes a [host](../glossary.md#term-host) numeric
[sample](../glossary.md#term-sample-machine-learning-and-hyperparameters)/sequence preparation boundary with only a [Model](../glossary.md#term-model) [API](../glossary.md#term-application-programming-interface-api) [dependency](../glossary.md#term-dependency). Its initial foundation
added no production types or batching API; build registration and enforcement belong to
[Data0001](../planning/extensions/data/tasks/0001-data-extension-boundary.md).

The implemented [Data sequence batching API](../reference/data/sequences.md) provides copied host
`ValidSequenceLengths`, logical `SequenceAxisOrder`, explicit `PaddingSide` and a
`PaddedSequenceBatch` record [borrowing](../glossary.md#term-borrowing) an existing [Tensor](../glossary.md#tensor). The record replaces the unpublished
right-only type without a compatibility [alias](../glossary.md#aliasing-and-arena). For caller-supplied values of [Shape](../glossary.md#shape) `[4,3]`,
the fluent form is:

```java
// Declare how existing values are organized; the builder neither transposes nor copies them.
var batch = PaddedSequenceBatch.builder(values)
        .timeMajor().rightPadded().validLengths(3, 2, 1).build();
```

The lengths describe valid prefixes in the three original rows and derive [padding](../glossary.md#padding) lengths
`[1,2,3]`. `.batchMajor()` instead requires values of Shape `[3,4]`; neither choice transposes or
copies data. [Axis](../glossary.md#term-axis) order, side and counts must be selected explicitly. `.leftPadded()` instead
declares valid suffixes, without moving values. The reusable builder
snapshots lengths at the setter and delegates geometry checks to the record constructor; it
introduces no storage [owner](../glossary.md#term-owner) or pipeline. The direct constructor and [immutable](../glossary.md#term-immutable) lengths remain
available to advanced callers. Zeros are ordinary data: declaring lengths does not inspect or
prove padding contents.

`LongSequenceBatcher` requires an explicit padding value, side and order, then copies host long-array
rows into one fresh [contiguous](../glossary.md#term-contiguous) [INT64](../glossary.md#data-type--datatype) [heap](../glossary.md#term-heap) Tensor through public Model allocation. It pads to the
[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) maximum without [truncation](../glossary.md#term-truncation-toward-zero), retains no caller arrays and introduces no close protocol.
Incremental geometry checks reject oversized input before batch-sized [metadata](../glossary.md#term-metadata) or output allocation;
the cap is `Integer.MAX_VALUE` [elements](../glossary.md#term-element), not bytes. For input rows `[2,3]` and `[1]`, left/batch-major
padding with zero produces `[[2,3],[0,1]]` and counts `[2,1]`. Data owns numeric placement, not [token](../glossary.md#term-token)
meaning, model execution or [runtime](../glossary.md#term-runtime) length [binding](../glossary.md#term-binding). Floating/sample batching remains deferred.

Neither metadata construction path reads [host storage](../glossary.md#host-storage) or promises its [liveness](../glossary.md#term-liveness). The implemented
`PaddedSequenceBatch.toScanLengthsTensor()` method is a separate opt-in [materialization](../glossary.md#materialization) step,
used by the [numeric inference workflow](../reference/data/sequences.md#complete-numeric-inference-example)<span data-glossary-navigation="true"> ([inference glossary](../glossary.md#term-inference))</span>.
For a RIGHT/TIME_MAJOR batch with counts `[3,1,0]`, it creates a fresh [canonical](../glossary.md#term-canonical-form) non-[gradient](../glossary.md#term-gradient)
INT64 `[3]` heap Tensor holding those counts. The application passes `batch.values()` and that
Tensor to the existing `model.compute(values, lengths)` boundary; Data does not bind or execute.
LEFT is rejected before BATCH_MAJOR, and both checks precede output allocation. The method never
reads or repacks values, so metadata-only values can export counts but still cannot execute
without live supported storage. Mutating an export does not change counts or another export.

This separates immutable host declarations from [mutable](../glossary.md#term-mutable) execution inputs without making [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) depend
on Data or adding another prepared/[resource owner](../glossary.md#ownership-borrowing-and-lease). The first proof uses canonical [FLOAT32](../glossary.md#term-float32)
`[T,B,F]` numeric values and existing scan Models; Text IDs → [Embedding](../glossary.md#embedding) → scan requires a separate
integer-input/Embedding builder decision. [ADR 0044](../design/decisions/0044-opt-in-scan-length-tensor-export.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span>
explains the choice and its explicit [O(B)](../glossary.md#memory-management-atomic-counters-and-complexity) count-copy cost. Model retains Tensor/storage semantics;
NN, Training and [Engine](../glossary.md#term-engine) retain their existing roles. [ADR 0042](../design/decisions/0042-data-extension-boundary.md) explains the empty foundation
and why exporting Model does not require a placeholder [Data type](../glossary.md#data-type--datatype).

The shared left/right design preserves [descriptor](../glossary.md#term-descriptor)-only metadata checks independently of allocation.
[ADR 0043](../design/decisions/0043-whitespace-text-and-padded-integral-batches.md) explains this
migration and the implemented Text consumer. LEFT preparation does not
widen the scan's right-padded valid-prefix contract.

### `extensions/text`

The implemented [Text API](../reference/text/tokenization.md)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span> follows the
[Text contract](contracts/extensions-training.md#extensionstext): a concrete [frozen
whitespace tokenizer](../glossary.md#frozen-whitespace-tokenizer) and Data-delegating adapter. Text exports
only Data; [Model](../glossary.md#term-model) is available transitively. Text owns [token](../glossary.md#term-token)-ID meaning, while Data owns counts,
[padding](../glossary.md#padding) side, [axes](../glossary.md#term-axis) and numeric preparation. Neither Text nor Data runs a model.

For vocabulary `["Hello", "world"]`, left-padded rows for `"Hello world"` and `"missing"`
are `[[2,3],[0,1]]`, with counts `[2,1]`. This preserves case and uses [UNK](../glossary.md#frozen-whitespace-tokenizer)=1 for an unknown [run](../glossary.md#term-run).
An application can compose the resulting [Tensor](../glossary.md#tensor) with unchanged [Embedding](../glossary.md#embedding) and public [Engine](../glossary.md#term-engine);
the external integration project verifies actual numerical values, including an ordinary
nonzero [PAD](../glossary.md#frozen-whitespace-tokenizer) row. Adding [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Engine to Text to perform this work would reverse the boundary.
LEFT preparation does not change the scan's right-prefix semantics or bind [host](../glossary.md#term-host) counts at [runtime](../glossary.md#term-runtime).
See [ADR 0043](../design/decisions/0043-whitespace-text-and-padded-integral-batches.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span> for migration
and the [Text API example](../reference/text/tokenization.md#complete-example) for caller-owned composition.
Vision remains separately gated.

### `extensions/nn`

Owns stateful neural-network composition: [`Module`](../glossary.md#term-module), [`Parameter`](../glossary.md#term-parameter), [`Buffer`](../glossary.md#term-buffer), module-tree traversal,
train/[eval mode](../glossary.md#inference-and-evaluation), forward context, layers, blocks, and neural-network functional conveniences.
A parameter is a module-owned trainable value; a buffer is persistent module state that an
[optimizer](../glossary.md#term-optimizer) does not update. `train()` and `eval()` are module forward-behavior modes, so their
propagation belongs here rather than in an optimizer.

`extensions/nn` composes the generic [tensor](../glossary.md#tensor) semantics supplied by `modules/model`. It does not
own [autograd](../glossary.md#autograd) construction, optimizer algorithms, training-step orchestration, [backend](../glossary.md#backend) storage,
[kernel](../glossary.md#kernel) selection, or [concrete backend](../glossary.md#concrete-backend) dependencies.

The implemented [closed declarative boundary](contracts/extensions-training.md#closed-declarative-models)
also places semantic description and reconstruction in [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context). [Model](../glossary.md#term-model) supplies semantics; the separate
public Training edge supplies generic live [admission](../glossary.md#term-admission) and sessions. NN0026 is independently reviewed
and coordinator-validated. For an ordinary nested Sequential [MLP](../glossary.md#term-multilayer-perceptron-mlp), description records
typed configuration, ordered numeric paths and current Tensor-[identity](../glossary.md#term-identity-meanings) groups; reconstruction
returns the same paths and shared [descriptor](../glossary.md#term-descriptor)-only references, then existing strict state load supplies
values. This avoids random throwaway [weights](../glossary.md#term-weight) and a [persistence](../glossary.md#term-persistence)-owned layer interpreter. Description
after a legal [binding](../glossary.md#term-binding) replacement observes the new groups under caller [quiescence](../glossary.md#term-quiescence), not an old [cache](../glossary.md#term-cache).
The [selected ADR scope](../design/decisions/0038-declarative-model-and-training-resume.md#nn-decision-selected-for-review)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr); [scope glossary](../glossary.md#term-scope))</span>
explains why custom callbacks, physical-overlap reconstruction and backend execution stay outside it.
File formats and Training-owned [run](../glossary.md#term-run) proposals still require their own decisions; no new [dependency](../glossary.md#term-dependency)
or runner is introduced by this NN capability.

The existing static [RNN](../glossary.md#term-recurrent-neural-network-rnn), [GRU](../glossary.md#term-gated-recurrent-unit-gru), [LSTM](../glossary.md#term-long-short-term-memory-lstm), and bidirectional containers retain their snapshotted Java
`long[]`, static unroll, compact per-step output, and final-state contracts. NN does not own the
fixed [scan](../glossary.md#term-fixed-recurrent-scan) [operation](../glossary.md#operation) or its prepared loop, and no existing container is silently redirected to
it. The additive one-directional `forwardScan` methods delegate to Model's matching fixed scan
constructor using that same owned cell's current parameters, explicit [runtime](../glossary.md#term-runtime)-length Tensor,
states and direction. For example, LSTM returns dense outputs and explicit final hidden/cell
Tensors; NN retains none as implicit next-call state. Automatic cells validate descriptors and
bind even for empty scans through their existing cell-local [publication](../glossary.md#term-publication) gate, not a dummy
forward. This keeps parameter initialization in NN and fixed transition meaning in Model.
The public [Engine](../glossary.md#term-engine) owns separate [compile](../glossary.md#compile)/[prepare](../glossary.md#prepare)/runValues execution; [CPU](../glossary.md#term-cpu) [qualification](../glossary.md#term-qualification) remains
fully static [homogeneous](../glossary.md#term-homogeneous) [FLOAT32](../glossary.md#term-float32)/[FLOAT64](../glossary.md#term-float64)/[BFLOAT16](../glossary.md#term-bfloat16)/[FLOAT16](../glossary.md#term-float16) [scalar](../glossary.md#term-scalar) scans with resolved nonnegative
read [layouts](../glossary.md#layout) and [canonical](../glossary.md#term-canonical-form) outputs, independent of adjacent-operation execution support.
These expressions add no runner or [BPTT](../glossary.md#term-backpropagation-through-time-bptt). The approved NN0022B cutover below permits owned scan
[inference](../glossary.md#term-inference); existing static builder recipes and bidirectional containers remain unchanged.
Data integration is separate future work.

### Owned [tensor](../glossary.md#tensor) [Model](../glossary.md#term-model) composition

The implemented all-valid general-[scan](../glossary.md#general-differentiable-scan) workflow uses existing Model.[graph](../glossary.md#graph)/defineTensor and cached
compute/configureTraining/train [owners](../glossary.md#term-owner). Acquisition delegates to [Compiler](../glossary.md#term-compiler)'s bounded cold inventory
through [Checkpoint](../glossary.md#term-checkpoint)'s attachment-preserving in-memory copy/[rebind](../glossary.md#capture-graph-ports-and-reconstruction), independently of [CPU](../glossary.md#term-cpu) support.
[Canonical](../glossary.md#term-canonical-form) static [homogeneous](../glossary.md#term-homogeneous) [FLOAT32](../glossary.md#term-float32)/[FLOAT64](../glossary.md#term-float64) occurrences may compute and train where their actual
forward/reverse bodies qualify. All-valid means all static time steps participate, not arbitrary
body support. Fixed scan-training rejection and [padding](../glossary.md#padding) exclusion remain; v4 body files use the
same bounded cold inventory independently of execution support. See the
[extension contract](contracts/extensions-training.md#owned-tensor-models) and
[ADR0046](../design/decisions/0046-general-scan-cold-source-and-owned-model-direction.md).

The [owning NN contract](contracts/extensions-training.md#owned-tensor-models) places high-level
owned state and exclusion in [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context), without moving execution or [optimizer](../glossary.md#term-optimizer) algorithms there.
`Model.defineTensor` acquires a source after one eval [prototype](../glossary.md#term-prototype) forward; `Model.graph` acquires an
explicit boundary. Both copy each external state [identity](../glossary.md#term-identity-meanings) once through GraphCheckpoint and keep
one working set for compute, configured train and save. Load adopts final [heap](../glossary.md#term-heap) values directly.

```text
NN owned source -> Engine compile/prepare/runValues -> detached values
               -> TrainingSession.runValues       -> same weights + detached step
               -> GraphCheckpoint.save            -> source + current raw values
```

Each arrow delegates to the existing public [owner](../glossary.md#term-owner); save additionally requires wire [admission](../glossary.md#term-admission).
General bodies use the bounded v4 inventory below, not arbitrary [PURE](../glossary.md#term-pure) operations.
For a supported source without fixed scans, configuring training after compute
closes [inference](../glossary.md#term-inference) preparation before one native [parameter](../glossary.md#term-parameter) admission, preserving Tensor identity.
Rebuilding a source or replacing a [weight](../glossary.md#term-weight) per [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) would invalidate this [lifecycle](../glossary.md#lifecycle), not refresh it.
Mode, wrappers, paths and storage associations are guarded; inherited mutators remain caller-excluded.
Close waits for active work, then [releases](../glossary.md#commit-release-distribution-log-and-seed-meanings) sessions/preparations, [native memory](../glossary.md#aliasing-and-arena) and only a lazily
owned [Engine](../glossary.md#term-engine). A selected Engine is [borrowed](../glossary.md#ownership-borrowing-and-lease). See the [owned workflow](../user-guide/running-models.md#owned-static-model-workflow).

Selected default CNNs and static recurrent builders use the same source save path, not per-network
serializers. Their zero-state and row-index [constants](../glossary.md#term-constant) remain named CONSTANT [leaves](../glossary.md#term-leaf) outside the
[Module](../glossary.md#term-module) state dictionary. Advanced list Models may instead declare [runtime](../glossary.md#term-runtime) [INT64](../glossary.md#data-type--datatype) lengths as inputs
for cold scan copy/save/load/rebind. The inference-only scan builder uses that same boundary
and execution path, not a new runner.

For example, the builder workflow takes FLOAT32 `[4,3,2]` values
and INT64 `[3]` lengths `[4,2,0]`, composes `gruScan(8).lstmScan(6).finalHidden().linear(5)`,
and ends with `buildWithLengths()`. Each scan starts from its own zero state and consumes the
shared lengths. The final hidden port has [shape](../glossary.md#shape) `[3,6]`; the head produces `[3,5]` [detached](../glossary.md#detached-model-results) scores
through `compute(values, validLengths)`, also after save/loadGraph. The zero-length hidden row is
zero, but the ordinary head's [bias](../glossary.md#term-bias) may make its scores nonzero. `lastTimeStep()` would instead
select dense time index 3, not the semantic final hidden state for short rows.
Only scans skip padded recurrent transitions; ordinary layers retain dense semantics, without a
global [mask](../glossary.md#term-boolean-mask). Shape-valid composition/[persistence](../glossary.md#term-persistence) is separate from [backend](../glossary.md#backend) execution admission:
CPU [scalar](../glossary.md#term-scalar) scans support four homogeneous floating types and resolved nonnegative read [layouts](../glossary.md#layout);
this does not qualify every ordinary prefix [operation](../glossary.md#operation) or composition. See the
[complete runtime-length example](../user-guide/running-models.md#run-a-runtime-length-scan-model).

Cold list forward recomposes instantiated owned steps only when explicitly requested. Inference
uses fixed outputs and cached Engine preparations; it never reconstructs recipes per batch or
requires a prior forward call. The two-input compute convenience applies equally to an ordinary
two-input/one-output graph and follows declared positions, not inferred length roles. Existing
graph-model v3 stores those ordered input [descriptors](../glossary.md#term-descriptor), without new role [metadata](../glossary.md#term-metadata) or format changes.

Fixed-scan-containing Models must reject configure/train before the training-transition arrow above:
after lifecycle/live-source checks, but before freezing [binding](../glossary.md#term-binding), invoking an [objective](../glossary.md#term-objective) [callback](../glossary.md#term-callback),
creating an Engine, closing inference preparation or copying parameters to native storage. Waiting
for Compiler's [BPTT](../glossary.md#term-backpropagation-through-time-bptt) error would perform those side effects first. Static [unrolled](../glossary.md#backpropagation-through-time-and-recurrent-state) Models still
train normally; fixed scans remain inference-only even in a mixed fixed/general source.
The [owning contract](contracts/extensions-training.md#owned-tensor-models)
and [ADR 0041](../design/decisions/0041-runtime-length-scan-model-inference.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span> define and explain
this limited cutover; backend support, recurrence numerics and execution [ownership](../glossary.md#term-ownership-meanings) do not change.

### `modules/training` ([Gradle](../glossary.md#term-gradle) `:extensions:training`)

Owns public [backend](../glossary.md#backend)-neutral [optimizer](../glossary.md#term-optimizer) algorithms, reusable training sessions, training steps, and
[detached](../glossary.md#detached-model-results) in-memory optimizer/session state. The implemented [lifecycle](../glossary.md#lifecycle) provides `Optimizer`,
`Sgd`, [`Adam`](../glossary.md#adaptive-optimizer--adam-adamw), [`AdamW`](../glossary.md#adaptive-optimizer--adam-adamw), [`ParameterGroup`](../glossary.md#parameter-group--parametergroup), `GradientMode`, [`TrainingSession`](../glossary.md#training-session--trainingsession), `TrainingStep`, and
`TrainingState` over generic live [`TrainingParameters`](../glossary.md#live-parameter-source--trainingparameters) sources. [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s `Module.trainingParameters()`
retains its live tree and exact ordered paths/wrappers/[Tensors](../glossary.md#tensor); no frozen-map [validation](../glossary.md#term-validation) bypass
is allowed. Storage associations and bytes are checked by Training, not frozen by membership.
Ordered groups may assign different
algorithms to [parameters](../glossary.md#term-parameter) within one session and one global [optimizer step](../glossary.md#optimizer-and-stochastic-gradient-descent). Training does not own
`Parameter`, [`Buffer`](../glossary.md#term-buffer), layer behavior, or train/[eval mode](../glossary.md#inference-and-evaluation).

Training has exactly public [Model](../glossary.md#term-model)/[Engine](../glossary.md#term-engine) edges and no NN [dependency](../glossary.md#term-dependency); NN has exactly public
Model/Training/[Checkpoint](../glossary.md#term-checkpoint)/Engine edges. A Training-only caller can supply an ordinary Tensor [owner](../glossary.md#term-owner) without a [Module](../glossary.md#term-module).
Training uses one Engine [compile](../glossary.md#compile) and one Engine [prepared execution](../glossary.md#prepared-execution--preparedexecution); it
must not depend on [Runtime](../glossary.md#term-runtime), [Prepare](../glossary.md#prepare), tuning, or [concrete backend](../glossary.md#concrete-backend) modules or implement another
execution path. A [fused](../glossary.md#specialization-and-fusion) optimizer route on [Metal](../glossary.md#term-metal), for example, remains a Metal backend
prepare/[kernel](../glossary.md#kernel) concern rather than a `MetalOptimizerBridge` in training. See [Training
Graph](training-graph.md)<span data-glossary-navigation="true"> ([Training Graph glossary](../glossary.md#training-graph))</span>.

Training does not own fixed recurrent execution, valid-length handling, or [backpropagation through
time](../glossary.md#term-backpropagation-through-time-bptt). [Compiler](../glossary.md#term-compiler) owns the later [BPTT](../glossary.md#backpropagation-through-time-and-recurrent-state) decision and [gradient](../glossary.md#term-gradient) construction; Training remains the
consumer of published gradients and owner of optimizer orchestration.

<a id="extensionscheckpoint"></a>

### `modules/checkpoint` ([Gradle](../glossary.md#term-gradle) `:extensions:checkpoint`)

Generic [Checkpoint](../glossary.md#term-checkpoint) owns cold [graph](../glossary.md#graph)/value copy/save/load/[rebind](../glossary.md#capture-graph-ports-and-reconstruction) through GraphCheckpoint and the
[opaque](../glossary.md#term-opaque) [artifact](../glossary.md#term-artifact) package. Its public [Model](../glossary.md#term-model) edge and implementation-only [Compiler](../glossary.md#term-compiler) edge expose no
[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training/[Engine](../glossary.md#term-engine) [dependency](../glossary.md#term-dependency). A static input [prototype](../glossary.md#term-prototype) contributes only its [descriptor](../glossary.md#term-descriptor); named
[host](../glossary.md#term-host) [leaves](../glossary.md#term-leaf) contribute raw values by exact [identity](../glossary.md#term-identity-meanings). Load returns ordinary symbolic [Tensors](../glossary.md#tensor),
without execution or a second [ownership](../glossary.md#term-ownership-meanings) copy. See the [graph API](../reference/checkpoint/graph.md)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span>.

General-body in-memory copy/rebind now retains each [immutable](../glossary.md#term-immutable) scanBody attachment unchanged while
normalizing only outer [node](../glossary.md#node)/value IDs. Coincident numeric IDs do not cross body scopes; Compiler
alone preflights and reconstructs fresh body formals and [canonical](../glossary.md#term-canonical-form) ports per [occurrence](../glossary.md#term-occurrence). For
example, two uses of one source template still become two independent reconstructed bodies.
The exact [cold constructor inventory](contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)
is independent of [CPU](../glossary.md#term-cpu) [dtype](../glossary.md#data-type--datatype) support, not arbitrary [PURE](../glossary.md#term-pure)-body copying. V1/v2/v3 remain independently
closed to bodies. Implemented [v4 persistence](contracts/extensions-training.md#canonical-graph-model-version-4)<span data-glossary-navigation="true"> ([persistence glossary](../glossary.md#term-persistence))</span>
uses that same cold inventory; unsupported bodies reject before host copy/file effects.
NN uses it for graph/definition acquisition and save/load, while execution
still requires its independently qualified body and binder domain.

For example, a three-step [FLOAT32](../glossary.md#term-float32) `tanh(carry + step * weight)` source uses v4 with one inline
body and one raw named [weight](../glossary.md#term-weight) [payload](../glossary.md#artifact-envelope-and-atomic-publication), not three copies of the body or saved [activations](../glossary.md#term-activation). Loading
must delegate fresh scoped construction to Compiler; a Checkpoint interpreter or constructor table
would duplicate family ownership and risk [scope](../glossary.md#term-scope) [aliasing](../glossary.md#term-aliasing). Focused tests cover cold roundtrips and
loaded FLOAT32/[FLOAT64](../glossary.md#term-float64) execution/training; main accepted CP0013's final gates.
The [persistence decision](../design/decisions/0047-bounded-general-scan-graph-persistence.md) records
the compatibility tradeoff and pre-code gate. No new [module](../glossary.md#term-module) edge or execution [owner](../glossary.md#term-owner) is involved.

Checkpoint owns [v3 wire and normalization](contracts/extensions-training.md#canonical-graph-model-version-3)<span data-glossary-navigation="true"> ([normalization glossary](../glossary.md#term-normalization))</span>
and [v4 body records](contracts/extensions-training.md#canonical-graph-model-version-4),
not [operation](../glossary.md#operation) semantics or Tensor reconstruction. All ordered [producer](../glossary.md#term-producer) ports receive distinct
implicit value positions; iterative producer-aware [topology](../glossary.md#term-topology) retains unused sibling outputs while
rejecting dead producers. For example, a finalCell-only [LSTM](../glossary.md#term-long-short-term-memory-lstm) root still retains one three-output
producer. Descriptor-only source [preflight](../glossary.md#term-preflight) precedes named host-state import, and recapture precedes
return. No optimized CompileArtifacts, prepared/native/[backend](../glossary.md#backend)/[runtime](../glossary.md#term-runtime) state or input prototype
bytes are persisted. Published old graph bytes/readers remain compatible; execution and [optimizer](../glossary.md#term-optimizer)
[resume](../glossary.md#schema-serialization-and-checkpoint) are separate owner decisions.

### Retained [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [checkpoint](../glossary.md#term-checkpoint) glue

The old nine [model](../glossary.md#term-model) [persistence](../glossary.md#term-persistence) classes now belong to `extensions/nn`, package `nn.checkpoint`.
External old-model users need an explicit NN [dependency](../glossary.md#term-dependency) and migrated imports; no compatibility
forwarders exist. The following old-format behavior is unchanged.
Its [artifact](../glossary.md#term-artifact) [API](../glossary.md#term-application-programming-interface-api) saves and reads one bounded file of [opaque](../glossary.md#term-opaque) named byte payloads.
The file is forced and fully validated before one [atomic move](../glossary.md#term-atomic-move); success gives [atomic](../glossary.md#resource-transaction-and-rollback) live
visibility, not portable power-loss persistence of the rename. It does not interpret a [payload](../glossary.md#artifact-envelope-and-atomic-publication)
as a [Tensor](../glossary.md#tensor) or model state. The public model [facade](../glossary.md#term-facade) adds that interpretation: it [captures](../glossary.md#term-capture-meanings) one
complete NN [snapshot](../glossary.md#term-snapshot), materializes through bounded public [Engine](../glossary.md#term-engine) compute, and exposes [detached](../glossary.md#detached-model-results)
typed inspection. Load checks the caller fingerprints and complete file, then reconstructs [host](../glossary.md#term-host)
Tensors and asks NN to install the entire state dictionary. A fully bound [target](../glossary.md#term-target) defines v1
[alias](../glossary.md#aliasing-and-arena) groups; unbound/[lazy](../glossary.md#term-lazy-evaluation) targets fail. NN paths stay in [metadata](../glossary.md#term-metadata) while fixed ordinal artifact
keys carry bytes. For example, a [buffer](../glossary.md#term-buffer) named `重み` is stored as metadata path `重み` while its
bytes reside at `state/00000000`. See the [model checkpoint API](../reference/checkpoint/models.md),
[artifact API](../reference/checkpoint/artifacts.md),
[envelope decision](../design/decisions/0032-checkpoint-artifact-envelope.md), and
[model schema decision](../design/decisions/0033-model-candidate-schema.md). The
[public model checkpoint decision](../design/decisions/0034-public-model-checkpoint.md)
explains typed inspection, the direct Model dependency, and target-defined aliases.

The model-envelope API extends this same facade to [capture](../glossary.md#term-capture-meanings), inspect,
read [canonical](../glossary.md#term-canonical-form) state bytes, and load the existing `ArtifactEnvelope` without file I/O. The
[owning NN contract](contracts/extensions-training.md#retained-nn-checkpoint-apis) and
[ADR 0036](../design/decisions/0036-model-envelope-composition.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span> define that boundary.
The internal candidate and [codec](../glossary.md#term-codec) remain package-private; file calls keep their reread and [publication](../glossary.md#term-publication)
policies. Reusing the decoder prevents a standalone envelope from bypassing limits or [BOOL](../glossary.md#data-type--datatype)
[validation](../glossary.md#term-validation) merely because it did not pass through the file reader.

The [accepted whole-model contract](contracts/extensions-training.md#whole-model-definition-and-composition)
adds semantic definition bytes downstream of NN without changing edges. This CP0005B capability
is implemented and reviewed (Complete, 2026-10-15). It keeps existing ModelCheckpoint, accepts
ordinary supported Sequential or DeclarativeModel roots, and reconstructs only NN's closed [matrix](../glossary.md#term-matrix-meanings). Values-only APIs remain the
advanced foundation; they do not acquire automatic reconstruction of arbitrary Model callbacks.
The separate whole-model kind binds canonical definition bytes, complete values and exact aliases.
NN now owns old model bytes/versions/[preflight](../glossary.md#term-preflight) as well as meaning and strict installation;
Checkpoint retains only opaque artifacts and the separate generic [graph](../glossary.md#graph) format. Reusing public
component validation avoids inventing a trusted-image credential or private parsing interface.

Whole-model scenario: a [quiescent](../glossary.md#term-quiescent) Sequential contains two [FLOAT32](../glossary.md#term-float32) [Linear](../glossary.md#term-linear-meanings) layers whose
[weight](../glossary.md#term-weight) paths `0.weight` and `1.weight` share the same Tensor. Whole-model save retains an explicit
[identity](../glossary.md#term-identity-meanings) group as well as both per-path values; load first compares [schema](../glossary.md#term-schema)/configuration and group
bytes, then returns an alias-preserving [heap](../glossary.md#term-heap)-backed eval DeclarativeModel. Equal independent
weights remain independent. A corrupted group with unequal bytes fails before returning a model.
Saving a custom Model.define [callback](../glossary.md#term-callback) instead rejects before Engine/file work: [shapes](../glossary.md#shape) and weights
cannot establish that Java code's meaning. No Training state or native [run](../glossary.md#term-run) [lifetime](../glossary.md#term-lifetime) is implied.

Conceptual adapter scenario (requires a separately approved adapter): a model envelope contains
[parameter](../glossary.md#term-parameter) `layer.weight`, FLOAT32 Shape.of(2), with canonical bytes `3f800000 80000000`.
Inspection supplies its [descriptor](../glossary.md#term-descriptor); canonical-byte access supplies those exact bytes by NN path.
An optional adapter compares the complete parameter set, descriptors, and bytes with TrainingState.
A TrainingState value ending in `00000000` instead differs: positive and [negative zero](../glossary.md#nan-infinity-and-signed-zero) are not
interchangeable stored values. Rejection precedes publication of the private replacement.
Matching bytes alone do not establish [optimizer](../glossary.md#term-optimizer)/session compatibility; Training still validates
restore, and NN still owns installation. The model operations are available; the adapter scenario
is not a runnable current API. The [file-free API example](../reference/checkpoint/values-examples.md#file-free-example)
demonstrates public descriptor and raw-byte access without implementing that adapter.

[Exact training continuation](../glossary.md#exact-training-continuation) is a different problem: it also requires Training-owned state and
caller-owned progress or random state. An optional downstream Training Checkpoint adapter is
reserved for that coordination after a separate gate; it is not included or implemented today.
Putting that adapter's state into the model-only project would couple [inference](../glossary.md#term-inference) persistence to
Training restore rules and blur which [owner](../glossary.md#term-owner) validates restored values. NN's narrow [admission](../glossary.md#term-admission) edge
does not grant Checkpoint that [ownership](../glossary.md#term-ownership-meanings). See the [checkpoint boundary
decision](../design/decisions/0031-model-checkpoint-extension-boundary.md).
The selected first adapter persists only Model + TrainingState and restores saved state, not
automatic exact continuation. Modes, effective contexts, input cursor, external [RNG](../glossary.md#graph-rng-state--graphrngstate), scheduler,
and scaler stay application-owned. The adapter remains Draft, absent from the build, and
unimplemented; 0005A does not authorize its codec or inclusion.

### `extensions/onnx`

Owns [ONNX](../glossary.md#term-onnx) import, export, and mapping to and from the [model](../glossary.md#term-model). It remains outside the [runtime](../glossary.md#term-runtime) [hot path](../glossary.md#term-hot-path) and does not own [backend](../glossary.md#backend) [lowering](../glossary.md#lowering), [kernel](../glossary.md#kernel) selection, execution, or [residency](../glossary.md#residency).

## Tooling

`tools/benchmarks` owns fixed reproducible workloads and observational `BenchmarkReport`
evidence. It compares [commits](../glossary.md#commit-release-distribution-log-and-seed-meanings), models, and environments without selecting or mutating production
settings.

`tools/tuning` coordinates one explicit [model](../glossary.md#term-model)-[autotuning](../glossary.md#autotuning--model-autotuning) workflow. It reuses or measures [canonical](../glossary.md#term-canonical-form)
local workload [signatures](../glossary.md#term-signature), then measures a bounded set of complete valid [graph](../glossary.md#graph) and plan
candidates end to end. [Compiler](../glossary.md#term-compiler), [planning](../glossary.md#term-planning), [prepare](../glossary.md#prepare), and concrete [backends](../glossary.md#backend) generate candidates for
their own decisions. Tuning owns measurement, explicit [cache](../glossary.md#term-cache) coordination, and selection, not
semantics, [ownership](../glossary.md#term-ownership-meanings) policy, [lowering](../glossary.md#lowering), or route vocabulary.

Future results use an explicit file-backed reusable workload cache plus a model-specific plan
cache or prepared-plan record. A representative model corpus may pre-[seed](../glossary.md#term-seed) the same workload cache;
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

The [dependency](../glossary.md#term-dependency) implications of these responsibilities are detailed in [Dependency Rules](dependency-rules.md).
