# Synaptik Architecture Contract

This file is the authoritative architecture root and the repository's sole authority index. The
global rules in this root and the six scoped contracts explicitly incorporated below together form
the complete architecture contract.

All contributors and agents must preserve these boundaries unless this root and, when scoped
detail is affected, the one owning scoped contract are explicitly updated as part of the same
change.

For agent working instructions, see [`AGENTS.md`](AGENTS.md). For explanatory architecture
navigation, see [current architecture documentation](docs/architecture/current-architecture-plan.md).

## Authority, incorporation, and precedence

Only the following six files are incorporated as normative, and only within their stated,
non-overlapping scopes:

1. [Foundational module and numerical-semantics contract](docs/architecture/contracts/foundational-modules.md)
2. [Fixed and general differentiable-scan contract](docs/architecture/contracts/recurrent-scan.md)
3. [Compiler and automatic-differentiation contract](docs/architecture/contracts/compiler-autograd.md)
4. [Runtime, Prepare, and Engine contract](docs/architecture/contracts/runtime-prepare-engine.md)
5. [Backend execution contract](docs/architecture/contracts/backend-execution.md)
6. [Extensions and training contract](docs/architecture/contracts/extensions-training.md)

All other files under `docs/`, including architecture explanations, architecture decision records
(ADRs), design notes, guides, plans, and task briefs, are explanatory or coordinative evidence.
Source comments and tests are also evidence. None can override the architecture contract.

The global invariants and dependency rules in this root apply everywhere and take precedence over
a scoped contract on conflict. Each scoped contract is authoritative only for its stated scope and
must point back here; no scoped file has independent authority. Any overlap, contradiction,
missing applicable scope, or ambiguity requires contributors and agents to stop and make an
explicit coordinated architecture update. They must not choose an interpretation silently.

## Java namespace

All production Java packages must use:

```text
io.github.pho001.synaptik.*
```

Gradle module directories may remain short and do not need the `synaptik-` prefix.

Examples:

```text
modules/compiler
modules/runtime
modules/backend/cpu
modules/backend/metal
modules/training
extensions/data
extensions/text
extensions/nn
```

Example Java packages:

```text
io.github.pho001.synaptik.model
io.github.pho001.synaptik.compiler
io.github.pho001.synaptik.runtime
io.github.pho001.synaptik.backend.cpu
io.github.pho001.synaptik.backend.metal
io.github.pho001.synaptik.data
io.github.pho001.synaptik.text
io.github.pho001.synaptik.nn
```

## Repository layout

The intended high-level repository layout is:

```text
ComputationalGraph/
  AGENTS.md
  ARCHITECTURE.md
  README.md
  build-logic/

  docs/
    index.md
    getting-started.md
    architecture/
    design/
    user-guide/
    backend-guide/
    developer-guide/
    api/

  modules/
    trace/
    backend-contract/
    model/
    config/
    planning/
    compiler/
    runtime/
    prepare/
    engine/
    training/
    checkpoint/
    backend/
      openblas-provider/
      cpu/
      metal/
        macos-arm64/
      cuda/

  extensions/
    data/
    text/
    nn/
    onnx/

  tools/
    tuning/
    benchmarks/
    cli/

  testing/
    architecture-tests/
    backend-conformance/
    integration-tests/
```

The physical backend group retains separate `:backends:*` Gradle projects through explicit
settings mappings, including the `:backends` parent at `modules/backend`. The backend-neutral
`modules/backend-contract` remains separate. CUDA has only its existing marker and merged native
README scaffold; no executable native CUDA project or platform tree is implied by this layout.
Training is physically `modules/training`, retaining Gradle project `:extensions:training` through
an explicit settings mapping and retaining its Java packages. Its owning contract remains in
the incorporated extensions/training contract below. Checkpoint is physically `modules/checkpoint`,
retaining `:extensions:checkpoint` through an explicit settings mapping.

Data is authorized at physical `extensions/data`, Gradle `:extensions:data`, with namespace
`io.github.pho001.synaptik.data`. The owning Data contract authorizes position-independent valid
counts, explicit padding side/sequence-axis order, a borrowed padded batch and INT64 host sequence
batching. The implemented padded API replaces the unpublished right-only API and supports
left/right padding and INT64 host allocation. The implemented opt-in
`PaddedSequenceBatch.toScanLengthsTensor()` export materializes fresh ordinary heap INT64 counts
for RIGHT/TIME_MAJOR batches only. Metadata construction
and accessors remain allocation-free with respect to Tensors; Data gains no execution owner.
Text is registered at `extensions/text`, Gradle `:extensions:text`, namespace
`io.github.pho001.synaptik.text`, for a frozen whitespace tokenizer and Data-delegating batcher.
Both are implemented. Vision remains separately gated.

## Core lifecycle

The core lifecycle is:

```text
Tensor expression
  -> GraphCompiler
  -> CompileArtifacts
     - CompiledGraphModel
     - PlannedPartition[]
     - LogicalMemoryPlan
     - PublicationPlan
  -> prepare
  -> BackendPartitionAnalysis[]
  -> PreparedMemoryPlan with assigned buffer/workspace slots
  -> PreparedPartition[]
  -> PreparedSchedule
  -> PreparedExecution
  -> RunState
  -> execute
```

Given an output expression `output` and its caller-owned input Tensor `input`, the current
ordinary public API shape is:

```java
try (Engine engine = Engine.standard()) {
    CompiledGraph graph = engine.compile(List.of(output));
    try (PreparedExecution prepared = engine.prepare(graph);
            RunResult result = engine.run(prepared, List.of(input))) {
        // Inspect publication metadata or explicitly materialize an occurrence while open.
    }
}
```

The Engine owns compile, prepare, and run orchestration. Its `PreparedExecution` is the sole
ordinary public prepared owner and borrows its Engine's composition lifetime; each run result is
a distinct closeable owner. `run` binds exact compiled Tensor identities in arbitrary order;
`runValues` binds fresh values in final compiled input order. Static templates require exact
fully static descriptors. The accepted symbolic-batch contract permits explicitly qualified
templates to bind one named batch B from actual static descriptors through Compiler-owned metadata
admission, retaining fixed feature/type/parameter semantics and immutable code, routes and resource
identities. Runtime receives only numeric recipes/run geometry, never Model or Compiler metadata.
[Prepare0013](docs/planning/modules/prepare/tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result)
implements bounded ordinary homogeneous FLOAT32 CPU forward and actual first-order backward:
rank-at-most-two boundaries plus qualified automatic-differentiation views, generated scalar,
single-range execution, and operation-qualified B=0. Final independent reviews and the native-enabled
integration gate passed; MAIN accepted the capability on 2026-10-24.
The [bounded symbolic Scan contract](docs/architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan)
has a Model source foundation under
[Model0042](docs/planning/modules/model/tasks/0042-symbolic-scan-source-foundation.md), independently reviewed
and MAIN accepted 2026-10-25. Static source remains first-class; named symbolic shapes retain unresolved
layouts. Compiler0014 implements scoped inference, actual FLOAT32/FLOAT64 first-order AD and
fresh all-port metadata binding, independently reviewed and MAIN accepted 2026-10-25 after the
named compatibility checkpoint. Prepare0014 implements bounded homogeneous FLOAT32 symbolic
Scan forward and actual Compiler-generated first-order reverse through Prepare/CPU/Engine,
with fixed T/features/state widths, generated scalar and single-range execution. Final
qualification is tracked in its [task](docs/planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md);
symbolic cold/copy/wire and NN/training remain closed. Engine geometry memoization is not implemented.
The [symbolic-length Compiler qualification](docs/architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates)
is implemented by Compiler0015 for fixed T/features, one named B, FLOAT32/FLOAT64 inference,
actual first-order sibling AD and complete fresh scoped all-port metadata binding. Its completed
acceptance is recorded in the [task result](docs/planning/modules/compiler/tasks/0015-symbolic-length-scan-inference-ad-and-binding.md#result).
Prepare0015 implements the bounded combined Prepare/CPU/Engine path: one named B, fixed T/features,
homogeneous FLOAT32/FLOAT64 forward and actual Compiler reverse, including only the CPU owner's
bounded outer/control prerequisite. Owning validation and independent code/structural review passed;
root validation and the sole bounded probe completed;
[Prepare0015 Result](docs/planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
records evidence and limits; final MAIN acceptance remains pending.
This capability changes no runtime lifecycle or resource protocol; cold/copy/wire,
NN/Training and Engine memoization remain closed.
Model0041's coordinated authority cutover is Complete, MAIN accepted 2026-10-25 after independent APPROVE.
NN/TrainingSession, cold/copy/wire,
native/mixed, wider dtype, variable T, higher-order, vector/parallel and tuning remain separately gated.
[Runtime/Prepare/Engine](docs/architecture/contracts/runtime-prepare-engine.md#modulesruntime)
owns sizing/admission/publication; Compiler and CPU own their respective binding semantics below.
Caller input storage remains borrowed for the result lifetime. Runtime retains unique authority
over persistent prepared resources and run leases; the Engine handle adds no second lease protocol.
Ordinary and advanced Engine result publication is transactional through wrapper construction and
Engine registration. Failed publication rolls back the completed run, preserves the primary failure
and required cleanup suppression, and always releases Engine admission. Result closure releases
Engine/Runtime owners' direct references to fresh invocation payloads even when cleanup fails,
without closing borrowed caller storage; immutable publication metadata and Runtime plan/count/
geometry inspection remain available. [Engine0022](docs/planning/modules/engine/tasks/0022-result-publication-and-closed-payload-cleanup.md#result)
implements these existing guarantees after independent review and actual before/after proof,
without a new lifecycle or API.
The accepted [Engine0023 corrective contract](docs/architecture/contracts/runtime-prepare-engine.md#reentrant-engine-close-and-lease-admission)
requires self-close from an admitted calling thread to fail before shutdown/wait, while cleanup-owner
recursive close is idempotent and other closers retain synchronous completion/failure replay.
Lease-token allocation publishes no lifetime count on failure. Engine0023 implements this correction
in existing owners, with source-matched actual before/after qualification; release acceptance is separate.
The Training module's public `TrainingSession` deliberately depends on this public Engine
lifecycle: it borrows an Engine, owns one Engine prepared execution, and adds optimizer state and
parameter-update orchestration without introducing another compiler, preparer, scheduler, runner,
or backend adapter.
Training's `run` preserves exact-ID, arbitrary-order external binding; `runValues` accepts stored
values in `TrainingSession.inputs()` order with exact full static descriptors. Both insert the
same captured parameters and private cotangent seed and share one optimizer transaction. Fresh
external storage is borrowed only through the synchronous call and internal result closure;
Training retains its stronger rejection of nonempty input overlap.

Owned tensor Models in NN compose this same lifecycle: builder-owned leaves and ordinary composition,
or cold GraphCheckpoint acquisition for definition/graph/load paths, establish one private working
state, then cached public Engine preparations and an optional TrainingSession
consume fresh values. NN owns exclusion and its native Arena, not execution or optimizer algorithms.
Construction, cold forward and save create no Engine. The owning rules are in
[extensions/nn](docs/architecture/contracts/extensions-training.md#owned-tensor-models).

Owned Models compose fixed and general recurrent scans through that same public Engine lifecycle
and may configure first-order training when the actual forward/Compiler-AD/backend closure qualifies.
Explicit `ModelBuilder` scan recipes share runtime lengths and zero initial states;
`buildWithLengths()` returns the existing list-boundary Model. The generic two-input, one-output
compute convenience uses the same cached execution path, including after load. Fixed/general
recipes retain their original storage and rounding semantics; NN neither rewrites them nor owns
a separate adjoint implementation. The [owning NN contract](docs/architecture/contracts/extensions-training.md#owned-tensor-models)
defines exact objective, lifecycle, source, execution and persistence admission.

All-valid general-scan owned Models use existing `Model.graph`/`defineTensor`, `compute` and
`configureTraining`/`train`, with no new execution owner or dependency. Bounded Compiler cold body
reconstruction, Checkpoint attachment-preserving in-memory copy/rebind and NN workflow support are
implemented. Acquisition delegates to that cold inventory independently of CPU capability;
compute and first-order training qualify canonical static homogeneous FLOAT32/FLOAT64 occurrences
through the existing Engine/Training owners. All-valid means every static time step participates,
not arbitrary body support. Implemented general-body persistence uses the bounded
[graph-model v4 contract](docs/architecture/contracts/extensions-training.md#canonical-graph-model-version-4);
codec/consumer evidence and main's accepted final gates are recorded in CP0013. V1/v2/v3
remain body-free. The [owning extension contract](docs/architecture/contracts/extensions-training.md#owned-tensor-models)
defines the separate source, execution and persistence admissions.

The static homogeneous FLOAT32/FLOAT64 length-aware cold intersection in the
[Compiler contract](docs/architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)
is implemented and accepted under Compiler0018, including Checkpoint copy/rebind and inherited
NN graph/definition acquisition and explicit cold forward. Independent code/docs review and
affected-module validation passed. These cold effects alone do not qualify complete owned
numerical workflows. [Checkpoint0014](docs/planning/extensions/checkpoint/tasks/0014-length-aware-general-scan-v5.md#result)
implements selective static length-aware v5 persistence with independent review and loaded
F32/F64 Model numerical proof. Symbolic cold, wider per-value consumers and NN0033's public
recurrent-composition workflow remain separate capabilities.

NN0033 implements additive `RnnSequence.forwardGeneralScan` and the FLOAT32/FLOAT64 `rnnScan`
builder cutover through existing length-aware general Scan. Independent reviews, affected gates
and public forward/repeated-training/v5 save/load/fresh-training proof are accepted in its
[Result](docs/planning/extensions/nn/tasks/0033-trainable-runtime-length-rnn-workflow.md#result).
There is no new execution owner or higher-order grant. Ordinary static recurrence, direct fixed
forwardScan, low fixed construction and GRU/LSTM retain their recipes; qualified fixed and mixed
sources use the same configured TrainingSession and Compiler first-order AD.

## Core invariants

The following invariants must remain true:

- `Tensor` is public model state with immutable identity, descriptor, and expression provenance.
- `Tensor` is not an IR node.
- `Tensor` has no gradient field, backward method, or gradient-lifecycle state.
- `Operation` owns semantic behavior but never backend support.
- `Operation` must not expose `supportedBackends()`.
- Model owns one profile-free semantic contract per operation family and dtype: formulas, guards,
  contributor domains, precision and rounding sites, special values, and observable results.
  There is no numerical selector, public accuracy envelope, runtime numerical certificate, or
  production tolerance policy. Exact structure, represented-value comparison and extrema,
  discrete decisions, casts, raw movement and selection, saved/state values, and publication do
  not acquire arithmetic latitude. Finite arithmetic tolerances qualify backend routes only in
  tests, never Compiler transformations or runtime decisions.
- Denormals-are-zero (DAZ) at named floating arithmetic primitive inputs and flush-to-zero (FTZ)
  at named primitive results are permitted for FLOAT32, BFLOAT16, FLOAT16, FLOAT8_E4M3, and
  FLOAT8_E5M2, including low multiplication, but forbidden for FLOAT64. DAZ uses same-signed zero without changing stored
  bits; FTZ obeys the family's existing zero-sign rule. Neither applies at stored comparison,
  extrema/winner, raw movement, cast, selection, index/mask/state, or exact unary ABS, NEG, SIGN,
  FLOOR, CEIL, and RELU sites. Composite and generated-gradient formulas inherit only their
  named arithmetic sites' permissions; they gain no blanket output tolerance.
- Model-level `FLOAT16` semantics remain at ordinal 6; `FLOAT8_E4M3` and `FLOAT8_E5M2`
  append ordinals 7 and 8 without changing existing identities. Each low format is a distinct
  storage type. Unlike low floating types promote to `FLOAT32`; a `FLOAT64` operand promotes
  to `FLOAT64`. CPU execution follows that promotion; Metal capability remains independent.
  Homogeneous low arithmetic uses `FLOAT32` working/accumulator values and one final ties-to-even narrowing. Raw
  storage/movement, mapping, guards, predicates, indices, masks, state, selection, casts,
  public/saved values, and final conversion remain exact. The family declares arithmetic
  zero-sign, and NaN class/domain rules; NaN payload/sign are not accuracy requirements.
  Operation-local reassociation/FMA and fusion across only unobservable single-use intermediates
  are permitted, never across public, saved, fan-out, predicate, index, mask, selection, or state
  boundaries.
- The user-approved CPU low-precision parity cutover supersedes the temporary CPU0078
  BFLOAT16/FLOAT16 arithmetic lock. BFLOAT16, FLOAT16 and both OFP8 types use the existing
  CPU owners for numerical, exact, derivative, recurrent and symbolic execution. There is no
  runtime unlock flag, second AD/executor or hidden quantization scale. Native oneDNN is
  an occurrence-specific acceleration of this CPU capability, not its availability condition.
  Unsupported native configurations use the ordinary CPU plan selected before execution.
  Metal gains no FP8 capability or changed numerical permissions from this CPU cutover.
- Floating scatter MUL combines its base and every addressed update exactly once; a rounded
  multiplication tree is permitted, including intermediate overflow and ensuing NaN. Integral
  modular multiplication and unaddressed raw cells remain exact.
- First-class SOFTMAX and LOG_SOFTMAX use the Model-defined maximum-shifted finite-slice formula; finite
  exponentials are not evaluated on unshifted inputs merely to reproduce incidental overflow.
  Exact stored-value maximum selection, named working-precision sites, and the families' separate
  NaN, infinity, and empty-slice rules remain mandatory.
  [CPU0077's finite-softmax admission correction](docs/architecture/contracts/backend-execution.md#cpu0077-finite-softmax-admission-parity)
  aligns the F64 finite-input domain with F32 without changing these Model sites or generated
  arithmetic. The CPU owner defines runtime validation and unchanged cache/ABI consequences.
- Canonical checked-in capability ledgers use actual CPU and Metal provider queries. The v1
  `ACCELERATOR` true and false answers, independent low-type queries, negative predicates,
  occurrence identity, and provenance are immutable historical cutover evidence. The profile-free
  v2 ledger instead records actual current provider answers over its representative occurrences;
  it may change after a reviewed post-cutover capability change and is not required to remain
  an identical projection of v1. A new occurrence requires its own dtype-specific true provider
  answer, qualified backend route, updated current ledger and conformance tests, and an append-only
  compatibility identity where changed behavior makes prior inputs incompatible. FLOAT32 support
  alone never grants BFLOAT16 or FLOAT16 support. Former target columns are not provider facts.
  Provider evidence contains no route, runtime, device, or generated-backward ownership.
  Metal native ABI 7 exposes thirteen symbols. The cutover advanced the Metal program to schema
  20 with a 124-byte header, the CPU generator envelope to schema 69, and coordinated Metal
  workload/exact-policy/candidate/compatibility/route/codec identity to 30. Schema 19, CPU schema
  68, and Metal identity 29 are incompatible pre-cutover baselines. Metal identity 30 is the
  historical cutover value; identity 31 is the superseded FLOAT32 `EXP` value, identity 32 is the
  superseded low-`EXP` value, identity 33 is the superseded FLOAT32 `SIGMOID` value, and
  identity 34 is the superseded low-`SIGMOID` value and identity 35 is the superseded native-boundary
  and exact low-value repair value. Identity 36 corrected standalone scalar RELU/MUL; 37 added
  dense-view MATMUL; 38 added materializing RESHAPE; 39 added output-owned overlapping FLOAT32
  image folds for Conv3d training. Current coordinated Metal identity 40 admits shared FLOAT32
  Pool2d/Pool3d overlap through the existing pooling/window closure, retaining FLOAT64/low
  non-overlap boundaries. Native ABI 7, program schema 20, the 124-byte header, thirteen exports
  and the 87,929-byte shader corpus remain unchanged. CPU schema 70 corrected activation extremes
  and low FLOAT32 working sites. The current CPU envelope is schema 71, adding the bounded
  FLOAT32 pointwise general-scan family. CPU0031 adds a distinct shaped specialization within
  schema 71 while preserving old pointwise keys and generated bytes. CPU0032 adds a versioned
  MATMUL-only specialization within the same envelope, preserving pointwise and shaped keys/bytes;
  canonical identity remains geometry-free. CPU0033 adds homogeneous FLOAT64 scan specializations
  within schema/family 71, retaining the shaped-v1/MATMUL-v1 projections and payloads and accepted
  FLOAT32 keys/bytes; existing dtype and baked resource facts distinguish the typed variants.
  CPU0034 advanced only the baked shaped and MATMUL specialization payloads to v2 for typed
  coordinate induction and proved post-step emission-carry transitions in both dtypes. CPU0035
  advanced those same sole baked payloads to v3 for proved redundant factor, address and binding
  cleanup in both dtypes. CPU0036 advances those payloads to v4 for top-tested staged loop control
  and dominating final-copy bases. Both specialization identity streams and staged keys/bytes
  advance; global schema/family 71,
  geometry-free canonical v1 projections and existing pointwise/fixed identities remain unchanged.
  Staged v1/v2/v3 artifacts are now stale; earlier retention statements describe historical acceptance.
  CPU0039 advanced only symbolic Scan's baked payload to `|scan-symbolic-batch-v2=` for
  coefficient-proved fixed geometry, int-bounded symbolic contractions and generation-only
  cell-scoped affine offset sharing. CPU0041 advances only that payload to the current
  `|scan-symbolic-batch-v3=` for direct immutable entry-carrier bindings on produced cuts backed
  by demanded ports. Checked foreign-memory access and independent formal-carry selection remain
  unchanged. Both specialization streams reject symbolic baked v1/v2 artifacts; canonical
  `:symbolic-batch-v1`, schema/family 71, static shaped/MATMUL v4 and
  pointwise/fixed/ordinary keys and generated bytes remain unchanged.
  CPU0042 adds fixed-geometry length mode with canonical `:length-aware-v1` and baked
  `|scan-length-aware-v1=` in both specialization streams. Schema/family 71 and all retained
  all-valid static/symbolic/fixed/ordinary keys and generated bytes remain unchanged.
  Prepare0015 implements separate combined
  `:symbolic-length-v1` and bounded outer
  `ordinary-batch-typed-v1` namespaces under the
  [CPU owner](docs/architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract).
  CPU0044 advanced only the combined baked payload to v4 for four proved code-shape redundancy
  fixes. CPU0045 advanced that payload to `|scan-symbolic-length-v5=` in both
  specialization streams for cell-scoped read/write offset sharing and proved zero-base omission
  after raw initialization of a stable own-final-port carry without a bank. Each derived view
  retains its own nominal stride recipe; changing/bank-backed and noncombined bindings are unchanged.
  CPU0058 advanced only this payload to `|scan-symbolic-length-v6=` for nonbanked
  scratch cuts whose complete `offset + offsetPerBatch * B` recipe has both coefficients zero.
  Their exact origin retains its entry scratch binding through the inclusive cut lifetime;
  derived views keep their own relative mappings and cell-scoped offset caching is unchanged.
  Port/time-backed and bank-backed cuts do not gain this scratch proof.
  CPU0060 advanced only this payload to `|scan-symbolic-length-v7=` for flat scalar
  initialization, initialized empty-length publication, coordinate-nested affine prefixes,
  proved active-row direct contractions and raw-singleton SUM-to-Shape loops.
  CPU0064 advances only this payload to current `|scan-symbolic-length-v9=` for direct stable
  carrier/temporal-base bindings, affine geometry derived from admitted explicit B, proved
  indexed contractions and bank-free scheduling checked against dependencies, physical output
  aliases and inclusive scratch lifetimes. Unproved cases retain their existing paths.
  Combined baked v1 through v8 artifacts are stale; v8 was withdrawn. Canonical `:symbolic-length-v1`,
  the `typed-explicit-b-v1` numeric pack,
  schema/family 71 and all noncombined identities and generated bytes remain unchanged.
  The implemented [CPU0071 static MATMUL compatibility cutover](docs/architecture/contracts/backend-execution.md#cpu0071-approved-static-matmul-compatibility-cutover)
  narrowly qualifies the static retention promises in this identity history.
  Independently reviewed and MAIN accepted on 2026-10-29, it changes only the two approved
  static retained-MATMUL namespaces; unaffected controls retain their identities and bytes.
  The owning section defines its exact limits without changing Scan semantics or Runtime ownership.
  The user-approved [CPU0073 row-map/multiple-terminal mechanism](docs/architecture/contracts/backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)
  is implemented, independently reviewed and MAIN accepted on 2026-10-30. It uses only selected
  eligible ordinary class projection 72 and selective ordinary/Scan row-fusion-v1 suffixes; global envelope 71 and
  Scan projection 71 stay unchanged. Its owning section defines the exact types, signatures,
  proof/resource boundaries and noneligible key/ClassFile retention; no broader exception follows.
  No broader dtype/family capability follows from these identities.
  CPU0074's [selected row-region execution exception](docs/architecture/contracts/backend-execution.md#cpu0074-selected-row-region-execution-exception)
  is implemented, independently reviewed and MAIN accepted on 2026-10-30. Only new selected
  exec-v1 realizations may use bounded feature SIMD or synchronous
  row-partial workers; scalar CPU0073 remains unchanged except for the user-approved
  [CPU0075 selective scalar emission/cache cutover](docs/architecture/contracts/backend-execution.md#cpu0075-selective-scalar-load-and-address-reuse).
  That implemented, independently reviewed cutover replaces only affected ordinary/Scan scalar row
  emission for within-row external-load reuse and constant address folding/sharing, with scoped
  markers in both identity streams. Unaffected SIMD/nonregional artifacts, global schema,
  projections, mathematics, APIs, ABIs and lifecycle remain unchanged; no parallel old emitter
  is retained. CPU0075's [Result](docs/planning/backends/cpu/tasks/0075-row-fusion-scalar-load-address-reuse.md#result)
  records final validation, reduced load sites and JIT-qualified timing; no material Scan speedup
  or C1 repair is established. The CPU0074 exception permits only its typed
  run-bound dispatch and scheduler metadata, not a body interpreter, another executor or public
  numerical mode. One RunState owns the context and scratch; all started workers join before
  merge, carry transition, scratch reuse or reservation release. Runtime's synchronous ownership
  and staged Prepare remain unchanged. Functional capability is Complete; bounded performance
  comparison was collected, not steady-state profitability qualified. Scalar/thread-count-one
  defaults remain unchanged; no automatic rollout or universal speed/parity claim follows.
  The user-approved [CPU0076 invariant external-cell staging exception](docs/architecture/contracts/backend-execution.md#cpu0076-prospective-invariant-external-cell-load-staging)
  qualifies CPU0075's within-row-only prohibition and scalar retention. It is implemented and
  independently source/protocol/result/documentation reviewed; final CPU/Javadoc and named
  Engine gates passed, as recorded in the [CPU0076 Result](docs/planning/backends/cpu/tasks/0076-row-invariant-load-staging.md#result).
  It permits only proved scalar load placement in existing owners;
  Scan activity/traversal, separate per-time carry ADDs, existing resource limits and helper
  ABIs remain unchanged. Complete-body/rank and lifetime-budget failures retain checked emission;
  bounded B256 improvement does not establish universal speed, parity or C1 repair.
  Schemas 70 and earlier remain stale. Changed
  serialized/generated identities advance append-only
  and reject stale inputs. The Metal profile
  wire is absent in both Java and native schema-20 decoding; source and package authentication
  remain required.
  Every low-precision PREPARE trace reports the selected custom route and ordered logical boundary
  dtype tuple. It has no accumulator or working-type field: those Model
  arithmetic guarantees are semantic rules, not diagnostic route facts. The trace reports
  execution facts, not hypothetical candidates or removed certificate state. Metal has no
  executable/compilation/preparation cache. Tuning compatibility is session-only and binds identity
  versions, ABI, a random context nonce,
  versioned program/plan semantics, explicit dtype wires and descriptors; candidate and
  complete-plan bytes additionally bind the selected route and phase-one decision. A new context
  has a new nonce, and tools never persist session-scoped decisions.
- Engine, Config, Planning queries, Compiler artifacts, Prepare contexts, Trace, and backend/native
  identities carry no numerical-profile selector. CPU routes obey the temporary CPU-only
  low-arithmetic lock above. Metal's
  currently supported domain includes exact unary, affine,
  canonicalization, indexing, BOOL-domain, movement, replacement/fold/aggregate,
  ordering/top-K/arg-extrema, comparison, and maximum-pooling occurrences. It also
  realizes no-gradient INT32/INT64 `MATMUL` pairs with INT64-dominant promotion and modular result
  arithmetic. The supported homogeneous FLOAT32 occurrences preserved at the cutover retain their
  independently queried BFLOAT16 and FLOAT16 ownership where declared; later FLOAT32 additions
  do not imply either low answer. Existing supported low occurrences remain supported. The
  preserved low domain includes the documented arithmetic, scalar, reduction, scan,
  MATMUL, MSE, convolution, average-pooling, dropout, L1, ScatterAdd, and singleton-variance
  domains. Direct BFLOAT16/FLOAT16 mixed-low operations remain unsupported; explicit casts to
  FLOAT32 are the sole mixed-low path. At the Task-0074 checkpoint (2026-10-09), the bounded
  no-gradient canonical FLOAT32 `EXP` occurrence used a fixed direct MPSGraph route; independently
  queried BFLOAT16 and FLOAT16 `EXP` provider answers were false, as were FLOAT64 and
  gradient-bearing `EXP` answers. A later homogeneous BFLOAT16 or FLOAT16 `EXP` occurrence could be
  admitted only after its dtype has a separately qualified custom step satisfying Model semantics,
  FLOAT32 working evaluation with one final low narrowing, and the provider, current-ledger,
  conformance, and compatibility-identity requirements above. Task 0075 has now qualified both
  homogeneous low types independently in that bounded canonical, positive-static-rank `1..16`,
  no-gradient, same-type/same-Shape domain. Current v2 answers are true for both; historical v1
  `EXP` remains false. FLOAT64, gradient-bearing, scalar-rank, noncanonical, and over-limit `EXP`
  remain unsupported. FLOAT32 `EXP` keeps its separate direct MPSGraph route; each low-valued
  `EXP` uses a typed custom step with FLOAT32 working evaluation and one final low narrowing.
  Task 0076 admitted bounded canonical no-gradient FLOAT32 `SIGMOID` through the fixed
  sign-guarded composed MPSGraph route at wire 64. Its exact stored-bit guard selects the stable
  Model branch. Task 0077 now admits each homogeneous BFLOAT16 and FLOAT16 `SIGMOID`
  independently in the same bounded canonical no-gradient domain, with checked two-byte spans.
  Current v2 records all three types true; historical v1 stays false. Each low type has its own
  sign-guarded custom step: exact stored-bit guard, one FLOAT32 `EXP`, branch-selected numerator,
  FLOAT32 addition/division, and one final low narrowing. FLOAT64, gradient-bearing, scalar-rank,
  noncanonical, mixed-low, and over-limit `SIGMOID` remain false. An explicit low-to-FLOAT32 cast
  may place the FLOAT32 step inside a low-containing custom partition without changing its route.
  Every partition containing BFLOAT16 or FLOAT16 values selects one fixed custom program and
  exposes only the `CUSTOM_PROGRAM` candidate.
  This includes raw-preserving `RESHAPE`, simple `PERMUTE`, materializing `CONTIGUOUS`, `SLICE`,
  `CONCAT`, and `TILE`. The partition route remains `CUSTOM_PROGRAM`: every operation consuming
  or producing a BFLOAT16 or FLOAT16 value uses a custom step, never MPSGraph or a hidden
  BFLOAT16-to-FLOAT16 substitution. Any FLOAT32-only operation in that same partition may use an
  internal MPSGraph boundary step under the existing FLOAT32 policy, whether on an independent
  FLOAT32 branch or after an explicit low-to-FLOAT32 cast; this does not add a partition candidate
  or change its route. No generated-pointwise or anchor-epilogue fusion occurs anywhere in a
  low-containing partition. Classic MPS, MPP, CPU, retry, and fallback are not low-precision
  alternatives.
  The implemented MATMUL operand baseline is canonical storage or authenticated local
  identity-prefix, last-two-axis transpose. The accepted
  [Metal0080 dense-view extension](docs/architecture/contracts/backend-execution.md#accepted-metal-dense-view-matmul-extension)
  also permits exact row-major, zero-offset operands with span equal to logical count regardless
  of logical isView, without broadening generic canonicalStorage or erasing physical/provenance
  facts. Its reviewed profile37 candidate preceded the identity38 materializing-RESHAPE cutover;
  identity39 added FLOAT32 image-fold accumulation for Conv3d training; current identity40 also
  admits shared FLOAT32 Pool2d/Pool3d overlap.
  Existing kernels, routes, typed bounds and independent validation remain.
  Conv1d and Pool1d use only authenticated local singleton-height compositions. Existing rank-two
  FLOAT32 matrix products retain direct MPSGraph; newly admitted MATMUL forms and
  convolution/pooling rows use the fixed custom program. Gradient-bearing low arithmetic, scalar,
  reduction, scan, dropout, non-overlapping maximum-pool, and other admitted differentiable
  occurrences retain compiler-generated first-order and owned higher-order closure. The
  implemented Conv3d training extension reuses Compiler's grouped convolution algebra and
  output-owned FLOAT32 image-fold accumulation, qualified on a Metal-only Engine. FLOAT64 and low
  image folds retain non-overlap placement; overlapping FOLD_AXIS, attention, convolution transpose,
  and every unsupported operation occurrence remain fail-closed.
  Metal indexing validates complete bounds and
  scatter target uniqueness before dispatch or target writes and leaves targets unchanged on
  failure. Canonical cross-owner transfer supports all seven current data types at ranks `0..16`;
  transfer coverage does not widen operation capability. Runtime and Trace remain profile-free.
- The accepted [ordinary materializing RESHAPE extension](docs/architecture/contracts/backend-execution.md#accepted-ordinary-materializing-reshape-extension)
  and [ADR0062](docs/design/decisions/0062-ordinary-materializing-reshape-closure.md) distinguish
  final canonical non-view values from exact inferred views, without changing raw source/wire
  inference or ordinary derivatives. Complete Compiler0019 owns final logical geometry and Complete
  CPU0079 owns existing affine-copy admission. The Metal0081 identity38 checkpoint passed its
  source-matched native/package/GPU and unchanged original-typed consumer proof; current Conv3d
  and Pool3d training use identity40, with the joined release checkpoint remaining separate. No consumer copy,
  low-arithmetic unlock or blanket CPU identity bump follows.
- `CompiledGraphModel` is immutable compile-time graph state.
- The sole bounded-body exception is the fixed-count general differentiable scan specified by
  [the recurrent-scan semantic owner](docs/architecture/contracts/recurrent-scan.md#general-differentiable-scan).
  It permits ordinary Tensor source-body containment and one independently scoped compiled body
  per occurrence, not arbitrary graph regions, while loops, nested scans, callbacks, or another
  execution owner. That owner also defines Model source admission separately from execution
  qualification and conservative outer result eligibility. Model source/IR, Planning query
  transport, Compiler scoped capture/first-order AD/demand and Prepare scoped projection/static
  binding/demand validation are implemented. CPU0030 implements bounded FLOAT32 same-shape
  pointwise forward execution and its independently qualified actual Compiler-generated reverse
  sibling. CPU0031 adds qualified homogeneous FLOAT32 broadcast, SUM/sumToShape and bounded
  shape/layout closure, including independently qualified actual reverse bodies. CPU0032 adds
  static homogeneous FLOAT32 rank-two MATMUL inside that emitted scan, with each actual forward
  and Compiler-generated reverse body/binder independently qualified. CPU0033 implements the
  homogeneous FLOAT64 counterpart across the complete bounded declared algebra and actual
  first-order reverse closure. Both dtypes retain whole-occurrence qualification, static T,
  canonical boundaries and proved zero-offset scalar-all-zero reads. The approved CPU0037
  [read-boundary exception](docs/architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification)
  additionally permits exactly canonical-stride, zero-offset dense actual reads marked as views.
  CPU0037 implements this exception with forward/actual-reverse qualification; main accepted
  its final capability gates on 2026-10-24. Formals and output/write boundaries remain
  non-view canonical; Tensor metadata/equality, source, ownership and NN migration are unchanged.
  No arbitrary operation or layout is admitted. Bounded constructor-backed cold reconstruction and attachment-preserving Checkpoint
  in-memory copy/rebind are implemented independently of backend qualification. The approved
  all-valid owned NN workflow is implemented through the existing Engine/Training owners, without
  widening that cold inventory or the separately qualified execution domain.
  The user-approved length-aware extension is specified by the same
  [scan semantic owner](docs/architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan).
  Model0038 implements its bounded Model source API, row validation and structural IR counts.
  Compiler0013 implements independent length-mode inference, bounded first-order reverse
  construction and final role/demand queries. Original-source AD requires every carry and every
  original sequence to be ROW(0); generated reverse inputs retain explicit mixed roles.
  Prepare0011 implements static boundary/role validation and complete-graph demand projection
  through the existing declaration/assignment/finalization handoff, including actual generated
  reverse scans with mixed roles. It reads no length values and grants no backend execution capability.
  CPU0042 implements fixed T/B/features FLOAT32/FLOAT64 length-aware forward and actual
  Compiler-generated first-order reverse through the existing generated scalar, single-range
  path, including producer-ordered preflight and activity-dominated skipping. MAIN accepted
  the independently reviewed capability on 2026-10-26; its
  [Result](docs/planning/backends/cpu/tasks/0042-length-aware-general-scan-execution.md#result)
  owns validation and timing limits. Cold capture/reconstruction and encoding still reject;
  [Model0044](docs/planning/modules/model/tasks/0044-symbolic-length-scan-source-foundation.md)
  implements the user-approved static-or-one-named-B length-mode Model source foundation; its
  bounded source authority was accepted on 2026-10-26. Independent source/docs review and MAIN's
  configured root test/Javadoc checkpoint passed; MAIN accepted Model0044 as Complete on 2026-10-26.
  Model0045 implements [per-value typed length-aware Model source](docs/architecture/contracts/recurrent-scan.md#accepted-per-value-typed-length-aware-source)
  and explicit ordinary CAST with stable carry descriptors and exact SHARED contribution typing.
  Independent source/consumer-guard review passed; MAIN accepted Model0045 as Complete, source only,
  on 2026-11-01 using retained coverage and three approved test-only recoveries. Its
  [Result](docs/planning/modules/model/tasks/0045-per-value-typed-length-scan-source.md#result)
  retains the original failed commands; no clean root PASS is claimed.
  This changes no downstream inference, AD or execution domain.
  The implemented [empty-carry/explicit-mode source cutover](docs/architecture/contracts/recurrent-scan.md#accepted-empty-carry-and-explicit-length-mode-source)
  permits true carry-free source and zero-ROW length source through Model0046.
  Compiler/Prepare/CPU and cold/raw codecs independently retain their nonempty/positive-ROW domains.
  No dummy cotangent, new executor or wire/identity expansion follows.
  The accepted [per-output ordinary coordinate contract](docs/architecture/contracts/recurrent-scan.md#bounded-row-role-analysis)
  is implemented by Model0048, with one semantic recomputation seam after independent descriptor
  checking and intentionally broader homogeneous forward-coordinate inference. This is not a
  wider typed AD, execution, cold or wire grant; retained consumers keep independent admission.
  Compiler0015's bounded symbolic-length inference/actual first-order AD/fresh scoped all-port binding
  are unchanged within that retained qualification. Prepare0015 implements bounded combined
  projection/CPU/Engine execution under the
  [existing implementation gates](docs/architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates).
  Owning validation and independent code/structural review passed; root validation and the sole bounded probe
  completed;
  [Prepare0015 Result](docs/planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
  records evidence and limits; final MAIN acceptance remains pending.
  No new type/API, dependency, Runtime protocol, resource/lease owner or Engine memo is selected;
  Symbolic cold/copy/wire and wider per-value NN/Training workflows remain closed; the accepted
  static homogeneous cold/v5 intersection above is a separate capability.
  Old all-valid paths remain unchanged. Bounded all-valid body persistence belongs only to
  [Checkpoint's v4 owner](docs/architecture/contracts/extensions-training.md#canonical-graph-model-version-4),
  and is implemented without granting execution, AD or migration capability.
- Implemented coordinated all-PURE Scan cutover: the
  [Scan semantic owner](docs/architecture/contracts/recurrent-scan.md#general-ordinary-unit-scan-cutover),
  [Compiler proof/component-eight boundary](docs/architecture/contracts/compiler-autograd.md#controlled-scan-derivative-metadata),
  [Planning carrier](docs/architecture/contracts/foundational-modules.md#general-scan-activity-transport),
  [Prepare projection](docs/architecture/contracts/runtime-prepare-engine.md#general-scan-activity-projection)
  and [CPU realization](docs/architecture/contracts/backend-execution.md#general-scan-ordinary-unit-realization)
  jointly implement the mandatory all-valid/length first-production continuation. MAIN accepted
  the exact coordinated Ready/source after independent authority/interface and implementation
  reviews, actual selected forward/backward, typed/activity/boundary/empty/raw consumers and the
  unfiltered twelve-project test/Javadoc gate. The explicit neutral APIs/focused CPU owners
  supersede the earlier no-new-type/body-unit restrictions for this continuation, not a seventh
  authority, new Runtime executor/resource owner, cold/wire widening, Metal Scan grant or low-CPU
  arithmetic unlock. Qualified old recipes retain their own exact identities and byte replay.
  The Scan owner also defines local activity versus materialized Tensor boundaries; its reviewed
  clarification adds no global sparse-AD transport or public/runtime interface.
  A separate reviewed source-proof amendment selects the scoped four-argument ordinary guard on
  existing Model ScanBody, with unchanged three-argument ordinary inference and unchanged neutral/
  Runtime fields; named all-valid reshape count proof remains distinct from length ROW locality.
  A further independently reviewed prerequisite shares existing ordinary computed logical-layout
  closure through Model `ScanBody.closeOrdinaryLayouts(...)` and the separate geometry-independent
  `hasCanonicalContiguousResults(Operation)` fact. Compiler and CPU consume the same owners;
  symbolic classification, independent inference, exact captured/activity graphs and strict
  actual/formal boundaries remain unchanged. This adds no production type or Runtime field.
  A separately reviewed complete-graph publication prerequisite generalizes the existing
  producerless-constant contribution owner to producerless published caller or constant values.
  Its renamed five-component resource carries an existing Runtime ByteSize recipe; caller order,
  exact source/publication roles and complete-graph batch authority remain explicit. Existing
  CPU cold hooks authenticate genuinely fixed selections inside a complete batch-required plan
  without changing Runtime defaults, generated identities or execution protocols. No fake
  producer/reverse Scan, new executor, no-node grant or low-arithmetic unlock follows.
- `CompileArtifacts` are immutable compile-time output.
- `PreparedExecution`, its prepared memory/schedule/executable recipes, and immutable persistent
  prepared resources are immutable and reusable across runs.
  The sole accepted future exception is one bounded metadata-only replacement slot on the exact
  Engine prepared owner under the
  [geometry-memo contract](docs/architecture/contracts/runtime-prepare-engine.md#accepted-engine-geometry-memo).
  It cannot change prepared meaning or retain invocation resources. Encapsulated immutable numeric
  facts may be shared; mutable invocation arrays and every run's resources remain isolated.
  This exception is unimplemented and adds neither a Runtime cache nor a second lease protocol.
- `PreparedExecution` is the unique owner of persistent prepared resources. It has an explicit,
  idempotent close lifecycle even though its logical recipe remains immutable.
- Each active logical execution of one `PreparedExecution` has exactly one mutable `RunState`
  covering the complete heterogeneous run. Concurrent runs use distinct `RunState` instances and
  isolated mutable state and run-owned resources.
- `PreparedExecutable` computes only its prepared region.
- Runtime hot path must not see `Operation` or `CompiledNode`.
- Compiler must not allocate physical buffers.
- Planning must not select concrete kernels.
- Planning scoring selects backend ownership, not implementation routes.
- Planning may consume backend-neutral cost estimates, but it must not interpret backend route,
  vector, thread, tile, kernel, or other implementation parameters.
- Backend prepare owns backend-specific lowering and kernel selection.
- Metal route comparison begins only after candidates qualify for the same occurrence domain and
  Model semantics: exact structure/stored/discrete/special conformance and explicit test-only
  finite oracles, metrics, thresholds, and cancellation boundaries where arithmetic is finite.
  No sample grants provider capability or licenses a Compiler rewrite. A single survivor needs no
  comparative cost gate. Multiple survivors may be selected
  without timing only by strict structural dominance: no more compute dispatches, no more
  route-owned temporary bytes, and at least one strict improvement, using exact facts. An explicit
  custom route may use a source-bound declaration only when retained source owns every dispatch and
  resource; opaque routes including MPSGraph require supported actual-internal observation and
  never inferred facts. If multiple nondominated survivors remain, selection stays pending an
  authorized controlled environment. Local-device timing is diagnostic only and never
  qualification, route-selection, or tuning-identity authority. Any selected route is fixed during
  cold preparation, with no hot-path benchmark, retry, fallback, or matrix.
- Backend preparation is staged: backend analysis and exact shared-resource declaration precede
  shared slot assignment, and backend finalization follows slot assignment.
- Shared Prepare assigns one logical buffer slot per materialized graph value and one deterministic
  representation position per participating backend owner. Cross-owner availability is realized
  only by explicit prepared transfer steps between those positions.
- Model autotuning, when requested, must complete before runtime hot-path execution.
- Runtime profiling is passive observation and must not select or mutate execution settings.
- Runtime executes prepared schedules only.
- Engine is the composition root.
- Concrete backends must not depend on engine.
- Runtime must not depend on concrete backend implementations.

## Scope-indexed normative contracts

Use this table to find the one detailed contract that owns a concern. The linked heading is the
normative destination of the corresponding former root detail.

The following compatibility anchors keep existing focused links routed to this index while task
briefs and maintained navigation adopt the exact scoped destinations:

<a id="fixed-recurrent-scan-without-graph-regions"></a>
<a id="fixed-family-and-planned-model-surface"></a>
<a id="fixed-family-and-current-model-surface"></a>
<a id="static-shape-and-runtime-value-boundary"></a>
<a id="valid-lengths-traversal-and-outputs"></a>
<a id="purity-and-lifecycle-ownership"></a>
<a id="performance-and-migration-invariants"></a>
<a id="module-responsibilities"></a>
<a id="modulestrace"></a>
<a id="modulesbackend-contract"></a>
<a id="modulesmodel"></a>
<a id="modulesconfig"></a>
<a id="modulesplanning"></a>
<a id="partition-scoring"></a>
<a id="modulescompiler"></a>
<a id="compile-artifacts"></a>
<a id="training-graph-model"></a>
<a id="compiler-owned-automatic-differentiation"></a>
<a id="modulesruntime"></a>
<a id="modulesprepare"></a>
<a id="concrete-backend-modules"></a>
<a id="performance-evidence-and-optimization-tooling"></a>
<a id="cpu-backend-routes"></a>
<a id="metal-backend"></a>
<a id="openblas-provider"></a>
<a id="modulesengine"></a>
<a id="runtime-service-locator"></a>
<a id="reflective-backend-plugin-discovery"></a>
<a id="extensionsnn"></a>
<a id="extensionstraining"></a>
<a id="extensionsonnx"></a>
<a id="compile-lifecycle"></a>
<a id="prepare-lifecycle"></a>
<a id="run-lifecycle"></a>
<a id="optimizertraining-lifecycle"></a>

| Concern, module, or boundary | Normative contract and heading |
|---|---|
| Trace | [Foundational modules — `modules/trace`](docs/architecture/contracts/foundational-modules.md#modulestrace) |
| Backend Contract | [Foundational modules — `modules/backend-contract`](docs/architecture/contracts/foundational-modules.md#modulesbackend-contract) |
| Model and Tensor producer ownership | [Foundational modules — `modules/model`](docs/architecture/contracts/foundational-modules.md#modulesmodel) |
| Model numerical semantics and allowed results | [Foundational modules — profile-free numerical semantics](docs/architecture/contracts/foundational-modules.md#profile-free-numerical-semantics) |
| Config | [Foundational modules — `modules/config`](docs/architecture/contracts/foundational-modules.md#modulesconfig) |
| Planning responsibility | [Foundational modules — `modules/planning`](docs/architecture/contracts/foundational-modules.md#modulesplanning) |
| Fixed recurrent scan | [Recurrent scan — fixed recurrent scan without graph regions](docs/architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions) |
| Complete fixed/general scan semantics, formals, residuals, traversal and exclusions | [Recurrent scan — general differentiable scan](docs/architecture/contracts/recurrent-scan.md#general-differentiable-scan); `recurrent-scan.md` is the sole complete scan semantic owner. Other scoped contracts own only their module boundaries and link to it. |
| Implemented CPU row-map/multiple-terminal fusion mechanism and selective identities | [Backend execution — CPU0073 accepted row-map and multiple-terminal reduction fusion](docs/architecture/contracts/backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion); CPU-internal physical lowering only, not a new semantic graph or Runtime protocol. Scan semantics remain with the Scan owner. |
| Implemented scalar invariant external-cell staging | [Backend execution — CPU0076 invariant external-cell load staging](docs/architecture/contracts/backend-execution.md#cpu0076-prospective-invariant-external-cell-load-staging); narrow load-placement/identity exception to CPU0075, not carry promotion, new capability or lifecycle authority. |
| Implemented and MAIN-accepted selected row-fusion SIMD and parallel execution | [Backend execution — CPU0074 selected row-region execution exception](docs/architecture/contracts/backend-execution.md#cpu0074-selected-row-region-execution-exception); exact internal seams/identities only. [Scan semantics](docs/architecture/contracts/recurrent-scan.md#cpu0074-selected-terminal-grouping-exception) and [run-owned reservation](docs/architecture/contracts/runtime-prepare-engine.md#cpu0074-run-owned-synchronous-reservation) remain with their respective owners. CPU0074 is Complete; steady-state profitability is not qualified and production defaults remain unchanged. |
| Length-aware source/IR, Compiler first-order reverse, Prepare projection and F32/F64 CPU execution qualifications | [Recurrent scan — length-aware whole-batch scan](docs/architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan); Model0044 source and Compiler0015 inference/actual AD/scoped all-port binding are implemented. CPU0042 implements fixed geometry; Prepare0015 implements the one-B combined path, with root validation and the sole bounded probe completed (evidence and limits: [Prepare0015 Result](docs/planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)); final MAIN acceptance remains pending. No cold/persistence/NN/Training admission. |
| Implemented per-value typed length-aware Model source | [Recurrent scan — accepted per-value typed length-aware source](docs/architecture/contracts/recurrent-scan.md#accepted-per-value-typed-length-aware-source); Model0045 Complete, source only, MAIN accepted 2026-11-01 after independent APPROVE, retained coverage and three approved test-only recoveries ([Result](docs/planning/modules/model/tasks/0045-per-value-typed-length-scan-source.md#result)). No clean root PASS; existing consumer domains and fixed/unrolled APIs remain unchanged. |
| Implemented empty-carry and explicit length-mode source | [Recurrent scan — accepted empty-carry and explicit length-mode source](docs/architecture/contracts/recurrent-scan.md#accepted-empty-carry-and-explicit-length-mode-source); Model0046 source/caller/guard cutover is integrated and source-qualified. Compiler/Prepare/CPU/cold/raw-codec guards, old qualified CPU artifact bytes and current capability rows remain independently bounded; no typed execution or persistence widening. |
| Implemented ordinary per-output Scan coordinate semantics | [Recurrent scan — bounded row-role analysis](docs/architecture/contracts/recurrent-scan.md#bounded-row-role-analysis); Model0048 shares descriptor-only semantic recomputation, canonical value/control dependencies and completely proved intermediate axes/ranks without a new math whitelist. Descriptor inference and downstream effective admission remain independent. |
| Implemented all-PURE Scan continuation | [Scan semantics](docs/architecture/contracts/recurrent-scan.md#general-ordinary-unit-scan-cutover), [Compiler proof/metadata](docs/architecture/contracts/compiler-autograd.md#controlled-scan-derivative-metadata), [Planning neutral carrier](docs/architecture/contracts/foundational-modules.md#general-scan-activity-transport), [Prepare projection](docs/architecture/contracts/runtime-prepare-engine.md#general-scan-activity-projection), [CPU realization](docs/architecture/contracts/backend-execution.md#general-scan-ordinary-unit-realization). Compiler0017/CPU0081 are integrated and source/consumer-qualified; static and one-B all-valid/length composition use ordinary owners, exact selected VJPs and structural activity. Cold/wire, Metal Scan and Java low arithmetic remain independently bounded. |
| Historical one-B Scan qualifications | [Recurrent scan — accepted symbolic-batch scan](docs/architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan); sole Scan semantic owner. Model0042 source, Compiler0014 all-valid inference/AD/binding and Prepare0014 all-valid FLOAT32 execution remain qualified legacy recipes. Prepare0015's combined F32/F64 path did not itself add all-valid symbolic F64; the implemented general continuation above is separately qualified. Symbolic cold/copy/wire/NN/Training remain closed. |
| Partition scoring | [Compiler/autograd — partition scoring](docs/architecture/contracts/compiler-autograd.md#partition-scoring) |
| Compiler responsibility | [Compiler/autograd — `modules/compiler`](docs/architecture/contracts/compiler-autograd.md#modulescompiler) |
| Compile artifacts | [Compiler/autograd — compile artifacts](docs/architecture/contracts/compiler-autograd.md#compile-artifacts) |
| Training graph and compiler-owned autograd | [Compiler/autograd — training graph model](docs/architecture/contracts/compiler-autograd.md#training-graph-model) |
| Compile lifecycle | [Compiler/autograd — compile lifecycle](docs/architecture/contracts/compiler-autograd.md#compile-lifecycle) |
| Cold source admission and family reconstruction, including the bounded general-SCAN/body exception to the unchanged flat source inventory | [Compiler/autograd — compile lifecycle](docs/architecture/contracts/compiler-autograd.md#compile-lifecycle) |
| Runtime | [Runtime/Prepare/Engine — `modules/runtime`](docs/architecture/contracts/runtime-prepare-engine.md#modulesruntime) |
| Prepare | [Runtime/Prepare/Engine — `modules/prepare`](docs/architecture/contracts/runtime-prepare-engine.md#modulesprepare) |
| Engine and explicit composition | [Runtime/Prepare/Engine — `modules/engine`](docs/architecture/contracts/runtime-prepare-engine.md#modulesengine) |
| Prepare and run lifecycles | [Runtime/Prepare/Engine — prepare lifecycle](docs/architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle) |
| Symbolic-batch metadata admission, numeric sizing/creation and actual publication | [Runtime/Prepare/Engine — accepted symbolic-batch resource contract](docs/architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract); [Compiler metadata binding](docs/architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding); [CPU numeric recipes](docs/architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract). Prepare0015 implements combined/typed execution; root validation and the sole bounded probe completed (evidence and limits: [Prepare0015 Result](docs/planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)); final MAIN acceptance remains pending. Scan semantics stay with their owner. No dependency edge or NN/persistence capability follows. |
| Accepted future metadata-only geometry reuse, admission and publication/close coordination | [Runtime/Prepare/Engine — accepted Engine geometry memo](docs/architecture/contracts/runtime-prepare-engine.md#accepted-engine-geometry-memo); exact Engine preparation only, numeric-only Runtime, unimplemented. |
| Service-locator and plugin-discovery prohibitions | [Runtime/Prepare/Engine — runtime service locator](docs/architecture/contracts/runtime-prepare-engine.md#runtime-service-locator) |
| Concrete CPU, Metal, and CUDA backends | [Backend execution — concrete backend modules](docs/architecture/contracts/backend-execution.md#concrete-backend-modules) |
| Benchmarking and model-autotuning tools | [Backend execution — performance evidence and optimization tooling](docs/architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling) |
| CPU routes and generated JVM bytecode | [Backend execution — CPU backend routes](docs/architecture/contracts/backend-execution.md#cpu-backend-routes) |
| Metal | [Backend execution — Metal backend](docs/architecture/contracts/backend-execution.md#metal-backend) |
| OpenBLAS provider | [Backend execution — OpenBLAS provider](docs/architecture/contracts/backend-execution.md#openblas-provider) |
| Data host counts, padded batches, integral sequence batching and opt-in scan-length Tensor export | [Extensions/training — `extensions/data`](docs/architecture/contracts/extensions-training.md#extensionsdata) |
| Frozen whitespace tokenization and Data-delegating text batching | [Extensions/training — `extensions/text`](docs/architecture/contracts/extensions-training.md#extensionstext) |
| NN stateful composition, declarative reconstruction and retained old model persistence | [Extensions/training — `extensions/nn`](docs/architecture/contracts/extensions-training.md#extensionsnn) |
| Owned Models, fixed/general recurrent inference and qualified first-order training | [Extensions/training — owned tensor Models](docs/architecture/contracts/extensions-training.md#owned-tensor-models) |
| Training | [Extensions/training — `modules/training`](docs/architecture/contracts/extensions-training.md#modulestraining) |
| Generic graph/value persistence and opaque artifact envelope | [Extensions/training — `extensions/checkpoint`](docs/architecture/contracts/extensions-training.md#extensionscheckpoint) |
| Selective graph-model v1/v2 wire compatibility | [Extensions/training — canonical graph-model version 2](docs/architecture/contracts/extensions-training.md#canonical-graph-model-version-2) |
| Graph-model v3 wire, multi-output normalization and selective v1/v2/v3 compatibility | [Extensions/training — canonical graph-model version 3](docs/architecture/contracts/extensions-training.md#canonical-graph-model-version-3) |
| Bounded general-scan persistence, scoped inline bodies and selective v4 compatibility | [Extensions/training — canonical graph-model version 4](docs/architecture/contracts/extensions-training.md#canonical-graph-model-version-4) |
| Implemented selective static length-aware v5 wire and old v1–v4 compatibility | [Extensions/training — canonical graph-model version 5](docs/architecture/contracts/extensions-training.md#canonical-graph-model-version-5) |
| Retained NN model persistence and optional Training Checkpoint adapter reservation | [Extensions/training — retained NN checkpoint APIs](docs/architecture/contracts/extensions-training.md#retained-nn-checkpoint-apis) |
| ONNX | [Extensions/training — `extensions/onnx`](docs/architecture/contracts/extensions-training.md#extensionsonnx) |
| Optimizer/training lifecycle | [Extensions/training — optimizer/training lifecycle](docs/architecture/contracts/extensions-training.md#optimizertraining-lifecycle) |
| CLI | [Backend execution — other tool scopes](docs/architecture/contracts/backend-execution.md#other-tool-scopes) |
| Native projects | [Backend execution — concrete backend modules](docs/architecture/contracts/backend-execution.md#concrete-backend-modules) |
| Architecture, conformance, and integration testing | [Testing requirements](#testing-requirements) |

## Module-ownership routing

This compact routing summary does not replace the linked details:

| Scope | Owner |
|---|---|
| Model operation semantics and immutable graph model | Model |
| Backend-neutral capability, ownership, partitioning, and logical requirements | Planning |
| Graph transformations, autograd, publication, and compile artifacts | Compiler |
| Shared transition contracts, slot assignment, and prepared validation | Prepare |
| Concrete lowering, fusion, routes, storage, and backend execution | Each concrete backend |
| Prepared schedules, residency, publication, and per-run mutable state | Runtime |
| Explicit composition and public lifecycle | Engine |
| Typed diagnostics | Trace |
| Host numeric sequence preparation, without execution or new storage-lifecycle ownership | Data (`extensions/data`); INT64 batching and opt-in scan-length export implemented, broader numeric/sample batching deferred |
| Frozen vocabulary, whitespace tokenization and text-to-Data adaptation | Text (`extensions/text`); no NN or execution responsibility |
| Stateful modules, parameters, buffers, train/eval behavior, declarative and owned tensor Models, model persistence | NN; composes public Engine/Training/Checkpoint owners, never another execution pipeline |
| Generic live parameter admission, optimizer algorithms, explicit pre-session native parameter copies, and training orchestration | Training (`modules/training`, Gradle `:extensions:training`); NN owns its live Module adapter |
| Generic cold graph/value copy/save/load/rebind and bounded opaque artifact envelope | Checkpoint (`modules/checkpoint`, Gradle `:extensions:checkpoint`); Compiler owns raw capture/reconstruction; no NN, Training or Engine dependency |
| Optional stored Model + TrainingState restoration; broader exact continuation requires all affecting state | Future Training Checkpoint adapter, after its separate gate; each state owner retains validation and restore authority |
| Exact training continuation claim and cross-owner restore coordination | Each state owner retains its state; the caller stages an unpublished replacement and controls exclusion and publication under the [Training lifecycle contract](docs/architecture/contracts/extensions-training.md#optimizertraining-lifecycle) |
| Fixed benchmark evidence | `tools/benchmarks` |
| Model-autotuning measurement, cache coordination, and selection | `tools/tuning` |

## Dependency rules

The intended dependency direction is:

```text
trace
backend-contract
config
model
  -> planning
  -> compiler
```

Runtime/prepare/backend side:

```text
model
config
backend-contract
trace
planning
compiler
  -> runtime
  -> prepare
  -> concrete backends
```

Composition:

```text
compiler
runtime
prepare
modules/backend/cpu
modules/backend/metal
  -> engine
```

The supported first-production Engine composition depends on the implemented CPU and Metal
backends, not the unused CUDA marker. CUDA's project mapping and source remain available for
future development; removing that unused Engine edge does not remove a supported backend or
authorize CUDA publication. Public inward types must remain visible through genuine API exports;
internal implementation use alone does not justify an API edge.

Neural-network composition and training use this direction (Training retains Gradle identity
`:extensions:training` at physical `modules/training`):

```text
modules/model
  -> extensions/data (Model API export only)
  -> extensions/text (through Data API export only)

modules/model + public modules/engine facade
  -> modules/training

modules/model + modules/compiler (cold implementation only)
  -> modules/checkpoint

modules/model + modules/training + modules/checkpoint + public modules/engine facade
  -> extensions/nn

modules/training + extensions/checkpoint
  -> optional future extensions/training-checkpoint
```

Concrete rules:

- `modules/trace` must not depend on model, planning, compiler, runtime, prepare, engine, or concrete backends.
- `modules/model` must not depend on planning, compiler, runtime, prepare, engine, or concrete backends.
- `modules/config` must not depend on concrete backend implementations.
- `modules/planning` must not depend on concrete backends, runtime, prepare, or engine.
- `modules/compiler` must not depend on runtime, prepare, engine, or concrete backends.
- `modules/runtime` must not depend on concrete backends or engine.
- `modules/prepare` must not depend on concrete backend implementations.
- Concrete backends must not depend on `modules/engine`.
- `modules/backend/openblas-provider` must not depend on compiler, planning, runtime, prepare, engine, or Tensor API.
- `extensions/data` has exactly `api(project(":modules:model"))` as its sole project dependency.
  No other project edge is permitted in any configuration, including tests. Inward modules and
  existing extensions must not depend on Data, except the separately authorized Text edge below.
  This changes no Model, NN, Training, runtime-input binding or storage-ownership contract.
- `extensions/text` has exactly `api(project(":extensions:data"))` as its sole project dependency
  across all configurations, including tests. Model is exported transitively by Data; Text has
  no direct Model edge and may use only public Data/Model APIs. No NN, Engine, Training,
  Checkpoint, Compiler, Runtime, Prepare, tuning or backend dependency/import is permitted.
  Inward modules and existing extensions, including NN, have no Text edge. The integration-test
  project may add only `testImplementation(project(":extensions:text"))` for Text integration;
  it composes its existing NN/Engine edges externally. Vision remains unauthorized.
- `extensions/nn` has exactly public Model, Training, Checkpoint and Engine project edges.
  Training supplies live Module admission and sessions; Checkpoint supplies generic artifacts/graph persistence;
  Engine supplies cached public execution and old persistence materialization. NN has no
  direct Compiler, Runtime, Prepare, Data, Text or concrete-backend edge, and owns no second execution pipeline.
- `modules/training` has exactly the public Model and Engine project edges and must not depend
  on NN. The explicit public Engine composition edge does not permit a dependency on Compiler,
  Runtime, Prepare, tuning, or a concrete backend, use of Engine internals, or a second execution
  pipeline.
- `modules/training` must not depend directly on concrete backend modules.
- `modules/checkpoint` (Gradle `:extensions:checkpoint`) has exactly `api(Model)` and
  `implementation(Compiler)` project edges. Compiler access is only the cold source boundary;
  public Checkpoint signatures expose no Compiler types. Checkpoint has no NN, Training, Engine,
  Runtime, Prepare, tuning or concrete-backend edge, including tests. NN may depend on Checkpoint;
  other inward modules and Training may not. It reads Model host storage, never executes.
- The optional `extensions/training-checkpoint` adapter may depend on Training and Checkpoint to
  coordinate durable training continuation, but is not included or implemented before its later
  adapter gate. It must not move model-only persistence into Training or make Checkpoint depend on
  Training.
- `extensions/onnx` must not depend on runtime hot-path execution internals.

<a id="documentation"></a>

## Documentation authority and architecture updates

The repository distinguishes authority from explanation and process:

```text
ARCHITECTURE.md
  authoritative root and sole authority index

docs/architecture/contracts/
  the six scoped normative contracts incorporated by the root

docs/
  explanatory architecture, guides, examples, ADRs, and design notes

docs/planning/
  non-authoritative implementation plans

AGENTS.md
  agent working instructions
```

When an architecture decision changes, update in the same change:

1. this root and the one owning scoped contract;
2. the relevant explanatory file under `docs/architecture/`;
3. an ADR under `docs/design/decisions/`, when the decision is significant; and
4. architecture tests under `testing/architecture-tests/`, when dependency rules change.

A change must not create overlapping scoped authority. If ownership would cross two scoped
contracts, update the root index to establish one unambiguous owner before implementation.

## Explicit non-goals

Do not add these unless this document is updated first:

- compile-time physical schedule
- standalone artifacts/program module
- shared `backend.lowering` module
- runtime service locator
- reflective backend plugin discovery as the core backend mechanism
- separate `cpu-scalar`, `cpu-vector`, or `cpu-blas` backend modules
- `MetalOptimizerBridge` in training
- `Map<String,String>` as the primary trace model
- backend-specific kernel/runtime scoring in planning
- benchmark-driven production-setting mutation
- model autotuning or tuning-cache mutation in the runtime hot path
- planning interpretation of backend routes, vector parameters, threads, chunks, or tiles
- `Tensor.gradient`, `Tensor.backward`, mutable Tensor gradient fields, or hidden thread-local
  gradient/compilation scope
- model-owned derivative rules
- placeholder Tensors that stand in for already-captured `ValueId` values
- a `ValueId`-to-Tensor conversion map except call-local scope-qualified tables used only after complete cold
  source preflight for bounded family-constructor reconstruction under the
  [Compiler contract](docs/architecture/contracts/compiler-autograd.md#compile-lifecycle);
  discard those tables on return or failure, never use them for autograd or runtime
- a second low-level gradient algebra language
- a public compiler gradient-rule registry or facade
- arbitrary graph regions, if/while control flow, recursion, nested scans, or parallel/associative
  general scan; the bounded fixed-count scan exception above grants none of these

## Future extensions allowed only with architecture update

The following may be added later, but only with an explicit update to this document:

- `modules/compiler-api`
- `modules/program`
- `LogicalSchedulePlan`
- source-generated CPU fused kernels
- external plugin ecosystem
- `ServiceLoader` as an optional engine-level convenience layer
- more advanced segment-level partition scoring

## Testing requirements

Architecture-sensitive changes must include or update architecture tests under:

```text
testing/architecture-tests/
```

Architecture tests should enforce:

- trace does not depend on other modules
- model does not depend on planning/compiler/runtime/prepare/engine/backend
- config does not depend on concrete backends
- planning does not depend on concrete backend/runtime/prepare/engine
- compiler does not depend on runtime/prepare/engine/concrete backend
- runtime does not depend on concrete backend/engine
- prepare does not depend on concrete backend implementations
- backends do not depend on engine
- openblas-provider does not depend on compiler/planning/runtime/prepare/engine/Tensor API
- `:extensions:data` is included at `extensions/data` with exactly the Model API project edge
  across all configurations, no reverse inward/existing-extension edge except authorized Text,
  and no Vision inclusion;
  Data Java sources, when present, use `io.github.pho001.synaptik.data` and only Model project imports;
  boundary checks retain test-local positive/negative fixtures and an isolated Data-only consumer
  that constructs and inspects the permitted Data API and compiles Model Tensor declarations
  through the exported edge, without production placeholders or an NN classpath
- `:extensions:text` is included at `extensions/text` with exactly the Data API project edge
  across all configurations and public Data/Model imports only; isolated Text-only consumers
  compile Data/Model declarations transitively without NN/Engine. Negative fixtures reject
  forbidden edges/imports, including NN-to-Data/Text. Text's numerical integration is exclusively
  in `testing/integration-tests` through its test-only Text edge, never a Text-to-NN/Engine edge.
- `Operation` does not expose `supportedBackends()`
- runtime hot path does not use `Operation` or `CompiledNode`
- planning scoring does not reference concrete kernel classes
- `extensions/nn` has exactly public Model/Training/Checkpoint/Engine edges, with no direct
  Compiler/Runtime/Prepare/backend edge; its live adapter preserves ordered path/wrapper/Tensor membership,
  and owned Models delegate execution, training and persistence to their existing public owners
- physical `modules/training` maps to `:extensions:training`, has exactly public Model/Engine
  edges, and has no NN imports, NN-taking overloads, or test-only NN dependency; the public Engine
  exception is role-specific and never exempts other inward modules
- the included `:extensions:checkpoint` project maps to `modules/checkpoint`, has exactly public
  Model and private cold Compiler edges, NN-free sources/tests and public signatures, no old source
  location or forwarding packages, and no included Training Checkpoint adapter before its separate gate

Backend behavior changes should include or update backend conformance tests under:

```text
testing/backend-conformance/
```

End-to-end behavior changes should include or update integration tests under:

```text
testing/integration-tests/
```

## Final summary

The architecture is:

```text
model      = clean computational model
planning   = backend-neutral intent, capability, scoring, ownership, logical memory
compiler   = graph transformations, autograd, compile artifacts
prepare    = transition from compile artifacts to executable runtime
backend    = concrete lowering, fusion, kernel selection, storage
runtime    = hot-path execution, residency, publication
engine     = composition root and public lifecycle
trace      = typed diagnostic leaf
nn         = stateful neural-network composition, parameters, buffers, and train/eval behavior
training   = optimizer algorithms and training orchestration over nn-declared parameters
```

The most important invariant is:

```text
CompileArtifacts are an immutable recipe.
Planning scoring selects backend ownership, not kernel implementation.
Backend analysis selects a route and declares exact shared requirements before slot assignment.
Backend finalization constructs executable state only after slot assignment.
PreparedExecution recipes are immutable and reusable across runs.
Exactly one RunState owns the mutable state of each active complete logical run.
PreparedExecutable computes only its prepared region.
```
