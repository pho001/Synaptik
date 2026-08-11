# CPU Backend Master Plan

## Goal

Implement truthful CPU capability reporting, backend-owned preparation, portable correctness and
SIMD routes, optional native routes, storage, workspace, and execution.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- CPU capability provider
- partition lowering, specialization, and fusion
- computation-oriented partition execution units and a CPU-private loop-oriented kernel IR
- portable generated scalar and JDK Vector API elementwise/reduction strategies
- standard JDK Class-File API generation for portable scalar and Vector API computation kernels
- optional OpenBLAS routes for supported BLAS-compatible linear algebra
- distinct Intel oneMKL BLAS/VML and oneDNN integrations
- Apple Accelerate BLAS, vDSP, and vForce integrations
- distinct AMD AOCL-BLAS/AOCL-LibM and later optional ZenDNN integrations
- other specialized and fused routes only when a concrete capability justifies them
- CPU executables, storage, workspace, scheduling, and tracing
- deterministic generated-class identity, verification, and process-local compatible reuse, with
  optional trusted-root filesystem persistence as a cold-path policy
- joint route and physical-representation selection across relevant CPU dataflow and partition uses
- typed, version-controlled, tested route candidate generators and compatible workload-cache
  lookup during prepare

## Out of scope

- global compiler logic
- public Tensor ownership
- engine dependency
- separate CPU route backends

## Module invariants

- Planning selects CPU ownership; CPU prepare selects the route.
- All CPU routes remain inside one concrete backend.
- CPU backend never depends on engine.
- The portable route is the bytecode-first Java Class-File API plus Vector API production baseline
  and the always-available semantic fallback for every occurrence it supports. The scalar
  reference is for conformance and fail-closed checking, not a Runtime operation/IR interpreter.
- The next implementation task atomically replaces the provisional per-node portable pipeline.
  CPU analysis lowers one complete partition into computation-oriented execution units, performs
  safe fusion before exact resource declaration, selects a realization route, and finalizes one
  partition-level executable. One `OperationKind` must not imply one lowering, emitter,
  invocation, or kernel.
- `LogicalMemoryPlan` remains logical and complete for graph values. A graph value receives a
  physical Runtime slot only when backend analysis emits a `PreparationResourceRequirement.Buffer`.
  Same-unit fused intermediates remain graph and IR values without declarations or slots.
- Java 26 `java.lang.classfile.CodeBuilder` is the selected current implementation direction for
  every portable CPU computation kernel. This is non-authoritative planning: the architecture
  permits generated JVM-bytecode CPU computation kernels without making that builder or another
  generation library an invariant.
- Default portable classes use shape-polymorphic primitive `start`/`end` loops. Concrete compatible
  extents and element count are cold-bound facts and do not enter class/cache identity unless an
  explicit evidence-selected fixed-shape specialization consumes a bounded specialization budget.
- One route-independent `CpuKernelIr` records typed boundary/virtual values, ordered semantics,
  access-plan form, universal loop model, and stores. Route, thread count, vector species,
  artifact root, Runtime/graph identities, generator versions, and instance bindings are separate.
- Every currently selected executable Model operation semantic must gain truthful portable
  generated coverage before the portable capability milestone closes. Metadata-only or zero-work
  view occurrences need no generated computation. Unsupported executable semantics fail closed
  and are not advertised until their generated coverage exists.
- OpenBLAS is a narrow cross-platform FLOAT32/FLOAT64 native fallback for supported BLAS-compatible
  linear algebra. It is neither a universal fallback nor the preferred CPU route by identity.
  BFLOAT16 requires a separately verified version, instruction-set architecture (ISA), and
  operation route. FLOAT16 support is neither broad nor baseline by assumption.
- CPU never offloads internally to MPSGraph or a custom Metal kernel. Planning must first select
  Metal ownership, after which the separate Metal backend owns those routes.
- CPU owns capability truth, provider coordination, route selection, fallback, thread/lifetime
  coordination, and tuning. Native provider layers remain low-level ABI/lifetime leaves.
- Common whole-partition lowering, fusion legality/profitability, `CpuKernelIr`, access plans,
  materialization accounting, numerical/determinism filtering, and representation planning are
  route-independent. Native provider adapters do not interpret graphs, plan broadcasting or
  fusion, or own shared resource lifetimes.
- Fusion legality is a fail-closed correctness decision made before profitability. Profitability
  ranks only complete legal CPU candidates and may select a split plan even when a fused plan is
  legal. Safe bounded heuristics are the default; optional model autotuning may later measure
  eligible complete candidates before Runtime, but Runtime never searches or revises the choice.
- Planning selects one CPU backend owner and one `BackendId("cpu")`. CPU Prepare chooses among the
  portable baseline, OpenBLAS where narrowly eligible, and vendor/platform-specialized peer routes
  by exact capabilities and whole-plan cost. There is no fixed vendor priority.
- ARM selection is capability-first. Apple Silicon may select Accelerate; another ARM target uses
  portable code generation unless a later task adds an explicitly verified provider.
- CPU candidate generators return complete valid route-specific configurations; shared tuning
  sees them opaquely.
- Safe CPU heuristics remain correct when tuning is disabled or a compatible cache entry is
  absent.
- Candidate eligibility is filtered for exact operation semantics and required determinism before
  any performance comparison. Current exact/default semantics do not permit vendor fast- or
  relaxed-math routines.
- Internal portable and vendor routes may implement fast exponential, hyperbolic-tangent, or
  similar algorithms, but an exact/default candidate is eligible only when it satisfies the
  ordinary operation's conformance contract. A genuinely relaxed approximation requires explicit
  caller permission; CPU must not infer it from hardware, provider availability, workload size,
  tuning objectives, or benchmark results.
- Each eligible operation family fixes its logical input, accumulation or other intermediate, and
  output types before CPU route selection. FLOAT32 accumulation is the expected default for
  numerically sensitive 16-bit work; exceptions require an explicit Model semantic contract.
- `DataType` availability and a matching storage carrier never advertise or select a CPU route.
  Every 16-bit candidate requires exact ABI, ISA/hardware, operation, layout, numerical,
  determinism, and resource filtering before route/workload benchmarking, safe heuristics, or
  compatible tuning evidence can compare it.
- Memory-segment storage and execution route are orthogonal. Scalar Java, JDK Vector API, and
  Foreign Function and Memory (FFM) native-provider calls may use the same native-backed
  `MemorySegment` without copying merely because the route changes.
- Run-owned internal CPU buffers use aligned native off-heap `MemorySegment` storage as the
  canonical interoperable CPU representation. Complete task 0001 implements its exact alignment,
  arena/lifetime ownership, access, cleanup, zero-size, and allocation contracts.
- Borrowed caller inputs are handled per value and use. Compatible heap-backed inputs may remain
  heap-backed. CPU preparation introduces at most one necessary native materialization for an
  exact selected downstream FFM route and reuses it across compatible consumers.
- Ordered carrier access form is an explicit code-shaping structural specialization. Each portable
  generated class has one direct static entry signature for its actual ordered primitive-array or
  `MemorySegment` pattern; that pattern participates in compatibility/class identity. Exact
  carrier objects and byte offsets remain cold bindings. CPU analysis receives and validates the
  backend-owned prepared pattern, finalization emits or loads its one artifact, and Runtime binding
  only accepts matching concrete carriers. This avoids emitting every possible carrier combination
  in each class while preserving direct hot code without Runtime generation, storage discovery, or
  dispatch.
- `CpuPartitionAnalysisInputs.DEFAULT` disables lowering-manifest retention and uses an empty
  explicit carrier list as the policy "one exact `MEMORY_SEGMENT` form per lowering-derived
  boundary." Explicit CPU analysis inputs may immutably select an ordered typed heap/segment
  pattern whose count and order must equal the derived boundary list. Boundary cardinality is not
  fixed by the analysis-input contract.
- Portable `MemorySegment` storage is representation-only and accepts the logical data type. A
  two-byte representation does not imply executable arithmetic or Vector support. The current
  generated specialization admits Java Vector lanes only for FLOAT64, FLOAT32, INT32, and INT64;
  BFLOAT16 and future FLOAT16 need separately established routes.
- Portable execution has exactly four strategies: scalar, vector, parallel-scalar, and
  parallel-vector. Scalar/vector is the compute axis; single-thread/parallel is the orchestration
  axis. Every generated kernel accepts `start` and `end`; workers dispatch chunks outside it.

## Allowed dependencies

- modules/model
- modules/config
- modules/planning
- modules/runtime
- modules/prepare
- modules/backend-contract
- modules/trace
- backends/openblas-provider

## Forbidden dependencies

- modules/engine

## Package structure

```text
io.github.pho001.synaptik.backend.cpu/
  CpuCapabilityProvider       sole public truthful fail-closed CPU capability provider
  package-info.java           public package boundary and current status
  internal/
    memory/                   representations and cold binding
    prepare/                  analysis and finalization lifecycle
    lowering/                 whole-partition unit formation and fusion
    ir/                       canonical IR and normalized access plans
    codegen/emit/             portable Class-File generation and direct loop emission
    route/portable/           portable route selection/realization plan
    cache/                    structural identity and optional persistence
    executable/               prepared partition execution and CPU-private worker orchestration
    reference/                conformance/fail-closed scalar reference
```

Task 0005A adopts this structure atomically. Java subpackages are separate access domains, not
friends, so only the minimal cross-package contracts are technically public below `.internal` and
are explicitly unsupported API. `CpuCapabilityProvider` remains the sole supported public CPU
type. No JPMS export, service locator, registry, broad facade, compatibility bridge, or retained
flat pipeline is permitted.

Later tasks add only their concrete leaves: `route/nativeblas/{openblas,accelerate,mkl,aocl}` for
BLAS-compatible calls and `route/nativeops/{accelerate,mkl,onednn,aocl,zendnn}` for vDSP/vForce,
VML, oneDNN, AOCL-LibM, and ZenDNN. These target locations are not placeholder packages and are not
created by 0005A. All consume the common analysis above; none creates another backend identity.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [CPU capability, representation, binding, and parallel foundation](tasks/0001-cpu-capability-representation-binding-and-parallel-foundation.md) | Superseded | Stable planning, runtime, prepare, backend-contract, and trace contracts | Historical foundation replaced atomically by 0005A; retained as validation and design evidence. |
| 0002 | [Portable Class-File API generator foundation](tasks/0002-portable-class-file-api-generator-foundation.md) | Superseded | 0001; generated JVM-bytecode CPU-kernel architecture contract; Java 26 Class-File and Vector API toolchain | Historical generator foundation replaced atomically by 0005A; retained as validation and design evidence. |
| 0003 | [Durable generated-kernel artifact store and cold loading](tasks/0003-bounded-generated-artifact-cache-and-cold-finalization.md) | Superseded | 0002; stable CPU finalization and artifact compatibility; explicit trusted local root | Historical mandatory durable-store design replaced by 0005A's optional persistence policy. |
| 0004 | [Typed portable analysis, specialization, and finalization](tasks/0004-typed-portable-analysis-specialization-and-finalization.md) | Superseded | 0001–0003 | Historical per-node candidate/finalization architecture replaced atomically by 0005A. |
| 0005 | [Dense ADD and partition-sequence execution](tasks/0005-dense-add-and-partition-sequence-execution.md) | Superseded | 0002–0004 | Historical per-node dense ADD route replaced atomically by 0005A's fused partition kernel. |
| 0005A | [Atomic partition-kernel architecture reset](tasks/0005a-atomic-partition-kernel-architecture-reset.md) | Complete | 0001–0005; current shared Prepare contracts | Adopted structured internals and only the portable route leaf; replaced the per-node path with whole-partition lowering, route-independent IR, universal start/end Class-File generation, exact declarations, and one partition executable; proved shape-polymorphic FLOAT64 ADD-to-GELU-to-MUL fusion. |
| 0005B | [Universal access plans and right-aligned broadcasting](tasks/0005b-universal-access-plans-and-right-aligned-broadcasting.md) | Complete | 0005A | Delivered one normalized per-value access system over current ShapeBroadcast/LayoutDescriptor semantics, complete static scalar/rank-expanded/multi-axis/zero/strided/heap/segment/mixed-carrier support, one direct entry per ordered carrier-pattern specialization, and five distinct dense-to-general-odometer scalar state machines. |
| 0005C | [Vector and parallel portable strategies](tasks/0005c-vector-and-parallel-portable-strategies.md) | Complete | 0005B | Added preferred-species FLOAT64 vector, parallel-scalar, and parallel-vector realization over universal start/end kernels; direct contiguous runs vectorize without gather, and explicit caller-owned CPU-private workers execute deterministic disjoint chunks. |
| 0005D | [Materialization, specialization, and persistence evidence gate](tasks/0005d-materialization-specialization-and-persistence-evidence-gate.md) | Complete | 0005C | Added at most one CPU-internal contiguous materialization before assignment, enforced four-candidate/one-artifact/zero-shape/zero-unroll budgets, and recorded a `KEEP_DISABLED` opt-in persistence-evidence verdict. |
| 0005E | [Portable pointwise types, carriers, and semantic-family expansion](tasks/0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md) | Complete | 0005D | Delivered the first bounded five-type core pointwise increment through one nineteen-opcode family pipeline, derived-boundary typed carriers, and the completed unit/IR/access/materialization/route/artifact/executable architecture; unsupported rows remain fail-closed. |
| 0005F | [Floating division and exact scalar-power realization](tasks/0005f-floating-division-and-exact-scalar-power-realization.md) | Complete | 0005E | Added exact/default same-typed FLOAT32/FLOAT64 binary and scalar DIV plus direct scalar `POW`; retained semantic `POW` while selecting only the proved positive-one, identity, one-multiply square, or one-division reciprocal realizations. All three family opcodes preserve the completed pointwise boundaries, and reciprocal power remains semantically distinct from DIV. |
| 0005G | [Extrema, clamp, Tensor power, and logical coverage](tasks/0005g-extrema-clamp-tensor-power-and-logical-coverage.md) | Complete | 0005F; Model 0018T/0018U/0025A | Added exact same-typed binary/scalar MIN/MAX, first-class floating CLAMP, direct floating Tensor/Tensor POW, and canonical-BOOL AND/OR/NOT through the existing family pipeline. Every new row is scalar or parallel-scalar; one-instruction CLAMP, completed budgets, and fail-closed cross-type CAST/unary boundaries remain preserved. |
| 0005H | [Portable unary, transcendental, and activation closure](tasks/0005h-portable-unary-transcendental-and-activation-closure.md) | Complete | 0005G; Model 0018P/0018T1/0019A semantics; Java 26 math/Vector contracts | Closed all nineteen FLOAT32/FLOAT64 unary kinds through the existing family pipeline, preserved completed NEG and classification rows, extended/corrected GELU, selected an explicit scalar/vector algorithm, special-value, and tolerance matrix, advanced to schema 8, and retained cross-type CAST fail-closed. |
| 0005I | [FLOAT32 vector parity and vector-emission boundary](tasks/0005i-float32-vector-parity-and-vector-emission-boundary.md) | Complete | 0005H; Java 26 `FloatVector`/`DoubleVector`; completed pointwise access, specialization, and numerical contracts | Added preferred-species FLOAT32 parity for the exact twenty-one current vector-eligible pointwise opcodes, kept every unsupported row on scalar fallback, split operation-level vector bytecode emission from pure vector math, and made the selected ERF/GELU coefficient provenance and binary32 derivation auditable. |
| 0005J | [Bounded pointwise coverage and parity hardening](tasks/0005j-bounded-pointwise-coverage-and-parity-hardening.md) | Complete | 0005I; current Model pointwise semantics; Java 26 Byte/Int/Long/Float/Double Vector API contracts | Added exact floating extrema/clamp/ReLU/sign/cast, signed-integral arithmetic/extrema/cast, canonical-BOOL logic/cast, and virtual floating-mask-to-WHERE preferred-species parity; schema 10, deterministic scalar fallback, and all completed budgets remain explicit. |
| 0006 | [Portable static affine views and boundary materialization](tasks/0006-portable-static-affine-views-and-boundary-materialization.md) | Complete | 0005J | Folds bounded straight-line static RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE, SELECT, and normalized SLICE mappings without computation or slots when they remain internal; otherwise generates one exact portable scalar or parallel-scalar boundary copy, including CONTIGUOUS, for all six Model data types through the resulting seven carrier forms. The new `SHORT_ARRAY` form copies BFLOAT16 represented bits only and does not advertise BFLOAT16 arithmetic or numerical support. |
| 0006A | [Portable pad, tile, and tensor-composition movement](tasks/0006a-portable-pad-tile-and-tensor-composition-movement.md) | Complete | 0006 | Added one fully static resolved-layout PAD, TILE, CONCAT, or STACK node through compact CPU-private movement IR, unique multi-input declarations, exact represented-bit scalar/parallel-scalar generation, and one materialized injective output; schema 12 records movement identity. |
| 0006A1 | [Portable static window extraction](tasks/0006a1-portable-static-window-extraction.md) | Complete | 0006A | Added one-node fully static resolved-layout UNFOLD_AXIS for all six current types and both ordered floating-only UNFOLD2D variants through the completed movement foundation, with exact axis/NCHW mapping, dilation, floor/ceil grids, conceptual-zero and typed-padding represented bits, compact unequal-rank geometry, and schema 13; no fold accumulation. |
| 0006A2 | [Portable gather and one-hot indexing](tasks/0006a2-portable-gather-and-one-hot-indexing.md) | Complete | 0006A1 | Added GATHER, GATHER_ELEMENTS, GATHER_ND, and ONE_HOT with INT32/INT64 carriers, a complete pre-write execution-time index-validation pass, deterministic first-invalid-index failure, scalar or parallel-scalar writes, and schema 14; no wrap, clamp, default selection, partial Gather output, or all-false invalid one-hot row. |
| 0006B | Portable functional update, scatter, and overlap-fold coverage | Draft | 0006A2 | Add SLICE_UPDATE, SCATTER_ELEMENTS, SCATTER_ADD, SCATTER_ND, FOLD_AXIS, and FOLD2D with exact base participation, bounds checks, duplicate-target policy, overlap accumulation, determinism, and safe split fallback. |
| 0006C | Portable stable ordering and selection coverage | Draft | 0006B | Add stable SORT and ARGSORT plus two-output TOP_K with exact NaN, signed-zero, tie, empty-axis, output-order, workspace, and multi-store behavior. |
| 0006D | Portable explicit-state RNG and dropout coverage | Draft | 0006C | Select and version one CPU-private portable counter-based generator configuration, materialize INITIAL_STATE, and add three-output DROPOUT with exact state advancement, auxiliary mask, bounded replay, and multi-store behavior. |
| 0007 | Portable reduction, scan, statistics, and normalization family coverage | Draft | 0006D; completed 0005A–0005J | Generate family-specific range, tile, partial-reduction, and combine bodies for aggregates, arg extrema, scans, softmax/log-softmax, statistics, and normalization with exact semantics and determinism. |
| 0008 | Portable linear algebra, convolution, pooling, attention, and loss coverage | Draft | 0002–0007 | Generate the remaining portable executable families. Establish a bounded initial epilogue direction for MATMUL or convolution followed by an optional compatible bias ADD and at most one already-supported exact pointwise activation or clamp, only when single-use dataflow, Shape/layout, numerical order, publication, and resource rules preserve semantics; all other forms split safely. |
| 0008A | General partition-DAG computation-unit decomposition and bounded fusion | Draft | 0006–0008 | Decompose one CPU-owned partition directed acyclic graph (DAG) into computation units, then admit bounded vertical and horizontal fusion only across legal edges. Bound fan-out, indexing complexity, generated-code size, simultaneously live values, and unit/candidate count; preserve a deterministic materialized split fallback whenever fusion is illegal, over budget, or unprofitable. |
| 0008B | Typed specialized-subgraph and epilogue recognition | Draft | 0007–0008A | Add CPU-private typed recognition for a selected closed set of specialized subgraphs: initially MATMUL, convolution, and reduction epilogues plus explicit semantic kernels. Add no public pattern registry or domain-specific language (DSL), no new Model kinds, and no recognition that silently turns decomposed softmax into stable `SOFTMAX`. Unrecognized or ineligible graphs retain ordinary decomposed units. |
| 0008C | Bounded fusion profitability and typed decision facts | Draft | 0008A–0008B | Rank only complete legal fused and split candidates with bounded safe no-measurement heuristics by default. Retain typed cold accepted, rejected, and selected decision facts, including legality rejection separately from profitability rejection, for later Trace backend payload translation and tuning inspection without exposing a public registry or moving selection into Runtime. |
| 0009 | Portable generated-coverage closure checkpoint | Draft | 0001–0008C, explicitly including 0005A–0005J and 0008A–0008C; complete current selected Model semantic inventory | Prove the bytecode/Vector portable route is the truthful supported semantic baseline and fallback, including safe general DAG decomposition, bounded recognition/fusion, deterministic split fallback, and typed cold decision evidence. Classify metadata-only work, prove unsupported work fails closed, and close capability/conformance before native peer-route expansion. |
| 0010 | Narrow OpenBLAS BLAS-compatible native route | Draft | 0005A; 0009; completed OpenBLAS provider | Add only `route.nativeblas.openblas` for eligible BLAS-compatible linear algebra, preserving portable alternatives and using shared lowering, representations, exact filtering, materialization accounting, and whole-plan transition cost; never treat OpenBLAS as universal or preferred. |
| 0011 | Intel oneMKL BLAS and VML peer routes | Draft | 0005A; 0009; concrete Intel CPU use case and supported oneMKL ABI evidence | Add distinct `route.nativeblas.mkl` BLAS and `route.nativeops.mkl` VML leaves over shared analysis, without duplicating graph interpretation, fusion, access planning, or lifecycle ownership. |
| 0012 | Intel oneDNN partition peer routes | Draft | 0005A; 0009; stable common CPU lowering; concrete DNN/ML use case and supported oneDNN ABI evidence | Add `route.nativeops.onednn` as a distinct eligible partition route over common lowering/IR and whole-plan cost, without collapsing it into oneMKL or portable code generation. |
| 0013 | Apple Accelerate peer routes | Draft | 0005A; 0009; concrete Apple CPU use case and supported Accelerate ABI evidence | Add `route.nativeblas.accelerate` for BLAS and `route.nativeops.accelerate` for vDSP/vForce over shared analysis; Apple Silicon is capability-selected, while MPSGraph and Metal kernels remain outside CPU. |
| 0014 | AMD AOCL-BLAS and AOCL-LibM peer routes | Draft | 0005A; 0009; concrete AMD CPU use case and supported AOCL ABI evidence | Add distinct `route.nativeblas.aocl` and `route.nativeops.aocl` leaves over shared analysis and whole-plan cost, preserving the portable fallback and avoiding provider-owned lowering. |
| 0015 | Optional AMD ZenDNN partition peer routes | Draft | 0014; 0005A; 0009; stable common CPU lowering; concrete ZenDNN use case and integration evidence | Add `route.nativeops.zendnn` only for verified eligible DNN partitions, distinct from AOCL and portable generation and without another backend identity. |
| 0016 | Compatible CPU workload-tuning-cache selection | Draft | 0004; 0010–0015 as implemented; deferred Prepare opaque-candidate handoff; stable tuning-artifact compatibility | Reuse only compatible persistent selected-route evidence while retaining exact filtering and safe heuristic fallback; keep selected-route evidence distinct from the generated-class artifact store and add no measurement or tuning-cache mutation to CPU prepare. |
| 0017 | Explicit relaxed numerical candidate consumption | Draft | Config 0006; 0005F; stable exact portable and implemented peer-route consumers | Admit and compare relaxed portable or vendor candidates only under explicit caller permission, keep common analysis authoritative for eligibility and selected realization plans, and include numerical mode in compatibility/manifests without hot-path policy lookup. |


## Milestones

- CPU representation, binding, generator, and durable artifact-store foundation
- Portable generated family coverage and closure
- Optional native routes and complete candidate selection
- Tuning-cache integration and conformance

## Current status

CPU 0005A is Complete. It atomically replaced the provisional work from CPU 0001–0005, whose
task records are now Superseded but remain preserved as historical evidence. The current module
has exactly one detailed implemented partition-kernel slice and one route leaf,
`internal.route.portable`. Detailed CPU 0005B and detailed
[CPU 0005C Vector and parallel portable strategies](tasks/0005c-vector-and-parallel-portable-strategies.md)
are Complete. Detailed
[CPU 0005D Materialization, specialization, and persistence evidence gate](tasks/0005d-materialization-specialization-and-persistence-evidence-gate.md)
is Complete; detailed
[CPU 0005E Portable pointwise types, carriers, and semantic-family expansion](tasks/0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md)
is Complete. Detailed
[CPU 0005F Floating division and exact scalar-power realization](tasks/0005f-floating-division-and-exact-scalar-power-realization.md)
is Complete. Detailed
[CPU 0005G Extrema, clamp, Tensor power, and logical coverage](tasks/0005g-extrema-clamp-tensor-power-and-logical-coverage.md)
is Complete. Detailed
[CPU 0005H Portable unary, transcendental, and activation closure](tasks/0005h-portable-unary-transcendental-and-activation-closure.md)
is Complete. Detailed
[CPU 0005I FLOAT32 vector parity and vector-emission boundary](tasks/0005i-float32-vector-parity-and-vector-emission-boundary.md)
is Complete. Detailed
[CPU 0005J Bounded pointwise coverage and parity hardening](tasks/0005j-bounded-pointwise-coverage-and-parity-hardening.md)
is Complete. Detailed
[CPU 0006 Portable static affine views and boundary materialization](tasks/0006-portable-static-affine-views-and-boundary-materialization.md)
is Complete. The fresh operation inventory first split the former broad row into this static
resolved-layout affine slice and later non-affine/index, functional scatter/fold, ordering, and
explicit-state random work. Planning for the next frontier found that the remaining
non-affine/index row still combined three distinct dependencies. Detailed
[CPU 0006A Portable pad, tile, and tensor-composition movement](tasks/0006a-portable-pad-tile-and-tensor-composition-movement.md)
is now `Complete`; detailed
[CPU 0006A1 Portable static window extraction](tasks/0006a1-portable-static-window-extraction.md)
is `Complete`, and detailed
[CPU 0006A2 Portable gather and one-hot indexing](tasks/0006a2-portable-gather-and-one-hot-indexing.md)
is `Complete`. It delivers Gather/one-hot index loading plus complete pre-write execution-time
validation. CPU 0006B and every later CPU task remain `Draft` without
detailed specifications.

Superseded task 0001 remains preserved through its sole detailed CPU
[capability, representation, binding, and parallel foundation](tasks/0001-cpu-capability-representation-binding-and-parallel-foundation.md)
specification. It introduces one truthful provider with stable `cpu` identity and no advertised
operation semantics, package-private aligned native representation and typed cold-binding
infrastructure, and a package-private bounded worker/range foundation. It does not add a CPU
preparer, executable semantic coverage, a route, generated code, OpenBLAS integration, or another
module change. Superseded
[task 0002 Portable Class-File API generator foundation](tasks/0002-portable-class-file-api-generator-foundation.md)
now adds the backend-private deterministic generator foundation and synthetic probes.
`CpuPortableExecutionMode.emit(CodeBuilder, CpuKernelSpecialization,
CpuFamilyKernelEmitter)` owns only structural scalar-versus-Vector emitter construction and
dispatch, while `CpuClassFileKernelGenerator` delegates without that switch. The trusted
`CpuLoweringFingerprint.fromDigest(byte[])` factory retains only an exact defensively copied
32-byte derived digest without rehashing. Scalar modes require `Tail.NONE`; Vector modes permit
`NONE`, `SCALAR`, and `MASKED`. Baked primitive-array byte offsets require data-type-width
alignment, exact segments retain baked zero offset, and dynamic array offsets remain cold typed
invocation values. Equal requests produce identical bytes and distinct hidden artifacts because
the direct generator path remains fresh. At the CPU-0002 checkpoint, CPU advertised and executed
no Model operation.
Superseded
[task 0003 Durable generated-kernel artifact store and cold loading](tasks/0003-bounded-generated-artifact-cache-and-cold-finalization.md)
adds one CPU-private durable reuse source: a
model-independent filesystem store under an explicit caller/composition-supplied trusted local
root. Deterministic content-addressed paths retain exact compatibility metadata and verified
class bytes in a self-contained envelope published by forced temporary-file write and atomic
move. Stored hits are accepted only after full structural metadata, checksum, class-file, and
entry-shape validation. Process-local equal requests single-flight, while loaded artifacts are
interned only weakly with stale-key cleanup; no strong completed LRU, expiry, or age-based
invalidation exists. The generator now separates deterministic byte emission from revalidated
stored-byte definition, and exact class-shape validation remains generator-owned. No production
operation-family source uses the store yet. Superseded CPU 0004 historically owned the CPU-private
integration, explicit root seam, and prepared-executable strong retention. Detailed
[task 0004 Typed portable analysis, specialization, and finalization](tasks/0004-typed-portable-analysis-specialization-and-finalization.md)
is Superseded. It added only the CPU-private staged lifecycle foundation: deterministic selection
from a direct typed candidate source, exact declarations before assignment, artifact-store use
only during finalization afterward, immutable generated-kernel/direct-handle retention, and
family-owned direct typed cold binding. CPU analysis preserves backend-declared byte geometry;
shared Prepare validates that geometry against assigned slots, and no dense-layout or
materialization policy was introduced. Its tests remain bounded and synthetic,
`CpuCapabilityProvider` remains fail-closed, and no real Model operation family is implemented or
advertised. CPU task 0005 is Superseded through its sole detailed
[Dense ADD and partition-sequence execution](tasks/0005-dense-add-and-partition-sequence-execution.md)
specification. It solves maximal-partition truth with an ordered node-kernel sequence while
limiting first executable coverage to static canonical dense FLOAT64/FLOAT32/INT32/INT64 ADD
through scalar single-thread native segments. Implementation context
`019fd19a-4262-7580-9674-226595356fbc` recorded the sole final CPU 31-test pass; the clean
documentation pass reused that evidence because it changed no executable Java behavior.

Detailed
[task 0005A Atomic partition-kernel architecture reset](tasks/0005a-atomic-partition-kernel-architecture-reset.md)
is Complete. It delivered the intentional atomic capability reset before family expansion:
structured internal packages with only `route.portable`, a route-neutral selected-plan
seam, whole-partition lowering, pre-declaration fusion, route-independent IR, universal primitive
start/end loops, normal in-memory Class-File generation, optional cold persistence, and one
partition executable. Its exact proving chain is canonical-dense
FLOAT64 ADD -> exact GELU -> MUL in one unit, with two logical graph/IR intermediates and no
physical slots for either. The same generated bytes and loaded compatibility identity must serve
two compatible extents. Detailed
[task 0005B Universal access plans and right-aligned broadcasting](tasks/0005b-universal-access-plans-and-right-aligned-broadcasting.md)
is Complete. It adds right-aligned static broadcast/layout normalization, exact per-boundary
declaration and accessed-range spans, complete write-injectivity proof, five generated scalar
state machines, and all sixteen ordered heap/segment carrier specializations. Detailed CPU 0005C
is Complete. CPU 0005D, detailed CPU 0005E, detailed CPU 0005F, and detailed CPU 0005G are also
Complete. Detailed CPU 0005H, detailed CPU 0005I, and detailed CPU 0005J are `Complete`;
task 0006 and detailed task 0006A are `Complete`, detailed task 0006A1 is `Complete`, detailed
task 0006A2 is `Complete`, and tasks 0006B–0017 remain `Draft` without
detailed specifications.
CPU 0005C preserves that exact slice and implements cold selection among all four portable
strategies. It uses the preferred Java 26 FLOAT64 species only for direct contiguous runs and
scalar broadcasts, scalar tails and general-odometer fallback, configured/available parallelism
bounds, deterministic disjoint chunks, and one explicit caller-owned CPU-private worker group
borrowed by finalization/execution. Strategy/species alter generated identity; extents, chunk
configuration, and worker identity remain cold facts. It adds no gather, masked tail,
materialization, shared lifecycle, Config surface, or performance claim.

The CPU 0005C implementation context's final corrected `./gradlew :backends:cpu:test` run passed
18 suites and 49 tests with zero failures, errors, or skips. Clean documentation context
`/root/cpu_0005c_docs` reused that evidence because no executable Java or test changed afterward,
finalized affected Javadocs, package summaries, the CPU guide, glossary, and planning records, and
passed CPU Javadoc plus the recorded Markdown, exact-scope, status, forbidden-vocabulary, and
whitespace gates.

Detailed CPU 0005D is Complete. CPU analysis can now compare direct access with at most one
eligible contiguous input copy, append exact workspace ID `0` before assignment, and retain the
original source separately from the adjusted generated consumer pattern. Finalization resolves
every assignment before one artifact lookup; execution copies once before consumer work. The
specialization budget is four complete candidates, one realized artifact, zero fixed-shape
variants, and zero unrolled variants. Optional explicit-root persistence uses one bounded verified
current-schema envelope. The recorded six-fixture Oracle JDK 26.0.1 evidence verdict is
`KEEP_DISABLED`, so default persistence remains off. Class bytes, JVM JIT machine code/profile,
and the future workload tuning cache remain distinct.

Implementation evidence includes the final 11-suite/40-test focused matrix, focused 8-, 13-, and
7-test runs, the sole final ordinary 21-suite/62-test CPU pass with one skipped evidence method,
and the sole successful explicit 1-test evidence run. Clean documentation context
`/root/cpu_0005d_docs` reused those stabilized results, finalized the exact 21 production/package
paths and five Markdown records, and passed CPU Javadoc plus Markdown, exact 38-path scope,
status, terminology, structure/order, persistence, and whitespace gates.

Detailed CPU 0005E is Complete. One through eight fully static connected straight-line pointwise
occurrences lower into one nineteen-opcode CPU-private sequence, one canonical IR, one artifact,
and one partition executable. The admitted matrix uses FLOAT64, FLOAT32, INT32, INT64, and BOOL
with exact typed array or segment carriers; boundaries are derived deterministically, internal
single-use results stay virtual, and exactly one final store is materialized. Scalar and
parallel-scalar generated execution cover every admitted row, while vector compute remains
FLOAT64 numeric-only. Same-type CAST preserves represented values; cross-type CAST and excluded
pointwise or later semantic families remain fail-closed. The generator schema is 5, default
persistence remains disabled, and no public API, common lowering, shared module, architecture,
dependency, build, conformance, or integration contract changed. Implementation evidence includes
the required focused matrix, a broader focused regression batch, and the sole final 24-suite/
73-test CPU pass with zero failures/errors and one opt-in evidence skip. Clean documentation
context `/root/cpu_0005e_docs` reused that evidence, finalized affected Javadocs/package summaries
and five Markdown records, and passed CPU Javadoc plus Markdown, exact 38-path scope, semantic,
status, and whitespace gates.

Detailed CPU 0005F is Complete. It extends the family pipeline to twenty-two opcodes with exact
same-typed FLOAT32/FLOAT64 binary DIV, scalar DIV, and scalar POW. Common lowering classifies exact
exponent bits once as direct, positive one, identity, square, or reciprocal; reciprocal remains
semantic scalar POW. FLOAT64 DIV and special power retain existing conditional vector eligibility,
while direct power and FLOAT32 use scalar compute with optional parallel orchestration. Schema 6,
canonical IR, specialization, and the cold manifest retain realization compatibility. The
implementation context passed compile-test, focused generated-kernel validation, a final 16-test
IR/lowering/generated run, and the sole final 25-suite/102-test CPU run with one opt-in timing
skip and no failures or errors. Clean documentation context `/root` reused that Java evidence and
passed CPU Javadoc, Markdown, exact 31-path scope, semantic/status, and whitespace gates.

Detailed CPU 0005G is Complete. It extends the closed family pipeline to 31 opcodes with exact
same-typed FLOAT32/FLOAT64/INT32/INT64 binary and scalar extrema, first-class FLOAT32/FLOAT64
range CLAMP, direct same-typed FLOAT32/FLOAT64 Tensor power, and canonical-BOOL AND/OR/NOT.
CLAMP remains one occurrence and one IR instruction with exact ordered raw bounds. All nine new
opcodes use scalar or parallel-scalar compute, while existing vector coverage and all fusion,
access, materialization, specialization, artifact, and lifecycle budgets remain unchanged.
Generator schema 7 rejects every older envelope without migration. Implementation context
`/root/cpu_0005g_impl` passed the focused 9-suite/41-test command and exactly one final
25-suite/106-test CPU command with zero failures/errors and one existing opt-in persistence-timing
skip. Clean documentation context `/root/cpu_0005g_docs` reused that evidence, finalized affected
Javadocs and five Markdown records, and passed CPU Javadoc, Markdown, exact authorized-scope,
semantic/status, and whitespace gates without changing executable Java.

Detailed CPU 0005H is Complete. It extends the closed family pipeline to exactly 48 opcodes and
schema 8, with one distinct opcode for each of the nineteen same-typed FLOAT32/FLOAT64 unary
semantics and the existing three floating classifications unchanged. Scalar and parallel-scalar
cover every unary row; the selected FLOAT64 lane-operator and ERF/GELU rows also admit vector and
parallel-vector compute, while vector-ineligible chains select scalar compute. GELU now covers
both precisions and maps negative infinity to negative zero; stable sigmoid, tanh-approximation
GELU, and SiLU preserve their fixed exceptional-value contracts. One-through-eight fusion, five
access regimes, one-copy/four-candidate/one-artifact materialization budgets, zero fixed-shape and
unroll budgets, same-type CAST, and fail-closed cross-type CAST/BFLOAT16 remain unchanged.

Implementation context `/root/cpu_0005h_impl` passed CPU compilation, test compilation, the
required focused nine-class command with 43 tests, and exactly one new final 25-suite/108-test CPU
command with zero failures/errors and one expected opt-in persistence-timing skip after correcting
FLOAT32 RSQRT to perform its square root and reciprocal before one final narrowing. This supersedes
the earlier 42-test/107-test evidence. Clean documentation context `/root/cpu_0005h_docs` reopened,
reused the restabilized Java evidence, finalized the
affected Javadocs/package summaries, CPU guide, glossary, and planning records, and passed CPU
Javadoc, Markdown/link/anchor/formatting, exact authorized-scope, semantic/status, and whitespace
gates without rerunning Java tests.

Detailed CPU 0005I is Complete. It closes the intentional FLOAT64-only vector realization before
family expansion with preferred-species FLOAT32 parity for the exact existing twenty-one eligible
opcodes. Package-private final `CpuVectorInstructionEmitter` owns the closed family switch, while
package-private final `CpuVectorMath` owns pure `FloatVector`/`DoubleVector` formulas. The retained
Cephes binary64 ERF family now has exact source-stable binary32 rounded tables, independent-oracle
and scalar-differential evidence, and explicit FLOAT32 bounds. Schema 9 rejects every older
envelope. Scalar fallback and tails, FLOAT64-only materialization, one-through-eight fusion, the
four-candidate/one-artifact/zero-fixed-shape/zero-unroll budgets, optional persistence policy,
capability, public API, modules, dependencies, and architecture remain unchanged.

The implementation pass's revised focused eleven-class command passed 11 suites/62 tests. The
final CPU command passed 26 suites/117 tests with zero failures/errors and one expected opt-in
persistence-evidence skip. Clean documentation context `/root` reused that evidence, finalized
the affected Javadocs, CPU guide, glossary, and planning records, and passed CPU Javadoc plus
local Markdown, official-link, exact-scope/status/inventory, formatting, and whitespace gates
without rerunning Java tests. CPU 0005J and detailed CPU 0006 are Complete. CPU 0006A's later
completion is recorded below; detailed CPU 0006A1 and CPU 0006A2 are `Complete`, and CPU 0006B
and later tasks remain `Draft` without detailed
specifications.

Detailed CPU 0005J is Complete. It preserves the exact forty-eight-opcode semantic inventory and
adds preferred-species FLOAT32/FLOAT64 extrema, clamp, ReLU, sign, and same-type cast;
INT32/INT64 modular arithmetic, signed extrema, and same-type cast; canonical BOOL logic and
same-type cast; and virtual floating comparison/classification masks through logical masks into
floating WHERE. A scalar/all-zero external BOOL condition may become an all-true or all-false
mask, while materialized masks and non-scalar external conditions remain scalar. Schema 10 keys
the exact typed species and mask topology. General odometers, too-short runs, unsupported
opcode/type pairs, direct scalar power, and unsafe mask boundaries deterministically select scalar
or parallel-scalar compute. The implementation preserves one-through-eight straight-line fusion,
one final store, scalar tails, FLOAT64-only one-input materialization, four candidates, one
artifact, and zero fixed-shape/unroll variants. The implementation pass compiled production and
tests; its required twelve-class matrix passed 12 suites/76 tests; additional focused generated-
kernel, vector-math, specialization, and preparation runs passed; and its sole final CPU suite
passed 26 suites/125 tests with zero failures or errors and one skipped opt-in persistence-
evidence test. The clean documentation pass reused that stable Java evidence, finalized the
authorized Javadocs and documentation, and passed CPU Javadoc, Markdown, exact-scope, semantic,
status, unchanged-layer, and whitespace checks without rerunning Java tests.

Detailed CPU 0006 is Complete. It composes one-through-eight connected static resolved-layout
affine view occurrences into one source-to-result mapping, keeps same-unit internal view values
virtual without declarations or Runtime slots, and emits one exact scalar or parallel-scalar
represented-bit boundary copy. The result retains its resolved offsets and positive or zero
strides; non-injective layouts use deterministic distinct-address writes only after proving that
repeated logical coordinates select the same source value. All six Model data types copy across
array, segment, and mixed carriers, with the seventh `SHORT_ARRAY` form limited to raw BFLOAT16
payloads. Schema 11 distinguishes affine structure and carrier compatibility. The implementation
pass's focused command and sole final CPU command passed; the final suite recorded 29 suites and
140 tests with zero failures, zero errors, and one skipped opt-in persistence-evidence test on
OpenJDK 26.0.1+8-34. Clean documentation context `/root/cpu_0006_docs` reused that stabilized Java
evidence, finalized Javadocs/package summaries, the CPU guide, glossary, and planning records,
and passed CPU Javadoc plus Markdown, exact-scope/status, and whitespace validation without
rerunning Java tests.

Detailed CPU 0006A is Complete. It adds one exact fully static resolved-layout PAD, TILE, CONCAT,
or STACK occurrence with one-through-sixteen ordered semantic inputs, first-occurrence unique
resource declarations, compact cold movement geometry, all-six-type represented-bit scalar or
parallel-scalar generation, and one distinct injective output. Generator schema 12 records family,
rank, occurrence mapping, and exact PAD bits. The focused command passed 8 suites/43 tests and the
sole final CPU command passed 32 suites/153 tests with zero failures or errors and one skipped
opt-in persistence test on Java 26.0.2, HotSpot 26.0.2+10-55. Clean documentation context `/root`
reused that stabilized evidence, finalized Javadocs/package summaries, the CPU guide, glossary,
and planning records, and passed CPU Javadoc plus Markdown, exact-scope/status/inventory, forbidden-
change, and whitespace validation without rerunning Java tests. Detailed CPU 0006A1 is `Complete`;
detailed CPU 0006A2 is `Complete`, and CPU 0006B and every later task
remain `Draft` without detailed specifications.

Detailed CPU 0006A1 is Complete. It extends the same movement pipeline with one fully static,
resolved-layout UNFOLD_AXIS occurrence for all six represented types or one floating UNFOLD2D
occurrence using direct positive-zero or exact matching typed padding. The implementation keeps
axis/NCHW mapping, dilation, floor/ceil grids, unequal-rank strides, and arbitrary-range start
state in compact cold geometry; emitted loops use carry/reset odometers and schema 13 records only
code-shaping identity. The focused matrix passed 10 suites/63 tests, and the sole final CPU suite
passed 32 suites/163 tests with one expected persistence skip on OpenJDK 26.0.1+8-34. Clean
documentation context `019ff061-4aa7-7f62-b2c5-7a1aed7b16f4` reused that stable evidence,
finalized Javadocs, the CPU guide,
glossary, and planning records, and passed CPU Javadoc plus scope, schema, status, Markdown, and
whitespace gates without changing executable Java.

Detailed CPU 0006A2 is Complete. It adds one fully static resolved-layout GATHER,
GATHER_ELEMENTS, GATHER_ND, or ONE_HOT occurrence through distinct mixed-type indexing IR,
compact geometry, deterministic complete run-bound validation, and one schema-14 output-writing
artifact. Unique inputs plus one output are declared with no materialization or workspace; valid
zero-output execution makes no generated call or worker submission. Implementation context
`019ff098-313c-7b53-8a03-df9f31fcf71f` established the implementation. Mandatory clean audit/fix
context `019ff0bc-5996-7c30-903e-6f32d1b53a36` then found no production defect, expanded the
allowlisted regression matrix, and recorded the final focused 12-suite/71-test pass plus final CPU
35-suite/183-test pass with one expected persistence-evidence skip and zero failures/errors on
Oracle OpenJDK 26.0.1, Runtime/VM 26.0.1+8-34. Mandatory clean documentation context
`019ff0c9-83cf-7d82-8f3c-e61be7a30269` independently finalized Javadocs, package summaries, the CPU
guide, glossary, and planning/status records and reused that stable executable evidence without
rerunning Java tests.

The reset was a working-tree replacement, not deletion of history. CPU 0001–0005 are Superseded
with all recorded evidence preserved; the repository contains no old/new dual pipeline.

Future Model task 0026 is required before any CPU route advertises FLOAT16. That dependency does
not block tasks 0003–0004 or later current-type work: generated-artifact storage, portable
analysis/finalization, and current-type family coverage may proceed while remaining fail-closed for
FLOAT16.

The separately authorized architecture synchronization remains complete: the authoritative
contract and focused explanations permit implementation-neutral generated JVM-bytecode CPU
computation kernels while preserving CPU backend and Prepare ownership. Implementation context
`019fc815-42aa-7de2-8970-a2fcab3a390e` recorded the final focused 3-suite/18-test pass and the
sole final CPU 9-suite/34-test pass, both with zero failures, errors, or skips. The clean
documentation pass reused that executable evidence and finalized CPU Javadoc, guide, glossary,
and planning validation. The Class-File API and Vector API remain selected CPU-internal
implementation choices rather than architecture invariants.

Implementation context `019fc96e-494b-74f2-b6e9-5b55d649cd6c` recorded CPU 0003's focused
3-suite/30-test pass and sole final CPU 10-suite/48-test pass, both with zero failures, errors, or
skips. The mandatory clean documentation pass reused that evidence, finalized the authorized
Javadocs, guide, glossary, and planning records, and changed no executable Java.

The CPU 0004 implementation pass recorded a final focused three-suite/24-test pass and the sole
stabilized CPU module 13-suite/72-test pass, both with zero failures, errors, or skips. Mandatory
clean documentation context `/root/cpu_0004_docs` reused that evidence, finalized all affected
production/package Javadocs, the CPU guide, glossary, task/master/roadmap records, and passed CPU
Javadoc plus exact-scope, Markdown, surface, status, excluded-path, and whitespace checks without
changing executable Java.

## Deferred pointwise follow-ups

These named follow-ups are deliberately non-blocking planning candidates rather than ordered task
rows. They do not delay CPU 0006, reductions in CPU 0007, or matrix multiplication in CPU 0008.
Create a detailed specification only if the stated trigger becomes current and the roadmap inserts
the work at the active frontier.

- **Remaining pointwise vector-policy closure** — reconsider vector FLOOR/CEIL, stable complex
  activations, materialized comparison/classification masks, general external BOOL-mask loading
  for floating WHERE, and other evidence-backed pointwise rows after CPU 0005J. The trigger is a
  complete exceptional-value/numerical proof or a direct portable operator plus a bounded carrier/
  mask representation that does not require per-lane scalar reinsertion, broad materialization, or
  a shared Runtime mask contract.
- **Cross-type CAST execution** — implement exact CPU conversions only after a Model-owned contract
  defines source/target compatibility and observable rounding, truncation or saturation, overflow,
  underflow, NaN, infinity, signed-zero, BFLOAT16, and BOOL conversion behavior. Java primitive or
  Vector API conversion availability is not that semantic authority.

## Open questions

- Exact route-specific configuration records, target fingerprints, and candidate-schema versions
  wait for implemented CPU routes and the shared opaque orchestration consumer.
- Vendor ABI/lifetime layers remain inside the CPU backend unless a later explicit architecture
  decision authorizes another provider module and dependency edge. This plan does not add such a
  module.
- Exact oneDNN partition/fusion coverage and ZenDNN coverage wait for concrete integration use
  cases. Apple Basic Neural Network Subroutines (BNNS) are deferred without a task row until a
  concrete integration spike or use case establishes scope and advantage.
- A future explicit numerical-policy contract must define any relaxed/fast-math permission before
  such a vendor candidate can become eligible. Current exact/default semantics admit none.
- Exact/default floating division and scalar-power realization are independent of that future
  permission. Complete 0005F adds primitive same-typed FLOAT32/FLOAT64 binary and scalar DIV and
  selects only its proved positive-one, identity, one-multiply square, and one-division reciprocal
  power realizations; every other admitted exponent retains direct power. Reciprocal power remains
  semantic `SCALAR_POW`, not DIV. Draft 0017 later consumes Config 0006 permission for genuinely
  relaxed candidates and therefore does not block exact work.
- Historical task 0001 implements the exact native `MemorySegment` representation, shared-arena
  ownership, zero-size/alignment/allocation/cleanup rules, borrowed heap/native cold binding,
  direct typed invocation seam, and worker evidence. Task 0005A retains the representation and
  binding contracts but removes unused worker types; completed task 0005C implements the minimum
  caller-owned worker group required by its parallel strategies. Route-specific
  representations and materializations still wait for their ordered tasks.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- The former broad 0006A row was dependency-split before implementation. Complete 0006A owns one-node
  static PAD/TILE/CONCAT/STACK movement and unique multi-input binding; detailed Complete 0006A1
  owns window extraction; detailed Complete 0006A2 owns Gather/one-hot plus complete pre-write
  index validation. All three slices have detailed task specifications;
  0006B and later remain Draft without detailed specifications.
- CPU 0006A declares each distinct composition input `ValueId` once in first-occurrence order and
  retains a CPU-private ordered occurrence-to-boundary map. Repeated semantic inputs do not force
  duplicate shared requirements. The one output is separate, injective, and materialized.
- Value-dependent index work begins only in 0006A2. That slice must validate all INT32/INT64 index
  values before any output write and deterministically reject the first invalid logical index;
  invalid indices never wrap, clamp, select defaults, expose partial Gather output, or become an
  all-false one-hot row.
- Task 0005A replaces the provisional per-node implementation atomically. The permanent model is
  partition -> computation-oriented execution units -> canonical loop IR -> route realization ->
  one partition executable. There is no compatibility layer between the two designs.
- CPU lowering operates over the complete owned partition and supports pointwise/broadcast fusion,
  metadata/view folding, reduction/scan and stable multi-pass units, matrix/convolution epilogues,
  and explicit semantic kernels as their Draft tasks land. Fusion legality and profitability are
  separate; neither may cross alias/state/random/publication/fan-out/partition or numerical-order
  boundaries. Decomposed softmax is not silently recognized as stable `SOFTMAX`.
- Draft 0008A first generalizes the current straight-line unit boundary to bounded partition-DAG
  decomposition with deterministic materialized split fallback. Draft 0008B then owns only a
  closed typed CPU-private recognition set, and Draft 0008C owns profitability ranking and cold
  decision facts. Legality rejects a candidate that cannot preserve semantics or resource rules;
  profitability may reject a legal candidate because of indexing complexity, fan-out, code size,
  live-value pressure, materialization, route eligibility, or estimated complete-plan cost.
- Draft 0008C uses safe deterministic no-measurement heuristics for ordinary preparation. Later
  Config 0006A supplies declarative model-autotuning inputs, Prepare 0004 carries candidates and
  compatible decisions opaquely, CPU 0016 consumes compatible workload-cache selections, and
  Tuning 0001–0002 may measure eligible complete candidates when the user explicitly requests
  model autotuning. Autotuning is not the default fusion-profitability mechanism, and no search,
  cache mutation, or choice occurs in Runtime. Draft 0008C's typed cold decision facts remain
  CPU-owned; later Trace backend payloads and tuning inspection may translate or consume them
  without making Trace or tuning the decision owner.
- One route-independent `CpuKernelIr` contains typed boundary/virtual values, ordered semantics,
  structural access-plan form, universal start/end loop model, and stores. Selected route, thread
  count, vector species, cache root, slots/segments, graph identities, generator versions, and
  instance bindings remain outside it.
- One normalized per-value CPU access plan eventually covers current right-aligned
  `ShapeBroadcast` and `LayoutDescriptor` semantics: scalars, rank/singleton/multi-axis/zero
  broadcast, multi-input masks and parameters, differing fused patterns, offsets, positive and
  zero strides, and heap/segment/mixed carriers. CPU 0005B covers only fully static shapes because
  current Prepare has no exact dynamic binding; a future explicit binding contract is required for
  dynamic or symbolic dimensions. It does not duplicate elementwise, `WHERE`, or fusion planners,
  and broadcast gradients remain `SUM_TO_SHAPE` plus later reduction coverage.
- CPU 0005B implements five ordered scalar regimes: dense linear, all-zero/scalar, last-axis bias,
  block/outer with a contiguous inner loop, and the complete general positive/zero-strided
  odometer. Each emits its own primitive state machine without hot cursor objects, virtual calls,
  or per-element division/modulo. Completed 0005C vectorizes only direct contiguous runs and scalar
  broadcasts, uses scalar fallback for general odometers, and promises no gather. Complete 0005D
  owns optional materialization; full semantics do not promise universal vectorization.
- Before shared assignment, CPU analysis may compare direct access with internal contiguous
  materialization using copy cost, kernel benefit, reuse/fan-out, vendor eligibility, memory, and
  repeated-run expectation. A selected copy resource is declared exactly before assignment and
  lowered without changing the Model graph; shared Prepare remains CPU-blind.
- Task 0005A replaces the flat execution package with cohesive `.internal` root contracts and
  `memory`, `prepare`, `lowering`, `ir`, `codegen.emit`, `route.portable`, `cache`, `executable`,
  and `reference` packages. It creates no native/provider leaf. Minimal cross-package contracts are
  technically public but unsupported; only `CpuCapabilityProvider` is supported public API.
- Physical CPU buffers are declared only for post-fusion unit/partition boundaries. Planning's
  logical requirement for every graph value does not force a buffer declaration or slot; a
  same-unit virtual intermediate remains represented in graph and IR only.
- Shared Prepare adds only fail-closed cross-planned-partition completeness: a producer, when
  present, and every distinct external consumer must each declare the crossing value. Shared code
  does not inspect CPU units or choose fusion/materialization.
- CPU task 0001 exposes only `CpuCapabilityProvider`. Its stable `BackendId("cpu")` is identity,
  not availability or readiness, and `supports` remains unconditionally false until later tasks
  deliver and test exact executable semantic coverage.
- Historical task 0001 placed physical representations, typed cold arguments, specialization, and
  worker/range coordination in one flat execution package. Task 0005A preserves its validated
  representation/lifetime evidence while replacing that package and removing foundations with no
  current consumer. Run-owned buffers retain exact aligned native allocation and borrowed Model
  storage remains non-owning.
- Task 0001 binds each selected representation independently. The six Model data types map to
  their exact observable primitive-array carriers with retained carrier-relative byte offsets.
  When the matching carrier is unavailable, `CpuBufferArgument.Segment` retains the exact segment
  or slice without asserting native provenance; this covers genuine native segments and JDK 26
  read-only heap segments whose `heapBase()` is empty. Route-specific bound invocations must copy
  direct typed fields out of cold arrays and perform no storage discovery in the hot call.
- Historical task 0001 proved bounded worker and range behavior. Task 0005A removes those unused
  production types while retaining their evidence. Completed 0005C implements one explicit
  caller-owned CPU-private worker group, borrowed by finalized executables, and reintroduces no
  public executor facade, shared lifecycle, or general task system.
- Java 26 `java.lang.classfile.CodeBuilder` is the selected primary generation mechanism for all
  portable CPU computation kernels. This remains a current planning choice rather than an
  architecture invariant. Native vendor providers remain separate optional routes; the generator
  does not replace OpenBLAS, oneMKL, oneDNN, Accelerate, AOCL, or ZenDNN.
- Portable capability is truthful and fail-closed. Every selected executable Model semantic must
  have generated coverage before the portable milestone closes; metadata-only or zero-work view
  occurrences require no generated computation and unsupported executable semantics are not
  advertised.
- A default generated class is specialized to one canonical computation topology plus structural
  access regime, fusion form, types, selected route/strategy/configuration, numerical/determinism
  policy, and compatibility facts. Compatible concrete extents, element count, offsets, carriers,
  addresses, slots, graph identities, and run identity are cold-bound and excluded. Fixed-shape or
  unrolled variants require explicit later evidence and consume a bounded specialization budget.
- Cold binding resolves observable heap primitive carriers, exact `MemorySegment` values whose
  matching carrier is unavailable, and mixed signatures into direct typed entry-point arguments. Generated
  hot code performs no heap-base discovery, generic type check, route choice, cache lookup, or
  storage-kind, data-type, layout, vector, parallel, broadcast, or operation switch.
- The portable execution matrix is exactly scalar, vector, parallel-scalar, and parallel-vector.
  Generated units accept primitive start/end bounds; orchestration owns chunks outside the inner
  loop. Family-specific lowering may produce range, tile, partial-reduction, multi-pass, and combine
  bodies without changing those orthogonal compute/orchestration axes.
- Operation semantics are lowered once through family-specific typed lowerers. Shared scalar,
  vector, heap, segment, range, tile, and reduction emission primitives adapt that lowering to
  storage and execution modes; the design neither duplicates semantics across heap/native and
  scalar/vector modes nor concentrates all families in one generator.
- CPU analysis selects one complete valid route, access regime, representation, specialization,
  strategy, and materialization plan. Finalization after shared slot assignment deterministically
  realizes that selection. Without a trusted root it emits, verifies, and defines in memory; with
  a root it may verify and define a compatible stored hit or emit a miss. Runtime invokes only the
  cold-resolved handle.
- Task 0004 represents each selected portable buffer form through the existing
  `CpuKernelSpecialization.Argument` carrier/access vocabulary rather than a duplicate storage
  enum. One direct typed candidate source supplies deterministic preference order; CPU 0004 adds
  no registry, service locator, universal priority, tuning evidence, benchmark, or broad cost
  model.
- `CpuBorrowedBuffer` remains CPU 0001's non-owning `HostTensorStorage` representation/lifetime
  boundary and begins production portable cold-binding use in task 0004. Direct
  `HostTensorStorage`-to-argument binding is rejected because it would bypass Runtime's
  `BufferRepresentation` lifecycle boundary. The borrowed wrapper is not an artifact cache.
- Because generated method signatures vary, task 0004 keeps signature-specific invocation
  construction family-owned through a typed cold binder. The immutable portable executable
  strongly retains both `CpuGeneratedKernel` and its exact direct handle; a bound invocation
  retains direct typed carrier/workspace/worker fields and performs no hot argument
  classification.
- Historical task 0004 received an explicit trusted root and worker group. Task 0005A makes the
  root optional and removes the unused parallel object; neither becomes a public Config API.
- Deterministic generated bytes, structural identity, verification, and safe in-memory definition
  are mandatory. Filesystem persistence under an explicit trusted root is optional cold-path
  policy. Task 0005D recorded `KEEP_DISABLED`, so the default remains persistence-free. A hit can
  avoid emission but cannot preserve JIT machine code or profile.
- The generated-artifact key includes a generator schema/version and exactly every fact that
  changes bytecode or compatibility. Thread count or chunk size is keyed only when emitted code
  changes; otherwise it remains prepared parallel-execution configuration. Key construction and
  equality are deterministic and must be validated directly.
- The reset stores and compares the complete canonical IR/specialization structure plus generator,
  CPU/JDK/Class-File, generated-class, entry-name, and entry-descriptor compatibility facts. A path
  digest or checksum alone cannot establish a hit. Concrete compatible extents and every instance,
  graph, slot, address, run, handle, class-loader, and store identity are excluded by default.
- Compatible age never invalidates an artifact. The store has no time-to-live, hit-rate policy,
  automatic disk eviction, quota, background service, or valid-entry maintenance. Corrupt or
  incompatible bytes are never defined and may be replaced only through verified generation and
  forced temporary-file plus atomic-move publication.
- Equal requests share process-local compatible realization. A prepared executable reachable from
  `PreparedExecution` strongly retains the class, lookup, and exact handle for active use. The
  optional store is not a correctness owner and the reset has no legacy-schema migration reader.
- Checksums and structural verification detect accidental corruption but do not authenticate
  executable bytes. The artifact root is an explicit trusted-local-cache security boundary whose
  write isolation and administration belong to the caller/composition owner.
- The generated-class artifact store is distinct from the persistent tuning cache: the former
  reuses exact verified executable class bytes after route selection, while the latter records
  compatible route/configuration-selection evidence. Neither performs Runtime hot-path work.
- Matrix-multiplication candidates may include supported JDK Vector API species and strategy,
  unroll, tile, parallelism, and OpenBLAS thread configurations derived and pruned from target
  capabilities, workload facts, and budget.
- Scalar/vector and single/parallel are typed strategies inside the portable route. OpenBLAS and
  vendor integrations are typed peer-route configurations, never booleans in
  `Map<String,Object>`. Operation family does not key one universal configuration.
- Concrete CPU analysis generates a complete typed candidate set for each operation occurrence or
  partition. It filters by platform and provider availability, operation and attributes, data
  type, `Shape`, layout, exact numerical and determinism compatibility, and resource validity
  before comparing call overhead, safe workload heuristics, or a compatible tuning-cache entry.
- Common CPU analysis also owns exact division eligibility, exponent classification, numerical
  eligibility, and any selected `POW` realization plan. Emitters and vendor adapters consume those
  decisions and never invent independent fast-math behavior. Complete CPU 0005F admits same-typed
  FLOAT32/FLOAT64 binary/scalar DIV and scalar power with a direct fallback for every exponent.
  Exact typed positive or negative zero may produce positive one, positive one may select
  identity, positive two may select one typed multiply, and negative one may select one typed
  division while remaining semantic scalar power. Other integral exponents retain direct power:
  multiply chains and exponentiation by squaring have unproved intermediate rounding, overflow,
  and underflow behavior. `POW(0.5)` is not silently replaced by `SQRT`. Tensor exponents require
  compiler-owned immutable uniform-constant facts and are never inferred from Tensor storage or
  factory history.
- The selected numerical mode and every realization-changing `POW` plan participate in
  specialization/cache compatibility and the cold lowering manifest. The hot path performs no
  policy lookup. These rules apply equally to forward and compiler-generated gradient operations;
  CPU has no gradient-specific numerical policy.
- Exact semantics, determinism, data type, `Shape`, layout, alignment, lifetime, and provider
  eligibility are hard filters before cost comparison. CPU preparation then chooses routes and
  physical representations jointly over the relevant CPU dataflow and partition uses. It compares
  complete valid plan cost, including kernel time, Java/native call overhead, allocation,
  copying/materialization, packing or reorder work, and resource requirements; it must not select
  a locally fastest kernel when the resulting transition plan is worse.
- Native-backed CPU representations remain directly usable by scalar Java and Vector API routes.
  Crossing into an FFM provider call alone causes no copy. A preceding Java kernel may write
  directly into the selected native output buffer so a downstream compatible native consumer
  needs no intermediate materialization.
- Specialized opaque or prepacked layouts, CPU-to-device transfer, incompatible layout or
  alignment, and explicit heap export may still require distinct representations or explicit
  materialization. The canonical internal representation is not a claim that all model data or
  all external inputs are native.
- Compile and Planning remain logical and backend-neutral. Concrete CPU preparation owns route
  plus representation requirements; shared Prepare assigns and reconciles slots and explicit
  materializations across the complete prepared uses; Runtime executes that prepared schedule and
  tracks representation validity and residency.
- CPU uses canonical native internal buffers plus per-value, use-aware handling of borrowed and
  specialized representations. It does not introduce an all-Java versus all-native model mode.
- The Model `HostTensorStorage` contract is unchanged: `MemorySegmentStorage` remains borrowed and
  accepts compatible heap-backed and native-backed segments without gaining Runtime ownership,
  alignment, allocation, or route semantics.
- There is no fixed global vendor priority. A small workload may select Vector API or scalar even
  when a compatible native provider is available.
- The portable Class-File/Vector route remains the supported semantic baseline and fallback even
  after native routes land. OpenBLAS is only a narrow BLAS-compatible fallback. Accelerate,
  oneMKL/oneDNN, and AOCL/ZenDNN are exact-capability peer families compared by whole-plan cost,
  not a priority ladder and not separate `BackendId` values.
- Intel oneMKL and oneDNN remain separate low-level provider/integration boundaries because they
  serve standalone math and DNN-partition concerns respectively. AOCL-BLAS/AOCL-LibM and ZenDNN
  likewise remain distinct. CPU, not any provider leaf, owns capability reporting, coordination,
  fallback, route choice, and tuning.
- On Apple CPU, Accelerate BLAS, vDSP, and vForce are CPU routes. MPSGraph and custom Metal
  kernels remain exclusively Metal-backend routes after Planning selects Metal ownership; CPU
  never treats Metal as an internal optimization route.
- ARM has no hard-coded provider default. Apple Silicon may admit the Accelerate candidates above;
  other ARM systems retain portable code generation unless an explicitly verified provider task
  adds an eligible peer.
- On AMD CPU, portable scalar and Vector API routes remain available alongside AOCL-BLAS rather
  than becoming a lower-priority emergency path. AOCL-LibM enters only eligible sufficiently
  large vector-math candidate sets; ZenDNN remains a separate later DNN-partition option.
- Intel oneMKL/oneDNN and AMD AOCL/ZenDNN native candidates prioritize BFLOAT16 where the exact
  ABI, ISA/hardware, operation, and measured workload benefit are established. FLOAT16 enters only
  after Model task 0026 and the same exact evidence; it is not inferred from a 16-bit carrier or
  vendor availability.
- Vendor fast- or relaxed-math entry points require a future explicit numerical policy. They are
  not candidates under current exact/default semantics.
- Benchmarks, safe heuristics, and tuning may compare only candidates already eligible under the
  ordinary operation contract and explicit caller permission. Measurement never grants relaxed
  permission.

## Risks

- Leaking route selection into planning or splitting CPU routes into false backends.
- Exposing private CPU knobs through string dispatch, reflection annotations, a central registry,
  or a generic configuration language.
- Treating installed native libraries as a priority list rather than generating and filtering the
  valid candidates for the exact occurrence or partition.
- Moving capability truth, fallback, lifetime coordination, or tuning into a low-level vendor
  provider.
- Treating route choice and storage choice as coupled global modes, or choosing kernels locally
  without accounting for required representation transitions across consumers.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
