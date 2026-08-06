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
- `CpuPartitionAnalysisInputs.DEFAULT` remains the compatibility input for the current fixed
  proving topology: lowering-manifest retention is disabled and its four ordered boundary forms are
  all `MEMORY_SEGMENT`. Explicit CPU analysis inputs may immutably select any four-entry pattern in
  0005B. The default's length follows today's three inputs plus one output and is not a permanent
  fused-unit boundary-count contract.
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
    executable/               prepared partition execution
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
| 0005C | Vector and parallel portable strategies | Draft | 0005B | Deliver vector, parallel-scalar, and parallel-vector over universal start/end kernels, external chunk dispatch, correct vector chunks/tails, and no hot dispatch or cursor allocation. |
| 0005D | Materialization, specialization, and persistence evidence gate | Draft | 0005C | Compare direct access with CPU-internal contiguous materialization before assignment, enforce specialization budgets, and benchmark whether optional trusted-root class-byte persistence should remain disabled by default. |
| 0005E | Portable pointwise types, carriers, and semantic-family expansion | Draft | 0005D | Expand exact pointwise, comparison, selection, and cast coverage through the completed unit/IR/access architecture without duplicating planners or storage matrices. |
| 0005F | Exact scalar-power strength reduction | Draft | 0005E | Classify exact typed scalar exponents once in common route-independent CPU analysis and retain semantic `POW` in the compiled graph while selecting multiply, reciprocal, or exponentiation-by-squaring realizations only under an exact/default conformance proof; this task does not depend on relaxed Config 0006. |
| 0006 | Portable layout, indexing, ordering, and random family coverage | Draft | 0005E | Generate truthful coverage for executable layout transforms, slicing, padding, tiling, composition, windows, gather/scatter, ordering, one-hot, and explicit-state random/dropout work; metadata-only or zero-work views remain computation-free. |
| 0007 | Portable reduction, scan, statistics, and normalization family coverage | Draft | 0002–0006 | Generate family-specific range, tile, partial-reduction, and combine bodies for aggregates, arg extrema, scans, softmax/log-softmax, statistics, and normalization with exact semantics and determinism. |
| 0008 | Portable linear algebra, convolution, pooling, attention, and loss coverage | Draft | 0002–0007 | Generate the remaining portable executable families, including exact fused-partition forms only where the same lowering contracts preserve semantics across storage and execution modes. |
| 0009 | Portable generated-coverage closure checkpoint | Draft | 0001–0008; complete current selected Model semantic inventory | Prove the bytecode/Vector portable route is the truthful supported semantic baseline and fallback, classify metadata-only work, prove unsupported work fails closed, and close capability/conformance before native peer-route expansion. |
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
`internal.route.portable`. Detailed CPU 0005B is Complete; CPU 0005C is the next Draft frontier,
and every later CPU task remains Draft without a detailed specification.

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
state machines, and all sixteen ordered heap/segment carrier specializations. CPU 0005C–0005E and
tasks 0006–0017 remain `Draft` without detailed specifications.

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
- Exact/default scalar-power strength reduction is independent of that future permission. Draft
  0005F may select only realizations proved conforming to semantic `POW`; Draft 0017 later consumes
  Config 0006 permission for genuinely relaxed candidates and therefore does not block exact work.
- Historical task 0001 implements the exact native `MemorySegment` representation, shared-arena
  ownership, zero-size/alignment/allocation/cleanup rules, borrowed heap/native cold binding,
  direct typed invocation seam, and worker evidence. Task 0005A retains the representation and
  binding contracts but removes unused worker types; task 0005C introduces the workers required by
  implemented parallel strategies. Route-specific representations and materializations still wait
  for their ordered tasks.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Task 0005A replaces the provisional per-node implementation atomically. The permanent model is
  partition -> computation-oriented execution units -> canonical loop IR -> route realization ->
  one partition executable. There is no compatibility layer between the two designs.
- CPU lowering operates over the complete owned partition and supports pointwise/broadcast fusion,
  metadata/view folding, reduction/scan and stable multi-pass units, matrix/convolution epilogues,
  and explicit semantic kernels as their Draft tasks land. Fusion legality and profitability are
  separate; neither may cross alias/state/random/publication/fan-out/partition or numerical-order
  boundaries. Decomposed softmax is not silently recognized as stable `SOFTMAX`.
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
  or per-element division/modulo. Draft 0005C owns vector gather and Draft 0005D owns optional
  materialization; full semantics do not promise universal vectorization.
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
  production types while retaining their evidence; task 0005C reintroduces only the orchestration
  required by implemented parallel-scalar and parallel-vector strategies.
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
  policy and disabled by default until task 0005D supplies benchmark evidence. A hit can avoid
  emission but cannot preserve JIT machine code or profile.
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
- Common CPU analysis also owns exponent classification, numerical eligibility, and any selected
  `POW` realization plan. Emitters and vendor adapters consume that decision and never invent
  independent fast-math behavior. Scalar `POW(2)`, `POW(-1)`, and other exact typed small integral
  exponents may select multiply, reciprocal, or exponentiation by squaring after proof; `POW(0.5)`
  is not silently replaced by `SQRT`. Tensor exponents require compiler-owned immutable uniform-
  constant facts and are never inferred from Tensor storage or factory history.
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
