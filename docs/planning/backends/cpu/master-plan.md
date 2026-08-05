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
- scalar correctness routes and JDK Vector API elementwise/reduction routes
- standard JDK Class-File API generation for portable scalar and Vector API computation kernels
- optional OpenBLAS routes for supported BLAS-compatible linear algebra
- distinct Intel oneMKL BLAS/VML and oneDNN integrations
- Apple Accelerate BLAS, vDSP, and vForce integrations
- distinct AMD AOCL-BLAS/AOCL-LibM and later optional ZenDNN integrations
- other specialized and fused routes only when a concrete capability justifies them
- CPU executables, storage, workspace, scheduling, and tracing
- a CPU-owned durable filesystem store of compatible generated class bytes plus process-local
  single-flight and weak loaded-artifact interning
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
- Portable scalar code is the correctness baseline. JDK Vector API kernels are the portable
  optimized baseline for eligible elementwise and reduction work.
- Java 26 `java.lang.classfile.CodeBuilder` is the selected current implementation direction for
  every portable CPU computation kernel. This is non-authoritative planning: the architecture
  permits generated JVM-bytecode CPU computation kernels without making that builder or another
  generation library an invariant.
- Every currently selected executable Model operation semantic must gain truthful portable
  generated coverage before the portable capability milestone closes. Metadata-only or zero-work
  view occurrences need no generated computation. Unsupported executable semantics fail closed
  and are not advertised until their generated coverage exists.
- OpenBLAS is primarily an optional FLOAT32/FLOAT64 native fallback for supported BLAS-compatible
  linear algebra; it is never a universal CPU fallback. BFLOAT16 requires a separately verified
  version, instruction-set architecture (ISA), and operation route. FLOAT16 support is neither
  broad nor baseline by assumption.
- CPU never offloads internally to MPSGraph or a custom Metal kernel. Planning must first select
  Metal ownership, after which the separate Metal backend owns those routes.
- CPU owns capability truth, provider coordination, route selection, fallback, thread/lifetime
  coordination, and tuning. Native provider layers remain low-level ABI/lifetime leaves.
- CPU candidate generators return complete valid route-specific configurations; shared tuning
  sees them opaquely.
- Safe CPU heuristics remain correct when tuning is disabled or a compatible cache entry is
  absent.
- Candidate eligibility is filtered for exact operation semantics and required determinism before
  any performance comparison. Current exact/default semantics do not permit vendor fast- or
  relaxed-math routines.
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
- Storage kind is a cold-binding specialization fact. Portable generated kernels support heap
  primitive carriers or heap-backed segments, native off-heap `MemorySegment` representations,
  and mixed per-input/per-output signatures without discovering storage or selecting a route in
  generated hot code.
- Portable `MemorySegment` storage is representation-only and accepts the logical data type. A
  two-byte representation does not imply executable arithmetic or Vector support. The current
  generated specialization admits Java Vector lanes only for FLOAT64, FLOAT32, INT32, and INT64;
  BFLOAT16 and future FLOAT16 need separately established routes.
- Portable execution has exactly four modes: scalar single-thread, scalar parallel, Vector API
  single-thread, and Vector API parallel. Shared CPU parallel infrastructure owns workers, chunk
  dispatch, cancellation, synchronization, and failure propagation; generated classes do not own
  thread pools or schedulers.

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
  CpuCapabilityProvider       public fail-closed CPU capability identity
  package-info.java           public package boundary and current status
  execution/                  package-private representations, cold binding, workers,
                              generated-kernel specialization/emission/artifact machinery,
                              durable artifact storage, weak loaded reuse, typed portable
                              analysis/finalization, and CPU executable recipes
```

Tasks 0002–0004 extend only the existing package-private `execution` package so generated-kernel,
artifact-store, typed analysis/finalization, and cold-binding code can reuse task 0001's arguments
and workers without widening them or creating a cross-package facade. Later family-specific
placement must be planned by its owning detailed task before that family becomes Ready.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [CPU capability, representation, binding, and parallel foundation](tasks/0001-cpu-capability-representation-binding-and-parallel-foundation.md) | Complete | Stable planning, runtime, prepare, backend-contract, and trace contracts | Established truthful fail-closed capability; canonical aligned native internal buffers; typed-array, exact-segment, and mixed binding; direct typed invocation; and shared worker/chunk/cancellation/failure infrastructure without executable semantic coverage claims. |
| 0002 | [Portable Class-File API generator foundation](tasks/0002-portable-class-file-api-generator-foundation.md) | Complete | 0001; generated JVM-bytecode CPU-kernel architecture contract; Java 26 Class-File and Vector API toolchain | Established generator schema/versioning, exact typed specialization descriptors, mode-owned structural scalar/vector emission dispatch, family-specific emitter contracts, shared scalar/vector/heap/segment/range/tile/reduction emitters, hidden-class definition, and the exact four-mode execution matrix without a god generator or operation coverage. |
| 0003 | [Durable generated-kernel artifact store and cold loading](tasks/0003-bounded-generated-artifact-cache-and-cold-finalization.md) | Complete | 0002; stable CPU finalization and artifact compatibility; explicit trusted local root | Added a model-independent filesystem store for exact compatible generated class bytes, atomic cross-process-safe publication, full validation, process-local single-flight, weak loaded-artifact interning, and cold definition/handle resolution for later post-slot finalization. |
| 0004 | [Typed portable analysis, specialization, and finalization](tasks/0004-typed-portable-analysis-specialization-and-finalization.md) | Complete | 0001–0003 | Connected staged Prepare to CPU-private typed portable selection, exact pre-assignment declarations, post-assignment artifact finalization, immutable generated-executable retention, and direct per-run cold binding using bounded synthetic/fail-closed coverage only. |
| 0005 | [Dense ADD and partition-sequence execution](tasks/0005-dense-add-and-partition-sequence-execution.md) | Complete | 0002–0004 | Delivered the first truthful static canonical dense FLOAT64/FLOAT32/INT32/INT64 ADD route and generalized CPU-private recipes so any maximal partition made only of advertised occurrences executes as an ordered scalar single-thread native-segment kernel sequence. |
| 0005A | Remaining portable pointwise arithmetic, carriers, and modes | Draft | 0005 | Extend the proven sequence contract with the remaining exact arithmetic matrix plus heap/mixed carriers and Vector/parallel candidates; keep later pointwise semantic families separately bounded. |
| 0006 | Portable layout, indexing, ordering, and random family coverage | Draft | 0002–0005 | Generate truthful coverage for executable layout transforms, slicing, padding, tiling, composition, windows, gather/scatter, ordering, one-hot, and explicit-state random/dropout work; metadata-only or zero-work views remain computation-free. |
| 0007 | Portable reduction, scan, statistics, and normalization family coverage | Draft | 0002–0006 | Generate family-specific range, tile, partial-reduction, and combine bodies for aggregates, arg extrema, scans, softmax/log-softmax, statistics, and normalization with exact semantics and determinism. |
| 0008 | Portable linear algebra, convolution, pooling, attention, and loss coverage | Draft | 0002–0007 | Generate the remaining portable executable families, including exact fused-partition forms only where the same lowering contracts preserve semantics across storage and execution modes. |
| 0009 | Portable generated-coverage closure checkpoint | Draft | 0001–0008; complete current selected Model semantic inventory | Prove every advertised executable Model semantic has truthful portable generated coverage or a documented metadata-only/zero-work classification, prove unsupported work fails closed, and run the CPU capability/conformance checkpoint before the portable milestone closes. |
| 0010 | Optional portable OpenBLAS linear-algebra route | Draft | 0001; 0004; 0009; completed OpenBLAS provider | Add OpenBLAS only for eligible BLAS-compatible linear algebra, preserving generated portable alternatives, exact filtering and full transition costing, and never generating an OpenBLAS replacement. |
| 0011 | Intel oneMKL BLAS and VML provider routes | Draft | 0001–0004; 0009; concrete Intel CPU use case and supported oneMKL ABI evidence | Add distinct low-level oneMKL ABI/lifetime leaves and CPU-owned BLAS/VML routes without generating vendor replacements. |
| 0012 | Intel oneDNN DNN-partition routes | Draft | 0001–0004; 0009; stable CPU partition lowering; concrete DNN/ML use case and supported oneDNN ABI evidence | Add a distinct oneDNN provider boundary and eligible DNN/ML partition/fusion routes without collapsing it into oneMKL or generated portable code. |
| 0013 | Apple Accelerate routes | Draft | 0001–0004; 0009; concrete Apple CPU use case and supported Accelerate ABI evidence | Add low-level Accelerate BLAS, vDSP, and vForce leaves plus eligible CPU routes; generate no replacements and do not call MPSGraph or Metal kernels from CPU. |
| 0014 | AMD AOCL-BLAS and AOCL-LibM routes | Draft | 0001–0004; 0009; concrete AMD CPU use case and supported AOCL ABI evidence | Add distinct AOCL-BLAS and AOCL-LibM leaves and eligible CPU routes without generating vendor replacements. |
| 0015 | Optional AMD ZenDNN partition routes | Draft | 0014; 0004; 0009; stable CPU partition lowering; concrete ZenDNN use case and integration evidence | Add a distinct later ZenDNN provider and eligible DNN-partition route without collapsing it into AOCL or generated portable code. |
| 0016 | Compatible CPU workload-tuning-cache selection | Draft | 0004; 0010–0015 as implemented; deferred Prepare opaque-candidate handoff; stable tuning-artifact compatibility | Reuse only compatible persistent selected-route evidence while retaining exact filtering and safe heuristic fallback; keep selected-route evidence distinct from the generated-class artifact store and add no measurement or tuning-cache mutation to CPU prepare. |


## Milestones

- CPU representation, binding, generator, and durable artifact-store foundation
- Portable generated family coverage and closure
- Optional native routes and complete candidate selection
- Tuning-cache integration and conformance

## Current status

Task 0001 is Complete through its sole detailed CPU
[capability, representation, binding, and parallel foundation](tasks/0001-cpu-capability-representation-binding-and-parallel-foundation.md)
specification. It introduces one truthful provider with stable `cpu` identity and no advertised
operation semantics, package-private aligned native representation and typed cold-binding
infrastructure, and a package-private bounded worker/range foundation. It does not add a CPU
preparer, executable semantic coverage, a route, generated code, OpenBLAS integration, or another
module change. Complete
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
Complete
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
operation-family source uses the store yet. Complete CPU 0004 now owns the CPU-private
integration, explicit root seam, and prepared-executable strong retention. Detailed
[task 0004 Typed portable analysis, specialization, and finalization](tasks/0004-typed-portable-analysis-specialization-and-finalization.md)
is Complete. It adds only the CPU-private staged lifecycle foundation: deterministic selection
from a direct typed candidate source, exact declarations before assignment, artifact-store use
only during finalization afterward, immutable generated-kernel/direct-handle retention, and
family-owned direct typed cold binding. CPU analysis preserves backend-declared byte geometry;
shared Prepare validates that geometry against assigned slots, and no dense-layout or
materialization policy was introduced. Its tests remain bounded and synthetic,
`CpuCapabilityProvider` remains fail-closed, and no real Model operation family is implemented or
advertised. CPU task 0005 is Complete through its sole detailed
[Dense ADD and partition-sequence execution](tasks/0005-dense-add-and-partition-sequence-execution.md)
specification. It solves maximal-partition truth with an ordered node-kernel sequence while
limiting first executable coverage to static canonical dense FLOAT64/FLOAT32/INT32/INT64 ADD
through scalar single-thread native segments. Implementation context
`019fd19a-4262-7580-9674-226595356fbc` recorded the sole final CPU 31-test pass; the clean
documentation pass reused that evidence because it changed no executable Java behavior. CPU
0005A is the next Draft frontier, and tasks 0006–0016 remain Draft without detailed
specifications.

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
- Task 0001 implements the exact native `MemorySegment` representation, shared-arena ownership,
  zero-size/alignment/allocation/cleanup rules, borrowed heap/native cold binding, direct typed
  invocation seam, and worker lifecycle needed for implementation. Later route-specific
  representation requirements and materializations still wait for the route-selection tasks.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- CPU task 0001 exposes only `CpuCapabilityProvider`. Its stable `BackendId("cpu")` is identity,
  not availability or readiness, and `supports` remains unconditionally false until later tasks
  deliver and test exact executable semantic coverage.
- Task 0001 keeps physical representations, typed cold arguments, executable specialization, and
  worker/range coordination package-private in one CPU execution package. Run-owned internal
  buffers/workspaces use one shared `Arena` and exact aligned native allocation per
  representation, including zero-byte geometry; borrowed Model storage remains non-owning.
- Task 0001 binds each selected representation independently. The six Model data types map to
  their exact observable primitive-array carriers with retained carrier-relative byte offsets.
  When the matching carrier is unavailable, `CpuBufferArgument.Segment` retains the exact segment
  or slice without asserting native provenance; this covers genuine native segments and JDK 26
  read-only heap segments whose `heapBase()` is empty. Route-specific bound invocations must copy
  direct typed fields out of cold arrays and perform no storage discovery in the hot call.
- Task 0001's fixed CPU worker group owns bounded platform workers, deterministic contiguous range
  geometry, synchronous completion, cooperative range-boundary cancellation, first/suppressed
  failure propagation, interruption restoration, and idempotent shutdown. It selects no thread
  count from Config, tuning, or operation semantics.
- Java 26 `java.lang.classfile.CodeBuilder` is the selected primary generation mechanism for all
  portable CPU computation kernels. This remains a current planning choice rather than an
  architecture invariant. Native vendor providers remain separate optional routes; the generator
  does not replace OpenBLAS, oneMKL, oneDNN, Accelerate, AOCL, or ZenDNN.
- Portable capability is truthful and fail-closed. Every selected executable Model semantic must
  have generated coverage before the portable milestone closes; metadata-only or zero-work view
  occurrences require no generated computation and unsupported executable semantics are not
  advertised.
- A generated class is narrowly specialized to one exact operation or fused-partition fingerprint
  and every bytecode-relevant data type, per-input/per-output storage kind, layout, shape
  specialization, execution mode, Vector API species, unroll/tile/tail choice,
  numerical/determinism policy, and target fact. Runtime buffer identities and addresses,
  `TensorId`, `NodeId`, `ValueId`, model ID, and run ID are excluded. Exact dimensions are baked
  only when the selected specialization benefit justifies them; otherwise dimensions are typed
  invocation parameters.
- Cold binding resolves observable heap primitive carriers, exact `MemorySegment` values whose
  matching carrier is unavailable, and mixed signatures into direct typed entry-point arguments. Generated
  hot code performs no heap-base discovery, generic type check, route choice, cache lookup, or
  storage-kind, data-type, layout, vector, parallel, broadcast, or operation switch.
- The portable execution matrix is exactly scalar single-thread, scalar parallel, Vector API
  single-thread, and Vector API parallel. Family-specific lowering plus shared low-level emitters
  produce range, tile, partial-reduction, and combine bodies. Shared CPU parallel infrastructure,
  not generated classes, owns workers, chunk dispatch, cancellation, synchronization, and failure
  propagation.
- Operation semantics are lowered once through family-specific typed lowerers. Shared scalar,
  vector, heap, segment, range, tile, and reduction emission primitives adapt that lowering to
  storage and execution modes; the design neither duplicates semantics across heap/native and
  scalar/vector modes nor concentrates all families in one generator.
- CPU backend analysis selects one complete valid route, representation, specialization candidate,
  and prepared parallel configuration. Only CPU backend finalization after shared slot assignment
  consults the generated-artifact store and emits verified bytes only on a compatible stored miss.
  It validates stored bytes, defines the hidden class, and resolves the exact typed entry point;
  Runtime invokes only that cold-resolved handle.
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
- Task 0004 receives an explicit trusted artifact root and an already-owned worker group through
  a CPU-private finalizer constructor. It adds no public composition owner or default Config API,
  and finalization borrows rather than allocates or closes the worker group.
- The durable source of reusable generated kernels is one CPU-owned model-independent filesystem
  artifact store under an explicit caller/composition-supplied trusted local root. Separate store
  instances and later JVM processes sharing that root reuse exact compatible class bytes. Entries
  use deterministic content-addressed paths, exact compatibility metadata, bounded envelopes,
  checksums, full class-file verification, and atomic complete-file publication.
- The generated-artifact key includes a generator schema/version and exactly every fact that
  changes bytecode or compatibility. Thread count or chunk size is keyed only when emitted code
  changes; otherwise it remains prepared parallel-execution configuration. Key construction and
  equality are deterministic and must be validated directly.
- Task 0003 stores and compares the complete immutable `CpuKernelSpecialization` canonical
  structure plus generator, Java/Class-File, generated-class, entry-name, and entry-descriptor
  compatibility facts. A path digest, specialization fingerprint, Java hash, or checksum alone
  cannot establish a hit. Model, Tensor, graph, value, slot, storage, address, run, emitter,
  handle, class-loader, and store-instance identities are excluded.
- Compatible age never invalidates an artifact. The store has no time-to-live, hit-rate policy,
  automatic disk eviction, quota, background service, or valid-entry maintenance. Corrupt or
  incompatible bytes are never defined and may be replaced only through verified generation and
  forced temporary-file plus atomic-move publication.
- Equal requests share one process-local single-flight attempt across store instances. Loaded
  artifacts are interned only through weak values with reference-queue stale-key cleanup; there is
  no strong global completed LRU. A later prepared executable reachable from
  `PreparedExecution`, not the store, strongly retains the hidden class, lookup, and exact method
  handle for active use.
- Checksums and structural verification detect accidental corruption but do not authenticate
  executable bytes. The artifact root is an explicit trusted-local-cache security boundary whose
  write isolation and administration belong to the caller/composition owner.
- The generated-class artifact store is distinct from the persistent tuning cache: the former
  reuses exact verified executable class bytes after route selection, while the latter records
  compatible route/configuration-selection evidence. Neither performs Runtime hot-path work.
- Matrix-multiplication candidates may include supported JDK Vector API species and strategy,
  unroll, tile, parallelism, and OpenBLAS thread configurations derived and pruned from target
  capabilities, workload facts, and budget.
- Scalar, vector, and OpenBLAS are typed route configurations, not booleans in
  `Map<String,Object>`. Operation family selects a generator but does not key one universal
  configuration.
- Concrete CPU analysis generates a complete typed candidate set for each operation occurrence or
  partition. It filters by platform and provider availability, operation and attributes, data
  type, `Shape`, layout, exact numerical and determinism compatibility, and resource validity
  before comparing call overhead, safe workload heuristics, or a compatible tuning-cache entry.
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
- Intel oneMKL and oneDNN remain separate low-level provider/integration boundaries because they
  serve standalone math and DNN-partition concerns respectively. AOCL-BLAS/AOCL-LibM and ZenDNN
  likewise remain distinct. CPU, not any provider leaf, owns capability reporting, coordination,
  fallback, route choice, and tuning.
- On Apple CPU, Accelerate BLAS, vDSP, and vForce are CPU routes. MPSGraph and custom Metal
  kernels remain exclusively Metal-backend routes after Planning selects Metal ownership; CPU
  never treats Metal as an internal optimization route.
- On AMD CPU, portable scalar and Vector API routes remain available alongside AOCL-BLAS rather
  than becoming a lower-priority emergency path. AOCL-LibM enters only eligible sufficiently
  large vector-math candidate sets; ZenDNN remains a separate later DNN-partition option.
- Intel oneMKL/oneDNN and AMD AOCL/ZenDNN native candidates prioritize BFLOAT16 where the exact
  ABI, ISA/hardware, operation, and measured workload benefit are established. FLOAT16 enters only
  after Model task 0026 and the same exact evidence; it is not inferred from a 16-bit carrier or
  vendor availability.
- Vendor fast- or relaxed-math entry points require a future explicit numerical policy. They are
  not candidates under current exact/default semantics.

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
