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
- standard JDK Class-File API generation for portable scalar and Vector API computation kernels,
  subject to the architecture-synchronization prerequisite recorded below
- optional OpenBLAS routes for supported BLAS-compatible linear algebra
- distinct Intel oneMKL BLAS/VML and oneDNN integrations
- Apple Accelerate BLAS, vDSP, and vForce integrations
- distinct AMD AOCL-BLAS/AOCL-LibM and later optional ZenDNN integrations
- other specialized and fused routes only when a concrete capability justifies them
- CPU executables, storage, workspace, scheduling, and tracing
- a bounded concurrent CPU-owned in-memory cache of generated executable artifacts
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
- Subject to the required architecture synchronization, Java 26
  `java.lang.classfile.CodeBuilder` is the selected primary mechanism for every portable CPU
  computation kernel. This is a future planning decision, not a claim that the authoritative
  contract's current `CPU ASM` wording has already changed.
- Every currently selected executable Model operation semantic must gain truthful portable
  generated coverage before the portable capability milestone closes. Metadata-only or zero-work
  view occurrences need no generated computation. Unsupported executable semantics fail closed
  and are not advertised until their generated coverage exists.
- OpenBLAS is an optional portable native fallback for supported BLAS-compatible linear algebra;
  it is never a universal CPU fallback.
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
- Memory-segment storage and execution route are orthogonal. Scalar Java, JDK Vector API, and
  Foreign Function and Memory (FFM) native-provider calls may use the same native-backed
  `MemorySegment` without copying merely because the route changes.
- Run-owned internal CPU buffers use aligned native off-heap `MemorySegment` storage as the
  canonical interoperable CPU representation. Exact alignment, arena/lifetime ownership, access,
  cleanup, zero-size, and allocation declarations remain work for the next detailed CPU task;
  this Draft plan does not claim that representation is implemented.
- Borrowed caller inputs are handled per value and use. Compatible heap-backed inputs may remain
  heap-backed. CPU preparation introduces at most one necessary native materialization for an
  exact selected downstream FFM route and reuses it across compatible consumers.
- Storage kind is a cold-binding specialization fact. Portable generated kernels support heap
  primitive carriers or heap-backed segments, native off-heap `MemorySegment` representations,
  and mixed per-input/per-output signatures without discovering storage or selecting a route in
  generated hot code.
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

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | CPU capability, representation, binding, and parallel foundation | Draft | Stable planning, runtime, prepare, backend-contract, and trace contracts | Establish truthful fail-closed capability; canonical aligned native internal buffers; heap, native, and mixed representation binding; direct typed invocation; and shared worker/chunk/cancellation/failure infrastructure without executable semantic coverage claims. |
| 0002 | Portable Class-File API generator foundation | Draft | 0001; separate authorized architecture synchronization; Java 26 Class-File and Vector API toolchain | Establish generator schema/versioning, exact typed specialization descriptors, family-specific lowerer contracts, shared scalar/vector/heap/segment/range/tile/reduction emitters, hidden-class definition, and the exact four-mode execution matrix without a god generator. |
| 0003 | Bounded generated-artifact cache and cold finalization | Draft | 0002; stable CPU finalization and artifact compatibility | Add the CPU-owned bounded concurrent single-flight in-memory cache, deterministic complete key equality, hidden-class and typed-entry lifetime retention, and miss-only generation/definition during backend finalization after slot assignment. |
| 0004 | Typed portable analysis, specialization, and finalization | Draft | 0001–0003 | Generate complete typed portable candidates per operation occurrence or fused partition; select one valid route, representation, specialization, execution mode, and parallel configuration after exact compatibility and full transition-cost filtering; and cold-bind the finalized typed entry point. |
| 0005 | Portable elementwise and pointwise family coverage | Draft | 0002–0004 | Generate truthful scalar/Vector and single-thread/parallel coverage for arithmetic, comparison, logical, selection, cast, classification, and activation semantics across supported data, storage, layout, shape, broadcast, and tail specializations. |
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
| 0016 | Compatible CPU workload-tuning-cache selection | Draft | 0004; 0010–0015 as implemented; deferred Prepare opaque-candidate handoff; stable tuning-artifact compatibility | Reuse only compatible persistent selected-route evidence while retaining exact filtering and safe heuristic fallback; keep it distinct from the in-memory generated-artifact cache and add no measurement or cache mutation to CPU prepare. |


## Milestones

- CPU representation, binding, generator, and artifact-cache foundation
- Portable generated family coverage and closure
- Optional native routes and complete candidate selection
- Tuning-cache integration and conformance

## Current status

Draft at the selected target-strategy level. No task is Ready and no detailed task specification
exists. A separate planning step must refine only task 0001 when implementation begins; later rows
remain Draft.

Before task 0002 or any generated-kernel implementation task can become Ready, a separate
authorized architecture synchronization must replace the current ASM-specific wording in
`ARCHITECTURE.md` and `docs/architecture/module-boundaries.md` with an implementation-neutral
generated JVM bytecode route and assess whether an architecture decision record (ADR) or
architecture-test update is required. Until that coordinated change is authorized and complete,
the Class-File API direction remains selected non-authoritative planning only.

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
- Task 0001 must declare the exact native `MemorySegment` representation, allocation alignment,
  lifetime owner, cleanup, and binding contracts before it can become Ready. This master plan
  records the strategy only and does not select those details early.
- The first generated-kernel task cannot become Ready under the current authoritative `CPU ASM`
  and architecture-update-gated bytecode-generation wording. The prerequisite above must resolve
  that conflict; this planning task does not edit or reinterpret the contract.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Subject to the recorded architecture prerequisite, Java 26
  `java.lang.classfile.CodeBuilder` is the primary generation mechanism for all portable CPU
  computation kernels. Native vendor providers remain separate optional routes; the generator
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
- Cold binding resolves heap primitive carriers or heap-backed segments, native off-heap
  `MemorySegment` values, and mixed signatures into direct typed entry-point arguments. Generated
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
  consults the generated-artifact cache and emits/defines the selected class on a miss. Runtime
  invokes only the cold-bound typed entry point.
- The CPU backend owns one bounded concurrent in-memory generated-artifact cache with single-flight
  generation. A hit reuses the loaded artifact and its accumulated JVM/JIT profile. A cache value
  retains the hidden-class lifetime and typed entry point while cached or while a prepared
  executable reachable from `PreparedExecution` references it; eviction plus loss of every
  reference permits unloading. No unbounded static map, service locator, Runtime lookup, or Tensor
  ownership is permitted.
- The generated-artifact key includes a generator schema/version and exactly every fact that
  changes bytecode or compatibility. Thread count or chunk size is keyed only when emitted code
  changes; otherwise it remains prepared parallel-execution configuration. Key construction and
  equality are deterministic and must be validated directly.
- The generated-artifact cache is distinct from the persistent tuning cache: tuning selects a
  compatible candidate, while the artifact cache reuses generated executable code. The first
  artifact cache is memory-only. Persistent class-byte storage is deferred until strict Synaptik
  build, generator-schema, classfile/JDK, Vector API, and target compatibility, validation, and
  corruption-handling contracts are designed.
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
