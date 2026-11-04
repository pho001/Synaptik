# Task 0005B: Universal Access Plans and Right-Aligned [Broadcasting](../../../../glossary.md#broadcasting)

## Status

Complete

## Goal

Extend the whole-[partition](../../../../glossary.md#partition) [CPU](../../../../glossary.md#term-cpu) architecture established by task 0005A with one route-independent,
normalized per-value access-plan family. The same structural plans and cold instance [bindings](../../../../glossary.md#term-binding) must
drive common [lowering](../../../../glossary.md#lowering), the [scalar](../../../../glossary.md#term-scalar) reference realization, and portable generated scalar code for
the existing [FLOAT64](../../../../glossary.md#term-float64) `ADD -> exact GELU -> MUL` proving [topology](../../../../glossary.md#term-topology).

The task completes current [Model](../../../../glossary.md#term-model) right-aligned [broadcasting](../../../../glossary.md#broadcasting) and resolved-[layout](../../../../glossary.md#layout) access for fully
static [shapes](../../../../glossary.md#shape): scalar and [rank](../../../../glossary.md#term-rank) expansion, [singleton](../../../../glossary.md#term-singleton) and multi-[axis](../../../../glossary.md#term-axis) expansion, zero extents,
[canonical](../../../../glossary.md#term-canonical-form) dense and offset-dense layouts, arbitrary non-negative strided layouts, broadcast zero
[strides](../../../../glossary.md#term-stride), and [heap](../../../../glossary.md#term-heap)-array, exact-[`MemorySegment`](../../../../glossary.md#term-memorysegment), or mixed carriers. The ordered heap/segment access
pattern is an explicit code-shaping [specialization](../../../../glossary.md#term-specialization), while compatible instance extents, offsets,
strides, exact [carrier](../../../../glossary.md#term-storage-carrier) objects, [Runtime](../../../../glossary.md#term-runtime) slots, physical addresses, and [run](../../../../glossary.md#term-run) [identity](../../../../glossary.md#term-identity-meanings) remain outside
structural generated-class identity.

The resulting [mental model](../../../../glossary.md#term-mental-model) is:

```text
ShapeBroadcast + LayoutDescriptor
  -> CPU-normalized structural access plans
  -> structural ordered carrier pattern + cold geometry/carrier objects and exact accessed spans
  -> one generated or scalar-reference whole-partition execution
```

## [Scope](../../../../glossary.md#term-scope)

- Preserve the exact three-[node](../../../../glossary.md#node), one-unit [FLOAT64](../../../../glossary.md#term-float64) `ADD -> GELU -> MUL` topology from 0005A,
  including exact/default numerical mode, fixed operation order, one executable, two virtual
  intermediates, four boundary declarations, producer/consumer provenance, and primitive<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
  `start`/`end` bounds.
- Derive every [operation](../../../../glossary.md#operation) result [shape](../../../../glossary.md#shape) through the current `ShapeBroadcast` contract. Validate the
  unary GELU shape exactly and require the declared final output shape to equal the final MUL
  broadcast result.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu))</span>
- Normalize each of the three materialized inputs against the final unit iteration shape, composing
  the ADD result's later MUL expansion without materializing either virtual value.
- Normalize the materialized output against the same iteration shape. Permit canonical dense,
  offset-dense, and positive-strided writes, but reject any output geometry that maps two distinct
  logical output coordinates to one storage element. A zero stride on a size-zero or singleton
  axis is legal only when it cannot repeat a write.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis))</span>
- Expand `CpuAccessPlan` into the sole CPU access family. Its structural form records access kind,
  ordered normalized axis roles, iteration rank, contiguous-suffix structure, and one of these
  ordered scalar regimes:<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [normalized axis glossary](../../../../glossary.md#normalized-axis); [rank glossary](../../../../glossary.md#term-rank); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>

  1. `DENSE_LINEAR` — one linear address increment for canonical dense access, with a cold base
     offset;<span data-glossary-navigation="true"> ([linear glossary](../../../../glossary.md#term-linear-meanings))</span>
  2. `SCALAR_ALL_ZERO` — [scalar](../../../../glossary.md#term-scalar) or all-zero-[stride](../../../../glossary.md#term-stride) input access;
  3. `LAST_AXIS_BIAS` — one contiguous final indexed axis with every leading iteration axis
     broadcast, such as a right-aligned rank-one bias;<span data-glossary-navigation="true"> ([rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [bias glossary](../../../../glossary.md#term-bias))</span>
  4. `BLOCK_OUTER` — a contiguous trailing inner block with broadcast or positive-strided outer
     axes and carry/reset arithmetic outside that inner block;<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry))</span>
  5. `GENERAL_ODOMETER` — the complete scalar fallback for every other admitted combination of
     positive and broadcast-zero effective strides.
- Keep structural regime, iteration rank, axis-role pattern, contiguous-suffix structure, value
  kind, data type, ordered semantics, stores, and the ordered boundary carrier access pattern in
  specialization/class identity. Keep concrete extents, element count, source layout offset,
  effective stride values, starting coordinates and addresses, exact carrier objects, assigned
  slots, physical addresses, and run identity in immutable cold bindings only. Two instances with
  equal structural topology, axis roles, and carrier pattern but compatible different extents
  normally reuse deterministic class bytes and loaded identity. A different carrier pattern is an
  intentionally distinct specialization.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [specialization glossary](../../../../glossary.md#term-specialization); [identity glossary](../../../../glossary.md#term-identity-meanings); [layout glossary](../../../../glossary.md#layout); [run glossary](../../../../glossary.md#term-run); [bindings glossary](../../../../glossary.md#term-binding))</span>
- Make each cold access binding carry checked output extents, base element offset, effective
  right-aligned element strides, the primitive initial coordinate/address state for its exact
  `start`, and the exact accessed [element](../../../../glossary.md#term-element) span. Use checked `long` arithmetic for element and byte
  geometry.
- Use `LayoutDescriptor.storageOffset()`, `strides()`, `kind()`, and
  `referencedElementSpan()` as the resolved-layout source of truth. Do not infer a second layout
  classification or silently canonicalize a resolved layout to different geometry.
- Preserve the current four boundary values while allowing their declarations to have different
  byte spans. Declaration geometry is the checked referenced element span times the FLOAT64 width,
  not the final output logical element count.<span data-glossary-navigation="true"> ([referenced element span glossary](../../../../glossary.md#referenced-element-span))</span>
- Represent the actual ordered `double[]`/[`MemorySegment`](../../../../glossary.md#term-memorysegment) access form of every boundary position in
  `CpuKernelSpecialization` as an [immutable](../../../../glossary.md#term-immutable) ordered `List<CarrierAccess>`, where the exact nested
  enum `CarrierAccess` contains `DOUBLE_ARRAY` and `MEMORY_SEGMENT`. The list has one entry per
  ordered boundary position. The ordered carrier pattern is structural because it changes the
  generated entry descriptor and direct load/store bytecode. Exact carrier objects and their byte
  offsets remain cold invocation bindings.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Supply that exact ordered pattern through [backend](../../../../glossary.md#backend)-owned `CpuPartitionAnalysisInputs`, validate
  it against the four ordered boundary declarations during CPU analysis, and retain it in
  `CpuPartitionPreparationPlan` and `CpuKernelSpecialization`. Shared Prepare treats the backend
  input and plan opaquely. CPU finalization emits or loads the one matching artifact before Runtime
  execution; Runtime cold binding only accepts concrete carriers matching the prepared pattern and
  never generates, caches, or specializes code.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [artifact glossary](../../../../glossary.md#term-artifact); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Expand `CpuPartitionAnalysisInputs` to
  `CpuPartitionAnalysisInputs(boolean loweringManifestEnabled, List<CarrierAccess> carrierPattern)`
  while preserving `CpuPartitionAnalysisInputs.DEFAULT`. `DEFAULT` is exactly
  `loweringManifestEnabled == false` plus four ordered `MEMORY_SEGMENT` entries, matching the
  current 0005A proving topology and tests. This four-entry value is a compatibility default for
  today's fixed three-input/one-output boundary only; it is not a permanent boundary-count rule or
  architecture for later fused units.<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion))</span>
- Permit explicit user/composition-created `CpuPartitionAnalysisInputs` to supply any ordered
  non-null four-entry `DOUBLE_ARRAY`/`MEMORY_SEGMENT` pattern for 0005B without adding general
  Config or retaining physical carrier objects. Construction first rejects a null pattern
  reference, then snapshots the list immutably with null-element rejection in encounter order.
  After lowering establishes the ordered boundary list, CPU analysis compares pattern count with
  boundary count before consuming any pattern entry and then pairs entries by the same boundary
  index. A count mismatch fails closed before specialization or artifact work.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- Emit and verify exactly one direct static entry method in each generated class/artifact, with the
  exact ordered carrier signature selected for that prepared unit. <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>`CpuGeneratedKernel` retains
  that one exact handle. Do not emit unused carrier combinations, multiple alternative entries, a
  map, reflective lookup, public registry, or carrier switch in generated code.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [registry glossary](../../../../glossary.md#term-registry))</span>
- Generate the current four-boundary all-heap, all-segment, and fourteen mixed patterns on demand
  as distinct specializations when exercised. This is test coverage of sixteen possible current
  patterns, not sixteen methods in every generated class and not a fixed-width architectural
  assumption for later units with a different boundary count.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>
- Include the complete ordered [carrier](../../../../glossary.md#term-storage-carrier) pattern in `CpuKernelSpecialization.compatibilityBytes()`,
  structural key, generated binary identity, persisted artifact metadata, and class-file method
  descriptor verification. <span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [metadata glossary](../../../../glossary.md#term-metadata))</span>`CpuGeneratedKernelArtifactStore` must treat absent, reordered, or
  different carrier-pattern metadata or class bytes as incompatible and follow its existing safe
  miss/regeneration behavior.
- Emit direct primitive array-index or segment byte-address arithmetic for each structural access
  regime. A generated invocation must return on `start == end` before any carrier load, store, or
  element-address formation. Non-empty hot loops allocate no cursor or coordinate object and
  perform no semantic dispatch, map lookup, reflection, division, or modulo per element.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Initialize arbitrary [half-open](../../../../glossary.md#term-half-open-interval) `start` positions in the cold binding. Generated odometer loops
  consume primitive starting coordinates/addresses and use only increment, compare, subtract,
  carry, and reset arithmetic. This keeps the universal bound contract usable without division or
  modulo inside generated execution.
- Extend `CpuScalarReferenceKernel` to consume the same normalized access-plan bindings and carrier
  forms as generated scalar execution. It remains a conformance/fail-closed reference and is not a
  Runtime <span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>`Operation` or [canonical](../../../../glossary.md#term-canonical-form)-[IR](../../../../glossary.md#term-intermediate-representation-ir) interpreter.
- Perform cold write-legality and alias checks from actual accessed byte spans. Prove disjoint
  slices of the same carrier or segment as non-overlapping when their checked spans do not
  intersect. For the current proving slice, reject every actual input-output span intersection and
  every carrier relationship whose overlap cannot be disproved. Input-input overlap remains legal
  because all inputs are read-only.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [slice glossary](../../../../glossary.md#slice))</span>
- Accept `CpuBorrowedBuffer` and `CpuNativeBuffer` through their existing `CpuBufferArgument`
  classification. Preserve caller/run ownership, segment liveness/accessibility, read-only state,
  exact carrier-relative byte offsets, and exact selected regions without copying.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- Fail closed for every dynamic or symbolic dimension. Current shared Prepare supplies no exact
  dynamic binding: <span data-glossary-navigation="true"> ([Fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [dimension glossary](../../../../glossary.md#dimension))</span>[`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) rejects non-static projected shapes before CPU analysis. Add
  focused CPU capability/lowering evidence for this boundary and reserve dynamic support for a
  future explicit exact-binding contract; do not change shared Prepare in this task.
- Keep capability truth occurrence-local: ADD and MUL accept fully static right-broadcastable
  inputs and an exact broadcast-result output; GELU accepts equal fully static input/output shape.
  All descriptors require resolved admitted layouts. Complete-partition lowering retains the
  stricter topology, virtuality, provenance, alias, and resource checks.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [partition glossary](../../../../glossary.md#partition))</span>
- Update detailed Javadoc for every changed CPU contract and finalize the CPU backend guide and
  glossary in a distinct documentation-focused pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Any [operation](../../../../glossary.md#operation) beyond the exact [FLOAT64](../../../../glossary.md#term-float64) ADD, exact [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu), and MUL proving [topology](../../../../glossary.md#term-topology).
- Any change to GELU approximation, exact/default mode, evaluation order, tolerance, NaN,
  infinity, or signed-zero policy.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [tolerance glossary](../../../../glossary.md#term-tolerance); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Vector gather or any Vector API execution. CPU 0005C owns vector, parallel-scalar, and
  parallel-vector strategies.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [gather glossary](../../../../glossary.md#gather); [Vector API glossary](../../../../glossary.md#term-vector-api); [CPU glossary](../../../../glossary.md#term-cpu); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Worker creation, chunk scheduling, parallel binding, or changes to the scalar/single-thread
  selected strategy.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Direct-versus-contiguous materialization decisions, copy units, materialization resources, or
  cost comparison. CPU 0005D owns those decisions.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [materialization glossary](../../../../glossary.md#materialization))</span>
- Cost-gated gather. This task implements the complete scalar odometer fallback; gather remains a
  possible later vector/materialization realization, not a 0005B regime.
- Generated-class persistence policy, performance claims, tuning, benchmarks, fixed-shape
  specialization, unrolling, or a changed specialization budget.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [specialization glossary](../../../../glossary.md#term-specialization); [unrolling glossary](../../../../glossary.md#term-unrolling))</span>
- Native, [OpenBLAS](../../../../glossary.md#term-openblas), Accelerate, [oneMKL](../../../../glossary.md#term-onemkl), [oneDNN](../../../../glossary.md#term-onednn), [AOCL](../../../../glossary.md#term-aocl), [ZenDNN](../../../../glossary.md#term-zendnn), [Metal](../../../../glossary.md#term-metal), [CUDA](../../../../glossary.md#term-cuda), or vendor route work.
- POW rewriting, relaxed math, a new numerical mode, new configuration, or new capability family.
- Dynamic/symbolic execution, shared shape binding, or changes to Model, Compiler, Planning,
  Prepare, Runtime, Backend Contract, Config, Trace, Engine, architecture, dependencies, or Gradle.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- A public CPU access API, generic cursor abstraction, reusable cross-backend layout planner,
  registry, service locator, or broad facade.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [layout glossary](../../../../glossary.md#layout); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [facade glossary](../../../../glossary.md#term-facade))</span>
- [Backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), integration, end-to-end [Engine](../../../../glossary.md#term-engine), [persistence](../../../../glossary.md#term-persistence)-speed, or performance claims.
- Legacy source or package reuse. `legacy/pre-rewrite` may supply behavioral cases only.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Model,
  Planning, Runtime, Prepare, concrete backend ownership, CPU routes, and lifecycle sections.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [concrete backend glossary](../../../../glossary.md#concrete-backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md).
- [Planning Guide](../../../planning-guide.md).
- [CPU backend master plan](../master-plan.md)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>.
- [Completed CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md).
- [CPU backend guide](../../../../backend-guide/cpu-backend.md#atomic-partition-kernel-reset).
- [Glossary: CPU access plan](../../../../glossary.md#cpu-access-plan).

## Architecture constraints

- Planning continues to select only CPU ownership. CPU analysis owns access normalization,
  lowering, fusion, carrier-compatible portable realization, and exact resource declarations.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [normalization glossary](../../../../glossary.md#term-normalization); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Shared Prepare sees the CPU plan opaquely, assigns only the four declared buffers, and learns no
  broadcast, carrier, access-regime, alias, or materialization policy.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [buffers glossary](../../../../glossary.md#term-buffer); [alias glossary](../../../../glossary.md#aliasing-and-arena); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Runtime](../../../../glossary.md#term-runtime) receives one [immutable](../../../../glossary.md#term-immutable) [partition](../../../../glossary.md#partition)-level [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable); cold binding creates one
  typed <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>[`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) with direct references. Runtime does not inspect Model shapes/layouts or
  select a carrier/access regime.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout))</span>
- The Runtime [hot path](../../../../glossary.md#term-hot-path) sees no [`Operation`](../../../../glossary.md#operation) or `CompiledNode` and performs no graph inspection,
  backend discovery, route selection, kernel selection, reflection, or storage classification.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>
- `ShapeBroadcast` remains the [backend](../../../../glossary.md#backend)-independent [broadcasting](../../../../glossary.md#broadcasting) authority and
  `LayoutDescriptor` remains the resolved logical [element](../../../../glossary.md#term-element)-geometry authority.
- [Graph](../../../../glossary.md#graph) values and `LogicalMemoryPlan` entries remain unchanged. CPU virtuality alone prevents the
  two fused intermediates from receiving declarations and Runtime slots.<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion))</span>
- All changes remain in `backends/cpu` plus the listed explanatory/planning Markdown. No module or
  dependency boundary changes are authorized.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — truthful public [occurrence](../../../../glossary.md#term-occurrence) capability only.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — the sole canonical access-plan family and
  route-independent kernel IR.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [kernel glossary](../../../../glossary.md#kernel); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — right-aligned composition,
  structural normalization, fusion validation, and exact boundary facts.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [fusion glossary](../../../../glossary.md#term-fusion); [validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.memory` — existing direct heap/segment cold
  carrier classification.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — one specialization-selected
  direct carrier signature and direct scalar address/carry bytecode.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [signature glossary](../../../../glossary.md#term-signature); [carry glossary](../../../../glossary.md#term-carry); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema, structural compatibility, and
  verified artifact loading.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — opaque selected plan, exact span-based
  declarations, and post-assignment finalization.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — cold per-value binding, alias proof,
  and direct bound invocation.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — matching [scalar](../../../../glossary.md#term-scalar) [conformance](../../../../glossary.md#term-conformance) path.

Packages added or changed:

- No package is added.
- The responsibilities of the existing packages do not change.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan` — expanded in place as the one
  access contract. Its exact nested vocabulary is `Regime`, `AxisRole`, `AccessKind`, and
  [`Binding`](../../../../glossary.md#term-binding); do not create a second planner, cursor, or [layout](../../../../glossary.md#layout) abstraction.
- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr` — records each value's complete
  structural access plan and includes it in canonical identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering` — owns construction
  and composition of normalized plans from `ShapeBroadcast` and `LayoutDescriptor`.
- `io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument` — remains the sealed
  direct carrier vocabulary; 0005B uses only its existing <span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed))</span>`Doubles` and `Segment` variants.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuCarrierEmitter` and
  `CpuLoopEmitter` — own direct carrier and ordered-regime primitive emission. Do not add a generic
  emitter manager.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel` — owns the one
  verified exact static handle emitted for its specialization-selected carrier pattern.
- `io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization` — owns structural
  compatibility, including exact nested enum `CarrierAccess`, the immutable ordered carrier
  pattern, and its exact entry type.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs` — supplies
  the backend-owned exact ordered carrier pattern before analysis; its <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>`DEFAULT` remains the
  current four-`MEMORY_SEGMENT`, manifest-disabled compatibility value. Explicit instances snapshot
  their non-null pattern without retaining physical carriers. This does not add shared Config or
  permit Runtime carrier discovery.<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest); [snapshot glossary](../../../../glossary.md#term-snapshot); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan` — retains
  the validated pattern opaquely through shared assignment to finalization.
- `io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable` — owns exact
  per-value cold bindings, span/overlap validation, and the final direct invocation fields.
- `io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel` — owns the
  non-Runtime differential realization over the same bindings.

## Affected files

Expected production paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStore.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuLoopEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAccessPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuBufferArgument.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`

Expected test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFusedGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedKernelShapeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuShapePolymorphicArtifactTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAccessPlanTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuBufferBindingTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected explanatory and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0005b-universal-access-plans-and-right-aligned-broadcasting.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized by default.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 43 paths:

| Category | Maximum | Path accounting |
|---|---:|---|
| [CPU](../../../../glossary.md#term-cpu) production | 24 | The 24 existing production/package paths listed above |
| CPU tests | 14 | Thirteen existing tests plus one new `CpuAccessPlanTest` |
| Explanatory documentation | 2 | CPU [backend](../../../../glossary.md#backend) guide and glossary |
| [Planning](../../../../glossary.md#term-planning)/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **43** | **24 + 14 + 2 + 3** |

The ceiling increases by one existing production path because `CpuPartitionAnalysisInputs` must
supply the structural [carrier](../../../../glossary.md#term-storage-carrier) pattern before analysis. The simpler class [shape](../../../../glossary.md#shape) removes the
rejected exponential design without adding a new production type or package.

If another production or test path is necessary, stop and revise this specification before
implementation. Do not spend the ceiling on a broad helper, abstraction, or unrelated cleanup.

## Acceptance criteria

- CPU 0005A's exact topology lowers as one unit with one executable, four declarations, and no
  declaration or Runtime slot for either virtual intermediate.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- ADD and MUL output [shapes](../../../../glossary.md#shape) are exactly the current `ShapeBroadcast.broadcast` result; GELU
  preserves shape. Mismatched declared results and unprovable/dynamic compatibility fail closed.<span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Focused tables cover scalar, rank expansion, singleton expansion, multi-axis expansion, zero
  extents, dense, offset-dense, general positive strides, broadcast zero strides, and differing
  input/output plans.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank); [singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis); [strides glossary](../../../../glossary.md#term-stride))</span>
- `CpuAccessPlan` is the sole normalized access family and classifies every admitted plan into the
  five ordered regimes. `GENERAL_ODOMETER` executes every admitted case not captured by an earlier
  regime; there is no unsupported positive-strided remainder.
- Structural keys change for a regime, rank, axis-role pattern, contiguous-suffix form, value kind,
  data type, semantic, store, or ordered heap/segment carrier-pattern change. They do not change
  for compatible extents, concrete offset or stride values within the same structural form, exact
  carrier objects, slots, addresses, or run identity.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [data type glossary](../../../../glossary.md#data-type--datatype); [heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier); [run glossary](../../../../glossary.md#term-run); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- At least two compatible non-zero extents and a compatible zero-extent instance for one structural
  topology produce identical class bytes and reuse one loaded compatibility identity while their
  cold bindings remain distinct.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [bindings glossary](../../../../glossary.md#term-binding))</span>
- Every boundary declaration uses its own checked referenced storage span. Offset and strided
  access cannot exceed the declared/selected carrier region.
- Generated and reference execution agree for the exact fused semantics across every regime, with
  scalar, zero, rank-expanded, multi-axis, offset, positive-stride, and zero-stride cases.<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion))</span>
- All sixteen current four-boundary heap/segment patterns are accepted through on-demand
  specializations for compatible writable/read-only selections and produce the expected result.
  Each generated class has exactly one direct static entry method for its actual ordered pattern.
- `CpuPartitionAnalysisInputs.DEFAULT` has `loweringManifestEnabled == false` and exactly four
  ordered `MEMORY_SEGMENT` entries, so unchanged 0005A-style preparation fixtures select the
  current all-segment specialization. The four-entry default is documented and tested as proving-
  topology compatibility, not a general boundary-count invariant.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- Explicit analysis inputs accept each non-null four-entry pattern and snapshot caller list
  membership. Null pattern, null entry, and post-construction caller-list mutation behavior is
  exact; analysis rejects shorter or longer patterns before reading a pattern position, creating a
  specialization, or realizing an artifact, and maps a valid pattern to boundaries in declaration
  order.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Equal topology/access structure/carrier patterns with compatible different extents reuse class
  bytes and loaded identity. A different ordered carrier pattern changes the structural key,
  entry descriptor, generated bytes, and specialization identity without changing semantics.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Generated artifact compatibility and persistence metadata contain the complete ordered carrier
  pattern. A missing, reordered, or different pattern cannot be accepted as a compatible hit.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Cold binding resolves the specialization's one exact static handle and retains direct carrier
  objects and primitive geometry fields. Per-element code contains no cursor allocation, carrier
  or semantic dispatch, map access, reflection, division, or modulo.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- A concrete carrier pattern that differs from the prepared specialization fails cold binding.
  Binding performs no artifact generation, cache lookup, specialization, or handle adaptation.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Arbitrary valid `start`/`end` subranges produce correct addresses and results. Odometer
  initialization occurs in cold binding; generated carry/reset code remains primitive.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry))</span>
- A zero-element invocation returns before carrier load/store/address formation and does not touch
  output. Zero-byte legal carrier regions remain supported.
- Output write legality rejects every repeated-address write. Span checks accept proved-disjoint
  same-carrier slices and reject actual or ambiguous input-output overlap before execution.
- Heap and segment writability, liveness, thread accessibility, exact size, alignment, data type,
  byte offset, and ownership checks remain fail closed. No copy or materialization is introduced.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [alignment glossary](../../../../glossary.md#term-alignment); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [materialization glossary](../../../../glossary.md#materialization))</span>
- Exact GELU/mode, fixed ADD -> GELU -> MUL order, numerical oracle tolerance, special-value
  classifications, producer/consumer provenance, virtuality, and truthful whole-partition failure
  behavior remain unchanged from 0005A.<span data-glossary-navigation="true"> ([numerical oracle glossary](../../../../glossary.md#adr-ci-and-qualification); [tolerance glossary](../../../../glossary.md#term-tolerance); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [partition glossary](../../../../glossary.md#partition))</span>
- Capability reporting broadens only to the implemented fully static resolved occurrence matrix;
  complete lowering remains stricter. Dynamic/symbolic shapes, unresolved layouts, unsupported
  operations/types/routes/strategies, and unsafe topology facts remain rejected.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [lowering glossary](../../../../glossary.md#lowering); [layouts glossary](../../../../glossary.md#layout))</span>
- No shared Prepare, Runtime, Model, Compiler, Planning, Config, Trace, Backend Contract, Engine,
  Gradle, architecture, dependency, architecture-test, conformance-test, integration-test, native,
  or vendor path changes.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Trace glossary](../../../../glossary.md#trace); [Backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- A separate clean documentation-focused pass finalizes all affected Javadoc, CPU guide wording,
  glossary impact, links, examples, and planning/status evidence in the same overall change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [CPU](../../../../glossary.md#term-cpu) 0005B is `Complete` only after all implementation and documentation gates pass. CPU 0005A
  remains `Complete`; CPU 0005C and later remain `Draft` without detailed task specifications.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation context runs the affected-[module](../../../../glossary.md#term-module) validation once after executable Java stabilizes:

```bash
./gradlew :backends:cpu:test
```

The final [CPU](../../../../glossary.md#term-cpu) test [run](../../../../glossary.md#term-run) must include focused evidence for:

- `ShapeBroadcast`-derived ADD and MUL results and exact [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu) [shape](../../../../glossary.md#shape) preservation;
- all five regimes and the complete admitted positive/zero-[stride](../../../../glossary.md#term-stride) classification table;
- scalar, rank-expanded, singleton, multi-axis, zero-extent, offset, strided, and zero-stride
  generated/reference cases;<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank); [singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis); [extent glossary](../../../../glossary.md#term-extent))</span>
- all sixteen current ordered carrier patterns as on-demand specializations, including
  writability/liveness/size/alignment rejection and mixed carriers;<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [liveness glossary](../../../../glossary.md#term-liveness); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- exact `CpuPartitionAnalysisInputs.DEFAULT` [manifest](../../../../glossary.md#term-manifest) flag and four ordered `MEMORY_SEGMENT`
  entries; explicit all-heap/all-segment/mixed patterns; top-level and indexed null rejection;
  immutable membership snapshot; shorter/longer count rejection before specialization/artifact
  work; and exact boundary-order preservation;<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot); [specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- exactly one entry per generated class; equal-pattern compatible-extent class-byte/loaded-identity
  reuse; and different-pattern structural-key, method-descriptor, byte, and identity distinctions;<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- arbitrary ranges, cold odometer initialization, zero-element early return, declaration spans,
  repeated-write rejection, disjoint same-carrier slices, and hazardous/ambiguous overlap rejection;<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- exact numerical differential [vectors](../../../../glossary.md#term-vector) and unchanged [topology](../../../../glossary.md#term-topology)/[provenance](../../../../glossary.md#provenance)/virtuality/failure truth;
- dynamic/symbolic and every excluded semantic/[layout](../../../../glossary.md#layout)/route/strategy case failing closed; and
- stable one-entry generated class-file shape and artifact-store hit/miss/incompatibility
  verification after the generator schema change, including rejection of bytes or metadata for a
  different ordered carrier pattern.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [metadata glossary](../../../../glossary.md#term-metadata))</span>

The clean documentation-focused context reuses that successful test evidence unless it changes
executable behavior, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates, with recorded commands and results:

- every local Markdown [target](../../../../glossary.md#term-target) and explicit heading anchor in the five authorized Markdown files;
- balanced Markdown fences, final newlines, and absence of trailing whitespace;
- the exact 24-production/14-test inventory and 43-path ceiling;
- no Java change outside `backends/cpu`, no new package, no old flat `backend.cpu.execution`
  import/path, and no native/vendor placeholder;
- no generated/hot-path [`Operation`](../../../../glossary.md#operation), `CompiledNode`, map, reflection, cursor allocation,
  per-element division/modulo, storage discovery, route selection, or semantic dispatch;<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- task/master/roadmap status [synchronization](../../../../glossary.md#term-synchronization) and absence of 0005C-or-later detailed specs; and
- documentation language that distinguishes implemented 0005A/0005B behavior from later Draft
  work.

Repository-wide validation is deferred to the portable generated-coverage closure [checkpoint](../../../../glossary.md#term-checkpoint) and
[CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes only the CPU module and documentation. Architecture, [backend](../../../../glossary.md#backend)-
[conformance](../../../../glossary.md#term-conformance), and integration suites are not run because no [dependency](../../../../glossary.md#term-dependency) boundary, shared contract,
conformance claim, or end-to-end behavior is changed.

## Dependencies

- [CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md) is `Complete`.
- Current [Model](../../../../glossary.md#term-model) [`Shape`](../../../../glossary.md#shape), [`Dimension`](../../../../glossary.md#dimension), `ShapeBroadcast`, `LayoutDescriptor`, and
  `TensorDescriptor` contracts.
- Current [Planning](../../../../glossary.md#term-planning) `LogicalMemoryPlan` and `LogicalMemoryRequirement` contracts.
- Current Prepare staged analysis/declaration/assignment/finalization contracts, including the
  fully-static <span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>[`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) precondition.
- Current [Runtime](../../../../glossary.md#term-runtime) [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable), [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation), `BufferRepresentation`, representation
  binding, and <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>[`RunState`](../../../../glossary.md#run-state--runstate) contracts.
- Existing [CPU](../../../../glossary.md#term-cpu) `CpuBufferArgument`, whole-partition lowering/IR, portable artifact, and exact
  scalar reference foundation.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [lowering glossary](../../../../glossary.md#lowering); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [artifact glossary](../../../../glossary.md#term-artifact); [scalar glossary](../../../../glossary.md#term-scalar))</span>

## Follow-up tasks

- CPU 0005C: vector, parallel-scalar, and parallel-vector strategies over these access plans;
  includes any evidence-supported vector gather and external chunk binding.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [vector glossary](../../../../glossary.md#term-vector); [scalar glossary](../../../../glossary.md#term-scalar); [gather glossary](../../../../glossary.md#gather); [binding glossary](../../../../glossary.md#term-binding))</span>
- CPU 0005D: direct-versus-contiguous materialization, exact copy/resource units,
  specialization budgets, and persistence evidence.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [materialization glossary](../../../../glossary.md#materialization); [specialization glossary](../../../../glossary.md#term-specialization); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- CPU 0005E: broader portable pointwise types, carriers, and semantic families using the same
  access family.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- A future task may add dynamic/symbolic execution only after a shared exact-binding contract is
  explicitly designed and implemented. It is not implied by this task.

## Architecture impact

Expected impact: None.

This task implements [CPU](../../../../glossary.md#term-cpu)-private behavior already permitted by the [architecture contract](../../../../glossary.md#architecture-contract). If
implementation requires a shared dynamic-[binding](../../../../glossary.md#term-binding) contract, a [dependency](../../../../glossary.md#term-dependency) change, a new [module](../../../../glossary.md#term-module)
[owner](../../../../glossary.md#term-owner), or another architecture rule, stop and report the conflict instead of editing architecture.

## Implementation prompt

Use this prompt in a separate clean coding context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, completed CPU task 0005A, and this CPU task 0005B in
full. Inspect the current CPU implementation/tests and the directly relevant Model Shape/layout,
Planning logical-memory, Prepare staged-binding, and Runtime invocation/buffer contracts.

Implement task 0005B exactly as specified, within its exact path map and 43-path ceiling. Preserve
one direct static entry per ordered carrier-pattern specialization; do not emit an exponential
set of carrier combinations in a class or generate/specialize code during Runtime binding. Do not
break `CpuPartitionAnalysisInputs.DEFAULT`: it remains manifest-disabled with four ordered segment
entries for the current proving topology, while explicit inputs may select any validated four-entry
pattern. Do not implement out-of-scope work, architecture/shared-module changes, later task specs,
commits, or pushes. Stop if an architecture or scope conflict appears.

After clean implementation and the final CPU test run, hand the diff and exact test evidence to a
separate clean documentation-focused context. That pass must follow the documentation rules,
finalize affected Javadoc plus the authorized CPU guide/glossary/planning files, run Javadoc and
documentation validation, and update this task's evidence, notes, summary, and status. It must not
repeat successful Java tests unless executable behavior changes or a concrete risk is recorded.
```

## Local decisions

- Each of the five regimes emits a distinct generated primitive state machine. `DENSE_LINEAR`
  increments one address, `SCALAR_ALL_ZERO` keeps a [constant](../../../../glossary.md#term-constant) address, `LAST_AXIS_BIAS` uses one
  final-axis counter/reset, `BLOCK_OUTER` combines a contiguous-inner counter with outer carry,
  and only <span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [carry glossary](../../../../glossary.md#term-carry))</span>`GENERAL_ODOMETER` carries every iteration [axis](../../../../glossary.md#term-axis).
- Static write injectivity is decided completely for the bounded normalized geometry rather than
  by a conservative stride-order shortcut. Interleaved positive layouts such as extents <span data-glossary-navigation="true"> ([injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [stride glossary](../../../../glossary.md#term-stride); [layouts glossary](../../../../glossary.md#layout))</span>`[2, 3]`
  with strides `[3, 2]` are accepted; actual address collisions are rejected.
- Every cold binding stores the exact half-open accessed element interval for its selected logical
  range. Alias checks use those stored intervals in constant time and do not rescan elements.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [half-open glossary](../../../../glossary.md#term-half-open-interval); [element glossary](../../../../glossary.md#term-element); [Alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- The carrier-pattern list remains separate from physical carrier objects and is paired with the
  four boundary declarations only after lowering fixes their exact order.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [lowering glossary](../../../../glossary.md#lowering))</span>

## Known limitations

- Executable coverage remains the exact fully static [FLOAT64](../../../../glossary.md#term-float64) `ADD -> exact GELU -> MUL` topology,
  exact/default numerical mode, and scalar/single-thread strategy.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Dynamic or symbolic dimensions, unresolved layouts, Vector API and parallel strategies, gather,
  materialization, broader operation families, native/vendor routes, tuning, performance, backend
  conformance, and Engine integration remain outside this task.<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension); [layouts glossary](../../../../glossary.md#layout); [Vector API glossary](../../../../glossary.md#term-vector-api); [gather glossary](../../../../glossary.md#gather); [materialization glossary](../../../../glossary.md#materialization); [operation glossary](../../../../glossary.md#operation); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>
- The current compatibility default and explicit pattern input have four entries only because the
  proving topology has three materialized inputs and one output; this is not a permanent unit or
  fusion boundary-count contract.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context supplied the final executable evidence after remediation:

- `./gradlew :backends:cpu:test` passed 17 suites and 33 tests with zero skips, failures, or
  errors; 21 actionable tasks were reported, with 1 executed and 20 up-to-date.
- The final suite covers ShapeBroadcast-derived ADD/MUL results, exact GELU shape, all five
  distinct generated regime state machines, arbitrary-range generated/reference agreement,
  exhaustive small stride-pair output-injectivity comparison, exhaustive binding range/span
  comparison, all sixteen carrier patterns, exact declaration spans, cold binding/alias failures,
  compatible non-zero/zero extent reuse, and fail-closed excluded cases.<span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [shape glossary](../../../../glossary.md#shape); [stride glossary](../../../../glossary.md#term-stride); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [binding glossary](../../../../glossary.md#term-binding); [carrier glossary](../../../../glossary.md#term-storage-carrier); [alias glossary](../../../../glossary.md#aliasing-and-arena); [extent glossary](../../../../glossary.md#term-extent); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- `git diff --check` passed in the implementation context before documentation finalization.

Documentation-focused context `/root/cpu_0005b_docs` independently read and reviewed the final
24 authorized production/package paths, 14 authorized tests, architecture/[planning](../../../../glossary.md#term-planning) contracts,
[API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Backend](../../../../glossary.md#backend) Guide, Planning, General, and Example profiles, the [CPU](../../../../glossary.md#term-cpu) guide, glossary, and
planning records. It changed no executable Java and therefore reused the successful CPU test
evidence. Its final validation established:

- The generating `./gradlew :backends:cpu:javadoc && git diff --check` [run](../../../../glossary.md#term-run) passed with
  `BUILD SUCCESSFUL`; Gradle reported 11 actionable tasks, 2 executed and 9 up-to-date. Its five
  warnings were limited to the incubating Class-File API module and constructor-documentation
  warnings; no Javadoc error was reported. The final Javadoc confirmation also passed, reusing all
  11 up-to-date tasks.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [Class-File API glossary](../../../../glossary.md#term-class-file-api); [module glossary](../../../../glossary.md#term-module))</span>
- The five-file Markdown audit checked 685 local links and 290 explicit anchors with zero errors,
  while also checking balanced fences, final newlines, and trailing whitespace.
- The exact-scope audit found 24 authorized production paths, 14 authorized test paths, and five
  authorized documentation paths under the 43-path ceiling; 38 paths changed, with zero paths
  outside the map and zero Java paths outside <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`backends/cpu`.
- Package/import, native/vendor-placeholder, generated hot-path, status-synchronization, and
  later-detailed-spec audits passed. The only direct child of <span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>`internal.route` remains `portable`,
  and no CPU 0005C-or-later detailed task specification exists.

## Implementation notes

One normalized `CpuAccessPlan` now carries access kind, [axis](../../../../glossary.md#term-axis) roles, [contiguous](../../../../glossary.md#term-contiguous) suffix, and one of
five ordered regimes. Its cold [`Binding`](../../../../glossary.md#term-binding) carries concrete extents, base/effective [strides](../../../../glossary.md#term-stride),
[element](../../../../glossary.md#term-element) count, range, start [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)/address, [referenced span](../../../../glossary.md#referenced-element-span), and exact accessed subrange.
[Lowering](../../../../glossary.md#lowering) composes ADD and MUL broadcasts, requires exact [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu) [shape](../../../../glossary.md#shape), preserves the two virtual
intermediates, declares the four boundary spans independently, and proves output-write
[injectivity](../../../../glossary.md#storage-mapping-contiguity-and-injectivity). The generated and [scalar](../../../../glossary.md#term-scalar)-reference realizations consume the same normalized plans.

The generated [artifact](../../../../glossary.md#term-artifact) has one exact static entry for its prepared ordered [carrier](../../../../glossary.md#term-storage-carrier) pattern. All
sixteen current [heap](../../../../glossary.md#term-heap)/segment patterns are available on demand as distinct specializations; none
causes alternative entries in the same class. Cold binding validates carrier form, [liveness](../../../../glossary.md#term-liveness),
thread access, size, [alignment](../../../../glossary.md#term-alignment), writability, and exact input/output span overlap before retaining
direct [arguments](../../../../glossary.md#term-argument) and primitive geometry.

No shared [module](../../../../glossary.md#term-module), [architecture contract](../../../../glossary.md#architecture-contract), [dependency](../../../../glossary.md#term-dependency), [Gradle](../../../../glossary.md#term-gradle) configuration, package, native/vendor
placeholder, [conformance](../../../../glossary.md#term-conformance) test, or [integration test](../../../../glossary.md#term-integration-test) changed. [CPU](../../../../glossary.md#term-cpu) tests were not repeated in the
documentation context because only [Javadoc](../../../../glossary.md#term-javadoc) and Markdown changed after the recorded successful [run](../../../../glossary.md#term-run).

## Completion summary

- Completed changes: implemented and documented universal fully static CPU access plans,
  right-aligned broadcasting, five generated scalar regimes, exact range/span and write-injectivity
  decisions, and ordered heap/segment carrier specializations for the existing fused topology.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [broadcasting glossary](../../../../glossary.md#broadcasting); [scalar glossary](../../../../glossary.md#term-scalar); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier); [fused glossary](../../../../glossary.md#specialization-and-fusion); [topology glossary](../../../../glossary.md#term-topology))</span>
- Files changed or created: the authorized CPU production/tests and five documentation/planning
  paths recorded by this task; no path outside the 43-path map changed.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: reused the final 17-suite/33-test CPU pass; final CPU Javadoc, Markdown,
  scope/status/hot-path, and whitespace gates passed in the clean documentation context.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: `/root/cpu_0005b_docs` independently finalized affected Javadocs,
  package summaries, the CPU guide, glossary, and synchronized planning records.
- Documentation impact: current behavior and planned 0005C+ boundaries are now explicit, with a
  concrete access-regime example and no new authority claim.
- Javadoc review: affected public and technically public internal contracts now document access
  geometry, carrier patterns, ranges, ownership, results, and failures.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Glossary impact: the CPU implementation status, specialization/artifact/preparation/executable,
  and access-plan entries now describe the current whole-partition implementation.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact); [partition glossary](../../../../glossary.md#partition))</span>
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
