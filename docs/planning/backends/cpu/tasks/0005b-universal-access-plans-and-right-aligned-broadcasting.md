# Task 0005B: Universal Access Plans and Right-Aligned Broadcasting

## Status

Complete

## Goal

Extend the whole-partition CPU architecture established by task 0005A with one route-independent,
normalized per-value access-plan family. The same structural plans and cold instance bindings must
drive common lowering, the scalar reference realization, and portable generated scalar code for
the existing FLOAT64 `ADD -> exact GELU -> MUL` proving topology.

The task completes current Model right-aligned broadcasting and resolved-layout access for fully
static shapes: scalar and rank expansion, singleton and multi-axis expansion, zero extents,
canonical dense and offset-dense layouts, arbitrary non-negative strided layouts, broadcast zero
strides, and heap-array, exact-`MemorySegment`, or mixed carriers. The ordered heap/segment access
pattern is an explicit code-shaping specialization, while compatible instance extents, offsets,
strides, exact carrier objects, Runtime slots, physical addresses, and run identity remain outside
structural generated-class identity.

The resulting mental model is:

```text
ShapeBroadcast + LayoutDescriptor
  -> CPU-normalized structural access plans
  -> structural ordered carrier pattern + cold geometry/carrier objects and exact accessed spans
  -> one generated or scalar-reference whole-partition execution
```

## Scope

- Preserve the exact three-node, one-unit FLOAT64 `ADD -> GELU -> MUL` topology from 0005A,
  including exact/default numerical mode, fixed operation order, one executable, two virtual
  intermediates, four boundary declarations, producer/consumer provenance, and primitive
  `start`/`end` bounds.
- Derive every operation result shape through the current `ShapeBroadcast` contract. Validate the
  unary GELU shape exactly and require the declared final output shape to equal the final MUL
  broadcast result.
- Normalize each of the three materialized inputs against the final unit iteration shape, composing
  the ADD result's later MUL expansion without materializing either virtual value.
- Normalize the materialized output against the same iteration shape. Permit canonical dense,
  offset-dense, and positive-strided writes, but reject any output geometry that maps two distinct
  logical output coordinates to one storage element. A zero stride on a size-zero or singleton
  axis is legal only when it cannot repeat a write.
- Expand `CpuAccessPlan` into the sole CPU access family. Its structural form records access kind,
  ordered normalized axis roles, iteration rank, contiguous-suffix structure, and one of these
  ordered scalar regimes:

  1. `DENSE_LINEAR` — one linear address increment for canonical dense access, with a cold base
     offset;
  2. `SCALAR_ALL_ZERO` — scalar or all-zero-stride input access;
  3. `LAST_AXIS_BIAS` — one contiguous final indexed axis with every leading iteration axis
     broadcast, such as a right-aligned rank-one bias;
  4. `BLOCK_OUTER` — a contiguous trailing inner block with broadcast or positive-strided outer
     axes and carry/reset arithmetic outside that inner block;
  5. `GENERAL_ODOMETER` — the complete scalar fallback for every other admitted combination of
     positive and broadcast-zero effective strides.
- Keep structural regime, iteration rank, axis-role pattern, contiguous-suffix structure, value
  kind, data type, ordered semantics, stores, and the ordered boundary carrier access pattern in
  specialization/class identity. Keep concrete extents, element count, source layout offset,
  effective stride values, starting coordinates and addresses, exact carrier objects, assigned
  slots, physical addresses, and run identity in immutable cold bindings only. Two instances with
  equal structural topology, axis roles, and carrier pattern but compatible different extents
  normally reuse deterministic class bytes and loaded identity. A different carrier pattern is an
  intentionally distinct specialization.
- Make each cold access binding carry checked output extents, base element offset, effective
  right-aligned element strides, the primitive initial coordinate/address state for its exact
  `start`, and the exact accessed element span. Use checked `long` arithmetic for element and byte
  geometry.
- Use `LayoutDescriptor.storageOffset()`, `strides()`, `kind()`, and
  `referencedElementSpan()` as the resolved-layout source of truth. Do not infer a second layout
  classification or silently canonicalize a resolved layout to different geometry.
- Preserve the current four boundary values while allowing their declarations to have different
  byte spans. Declaration geometry is the checked referenced element span times the FLOAT64 width,
  not the final output logical element count.
- Represent the actual ordered `double[]`/`MemorySegment` access form of every boundary position in
  `CpuKernelSpecialization` as an immutable ordered `List<CarrierAccess>`, where the exact nested
  enum `CarrierAccess` contains `DOUBLE_ARRAY` and `MEMORY_SEGMENT`. The list has one entry per
  ordered boundary position. The ordered carrier pattern is structural because it changes the
  generated entry descriptor and direct load/store bytecode. Exact carrier objects and their byte
  offsets remain cold invocation bindings.
- Supply that exact ordered pattern through backend-owned `CpuPartitionAnalysisInputs`, validate
  it against the four ordered boundary declarations during CPU analysis, and retain it in
  `CpuPartitionPreparationPlan` and `CpuKernelSpecialization`. Shared Prepare treats the backend
  input and plan opaquely. CPU finalization emits or loads the one matching artifact before Runtime
  execution; Runtime cold binding only accepts concrete carriers matching the prepared pattern and
  never generates, caches, or specializes code.
- Expand `CpuPartitionAnalysisInputs` to
  `CpuPartitionAnalysisInputs(boolean loweringManifestEnabled, List<CarrierAccess> carrierPattern)`
  while preserving `CpuPartitionAnalysisInputs.DEFAULT`. `DEFAULT` is exactly
  `loweringManifestEnabled == false` plus four ordered `MEMORY_SEGMENT` entries, matching the
  current 0005A proving topology and tests. This four-entry value is a compatibility default for
  today's fixed three-input/one-output boundary only; it is not a permanent boundary-count rule or
  architecture for later fused units.
- Permit explicit user/composition-created `CpuPartitionAnalysisInputs` to supply any ordered
  non-null four-entry `DOUBLE_ARRAY`/`MEMORY_SEGMENT` pattern for 0005B without adding general
  Config or retaining physical carrier objects. Construction first rejects a null pattern
  reference, then snapshots the list immutably with null-element rejection in encounter order.
  After lowering establishes the ordered boundary list, CPU analysis compares pattern count with
  boundary count before consuming any pattern entry and then pairs entries by the same boundary
  index. A count mismatch fails closed before specialization or artifact work.
- Emit and verify exactly one direct static entry method in each generated class/artifact, with the
  exact ordered carrier signature selected for that prepared unit. `CpuGeneratedKernel` retains
  that one exact handle. Do not emit unused carrier combinations, multiple alternative entries, a
  map, reflective lookup, public registry, or carrier switch in generated code.
- Generate the current four-boundary all-heap, all-segment, and fourteen mixed patterns on demand
  as distinct specializations when exercised. This is test coverage of sixteen possible current
  patterns, not sixteen methods in every generated class and not a fixed-width architectural
  assumption for later units with a different boundary count.
- Include the complete ordered carrier pattern in `CpuKernelSpecialization.compatibilityBytes()`,
  structural key, generated binary identity, persisted artifact metadata, and class-file method
  descriptor verification. `CpuGeneratedKernelArtifactStore` must treat absent, reordered, or
  different carrier-pattern metadata or class bytes as incompatible and follow its existing safe
  miss/regeneration behavior.
- Emit direct primitive array-index or segment byte-address arithmetic for each structural access
  regime. A generated invocation must return on `start == end` before any carrier load, store, or
  element-address formation. Non-empty hot loops allocate no cursor or coordinate object and
  perform no semantic dispatch, map lookup, reflection, division, or modulo per element.
- Initialize arbitrary half-open `start` positions in the cold binding. Generated odometer loops
  consume primitive starting coordinates/addresses and use only increment, compare, subtract,
  carry, and reset arithmetic. This keeps the universal bound contract usable without division or
  modulo inside generated execution.
- Extend `CpuScalarReferenceKernel` to consume the same normalized access-plan bindings and carrier
  forms as generated scalar execution. It remains a conformance/fail-closed reference and is not a
  Runtime `Operation` or canonical-IR interpreter.
- Perform cold write-legality and alias checks from actual accessed byte spans. Prove disjoint
  slices of the same carrier or segment as non-overlapping when their checked spans do not
  intersect. For the current proving slice, reject every actual input-output span intersection and
  every carrier relationship whose overlap cannot be disproved. Input-input overlap remains legal
  because all inputs are read-only.
- Accept `CpuBorrowedBuffer` and `CpuNativeBuffer` through their existing `CpuBufferArgument`
  classification. Preserve caller/run ownership, segment liveness/accessibility, read-only state,
  exact carrier-relative byte offsets, and exact selected regions without copying.
- Fail closed for every dynamic or symbolic dimension. Current shared Prepare supplies no exact
  dynamic binding: `PrepareContext` rejects non-static projected shapes before CPU analysis. Add
  focused CPU capability/lowering evidence for this boundary and reserve dynamic support for a
  future explicit exact-binding contract; do not change shared Prepare in this task.
- Keep capability truth occurrence-local: ADD and MUL accept fully static right-broadcastable
  inputs and an exact broadcast-result output; GELU accepts equal fully static input/output shape.
  All descriptors require resolved admitted layouts. Complete-partition lowering retains the
  stricter topology, virtuality, provenance, alias, and resource checks.
- Update detailed Javadoc for every changed CPU contract and finalize the CPU backend guide and
  glossary in a distinct documentation-focused pass.

## Out of scope

- Any operation beyond the exact FLOAT64 ADD, exact GELU, and MUL proving topology.
- Any change to GELU approximation, exact/default mode, evaluation order, tolerance, NaN,
  infinity, or signed-zero policy.
- Vector gather or any Vector API execution. CPU 0005C owns vector, parallel-scalar, and
  parallel-vector strategies.
- Worker creation, chunk scheduling, parallel binding, or changes to the scalar/single-thread
  selected strategy.
- Direct-versus-contiguous materialization decisions, copy units, materialization resources, or
  cost comparison. CPU 0005D owns those decisions.
- Cost-gated gather. This task implements the complete scalar odometer fallback; gather remains a
  possible later vector/materialization realization, not a 0005B regime.
- Generated-class persistence policy, performance claims, tuning, benchmarks, fixed-shape
  specialization, unrolling, or a changed specialization budget.
- Native, OpenBLAS, Accelerate, oneMKL, oneDNN, AOCL, ZenDNN, Metal, CUDA, or vendor route work.
- POW rewriting, relaxed math, a new numerical mode, new configuration, or new capability family.
- Dynamic/symbolic execution, shared shape binding, or changes to Model, Compiler, Planning,
  Prepare, Runtime, Backend Contract, Config, Trace, Engine, architecture, dependencies, or Gradle.
- A public CPU access API, generic cursor abstraction, reusable cross-backend layout planner,
  registry, service locator, or broad facade.
- Backend-conformance, integration, end-to-end Engine, persistence-speed, or performance claims.
- Legacy source or package reuse. `legacy/pre-rewrite` may supply behavioral cases only.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Model,
  Planning, Runtime, Prepare, concrete backend ownership, CPU routes, and lifecycle sections.
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md).
- [Planning Guide](../../../planning-guide.md).
- [CPU backend master plan](../master-plan.md).
- [Completed CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md).
- [CPU backend guide](../../../../backend-guide/cpu-backend.md#atomic-partition-kernel-reset).
- [Glossary: CPU access plan](../../../../glossary.md#cpu-access-plan).

## Architecture constraints

- Planning continues to select only CPU ownership. CPU analysis owns access normalization,
  lowering, fusion, carrier-compatible portable realization, and exact resource declarations.
- Shared Prepare sees the CPU plan opaquely, assigns only the four declared buffers, and learns no
  broadcast, carrier, access-regime, alias, or materialization policy.
- Runtime receives one immutable partition-level `PreparedExecutable`; cold binding creates one
  typed `BoundInvocation` with direct references. Runtime does not inspect Model shapes/layouts or
  select a carrier/access regime.
- The Runtime hot path sees no `Operation` or `CompiledNode` and performs no graph inspection,
  backend discovery, route selection, kernel selection, reflection, or storage classification.
- `ShapeBroadcast` remains the backend-independent broadcasting authority and
  `LayoutDescriptor` remains the resolved logical element-geometry authority.
- Graph values and `LogicalMemoryPlan` entries remain unchanged. CPU virtuality alone prevents the
  two fused intermediates from receiving declarations and Runtime slots.
- All changes remain in `backends/cpu` plus the listed explanatory/planning Markdown. No module or
  dependency boundary changes are authorized.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — truthful public occurrence capability only.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — the sole canonical access-plan family and
  route-independent kernel IR.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — right-aligned composition,
  structural normalization, fusion validation, and exact boundary facts.
- `io.github.pho001.synaptik.backend.cpu.internal.memory` — existing direct heap/segment cold
  carrier classification.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — one specialization-selected
  direct carrier signature and direct scalar address/carry bytecode.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema, structural compatibility, and
  verified artifact loading.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — opaque selected plan, exact span-based
  declarations, and post-assignment finalization.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — cold per-value binding, alias proof,
  and direct bound invocation.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — matching scalar conformance path.

Packages added or changed:

- No package is added.
- The responsibilities of the existing packages do not change.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan` — expanded in place as the one
  access contract. Its exact nested vocabulary is `Regime`, `AxisRole`, `AccessKind`, and
  `Binding`; do not create a second planner, cursor, or layout abstraction.
- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr` — records each value's complete
  structural access plan and includes it in canonical identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering` — owns construction
  and composition of normalized plans from `ShapeBroadcast` and `LayoutDescriptor`.
- `io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument` — remains the sealed
  direct carrier vocabulary; 0005B uses only its existing `Doubles` and `Segment` variants.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuCarrierEmitter` and
  `CpuLoopEmitter` — own direct carrier and ordered-regime primitive emission. Do not add a generic
  emitter manager.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel` — owns the one
  verified exact static handle emitted for its specialization-selected carrier pattern.
- `io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization` — owns structural
  compatibility, including exact nested enum `CarrierAccess`, the immutable ordered carrier
  pattern, and its exact entry type.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs` — supplies
  the backend-owned exact ordered carrier pattern before analysis; its `DEFAULT` remains the
  current four-`MEMORY_SEGMENT`, manifest-disabled compatibility value. Explicit instances snapshot
  their non-null pattern without retaining physical carriers. This does not add shared Config or
  permit Runtime carrier discovery.
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

Expected explanatory and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0005b-universal-access-plans-and-right-aligned-broadcasting.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized by default.

## Maximum scope

This task may create or modify at most 43 paths:

| Category | Maximum | Path accounting |
|---|---:|---|
| CPU production | 24 | The 24 existing production/package paths listed above |
| CPU tests | 14 | Thirteen existing tests plus one new `CpuAccessPlanTest` |
| Explanatory documentation | 2 | CPU backend guide and glossary |
| Planning/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **43** | **24 + 14 + 2 + 3** |

The ceiling increases by one existing production path because `CpuPartitionAnalysisInputs` must
supply the structural carrier pattern before analysis. The simpler class shape removes the
rejected exponential design without adding a new production type or package.

If another production or test path is necessary, stop and revise this specification before
implementation. Do not spend the ceiling on a broad helper, abstraction, or unrelated cleanup.

## Acceptance criteria

- CPU 0005A's exact topology lowers as one unit with one executable, four declarations, and no
  declaration or Runtime slot for either virtual intermediate.
- ADD and MUL output shapes are exactly the current `ShapeBroadcast.broadcast` result; GELU
  preserves shape. Mismatched declared results and unprovable/dynamic compatibility fail closed.
- Focused tables cover scalar, rank expansion, singleton expansion, multi-axis expansion, zero
  extents, dense, offset-dense, general positive strides, broadcast zero strides, and differing
  input/output plans.
- `CpuAccessPlan` is the sole normalized access family and classifies every admitted plan into the
  five ordered regimes. `GENERAL_ODOMETER` executes every admitted case not captured by an earlier
  regime; there is no unsupported positive-strided remainder.
- Structural keys change for a regime, rank, axis-role pattern, contiguous-suffix form, value kind,
  data type, semantic, store, or ordered heap/segment carrier-pattern change. They do not change
  for compatible extents, concrete offset or stride values within the same structural form, exact
  carrier objects, slots, addresses, or run identity.
- At least two compatible non-zero extents and a compatible zero-extent instance for one structural
  topology produce identical class bytes and reuse one loaded compatibility identity while their
  cold bindings remain distinct.
- Every boundary declaration uses its own checked referenced storage span. Offset and strided
  access cannot exceed the declared/selected carrier region.
- Generated and reference execution agree for the exact fused semantics across every regime, with
  scalar, zero, rank-expanded, multi-axis, offset, positive-stride, and zero-stride cases.
- All sixteen current four-boundary heap/segment patterns are accepted through on-demand
  specializations for compatible writable/read-only selections and produce the expected result.
  Each generated class has exactly one direct static entry method for its actual ordered pattern.
- `CpuPartitionAnalysisInputs.DEFAULT` has `loweringManifestEnabled == false` and exactly four
  ordered `MEMORY_SEGMENT` entries, so unchanged 0005A-style preparation fixtures select the
  current all-segment specialization. The four-entry default is documented and tested as proving-
  topology compatibility, not a general boundary-count invariant.
- Explicit analysis inputs accept each non-null four-entry pattern and snapshot caller list
  membership. Null pattern, null entry, and post-construction caller-list mutation behavior is
  exact; analysis rejects shorter or longer patterns before reading a pattern position, creating a
  specialization, or realizing an artifact, and maps a valid pattern to boundaries in declaration
  order.
- Equal topology/access structure/carrier patterns with compatible different extents reuse class
  bytes and loaded identity. A different ordered carrier pattern changes the structural key,
  entry descriptor, generated bytes, and specialization identity without changing semantics.
- Generated artifact compatibility and persistence metadata contain the complete ordered carrier
  pattern. A missing, reordered, or different pattern cannot be accepted as a compatible hit.
- Cold binding resolves the specialization's one exact static handle and retains direct carrier
  objects and primitive geometry fields. Per-element code contains no cursor allocation, carrier
  or semantic dispatch, map access, reflection, division, or modulo.
- A concrete carrier pattern that differs from the prepared specialization fails cold binding.
  Binding performs no artifact generation, cache lookup, specialization, or handle adaptation.
- Arbitrary valid `start`/`end` subranges produce correct addresses and results. Odometer
  initialization occurs in cold binding; generated carry/reset code remains primitive.
- A zero-element invocation returns before carrier load/store/address formation and does not touch
  output. Zero-byte legal carrier regions remain supported.
- Output write legality rejects every repeated-address write. Span checks accept proved-disjoint
  same-carrier slices and reject actual or ambiguous input-output overlap before execution.
- Heap and segment writability, liveness, thread accessibility, exact size, alignment, data type,
  byte offset, and ownership checks remain fail closed. No copy or materialization is introduced.
- Exact GELU/mode, fixed ADD -> GELU -> MUL order, numerical oracle tolerance, special-value
  classifications, producer/consumer provenance, virtuality, and truthful whole-partition failure
  behavior remain unchanged from 0005A.
- Capability reporting broadens only to the implemented fully static resolved occurrence matrix;
  complete lowering remains stricter. Dynamic/symbolic shapes, unresolved layouts, unsupported
  operations/types/routes/strategies, and unsafe topology facts remain rejected.
- No shared Prepare, Runtime, Model, Compiler, Planning, Config, Trace, Backend Contract, Engine,
  Gradle, architecture, dependency, architecture-test, conformance-test, integration-test, native,
  or vendor path changes.
- A separate clean documentation-focused pass finalizes all affected Javadoc, CPU guide wording,
  glossary impact, links, examples, and planning/status evidence in the same overall change.
- CPU 0005B is `Complete` only after all implementation and documentation gates pass. CPU 0005A
  remains `Complete`; CPU 0005C and later remain `Draft` without detailed task specifications.

## Tests / validation

Implementation context runs the affected-module validation once after executable Java stabilizes:

```bash
./gradlew :backends:cpu:test
```

The final CPU test run must include focused evidence for:

- `ShapeBroadcast`-derived ADD and MUL results and exact GELU shape preservation;
- all five regimes and the complete admitted positive/zero-stride classification table;
- scalar, rank-expanded, singleton, multi-axis, zero-extent, offset, strided, and zero-stride
  generated/reference cases;
- all sixteen current ordered carrier patterns as on-demand specializations, including
  writability/liveness/size/alignment rejection and mixed carriers;
- exact `CpuPartitionAnalysisInputs.DEFAULT` manifest flag and four ordered `MEMORY_SEGMENT`
  entries; explicit all-heap/all-segment/mixed patterns; top-level and indexed null rejection;
  immutable membership snapshot; shorter/longer count rejection before specialization/artifact
  work; and exact boundary-order preservation;
- exactly one entry per generated class; equal-pattern compatible-extent class-byte/loaded-identity
  reuse; and different-pattern structural-key, method-descriptor, byte, and identity distinctions;
- arbitrary ranges, cold odometer initialization, zero-element early return, declaration spans,
  repeated-write rejection, disjoint same-carrier slices, and hazardous/ambiguous overlap rejection;
- exact numerical differential vectors and unchanged topology/provenance/virtuality/failure truth;
- dynamic/symbolic and every excluded semantic/layout/route/strategy case failing closed; and
- stable one-entry generated class-file shape and artifact-store hit/miss/incompatibility
  verification after the generator schema change, including rejection of bytes or metadata for a
  different ordered carrier pattern.

The clean documentation-focused context reuses that successful test evidence unless it changes
executable behavior, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates, with recorded commands and results:

- every local Markdown target and explicit heading anchor in the five authorized Markdown files;
- balanced Markdown fences, final newlines, and absence of trailing whitespace;
- the exact 24-production/14-test inventory and 43-path ceiling;
- no Java change outside `backends/cpu`, no new package, no old flat `backend.cpu.execution`
  import/path, and no native/vendor placeholder;
- no generated/hot-path `Operation`, `CompiledNode`, map, reflection, cursor allocation,
  per-element division/modulo, storage discovery, route selection, or semantic dispatch;
- task/master/roadmap status synchronization and absence of 0005C-or-later detailed specs; and
- documentation language that distinguishes implemented 0005A/0005B behavior from later Draft
  work.

Repository-wide validation is deferred to the portable generated-coverage closure checkpoint and
CI because this task changes only the CPU module and documentation. Architecture, backend-
conformance, and integration suites are not run because no dependency boundary, shared contract,
conformance claim, or end-to-end behavior is changed.

## Dependencies

- [CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md) is `Complete`.
- Current Model `Shape`, `Dimension`, `ShapeBroadcast`, `LayoutDescriptor`, and
  `TensorDescriptor` contracts.
- Current Planning `LogicalMemoryPlan` and `LogicalMemoryRequirement` contracts.
- Current Prepare staged analysis/declaration/assignment/finalization contracts, including the
  fully-static `PrepareContext` precondition.
- Current Runtime `PreparedExecutable`, `BoundInvocation`, `BufferRepresentation`, representation
  binding, and `RunState` contracts.
- Existing CPU `CpuBufferArgument`, whole-partition lowering/IR, portable artifact, and exact
  scalar reference foundation.

## Follow-up tasks

- CPU 0005C: vector, parallel-scalar, and parallel-vector strategies over these access plans;
  includes any evidence-supported vector gather and external chunk binding.
- CPU 0005D: direct-versus-contiguous materialization, exact copy/resource units,
  specialization budgets, and persistence evidence.
- CPU 0005E: broader portable pointwise types, carriers, and semantic families using the same
  access family.
- A future task may add dynamic/symbolic execution only after a shared exact-binding contract is
  explicitly designed and implemented. It is not implied by this task.

## Architecture impact

Expected impact: None.

This task implements CPU-private behavior already permitted by the architecture contract. If
implementation requires a shared dynamic-binding contract, a dependency change, a new module
owner, or another architecture rule, stop and report the conflict instead of editing architecture.

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
  increments one address, `SCALAR_ALL_ZERO` keeps a constant address, `LAST_AXIS_BIAS` uses one
  final-axis counter/reset, `BLOCK_OUTER` combines a contiguous-inner counter with outer carry,
  and only `GENERAL_ODOMETER` carries every iteration axis.
- Static write injectivity is decided completely for the bounded normalized geometry rather than
  by a conservative stride-order shortcut. Interleaved positive layouts such as extents `[2, 3]`
  with strides `[3, 2]` are accepted; actual address collisions are rejected.
- Every cold binding stores the exact half-open accessed element interval for its selected logical
  range. Alias checks use those stored intervals in constant time and do not rescan elements.
- The carrier-pattern list remains separate from physical carrier objects and is paired with the
  four boundary declarations only after lowering fixes their exact order.

## Known limitations

- Executable coverage remains the exact fully static FLOAT64 `ADD -> exact GELU -> MUL` topology,
  exact/default numerical mode, and scalar/single-thread strategy.
- Dynamic or symbolic dimensions, unresolved layouts, Vector API and parallel strategies, gather,
  materialization, broader operation families, native/vendor routes, tuning, performance, backend
  conformance, and Engine integration remain outside this task.
- The current compatibility default and explicit pattern input have four entries only because the
  proving topology has three materialized inputs and one output; this is not a permanent unit or
  fusion boundary-count contract.

## Validation evidence

Implementation context supplied the final executable evidence after remediation:

- `./gradlew :backends:cpu:test` passed 17 suites and 33 tests with zero skips, failures, or
  errors; 21 actionable tasks were reported, with 1 executed and 20 up-to-date.
- The final suite covers ShapeBroadcast-derived ADD/MUL results, exact GELU shape, all five
  distinct generated regime state machines, arbitrary-range generated/reference agreement,
  exhaustive small stride-pair output-injectivity comparison, exhaustive binding range/span
  comparison, all sixteen carrier patterns, exact declaration spans, cold binding/alias failures,
  compatible non-zero/zero extent reuse, and fail-closed excluded cases.
- `git diff --check` passed in the implementation context before documentation finalization.

Documentation-focused context `/root/cpu_0005b_docs` independently read and reviewed the final
24 authorized production/package paths, 14 authorized tests, architecture/planning contracts,
API/Javadoc, Backend Guide, Planning, General, and Example profiles, the CPU guide, glossary, and
planning records. It changed no executable Java and therefore reused the successful CPU test
evidence. Its final validation established:

- The generating `./gradlew :backends:cpu:javadoc && git diff --check` run passed with
  `BUILD SUCCESSFUL`; Gradle reported 11 actionable tasks, 2 executed and 9 up-to-date. Its five
  warnings were limited to the incubating Class-File API module and constructor-documentation
  warnings; no Javadoc error was reported. The final Javadoc confirmation also passed, reusing all
  11 up-to-date tasks.
- The five-file Markdown audit checked 685 local links and 290 explicit anchors with zero errors,
  while also checking balanced fences, final newlines, and trailing whitespace.
- The exact-scope audit found 24 authorized production paths, 14 authorized test paths, and five
  authorized documentation paths under the 43-path ceiling; 38 paths changed, with zero paths
  outside the map and zero Java paths outside `backends/cpu`.
- Package/import, native/vendor-placeholder, generated hot-path, status-synchronization, and
  later-detailed-spec audits passed. The only direct child of `internal.route` remains `portable`,
  and no CPU 0005C-or-later detailed task specification exists.

## Implementation notes

One normalized `CpuAccessPlan` now carries access kind, axis roles, contiguous suffix, and one of
five ordered regimes. Its cold `Binding` carries concrete extents, base/effective strides,
element count, range, start coordinates/address, referenced span, and exact accessed subrange.
Lowering composes ADD and MUL broadcasts, requires exact GELU shape, preserves the two virtual
intermediates, declares the four boundary spans independently, and proves output-write
injectivity. The generated and scalar-reference realizations consume the same normalized plans.

The generated artifact has one exact static entry for its prepared ordered carrier pattern. All
sixteen current heap/segment patterns are available on demand as distinct specializations; none
causes alternative entries in the same class. Cold binding validates carrier form, liveness,
thread access, size, alignment, writability, and exact input/output span overlap before retaining
direct arguments and primitive geometry.

No shared module, architecture contract, dependency, Gradle configuration, package, native/vendor
placeholder, conformance test, or integration test changed. CPU tests were not repeated in the
documentation context because only Javadoc and Markdown changed after the recorded successful run.

## Completion summary

- Completed changes: implemented and documented universal fully static CPU access plans,
  right-aligned broadcasting, five generated scalar regimes, exact range/span and write-injectivity
  decisions, and ordered heap/segment carrier specializations for the existing fused topology.
- Files changed or created: the authorized CPU production/tests and five documentation/planning
  paths recorded by this task; no path outside the 43-path map changed.
- Tests and validation: reused the final 17-suite/33-test CPU pass; final CPU Javadoc, Markdown,
  scope/status/hot-path, and whitespace gates passed in the clean documentation context.
- Documentation-agent review: `/root/cpu_0005b_docs` independently finalized affected Javadocs,
  package summaries, the CPU guide, glossary, and synchronized planning records.
- Documentation impact: current behavior and planned 0005C+ boundaries are now explicit, with a
  concrete access-regime example and no new authority claim.
- Javadoc review: affected public and technically public internal contracts now document access
  geometry, carrier patterns, ranges, ownership, results, and failures.
- Glossary impact: the CPU implementation status, specialization/artifact/preparation/executable,
  and access-plan entries now describe the current whole-partition implementation.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
