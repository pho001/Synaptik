# Task 0006: Portable Static Affine Views and Boundary Materialization

## Status

Complete

## Goal

Add the first bounded layout/indexing slice after completed CPU 0005J. CPU preparation will
compose a connected straight-line chain of fully static, resolved-layout affine view operations
into one source-to-result mapping. Intermediate view values remain graph values but require no
computation, buffer declaration, Runtime slot, or generated store when they have no publication,
fan-out, or partition-boundary obligation. When the final result must cross the unit boundary,
the portable route generates one exact scalar or parallel-scalar copy into the result's resolved
layout.

The admitted semantic set is deliberately closed:

```text
CONTIGUOUS
RESHAPE | EXPAND
PERMUTE | EXPAND_DIMS | SQUEEZE
SELECT
SLICE with SliceAttrs or CropToShapeAttrs
```

Every admitted occurrence must have fully static input/output Shapes and resolved input/output
layouts, and every Shape-valued attribute needed for mapping must also be fully static. All six
current Model data types are copied as represented bits. BFLOAT16 support is limited to raw
two-byte movement through the existing `CpuBufferArgument.Shorts` representation and a new
`CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY` form; it does not advertise BFLOAT16
pointwise arithmetic, conversion, numerical semantics, or Vector API support. The task adds no
numerical operation, index Tensor, duplicate-target behavior, ordering algorithm, random
algorithm, public API, shared physical-layout contract, or general partition-DAG decomposition.

Mental model:

```text
validated static affine chain
  -> compose exact logical-coordinate mappings during CPU analysis
  -> internal view values: descriptor/IR facts only, no work and no slot
  -> final boundary: one generated represented-bit copy into exact resolved output geometry
  -> one partition executable
```

This is the largest dependency-correct normal slice of the former broad 0006 row. Non-affine
movement/gather, functional scatter/fold, ordering, and explicit-state random work remain four
ordered Draft follow-ups in the CPU master plan and have no detailed specifications.

## Scope

### Fresh current-operation inventory

The inventory below is derived from the current Model `OperationKind` families and signatures,
not from the former broad CPU row. A classification describes the implementation role of an
eligible occurrence; a kind can have both a zero-work internal case and a materialized boundary
case.

| Current Model operation | Classification | CPU 0006 disposition |
|---|---|---|
| `ContiguousKind.CONTIGUOUS` | descriptor request when its input representation already satisfies the result; otherwise executable data movement | Admit only the resolved-layout static one-input/one-output occurrence; the current no-alias shared resource model means its final boundary is an exact copy. |
| `ShapeTransformKind.RESHAPE` | metadata-only affine coordinate reinterpretation internally; boundary copy when a distinct result slot is required | Admit resolved-layout static occurrences and compose ordered-logical-element mapping. |
| `ShapeTransformKind.EXPAND` | metadata-only zero-stride affine view internally; boundary materialization over distinct output addresses | Admit resolved-layout static occurrences; repeated logical positions must not cause parallel duplicate writes. |
| `AxisTransformKind.PERMUTE` | metadata-only stride/axis permutation internally; boundary copy when required | Admit resolved-layout static occurrences. |
| `AxisTransformKind.EXPAND_DIMS` | metadata-only extent-one axis insertion internally; boundary copy when required | Admit resolved-layout static occurrences. |
| `AxisTransformKind.SQUEEZE` | metadata-only extent-one axis removal internally; boundary copy when required | Admit resolved-layout static occurrences. |
| `SelectKind.SELECT` | metadata-only fixed-coordinate view internally; executable data movement at a boundary | Admit resolved-layout static occurrences with the already-normalized intrinsic coordinate. |
| `SliceKind.SLICE` with `SliceAttrs` | metadata-only affine view when the current result layout is resolved; otherwise executable extraction requiring a selected physical layout | Admit only fully static resolved-layout occurrences. Negative-step results are currently layout-unresolved and therefore remain fail-closed. |
| `SliceKind.SLICE` with `CropToShapeAttrs` | metadata-only target-relative affine view after exact attributes are static; otherwise requires unresolved binding/layout support | Admit only when target and prefix Shapes and both descriptors are fully static and resolved. |
| `PadKind.PAD`, `TileKind.TILE`, `TensorCompositionKind.CONCAT`/`STACK` | executable non-affine data movement | Defer to Draft 0006A. |
| `WindowTransformKind.UNFOLD_AXIS`/`UNFOLD2D` | executable non-affine window data movement, including explicit padding for `Unfold2dAttrs` | Defer to Draft 0006A. |
| `AxisGatherKind.GATHER`/`GATHER_ELEMENTS`, `GatherNdKind.GATHER_ND`, `OneHotKind.ONE_HOT` | executable value-dependent indexing/data generation with execution-time index bounds | Defer to Draft 0006A. |
| `SliceKind.SLICE_UPDATE` | functional base copy plus indexed replacement | Defer to Draft 0006B. |
| `AxisScatterKind.SCATTER_ELEMENTS`/`SCATTER_ADD`, `ScatterNdKind.SCATTER_ND` | executable functional scatter; `SCATTER_ELEMENTS`/`SCATTER_ND` admit `ScatterReduction.NONE`, `ScatterReduction.ADD`, `ScatterReduction.MUL`, `ScatterReduction.MAX`, and `ScatterReduction.MIN`, while `SCATTER_ADD` fixes ADD; duplicate semantics follow that exact variant | Defer to Draft 0006B. |
| `WindowTransformKind.FOLD_AXIS`/`FOLD2D` | executable overlap scatter-add | Defer to Draft 0006B rather than pulling reduction work into 0006 or general reductions into CPU 0007. |
| `OrderingKind.SORT`/`ARGSORT` | stable full-ordering algorithm with exact tie, NaN, and signed-zero rules | Defer to Draft 0006C. |
| `TopKKind.TOP_K` | deterministic selection algorithm with two outputs | Defer to Draft 0006C. |
| `GraphRngKind.INITIAL_STATE` | explicit-state descriptor/value initialization, not hidden Runtime state | Defer to Draft 0006D. |
| `DropoutKind.DROPOUT` | explicit-state random work with three outputs: value, auxiliary mask, and next state | Defer to Draft 0006D. |

Unsupported occurrences remain unadvertised and fail before artifact lookup. The deferred rows are
not authorization to implement their algorithms in this task.

### Affine occurrence and chain eligibility

- Extend `CpuCapabilityProvider` only for the exact admitted occurrence-local matrix. Require the
  operation's exact current attributes class and signature, fully static Shapes, resolved layouts,
  same input/output data type, and exact Model Shape/layout relationship. Do not reinterpret or
  repair Model descriptors.
- Accept every current `DataType`: FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, and BOOL. Copy raw
  represented values. Preserve floating NaN payload bits, infinities, both signed zeros, signed
  integral bit patterns, BFLOAT16 raw bits, and canonical BOOL bytes without conversion or
  arithmetic.
- Extend `CpuKernelSpecialization.CarrierAccess` from its current six forms to exactly seven:
  `DOUBLE_ARRAY`, `FLOAT_ARRAY`, `SHORT_ARRAY`, `INT_ARRAY`, `LONG_ARRAY`, `BYTE_ARRAY`, and
  `MEMORY_SEGMENT`. The exact heap mapping is FLOAT64→`DOUBLE_ARRAY`, FLOAT32→`FLOAT_ARRAY`,
  BFLOAT16→`SHORT_ARRAY`, INT32→`INT_ARRAY`, INT64→`LONG_ARRAY`, and BOOL→`BYTE_ARRAY`; every
  data type may instead use `MEMORY_SEGMENT` when the matching observable heap carrier is absent.
  `SHORT_ARRAY` binds only the already-existing `CpuBufferArgument.Shorts` raw-bit carrier.
- Keep the existing pointwise capability and lowering rejection for BFLOAT16 unchanged. Affine
  copy eligibility is a separate operation-family route whose load/store treats each BFLOAT16
  element as one opaque 16-bit payload; it performs no widening, narrowing, rounding, arithmetic,
  comparison, NaN handling, or vector-lane interpretation.
- Lower one through eight nodes only when they form one connected straight-line affine chain with
  exactly one external data input and one final output. Every non-final output must feed the next
  node exactly once, have no other consumer, not be a graph output, and remain inside the same
  planned partition.
- Permit `CONTIGUOUS` anywhere in the chain as an explicit materialization barrier in Model
  semantics, while still composing the logical mapping. It does not authorize alias reuse or
  elimination across a boundary.
- Reject mixed pointwise/affine chains, branches, joins, multiple external data inputs, multiple
  final outputs, repeated intermediate uses, and disconnected nodes. CPU 0008A remains the owner
  of general partition-DAG decomposition and bounded horizontal or general vertical fusion.
- Consume normalized axes, coordinates, permutations, lengths, signed steps, exact target Shapes,
  and prefix Shapes as captured. CPU validates them against projected descriptors but does not
  normalize caller syntax or invent compiler semantics.
- Require every `TargetShapeAttrs.targetShape`, `CropToShapeAttrs.targetShape`, and
  `CropToShapeAttrs.prefixShape` used by the admitted mapping to be fully static. Current Prepare
  supplies no binding substitution for symbolic attribute values.

### Metadata-only and physical-boundary rules

- Represent each internal affine result as a topology-local metadata value. Compose its mapping
  into the final source load and do not emit an instruction, class call, buffer declaration,
  workspace, Runtime slot, or store for that value.
- Retain every internal value and descriptor in the immutable graph and `LogicalMemoryPlan`.
  CPU-private virtuality changes only CPU declarations and lowering.
- Declare exactly the source and final result buffers for an eligible chain. A final result always
  needs a declaration because current shared preparation has no value-alias assignment and
  publication/cross-partition consumers address a result `ValueId`, not an input representation.
- Write the final result according to its exact resolved `LayoutDescriptor`, including offset,
  positive strides, zero strides, referenced element span, and view classification. Do not silently
  replace it with canonical dense geometry.
- For an injective result layout, the generated range domain is the logical result element range.
  For a zero-stride/non-injective result layout, derive a deterministic distinct-address domain and
  write each referenced result address at most once; allow parallel orchestration only when those
  writes are disjoint. Every omitted repeated logical coordinate must map to the same source value,
  or lowering fails closed.
- Preserve exact source/output accessed spans and reject any source/result overlap that cannot be
  proved safe. Do not add in-place view mutation or copy-direction heuristics.
- Zero-element results perform no carrier access. Scalar results use one logical element.

### CPU-private IR, lowering, and generated execution

- Add one small sealed CPU-private portable-kernel-IR role shared only by the existing pointwise
  IR and one new affine-copy IR. The existing pointwise structure and semantic inventory remain
  unchanged.
- The affine-copy IR records only topology-local type, the exact composed structural mapping,
  source and result access-plan forms, iteration/write-domain form, and one ordered store. It does
  not retain `Operation`, `CompiledNode`, graph or Runtime identities, concrete carriers, slots,
  addresses, route, worker, or artifact-store state.
- Keep family interpretation in a dedicated affine lowerer. `CpuPartitionLowering` dispatches
  cold analysis to the existing pointwise lowerer path or the new affine path; it must not become
  a monolithic switch containing both families' algorithms.
- Add a family-specific affine emitter using Java 26 Class-File API generation. Reuse the existing
  carrier load/store and primitive odometer facilities where their contracts match; do not
  duplicate one implementation per carrier or data type.
- Generated entry descriptors remain specialized to the exact ordered source/result carrier
  pattern. Admit the six exact primitive-array forms plus `MemorySegment`, producing exactly seven
  final `CarrierAccess` forms. For each logical data type, admit its matching array form and
  `MemorySegment`, including all-array, all-segment, and both mixed two-boundary patterns.
- Generated BFLOAT16 affine loads and stores use JVM `short` array operations or exact native-order
  two-byte `MemorySegment` access and preserve the 16 represented bits unchanged. These carrier
  primitives are available only to the affine copy emitter/reference path in this task; they do
  not make BFLOAT16 a supported pointwise scalar or vector lane type.
- Generated affine hot paths receive direct primitive carriers, cold geometry, and primitive
  `start`/`end` bounds. They perform no operation/graph/shape/layout/route/carrier switch, map
  lookup, reflection, allocation, virtual cursor call, or per-element division/modulo.
- Preserve exactly the existing scalar/vector compute axis and single/parallel orchestration
  vocabulary. Affine-copy bodies are scalar-compute in this task. `VECTOR_IF_ELIGIBLE` therefore
  selects deterministic scalar fallback, while eligible disjoint ranges may select
  parallel-scalar orchestration. Do not claim vector, gather, scatter, or masked-tail execution.
- Preserve the four-complete-candidate, one-realized-artifact, zero-fixed-shape, and zero-unroll
  ceilings. An affine chain forms one direct candidate and one artifact; it creates no optional
  materialization workspace because its boundary copy is the semantic realization itself.
- Advance the CPU generator schema exactly once. Include IR family, mapping topology, data type,
  source/result access structure, carrier pattern, write-domain form, execution compute form, and
  every other bytecode/compatibility-changing fact. Retain existing exclusions for compatible
  extents, offsets/stride magnitudes within one structure, slots, carriers, addresses, workers,
  graph/run identity, and artifact root.
- Keep optional generated-class persistence disabled by default and retain one current schema
  with no migration reader. Finalization realizes or reuses the selected artifact only after exact
  shared assignments are present.
- Extend the scalar reference path with the same composed mapping for differential evidence. It
  remains a test/conformance reference, never a Runtime operation or IR interpreter.

### Lifecycle and failure behavior

- CPU analysis performs Model/graph inspection, chain validation, mapping composition, strategy
  selection, specialization, and exact two-buffer declaration. Shared Prepare sees only the
  opaque selected plan and declarations.
- CPU finalization validates both assigned buffer positions before artifact access and constructs
  one immutable partition executable. It cannot change mapping, strategy, route, or resources.
- Cold binding validates exact type, carrier, size, alignment, accessibility, read-only output
  rejection, canonical BOOL inputs, and span non-overlap once. The bound invocation retains direct
  typed references.
- Runtime executes only the prepared invocation. It does not receive `Operation`, `CompiledNode`,
  Shape, layout, affine IR, or route choice and does not interpret a view.
- Generated/reference failures are deterministic and fail closed. No partial success changes the
  existing Runtime output-validity rules.

### Stop conditions

Stop implementation and report the exact conflict before broadening the diff if any of these is
required:

- assigning two distinct graph values to one shared Runtime slot or representation;
- attaching a CPU-selected physical layout to an unresolved value through a new shared contract;
- binding a dynamic/symbolic Shape or Shape-valued attribute during CPU preparation;
- changing Model, Compiler, Planning, shared Prepare, Runtime, public API, architecture, module
  dependencies, or Gradle;
- accepting a multi-output, random, ordering, scatter/fold, mixed-family, or general-DAG unit;
- adding a second route, workspace, artifact per analysis, package, or path outside the exact map;
  or
- discovering that the current captured descriptors do not determine one exact admitted mapping.

## Out of scope

- PAD, TILE, CONCAT, STACK, UNFOLD_AXIS, UNFOLD2D, GATHER, GATHER_ELEMENTS, GATHER_ND, ONE_HOT,
  SLICE_UPDATE, every scatter, FOLD_AXIS, FOLD2D, SORT, ARGSORT, TOP_K, INITIAL_STATE, or DROPOUT.
- Any dynamic or symbolic Shape execution, binding of Shape-valued attributes, negative-step
  layout invention, or a shared prepared physical-layout/alias contract.
- Multi-output CPU execution, state outputs, random algorithms, ordering workspaces, scatter
  duplicate handling, overlap accumulation, atomics, locks, or reductions.
- General DAG decomposition, horizontal fusion, general vertical fusion, fan-out materialization,
  or recognition/profitability work owned by CPU 0008A–0008C.
- CPU 0007 reductions/statistics/normalization; CPU 0008 matrix multiplication, convolution,
  pooling, attention, or loss; native/vendor routes; tuning; benchmarks; or relaxed math.
- Vector affine loads/stores, gather/scatter Vector API operations, masked tails, fixed-shape
  specialization, unrolling, or broadening CPU 0005D materialization.
- Model, Tensor API, Compiler, Config, Planning, shared Prepare, Runtime, Backend Contract, Trace,
  Engine, OpenBLAS provider, another backend, public API, architecture, dependency, Gradle,
  backend-conformance, or integration changes.
- A detailed specification for 0006A or later, a commit, or a push.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Runtime API](../../../../api/runtime-api.md)
- [Public API](../../../../api/public-api.md)
- [Glossary](../../../../glossary.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [Completed CPU 0005A architecture reset](0005a-atomic-partition-kernel-architecture-reset.md)
- [Completed CPU 0005B access and broadcasting](0005b-universal-access-plans-and-right-aligned-broadcasting.md)
- [Completed CPU 0005D materialization gate](0005d-materialization-specialization-and-persistence-evidence-gate.md)
- [Completed CPU 0005E family expansion](0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md)
- [Completed CPU 0005J parity hardening](0005j-bounded-pointwise-coverage-and-parity-hardening.md)
- [Model capability baseline](../../../modules/model/capabilities.md)
- [Model master plan](../../../modules/model/master-plan.md)

## Architecture constraints

- Model owns every operation meaning and descriptor. CPU consumes validated captured semantics; it
  does not normalize Tensor API requests, change provenance, infer a missing layout, or define
  public behavior.
- Planning selects only `BackendId("cpu")`. CPU analysis owns family lowering, metadata folding,
  route/strategy selection, specialization, and exact resource declarations.
- `LogicalMemoryPlan` remains complete. Only CPU analysis decides that an internal view value needs
  no `PreparationResourceRequirement.Buffer`; shared Prepare does not learn view semantics.
- Current shared contracts permit multiple outputs and multiple executable write selections, but
  the current CPU pipeline does not. This task does not change that CPU-private boundary because
  every admitted node has one output. Draft 0006C/0006D must extend it before TOP_K/DROPOUT.
- Current shared contracts do not assign two `ValueId` values to one physical slot or representation.
  Therefore only an internal same-unit view may be computation-free. A final/cross-partition/
  published view result is copied or rejected; the task must not fabricate shared aliasing.
- Runtime hot paths never see `Operation`, `CompiledNode`, graph IR, Shape, layout, route, or
  resource-selection policy. Generated hot paths never dispatch on any of them.
- Portable Class-File bytecode remains the production baseline and the scalar reference remains
  conformance-only. Scalar, vector, parallel-scalar, and parallel-vector reporting must remain
  exact even though this family admits only scalar compute in 0006.
- One partition still produces one `PreparedExecutable`. No general DAG fusion is authorized.
- Any need for a shared physical-layout contract, shared aliasing, another module change, or an
  architecture-rule change is a stop condition.

### Architecture and no-change conclusions

- `ARCHITECTURE.md`, focused architecture explanations, ADRs, and architecture tests need no
  change because ownership, dependency direction, lifecycle placement, and hot-path rules remain
  unchanged.
- Model and Tensor API need no change because all admitted kinds, attributes, Shapes, layouts,
  signatures, and provenance already exist and remain authoritative.
- Compiler needs no change because capture/inference/validation already preserve the exact
  operations, descriptors, and output positions consumed through `PrepareContext`.
- Planning and shared Prepare need no change because logical values remain complete, CPU analysis
  may omit same-unit buffer requirements, and shared assignment already treats the CPU plan
  opaquely.
- Runtime needs no change because one executable with two selected buffers and explicit read/write
  access already fits its cold-binding, validity, schedule, and ownership contracts.
- Backend Contract, Config, Trace, Engine, OpenBLAS, other backends, backend conformance,
  integration tests, and Gradle need no change because the task adds one CPU-private portable
  family with no composition or cross-module behavior.
- The CPU guide, affected Javadocs/package summaries, glossary, task, master plan, and roadmap do
  require the targeted documentation pass because executable CPU capability and internal
  terminology change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — occurrence-local truthful capability.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — portable kernel IR roles and structural
  access/mapping facts.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — family dispatch, affine chain
  validation, mapping composition, and boundary derivation.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — Class-File generation and direct
  primitive carrier/loop emission.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — deterministic strategy, declarations,
  opaque plan retention, and post-assignment finalization.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — selected portable IR and
  specialization facts.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema and structural compatibility.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — cold typed binding and direct
  partition invocation.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — differential reference mapping.

Packages added, removed, or moved:

- None.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPortableKernelIr` — new sealed internal
  structural role implemented by the existing pointwise IR and the new affine IR; it exposes only
  deterministic structural identity needed by the portable route.
- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr` — new immutable composed
  source-to-result mapping and distinct-write-domain contract.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAffineLayoutLowering` — new family
  lowerer because Model-kind interpretation and affine composition belong in lowering, not the
  generator or Runtime.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAffineCopyEmitter` — new
  family-specific bytecode emitter; shared carrier and loop primitives remain in their existing
  owners.
- Existing `CpuKernelSpecialization.CarrierAccess` gains exactly `SHORT_ARRAY`, and existing
  `CpuPreparedExecutable` cold binding recognizes `CpuBufferArgument.Shorts`; no new storage
  representation type is needed because task 0001 already established that exact raw carrier.
- Existing `CpuPartitionLowering` becomes only the small cold family dispatcher plus the retained
  pointwise path. It must not absorb the affine algorithm.

## Affected files

Expected CPU production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAffineCopyEmitter.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuLoopEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAccessPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAffineCopyIr.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAffineLayoutLowering.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`

Expected CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAffineCopyGeneratedKernelTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAffineCopyIrTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAffineLayoutLoweringTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected explanatory documentation and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0006-portable-static-affine-views-and-boundary-materialization.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review-only paths include Model, Compiler, Config, Planning, shared Prepare, Runtime, Backend
Contract, Trace, public API guides, architecture/ADRs/tests, Gradle, backend conformance,
integration tests, native routes, and later CPU tasks.

## Maximum scope

This task may create or modify at most 40 paths:

| Category | Maximum | Accounting |
|---|---:|---|
| CPU production/package | 22 | Exact eighteen existing paths plus four new focused internal types listed above |
| CPU tests | 13 | Exact ten existing tests plus three new focused tests listed above |
| Explanatory documentation | 2 | CPU backend guide and glossary |
| Planning/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **40** | **22 + 13 + 2 + 3** |

The larger-than-normal path count is required because one new kernel family must remain coherent
across capability truth, family lowering, route-neutral IR, Class-File emission, specialization
compatibility, staged finalization, cold binding, reference evidence, and documentation. The
semantic scope remains eight named one-output operations, one straight-line chain form, one source,
one result, one portable route, scalar compute, and no shared-module change. Unused authorized
paths remain unchanged. Adding `SHORT_ARRAY` changes only already-listed specialization, emitter,
executable, reference, and test paths. Advancing the generator schema from 10 to 11 also requires
the existing `CpuGeneratedKernelArtifactStoreTest` current-schema assertion to advance, so the
exact accounting is 22/13/2/3 with a 40-path ceiling. If another path, package, module, operation,
workspace, output, or execution route is required, stop and revise the plan rather than consuming
the ceiling.

## Acceptance criteria

- `CpuCapabilityProvider` returns true only for the exact admitted static/resolved occurrence
  matrix and false for every adjacent kind, wrong attributes, wrong arity, type mismatch, dynamic
  Shape/attribute, unresolved layout, and deferred operation.
- One through eight admitted nodes lower only as one exact straight-line one-input/one-output
  affine chain. Branches, joins, mixed pointwise nodes, fan-out, publication of an intermediate,
  cross-partition intermediate use, and disconnected nodes fail before artifact access.
- Every eligible internal affine result remains present in graph/logical-memory facts but has no
  CPU declaration, Runtime slot, workspace, computation instruction, or store.
- The final source and result are the only buffer declarations, in deterministic order. Final
  boundary work uses one affine IR, one generated class/artifact, one partition executable, and
  one bound invocation.
- Reshape, expand, permutation, singleton-axis insertion/removal, select, every admitted normalized slice,
  target-relative crop, and contiguous mappings match the current Model contracts for every
  admitted resolved-layout case.
- FLOAT64/FLOAT32/BFLOAT16/INT32/INT64/BOOL results preserve exact represented bits across array,
  segment, and mixed source/result carriers. No conversion, promotion, arithmetic, NaN canonicalization,
  signed-zero change, or BOOL truthiness is introduced.
- `CpuKernelSpecialization.CarrierAccess` has exactly seven forms after the task, including the new
  `SHORT_ARRAY`; its method descriptor is `short[]`, its only matching logical heap type is
  BFLOAT16, and cold binding maps the existing `CpuBufferArgument.Shorts` variant to it rather than
  falling through to `MEMORY_SEGMENT`. Wrong type/carrier pairs fail cold.
- BFLOAT16 affine fixtures prove exact raw-bit preservation for `short[]`→`short[]`,
  `short[]`→segment, segment→`short[]`, and segment→segment paths, including payloads representing
  zeros, infinities, and NaNs. Existing BFLOAT16 pointwise capability/lowering rejection remains
  regression-covered; no BFLOAT16 arithmetic, conversion, comparison, numerical, or vector claim
  is introduced.
- Exact offsets, positive/zero strides, accessed spans, scalar and zero-element Shapes, and
  distinct-address output domains are correct. Non-injective output layouts never create parallel
  duplicate writes, and unsafe aliasing fails cold.
- Scalar and parallel-scalar generated execution handle arbitrary legal `start`/`end` ranges.
  Vector preference falls back to scalar compute truthfully; no vector/gather/scatter claim appears.
- Generated hot code has direct typed arguments and contains no Model/graph/Shape/layout/route
  dispatch, map lookup, reflection, storage discovery, allocation, virtual cursor, or per-element
  division/modulo.
- The new sealed IR seam is CPU-private and minimal. Existing pointwise IR, all forty-eight
  opcodes, schema-10 behavior before the planned version advance, four execution-strategy names,
  one-copy materialization policy, and pointwise validation remain regression-covered.
- `CpuGeneratorSchema.CURRENT_VERSION` advances exactly from 10 to 11, and the existing
  `CpuGeneratedKernelArtifactStoreTest` current-schema assertion advances with it.
- Specialization and artifact identity contain every affine code-shaping compatibility fact and
  exclude all instance/runtime identities. Older schemas fail closed without migration; optional
  persistence stays disabled by default.
- No shared contract, public API, architecture, dependency, Gradle, conformance, integration,
  native route, tuning, CPU 0007/0008 family, or CPU 0008A fusion work changes.
- A separate clean documentation-focused pass finalizes affected Javadocs/package summaries, the
  CPU guide, glossary impact, this task evidence, CPU master plan, and roadmap. It reuses the
  implementation pass's successful Java evidence unless it changes executable behavior.
- CPU 0005A–0005J remain `Complete`; 0006 is `Complete` only after implementation and all
  validation passed; 0006A and later remain `Draft` without detailed specifications.

## Tests / validation

Implementation-focused validation:

```bash
./gradlew :backends:cpu:test --tests 'io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIrTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAffineLayoutLoweringTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAffineCopyGeneratedKernelTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest'
./gradlew :backends:cpu:test
```

The final CPU command is the sole normal affected-module Java evidence after executable code
stabilizes. Record suite/test counts, failures, errors, skips, Java 26 toolchain identity, and any
environmental limitation.

Required automated/manual checks include:

- exhaustive admitted/rejected operation/attribute/type/Shape/layout capability rows;
- one- and eight-node chains; intermediate no-declaration/no-slot proof; exactly two boundary
  declarations; and rejection of branch, join, fan-out, publication, mixed-family, and
  cross-partition intermediate cases;
- all eight named semantics, both slice attribute variants where static/resolved, scalar and
  zero-element cases, offsets, positive/zero strides, injective and distinct-address write domains;
- raw-bit fixtures for all six data types, every per-type array/segment pair, representative mixed
  carriers, arbitrary ranges, worker boundaries, and deterministic scalar fallback; BFLOAT16 must
  cover all four `SHORT_ARRAY`/`MEMORY_SEGMENT` source-result combinations and adversarial raw
  16-bit payloads without conversion;
- exact seven-form carrier inventory, `SHORT_ARRAY`→`short[]` entry descriptors, specialization-key
  equality/difference, `CpuBufferArgument.Shorts` cold recognition, wrong BFLOAT16 carrier
  rejection, and unchanged BFLOAT16 pointwise rejection;
- generated/reference differential results and cold alias/read-only/size/alignment/liveness checks;
- structural-key equality/difference, schema invalidation, no migration, one-artifact ceiling,
  the artifact-store current-schema assertion at 11, no default persistence change, and unchanged
  pointwise regression behavior;
- generated-bytecode or automated source-shape checks for forbidden hot-path dependencies. Add a
  stable automated assertion rather than relying on a repeated manual `javap` check.

Documentation-focused pass after executable evidence:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Also validate local Markdown links and anchors, balanced fences, final newlines, trailing
whitespace, the exact changed-path set and 40-path ceiling, canonical task headings, `Complete`
status/dependencies, 0005A–0005J `Complete`, 0006A+ `Draft`, and absence of any later detailed task
specification.

Repository-wide validation is deferred to CPU 0009 portable generated-coverage closure and CI.
No shared module, dependency, build, architecture, conformance, or integration contract changes in
this task, so a root `test` run would duplicate the focused CPU evidence without a recorded risk.

## Dependencies

- Completed CPU 0005A through 0005J, especially the partition reset, access/broadcast contracts,
  execution strategies, materialization/specialization budgets, typed carriers, and parity gates.
- Current Model layout/index operation signatures, static `Shape`, resolved `LayoutDescriptor`,
  typed scalar/carrier, and exact provenance contracts.
- Current Planning logical-memory and maximal same-owner partition contracts.
- Current staged Prepare projection/declaration/assignment/finalization contracts.
- Current Runtime multiple-selection access declarations, cold binding, validity, schedule, and
  per-run resource ownership contracts.

## Follow-up tasks

- CPU 0006A (Draft): non-affine PAD/TILE/composition/unfold, gather families, and one-hot.
- CPU 0006B (Draft): functional slice update, scatter families, and overlap-fold accumulation.
- CPU 0006C (Draft): stable SORT/ARGSORT and two-output TOP_K.
- CPU 0006D (Draft): INITIAL_STATE and three-output explicit-state DROPOUT.
- CPU 0007 (Draft): reductions, scans, statistics, softmax/log-softmax, and normalization after
  0006D.
- CPU 0008A (Draft): general partition-DAG decomposition and bounded fusion after family coverage.

These rows remain master-plan summaries. Do not create their detailed task files while 0006 is
the active implementation frontier.

## Architecture impact

Expected impact: None.

The current shared contracts are sufficient for this one-output slice and, later, for multiple
outputs and state values. The current CPU implementation's single output/final-store assumption is
CPU-private and deliberately left for 0006C/0006D. The absence of shared physical aliasing is not
silently repaired: it bounds computation-free view folding to same-unit internal values and
requires a final boundary copy or fail-closed result.

If implementation requires Runtime to inspect an operation/layout, shared Prepare to interpret an
affine plan, two `ValueId` values to share one slot, a new public physical-layout contract, or any
other architecture change, stop and report the exact decision rather than editing architecture.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are a clean-context implementation agent in the Synaptik repository.

Read completely:
- AGENTS.md
- ARCHITECTURE.md
- docs/planning/planning-guide.md
- docs/planning/backends/cpu/tasks/0006-portable-static-affine-views-and-boundary-materialization.md
- the directly referenced completed CPU and current Model/Planning/Prepare/Runtime contracts

Implement CPU 0006 exactly as specified. Do not implement out-of-scope operations, general DAG
fusion, shared contracts, public APIs, architecture, dependencies, Gradle, or later task specs.
Add exactly the seventh `CarrierAccess` form `SHORT_ARRAY` and use it only for opaque BFLOAT16
affine copying through the existing `CpuBufferArgument.Shorts`; preserve the current BFLOAT16
pointwise capability/lowering rejection and make no BFLOAT16 arithmetic, conversion, numerical,
or vector claim. Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly from 10 to 11 and update the
authorized existing `CpuGeneratedKernelArtifactStoreTest` current-schema assertion accordingly.
Stop if the task requires a shared physical-layout/alias contract or exceeds the exact 40-path map,
including its 13 CPU test paths.
Do not commit or push.

After executable Java stabilizes, run the focused and sole final CPU validation and record exact
evidence. Then hand the task spec, final diff, affected APIs/behavior, architecture constraints,
expected CPU guide/glossary/Javadoc work, and test evidence to a separate clean documentation-
focused agent. That pass must follow docs/developer-guide/documentation-rules.md, finalize the
authorized documentation and Javadocs in the same overall change, reuse successful Java evidence
unless executable behavior changes, run CPU Javadoc and documentation checks, and update this task
with evidence, notes, completion summary, and final status. Mark 0006 Complete only after every
gate passes; keep 0006A and later Draft without specs.
```

## Local decisions

- The former broad 0006 family is split because affine view composition, non-affine/index-tensor
  movement, duplicate-target functional updates, ordering algorithms, and explicit-state random
  execution have different IR, workspace, output, determinism, and validation boundaries.
- Resolved-layout static affine views are the first slice because current Prepare supplies static
  descriptors and current CPU access plans already model positive and zero strides. Unresolved
  physical layout selection is not inferred.
- Raw BFLOAT16 copying is safe within this slice because current storage already represents it as
  `CpuBufferArgument.Shorts` or an exact typed segment. The task adds the missing `SHORT_ARRAY`
  generated carrier form and opaque two-byte load/store only. Storage availability does not make
  BFLOAT16 pointwise arithmetic executable, so all existing numerical capability gates remain
  fail-closed.
- Same-unit internal affine values are the only zero-work form. Current shared slot assignment has
  no alias contract for distinct values, so a final view value is materialized.
- The task admits scalar and parallel-scalar execution only. This preserves the existing four-name
  strategy model and deterministic scalar fallback without claiming Vector API gather behavior.
- Shared multi-output and state contracts are adequate; later CPU tasks need CPU-private IR,
  executable, and emitter extensions rather than architecture changes.

## Known limitations

- Dynamic/symbolic Shapes and Shape-valued attributes remain unsupported by current Prepare.
- Negative-step slices normally have unresolved result layout and therefore remain unsupported in
  this task even though `SliceAttrs` can represent them.
- A mixed affine/pointwise or branched CPU-owned partition may fail CPU preparation until CPU
  0008A generalizes partition decomposition. Occurrence capability does not promise that every
  combination of individually supported nodes is one eligible unit.
- Boundary materialization preserves exact resolved layout geometry and may retain unused prefix
  or holes. This task adds no layout compaction, alias assignment, or representation pooling.
- No performance claim is made. Vector affine execution and broader materialization require later
  evidence and scope.
- BFLOAT16 support is representation-only affine movement. Pointwise arithmetic, conversion,
  comparison, reduction, ordering, random generation, and vector execution remain unsupported
  until their owning tasks establish exact numerical and determinism policies.

## Validation evidence

Planning context `/root`:

- Repository rules, architecture, focused lifecycle/boundary documentation, documentation General
  and Planning profiles, planning guide, roadmap, CPU master/completed-task history, current CPU
  source/test inventory, current Model operation/capability records, relevant public API/glossary
  contracts, and shared Planning/Prepare/Runtime contracts were reviewed before editing.
- A focused carrier audit confirmed six current `CarrierAccess` forms, the existing
  `CpuBufferArgument.Shorts` and BFLOAT16→`short[]` representation mapping, the missing generated
  `SHORT_ARRAY` specialization/binding form, and the current explicit BFLOAT16 pointwise rejection.
  The plan therefore scopes one new seventh carrier form solely to opaque affine copying.
- After the carrier correction, shell validation over the task, CPU master plan, roadmap, Git
  index, and untracked files passed: exact three-path scope; all twenty canonical headings in
  order; task/master `Ready` synchronization; exact 0006→0005J and 0007→0006D dependencies;
  0005A–0005J `Complete`; 0006A–0017 `Draft`; all twenty-nine inventoried layout/indexing/
  ordering/random operation names; exactly seven final carrier forms; no 0006A-or-later task spec;
  and absence of Java, Gradle, `ARCHITECTURE.md`, or `testing/` changes.
- A Ruby local-Markdown check resolved every relative file target and explicit heading anchor in
  the three changed planning files and passed balanced-fence, final-newline, and trailing-whitespace
  checks.
- `git diff --check` passed on the corrected final planning diff.
- Java tests and Javadoc were not run because this planning change modifies no Java or Javadoc.

Implementation context `/root/cpu_0006_impl`:

- The prescribed focused task command passed after implementation stabilized.
- The sole final `./gradlew :backends:cpu:test` passed on OpenJDK 26.0.1+8-34. XML evidence
  recorded 29 suites, 140 tests, zero failures, zero errors, and one skipped opt-in persistence-
  evidence test. No environmental limitation was reported.
- No executable Java changed after that final test evidence. The documentation pass therefore
  reused it as required instead of repeating the successful Java suites.

Documentation-focused context `/root/cpu_0006_docs`:

- Read the repository instructions and architecture contract; the current architecture index;
  planning guide and roadmap; documentation rules and General, API/Javadoc, Backend Guide,
  Planning, and Example profiles; CPU master plan and task 0006; completed CPU 0005A, 0005B,
  0005D, 0005E, and 0005J contracts; every final changed/created CPU production and test source;
  CPU guide and glossary; and directly relevant Model, Compiler, Planning, Prepare, Runtime,
  public API, architecture, build, conformance, integration, and native-route boundaries.
- Independently reviewed the final implementation and test diff. The pass finalized 17 authorized
  production/package Javadocs, the CPU guide, glossary, task record, CPU master plan, and roadmap.
  It documented the exact static/resolved affine chain, internal virtuality, final boundary copy,
  resolved offsets and positive/zero strides, distinct-address write domain, six-type raw-bit
  copying, BFLOAT16 representation-only `SHORT_ARRAY`, scalar/parallel-scalar behavior, vector
  fallback, schema 11, and cold lifecycle/failure boundaries without broadening supported scope.
- The first `./gradlew :backends:cpu:javadoc` attempt failed on a stale renamed `@param` and
  exposed incomplete new-type comments. After Javadoc-only correction, the required rerun passed
  with only the two expected incubating-Vector warnings. Inspection of generated pages confirmed
  rendered affine-copy, distinct-address, `SHORT_ARRAY`, portable-IR, route-plan, and executable
  contracts. No Java test was rerun because no executable behavior changed.
- The first three Ruby Markdown-check attempts failed in the checker itself: a Ruby regular-
  expression interpolation error, an incorrect `Dir.glob` invocation, and an incorrect GitHub
  duplicate-space slug rule. The corrected repository-local check passed 321 Markdown files,
  5,478 local links, 354 explicit heading
  anchors, balanced fences, final newlines, and trailing whitespace.
- Exact path-map validation passed for 34 changed paths within the exact 40-path map: 20 of 22
  authorized production/package paths, 9 of 13 authorized tests, both explanatory documents, and
  all 3 planning/status paths. Canonical headings, dependency/status synchronization, schema 11,
  exactly seven carrier forms, CPU 0005A–0005J `Complete`, CPU 0006A–0017 `Draft`, and absence of
  any later detailed task specification passed.
- `git diff --check` passed on the final combined change.
- Architecture and focused architecture pages, ADRs, architecture tests, public Tensor/Compile/
  Runtime/Training APIs, Model capabilities/master plan, shared modules, Gradle, backend
  conformance, integration tests, OpenBLAS/native routes, and other backends remain unchanged.
  The reason is specific: CPU 0006 consumes existing Model semantics and staged shared contracts
  entirely inside one CPU-private route, changes no dependency or public/shared contract, and adds
  no composed Engine or cross-backend behavior.

## Implementation notes

- Added a sealed CPU-private portable-IR role so the existing pointwise IR and the new affine-copy
  IR share only structural identity. Affine structure encodes mapping family and write-domain
  compatibility while exact composed addresses remain cold plan facts.
- Added a dedicated affine lowerer for one-through-eight connected static resolved-layout
  occurrences. Internal single-use values retain graph/logical-memory identity but create no CPU
  declaration or Runtime slot; source and final result are the only declared buffers.
- Final result materialization preserves the resolved layout. Repeated zero-stride coordinates are
  deduplicated into deterministic distinct-address writes only when they select one source value;
  ambiguous repetitions fail closed. Scalar and zero-element domains remain exact.
- Added scalar generated/reference represented-bit copying for FLOAT64, FLOAT32, BFLOAT16, INT32,
  INT64, and BOOL across array, segment, and mixed carriers. `SHORT_ARRAY` binds only existing raw
  BFLOAT16 `CpuBufferArgument.Shorts`; pointwise BFLOAT16 remains rejected.
- Preserved scalar/vector and single/parallel strategy vocabulary. Affine work selects scalar or
  parallel-scalar, and vector preference falls back deterministically without gather/scatter or
  masked-tail claims. Schema 11 invalidates older generated artifacts without migration.

## Completion summary

- Completed changes: implemented and documented bounded portable static affine view composition,
  CPU-private internal virtuality, and exact final boundary represented-bit materialization.
- Files changed or created: 34 authorized paths total—20 CPU production/package paths, 9 CPU test
  paths, 2 explanatory documentation paths, and 3 planning/status paths. The documentation pass
  finalized 17 of those production/package Javadocs plus all 5 authorized Markdown records.
- Tests and validation: reused the stabilized focused pass and sole final CPU 29-suite/140-test
  evidence; final CPU Javadoc, generated-page inspection, Markdown links/anchors/fences/newlines/
  whitespace, exact path map and ceiling, semantic/schema/status gates, and `git diff --check`
  passed.
- Documentation-agent review: clean context `/root/cpu_0006_docs`; complete with no executable
  behavior or test changes.
- Documentation impact: CPU guide, glossary, task, CPU master plan, and roadmap now describe the
  implemented scope and boundary accurately.
- Javadoc review: affected Javadocs and package summaries were finalized; authorized unchanged
  source paths retained accurate contracts and required no edit.
- Glossary impact: added the reusable CPU affine boundary-copy distinction and synchronized
  portable route, generated kernel, specialization, artifact, access, materialization, and
  virtual-intermediate terminology.
- Unresolved issues: None.
- Follow-up required: None for task 0006. CPU 0006A–0006D and later Draft rows retain all excluded
  non-affine/index/scatter/order/random, dynamic-layout, vector-affine, and general-DAG work.

Status: Complete
