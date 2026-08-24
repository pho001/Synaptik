# Task 0007A2: Portable Binding-Aware Sum-to-Shape Reduction

## Status

Complete

## Goal

Implement portable generated CPU execution for exactly one fully bound, fully static,
resolved-layout Model `AggregateReductionKind.SUM` occurrence carrying exact
`SumToShapeAttrs`. CPU analysis must prove the Model's right-aligned target-one-or-source-equal
obligation, derive leading and aligned reduction geometry, and execute all five current numeric
types through direct generated scalar or parallel-scalar hot loops.

The implementation reuses CPU 0007A1's exact floating SUM state, one-result-format rounding,
integral modular addition, output-cell ownership, workspace lifecycle, and direct optimal Java
oracle. It adds no dynamic-Shape execution or second reduction architecture.

```text
bound source Shape + exact bound target Shape
  -> prove target rank <= source rank and every aligned pair is equal or target one
  -> reduce every leading source axis
  -> preserve equal aligned axes; reduce and retain target-one aligned axes
  -> direct generated copy when no axis is reduced, otherwise direct generated SUM
```

## Scope

### Exact Model form and bound validation

- Admit only `AggregateReductionKind.SUM` with exactly one input, one output, and
  `SumToShapeAttrs`.
- Admit `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, and `INT64`; reject `BOOL` and every type,
  kind, arity, or attributes combination outside this matrix before route selection.
- Require fully static input, target, and output Shapes and resolved input/output layouts. This is
  the existing Prepare boundary: unresolved Model Dimensions must already have been bound and
  proved before CPU analysis. A dynamic or symbolic descriptor reaching CPU fails closed; CPU,
  generated code, and Runtime do not bind Dimensions.
- Require the output Shape to equal the exact bound `SumToShapeAttrs.targetShape()` and retain the
  input data type. Require target rank not to exceed input rank.
- Let `leading = inputRank - targetRank`. Every input axis below `leading` is reduced and removed.
  For target axis `t`, aligned input axis `i = leading + t` is handled in this exact order:
  equal bound extents preserve the coordinate without reduction; otherwise target extent one
  reduces input axis `i` and retains the target position; every other pair is rejected before
  finalization or execution.
- Canonical selected-axis membership is increasing input-axis order: all leading axes followed by
  aligned target-one axes whose source extent differs from one. Equality takes precedence, so an
  aligned source-one/target-one pair is preserved rather than treated as an arithmetic reduction.
- Equal bound input and target Shapes therefore select no reduction axes but still describe one
  explicit SUM occurrence with a distinct materialized output.

### Leading and aligned geometry

- Extend the existing `CpuAggregateIr`/`CpuAggregateLowering` family in place with a closed
  `SUM_TO_SHAPE` form and exact target-alignment facts. Do not create `CpuSumToShapeIr`, a second
  access planner, a generic reduction plan, or a runtime binding abstraction.
- Record enough cold geometry to map each flattened target coordinate to its right-aligned source
  coordinates and to visit the Cartesian product of selected leading/aligned axes in canonical
  row-major input-coordinate order. The mapping must not depend on `keepDimensions`, which is not
  a `SumToShapeAttrs` property.
- `outputCount` is the exact target element count. `domainCount` is the checked product of the
  selected source extents, or one when no axis is selected. Geometry and byte arithmetic use
  checked `long` operations and fail deterministically during capability/lowering/preparation.
- A selected zero extent makes `domainCount == 0`; every existing target coordinate receives the
  SUM empty identity. An equal aligned zero/zero pair is preserved and produces zero output
  coordinates. A zero extent on an unselected axis likewise makes `outputCount == 0`, so no write,
  workspace allocation, generated call, or worker submission occurs.
- Scalar input accepts only scalar target and follows the no-reduction copy path. Scalar target
  reduces every axis of a non-scalar input.
- Support every currently legal aggregate input/output layout and carrier combination: exact
  typed heap arrays, `MemorySegment`, and mixed input/output carriers; nonzero offsets;
  non-negative positive or zero read strides; arbitrary injective output layouts; and arbitrary
  valid half-open output-cell ranges. Do not narrow the established resolved-layout contract.

### Numerical, copy, and deterministic execution rules

- When at least one axis is actually reduced, inherit CPU 0007A1 SUM exactly:
  - floating finite values use the fixed-width signed superaccumulator in least-subnormal units
    and one round-to-nearest, ties-to-even conversion to the result type;
  - floating NaN, infinity, signed-zero, subnormal, overflow, underflow, and empty-domain behavior
    remains the schema-29 ordinary SUM contract;
  - `INT32` and `INT64` use same-width modular addition with zero identity.
- Reuse `CpuExactSumEmitter` as the sole generation-time exact floating accumulation/conversion
  owner. Do not duplicate its algorithm or add a runtime numerical helper.
- When no axis is reduced, copy the corresponding represented input value exactly. This includes
  raw floating NaN payload/sign/signaling bits and signed zero; it must not pass one value through
  SUM classification or rounding. The direct generated copy is still part of the aggregate
  artifact and must not call the affine-copy bridge or another prepared executable.
- Scalar and parallel-scalar execution partition only disjoint complete output-cell ranges. One
  worker owns the complete reduction domain and all mutable state for every output cell it writes.
  Never split or merge one reduction domain. Results are bit-identical across worker counts,
  chunking, repeated invocation, and concurrent reuse of one immutable prepared executable with
  distinct `RunState` instances.
- Generated dense heap forms use cold-proved primitive integer loop/address state where safe;
  every other supported layout/carrier/range retains a direct typed long-address fallback. Both
  must preserve the clean optimal Java algorithm and dataflow shape.

### Overlap, resources, and lifecycle

- Declare exactly one distinct input buffer and one distinct injective output buffer. Declare no
  materialization.
- Reject complete physical input/output overlap before any output mutation, including equal-Shape
  copy cases. Preserve carrier compatibility, liveness, writability, referenced-span, alignment,
  output-injectivity, and arbitrary-range validation at the established cold binding boundary.
- Floating rows with at least one selected reduction axis reuse workspace purpose
  `AGGREGATE_EXACT_STATE`, requirement ID zero, alignment eight, and the exact CPU 0007A1 SUM
  slice layout `8 + 8 * sumLimbCount` bytes. Limb count is derived from the bound `domainCount`.
  Analysis declares one disjoint slice per simultaneously executable output range before shared
  slot assignment and enforces `maximumAdditionalBytes` exactly.
- Integral reductions, all no-reduction represented-bit copies, and every zero-output occurrence
  declare zero workspace. A selected empty floating domain with positive output count still needs
  exact-state slices because the generated SUM path owns initialization and finalization.
- The immutable prepared executable captures geometry, specialization, assigned slot identities,
  artifact, handle, and worker policy only. It captures no writable accumulator, Arena, carrier,
  address, or run-owned workspace. Runtime continues to own allocation, isolation, cleanup, and
  publication.
- Complete buffer and scratch presence/size/alignment/access/physical-overlap validation before
  the first write. Failure mutates no output; preparation failure publishes no executable.

### Generated hot path and schema 43

- Extend the existing aggregate generated entry shapes:
  - reduction, integral: `(inputCarrier, outputCarrier, long[] geometry, long start, long end)void`;
  - reduction, floating: `(inputCarrier, outputCarrier, MemorySegment scratch, long[] geometry,
    long start, long end)void`;
  - no-reduction copy, all five types: the workspace-free integral-shaped entry.
- Input and output carriers are independently the exact matching `[D`, `[F`, `[S`, `[I`, `[J`,
  or `MemorySegment` form. No entry admits `Object`.
- Generated loops directly contain target-coordinate mapping, selected-axis traversal, typed
  loads/stores, modular addition or exact-sum limb work, and final conversion. Hot code contains no
  Model/Compiler/Runtime object, `Operation`, attributes, graph inspection, bridge to
  `CpuScalarReferenceKernel`, `CpuExactSumEmitter`, affine copy, method handle, collection,
  boxing, reflection, allocation, string/map lookup, route/type/form/carrier/layout switch,
  per-element virtual dispatch, or avoidable division/modulo.
- Advance `CpuGeneratorSchema.CURRENT_VERSION` from 42 to 43. This task changes generated bytes by
  adding the SUM-to-Shape form, target-alignment mapping, selected-axis traversal, and exact
  represented-copy body, and adds new form/mapping/resource facts to structural compatibility.
  Current-only compatibility therefore treats every schema-42 envelope as an incompatible safe
  miss; no migration reader or partial reuse is added. Existing schema-42 specializations must
  regenerate with unchanged semantics, and representative pre-A2 operations must be verified
  against preserved schema-42 bytes to detect unintended code-shape drift. The A1O ledger
  resource remains immutable historical evidence with `generated-schema=42`; its test must prove
  that this historical metadata is compatible evidence while independently requiring the current
  `CpuGeneratorSchema.CURRENT_VERSION` to be exactly 43.
- Include form, ranks/alignment, selected-axis membership, reduced-versus-copy mode, exact-state
  shape, access regimes, carriers, and every other byte/resource-changing fact in IR,
  specialization, entry-descriptor, manifest, and cache compatibility. Concrete carriers,
  offsets, addresses, slots, ranges, worker count/chunk size, graph identities, and run identity
  remain cold when they do not change emitted code.

### Correctness, code-shape, and performance evidence

- Add an independent test-only oracle that computes the right-aligned mapping itself. Reduced
  floating cases use the existing independent `BigInteger`/rational SUM oracle without sharing
  emitter logic; no-reduction cases compare raw represented bits directly.
- Cover all five types; scalar and lower-rank targets; leading-only, aligned-only, combined
  leading/aligned, equal-Shape/no-reduction, selected/unselected zero extents; dense/general
  layouts; offsets; positive and zero read strides; injective strided output; heap/segment/mixed
  carriers; partial/empty ranges; scalar/parallel worker counts; repeated/concurrent reuse;
  overlap/resource failures; cache hit/miss/corruption; and schema-42 safe misses.
- Floating adversarial vectors cover cancellation, both halfway parities, finite overflow,
  normal/subnormal boundaries, signed underflow, smallest subnormals, NaN classes, infinities,
  every zero-sign case, and empty domains. Integral vectors cover modular wrap and empty domains.
- Retain Class-File and `javap -c/-v` evidence for at least FLOAT64 combined leading/aligned
  reduction, BFLOAT16 general mixed-carrier reduction, INT64 aligned reduction, and FLOAT32
  equal-Shape copy. Assert exact descriptors, deterministic bytes, allowed member references,
  direct loop ownership, schema-43 compatibility, and forbidden helper/object leakage.
- Use a clean direct primitive Java baseline with the same mapping, exact floating state layout,
  reset, decoding, rounding, carrier work, and output-cell traversal. Measure these six dense
  heap-array cases on source `[64, 128, 256]` to target `[128, 1]`: reduced SUM for all five
  numeric types, plus FLOAT32 equal-Shape represented copy on `[1,048,576]`. Verify raw bits before
  timing. Run five isolated JVM forks, five warmup rounds, nine randomized measured rounds, adaptive
  batches of at least 25 ms, and `-Xms1g -Xmx1g`. Every fork and median-of-fork-medians
  generated/direct ratio must be `<= 1.15x`. This evidence is a release gate, never a tuning input.

## Out of scope

- Unresolved/dynamic Shape execution, a shared binding API, runtime Dimension values, or late
  route/resource selection.
- Ordinary full/single-/multi-axis or masked SUM changes; MEAN/PROD target-Shape forms; BOOL;
  arg-extrema; later reduction, softmax, or normalization families.
- Fusion, vector reduction, domain splitting/combination, native/OpenBLAS/vendor routes,
  measurement-driven preparation, tuning, or cache mutation outside existing cold artifact reuse.
- New Model/Compiler/Planning/Prepare/Runtime/Engine APIs, semantic changes, dependencies, Gradle
  changes, architecture contracts/ADRs/tests, another backend, conformance, or integration work.
- A second IR/lowering/emitter/resource hierarchy, generic helper/manager/service, public CPU API,
  or unrelated refactoring.
- Repair or reinterpretation of CPU 0007A1D's historical `Review needed` result.
- Editing or regenerating A1O's `operation-family-form-ledger-v2.tsv`; its
  `generated-schema=42` metadata records historical generated evidence and is not a current-schema
  assertion.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current-architecture-plan.md`](../../../../architecture/current-architecture-plan.md)
- [`performance-evidence-and-tuning.md`](../../../../architecture/performance-evidence-and-tuning.md)
- [`runtime-prepare-backend-boundary.md`](../../../../architecture/runtime-prepare-backend-boundary.md)
- [`planning-guide.md`](../../../planning-guide.md)
- [`documentation-rules.md`](../../../../developer-guide/documentation-rules.md)
- [`general documentation profile`](../../../../developer-guide/documentation/general-style.md)
- [`backend-guide documentation profile`](../../../../developer-guide/documentation/backend-guide-style.md)
- [`API/Javadoc documentation profile`](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [`planning documentation profile`](../../../../developer-guide/documentation/planning-style.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0005B`](0005b-universal-access-plans-and-right-aligned-broadcasting.md)
- [`CPU 0007A`](0007a-portable-ordinary-extrema-and-boolean-reductions.md)
- [`CPU 0007A0`](0007a0-generated-hot-path-parity-correction.md)
- [`CPU 0007A1`](0007a1-portable-ordinary-numerical-aggregate-reductions.md)
- [`CPU 0007A1A`](0007a1a-generated-scalar-body-self-containment.md)
- [`CPU 0007A1H`](0007a1h-numerical-aggregate-residual-parity.md)
- [`CPU 0007A1N`](0007a1n-multi-axis-min-residual-parity.md)
- [`CPU 0007A1O`](0007a1o-pointwise-ledger-evidence-reconciliation.md)
- [`Model 0023A`](../../../modules/model/tasks/0023a-binding-aware-sum-to-shape.md)
- [`Compiler 0005B`](../../../modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)

## Architecture constraints

- Model owns SUM/`SumToShapeAttrs` meaning, type and Shape semantics. Compiler owns retained
  binding constraints and proof. CPU consumes only the fully resolved occurrence during backend
  analysis and owns lowering, route choice, exact resources, specialization, generation, and
  execution.
- Shared Prepare remains CPU-blind: CPU analysis declares every buffer/workspace requirement
  before assignment; finalization resolves assignments and constructs the artifact without
  changing route or resources.
- Runtime executes one immutable prepared recipe with isolated per-run resources and performs no
  graph inspection, semantic dispatch, route selection, generation, or cache work.
- Direct generated bytecode must preserve the optimal clean Java semantic algorithm, hot-loop and
  dataflow shape, and avoidable-overhead profile. Any deviation requires a recorded technical
  reason and evidence.
- Preserve existing public API, module dependencies, and architecture. If a shared contract or
  architecture change proves necessary, stop before editing and request an explicit decision.

## Package impact

Existing packages used or changed:

- `io.github.pho001.synaptik.backend.cpu` — truthful occurrence-local capability and package status.
- `internal.ir` — existing aggregate IR gains the closed target-Shape form and mapping identity.
- `internal.lowering` — existing aggregate lowerer owns bound validation and geometry derivation.
- `internal.codegen.emit` — existing aggregate emitter owns direct mapping/copy/SUM bodies and
  reuses the existing exact-sum generation owner.
- `internal.cache` — schema 43 and exact specialization/artifact compatibility.
- `internal.prepare` and `internal.route.portable` — existing resource and selected-plan flow.
- `internal.executable` — existing cold binding, overlap validation, worker ranges, and invocation.
- `internal.reference` — independent conformance realization only.

No package is added, moved, or renamed.

Type placement:

- `CpuAggregateIr` — add `Form.SUM_TO_SHAPE` and its exact structural alignment/reduction facts;
  this existing type owns aggregate generated identity.
- `CpuAggregateLowering` and nested `Geometry` — own right-aligned bound validation, selected-axis
  derivation, target-coordinate mapping, counts, scratch shape, and packed primitive geometry.
- `CpuAggregateEmitter` — owns the direct target mapping, represented-copy, modular SUM, and exact
  floating SUM bytecode; `CpuExactSumEmitter` remains the only exact-sum limb/rounding emitter.
- `CpuCapabilityProvider` — advertises only valid fully static resolved occurrences.
- `CpuPartitionPreparationPlan`, `CpuPartitionPreparer`, `CpuPartitionFinalizer`, and
  `CpuPreparedExecutable` — retain the existing aggregate plan/resource/binding/execution roles;
  no new type is planned.
- `CpuGeneratorSchema` and `CpuKernelSpecialization` — own schema-43 compatibility and exact entry
  shape; no new cache type is planned.

## Affected files

Expected production and package contracts (20):

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuExactSumEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAggregateIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAggregateLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`

Expected tests (12):

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseLedgerEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAggregateIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAggregateLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected explanatory/planning paths (5):

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review `CpuClassFileKernelGenerator`, `CpuGeneratedKernel`, route-portable package Javadoc,
Model/Compiler SUM-to-Shape contracts, shared Prepare/Runtime APIs, architecture tests,
backend-conformance, integration tests, and Java 26 Gradle configuration. Change them only if an
acceptance criterion cannot otherwise be met and the task is first updated within its ceiling.

## Maximum scope

This task may create or modify at most 37 paths: the 20 production/package, 12 test, and five
documentation/planning paths listed above. It adds no production type or package. Unused allowance
is not permission for unrelated edits.

Stop before path 38, a new production type, a new public/technically-public API, another module,
another backend, a shared contract, an architecture/build change, or any later task. Update this
specification and request direction if current code proves the in-place aggregate extension
insufficient.

## Acceptance criteria

- Exact one-node SUM/`SumToShapeAttrs` support for all five numeric types; every other new
  combination remains fail closed.
- Fully static/resolved input, exact attrs target, and output agree; right-aligned rank and every
  bound pair are validated before route selection. Equality precedes target-one reduction.
- Leading, aligned-retained, aligned-preserved, scalar, equal-Shape, zero-extent, and empty-domain
  geometry matches the exact rules above with checked counts and no unresolved execution.
- Existing `CpuAggregateIr`/lowering/emitter/resource flow is extended in place; no parallel
  architecture, generic helper, or new production type appears.
- All five types preserve exact result type. Actual reductions reuse CPU 0007A1 SUM numerical and
  empty semantics; no-reduction occurrences preserve raw represented bits exactly.
- Dense integer-address and general long-address generated bodies cover every supported
  layout/carrier/range. Scalar and parallel-scalar results are bit-identical and one domain is
  never split.
- Exactly input/output buffers are declared, output is distinct and injective, no materialization
  exists, all physical overlap is rejected before mutation, and zero-output work declares no
  workspace or generated call.
- Floating actual reductions declare exact pre-assignment per-range run-owned state; integral and
  no-reduction copy cases declare zero workspace. Scratch isolation, validation, cleanup, repeated
  use, and concurrent use are proved.
- Schema advances exactly 42 to 43 for the planned generated-byte and compatibility changes;
  schema-42/corrupt artifacts are safe misses and no legacy reader exists. The A1O ledger resource
  retains its historical `generated-schema=42` metadata unchanged, while its integration test
  proves that schema 42 remains compatible historical evidence and separately asserts that the
  current `CpuGeneratorSchema.CURRENT_VERSION` is exactly 43.
- Generated descriptors, deterministic bytes, structural keys, manifests, allowed member
  references, and forbidden bridge/object/allocation/dispatch rules pass for the selected evidence
  matrix. Existing representative schema-42 semantic/code-shape controls do not regress.
- Independent semantic/oracle tests and all six five-fork `<= 1.15x` performance cases pass.
- A separate clean documentation-focused pass finalizes every affected Javadoc/package summary,
  CPU guide, glossary impact, task evidence, master plan, and roadmap in the same overall change.
- Documentation explicitly reviews and records no-change conclusions where correct for Model,
  Compiler, Planning, Prepare, Runtime, public APIs, architecture/ADRs/tests, other backends,
  conformance/integration, Gradle/build, and later tasks.
- CPU 0007A2 alone is `In progress` during implementation and becomes `Complete` only after every
  gate. No CPU task is `Ready`; CPU 0007A1D remains `Review needed`; completed A1 tasks remain
  unchanged; CPU 0007B and later remain `Draft` without detailed specifications.

## Tests / validation

Focused implementation command:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.CpuInternalPackageInventoryTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseLedgerEvidenceTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest
```

After executable Java stabilizes, run exactly one final CPU suite:

```bash
./gradlew :backends:cpu:test
```

Generated-code and performance evidence:

```bash
for classfile in <fresh-evidence-root>/generated-classes/*.class; do
  javap -c -p "$classfile"
  javap -v -p "$classfile"
  sha256sum "$classfile"
done
for fork in 1 2 3 4 5; do
  java --add-modules jdk.incubator.vector -Xms1g -Xmx1g \
    -cp <fresh-evidence-root>/classes:<required-test-and-main-classpath> \
    SumToShapeGeneratedDirectBenchmark
done
```

Record the fresh evidence root, exact compile/classpath commands, source and artifact checksums,
JDK/OS/CPU metadata, raw fork output, per-fork ratios, median-of-fork-medians, rejected samples,
descriptors, member references, and complete `javap` output. Do not reuse A1 performance results as
A2 evidence.

Documentation pass:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Also validate local Markdown links and explicit anchors, balanced fences, final newlines, trailing
whitespace, exact 37-path ceiling, package/type placement, current schema 43, unchanged historical
`generated-schema=42` ledger metadata, the ledger test's separate historical/current schema
assertions, task/master/roadmap status, unchanged CPU 0007A1D and completed A1 records, no later
task specification, empty staging area, and preservation of unrelated concurrent work.

Repository-wide validation is deferred to CPU 0009 and CI. Architecture, backend-conformance, and
integration suites are not run unless implementation changes a corresponding contract; such a
need is a stop condition, not implied scope.

## Dependencies

- Complete CPU 0005A–0005J, especially 0005B access/carrier/binding contracts.
- Complete CPU 0007A, 0007A0, and 0007A1 exact aggregate semantics, direct generated bodies,
  resource lifecycle, oracle, and performance boundary.
- Complete corrective CPU 0007A1A–0007A1C and 0007A1E–0007A1O accumulated schema-42 evidence.
  CPU 0007A1D remains historical `Review needed` and is not a dependency gate.
- Complete Model 0023A `SUM`/`SumToShapeAttrs` semantics and Compiler 0005B binding-constraint
  adoption.
- Existing static-shape Prepare projection, staged CPU analysis/declaration/assignment/finalization,
  Runtime cold binding/run-state ownership, and Java 26 Class-File toolchain.

All required dependencies are complete.

## Follow-up tasks

- CPU 0007B remains the next Draft row after A2 for arg-extrema coverage. Do not create its
  detailed specification in this task.
- CPU 0007C–0007F and later CPU work remain ordered Draft rows.
- Dynamic/symbolic CPU execution requires a future explicit shared exact-binding contract and is
  not implied by this task.
- CPU 0009 remains the portable generated-coverage capability checkpoint.

## Architecture impact

Expected impact: None.

This task implements a current Model semantic through existing CPU-private boundaries. If work
requires a new shared binding/resource contract, dependency direction, public surface, module
owner, architecture rule, or another package/type architecture, stop and report the exact conflict
before editing.

## Implementation prompt

Use this prompt in a separate clean-context task/thread:

```text
You are working in /Users/phujka/IdeaProjects/Synaptik. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md,
docs/planning/backends/cpu/master-plan.md, this CPU 0007A2 task in full, its linked completed CPU
tasks, and the current Model/Compiler SUM-to-Shape plus CPU aggregate/cache/resource contracts.

Implement CPU 0007A2 exactly within its 37-path ceiling. Preserve all concurrent user work. Extend
the existing aggregate pipeline in place, advance generated compatibility exactly to schema 43,
preserve the A1O ledger resource's historical `generated-schema=42` metadata, and update its now
permitted integration test to distinguish compatible historical schema evidence from exact current
schema 43. Preserve bytecode-first optimal direct-Java parity. Do not implement dynamic binding, later
tasks, shared/architecture/build changes, another IR hierarchy, or unrelated refactors. Stop and
report any architecture, dependency, scope, type-placement, or acceptance conflict.

Run the focused command, one final CPU suite, fresh Class-File/semantic/performance evidence, and
the exact static checks. Then hand the stable diff and evidence to a separate clean-context
documentation-focused agent. That pass must follow docs/developer-guide/documentation-rules.md,
inspect final source/tests/evidence, finalize affected Javadocs/package summaries, CPU guide,
glossary impact, this task, master plan, and roadmap, and reuse successful Java evidence unless it
changes executable behavior or records a concrete reason. Do not mark Complete until all gates pass.
```

## Local decisions

- Extend the existing aggregate IR/lowering/emitter/resource path with `Form.SUM_TO_SHAPE`; a
  second family would duplicate exact SUM state and lifecycle ownership.
- Equal aligned extents take precedence over target-one. This preserves the Model rule that a
  coordinate with no reduced axis is the corresponding input value, including exact represented
  bits for source-one/target-one.
- A no-reduction occurrence is a direct represented-bit aggregate copy with zero workspace. It
  remains one explicit distinct-output SUM artifact rather than an identity rewrite or affine
  prepared unit.
- Output-cell-only partitioning preserves scalar/parallel bit identity and avoids any new
  partial/combine resource contract.
- Schema 43 is required because A2 deliberately adds new generated form/mapping/copy bytes and new
  cache-visible structural/resource facts. A1O stayed at 42 precisely because it changed no
  production bytes; that rationale does not apply here.
- The A1O ledger resource's `generated-schema=42` fact describes the historical generated corpus,
  not the current generator version. A2 therefore permits
  `CpuPointwiseLedgerEvidenceTest.java` as its twelfth integration-test path and requires that test
  to validate historical schema-42 compatibility evidence separately from the exact current
  schema-43 assertion. The ledger resource and A1O history remain unchanged.
- No architecture, shared contract, conformance, integration, or build change is expected.

## Known limitations

- Only fully static, already-bound Shapes with resolved layouts execute.
- One reduction domain remains serial; parallelism is only across complete output cells.
- The portable route remains scalar or parallel-scalar for this family; no vector or native route
  is claimed.
- General layouts/carriers receive semantic and structural evidence, while comparative performance
  is limited to the six fixed dense cases.
- No claim extends to masked, advanced, softmax, normalization, or later families.

## Validation evidence

Implementation context `01a03437-bb19-7d30-ae22-5c599a677e6a` completed executable work on
2026-08-24. The corrected focused command from this task passed after the final emitter change:

```text
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest --tests io.github.pho001.synaptik.backend.cpu.CpuInternalPackageInventoryTest --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseLedgerEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIrTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest
BUILD SUCCESSFUL in 1s; 22 actionable tasks, one executed and 21 up-to-date.
```

The single final CPU suite was then run exactly once after executable stabilization:

```text
./gradlew :backends:cpu:test
BUILD SUCCESSFUL in 3s; 22 actionable tasks, one executed and 21 up-to-date.
```

Development-focused aggregate, lowering, IR, capability, generated-kernel, preparation,
executable, cache, and ledger selections also passed. The pre-correction ledger run that required
current schema 42 failed as recorded previously; the corrected test now independently requires
historical ledger schema 42, current generator schema 43, and `historical <= current`.

Fresh Class-File and performance evidence root:

```text
/private/tmp/synaptik-cpu-0007a2-elMsQh5P
```

Evidence was compiled and generated with:

```text
A2_CP=$(find . -type d -path '*/build/classes/java/*' | sort | paste -sd ':' -)
javac --add-modules jdk.incubator.vector -cp "$A2_CP" -d /private/tmp/synaptik-cpu-0007a2-elMsQh5P/classes backends/cpu/build/a2-evidence-src/SumToShapeEvidenceFactory.java backends/cpu/build/a2-evidence-src/SumToShapeGeneratedDirectBenchmark.java
javac -d /private/tmp/synaptik-cpu-0007a2-elMsQh5P/classes backends/cpu/build/a2-evidence-src/SumToShapeIntegralOracle.java
java --add-modules jdk.incubator.vector -cp "/private/tmp/synaptik-cpu-0007a2-elMsQh5P/classes:$A2_CP" io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.SumToShapeEvidenceFactory /private/tmp/synaptik-cpu-0007a2-elMsQh5P/generated-classes
for classfile in /private/tmp/synaptik-cpu-0007a2-elMsQh5P/generated-classes/*.class; do javap -c -p "$classfile"; javap -v -p "$classfile"; shasum -a 256 "$classfile"; done
```

The four required artifacts and two integral benchmark-shape diagnostic artifacts are deterministic,
schema 43, and have these exact descriptors and SHA-256 values:

| Artifact | Descriptor | SHA-256 |
| --- | --- | --- |
| FLOAT64 combined | `([D[DLjava/lang/foreign/MemorySegment;[JJJ)V` | `ebcf8cf5a554452059b9383cf4606fb28f80abf1df4fae0c0f0d6298e2ee20c9` |
| BFLOAT16 general mixed | `(Ljava/lang/foreign/MemorySegment;[SLjava/lang/foreign/MemorySegment;[JJJ)V` | `4856feb45463c907b3a9201c64ca7951113403ec2a20b963bc69fd1f4a2f71c4` |
| INT64 aligned | `([J[J[JJJ)V` | `3538405662a45b785f2fadd047beaa7cc29bd74baeab3708d9800cc47c81c9d4` |
| FLOAT32 represented copy | `([F[F[JJJ)V` | `d8704e5a2e91b9f6a8d0159dcb422f3239e43f5f59a1502d26349b9686b256b7` |
| INT32 benchmark | `([I[I[JJJ)V` | `0e060ec1ab3fe52965dd9801455ed4af6dfdcde919f89fc505d05a3453f51d2f` |
| INT64 benchmark | `([J[J[JJJ)V` | `d7c5aa66ecdc49c48be04f1be1d0a099cdaa6169e25a9d7b62833d20da8756b5` |

Complete `javap -c/-v` output is adjacent to every class. Integral benchmark decompilation was
compared directly with `javac` output for the original fast one-accumulator Java oracle. The
performance defect was a generated bottom-tested inner counted loop: a direct diagnostic measured
the otherwise equivalent pre-tested variants at about 154–161 microseconds and the bottom-tested
variant at about 586 microseconds. The generated integral body now uses the same pre-tested loop,
single same-width accumulator, affine `base + inner` address, one load/add per element, and one
store per cell as the faster Java oracle. An interim four-accumulator direct comparator was
rejected because it slowed the clean-Java oracle; it is not accepted evidence. The integral artifacts
have no member references. Floating references are limited to typed `MemorySegment` state/access,
native-order layout where required, raw-bit conversion, `Long.compareUnsigned`, and
`Long.numberOfLeadingZeros`; no generated artifact contains a Synaptik helper, bridge, allocation,
boxing, reflection, collection, `Object` entry, bootstrap, or runtime semantic dispatch.

The accepted performance command was:

```text
for fork in 1 2 3 4 5; do java --add-modules jdk.incubator.vector -Xms1g -Xmx1g -cp "/private/tmp/synaptik-cpu-0007a2-elMsQh5P/classes:$A2_CP" io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.SumToShapeGeneratedDirectBenchmark; done
```

It used the original optimal simple nested integral comparator with one accumulator. Every fork and
median-of-fork-medians ratio passed `<= 1.15x`:

| Case | Fork 1 | Fork 2 | Fork 3 | Fork 4 | Fork 5 | Median-of-fork-medians |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FLOAT64 SUM | 0.907139379 | 0.922482808 | 0.909809078 | 0.919686346 | 0.912718408 | 0.915170586 |
| FLOAT32 SUM | 0.370789029 | 0.376636497 | 0.364385761 | 0.360382198 | 0.346378024 | 0.370789029 |
| BFLOAT16 SUM | 0.395161815 | 0.376955463 | 0.425234192 | 0.426668879 | 0.386681676 | 0.394126915 |
| INT32 SUM | 1.006048092 | 0.997510847 | 1.010224426 | 1.005217876 | 1.000896052 | 1.006048092 |
| INT64 SUM | 1.071772468 | 1.003376732 | 0.997980699 | 0.867376737 | 0.999205938 | 0.994940110 |
| FLOAT32 copy | 0.409918180 | 0.442128533 | 0.413142526 | 0.393407481 | 0.405984260 | 0.410870595 |

No sample was rejected from the accepted run. The interrupted pre-fix five-fork command and all
four-accumulator comparator runs are rejected diagnostic evidence and do not contribute to these
ratios. Raw accepted output SHA-256 is
`ce94c9adf8c7cd0d3106b701a4aff6aceeedd562fc2af077c14aeaa5a49e16c9`; benchmark-source
SHA-256 is `d7877cccb296818ff7636a976a8eca553005f1ff1f68326ba0df563d3e725c4f`.
The environment was OpenJDK 26.0.1+8-34, macOS/Darwin 25.5.0 ARM64, Apple M3 Max, 16 cores
(12 performance and four efficiency), and 64 GB RAM.

The mandatory clean documentation finalization ran in context
`01a03461-6786-75f0-ae15-798bf768ba1e`. It independently inspected the final diff, all listed
production/package contracts, tests, Model/Compiler semantic owners, shared Prepare/Runtime/public
boundaries, and the complete fresh evidence set. It changed no executable Java and therefore
reused the successful focused and one final CPU-suite evidence above.

Static checks passed: CPU Javadoc, task-required Markdown links/anchors/fences/final-newlines and
trailing-whitespace checks, exact allowed-path/scope/status checks, `git diff --check`,
`git diff --cached --check`, current schema exactly 43,
all fresh compatibility envelopes begin with 43, and the historical ledger remains byte-identical
at SHA-256 `41774384007bdb17b011cb8fbaae3b6928baa6ac8c04027a82540977f046031a`
with `generated-schema=42`. Staging is empty. The worktree has 22 total changed/untracked paths,
including preserved unrelated CPU 0007A1O/planning work; A2 itself remains below its 37-path ceiling.

## Implementation notes

The existing aggregate flow now owns bound right-aligned geometry, source/target structural
identity, direct no-reduction represented-bit copy, general typed coordinate odometers, and the
specialized dense benchmark form. Floating paths retain `CpuExactSumEmitter` as the sole exact
accumulation/conversion emitter. Integral dense loops use the optimization-relevant pre-tested
counted-loop shape demonstrated by the clean Java oracle. Existing preparation/finalization and
executable resource owners required no production change.

## Completion summary

- Completed changes: Full in-place SUM-to-Shape capability validation, structural IR identity,
  checked lowering/geometry, typed general and dense generated execution, raw represented copy,
  exact resource selection, scalar reference behavior, schema-43 cache compatibility, historical
  ledger/current-schema separation, and semantic/resource/cache/lifecycle/code-shape/performance
  tests and evidence.
- Implementation-owned files changed or created: `CpuCapabilityProvider.java`,
  `CpuGeneratorSchema.java`, `CpuAggregateEmitter.java`, `CpuAggregateIr.java`,
  `CpuAggregateLowering.java`, `CpuScalarReferenceKernel.java`, `CpuCapabilityProviderTest.java`,
  `CpuGeneratedKernelArtifactStoreTest.java`, `CpuKernelSpecializationTest.java`,
  `CpuAggregateGeneratedKernelTest.java`, `CpuPointwiseLedgerEvidenceTest.java`,
  `CpuPreparedExecutableTest.java`, `CpuAggregateIrTest.java`, `CpuAggregateLoweringTest.java`,
  `CpuPartitionFinalizerTest.java`, `CpuPartitionPreparerTest.java`, and this task record.
- Tests and validation: Corrected focused selection passed; the one final CPU suite passed; fresh
  semantic, Class-File, deterministic, member-reference, descriptor, compatibility, ledger,
  static, and six-case five-fork performance gates passed.
- Documentation finalization: Context `01a03461-6786-75f0-ae15-798bf768ba1e` finalized
  `CpuAggregateIr` plus the CPU provider, cache, emitter, executable, IR, lowering, and prepare
  package contracts; updated the CPU backend guide, glossary, this task, CPU master plan, and
  roadmap. The guide now records fully static right-aligned five-type support, exact floating and
  modular integral semantics, raw represented-copy behavior, output-cell-only parallelism,
  overlap/resources/lifecycle, direct general and guarded dense generated forms, schema 43, and
  the explicit dynamic/vector/native/fusion exclusions.
- Javadoc review: `CpuCapabilityProvider`, `CpuGeneratorSchema`, `CpuAggregateEmitter`,
  `CpuAggregateIr`, `CpuAggregateLowering`, and `CpuScalarReferenceKernel` accurately document the
  changed contracts. `CpuKernelSpecialization`, `CpuExactSumEmitter`, `CpuPreparedExecutable`,
  `CpuPartitionLowering`, `CpuPartitionFinalizer`, `CpuPartitionPreparationPlan`, and
  `CpuPartitionPreparer` remain accurate without edits; their behavior and ownership did not
  change. Updated package summaries are limited to the provider, cache, emitter, executable, IR,
  lowering, and prepare packages where ordinary-only or schema-42 wording became stale.
- Glossary impact: The established Sum-to-Shape term needed only a current-boundary clarification:
  Model construction still performs no binding or execution, while CPU schema 43 executes only
  fully bound static occurrences. No new term was introduced.
- No-change conclusions: Model and Compiler SumToShape semantics, Planning capability contracts,
  shared Prepare analysis/finalization, Runtime and `RunState`, public Tensor/Compile/Training APIs,
  `ARCHITECTURE.md`, architecture plans/ADRs/tests, other backends, backend conformance and
  integration tests, Gradle/build configuration, and later task specifications remain unchanged.
  The CPU-private implementation extends the existing aggregate path without changing those
  boundaries. CPU 0007A1D remains `Review needed`; CPU 0007A1C, CPU 0007A1O, and all other
  completed A1 tasks remain `Complete`. CPU 0007B is the next ordered Draft planning follow-up;
  no CPU task is Ready and no detailed 0007B specification was created.
- Unresolved implementation issues: None.
- Follow-up required: None for CPU 0007A2. CPU 0007B remains separate Draft planning work.

Status: Complete
