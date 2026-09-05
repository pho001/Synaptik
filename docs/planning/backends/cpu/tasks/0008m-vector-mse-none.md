# Task 0008M: Vector MSE `NONE`

## Status

Complete

## Goal

Add Java 26 Vector API and parallel-vector execution for the existing portable CPU
mean-squared-error (MSE) `NONE` form when prediction, target, and result are same-typed
`FLOAT32` or same-typed `FLOAT64` contiguous values. Each full vector computes the existing
per-element formula and the current scalar body owns every remainder:

```text
difference = prediction - target
loss = difference * difference
```

The change extends only the compute strategy for an already implemented first-class loss.
`SUM` and `MEAN` remain scalar because horizontal vector reduction would change the current CPU
loss body's defined increasing-order, left-associated accumulation. Categorical loss, BFLOAT16,
mixed floating promotion, and non-contiguous MSE remain on their existing scalar paths.

## Scope

### Exact semantic and eligibility boundary

Admit vector compute only when all of these facts are true:

- the occurrence is exactly `LossKind.MEAN_SQUARED_ERROR` with
  `MeanSquaredErrorAttrs(LossReduction.NONE)`;
- prediction, target, and output are all `FLOAT32`, or all are `FLOAT64`;
- Shapes are fully static and positionally exact, the output has the same Shape, and every
  prediction, target, and output access is dense row-major contiguous with a non-negative cold
  element offset;
- the output is injective and every element count, referenced span, byte offset, range, and
  vector-address calculation is representable under the existing lowering/binding checks;
- the Java 26 preferred species for the one lane type has more than one lane and the complete
  logical element count contains at least one full species;
- CPU analysis inputs request `VECTOR_IF_ELIGIBLE`; and
- existing carrier, span, accessibility, native-order segment, writability, and overlap checks
  succeed before invocation or worker submission.

The exact vector type matrix is therefore two rows:

| Prediction | Target | Result/lanes | Vector form |
|---|---|---|---|
| `FLOAT32` | `FLOAT32` | `FLOAT32` / preferred `FloatVector` species | eligible |
| `FLOAT64` | `FLOAT64` | `FLOAT64` / preferred `DoubleVector` species | eligible |

Every other already admitted loss form remains supported through its current scalar compute:

- MSE `SUM` and `MEAN`, including dense large inputs;
- BFLOAT16 MSE and either mixed ordered floating pair;
- MSE `NONE` with any non-contiguous prediction, target, or output access;
- dense-target and index-target categorical loss for every reduction; and
- zero-element, rank-zero, or otherwise shorter-than-one-species MSE `NONE` work.

`VECTOR_IF_ELIGIBLE` is a preference, not a requirement. An ineligible vector form selects
`SCALAR` or `PARALLEL_SCALAR` according to the existing range policy; it does not lose CPU
capability, decompose the loss, materialize a value, or fail an otherwise valid occurrence.

### Shapes, layouts, carriers, ranges, and aliases

- Vector eligibility is rank-polymorphic over any fully static Shape. A rank-zero Shape contains
  one element and selects scalar. Any zero extent produces zero work and no generated call or
  worker submission.
- Dense layouts may have zero or positive cold storage offsets. Surrounding storage is untouched.
  Positive/zero strides that do not form the exact dense row-major layout retain the scalar path;
  there is no gather, scatter, lane map, or automatic contiguous copy.
- Each unique boundary is either its exact primitive array (`float[]` or `double[]`) or an exact
  native-order accessible `MemorySegment`. Every ordered array/segment pattern is eligible.
  Segment element offsets are converted to byte offsets with the lane type's exact width.
- The direct generated signature remains one typed carrier parameter per unique boundary,
  followed by `long[] geometry`, primitive `long start`, and primitive `long end`, returning
  `void`. Distinct inputs have three carrier parameters; an exact repeated input role has one
  shared read parameter plus one output parameter. There is no `Object`, scratch, worker, Shape,
  layout, operation, or strategy parameter.
- Any valid half-open `[start,end)` inside the logical element domain is accepted. Full vectors
  start at the exact cold base plus `start`; the loop executes unmasked complete preferred-
  species chunks, then the existing scalar formula handles each remaining element exactly once.
  Empty and short requested ranges execute no vector access.
- Read-only prediction and target storage may be identical or overlap. Exact repeated graph input
  roles retain the existing role-to-unique-boundary map `[0,0]`; distinct roles retain `[0,1]`
  even when their bound storage overlaps. Output may share an underlying carrier only when its
  validated byte span is disjoint from every input span. Any output/prediction or output/target
  overlap is rejected before output mutation or worker submission. The one-output family has no
  output/output alias case.

### Vector body and scalar tail

For each full chunk, the generated body and its optimal clean Java oracle must:

1. load prediction and target with the matching preferred species through generation-time-
   selected array or native-order segment calls;
2. compute one typed vector subtraction in prediction/target order;
3. multiply that difference vector by itself, without fused multiply-add, reassociation,
   widening, reduction, or horizontal lane operation;
4. store the typed vector directly to the selected output carrier; and
5. advance only primitive ordinal/address state by the preferred lane count.

After the last full chunk, the scalar tail performs the existing typed subtraction, multiply, and
store in increasing logical-element order. There is no masked tail, temporary array, per-lane
extraction/insertion, scalar callback per lane, or call to a Synaptik-owned vector helper.

`FLOAT32` performs binary32 subtraction and multiplication. `FLOAT64` performs binary64
subtraction and multiplication. This preserves the existing MSE `NONE` NaN classification,
infinity behavior, signed-zero squaring, subnormal/underflow behavior, and overflow behavior.
Tests compare raw bits where the existing semantics define them and otherwise use the same
classification/tolerance rules as CPU 0008I. The task introduces no cross-backend bitwise promise.

### Strategy selection, parallelism, and run isolation

- CPU analysis selects `VECTOR` when the exact vector conditions hold and the existing selected
  range count is one. It selects `PARALLEL_VECTOR` when the same conditions hold and the existing
  bounded range policy selects at least two ranges.
- Parallel-vector reuses the same vector generated artifact. Configured/available parallelism,
  minimum elements per worker, selected range count, worker identity, and chunk boundaries remain
  prepared or invocation facts outside class identity.
- Existing quotient/remainder chunking produces ascending, non-empty, disjoint ranges that cover
  the requested interval exactly once. Each worker owns only its output range; scalar tails are
  local to that range.
- Existing inline one-chunk behavior, nested-submission rejection, worker accessibility proof,
  synchronous join, deterministic failure selection/suppression, interruption behavior, and
  caller-owned `CpuWorkerGroup` lifetime remain unchanged.
- `CpuPreparedExecutable` remains immutable and reusable. Concurrent runs and concurrent bound
  invocations retain distinct geometry, ranges, carrier bindings, and Runtime `RunState` objects;
  they share only immutable generated artifacts and explicitly borrowed worker capacity.

### Schema, class identity, direct signatures, and cache facts

- Advance `CpuGeneratorSchema.CURRENT_VERSION` from 61 to 62.
- Use class-identity schema 62 only for a vector-compute `CpuLossIr` whose exact semantic form is
  same-typed FLOAT32/FLOAT64 MSE `NONE` and whose three occurrence accesses are dense. Preserve
  class-identity schema 58 and byte-identical generated classes for every scalar loss form.
  Preserve schema 52, 54-57, and 59-61 projections and generated bytes for all unchanged families.
- Preserve the existing route-independent `CpuLossIr` semantic/range/role fingerprint for scalar
  compatibility. Vector code shape is distinguished by specialization compute `VECTOR`, exact
  preferred-species bit size, ordered boundary types/carriers, class schema 62, and the existing
  loss fingerprint. In particular, preserve the existing
  `workspace=NONE:realization=DIRECT_SCALAR` fingerprint text byte-for-byte: despite that
  historical token, compute strategy remains a specialization fact, and changing the token would
  invalidate the schema-58 scalar inventory. Do not add a second loss IR, vector-loss operation
  kind, or strategy field to Model semantics.
- Code-shaping facts are the existing MSE `NONE` family/reduction/range/role identity, same lane
  type, exact ordered unique-boundary carrier pattern, vector compute, preferred-species bit size,
  and schema 62. Parallel orchestration does not create another class.
- Rank, extents, element count, cold base offsets, concrete carrier instances/addresses, slots,
  graph/value/run identities, start/end bounds, selected range count, chunking, worker identity,
  and artifact root remain outside generated class identity.
- `compatibilityBytes()` advances with envelope 62, so every older persisted envelope is a safe
  miss. `classIdentityBytes()` retains prior projections for unchanged classes. Cache tests prove
  equal vector requests reuse one process-local artifact, carrier order/species/type/role changes
  miss, parallel-vector reuses vector identity, and cold geometry/run facts do not change the key.
- Generated schema-62 classes remain final, field-free, constructor-free hidden classes. They
  retain one public direct typed `invoke` entry and the current loss-private direct-body members;
  cold binding resolves the exact contiguous body before execution. No method-handle lookup or
  helper selection occurs in the generated loop or worker call.

The current generic instruction-free-family path can classify homogeneous loss IR as vector-
eligible even though schema-58 `CpuLossEmitter` emits scalar arithmetic. Replace that accidental
generic eligibility only with the exact family-specific predicate above: MSE `NONE` gains vector
compute, while MSE reductions and both categorical families select scalar or parallel-scalar.
This correction must not alter the schema-58 scalar fingerprint, direct signatures, or bytes.

### Exact finite validation matrices

The implementation must encode these finite matrices as named rows, assert their counts, and fail
on a missing or duplicate row. “All variants” is not acceptable evidence.

#### Generated identity and semantic execution matrix

Retain exactly 24 schema-62 Class-File dossiers:

```text
distinct prediction/target boundaries:
  2 lane types * 2^3 ordered array/segment assignments = 16 classes

shared prediction/target boundary ([0,0]):
  2 lane types * 2^2 ordered array/segment assignments = 8 classes

total = 16 + 8 = 24 classes
```

Execute every one of the 24 identities in exactly two strategy/range scenarios:

```text
24 classes * (VECTOR zero-offset exact-multiple
              + PARALLEL_VECTOR positive-offset tail) = 48 normal cases
```

The exact-multiple case uses `4 * lanes`; the tail case uses `9 * lanes + 3`, positive boundary
offsets `[3,5,7]` for distinct roles or `[3,7]` for shared roles, four configured/available
workers, and a minimum-elements setting that selects at least two non-empty chunks. Each case
checks pre/post sentinels and exact range coverage.

Add these separately counted focused matrices:

```text
selection/fallback:
  2 types * (SUM, MEAN, strided prediction, strided target,
             strided output, whole-domain shorter than one species) = 12 cases

shape/range:
  2 types * (zero extent, rank zero, empty subrange,
             one-element subrange, arbitrary non-zero-start tail) = 10 cases

exceptional values:
  2 types * (all-array, all-segment) = 4 cases

legal aliasing:
  2 types * (shared-role array, shared-role segment,
             overlapping distinct read arrays, overlapping distinct read segments,
             same-array disjoint output, same-segment disjoint output) = 12 cases

pre-write rejection:
  2 types * (output/prediction array overlap, output/target array overlap,
             output/prediction segment overlap, output/target segment overlap,
             insufficient span, misaligned segment offset, read-only output,
             inaccessible segment) = 16 cases

concurrent isolation:
  2 types * (VECTOR, PARALLEL_VECTOR) = 4 cases

focused semantic total = 48 + 12 + 10 + 4 + 12 + 16 + 4 = 106 cases
```

The four exceptional-value rows each contain finite ordinary values, `+0.0`, `-0.0`, positive and
negative infinity, NaN, maximum finite values that square to infinity, minimum-normal/subnormal
values that exercise underflow, and lane-boundary plus scalar-tail positions.

Retain exactly nine unchanged scalar controls: FLOAT32/FLOAT64 MSE `SUM` and `MEAN` (four),
BFLOAT16 MSE `NONE` (one), both ordered FLOAT32/FLOAT64 mixed MSE `NONE` forms (two), FLOAT32 dense
categorical `NONE` (one), and FLOAT32/INT32 index categorical `NONE` without ignore (one). Each
control remains scalar compute with schema 58. A pre-edit scalar-loss baseline generated from the
clean task-start commit and the post-edit complete 792-class schema-58 inventory must have the
same keys and Class-File SHA-256 values; the comparison is retained under the evidence root.

#### Class-File and forbidden-overhead inspection

For all 24 schema-62 classes retain the Class-File, complete `javap -c -v -p`, descriptor/member
inventory, normalized instruction/call-owner record, SHA-256, and deterministic second emission.
Inspection must prove:

- exact direct signatures for every distinct/shared ordered carrier pattern;
- preferred `FloatVector` or `DoubleVector` loads, one vector subtraction, one vector multiply,
  direct vector store, lane-count ordinal advance, and the scalar tail;
- no `reduceLanes`, horizontal reduction, FMA, masked-tail load/store, gather/scatter,
  `intoArray` temporary, per-lane extraction, or scalar callback in a full chunk;
- no allocation, boxing, reflection, `invokedynamic`, dynamic constant/bootstrap, collection,
  map/string/opcode dispatch, monitor, method-handle lookup, graph/layout/operation/cache/route/
  resource/worker lookup, fallback/reference call, or Synaptik-owned hot helper; and
- unchanged scalar loss and prior-family class projections/hashes as described above.

#### Generated-versus-optimal-clean-Java performance matrix

Measure exactly six rows per lane type:

| Row per type | Boundaries | Offsets | Strategy | Range |
|---|---|---|---|---|
| 1 | distinct array/array/array | zero | `VECTOR` | exact multiple |
| 2 | distinct segment/segment/segment | positive | `PARALLEL_VECTOR` | tail |
| 3 | distinct array/segment/array | positive | `VECTOR` | tail |
| 4 | distinct segment/array/segment | zero | `PARALLEL_VECTOR` | exact multiple |
| 5 | shared input array, output segment | positive | `VECTOR` | tail |
| 6 | shared input segment, output array | zero | `PARALLEL_VECTOR` | exact multiple |

```text
6 rows * 2 types = 12 performance rows
12 rows * 5 fixed fresh forks * 9 retained symmetric sample pairs = 540 sample pairs
12 rows * 5 forks = 60 fork medians
12 median-of-fork-medians aggregates
```

Use a shape-polymorphic dense rank-two workload whose element count is
`8192 * preferredLanes` for exact rows and `8192 * preferredLanes + 3` for tail rows. Positive
offset is seven elements. Shared-input rows bind one exact input boundary. The fixture is cold
geometry, not fixed-shape specialization.

The ordinary clean Java 26 oracle has the exact generated typed signature, preferred species,
carrier order, cold `long[]` geometry, and `start`/`end` bounds. It is compiled by `javac` before
timing and uses the same full-vector/scalar-tail algorithm, load/store order, primitive address
state, parallel ranges, and output consumption as the generated side. Source and decompilation
inspection must reject a slower oracle, dispatcher, helper bridge, allocation, boxing,
reflection, fallback/reference call, fixed fixture trip count, or different algorithm.

Each fork uses Java 26, fixed `-Xms1g -Xmx1g`, C2-only synchronous compilation
(`-XX:-TieredCompilation -Xbatch`), a recorded deterministic row/order seed, exact pre/post output
verification, and exactly five warmup pairs before final calibration. After warmup, conservatively
double one shared iteration count until the two-invocation generated aggregate and the two-
invocation direct aggregate are each at least 50 ms. Retain nine randomized symmetric
`G-D-D-G` or `D-G-G-D` AB/BA pairs, and require each of the four individual retained timings in
every pair to be at least 25 ms; the 50 ms calibration rule is only the conservative precondition,
not a substitute for checking every side. No retry, discarded sample, replacement fork, outlier
filter, threshold change, asymmetric work, or production selection is permitted.

Every retained pair ratio, fork median, and aggregate generated/direct ratio must be `<= 1.15x`.
Retain raw four-timing samples, iteration counts, checksums, environment/JDK/OS/CPU/heap facts,
commands, source snapshots, oracle bytes, generated classes, decompilation, summaries, and a
complete SHA-256 manifest under one explicit untracked evidence root.

## Out of scope

- Vector or parallel-vector MSE `SUM`/`MEAN`; horizontal reduction, reassociation, partial
  reductions, combine kernels, workspace, and deterministic partial-reduction parallelism.
- Dense/index categorical SIMD; softmax/log-softmax/attention numerical changes; CPU 0008O's
  stable-reduction eligibility work.
- BFLOAT16/FLOAT16 SIMD, mixed-type vector conversion/promotion, integral/BOOL work, relaxed math,
  FMA, masked vector tails, gather/scatter, or non-contiguous vector access.
- Materialization, packing, fixed-shape specialization, unrolling, native/vendor/OpenBLAS routes,
  autotuning, tuning-cache mutation, benchmark-selected production behavior, or persistence
  policy.
- Loss fusion, decomposed-loss recognition, public API, operation, gradient, Training, Model,
  Compiler, Planning, Config, Backend Contract, Trace, shared Prepare, Runtime, Engine, other
  backend, module dependency, architecture, ADR, build-toolchain, conformance, or integration
  change.
- CPU 0008N implementation or detailed specification; CPU 0008N becomes the sole Ready frontier
  after this task completes, without a detailed task file.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially concrete CPU route,
  staged Prepare, Runtime/run isolation, and generated-code performance discipline.
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md).
- [Planning Guide](../../../planning-guide.md).
- [CPU backend master plan](../master-plan.md).
- [CPU 0008I portable loss-family execution](0008i-portable-loss-family-execution.md).
- [CPU 0008L pointwise SIMD mask/output closure](0008l-pointwise-simd-mask-output-closure.md).
- [CPU 0005C vector and parallel portable strategies](0005c-vector-and-parallel-portable-strategies.md).
- [CPU 0005I FLOAT32 vector parity](0005i-float32-vector-parity-and-vector-emission-boundary.md).
- [CPU 0007A0 generated hot-path parity](0007a0-generated-hot-path-parity-correction.md).
- [CPU 0007A1A generated scalar-body self-containment](0007a1a-generated-scalar-body-self-containment.md).
- [CPU 0008B partition-DAG decomposition](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md).
- [CPU 0008E1 shared partition-DAG adoption](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md).
- [CPU backend guide](../../../../backend-guide/cpu-backend.md).
- [Glossary](../../../../glossary.md).

## Architecture constraints

- Model owns MSE meaning and Compiler owns inference/gradients. Planning continues to select only
  CPU ownership. CPU analysis alone owns vector eligibility, preferred species, strategy,
  specialization, direct carrier signature, and scalar fallback.
- CPU analysis selects the route/strategy and declares unchanged buffers before shared assignment;
  finalization realizes that immutable decision. Shared Prepare and Runtime do not interpret loss,
  vectors, species, carriers, ranges, or workers.
- Generated code receives direct typed carriers plus primitive cold geometry/ranges. Runtime hot
  execution sees no `Operation`, `CompiledNode`, graph, strategy, or backend lookup.
- Prepared recipes remain immutable/reusable and each active run has isolated mutable `RunState`.
  Existing CPU-private caller-owned workers are the only shared orchestration resource.
- The `AGENTS.md` generated-code rule is mandatory: directly emitted bytecode must preserve the
  optimal clean Java specialized algorithm, full-vector/scalar-tail loop and dataflow shape, and
  avoidable-overhead profile. Any deviation requires a source-backed reason and evidence.
- Any public/shared contract, new dependency, resource kind, architecture decision, or need to
  change current MSE semantics is a stop condition.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.ir` — unchanged loss semantic/range/role identity.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — exact vector eligibility, strategy,
  species, and schema selection.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct typed MSE vector body,
  scalar tail, structural evidence, and test-only optimal Java oracle.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema-62 compatibility and class identity.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — unchanged cold helper binding,
  overlap validation, ranged calls, workers, failures, and run isolation.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — unchanged scalar semantic oracle.

Packages added or changed:

- No package or responsibility is added or moved.

Type placement:

- No production type is added. Extend the existing `CpuLossEmitter`, `CpuPartitionPreparer`,
  `CpuKernelSpecialization`, `CpuGeneratorSchema`, and `CpuLossIr` documentation/owners only.
- `...internal.codegen.emit.CpuVectorMseEvidenceTest` — package-private exhaustive schema-62
  semantic, Class-File, scalar-preservation, and retained-evidence owner.
- `...internal.codegen.emit.CpuVectorMsePerformanceOracle` — package-private test-only `javac`
  compiler/binder for the optimal direct Java Vector API methods; unreachable from production.
- `...internal.codegen.emit.CpuVectorMsePerformanceTest` — package-private opt-in fixed-matrix,
  fixed-fork performance and manifest owner.

## Affected files

Expected production/Javadoc paths (5):

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuLossEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuLossIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`

Expected test/evidence/build paths (18):

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuBatchNormTrainingEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv2dEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv3dEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuLossEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuLossGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPartitionDagGeneratedEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseLedgerEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseMaskEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorMseEvidenceTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorMsePerformanceOracle.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorMsePerformanceTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/build.gradle.kts` — guarded structural/performance evidence properties only.

The build file forwards exactly
`synaptik.cpu.vectorMse.structuralEvidenceRoot`,
`synaptik.cpu.vectorMse.performance`, and
`synaptik.cpu.vectorMse.performanceEvidenceRoot` when explicitly supplied. It adds no ordinary
test behavior or environment-variable alias.

Expected documentation/planning paths (5):

- `docs/backend-guide/cpu-backend.md` — update the current portable-loss strategy and
  compatibility-schema account, which currently ends at schema 61/CPU 0008L.
- `docs/glossary.md` — update only the existing `CPU portable route` entry, which currently ends
  at CPU 0008L/schema 61; add no term or heading.
- `docs/planning/backends/cpu/tasks/0008m-vector-mse-none.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Reviewed and expected unchanged: `CpuCapabilityProvider`, `CpuClassFileKernelGenerator`,
`CpuCarrierEmitter`, `CpuVectorInstructionEmitter`, `CpuLoopEmitter`, `CpuScalarEmitter`,
`CpuLossLowering`, `CpuLossReferenceKernel`, `CpuPortableRoutePlan`, `CpuGeneratedKernel`,
`CpuPreparedExecutable`, package summaries, existing scalar loss performance owners, Model/
Compiler/Training APIs, shared Prepare/Runtime, architecture pages/ADRs/tests,
backend-conformance/integration tests, other backends, and root Java 26 Gradle configuration.

## Maximum scope

At most 28 paths: five production/Javadoc, 18 test/evidence/build, and five documentation/planning.
At most three package-private test types and no production type or package may be added.

This exceeds the normal task range because one cohesive generated-code increment must update
eligibility, schema/cache identity, direct vector emission, scalar-preservation controls,
semantic/Class-File/performance evidence, hard-coded current-envelope assertions, Javadocs, and
backend documentation together. The prior-family evidence edits authorize only current-envelope
assertion maintenance; they must not change prior-family behavior or generated bytes. A listed
review path need not change merely because it is named. Stop and replan if another production
owner, test type, package, shared/public path, resource, route, or architecture edit is required.

## Acceptance criteria

1. CPU analysis selects vector compute for exactly same-typed FLOAT32/FLOAT64 contiguous MSE
   `NONE` under vector preference and the species/count gates; every other existing loss form
   retains scalar compute or existing scalar fallback without losing capability.
2. Generated FLOAT32/FLOAT64 full chunks implement exactly typed subtract, self-multiply, and
   direct store; scalar tails preserve the current scalar formula and visit every remainder once.
   No reduction, FMA, widening, reassociation, masked tail, gather/scatter, or per-lane callback is
   present.
3. The exact 106-case focused semantic matrix passes across all 24 identities, both vector
   strategies, every ordered array/segment pattern, offsets, exact/tail/short/zero/rank-zero/
   arbitrary ranges, exceptional values, legal aliases, pre-write rejection, and concurrency.
4. Vector and parallel-vector use the same direct typed artifact. Existing worker chunking,
   joining, failure behavior, accessibility checks, immutable prepared recipes, and run isolation
   remain unchanged.
5. Schema advances 61 to 62 only for changed vector MSE bytes. All 24 schema-62 identities are
   unique and deterministic; scalar loss remains schema 58 and the complete 792-class scalar
   inventory is byte-identical to its clean task-start baseline. Every other prior class projection
   remains unchanged, while compatibility envelope 62 safely misses older persisted entries.
6. Direct signatures and cache keys contain every code-shaping type/carrier/role/compute/species
   fact and exclude every cold extent/offset/address/range/chunk/worker/run fact listed in Scope.
7. The exact 24 retained Class-File dossiers pass complete descriptor/member/decompilation,
   deterministic-byte, algorithm/loop/tail, call-owner, and forbidden-overhead inspection.
8. All 540 retained symmetric sample pairs, 60 fork medians, and 12 aggregates in the exact
   performance matrix are `<= 1.15x`; every retained generated and direct side is at least 25 ms,
   and the protocol uses warmup before conservative post-warmup calibration, fixed forks, and no
   retry/discard.
9. MSE `SUM`/`MEAN` stay scalar and their increasing-order left-associated accumulation remains
   unchanged. No categorical, BFLOAT16, mixed-type, non-contiguous, materialized, native, or
   partial-reduction vector form becomes selectable.
10. No public/shared API, resource, dependency, architecture, conformance, integration, toolchain,
    tuning, persistence-policy, or unrelated generated-family behavior changes.
11. A distinct clean documentation-focused pass finalizes affected Javadocs, the CPU guide,
    glossary impact, task evidence, CPU master plan, and roadmap before completion. It records
    reasoned no-change conclusions and reuses successful Java evidence unless it changes executable
    behavior.
12. On completion CPU 0008M becomes `Complete`; CPU 0008N becomes the sole `Ready` CPU frontier.
    No 0008N detailed task is created in the 0008M completion pass.

## Tests / validation

### Implementation commands

Capture the clean scalar baseline before production edits, then run focused tests while
implementing. The implementation context must replace the focused command only if an expected
owner proves unnecessary; it may not silently omit a changed owner.

```bash
CPU_0008M_EVIDENCE_ROOT="$(mktemp -d '/private/tmp/synaptik-cpu-0008m-XXXXXXXX')"
printf '%s\n' "$CPU_0008M_EVIDENCE_ROOT" > "$CPU_0008M_EVIDENCE_ROOT/evidence-root.txt"
mkdir -p "$CPU_0008M_EVIDENCE_ROOT/scalar-before"
touch "$CPU_0008M_EVIDENCE_ROOT/scalar-before/RUN-STRUCTURAL-EVIDENCE"
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuLossEvidenceTest -Dsynaptik.cpu.loss.structuralEvidenceRoot="$CPU_0008M_EVIDENCE_ROOT/scalar-before" --rerun-tasks

./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuBatchNormTrainingEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGeneratorTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuConv2dEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuConv3dEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuLossGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPartitionDagGeneratedEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseLedgerEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseMaskEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuVectorMseEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest
./gradlew :backends:cpu:test --rerun-tasks
```

`CpuLossEvidenceTest` consumes and deletes the exact `RUN-STRUCTURAL-EVIDENCE` marker in the
property-selected directory. Capture this baseline before any production edit. Do not weaken,
reconstruct after the edit, or skip the pre/post 792-class key and Class-File hash comparison.

### Retained structural and performance evidence

Use the same explicit root captured above:

```bash
touch "$CPU_0008M_EVIDENCE_ROOT/RUN-VECTOR-MSE-STRUCTURAL-EVIDENCE"
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuVectorMseEvidenceTest -Dsynaptik.cpu.vectorMse.structuralEvidenceRoot="$CPU_0008M_EVIDENCE_ROOT" --rerun-tasks

mkdir -p "$CPU_0008M_EVIDENCE_ROOT/scalar-after"
touch "$CPU_0008M_EVIDENCE_ROOT/scalar-after/RUN-STRUCTURAL-EVIDENCE"
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuLossEvidenceTest -Dsynaptik.cpu.loss.structuralEvidenceRoot="$CPU_0008M_EVIDENCE_ROOT/scalar-after" --rerun-tasks

./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuVectorMsePerformanceTest -Dsynaptik.cpu.vectorMse.performance=true -Dsynaptik.cpu.vectorMse.performanceEvidenceRoot="$CPU_0008M_EVIDENCE_ROOT" --rerun-tasks
```

Retain exact commands, task outcomes/test counts, 24 vector classes and `javap` reports, both
792-class scalar inventories and their equality report, 540 sample-pair records, 60 fork medians,
12 aggregates, source/oracle snapshots, and complete manifests/checksums. The evidence root is
untracked and must not be staged. `CpuVectorMseEvidenceTest` must read both scalar inventory roots,
compare the complete key-to-SHA-256 maps, and write the retained equality report; directory counts
or schema labels alone are insufficient.

### Documentation and final checks

After executable Java stabilizes, the implementation context runs the full CPU suite once. The
separate documentation context reuses that evidence unless it changes executable behavior, then
runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

Also validate all local links and heading anchors in the five documentation/planning files;
heading uniqueness; balanced Markdown fences; final newlines; trailing whitespace; the exact
`16 + 8 = 24`, `48 + 12 + 10 + 4 + 12 + 16 + 4 = 106`, and `12 * 5 * 9 = 540` arithmetic;
schema `61 -> 62`; the 5 + 18 + 5 = 28 path ceiling; exact type placement; no tracked evidence;
empty staging; and synchronized order/status with 0008M Complete, CPU 0008N as the sole Ready
frontier, and CPU 0008O onward Draft.

Repository-wide Java validation is deferred to CPU 0009/CI because this task changes only one
CPU-private route increment. Architecture, backend-conformance, and integration suites are not
run unless a shared/dependency/public boundary changes, which is a stop-and-replan condition.

## Dependencies

- [CPU 0008I](0008i-portable-loss-family-execution.md) — Complete; atomic scalar loss semantics,
  geometry, direct signatures, 792-class inventory, validation, and scalar oracle.
- [CPU 0008L](0008l-pointwise-simd-mask-output-closure.md) — Complete; current schema 61,
  preferred-species array/segment generation, scalar tails, retained Class-File evidence, and the
  corrected warmup/calibration/performance protocol.
- [CPU 0005C](0005c-vector-and-parallel-portable-strategies.md) and
  [CPU 0005I](0005i-float32-vector-parity-and-vector-emission-boundary.md) — Complete; four
  strategies, FLOAT32/FLOAT64 preferred species, direct carriers, ranges, workers, and fallback.
- [CPU 0007A0](0007a0-generated-hot-path-parity-correction.md) and
  [CPU 0007A1A](0007a1a-generated-scalar-body-self-containment.md) — Complete; generated/direct
  algorithmic parity and forbidden-overhead evidence discipline.
- [CPU 0008B](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md)
  and [CPU 0008E1](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md) — Complete;
  atomic numerical-order barriers and shared partition-DAG adoption.
- Java 26.0.1 Class-File and incubating Vector APIs through the existing CPU build.

## Follow-up tasks

- CPU 0008N is the sole Ready frontier after 0008M completes. It owns only measured
  profitable FLOAT32/FLOAT64 Conv2d/Conv3d accumulation-axis SIMD and requires its benchmark-axis
  spike; do not detail or implement it here.
- CPU 0008O remains the later Draft stable-reduction numerical spike for softmax, categorical
  loss, and attention.
- CPU 0008P remains the later Draft deterministic partial-reduction architecture with per-worker
  workspace and combine execution.
- CPU 0009 remains the generated-coverage closure checkpoint and must inventory schema-62 vector
  MSE evidence separately from CPU 0008I's still-missing corrected full scalar-loss performance
  gate.

## Architecture impact

Expected impact: None.

Required no-change conclusions:

- Architecture contract/pages/ADRs/tests remain unchanged because ownership, lifecycle,
  dependencies, generated-code authority, and run isolation do not change.
- Model loss semantics, `LossKind`, `LossReduction`, `MeanSquaredErrorAttrs`, Tensor/Compiler/
  Training APIs, and capability-query contracts remain unchanged because CPU support and output
  meaning already exist.
- Planning, shared Prepare, and Runtime remain unchanged; current projected facts, declarations,
  slot assignment, finalization, cold binding, publication, worker ranges, and run state suffice.
- `CpuCapabilityProvider`, `CpuLossLowering`, `CpuPreparedExecutable`, `CpuWorkerGroup`, and scalar
  reference semantics remain behaviorally unchanged. CPU-private analysis changes only strategy
  eligibility for the exact form.
- MSE `SUM`/`MEAN`, categorical losses, prior scalar loss forms, other generated families, native
  routes, materialization/tuning policy, and other backends remain behaviorally unchanged.
- Backend-conformance and integration tests remain unchanged because no shared/public behavior
  owner or Engine path changes; CPU-private semantic and generated evidence owns this increment.
- Module dependencies, plugins, source sets, and root Java 26 toolchain remain unchanged. CPU
  Gradle may forward only guarded evidence properties to the test JVM.
- The CPU guide and existing `CPU portable route` glossary entry receive only the narrow current-
  behavior/schema update identified under Affected files. No architecture page, ADR, public API
  guide, new glossary term, or unrelated guide section changes; discovering such a need is a
  stop-and-replan condition.

Re-evaluate every conclusion against the final diff and stop if one is false.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean-context implementation agent for Synaptik CPU task 0008M.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, task 0008M,
completed CPU 0008I and 0008L plus the directly linked prerequisites, and every affected current
loss/vector/cache/executable/test owner in full. Capture the required clean scalar-loss baseline
before production edits. Implement the Ready specification within its exact path/type ceilings.

Preserve scalar loss semantics and bytes, scalar tails/fallback, direct typed signatures, ordered
carrier roles, cold geometry, worker/run isolation, and selective schema identity. Stop on any
architecture, shared-contract, accumulation-order, Vector API, optimal-clean-Java parity,
evidence, or scope conflict. Do not commit, push, stage, reset, or modify unrelated work.

After Java/Class-File/performance gates stabilize, hand the exact diff and retained evidence to a
distinct clean documentation-focused agent under docs/developer-guide/documentation-rules.md.
Do not mark Complete before every criterion and that pass succeed; then make CPU 0008N the sole
Ready frontier without a detailed task.
```

## Local decisions

- The implementation keeps `CpuLossIr` route-independent and preserves its historical
  `workspace=NONE:realization=DIRECT_SCALAR` fingerprint. Vector compute and the preferred species
  remain specialization facts rather than Model or loss-IR semantics.
- `VECTOR` and `PARALLEL_VECTOR` use the same schema-62 class. Parallel range count, worker
  identity, geometry, offsets, and invocation bounds remain cold facts.
- Vector chunks use one subtraction followed by self-multiplication. FMA, reduction, widening,
  reassociation, and masked tails remain excluded; the existing scalar formula owns every tail.
- Schema 62 is selected only for same-typed contiguous FLOAT32/FLOAT64 MSE `NONE`. Every scalar
  loss stays on schema 58, including `SUM`, `MEAN`, BFLOAT16, mixed types, categorical losses,
  non-contiguous access, and short domains.

## Known limitations

- Only same-typed contiguous FLOAT32/FLOAT64 MSE `NONE` gains SIMD. Every reduction, mixed type,
  BFLOAT16, categorical, and non-contiguous form remains scalar.
- Preferred species and observed performance are target/JVM-specific. This task establishes the
  fixed recorded Java 26 forms, not a hardware-intrinsic or universal speedup promise.
- Scalar tails remain required for short ranges and arbitrary chunk boundaries; there is no
  masked-tail path.
- CPU 0008I's corrected full 792-class by five-fork scalar-loss performance evidence remains
  missing and owned by CPU 0009. The 0008M scalar hash-preservation check and vector MSE evidence
  do not close or relabel that gap.

## Validation evidence

Documentation-focused context: `01a0730d-6446-7150-84ad-328833bdf27a`.

- Documentation profiles: General, API/Javadoc, Planning, backend-guide, and example-format.
- Reviewed contracts and owners: `AGENTS.md`, `ARCHITECTURE.md`, current architecture index,
  documentation rules/profiles, Planning Guide, this task, CPU master plan, roadmap, CPU guide,
  glossary, all five changed production owners and their Javadocs, all three new test/evidence
  owners, guarded CPU Gradle evidence-property forwarding, relevant package summaries, and the
  public/shared boundaries named under Architecture impact.
- Implementation contexts: `01a072d0-c30a-7821-90ec-3294b221b2d1` and continuation
  `01a072ec-6d5d-7892-b82c-130cfd882597`. Documentation context:
  `01a0730d-6446-7150-84ad-328833bdf27a`.
- Retained evidence root: `/private/tmp/synaptik-cpu-0008m-final-Ad4Cnvnl`. Task-start commit:
  `34b637dbf519712c45d154a88d6770b69b7ff461`; task-start tree SHA-256:
  `04d49430747cd1991005363c8b4b973f2a42b8a16074a0abed7bc002889cee99`.
- Structural evidence passed for exactly 24 schema-62 classes, keys, and hashes: 12 FLOAT32 and
  12 FLOAT64 identities, 24 complete `javap` dossiers, and an exactly eight-column dossier CSV.
  The 1,626-record structural manifest SHA-256 is
  `4612618d4c3c594ca52ea19c61c8e5a7966235cfc14b8f7a32d15f3e11c0e563`; command-record SHA-256
  is `2f49f967f19d523bbb50ecc8cdfd5b2e50c0a357225d472a7b70b1f12e592771`; dossier SHA-256 is
  `824e1a741d2dfc74bcfa34954273f87ee41805ea81cfccb64094ff740bb3f9d3`; combined-`javap`
  SHA-256 is `60c0b467286e0d7fabae28eb27ad64bd09ccea4ecb5ef76454808c86ab91bcf6`; and source-snapshot
  SHA-256 is `edab5873c40eec28d84b9616436155be2b115db9b5da4d036a089d15dd19e7a4`.
  Inspection found exact typed subtraction followed by self-multiplication, direct stores, typed
  scalar tails, shared `VECTOR`/`PARALLEL_VECTOR` identity, and no forbidden hot overhead.
- Scalar preservation passed for exactly 792 before and 792 after schema-58 classes. Both
  inventory SHA-256 values are
  `ce863c65922a5de76527eeaf48f7d1d969914765c3450508859846d39354fe2b`; key-map SHA-256 is
  `84eabd309ee018d743f35ead5f90a705723f5d379e2f19680c3039d24233b093`; equality-report
  SHA-256 is `86818e9e555c226a3828cb49e9cac85963176aa7ff27aeb720316cfed32bed65`.
- The authoritative performance command completed in 1 minute 36 seconds. All
  `12 * 5 * 9 = 540` retained pairs, 60 fork medians, and 12 aggregates passed with 4,096
  iterations. Pair ratios ranged from `0.898334532720123x` to `1.050397665425858x`; fork medians
  ranged from `0.919616703782793x` to `1.010812953112698x`; aggregates ranged from
  `0.924263487813383x` to `1.009408829343164x`. Individual timings ranged from 25,521,209 to
  33,990,208 ns. There were zero threshold or timing-floor violations, retry and discard were
  false, and exact equality checks passed. The 107-record performance manifest SHA-256 is
  `f2e104742250c82ffa28521f8ae390f34c9350d20b5192a0b7ecbdf033c20265`; aggregate SHA-256 is
  `2b3933e69aefc9cf1feded4d973b09e7de80e02162f6a22d6a82f46c0d5b5b52`.
- Final validation after evidence repair passed: the exact focused command ran 164 tests with one
  guard skip and zero failures/errors; the guarded structural run activated and passed all 11
  tests; and the full CPU run passed 139 suites and 730 tests with 21 expected opt-in skips and
  zero failures/errors.
- Javadoc generation passed after the `CpuPartitionPreparer` class-Javadoc correction, with 53
  pre-existing unrelated warnings. All five changed production owners now document their
  affected contracts accurately.
- Package/public/shared review: existing package summaries remain accurate at their ownership
  level and need no 0008M-specific edit. No public API, Model/Compiler/Training contract,
  capability-query contract, shared Prepare/Runtime contract, architecture page/ADR/test,
  backend-conformance/integration owner, other backend, dependency, materialization policy,
  tuning policy, or root Java 26 toolchain document changes.
- Documentation validation in context `01a0730d-6446-7150-84ad-328833bdf27a` passed the prior
  979-link/310-fragment check. The final five-file rerun passed local targets and anchors, unique
  effective headings, balanced fences, final newlines, no carriage returns or trailing
  whitespace, exact arithmetic/status/schema checks, `git diff --check`,
  `git diff --cached --check`, and empty staging. Exact scope is 24 of the authorized maximum 28
  paths: five production, 14 test/evidence/build, and five documentation/planning, with no path
  outside the task inventory and no tracked evidence artifact.

## Implementation notes

- Production changes are confined to the five authorized owners. CPU analysis removes generic
  loss vector eligibility and admits only the exact MSE predicate; emission uses direct typed
  preferred-species loads/stores and a scalar tail; specialization advances the compatibility
  envelope and selects schema 62 only for the vector form.
- The three new package-private test owners encode the exact 106-case semantic matrix, 24
  structural dossiers, scalar inventory equality, independently compiled clean-Java oracle, and
  12-row/five-fork performance protocol. CPU Gradle forwards only the three explicitly supplied
  `synaptik.cpu.vectorMse.*` properties.
- Documentation finalization updates only the five authorized Markdown paths. No Java, test,
  build, architecture, API, conformance, integration, package-summary, or evidence file was
  changed by the documentation context.

## Completion summary

- Completed changes: implemented and documented the bounded same-typed contiguous FLOAT32/FLOAT64
  vector and parallel-vector MSE `NONE` increment with scalar tails and selective schema 62.
- Files changed or created: see the exact 5 + 18 + 5 task inventory; documentation finalization
  changed only the five authorized Markdown paths.
- Tests and validation: exact structural, scalar-preservation, five-fork performance, focused,
  guarded structural, full CPU, Javadoc, Markdown, status, arithmetic, and scope gates passed as
  recorded above.
- Documentation-agent review: completed for the CPU guide, glossary entry, planning records,
  production/test/build contracts, package summaries, and public/shared no-change boundaries.
- Documentation impact: narrow CPU-private strategy and schema explanation only.
- Javadoc review: all five affected production-owner contracts are accurate; generation passed
  after the preparer class-Javadoc correction.
- Glossary impact: updated only the existing `CPU portable route` entry; no reusable term or
  heading was added.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
