# Task 0007A0E: Ordering Generated-Loop Parity

## Status

Complete

## Goal

Replace the bridge-only generated comparison, merge, selection, and output work for the completed
CPU stable-ordering family with carrier-, represented-type-, family-, access-, comparison-,
output-, and loop-specialized bodies embedded in each generated class. `SORT`, `ARGSORT`, and
`TOP_K` must preserve every CPU 0006C semantic, layout, carrier, range, resource, parallelism,
validation, and lifecycle contract while two independently gated dense heap-array FLOAT32 cases
reach reproducible near-parity with equivalent direct primitive stable-merge implementations.

This is one cohesive CPU-private generated-code correction. The three families share
`CpuOrderingIr`, compact cold geometry, `CpuOrderingEmitter`, stable bottom-up merge over two
assigned INT64 scratch regions, exact comparison classes, and one generated-artifact compatibility
boundary. The task changes generated code shape only; completed CPU 0006C capability, operation
semantics, algorithm, scratch declaration, and observable behavior remain unchanged.

## Scope

### Mandatory pre-edit source and Class-File baseline

- Before changing production Java, create one fresh isolated directory under `/private/tmp` and
  capture the current schema-26 source, generated classes, semantics, and raw timing for the two
  exact performance cases defined below.
- Also capture representative current classes for dense ARGSORT and a general-layout or mixed
  heap-array/`MemorySegment` ordering form so the baseline covers every output role and both loop-
  addressing categories before implementation chooses locals or control flow.
- Retain the probe and summarizer source; exact compile, generation, decompilation, execution, and
  summary commands; complete environment facts; generated class bytes; complete `javap -c -p`
  and `javap -v -p`; entry descriptors; class sizes; SHA-256 values; constant-pool member
  references; semantic verification; raw samples; per-fork summaries; and aggregate summaries.
- Reproduce baseline evidence freshly. Planning audit context
  `01a018e2-75c0-7272-b0b9-1ebb6ae7f856` observed these schema-26 classes under
  `/private/tmp/synaptik-cpu-0007a0e-planning.FVb4Eg`:
  - dense ascending FLOAT32 SORT: descriptor
    `([F[FLjava/lang/foreign/MemorySegment;[JJJ)V`, 489 bytes, SHA-256
    `55c2a18352b72827351a1a87476b4075acf3fdf36bb0c93c615c82d7933b0d7a`;
  - dense descending FLOAT64 ARGSORT: descriptor
    `([D[JLjava/lang/foreign/MemorySegment;[JJJ)V`, 489 bytes, SHA-256
    `2e7f10e26fa71e0aae7f71880624bba36552f5f70a246b9dcc957c2f771b4e88`;
  - dense sorted-largest FLOAT32 TOP_K: descriptor
    `([F[F[JLjava/lang/foreign/MemorySegment;[JJJ)V`, 492 bytes, SHA-256
    `e0d932fd29343fc7aedf703266a4eb89ee55b5973aeaeedc3996ca74b823f3b5`;
  - mixed-carrier unsorted-smallest BFLOAT16 TOP_K: descriptor
    `(Ljava/lang/foreign/MemorySegment;[SLjava/lang/foreign/MemorySegment;`
    `Ljava/lang/foreign/MemorySegment;[JJJ)V`, 554 bytes, SHA-256
    `4ffc4a2eca673e9ca944c16b448135d2750cd5e3b1193800472c93c0c7819de1`.
- Each audited class has exactly one method reference, to
  `CpuOrderingEmitter.execute(Object,Object,Object,MemorySegment,long[],long,long)`. The helper
  allocates one `int[]` layout table per invocation; reconstructs family, `DataType`, carrier, and
  access decisions at runtime; performs generic `Object` reads/writes; and calls comparison,
  address, copy, and index-output helpers inside ordering work. Treat these observations as
  planning evidence only and retain a fresh implementation baseline.
- Time only warmed stable-merge, selected-pair ordering, and output work. Exclude lowering,
  generation, verification, class definition, artifact lookup, preparation/finalization, cold
  binding, geometry packing, carrier/scratch allocation, validation, input initialization,
  expected-result construction, verification, summarization, and sink observation from both
  generated and direct timed forms.

### Embedded specialized ordering bodies

- Change `CpuClassFileKernelGenerator` and `CpuOrderingEmitter` so every generated entry embeds
  initialization of logical indices, stable bottom-up merge passes, selected-output ordering,
  represented value/index writes, and exact carrier access instead of calling
  `CpuOrderingEmitter.execute` or any equivalent generic bridge.
- Pass the existing structural ordering IR to the focused emitter. Select at generation time:
  - family: `SORT`, `ARGSORT`, or `TOP_K`;
  - represented type: FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, or BOOL;
  - direction: ascending/descending for full ordering or smallest/largest for TOP_K;
  - TOP_K output mode: sorted or deterministic increasing-original-index unsorted;
  - ordered input, values-output, and INT64-index-output carrier forms independently;
  - dense heap-array integer addressing or typed general long addressing; and
  - one-output versus ordered two-output generated stores.
- Preserve the universal primitive `long start, long end` entry. Bounds continue to identify
  complete logical-axis slices; arbitrary legal disjoint slice ranges remain valid, and one slice
  is never split between ranges.
- Hoist the family, type, direction, sorted flag, boundary ranks/roles, axis role, packed-layout
  positions, scratch-region facts, and other invocation-invariant packed values into primitive
  locals before the slice/merge/output loops. Concrete extents, axis, `k`, offsets, stride
  magnitudes, carrier-relative bases, scratch offsets, ranges, and range index remain cold,
  invocation-private, shape-polymorphic facts.
- Generated hot work must contain no `Object` descriptor or cast, `DataType.values`, family/type/
  carrier ordinal dispatch, array-versus-segment or carrier-form test, reflection, map/string
  dispatch, graph/Runtime/backend/route/cache lookup, boxing, comparator/object array, or per-
  invocation, per-slice, per-merge, or per-element allocation.
- Eliminate avoidable hot helper calls. Dense heap-array classes should normally have no method
  reference. General `MemorySegment` forms may reference only the precise predefined native-order
  primitive layout accessors and any closed, explicitly justified primitive bit conversion that
  cannot be emitted directly. Repeated `ValueLayout`/byte-order construction is forbidden.

### Exact stable merge and comparison behavior

- Preserve one bottom-up stable merge over two assigned primitive INT64 index regions. Initialize
  the source region with increasing logical-axis indices for every slice. For equal comparison
  classes, including every NaN pair, choose the left/logically earlier index.
- FLOAT64, FLOAT32, and BFLOAT16 place every NaN after every non-NaN in both directions. NaN sign,
  payload, and signaling state do not change comparison, but selected values retain their exact
  represented bits.
- Non-NaN floating values use the requested numerical direction. Negative zero precedes positive
  zero ascending and follows it descending. Preserve infinities, subnormals, and equal finite
  values exactly.
- INT32 and INT64 use signed order. BOOL uses canonical `false < true`. Descending reverses only
  the non-NaN value order and never reverses logical-index stability.
- `SORT` writes represented values in complete stable order. `ARGSORT` writes zero-based INT64
  axis coordinates. `TOP_K` selects the first `k` pairs from the complete requested stable order,
  writes values then indices from one invocation, and does not substitute partial selection.
- Sorted TOP_K retains stable value order. Unsorted TOP_K preserves the selected set and orders it
  by increasing original logical-axis index using the completed deterministic algorithm. Do not
  change to unstable selection, heap selection, quickselect, radix sort, sorting network, library
  sort, comparator sort, or another algorithm in this correction.
- Copy exact represented bits from the selected input coordinate. Do not canonicalize NaNs,
  normalize signed zero, numerically convert BFLOAT16, widen/narrow integral values, or derive
  output indices from physical addresses.

### Dense and general access forms

- Dense heap-array forms use `DENSE_HEAP_ARRAY_INT`. Narrow universal bounds, bases, extents,
  output counts, scratch-region positions, and every proved array index once before hot work, then
  keep primitive integer slice, merge, comparison, and output-address state without repeated
  `l2i`, exact-arithmetic helpers, rank decisions, division, or modulo where the dense geometry
  makes them avoidable.
- Generate typed `GENERAL_LONG` forms for every supported `MemorySegment`, mixed-carrier,
  arbitrary non-negative layout, zero-stride input, offset/strided injective output, large range,
  or otherwise unproved case. Retain only checked long coordinate/address arithmetic required by
  the selected mapping and access plans.
- A failed dense proof selects the general typed form; it never narrows CPU 0006C support. Input,
  values output, and index output may independently use their exact primitive array or a
  native-order `MemorySegment`.
- Direct scratch `MemorySegment` access is permitted and required. Use the existing assigned
  native-order INT64 layout and exact disjoint two-region contract without constructing layouts or
  byte-order variants repeatedly.
- Scalar and parallel-scalar prepared plans reuse the same compatible scalar artifact. All
  coordinate, merge, and scratch state is invocation-private; generated classes retain no mutable
  run state.

### Completed semantics, resources, validation, and lifecycle

- Preserve exactly one lowered unit and one generated artifact; boundaries `[input, values]` for
  SORT, `[input, indices]` for ARGSORT, and `[input, values, indices]` for TOP_K; one exact
  run-owned workspace; no materialization; and scalar or parallel-scalar strategy.
- Preserve `Geometry.workspaceBytes(selectedRangeCount)` and its two axis-extent INT64 regions per
  selected range, requirement ID `0`, INT64 alignment, shared assignment, per-range offset, run
  ownership, cleanup, and accessibility. Scratch never becomes a graph buffer or persistent
  resource and never escapes the run.
- Preserve complete physical overlap rejection for every input/output pair and the TOP_K output/
  output pair before scratch mutation, output write, generated invocation, or worker submission.
- Preserve carrier type, size, byte alignment, accessibility, read-only/writability, complete
  referenced-span, assignment, and range validation. Preserve arbitrary legal disjoint ranges,
  complete-slice worker ownership, and scalar/parallel-scalar determinism.
- Preserve empty selected axes, empty outer dimensions, `k == 0`, singleton axes, `k == extent`,
  all-equal values, all-NaN values, duplicates, every axis, every legal layout/carrier combination,
  and failure timing established by CPU 0006C.
- Preserve lowering, route, capability, cache, specialization, preparation, finalization,
  executable, and independent reference contracts unless an allowlisted Javadoc/package summary
  must be finalized to describe the embedded body accurately.
- CPU analysis remains deterministic and measurement-free. Finalization generates or reuses the
  selected class only after shared slot assignment. Runtime receives an immutable prepared
  executable and performs no ordering interpretation, generation, specialization, lookup,
  fallback, tuning, or benchmark-driven mutation.

### Generated compatibility transition

- Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly from `26` to `27`. Schema 27 is the first
  compatibility version whose ordering classes embed carrier-, type-, family-, direction-,
  access-, comparison-, output-, and loop-specialized stable-merge bodies.
- Treat schema 26 as an incompatible safe miss. Extend the artifact-store regression with an
  otherwise valid schema-26 envelope and current metadata/class bytes, prove it is not loaded,
  then prove lookup regenerates and publishes schema-27 bytes. Add no migration reader, converter,
  compatibility alias, or legacy bridge.
- Preserve structural identity for family, represented type, direction/sorted flags, boundary
  roles/types/access/carriers, scratch-bearing entry shape, and dense/general code shape. Preserve
  compatible shape-polymorphic reuse: concrete extents, axis, `k`, offsets, stride magnitudes,
  scratch sizes/offsets, range count/bounds, carrier instances, slots, workers, graph/run identity,
  and artifact root remain cold.

### Stable automated semantic and Class-File evidence

- Add Java Class-File model assertions for all three families, all six represented types,
  ascending/descending or smallest/largest, sorted/unsorted TOP_K, one/two outputs, every exact
  heap-array form, all-`MemorySegment`, mixed carriers, dense integer and general long forms,
  offset/strided/zero-stride layouts, arbitrary legal ranges, and scalar/parallel artifact reuse in
  proportion to risk.
- Assert direct typed descriptors, embedded initialization/merge/comparison/output loops, exact
  scratch access, values/index output roles, and dense/general addressing. Reject
  `CpuOrderingEmitter.execute` and equivalent bridges, `Object` descriptors/casts,
  `DataType.values`, generic family/type/carrier dispatch, allocations, boxing, comparator/library
  sorting, reflection, map/string dispatch, graph/Runtime/backend/cache references, and avoidable
  helper calls through Java Class-File model inspection rather than source-text matching.
- Use a closed explicit member-reference allowlist for each representative class. Dense classes
  must reject unexpected member references; segment forms may admit only the precise carrier/
  scratch primitives and explicitly justified bit conversions.
- Retain complete `javap` only as audit evidence. Durable regression gates use the Java Class-File
  API and stable structural assertions, not absolute bytecode offsets or planning-observed sizes.
- Preserve or extend independent semantic differentials against `CpuScalarReferenceKernel` or a
  test-local oracle that does not call/share the generated emitter implementation. Cover every
  type/family/direction/mode, raw NaN payloads/signs, signed zeros, infinities, duplicates, logical
  stability, exact selected bits, INT64 indices, general layouts, heap/segment/mixed carriers,
  partial ranges, exact scratch isolation, empty/edge cases, overlap/failure timing, and scalar/
  parallel bitwise parity.

### Independent fixed performance cases

Use two independent cases. Each uses dense heap-array FLOAT32 input/output carriers, contiguous
layouts, shape `[64,1024]`, normalized axis `1`, 64 complete slices, the complete slice range
`[0,64)`, and one range's existing 16,384-byte two-region INT64 `MemorySegment` scratch.

Initialize each input once per fork from global ordinal `i`, `slice = i / 1024`, and
`position = i % 1024`:

- let `selector = (position + 17 * slice) % 127`;
- selector `0` is a quiet NaN constructed from raw bits
  `0x7fc00000 | (((31 * position + slice + 1) & 0x003fffff))`, with alternating sign by slice;
- selector `1` is raw negative zero and selector `2` is raw positive zero; and
- every other value is `(float) ((((37 * position + 19 * slice) % 257) - 128) * 0.25)`.

This deterministic pattern contains repeated finite values, multiple raw NaN payloads/signs,
both signed zeros, and stable ties in every representative run. Verify the exact raw-bit outputs,
logical indices, unchanged input, and scratch bounds before and after timing.

#### Case 1: stable ascending SORT

- Operation: `SORT`, shape `[64,1024]`, axis `1`, `descending == false`.
- Output: FLOAT32 `[64,1024]` values.
- The direct body must use the same two scratch regions, increasing logical-index initialization,
  stable bottom-up merge widths, left-on-equality FLOAT32 comparison contract, slice traversal,
  and represented output copy as the generated body. It may use primitive locals and direct array/
  scratch access only.

#### Case 2: sorted-largest two-output TOP_K

- Operation: `TOP_K`, input shape `[64,1024]`, output shapes `[64,64]`, axis `1`, `k == 64`,
  `largest == true`, and `sorted == true`.
- Outputs: FLOAT32 values then INT64 logical-axis indices.
- The direct body must perform the same complete stable merge of every 1024-element slice, select
  the first 64 pairs, and write both outputs in the same order with the same scratch/traversal
  semantics. It must not use partial selection, a heap, quickselect, library sorting, precomputed
  order, or omit either output.

### Reproducible performance protocol and gate

- Use one standalone direct-Java probe outside Gradle and JUnit. Run at least five fresh JVM forks
  with fixed `-Xms1g -Xmx1g`. Each fork runs at least five warmup batches and nine measurement
  rounds per case, with adaptive repetition counts making each timed sample at least 25 ms.
- Randomize case order and generated/direct order within every batch with deterministic seed
  `0x5A17D00D7A0E27L`, reset identically in each fork. The summarizer reports each case separately.
  Consume an observable sink only outside timed regions.
- Report generated/direct medians and ratio for every fork of each case. For each case, compute the
  aggregate as the median of its generated per-fork medians divided by the median of its direct
  per-fork medians. Every fork ratio and the aggregate ratio for SORT must be `<= 1.15x`; every
  fork ratio and aggregate for TOP_K must independently be `<= 1.15x`.
- Never average forks, drop an outlier, allow one case or the aggregate to hide a failing fork,
  widen the threshold, change the fixed cases/pattern after baseline, or substitute another
  direction, `k`, mode, shape, range, algorithm, carrier, or scratch contract.
- Use identical probe/summarizer source, data, direct bodies, commands, environment, heap,
  warmup/round/fork/adaptive protocol, seed, verification, and sink rules for baseline and final
  evidence. Record source SHA-256 and generated-class SHA-256 for both phases.
- Performance evidence is observational and must not select or mutate ordinary preparation or
  Runtime behavior. If either case has one failing fork or aggregate, keep implementation status
  `In progress` or `Review needed`, inspect generated/JIT shape, and correct the implementation.
- ARGSORT, every represented type, descending/smallest order, unsorted TOP_K, edge cases, general
  layouts, `MemorySegment`/mixed carriers, arbitrary ranges, and parallel-scalar forms remain in
  semantic and Class-File gates. Add another timed case only if implementation evidence identifies
  a concrete risk that those stable gates cannot cover; do not weaken or replace either fixed gate.

## Out of scope

- New Model semantics, operations, attributes, types, Shapes, ordering policies, capability rows,
  public APIs, or gradient behavior
- Any ordering algorithm redesign, partial TOP_K selection, unstable/custom comparison, vector or
  native sorting, runtime algorithm choice, fixed-shape specialization, unrolling, or relaxed
  numerical policy
- Pointwise, affine/movement, indexing, scatter, fold, random/dropout, scan, aggregate, reduction,
  normalization, linear algebra, native-provider, or later family implementation
- New routes, workspace kinds, scratch declaration/lifecycle, materialization, representation
  policy, tuning controls, tuning-cache behavior, public configuration, or Runtime mutation
- Architecture, module/dependency boundary, shared Model/Compiler/Planning/Prepare/Runtime, Config,
  Trace, Engine, other backend, NN, Training, Gradle/build, Java version, architecture-test,
  backend-conformance, integration-test, or unrelated documentation changes
- Detailed specifications or implementation for CPU 0007A0F, CPU 0007A1, or any later task
- Commit, push, staging, revert, deletion, or modification of unrelated concurrent work

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Runtime/Prepare/Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [CPU 0006C portable stable ordering and selection](0006c-portable-stable-ordering-and-selection.md)
- [CPU 0007A0 generated hot-path parity correction](0007a0-generated-hot-path-parity-correction.md)
- [CPU 0007A0A affine and movement parity](0007a0a-affine-and-movement-generated-loop-parity.md)
- [CPU 0007A0B indexing parity](0007a0b-indexing-generated-loop-parity.md)
- [CPU 0007A0C scatter parity](0007a0c-scatter-generated-loop-parity.md)
- [CPU 0007A0D fold parity](0007a0d-fold-generated-loop-parity.md)

## Architecture constraints

- Model owns stable ordering meaning. This task changes only CPU-private generated realization and
  cannot reinterpret, narrow, or extend that semantic contract.
- Planning selects CPU ownership only. CPU analysis owns deterministic lowering, specialization,
  strategy, and exact declarations; CPU finalization owns compatible generation/reuse after shared
  assignment; Runtime executes the immutable prepared result only.
- Runtime hot code receives typed direct carriers, the assigned scratch segment, and primitive
  cold geometry. No `Operation`, `CompiledNode`, Tensor, graph object, backend discovery,
  validation decision, route/cache policy, or benchmark evidence reaches it.
- Prepared recipes and generated artifacts remain immutable and reusable. Concurrent runs use
  distinct `RunState` objects; every active range has disjoint output slices and assigned scratch.
- Complete CPU 0006C semantics, general layouts/carriers/ranges, exact scratch ownership,
  deterministic scalar/parallel behavior, and the Runtime/Prepare boundary remain unchanged.
- Stop if parity requires a public/shared contract, architecture, dependency, semantic, route,
  resource, materialization, Runtime-policy, algorithm, or broader-module change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — embedded stable-merge,
  comparison, output, typed carrier, scratch, and dense/general Class-File emission.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema and artifact compatibility.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — existing family, represented type,
  direction/output-order, boundary-access, and scratch-policy identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — existing validated logical-slice,
  layout, axis, `k`, and exact scratch geometry.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — existing deterministic declarations,
  specialization, strategy, workspace policy, and final assignment validation.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — existing carrier/overlap binding,
  geometry packing, range partitioning, scratch slicing, and worker invocation.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent semantic oracle only.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — current generated-route and
  artifact lifecycle documentation.

Packages added or changed:

- No package is added, removed, moved, exported, or made supported API.

Type placement:

- `CpuOrderingEmitter` remains the sole owner of generated stable merge, comparison classes,
  selected-pair ordering, and values/index stores. It becomes an embedded-body emitter rather
  than a generic runtime bridge owner.
- `CpuClassFileKernelGenerator` passes existing structural ordering IR to the focused emitter; it
  does not acquire ordering semantics, packed geometry, scratch ownership, or Runtime state.
- `CpuCarrierEmitter` may supply generation-time typed load/store instructions but does not own
  ordering comparison, merge, output-role, or resource policy.
- `CpuOrderingIr`, `CpuOrderingLowering.Geometry`, preparation, executable, and reference owners
  retain their current responsibilities. Add no dispatcher, registry, comparator abstraction,
  manager, broad helper, or facade.

## Affected files

Exact implementation allowlist:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrderingEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrderingGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0007a0e-ordering-generated-loop-parity.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Existing `CpuOrderingIr`, `CpuOrderingLowering`, `CpuPartitionPreparationPlan`,
`CpuPreparedExecutable`, `CpuScalarReferenceKernel`, their focused tests, build files,
architecture documents, and shared modules are inspection-only unless the task is stopped and
replanned.

## Maximum scope

This task may modify at most 18 repository paths, all drawn from the exact allowlist above. It may
create no production or test type and no later task specification. Temporary probe/evidence files
must remain within the one fresh isolated `/private/tmp` directory and are not repository paths.

If implementation requires any non-allowlisted repository path, more than 18 changed paths, a new
type, or a broader contract, stop before editing it and request a planning update. Do not consume
the allowlist with unrelated cleanup.

## Acceptance criteria

- Current schema-26 source and dense/general generated classes are freshly retained and audited
  before production edits with exact source/commands/environment, descriptors, bytes, SHA-256,
  complete `javap`, member references, semantics, and raw timing.
- Every admitted SORT, ARGSORT, and TOP_K specialization embeds typed stable merge, comparison,
  optional selected-index ordering, and correct one/two-output work with no generic bridge.
- Dense heap-array forms use cold-proved primitive integer state; every admitted segment, mixed-
  carrier, arbitrary-layout, zero-stride, strided-output, large-range, or unproved form retains
  correct typed general-long execution.
- NaN-last, directional signed zero, infinities, signed integral/BOOL order, left-on-equality
  logical stability, exact represented bits, INT64 logical indices, sorted/unsorted TOP_K, empty
  and edge behavior, partial ranges, and scalar/parallel determinism are independently proved.
- Existing capability, lowering validation, ordered boundaries, exact assigned two-region scratch,
  no materialization, carrier/overlap/assignment checks, range ownership, and immutable lifecycle
  remain unchanged.
- Schema advances exactly from 26 to 27; schema-26 persistence is an incompatible safe miss that
  regenerates schema-27 bytes; concrete invocation geometry remains shape-polymorphically reusable.
- Stable Java Class-File assertions cover the required family/type/direction/mode/carrier/access/
  output/range matrix and reject `Object`, generic dispatch, allocation, forbidden lookup, generic
  bridge, library sort, and avoidable helper references.
- Every SORT fork and its aggregate independently pass `<= 1.15x`; every two-output TOP_K fork and
  its aggregate independently pass `<= 1.15x`, using unchanged equivalent direct bodies and exact
  pre/post verification.
- Baseline and final evidence remain under one fresh isolated `/private/tmp` directory and include
  all required source, commands, environment, class files, decompilation, checksums, member
  references, semantic results, raw samples, and summaries.
- Only allowlisted paths change, no more than 18 repository paths change, no GSD artifact exists,
  and no CPU 0007A0F or later detailed task specification exists.
- A distinct clean documentation-focused context independently reviews final source, tests,
  Class-File evidence, and performance evidence; finalizes affected Javadocs/package summaries,
  CPU guide, glossary impact, this task, master plan, and roadmap; and records reasoned no-change
  conclusions before completion.

## Tests / validation

Run focused tests for every changed test owner while iterating. The final focused command must name
every changed test class and cover stable Class-File shape, all three families and six types,
dense/general and heap/segment/mixed carriers, both directions, TOP_K modes, raw special values,
arbitrary layouts/ranges, exact scratch isolation, independent differentials, scalar/parallel
reuse, preparation/finalization, specialization, and schema/persistence compatibility.

After executable Java stabilizes, run exactly one authoritative affected-module command:

```bash
./gradlew :backends:cpu:test
```

Run the isolated baseline/final probe separately from Gradle/JUnit with the exact five-fork,
two-case protocol. Timing is retained observational evidence, not an ordinary unit test.

The distinct clean documentation pass reuses successful Java and timing evidence unless it
changes executable Java/probe behavior or records a concrete stale-evidence reason. It runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also inspects rendered affected Javadocs and validates local Markdown links and heading
anchors, balanced fences, terminal newlines, carriage returns, trailing whitespace, exact changed-
path allowlist/count, package/type placement, schema-27 history and schema-26 rejection, generated
descriptors and member-reference allowlists, retained baseline/final source/checksums/protocol/
ratios, task/master/roadmap status and dependency order, absence of later specs, preservation of
concurrent scope, and absence of GSD artifacts.

Repository-wide, architecture, backend-conformance, and integration validation remains deferred
to CPU 0009 or continuous integration because this task changes one concrete backend realization
and no shared or architecture contract. Stop and replan if implementation makes that conclusion
false.

## Dependencies

- [CPU 0006C](0006c-portable-stable-ordering-and-selection.md) is `Complete` and owns the exact
  current ordering semantics, type/family matrix, stable merge and scratch policy, layouts,
  carriers, ranges, validation, reference oracle, worker safety, and schema-18 history preserved
  here.
- [CPU 0007A0](0007a0-generated-hot-path-parity-correction.md) is `Complete` and supplies shared
  typed carrier emission, dense integer proof, stable Class-File testing, and the retained
  five-fork evidence pattern.
- [CPU 0007A0A](0007a0a-affine-and-movement-generated-loop-parity.md) is `Complete` and establishes
  the dense/general correction boundary and schema-23 history.
- [CPU 0007A0B](0007a0b-indexing-generated-loop-parity.md) is `Complete` and establishes embedded
  mixed-carrier indexed mapping without generic bridges and schema-24 history.
- [CPU 0007A0C](0007a0c-scatter-generated-loop-parity.md) is `Complete` and establishes embedded
  output/contribution bodies with declared scratch and schema-25 history.
- [CPU 0007A0D](0007a0d-fold-generated-loop-parity.md) is `Complete` and establishes current schema
  26, direct predefined segment layouts, and embedded typed mapping/addition bodies.
- Current Java 26 Class-File API, shared Prepare/Runtime, CPU worker, direct carrier binding,
  scratch assignment, and generated-artifact store contracts are complete and unchanged.

## Follow-up tasks

- CPU 0007A0F remains the next ordered `Draft` random/dropout correction without a detailed
  specification. Do not create it in this task.
- CPU 0007A1 remains `Draft` and explicitly depends on CPU 0007A0F before semantic-family
  expansion resumes.
- CPU 0009 retains the generated-coverage and conformance checkpoint.

## Architecture impact

Expected impact: None.

The task changes only CPU-owned generated ordering code shape and current-only artifact
compatibility under the existing analysis/finalization/binding lifecycle. If implementation
requires an architecture, public/shared contract, module, dependency, semantic, algorithm, route,
resource, materialization, or Runtime-policy change, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Implement Synaptik CPU task 0007A0E exactly from
docs/planning/backends/cpu/tasks/0007a0e-ordering-generated-loop-parity.md. Do not use GSD. Do not
commit, push, stage, revert, delete, or modify unrelated work.

Read in full AGENTS.md, ARCHITECTURE.md, the directly relevant architecture/performance/Runtime-
Prepare-Backend documents, documentation rules and applicable profiles, planning guide, roadmap,
CPU master plan, task 0007A0E, completed tasks 0006C/0007A0/0007A0A/0007A0B/0007A0C/0007A0D,
and every affected or directly relevant ordering IR/lowering/route/specialization/generator/
emitter/geometry/scratch/preparation/finalization/executable/reference/test/cache/schema/build file.
Inspect the complete worktree and index first and preserve concurrent unrelated changes.

Capture the required fresh schema-26 source/Class-File/semantic/performance baseline under one
isolated /private/tmp directory before production edits. Implement the exact typed dense/general
SORT/ARGSORT/TOP_K bodies, schema-27 compatibility, stable Class-File and semantic tests, focused
validation, one authoritative CPU suite, and both fixed five-fork per-fork/aggregate parity gates.
Stop on any architecture, scope, semantic, shared-contract, algorithm, route, resource, lifecycle,
or unresolved performance conflict.

After executable Java and probe evidence stabilize, hand the same uncommitted diff and exact
evidence to a distinct clean documentation-focused context following
docs/developer-guide/documentation-rules.md. That context must independently inspect final source,
tests, generated class shape, and evidence; finalize affected Javadocs/package summaries, CPU
guide, glossary impact, task/master/roadmap, rendered pages, Markdown, exact scope, schema,
status/order, and performance evidence; and not repeat successful Java or timing runs unless it
changes executable/probe behavior or records a concrete stale-evidence reason. Do not mark
Complete until both contexts and every acceptance criterion succeed.

Return exact changed paths, commands/results, environment, class descriptors/sizes/SHA-256/member
references, every SORT and TOP_K per-fork/aggregate ratio, documentation/Javadoc/glossary impact
and reasoned no-change conclusions, context IDs, unresolved issues, and exactly Status: Complete
or Status: Incomplete with Follow-up required: <specific follow-up>.
```

## Local decisions

- Keep SORT, ARGSORT, and TOP_K together because they share one bridge-only emitter, one stable
  comparison and bottom-up merge contract, one assigned scratch policy, one/two-output boundary,
  and one schema transition. Splitting them would leave current ordering families behind the same
  defect.
- Advance exactly to schema 27 because generated ordering bodies change while schema 26 already
  identifies embedded fold bodies. A schema-26 bridge-only ordering artifact must never satisfy a
  schema-27 direct-loop request.
- Use `[64,1024]` axis-1 FLOAT32 cases to exercise repeated complete slices, stable comparison,
  scratch reuse, and output work without fixed-shape production specialization. Use full ascending
  SORT and sorted-largest TOP_K with `k == 64` as independently gated one-output and two-output
  cases.
- Require both every per-fork ratio and each case's aggregate to pass `<= 1.15x`; aggregate success
  or the other benchmark cannot conceal one failing process or case.
- Keep ARGSORT, all other types/directions/modes, general layouts, segment/mixed carriers, edge
  cases, arbitrary ranges, and parallel forms in stable semantic/Class-File gates. The two named
  performance claims do not generalize beyond their exact cases.
- Direct scratch segment access uses the existing predefined native-order INT64 layout and assigned
  two-region lifecycle; no resource or algorithm redesign is authorized.

## Known limitations

- The timing gates cover only dense heap-array FLOAT32 ascending SORT and sorted-largest two-output
  TOP_K on the recorded environment. They make no parity claim for ARGSORT, other types,
  directions, unsorted TOP_K, general layouts, segment/mixed carriers, parallel orchestration,
  every CPU, or every JVM.
- General and unmeasured forms receive stable semantic and Class-File validation but no numerical
  performance threshold in this task.
- Only the existing one-node fully static resolved-layout CPU ordering capability is corrected.
  Dynamic layouts, custom ordering, vector/native sorting, multi-node fusion, and new semantics
  remain unsupported or deferred as before.
- The completed stable bottom-up merge and full-order TOP_K algorithm remain intentionally
  unchanged. This generated-shape task does not claim optimal selection complexity.

## Validation evidence

Planning context `01a018e2-75c0-7272-b0b9-1ebb6ae7f856` inspected the clean worktree, current
schema-26 source, ordering IR/lowering/geometry/emitter/generator/specialization/preparation/
finalization/executable/reference/tests/cache/build contracts, completed task history, and the
four generated Class-File forms recorded in Scope. `./gradlew :backends:cpu:testClasses` passed
with 11 tasks up-to-date. Complete `javap -c -p` and `javap -v -p` confirmed typed direct entry
descriptors followed by the sole generic bridge member reference. This evidence fixes the
allowlist and justifies schema 27 but must be reproduced as a fresh retained implementation
baseline before production edits.

Implementation validation evidence:

- The clean implementation context retained all evidence under
  `/private/tmp/synaptik-cpu-0007a0e-implementation.dMNFPD`. The environment was OpenJDK
  26.0.1+8-34 on macOS 26.5.2 arm64; context ID
  `01a018ec-ea40-7f21-b2be-1ec07564d0b8`. `OrderingPerformanceProbe.java` has SHA-256
  `efc15fecc64380adad13999cfda62454e07040d86d2738cdbd34a3f5f2c72e3e`; the retained
  summarizer has SHA-256 `2ee59c1c88e95ddb2ca96f850b88a2ce198ebbd984ae43b5a0adafceb66226e8`.
- Fresh schema-26 baseline classes retained their typed entry descriptors and bridge-only shape:
  dense FLOAT32 SORT was 489 bytes, SHA-256
  `55c2a18352b72827351a1a87476b4075acf3fdf36bb0c93c615c82d7933b0d7a`; dense FLOAT64
  ARGSORT was 489 bytes, `2e7f10e26fa71e0aae7f71880624bba36552f5f70a246b9dcc957c2f771b4e88`;
  dense FLOAT32 TOP_K was 492 bytes,
  `e0d932fd29343fc7aedf703266a4eb89ee55b5973aeaeedc3996ca74b823f3b5`; and mixed BFLOAT16
  TOP_K was 554 bytes, `62a7366e1a4fb101822e166c2aec1888eaba13a04e4aa871c79b6dc308138bed`.
  Each had only the generic `CpuOrderingEmitter.execute(Object,Object,Object,...)` method
  reference. Baseline SORT fork ratios were `5.000745397x`, `4.888217534x`, `4.937615787x`,
  `4.986715932x`, and `4.936871843x`, aggregate `4.974047838x`; TOP_K ratios were
  `4.470562353x`, `5.060709708x`, `4.550869282x`, `4.815527022x`, and `4.991654916x`,
  aggregate `4.815527022x`.
- Final schema-27 descriptors are unchanged. Dense FLOAT32 SORT is 2,924 bytes, SHA-256
  `5451f7139693a48092c538a8c158ec4dbd2ed86ef900861c9a325051c3cecdaf`; dense FLOAT64
  ARGSORT is 2,878 bytes, `1a7d36900b6a198db205d8deb7f09e537fb51fc0e1f06f570f43167f8f766893`;
  dense FLOAT32 TOP_K is 3,485 bytes,
  `5ed4abf07e0d1977744c6e27d5dc7bb9d30e3804b2d7e3fa3433871101fba9ca`; and mixed BFLOAT16
  TOP_K is 4,968 bytes, `ec8cc402609ea0298a726d7e5b6d1b5ce5227b87a50236353b6647c9e3b32334`.
  Complete retained `javap -c -p` and `javap -v -p` show direct loops and no bridge or Object
  descriptor. Dense member references are exactly `ValueLayout.JAVA_LONG` plus
  `MemorySegment.get/set` for assigned scratch. The mixed class additionally references exact
  predefined SHORT/LONG unaligned layouts and the justified `Float.intBitsToFloat` BFLOAT16
  conversion.
- Final SORT fork ratios were `1.092722590x`, `1.077727976x`, `1.090165540x`, `1.049603676x`,
  and `1.089690637x`, aggregate `1.090165540x`. Final TOP_K fork ratios were `0.962180009x`,
  `0.969660236x`, `0.936075945x`, `1.018861174x`, and `1.020688132x`, aggregate
  `0.965951020x`. Every independent fork and aggregate passed `<= 1.15x` with unchanged source,
  data, direct bodies, heap, seed, warmup, measurement, adaptive, verification, and sink rules.
- The final focused command named all five changed test owners and passed. The single authoritative
  `./gradlew :backends:cpu:test` passed 320 tests with 1 expected skip and zero failures/errors.
  Class-File and semantic tests cover all types/families/directions/TOP_K modes, typed descriptors,
  dense/general forms, heap/segment/mixed carriers, zero-stride and offset/strided layouts,
  partial ranges, represented special values, scalar/parallel lifecycle, and schema-26 rejection.
  No Java suite or timing probe was repeated by the documentation-only finalization.
- Mandatory clean review/fix context `01a01905-b866-70f3-89c7-181adcabe2c5` independently audited
  the executable diff and refactored the compressed lower half of `CpuOrderingEmitter` plus dense
  touched-test setup into normal readable Java without changing the emitted Class-File builder
  sequence. The focused five-owner command passed, followed by exactly one authoritative
  `./gradlew :backends:cpu:test`; the latter again passed 320 tests with 1 expected skip and zero
  failures/errors. The retained `BaselineClasses.java` harness regenerated all four representative
  classes under `review-final/`: every descriptor, size, SHA-256 value, complete `javap -c -p`,
  complete `javap -v -p` body, and closed member-reference allowlist exactly matched the retained
  final evidence. Generated bytes were therefore unchanged, so the accepted five-fork SORT and
  TOP_K timing evidence above was reused rather than rerun. Final allowlist, later-specification,
  GSD, index, and whitespace checks passed.
- Mandatory clean documentation context `01a01914-1a96-7cc3-9215-c69a7531f30e` independently
  reviewed the final source, every changed Java test, schema transition, and retained Class-File
  and timing evidence. It finalized the schema-27 CPU guide, confirmed the provisional Java and
  package Javadocs against the implementation without changing executable Java, and inspected
  every affected rendered Javadoc page after the exact CPU Javadoc command passed. Local Markdown
  links and anchors, headings, code fences, final newlines, whitespace, schema/status/order,
  the exact 18-path allowlist, changed-path containment, absence of a later 0007A0F specification,
  absence of GSD artifacts, and the unstaged index all passed. `git diff --check` passed.
- Glossary review required no edit: schema 27 changes the private generated implementation and
  compatibility version, but introduces no new user-facing term or synonym and changes none of
  the existing stable-ordering semantics. Architecture, ADRs, build files, module boundaries,
  capabilities, public APIs, backend conformance, integration tests, Compiler, Runtime, Engine,
  gradients, related ordering/lowering/geometry/preparation/finalization/executable contracts,
  and other modules likewise required no documentation or implementation change.

## Implementation notes

Schema 27 now embeds typed stable merge, comparison, unsorted selected-index ordering, and
represented output stores. Existing lowering geometry, two-region assigned scratch, preparation,
finalization, executable binding, overlap validation, range ownership, and reference contracts
were preserved without changes. The clean documentation context independently confirmed these
implementation claims against the final source and retained evidence.

## Completion summary

Schema 27 now embeds typed direct stable-ordering loops for SORT, ARGSORT, and TOP_K while
preserving CPU 0006C semantics, exact two-region scratch, validation, carrier/layout coverage,
parallel range ownership, and lifecycle. Both clean executable contexts and the clean
documentation context completed their required reviews. The focused five-owner tests and the
single final CPU suite passed; all retained Class-File, member-reference, compatibility, and
five-fork performance gates passed. The CPU guide and planning state are synchronized, the
glossary no-change conclusion is recorded, and CPU 0007A0F is the sole next ordered Draft frontier
with CPU 0007A1 explicitly behind it.

Status: Complete
