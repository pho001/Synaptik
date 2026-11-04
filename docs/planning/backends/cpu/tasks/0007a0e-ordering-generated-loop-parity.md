# Task 0007A0E: Ordering Generated-Loop Parity

## Reader context

The stable ordering algorithm and its workspace did not change here; its comparison, merge,
selection, and stores moved into generated typed bodies. CPU lowering still described complete
logical-axis slices, and shared assignment still supplied two INT64 merge-index regions per
executing range. Those scratch indices named source positions, while value stores preserved the
selected representations.

The emitter specialized on known type/family/carrier facts before execution. Runtime binding
supplied current storage and whole-slice ranges; it did not reinterpret SORT or choose a
different merge algorithm on each element. The local FLOAT32 comparisons and schema-27 receipts
remain historical. The current sorting reference clarifies stability and TOP_K value/index
pairing, and the generator explanation locates the realization work.

Current reading alongside this historical record:

- [Stable sort and argsort semantics](../../../../reference/tensor/sort-topk.md#stable-sort-and-argsort-expressions)
  — explains logical indices, equal-value stability, NaN placement, and represented-bit selection.
- [CPU generated-code ownership](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit)
  — separates generation-time emitters, emitted typed work, and JVM compilation.

## Status

Complete

## Goal

Replace the bridge-only generated comparison, merge, selection, and output work for the completed
[CPU](../../../../glossary.md#term-cpu) stable-ordering family with [carrier](../../../../glossary.md#term-storage-carrier)-, represented-type-, family-, access-, comparison-,
output-, and loop-specialized bodies embedded in each generated class. `SORT`, `ARGSORT`, and
`TOP_K` must preserve every CPU 0006C semantic, [layout](../../../../glossary.md#layout), carrier, range, resource, parallelism,
[validation](../../../../glossary.md#term-validation), and [lifecycle](../../../../glossary.md#lifecycle) contract while two independently gated dense [heap](../../../../glossary.md#term-heap)-array [FLOAT32](../../../../glossary.md#term-float32) cases
reach reproducible near-parity with equivalent direct primitive stable-merge implementations.

This is one cohesive CPU-private generated-code correction. The three families share
`CpuOrderingIr`, compact cold geometry, `CpuOrderingEmitter`, stable bottom-up merge over two
assigned [INT64](../../../../glossary.md#data-type--datatype) scratch regions, exact comparison classes, and one generated-[artifact](../../../../glossary.md#term-artifact) compatibility
boundary. The task changes generated code [shape](../../../../glossary.md#shape) only; completed CPU 0006C capability, [operation](../../../../glossary.md#operation)
semantics, algorithm, scratch declaration, and observable behavior remain unchanged.

## [Scope](../../../../glossary.md#term-scope)

### Mandatory pre-edit source and Class-File baseline

- Before changing production Java, create one fresh isolated directory under `/private/tmp` and
  capture the current schema-26 source, generated classes, semantics, and raw timing for the two
  exact performance cases defined below.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Also capture representative current classes for dense ARGSORT and a general-layout or mixed
  heap-array/<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment) ordering form so the baseline covers every output role and both loop-
  addressing categories before implementation chooses locals or control flow.
- Retain the probe and summarizer source; exact compile, generation, decompilation, execution, and
  summary commands; complete environment facts; generated class bytes; complete <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [decompilation glossary](../../../../glossary.md#term-decompilation))</span>`javap -c -p`
  and `javap -v -p`; entry descriptors; class sizes; SHA-256 values; constant-pool member
  references; semantic verification; raw samples; per-fork summaries; and aggregate summaries.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- Reproduce baseline evidence freshly. [Planning](../../../../glossary.md#term-planning) audit context
  `01a018e2-75c0-7272-b0b9-1ebb6ae7f856` observed these [schema](../../../../glossary.md#term-schema)-26 classes under
  `/private/tmp/synaptik-cpu-0007a0e-planning.FVb4Eg`:
  - dense ascending [FLOAT32](../../../../glossary.md#term-float32) SORT: [descriptor](../../../../glossary.md#term-descriptor)
    `([F[FLjava/lang/foreign/MemorySegment;[JJJ)V`, 489 bytes, [SHA-256](../../../../glossary.md#term-sha-256)
    `55c2a18352b72827351a1a87476b4075acf3fdf36bb0c93c615c82d7933b0d7a`;
  - dense descending [FLOAT64](../../../../glossary.md#term-float64) ARGSORT: descriptor
    `([D[JLjava/lang/foreign/MemorySegment;[JJJ)V`, 489 bytes, SHA-256
    `2e7f10e26fa71e0aae7f71880624bba36552f5f70a246b9dcc957c2f771b4e88`;
  - dense sorted-largest FLOAT32 TOP_K: descriptor
    `([F[F[JLjava/lang/foreign/MemorySegment;[JJJ)V`, 492 bytes, SHA-256
    `e0d932fd29343fc7aedf703266a4eb89ee55b5973aeaeedc3996ca74b823f3b5`;
  - mixed-[carrier](../../../../glossary.md#term-storage-carrier) unsorted-smallest [BFLOAT16](../../../../glossary.md#term-bfloat16) TOP_K: descriptor
    `(Ljava/lang/foreign/MemorySegment;[SLjava/lang/foreign/MemorySegment;`
    `Ljava/lang/foreign/MemorySegment;[JJJ)V`, 554 bytes, SHA-256
    `4ffc4a2eca673e9ca944c16b448135d2750cd5e3b1193800472c93c0c7819de1`.
- Each audited class has exactly one method reference, to
  `CpuOrderingEmitter.execute(Object,Object,Object,MemorySegment,long[],long,long)`. The helper
  allocates one `int[]` [layout](../../../../glossary.md#layout) table per invocation; reconstructs family, [`DataType`](../../../../glossary.md#data-type--datatype), carrier, and
  access decisions at runtime; performs generic <span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>`Object` reads/writes; and calls comparison,
  address, copy, and index-output helpers inside ordering work. Treat these observations as
  planning evidence only and retain a fresh implementation baseline.
- Time only warmed stable-merge, selected-pair ordering, and output work. Exclude lowering,
  generation, verification, class definition, artifact lookup, preparation/finalization, cold
  binding, geometry packing, carrier/scratch allocation, validation, input initialization,
  expected-result construction, verification, summarization, and sink observation from both
  generated and direct timed forms.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [artifact glossary](../../../../glossary.md#term-artifact); [lookup glossary](../../../../glossary.md#term-lookup); [binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation))</span>

### Embedded specialized ordering bodies

- Change `CpuClassFileKernelGenerator` and `CpuOrderingEmitter` so every generated entry embeds
  initialization of logical indices, stable bottom-up merge passes, selected-output ordering,
  represented value/index writes, and exact carrier access instead of calling
  `CpuOrderingEmitter.execute` or any equivalent generic bridge.
- Pass the existing structural ordering [IR](../../../../glossary.md#term-intermediate-representation-ir) to the focused emitter. Select at generation time:
  - family: `SORT`, `ARGSORT`, or `TOP_K`;
  - represented type: [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), [INT32](../../../../glossary.md#data-type--datatype), [INT64](../../../../glossary.md#data-type--datatype), or [BOOL](../../../../glossary.md#data-type--datatype);
  - direction: ascending/descending for full ordering or smallest/largest for TOP_K;
  - TOP_K output mode: sorted or deterministic increasing-original-index unsorted;
  - ordered input, values-output, and INT64-index-output [carrier](../../../../glossary.md#term-storage-carrier) forms independently;
  - dense [heap](../../../../glossary.md#term-heap)-array integer addressing or typed general long addressing; and
  - one-output versus ordered two-output generated stores.
- Preserve the universal primitive `long start, long end` entry. Bounds continue to identify
  complete logical-axis slices; arbitrary legal disjoint slice ranges remain valid, and one slice
  is never split between ranges.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [slice glossary](../../../../glossary.md#slice))</span>
- Hoist the family, type, direction, sorted flag, boundary ranks/roles, axis role, packed-layout
  positions, scratch-region facts, and other invocation-invariant packed values into primitive
  locals before the slice/merge/output loops. Concrete extents, axis, <span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [invariant glossary](../../../../glossary.md#term-invariant))</span>`k`, offsets, stride
  magnitudes, carrier-relative bases, scratch offsets, ranges, and range index remain cold,
  invocation-private, shape-polymorphic facts.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [shape glossary](../../../../glossary.md#shape))</span>
- Generated hot work must contain no `Object` [descriptor](../../../../glossary.md#term-descriptor) or cast, `DataType.values`, family/type/
  carrier ordinal dispatch, array-versus-segment or carrier-form test, reflection, map/string
  dispatch, graph/Runtime/backend/route/cache lookup, boxing, comparator/object array, or per-
  invocation, per-slice, per-merge, or per-element allocation.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [graph glossary](../../../../glossary.md#graph); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [cache glossary](../../../../glossary.md#term-cache); [lookup glossary](../../../../glossary.md#term-lookup); [element glossary](../../../../glossary.md#term-element))</span>
- Eliminate avoidable hot helper calls. Dense heap-array classes should normally have no method
  reference. General [`MemorySegment`](../../../../glossary.md#term-memorysegment) forms may reference only the precise predefined native-order
  primitive layout accessors and any closed, explicitly justified primitive bit conversion that
  cannot be emitted directly. Repeated `ValueLayout`/byte-order construction is forbidden.

### Exact stable merge and comparison behavior

- Preserve one bottom-up stable merge over two assigned primitive INT64 index regions. Initialize
  the source region with increasing logical-axis indices for every slice. For equal comparison
  classes, including every NaN pair, choose the left/logically earlier index.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [slice glossary](../../../../glossary.md#slice); [NaN glossary](../../../../glossary.md#term-nan))</span>
- FLOAT64, FLOAT32, and BFLOAT16 place every NaN after every non-NaN in both directions. NaN sign,
  payload, and signaling state do not change comparison, but selected values retain their exact
  represented bits.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [represented bits glossary](../../../../glossary.md#term-represented-bits))</span>
- Non-NaN floating values use the requested numerical direction. Negative zero precedes positive
  zero ascending and follows it descending. Preserve infinities, subnormals, and equal finite
  values exactly.<span data-glossary-navigation="true"> ([Negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [subnormals glossary](../../../../glossary.md#term-subnormal); [finite glossary](../../../../glossary.md#term-finite))</span>
- [INT32](../../../../glossary.md#data-type--datatype) and [INT64](../../../../glossary.md#data-type--datatype) use signed order. [BOOL](../../../../glossary.md#data-type--datatype) uses [canonical](../../../../glossary.md#term-canonical-form) `false < true`. Descending reverses only
  the non-NaN value order and never reverses logical-index stability.
- `SORT` writes represented values in complete stable order. `ARGSORT` writes zero-based INT64
  axis coordinates. <span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`TOP_K` selects the first `k` pairs from the complete requested stable order,
  writes values then indices from one invocation, and does not substitute partial selection.
- Sorted TOP_K retains stable value order. Unsorted TOP_K preserves the selected set and orders it
  by increasing original logical-axis index using the completed deterministic algorithm. Do not
  change to unstable selection, heap selection, quickselect, radix sort, sorting network, library
  sort, comparator sort, or another algorithm in this correction.<span data-glossary-navigation="true"> ([axis index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [heap glossary](../../../../glossary.md#term-heap))</span>
- Copy exact represented bits from the selected input coordinate. Do not canonicalize NaNs,
  normalize signed zero, numerically convert BFLOAT16, widen/narrow integral values, or derive
  output indices from physical addresses.<span data-glossary-navigation="true"> ([signed zero glossary](../../../../glossary.md#term-signed-zero))</span>

### Dense and general access forms

- Dense [heap](../../../../glossary.md#term-heap)-array forms use `DENSE_HEAP_ARRAY_INT`. Narrow universal bounds, bases, extents,
  output counts, scratch-region positions, and every proved array index once before hot work, then
  keep primitive integer slice, merge, comparison, and output-address state without repeated<span data-glossary-navigation="true"> ([array index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [slice glossary](../../../../glossary.md#slice))</span>
  `l2i`, exact-arithmetic helpers, rank decisions, division, or modulo where the dense geometry
  makes them avoidable.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>
- Generate typed `GENERAL_LONG` forms for every supported [`MemorySegment`](../../../../glossary.md#term-memorysegment), mixed-carrier,
  arbitrary non-negative layout, zero-stride input, offset/strided injective output, large range,
  or otherwise unproved case. Retain only checked long coordinate/address arithmetic required by
  the selected mapping and access plans.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [stride glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- A failed dense proof selects the general typed form; it never narrows CPU 0006C support. Input,
  values output, and index output may independently use their exact primitive array or a
  native-order <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>`MemorySegment`.
- Direct scratch `MemorySegment` access is permitted and required. Use the existing assigned
  native-order INT64 layout and exact disjoint two-region contract without constructing layouts or
  byte-order variants repeatedly.<span data-glossary-navigation="true"> ([INT64 glossary](../../../../glossary.md#data-type--datatype))</span>
- Scalar and parallel-scalar prepared plans reuse the same compatible scalar artifact. All
  coordinate, merge, and scratch state is invocation-private; generated classes retain no mutable
  run state.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [artifact glossary](../../../../glossary.md#term-artifact); [mutable glossary](../../../../glossary.md#term-mutable); [run state glossary](../../../../glossary.md#run-state--runstate))</span>

### Completed semantics, resources, [validation](../../../../glossary.md#term-validation), and [lifecycle](../../../../glossary.md#lifecycle)

- Preserve exactly one lowered unit and one generated [artifact](../../../../glossary.md#term-artifact); boundaries `[input, values]` for
  SORT, `[input, indices]` for ARGSORT, and `[input, values, indices]` for TOP_K; one exact
  run-owned workspace; no materialization; and scalar or parallel-scalar strategy.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [materialization glossary](../../../../glossary.md#materialization); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Preserve `Geometry.workspaceBytes(selectedRangeCount)` and its two axis-extent INT64 regions per
  selected range, requirement ID <span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>`0`, INT64 alignment, shared assignment, per-range offset, run
  ownership, cleanup, and accessibility. Scratch never becomes a graph buffer or persistent
  resource and never escapes the run.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [graph glossary](../../../../glossary.md#graph); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Preserve complete physical overlap rejection for every input/output pair and the TOP_K output/
  output pair before scratch mutation, output write, generated invocation, or worker submission.
- Preserve carrier type, size, byte alignment, accessibility, read-only/writability, complete
  referenced-span, assignment, and range validation. Preserve arbitrary legal disjoint ranges,
  complete-slice worker ownership, and scalar/parallel-scalar determinism.<span data-glossary-navigation="true"> ([byte alignment glossary](../../../../glossary.md#term-byte-alignment))</span><span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [slice glossary](../../../../glossary.md#slice))</span>
- Preserve empty selected [axes](../../../../glossary.md#term-axis), empty outer [dimensions](../../../../glossary.md#dimension), `k == 0`, [singleton](../../../../glossary.md#term-singleton) axes, `k == extent`,
  all-equal values, all-NaN values, duplicates, every axis, every legal layout/carrier combination,
  and failure timing established by CPU 0006C.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [layout glossary](../../../../glossary.md#layout); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Preserve lowering, route, capability, cache, specialization, preparation, finalization,
  executable, and independent reference contracts unless an allowlisted Javadoc/package summary
  must be finalized to describe the embedded body accurately.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [cache glossary](../../../../glossary.md#term-cache); [specialization glossary](../../../../glossary.md#term-specialization); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- CPU analysis remains deterministic and measurement-free. Finalization generates or reuses the
  selected class only after shared slot assignment. Runtime receives an immutable prepared
  executable and performs no ordering interpretation, generation, specialization, lookup,
  fallback, tuning, or benchmark-driven mutation.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [lookup glossary](../../../../glossary.md#term-lookup))</span>

### Generated compatibility transition

- Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly from `26` to `27`. Schema 27 is the first
  compatibility version whose ordering classes embed carrier-, type-, family-, direction-,
  access-, comparison-, output-, and loop-specialized stable-merge bodies.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Treat schema 26 as an incompatible safe miss. Extend the artifact-store regression with an
  otherwise valid schema-26 envelope and current metadata/class bytes, prove it is not loaded,
  then prove lookup regenerates and publishes schema-27 bytes. Add no migration reader, converter,
  compatibility alias, or legacy bridge.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [regression glossary](../../../../glossary.md#term-regression); [metadata glossary](../../../../glossary.md#term-metadata); [lookup glossary](../../../../glossary.md#term-lookup); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Preserve structural identity for family, represented type, direction/sorted flags, boundary
  roles/types/access/carriers, scratch-bearing entry shape, and dense/general code shape. Preserve
  compatible shape-polymorphic reuse: concrete extents, axis, <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [shape glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis))</span>`k`, offsets, stride magnitudes,
  scratch sizes/offsets, range count/bounds, carrier instances, slots, workers, graph/run identity,
  and artifact root remain cold.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [graph glossary](../../../../glossary.md#graph); [run glossary](../../../../glossary.md#term-run))</span>

### Stable automated semantic and Class-File evidence

- Add Java Class-File model assertions for all three families, all six represented types,
  ascending/descending or smallest/largest, sorted/unsorted TOP_K, one/two outputs, every exact
  heap-array form, all-<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [heap glossary](../../../../glossary.md#term-heap))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment), mixed carriers, dense integer and general long forms,
  offset/strided/zero-stride layouts, arbitrary legal ranges, and scalar/parallel artifact reuse in
  proportion to risk.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [layouts glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Assert direct typed descriptors, embedded initialization/merge/comparison/output loops, exact
  scratch access, values/index output roles, and dense/general addressing. Reject
  `CpuOrderingEmitter.execute` and equivalent bridges, `Object` [descriptors](../../../../glossary.md#term-descriptor)/casts,
  `DataType.values`, generic family/type/carrier dispatch, allocations, boxing, comparator/library
  sorting, reflection, map/string dispatch, graph/Runtime/backend/cache references, and avoidable
  helper calls through Java Class-File model inspection rather than source-text matching.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [dispatch glossary](../../../../glossary.md#term-dispatch); [graph glossary](../../../../glossary.md#graph); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [cache glossary](../../../../glossary.md#term-cache))</span>
- Use a closed explicit member-reference allowlist for each representative class. Dense classes
  must reject unexpected member references; segment forms may admit only the precise carrier/
  scratch primitives and explicitly justified bit conversions.
- Retain complete `javap` only as audit evidence. Durable regression gates use the Java Class-File
  API and stable structural assertions, not absolute bytecode offsets or planning-observed sizes.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression); [Class-File API glossary](../../../../glossary.md#term-class-file-api); [bytecode glossary](../../../../glossary.md#term-bytecode); [planning glossary](../../../../glossary.md#term-planning))</span>
- Preserve or extend independent semantic differentials against `CpuScalarReferenceKernel` or a
  test-local oracle that does not call/share the generated emitter implementation. Cover every
  type/family/direction/mode, raw NaN payloads/signs, signed zeros, infinities, duplicates, logical
  stability, exact selected bits, INT64 indices, general layouts, heap/segment/mixed carriers,
  partial ranges, exact scratch isolation, empty/edge cases, overlap/failure timing, and scalar/
  parallel bitwise parity.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [NaN glossary](../../../../glossary.md#term-nan); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>

### Independent fixed performance cases

Use two independent cases. Each uses dense [heap](../../../../glossary.md#term-heap)-array [FLOAT32](../../../../glossary.md#term-float32) input/output carriers, [contiguous](../../../../glossary.md#term-contiguous)
[layouts](../../../../glossary.md#layout), [shape](../../../../glossary.md#shape) `[64,1024]`, [normalized axis](../../../../glossary.md#normalized-axis) `1`, 64 complete slices, the complete [slice](../../../../glossary.md#slice) range
`[0,64)`, and one range's existing 16,384-byte two-region [INT64](../../../../glossary.md#data-type--datatype) [`MemorySegment`](../../../../glossary.md#term-memorysegment) scratch.

Initialize each input once per [fork](../../../../glossary.md#term-fork-meanings) from global ordinal `i`, `slice = i / 1024`, and
`position = i % 1024`:

- let `selector = (position + 17 * slice) % 127`;
- selector `0` is a [quiet NaN](../../../../glossary.md#nan-infinity-and-signed-zero) constructed from [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity)
  `0x7fc00000 | (((31 * position + slice + 1) & 0x003fffff))`, with alternating sign by slice;
- selector `1` is raw [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) and selector `2` is raw [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero); and
- every other value is `(float) ((((37 * position + 19 * slice) % 257) - 128) * 0.25)`.

This deterministic pattern contains repeated [finite](../../../../glossary.md#term-finite) values, multiple raw [NaN](../../../../glossary.md#term-nan) payloads/signs,
both signed zeros, and stable ties in every representative [run](../../../../glossary.md#term-run). Verify the exact raw-bit outputs,
logical indices, unchanged input, and scratch bounds before and after timing.

#### Case 1: stable ascending SORT

- [Operation](../../../../glossary.md#operation): `SORT`, [shape](../../../../glossary.md#shape) `[64,1024]`, [axis](../../../../glossary.md#term-axis) `1`, `descending == false`.
- Output: [FLOAT32](../../../../glossary.md#term-float32) `[64,1024]` values.
- The direct body must use the same two scratch regions, increasing logical-index initialization,
  stable bottom-up merge widths, left-on-equality FLOAT32 comparison contract, slice traversal,
  and represented output copy as the generated body. It may use primitive locals and direct array/
  scratch access only.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>

#### Case 2: sorted-largest two-output TOP_K

- [Operation](../../../../glossary.md#operation): `TOP_K`, input [shape](../../../../glossary.md#shape) `[64,1024]`, output shapes `[64,64]`, [axis](../../../../glossary.md#term-axis) `1`, `k == 64`,
  `largest == true`, and `sorted == true`.
- Outputs: [FLOAT32](../../../../glossary.md#term-float32) values then [INT64](../../../../glossary.md#data-type--datatype) logical-axis indices.
- The direct body must perform the same complete stable merge of every 1024-element slice, select
  the first 64 pairs, and write both outputs in the same order with the same scratch/traversal
  semantics. It must not use partial selection, a heap, quickselect, library sorting, precomputed
  order, or omit either output.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [slice glossary](../../../../glossary.md#slice); [heap glossary](../../../../glossary.md#term-heap))</span>

### Reproducible performance protocol and gate

- Use one standalone direct-Java probe outside Gradle and JUnit. Run at least five fresh JVM forks
  with fixed <span data-glossary-navigation="true"> ([JVM forks glossary](../../../../glossary.md#java-preview-incubator-modules-and-compiler-tiers))</span><span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [JUnit glossary](../../../../glossary.md#term-junit); [Run glossary](../../../../glossary.md#term-run); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>`-Xms1g -Xmx1g`. Each fork runs at least five warmup batches and nine measurement
  rounds per case, with adaptive repetition counts making each timed sample at least 25 ms.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Randomize case order and generated/direct order within every [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) with deterministic [seed](../../../../glossary.md#term-seed)
  `0x5A17D00D7A0E27L`, reset identically in each fork. The summarizer reports each case separately.
  Consume an observable sink only outside timed regions.
- Report generated/direct medians and ratio for every fork of each case. For each case, compute the
  aggregate as the median of its generated per-fork medians divided by the median of its direct
  per-fork medians. Every fork ratio and the aggregate ratio for SORT must be <span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median))</span>`<= 1.15x`; every
  fork ratio and aggregate for TOP_K must independently be `<= 1.15x`.
- Never average forks, drop an outlier, allow one case or the aggregate to hide a failing fork,
  widen the threshold, change the fixed cases/pattern after baseline, or substitute another
  direction, <span data-glossary-navigation="true"> ([outlier glossary](../../../../glossary.md#term-outlier))</span>`k`, mode, [shape](../../../../glossary.md#shape), range, algorithm, [carrier](../../../../glossary.md#term-storage-carrier), or scratch contract.
- Use identical probe/summarizer source, data, direct bodies, commands, environment, heap,
  warmup/round/fork/adaptive protocol, seed, verification, and sink rules for baseline and final
  evidence. Record source SHA-256 and generated-class SHA-256 for both phases.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [SHA-256 glossary](../../../../glossary.md#term-sha-256))</span>
- Performance evidence is observational and must not select or mutate ordinary preparation or
  Runtime behavior. If either case has one failing fork or aggregate, keep implementation status<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
  `In progress` or `Review needed`, inspect generated/[JIT](../../../../glossary.md#term-just-in-time-compilation-jit) shape, and correct the implementation.
- ARGSORT, every represented type, descending/smallest order, unsorted TOP_K, edge cases, general
  layouts, <span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment)/mixed carriers, arbitrary ranges, and parallel-scalar forms remain in
  semantic and Class-File gates. Add another timed case only if implementation evidence identifies
  a concrete risk that those stable gates cannot cover; do not weaken or replace either fixed gate.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- New Model semantics, operations, attributes, types, Shapes, ordering policies, capability rows,
  public APIs, or gradient behavior<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Shapes glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Any ordering algorithm redesign, partial TOP_K selection, unstable/custom comparison, vector or
  native sorting, runtime algorithm choice, fixed-shape specialization, unrolling, or relaxed
  numerical policy<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [runtime glossary](../../../../glossary.md#term-runtime); [specialization glossary](../../../../glossary.md#term-specialization); [unrolling glossary](../../../../glossary.md#term-unrolling))</span>
- Pointwise, affine/movement, indexing, scatter, fold, random/dropout, scan, aggregate, reduction,
  normalization, linear algebra, native-provider, or later family implementation<span data-glossary-navigation="true"> ([Pointwise glossary](../../../../glossary.md#term-pointwise); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [scan glossary](../../../../glossary.md#term-cumulative-scan); [normalization glossary](../../../../glossary.md#term-normalization); [linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- New routes, workspace kinds, scratch declaration/lifecycle, materialization, representation
  policy, tuning controls, tuning-cache behavior, public configuration, or Runtime mutation<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [materialization glossary](../../../../glossary.md#materialization); [cache glossary](../../../../glossary.md#term-cache))</span>
- Architecture, module/dependency boundary, shared Model/Compiler/Planning/Prepare/Runtime, Config,
  Trace, Engine, other backend, NN, Training, Gradle/build, Java version, architecture-test,
  backend-conformance, integration-test, or unrelated documentation changes<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Detailed specifications or implementation for [CPU](../../../../glossary.md#term-cpu) 0007A0F, CPU 0007A1, or any later task
- [Commit](../../../../glossary.md#term-commit), push, staging, revert, deletion, or modification of unrelated concurrent work

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Runtime/Prepare/Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU 0006C portable stable ordering and selection](0006c-portable-stable-ordering-and-selection.md)
- [CPU 0007A0 generated hot-path parity correction](0007a0-generated-hot-path-parity-correction.md)
- [CPU 0007A0A affine and movement parity](0007a0a-affine-and-movement-generated-loop-parity.md)
- [CPU 0007A0B indexing parity](0007a0b-indexing-generated-loop-parity.md)
- [CPU 0007A0C scatter parity](0007a0c-scatter-generated-loop-parity.md)
- [CPU 0007A0D fold parity](0007a0d-fold-generated-loop-parity.md)

## Architecture constraints

- Model owns stable ordering meaning. This task changes only CPU-private generated realization and
  cannot reinterpret, narrow, or extend that semantic contract.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Planning selects CPU ownership only. CPU analysis owns deterministic lowering, specialization,
  strategy, and exact declarations; CPU finalization owns compatible generation/reuse after shared
  assignment; Runtime executes the immutable prepared result only.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Runtime hot code receives typed direct carriers, the assigned scratch segment, and primitive
  cold geometry. No [`Operation`](../../../../glossary.md#operation), `CompiledNode`, Tensor, graph object, backend discovery,
  validation decision, route/cache policy, or benchmark evidence reaches it.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend); [validation glossary](../../../../glossary.md#term-validation); [cache glossary](../../../../glossary.md#term-cache))</span>
- Prepared recipes and generated artifacts remain immutable and reusable. Concurrent runs use
  distinct <span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact))</span>[`RunState`](../../../../glossary.md#run-state--runstate) objects; every active range has disjoint output slices and assigned scratch.
- Complete CPU 0006C semantics, general layouts/carriers/ranges, exact scratch ownership,
  deterministic scalar/parallel behavior, and the Runtime/Prepare boundary remain unchanged.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Stop if parity requires a public/shared contract, architecture, dependency, semantic, route,
  resource, materialization, Runtime-policy, algorithm, or broader-module change.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [materialization glossary](../../../../glossary.md#materialization); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — embedded stable-merge,
  comparison, output, typed carrier, scratch, and dense/general Class-File emission.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — [schema](../../../../glossary.md#term-schema) and [artifact](../../../../glossary.md#term-artifact) compatibility.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — existing family, represented type,
  direction/output-order, boundary-access, and scratch-policy identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — existing validated logical-slice,
  layout, axis, <span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [layout glossary](../../../../glossary.md#layout); [axis glossary](../../../../glossary.md#term-axis))</span>`k`, and exact scratch geometry.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — existing deterministic declarations,
  specialization, strategy, workspace policy, and final assignment validation.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — existing carrier/overlap binding,
  geometry packing, range partitioning, scratch slicing, and worker invocation.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent semantic [oracle](../../../../glossary.md#term-test-oracle) only.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — current generated-route and
  artifact lifecycle documentation.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>

Packages added or changed:

- No package is added, removed, moved, exported, or made supported [API](../../../../glossary.md#term-application-programming-interface-api).

Type placement:

- `CpuOrderingEmitter` remains the sole owner of generated stable merge, comparison classes,
  selected-pair ordering, and values/index stores. It becomes an embedded-body emitter rather
  than a generic runtime bridge owner.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- `CpuClassFileKernelGenerator` passes existing structural ordering IR to the focused emitter; it
  does not acquire ordering semantics, packed geometry, scratch ownership, or Runtime state.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `CpuCarrierEmitter` may supply generation-time typed load/store instructions but does not own
  ordering comparison, merge, output-role, or resource policy.
- `CpuOrderingIr`, `CpuOrderingLowering.Geometry`, preparation, executable, and reference owners
  retain their current responsibilities. Add no dispatcher, registry, comparator abstraction,
  manager, broad helper, or facade.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [facade glossary](../../../../glossary.md#term-facade))</span>

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
architecture documents, and shared [modules](../../../../glossary.md#term-module) are inspection-only unless the task is stopped and
replanned.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may modify at most 18 repository paths, all drawn from the exact allowlist above. It may
create no production or test type and no later task specification. Temporary probe/evidence files
must remain within the one fresh isolated `/private/tmp` directory and are not repository paths.

If implementation requires any non-allowlisted repository path, more than 18 changed paths, a new
type, or a broader contract, stop before editing it and request a [planning](../../../../glossary.md#term-planning) update. Do not consume
the allowlist with unrelated cleanup.

## Acceptance criteria

- Current schema-26 source and dense/general generated classes are freshly retained and audited
  before production edits with exact source/commands/environment, descriptors, bytes, SHA-256,
  complete <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [descriptors glossary](../../../../glossary.md#term-descriptor); [SHA-256 glossary](../../../../glossary.md#term-sha-256))</span>`javap`, member references, semantics, and raw timing.
- Every admitted SORT, ARGSORT, and TOP_K specialization embeds typed stable merge, comparison,
  optional selected-index ordering, and correct one/two-output work with no generic bridge.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
- Dense heap-array forms use cold-proved primitive integer state; every admitted segment, mixed-
  carrier, arbitrary-layout, zero-stride, strided-output, large-range, or unproved form retains
  correct typed general-long execution.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [stride glossary](../../../../glossary.md#term-stride))</span>
- NaN-last, directional signed zero, infinities, signed integral/BOOL order, left-on-equality
  logical stability, exact represented bits, INT64 logical indices, sorted/unsorted TOP_K, empty
  and edge behavior, partial ranges, and scalar/parallel determinism are independently proved.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [signed zero glossary](../../../../glossary.md#term-signed-zero); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [BOOL glossary](../../../../glossary.md#data-type--datatype); [represented bits glossary](../../../../glossary.md#term-represented-bits); [INT64 glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Existing capability, lowering validation, ordered boundaries, exact assigned two-region scratch,
  no materialization, carrier/overlap/assignment checks, range ownership, and immutable lifecycle
  remain unchanged.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [validation glossary](../../../../glossary.md#term-validation); [materialization glossary](../../../../glossary.md#materialization); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Schema advances exactly from 26 to 27; schema-26 persistence is an incompatible safe miss that
  regenerates schema-27 bytes; concrete invocation geometry remains shape-polymorphically reusable.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [shape glossary](../../../../glossary.md#shape))</span>
- Stable Java Class-File assertions cover the required family/type/direction/mode/carrier/access/
  output/range matrix and reject <span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>`Object`, generic dispatch, allocation, forbidden lookup, generic
  bridge, library sort, and avoidable helper references.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Every SORT [fork](../../../../glossary.md#term-fork-meanings) and its aggregate independently pass `<= 1.15x`; every two-output TOP_K fork and
  its aggregate independently pass `<= 1.15x`, using unchanged equivalent direct bodies and exact
  pre/post verification.
- Baseline and final evidence remain under one fresh isolated `/private/tmp` directory and include
  all required source, commands, environment, class files, decompilation, checksums, member
  references, semantic results, raw samples, and summaries.<span data-glossary-navigation="true"> ([decompilation glossary](../../../../glossary.md#term-decompilation); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Only allowlisted paths change, no more than 18 repository paths change, no GSD artifact exists,
  and no CPU 0007A0F or later detailed task specification exists.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- A distinct clean documentation-focused context independently reviews final source, tests,
  Class-File evidence, and performance evidence; finalizes affected Javadocs/package summaries,
  CPU guide, glossary impact, this task, master plan, and roadmap; and records reasoned no-change
  conclusions before completion.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused tests for every changed test [owner](../../../../glossary.md#term-owner) while iterating. The final focused command must name
every changed test class and cover stable Class-File [shape](../../../../glossary.md#shape), all three families and six types,
dense/general and [heap](../../../../glossary.md#term-heap)/segment/mixed carriers, both directions, TOP_K modes, raw special values,
arbitrary [layouts](../../../../glossary.md#layout)/ranges, exact scratch isolation, independent differentials, [scalar](../../../../glossary.md#term-scalar)/parallel
reuse, preparation/finalization, [specialization](../../../../glossary.md#term-specialization), and [schema](../../../../glossary.md#term-schema)/[persistence](../../../../glossary.md#term-persistence) compatibility.

After executable Java stabilizes, run exactly one authoritative affected-[module](../../../../glossary.md#term-module) command:

```bash
./gradlew :backends:cpu:test
```

Run the isolated baseline/final probe separately from [Gradle](../../../../glossary.md#term-gradle)/[JUnit](../../../../glossary.md#term-junit) with the exact five-[fork](../../../../glossary.md#term-fork-meanings),
two-case protocol. Timing is retained observational evidence, not an ordinary [unit test](../../../../glossary.md#term-unit-test).

The distinct clean documentation pass reuses successful Java and timing evidence unless it
changes executable Java/probe behavior or records a concrete stale-evidence reason. It runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also inspects rendered affected Javadocs and validates local Markdown links and heading
anchors, balanced fences, terminal newlines, carriage returns, trailing whitespace, exact changed-
path allowlist/count, package/type placement, schema-27 history and schema-26 rejection, generated
[descriptors](../../../../glossary.md#term-descriptor) and member-reference allowlists, retained baseline/final source/checksums/protocol/
ratios, task/master/roadmap status and [dependency](../../../../glossary.md#term-dependency) order, absence of later specs, preservation of
concurrent [scope](../../../../glossary.md#term-scope), and absence of GSD [artifacts](../../../../glossary.md#term-artifact).

Repository-wide, architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration validation remains deferred
to [CPU](../../../../glossary.md#term-cpu) 0009 or [continuous integration](../../../../glossary.md#term-continuous-integration-ci) because this task changes one [concrete backend](../../../../glossary.md#concrete-backend) realization
and no shared or [architecture contract](../../../../glossary.md#architecture-contract). Stop and replan if implementation makes that conclusion
false.

## Dependencies

- [CPU 0006C](0006c-portable-stable-ordering-and-selection.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> is `Complete` and owns the exact
  current ordering semantics, type/family matrix, stable merge and scratch policy, layouts,
  carriers, ranges, validation, reference oracle, worker safety, and schema-18 history preserved
  here.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [layouts glossary](../../../../glossary.md#layout); [validation glossary](../../../../glossary.md#term-validation); [oracle glossary](../../../../glossary.md#term-test-oracle); [schema glossary](../../../../glossary.md#term-schema))</span>
- [CPU 0007A0](0007a0-generated-hot-path-parity-correction.md) is `Complete` and supplies shared
  typed carrier emission, dense integer proof, stable Class-File testing, and the retained
  five-fork evidence pattern.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- [CPU 0007A0A](0007a0a-affine-and-movement-generated-loop-parity.md) is `Complete` and establishes
  the dense/general correction boundary and schema-23 history.
- [CPU 0007A0B](0007a0b-indexing-generated-loop-parity.md) is `Complete` and establishes embedded
  mixed-carrier indexed mapping without generic bridges and schema-24 history.
- [CPU 0007A0C](0007a0c-scatter-generated-loop-parity.md) is `Complete` and establishes embedded
  output/contribution bodies with declared scratch and schema-25 history.
- [CPU 0007A0D](0007a0d-fold-generated-loop-parity.md) is `Complete` and establishes current schema
  26, direct predefined segment layouts, and embedded typed mapping/addition bodies.
- Current Java 26 Class-File API, shared Prepare/Runtime, CPU worker, direct carrier binding,
  scratch assignment, and generated-artifact store contracts are complete and unchanged.<span data-glossary-navigation="true"> ([Class-File API glossary](../../../../glossary.md#term-class-file-api); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding); [artifact glossary](../../../../glossary.md#term-artifact))</span>

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0007A0F remains the next ordered `Draft` random/dropout correction without a detailed
  specification. Do not create it in this task.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- CPU 0007A1 remains `Draft` and explicitly depends on CPU 0007A0F before semantic-family
  expansion resumes.
- CPU 0009 retains the generated-coverage and [conformance](../../../../glossary.md#term-conformance) [checkpoint](../../../../glossary.md#term-checkpoint).

## Architecture impact

Expected impact: None.

The task changes only [CPU](../../../../glossary.md#term-cpu)-owned generated ordering code [shape](../../../../glossary.md#shape) and current-only [artifact](../../../../glossary.md#term-artifact)
compatibility under the existing analysis/finalization/[binding](../../../../glossary.md#term-binding) [lifecycle](../../../../glossary.md#lifecycle). If implementation
requires an architecture, public/shared contract, [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency), semantic, algorithm, route,
resource, [materialization](../../../../glossary.md#materialization), or [Runtime](../../../../glossary.md#term-runtime)-policy change, stop and report the conflict.

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
  defect.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Advance exactly to schema 27 because generated ordering bodies change while schema 26 already
  identifies embedded fold bodies. A schema-26 bridge-only ordering artifact must never satisfy a
  schema-27 direct-loop request.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Use `[64,1024]` axis-1 FLOAT32 cases to exercise repeated complete slices, stable comparison,
  scratch reuse, and output work without fixed-shape production specialization. Use full ascending
  SORT and sorted-largest TOP_K with <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [FLOAT32 glossary](../../../../glossary.md#term-float32); [shape glossary](../../../../glossary.md#shape); [specialization glossary](../../../../glossary.md#term-specialization))</span>`k == 64` as independently gated one-output and two-output
  cases.
- Require both every per-[fork](../../../../glossary.md#term-fork-meanings) ratio and each case's aggregate to pass `<= 1.15x`; aggregate success
  or the other benchmark cannot conceal one failing process or case.
- Keep ARGSORT, all other types/directions/modes, general layouts, segment/mixed carriers, edge
  cases, arbitrary ranges, and parallel forms in stable semantic/Class-File gates. The two named
  performance claims do not generalize beyond their exact cases.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout))</span>
- Direct scratch segment access uses the existing predefined native-order INT64 layout and assigned
  two-region lifecycle; no resource or algorithm redesign is authorized.<span data-glossary-navigation="true"> ([INT64 glossary](../../../../glossary.md#data-type--datatype); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Known limitations

- The timing gates cover only dense heap-array FLOAT32 ascending SORT and sorted-largest two-output
  TOP_K on the recorded environment. They make no parity claim for ARGSORT, other types,
  directions, unsorted TOP_K, general layouts, segment/mixed carriers, parallel orchestration,
  every CPU, or every JVM.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [FLOAT32 glossary](../../../../glossary.md#term-float32); [layouts glossary](../../../../glossary.md#layout); [CPU glossary](../../../../glossary.md#term-cpu); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- General and unmeasured forms receive stable semantic and Class-File validation but no numerical
  performance threshold in this task.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Only the existing one-node fully static resolved-layout CPU ordering capability is corrected.
  Dynamic layouts, custom ordering, vector/native sorting, multi-node fusion, and new semantics
  remain unsupported or deferred as before.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [vector glossary](../../../../glossary.md#term-vector); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- The completed stable bottom-up merge and full-order TOP_K algorithm remain intentionally
  unchanged. This generated-shape task does not claim optimal selection complexity.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `01a018e2-75c0-7272-b0b9-1ebb6ae7f856` inspected the clean worktree, current
[schema](../../../../glossary.md#term-schema)-26 source, ordering [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/geometry/emitter/generator/[specialization](../../../../glossary.md#term-specialization)/preparation/
finalization/executable/reference/tests/[cache](../../../../glossary.md#term-cache)/build contracts, completed task history, and the
four generated Class-File forms recorded in [Scope](../../../../glossary.md#term-scope). `./gradlew :backends:cpu:testClasses` passed
with 11 tasks up-to-date. Complete `javap -c -p` and `javap -v -p` confirmed typed direct entry
[descriptors](../../../../glossary.md#term-descriptor) followed by the sole generic bridge member reference. This evidence fixes the
allowlist and justifies schema 27 but must be reproduced as a fresh retained implementation
baseline before production edits.

Implementation validation evidence:

- The clean implementation context retained all evidence under
  `/private/tmp/synaptik-cpu-0007a0e-implementation.dMNFPD`. The environment was OpenJDK
  26.0.1+8-34 on macOS 26.5.2 arm64; context ID<span data-glossary-navigation="true"> ([arm64 glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary))</span>
  `01a018ec-ea40-7f21-b2be-1ec07564d0b8`. `OrderingPerformanceProbe.java` has [SHA-256](../../../../glossary.md#term-sha-256)
  `efc15fecc64380adad13999cfda62454e07040d86d2738cdbd34a3f5f2c72e3e`; the retained
  summarizer has SHA-256 `2ee59c1c88e95ddb2ca96f850b88a2ce198ebbd984ae43b5a0adafceb66226e8`.
- Fresh schema-26 baseline classes retained their typed entry descriptors and bridge-only shape:
  dense FLOAT32 SORT was 489 bytes, SHA-256<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
  `55c2a18352b72827351a1a87476b4075acf3fdf36bb0c93c615c82d7933b0d7a`; dense FLOAT64
  ARGSORT was 489 bytes, <span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64))</span>`2e7f10e26fa71e0aae7f71880624bba36552f5f70a246b9dcc957c2f771b4e88`;
  dense FLOAT32 TOP_K was 492 bytes,
  `e0d932fd29343fc7aedf703266a4eb89ee55b5973aeaeedc3996ca74b823f3b5`; and mixed BFLOAT16
  TOP_K was 554 bytes, <span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>`62a7366e1a4fb101822e166c2aec1888eaba13a04e4aa871c79b6dc308138bed`.
  Each had only the generic `CpuOrderingEmitter.execute(Object,Object,Object,...)` method
  reference. Baseline SORT fork ratios were `5.000745397x`, `4.888217534x`, `4.937615787x`,
  `4.986715932x`, and `4.936871843x`, aggregate `4.974047838x`; TOP_K ratios were
  `4.470562353x`, `5.060709708x`, `4.550869282x`, `4.815527022x`, and `4.991654916x`,
  aggregate `4.815527022x`.
- Final schema-27 descriptors are unchanged. Dense [FLOAT32](../../../../glossary.md#term-float32) SORT is 2,924 bytes, SHA-256
  `5451f7139693a48092c538a8c158ec4dbd2ed86ef900861c9a325051c3cecdaf`; dense FLOAT64
  ARGSORT is 2,878 bytes, `1a7d36900b6a198db205d8deb7f09e537fb51fc0e1f06f570f43167f8f766893`;
  dense FLOAT32 TOP_K is 3,485 bytes,
  `5ed4abf07e0d1977744c6e27d5dc7bb9d30e3804b2d7e3fa3433871101fba9ca`; and mixed BFLOAT16
  TOP_K is 4,968 bytes, `ec8cc402609ea0298a726d7e5b6d1b5ce5227b87a50236353b6647c9e3b32334`.
  Complete retained `javap -c -p` and `javap -v -p` show direct loops and no bridge or Object
  descriptor. Dense member references are exactly `ValueLayout.JAVA_LONG` plus
  `MemorySegment.get/set` for assigned scratch. The mixed class additionally references exact
  predefined SHORT/LONG unaligned layouts and the justified <span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout))</span>`Float.intBitsToFloat` BFLOAT16
  conversion.
- Final SORT [fork](../../../../glossary.md#term-fork-meanings) ratios were `1.092722590x`, `1.077727976x`, `1.090165540x`, `1.049603676x`,
  and `1.089690637x`, aggregate `1.090165540x`. Final TOP_K fork ratios were `0.962180009x`,
  `0.969660236x`, `0.936075945x`, `1.018861174x`, and `1.020688132x`, aggregate
  `0.965951020x`. Every independent fork and aggregate passed `<= 1.15x` with unchanged source,
  data, direct bodies, heap, seed, warmup, measurement, adaptive, verification, and sink rules.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [seed glossary](../../../../glossary.md#term-seed); [warmup glossary](../../../../glossary.md#term-warmup))</span>
- The final focused command named all five changed test [owners](../../../../glossary.md#term-owner) and passed. The single authoritative
  `./gradlew :backends:cpu:test` passed 320 tests with 1 expected skip and zero failures/errors.
  Class-File and semantic tests cover all types/families/directions/TOP_K modes, typed descriptors,
  dense/general forms, heap/segment/mixed carriers, zero-stride and offset/strided layouts,
  partial ranges, represented special values, scalar/parallel lifecycle, and schema-26 rejection.
  No Java suite or timing probe was repeated by the documentation-only finalization.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [scalar glossary](../../../../glossary.md#term-scalar); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Mandatory clean review/fix context `01a01905-b866-70f3-89c7-181adcabe2c5` independently audited
  the executable diff and refactored the compressed lower half of `CpuOrderingEmitter` plus dense
  touched-test setup into normal readable Java without changing the emitted Class-File builder
  sequence. The focused five-owner command passed, followed by exactly one authoritative<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
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
  absence of GSD artifacts, and the unstaged index all passed. <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc); [artifacts glossary](../../../../glossary.md#term-artifact))</span>`git diff --check` passed.
- Glossary review required no edit: schema 27 changes the private generated implementation and
  compatibility version, but introduces no new user-facing term or synonym and changes none of
  the existing stable-ordering semantics. Architecture, ADRs, build files, module boundaries,
  capabilities, public APIs, backend conformance, integration tests, Compiler, Runtime, Engine,
  gradients, related ordering/lowering/geometry/preparation/finalization/executable contracts,
  and other modules likewise required no documentation or implementation change.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [gradients glossary](../../../../glossary.md#term-gradient))</span>

## Implementation notes

[Schema](../../../../glossary.md#term-schema) 27 now embeds typed stable merge, comparison, unsorted selected-index ordering, and
represented output stores. Existing [lowering](../../../../glossary.md#lowering) geometry, two-region assigned scratch, preparation,
finalization, executable [binding](../../../../glossary.md#term-binding), overlap [validation](../../../../glossary.md#term-validation), range [ownership](../../../../glossary.md#term-ownership-meanings), and reference contracts
were preserved without changes. The clean documentation context independently confirmed these
implementation claims against the final source and retained evidence.

## Completion summary

[Schema](../../../../glossary.md#term-schema) 27 now embeds typed direct stable-ordering loops for SORT, ARGSORT, and TOP_K while
preserving [CPU](../../../../glossary.md#term-cpu) 0006C semantics, exact two-region scratch, [validation](../../../../glossary.md#term-validation), [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout) coverage,
parallel range [ownership](../../../../glossary.md#term-ownership-meanings), and [lifecycle](../../../../glossary.md#lifecycle). Both clean executable contexts and the clean
documentation context completed their required reviews. The focused five-[owner](../../../../glossary.md#term-owner) tests and the
single final CPU suite passed; all retained Class-File, member-reference, compatibility, and
five-[fork](../../../../glossary.md#term-fork-meanings) performance gates passed. The CPU guide and [planning](../../../../glossary.md#term-planning) state are synchronized, the
glossary no-change conclusion is recorded, and CPU 0007A0F is the sole next ordered Draft frontier
with CPU 0007A1 explicitly behind it.

Status: Complete
