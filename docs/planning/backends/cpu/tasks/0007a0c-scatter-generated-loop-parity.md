# Task 0007A0C: Scatter Generated-Loop Parity

## Status

Complete

## Goal

Replace the bridge-only generated output and contribution work for the completed [CPU](../../../../glossary.md#term-cpu) functional
scatter family with [carrier](../../../../glossary.md#term-storage-carrier)-, type-, reduction-, access-, and family-specialized loops embedded
in each generated class. `SCATTER_ELEMENTS`, [Gather](../../../../glossary.md#gather)-compatible `SCATTER_ADD`, and `SCATTER_ND`
must preserve every CPU 0006B1 semantic, [validation](../../../../glossary.md#term-validation), [layout](../../../../glossary.md#layout), carrier, range, resource, worker,
scratch, and [lifecycle](../../../../glossary.md#lifecycle) contract while two representative dense cases independently reach
measurable near-parity with equivalent direct Java loops.

This is one cohesive CPU-private correction. The three forms share `CpuScatterIr`, compact cold
geometry, deduplicated boundary roles, `CpuScatterEmitter`, [canonical](../../../../glossary.md#term-canonical-form) contribution scanning,
represented reductions, optional exact floating-product scratch, and one generated-[artifact](../../../../glossary.md#term-artifact)
compatibility boundary. The task changes generated realization only; completed CPU 0006B1
capability and semantics remain unchanged.

## [Scope](../../../../glossary.md#term-scope)

### Mandatory pre-edit source and [bytecode](../../../../glossary.md#term-bytecode) baseline

- Before changing production Java, create a fresh isolated `/tmp` evidence directory and capture
  representative current schema-24 generated classes for:<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [schema glossary](../../../../glossary.md#term-schema))</span>
  - dense unique [INT32](../../../../glossary.md#data-type--datatype)-index `SCATTER_ELEMENTS + NONE` over [FLOAT32](../../../../glossary.md#term-float32) [heap](../../../../glossary.md#term-heap) arrays;
  - dense duplicate-index INT32 `SCATTER_ADD` over FLOAT32 heap arrays;
  - one `SCATTER_ND` general-[layout](../../../../glossary.md#layout) or segment/mixed-[carrier](../../../../glossary.md#term-storage-carrier) form; and
  - one floating `MUL` form whose entry includes the declared scratch [`MemorySegment`](../../../../glossary.md#term-memorysegment).
- Retain the probe source, exact compile/generate/decompile/run/summarize commands, Java and
  operating-system facts, class bytes, complete <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run))</span>`javap -c -p` and `javap -v -p` output, sizes,
  SHA-256 checksums, raw fork samples, summaries, ratios, and verification results together.<span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256); [fork glossary](../../../../glossary.md#term-fork-meanings); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Record the current generated entry descriptors and constant-pool method references. The known
  current shape is a typed outer descriptor whose body only loads arguments and invokes<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [constant glossary](../../../../glossary.md#term-constant); [shape glossary](../../../../glossary.md#shape); [arguments glossary](../../../../glossary.md#term-argument))</span>
  `CpuScatterEmitter.execute2*`, `execute3*`, or `execute4*` through `Object` parameters; do not
  rely on this planning statement in place of the required fresh baseline.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [planning glossary](../../../../glossary.md#term-planning))</span>
- Time only warmed, already-validated output/contribution work. Exclude lowering, generation,
  class definition, artifact lookup, cold binding, allocation, input initialization, complete
  bounds/duplicate validation, result verification, and sink observation from both timed forms.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [artifact glossary](../../../../glossary.md#term-artifact); [lookup glossary](../../../../glossary.md#term-lookup); [binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation))</span>

### Embedded specialized scatter work

- Change `CpuClassFileKernelGenerator` and `CpuScatterEmitter` so each generated entry embeds the
  selected scatter family, reduction, data type, index width, carrier accesses, contribution
  matching, represented reduction, and output store rather than calling an <span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`Object` bridge for
  output-cell or contribution work.
- Select `SCATTER_ELEMENTS`, [Gather](../../../../glossary.md#gather)-compatible `SCATTER_ADD`, or `SCATTER_ND`; `NONE`, `ADD`,
  `MUL`, `MIN`, or `MAX`; FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, or canonical BOOL where
  admitted; INT32 or INT64 indices; and each ordered heap/segment carrier form at generation time
  from existing typed CPU-private IR and specialization facts.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [heap glossary](../../../../glossary.md#term-heap); [CPU glossary](../../../../glossary.md#term-cpu); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Preserve unique boundary declaration and the exact three-role occurrence map when data,
  indices, or updates reuse the same <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot). The generated class must address each semantic
  role through its mapped direct typed boundary without reconstructing generic carriers.
- Preserve the universal primitive `long start, long end` entry. Each arbitrary legal half-open
  range owns disjoint output coordinates, reads the functional base, scans matching logical
  update contributions in canonical row-major order, and writes every owned output coordinate
  exactly once.<span data-glossary-navigation="true"> ([half-open glossary](../../../../glossary.md#term-half-open-interval); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- Generate direct typed array or native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment) loads and stores. Use no hot
  [`DataType`](../../../../glossary.md#data-type--datatype), `ScatterReduction`, family, carrier, array-versus-segment, string, map, reflection,
  graph, Runtime, route, or cache dispatch and no boxing or per-output/per-contribution allocation.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [Runtime glossary](../../../../glossary.md#term-runtime); [cache glossary](../../../../glossary.md#term-cache); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Hoist family, reduction, ranks, boundary positions, axis/batch/tuple roles, and invariant packed-
  geometry positions out of the output and contribution loops when structural specialization
  makes them invariant. Concrete extents, offsets, stride magnitudes, ranges, and scratch offsets
  remain cold shape-polymorphic geometry.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [invariant glossary](../../../../glossary.md#term-invariant); [stride glossary](../../../../glossary.md#term-stride); [shape glossary](../../../../glossary.md#shape))</span>

### Dense and general generated forms

- For cold-proved dense [heap](../../../../glossary.md#term-heap)-array boundaries, use the current `DENSE_HEAP_ARRAY_INT` category.
  Narrow universal bounds, bases, and other proved Java-array-index facts once before hot work,
  then retain primitive integer output/update/address state without per-contribution `l2i` or
  generic address helpers.
- Dense `SCATTER_ELEMENTS + NONE` must copy base for unaddressed outputs or the sole validated
  update for addressed outputs with no reduction path. Its unique-target timed case must not do
  duplicate detection inside the generated entry.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Dense [FLOAT32](../../../../glossary.md#term-float32) `SCATTER_ADD` must include base once and add every duplicate contribution once in
  canonical logical update order using binary32 addition after each contribution. Its timed
  direct baseline must use the same output-domain scan and the same ordered binary32 work; an
  update-centric, pre-grouped, reordered, atomic, widened, or otherwise semantically easier
  direct algorithm is forbidden.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [scan glossary](../../../../glossary.md#term-cumulative-scan); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- Generate typed `GENERAL_LONG` forms for every supported [`MemorySegment`](../../../../glossary.md#term-memorysegment), mixed-carrier,
  arbitrary non-negative layout, zero-stride read, offset/strided injective output, large range,
  and otherwise unproved case. Use primitive carry/reset state and only coordinate arithmetic
  genuinely required by the structural form.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [stride glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective); [carry glossary](../../../../glossary.md#term-carry); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- A failed dense proof selects the correct general form and never rejects a CPU 0006B1-supported
  occurrence. Preserve scalar and zero domains, empty contributions, scalar indices where legal,
  both index widths, all current families/reductions/types, and arbitrary legal ranges.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [occurrence glossary](../../../../glossary.md#term-occurrence); [scalar glossary](../../../../glossary.md#term-scalar))</span>

### Semantic and [validation](../../../../glossary.md#term-validation) invariants

- Preserve functional behavior: inputs are never mutated; the result has the exact data Shape and
  type; an unaddressed output copies exact base bits; <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>`NONE` replaces without base participation;
  arithmetic target groups include base exactly once and every duplicate contribution exactly
  once.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Preserve complete deterministic logical [row-major](../../../../glossary.md#term-row-major) bounds validation, followed for
  `SCATTER_ELEMENTS + NONE` and `SCATTER_ND + NONE` by complete deterministic target-uniqueness
  validation, before any generated call, worker submission, scratch mutation, or output write.
- Preserve exact exception types, messages, logical ordinals, selected axes/extents, first-later
  duplicate reporting, independent empty-index/zero-output domains, and complete no-write behavior
  for every invalid binding.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [binding glossary](../../../../glossary.md#term-binding))</span>
- Generated work consumes only validated indices. It must not normalize, wrap, clamp, ignore,
  default, or revalidate indices; choose first/last duplicate replacement; or expose partial output.
- Preserve canonical logical contribution order independently of physical layouts and worker
  scheduling. FLOAT64/FLOAT32 add in the same format, BFLOAT16 rounds after each addition,
  INT32/INT64 arithmetic is modular, extrema use exact signed/floating rules, and <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [layouts glossary](../../../../glossary.md#layout); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>`NONE` preserves
  represented bits.<span data-glossary-navigation="true"> ([represented bits glossary](../../../../glossary.md#term-represented-bits))</span>
- Preserve scalar/parallel-scalar bitwise parity, disjoint output ownership, read-only shared
  inputs, invocation-private coordinate state, optional disjoint scratch slices, and absence of
  atomics, locks, cross-range partials, barriers, or merge work.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>

### Exact floating-product scratch

- Preserve the CPU 0006B1 once-rounded abstract floating product for FLOAT64, FLOAT32, and
  BFLOAT16 <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>`MUL`, including base participation, all finite factors, ties-to-even, overflow,
  subnormal/underflow, sign parity, signed zero, infinity, NaN, and zero-times-infinity behavior.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [ties-to-even glossary](../../../../glossary.md#term-ties-to-even); [overflow glossary](../../../../glossary.md#term-overflow); [subnormal glossary](../../../../glossary.md#term-subnormal); [underflow glossary](../../../../glossary.md#term-underflow); [signed zero glossary](../../../../glossary.md#term-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [NaN glossary](../../../../glossary.md#term-nan))</span>
- Preserve exactly one declared `SCATTER_PRODUCT` workspace when and only when a non-empty
  floating-product calculation is possible. Retain checked per-range slice sizing, eight-byte
  alignment, fixed metadata, precision-derived primitive limbs, disjoint range offsets, and no
  materialization or second workspace.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [byte alignment glossary](../../../../glossary.md#term-byte-alignment); [metadata glossary](../../../../glossary.md#term-metadata); [precision glossary](../../../../glossary.md#term-precision); [materialization glossary](../../../../glossary.md#materialization))</span>
- Embed the type-specialized factor classification, significand multiplication, exponent state,
  and final rounding operations in generated work. Calls are allowed only to the minimal JDK or<span data-glossary-navigation="true"> ([significand glossary](../../../../glossary.md#term-significand); [exponent glossary](../../../../glossary.md#term-exponent); [rounding glossary](../../../../glossary.md#term-rounding); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>
  [`MemorySegment`](../../../../glossary.md#term-memorysegment) primitives necessary to realize those operations; no generic scatter helper,
  `Object` access, family/type/reduction dispatcher, `BigInteger`, collection, array, cursor,
  table, or per-factor/per-output allocation may enter the hot path.<span data-glossary-navigation="true"> ([hot path glossary](../../../../glossary.md#term-hot-path))</span>
- Do not include floating `MUL` in either parity ratio. Record its scratch-bearing class shape,
  semantic differential coverage, allocation/dispatch evidence, and workspace accounting
  separately so a scratch-free benchmark cannot hide regression in this distinct contract.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [dispatch glossary](../../../../glossary.md#term-dispatch); [regression glossary](../../../../glossary.md#term-regression))</span>

### Resources, preparation, finalization, and [Runtime](../../../../glossary.md#term-runtime) boundary

- Preserve one lowered unit, one artifact, two through four exact buffers after input
  deduplication, one distinct injective output, no input materialization, scalar or parallel-
  scalar strategy, and only the optional exact product workspace above.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [buffers glossary](../../../../glossary.md#term-buffer); [injective glossary](../../../../glossary.md#term-injective); [materialization glossary](../../../../glossary.md#materialization); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Preserve complete carrier size/alignment/accessibility/writability validation, canonical BOOL
  input validation, full referenced-span output/input non-overlap, permitted input/input overlap,
  exact assignment validation, one artifact lookup after shared slot assignment, and prepared
  resource ownership.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership))</span><span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [alignment glossary](../../../../glossary.md#term-alignment); [validation glossary](../../../../glossary.md#term-validation); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [lookup glossary](../../../../glossary.md#term-lookup); [prepared resource glossary](../../../../glossary.md#prepared-resource--preparedresource))</span>
- CPU analysis remains deterministic and measurement-free. CPU finalization generates or reuses
  the selected class after shared assignment. Runtime receives an immutable prepared executable
  and performs no scatter interpretation, generation, specialization, cache lookup, route choice,
  tuning, fallback, or benchmark-driven mutation.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [immutable glossary](../../../../glossary.md#term-immutable); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [specialization glossary](../../../../glossary.md#term-specialization); [cache glossary](../../../../glossary.md#term-cache))</span>

### Generated compatibility

- Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly from `24` to `25`. Generated scatter bytes
  and their compatibility envelope necessarily change because the output/contribution body moves
  into the class.
- Schema 25 records the embedded scatter family/type/reduction/carrier/access body and the existing
  optional scratch-bearing entry distinction. Treat schema 24 as an incompatible safe miss. In<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
  `CpuGeneratedKernelArtifactStoreTest`, write an otherwise valid envelope with version 24 and
  valid current metadata/class bytes, prove it is not loaded, and prove lookup regenerates and
  publishes the schema-25 artifact. Add no migration reader, converter, compatibility alias, or
  legacy execution bridge.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [lookup glossary](../../../../glossary.md#term-lookup); [artifact glossary](../../../../glossary.md#term-artifact); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Preserve compatible shape-polymorphic reuse. Concrete axes, batch count, tuple depth, extents,
  layout magnitudes, counts, ranges/chunks, values, validation results, scratch byte count/offset,
  carrier objects, byte offsets, slots, workers, and run identity remain cold.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [axes glossary](../../../../glossary.md#term-axis); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [tuple depth glossary](../../../../glossary.md#term-tuple-depth); [layout glossary](../../../../glossary.md#layout); [validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

### Stable automated evidence

- Add Class-File model tests for every scatter family, all five reduction modes where admitted,
  every data type, both index widths, heap-array and <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [data type glossary](../../../../glossary.md#data-type--datatype); [heap glossary](../../../../glossary.md#term-heap))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment) carriers, mixed and deduplicated
  boundary patterns, dense integer and general long forms, arbitrary layouts/ranges, scratch-free
  and scratch-bearing entries, and scalar/parallel prepared use in proportion to each risk.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Prove generated classes contain typed loops, matching operations, reduction instructions, and
  direct stores. Prove absence of `CpuScatterEmitter.execute*` or an equivalent bridge, `Object`
  descriptors/casts, generic family/type/reduction/carrier dispatch, <span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [carrier glossary](../../../../glossary.md#term-storage-carrier); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>`DataType.values`,
  `ScatterReduction.values`, per-element allocation, boxing, reflection, string/map dispatch,
  graph/Runtime/backend/cache lookup, and avoidable helper calls. Allow only the closed necessary
  carrier/JDK/<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [graph glossary](../../../../glossary.md#graph); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [cache glossary](../../../../glossary.md#term-cache); [lookup glossary](../../../../glossary.md#term-lookup); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`MemorySegment` primitives asserted explicitly by the tests.
- Test both dense and general class shapes without source-text matching, fragile absolute bytecode
  offsets, or manual-only <span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>`javap`. The retained decompilation is audit evidence; automated
  Class-File assertions are the durable gate.<span data-glossary-navigation="true"> ([decompilation glossary](../../../../glossary.md#term-decompilation))</span>
- Preserve or extend independent semantic differential evidence. Expected scatter results must
  come from `CpuScalarReferenceKernel` or test-local oracle code that neither calls generated
  helpers nor shares the generated reduction/scratch implementation.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>

### Performance acceptance protocol

- Before production edits and again after stabilization, run these two independent dense heap-
  array cases. Both use rank-one contiguous FLOAT32 data and updates, INT32 indices, axis zero,
  data extent 1,024, update extent 1,024, and therefore exactly 1,048,576 output/candidate-
  contribution examinations under the retained output-domain algorithm. Initialize<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [heap glossary](../../../../glossary.md#term-heap); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [contiguous glossary](../../../../glossary.md#term-contiguous); [FLOAT32 glossary](../../../../glossary.md#term-float32); [INT32 glossary](../../../../glossary.md#data-type--datatype); [axis glossary](../../../../glossary.md#term-axis); [extent glossary](../../../../glossary.md#term-extent))</span>
  `data[i] = (float) ((i % 251 - 125) * 0.25)` and
  `updates[i] = (float) ((i % 127 - 63) * 0.125)` in both forms:
  - unique-[target](../../../../glossary.md#term-target) `SCATTER_ELEMENTS + NONE`, with indices
    `indices[i] = (5 * i + 17) & 1023`, a deterministic [permutation](../../../../glossary.md#permutation) of every target; and
  - duplicate-index [Gather](../../../../glossary.md#gather)-compatible `SCATTER_ADD`, with `indices[i] = i >>> 1`, so targets
    0 through 511 each receive exactly two FLOAT32 contributions in logical update order and
    targets 512 through 1,023 retain their base value.
- Use the exact same case definitions, direct algorithms, deterministic inputs, fixed heap,
  compiler/run commands, and measurement/summarization code for baseline and final evidence.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>
- Use five fresh [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) [forks](../../../../glossary.md#term-jvm-fork), fixed `-Xms1g -Xmx1g`, at least five randomized warmup batches per
  fork, nine randomized measurement rounds per fork, and adaptive batches lasting at least 25
  milliseconds. Verify exact inputs and outputs before and after measurement and consume an
  observed sink outside the timed region.<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- Report every generated and direct per-fork median and ratio. Acceptance is independent per case:
  the median of generated per-fork medians must be no greater than <span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median))</span>`1.15x` the median of the
  corresponding direct-loop per-fork medians. Do not average cases, change shapes or work after
  measurement, widen the threshold, or substitute a passing family for a failing one.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape))</span>
- Keep timing outside Gradle/JUnit. Evidence is observational and cannot select or mutate
  production behavior. If either case exceeds <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [JUnit glossary](../../../../glossary.md#term-junit))</span>`1.15x`, retain `In progress` or `Review needed`,
  inspect generated and just-in-time code shape, and correct the implementation.

## Out of [scope](../../../../glossary.md#term-scope)

- New Model semantics, operations, attrs, reductions, types, Shapes, negative-index policy,
  capabilities, public APIs, or historical scatter forms<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Shapes glossary](../../../../glossary.md#shape))</span>
- Fold, ordering, random/dropout, pointwise, affine/movement, indexing, scan, aggregate, vector
  scatter/reduction, native-provider, or later semantic-family work<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [pointwise glossary](../../../../glossary.md#term-pointwise); [scan glossary](../../../../glossary.md#term-cumulative-scan); [vector glossary](../../../../glossary.md#term-vector))</span>
- New routes, resources, workspace kinds, materialization policy, public tuning controls,
  tuning-cache changes, fixed-shape classes, unrolling, algorithm substitution, relaxed numerics,
  in-place output, negative strides, dynamic Shapes, atomics, or update-centric execution<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [cache glossary](../../../../glossary.md#term-cache); [unrolling glossary](../../../../glossary.md#term-unrolling); [strides glossary](../../../../glossary.md#term-stride))</span>
- Architecture, module-boundary, dependency, shared Model/Compiler/Planning/Prepare/Runtime,
  Config, Trace, Engine, Gradle/build, Java-version, architecture-test, backend-conformance,
  integration-test, other-backend, NN, training, or unrelated documentation changes<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Detailed specifications or implementation for [CPU](../../../../glossary.md#term-cpu) 0007A0D–0007A0F or CPU 0007A1
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
- [CPU 0006B1 portable functional scatter](0006b1-portable-functional-scatter.md)
- [CPU 0007A0 generated hot-path parity correction](0007a0-generated-hot-path-parity-correction.md)
- [CPU 0007A0A affine and movement parity](0007a0a-affine-and-movement-generated-loop-parity.md)
- [CPU 0007A0B indexing parity](0007a0b-indexing-generated-loop-parity.md)

## Architecture constraints

- Model owns scatter meaning. This task changes only CPU-private generated realization and cannot
  reinterpret, narrow, or extend a semantic contract.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Planning selects CPU ownership only. CPU analysis owns deterministic lowering, specialization,
  strategy, and exact declarations; CPU finalization owns compatible generation/reuse after shared
  assignment; Runtime executes the immutable prepared result only.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Generated hot code receives typed direct carriers and primitive cold geometry. No [`Operation`](../../../../glossary.md#operation),
  `CompiledNode`, Tensor, graph object, backend discovery, route/cache policy, validation decision,
  or benchmark evidence reaches it.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend); [cache glossary](../../../../glossary.md#term-cache); [validation glossary](../../../../glossary.md#term-validation))</span>
- Prepared recipes remain [immutable](../../../../glossary.md#term-immutable) and reusable. Concurrent runs use distinct [`RunState`](../../../../glossary.md#run-state--runstate) and
  invocation-private geometry/scratch state; generated classes retain no mutable run state.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- Complete validation-before-write, canonical contribution order, general carrier/layout/range
  support, exact resource declarations, and Runtime/Prepare ownership remain unchanged.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Stop if parity requires a public/shared contract, architecture, dependency, semantic, route,
  resource, materialization, Runtime-policy, or broader-module change.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [materialization glossary](../../../../glossary.md#materialization); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — typed scatter family,
  contribution, reduction, scratch, carrier, and dense/general Class-File emission.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — exact [schema](../../../../glossary.md#term-schema) and [artifact](../../../../glossary.md#term-artifact) compatibility.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — existing closed scatter family,
  reduction, occurrence-map, boundary-type/access, and scratch-signature identity.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [signature glossary](../../../../glossary.md#term-signature); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — existing compact cold geometry,
  canonical mapping, and exact scratch sizing.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — existing deterministic plan,
  declarations, specialization, and final assignment checks.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — existing complete validation-
  before-write, direct binding, geometry packing, range/worker invocation, and scratch slicing.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [binding glossary](../../../../glossary.md#term-binding))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent [oracle](../../../../glossary.md#term-test-oracle) evidence only.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — generated route Javadoc and
  existing direct scratch-bearing entry contract.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

Packages added or changed:

- No package is added, removed, moved, exported, or made supported [API](../../../../glossary.md#term-application-programming-interface-api).

Type placement:

- `CpuScatterEmitter` remains the sole owner of generated scatter mapping, typed reduction, and
  exact-product emission. It becomes an embedded-body emitter rather than a generic bridge owner.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- `CpuClassFileKernelGenerator` passes the existing structural `CpuKernelIr` to the focused
  emitter; it does not acquire scatter semantics or Runtime state.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- `CpuCarrierEmitter` remains the unchanged focused owner of primitive-array and native-order
  segment instruction emission; do not move scatter semantics into it.
- Existing `CpuScatterIr`, `CpuScatterLowering.Geometry`, preparation, executable validation, and
  reference types retain their current owners and contracts. Add no dispatcher, registry,
  manager, facade, generic math utility, or new public type.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [facade glossary](../../../../glossary.md#term-facade))</span>

## Affected files

Expected production and package-documentation paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`

Expected [CPU](../../../../glossary.md#term-cpu) test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0007a0c-scatter-generated-loop-parity.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized. An expected path may remain unchanged after explicit review. Before
substituting a path, update this specification with the replacement and reason while keeping the
same 20-path ceiling and owning [module](../../../../glossary.md#term-module)/documentation area.

Review without modification and record explicit conclusions for:

- `ARCHITECTURE.md`, focused architecture pages, ADRs, and architecture tests;
- Model scatter operations, Tensor/API and Compile API documentation, Compiler, shared Planning,
  Prepare, Runtime, Config, Trace, Engine, backend conformance, integration, Gradle/build, native
  routes, other backends, NN, and training;<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- `CpuScatterIr`, `CpuScatterLowering`, `CpuPartitionPreparationPlan`, `CpuPreparedExecutable`,
  `CpuScalarReferenceKernel`, `CpuContiguousWorkspace`, and their package contracts when their
  existing semantics, validation, scratch, and lifecycle Javadocs remain accurate; and<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- CPU 0007A0D–0007A0F and 0007A1, which remain Draft without detailed specifications.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 20 repository paths: 7 production/package-documentation
paths, 8 [CPU](../../../../glossary.md#term-cpu) test paths, and 5 documentation/[planning](../../../../glossary.md#term-planning) paths listed above. Stop and revise this
specification before a 21st path, a new package, an executable path outside `backends/cpu`, or a
documentation path outside the allowlist. Unused capacity does not authorize cleanup.

## Acceptance criteria

- The complete pre-edit schema-24 source/Class-File audit and both timing baselines are retained
  before production edits; every class descriptor, constant-pool bridge, checksum, command,
  environment fact, raw fork sample, summary, and exact verification result is reproducible.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [descriptor glossary](../../../../glossary.md#term-descriptor); [constant glossary](../../../../glossary.md#term-constant); [checksum glossary](../../../../glossary.md#term-checksum); [fork glossary](../../../../glossary.md#term-fork-meanings); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Generated classes for all three scatter families embed carrier/type/reduction/access-specialized
  output and contribution loops and contain no <span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`CpuScatterEmitter.execute*` or equivalent
  `Object` bridge.
- Dense heap-array forms use direct typed integer loop/address state; every segment, mixed,
  arbitrary-layout, large-range, or otherwise unproved form retains a correct typed general-long
  generated fallback.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [layout glossary](../../../../glossary.md#layout))</span>
- Every CPU 0006B1 family, reduction, type, index width, occurrence-map/deduplication, scalar/zero
  domain, layout, carrier, range, parallel, overlap, validation, and resource behavior remains
  correct against an independent oracle.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [occurrence glossary](../../../../glossary.md#term-occurrence); [scalar glossary](../../../../glossary.md#term-scalar); [validation glossary](../../../../glossary.md#term-validation); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Complete bounds then `NONE` uniqueness validation remains on the bound invoking thread before
  generated work or worker submission; every invalid case leaves output and scratch unchanged.
- Exact floating-product scratch preserves once-rounded semantics, one declared/disjoint
  workspace, checked size/alignment, and allocation-free generated work; it is verified separately
  from the scratch-free performance cases.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
- Stable Class-File tests prove typed generated operations and the absence of bridges, `Object`
  carrier descriptors/casts, generic family/type/reduction dispatch, allocation, reflection,
  boxing, string/map dispatch, Runtime/backend/cache lookup, and avoidable helper calls.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [cache glossary](../../../../glossary.md#term-cache); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Generated compatibility advances exactly `24 -> 25`; schema 24 is an incompatible safe miss,
  no migration reader exists, and compatible concrete geometry still reuses one artifact.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Unique replacement `SCATTER_ELEMENTS` and duplicate-index [FLOAT32](../../../../glossary.md#term-float32) `SCATTER_ADD` each pass the
  unchanged independent `<= 1.15x` median-of-fork-medians gate under the complete five-fork
  protocol and equivalent semantic/algorithmic work.<span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median))</span>
- No public API, capability, semantic, route, materialization, resource kind, architecture,
  dependency, shared-module, Runtime-tuning, build, conformance, integration, later-task, or
  unrelated change is present.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [materialization glossary](../../../../glossary.md#materialization); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- A distinct clean documentation-focused context independently reviews the final diff and
  evidence, finalizes affected Javadocs/package summaries, CPU guide, glossary impact, task,
  master plan, and roadmap, and records every no-change conclusion before completion.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused tests for every changed test [owner](../../../../glossary.md#term-owner) while iterating. The final focused command must
name every changed test class and cover Class-File [shape](../../../../glossary.md#shape), scatter semantics, independent [oracle](../../../../glossary.md#term-test-oracle),
validation-before-write/worker submission, arbitrary range/parallel execution, scratch slicing,
preparation/finalization, [specialization](../../../../glossary.md#term-specialization), and [schema](../../../../glossary.md#term-schema)/[persistence](../../../../glossary.md#term-persistence) compatibility.

After executable Java stabilizes, run exactly one authoritative affected-[module](../../../../glossary.md#term-module) command:

```bash
./gradlew :backends:cpu:test
```

Run the isolated baseline/final performance probe separately from [Gradle](../../../../glossary.md#term-gradle)/[JUnit](../../../../glossary.md#term-junit) using the retained
commands and unchanged five-[fork](../../../../glossary.md#term-fork-meanings) protocol. Timing is evidence only and is not an ordinary test.

The distinct clean documentation pass reuses successful Java and timing evidence unless it
changes executable Java or probe behavior, or records a concrete stale-evidence reason. It runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also inspects rendered affected Javadocs; validates local Markdown links/anchors, balanced
fences, terminal newlines, trailing whitespace, exact changed-path allowlist/count, package/type
placement, schema-25 history and schema-24 rejection, stable [bytecode](../../../../glossary.md#term-bytecode) claims, baseline/final
protocol/ratios/checksums, status/[dependency](../../../../glossary.md#term-dependency) [synchronization](../../../../glossary.md#term-synchronization), and exact
0007A0C/0007A0D–0007A0F/0007A1 ordering.

Repository-wide, architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration validation remains deferred
to [CPU](../../../../glossary.md#term-cpu) 0009 or [continuous integration](../../../../glossary.md#term-continuous-integration-ci) because this task changes one [concrete backend](../../../../glossary.md#concrete-backend) and no
shared or [architecture contract](../../../../glossary.md#architecture-contract). Stop and replan if implementation makes that conclusion false.

## Dependencies

- [CPU 0006B1](0006b1-portable-functional-scatter.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> is `Complete` and owns the exact current
  scatter semantics, validation-before-write order, occurrence mapping, canonical contributions,
  represented reductions, arbitrary layout/carrier/range behavior, exact-product scratch,
  reference oracle, worker safety, declarations, and schema-16 history preserved here.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [occurrence glossary](../../../../glossary.md#term-occurrence); [canonical glossary](../../../../glossary.md#term-canonical-form); [layout glossary](../../../../glossary.md#layout); [carrier glossary](../../../../glossary.md#term-storage-carrier); [oracle glossary](../../../../glossary.md#term-test-oracle); [schema glossary](../../../../glossary.md#term-schema))</span>
- [CPU 0007A0](0007a0-generated-hot-path-parity-correction.md) is `Complete` and supplies the
  shared typed carrier emission, dense integer loop/address proof, stable Class-File testing, and
  retained five-fork performance protocol.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- [CPU 0007A0A](0007a0a-affine-and-movement-generated-loop-parity.md) is `Complete` and establishes
  the current dense/general correction pattern and schema-23 history.
- [CPU 0007A0B](0007a0b-indexing-generated-loop-parity.md) is `Complete` and establishes current
  schema 24 plus embedded mixed-type indexing loops while preserving separate pre-write validation.
- Current Java 26 Class-File API, shared Prepare/Runtime, CPU worker, direct carrier binding,
  exact workspace, and generated-artifact cache contracts are complete and unchanged.<span data-glossary-navigation="true"> ([Class-File API glossary](../../../../glossary.md#term-class-file-api); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding); [artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache))</span>

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0007A0D remains the next `Draft` corrective row and owns fold generated-loop parity after
  this task completes; do not create its detailed specification here.
- CPU 0007A0E and CPU 0007A0F remain ordered `Draft` corrections for ordering and random/[dropout](../../../../glossary.md#dropout--inverted-dropout).
- CPU 0007A1 remains `Draft` and explicitly depends on CPU 0007A0F before semantic-family work
  resumes.
- CPU 0009 retains the generated-coverage and [conformance](../../../../glossary.md#term-conformance) [checkpoint](../../../../glossary.md#term-checkpoint).

## Architecture impact

Expected impact: None.

The task changes only [CPU](../../../../glossary.md#term-cpu)-owned generated scatter code [shape](../../../../glossary.md#shape) and current-only [artifact](../../../../glossary.md#term-artifact)
compatibility under the existing analysis/finalization/[binding](../../../../glossary.md#term-binding) [lifecycle](../../../../glossary.md#lifecycle). If implementation
requires an architecture, public/shared contract, [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency), semantic, route, resource,
[materialization](../../../../glossary.md#materialization), or [Runtime](../../../../glossary.md#term-runtime)-policy change, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Implement Synaptik CPU task 0007A0C exactly from
docs/planning/backends/cpu/tasks/0007a0c-scatter-generated-loop-parity.md. Use a separate clean
implementation context. Do not use GSD. Do not commit, push, stage, revert, delete, or modify
unrelated work.

Read in full AGENTS.md, ARCHITECTURE.md, the current architecture plan and directly relevant
architecture/performance/Runtime-Prepare-Backend documents, documentation rules and applicable
profiles, planning guide, roadmap, CPU master plan, task 0007A0C, completed tasks
0006B1/0007A0/0007A0A/0007A0B, and every affected or directly relevant current CPU source/test/
package file. Inspect the complete worktree diff and preserve concurrent unrelated changes.

Record the required schema-24 generated-class decompilation and both performance baselines before
production edits. Then implement embedded typed scatter output/contribution/reduction/scratch
work, exact schema 25 compatibility, stable Class-File and semantic tests, focused validation,
one authoritative CPU suite, and the unchanged isolated five-fork per-case parity gates. Preserve
all CPU 0006B1 validation-before-write, canonical-order, exact-product, carrier/layout/range,
worker, declaration, and lifecycle contracts. Stop on any architecture, scope, semantic, shared-
contract, route, resource, lifecycle, or unresolved performance conflict.

After executable Java and probe evidence stabilize, hand the same uncommitted diff and exact
evidence to a distinct clean documentation-focused context following
docs/developer-guide/documentation-rules.md. That context must independently inspect final source,
tests, class shape, and evidence; finalize affected Javadocs/package summaries, CPU guide,
glossary impact, task/master/roadmap, rendered pages, Markdown, exact scope, schema, status/order,
performance evidence, and whitespace; and not repeat successful Java or timing runs unless it
changes executable/probe behavior or records a concrete stale-evidence reason. Do not mark
Complete until both contexts and every acceptance criterion succeed.

Return exact changed paths, commands/results, environment and per-fork/per-case ratios,
documentation/Javadoc/glossary impact and explicit no-change conclusions, context IDs,
unresolved issues, and exactly Status: Complete or Status: Incomplete with
Follow-up required: <specific follow-up>.
```

## Local decisions

- Keep all three scatter families and every reduction together because they share one bridge-only
  emitter, one output-domain contribution algorithm, one scratch-signature boundary, and one
  compatibility schema.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [schema glossary](../../../../glossary.md#term-schema))</span>
- Measure already-validated writers so timing isolates the bridge/output-contribution defect while
  production retains mandatory validation. Direct loops perform identical output-domain mapping
  and canonical contribution work.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Require separate replacement and duplicate-add gates because represented-bit selection and
  ordered reduction have different control flow; neither passing case proves the other.<span data-glossary-navigation="true"> ([represented-bit glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Advance directly to schema 25 because generated scatter bytes necessarily change. Schema 24 is
  the single prior incompatible envelope and concrete geometry remains cold.
- Keep floating-product scratch outside timing and inside semantic/Class-File/resource evidence so
  the benchmark neither simplifies nor conceals its distinct exact arithmetic contract.

## Known limitations

- Performance gates cover two representative large dense heap-array cases on the recorded local
  environment. They do not claim parity for every scatter family, reduction, type, carrier,
  layout, range, index width, CPU, or Java virtual machine.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [CPU glossary](../../../../glossary.md#term-cpu); [Java virtual machine glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- General segment/mixed/arbitrary-layout and floating-product forms receive semantic, resource,
  and stable Class-File validation but no timing threshold in this task.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Complete bounds and replacement-uniqueness validation intentionally remains a separate scalar
  execution-time pass; generated output work reads validated indices again.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Only the existing one-node fully static resolved-layout CPU scatter capability is corrected.
  Multi-node fusion, dynamic layouts, vector scatter, native routes, and new semantics remain out
  of scope.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [fusion glossary](../../../../glossary.md#term-fusion); [vector glossary](../../../../glossary.md#term-vector); [scope glossary](../../../../glossary.md#term-scope))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `01a00085-1280-7bf1-be61-f6e7b1cf1e4f` read the governing repository,
architecture, performance, [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[backend](../../../../glossary.md#backend), planning, and documentation instructions;
the [CPU](../../../../glossary.md#term-cpu) master plan and roadmap; completed CPU 0006B1/0007A0/0007A0A/0007A0B tasks; and directly
relevant scatter [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/emitter/generator/[specialization](../../../../glossary.md#term-specialization)/[cache](../../../../glossary.md#term-cache)/prepare/executable/reference
source, tests, package summaries, CPU guide, glossary, and retained benchmark evidence.

The pre-edit audit confirmed [schema](../../../../glossary.md#term-schema) 24 is current and `CpuScatterEmitter.emit` still emits only an
`invokestatic execute2*`, `execute3*`, or `execute4*`. A retained representative current class at
`/tmp/synaptik-generated-bytecode-audit/scatter.class` is 570 bytes with [SHA-256](../../../../glossary.md#term-sha-256)
`b3dc572bf64a3348a8ce70fde16e0d9f097a899e1224cc32d0eaeaf88e2d82ae`; complete `javap`
shows its sole typed segment entry loads [arguments](../../../../glossary.md#term-argument) and calls
`CpuScatterEmitter.execute4(Object,Object,Object,Object,long[],long,long)`. Source and helper
[bytecode](../../../../glossary.md#term-bytecode) show generic [carrier](../../../../glossary.md#term-storage-carrier) selection, [`DataType`](../../../../glossary.md#data-type--datatype)/reduction [lookup](../../../../glossary.md#term-lookup), per-output logical update
scans, generic [represented-bit](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) read/write/reduce helpers, and the separate exact-product scratch
implementation behind that bridge.

Planning-only validation confirmed the required section order and [`Ready`](../../../../glossary.md#term-ready) status, 20 unique
allowlisted paths, 30 unique generated heading anchors, existing [targets](../../../../glossary.md#term-target) for every local Markdown
link, balanced fences, terminal LF newlines, no carriage returns or trailing whitespace, exact
schema/status/[dependency](../../../../glossary.md#term-dependency) order, one CPU `Ready` row, and no detailed 0007A0D–0007A0F or 0007A1
task. The final changed-path inspection contains only this task, the CPU master plan, and the
CPU-specific roadmap [synchronization](../../../../glossary.md#term-synchronization) in addition to preserved concurrent [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) work. Tracked
planning changes pass `git diff --check`. No Java test, [Javadoc](../../../../glossary.md#term-javadoc), or performance command is [run](../../../../glossary.md#term-run) for
this planning-only change.

## Implementation notes

- `CpuClassFileKernelGenerator` now passes the [canonical](../../../../glossary.md#term-canonical-form) structural [IR](../../../../glossary.md#term-intermediate-representation-ir) to
  `CpuScatterEmitter`. The emitter parses only generation-time structural facts and embeds the
  selected `SCATTER_ELEMENTS`, [Gather](../../../../glossary.md#gather)-compatible `SCATTER_ADD`, or `SCATTER_ND` family, data and
  index types, reduction, occurrence mapping, carrier access, contribution matching, and output
  store in the generated class. The former <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`execute2*`/`execute3*`/`execute4*` generic bridges and
  their generic represented-value access/reduction helpers were removed.
- Cold-proved dense [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [heap](../../../../glossary.md#term-heap)-array `SCATTER_ELEMENTS` and `SCATTER_ADD` entries narrow the
  universal `long` bounds and bases once, then retain primitive integer output/update/address
  state. Typed general-long emission remains for [`MemorySegment`](../../../../glossary.md#term-memorysegment), mixed-carrier, arbitrary-layout,
  large-range, and otherwise unproved forms. A failed dense proof therefore changes generated
  shape rather than narrowing supported CPU 0006B1 behavior.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [shape glossary](../../../../glossary.md#shape); [narrowing glossary](../../../../glossary.md#term-narrowing); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Generated work still consumes only validated indices. Complete deterministic row-major bounds
  validation, followed by complete deterministic target-uniqueness validation for replacement
  forms, remains in cold binding before any generated call, worker submission, scratch mutation,
  or output write. Each generated range reads the functional base, scans contributions in
  canonical logical order, and writes each owned output coordinate exactly once.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [validation glossary](../../../../glossary.md#term-validation); [target glossary](../../../../glossary.md#term-target); [binding glossary](../../../../glossary.md#term-binding); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), and [BFLOAT16](../../../../glossary.md#term-bfloat16) `MUL` entries now embed the exact-product classification,
  sign/exponent/significand state, primitive-limb multiplication, and ties-to-even final rounding.
  They use the one existing checked, aligned, disjoint <span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent); [significand glossary](../../../../glossary.md#term-significand); [limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs); [ties-to-even glossary](../../../../glossary.md#term-ties-to-even); [rounding glossary](../../../../glossary.md#term-rounding))</span>`SCATTER_PRODUCT` slice and contain no
  generic scatter helper, <span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>`BigInteger`, collection, array allocation, or per-factor/per-output
  allocation. BFLOAT16 ADD/MIN/MAX paths likewise embed direct conversion, arithmetic/extrema, and
  ties-to-even narrowing without ordinal dispatch.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- `CpuGeneratorSchema.CURRENT_VERSION` advances exactly from 24 to 25. Schema-24 envelopes are
  incompatible safe misses and regenerate schema-25 bytes; there is no migration reader,
  converter, compatibility alias, or legacy execution bridge. Concrete geometry, values, ranges,
  scratch offsets/sizes, carriers, slots, workers, and run identity remain cold compatible facts.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [alias glossary](../../../../glossary.md#aliasing-and-arena); [run glossary](../../../../glossary.md#term-run); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Performance evidence remains observational. The two scratch-free timed class files remained
  byte-identical after the final exact-product audit/fix, so their accepted five-fork evidence was
  reused. No timing threshold or universal performance claim applies to general layouts, segment
  or mixed carriers, other families/types/reductions/index widths, or scratch-bearing forms.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>

## Completion summary

Completed changes:

- Replaced bridge-only generated scatter work with typed direct dense/general loops for all three
  current scatter families and every admitted represented reduction/type/carrier combination.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Embedded exact floating-product scratch state, advanced current generated compatibility from
  schema 24 to schema 25, and preserved complete validation-before-write and Runtime/Prepare
  ownership boundaries.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Finalized affected implementation Javadocs, package summaries, CPU guide, existing glossary
  entries, this task, the CPU master plan, and the roadmap. CPU 0007A0D is the sole next <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>`Draft`
  corrective frontier; CPU 0007A0E–0007A0F remain `Draft` without detailed specifications, and
  CPU 0007A1 remains behind CPU 0007A0F.

Files changed or created:

- Exactly 17 repository paths changed or were created: seven production/package-documentation
  paths, five CPU test paths, and five documentation/planning paths, all in the exact 20-path
  allowlist.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0007a0c-scatter-generated-loop-parity.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Tests and [validation](../../../../glossary.md#term-validation):

- Reused final implementation/audit focused evidence: the exact eight changed/risk-owner suites
  passed 105 tests with zero skips, failures, or errors.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Reused authoritative `./gradlew :backends:cpu:test` evidence: 53 suites and 312 tests, zero
  failures/errors, and one existing skip. No executable Java or probe behavior changed during the
  clean documentation pass, so the Java suite was not repeated.
- Reused the retained Java 26.0.1, macOS 26.5.2 aarch64 five-fork probe. Every individual fork was
  below <span data-glossary-navigation="true"> ([aarch64 glossary](../../../../glossary.md#term-aarch64))</span>`1.15x`; generated/direct [median](../../../../glossary.md#term-median)-of-[fork](../../../../glossary.md#term-fork-meanings)-medians ratios were `0.979533` for unique
  `SCATTER_ELEMENTS + NONE` and `0.983230` for duplicate-index [FLOAT32](../../../../glossary.md#term-float32) `SCATTER_ADD`.
- Representative final [schema](../../../../glossary.md#term-schema)-25 [SHA-256](../../../../glossary.md#term-sha-256) checksums are
  `557ba806c649f2e758b8013cd9090c1dd18dfd3a3ef57694e1be12b39da3b61e` for general-segment
  BFLOAT16 ADD, <span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>`13f20bacbab22a25aa741adaa78405407e8ff531b525c577c3528ad3a3837c95`
  for exact MUL scratch, `bbcf2cc940567dedc581530e8177964b95de274bc5886da96493b453c3b6c59c`
  for dense Elements NONE, <span data-glossary-navigation="true"> ([Elements glossary](../../../../glossary.md#term-element))</span>`8b1b17b3fa83135279eadff55134e13f1285b92938642cee4bbd65b02c4e5194`
  for dense Scatter Add, and<span data-glossary-navigation="true"> ([Scatter Add glossary](../../../../glossary.md#scatter-add))</span>
  `ad1e3934e5ae9f1c2f7fc1c20c47e88575ff9fd175aed26e2783585056217631`
  for general-segment Scatter-ND.<span data-glossary-navigation="true"> ([Scatter-ND glossary](../../../../glossary.md#scatter-nd))</span>
- Clean documentation context `01a000c9-bb65-7d52-82ba-29ce0c26157a` ran
  `./gradlew :backends:cpu:javadoc` successfully with only the two expected incubating-module
  warnings and inspected the rendered affected type and package pages. Final Markdown
  link/anchor/fence/newline/whitespace, exact allowlist/path-count, schema/status/order/evidence,
  checksum, and <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [checksum glossary](../../../../glossary.md#term-checksum))</span>`git diff --check` gates passed.

Documentation impact:

- The CPU guide now explains the embedded direct-loop boundary, dense/general distinction,
  validation-before-write ownership, inline exact-product scratch, schema-25 compatibility, and
  the bounded meaning of the retained performance evidence.
- Existing glossary entries were updated for the changed CPU generated-kernel and specialization
  boundaries. No new reusable project term was introduced, so no new glossary heading was needed.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Javadocs and package summaries now describe the direct scatter body, optional inline
  exact-product state, schema 25, and the separation between generated work and cold validation.

Reasoned no-change conclusions:

- Public Tensor/API and Compile API documentation, Model scatter semantics, Compiler, and shared
  Planning/Prepare/Runtime/Config/Trace/Engine remain unchanged because this correction changes
  only CPU-private generated realization after the existing semantic and lifecycle boundaries.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `ARCHITECTURE.md`, focused architecture pages, ADRs, dependency rules, and architecture tests
  remain unchanged because no ownership, module boundary, dependency, resource kind, or Runtime
  policy changed.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Backend conformance and integration tests remain unchanged and their repository-wide checkpoint
  remains CPU 0009/CI because no public behavior or cross-backend contract changed.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>
- Gradle/build configuration and the Java version remain unchanged; existing Java 26 Class-File
  and incubating Vector API configuration already covers the implementation.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [Vector API glossary](../../../../glossary.md#term-vector-api))</span>
- Native routes, other backends, NN, and training remain unchanged because this task neither
  changes capability advertising nor adds a peer route, public semantic, gradient, or training
  behavior. Concurrent NN planning work was preserved and excluded from the CPU allowlist count.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- `CpuScatterIr`, `CpuScatterLowering`, `CpuPartitionPreparationPlan`, `CpuPreparedExecutable`,
  `CpuScalarReferenceKernel`, `CpuContiguousWorkspace`, and their package contracts remain accurate:
  their semantic occurrence mapping, compact geometry, validation, workspace, oracle, binding,
  and lifecycle responsibilities did not change.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [oracle glossary](../../../../glossary.md#term-test-oracle); [binding glossary](../../../../glossary.md#term-binding))</span>

Unresolved issues: None.

Follow-up required: None.

Status: Complete
