# Task 0007A0D: Fold Generated-Loop Parity

## Reader context

A fold emitter translates a known overlap mapping into code; it is not a fold evaluator
consulted for every input contribution. Here the generated class began to contain the selected
family mapping, typed addition, and carrier access directly instead of calling a generic static
body. A cold dense-array proof enabled integer cursors, while arbitrary layouts and
segment/mixed carriers retained long addressing.

Each output cell still began at positive zero and consumed its historical canonical contribution
order. Parallelism remained disjoint output ownership, with no new atomics, scratch, or partial
reductions. The preserved BFLOAT16 per-contribution rounding and measured dense FOLD_AXIS case
are checkpoint-specific, not a universal numerical or speed promise. Current window semantics
and emission ownership explain the stable boundaries.

Current reading alongside this historical record:

- [Window-transform expressions](../../../../reference/tensor/windows.md#window-transform-expressions)
  — distinguishes extraction, columns geometry, and overlap accumulation.
- [CPU generated-code ownership](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit)
  — separates generation-time emitters, emitted typed work, and JVM compilation.

## Status

Complete

## Goal

Replace the bridge-only generated contribution work for the completed [CPU](../../../../glossary.md#term-cpu) overlap-fold family
with [carrier](../../../../glossary.md#term-storage-carrier)-, type-, family-, access-, and loop-form-specialized bodies embedded in each generated
class. `FOLD_AXIS` and `FOLD2D` must preserve every CPU 0006B2 semantic, [layout](../../../../glossary.md#layout), carrier, range,
resource, parallelism, [validation](../../../../glossary.md#term-validation), and [lifecycle](../../../../glossary.md#lifecycle) contract while one representative dense
overlapping [FLOAT32](../../../../glossary.md#term-float32) `FOLD_AXIS` case reaches reproducible near-parity with the equivalent direct
Java loop.

This is one cohesive CPU-private realization correction. Both fold families share `CpuFoldIr`,
compact cold geometry, `CpuFoldEmitter`, positive-zero output initialization, [canonical](../../../../glossary.md#term-canonical-form) input-row-
major contribution scanning, represented addition, disjoint output ranges, and one generated-
[artifact](../../../../glossary.md#term-artifact) compatibility boundary. The task changes generated code [shape](../../../../glossary.md#shape) only; completed CPU 0006B2
capability and semantics remain unchanged.

## [Scope](../../../../glossary.md#term-scope)

### Mandatory pre-edit source and Class-File baseline

- Before changing production Java, create a fresh isolated `/tmp` evidence directory and capture
  current schema-25 generated classes for:<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [schema glossary](../../../../glossary.md#term-schema))</span>
  - the exact dense [heap](../../../../glossary.md#term-heap)-array [FLOAT32](../../../../glossary.md#term-float32) `FOLD_AXIS` performance case defined below;
  - one dense [INT32](../../../../glossary.md#data-type--datatype) or [INT64](../../../../glossary.md#data-type--datatype) `FOLD_AXIS` form;
  - one general-[layout](../../../../glossary.md#layout) or mixed heap-array/[`MemorySegment`](../../../../glossary.md#term-memorysegment) [BFLOAT16](../../../../glossary.md#term-bfloat16) `FOLD_AXIS` form; and
  - one `FOLD2D` form with nonzero [padding](../../../../glossary.md#padding) and either [dilation](../../../../glossary.md#term-dilation) or ceil mode.
- Retain the probe source, exact compile/generate/decompile/run/summarize commands, Java virtual
  machine and operating-system facts, class bytes, complete <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run); [Java virtual machine glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>`javap -c -p` and `javap -v -p`
  output, sizes, SHA-256 checksums, raw fork samples, summaries, ratios, and verification results.<span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256); [fork glossary](../../../../glossary.md#term-fork-meanings); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Record every generated entry descriptor and constant-pool method reference. Planning observed
  the dense performance form as a 398-byte schema-25 class with entry descriptor <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [constant glossary](../../../../glossary.md#term-constant); [Planning glossary](../../../../glossary.md#term-planning))</span>`([F[F[JJJ)V`,
  SHA-256 `b82d17950751985e8b250e1aee6b24ae45cec547cf1ba3fc9357816e126c5a26`, and a body
  containing typed argument loads followed by<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
  `CpuFoldEmitter.execute(Object,Object,long[],long,long)`. Reproduce this baseline freshly rather
  than treating planning evidence as implementation evidence.
- Audit `CpuFoldEmitter`, `CpuClassFileKernelGenerator`, the packed geometry, and the generated
  classes before selecting locals or control flow. Do not assume that fold shares the final
  scatter emitter shape merely because both scan contributions by output.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Time only warmed fold output/contribution work. Exclude lowering, generation, class definition,
  artifact lookup, cold binding, geometry packing, allocation, input initialization, overlap and
  carrier validation, result verification, and sink observation from both timed forms.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [artifact glossary](../../../../glossary.md#term-artifact); [lookup glossary](../../../../glossary.md#term-lookup); [binding glossary](../../../../glossary.md#term-binding); [carrier glossary](../../../../glossary.md#term-storage-carrier); [validation glossary](../../../../glossary.md#term-validation))</span>

### Embedded typed fold bodies

- Change `CpuClassFileKernelGenerator` and `CpuFoldEmitter` so each generated entry embeds the
  selected `FOLD_AXIS` or `FOLD2D` coordinate mapping, represented type, carrier accesses,
  positive-zero initialization, contribution matching, sequential represented addition, and
  output store instead of calling <span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`CpuFoldEmitter.execute` or an equivalent generic bridge.
- Select at generation time from existing [CPU](../../../../glossary.md#term-cpu)-private typed structural facts:
  - `FOLD_AXIS` for [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), [INT32](../../../../glossary.md#data-type--datatype), and [INT64](../../../../glossary.md#data-type--datatype);
  - `FOLD2D` for FLOAT64, FLOAT32, and BFLOAT16 only;
  - the exact matching primitive [heap](../../../../glossary.md#term-heap)-array [carrier](../../../../glossary.md#term-storage-carrier) or native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment) independently
    for input and output; and
  - cold-proved dense heap-array integer addressing or typed general long addressing.
- Preserve the universal primitive `long start, long end` entry. Each arbitrary legal half-open
  range owns disjoint flattened output ordinals, initializes each owned output coordinate from
  represented positive zero, scans all logical input positions in canonical row-major order,
  includes exactly the matching contributions, and writes the owned output coordinate once.<span data-glossary-navigation="true"> ([half-open glossary](../../../../glossary.md#term-half-open-interval); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- Generate direct typed array or native-order `MemorySegment` loads and stores. Generated hot work
  must contain no `Object` carrier, cast, [`DataType`](../../../../glossary.md#data-type--datatype) lookup, family/type/carrier ordinal dispatch,
  array-versus-segment test, map, string dispatch, reflection, graph object, Runtime policy, route
  or cache lookup, boxing, or per-output/per-contribution allocation.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch); [graph glossary](../../../../glossary.md#graph); [Runtime glossary](../../../../glossary.md#term-runtime); [cache glossary](../../../../glossary.md#term-cache))</span>
- Hoist family, type, ranks, axis role, kernel roles, and structurally invariant packed-geometry
  positions out of the output and contribution loops. Concrete extents, offsets, stride
  magnitudes, window/step/padding/dilation/grid values, carrier bases, and ranges remain cold
  shape-polymorphic geometry.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [kernel glossary](../../../../glossary.md#kernel); [invariant glossary](../../../../glossary.md#term-invariant); [stride glossary](../../../../glossary.md#term-stride); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation); [shape glossary](../../../../glossary.md#shape))</span>

### Exact FOLD_AXIS behavior

- Preserve the current rank relationship: the input rank is the output rank plus one, the final
  input dimension is the positive window size, and the selected normalized output axis receives<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [dimension glossary](../../../../glossary.md#dimension); [axis glossary](../../../../glossary.md#term-axis))</span>
  `inputCoordinate[axis] * step + inputCoordinate[last]`. Every other input coordinate maps to
  the output coordinate at the same axis position.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Preserve scalar outputs and inputs where admitted, zero-sized logical domains, windows that
  leave output positions uncovered, steps greater than one, overlap, every normalized axis, and
  arbitrary legal partial output ranges.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [normalized axis glossary](../../../../glossary.md#normalized-axis))</span>
- For dense [heap](../../../../glossary.md#term-heap) arrays, use `DENSE_HEAP_ARRAY_INT`. Narrow universal bounds, bases, counts, and
  every proved array-index fact once before hot work, then retain primitive integer output,
  contribution, coordinate, and address state without per-contribution `l2i`, generic address
  helpers, or repeated rank/type decisions.
- The dense form may specialize the current output-domain traversal but must not change its
  semantic order. In particular, it may derive matching contribution coordinates directly only
  if automated tests prove that the contributions are visited in the exact same logical input-
  row-major order for every owned output and that BFLOAT16 and floating results are bitwise
  unchanged. The performance baseline must use the same final selected traversal on both sides.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>

### Exact FOLD2D behavior

- Preserve [canonical](../../../../glossary.md#term-canonical-form) [rank](../../../../glossary.md#term-rank)-three columns input to rank-four [NCHW](../../../../glossary.md#term-nchw) output mapping. Decode the input
  `(batch, channel-times-kernel, column)` coordinates into channel, kernel height/width, and
  output-column height/width using the retained kernel and column-grid geometry.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [channel glossary](../../../../glossary.md#term-channel); [kernel glossary](../../../../glossary.md#kernel))</span>
- Map height and width exactly as `oh * stride - padding + kernelCoordinate * dilation`. Exclude
  every contribution whose mapped height or width is outside the NCHW output; never clamp, wrap,
  reflect, or materialize padded values.
- Preserve floor and ceil grid modes already validated by lowering, nonzero symmetric padding,
  dilation, overlap, multiple batches/channels, scalar and zero domains, arbitrary legal ranges,
  and the exact floating-only type boundary.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- `FOLD2D` receives stable semantic and Class-File coverage but no separate performance threshold
  in this task. Its unmeasured status must not permit an `Object` bridge or generic [dispatch](../../../../glossary.md#term-dispatch).

### Typed arithmetic requirements

- FLOAT64 contributions add with one binary64 addition per contribution; FLOAT32 contributions
  add with one binary32 addition per contribution. Preserve Java special-value, signed-zero,
  subnormal, overflow, infinity, NaN, cancellation, and sequential-rounding behavior.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [FLOAT32 glossary](../../../../glossary.md#term-float32); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [subnormal glossary](../../../../glossary.md#term-subnormal); [overflow glossary](../../../../glossary.md#term-overflow); [infinity glossary](../../../../glossary.md#term-infinity); [NaN glossary](../../../../glossary.md#term-nan); [cancellation glossary](../../../../glossary.md#term-cancellation); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- BFLOAT16 starts from represented positive zero, widens each represented contribution to
  FLOAT32, adds to the widened current BFLOAT16 accumulator, and narrows immediately after every
  contribution with the existing round-to-nearest, ties-to-even policy and NaN quieting. Do not
  accumulate several contributions in FLOAT32 or widen the final rounding point.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [accumulator glossary](../../../../glossary.md#accumulator-type); [round-to-nearest, ties-to-even glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- INT32 and INT64 start at zero and use Java two's-complement modular addition after every
  contribution. Do not widen INT32 accumulation, reject overflow, saturate, or use exact-real
  summation.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [two's-complement glossary](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks))</span>
- Preserve canonical input-row-major contribution order independently of physical layouts and
  worker scheduling. Do not use atomics, output-centric reordering that changes contribution
  order, partial sums, tree combination, vector reduction, widened accumulation, or parallel
  contributions to one output.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major); [layouts glossary](../../../../glossary.md#layout); [vector glossary](../../../../glossary.md#term-vector))</span>

### Dense and general access forms

- Generate typed `GENERAL_LONG` forms for every supported [`MemorySegment`](../../../../glossary.md#term-memorysegment), mixed-carrier,
  arbitrary non-negative input layout, zero-stride input read, offset/strided injective output,
  large range, and otherwise unproved case. Use primitive carry/reset state and only the checked
  long coordinate/address arithmetic required by the selected mapping and access plans.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [stride glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective); [carry glossary](../../../../glossary.md#term-carry); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- A failed dense proof selects the correct general form and never rejects a CPU 0006B2-supported
  occurrence. Input and output carriers may independently be their exact heap-array form or a
  native-order segment; mixed forms remain supported.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [occurrence glossary](../../../../glossary.md#term-occurrence); [heap glossary](../../../../glossary.md#term-heap))</span>
- Preserve arbitrary non-negative resolved input layouts, including zero-stride reads, and every
  validated injective non-negative output layout. Negative strides and unresolved or dynamic
  layouts remain unsupported as before.
- Scalar and parallel-scalar prepared plans must reuse the same compatible scalar artifact.
  Parallel orchestration partitions only the output domain into disjoint ranges; all coordinate
  state is invocation-private and no generated class retains mutable run state.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [artifact glossary](../../../../glossary.md#term-artifact); [mutable glossary](../../../../glossary.md#term-mutable); [run state glossary](../../../../glossary.md#run-state--runstate))</span>

### [Validation](../../../../glossary.md#term-validation), resources, preparation, and [lifecycle](../../../../glossary.md#lifecycle)

- Preserve exactly one lowered unit, one generated artifact, one distinct input and one distinct
  injective output buffer, no input materialization, no workspace, and scalar or parallel-scalar
  strategy. The output remains distinct from and physically non-overlapping with the complete
  referenced input span.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [injective glossary](../../../../glossary.md#term-injective); [buffer glossary](../../../../glossary.md#term-buffer); [materialization glossary](../../../../glossary.md#materialization); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Preserve complete carrier type, size, alignment, accessibility, read-only/writability, range,
  assignment, and referenced-span validation before any generated call, output write, or worker
  submission. Preserve permitted invocation-private geometry and read-only input sharing.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- Preserve exact lowering validation for one fully static resolved-layout occurrence, supported
  family/type/Shape relationships, checked geometry arithmetic, non-negative layouts, injective
  output, and capability fail-closed behavior. Generated code consumes these validated facts and
  does not reinterpret or repeat policy validation.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [layout glossary](../../../../glossary.md#layout); [occurrence glossary](../../../../glossary.md#term-occurrence); [Shape glossary](../../../../glossary.md#shape); [fail-closed behavior glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- CPU analysis remains deterministic and measurement-free. CPU finalization generates or reuses
  the selected class only after shared slot assignment. Runtime receives an immutable prepared
  executable and performs no fold interpretation, generation, specialization, artifact lookup,
  route choice, tuning, fallback, or benchmark-driven mutation.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [specialization glossary](../../../../glossary.md#term-specialization); [lookup glossary](../../../../glossary.md#term-lookup))</span>

### Generated compatibility transition

- Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly from `25` to `26`. Schema 26 is the first
  compatibility version whose fold classes embed carrier-, type-, family-, access-, mapping-, and
  addition-specialized contribution bodies. This is a required incompatible transition because
  generated fold bytes and their executable dispatch boundary change; retaining schema 25 would
  permit a persisted bridge-only class to satisfy a direct-loop request incorrectly.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Treat [schema](../../../../glossary.md#term-schema) 25 as an incompatible safe miss. In `CpuGeneratedKernelArtifactStoreTest`, write an
  otherwise valid envelope with version 25 and valid current metadata/class bytes, prove it is not
  loaded, then prove lookup regenerates and publishes schema-26 bytes. Add no migration reader,
  converter, compatibility alias, or legacy execution bridge.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [lookup glossary](../../../../glossary.md#term-lookup); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Preserve compatible shape-polymorphic reuse. Concrete extents, axes, window sizes, steps,
  padding, dilation, column grids, layout magnitudes, ranges, carriers, base offsets, buffer slots,
  workers, and run identity remain cold and absent from the structural specialization.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [axes glossary](../../../../glossary.md#term-axis); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation); [layout glossary](../../../../glossary.md#layout); [buffer glossary](../../../../glossary.md#term-buffer); [run glossary](../../../../glossary.md#term-run); [identity glossary](../../../../glossary.md#term-identity-meanings); [specialization glossary](../../../../glossary.md#term-specialization))</span>

### Stable automated Class-File and semantic evidence

- Add Java Class-File [model](../../../../glossary.md#term-model) assertions for both families, every admitted type, [heap](../../../../glossary.md#term-heap)-array and
  [`MemorySegment`](../../../../glossary.md#term-memorysegment) input/output, mixed carriers, dense integer and general long forms, zero-stride
  input, strided output, arbitrary ranges, and scalar/parallel prepared reuse in proportion to
  risk.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Prove generated classes contain typed loops, family-specific coordinate matching, exact typed
  additions, and direct typed stores. Prove absence of <span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`CpuFoldEmitter.execute` or an equivalent
  bridge, `Object` [descriptors](../../../../glossary.md#term-descriptor)/casts, `DataType.values`, generic family/type/carrier dispatch,
  allocation, boxing, reflection, string/map dispatch, graph/Runtime/backend/cache lookup, and
  avoidable per-element helper calls. Allow only a closed set of explicitly asserted JDK and<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [dispatch glossary](../../../../glossary.md#term-dispatch); [graph glossary](../../../../glossary.md#graph); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [cache glossary](../../../../glossary.md#term-cache); [lookup glossary](../../../../glossary.md#term-lookup); [element glossary](../../../../glossary.md#term-element); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>
  `MemorySegment` primitives needed for carriers and [BFLOAT16](../../../../glossary.md#term-bfloat16) conversion/[rounding](../../../../glossary.md#term-rounding).
- Assert dense and general class shapes through the Java Class-File API, not source-text matching,
  fragile absolute bytecode offsets, or manual-only <span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [Class-File API glossary](../../../../glossary.md#term-class-file-api); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>`javap`. Retained decompilation is audit
  evidence; automated assertions are the durable regression gate.<span data-glossary-navigation="true"> ([decompilation glossary](../../../../glossary.md#term-decompilation); [regression glossary](../../../../glossary.md#term-regression))</span>
- Preserve or extend independent semantic differentials against `CpuScalarReferenceKernel` or
  test-local oracle code that does not call the generated emitter or share its mapping/addition
  implementation. Cover overlap, uncovered positive zero, partial ranges, both integral modular
  boundaries, BFLOAT16 per-contribution rounding, floating special values and cancellation,<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [cancellation glossary](../../../../glossary.md#term-cancellation))</span>
  `FOLD2D` padding exclusion/dilation/ceil behavior, mixed carriers, zero-stride input, strided
  output, and scalar/parallel bitwise parity.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation))</span>

### Independent reproducible performance acceptance protocol

- Use exactly one dense [heap](../../../../glossary.md#term-heap)-array [FLOAT32](../../../../glossary.md#term-float32) `FOLD_AXIS` case: output [Shape](../../../../glossary.md#shape) `[1024]`, [axis](../../../../glossary.md#term-axis) `0`,
  output size `1024`, window size `16`, step `8`, input Shape `[127,16]`, contiguous layouts, and
  the complete output range <span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [layouts glossary](../../../../glossary.md#layout))</span>`[0,1024)`. This creates deterministic overlap: interior covered
  positions receive two contributions and no output is processed by more than one range.
- Initialize the 2,032 input values once per [fork](../../../../glossary.md#term-fork-meanings) as
  `input[i] = (float) ((i % 251 - 125) * 0.25)`. Before timing, verify the generated and direct
  outputs bitwise against an independent oracle and verify that the fixed input remains unchanged.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- The direct Java body must implement the exact canonical output-domain algorithm used for the
  generated acceptance form: for each output ordinal in <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>`[0,1024)`, start with `+0.0f`; [scan](../../../../glossary.md#term-cumulative-scan)
  `inputOrdinal` from `0` through `2031`; derive
  `window = inputOrdinal / 16`, `within = inputOrdinal - window * 16`, and
  `target = window * 8 + within`; add `input[inputOrdinal]` with [binary32](../../../../glossary.md#floating-point-precision-and-rounding) addition exactly when
  `target == outputOrdinal`; then store once. The generated timed form must perform semantically
  and algorithmically equivalent mapping and ordered arithmetic. Do not compare with an
  input-centric scatter, precomputed incidence table, grouped list, vector reduction, reordered
  traversal, widened accumulator, or otherwise easier direct algorithm.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [accumulator glossary](../../../../glossary.md#accumulator-type))</span>
- Use a standalone direct-Java probe outside [Gradle](../../../../glossary.md#term-gradle) and [JUnit](../../../../glossary.md#term-junit). [Run](../../../../glossary.md#term-run) five fresh [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) [forks](../../../../glossary.md#term-jvm-fork) with fixed
  `-Xms1g -Xmx1g`, at least five randomized warmup batches per fork, nine randomized measurement
  rounds per fork, and adaptive batch counts that make each timed sample last at least 25 ms.
  Randomize generated/direct order within each batch with deterministic seed<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [seed glossary](../../../../glossary.md#term-seed))</span>
  `0x5A17D00D7A0D26L` reset identically in every fork. Consume an observable sink outside the
  timed region and verify inputs and bitwise outputs again afterward.
- Report every generated and direct per-fork median and ratio plus the aggregate median of the
  five generated per-fork medians divided by the aggregate median of the five direct per-fork
  medians. Acceptance requires every per-fork ratio and the aggregate ratio to be <span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median))</span>`<= 1.15x`.
  Do not average forks, drop an outlier, change the case after baseline measurement, widen the
  threshold, or substitute another family/type/shape/range.<span data-glossary-navigation="true"> ([outlier glossary](../../../../glossary.md#term-outlier))</span>
- Use the exact same source, deterministic data, shape, range, direct algorithm, heap, commands,
  warmup/measurement protocol, random seed, and summarizer for pre-edit and final evidence.
  Performance evidence is observational and must never select or mutate production behavior. If
  any per-fork or aggregate gate fails, keep the task <span data-glossary-navigation="true"> ([measurement protocol glossary](../../../../glossary.md#term-measurement-protocol); [random seed glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>`In progress` or `Review needed`, inspect
  generated and just-in-time code shape, and correct the implementation.

## Out of [scope](../../../../glossary.md#term-scope)

- New Model semantics, operations, attrs, types, Shapes, padding policy, capabilities, public APIs,
  gradient behavior, or historical fold forms<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Shapes glossary](../../../../glossary.md#shape); [padding glossary](../../../../glossary.md#padding); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Ordering, random/dropout, pointwise, affine/movement, indexing, scatter, scan, aggregate, vector
  fold/reduction, native-provider, or later semantic-family implementation<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [pointwise glossary](../../../../glossary.md#term-pointwise); [scan glossary](../../../../glossary.md#term-cumulative-scan); [vector glossary](../../../../glossary.md#term-vector))</span>
- New routes, resources, workspace kinds, materialization policy, public tuning controls, tuning-
  cache changes, fixed-shape classes, unrolling, algorithm substitution, relaxed numerics, in-
  place output, negative strides, dynamic Shapes, atomics, or contribution-parallel execution<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [cache glossary](../../../../glossary.md#term-cache); [unrolling glossary](../../../../glossary.md#term-unrolling); [strides glossary](../../../../glossary.md#term-stride))</span>
- Architecture, module-boundary, dependency, shared Model/Compiler/Planning/Prepare/Runtime,
  Config, Trace, Engine, Gradle/build, Java-version, architecture-test, backend-conformance,
  integration-test, other-backend, NN, training, or unrelated documentation changes<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Detailed specifications or implementation for CPU 0007A0E, CPU 0007A0F, CPU 0007A1, or any
  later task<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
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
- [CPU 0006B2 portable overlap fold](0006b2-portable-overlap-fold.md)
- [CPU 0007A0 generated hot-path parity correction](0007a0-generated-hot-path-parity-correction.md)
- [CPU 0007A0A affine and movement parity](0007a0a-affine-and-movement-generated-loop-parity.md)
- [CPU 0007A0B indexing parity](0007a0b-indexing-generated-loop-parity.md)
- [CPU 0007A0C scatter parity](0007a0c-scatter-generated-loop-parity.md)

## Architecture constraints

- Model owns fold meaning. This task changes only CPU-private generated realization and cannot
  reinterpret, narrow, or extend the semantic contract.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Planning selects CPU ownership only. CPU analysis owns deterministic lowering, specialization,
  strategy, and exact declarations; CPU finalization owns compatible generation/reuse after shared
  assignment; Runtime executes the immutable prepared result only.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- [Runtime](../../../../glossary.md#term-runtime) hot code receives typed direct carriers and primitive cold geometry. No [`Operation`](../../../../glossary.md#operation),
  `CompiledNode`, Tensor, graph object, backend discovery, route/cache policy, validation decision,
  or benchmark evidence reaches it.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend); [cache glossary](../../../../glossary.md#term-cache); [validation glossary](../../../../glossary.md#term-validation))</span>
- Prepared recipes and generated artifacts remain immutable and reusable. Concurrent runs use
  distinct <span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact))</span>[`RunState`](../../../../glossary.md#run-state--runstate) objects and invocation-private packed [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) state.
- General layouts/carriers/ranges, canonical contribution order, exact represented arithmetic,
  disjoint output ownership, zero resources, and the Runtime/Prepare ownership boundary remain
  unchanged.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [canonical glossary](../../../../glossary.md#term-canonical-form); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Stop if parity requires a public/shared contract, architecture, dependency, semantic, route,
  resource, materialization, Runtime-policy, or broader-module change.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [materialization glossary](../../../../glossary.md#materialization); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — typed family mapping,
  contribution, arithmetic, carrier, and dense/general Class-File emission.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — [schema](../../../../glossary.md#term-schema) and [artifact](../../../../glossary.md#term-artifact) compatibility.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — existing fold family, represented type,
  addition policy, and access identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — existing validated compact layout,
  range, axis, and NCHW geometry.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [axis glossary](../../../../glossary.md#term-axis); [NCHW glossary](../../../../glossary.md#term-nchw))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — existing deterministic declarations,
  strategy, specialization, and final assignment checks.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — existing binding, overlap
  validation, geometry packing, range partitioning, and worker invocation.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent semantic [oracle](../../../../glossary.md#term-test-oracle) only.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — current generated-route
  boundary documentation.

Packages added or changed:

- No package is added, removed, moved, exported, or made supported [API](../../../../glossary.md#term-application-programming-interface-api).

Type placement:

- `CpuFoldEmitter` remains the sole owner of generated fold coordinate matching and represented
  addition. It becomes an embedded-body emitter rather than a generic bridge owner.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- `CpuClassFileKernelGenerator` passes the existing structural `CpuKernelIr` to the focused
  emitter; it does not acquire fold semantics, geometry packing, or Runtime state.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- `CpuCarrierEmitter` may be reused for closed typed carrier instructions but does not acquire fold
  mapping or arithmetic policy.
- `CpuFoldIr`, `CpuFoldLowering.Geometry`, preparation, executable validation, and reference types
  retain their current owners. Add no dispatcher, registry, manager, or broad helper.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry))</span>

## Affected files

Exact implementation allowlist:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFoldEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFoldGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0007a0d-fold-generated-loop-parity.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Existing `CpuFoldIr`, `CpuFoldLowering`, `CpuPartitionPreparationPlan`, `CpuPreparedExecutable`,
`CpuScalarReferenceKernel`, their focused tests, build files, architecture documents, and shared
[modules](../../../../glossary.md#term-module) are inspection-only unless the task is stopped and replanned.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may modify at most 18 repository paths, all drawn from the exact allowlist above. It may
create no production or test type and no later task specification. Temporary probe/evidence files
must remain under one isolated `/tmp` directory and are not repository paths.

If implementation requires any non-allowlisted repository path, more than 18 changed paths, a new
type, or a broader contract, stop before editing it and request a [planning](../../../../glossary.md#term-planning) update. Do not spend the
allowlist on unrelated cleanup.

## Acceptance criteria

- Current schema-25 source and generated dense/general classes are freshly captured and audited
  before production edits, including exact typed descriptors, bytes, checksums, full <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>`javap`, and
  the observed bridge call.
- Every admitted `FOLD_AXIS` and `FOLD2D` specialization embeds typed family/mapping,
  contribution, addition, and store work with no <span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>`CpuFoldEmitter.execute` or equivalent generic
  hot bridge.
- Dense heap-array forms use cold-proved integer state; every admitted segment, mixed-carrier,
  arbitrary-layout, zero-stride-read, strided-output, large-range, or otherwise unproved form
  retains correct typed general-long execution.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [stride glossary](../../../../glossary.md#term-stride))</span>
- Positive-zero initialization, canonical input-row-major order, binary64/binary32 sequential
  rounding, BFLOAT16 rounding after every contribution, modular INT32/INT64 addition, <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [rounding glossary](../../../../glossary.md#term-rounding); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>`FOLD2D`
  padding exclusion, partial ranges, and scalar/parallel bitwise parity are independently proved.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Existing capability, lowering validation, exact two-buffer/no-workspace/no-materialization
  declarations, carrier/overlap/assignment checks, disjoint output ranges, immutable lifecycle,
  and Runtime/Prepare ownership remain unchanged.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [validation glossary](../../../../glossary.md#term-validation); [buffer glossary](../../../../glossary.md#term-buffer); [materialization glossary](../../../../glossary.md#materialization); [immutable glossary](../../../../glossary.md#term-immutable); [lifecycle glossary](../../../../glossary.md#lifecycle); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Schema advances exactly from 25 to 26; schema-25 persistence is an incompatible safe miss that
  regenerates schema-26 bytes; concrete geometry remains shape-polymorphically reusable.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [shape glossary](../../../../glossary.md#shape))</span>
- Stable Java Class-File and semantic tests cover the stated family/type/carrier/layout/range/
  parallelism matrix and fail if the generic bridge, hot dispatch, allocation, or forbidden
  references return.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- All five dense [FLOAT32](../../../../glossary.md#term-float32) `FOLD_AXIS` forks and the aggregate median-of-fork-medians ratio pass the
  unchanged independent <span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>`<= 1.15x` direct-Java gate with exact pre/post verification.
- Only allowlisted paths change, no more than 18 repository paths change, and no later detailed
  task specification exists.
- A distinct clean documentation-focused context independently reviews final source, tests,
  generated class evidence, and performance evidence; finalizes affected Javadocs/package
  summaries, CPU guide, glossary impact, this task, CPU master plan, and roadmap; and records
  reasoned no-change conclusions before completion.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused tests for each changed test [owner](../../../../glossary.md#term-owner) while iterating. The final focused command must name
every changed test class and cover Class-File [shape](../../../../glossary.md#shape), both fold families and all admitted types,
dense/general and mixed carriers, arbitrary [layouts](../../../../glossary.md#layout)/ranges, independent [oracle](../../../../glossary.md#term-test-oracle) differentials,
parallel reuse, preparation/finalization, [specialization](../../../../glossary.md#term-specialization), and [schema](../../../../glossary.md#term-schema)/[persistence](../../../../glossary.md#term-persistence) compatibility.

After executable Java stabilizes, run exactly one authoritative affected-[module](../../../../glossary.md#term-module) command:

```bash
./gradlew :backends:cpu:test
```

Run the isolated baseline/final performance probe separately from [Gradle](../../../../glossary.md#term-gradle)/[JUnit](../../../../glossary.md#term-junit) with the exact
five-[fork](../../../../glossary.md#term-fork-meanings) protocol. Timing is evidence, not an ordinary test.

The distinct clean documentation pass reuses successful Java and timing evidence unless it
changes executable Java or probe behavior, or records a concrete stale-evidence reason. It runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also inspects rendered affected Javadocs; validates local Markdown links and anchors, balanced
fences, terminal newlines, carriage returns, trailing whitespace, exact changed-path allowlist and
count, package/type placement, schema-26 history and schema-25 rejection, generated [descriptors](../../../../glossary.md#term-descriptor)
and forbidden references, baseline/final protocol and ratios, status/[dependency](../../../../glossary.md#term-dependency) [synchronization](../../../../glossary.md#term-synchronization),
and exact 0007A0C/0007A0D/0007A0E/0007A0F/0007A1 order.

Repository-wide, architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration validation remains deferred
to [CPU](../../../../glossary.md#term-cpu) 0009 or [continuous integration](../../../../glossary.md#term-continuous-integration-ci) because this task changes one [concrete backend](../../../../glossary.md#concrete-backend) and no shared
or [architecture contract](../../../../glossary.md#architecture-contract). Stop and replan if implementation makes that conclusion false.

## Dependencies

- [CPU 0006B2](0006b2-portable-overlap-fold.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> is `Complete` and owns the exact current fold
  semantics, family/type matrix, positive-zero initialization, canonical contributions, padding
  exclusion, arbitrary layout/carrier/range behavior, validation, zero resources, reference
  oracle, worker safety, and schema-17 history preserved here.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [canonical glossary](../../../../glossary.md#term-canonical-form); [padding glossary](../../../../glossary.md#padding); [layout glossary](../../../../glossary.md#layout); [carrier glossary](../../../../glossary.md#term-storage-carrier); [validation glossary](../../../../glossary.md#term-validation); [oracle glossary](../../../../glossary.md#term-test-oracle); [schema glossary](../../../../glossary.md#term-schema))</span>
- [CPU 0007A0](0007a0-generated-hot-path-parity-correction.md) is `Complete` and supplies shared
  typed carrier emission, dense integer proof, stable Class-File testing, and the retained
  five-fork evidence pattern.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- [CPU 0007A0A](0007a0a-affine-and-movement-generated-loop-parity.md) is `Complete` and establishes
  the dense/general correction boundary and schema-23 history.
- [CPU 0007A0B](0007a0b-indexing-generated-loop-parity.md) is `Complete` and establishes embedded
  mixed-carrier indexed mapping without generic bridges and schema-24 history.
- [CPU 0007A0C](0007a0c-scatter-generated-loop-parity.md) is `Complete` and establishes current
  schema 25 plus typed generated contribution/addition bodies while preserving a separate semantic
  and lifecycle contract.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Current Java 26 Class-File API, shared Prepare/Runtime, CPU worker, direct carrier binding, and
  generated-artifact store contracts are complete and unchanged.<span data-glossary-navigation="true"> ([Class-File API glossary](../../../../glossary.md#term-class-file-api); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding); [artifact glossary](../../../../glossary.md#term-artifact))</span>

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0007A0E remains the next ordered `Draft` correction for ordering after this task completes;
  do not create its detailed specification here.
- CPU 0007A0F remains the subsequent `Draft` random/[dropout](../../../../glossary.md#dropout--inverted-dropout) correction without a detailed spec.
- CPU 0007A1 remains `Draft` and explicitly behind CPU 0007A0F before semantic-family expansion
  resumes.
- CPU 0009 retains the generated-coverage and [conformance](../../../../glossary.md#term-conformance) [checkpoint](../../../../glossary.md#term-checkpoint).

## Architecture impact

Expected impact: None.

The task changes only [CPU](../../../../glossary.md#term-cpu)-owned generated fold code [shape](../../../../glossary.md#shape) and current-only [artifact](../../../../glossary.md#term-artifact) compatibility
under the existing analysis/finalization/[binding](../../../../glossary.md#term-binding) [lifecycle](../../../../glossary.md#lifecycle). If implementation requires an
architecture, public/shared contract, [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency), semantic, route, resource,
[materialization](../../../../glossary.md#materialization), or [Runtime](../../../../glossary.md#term-runtime)-policy change, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Implement Synaptik CPU task 0007A0D exactly from
docs/planning/backends/cpu/tasks/0007a0d-fold-generated-loop-parity.md. Do not use GSD. Do not
commit, push, stage, revert, delete, or modify unrelated work.

Read in full AGENTS.md, ARCHITECTURE.md, the directly relevant architecture/performance/Runtime-
Prepare-Backend documents, documentation rules and applicable profiles, planning guide, roadmap,
CPU master plan, task 0007A0D, completed tasks 0006B2/0007A0/0007A0A/0007A0B/0007A0C, and every
affected or directly relevant current CPU source/test/package/documentation file. Inspect the
complete worktree and index and preserve concurrent unrelated changes.

Capture the required current schema-25 source/Class-File and performance baselines before
production edits. Implement the exact typed dense/general fold bodies, schema-26 compatibility,
stable Class-File and semantic tests, focused validation, one authoritative CPU suite, and the
fixed five-fork per-fork/aggregate parity gates. Stop on any architecture, scope, semantic,
shared-contract, route, resource, lifecycle, or unresolved performance conflict.

After executable Java and probe evidence stabilize, hand the same uncommitted diff and exact
evidence to a distinct clean documentation-focused context following
docs/developer-guide/documentation-rules.md. That context must independently inspect final source,
tests, class shape, and evidence; finalize affected Javadocs/package summaries, CPU guide,
glossary impact, task/master/roadmap, rendered pages, Markdown, exact scope, schema, status/order,
and performance evidence; and not repeat successful Java or timing runs unless it changes
executable/probe behavior or records a concrete stale-evidence reason. Do not mark Complete until
both contexts and every acceptance criterion succeed.

Return exact changed paths, commands/results, environment, every per-fork and aggregate ratio,
documentation/Javadoc/glossary impact and reasoned no-change conclusions, context IDs, unresolved
issues, and exactly Status: Complete or Status: Incomplete with
Follow-up required: <specific follow-up>.
```

## Local decisions

- Keep `FOLD_AXIS` and `FOLD2D` together because they share one bridge-only emitter, one canonical
  contribution/addition contract, one zero-resource output-domain execution model, and one schema
  transition. Splitting them would leave one current fold family behind the same defect.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [model glossary](../../../../glossary.md#term-model); [schema glossary](../../../../glossary.md#term-schema))</span>
- Advance exactly to schema 26 because fold class bodies change while schema 25 already identifies
  the completed embedded scatter boundary. A schema-25 fold artifact must never masquerade as an
  embedded fold body.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Use the fixed `[127,16] -> [1024]`, step-8, window-16 axis case because it is dense, overlapping,
  deterministic, large enough for stable repeated work, and admits an exact direct Java version of
  the current canonical output-domain traversal without timing validation or allocation.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [validation glossary](../../../../glossary.md#term-validation))</span>
- Require both every per-[fork](../../../../glossary.md#term-fork-meanings) ratio and the aggregate ratio to pass `<= 1.15x`; aggregate success
  cannot hide one unstable or materially regressed fresh process.
- Keep `FOLD2D`, BFLOAT16, integral, general-layout, segment, and parallel forms in stable semantic
  and Class-File gates rather than multiplying timing cases. The single performance case bounds
  only its named dense FLOAT32 form.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [layout glossary](../../../../glossary.md#layout); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>

## Known limitations

- The timing gate covers one representative dense [heap](../../../../glossary.md#term-heap)-array [FLOAT32](../../../../glossary.md#term-float32) `FOLD_AXIS` case on the
  recorded local environment. It does not claim parity for `FOLD2D`, other types, general layouts,
  segment or mixed carriers, every range, CPU, or Java virtual machine.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [CPU glossary](../../../../glossary.md#term-cpu); [Java virtual machine glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- General and two-dimensional forms receive stable Class-File and semantic validation but no
  numerical performance threshold in this task.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Only the existing one-node fully static resolved-layout CPU fold capability is corrected.
  Multi-node fusion, dynamic layouts, vector fold, native routes, and new semantics remain out of
  scope.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [fusion glossary](../../../../glossary.md#term-fusion); [vector glossary](../../../../glossary.md#term-vector); [scope glossary](../../../../glossary.md#term-scope))</span>
- Output-domain execution remains intentionally quadratic in output count times input count for
  the fixed parity baseline. Algorithm redesign is not hidden inside this generated-shape task.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `01a000ec-f972-74f1-8696-37cc35c383a8` established the exact task and baseline.
Implementation context `01a000f9-8118-73a0-bcc6-2f43e4f271fc` completed the [schema](../../../../glossary.md#term-schema)-26 generated
fold bodies, and audit/fix context `01a00197-8794-7ca0-8849-ba3e1de1a0c0` stabilized direct
segment [layout](../../../../glossary.md#layout) access and the final evidence. Documentation context
`01a001a3-2387-7c21-b04b-de96079c5959` independently reviewed the final source, tests, Javadocs,
[CPU](../../../../glossary.md#term-cpu) guide, glossary, planning records, and retained generated/timing evidence.

The pre-edit schema-25 evidence under
`/private/tmp/synaptik-cpu-0007a0d.4tsSf1/baseline` contains the unchanged probe, environment,
commands, raw [forks](../../../../glossary.md#term-fork-meanings), manifests, class bytes, and complete `javap -c -p`/`javap -v -p` output.
Dense [FLOAT32](../../../../glossary.md#term-float32) and [INT64](../../../../glossary.md#data-type--datatype) classes were 398 bytes; the mixed [BFLOAT16](../../../../glossary.md#term-bfloat16) and padded/dilated [FLOAT64](../../../../glossary.md#term-float64)
FOLD2D classes were 429 bytes. Every class referenced
`CpuFoldEmitter.execute(Object,Object,long[],long,long)`. The fixed FLOAT32 baseline ratios were
`4.959644x`, `5.016241x`, `4.962951x`, `4.945605x`, and `5.001992x`, with aggregate
`4.973988x`.

The stabilized final evidence is under
`/private/tmp/synaptik-cpu-0007a0d.4tsSf1/final-segment-fix`. Dense FLOAT32 has [descriptor](../../../../glossary.md#term-descriptor)
`([F[F[JJJ)V`, 707 bytes, and [SHA-256](../../../../glossary.md#term-sha-256)
`fae76f5667c6bbdb544509977df2267c5753cc27d6dc0a97b98dbce066330c2e`; dense INT64 has
descriptor `([J[J[JJJ)V`, 699 bytes, and SHA-256
`8596bbe6d9ad23f0f082490699c30f532a75a9b6dbcb2e6c23848153dbcd45b3`. Neither has a method
reference. Mixed BFLOAT16 has descriptor `(Ljava/lang/foreign/MemorySegment;[S[JJJ)V`, 1,438
bytes, and SHA-256 `e2c0dc935ec0793a6513ac9232bbaa8ee9bef3ff2c21a75b2f12be1fd45f2484`; its only method
references are `MemorySegment.get` and the two required `Float` bit conversions. Padded/dilated
FLOAT64 FOLD2D has descriptor `(Ljava/lang/foreign/MemorySegment;[D[JJJ)V`, 1,586 bytes, and
SHA-256 `666c54f454310f6dc9a8c9fbc0030989e2ce52a4ed290f396da15954b2d7bf5e`; its only method
reference is `MemorySegment.get`. Complete [decompilation](../../../../glossary.md#term-decompilation) contains no `ByteOrder`, `nativeOrder`,
`withOrder`, `CpuFoldEmitter`, Object descriptor/checkcast, allocation, reflection, map, [Runtime](../../../../glossary.md#term-runtime),
or [cache](../../../../glossary.md#term-cache) reference. Segment access uses predefined native-order `JAVA_*_UNALIGNED` layouts.

The unchanged standalone probe has SHA-256
`d727104d530ffc99b689df21529e5de5222985679e4347a647e5491bec40c857`. On OpenJDK
26.0.1+8-34, macOS 26.5.2, Apple M3 Max [aarch64](../../../../glossary.md#term-aarch64), the stabilized [fork](../../../../glossary.md#term-fork-meanings) ratios were `0.929879x`,
`0.923513x`, `0.925827x`, `0.927054x`, and `0.926274x`; aggregate was `0.926451x`. Every fork
passed semantic verification and the `<= 1.15x` threshold. This establishes parity only for the
fixed dense FLOAT32 FOLD_AXIS case.

The final focused command named all five changed test classes and passed 50 tests. The authoritative
`./gradlew :backends:cpu:test` [run](../../../../glossary.md#term-run) passed 53 suites and 317 tests with zero failures or errors and
one existing expected opt-in [persistence](../../../../glossary.md#term-persistence)-evidence skip. No executable Java changed after those
runs. The documentation pass reused them and did not rerun Java tests or timing. It ran
`./gradlew :backends:cpu:javadoc`, inspected the generated pages for all seven changed production
contracts, and completed Markdown link/anchor/fence, newline/CRLF/trailing-whitespace,
schema/status/order/context/[checksum](../../../../glossary.md#term-checksum)/ratio, exact allowlist/count, later-spec absence, GSD-[artifact](../../../../glossary.md#term-artifact),
concurrent-[scope](../../../../glossary.md#term-scope)-preservation, `git diff --check`, and `git diff --cached --check` validation.
The final CPU task scope contains exactly 17 changed paths drawn from the 18-path allowlist;
`CpuClassFileKernelGeneratorTest` is the sole unchanged allowlisted path. All task-local Markdown
[targets](../../../../glossary.md#term-target) resolve, no changed text adds an unresolved anchor, and the six task fence markers balance.

No architecture document or [architecture decision record](../../../../glossary.md#term-architecture-decision-record-adr) changes: this is a CPU-private generated
realization under the existing analysis/finalization/runtime boundary, with no [ownership](../../../../glossary.md#term-ownership-meanings),
[dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), route, or resource change. Architecture tests therefore need no update.
[Backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) and integration tests need no task-local change because the completed CPU
0006B2 observable semantics and capability surface are unchanged; the focused CPU semantic and
Class-File tests cover the changed realization, and the broader [checkpoint](../../../../glossary.md#term-checkpoint) remains CPU 0009.
[Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), Planning, [Prepare](../../../../glossary.md#prepare), Runtime, Config, [Trace](../../../../glossary.md#trace), [Engine](../../../../glossary.md#term-engine), other backends, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), Training,
public [API](../../../../glossary.md#term-application-programming-interface-api) guides, and [Gradle](../../../../glossary.md#term-gradle)/build configuration are unchanged because no shared contract,
dependency, configuration, public API, or build input moved. All seven changed production
Javadocs/package summaries were reviewed against source and tests. `CpuFoldEmitter` and
`CpuClassFileKernelGenerator` were finalized for represented arithmetic, segment layouts, and the
absence of a generic bridge; the schema/cache and portable-route contracts already accurately
state current-only compatibility, structural [identity](../../../../glossary.md#term-identity-meanings), lifecycle, [parameters](../../../../glossary.md#term-parameter), [nullability](../../../../glossary.md#term-nullability), and
failures. Existing glossary entries required current-schema updates, but no new reusable term was
introduced. Concurrent NN planning and implementation paths were preserved without edits by this
documentation context.

## Implementation notes

- `CpuClassFileKernelGenerator` now passes the structural fold [IR](../../../../glossary.md#term-intermediate-representation-ir) to `CpuFoldEmitter`. Generated
  FOLD_AXIS/FOLD2D entries embed family mapping, type-specific addition, and direct typed carrier
  access instead of calling a generic static bridge.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Dense rank-one heap-array FOLD_AXIS uses a direct integer output/input traversal with a zero-base
  fast form and a nonzero-base form. Other proved dense heap-array forms retain integer geometry;
  arbitrary layouts, segments, mixed carriers, and otherwise unproved forms retain typed long
  addressing.<span data-glossary-navigation="true"> ([rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [heap glossary](../../../../glossary.md#term-heap); [layouts glossary](../../../../glossary.md#layout))</span>
- MemorySegment reads and writes use predefined native-order unaligned layouts directly. BFLOAT16
  keeps its Float bit conversion and immediate per-contribution round-to-nearest-even behavior.<span data-glossary-navigation="true"> ([MemorySegment glossary](../../../../glossary.md#term-memorysegment); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [round-to-nearest glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- Schema 26 invalidates schema-25 artifacts without a migration reader. Concrete geometry remains
  invocation-private packed primitive data loaded into primitive locals before hot loops.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [artifacts glossary](../../../../glossary.md#term-artifact))</span>

## Completion summary

- Completed changes: Embedded typed schema-26 FOLD_AXIS/FOLD2D bodies with dense integer and
  general typed-long forms, direct predefined segment layouts, schema-25 rejection, and durable
  Class-File/semantic coverage.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [layouts glossary](../../../../glossary.md#layout))</span>
- Files changed or created: Seven production sources/package summaries, five tests, the CPU guide,
  glossary, this task, CPU master plan, and roadmap; all are within the exact 18-path allowlist.
- Tests and validation: Focused 5-suite/50-test and authoritative 53-suite/317-test CPU evidence
  passed; all five performance forks and aggregate passed; final Javadoc and documentation/static
  gates passed.<span data-glossary-navigation="true"> ([forks glossary](../../../../glossary.md#term-fork-meanings))</span><span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Documentation-agent review: Context `01a001a3-2387-7c21-b04b-de96079c5959` completed the
  independent API-Javadoc, backend-guide, planning, example/evidence, and glossary review.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: CPU guide and planning records now describe schema-26 generated fold
  bodies and the bounded fixed-case performance result without expanding current capability.
- Javadoc review: All seven changed production contracts were reviewed; focused emitter/generator
  and package summaries were finalized, while cache and portable-route Javadocs already accurately
  documented schema/current-only identity, lifecycle, nullability, parameters, and failures.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [nullability glossary](../../../../glossary.md#term-nullability); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- Glossary impact: Existing generated-kernel, specialization, and CPU overlap-fold entries were
  updated for schema 26; no new reusable term was introduced.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Unresolved issues: None.
- Follow-up required: None. [CPU](../../../../glossary.md#term-cpu) 0007A0E remains the sole next ordered Draft frontier.

Status: Complete
