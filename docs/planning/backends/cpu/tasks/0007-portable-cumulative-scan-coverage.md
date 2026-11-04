# Task 0007: Portable [Cumulative Scan](../../../../glossary.md#cumulative-scan) Coverage

## Status

Complete

## Goal

Add the first bounded [slice](../../../../glossary.md#slice) of the former broad [CPU](../../../../glossary.md#term-cpu) 0007 frontier: execute exactly one fully
static, resolved-[layout](../../../../glossary.md#layout) current [Model](../../../../glossary.md#term-model) `CUM_SUM` or `CUM_PROD` [occurrence](../../../../glossary.md#term-occurrence) through the portable
generated-[kernel](../../../../glossary.md#kernel) route.

This slice completes the existing `CumulativeScanKind` family across its five numeric data types
and all inclusive/exclusive and forward/reverse modes. Each independent [scan](../../../../glossary.md#term-cumulative-scan) slice is traversed
sequentially in logical [axis](../../../../glossary.md#term-axis) order while distinct slices may execute in parallel. This preserves
one deterministic [finite](../../../../glossary.md#term-finite)-[precision](../../../../glossary.md#term-precision) order per output and establishes the range/body seam needed by
later reduction work without mixing aggregate combination, statistics, [softmax](../../../../glossary.md#softmax--log-softmax), or [normalization](../../../../glossary.md#term-normalization)
into the same task.

## [Scope](../../../../glossary.md#term-scope)

### Exact [occurrence](../../../../glossary.md#term-occurrence) and type [matrix](../../../../glossary.md#term-matrix-meanings)

- Admit exactly one [CPU](../../../../glossary.md#term-cpu)-owned [node](../../../../glossary.md#node) whose kind is `CumulativeScanKind.CUM_SUM` or
  `CumulativeScanKind.CUM_PROD`, whose attributes are exact `CumulativeScanAttrs`, and whose
  ordered boundary is one input followed by one output.
- Support [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), [INT32](../../../../glossary.md#data-type--datatype), and [INT64](../../../../glossary.md#data-type--datatype). Reject [BOOL](../../../../glossary.md#data-type--datatype) and every other type.
- Require fully static input/output Shapes, the exact same Shape and data type, resolved layouts
  with non-negative offsets and strides, a valid normalized axis, and an injective output layout.
  The input may use any current supported resolved layout, including offset, positive-strided,
  interleaved, transposed, and zero-strided reads.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [data type glossary](../../../../glossary.md#data-type--datatype); [layouts glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride); [normalized axis glossary](../../../../glossary.md#normalized-axis); [injective glossary](../../../../glossary.md#term-injective))</span>
- Reject scalar input because no valid scan axis exists. Accept zero extents; no output element is
  written and no worker is submitted when the logical output is empty.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [scan glossary](../../../../glossary.md#term-cumulative-scan); [axis glossary](../../../../glossary.md#term-axis); [element glossary](../../../../glossary.md#term-element))</span>
- Keep output and input as distinct complete physical bindings. Reject any complete-span overlap
  before generated execution or worker submission; this task does not introduce in-place scan
  semantics.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding))</span>

### Logical traversal and [finite](../../../../glossary.md#term-finite)-[precision](../../../../glossary.md#term-precision) realization

- A scan slice fixes every logical coordinate except the selected axis. Slice ordinals use the
  row-major order of the non-axis coordinates and are independent of physical layout.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [slice glossary](../../../../glossary.md#slice); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [axis glossary](../../../../glossary.md#term-axis); [row-major glossary](../../../../glossary.md#term-row-major); [layout glossary](../../../../glossary.md#layout))</span>
- Forward mode visits axis coordinates from zero upward; reverse mode visits them from the last
  coordinate downward without changing output coordinates.<span data-glossary-navigation="true"> ([Forward mode glossary](../../../../glossary.md#term-forward-mode); [reverse mode glossary](../../../../glossary.md#term-reverse-mode))</span>
- Inclusive mode updates the accumulator with the current input and then writes it. Exclusive mode
  writes the current accumulator and then updates it. The identities are represented positive zero
  for <span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`CUM_SUM` and represented positive one for `CUM_PROD`.
- INT32 and INT64 use exact-width two's-complement modular addition or multiplication after every
  visited value.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [two's-complement glossary](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks))</span>
- FLOAT32 and FLOAT64 perform one ordinary same-type IEEE 754 addition or multiplication for each
  visited value and retain that rounded result as the next accumulator. BFLOAT16 widens each exact
  represented input and accumulator to FLOAT32, performs one FLOAT32 operation, and rounds once
  back to BFLOAT16 after every visited value. No reassociation, wider retained accumulator,
  reciprocal rewrite, fused operation, vector prefix algorithm, or cross-slice combination is
  permitted.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [IEEE 754 glossary](../../../../glossary.md#term-ieee-754); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [operation glossary](../../../../glossary.md#operation); [reassociation glossary](../../../../glossary.md#term-reassociation); [fused glossary](../../../../glossary.md#specialization-and-fusion); [vector glossary](../../../../glossary.md#term-vector))</span>
- This CPU-private evaluation order must satisfy the Model special-value and identity contract.
  NaN payload choice remains unspecified. Scalar and parallel execution must nevertheless be
  bitwise identical because both execute the same per-slice order.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [evaluation glossary](../../../../glossary.md#term-evaluation); [Model glossary](../../../../glossary.md#term-model); [identity glossary](../../../../glossary.md#term-identity-meanings); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [Scalar glossary](../../../../glossary.md#term-scalar))</span>

### Generated execution and resources

- Add focused [CPU](../../../../glossary.md#term-cpu)-private `CpuScanIr`, `CpuScanLowering`, and `CpuScanEmitter` owners in the
  existing IR, lowering, and Class-File emission packages.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering))</span>
- Lower one occurrence to one computation unit and one generated artifact. The boundary list and
  declared buffers are exactly <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [artifact glossary](../../../../glossary.md#term-artifact); [buffers glossary](../../../../glossary.md#term-buffer))</span>`[input, output]`; there is no virtual result, materialization,
  workspace, partial buffer, carry buffer, or persistent mutable state.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [carry glossary](../../../../glossary.md#term-carry); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Generated primitive `start`/`end` bounds denote a contiguous range of independent scan-slice
  ordinals. The generated entry bridges directly to a CPU-owned static scan body, which
  reconstructs the corresponding non-axis coordinates once per slice and walks the selected axis
  without allocating an object per slice or element.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [scan glossary](../../../../glossary.md#term-cumulative-scan); [slice glossary](../../../../glossary.md#slice); [scan body glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [axis glossary](../../../../glossary.md#term-axis); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [element glossary](../../../../glossary.md#term-element))</span>
- Scalar execution covers all slice ordinals directly. Parallel-scalar execution partitions only
  the independent slice domain into deterministic disjoint ranges. Never split one scan slice
  across workers. Vector and parallel-vector scan bodies are out of scope.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [Vector glossary](../../../../glossary.md#term-vector); [scope glossary](../../../../glossary.md#term-scope))</span>
- When the selected axis is non-empty but the product of the non-axis extents is one, execution is
  scalar. When any non-axis extent or the selected extent is zero, no generated call or worker
  submission occurs.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- Cold lowering computes and validates checked element counts, slice counts, byte spans, axis
  geometry, and address bounds. Cold binding validates carrier/type compatibility, byte sizes,
  alignment, writeability, complete spans, and non-overlap before returning direct typed fields.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [carrier glossary](../../../../glossary.md#term-storage-carrier); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- The [hot path](../../../../glossary.md#term-hot-path) performs no [`Operation`](../../../../glossary.md#operation)/`CompiledNode` access, graph inspection, layout discovery,
  map lookup, boxing, reflection, string dispatch, synchronization, allocation, division/modulo
  per element, route selection, or cache lookup.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [layout glossary](../../../../glossary.md#layout); [lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch); [synchronization glossary](../../../../glossary.md#term-synchronization); [cache glossary](../../../../glossary.md#term-cache))</span>

### Capability, compatibility, and evidence

- Capability reporting is true only for the exact occurrence matrix above and remains false for
  all aggregate reductions, arg extrema, masked reductions, statistical reductions, softmax,
  layer/RMS/batch normalization, and unsupported scan rows.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Advance generated compatibility from [schema](../../../../glossary.md#term-schema) 19 to schema 20 with no migration reader.
- Structural IR, fingerprint, specialization, and compatibility identity include the scan kind,
  data type, normalized axis role, exclusive/reverse modes, rank, boundary roles/count, access-plan
  structure, finite-precision rule, execution compute mode, carrier pattern, and absence of
  workspace. Concrete extents, offsets, stride magnitudes, slots, carrier objects, addresses,
  worker identity, run identity, and selected range count remain cold when they do not alter
  emitted bytes.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [specialization glossary](../../../../glossary.md#term-specialization); [identity glossary](../../../../glossary.md#term-identity-meanings); [data type glossary](../../../../glossary.md#data-type--datatype); [normalized axis glossary](../../../../glossary.md#normalized-axis); [rank glossary](../../../../glossary.md#term-rank); [finite glossary](../../../../glossary.md#term-finite); [precision glossary](../../../../glossary.md#term-precision); [carrier glossary](../../../../glossary.md#term-storage-carrier); [stride glossary](../../../../glossary.md#term-stride); [run glossary](../../../../glossary.md#term-run))</span>
- Extend capability, lowering dispatch, sealed IR permits/encoding, portable route, preparation,
  finalization, generator dispatch/validation, executable binding, scalar reference, schema, and
  focused tests only as required for this family.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [dispatch glossary](../../../../glossary.md#term-dispatch); [sealed glossary](../../../../glossary.md#term-sealed); [validation glossary](../../../../glossary.md#term-validation); [binding glossary](../../../../glossary.md#term-binding); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Test generated results against an independent reference for all five types, both kinds, all four
  modes, arbitrary layouts/carriers, empty and singleton axes, modular overflow, signed zero, NaN,
  infinities, and scalar/parallel parity. Reference tests must not invoke production scan helpers.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [singleton glossary](../../../../glossary.md#term-singleton); [axes glossary](../../../../glossary.md#term-axis); [overflow glossary](../../../../glossary.md#term-overflow); [signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- After executable Java stabilizes, hand the uncommitted diff and exact CPU-test evidence to a
  separate clean documentation-focused context. That pass finalizes affected Javadocs/package
  summaries, the CPU backend guide, glossary impact, this task evidence, CPU master plan, and
  roadmap.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Full, single-axis, multi-axis, masked, target-Shape, arg-extrema, logarithmic, statistical, or
  norm reductions<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [norm glossary](../../../../glossary.md#term-norm))</span>
- softmax, log-softmax, layer normalization, RMS normalization, batch normalization, attention,
  loss, pooling, convolution, or linear algebra<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [layer normalization glossary](../../../../glossary.md#layer-normalization); [RMS normalization glossary](../../../../glossary.md#rms-normalization); [batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training); [loss glossary](../../../../glossary.md#loss); [pooling glossary](../../../../glossary.md#term-pooling); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- more than one node, scan fusion, scan/reduction fusion, general partition-DAG decomposition,
  materialized splits, partial scans, cross-worker prefix combination, vector scan, or native routes<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [scan glossary](../../../../glossary.md#term-cumulative-scan); [fusion glossary](../../../../glossary.md#term-fusion); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [vector glossary](../../../../glossary.md#term-vector))</span>
- in-place or overlapping input/output execution, negative physical strides, dynamic Shapes or
  layouts, runtime-bound axis geometry, or a shared Runtime scan primitive<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [layouts glossary](../../../../glossary.md#layout); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- a backend-neutral or cross-backend floating accumulation order, NaN payload promise, bitwise
  result promise, accuracy policy, or public scan configuration<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Model, Compiler, shared Runtime/Prepare, Config, Trace, backend-contract, architecture,
  architecture-test, Gradle/dependency, backend-conformance, integration, Engine, NN, training,
  tuning, benchmark, persistence-policy, or public API changes<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Trace glossary](../../../../glossary.md#trace); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [persistence glossary](../../../../glossary.md#term-persistence); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- per-element allocation, boxing, reflection, string/map dispatch, synchronization, or a generic
  reduction/scan registry, manager, service, cursor, or interpreter<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [dispatch glossary](../../../../glossary.md#term-dispatch); [synchronization glossary](../../../../glossary.md#term-synchronization); [registry glossary](../../../../glossary.md#term-registry))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU 0005A partition-kernel reset](0005a-atomic-partition-kernel-architecture-reset.md)<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [kernel glossary](../../../../glossary.md#kernel))</span>
- [CPU 0005C vector and parallel strategies](0005c-vector-and-parallel-portable-strategies.md)<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>
- [CPU 0005D evidence gate](0005d-materialization-specialization-and-persistence-evidence-gate.md)
- [CPU 0005E typed portable family expansion](0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md)
- [CPU 0006 static affine views](0006-portable-static-affine-views-and-boundary-materialization.md)<span data-glossary-navigation="true"> ([views glossary](../../../../glossary.md#term-view-meanings))</span>
- [CPU 0006B2 overlap fold](0006b2-portable-overlap-fold.md)
- [CPU 0006C ordering](0006c-portable-stable-ordering-and-selection.md)
- [CPU 0006D explicit-state RNG and dropout](0006d-portable-explicit-state-rng-and-dropout.md)<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- [Model 0023E cumulative scan normalization and product](../../../modules/model/tasks/0023e-cumulative-scan-normalization-and-product.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [cumulative scan glossary](../../../../glossary.md#cumulative-scan); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)

## Architecture constraints

- Model owns the scan kind, axis, inclusion/direction modes, type domain, exact-width integral
  arithmetic, and abstract special-value meaning. CPU owns only its finite-precision realization,
  private IR, generated route, resources, binding, and execution.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [scan glossary](../../../../glossary.md#term-cumulative-scan); [axis glossary](../../../../glossary.md#term-axis); [CPU glossary](../../../../glossary.md#term-cpu); [finite glossary](../../../../glossary.md#term-finite); [precision glossary](../../../../glossary.md#term-precision); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [binding glossary](../../../../glossary.md#term-binding))</span>
- CPU analysis deterministically lowers one supported occurrence, declares exactly two buffers and
  zero workspace before CPU-blind shared assignment, and finalizes one artifact only after slot
  assignments exist.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [buffers glossary](../../../../glossary.md#term-buffer); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Prepared recipes remain [immutable](../../../../glossary.md#term-immutable) and reusable. Concurrent runs use distinct [`RunState`](../../../../glossary.md#run-state--runstate) values;
  the scan executable owns no mutable prepared accumulator or shared scratch.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [accumulator glossary](../../../../glossary.md#accumulator-type))</span>
- Runtime receives only a prepared executable with cold-bound direct carrier references and
  primitive geometry. It performs no scan interpretation, graph work, route selection, or
  resource discovery.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [carrier glossary](../../../../glossary.md#term-storage-carrier); [graph glossary](../../../../glossary.md#graph))</span>
- Work remains inside `backends/cpu`, adds no dependency or package, and preserves the current
  generated portable route and optional artifact lifecycle.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Capability truth may not exceed complete lowering, declaration, assignment, generation,
  binding, overlap validation, execution, and independent reference evidence.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [validation glossary](../../../../glossary.md#term-validation))</span>
- Stop if exact execution requires a Model/compiler/shared Runtime or Prepare change, if BFLOAT16
  per-step rounding conflicts with the current Model contract, or if a slice cannot be kept within
  one worker without changing the prepared execution boundary.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [rounding glossary](../../../../glossary.md#term-rounding); [slice glossary](../../../../glossary.md#slice); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — sole supported [fail-closed](../../../../glossary.md#term-fail-closed) capability provider.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — [CPU](../../../../glossary.md#term-cpu)-private structural [scan](../../../../glossary.md#term-cumulative-scan) [IR](../../../../glossary.md#term-intermediate-representation-ir).
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — one-[node](../../../../glossary.md#node) [validation](../../../../glossary.md#term-validation) and cold geometry.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct Class-File scan emission.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare`, `.route.portable`, `.cache`,
  `.executable`, `.memory`, and `.reference` — existing staged realization, binding, execution,
  and independent reference boundaries.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>

Packages added or changed:

- No package is added, removed, moved, or exported. Existing CPU-internal packages gain only the
  focused scan-family types and direct integration described here.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanIr` — immutable scan semantics,
  boundary roles, numerical order, slice-domain, and compatibility identity.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [slice glossary](../../../../glossary.md#slice); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScanLowering` — exact one-node
  revalidation, declarations, access plans, and compact slice/axis geometry.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuScanEmitter` — allocation-free
  generated per-slice traversal, typed accumulation, and output stores.

Tests mirror these production packages. No generic utility package or new public CPU type is
permitted.

## Affected files

Expected production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuScanIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuScanLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScanEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`

Expected [CPU](../../../../glossary.md#term-cpu) tests:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuScanIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuScanLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScanGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuBufferBindingTest.java`

Expected documentation/[planning](../../../../glossary.md#term-planning) paths during implementation:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No [Model](../../../../glossary.md#term-model)/[compiler](../../../../glossary.md#term-compiler)/shared [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[API](../../../../glossary.md#term-application-programming-interface-api)/architecture/[Gradle](../../../../glossary.md#term-gradle)/[conformance](../../../../glossary.md#term-conformance)/integration path is
expected to change.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 42 paths: 24 production/package paths, 13 [CPU](../../../../glossary.md#term-cpu) test paths,
and 5 documentation/[planning](../../../../glossary.md#term-planning) paths. Stop before a 43rd path, a new package, or any path outside
the listed CPU [module](../../../../glossary.md#term-module) and documentation/planning set. A stale existing CPU inventory test may
replace another listed test path but must not increase the ceiling.

If implementation requires a shared contract, public [API](../../../../glossary.md#term-application-programming-interface-api), another semantic family, another
workspace, partial-[scan](../../../../glossary.md#term-cumulative-scan) combination, or more paths, stop and propose a focused follow-up or
architecture decision. Do not hide incomplete scan acceptance behind a follow-up.

## Acceptance criteria

- Capability and lowering admit exactly the one-node static/resolved five-type CUM_SUM/CUM_PROD
  matrix with all four modes and reject every unsupported row truthfully.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [node glossary](../../../../glossary.md#node); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Boundary roles, declarations, assignments, generated parameters, direct bound fields, prepared
  access declarations, and stores are exactly one read input followed by one write output.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>
- Forward/reverse and inclusive/exclusive traversal, identities, exact-width integral wrap, and
  the specified per-step floating/BFLOAT16 rounding order match this specification.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- Scalar and parallel-scalar execution are bitwise identical for every output because a scan slice
  is never divided between workers.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [scan glossary](../../../../glossary.md#term-cumulative-scan); [slice glossary](../../../../glossary.md#slice))</span>
- Dense, offset, positive-strided, transposed/interleaved, and zero-strided reads; injective
  non-dense writes; heap/native/mixed carriers; empty/singleton axes; modular overflow; finite
  values; signed zeros; NaNs; and infinities agree with an independent reference.<span data-glossary-navigation="true"> ([injective glossary](../../../../glossary.md#term-injective); [heap glossary](../../../../glossary.md#term-heap); [singleton glossary](../../../../glossary.md#term-singleton); [axes glossary](../../../../glossary.md#term-axis); [overflow glossary](../../../../glossary.md#term-overflow); [finite glossary](../../../../glossary.md#term-finite); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Complete input/output overlap fails during cold binding before output mutation or worker
  submission, including spans whose physical intersection lies outside a selected range prefix.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Zero workspace/materialization/scratch, bounded candidates, one realized artifact, immutable
  prepared state, and allocation-free hot loops are enforced by construction and tests.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [artifact glossary](../../../../glossary.md#term-artifact); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Schema 20 rejects schema-19 artifacts. Every semantic/code-shaping scan fact is included in
  deterministic structural/fingerprint/compatibility identity and every listed cold fact remains
  excluded when it does not change emitted bytes.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Existing pointwise, movement, indexing, scatter, fold, ordering, random, worker, cache, and
  persistence behavior remains unchanged.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- A separate documentation-focused context finalizes affected Javadocs, package summaries, CPU
  guide, glossary impact, planning evidence, and explicit API/architecture/conformance no-change
  conclusions before this task becomes Complete.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) focused tests for capability, [scan](../../../../glossary.md#term-cumulative-scan) [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/generated execution,
all data types and modes, arbitrary [layouts](../../../../glossary.md#layout)/carriers, preparation/finalization, [binding](../../../../glossary.md#term-binding)/overlap,
[scalar](../../../../glossary.md#term-scalar)/parallel determinism, [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings), and independent reference comparison. After
executable Java stabilizes, run one final [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :backends:cpu:test
```

The separate clean documentation pass reuses that evidence unless it changes executable Java and
runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates affected Markdown links and anchors, balanced fences, one terminal newline,
trailing whitespace, generated Javadocs, exact changed-path membership and ceiling, package/type
placement, schema 20, scan [matrix](../../../../glossary.md#term-matrix-meanings) and modes, task/master/roadmap status coherence, and absence of
a detailed [CPU](../../../../glossary.md#term-cpu) 0007A task file.

Repository-wide tests, architecture tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), and integration tests are deferred
to CPU 0009 or [continuous integration](../../../../glossary.md#term-continuous-integration-ci). Run them here only if implementation unexpectedly changes
a repository-wide, [dependency](../../../../glossary.md#term-dependency), architecture, or reusable cross-backend contract; such a change is
outside [scope](../../../../glossary.md#term-scope) and normally requires stopping first.

## Dependencies

- CPU 0005A–0006D: Complete, including typed carriers, arbitrary resolved-layout access, generated
  start/end entries, scalar/parallel-scalar workers, cold overlap validation, and schema 19.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [layout glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [validation glossary](../../../../glossary.md#term-validation); [schema glossary](../../../../glossary.md#term-schema))</span>
- [Model](../../../../glossary.md#term-model) 0023E: Complete current `CumulativeScanKind`/`CumulativeScanAttrs` family, five-type domain,
  four modes, identities, integral modular product, and floating special-value semantics.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- Current Compiler capture/inference and shared Prepare/Runtime cold-binding contracts: Complete
  and unchanged; inspected only to confirm the one-input/one-output occurrence and direct prepared
  execution boundary.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-normative))</span><span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [occurrence glossary](../../../../glossary.md#term-occurrence); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>

## Follow-up tasks

- Draft CPU 0007A owns ordinary aggregate reductions, including full, single-axis, multi-axis,
  and binding-aware SUM forms. It has no detailed specification yet.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [binding glossary](../../../../glossary.md#term-binding))</span>
- Draft CPU 0007B owns arg-min and arg-max; Draft CPU 0007C owns two-input masked SUM/MEAN; Draft
  CPU 0007D owns advanced logarithmic, statistical, and norm reductions.<span data-glossary-navigation="true"> ([MEAN glossary](../../../../glossary.md#term-mean); [norm glossary](../../../../glossary.md#term-norm))</span>
- Draft CPU 0007E owns stable SOFTMAX/LOG_SOFTMAX, and Draft CPU 0007F owns layer, RMS, and batch
  normalization after the required aggregate/statistical foundations.<span data-glossary-navigation="true"> ([SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax); [batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training))</span>
- [CPU](../../../../glossary.md#term-cpu) 0009 retains the portable generated-coverage and [conformance](../../../../glossary.md#term-conformance) [checkpoint](../../../../glossary.md#term-checkpoint).

## Architecture impact

Expected impact: None.

This task uses the existing concrete-[backend ownership](../../../../glossary.md#backend-ownership) of [lowering](../../../../glossary.md#lowering), route selection, generated
[artifacts](../../../../glossary.md#term-artifact), storage, and execution. It changes no [module](../../../../glossary.md#term-module) responsibility or [dependency direction](../../../../glossary.md#java-platform-and-build-tools).
If implementation requires architecture, another module, or a shared contract change, stop and
report the exact conflict instead of editing around it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Implement Synaptik CPU task 0007 exactly from its Ready specification. Do not use GSD. Read
AGENTS.md, ARCHITECTURE.md, the current architecture plan, documentation rules and applicable
profiles, planning guide, roadmap, CPU master plan, task 0007, completed CPU tasks
0005A/0005C/0005D/0005E/0006/0006B2/0006C/0006D, Model task 0023E, and every affected or directly
relevant scan/CPU source and test in full before editing.

Deliver one-node static resolved five-type CUM_SUM/CUM_PROD through the generated portable route
with all four modes, exact per-step typed accumulation, BFLOAT16 round-after-each-operation,
logical-slice scalar/parallel determinism, arbitrary supported layouts/carriers, complete
pre-mutation overlap rejection, zero workspace/materialization, independent reference evidence,
truthful capability, and schema 20. Preserve every exclusion and the 42-path ceiling. Stop on any
architecture, shared-contract, Model-semantic, numerical, resource, or scope conflict.

Run focused tests and one final ./gradlew :backends:cpu:test after executable Java stabilizes. Do
not commit or push. Then hand the uncommitted diff and exact Java evidence to a distinct clean
documentation-focused context following docs/developer-guide/documentation-rules.md. That pass
must finalize affected Javadocs/package summaries, CPU guide, glossary impact, task/master/
roadmap, no-change conclusions, Javadoc, Markdown, exact-scope, and whitespace validation without
repeating the successful Java suite unless executable behavior changes. Do not mark Complete until
both passes and every acceptance criterion succeed. Return exact evidence, changed paths,
unresolved issues, context ID if available, and Status: Complete or Status: Incomplete with
required follow-up.
```

## Local decisions

- Select the closed cumulative-scan family first because it is the smallest complete Model family
  in broad CPU 0007, has no dependency on aggregate combination, and can parallelize only across
  independent slices while preserving one exact traversal order.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Use one typed rounded accumulator per slice. BFLOAT16 rounds after each operation; FLOAT32 and
  FLOAT64 retain their own formats; integral arithmetic wraps at its own width.<span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type); [slice glossary](../../../../glossary.md#slice); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [operation glossary](../../../../glossary.md#operation); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
- Use slice ordinals for generated and worker ranges. This prevents cross-worker prefix state and
  makes scalar/parallel results identical without workspace.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Reject input/output overlap rather than introduce an in-place traversal/alias contract that the
  Model and current CPU prepared boundary do not promise.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Keep vector scan, partial-scan combination, and all aggregate/statistical/normalization families
  outside this first slice.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [normalization glossary](../../../../glossary.md#term-normalization))</span>

## Known limitations

- Coverage is one fully static, resolved-[layout](../../../../glossary.md#layout) [occurrence](../../../../glossary.md#term-occurrence) with non-negative physical [strides](../../../../glossary.md#term-stride).
- Floating bit patterns are stable only for this exact CPU per-step realization and execution
  strategy, not a Model, cross-backend, or future-version promise; NaN payloads are unspecified.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend); [NaN glossary](../../../../glossary.md#term-nan))</span>
- Parallelism requires at least two independent non-axis slices. One long scan slice remains
  scalar in this task.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [scan glossary](../../../../glossary.md#term-cumulative-scan); [slice glossary](../../../../glossary.md#slice); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- No [aggregate reduction](../../../../glossary.md#aggregate-reduction), [softmax](../../../../glossary.md#softmax--log-softmax), or [normalization](../../../../glossary.md#term-normalization) capability is advertised by this task.

## [Validation](../../../../glossary.md#term-validation) evidence

- [Planning](../../../../glossary.md#term-planning) context `019ffcbc-6b82-7d43-bf99-cb71ef09e77f` produced this bounded task. Initial
  implementation context `019ffcc6-534a-7cb0-a008-3e9dd9f0e5a3` delivered the scan route, and
  first documentation context <span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>`019ffcd8-55ac-7482-a08d-4638dae6fa4b` performed the initial
  documentation pass.
- Corrective implementation/audit context `019ffcea-9e5e-7d60-9c45-f5664c8c4d4c` aligned
  capability and lowering on one bounded injectivity decision. Layouts with at most 1,000,000
  elements use exact address enumeration; larger layouts use the bounded monotone-span sufficient
  proof. The focused regression passed 1 suite and 3 tests. It proves that Shape <span data-glossary-navigation="true"> ([injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [Layouts glossary](../../../../glossary.md#layout); [elements glossary](../../../../glossary.md#term-element); [regression glossary](../../../../glossary.md#term-regression); [Shape glossary](../../../../glossary.md#shape))</span>`[3, 2]` with
  strides `[2, 3]` is [injective](../../../../glossary.md#term-injective) and accepted by capability and [lowering](../../../../glossary.md#lowering), while [strides](../../../../glossary.md#term-stride) `[1, 1]`
  collide and are rejected.
- After the correction stabilized, the final focused CPU 0007 matrix passed 11 suites and 109
  tests with no skips, failures, or errors. The sole latest authoritative<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
  `./gradlew :backends:cpu:test` passed 50 suites and 292 tests with one existing opt-in skip and
  no failures or errors. The environment was OpenJDK 26.0.1+8-34 with Gradle wrapper 9.6.1.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Mandatory clean documentation context `019ffcf5-ac50-7920-b528-1ea57c175e96` independently
  reviewed and finalized the affected contracts without changing executable Java or tests. It
  used the Backend Guide, API/Javadoc, Planning, Guide, Glossary, and Example profiles with the
  general documentation style and reused the stable Java evidence above.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `./gradlew :backends:cpu:javadoc` passed after final Java/package documentation edits. Javadoc
  reported only the expected two incubating-module warnings for <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>`jdk.incubator.vector`. Rendered
  pages inspected were `CpuScanIr` and `CpuScanIr.Kind`, `CpuScanLowering`,
  `CpuScanLowering.Layout`, `CpuScanLowering.Geometry`, `CpuScanEmitter`,
  `CpuCapabilityProvider`, `CpuPreparedExecutable`, and the CPU root, IR, lowering, emitter,
  cache, portable-route, prepare, executable, and reference package summaries. The rendered
  contracts cover type/mode identity, BFLOAT16 per-step rounding, whole-slice ranges, exact two-
  boundary/no-workspace ownership, complete overlap rejection, and schema 20.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [cache glossary](../../../../glossary.md#term-cache); [prepare glossary](../../../../glossary.md#prepare); [identity glossary](../../../../glossary.md#term-identity-meanings); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [rounding glossary](../../../../glossary.md#term-rounding); [slice glossary](../../../../glossary.md#slice); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [schema glossary](../../../../glossary.md#term-schema))</span>
- A targeted deterministic five-file Markdown checker passed local file targets and generated
  heading anchors, balanced backtick/tilde fences, one terminal newline, and trailing whitespace
  for the CPU guide, glossary, task, CPU master plan, and roadmap. Manual review recalculated the<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
  `[1, 2, 3]` scan example, confirmed terminology against the existing cumulative-scan glossary
  entry, and found no conceptual code presented as runnable API.
- The final CPU-attributable delta is 39 listed paths: 23 production/package paths, 11 CPU test
  paths, and the five permitted documentation/planning paths. The shared glossary and roadmap
  retain concurrent NN hunks, and unrelated NN planning/source/test paths are excluded from the
  CPU count. No CPU 0007A task file exists. The historical schema/status check confirmed CPU 0007 Complete,
  CPU 0007A Draft without a detailed specification, and generated schema 20. <span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>`git diff --check`
  passed. No commit or push was performed.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit))</span>

## Implementation notes

- Added focused `CpuScanIr`, `CpuScanLowering`, and `CpuScanEmitter` owners. Lowering revalidates
  exactly one current CUM_SUM/CUM_PROD occurrence, derives checked complete layouts plus an
  independent non-axis slice count, declares <span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Lowering glossary](../../../../glossary.md#lowering); [occurrence glossary](../../../../glossary.md#term-occurrence); [layouts glossary](../../../../glossary.md#layout); [axis glossary](../../../../glossary.md#term-axis); [slice glossary](../../../../glossary.md#slice))</span>`[input, output]`, and retains zero workspace and
  materialization.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- The generated two-boundary entry bridges to a CPU-owned static scalar body. That body
  reconstructs non-axis coordinates once per slice and performs one forward or reverse
  sequential axis traversal. Inclusive mode updates then writes; exclusive
  mode writes then updates. INT32/INT64 wrap naturally, FLOAT32/FLOAT64 retain their own format,
  and BFLOAT16 rounds back after every operation.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [scalar glossary](../../../../glossary.md#term-scalar); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [operation glossary](../../../../glossary.md#operation))</span>
- Preparation may choose scalar or parallel-scalar orchestration from the slice count. Each
  generated range owns complete slices only. Empty axes or any empty non-axis domain submit no
  generated call or worker work, and a single slice remains scalar.
- Cold binding validates complete carriers and complete input/output physical spans before any
  generated call or worker submission. The prepared recipe retains direct typed fields and
  immutable cold <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [immutable glossary](../../../../glossary.md#term-immutable))</span>`CpuScanLowering.Geometry`. For each invocation range, `Geometry.pack` creates a
  fresh mutable <span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>`long[]` that owns that range's coordinate state. The CPU-owned scan body inspects
  no Model or graph objects and performs no allocation, route/cache lookup, reflection, boxing,
  synchronization, or per-element division/modulo; generated bytecode is only its direct bridge.<span data-glossary-navigation="true"> ([scan body glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [Model glossary](../../../../glossary.md#term-model); [graph glossary](../../../../glossary.md#graph); [cache glossary](../../../../glossary.md#term-cache); [lookup glossary](../../../../glossary.md#term-lookup); [synchronization glossary](../../../../glossary.md#term-synchronization); [element glossary](../../../../glossary.md#term-element); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Added an independent scalar reference implementation and differential evidence across all five
  types, both kinds, and all four modes, plus generated and prepared-execution coverage for
  overflow, signed zero, infinities, NaN, BFLOAT16 per-step rounding, slice parallelism, and
  pre-mutation overlap rejection. Generated compatibility is schema 20.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [signed zero glossary](../../../../glossary.md#term-signed-zero); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [rounding glossary](../../../../glossary.md#term-rounding); [schema glossary](../../../../glossary.md#term-schema))</span>

## Completion summary

- Completed changes: Implemented and documented one static resolved-layout five-type CUM_SUM/
  CUM_PROD occurrence with all four modes, sequential typed accumulation, scalar or whole-slice
  parallel-scalar execution, complete pre-mutation overlap rejection, zero workspace/
  materialization, and schema 20.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [occurrence glossary](../../../../glossary.md#term-occurrence); [scalar glossary](../../../../glossary.md#term-scalar); [slice glossary](../../../../glossary.md#slice); [materialization glossary](../../../../glossary.md#materialization); [schema glossary](../../../../glossary.md#term-schema))</span>
- Files changed or created: exactly 23 of the production/package paths listed under Affected
  files (all except unchanged `CpuKernelSpecialization.java`); exactly 11 of the listed CPU tests
  (all except unchanged `CpuClassFileKernelGeneratorTest.java` and `CpuBufferBindingTest.java`);
  and exactly `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task,
  `docs/planning/backends/cpu/master-plan.md`, and [CPU](../../../../glossary.md#term-cpu)-specific roadmap hunks in
  `docs/planning/roadmap.md`. No other path is attributable to CPU 0007.
- Tests and [validation](../../../../glossary.md#term-validation): Reused the [scope](../../../../glossary.md#term-scope)-stable implementation evidence from context
  `019ffcea-9e5e-7d60-9c45-f5664c8c4d4c` (corrective 1 suite/3 tests, focused 11 suites/109 tests,
  and final CPU 50 suites/292 tests, one existing skip, no failures/errors). Final CPU Javadoc,
  rendered-page inspection,
  Markdown/link/anchor/fence/example/terminology/newline/whitespace checks, exact 39-path inventory,
  status/schema/no-0007A-spec checks, and <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check` passed.
- Documentation-agent review: Mandatory separate clean documentation context
  `019ffcf5-ac50-7920-b528-1ea57c175e96` completed the final review. It changed no executable
  behavior or tests.
- Documentation impact: Finalized the CPU guide's ninth-family scan contract, current/planned
  boundaries, and schema-20 summary; synchronized this task, CPU master plan, and CPU roadmap.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Javadoc review: Finalized the three new scan contracts, affected integration-type summaries,
  constructor/component/failure documentation, and nine materially affected package summaries.
  Existing unaffected member contracts remained accurate and required no mechanical restatement.
- Glossary impact: Updated the existing cumulative-scan entry and implementation-status paragraph
  for the bounded CPU route; no new project term was introduced. Concurrent NN glossary hunks were
  preserved.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Unresolved issues: None for CPU 0007.
- Follow-up required: None for CPU 0007. CPU 0007A remains Draft without a detailed specification.

Status: Complete
