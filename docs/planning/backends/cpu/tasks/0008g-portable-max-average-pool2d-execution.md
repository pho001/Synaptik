# CPU Task 0008G: Portable Max/Average Pool2d Execution

## Status

Complete.

## Goal

Implement the complete current static Model `MAX_POOL2D` and `AVERAGE_POOL2D` families on the CPU
portable generated-bytecode route. Preserve the exact NCHW literal floor/ceiling grid, padding,
dilation, extrema, fixed-divisor, accumulator, special-value, empty-domain, layout, carrier,
alias, and publication contracts already established by Model tasks 0020A and 0020A1 and adopted
by Compiler task 0005D.

The implementation must provide one generated scalar production form for every admitted case,
reuse the existing caller-owned CPU workers only across disjoint complete output cells, and prove
the generated body against an equally specialized optimal clean Java oracle. It must not add
pooling epilogue fusion, a native route, a public/shared abstraction, hidden materialization, or a
Runtime operation interpreter.

## Scope

### Current source-backed baseline

- `Pool2dKind` contains exactly `MAX_POOL2D` and `AVERAGE_POOL2D`. Each kind accepts only its own
  nine-component attribute record, exactly one input, and exactly one output.
- `MaxPool2dAttrs` and `AveragePool2dAttrs` contain positive kernel, stride, and dilation
  components; non-negative symmetric padding per side; and one literal `ceilMode` flag. There is
  no asymmetric padding, automatic padding, count-padding flag, divisor override, padding value,
  or runtime geometry input.
- Model accepts BFLOAT16, FLOAT32, and FLOAT64 rank-four NCHW input `[N,C,H,W]`. Output preserves
  the exact input type, gradient eligibility, batch extent, and channel extent and has Shape
  `[N,C,Hout,Wout]`.
- Model construction may retain symbolic spatial extents. CPU execution remains narrower: this
  task admits only fully static Shapes and resolved non-negative layouts whose exact geometry is
  independently revalidated during CPU capability analysis and lowering.
- Compiler structured inference revalidates the one-input descriptor, attributes, rank, type,
  exact output Shape, and gradient eligibility. Compiler 0005D owns the already completed
  pooling-gradient formulas; CPU executes the resulting ordinary forward operations and does not
  implement autograd.
- CPU capability currently rejects both pooling kinds. The shared `PartitionDag` already carries
  authoritative partition-local topology. CPU 0008B treats numerical semantic families as atomic
  units and materializes legal splits around them. CPU 0008C explicitly excludes pooling
  recognition and pooling epilogues. CPU 0008D ranks only existing recognized topology choices,
  and CPU 0008E retains pointwise materializations as candidate-only while ordinary preparation
  remains direct.
- Generator compatibility is currently schema 54. Unchanged families retain schema-52 class
  identity, and MATMUL retains schema-54 class identity. A new Pool2d emitted form must not reuse
  either identity projection.

### Exact admitted operations and geometry

- Admit only `Pool2dKind.MAX_POOL2D` paired with `MaxPool2dAttrs` and
  `Pool2dKind.AVERAGE_POOL2D` paired with `AveragePool2dAttrs`. Reject a kind/attributes mismatch,
  input/output count mismatch, or any other pooling kind or attribute representation.
- Revalidate exact input/output descriptors: one BFLOAT16, FLOAT32, or FLOAT64 rank-four input;
  one same-type, same-gradient-eligibility rank-four output; fully static extents; resolved layouts;
  non-negative storage offsets and strides; and an injective output layout.
- For each spatial axis, with input extent `D`, kernel-position count `k`, symmetric padding per
  side `p`, dilation `d`, and stride `s`, compute with checked `long` arithmetic:

  ```text
  effectiveKernel = d * (k - 1) + 1
  numerator       = D + 2 * p - effectiveKernel
  floor output    = floor(numerator / s) + 1
  ceil output     = ceil(numerator / s) + 1
  ```

- Reject a negative numerator, arithmetic overflow, or any disagreement between the recomputed
  `[N,C,Hout,Wout]` and the supplied result Shape. Ceil mode is the literal symmetric padded
  grid: do not remove a terminal window whose start lies entirely in trailing padding.
- Output coordinate `(n,c,oh,ow)` has input-window start
  `(oh * strideHeight - paddingHeight, ow * strideWidth - paddingWidth)`. Kernel coordinates are
  visited in increasing height and then increasing width order, and sample positions add the
  corresponding dilation. Use checked arithmetic for every derived extent, product, flattened
  count, byte span, address, range endpoint, and geometry array component.
- The CPU may reject a Model-valid occurrence whose exact kernel-position product, flattened
  output-cell count, address span, or invocation geometry cannot be represented safely in its
  checked `long`/carrier boundary. Such rejection is fail-closed capability truth, not a change
  to Model semantics.

### Maximum-pooling numerical contract

- Padding positions do not participate in maximum selection. An output window with no in-bounds
  sampled coordinate stores exact negative infinity in the input/result type.
- For a window containing at least one in-bounds sample, any NaN wins over every non-NaN and the
  first logical NaN wins among multiple NaNs. Otherwise ordinary numerical maximum applies,
  positive zero wins over negative zero, infinities use ordinary order, and equal candidates keep
  the first increasing kernel-height/kernel-width sample.
- Preserve the selected non-NaN represented value exactly. BFLOAT16 does not promote to an
  accumulator for max; compare its decoded represented value and retain the selected BFLOAT16
  bits for non-NaN values. NaN payload/sign and signaling preservation remain unspecified by
  Model, so tests must assert the selected NaN class and first-winner behavior without inventing a
  stronger portable payload guarantee.
- The generated and direct-oracle implementations use the same one-pass first-winner state over
  increasing logical kernel coordinates. No indices output, hidden winner buffer, unpooling
  state, or workspace is created.

### Average-pooling numerical contract

- The divisor is always the positive mathematical product `kernelHeight * kernelWidth`. Dilation
  changes coordinates, not the number of divisor positions. Every in-bounds coordinate
  contributes its input value; every out-of-bounds coordinate contributes conceptual exact
  positive zero and still counts once in the divisor.
- BFLOAT16 and FLOAT32 accumulate and divide in FLOAT32. FLOAT64 accumulates and divides in
  FLOAT64. BFLOAT16 narrows once after the final division. Perform one final division by the exact
  positive divisor; there is no valid-sample count, divisor override, integer accumulation,
  mixed-type promotion, or zero-divisor case.
- The generated scalar form and the optimal clean Java oracle use the same increasing
  kernel-height/kernel-width accumulation order. This deterministic CPU implementation choice is
  within Model's permitted reassociation boundary and does not claim bitwise equality with other
  backends.
- Preserve the Model special-value classes: any in-bounds NaN produces NaN; opposing infinities
  produce NaN; otherwise a present infinity keeps its sign. Conceptual padding introduces no NaN
  or infinity. An exact-zero finite result is negative zero only when every divisor contribution
  is an in-bounds negative zero; cancellation, any positive zero, or any conceptual padding
  produces positive zero. An all-padding window stores exact positive zero.
- Implement the signed-zero rule explicitly rather than assuming an incidental Java addition
  order will distinguish cancellation, conceptual padding, and an all-negative-zero domain.
  Preserve only the special-value and rounding guarantees selected by Model; do not promise NaN
  payload/sign preservation or cross-backend finite bit identity.

### Static layouts, carriers, zero extents, and aliasing

- Support arbitrary resolved non-negative input and output strides that pass checked address-span
  validation. Dense layout is not a semantic prerequisite. A read-only input may be non-injective;
  the output must be injective because every logical output cell is written exactly once.
- Support the exact same-type carrier combinations for each kind: BFLOAT16 `short[]` or
  `MemorySegment`, FLOAT32 `float[]` or `MemorySegment`, and FLOAT64 `double[]` or
  `MemorySegment`, including all-array, all-segment, and mixed input/output forms. Carrier objects,
  base addresses, storage offsets, extents, strides, and run identities remain cold invocation
  facts outside generated class identity.
- Reject every physical output/input overlap before output mutation or worker submission. This
  task does not select an in-place exception, even for a one-by-one kernel, because one uniform
  disjoint-output contract is the bounded safe fallback. There is only one input and one output,
  so no input/input or output/output pair policy is needed.
- Static zero batch or channel extents produce zero output cells and submit no worker work. Static
  zero height or width remains admissible only when the checked padding/effective-kernel formula
  yields valid positive spatial output extents; those windows use the all-padding result policy.
  Valid geometry never creates a zero spatial output extent under the current Model formulas.
- Empty output performs no generated invocation, allocation, write, or validity mutation inside
  the CPU child. The surrounding existing Runtime contract continues to own partition-level
  validity transitions.

### CPU-private lowering, generation, and execution

- Add one focused immutable CPU-private `CpuPool2dIr` carrying only code-shaping family, data type,
  scalar numerical form, and exact input/output access regimes. Add one `CpuPool2dLowering` that
  owns checked semantic geometry and range facts. Do not add a public type, shared lowering
  abstraction, pooling registry, generic window interpreter, or common `PoolNd` framework.
- Use one generated realization, `DIRECT_SCALAR`, for both kinds and all admitted cases. The same
  generated class body is reused by scalar and parallel-scalar orchestration for compatible
  code-shaping facts. No Vector API, tiling, unrolling, fixed-Shape class, window materialization,
  `UNFOLD2D` decomposition, or interpreted scalar Runtime fallback is selected in this task.
- The generated portable body is the production semantic baseline and safe fallback. The separate
  `CpuPool2dReferenceKernel` is a conformance and performance oracle only; Runtime must never call
  it to interpret an operation or IR.
- Thread the family through the existing CPU-private unit lowering, portable specialization,
  preparation plan, finalizer, generated Class-File dispatcher, executable geometry binding, and
  artifact cache. Use the established primitive geometry-array plus typed-carriers plus
  `start`/`end` entry boundary; add no new Runtime or shared Prepare contract.
- Advance `CpuGeneratorSchema.CURRENT_VERSION` from 54 to 55. Pool2d class identity uses a new
  schema-55 projection containing every code-shaping pooling fact. Unchanged pre-0008F families
  retain their schema-52 class identity and byte-identical class bytes; MATMUL retains its
  schema-54 class identity and byte-identical class bytes. All compatibility envelopes advance to
  current-only schema 55, so schema-54 envelopes become safe misses.
- Generated classes remain final and field-free with one exact static entry. Their hot body may
  contain primitive index arithmetic, direct array/segment access, type-specific comparisons or
  accumulation, one average division, and a direct store only. It must contain no allocation,
  boxing, reflection, `Map` lookup, graph/value traversal, operation switch, route selection,
  cache access, layout construction, helper dispatch, or worker management.

### Partition DAG, representation, publication, and ranges

- The shared immutable `PartitionDag` remains the sole structural source. Pooling becomes one
  atomic executable seed in CPU-owned decomposition. CPU does not reconstruct complete partition
  adjacency or retain another graph.
- CPU 0008B's baseline decomposition and strict unit order remain authoritative. A pooling unit is
  a numerical-family barrier. Surrounding pointwise or affine work remains a separate valid unit
  with an ordinary declared boundary unless an already implemented rule independently proves a
  different topology.
- Pooling epilogue fusion is explicitly out of scope. CPU 0008C's closed recognizer does not
  include pooling, and its task explicitly deferred pooling execution to 0008G. Do not infer an
  ADD, activation, CLAMP, affine view, decomposed reduction, or another suffix around Pool2d.
  CPU 0008D therefore receives no new pooling fused/split alternative or profitability fact.
- Select direct input/output representation as the only ordinary Pool2d plan. Do not broaden CPU
  0008E's pointwise materialization grammar, promote a materialized input, or introduce a hidden
  contiguous copy. Existing materialized pointwise variants remain complete candidate-only facts
  and ordinary preparation remains direct. A future Pool2d materialization candidate requires a
  separately planned complete candidate, resource, and end-to-end evidence boundary.
- Declare exactly the input read and output write buffer requirements before shared slot
  assignment. Declare no workspace, partial, index, count, winner, column, combine, padding, or
  scratch resource. Finalization may only generate/load the already selected Pool2d artifact and
  bind the already assigned slots.
- Flatten the logical output-cell domain in stable NCHW order. Each half-open generated range owns
  complete output cells and every cell's complete kernel window. Reuse existing caller-owned
  `CpuWorkerGroup`, configured range-count, and minimum-work policy; do not add a Pool2d threshold
  constant. Parallel execution partitions only output cells, never kernel coordinates, and has no
  partial reduction, combine phase, atomics, locks, nested workers, or worker-count-dependent
  arithmetic.
- Runtime receives only immutable prepared geometry, typed buffer/workspace selections, and the
  generated handle. It performs no graph interpretation, allocation policy, geometry derivation,
  alias decision, route search, materialization decision, or pooling dispatch.

### Performance and generated-code evidence

- Every generated pooling specialization must follow the semantic algorithm, hot-loop/dataflow
  shape, carrier/layout case, and avoidable-overhead profile of an optimal clean Java
  implementation at the same Shape-polymorphic specialization boundary. The oracle receives the
  same primitive geometry, typed carriers, and half-open output-cell range.
- Semantic validation precedes timing. Retain a bounded evidence ledger that covers both kinds,
  all three types, dense and general positive-stride layouts, padding, dilation, floor and literal
  ceil grids, heap arrays, segments, mixed carriers, all-padding windows, and scalar/parallel
  range subdivision. Special-value and zero-domain cases are semantic evidence; do not time tiny
  exceptional cases as performance representatives.
- Time at least these six hot representatives in five fresh isolated Java 26 forks: dense
  FLOAT64 max 3-by-3; padded/dilated FLOAT32 max with a ceil tail; mixed-carrier BFLOAT16 max;
  dense FLOAT64 average 3-by-3; padded/dilated FLOAT32 average with a ceil tail; and mixed-carrier
  BFLOAT16 average. If a general-layout code shape is distinct from those rows, add one
  representative for each affected kind without changing production selection policy.
- Use the established CPU evidence protocol: fixed heap, deterministic inputs, randomized
  generated/direct order, at least five warmup batches, nine measured rounds, adaptive batches of
  at least 25 ms, exact pre/post semantic verification, no retry or discarded measured miss, and
  generated/direct ratio `<= 1.15x` for every accepted fork and each row's aggregate median. This
  existing repository gate is reused; no new performance threshold or production constant is
  introduced.
- Retain generated bytes and digest, specialization/lowering manifests, entry descriptor,
  Class-File parse, complete `javap -c -v -p`, constant-pool/member report, forbidden-reference
  scan, semantic checksum, environment, direct-oracle identity/source, fork reports, aggregate,
  and a checksummed evidence manifest outside the repository.
- Retain byte-identical generated controls for representative schema-52 pointwise, affine-copy,
  reduction, Conv2d, and Conv3d forms and one schema-54 MATMUL form. Compatibility changes to 55;
  their class projections, binary names, emitted bodies, and class bytes do not.

### Documentation and Javadoc

- After executable Java and generated evidence stabilize, use a separate clean
  documentation-focused agent/thread. It receives this task, the actual diff, exact test and
  evidence results, affected APIs/behavior, architecture constraints, expected documentation,
  and validation commands.
- Finalize meaningful Javadoc for every changed CPU type and member, including NCHW coordinates,
  both numerical policies, geometry, carrier/layout/alias rules, range ownership, resource and
  lifecycle boundaries, nullability, results, and expected failures.
- Update `docs/backend-guide/cpu-backend.md` with the implemented direct portable Pool2d boundary,
  max/average semantic distinction, static/layout/carrier limits, range/resource model, safe
  fallback, and no-fusion/no-materialization boundary. Review `docs/glossary.md`; change it only
  if implementation changes a reusable term, otherwise record why its existing pooling entry is
  accurate.
- Synchronize this task, the CPU master plan, and roadmap only after all executable,
  performance, documentation, status, and scope gates pass. The documentation context reuses
  successful Java tests unless it changes executable Java or records a concrete stale-evidence
  risk.

## Out of scope

- Native/OpenBLAS, Accelerate, oneMKL, oneDNN, AOCL, ZenDNN, JNI, or Foreign Function and Memory
  downcall pooling routes.
- Backward/autograd implementation, saved max indices, hidden outputs, unpooling, gradient
  formulas, training orchestration, or changes to Compiler 0005D.
- Dynamic or binding-dependent Shapes, unresolved layouts, negative storage strides, runtime
  geometry, sparse, quantized, complex, FLOAT16, integral, BOOL, or mixed-type pooling.
- Global, adaptive, one-dimensional, three-dimensional, channels-last, asymmetric, automatic, or
  transposed pooling; configurable count-padding, valid-sample average, divisor override, or
  padding value.
- Pooling epilogue recognition/fusion, decomposed pooling recognition, graph rewriting,
  `UNFOLD2D` materialization, column buffers, winner/index buffers, input materialization,
  vectorization, tiling, unrolling, or fixed-Shape specialization.
- New public/shared APIs or abstractions; changes to Model, Compiler, Planning, Prepare, Runtime,
  Config, Trace, Backend Contract, OpenBLAS provider, Engine, architecture, ADRs, dependencies,
  Gradle, architecture tests, backend conformance, or integration tests.
- Global autotuning policy, tuning-cache lookup/mutation, measurement-driven preparation,
  Runtime profiling selection, or any CPU 0010-plus native/tuning work.
- Attention, loss, or any CPU 0008H-or-later task specification or implementation.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [CPU backend master plan](../master-plan.md)
- [CPU 0008B partition-DAG decomposition](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md)
- [CPU 0008C typed recognition](0008c-typed-specialized-subgraph-and-epilogue-recognition.md)
- [CPU 0008D fusion profitability](0008d-bounded-fusion-profitability-and-typed-decision-facts.md)
- [CPU 0008E representation candidates](0008e-bounded-multi-input-materialization-and-representation-reuse.md)
- [CPU 0008E1 shared DAG adoption](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md)
- [CPU 0008F portable MATMUL](0008f-portable-matmul-execution-and-bounded-linear-epilogues.md)
- [Model 0020A max Pool2d](../../../modules/model/tasks/0020a-nchw-max-pool2d-semantics-and-tensor-expression.md)
- [Model 0020A1 average Pool2d](../../../modules/model/tasks/0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md)
- [Compiler 0005D pooling gradients](../../../modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)

## Architecture constraints

- `ARCHITECTURE.md` remains authoritative. Planning selects CPU ownership; CPU analysis owns
  exact lowering, route/representation selection, and complete resource declaration; shared
  Prepare assigns slots; CPU finalization constructs the executable; Runtime invokes prepared
  work only.
- The shared `PartitionDag` is the sole complete partition-structure source. CPU-private Pool2d
  geometry and IR retain only exact unit-local semantic and code-shaping facts.
- Generated JVM bytecode is the portable production route and truthful fallback. Reference Java
  is an oracle and never a Runtime interpreter.
- All semantic validation, geometry, layout, carrier, address-span, injectivity, alias, range,
  resource, schema, and route decisions are cold. Generated hot code receives primitive geometry
  and direct typed carriers only.
- Each generated specialization preserves the optimal clean Java algorithm and avoidable-overhead
  profile at the same specialization boundary. A general fallback does not excuse avoidable
  overhead in a proved case.
- CPU 0008B–0008E1 ownership remains unchanged: shared DAG structure; CPU-owned unit formation,
  recognition, profitability, and representation candidates; candidate-only materialization;
  strict inter-unit execution; and no Runtime topology or policy interpretation.
- Capability advertisement becomes true only after the complete admitted matrix is implemented
  and validated. Any unsupported, dynamic, unresolved, unsafe, overflowed, or mismatched
  occurrence remains fail-closed.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — sole public capability provider.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — new immutable Pool2d code-shaping facts.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — checked Pool2d semantic geometry and
  direct scalar selection.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct generated Pool2d body.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — optimal clean Java oracle.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema and specialization identity.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — resource declaration, assignment
  handoff, and finalization facts.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — cold typed geometry binding and
  prepared range invocation.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — generated portable plan.

Packages added or changed:

- No package is added. Existing CPU-private packages gain focused Pool2d leaves only.

Type placement:

- `...internal.ir.CpuPool2dIr` — immutable Pool2d family/type/access class identity.
- `...internal.lowering.CpuPool2dLowering` — descriptor validation, exact NCHW geometry, range,
  and direct scalar plan owner.
- `...internal.codegen.emit.CpuPool2dEmitter` — generated direct scalar max/average body.
- `...internal.reference.CpuPool2dReferenceKernel` — same-boundary semantic and performance
  oracle, never production dispatch.
- Existing preparation, cache, generator, executable, and capability owners carry only the
  minimal new family branch/facts required by those responsibilities.

## Affected files

Expected CPU production owners:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPool2dEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java`
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPool2dIr.java`
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPool2dLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionDagDecomposer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuPool2dReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`

Expected focused test owners:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuBatchNormTrainingEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv2dEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv3dEvidenceTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPool2dGeneratedKernelTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPool2dPerformanceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPartitionDagGeneratedEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseLedgerEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPool2dIrTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPool2dLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuPool2dReferenceTest.java`

If implementation proves a named existing suite is not the current owner, substitute one matching
CPU test path and record the source-backed reason without increasing the maximum scope.

The five existing evidence owners added above are mechanical schema invalidation controls only:
their stale assertions must advance from current schema 54 to current schema 55 after Pool2d
changes the compatibility envelope. They do not widen Pool2d behavioral scope, add evidence rows,
or authorize changes to their underlying completed families. The focused
`CpuClassFileKernelGeneratorTest`, `CpuShapePolymorphicArtifactTest`, and
`CpuPartitionDagDecomposerTest` suites remain required validation owners below but are not
expected changed paths.

The targeted documentation pass may update affected CPU package/type Javadocs,
`docs/backend-guide/cpu-backend.md`, `docs/glossary.md` only if a reusable term changes, this task,
the CPU master plan, and the roadmap. No other documentation path is authorized.

## Maximum scope

The implementation may create or modify at most 38 repository paths:

- exactly 16 CPU production paths, including the sealed `CpuPortableKernelIr` owner required for
  `CpuPool2dIr` to join its hierarchy;
- exactly 17 CPU test paths, comprising the 12 Pool2d implementation/evidence paths and the five
  mechanical schema-55 invalidation controls named above; and
- exactly five possible documentation/planning paths: CPU backend guide, conditional glossary,
  this task, CPU master plan, and roadmap.

The ceiling is justified by one cohesive generated family crossing the existing capability, IR,
lowering, preparation, schema, generation, execution, oracle, and evidence seams. It is a ceiling,
not a target. No Model, Compiler, Planning, Prepare, Runtime, Config, Trace, Backend Contract,
OpenBLAS provider, Engine, architecture, ADR, architecture-test, backend-conformance,
integration-test, resource, generated-source, or build path may change. If a production path
beyond the explicitly inventoried 16, an eighteenth test path, a sixth documentation path, or any
shared/public change is necessary, stop and revise the plan before editing.

## Acceptance criteria

- CPU capability admits all and only fully static, resolved-layout, supported-carrier exact
  `MAX_POOL2D`/`MaxPool2dAttrs` and `AVERAGE_POOL2D`/`AveragePool2dAttrs` occurrences across
  BFLOAT16, FLOAT32, and FLOAT64 with exact one-input/one-output descriptor relationships.
- Checked cold lowering reproduces the literal NCHW floor/ceil Shape, exact starts and dilated
  coordinates, and rejects kind/attrs, rank, type, Shape, gradient, layout, span, injectivity,
  overflow, and unsupported-carrier mismatches before artifact access or writes.
- Generated max pooling preserves excluded padding, all-padding negative infinity, first logical
  NaN/equal winner, positive-over-negative zero, infinity order, and original selected BFLOAT16
  bits across every admitted layout/carrier/range form.
- Generated average pooling preserves fixed count-padding, FLOAT32 accumulation/division for
  BFLOAT16/FLOAT32, FLOAT64 accumulation/division for FLOAT64, one final division and BFLOAT16
  narrowing, NaN/opposing/single-infinity classes, exact signed-zero rule, and all-padding
  positive zero.
- Zero N/C domains perform zero work; zero spatial input with valid padded geometry produces exact
  all-padding outputs. Output is injective and completely disjoint from input before any write or
  worker submission. All-array, all-segment, and mixed carriers pass.
- Exactly one generated `DIRECT_SCALAR` form is the production fallback for every admitted case.
  Scalar and parallel-scalar execution share its class and are equal across range subdivisions and
  worker counts. Each range owns complete disjoint output cells and complete windows; there is no
  partial/combine state, workspace, allocation, atomics, locks, or nested parallelism.
- Pooling is one atomic shared-DAG-derived CPU unit. No pooling epilogue fact, fusion candidate,
  materialization candidate/promotion, `UNFOLD2D` decomposition, graph reconstruction, or Runtime
  interpretation is introduced. Surrounding work retains the exact valid CPU 0008B split.
- Preparation declares only the input and output buffers before assignment. Finalization adds no
  resources or policy and realizes only the selected artifact. Runtime receives cold-bound typed
  carriers, primitive geometry, and ranges.
- Schema advances exactly to 55. Stale schema-54 envelopes fail safely; Pool2d identity includes
  every code-shaping fact; unchanged schema-52 families and schema-54 MATMUL retain exact binary
  names and byte-identical class bytes.
- Raw Class-File parsing and complete `javap -c -v -p` inspection prove one final field-free class
  and static entry per selected artifact, expected direct primitive operations, and no forbidden
  helper, allocation, boxing, reflection, map, cache, graph, route, layout-construction, or worker
  reference.
- Generated and optimal clean Java implementations use the same Shape-polymorphic specialized
  algorithm, accumulation/selection order, carrier/layout case, geometry, and range boundary.
  Every required performance fork and aggregate passes the established `<= 1.15x` gate.
- Capability, preparation, schema, execution, semantic, generated, reference, performance,
  Javadoc, Markdown, exact-scope, status, and whitespace validation pass. No Java test suite is
  repeated by the documentation context absent executable change or a recorded concrete risk.
- No public API, shared module, dependency, architecture rule, Gradle configuration, native route,
  conformance/integration boundary, attention/loss behavior, or CPU 0008H-or-later task file
  changes.
- A separate documentation-focused context finalizes affected Javadocs, CPU guide, glossary
  impact, task evidence/status, master plan, and roadmap before this task becomes `Complete`.

## Tests / validation

### Tier 1: focused implementation validation

After executable Java stabilizes, run this exact focused command once:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.CpuInternalPackageInventoryTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool2dIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool2dLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionDagDecomposerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGeneratorTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPool2dGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuShapePolymorphicArtifactTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuPool2dReferenceTest
```

The focused suites must cover both kinds and attribute pairings; exact floor/ceil and all-padding
geometry; all types; special values and signed zero; zero extents; arbitrary positive strides;
arrays, segments, and mixed carriers; overlap/injectivity failures before writes; scalar/parallel
range ownership; schema/specialization/cache/finalizer propagation; exact descriptors; and
unchanged split/recognition/materialization behavior.

### Tier 2: generated-code and performance evidence

Run semantic/Class-File checks through the Tier-1 generated suite. Then create one fresh absolute
evidence directory and run the opt-in five-fork evidence owner exactly once:

```bash
SYNAPTIK_CPU_POOL2D_PERFORMANCE=true \
SYNAPTIK_CPU_POOL2D_EVIDENCE_ROOT=<fresh-absolute-evidence-directory> \
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPool2dPerformanceTest \
  --rerun-tasks
```

`CpuPool2dPerformanceTest` must launch and retain exactly five fresh fixed-heap Java 26 subprocess
forks, aggregate all measured rows, reject only a whole pre-measurement environmental/control
failure with a recorded reason, and fail the Gradle command on any semantic, structural,
per-fork, aggregate, inventory, or manifest failure. The full CPU suite runs this owner in a
non-measuring assumption/skip mode when the opt-in environment variable is absent.

Inspect every retained generated class with the JDK Class-File API and:

```bash
javap -c -v -p <each-retained-generated-pool2d-class>
```

Record the exact evidence directory, manifest digest, row inventory, environment, all fork and
aggregate ratios, rejected pre-measurement forks if any, and zero forbidden-reference findings in
this task before completion. Do not rerun, replace, or relabel a measured failure.

### Tier 3: CPU capability checkpoint

After focused and performance evidence stabilizes, run the complete CPU module suite exactly once:

```bash
./gradlew :backends:cpu:test
```

This module checkpoint is required because capability, family lowering, preparation, schema,
generated execution, cache compatibility, carrier binding, and parallel ranges change together.
The checkpoint must also exercise the five inventoried evidence controls whose only authorized
edit is advancing their stale current-schema assertion from 54 to 55; those mechanical edits do
not add Pool2d cases or reopen the completed batch-normalization, Conv2d, Conv3d, partition-DAG,
or pointwise evidence scopes.
Do not run repository-wide, architecture, backend-conformance, integration, Model, or Compiler
test suites unless implementation discovers an actual shared/boundary change. The Model and
Compiler contracts are read-only prerequisites and their completed evidence is reused.

### Tier 4: documentation and repository hygiene

The separate documentation-focused context runs after final Javadoc edits:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates rendered affected Javadoc pages; local Markdown links and heading anchors;
balanced fences; terminology; final newlines and trailing whitespace; exact package/type
placement; schema 55 and preserved 52/54 projections; exact changed-path inventory within the
38-path ceiling with exactly 16 production and 17 test paths; no generated/evidence leakage; and
synchronized 0008F Complete, 0008G Complete, 0008H Draft status. The implementation retained
0008G Ready and 0008H Draft until these gates passed; 0008H remains Draft.

Repository-wide validation is deferred to CPU 0009 or CI because this task changes one concrete
backend module without changing dependencies, architecture, shared build configuration, or a
shared/public contract.

## Dependencies

- [CPU 0008F](0008f-portable-matmul-execution-and-bounded-linear-epilogues.md) — Complete.
- [CPU 0008E1](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md) — Complete.
- [CPU 0008E](0008e-bounded-multi-input-materialization-and-representation-reuse.md) — Complete.
- [CPU 0008D](0008d-bounded-fusion-profitability-and-typed-decision-facts.md) — Complete.
- [CPU 0008C](0008c-typed-specialized-subgraph-and-epilogue-recognition.md) — Complete.
- [CPU 0008B](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md) — Complete.
- [Model 0020A](../../../modules/model/tasks/0020a-nchw-max-pool2d-semantics-and-tensor-expression.md) — Complete.
- [Model 0020A1](../../../modules/model/tasks/0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md) — Complete.
- [Compiler 0005D](../../../modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md) — Complete.

## Follow-up tasks

- CPU 0008H remains Draft and owns portable scaled-dot-product attention execution. Do not create
  its detailed specification or implement it in this task.
- CPU 0008I remains Draft and owns portable loss-family execution.
- CPU 0009 remains Draft and owns the portable generated-coverage closure checkpoint after the
  ordered 0008H and 0008I families complete.
- Any vectorized/tiled Pool2d realization, Pool2d input materialization, native DNN route,
  adaptive/global pooling, or pooling epilogue fusion requires a separately planned task with its
  own eligibility, resource, numerical, and performance evidence.

## Architecture impact

Expected impact: None.

This task adds one CPU-private portable family within the existing concrete-backend ownership of
lowering, route selection, exact declarations, finalization, generation, and execution. It uses
the existing shared `PartitionDag`, Prepare slot assignment, Runtime prepared-executable boundary,
and dependency direction. It changes no architecture rule or dependency, so no
`ARCHITECTURE.md`, focused architecture page, architecture decision record (ADR), or architecture
test update is required. If implementation proves any shared/public contract or architecture
change necessary, stop and report the conflict before editing outside this task.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are implementing Synaptik CPU task 0008G in a separate clean context.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md, the CPU master plan, completed CPU
tasks 0008B-0008F, Model tasks 0020A-0020A1, Compiler task 0005D, documentation rules/profiles,
this task file, and every directly affected source/test owner before editing.

Implement this task exactly as specified. Preserve the shared PartitionDag and CPU 0008B-0008E1
ownership boundaries. Add only direct generated scalar/parallel-scalar max and fixed-count average
Pool2d execution, schema 55 identity, checked static NCHW geometry, direct carriers, disjoint
output-cell ranges, and the same-boundary optimal clean Java oracle. Do not add fusion,
materialization, vector/tiled/native routes, backward work, dynamic geometry, shared/public APIs,
or CPU 0008H-or-later planning.

Run the exact focused, five-fork evidence, full CPU checkpoint, and scope checks. After executable
behavior stabilizes, hand the actual diff and exact Java/evidence results to a distinct clean
documentation-focused context. That context must finalize affected Javadocs, CPU guide, glossary
impact, planning evidence/status, and documentation validation in the same overall change without
repeating successful Java tests unless executable Java changes or a concrete risk is recorded.

Do not stage, commit, or push. Stop on architecture/shared-module need, scope overflow, or a
failed fixed evidence gate rather than widening scope or changing policy silently. Do not mark
0008G Complete until every acceptance, evidence, documentation, exact-path, and whitespace gate
passes; keep 0008H and later Draft.
```

## Local decisions

- Pooling is one cohesive CPU family because max and average share exact NCHW geometry, carriers,
  resources, output-cell ownership, and generated entry shape while retaining separate family-
  specific numerical bodies.
- The first portable implementation has exactly one direct scalar code shape. Parallelism is
  outer output-cell orchestration using existing workers; no vector or tile realization is
  justified by current source/evidence before implementation.
- Increasing kernel-height/kernel-width traversal is selected for generated/reference parity and
  deterministic CPU results. This is compatible with average pooling's permitted reassociation
  and max pooling's required first-winner order.
- Direct representations are complete and selected. Pool2d does not broaden the candidate-only
  materialization program established by CPU 0008E.
- Pooling epilogue fusion is not in scope because CPU 0008C explicitly excludes pooling
  recognition and current code has no pooling fact or executable form to consume.
- Schema 55 is required because Pool2d adds a new emitted family and code-shaping identity.
  Existing schema-52 and MATMUL schema-54 class projections remain stable.

## Known limitations

- Only fully static, resolved-layout NCHW BFLOAT16/FLOAT32/FLOAT64 Pool2d executes. Model symbolic
  Shape construction remains valid but is outside this CPU execution boundary.
- The first route is scalar or parallel-scalar only. General positive-stride layouts and every
  array/segment mixture are correctness fallbacks, not broad performance claims.
- Average finite non-zero results are deterministic for this selected CPU traversal but are not a
  promise of bitwise equality across backends because Model permits reassociation.
- No pooling fusion, input materialization, native route, saved max indices, or backward execution
  is added.

## Validation evidence

The implementation context passed the focused Pool2d and affected CPU tests, the schema-54
regression controls, generated structural scans, and the one required full CPU checkpoint: 596
tests, zero failures, zero errors, and four expected skips. No successful Java test suite was
repeated by the documentation context.

The retained evidence root is
`/private/tmp/synaptik-cpu-0008g-evidence-20260830-001`. Its checksum manifest validates 32 listed
artifacts, and `manifest.digest` validates the manifest content, for 33 checksum-valid evidence
entries. It contains five successful fork CSVs, six aggregate rows, no `failed-forks` directory,
the byte-identical direct-oracle source, generated classes, `javap` output, member scans,
specializations, and environment facts. The six generated/direct aggregate medians are
`0.998144202`, `1.010875034`, `1.051055579`, `1.074063542`, `1.106718493`, and `0.931805101`; all
passed the fixed `<= 1.15x` gate.

Clean documentation context `01a05161-2d20-7b70-9168-83269bf208b1` independently reconciled the
final implementation and tests
with the architecture, Model operation contracts, CPU guide, glossary, and affected Javadocs. The
final `:backends:cpu:javadoc` task passed with only the two expected incubating-module warnings.
Local Markdown links and anchors and all four changed Markdown fence counts passed. Exact scope is
37 paths: 16 production/Javadoc, 17 tests/evidence, and four documentation/planning paths. The
`git diff --check` validation passed, and no retained evidence artifact is present in the
repository.

## Implementation notes

- `CpuPool2dLowering` revalidates exact static NCHW descriptors, literal floor/ceil geometry,
  checked layouts, output injectivity, fixed divisor, and flattened output-cell count. It threads
  immutable geometry through the existing unit preparation/finalization/executable boundary with
  exactly one input, one output, and zero workspace.
- `CpuPool2dIr` and `CpuPool2dEmitter` add one `DIRECT_SCALAR` code shape for MAX and AVERAGE over
  BFLOAT16, FLOAT32, and FLOAT64 typed arrays, segments, and mixed carriers. Scalar and
  parallel-scalar orchestration invoke the same generated body over disjoint half-open ranges of
  complete output cells.
- The generated max and average loops implement the Model-owned padding, first-winner, signed-zero,
  accumulator, division, and BFLOAT16 narrowing rules without window materialization, hot helper
  dispatch, partial results, or a combine step. Cold executable validation rejects overlap before
  any write or worker submission and avoids invocation for empty output.
- Schema 55 owns the Pool2d projection. Existing schema-52 family bytes and MATMUL schema-54 bytes
  remain stable, while older compatibility envelopes are safe misses.
- `CpuPool2dReferenceKernel` is the optimal clean Java test/performance oracle only; Runtime never
  calls it as an interpreter or fallback. No planned deviation, fusion, materialization, vector,
  native, backward, public/shared, dependency, or architecture change was required.

## Completion summary

Completed portable generated MAX_POOL2D and fixed-divisor AVERAGE_POOL2D execution for the full
admitted static NCHW BFLOAT16/FLOAT32/FLOAT64 boundary. The change includes checked lowering and
geometry, schema-55 identity, scalar/parallel-scalar typed generated loops, cold carrier/alias
validation, an optimal clean Java oracle, semantic/structural/performance coverage, affected
Javadocs, the CPU backend guide, and synchronized task/master/roadmap evidence.

Validation passed at the task's focused, full CPU, structural, schema, performance, Javadoc,
Markdown, checksum, exact-path, and whitespace tiers. The existing glossary pooling entry remains
accurate and required no edit. Shared API, architecture, ADR, architecture tests, backend
conformance, integration, Gradle, Compiler/Training guides, and unrelated operation contracts also
required no change because the implementation is confined to the existing CPU-private portable
route and does not alter their contracts. No unresolved issue or required follow-up remains for
0008G. CPU 0008H remains Draft and is the next planned family task.

Status: Complete
