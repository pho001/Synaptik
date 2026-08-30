# CPU Task 0008G1: Portable Pool1d Composition Validation and Pool3d Generated Execution

## Status

Complete.

## Goal

Validate and execute the exact visible NCW Pool1d composition without inventing a Pool1d
operation kind, capability, IR family, or generated body, and add first-class portable CPU
execution for the complete current static NCDHW `MAX_POOL3D` and `AVERAGE_POOL3D` families.

The Pool1d path must recognize only
`EXPAND_DIMS(axis 2) -> POOL2D -> SQUEEZE(axis 2)`, keep both rank-edit results virtual, and reuse
the existing schema-55 Pool2d generated body. The Pool3d path must use a rank-five direct
complete-output-cell scalar body, with caller-owned parallelism only across disjoint output cells,
and prove that generated code has the same semantic algorithm, loop/dataflow shape, and
avoidable-overhead profile as an optimal clean Java implementation of the same specialization.

## Scope

### Source-backed baseline

- Model task 0025I exposes Pool1d only as three ordinary graph occurrences. The middle Pool2d has
  height geometry `(kernel, stride, padding, dilation) = (1, 1, 0, 1)` and the requested width
  geometry and literal ceil mode. There is no Pool1d operation kind or signature.
- CPU capability already reports the exact affine `EXPAND_DIMS` and `SQUEEZE` occurrences and the
  exact Pool2d occurrence independently. The current partition decomposer can seed the visible
  nodes separately, but `CpuPartitionLowering` has no exact three-node Pool1d recognizer, so the
  views are not yet preserved as one Pool2d-backed executable unit.
- Model task 0025J defines first-class rank-five `MAX_POOL3D` and `AVERAGE_POOL3D`; Model 0025K
  defines the separate general three-dimensional unfold/fold algebra; Compiler 0006B1 and 0006B2
  capture, validate, and differentiate the Pool3d signatures. None of those completions advertises
  current CPU Pool3d ownership or execution.
- The current CPU provider reports Pool2d but not Pool3d. CPU preparation has a dedicated Pool2d
  IR/lowering/emitter/reference family and a rank-five Conv3d lowering/emitter that demonstrates
  checked NCDHW output decoding and depth-aware address arithmetic. Neither is a generic PoolNd
  bridge.
- Generator schema 55 belongs to Pool2d. Unchanged families project schema 52, MATMUL projects
  schema 54, and Pool2d projects schema 55. Pool3d needs a new identity projection; recognized
  Pool1d must retain the existing Pool2d projection.

### Exact Pool1d composition validation

- Recognize exactly three nodes in strict order: rank-three NCW `EXPAND_DIMS` with normalized
  axis 2, matching rank-four Pool2d, then `SQUEEZE` with normalized axis 2. Both graph edges must be
  the exact producer-to-consumer values; both intermediate values must be private to this chain,
  unpublished, and single-use.
- Require the expanded descriptor to be `[N,C,1,W]` with the exact affine view layout derived by
  the current CPU affine rules. Require the squeezed output to be `[N,C,Wout]` with the exact view
  layout obtained by removing axis 2. Revalidate type, gradient eligibility, fully static Shape,
  resolved non-negative layout, and exact descriptor identity relationships at lowering.
- Accept only `MAX_POOL2D` with `MaxPool2dAttrs` or `AVERAGE_POOL2D` with
  `AveragePool2dAttrs`. Height attributes must be exactly `(1,1,0,1)`; width attributes and
  `ceilMode` remain exact. Reject every mismatch, extra edge, fan-out, publication, reordered node,
  extra affine node, different axis, or noncanonical height geometry.
- Lower the exact chain as one Pool2d-backed executable unit. The external rank-three input and
  output are the only declared boundary values. The expanded and squeezed values remain virtual
  views, with no slot, copy, workspace, materialization, generated invocation, or Runtime step.
- Reuse `CpuPool2dIr`, `CpuPool2dLowering.Geometry`, `CpuPool2dEmitter`, and schema-55 class
  identity. Translate the validated external NCW layouts into the exact rank-four singleton-height
  input/output layouts required by the existing Pool2d lowering. Do not create a Pool1d IR,
  emitter, reference kernel, schema, cache key, route, or fallback.
- Capability reporting remains occurrence-local and truthful: query and assert support separately
  for the actual `EXPAND_DIMS`, Pool2d, and `SQUEEZE` occurrences. Do not add a synthetic Pool1d
  query or capability. A supported three-component capability conjunction does not by itself
  guarantee that the topology recognizer accepts a malformed chain.
- If exact recognition fails, retain the established legal partition-DAG decomposition rather
  than silently treating a near match as Pool1d. The recognized fast path must not hide generic
  dispatch, an interpreted bridge, or an unreported operation.

### Pool3d admitted operations and geometry

- Admit exactly `MAX_POOL3D` paired with `MaxPool3dAttrs` and `AVERAGE_POOL3D` paired with
  `AveragePool3dAttrs`, one input and one output, and no other pooling or window kind.
- Admit only BFLOAT16, FLOAT32, and FLOAT64 rank-five input `[N,C,D,H,W]` and same-type,
  same-gradient-eligibility output `[N,C,Dout,Hout,Wout]`. Shapes must be fully static, layouts
  resolved with non-negative offsets and strides, and output layout injective. Read-only input may
  be non-injective.
- For each depth, height, and width axis, compute with checked `long` arithmetic:

  ```text
  effective = dilation * (kernel - 1) + 1
  numerator = inputExtent + 2 * padding - effective
  floorOut  = floor(numerator / stride) + 1
  ceilOut   = ceil(numerator / stride) + 1
  ```

  Reject negative numerator, overflow, or output-Shape disagreement. Literal ceil mode retains a
  terminal all-padding window; it does not apply a framework-specific tail-removal rule.
- Output `(n,c,od,oh,ow)` starts at
  `(od*strideDepth-paddingDepth, oh*strideHeight-paddingHeight,
  ow*strideWidth-paddingWidth)`. Visit kernel positions in increasing depth, then height, then
  width order, applying the corresponding dilation.
- Support static zero batch or channel as zero work. Zero input depth, height, or width is
  admissible only when the checked padding geometry produces the supplied positive spatial output;
  its windows follow the all-padding rules. Reject every unrepresentable flattened cell count,
  address span, byte offset, range endpoint, or invocation geometry.
- The Model divisor is a mathematical product. CPU lowering may fail closed if the exact positive
  `kernelDepth * kernelHeight * kernelWidth` cannot be represented by the primitive invocation
  boundary, but must not reinterpret, saturate, wrap, or reduce that divisor.

### Maximum Pool3d contract

- Exclude padding from selection. A window with no in-bounds sampled coordinate writes exact
  negative infinity in the result type.
- Any eligible NaN wins over every non-NaN, and the first eligible NaN wins. Otherwise use ordinary
  numeric maximum, rank positive zero above negative zero, and retain the first equal winner in
  increasing depth-height-width order. Real negative infinity is distinct from the all-padding
  state.
- Preserve the selected represented non-NaN value. For BFLOAT16, compare decoded values but retain
  the selected BFLOAT16 bits; do not introduce an accumulator narrowing. NaN payload/sign and
  signaling preservation remain unspecified beyond NaN class and first-winner selection.
- Use one-pass state shared across all three kernel loops. Do not allocate or publish winner
  indices, masks, counts, partials, or workspace.

### Average Pool3d contract

- Divide by the fixed positive mathematical count-padding divisor
  `kernelDepth * kernelHeight * kernelWidth`. Dilation changes sampled coordinates, not the
  divisor. Every logical kernel position counts; an out-of-bounds position contributes conceptual
  exact positive zero.
- BFLOAT16 and FLOAT32 accumulate and divide in FLOAT32; FLOAT64 accumulates and divides in
  FLOAT64. BFLOAT16 narrows once after the one final division. The direct and generated CPU forms
  use identical increasing depth-height-width accumulation order.
- Preserve the Model exceptional classes: an eligible NaN produces NaN; opposing infinities
  produce NaN; otherwise a present infinity retains its sign. An exact finite zero is negative
  only when every divisor position is an in-bounds negative zero. Cancellation, any positive zero,
  any padding position, or an all-padding window produces positive zero.
- Implement the signed-zero rule explicitly. Do not claim NaN payload/sign preservation,
  cross-backend finite bit identity, reassociation, a valid-sample divisor, or a divisor override.

### Lowering, generated route, and execution ownership

- Add focused CPU-private `CpuPool3dIr`, `CpuPool3dLowering`, `CpuPool3dEmitter`, and
  `CpuPool3dReferenceKernel`. The reference kernel is test/performance evidence only and must be
  unreachable from production Runtime execution.
- Select exactly `DIRECT_SCALAR`. Flatten output in stable NCDHW order; each half-open worker range
  owns complete output cells and every cell's complete depth-height-width window. Parallelize only
  across those cells through the existing caller-owned worker group. Add no depth/kernel split,
  partial reduction, combine, atomics, locks, nested workers, or Pool3d-specific worker threshold.
- Reuse the proven Pool2d generated design for carrier handling, max/average state, all-padding,
  average signed-zero tracking, preparation, and direct-vs-generated evidence. The extra depth
  dimension requires rank-specific output decoding, `kd` loop, depth bounds, depth strides,
  address arithmetic, and max first-winner/average accumulator state spanning all three loops.
  Implement those in the Pool3d emitter/reference; do not route through Pool2d, Conv3d, a generic
  PoolNd Runtime helper, unfold/fold, reflection, or an operation switch.
- Use one general checked-long rank-five address body for all admitted resolved layouts, matching
  the established Pool2d design. Do not add an unsupported dense-int, vector, tiled, native,
  packed, or materialized algorithm in this task. Access plans and carriers remain class-identity
  facts even though the emitted body has one address form.
- Support array, native-order `MemorySegment`, and all four input/output carrier pairings for each
  type. Reject output/input physical overlap before any write or worker submission. Declare exactly
  one input read and one output write, and zero workspace, materialization, scratch, columns,
  indices, counts, winners, or padding buffers.
- Thread Pool3d geometry through `LoweredPartition`, portable route planning, preparation,
  finalization, generated dispatch, prepared executable binding, and cache identity. Runtime sees
  only prepared primitive geometry, carriers/slots, range, and generated handle; it must not see
  Model `Operation`, `CompiledNode`, graph topology, or layout objects.
- Advance `CpuGeneratorSchema.CURRENT_VERSION` from 55 to 56. Pool3d uses schema 56 with every
  code-shaping kind/type/access/carrier/algorithm fact. Pool1d-recognized bodies and ordinary
  Pool2d retain schema 55; MATMUL retains 54; unchanged families retain 52. Advance compatibility
  envelopes to current-only 56 and prove old envelopes safe-miss without changing the retained
  class projections or bytes of unchanged controls.
- Generated classes remain final, field-free, and constructor-free, with one typed static entry.
  Their hot loops may contain primitive indexing, direct typed array/segment access, comparisons,
  accumulation, one average division, and one store. They must contain no allocation, boxing,
  reflection, collection/map lookup, string dispatch, graph/layout/operation lookup, cache/route
  selection, worker management, fallback call, or Synaptik-owned hot helper call.

### Capability and fallback boundaries

- Extend `CpuCapabilityProvider` only for exact Pool3d kind/attribute pairs and the static
  descriptor/geometry boundary above. A `true` answer means the ordinary generated portable route
  can prepare that occurrence; capability must fail closed for mismatches or unrepresentable CPU
  geometry.
- Pool1d capability remains the conjunction of the three visible component queries. No public
  Model API, shared capability result, or backend-contract change is permitted.
- Pool3d becomes an atomic numerical seed in the partition DAG. No pooling epilogue fusion,
  pointwise materialization candidate, external affine suffix, or general DAG reconstruction is
  added. Surrounding unsupported or separately executable work follows existing decomposition and
  preparation fallback rules.
- The generated portable Pool3d body is the production semantic baseline and safe CPU route.
  There is no interpreted reference fallback, generic bridge, Vector API route, OpenBLAS route,
  native peer, or Runtime operation dispatch.

## Exclusions

- Attention, CPU 0008H details, losses, Conv3d changes, Conv3d gradients, Pool3d gradients,
  unfold/fold execution, unpooling, saved indices, and training-specific CPU state.
- New public Model, Compiler, Planning, Prepare, Runtime, backend-contract, or capability APIs;
  new Pool1d/PoolNd operations; architecture or module-boundary changes.
- Pooling fusion, materialization, workspace, vector/native execution, autotuning, dynamic or
  symbolic execution Shapes, unresolved layouts, negative strides, in-place output, or
  cross-backend bitwise/performance guarantees.
- Unrelated refactors, schema cleanup, generic window abstractions, or 0008H task specification.

## Architecture references and constraints

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) is authoritative, especially Model semantic
  ownership, Planning occurrence ownership, backend Prepare ownership, finalization after shared
  slot assignment, and Runtime's prepared-schedule-only boundary.
- [`current-architecture-plan.md`](../../../../architecture/current-architecture-plan.md) explains
  the current module topology; this task changes no dependency direction or public ownership.
- [`planning-guide.md`](../../../planning-guide.md) governs Ready state, clean implementation and
  documentation contexts, validation tiers, completion evidence, and path-ceiling escalation.
- [Model 0025I](../../../modules/model/tasks/0025i-ncw-max-average-pool1d-composition.md),
  [0025J](../../../modules/model/tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md),
  and [0025K](../../../modules/model/tasks/0025k-public-ncdhw-unfold3d-and-fold3d-window-transforms.md)
  are the semantic contracts. [Compiler 0006B1](../../../modules/compiler/tasks/0006b1-pool3d-and-3d-window-forward-adoption-and-explicit-gradient-boundary.md)
  and [0006B2](../../../modules/compiler/tasks/0006b2-pool3d-and-3d-window-gradient-closure.md)
  are the capture/inference/gradient contracts.
- [CPU 0008G](0008g-portable-max-average-pool2d-execution.md) is the implementation and evidence
  precedent. Reuse its proven private design without turning it into a shared PoolNd framework.
- Preserve generated-code discipline from `AGENTS.md`: the optimal clean Java specialized case is
  the design/review oracle, and any hot-loop/dataflow deviation needs an explicit technical reason
  and evidence.

## Affected files and package impact

Expected production/Javadoc paths are the following 17; the category ceiling is 18:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `.../internal/cache/CpuGeneratorSchema.java`
- `.../internal/cache/CpuKernelSpecialization.java`
- `.../internal/codegen/emit/CpuClassFileKernelGenerator.java`
- new `.../internal/codegen/emit/CpuPool3dEmitter.java`
- `.../internal/executable/CpuPreparedExecutable.java`
- new `.../internal/ir/CpuPool3dIr.java`
- `.../internal/ir/CpuPortableKernelIr.java`
- new `.../internal/lowering/CpuPool1dCompositionLowering.java`
- `.../internal/lowering/CpuPartitionDagDecomposer.java`
- `.../internal/lowering/CpuPartitionLowering.java`
- new `.../internal/lowering/CpuPool3dLowering.java`
- `.../internal/prepare/CpuPartitionFinalizer.java`
- `.../internal/prepare/CpuPartitionPreparationPlan.java`
- `.../internal/prepare/CpuPartitionPreparer.java`
- new `.../internal/reference/CpuPool3dReferenceKernel.java`
- `.../internal/route/portable/CpuPortableRoutePlan.java`

The one-path production reserve is only for unavoidable mechanical propagation through an
existing CPU-private schema/generated carrier discovered during implementation. It cannot hold a
new abstraction, route, public API, algorithm, or module boundary, and its reason must be recorded.

Expected CPU test/evidence paths, maximum 21:

- update `CpuCapabilityProviderTest`, `CpuInternalPackageInventoryTest`,
  `CpuGeneratedKernelArtifactStoreTest`, `CpuKernelSpecializationTest`,
  `CpuClassFileKernelGeneratorTest`, `CpuPartitionDagDecomposerTest`,
  `CpuPartitionPreparerTest`, `CpuPartitionFinalizerTest`, `CpuPreparedExecutableTest`, and the
  schema/projection controls in `CpuBatchNormTrainingEvidenceTest`, `CpuConv2dEvidenceTest`,
  `CpuConv3dEvidenceTest`, `CpuPartitionDagGeneratedEvidenceTest`, and
  `CpuPointwiseLedgerEvidenceTest`;
- add focused `CpuPool1dCompositionLoweringTest`, `CpuPool3dIrTest`,
  `CpuPool3dLoweringTest`, `CpuPool3dReferenceTest`, `CpuPool3dGeneratedKernelTest`, and
  `CpuPool3dPerformanceTest`;
- use the final remaining path only for an existing shape-polymorphism or persistence control when
  implementation evidence shows that schema-56 projection reaches it.

Expected documentation/planning paths, maximum 9:

- `docs/backend-guide/cpu-backend.md`, `docs/api/tensor-api.md`, `docs/api/compile-api.md`, and
  `docs/glossary.md` when the documentation-focused review confirms affected current/future text;
- this task, the CPU master plan, Model capabilities, Model master plan, and roadmap.

Maximum scope is 48 changed paths: at most 18 production/Javadoc, 21 tests/evidence, and 9
documentation/planning paths. Generated evidence artifacts outside tracked source do not count.
Stop and return to planning before exceeding any category or the total; do not hide excess scope
as mechanical churn.

Package impact is confined to `backends:cpu` implementation/test packages and explanatory/planning
documentation. No module dependency, exported package, service registration, or architecture test
rule changes are expected.

## Acceptance criteria

1. Exact valid Pool1d chains lower as one Pool2d-backed executable with only external NCW
   boundaries, virtual rank edits, schema-55 identity, and byte-identical generated Pool2d class
   bytes for matched specialization facts.
2. Every malformed or nonprivate Pool1d near-match is rejected by recognition and safely follows
   existing decomposition. Capability tests report only the actual expand, Pool2d, and squeeze
   components; no Pool1d kind/query/IR/schema appears.
3. CPU capability and lowering accept exactly the stated static Pool3d matrix and independently
   reject kind/attrs, type, rank, gradient, Shape, layout, injectivity, geometry, overflow, and
   representability mismatches.
4. Direct reference and generated Pool3d results agree for both kinds, all three types, floor and
   ceil, asymmetric per-axis parameter values, padding, dilation, stride, dense/noninjective-input/
   general layouts, arrays, segments, mixed carriers, zero domains, all-padding windows, overlap
   rejection, scalar ranges, and parallel range subdivision.
5. Maximum tests prove first depth-height-width ties, first NaN, positive-over-negative zero,
   infinities, real negative infinity versus all-padding, BFLOAT16 selected-bit retention, and no
   winner state. Average tests prove fixed divisor, padding participation, one final BFLOAT16
   narrowing, accumulation domain/order, NaN, opposing/single infinities, cancellation,
   all-negative zero, positive zero, padding-forced positive zero, and all-padding positive zero.
6. Pool3d uses zero workspace/materialization and disjoint complete-output-cell ranges. Prepared
   Runtime state contains no Model/Compiler graph object and production cannot call the reference
   kernel or a generic fallback bridge.
7. Schema 56 is current-only compatible; Pool3d identity includes all code-shaping facts. Existing
   schema-52, schema-54, and schema-55 controls keep their intended projections and generated
   bytes. Recognized Pool1d adds no projection.
8. Generated structural evidence enumerates all 24 body families (`2 kinds * 3 data types * 4
   input/output carrier pairings`) because Pool3d deliberately has one general-long access body.
   For every family, retain the full Class-File and complete `javap -c -v -p` output and verify one
   typed entry, depth-height-width hot loops, direct load/store, and the forbidden-structure list.
   Tests fail if production selection emits an unenumerated body family.
9. The generated code has no avoidable helper dispatch, allocation, boxing, reflection,
   map/collection/string dispatch, cache/route/fallback overhead, or Synaptik-owned hot call. Any
   generation-time source helper must disappear into direct bytecode; no hidden generic bridge is
   accepted.
10. Five isolated fixed-heap Java 26 forks compare generated and direct optimal clean Java using
    randomized generated/direct order, five warmup rounds, nine measured rounds, adaptive batches
    of at least 25 ms, and no retry/discard. Every fork and aggregate generated/direct ratio is
    `<= 1.15x` for at least dense FLOAT64 max/average 3x3x3, padded+dilated+ceil FLOAT32
    max/average, and mixed-carrier BFLOAT16 max/average. Add a row only if implementation creates
    another emitted hot-loop family.
11. Documentation accurately distinguishes current Pool1d composition optimization from component
    capability and current Pool3d CPU execution from Model/Compiler semantics. A clean
    documentation-focused agent finalizes affected Javadoc, CPU guide, Tensor/Compile API text,
    glossary impact, and reasoned no-change conclusions.
12. The implementation stays within the path ceiling, contains no unrelated attention/Conv3d/
    gradient/API/architecture work, and all required validation and evidence pass before the task
    status becomes Complete.

## Validation

### Tier 1: focused semantic, lowering, preparation, and schema validation

Run the focused CPU test owners for capability, Pool1d recognition, Pool3d reference/IR/lowering/
generation, DAG decomposition, preparation/finalization/executable binding, cache/schema controls,
artifact persistence, and unchanged generated-family projections. Record the exact Gradle
`--tests` command and test count in implementation evidence.

### Tier 2: generated structure and performance

- Generate and retain all 24 Pool3d family Class-Files and full `javap -c -v -p` reports under an
  explicit evidence directory. Run both semantic bytecode inspection and independent forbidden
  constant-pool/instruction/call-owner scans for every file.
- Run `CpuPool3dPerformanceTest` only under an explicit opt-in environment flag and evidence-root
  property, in five fresh 1 GiB Java 26 forks with the methodology and gates above. Retain raw
  samples, medians, ratios, fork metadata, generated classes, and decompilation. Do not rerun a
  failed fork to manufacture acceptance.

### Tier 3: module checkpoint

Run `./gradlew :backends:cpu:test` once after focused validation because this task changes CPU
capability breadth, partition lowering, preparation plumbing, and generator compatibility. Run
`./gradlew :backends:cpu:javadoc` after the documentation/Javadoc pass. Run `git diff --check` and
the repository's Markdown link/anchor/fence validation commands.

### Tier 4: boundary conclusions

- Backend conformance and integration source modules are currently marker modules with no pooling
  harness. Do not add placeholder tests. Record a reasoned no-change conclusion unless the
  implementation introduces a callable shared harness or end-to-end path, in which case stop and
  replan the added scope.
- No architecture/module dependency changes are planned, so architecture-test changes and a full
  repository test are not required. If implementation changes a dependency boundary, public
  contract, shared build configuration, or multiple modules, stop and obtain a revised plan before
  running the corresponding higher tier.
- The documentation-focused agent must not repeat successful Java suites unless it changes
  executable Java behavior or finds a concrete reason. It owns documentation validation and may
  reuse implementation evidence.

Before marking Complete, verify exactly one detailed Ready/Complete CPU 0008G1 task exists, all
links resolve, CPU 0008G is Complete, Model 0025I/J/K and Compiler 0006B1/B2 remain Complete, CPU
0008H remains Draft without a detailed task file, and roadmap/master-plan order agrees.

## Dependencies

- CPU 0008G, Model 0025I, Model 0025J, Model 0025K, Compiler 0006B1, and Compiler 0006B2 are
  Complete.
- Existing CPU 0008A–0008G partition-DAG, recognition, preparation, direct-representation,
  generated-code, evidence, and Pool2d contracts remain authoritative.
- Java 26 Class-File generation/decompilation, fixed-heap fork harness, artifact retention, typed
  carriers, and caller-owned worker infrastructure already exist.

## Follow-ups

- CPU 0008H remains Draft and owns scaled-dot-product attention only after this task completes. Do
  not create its detailed task specification here.
- CPU 0009 later closes the portable generated-coverage capability/conformance checkpoint.
- Pooling fusion, vector/native Pool3d, materialization, dynamic execution, and any shared
  conformance harness require separately planned evidence-backed tasks.

## Architecture impact

No architecture change is intended. The task fills an existing CPU-backend implementation slot:
Model keeps semantic ownership, Planning keeps occurrence ownership, CPU Prepare chooses and binds
the generated route, and Runtime receives only prepared execution state. Any need for a new shared
API, module edge, Runtime interpreter, Pool1d operation, or generic bridge is an architectural
uncertainty and must stop implementation for clarification.

## Implementation prompt

Use a mandatory separate clean implementation context. Give it this task, `AGENTS.md`, the
authoritative architecture/planning documents, Model 0025I/J/K, Compiler 0006B1/B2, CPU 0008G,
and the directly affected CPU sources/tests. Tell it to implement only the bounded Java/test work,
run Tiers 1–3 through the CPU module checkpoint, retain all generated/performance evidence, avoid
documentation finalization, make no commit or push unless separately authorized, and return the
required completion summary with exact changed paths, commands, results, unresolved issues, and
`Status: Complete` or `Status: Incomplete`.

After implementation succeeds, use a distinct mandatory clean documentation-focused context. Give
it the final diff, task, relevant documentation rules and General/API-Javadoc/Planning/Example
profiles, CPU guide, Tensor/Compile API docs, glossary, and affected Javadoc. It must independently
review and finalize documentation/Javadoc, record explicit no-change conclusions, run CPU Javadoc
and documentation validation without repeating successful Java tests absent cause, enforce the
48-path ceiling and task status gates, and return the same required completion-summary format.

## Local decisions

- Pool1d is exact topology recognition plus Pool2d reuse, never a capability or operation family.
- Pool3d uses a dedicated rank-five direct scalar generated/reference loop. Pool2d supplies the
  numerical/evidence pattern; Conv3d supplies evidence that rank-five decode and depth addressing
  need rank-specific code. Neither is called as a hot-path bridge.
- Pool3d has one general-long address form. Therefore the exhaustive emitted-body inventory is 24
  kind/type/carrier families, while semantic tests independently cover dense and general layouts.
- Deterministic CPU average accumulation is depth-major, then height, then width, matching the
  direct oracle. This is an implementation choice within the Model contract, not a new Model rule.
- Direct representation, zero workspace/materialization, complete-output-cell parallelism, and
  schema 56 are the only selected route/resource/compatibility choices.

## Known limitations

- Only fully static, resolved non-negative-layout BFLOAT16/FLOAT32/FLOAT64 occurrences are admitted.
- Output/input overlap, unrepresentable geometry/divisor/address domains, and noninjective outputs
  fail closed even when abstract Model semantics are otherwise meaningful.
- No vector/native/materialized Pool3d route, Pool3d fusion, saved winner state, or Pool1d-specific
  artifact exists. Performance evidence is bounded to retained representatives and environment.
- The shared backend-conformance and integration modules have no applicable executable pooling
  harness at planning time.

## Evidence

Planning evidence inspected the clean current worktree; authoritative architecture/planning
documents; completed Model 0025I/J/K and Compiler 0006B1/B2; completed CPU 0008G; CPU capability,
affine, partition-DAG, Pool2d, Conv3d, schema, preparation, generated artifact, decompilation, and
benchmark sources/tests; backend capability documentation; CPU guide; Tensor/Compile API docs;
glossary; and the conformance/integration marker modules. This evidence supports the bounded
algorithm and ownership choices above. It is not implementation evidence.

## Notes

- Preserve the exact source-backed names and status ordering. Do not broaden this task because a
  nearby Pool3d window, gradient, Conv3d, or attention facility exists.
- If the final source proves that any stated 24-family body inventory or schema projection cannot
  be expressed without a materially different hot loop or public boundary, stop and amend this
  Ready task rather than silently changing the design.

## Validation evidence

Implementation-context validation completed on Java 26.0.1/OpenJDK 64-Bit Server VM:

- Tier 1 used one focused `./gradlew :backends:cpu:test` invocation with the 20 exact owners
  `CpuCapabilityProviderTest`, `CpuInternalPackageInventoryTest`,
  `CpuPool1dCompositionLoweringTest`, `CpuPool3dIrTest`, `CpuPool3dLoweringTest`,
  `CpuPool3dReferenceTest`, `CpuPool3dGeneratedKernelTest`,
  `CpuPartitionDagDecomposerTest`, `CpuPartitionDagResourceTest`, `CpuPartitionPreparerTest`,
  `CpuPartitionFinalizerTest`, `CpuPreparedExecutableTest`,
  `CpuGeneratedKernelArtifactStoreTest`, `CpuKernelSpecializationTest`,
  `CpuClassFileKernelGeneratorTest`, `CpuBatchNormTrainingEvidenceTest`,
  `CpuConv2dEvidenceTest`, `CpuConv3dEvidenceTest`,
  `CpuPartitionDagGeneratedEvidenceTest`, and `CpuPointwiseLedgerEvidenceTest`. Result: 20 suites,
  181 tests, 0 skipped, 0 failures, and 0 errors.
- Tier 2 used the opt-in properties `pool3dPerformance=true` and
  `pool3dEvidenceRoot=/tmp/synaptik-cpu-0008g1-pool3d-evidence-20260830c` through a temporary
  test-environment init script, with `:backends:cpu:test --tests
  io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPool3dPerformanceTest
  --rerun-tasks`. The temporary init script was removed after the run. Exactly five fresh `-Xms1g
  -Xmx1g` forks passed; every individual row was at most `1.002680589x`. Median-of-five aggregate
  ratios were dense FLOAT64 max `0.955486129x`, padded/dilated/ceil FLOAT32 max `0.866583071x`,
  mixed-carrier BFLOAT16 max `0.988159577x`, dense FLOAT64 average `1.001036519x`,
  padded/dilated/ceil FLOAT32 average `0.966994679x`, and mixed-carrier BFLOAT16 average
  `0.932421527x`.
- Two earlier development evidence roots failed the per-fork FLOAT32 max gate before the clean
  Java oracle was corrected to preserve the generated general-long address/dataflow shape. They
  were not retried, averaged, or included in the accepted root. The final retained run followed
  the source correction and used a fresh evidence root.
- The accepted root contains five raw fork CSV files, all 24 distinct generated Class-Files, 24
  complete `javap -c -v -p` reports, 24 member-reference scans, 24 specialization snapshots, the
  direct-oracle source, environment metadata, aggregate results, and a 104-entry SHA-256 manifest.
  The manifest digest is
  `80457c6c181e0ae80e6c33c264ddbc02202c79bd3f67e76cf017172d540cf85c`.
  All generated families were final, field-free, constructor-free, single-entry bodies with no
  Synaptik, collection/map, reflection, allocation, boxing, string-dispatch, route/cache, or
  fallback hot call. Segment families retain only their required direct `MemorySegment.get/set`
  calls.
- Tier 3 ran the required final `./gradlew :backends:cpu:test` checkpoint once after focused
  validation: 120 suites, 618 tests, 5 intentional opt-in skips, 0 failures, and 0 errors.
- Schema controls passed with 56 current-only, Pool3d at 56, Pool1d-recognized and ordinary Pool2d
  at 55, MATMUL at 54, and unchanged generated families at 52. The recognized Pool1d and direct
  Pool2d specializations produced byte-identical generated classes.
- Implementation stayed at 42 changed paths: 17 production/Javadoc, 20 tests/evidence, and 5
  planning paths. `git diff --check` passed. No architecture/module dependency, shared build,
  backend-conformance, or integration harness changed, so architecture-test, conformance,
  integration, and repository-wide validation remain reasoned no-change conclusions.
- CPU Javadoc, explanatory/API/glossary review, Markdown links/anchors/fences, final path audit,
  status/order audit, and the overall Complete transition remain owned by the mandatory independent
  documentation-focused context.

Independent documentation context `01a053e7-8514-7622-b16c-44ef10c39460` used the General,
API/Javadoc, Planning, Backend Guide, and Example profiles. It reviewed all 17 affected production
contracts, all 20 affected tests/evidence owners, the CPU guide, Tensor and Compile API references,
glossary, Model capability/master-plan records, CPU master plan, roadmap, task lineage, and status
gates. It changed no executable Java behavior and did not repeat the successful Java suites.

- The first `./gradlew :backends:cpu:javadoc` completed but reported missing record-component and
  compatibility-constructor documentation in the provisional Javadocs. The documentation context
  corrected those comments and one inaccurate reference-oracle overlap claim. The final
  `./gradlew :backends:cpu:javadoc --rerun-tasks` passed with exactly the two expected
  `jdk.incubator.vector` warnings and no documentation warnings. Rendered `CpuPool3dIr`,
  `CpuPool3dLowering.Geometry`, and `CpuPool3dReferenceKernel` pages exist and contain the
  finalized component and failure contracts.
- The CPU guide now documents exact Pool1d recognition/schema-55 reuse and the bounded direct
  Pool3d schema-56 route. Tensor and Compile API text preserves Model/Compiler ownership while
  identifying only the fully static resolved-layout non-gradient CPU subset. The glossary now
  distinguishes visible Pool1d component capability from CPU topology recognition and records the
  bounded Pool3d execution boundary. No new reusable term was introduced.
- A targeted Ruby checker validated local Markdown targets, GitHub-style heading anchors, and
  balanced backtick/tilde fences in all nine changed Markdown files. Early checker-development
  invocations had a Ruby interpolation error and then an incomplete/incorrect slug calculation;
  neither was accepted. The corrected final invocation passed all nine files.
- Final scope is 46 paths: 17 production/Javadoc, 20 tests/evidence, and 9 documentation/planning.
  This remains below the 48-path total and within every category ceiling. No retained evidence
  artifact entered the repository.
- Final status/order inspection confirmed CPU 0008G, Model 0025I/J/K, and Compiler 0006B1/B2
  remain Complete; CPU 0008G1 is the sole detailed task at this position and is Complete; CPU
  0008H remains Draft with no detailed task file. CPU master plan, Model records, and roadmap agree.
- `git diff --check` passed after final content/status edits. Architecture, ADR, package-summary,
  Tensor/Compile Java API, Training API/documentation, backend-conformance, integration,
  architecture-test, Gradle, and other-module changes were not required: the implementation stays
  inside the existing CPU-private prepare/finalize/executable boundary and changes no shared or
  public Java contract.

## Implementation notes

The implementation context added strict three-node Pool1d composition recognition and reuses the
existing Pool2d IR, geometry, schema-55 specialization, and generated body with only external NCW
boundaries. Both rank-four intermediates remain virtual; malformed topology, affine descriptors,
height geometry, or publication obligations fail recognition, and a valid near-match decomposes
through the existing partition DAG. No Pool1d kind, capability, IR, schema, or artifact was added.

Pool3d now has dedicated CPU-private IR, lowering, clean-Java reference, and Class-File emitter
owners. Capability and lowering admit only non-gradient static resolved NCDHW BFLOAT16/FLOAT32/
FLOAT64 max and fixed-divisor average occurrences, including noninjective read-only inputs and
injective outputs. Preparation, schema projection, persistence, finalization, scalar/caller-parallel
execution, overlap validation, and complete-output-cell range ownership carry checked rank-five
geometry with zero workspace or materialization. Generated max and average bodies preserve
depth-height-width/width-fastest traversal, padding/dilation/stride, special-value and signed-zero
rules, direct typed array/segment access, and the general-long address shape of the direct oracle.

The independent documentation-focused context finalized all affected production Javadocs,
including record-component and compatibility-constructor contracts, and narrowed the reference
oracle's failure documentation to its actual checks. It finalized the CPU guide, Tensor API,
Compile API, glossary, capabilities, master plans, roadmap, and this task without changing
executable Java behavior or tests.

## Completion summary

Completed exact visible NCW Pool1d topology recognition with virtual rank edits and byte-identical
schema-55 Pool2d reuse, plus direct schema-56 NCDHW max/average Pool3d generated execution for the
bounded static BFLOAT16/FLOAT32/FLOAT64 matrix. Implementation evidence passed 181 focused tests,
the 618-test CPU checkpoint, all 24 structural family scans, and all retained five-fork performance
gates; the two earlier FLOAT32 maximum development-root failures remain explicitly pre-fix and are
not acceptance evidence.

The independent documentation context finalized four explanatory/API/glossary documents, five
planning documents, and affected Javadocs; passed final CPU Javadoc, nine-file Markdown
link/anchor/fence validation, exact 46-path scope, status/order, and whitespace checks; and recorded
reasoned no-change conclusions for shared/public APIs, package summaries, architecture/ADR/tests,
backend conformance, integration, Gradle, Training documentation, and other modules. No commit or
push was made. Unresolved issues: None. Follow-up required for CPU 0008G1: None. Draft CPU 0008H
remains separate and has no detailed task.

Status: Complete
