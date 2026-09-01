# CPU Task 0008H: Portable Scaled Dot-Product Attention Execution

## Status

Complete.

## Goal

Execute exactly the current first-class one- and two-output scaled dot-product attention
occurrences on the portable CPU route. Support the optional right-broadcast BOOL mask, top-left
causal eligibility, explicit or embedding-derived scale, all current floating type promotions,
and the complete Model exceptional-value contract through one direct scalar generated family.

The implementation must preserve attention as one atomic numerical unit, publish the requested
output slots exactly once, use one declared per-range score/weight workspace, and partition only
complete broadcast-batch/query rows. Generated code must preserve the semantic algorithm,
hot-loop/dataflow shape, and avoidable-overhead profile of an optimal clean Java implementation of
the same exact type, eligibility, output, carrier, layout, and range specialization.

## Scope

### Source-backed occurrence and descriptor contract

- Admit only `ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION` paired with exact
  `ScaledDotProductAttentionAttrs`.
- Admit exactly these ordered occurrence forms:

  ```text
  inputs  [query, key, value]
  outputs [output]

  inputs  [query, key, value, mask]
  outputs [output]

  inputs  [query, key, value]
  outputs [output, weights]

  inputs  [query, key, value, mask]
  outputs [output, weights]
  ```

  The one-output form has slot zero only. The two-output form has output at slot zero and
  canonical normalized weights at slot one from the same exact producer. There is no hidden
  output, weights reconstruction, scores output, saved-value output, or weights-only form.
- Revalidate the current Model/Compiler Shape contract exactly:

  ```text
  query   [...queryBatch, L, E]
  key     [...keyBatch,   S, E]
  value   [...valueBatch, S, Ev]
  score   [...broadcastBatch, L, S]
  output  [...broadcastBatch, L, Ev]
  weights [...broadcastBatch, L, S]
  ```

  Query, key, and value rank are at least two. Their batch prefixes broadcast right-aligned
  together. Query/key `E` are equal and positive; key/value `S` are equal. The optional mask is
  exact BOOL, has rank no greater than score rank, and right-broadcasts exactly to score Shape;
  rank-zero masks are valid.
- Require fully static Shapes and resolved layouts with non-negative storage offsets and strides.
  Both outputs must have injective write layouts. Read-only query, key, value, and mask layouts may
  be non-injective. Reject unresolved/dynamic dimensions, negative strides, noninjective outputs,
  Shape/layout disagreement, or unrepresentable counts, spans, addresses, byte sizes, range
  endpoints, and workspace arithmetic.
- Require query, key, and value to be BFLOAT16, FLOAT32, or FLOAT64. Promote query/key first, then
  value, exactly as Model and Compiler do. Output and optional weights have that promoted type.
  Explicit scale must have that exact type; its already-validated value remains finite and
  positive. No cast, FLOAT16, integral, BOOL-data, or quantized attention form is added.
- Revalidate descriptor metadata, not only Shape and type. Output gradient eligibility is exact
  query/key/value OR. Weights eligibility is exact query/key OR. Mask does not contribute to
  either result. Forward CPU execution accepts either valid output count regardless of whether a
  descriptor requests gradients; it adds no autograd or training-mode route.
- Preserve duplicate semantic input roles. Query, key, value, and mask positions may refer to a
  shared read-only boundary value. Lowering records occurrence-position-to-unique-boundary mapping,
  while ordered unique inputs precede the one or two distinct output boundaries.

### Eligibility and numerical semantics

- For each broadcast-batch coordinate and query position `i`, key position `j` is eligible when
  the optional broadcast mask is true and, when `causal` is true, `j <= i`. Explicit and causal
  eligibility combine by logical AND. There is no causal offset, right alignment, sliding window,
  additive mask, bias, or numeric mask.
- Test eligibility before reading query/key components for that score. Test it again before value
  arithmetic. An excluded position receives positive-zero weight and its query/key/value special
  values cannot affect the row.
- For each eligible `j`, calculate the increasing-`E` dot product and multiply by scale in the
  selected accumulation domain. Then apply these exact row classifications in order:

  1. no eligible positions: all requested weights and output components are positive zero;
  2. any eligible NaN score: every eligible requested weight and every non-empty output component
     is NaN, while excluded requested weights are positive zero;
  3. one or more eligible positive-infinity scores: each positive-infinity position receives
     `1 / count`, every other eligible position receives positive-zero weight, and output uses the
     ordinary eligible-position weighted-value rule;
  4. all eligible scores are negative infinity: all requested weights and output components are
     positive zero without value reads;
  5. otherwise use stable max-shift softmax; eligible negative infinity receives positive-zero
     weight and finite eligible weights total one ideally.
- Ordinary output accumulation visits every eligible value row even when its eligible weight is
  zero. Therefore eligible value NaN propagates through ordinary multiply/add; only an excluded
  position is skipped. The mandated no-eligible and all-negative-infinity cases write positive
  zero directly.
- FLOAT64 score dots, shifted-exponential sums, weights, and output sums use binary64. FLOAT32 and
  BFLOAT16 use binary32 for all four, decoding BFLOAT16 inputs before arithmetic. BFLOAT16 narrows
  requested weights and output once at their final stores with the established round-to-nearest,
  ties-to-even conversion.
- Freeze one CPU algorithm for generated/direct identity. Visit `E`, then `S`, then `Ev` logical
  coordinates in increasing order. Use ordinary left-associated primitive multiply/add for dot
  and output sums. In an ordinary softmax row, find the maximum, evaluate
  `StrictMath.exp(score - maximum)` and narrow that result to binary32 exactly once for the
  binary32 domain, accumulate the denominator left-associated, then divide. Do not introduce
  Kahan state, reassociation, explicit fused multiply-add, vector reduction, partial combination,
  or a second rounding point. This is a CPU implementation choice allowed by Model, not a new
  cross-backend bitwise promise.
- Resolve scale before generated execution. An explicit FLOAT64 scale retains its binary64 value;
  an explicit FLOAT32/BFLOAT16 scale is decoded to binary32. The semantic default becomes
  `1.0d / StrictMath.sqrt((double) E)` for binary64 and one binary32 narrowing of that quotient for
  the binary32 domain. Generated code receives the resolved primitive scale; it performs no
  square root, attribute inspection, optional dispatch, or type decision.
- Preserve empty-domain meaning. `L == 0` and a zero broadcast-batch extent create no rows.
  `S == 0` writes positive-zero output for every existing output component and has no weight
  element. `Ev == 0` creates no output element, but a two-output occurrence still calculates and
  publishes weights. Static `E == 0` is rejected. A row with neither an output component nor a
  requested weight element is zero work.

### Lowering, workspace, generated route, and parallelism

- Add focused CPU-private `CpuAttentionIr`, `CpuAttentionLowering`, `CpuAttentionEmitter`,
  `CpuAttentionMaskValidator`, and `CpuAttentionReferenceKernel` owners. The reference kernel is
  test/performance evidence only and must be unreachable from production Runtime execution.
- Select exactly one rank-polymorphic checked-long `DIRECT_SCALAR` realization. Concrete ranks,
  extents, right-aligned batch/mask mappings, offsets, strides, scale bits, row counts, and scratch
  offsets remain cold primitive geometry. The generated class specializes kind, ordered boundary
  types, mask presence, causal form, output count, accumulator form, scratch-bearing entry shape,
  and exact carriers; it does not specialize arbitrary Shape magnitudes.
- Use one scratch slice per selected scalar or caller-parallel range. A non-empty slice holds
  exactly `S` accumulation-domain values: raw eligible scores after the score pass, then normalized
  weights in place. Its byte size is `alignUpExact(S * accumulatorBytes, 8)`, where
  `accumulatorBytes` is eight for FLOAT64 and four otherwise. Total workspace is exactly
  `scratchSliceBytes * selectedRangeCount`, aligned to eight bytes. `S == 0` or zero work declares
  no workspace. Overflow fails during analysis.
- Declare the exact workspace during CPU analysis, before shared slot/resource assignment. Shared
  Prepare assigns it without understanding attention. CPU finalization verifies the same
  declaration, selected range count, slice size, and scratch-bearing generated signature. Binding
  gives each selected range one disjoint slice; no hidden array, score tensor, weight tensor,
  allocation, recomputation buffer, thread-local state, or retained cross-run state is allowed.
- The generated body uses a score pass over eligible `S`, the row-classification state above, an
  in-place score-to-weight pass when required, an increasing-`S` weighted sum for each increasing
  `Ev`, and an optional direct increasing-`S` weights store. The one-output form does not store or
  publish weights; the two-output form stores slot one from the same scratch values used for slot
  zero. No MATMUL/SOFTMAX decomposition or helper call is present.
- Flatten work as complete broadcast-batch/query rows in stable logical row-major order. One range
  owns each row's full `E`, `S`, and `Ev` domains plus its private scratch slice. Reuse the existing
  scalar/caller-parallel selection and fixed caller-owned worker group. Add no split within a row,
  nested workers, atomics, locks, partial scores, partial sums, combine phase, attention-specific
  threshold, or worker-owned allocation.
- Thread attention geometry through `LoweredPartition`, portable route planning, analysis,
  preparation, finalization, generated dispatch, and prepared executable binding. Runtime sees
  only assigned slots/resources, direct carriers, primitive geometry, ranges, and a generated
  handle. It never sees `Operation`, `CompiledNode`, Tensor/provenance, layouts, masks as semantic
  objects, or route selection.
- Support primitive arrays, native-order `MemorySegment`, and every legal per-boundary mixed
  carrier pattern: `short[]`/segment for BFLOAT16, `float[]`/segment for FLOAT32,
  `double[]`/segment for FLOAT64, and `byte[]`/segment for BOOL. The scratch carrier is always the
  exact assigned `MemorySegment`.

### Publication, binding, and overlap behavior

- Declare one read per unique query/key/value/optional-mask boundary and one write for each exact
  occurrence output. Preserve role mapping when a read value appears in multiple semantic
  positions. A two-output occurrence has one atomic CPU execution unit with two output writes;
  it is not two units and does not infer publication from consumer count.
- Validate liveness, carrier class, native byte order, access mode, exact data type, alignment,
  logical geometry, complete referenced spans, output writability/injectivity, and checked range
  geometry on the invoking thread. Validate every logically represented mask byte as canonical
  `0` or `1`, once per distinct mask coordinate rather than once per broadcast use.
- Complete all carrier, mask, workspace, and overlap validation before scratch mutation, output
  writes, generated calls, or worker submission. Invalid input leaves every output and the
  workspace unmodified and submits no worker work.
- Read-only query, key, value, and mask spans may overlap each other. Reject each output span
  against every input span, reject output/output overlap for the two-output form, and reject the
  workspace against every input and output span. In-place or partially overlapping attention is
  unsupported even when a particular mask would avoid some reads.
- Each selected parallel range has disjoint logical output rows, disjoint injective output spans,
  and a disjoint scratch slice. Join synchronously through the existing worker group. Preserve its
  deterministic worker failure, interruption, close, nested-submission, and shutdown behavior;
  attention adds no lifecycle policy.
- The enclosing prepared partition publishes output validity only after the attention invocation
  and every submitted range complete successfully. A validation failure is pre-write. A later
  generated/worker failure publishes no output validity; physical bytes already written by a
  completed range may be partial and remain unobservable through the invalid run state, matching
  the established atomic prepared-partition boundary rather than promising transactional memory.

### Partition-DAG, capability, schema, and fail-closed boundaries

- Extend `CpuCapabilityProvider` only for the exact first-class occurrence matrix above. A `true`
  answer means ordinary CPU preparation can realize that occurrence through the generated
  portable route with truthful buffers/workspace. Capability fails closed for every kind/attrs,
  cardinality, descriptor, type, gradient, Shape, layout, broadcast, injectivity, and
  representability mismatch.
- Seed an admitted first-class attention occurrence as one atomic numerical unit and one
  numerical-order barrier in the existing partition DAG. Do not fuse an external ADD, activation,
  normalization, mask, softmax, MATMUL, or other suffix/prefix; do not offer a 0008E
  materialization candidate. Surrounding supported units follow the established 0008B split and
  atomic sequential publication rules.
- Never recognize a decomposed `MATMUL -> mask/WHERE -> SOFTMAX -> MATMUL` or another approximate
  topology as attention. Near matches remain their actual first-class component occurrences and
  must not inherit attention's mask, infinity, NaN, output-count, or resource semantics.
- Advance `CpuGeneratorSchema.CURRENT_VERSION` from 56 to 57. Attention uses schema 57 with every
  code-shaping type, eligibility, output-count, accumulator, scratch-entry, semantic-position, and
  carrier fact. Scale value, extents, ranks, layouts, broadcast mappings, addresses, selected
  ranges, slots, and workspace identity remain cold and do not fragment class identity.
- Compatibility is current-only schema 57. Existing families retain their exact projections and
  generated bytes: unchanged families schema 52, MATMUL 54, Pool2d 55, and Pool3d 56. Every older
  persistent envelope is an incompatible safe miss; no legacy reader, converter, or migration is
  added.
- The direct generated scalar body is the safe production fallback for every admitted static
  layout/carrier form. If its complete proof or exact resource assignment fails, preparation fails
  before realization. There is no interpreted reference fallback, generic attention bridge,
  component decomposition fallback, Runtime operation switch, Vector API route, OpenBLAS/native
  route, materialized route, retry, or silent slower helper path.
- Generated classes are final, field-free, and constructor-free with one public typed static
  entry. Their hot work may contain primitive address arithmetic, typed array/segment access,
  eligibility branches, primitive classification, `StrictMath.exp`, embedded BFLOAT16 conversion,
  scratch loads/stores, and direct output stores. It may not contain allocation, boxing,
  reflection, method-handle lookup, `invokedynamic`, collection/map/string dispatch, monitor use,
  graph/layout/operation lookup, route/cache/resource selection, worker management, generic
  `Object` entry descriptors, reference/fallback calls, or any Synaptik-owned hot helper call.

## Exclusions

- New Model, Compiler, Planning, Prepare, Runtime, backend-contract, Tensor, Compile, or Training
  API; new semantic kind/attrs/signature; changed attention or gradient formulas.
- Attention dropout, RNG state, training execution, saved intermediates, caches, key/value caches,
  grouped-query, multi-query, sparse, packed, block, flash, additive-bias, offset-causal,
  cross-attention special cases, FLOAT16, integral, quantized, or native attention.
- Dynamic or symbolic execution, unresolved layouts, negative strides, output aliasing,
  materialization, packing, tiling, vectorization, native/OpenBLAS execution, autotuning, or a
  cross-backend bitwise/performance guarantee.
- Generic attention decomposition, recognition of decomposed graphs, new fusion, loss execution,
  gradient execution, compiler saved-value lifetime, hidden weights, or an interpreted fallback.
- Unrelated refactors, schema cleanup, broad tensor-contraction/normalization abstractions, module
  dependency changes, architecture changes, CPU 0008I implementation, or a detailed later task.

## Architecture references and constraints

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) is authoritative, especially Model semantic
  ownership, Planning occurrence ownership, CPU Prepare route/resource ownership, staged analysis
  before shared assignment and finalization, Runtime's prepared-schedule-only boundary, and the
  generated-code performance discipline.
- [`current-architecture-plan.md`](../../../../architecture/current-architecture-plan.md) explains
  the current module topology. This task changes no public/shared ownership or dependency edge.
- [`planning-guide.md`](../../../planning-guide.md) governs Ready state, path ceilings, isolated
  implementation/documentation contexts, validation tiers, and completion evidence.
- [Model 0019E](../../../modules/model/tasks/0019e-scaled-dot-product-attention.md) owns the exact
  attention Shapes, promotion, mask/causal/scale, exceptional values, accumulation domains, and
  one-output construction. [Model 0023F](../../../modules/model/tasks/0023f-scaled-dot-product-attention-weights-output.md)
  owns the same-occurrence canonical weights output and its descriptor metadata.
- [Compiler 0005D](../../../modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
  owns capture/inference revalidation and exact two-output gradient use. It deliberately fails
  one-output gradient requests closed; this task implements forward execution only.
- [CPU 0008B](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md)
  through [0008E1](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md) own the
  shared partition-DAG/unit/resource flow. [CPU 0008F](0008f-portable-matmul-execution-and-bounded-linear-epilogues.md)
  and [CPU 0007E](0007e-portable-stable-softmax-and-log-softmax-coverage.md) are algorithmic
  precedents, not decomposed production implementations of attention. [CPU 0008G1](0008g1-portable-pool1d-composition-validation-and-pool3d-generated-execution.md)
  supplies the current schema and evidence baseline.
- Preserve `AGENTS.md` generated-code discipline. The optimal clean Java implementation of each
  exact specialization is the design/review/performance oracle, and any semantic algorithm,
  hot-loop/dataflow, or avoidable-overhead deviation requires an explicit reason and evidence.

### Package, type, and test map

- `io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider` owns truthful public capability
  answers for exact attention occurrences; no new public type or method is added.
- `...internal.ir.CpuAttentionIr` owns the closed direct-scalar code identity. Existing
  `CpuPortableKernelIr`, `CpuFusionDecision`, and `CpuSpecializedSubgraph` add only the attention
  family/barrier and exact `ATTENTION_ROW_STATE` workspace vocabulary needed by current CPU-private
  decision facts.
- `...internal.lowering.CpuAttentionLowering` owns occurrence/descriptor revalidation, static
  broadcast/layout geometry, scale resolution, semantic-role mapping, row/range facts, and exact
  scratch size. Existing partition lowering/decomposition, recognizer, and profitability owners
  propagate the atomic barrier and workspace role without recognizing or fusing attention.
- `...internal.codegen.emit.CpuAttentionEmitter` owns the complete direct generated algorithm.
  `CpuClassFileKernelGenerator` performs only generation-time dispatch/validation and must not
  absorb attention arithmetic.
- `...internal.executable.CpuAttentionMaskValidator` owns canonical logical BOOL validation;
  `CpuPreparedExecutable` owns carrier/span/overlap validation, cold primitive geometry packing,
  scratch slicing, and existing scalar/parallel orchestration.
- `...internal.reference.CpuAttentionReferenceKernel` owns independent clean Java semantic and
  performance evidence only. It is not a production fallback or a production dependency.
- Existing cache, portable-route, preparation, and finalization types own schema-57 identity,
  exact `ATTENTION_ROW_STATE` declaration/assignment verification, and generated-handle
  realization in their current layers.
- Focused tests map one owner per responsibility: capability; IR identity; lowering/geometry;
  independent reference semantics; canonical-mask validation; generated execution/binding;
  exhaustive Class-File evidence; and the bounded 992-row five-fork performance matrix. Existing DAG,
  preparation/finalization, executable, artifact-store, schema, and unchanged-family tests remain
  cross-owner controls rather than a second attention oracle.

## Affected files and package impact

Expected production/Javadoc paths are at most 30:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`;
- existing cache owners `CpuGeneratorSchema`, `CpuKernelSpecialization`, and their package docs;
- existing generator owner `CpuClassFileKernelGenerator`, new `CpuAttentionEmitter`, and package
  docs;
- existing executable owner `CpuPreparedExecutable`, new `CpuAttentionMaskValidator`, and package
  docs;
- new `CpuAttentionIr`, existing `CpuPortableKernelIr`, `CpuFusionDecision`,
  `CpuSpecializedSubgraph`, and IR package docs;
- new `CpuAttentionLowering`, existing `CpuPartitionLowering` and
  `CpuPartitionDagDecomposer`, `CpuFusionProfitabilitySelector`,
  `CpuSpecializedSubgraphRecognizer`, and lowering package docs;
- existing `CpuPartitionPreparationPlan`, `CpuPartitionPreparer`, `CpuPartitionFinalizer`, and
  preparation package docs;
- new `CpuAttentionReferenceKernel` and reference package docs;
- existing `CpuPortableRoutePlan` and portable-route package docs.

The remaining production reserve covers only unavoidable propagation through an existing
CPU-private prepared-unit carrier or generated-schema owner discovered during implementation. It
cannot hold a new route, public/shared API, generic abstraction, second algorithm, or module edge.

Expected CPU test/evidence paths are at most 30:

- update capability, public/internal inventory, schema, specialization, artifact-store,
  Class-File generator, partition-DAG, lowering, preparation/finalization, prepared-executable,
  overlap/resource, and unchanged-family schema-projection controls;
- add focused `CpuAttentionIrTest`, `CpuAttentionLoweringTest`,
  `CpuAttentionReferenceKernelTest`, `CpuAttentionMaskValidatorTest`,
  `CpuAttentionGeneratedKernelTest`, `CpuAttentionEvidenceTest`, and
  `CpuAttentionPerformanceTest`;
- update the existing schema-52, MATMUL-54, Pool2d-55, and Pool3d-56 evidence controls only as
  needed to prove unchanged projections and bytes.

Expected documentation/planning paths are at most nine:

- `docs/backend-guide/cpu-backend.md`, `docs/api/tensor-api.md`, `docs/api/compile-api.md`, and
  `docs/glossary.md` when the later clean documentation review confirms affected current/future
  text;
- this task, the CPU master plan, Model capabilities, Model master plan only if independently
  stale, and roadmap.

Maximum implementation scope is 69 changed paths: at most 30 production/Javadoc, 30 test/evidence,
and nine documentation/planning paths. Generated Class-Files, decompilation, benchmark CSV, and
manifests under the explicit untracked evidence root do not count. Stop and return to planning
before exceeding a category or total; do not hide scope as package-info or mechanical churn.

Package impact is confined to existing `backends:cpu` implementation/test packages plus
explanatory and planning documentation. No module dependency, exported package, service
registration, architecture test, shared backend-contract, conformance, or integration change is
expected.

## Acceptance criteria

1. CPU capability and lowering accept exactly the current first-class three/four-input and
   one/two-output static attention occurrences and independently reject every stated kind/attrs,
   cardinality, descriptor, type, Shape, layout, broadcast, injectivity, and representability
   mismatch.
2. Boundary order, duplicate-role mapping, output-slot order, gradient metadata checks,
   publications, buffers, and workspace declarations exactly match each occurrence. One-output
   execution has no hidden weights boundary; two-output execution publishes both slots from one
   atomic unit.
3. Reference and generated results agree across all 27 ordered query/key/value floating type
   triples, default/explicit scale, mask absent/present, causal false/true, one/two outputs,
   ranks two and higher, unequal broadcast prefixes, scalar/lower-rank masks, dense/general/
   broadcast input layouts, injective strided outputs, every legal duplicate semantic-input
   mapping, arrays, segments, every mixed carrier pattern, scalar ranges, and caller-parallel
   subdivisions.
4. Semantic tests prove exclusion before query/key/value reads; no-eligible, `S == 0`, and
   all-negative-infinity positive zero; eligible score NaN; positive-infinity tie splitting;
   finite stable max shift; eligible negative-infinity zero weight; eligible value NaN at zero
   weight; infinities and signed zero; sequential accumulation/narrowing; and `L == 0`, `Ev == 0`,
   and zero batch/head axes.
5. Analysis declares exactly the aligned `S`-value scratch slice per selected non-empty range and
   no workspace for `S == 0` or zero work. Finalization and binding preserve exact size/alignment,
   scratch-bearing entry shape, disjoint slices, and no hidden allocation/materialization/state.
6. Binding validates canonical mask values and every carrier/span/overlap fact before mutation or
   worker submission. Input/input sharing is legal; output/input, output/output, and
   workspace/buffer overlap fail without modifying outputs or scratch.
7. Attention is one numerical-order barrier and one multi-store unit when applicable. No external
   fusion/materialization or decomposed-attention recognition occurs; malformed attention fails
   closed and surrounding actual component operations retain established DAG handling.
8. Schema 57 is current-only and attention identity contains every code-shaping fact. Existing
   schema-52, MATMUL-54, Pool2d-55, and Pool3d-56 projections and generated bytes remain unchanged.
9. Exhaustive generated inventory contains exactly 11,880 attention specializations. For distinct
   semantic inputs, the 27 ordered type triples contribute 7,776 classes: two causal forms times
   the carrier-pattern sum `16 + 32 + 32 + 64` for unmasked one-output, masked one-output,
   unmasked two-output, and masked two-output entries. Legal duplicate query/key/value roles add
   3,888 two-unique-input and 216 one-unique-input classes. Equivalently, the type-compatible role
   partitions comprise 27 three-input mappings, 27 two-input mappings, and three one-input
   mappings, each contributing `36 * 2^uniqueFloatingInputs` classes. Default versus explicit
   scale and concrete geometry do not create another class. Tests fail on a missing, duplicate,
   or unenumerated specialization.
10. Retain the complete Class-File and full `javap -c -v -p` report for all 11,880
    specializations.
    Every report proves one public typed static entry, no fields or constructors, the exact
    scratch/geometry/range signature, direct score/normalization/value/optional-weight loops,
    direct typed loads/stores, and the complete forbidden-instruction/descriptor/call-owner scan.
11. No generated specialization contains avoidable allocation, boxing, reflection,
    `invokedynamic`, generic object access, collection/map/string dispatch, monitor use, graph or
    layout inspection, worker/cache/route/resource selection, reference/fallback calls, or a
    Synaptik-owned hot helper. Embedded BFLOAT16 work and JDK primitive/`StrictMath.exp` calls are
    the only nontrivial scalar facilities allowed by this task.
12. A bounded 992-row matrix, not all 11,880 specialization identities, passes generated-versus-
    optimal-clean-Java performance evidence in each of five fresh isolated forks. The matrix is
    complete over emitted performance-relevant factors:

    - 912 core rows cross all 57 type-compatible semantic-role mappings from criterion 9 with all
      eight mask-presence, causal, and output-count modes and both uniform carrier regimes (all
      primitive arrays and all `MemorySegment` boundaries); and
    - 80 mixed-carrier rows use distinct FLOAT32 query/key/value roles and, for every one of the
      eight modes, flip each active unique boundary once from array to segment and once from
      segment to array. The active-boundary counts are four, five, five, and six for unmasked one-
      output, masked one-output, unmasked two-output, and masked two-output forms respectively,
      so `2 causal forms * (4 + 5 + 5 + 6) boundaries * 2 flip directions == 80`.

    Together the rows cover every emitted score/normalization/value/optional-weight loop shape,
    the sole exact/default numerical mode, all 27 ordered type triples and their binary32/binary64
    accumulation, BFLOAT16 decode/final-narrowing paths, all mask/causal/output-count forms, every
    semantic-role duplicate mapping, and array and segment access at every active boundary. The
    direct side uses the same exact carrier signature, geometry, algorithm, traversal,
    arithmetic, scratch shape, and stores with no adapter, callback, reflection, or production/
    reference helper in timed work. Every row in every fork and each row's median-of-fork-medians
    aggregate ratio is `<= 1.15x`.
13. A deterministic performance-equivalence manifest maps every one of the 11,880 exact
    specialization identities to its normalized loop-skeleton hash and ordered boundary-access
    fragment hashes. Normalization may remove only binary/class names, constant-pool indexes, and
    mechanically shifted local-slot numbers. It must retain control-flow edges, loop nesting and
    order, primitive arithmetic/conversion/classification instructions, calls and call owners,
    semantic-role alias mapping, mask/causal/output-count branches, stores, and each boundary's
    data type and array/segment access fragment. The scan must prove that carrier choice changes
    only the corresponding direct access fragment and entry descriptor, never the surrounding
    loop/dataflow skeleton. Every distinct skeleton and every distinct role/type/carrier access
    fragment must occur in at least one of the 992 timed rows; an unclassified or uncovered body
    component fails the task. This structural decomposition is the proof that unmeasured
    multi-mixed carrier permutations compose already inspected and timed code shapes rather than
    introducing another hot algorithm or access mechanism.
14. Documentation accurately distinguishes Model semantics, Compiler inference/gradient use, and
    current CPU forward execution. A clean documentation-focused agent finalizes affected
    Javadoc, CPU guide, Tensor/Compile API text, glossary impact, planning evidence, and reasoned
    no-change conclusions.
15. The implementation stays within every path ceiling, changes no public/shared API or
    architecture boundary, leaves CPU 0008G1 Complete and CPU 0008I Draft without a detailed
    specification, and passes every required validation before this task becomes Complete.

## Validation

### Tier 1: focused semantics, capability, lowering, preparation, and schema

Run the focused CPU test owners for attention capability, IR, reference semantics, mask
validation, generated execution, partition-DAG barriers, preparation/finalization, resources,
prepared binding, overlap, schema/specialization, persistence, and unchanged-family projections.
Record the exact Gradle `--tests` command, suite/test count, failures, errors, and skips.

Include deterministic differential seeds and explicit independent expected values. A production
reference-kernel call is a failure, not a differential oracle. Verify invalid capability and
prepare cases separately so a broad `false` answer cannot conceal an admitted-form regression.

### Tier 2: complete generated structure and performance

- Generate all 11,880 attention Class-Files under one explicit evidence root and retain full
  `javap -c -v -p` output for each. Verify deterministic names/hashes, schema projection, class
  version, access flags, member inventory, entry descriptors, stack/local validity, scratch
  signature, store count, direct loops, allowed JDK calls, and the forbidden list above. Independently
  scan raw constant pools, descriptors, instructions, and call owners; do not rely only on text
  search of decompilation. Retain the exact-to-normalized performance-equivalence manifest,
  normalized loop-skeleton and boundary-access fragment inventories, their hashes, and the
  complete 11,880-to-992 coverage report required by criterion 13.
- Run `CpuAttentionPerformanceTest` only with an explicit opt-in environment flag and evidence-root
  property. Run exactly the 992 rows from criterion 12 in each of five fresh Java 26 forks with
  fixed `-Xms1g -Xmx1g`, identical generated/direct inputs, a fixed hot dense rank-four workload
  with query, key, and value logical Shapes all
  `[2,2,32,64]`, and a `[1,2,32,32]` broadcast mask for masked forms. Equal Shapes make every
  legal type-compatible duplicate-role mapping benchmarkable; separate semantic tests retain
  unequal batch prefixes and `Ev`. Carrier storage and output count vary with the exact
  specialization.
- In every fork and for every performance row, use five warmup rounds and nine measured rounds,
  randomize generated/direct order per round from a recorded seed, choose an adaptive batch whose
  measured side is at least 25 ms, consume outputs to prevent dead-code elimination, and retain
  all raw times. Do not retry, discard, replace, cherry-pick, or rerun a failed sample. Report each
  row/fork ratio plus the row's median-of-fork-medians aggregate; every ratio must be `<= 1.15x`.
- Retain fork commands, JDK/OS/architecture/heap metadata, CPU identity, raw CSV, direct-oracle
  source identity, specialization manifest, Class-Files, decompilation, forbidden-scan report,
  normalized equivalence/coverage manifests, hashes, and summary. A task result with an uncovered
  loop skeleton, numerical/type path, role mapping, mode, boundary-access fragment, individual
  row/fork ratio, or row aggregate is incomplete.

### Tier 3: CPU module checkpoint and documentation

Run `./gradlew :backends:cpu:test` once after focused validation because the task changes CPU
capability breadth, partition lowering, resource assignment, prepared execution, and generator
compatibility. Run `./gradlew :backends:cpu:javadoc` after the clean documentation/Javadoc pass.
Run the repository Markdown link/anchor/fence checks, exact changed-path inventory, status/order
checks, `git diff --check`, and a tracked-artifact scan that confirms performance evidence did not
enter source control.

### Tier 4: boundary conclusions

- Backend conformance and integration remain marker modules without a callable attention harness.
  Do not add placeholders. Record a reasoned no-change conclusion unless implementation exposes a
  real shared/end-to-end harness, in which case stop and replan the added scope.
- No public/shared API, module dependency, build configuration, or architecture rule changes are
  planned, so architecture-test changes and repository-wide tests are not required. If one occurs,
  stop and obtain a revised task before applying the corresponding higher tier.
- The documentation-focused context must not repeat successful Java suites unless it changes
  executable Java behavior or identifies a concrete cause. It may reuse implementation evidence
  and owns Javadoc/documentation validation.

Before marking Complete, verify one detailed CPU 0008H task exists, it changes Ready to Complete
only after all evidence passes, CPU 0008G1 remains Complete, CPU 0008I remains Draft without a task
file, Model 0019E/0023F and Compiler 0005D remain Complete, all links resolve, and CPU master-plan
and roadmap status/order agree.

## Dependencies

- CPU 0008G1, Model 0019E, Model 0023F, and Compiler 0005D are Complete.
- Existing CPU 0008B–0008E1 partition-DAG, unit/resource, publication, and shared-DAG adoption;
  CPU 0008F numerical contraction; CPU 0007E stable normalization; and CPU 0008G1 schema/evidence
  infrastructure are required implementation precedents.
- Existing Java 26 Class-File generation/decompilation, current-only artifact persistence,
  primitive array/segment carriers, shared resource assignment, fixed worker group, and
  fixed-heap fork harness remain available.

The order is correct because Model first owns the exact forward occurrence and weights semantics,
Compiler already preserves/revalidates those occurrences and consumes only canonical two-output
weights for gradients, and CPU 0008B–0008G1 now supply the necessary atomic-unit, multi-output,
resource, generated-code, and evidence machinery. Loss execution in CPU 0008I can follow without
being entangled with attention's distinct row scratch, mask, and exceptional-value policy.

## Follow-ups

- CPU 0008I remains Draft and owns the three current loss families after attention execution is
  Complete. Do not create its detailed task specification in this work.
- CPU 0009 later closes the full portable generated-coverage checkpoint.
- Attention vector/native/packed/flash routes, materialization, fusion, dropout, dynamic execution,
  and any real shared conformance/integration harness require separate source-backed planning.

## Architecture impact

No architecture change is intended. The task fills an existing CPU backend implementation slot:
Model owns attention meaning and output descriptors, Compiler owns capture/inference/gradients,
Planning owns the CPU occurrence assignment, CPU analysis/Prepare owns route and exact resources,
shared Prepare assigns them, CPU finalization binds the chosen generated artifact, and Runtime
invokes only prepared primitive state. Any need for a new public/shared contract, dependency edge,
Runtime interpreter, decomposed-attention recognizer, or unplanned resource kind is architectural
uncertainty and must stop implementation for clarification.

## Implementation prompt

Use a mandatory separate clean implementation context. Require it to read `AGENTS.md`,
`ARCHITECTURE.md`, `docs/planning/planning-guide.md`, and this exact task,
`docs/planning/backends/cpu/tasks/0008h-portable-scaled-dot-product-attention-execution.md`, before
implementation, then inspect Model 0019E/0023F, Compiler 0005D, CPU 0007E/0008B–0008G1, and the
directly affected CPU source/tests. Tell it to implement only the bounded Java/test/evidence work,
preserve CPU 0008G1 and old schema projections, run Tiers 1–3 through the CPU checkpoint, retain
all 11,880 structural artifacts and the exact 992-row five-fork performance/equivalence artifacts,
make no commit/push unless separately authorized, and return exact changed paths, commands,
results, evidence root/hashes, unresolved
issues, and the required completion status. It must stop and return to planning before editing if
source conflicts with `ARCHITECTURE.md` or this task, or if implementation needs work outside the
stated scope or path ceiling.

After implementation succeeds, use a distinct mandatory clean documentation-focused context.
Give it the final diff, task, documentation rules and General/API-Javadoc/Backend-guide/Planning/
Example profiles, affected Javadocs, CPU guide, Tensor/Compile APIs, glossary, Model capabilities,
master plan, and roadmap. It must independently finalize documentation, record explicit no-change
conclusions, run CPU Javadoc and documentation validation without repeating successful Java tests
absent cause, enforce the 69-path ceiling/status gates, and return the required completion summary.

## Local decisions

- One- and two-output forms stay in one task because they are one semantic kind and share the same
  score/weight state; splitting them would duplicate the numerical, mask, workspace, and generated
  review while leaving the Compiler's canonical weights occurrence unusable.
- Attention is one atomic rank-polymorphic checked-long scalar family. Arbitrary static rank/layout
  facts remain cold geometry so the generated inventory is finite and no rank-specific policy or
  unsupported signature is invented.
- Per-range `S` scratch is the smallest resource that preserves one score pass, stable
  normalization, reuse of the exact normalized weights for output and slot-one publication, and
  independent complete-row parallelism. It avoids an `L*S` tensor and avoids recomputing dots.
- Causal form, mask presence, output count, type triple, and carriers are generation-time facts.
  Scale and Shape values are cold because they alter values/addresses but not loop/dataflow shape.
- The complete emitted specialization inventory is 11,880, including type-compatible duplicate
  semantic input-role mappings. Every class receives full structural evidence. Performance uses
  the exact 992-row factor-complete matrix because the other 10,888 identities vary only by
  structurally proved compositions of already covered loop skeletons and direct boundary-access
  fragments; they do not receive redundant timing merely because their descriptors or multiple
  carrier positions differ.

## Known limitations

- Only fully static, resolved non-negative-layout BFLOAT16/FLOAT32/FLOAT64 attention executes.
  Dynamic Shapes, negative strides, output aliasing, and unrepresentable primitive geometry fail
  closed even when abstract Model semantics are meaningful.
- The selected implementation is scalar generated code with caller-owned row parallelism and one
  exact per-range workspace. It provides no vector/native/packed/flash route or cross-backend
  finite-bit/performance promise.
- Structural evidence remains exhaustive because every type/eligibility/output/carrier identity
  is a generated specialization. Performance evidence is bounded to 992 rows whose normalized
  factor coverage proves the omitted carrier permutations add no new hot-loop/dataflow or access
  component. The fixed workload proves generated/direct overhead parity, not universal workload
  performance or an autotuning threshold.
- Shared backend-conformance and integration modules have no applicable attention harness at
  planning time.

## Evidence

Planning inspected the current uncommitted planning worktree; authoritative architecture/planning/documentation
rules; CPU master plan and completed 0007E/0008B–0008G1 tasks; Model 0019E/0023F source, tests,
Tensor API, capabilities, master plan, and glossary; Compiler 0005D and current attention
capture/inference/preflight/gradient source/tests; backend capability contracts; Compile API and
CPU guide; and current CPU capability, IR, lowering, partition-DAG, representation, preparation,
resource, generated Class-File, cache/schema, prepared execution, overlap, reference, and
performance infrastructure. This supports the exact bounded decisions above and is not
implementation evidence.

## Notes

- Planning documents are non-authoritative. If source or `ARCHITECTURE.md` contradicts this task,
  stop and amend the Ready plan before implementing.
- Do not reduce the exhaustive Class-File matrix or the 992-row performance-factor matrix by
  treating mixed-carrier access, one output, masking, causal eligibility, or mixed floating types
  as cold. Only multi-mixed carrier permutations proved compositional by criterion 13 omit timing.
- Do not broaden the task because nearby MATMUL, softmax, loss, autograd, or native facilities
  exist. They have distinct ownership and semantics.

## Validation evidence

Implementation evidence, supplied to and independently inspected by the documentation context:

- Focused attention owner suites previously passed 64 tests with no failures. This documentation
  pass reuses that evidence and does not rerun Java tests because it changes comments and Markdown
  only.
- Exhaustive structural evidence is retained at
  `/tmp/synaptik-cpu-0008h-evidence/structural-delta-review-20260901a`: `BUILD SUCCESSFUL`,
  11,880 generated classes, 992 timed rows, 456 normalized skeletons, and 312 ordered boundary
  fragments. Verifier, member, constant-pool, call-owner, forbidden scan, and 11,880-to-992
  skeleton/fragment coverage passed. Inventory SHA-256
  `1d0dc5b397429da4e876252b82b05bf0e9e8a63b8100c9638aadfebea78921a2`; fragment inventory
  `2614f63804ad5f5432498e33155779ce5e6b8faca430787cf8566bfad5699232`; loop skeleton
  `8f192aed0dedf35283bfaf1647374fcdd24b0ccb6fa4f571dca91ae554cbb9b5`; timed components
  `fb60a90a470503dc60e7f07b5c3408f2cb96a3149be43ebb74495517f418cb26`; manifest
  `9f8fa203f5700528790c31382cff1270cdc4d2336e4c43d549738f583e7701e0`; and javap
  `9e3c25237f9e2f82a93530c449ee8548de93ba23d32a6414e1dfaac9b3965860`. The 312 fragments are
  broader valid compositional coverage, not a 252-fragment requirement.
- Authoritative performance evidence is retained at
  `/tmp/synaptik-cpu-0008h-evidence/performance-authoritative-final-20260901b`. With an explicit
  evidence root, `SYNAPTIK_CPU_ATTENTION_PERFORMANCE=true`, `--no-daemon`, and `--rerun-tasks`,
  `CpuAttentionPerformanceTest` completed `BUILD SUCCESSFUL` in 5h41m29s: five fresh forks times
  992 rows equals 4,960 row/fork results and 992 aggregates. Every ratio was `<= 1.15`; the worst
  individual ratio was `1.090399240` and the worst median aggregate ratio was `1.004388537`.
  Generated/direct semantic equivalence and driver preparation passed. Manifest digest:
  `e5fa969f3af2a6dfe728004fb3c9a0afdb1195cc41fdaca3b3fa01d56c2b98c7`.
- An older CPU checkpoint reported 628 tests: 620 passed, 7 skipped, and one failure from a
  missing retained historical CPU 0008E evidence CSV, not an attention assertion. It remains as
  historical evidence only. The immediate pre-final discovery run also reported 628 tests with
  two failures and seven skips: (1) the same missing retained historical CPU 0008E CSV/control
  artifacts and (2) one unrelated noisy
  `CpuPartitionDagGeneratedEvidenceTest` performance sample, where fork 0 had no accepted sample
  at or below `1.15`. The latter was neither an attention-correctness failure nor a failure of
  CPU 0008H's authoritative 992-row performance matrix.
- The coordinator restored the exact immutable historical artifacts outside the worktree from the
  original agent log: both CSV SHA-256 values matched their committed expectations, and the two
  regenerated control Class-Files had their expected hashes. None is tracked or part of this
  change. Both affected owners then passed together with `BUILD SUCCESSFUL in 8s`:
  `./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRepresentationPlannerTest.retainedEvidenceAndGeneratedClassControlsRemainExactWithoutRerunningForks --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPartitionDagGeneratedEvidenceTest.fiveFixedHeapForksAndAggregateStayWithinGate`.
  The final authoritative `./gradlew :backends:cpu:test` checkpoint then passed with `BUILD
  SUCCESSFUL in 17s`: 127 suites, 628 tests, 0 failures, 0 errors, and 7 expected skips.

Documentation context `01a05d40-07bf-7501-a9e1-41df7a43f793` selected the API/Javadoc, backend-guide, planning, general, and
example profiles. It reviewed the complete implementation diff; all affected production types and
surrounding contracts; attention tests and generated evidence owners; the CPU guide; Tensor and
Compile API references; glossary; Model 0019E/0023F; Compiler 0005D; CPU 0007E/0008B/0008E1/
0008F/0008G/0008G1; CPU master plan; roadmap; and architecture/planning rules. It finalizes the
Javadoc and documentation described below. Final `./gradlew :backends:cpu:javadoc` was
`BUILD SUCCESSFUL` (the module retains 87 pre-existing broad-record Javadoc warnings; no
attention-specific warning remains). The repository Markdown validator passed the five changed
non-Tensor Markdown files. Its full Tensor API run reports only that file's existing repeated
example headings; a targeted equivalent local-link/fence/final-newline/trailing-whitespace check
for the changed Tensor API passed. `git diff --check` passed.

## Implementation notes

- CPU lowering keeps attention atomic and records only CPU-private direct-route identity and cold
  geometry. Generated schema 57 is a direct scalar execution route, with one scratch slice per
  selected range; it is not a Model semantic implementation, Compiler gradient path, or Runtime
  route selector.
- Documentation now distinguishes the Model's backend-independent construction, Compiler's
  capture/inference/gradient ownership, and the current CPU forward subset. It explicitly excludes
  dropout, dynamic shapes, negative strides, overlap/in-place execution, decomposed recognition,
  fusion, materialization, vector/native/packed/flash routing, and universal-performance claims.
- The CPU guide adds the current family; Tensor/Compile API and glossary wording no longer says
  that all attention execution remains planned. Their public semantic and gradient contracts did
  not change.

## Completion summary

- Documentation/Javadoc pass: finalized the CPU-private attention emitter, mask validator, IR,
  and lowering Javadocs; added the current bounded CPU route to the CPU guide; narrowed stale
  Tensor/Compile API and glossary execution statements; and synchronized the CPU master-plan row
  and roadmap status/order after the final checkpoint. Documentation-pass paths are
  `CpuAttentionEmitter`, `CpuAttentionMaskValidator`, `CpuAttentionIr`, `CpuAttentionLowering`,
  `CpuPartitionLowering`, `cpu-backend.md`, `tensor-api.md`, `compile-api.md`, `glossary.md`, the
  CPU master plan, roadmap, and this task record.
- No-change conclusions: `ARCHITECTURE.md`, architecture explanations/tests, backend-conformance,
  integration, Gradle/build configuration, Model capabilities/master plan, and other modules need
  no change. The architecture and dependency edges are unchanged; conformance/integration have no
  callable attention harness; build configuration is unaffected; Model ownership/capability text
  and Compiler public gradient text already describe the current semantic/gradient contracts.
  CPU master-plan and roadmap status/order are synchronized: CPU 0008H is Complete, CPU 0008G1
  remains Complete, CPU 0008I is the next Draft CPU family frontier without a task file, and Model 0019E/0023F plus Compiler
  0005D remain Complete.
- Documentation/Javadoc validation is complete. The final combined changed-path inventory is 46 paths (34 tracked
  modifications and 12 new files), below the 69-path ceiling; the untracked files are only the
  expected attention source/tests, and no retained `/tmp` evidence is tracked.
- No unresolved issue or follow-up remains for CPU 0008H. CPU 0008I is the next separate Draft
  loss-family task and intentionally has no detailed task file.
