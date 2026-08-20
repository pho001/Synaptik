# Task 0007A1: Portable Ordinary Numerical Aggregate Reductions

## Status

Complete

## Goal

Implement portable generated CPU execution for exactly one fully static, resolved-layout current
Model `SUM`, `MEAN`, or `PROD` ordinary aggregate-reduction occurrence in full-reduction,
single-axis, and multi-axis forms. The result must have a direct generated hot path, an exact and
independently testable numerical target, truthful pre-assignment workspace requirements, immutable
prepared execution, invocation-private mutable state, deterministic output-cell partitioning, and
scalar/parallel bit identity.

## Scope

### Closed Model semantic inventory

- Admit `SUM` for `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, and `INT64`.
- Admit `MEAN` only for `FLOAT64`, `FLOAT32`, and `BFLOAT16`. Integral `MEAN` remains a Model
  validation failure and must not enter CPU lowering.
- Admit `PROD` for `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, and `INT64`.
- Admit exactly the current ordinary forms: full reduction with `NoOperationAttrs`, one-axis
  reduction with `AxisReductionAttrs`, and ordered multi-axis reduction with
  `MultiAxisReductionAttrs`. `SumToShapeAttrs` is not an ordinary form and remains entirely in CPU
  0007A2.
- Preserve the input element type as the output element type. The output Shape is the current Model
  inferred Shape: a full reduction is scalar; a selected axis is removed when `keepDim` is false
  and retained at extent one when true; a multi-axis reduction applies that rule to the distinct
  selected axes; an empty multi-axis list selects no axes and is a point-domain identity mapping.
- Consume Model-normalized axes. Negative axes normalize against input rank, out-of-range axes fail,
  and duplicate normalized multi-axes fail in Model validation. Attribute order is retained as Model
  metadata but does not define a sequential typed fold; CPU lowering canonicalizes an axis-membership
  mask for geometry and traverses each logical reduction domain in row-major input-coordinate order.
- A rank-zero input has one logical value. Full reduction and any legal point domain therefore have
  domain size one. A selected zero-extent axis gives domain size zero. An unselected zero-extent axis
  gives zero output cells, so execution has an empty output range and performs no writes.
- Reject an operation/type/form combination absent from this inventory before route selection or
  finalization. Preserve all current Model failures for wrong attribute class, wrong arity, invalid
  axis, duplicate axis, invalid `keepDim` use, wrong output type, or wrong output Shape.

### Exact numerical target

Floating `SUM` is the exact real sum of the selected finite inputs followed by one conversion to the
result format using round-to-nearest, ties-to-even. It is not a left fold in the result type.
Floating `MEAN` is that exact real sum divided by the exact positive integer domain size and then
converted once using round-to-nearest, ties-to-even; it must never round a `SUM` result before the
division. Floating `PROD` is the exact real product of the finite nonzero inputs followed by one
conversion using round-to-nearest, ties-to-even. Traversal order may not change any finite result.

For `SUM` and `MEAN`, use a fixed-width signed two's-complement superaccumulator in units of the
format's least positive subnormal:

| Type | Precision `p` | Unit exponent `emin` | Maximum finite exponent `emax` |
| --- | ---: | ---: | ---: |
| `FLOAT64` | 53 | -1074 | 1023 |
| `FLOAT32` | 24 | -149 | 127 |
| `BFLOAT16` | 8 | -133 | 127 |

For a reduction-domain cardinality `N`, let `valueBits = emax + 1 - emin`,
`signedBits = valueBits + ceilLog2(max(1, N)) + 1`, and
`sumLimbCount = ceil(signedBits / 64)`. Decode each finite value into its signed integer coefficient
of `2^emin` and add it to the little-endian `long` limb array with explicit carry and sign extension.
The chosen width must represent every exact sum of `N` finite values without overflow. Final
`SUM` conversion rounds `A * 2^emin`; final nonempty `MEAN` conversion divides the magnitude of `A`
by `N`, using quotient, remainder, guard, and sticky information to implement ties-to-even directly
for normal and subnormal outputs. Do not use `double`, `float`, `BigInteger`, or a result-format
intermediate as the generated accumulator.

For floating `PROD`, decode each finite nonzero operand into sign, an unsigned integer significand,
and a power-of-two exponent. XOR signs, sum exponents exactly in a signed `long` after analysis has
proved the maximum domain cannot overflow that sum, and multiply significands into a little-endian
unsigned limb array. Let `productLimbCount = max(1, ceil(p * max(1, N) / 64))`. That width represents
the exact product significand. Normalize once and use discarded bits as guard/sticky information for
one ties-to-even conversion, including normal/subnormal boundaries and overflow. Extract the existing
generation-time exact-product emission from `CpuScatterEmitter` into the focused package-private
`CpuExactProductEmitter`; both scatter product and ordinary product must emit direct limb operations
from that owner. The generated class must not call a runtime numerical helper.

Special values are classified before finite accumulation:

- Any floating NaN makes the output the canonical positive quiet NaN for schema 29:
  `0x7ff8000000000000L`, `0x7fc00000`, or BFLOAT16 bits `0x7fc0`. Input NaN payload, sign, and
  signaling state are deliberately not preserved because the Model contract leaves them unspecified.
- For `SUM` and nonempty `MEAN`, both positive and negative infinity produce canonical NaN; otherwise
  an infinity produces infinity of the observed sign. Finite exact overflow rounds to signed infinity.
- An empty floating `SUM` is positive zero. A nonempty exact-zero `SUM` or `MEAN` is negative zero
  only when every selected operand is negative zero; cancellation, any positive zero, or any nonzero
  finite contribution yields positive zero. Empty `MEAN` is canonical NaN.
- For `PROD`, NaN wins; any zero together with any infinity produces canonical NaN; otherwise an
  infinity produces signed infinity and a zero produces signed zero, with sign equal to the XOR of
  all operand sign bits. Empty floating `PROD` is positive one.
- Subnormal inputs participate exactly. A finite subnormal result and finite underflow-to-zero result
  use the same ties-to-even conversion and retain the exact result sign.

Integral `SUM` and `PROD` use same-width two's-complement modular arithmetic. Generated `INT32`
uses `iadd`/`imul`; generated `INT64` uses `ladd`/`lmul`. Overflow wraps modulo `2^32` or `2^64`.
Empty `SUM` is zero and empty `PROD` is one. No wider, floating, boxed, or saturating accumulator is
permitted.

### Geometry, parallelism, and failure timing

- Extend `CpuAggregateIr` and lowering with closed facts for family, represented type, form,
  `keepDim`, normalized reduced-axis membership, output geometry, domain cardinality, access form,
  carrier category, state shape, and workspace slice size. Facts used by generated code must be cold
  immutable data and part of structural identity where they change code or resource shape.
- Retain the existing typed dense integer-address form for eligible heap arrays and typed general
  long-address form for all legal layouts/carriers. General execution must cover nonzero offsets,
  positive/negative strides where currently legal, zero-stride reads, and arbitrary injective output
  layouts.
- Parallelize only by disjoint output-cell ranges. A worker owns every input visit and all state for
  each output cell in its half-open range. Never partition one reduction domain and never create a
  partial-result merge; this makes scalar and every worker count bit-identical.
- Accept arbitrary valid subranges, including empty ranges. An empty range performs no write and no
  state initialization. Repeated and concurrent invocations of one prepared executable must not
  share mutable accumulator state.
- Complete binding validation before the first output mutation: carrier compatibility, layout bounds,
  output injectivity, full physical input/output overlap rejection, scratch presence/size/alignment,
  and full physical scratch overlap rejection against every input and output. Preserve the current
  complete-overlap rule rather than checking only logical element intersections.
- Overflow while computing geometry, cardinality, workspace requirements, slice offsets, or an exact
  product exponent bound is a deterministic prepare failure, never wraparound or a late hot-path
  failure. A failed prepare publishes no executable; a failed invocation validation mutates no output.

### Resource and lifecycle contract

Integral rows declare no workspace. Floating rows declare one CPU-private run-owned workspace use,
workspace ID zero, alignment eight, with one disjoint slice for each simultaneously executable
output-cell worker range. The workspace is not materialization and must be represented by a new
closed `WorkspaceUse` value such as `AGGREGATE_EXACT_STATE`, not by overloading scatter or
materialization ownership.

For `SUM` and `MEAN`, one slice is `8 + 8 * sumLimbCount` bytes: the first eight bytes are a flags
word containing the observed NaN, positive-infinity, negative-infinity, positive-zero,
negative-zero, and nonzero-finite bits; limbs begin at byte eight. For `PROD`, one slice is
`24 + 8 * productLimbCount` bytes: flags/sign at byte zero, exact signed exponent at byte eight,
used-limb count at byte sixteen, and limbs at byte twenty-four. Reserved flag bits must be zero.
The sum/mean flag positions are respectively bits zero through five in that listed order; bits six
through 63 are zero. The product flag positions are bit zero NaN, bit one zero, bit two infinity,
and bit three accumulated negative-sign parity; bits four through 63 are zero. Sum/mean reset flags
and all limbs to zero. Product resets flags and exponent to zero, used-limb count to one, limb zero
to one, and every remaining limb to zero. Each product operand XORs its sign into bit three before
classifying its magnitude. Every output cell performs the applicable reset before reading its
domain; no state survives the cell or invocation.

Analysis computes the maximum simultaneously required slices from the selected worker policy and
proves `sliceBytes * sliceCount` with exact arithmetic against
`CpuPartitionAnalysisInputs.maximumAdditionalBytes`. The exact requirement exists before slot
assignment. Preparation assigns the slot; finalization only resolves the assigned segment and builds
the artifact. The immutable executable captures no Arena, writable state segment, mutable array, or
worker-local accumulator. Runtime owns allocation and teardown through the existing run lifecycle.
`sliceCount` is `min(selectedWorkerCount, outputCellCount)`; zero output cells require zero bytes,
no assigned scratch slot, and no generated call. A positive output count requires at least one slice.
Scalar execution uses the same scratch-bearing floating entry shape and exact algorithm rather than
a second numerical implementation. Before each generated call, the executable passes that worker's
already assigned disjoint slice as the `scratch` argument, so the entry needs no worker identifier
and cannot address another worker's state.

### Generated artifact and cache contract

Generated code is direct from the first implementation. Specialize family, represented type,
full/single-/multi-axis form, dense/general access, carrier, state layout, and loop/code shape at
generation time. The JVM entries are exactly:

- integral: `(inputCarrier, outputCarrier, long[] geometry, long start, long end)void`;
- floating: `(inputCarrier, outputCarrier, MemorySegment scratch, long[] geometry, long start,
  long end)void`.

In descriptor grammar those entries are `(<I><O>[JJJ)V` and
`(<I><O>Ljava/lang/foreign/MemorySegment;[JJJ)V`. For the represented type, `<I>` and `<O>` are
independently the selected `[D`, `[F`, `[S`, `[I`, or `[J` primitive-array descriptor, or
`Ljava/lang/foreign/MemorySegment;`; only the matching represented-type array descriptor is legal.
This includes the existing mixed array/segment bindings and never admits `Object`. Geometry is
immutable prepared metadata. There is no operation,
graph, enum-family, represented-type, carrier, or state-kind dispatch in the generated hot path; no
boxing, reflection, map/string lookup, per-cell/per-element allocation, virtual numerical strategy,
or avoidable helper call.

Schema advances exactly from 28 to 29. Extend `CpuKernelSpecialization` and lowering fingerprints
with every code-shape/resource fact. Shape-polymorphic structural identity may omit cold invocation
facts only when bytecode, descriptor, resource size, and semantic result do not change; domain
cardinality and limb count are structural for floating rows. Persistent schema-28 or malformed
artifacts are safe misses and are never migrated or partially reused. Stable Class-File API tests
assert exact descriptors, typed carrier bytecodes, direct limb bodies, allowed member references,
and forbidden bridges/dispatch/allocation.

The closed member-reference allowlist for new aggregate artifacts is primitive carrier access and raw
bit conversion, `Long.numberOfLeadingZeros`, `Long.compareUnsigned`, `Long.divideUnsigned`,
`Long.remainderUnsigned`, `Math.unsignedMultiplyHigh`, and typed `MemorySegment.get`/`set` with the
already approved layouts for general carriers. A required reference outside this list is a task-scope
stop requiring explicit review. Ordinary aggregate artifacts must not reference Model, Compiler,
Runtime dispatch, operation attributes, collection/boxing APIs, `BigInteger`, or reference kernels.

### Independent correctness and performance evidence

Build test-only numerical oracles that do not reuse either exact emitter or generated rounding
helpers. The oracle uses `BigInteger` for exact signed sums and products and an independently written
integer/rational ties-to-even conversion. It compares raw result bits, with the canonical schema-29
NaN policy above.

Cover every family/type/form/carrier category and both dense/general code shapes. Required adversarial
data includes catastrophic cancellation, halfway ties on both parity outcomes, finite overflow,
normal/subnormal boundaries, underflow to signed zero, smallest subnormals, mixed infinities, every
zero sign combination, NaN sign/payload/signaling classes, `INT32`/`INT64` modular wrap, empty
domains, scalar inputs, empty multi-axis lists, every axis position, unsorted multi-axis lists,
keep-dim true/false, zero selected/unselected extents, offsets, positive/negative strides,
zero-stride reads, partial/empty ranges, repeated and concurrent prepared-executable reuse, worker
counts, cache hits/misses/corruption, schema 28 safe miss, and resource/lifecycle declarations.

Performance is a release gate, not a tuning input. Use dense heap-array Shape `[128, 2048]`, reduce
axis one without keeping the dimension, and test these 13 materially distinct paths: floating
`SUM`, `MEAN`, and `PROD` for each of `FLOAT64`, `FLOAT32`, and `BFLOAT16`; integral `SUM` and
`PROD` for `INT32` and `INT64`. The direct primitive-Java baseline must implement the same exact
limb algorithm, state reset, input decoding, final rounding, and the same preallocated
`MemorySegment` scratch layout/load/store work without invoking generated or emitter helpers. Verify
raw bits before timing. Run five isolated forks with five warmup
rounds and nine measured rounds per fork, adapting batch count until each sample is at least 25 ms,
with `-Xms1g -Xmx1g`. Record every fork median, the median of fork medians, JVM/OS/CPU metadata,
class-file evidence, and commands under `/private/tmp`. Every fork and the aggregate generated/direct
ratio for every case must be `<= 1.15`. Benchmark failures may diagnose or reject the implementation
but may not mutate production code automatically. General layouts and carriers are deliberately
unmeasured and instead require the structural and semantic gates above.

## Out of scope

- `SUM_TO_SHAPE`/`SumToShapeAttrs`, owned by CPU 0007A2.
- Arg extrema, masked reductions, logarithmic/statistical/norm reductions, softmax or normalization.
- Fusion, native/OpenBLAS routes, autotuning, runtime measurement, or dynamic unresolved geometry.
- New Model operations, attributes, public/shared numerical APIs, or changes to Model semantics.
- Broader backend-conformance or integration closure beyond a concrete failure exposed by this task.
- Parallel partitioning or merging within one reduction domain.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current-architecture-plan.md`](../../../../architecture/current-architecture-plan.md)
- [`performance-evidence-and-tuning.md`](../../../../architecture/performance-evidence-and-tuning.md)
- [`runtime-prepare-backend-boundary.md`](../../../../architecture/runtime-prepare-backend-boundary.md)
- [`planning-guide.md`](../../../planning-guide.md)
- [`documentation-rules.md`](../../../../developer-guide/documentation-rules.md)
- [`general documentation profile`](../../../../developer-guide/documentation/general-style.md)
- [`planning documentation profile`](../../../../developer-guide/documentation/planning-style.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0007A`](0007a-portable-ordinary-extrema-and-boolean-reductions.md)
- [`CPU 0007A0F`](0007a0f-random-and-dropout-generated-loop-parity.md)
- [`Model master plan`](../../../modules/model/master-plan.md)

## Architecture constraints

- Model owns operation meaning, validation, type inference, and Shape inference. CPU consumes those
  contracts and adds only CPU-private lowering, numerical representation, resource, and route facts.
- Analysis completes route selection, structural identity, geometry, exact workspace requirements,
  and specialization facts before slot assignment. Finalization alone constructs generated artifacts.
- Prepared execution is immutable. Runtime owns invocation storage and worker lifetime; CPU-private
  scratch is run-owned and never cached with an executable.
- Hot execution performs no graph/`Operation` inspection, runtime route selection, tuning, or
  unresolved semantic dispatch.
- Preserve current module dependencies and public APIs. If implementation proves a shared/public or
  architectural change unavoidable, stop and request clarification before editing architecture.

## Package impact

- `io.github.pho001.synaptik.backend.cpu`: capability advertisement and public package Javadoc only.
- `internal.ir`: closed numerical/state/resource facts on aggregate IR.
- `internal.lowering`: Model-form validation, normalized geometry, exact bounds, route eligibility.
- `internal.codegen.emit`: direct aggregate bodies and focused generation-time exact sum/product
  emitters; scatter product is migrated to the shared private product emitter without semantic change.
- `internal.cache`: schema 29 specialization and persistence compatibility.
- `internal.prepare` and `internal.route.portable`: exact pre-assignment requirements, slot resolution,
  and final artifact construction.
- `internal.executable`: validated scratch binding and output-cell worker ranges only.
- `internal.reference`: scalar semantic reference coverage, not a generated execution dependency.
- No new package, public facade, generic utility, or service/manager abstraction.
- Type placement: package-private `CpuExactSumEmitter` and `CpuExactProductEmitter` belong in
  `internal.codegen.emit` because they construct direct numerical bytecode and are neither runtime
  numerical services nor shared semantic APIs.

## Affected files

The implementation allowlist is the following 45 paths. A path is edited only if its current owner
requires the change; unused allowance is not permission for unrelated edits.

Production and package contracts (maximum 26):

1. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
2. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
3. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAggregateIr.java`
4. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
5. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAggregateLowering.java`
6. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
7. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
8. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateEmitter.java`
9. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuExactSumEmitter.java` (new)
10. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuExactProductEmitter.java` (new)
11. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterEmitter.java`
12. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
13. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
14. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
15. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
16. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`
17. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
18. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`
19. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`
20. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
21. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
22. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
23. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
24. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
25. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
26. `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`

Focused tests (maximum 14):

27. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
28. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
29. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAggregateIrTest.java`
30. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAggregateLoweringTest.java`
31. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateGeneratedKernelTest.java`
32. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterGeneratedKernelTest.java`
33. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
34. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
35. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
36. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
37. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
38. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
39. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`
40. `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuBufferBindingTest.java`

Documentation and planning (maximum 5):

41. `docs/backend-guide/cpu-backend.md`
42. `docs/glossary.md`
43. `docs/planning/backends/cpu/tasks/0007a1-portable-ordinary-numerical-aggregate-reductions.md`
44. `docs/planning/backends/cpu/master-plan.md`
45. `docs/planning/roadmap.md`

## Maximum scope

- Maximum changed or new paths: 45, including this task, master plan, and roadmap.
- Maximum new production types: two package-private generation-time emitters named above.
- Maximum generator schema transition: 28 to 29 exactly.
- Stop before a 46th path, a third new production type, a public/shared API, a module/dependency
  change, or a schema value other than 29; revise this task explicitly before proceeding.
- Architecture, backend-conformance, integration, other backend, Model, Compiler, Runtime, and build
  files are no-change decisions unless a concrete failing contract proves the task cannot be correct.
- `CpuGeneratedKernelArtifactStore` production code is inspection-only because its existing generic
  compatibility checks own schema safe misses; the focused store test proves schema-29 behavior.

## Acceptance criteria

- Every admitted operation/type/form produces Model-conforming raw result bits for all specified
  shapes, layouts, carriers, ranges, and exceptional values; every unadmitted combination fails cold.
- Exact finite `SUM`/`MEAN` and `PROD` follow the defined limb algorithms and one-rounding target;
  integral operations wrap at their declared width; empty-domain and zero-sign results are exact.
- Scalar and parallel execution are bit-identical because workers partition output cells only.
- Resource requirements, alignment, ownership, and maximum bytes are exact before slot assignment;
  preparation/finalization/run ownership and failure timing match the lifecycle contract.
- Generated artifacts have the exact typed descriptors and direct code shape, pass member-reference
  allowlists, and contain none of the forbidden bridges, dispatch, allocation, or helper dependencies.
- Schema 29 identity differentiates all structural facts and treats schema 28/corrupt artifacts as
  safe misses.
- Input/output/scratch validation completes before mutation and rejects all full physical overlap,
  carrier/layout/bounds/injectivity violations.
- All focused semantic, structural, cache, lifecycle, and performance gates pass. Every one of the
  13 performance cases passes every fork and aggregate at `<= 1.15`.
- A distinct clean documentation-focused agent finalizes affected Javadoc, package summaries, CPU
  guide, and glossary review after implementation. It records reasoned no-change conclusions where
  current prose remains exact and does not rerun successful Java suites without a concrete reason.
- The final implementation stays within the 45-path allowlist and reports exact changed paths.

## Tests / validation

Use the planning guide's affected-module tier. Do not substitute a reference-kernel comparison for
the independent numerical oracle.

1. Before production edits, record HEAD, clean/dirty state, schema 28, current fail-closed 0007A1
   capability result, existing aggregate and scatter-product generated descriptors/bytecode/member
   references, and five-fork direct-primitive timings for all 13 cases under one fresh
   `/private/tmp` directory. The capability has no pre-change generated timing; do not fabricate one.
2. Add focused unit/property tests for Model inventory rejection, normalized axis geometry, domain
   cardinality, state-size formulas, overflow rejection, all forms/types, identities, and raw bits.
3. Add independent BigInteger/rational oracle comparisons for all adversarial categories in Scope.
4. Add dense/general carrier/layout tests, arbitrary and empty ranges, overlap/injectivity failures,
   scalar/parallel bit identity, repeat/concurrent reuse, and pre-mutation failure assertions.
5. Add Class-File API structural tests for descriptors, opcodes, member-reference allowlists,
   forbidden patterns, and schema-29 specialization. Preserve scatter product semantics after emitter
   extraction.
6. Add persistence tests for schema-29 round trip and hit, schema-28 safe miss, corrupt/truncated safe
   miss, and structural-versus-cold identity facts.
7. Run focused CPU tests, then `./gradlew :backends:cpu:test`, CPU Javadoc generation, and the existing
   architecture test task only if the implementation touches a dependency/boundary contrary to the
   expected no-change decision. Do not run unrelated repository-wide tests absent a concrete trigger.
8. Run the fixed 13-case paired generated/direct five-fork performance protocol and exact Class-File
   inspection after implementation, using a temporary probe outside the repository; preserve raw
   logs and summarized per-fork/aggregate ratios under `/private/tmp`.
9. Run local Markdown links/anchors/fences, Javadoc links, glossary anchors, line endings, exact path
   count, schema/status, no-later-spec, `git diff --check`, and clean-index checks.

## Dependencies

- CPU 0007A is Complete and owns ordinary aggregate geometry and output-cell parallelism.
- CPU 0007A0 through 0007A0F are Complete; schema 28 and direct generated-family parity are the
  mandatory baseline.
- Model aggregate operations, attributes, validation, Shape/type inference, Tensor facade, semantic
  tests, and Javadoc are current authoritative semantics and are not modified by this task.
- CPU 0006B1 supplies exact product limb-generation precedent; extraction must preserve its completed
  scatter behavior.

## Follow-up tasks

- CPU 0007A2: binding-aware `SUM_TO_SHAPE` only.
- CPU 0007B–0007F: arg extrema, masked, advanced numerical, softmax, and normalization families.
- CPU 0009: broader portable generated-coverage closure and conformance checkpoint.

## Architecture impact

Expected impact: None.

No architecture, module-boundary, dependency-direction, or public-API change is planned. The task
extends existing CPU-private IR, lowering, generation, cache, prepare/finalize, execution, and
run-owned workspace mechanisms. A discovered need to alter an authoritative architecture rule is a
stop condition, not implied scope.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Work in a clean implementation context. Do not use GSD. Do not stage, commit, push, reset, stash,
clean, or alter unrelated work. Read repository instructions, all architecture references,
this task, its dependencies, current Model aggregate contracts/tests/Javadoc, and every affected CPU
owner before editing. Verify baseline HEAD/schema/worktree and preserve unrelated changes. First
capture baseline Class-File and 13-case performance evidence. Then implement exactly the closed
inventory and numerical/resource algorithms above: CPU-private analysis facts; exact requirements
before slot assignment; schema-29 typed specialization; direct generated dense-int and general-long
bodies; exact run-owned scratch; output-cell-only parallelism; validation before mutation; and
safe-miss persistence. Reuse the exact-product emitter only by extracting a focused generation-time
owner and proving existing scatter output/structure remains unchanged. Add independent BigInteger and
rational oracles that share no emitter rounding logic. Do not add `SUM_TO_SHAPE`, public APIs,
runtime dispatch, tuning, helpers in generated execution, or unrelated refactors. Run the specified
semantic, lifecycle, Class-File, cache, module, documentation, and five-fork performance validations.
After Java work, hand the same change to a distinct clean documentation-focused context to finalize
Javadoc/package summaries, `docs/backend-guide/cpu-backend.md`, and glossary impact. Mark this task
Complete only
after all gates pass and record exact evidence, changed paths, no-change decisions, context IDs, and
unresolved issues.
```

## Local decisions

- “Exact real” is executable here: fixed-width integer state in least-subnormal units for sums and an
  exact integer significand plus exponent for products, followed by one specified conversion.
- `MEAN` rounds the exact rational directly; it does not compose a rounded `SUM` and division.
- Output-cell-only partitioning is chosen because splitting a domain is unnecessary and would require
  a second combination/resource contract.
- Floating rows always use declared run-owned scratch, including scalar execution. Integral rows use
  primitive locals and no scratch.
- Domain cardinality and limb count are structural identity because they change scratch and emitted
  code; offsets, bound carriers, worker count, and invocation ranges remain cold invocation facts when
  they do not change those structures.
- Canonical positive quiet NaN makes Model-unspecified payload/sign behavior deterministic and
  testable in schema 29.
- The existing scatter exact-product body is extracted rather than duplicated; the new owner remains
  generation-time and package-private.
- No architecture/conformance/integration change is expected. The documentation pass must record
  explicit no-change conclusions for Model, Compiler, Runtime, other backends, architecture tests,
  backend conformance, integration tests, and build configuration.

## Known limitations

- The portable target deliberately does not preserve NaN payload, sign, or signaling state.
- One reduction domain is serial even under parallel execution; only output cells are parallelized.
- General layouts/carriers receive semantic and structural gates, not a comparative performance gate.
- No claim is made for numerical families or forms listed as out of scope.

## Validation evidence

Planning baseline: empty index at commit
`9db202a05a939d3dd5f0b5d8877e18fa84f4557f`; CPU 0007A0F Complete;
`CpuGeneratorSchema.CURRENT_VERSION` 28; CPU 0007A1 the sole Ready frontier; no later detailed CPU
task specification. The pre-existing CPU diff contained exactly this task, the CPU master plan, and
the roadmap. Concurrent non-CPU work remained excluded from every CPU scope check and was not
modified. Implementation context `01a019c6-6f14-7a70-8d09-ceb7f3c1cf3c` used the single fresh
evidence directory `/private/tmp/synaptik-cpu-0007a1-implementation.YeByo9`.
The concurrent workstream advanced repository HEAD to
`c7a7d4b6fb5c691352c7ebd5c0b3726a4bd1c923` after baseline capture; the CPU diff remained
uncommitted and isolated, and the index remained empty.

Pre-edit capability checks confirmed SUM/MEAN/PROD failed closed and schema 28 remained current.
Pre-edit Class-File evidence recorded:

| Artifact | Descriptor | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| aggregate MIN FLOAT64 dense | `([D[D[JJJ)V` | 527 | `38426da63747b23f07e9d50b60085f57f9a8ece8afae0547880568952cd54489` |
| aggregate MAX FLOAT32 general segment/array | `(MemorySegment;[F[JJJ)V` | 1,061 | `d0a49f4295dd696518f3d7897d18dd7a781b22bb60095246680b2c23e0d20c43` |
| scatter FLOAT64 product dense | `([D[I[D[DLjava/lang/foreign/MemorySegment;[JJJ)V` | 4,873 | `c84bb41485a1a762032d3a6645fb443d13b6460bc97c3aa80162b55aa23d8c56` |

The extrema artifacts referenced only their typed carrier/layout access and established
`CpuAggregateEmitter` selector; the scatter artifact referenced its prior direct exact-product
members. The standalone corrected direct-only source SHA-256 is
`a83ddc15bd523cf21f82533320bb6cc275d98befc6831393c02fa01ae4eb4f14`. Its truthful five-fork raw
log is `baseline-direct-final-raw.txt` with SHA-256
`3d46e4cac40b7decee120ee1a5c5ba0152a1b300f1efe751d1bdc83204ef8293`; the summary is
`baseline-direct-final-summary.txt` with SHA-256
`dac7575e14f069e74bbf9aa225a34c5ef7534718330ab9ad9548ca18da840c2b`. Median-of-fork-median
direct nanoseconds were: F64 SUM 6,014,385, MEAN 6,016,698, PROD 69,779,041; F32 SUM 1,441,635,
MEAN 1,452,986, PROD 33,078,916; BF16 SUM 1,714,802, MEAN 1,716,390, PROD 11,357,531; I32 SUM
24,112, PROD 36,153; I64 SUM 59,047, PROD 204,313. No unavailable pre-change generated timing is
claimed.

Final schema-29 Class-File evidence is under `final-generated-classes/`:

| Artifact | Descriptor | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| aggregate SUM FLOAT64 dense | `([D[DLjava/lang/foreign/MemorySegment;[JJJ)V` | 4,767 | `8506701439f0ec9cf6284a19e4a15e9db251087b0622adeda0078f87cc663abe` |
| aggregate MEAN FLOAT32 general segment/array | `(Ljava/lang/foreign/MemorySegment;[FLjava/lang/foreign/MemorySegment;[JJJ)V` | 5,811 | `b212f4ace35b8ae8abe5d211078e0e047d21e22a92d1c37eead7fde5741d5619` |
| aggregate PROD BFLOAT16 dense | `([S[SLjava/lang/foreign/MemorySegment;[JJJ)V` | 3,576 | `bcb4adfcb47c0d95f3f04c07c464df649597f4dc18e0d5229657e61295a5ce5a` |
| aggregate SUM INT32 dense | `([I[I[JJJ)V` | 389 | `2f17fde6825464d3848f893716e78079bf8a68c3c346089686b6c8342849c9f8` |
| aggregate PROD INT64 dense | `([J[J[JJJ)V` | 389 | `58c1dc76f46e62feeaf1b797fbf1d08f9de1bfbc030764f09937d100aa6ade4e` |
| extracted scatter FLOAT64 product dense | `([D[I[D[DLjava/lang/foreign/MemorySegment;[JJJ)V` | 4,896 | `d2dc5ed0ea700c6191ac24e4efe088784aa5daf824ad2697285748d6b3469ae3` |

The floating SUM/MEAN artifacts reference raw-bit conversion, `Long.compareUnsigned`,
`Long.numberOfLeadingZeros`, and typed state `MemorySegment.get`/`set`; the general artifact also
uses the established native-order FLOAT32 layout. PROD references the same typed state access plus
`Math.unsignedMultiplyHigh`. Integral artifacts have no member references. No generated artifact
references Model, Compiler, Runtime dispatch, collections, boxing, `BigInteger`, either new emitter,
or the scalar reference kernel. Full `javap -v`, `javap -c`, and extracted member-reference files
are adjacent to each class.

The final fixed paired probe source SHA-256 is
`c8b07320fc4e139493b2cb28310ddbcec4ad08cd177623791b216ddfd509af80`. Its raw five-fork log is
`final-paired-five-fork-raw.txt` with SHA-256
`eb4039617c5ebb5ee0986cd9d91ee5f5ee1aa696e305dbf0a5281eeb34ec0740`; summarized evidence is
`final-paired-summary.txt` with SHA-256
`b114ea1c3e3d0246c2f9e371b2c5246e879264502749c630171f306d151d65e4`.
Every per-fork and aggregate generated/direct ratio passed `<= 1.15`:

| Case | Fork 1 | Fork 2 | Fork 3 | Fork 4 | Fork 5 | Aggregate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F64 SUM | 0.733076982 | 0.729707281 | 0.733664360 | 0.735769322 | 0.729714387 | 0.731536337 |
| F64 MEAN | 0.764139147 | 0.763373946 | 0.762625229 | 0.763030907 | 0.763291586 | 0.762625229 |
| F64 PROD | 0.996687930 | 0.990920086 | 0.998487437 | 0.994790719 | 0.999317668 | 0.996687930 |
| F32 SUM | 0.627082516 | 0.603092784 | 0.626362853 | 0.598768343 | 0.583081609 | 0.602473414 |
| F32 MEAN | 0.643513268 | 0.637826364 | 0.646489683 | 0.613353423 | 0.585755520 | 0.617472831 |
| F32 PROD | 0.970634674 | 0.955702580 | 0.968927088 | 0.970842857 | 0.957671427 | 0.967382923 |
| BF16 SUM | 0.989706762 | 0.853140306 | 0.995825062 | 0.854998055 | 0.895019623 | 0.897626881 |
| BF16 MEAN | 0.995445036 | 0.880129506 | 0.988301652 | 0.701411093 | 0.955292747 | 0.904921513 |
| BF16 PROD | 0.960915394 | 0.947674726 | 0.954988312 | 0.963407443 | 0.956230520 | 0.957044954 |
| I32 SUM | 1.005484679 | 1.005955236 | 1.013971736 | 1.020012080 | 1.014359015 | 1.012310152 |
| I32 PROD | 1.039683406 | 1.034239650 | 1.026291712 | 1.039888204 | 1.035204151 | 1.039683406 |
| I64 SUM | 1.022757200 | 1.022036117 | 1.003376320 | 1.021563432 | 1.008419839 | 1.008976482 |
| I64 PROD | 1.023785433 | 1.007803203 | 1.020068380 | 1.050061417 | 1.019858191 | 1.020068380 |

The environment was OpenJDK 26.0.1+8-34 with `-Xms1g -Xmx1g`, macOS 26.5.2/Darwin 25.5.0
ARM64, Apple M3 Max, 16 cores (12 performance and 4 efficiency), and 64 GB RAM.

The authoritative validation commands were:

```text
javac -d /private/tmp/synaptik-cpu-0007a1-implementation.YeByo9/classes /private/tmp/synaptik-cpu-0007a1-implementation.YeByo9/AggregateDirectBaseline.java
for fork in 1 2 3 4 5; do java --add-modules jdk.incubator.vector -Xms1g -Xmx1g -cp /private/tmp/synaptik-cpu-0007a1-implementation.YeByo9/classes AggregateDirectBaseline; done
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest --tests io.github.pho001.synaptik.backend.cpu.CpuInternalPackageInventoryTest --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest
./gradlew :backends:cpu:test
./gradlew :backends:cpu:javadoc
for fork in 1 2 3 4 5; do java --add-modules jdk.incubator.vector -Xms1g -Xmx1g -cp /private/tmp/synaptik-cpu-0007a1-implementation.YeByo9/paired-classes:backends/cpu/build/classes/java/test:backends/cpu/build/classes/java/main:modules/model/build/classes/java/main:modules/backend-contract/build/classes/java/main:modules/config/build/classes/java/main:modules/planning/build/classes/java/main:modules/prepare/build/classes/java/main:modules/compiler/build/classes/java/main:modules/runtime/build/classes/java/main:modules/trace/build/classes/java/main AggregatePairedBenchmark; done
for classfile in /private/tmp/synaptik-cpu-0007a1-implementation.YeByo9/final-generated-classes/*.class; do javap -v -p "$classfile"; javap -c -p "$classfile"; sha256sum "$classfile"; done
git diff --check
git diff --cached --check
git status --short
```

The paired Java command was executed in five isolated JVMs.

Focused aggregate, lowering, preparation, executable, cache, capability, package-inventory, and
scatter tests passed during implementation. The single authoritative unfiltered command
`./gradlew :backends:cpu:test` passed (`BUILD SUCCESSFUL`, 21 actionable tasks, one executed and 20
up-to-date). Final `./gradlew :backends:cpu:javadoc` passed with only the two expected incubating
Vector API warnings. No dependency or architecture boundary changed, so architecture tests were
not triggered. Final scope, links/fences/line endings/status, whitespace, and empty-index checks are
recorded in the implementation handoff below.

The clean independent review/fix inspection context
`01a019ff-7d7b-73f1-8f87-b4963db097d6`, coordinated by
`01a019fe-c0e8-79e0-8c7f-c697e77c1445`, found no production defect. It added two focused
regressions only: raw-bit oracle comparison for arbitrary domain counts `0, 1, 2, 3, 5, 7, 9,
17` across FLOAT64/FLOAT32/BFLOAT16 SUM/MEAN/PROD, and genuinely concurrent reuse of one immutable
prepared executable with distinct `RunState` and run-owned workspace instances. Its six-class
focused matrix and both new regressions passed; the final aggregate generated-kernel class passed
13 of 13 tests. No executable Java changed after that review evidence.

Documentation context `01a01a09-3e83-7993-bb92-78931ce5349f` ran
`./gradlew :backends:cpu:javadoc` after final package-summary edits: `BUILD SUCCESSFUL`, 11
actionable tasks, two executed and nine up-to-date, with only the two expected incubating Vector
API warnings. Manual inspection covered the rendered provider, schema, aggregate IR/emitter,
exact-sum/product emitter, preparation-plan, executable, and nine affected package-summary pages;
their prose, parameter/return/throws sections, links, and markup were complete and showed no
implementation leakage. Targeted local Markdown link/anchor and fence checks passed for the five
changed Markdown documents; terminology, examples, UTF-8/line endings, and schema/status wording
were inspected manually. The final allowlist audit found exactly 35 CPU 0007A1 paths out of the
permitted 45, exactly two new production Java types, schema 29, no later detailed CPU task, and
Draft CPU 0007A2–0007F. `git diff --check` and `git diff --cached --check` passed, and the index was
empty. Concurrent NN paths, including its portions of the shared roadmap and glossary, remained
outside CPU validation and were neither edited nor reverted by this pass.

## Implementation notes

- Advertised exactly floating SUM/MEAN/PROD and modular integral SUM/PROD while keeping integral
  MEAN and all later reduction families fail closed.
- Added schema-29 aggregate IR/lowering facts for domain cardinality, exact limb count, scratch
  slice bytes, and resource policy. State sizes use checked formulas and structural identity embeds
  all generated-byte/resource facts.
- Added direct dense integer-address and typed general long-address generated bodies. Floating
  SUM/MEAN use a fixed-width signed superaccumulator and direct rational ties-to-even conversion;
  floating PROD uses exact significand limbs plus checked exponent and one final conversion.
  Integral rows use primitive modular locals without scratch.
- Extracted scatter's exact-product Class-File emission into package-private
  `CpuExactProductEmitter`, preserved scatter semantics, and added package-private
  `CpuExactSumEmitter`. Both are generation-time owners only.
- Declared one exact run-owned aligned workspace before slot assignment, bound disjoint per-range
  slices, validated size/alignment/access/physical overlap before mutation, reset each cell, and
  preserved immutable prepared-executable reuse and output-cell-only parallelism.
- Added a `BigInteger`/rational scalar oracle independent of generated helpers and expanded tests for
  inventory, forms, types, dense/general carriers, canonical NaN/signed zero, cancellation,
  ties-to-even parity, overflow, normal/subnormal boundaries, signed underflow, modular wrap,
  empty/partial ranges, parallel slices, repeated reuse, persistence, schema-28 safe misses, and
  Class-File descriptors/member ownership.
- Performance tuning embedded the structural limb width into generated bytes and retained aggregate
  product headers in primitive locals across each cell. This is static code-shape specialization,
  not runtime measurement or tuning.
- The implementation/review handoff occupied 24 allowlisted paths. Documentation finalization adds
  the CPU guide, glossary, and nine affected package summaries for a final 35 allowlisted paths.
  The change adds exactly the two permitted production types and advances schema exactly from 28 to
  29. `SUM_TO_SHAPE` and every later family remain out of scope.

## Completion summary

Executable Java, focused/authoritative tests, final Javadoc generation, Class-File inspection, the
fixed five-fork performance gate, independent code review, and clean documentation finalization
are complete. Implementation context: `01a019c6-6f14-7a70-8d09-ceb7f3c1cf3c`; independent review
context: `01a019ff-7d7b-73f1-8f87-b4963db097d6`; documentation context:
`01a01a09-3e83-7993-bb92-78931ce5349f`. Changed/new CPU implementation paths are:

- production: `CpuCapabilityProvider`, `CpuGeneratorSchema`, `CpuAggregateEmitter`, new
  `CpuExactProductEmitter`, new `CpuExactSumEmitter`, `CpuScatterEmitter`,
  `CpuPreparedExecutable`, `CpuAggregateIr`, `CpuAggregateLowering`,
  `CpuPartitionPreparationPlan`, `CpuPartitionPreparer`, and `CpuScalarReferenceKernel`;
- tests: `CpuCapabilityProviderTest`, `CpuInternalPackageInventoryTest`,
  `CpuGeneratedKernelArtifactStoreTest`, `CpuKernelSpecializationTest`,
  `CpuAggregateGeneratedKernelTest`, `CpuPreparedExecutableTest`, `CpuAggregateLoweringTest`,
  `CpuPartitionFinalizerTest`, and `CpuPartitionPreparerTest`;
- documentation/Javadoc/planning: nine affected CPU package summaries,
  `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task, the CPU master plan, and the
  roadmap.

The documentation pass independently reconciled every changed production contract and focused test
with the rendered Javadocs, CPU guide, glossary, and planning records. Existing changed type/member
Javadocs already meaningfully cover the new aggregate inventory, numerical/state invariants,
lifecycle and workspace ownership, nullability, parameters, returns, and failure conditions. Nine
affected package summaries were stale and now describe schema-29 numerical/state/resource behavior.
The CPU guide now documents the complete current numerical
matrix, forms, exact-real/one-rounding target, special values, modular arithmetic, direct generated
execution, run-owned disjoint workspace, deterministic bit identity, schema-29 compatibility, and
fail-closed exclusions. The glossary needed no new reusable term, but its existing CPU portable
route and specialization entries were refined because their implementation-status and schema
statements were stale.

No Model, Compiler, Runtime, other-backend, public API guide, architecture-contract/current-
architecture/ADR, architecture-test, backend-conformance, integration-test, or build-configuration
change is correct outside the affected CPU package summaries. The capability adds no Model meaning
or public API; compiler
and Runtime contracts are consumed unchanged; CPU-private prepare/code generation changes neither
module ownership nor dependency direction; no other backend is affected; no shared conformance or
end-to-end surface was introduced; Java 26 and the existing CPU Javadoc configuration already cover
the two package-private emitters; and the updated CPU package summaries remain within their existing
responsibility boundaries. CPU 0007A2 and every later CPU task remain Draft, and no later detailed CPU
task specification exists.

- Completed changes: schema-29 ordinary numerical aggregates and their final explanatory,
  glossary, Javadoc-review, and planning evidence.
- Files changed or created by the documentation pass: nine affected CPU `package-info.java` files,
  `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task,
  `docs/planning/backends/cpu/master-plan.md`, and the CPU-only statements in
  `docs/planning/roadmap.md`. The final CPU 0007A1 change contains exactly 35 allowlisted paths.
- Tests and validation: reused the implementation and independent-review Java/Class-File/
  performance evidence above because executable Java did not change; final documentation and
  Javadoc commands are recorded above.
- Documentation-agent review: complete in clean context
  `01a01a09-3e83-7993-bb92-78931ce5349f` using the general, backend-guide, API/Javadoc, planning,
  and example profiles.
- Documentation impact: CPU guide, glossary status text, and synchronized planning evidence updated.
- Javadoc review: all changed production types/members and relevant package summaries reviewed;
  type/member prose required no edit, while nine package summaries were finalized and rendered.
- Glossary impact: existing CPU route/specialization entries refined; no new term added because
  exact accumulation, workspace, aggregate reduction, and generated-kernel lifecycle already have
  established vocabulary.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
