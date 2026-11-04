# Task 0008M: [Vector](../../../../glossary.md#term-vector) [MSE](../../../../glossary.md#mean-squared-error--mse) `NONE`

## Status

Complete

## Goal

Add Java 26 [Vector API](../../../../glossary.md#term-vector-api) and parallel-[vector](../../../../glossary.md#term-vector) execution for the existing portable [CPU](../../../../glossary.md#term-cpu)
[mean](../../../../glossary.md#term-mean)-squared-error ([MSE](../../../../glossary.md#mean-squared-error--mse)) `NONE` form when [prediction](../../../../glossary.md#term-prediction), [target](../../../../glossary.md#term-target), and result are same-typed
[`FLOAT32`](../../../../glossary.md#term-float32) or same-typed [`FLOAT64`](../../../../glossary.md#term-float64) [contiguous](../../../../glossary.md#term-contiguous) values. Each full vector computes the existing
per-[element](../../../../glossary.md#term-element) formula and the current [scalar](../../../../glossary.md#term-scalar) body owns every remainder:

```text
difference = prediction - target
loss = difference * difference
```

The change extends only the compute strategy for an already implemented first-class [loss](../../../../glossary.md#loss).
`SUM` and `MEAN` remain scalar because horizontal vector reduction would change the current CPU
loss body's defined increasing-order, left-associated accumulation. Categorical loss, [BFLOAT16](../../../../glossary.md#term-bfloat16),
mixed floating promotion, and non-contiguous MSE remain on their existing scalar paths.

## [Scope](../../../../glossary.md#term-scope)

### Exact semantic and eligibility boundary

Admit [vector](../../../../glossary.md#term-vector) compute only when all of these facts are true:

- the [occurrence](../../../../glossary.md#term-occurrence) is exactly `LossKind.MEAN_SQUARED_ERROR` with
  `MeanSquaredErrorAttrs(LossReduction.NONE)`;
- [prediction](../../../../glossary.md#term-prediction), [target](../../../../glossary.md#term-target), and output are all [`FLOAT32`](../../../../glossary.md#term-float32), or all are [`FLOAT64`](../../../../glossary.md#term-float64);
- Shapes are fully static and positionally exact, the output has the same Shape, and every
  prediction, target, and output access is dense row-major contiguous with a non-negative cold
  element offset;<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- the output is injective and every element count, referenced span, byte offset, range, and
  vector-address calculation is representable under the existing lowering/binding checks;<span data-glossary-navigation="true"> ([injective glossary](../../../../glossary.md#term-injective); [referenced span glossary](../../../../glossary.md#referenced-element-span); [lowering glossary](../../../../glossary.md#lowering); [binding glossary](../../../../glossary.md#term-binding))</span>
- the Java 26 preferred species for the one lane type has more than one lane and the complete
  logical element count contains at least one full species;
- [CPU](../../../../glossary.md#term-cpu) analysis inputs request `VECTOR_IF_ELIGIBLE`; and
- existing carrier, span, accessibility, native-order segment, writability, and overlap checks
  succeed before invocation or worker submission.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>

The exact vector type [matrix](../../../../glossary.md#term-matrix-meanings) is therefore two rows:

| Prediction | Target | Result/[lanes](../../../../glossary.md#term-lane) | Vector form |
|---|---|---|---|
| `FLOAT32` | `FLOAT32` | `FLOAT32` / preferred `FloatVector` species | eligible |
| `FLOAT64` | `FLOAT64` | `FLOAT64` / preferred `DoubleVector` species | eligible |

Every other already admitted [loss](../../../../glossary.md#loss) form remains supported through its current [scalar](../../../../glossary.md#term-scalar) compute:

- [MSE](../../../../glossary.md#mean-squared-error--mse) `SUM` and [`MEAN`](../../../../glossary.md#term-mean), including dense large inputs;
- [BFLOAT16](../../../../glossary.md#term-bfloat16) MSE and either mixed ordered floating pair;
- MSE `NONE` with any non-[contiguous](../../../../glossary.md#term-contiguous) prediction, target, or output access;
- dense-target and index-target categorical loss for every reduction; and
- zero-[element](../../../../glossary.md#term-element), [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), or otherwise shorter-than-one-species MSE `NONE` work.

`VECTOR_IF_ELIGIBLE` is a preference, not a requirement. An ineligible vector form selects
`SCALAR` or `PARALLEL_SCALAR` according to the existing range policy; it does not lose CPU
capability, decompose the loss, materialize a value, or fail an otherwise valid occurrence.

### [Shapes](../../../../glossary.md#shape), [layouts](../../../../glossary.md#layout), carriers, ranges, and [aliases](../../../../glossary.md#aliasing-and-arena)

- Vector eligibility is rank-polymorphic over any fully static Shape. A rank-zero Shape contains
  one element and selects scalar. Any zero extent produces zero work and no generated call or
  worker submission.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [rank glossary](../../../../glossary.md#term-rank); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [element glossary](../../../../glossary.md#term-element); [scalar glossary](../../../../glossary.md#term-scalar); [extent glossary](../../../../glossary.md#term-extent))</span>
- Dense layouts may have zero or positive cold storage offsets. Surrounding storage is untouched.
  Positive/zero strides that do not form the exact dense row-major layout retain the scalar path;
  there is no gather, scatter, lane map, or automatic contiguous copy.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [row-major glossary](../../../../glossary.md#term-row-major); [gather glossary](../../../../glossary.md#gather); [lane glossary](../../../../glossary.md#term-lane); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- Each unique boundary is either its exact primitive array (`float[]` or `double[]`) or an exact
  native-order accessible [`MemorySegment`](../../../../glossary.md#term-memorysegment). Every ordered array/segment pattern is eligible.
  Segment element offsets are converted to byte offsets with the lane type's exact width.
- The direct generated signature remains one typed carrier parameter per unique boundary,
  followed by <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [carrier glossary](../../../../glossary.md#term-storage-carrier); [parameter glossary](../../../../glossary.md#term-parameter))</span>`long[] geometry`, primitive `long start`, and primitive `long end`, returning
  `void`. Distinct inputs have three carrier parameters; an exact repeated input role has one
  shared read parameter plus one output parameter. There is no `Object`, scratch, worker, Shape,
  layout, operation, or strategy parameter.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Any valid [half-open](../../../../glossary.md#term-half-open-interval) `[start,end)` inside the logical element domain is accepted. Full vectors
  start at the exact cold base plus `start`; the loop executes unmasked complete preferred-
  species chunks, then the existing scalar formula handles each remaining element exactly once.
  Empty and short requested ranges execute no vector access.
- Read-only prediction and target storage may be identical or overlap. Exact repeated graph input
  roles retain the existing role-to-unique-boundary map <span data-glossary-navigation="true"> ([prediction glossary](../../../../glossary.md#term-prediction); [target glossary](../../../../glossary.md#term-target); [graph glossary](../../../../glossary.md#graph))</span>`[0,0]`; distinct roles retain `[0,1]`
  even when their bound storage overlaps. Output may share an underlying carrier only when its
  validated byte span is disjoint from every input span. Any output/prediction or output/target
  overlap is rejected before output mutation or worker submission. The one-output family has no
  output/output alias case.

### [Vector](../../../../glossary.md#term-vector) body and [scalar tail](../../../../glossary.md#term-scalar-tail)

For each full chunk, the generated body and its optimal clean Java [oracle](../../../../glossary.md#term-test-oracle) must:

1. load prediction and target with the matching preferred species through generation-time-
   selected array or native-order segment calls;
2. compute one typed vector subtraction in [prediction](../../../../glossary.md#term-prediction)/[target](../../../../glossary.md#term-target) order;
3. multiply that difference vector by itself, without fused multiply-add, reassociation,
   widening, reduction, or horizontal lane operation;<span data-glossary-navigation="true"> ([fused multiply-add glossary](../../../../glossary.md#term-fused-multiply-add-fma); [reassociation glossary](../../../../glossary.md#term-reassociation); [widening glossary](../../../../glossary.md#term-widening); [operation glossary](../../../../glossary.md#operation))</span>
4. store the typed vector directly to the selected output [carrier](../../../../glossary.md#term-storage-carrier); and
5. advance only primitive ordinal/address state by the preferred [lane](../../../../glossary.md#term-lane) count.

After the last full chunk, the scalar tail performs the existing typed subtraction, multiply, and
store in increasing logical-[element](../../../../glossary.md#term-element) order. There is no masked tail, temporary array, per-lane
extraction/insertion, [scalar](../../../../glossary.md#term-scalar) [callback](../../../../glossary.md#term-callback) per lane, or call to a Synaptik-owned vector helper.

[`FLOAT32`](../../../../glossary.md#term-float32) performs [binary32](../../../../glossary.md#floating-point-precision-and-rounding) subtraction and multiplication. [`FLOAT64`](../../../../glossary.md#term-float64) performs [binary64](../../../../glossary.md#floating-point-precision-and-rounding)
subtraction and multiplication. This preserves the existing [MSE](../../../../glossary.md#mean-squared-error--mse) `NONE` [NaN](../../../../glossary.md#term-nan) classification,
[infinity](../../../../glossary.md#term-infinity) behavior, [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) squaring, [subnormal](../../../../glossary.md#term-subnormal)/[underflow](../../../../glossary.md#term-underflow) behavior, and [overflow](../../../../glossary.md#term-overflow) behavior.
Tests compare [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) where the existing semantics define them and otherwise use the same
classification/[tolerance](../../../../glossary.md#term-tolerance) rules as [CPU](../../../../glossary.md#term-cpu) 0008I. The task introduces no cross-[backend](../../../../glossary.md#backend) bitwise promise.

### Strategy selection, parallelism, and [run](../../../../glossary.md#term-run) isolation

- [CPU](../../../../glossary.md#term-cpu) analysis selects [`VECTOR`](../../../../glossary.md#term-vector) when the exact vector conditions hold and the existing selected
  range count is one. It selects `PARALLEL_VECTOR` when the same conditions hold and the existing
  bounded range policy selects at least two ranges.
- Parallel-vector reuses the same vector generated artifact. Configured/available parallelism,
  minimum elements per worker, selected range count, worker identity, and chunk boundaries remain
  prepared or invocation facts outside class identity.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [elements glossary](../../../../glossary.md#term-element); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Existing quotient/remainder chunking produces ascending, non-empty, disjoint ranges that cover
  the requested interval exactly once. Each worker owns only its output range; scalar tails are
  local to that range.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Existing inline one-chunk behavior, nested-submission rejection, worker accessibility proof,
  synchronous join, deterministic failure selection/suppression, interruption behavior, and
  caller-owned <span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous))</span>[`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) [lifetime](../../../../glossary.md#term-lifetime) remain unchanged.
- `CpuPreparedExecutable` remains immutable and reusable. Concurrent runs and concurrent bound
  invocations retain distinct geometry, ranges, carrier bindings, and Runtime <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [carrier glossary](../../../../glossary.md#term-storage-carrier); [bindings glossary](../../../../glossary.md#term-binding); [Runtime glossary](../../../../glossary.md#term-runtime))</span>[`RunState`](../../../../glossary.md#run-state--runstate) objects;
  they share only immutable generated artifacts and explicitly borrowed worker capacity.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>

### [Schema](../../../../glossary.md#term-schema), class [identity](../../../../glossary.md#term-identity-meanings), direct [signatures](../../../../glossary.md#term-signature), and [cache](../../../../glossary.md#term-cache) facts

- Advance `CpuGeneratorSchema.CURRENT_VERSION` from 61 to 62.
- Use class-identity schema 62 only for a [vector](../../../../glossary.md#term-vector)-compute `CpuLossIr` whose exact semantic form is
  same-typed FLOAT32/FLOAT64 MSE <span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>`NONE` and whose three occurrence accesses are dense. Preserve
  class-identity schema 58 and byte-identical generated classes for every scalar loss form.
  Preserve schema 52, 54-57, and 59-61 projections and generated bytes for all unchanged families.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Preserve the existing route-independent `CpuLossIr` semantic/range/role fingerprint for scalar
  compatibility. Vector code shape is distinguished by specialization compute <span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [specialization glossary](../../../../glossary.md#term-specialization))</span>`VECTOR`, exact
  preferred-species bit size, ordered boundary types/carriers, class schema 62, and the existing
  loss fingerprint. In particular, preserve the existing
  `workspace=NONE:realization=DIRECT_SCALAR` fingerprint text byte-for-byte: despite that
  historical token, compute strategy remains a specialization fact, and changing the token would
  invalidate the schema-58 scalar inventory. Do not add a second loss IR, vector-loss operation
  kind, or strategy field to Model semantics.<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token); [operation glossary](../../../../glossary.md#operation); [Model glossary](../../../../glossary.md#term-model))</span>
- Code-shaping facts are the existing [MSE](../../../../glossary.md#mean-squared-error--mse) `NONE` family/reduction/range/role identity, same lane
  type, exact ordered unique-boundary carrier pattern, vector compute, preferred-species bit size,
  and schema 62. Parallel orchestration does not create another class.<span data-glossary-navigation="true"> ([lane glossary](../../../../glossary.md#term-lane); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Rank, extents, element count, cold base offsets, concrete carrier instances/addresses, slots,
  graph/value/run identities, start/end bounds, selected range count, chunking, worker identity,
  and artifact root remain outside generated class identity.<span data-glossary-navigation="true"> ([Rank glossary](../../../../glossary.md#term-rank); [element glossary](../../../../glossary.md#term-element); [graph glossary](../../../../glossary.md#graph); [run glossary](../../../../glossary.md#term-run); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- `compatibilityBytes()` advances with envelope 62, so every older persisted envelope is a safe
  miss. `classIdentityBytes()` retains prior projections for unchanged classes. Cache tests prove
  equal vector requests reuse one process-local artifact, carrier order/species/type/role changes
  miss, parallel-vector reuses vector identity, and cold geometry/run facts do not change the key.
- Generated schema-62 classes remain final, field-free, constructor-free hidden classes. They
  retain one public direct typed `invoke` entry and the current loss-private direct-body members;
  cold binding resolves the exact contiguous body before execution. No method-handle lookup or
  helper selection occurs in the generated loop or worker call.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [contiguous glossary](../../../../glossary.md#term-contiguous); [method-handle lookup glossary](../../../../glossary.md#term-method-handle-lookup))</span>

The current generic instruction-free-family path can classify [homogeneous](../../../../glossary.md#term-homogeneous) [loss](../../../../glossary.md#loss) [IR](../../../../glossary.md#term-intermediate-representation-ir) as vector-
eligible even though schema-58 `CpuLossEmitter` emits [scalar](../../../../glossary.md#term-scalar) arithmetic. Replace that accidental
generic eligibility only with the exact family-specific predicate above: MSE `NONE` gains vector
compute, while MSE reductions and both categorical families select scalar or parallel-scalar.
This correction must not alter the schema-58 scalar [fingerprint](../../../../glossary.md#term-fingerprint), direct signatures, or bytes.

### Exact [finite](../../../../glossary.md#term-finite) [validation](../../../../glossary.md#term-validation) [matrices](../../../../glossary.md#term-matrix-meanings)

The implementation must encode these finite matrices as named rows, assert their counts, and fail
on a missing or duplicate row. “All variants” is not acceptable evidence.

#### Generated [identity](../../../../glossary.md#term-identity-meanings) and semantic execution [matrix](../../../../glossary.md#term-matrix-meanings)

Retain exactly 24 [schema](../../../../glossary.md#term-schema)-62 Class-File dossiers:

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
workers, and a minimum-[elements](../../../../glossary.md#term-element) setting that selects at least two non-empty chunks. Each case
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

The four exceptional-value rows each contain [finite](../../../../glossary.md#term-finite) ordinary values, `+0.0`, `-0.0`, positive and
[negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero), [NaN](../../../../glossary.md#term-nan), maximum finite values that square to [infinity](../../../../glossary.md#term-infinity), minimum-normal/[subnormal](../../../../glossary.md#term-subnormal)
values that exercise [underflow](../../../../glossary.md#term-underflow), and [lane](../../../../glossary.md#term-lane)-boundary plus [scalar](../../../../glossary.md#term-scalar)-tail positions.

Retain exactly nine unchanged scalar controls: [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [MSE](../../../../glossary.md#mean-squared-error--mse) `SUM` and [`MEAN`](../../../../glossary.md#term-mean) (four),
[BFLOAT16](../../../../glossary.md#term-bfloat16) MSE `NONE` (one), both ordered FLOAT32/FLOAT64 mixed MSE `NONE` forms (two), FLOAT32 dense
categorical `NONE` (one), and FLOAT32/[INT32](../../../../glossary.md#data-type--datatype) index categorical `NONE` without ignore (one). Each
control remains scalar compute with schema 58. A pre-edit scalar-[loss](../../../../glossary.md#loss) baseline generated from the
clean task-start [commit](../../../../glossary.md#term-commit) and the post-edit complete 792-class schema-58 inventory must have the
same keys and Class-File [SHA-256](../../../../glossary.md#term-sha-256) values; the comparison is retained under the evidence root.

#### Class-File and forbidden-overhead inspection

For all 24 [schema](../../../../glossary.md#term-schema)-62 classes retain the Class-File, complete `javap -c -v -p`, [descriptor](../../../../glossary.md#term-descriptor)/member
inventory, normalized instruction/call-[owner](../../../../glossary.md#term-owner) record, [SHA-256](../../../../glossary.md#term-sha-256), and deterministic second emission.
Inspection must prove:

- exact direct [signatures](../../../../glossary.md#term-signature) for every distinct/shared ordered [carrier](../../../../glossary.md#term-storage-carrier) pattern;
- preferred `FloatVector` or `DoubleVector` loads, one vector subtraction, one vector multiply,
  direct vector store, lane-count ordinal advance, and the scalar tail;<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [scalar tail glossary](../../../../glossary.md#term-scalar-tail))</span>
- no `reduceLanes`, horizontal reduction, [FMA](../../../../glossary.md#term-fused-multiply-add-fma), masked-tail load/store, [gather](../../../../glossary.md#gather)/scatter,
  `intoArray` temporary, per-[lane](../../../../glossary.md#term-lane) extraction, or [scalar](../../../../glossary.md#term-scalar) [callback](../../../../glossary.md#term-callback) in a full chunk;
- no allocation, boxing, reflection, `invokedynamic`, dynamic constant/bootstrap, collection,
  map/string/opcode dispatch, monitor, method-handle lookup, graph/layout/operation/cache/route/
  resource/worker lookup, fallback/reference call, or Synaptik-owned hot helper; and<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [opcode glossary](../../../../glossary.md#term-opcode); [dispatch glossary](../../../../glossary.md#term-dispatch); [method-handle lookup glossary](../../../../glossary.md#term-method-handle-lookup); [graph glossary](../../../../glossary.md#graph); [layout glossary](../../../../glossary.md#layout); [operation glossary](../../../../glossary.md#operation); [cache glossary](../../../../glossary.md#term-cache))</span>
- unchanged scalar [loss](../../../../glossary.md#loss) and prior-family class projections/[hashes](../../../../glossary.md#term-hash) as described above.

#### Generated-versus-optimal-clean-Java performance [matrix](../../../../glossary.md#term-matrix-meanings)

Measure exactly six rows per [lane](../../../../glossary.md#term-lane) type:

| Row per type | Boundaries | Offsets | Strategy | Range |
|---|---|---|---|---|
| 1 | distinct array/array/array | zero | [`VECTOR`](../../../../glossary.md#term-vector) | exact multiple |
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

Use a [shape](../../../../glossary.md#shape)-polymorphic dense [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) workload whose [element](../../../../glossary.md#term-element) count is
`8192 * preferredLanes` for exact rows and `8192 * preferredLanes + 3` for tail rows. Positive
offset is seven elements. Shared-input rows bind one exact input boundary. The [fixture](../../../../glossary.md#term-fixture) is cold
geometry, not fixed-shape [specialization](../../../../glossary.md#term-specialization).

The ordinary clean Java 26 [oracle](../../../../glossary.md#term-test-oracle) has the exact generated typed [signature](../../../../glossary.md#term-signature), preferred species,
[carrier](../../../../glossary.md#term-storage-carrier) order, cold `long[]` geometry, and `start`/`end` bounds. It is compiled by `javac` before
timing and uses the same full-vector/[scalar](../../../../glossary.md#term-scalar)-tail algorithm, load/store order, primitive address
state, parallel ranges, and output consumption as the generated side. Source and [decompilation](../../../../glossary.md#term-decompilation)
inspection must reject a slower oracle, dispatcher, helper bridge, allocation, boxing,
reflection, fallback/reference call, fixed fixture trip count, or different algorithm.

Each [fork](../../../../glossary.md#term-fork-meanings) uses Java 26, fixed `-Xms1g -Xmx1g`, [C2](../../../../glossary.md#term-c2)-only [synchronous](../../../../glossary.md#term-synchronous) [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names)
(`-XX:-TieredCompilation -Xbatch`), a recorded deterministic row/order [seed](../../../../glossary.md#term-seed), exact pre/post output
verification, and exactly five [warmup](../../../../glossary.md#term-warmup) pairs before final calibration. After warmup, conservatively
double one shared iteration count until the two-invocation generated aggregate and the two-
invocation direct aggregate are each at least 50 ms. Retain nine randomized symmetric
`G-D-D-G` or `D-G-G-D` AB/BA pairs, and require each of the four individual retained timings in
every pair to be at least 25 ms; the 50 ms calibration rule is only the conservative precondition,
not a substitute for checking every side. No retry, discarded [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), replacement fork, [outlier](../../../../glossary.md#term-outlier)
filter, threshold change, asymmetric work, or production selection is permitted.

Every retained pair ratio, fork [median](../../../../glossary.md#term-median), and aggregate generated/direct ratio must be `<= 1.15x`.
Retain raw four-timing samples, iteration counts, checksums, environment/[JDK](../../../../glossary.md#term-java-development-kit-jdk)/OS/[CPU](../../../../glossary.md#term-cpu)/[heap](../../../../glossary.md#term-heap) facts,
commands, source snapshots, oracle bytes, generated classes, decompilation, summaries, and a
complete [SHA-256](../../../../glossary.md#term-sha-256) [manifest](../../../../glossary.md#term-manifest) under one explicit untracked evidence root.

## Out of [scope](../../../../glossary.md#term-scope)

- [Vector](../../../../glossary.md#term-vector) or parallel-vector [MSE](../../../../glossary.md#mean-squared-error--mse) `SUM`/[`MEAN`](../../../../glossary.md#term-mean); horizontal reduction, reassociation, partial
  reductions, combine kernels, workspace, and deterministic partial-reduction parallelism.<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation); [kernels glossary](../../../../glossary.md#kernel))</span>
- Dense/index categorical SIMD; softmax/log-softmax/attention numerical changes; CPU 0008O's
  stable-reduction eligibility work.<span data-glossary-navigation="true"> ([SIMD glossary](../../../../glossary.md#simd); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- BFLOAT16/FLOAT16 SIMD, mixed-type vector conversion/promotion, integral/BOOL work, relaxed math,
  FMA, masked vector tails, gather/scatter, or non-contiguous vector access.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [gather glossary](../../../../glossary.md#gather); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- Materialization, packing, fixed-shape specialization, unrolling, native/vendor/OpenBLAS routes,
  autotuning, tuning-cache mutation, benchmark-selected production behavior, or persistence
  policy.<span data-glossary-navigation="true"> ([Materialization glossary](../../../../glossary.md#materialization); [shape glossary](../../../../glossary.md#shape); [specialization glossary](../../../../glossary.md#term-specialization); [unrolling glossary](../../../../glossary.md#term-unrolling); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Loss fusion, decomposed-loss recognition, public API, operation, gradient, Training, Model,
  Compiler, Planning, Config, Backend Contract, Trace, shared Prepare, Runtime, Engine, other
  backend, module dependency, architecture, ADR, build-toolchain, conformance, or integration
  change.<span data-glossary-navigation="true"> ([Loss glossary](../../../../glossary.md#loss); [fusion glossary](../../../../glossary.md#term-fusion); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [operation glossary](../../../../glossary.md#operation); [gradient glossary](../../../../glossary.md#term-gradient); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [toolchain glossary](../../../../glossary.md#term-toolchain); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- CPU 0008N implementation or detailed specification; CPU 0008N becomes the sole Ready frontier
  after this task completes, without a detailed task file.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially concrete CPU route,
  staged Prepare, Runtime/run isolation, and generated-code performance discipline.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md).
- [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>.
- [CPU backend master plan](../master-plan.md)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>.
- [CPU 0008I portable loss-family execution](0008i-portable-loss-family-execution.md)<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss))</span>.
- [CPU 0008L pointwise SIMD mask/output closure](0008l-pointwise-simd-mask-output-closure.md)<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [SIMD glossary](../../../../glossary.md#simd); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>.
- [CPU 0005C vector and parallel portable strategies](0005c-vector-and-parallel-portable-strategies.md)<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>.
- [CPU 0005I FLOAT32 vector parity](0005i-float32-vector-parity-and-vector-emission-boundary.md)<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32))</span>.
- [CPU 0007A0 generated hot-path parity](0007a0-generated-hot-path-parity-correction.md).
- [CPU 0007A1A generated scalar-body self-containment](0007a1a-generated-scalar-body-self-containment.md)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>.
- [CPU 0008B partition-DAG decomposition](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md)<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>.
- [CPU 0008E1 shared partition-DAG adoption](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md).
- [CPU backend guide](../../../../backend-guide/cpu-backend.md).
- [Glossary](../../../../glossary.md).

## Architecture constraints

- Model owns MSE meaning and Compiler owns inference/gradients. Planning continues to select only
  CPU ownership. CPU analysis alone owns vector eligibility, preferred species, strategy,
  specialization, direct carrier signature, and scalar fallback.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [MSE glossary](../../../../glossary.md#mean-squared-error--mse); [Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [gradients glossary](../../../../glossary.md#term-gradient); [Planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [vector glossary](../../../../glossary.md#term-vector); [specialization glossary](../../../../glossary.md#term-specialization); [carrier glossary](../../../../glossary.md#term-storage-carrier); [signature glossary](../../../../glossary.md#term-signature); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- CPU analysis selects the route/strategy and declares unchanged buffers before shared assignment;
  finalization realizes that immutable decision. Shared Prepare and Runtime do not interpret loss,
  vectors, species, carriers, ranges, or workers.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [loss glossary](../../../../glossary.md#loss))</span>
- Generated code receives direct typed carriers plus primitive cold geometry/ranges. Runtime hot
  execution sees no [`Operation`](../../../../glossary.md#operation), `CompiledNode`, [graph](../../../../glossary.md#graph), strategy, or [backend](../../../../glossary.md#backend) [lookup](../../../../glossary.md#term-lookup).
- Prepared recipes remain [immutable](../../../../glossary.md#term-immutable)/reusable and each active [run](../../../../glossary.md#term-run) has isolated [mutable](../../../../glossary.md#term-mutable) [`RunState`](../../../../glossary.md#run-state--runstate).
  Existing CPU-private caller-owned workers are the only shared orchestration resource.
- The `AGENTS.md` generated-code rule is mandatory: directly emitted bytecode must preserve the
  optimal clean Java specialized algorithm, full-vector/scalar-tail loop and dataflow shape, and
  avoidable-overhead profile. Any deviation requires a source-backed reason and evidence.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape))</span>
- Any public/shared contract, new dependency, resource kind, architecture decision, or need to
  change current MSE semantics is a stop condition.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.ir` — unchanged [loss](../../../../glossary.md#loss) semantic/range/role [identity](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — exact vector eligibility, strategy,
  species, and schema selection.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct typed MSE vector body,
  scalar tail, structural evidence, and test-only optimal Java oracle.<span data-glossary-navigation="true"> ([MSE glossary](../../../../glossary.md#mean-squared-error--mse); [scalar tail glossary](../../../../glossary.md#term-scalar-tail))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — [schema](../../../../glossary.md#term-schema)-62 compatibility and class identity.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — unchanged cold helper binding,
  overlap validation, ranged calls, workers, failures, and run isolation.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — unchanged [scalar](../../../../glossary.md#term-scalar) semantic [oracle](../../../../glossary.md#term-test-oracle).

Packages added or changed:

- No package or responsibility is added or moved.

Type placement:

- No production type is added. Extend the existing `CpuLossEmitter`, `CpuPartitionPreparer`,
  `CpuKernelSpecialization`, `CpuGeneratorSchema`, and `CpuLossIr` documentation/[owners](../../../../glossary.md#term-owner) only.
- `...internal.codegen.emit.CpuVectorMseEvidenceTest` — package-private exhaustive schema-62
  semantic, Class-File, scalar-preservation, and retained-evidence owner.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- `...internal.codegen.emit.CpuVectorMsePerformanceOracle` — package-private test-only `javac`
  compiler/binder for the optimal direct Java Vector API methods; unreachable from production.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [Vector API glossary](../../../../glossary.md#term-vector-api))</span>
- `...internal.codegen.emit.CpuVectorMsePerformanceTest` — package-private opt-in fixed-matrix,
  fixed-fork performance and manifest owner.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [fork glossary](../../../../glossary.md#term-fork-meanings); [manifest glossary](../../../../glossary.md#term-manifest))</span>

## Affected files

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) paths (5):

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
test behavior or environment-variable [alias](../../../../glossary.md#aliasing-and-arena).

Expected documentation/[planning](../../../../glossary.md#term-planning) paths (5):

- `docs/backend-guide/cpu-backend.md` — update the current portable-loss strategy and
  compatibility-schema account, which currently ends at schema 61/CPU 0008L.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- `docs/glossary.md` — update only the existing [`CPU portable route`](../../../../glossary.md#cpu-portable-route) entry, which currently ends
  at CPU 0008L/schema 61; add no term or heading.
- `docs/planning/backends/cpu/tasks/0008m-vector-mse-none.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Reviewed and expected unchanged: `CpuCapabilityProvider`, `CpuClassFileKernelGenerator`,
`CpuCarrierEmitter`, `CpuVectorInstructionEmitter`, `CpuLoopEmitter`, `CpuScalarEmitter`,
`CpuLossLowering`, `CpuLossReferenceKernel`, `CpuPortableRoutePlan`, `CpuGeneratedKernel`,
`CpuPreparedExecutable`, package summaries, existing [scalar](../../../../glossary.md#term-scalar) [loss](../../../../glossary.md#loss) performance [owners](../../../../glossary.md#term-owner), [Model](../../../../glossary.md#term-model)/
[Compiler](../../../../glossary.md#term-compiler)/Training APIs, shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime), architecture pages/ADRs/tests,
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance)/integration tests, other backends, and root Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 28 paths: five production/[Javadoc](../../../../glossary.md#term-javadoc), 18 test/evidence/build, and five documentation/[planning](../../../../glossary.md#term-planning).
At most three package-private test types and no production type or package may be added.

This exceeds the normal task range because one cohesive generated-code increment must update
eligibility, [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings), direct [vector](../../../../glossary.md#term-vector) emission, [scalar](../../../../glossary.md#term-scalar)-preservation controls,
semantic/Class-File/performance evidence, hard-coded current-envelope assertions, Javadocs, and
[backend](../../../../glossary.md#backend) documentation together. The prior-family evidence edits authorize only current-envelope
assertion maintenance; they must not change prior-family behavior or generated bytes. A listed
review path need not change merely because it is named. Stop and replan if another production
[owner](../../../../glossary.md#term-owner), test type, package, shared/public path, resource, route, or architecture edit is required.

## Acceptance criteria

1. [CPU](../../../../glossary.md#term-cpu) analysis selects [vector](../../../../glossary.md#term-vector) compute for exactly same-typed [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [contiguous](../../../../glossary.md#term-contiguous) [MSE](../../../../glossary.md#mean-squared-error--mse)
   `NONE` under vector preference and the species/count gates; every other existing loss form
   retains scalar compute or existing scalar fallback without losing capability.<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [scalar glossary](../../../../glossary.md#term-scalar))</span>
2. Generated FLOAT32/FLOAT64 full chunks implement exactly typed subtract, self-multiply, and
   direct store; scalar tails preserve the current scalar formula and visit every remainder once.
   No reduction, FMA, widening, reassociation, masked tail, gather/scatter, or per-lane callback is
   present.<span data-glossary-navigation="true"> ([FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [widening glossary](../../../../glossary.md#term-widening); [reassociation glossary](../../../../glossary.md#term-reassociation); [gather glossary](../../../../glossary.md#gather); [lane glossary](../../../../glossary.md#term-lane); [callback glossary](../../../../glossary.md#term-callback))</span>
3. The exact 106-case focused semantic matrix passes across all 24 identities, both vector
   strategies, every ordered array/segment pattern, offsets, exact/tail/short/zero/rank-zero/
   arbitrary ranges, exceptional values, legal aliases, pre-write rejection, and concurrency.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
4. Vector and parallel-vector use the same direct typed artifact. Existing worker chunking,
   joining, failure behavior, accessibility checks, immutable prepared recipes, and run isolation
   remain unchanged.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [immutable glossary](../../../../glossary.md#term-immutable); [run glossary](../../../../glossary.md#term-run))</span>
5. Schema advances 61 to 62 only for changed vector MSE bytes. All 24 schema-62 identities are
   unique and deterministic; scalar loss remains schema 58 and the complete 792-class scalar
   inventory is byte-identical to its clean task-start baseline. Every other prior class projection
   remains unchanged, while compatibility envelope 62 safely misses older persisted entries.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema))</span>
6. Direct signatures and cache keys contain every code-shaping type/carrier/role/compute/species
   fact and exclude every cold extent/offset/address/range/chunk/worker/run fact listed in Scope.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [cache glossary](../../../../glossary.md#term-cache); [carrier glossary](../../../../glossary.md#term-storage-carrier); [extent glossary](../../../../glossary.md#term-extent); [Scope glossary](../../../../glossary.md#term-scope))</span>
7. The exact 24 retained Class-File dossiers pass complete descriptor/member/decompilation,
   deterministic-byte, algorithm/loop/tail, call-owner, and forbidden-overhead inspection.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [decompilation glossary](../../../../glossary.md#term-decompilation); [owner glossary](../../../../glossary.md#term-owner))</span>
8. All 540 retained symmetric sample pairs, 60 fork medians, and 12 aggregates in the exact
   performance matrix are <span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>`<= 1.15x`; every retained generated and direct side is at least 25 ms,
   and the protocol uses warmup before conservative post-warmup calibration, fixed forks, and no
   retry/discard.<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup))</span>
9. MSE `SUM`/[`MEAN`](../../../../glossary.md#term-mean) stay scalar and their increasing-order left-associated accumulation remains
   unchanged. No categorical, BFLOAT16, mixed-type, non-contiguous, materialized, native, or
   partial-reduction vector form becomes selectable.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
10. No public/shared API, resource, dependency, architecture, conformance, integration, toolchain,
    tuning, persistence-policy, or unrelated generated-family behavior changes.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance); [toolchain glossary](../../../../glossary.md#term-toolchain); [persistence glossary](../../../../glossary.md#term-persistence))</span>
11. A distinct clean documentation-focused pass finalizes affected Javadocs, the CPU guide,
    glossary impact, task evidence, CPU master plan, and roadmap before completion. It records
    reasoned no-change conclusions and reuses successful Java evidence unless it changes executable
    behavior.
12. On completion CPU 0008M becomes `Complete`; CPU 0008N becomes the sole [`Ready`](../../../../glossary.md#term-ready) CPU frontier.
    No 0008N detailed task is created in the 0008M completion pass.

## Tests / [validation](../../../../glossary.md#term-validation)

### Implementation commands

[Capture](../../../../glossary.md#term-capture-meanings) the clean [scalar](../../../../glossary.md#term-scalar) baseline before production edits, then [run](../../../../glossary.md#term-run) focused tests while
implementing. The implementation context must replace the focused command only if an expected
[owner](../../../../glossary.md#term-owner) proves unnecessary; it may not silently omit a changed owner.

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
reconstruct after the edit, or skip the pre/post 792-class key and Class-File [hash](../../../../glossary.md#term-hash) comparison.

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

Retain exact commands, task outcomes/test counts, 24 [vector](../../../../glossary.md#term-vector) classes and `javap` reports, both
792-class [scalar](../../../../glossary.md#term-scalar) inventories and their equality report, 540 [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)-pair records, 60 [fork](../../../../glossary.md#term-fork-meanings) medians,
12 aggregates, source/[oracle](../../../../glossary.md#term-test-oracle) snapshots, and complete manifests/checksums. The evidence root is
untracked and must not be staged. `CpuVectorMseEvidenceTest` must read both scalar inventory roots,
compare the complete key-to-[SHA-256](../../../../glossary.md#term-sha-256) maps, and write the retained equality report; directory counts
or [schema](../../../../glossary.md#term-schema) labels alone are insufficient.

### Documentation and final checks

After executable Java stabilizes, the implementation context runs the full [CPU](../../../../glossary.md#term-cpu) suite once. The
separate documentation context reuses that evidence unless it changes executable behavior, then
runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

Also validate all local links and heading anchors in the five documentation/[planning](../../../../glossary.md#term-planning) files;
heading uniqueness; balanced Markdown fences; final newlines; trailing whitespace; the exact
`16 + 8 = 24`, `48 + 12 + 10 + 4 + 12 + 16 + 4 = 106`, and `12 * 5 * 9 = 540` arithmetic;
[schema](../../../../glossary.md#term-schema) `61 -> 62`; the 5 + 18 + 5 = 28 path ceiling; exact type placement; no tracked evidence;
empty staging; and synchronized order/status with 0008M Complete, CPU 0008N as the sole [Ready](../../../../glossary.md#term-ready)
frontier, and CPU 0008O onward Draft.

Repository-wide Java [validation](../../../../glossary.md#term-validation) is deferred to CPU 0009/[CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes only one
CPU-private route increment. Architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration suites are not
[run](../../../../glossary.md#term-run) unless a shared/[dependency](../../../../glossary.md#term-dependency)/public boundary changes, which is a stop-and-replan condition.

## Dependencies

- [CPU 0008I](0008i-portable-loss-family-execution.md) — Complete; atomic scalar loss semantics,
  geometry, direct signatures, 792-class inventory, validation, and scalar oracle.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [scalar glossary](../../../../glossary.md#term-scalar); [loss glossary](../../../../glossary.md#loss); [signatures glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- [CPU 0008L](0008l-pointwise-simd-mask-output-closure.md) — Complete; current schema 61,
  preferred-species array/segment generation, scalar tails, retained Class-File evidence, and the
  corrected warmup/calibration/performance protocol.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [warmup glossary](../../../../glossary.md#term-warmup))</span>
- [CPU 0005C](0005c-vector-and-parallel-portable-strategies.md) and
  [CPU 0005I](0005i-float32-vector-parity-and-vector-emission-boundary.md) — Complete; four
  strategies, FLOAT32/FLOAT64 preferred species, direct carriers, ranges, workers, and fallback.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
- [CPU 0007A0](0007a0-generated-hot-path-parity-correction.md) and
  [CPU 0007A1A](0007a1a-generated-scalar-body-self-containment.md) — Complete; generated/direct
  algorithmic parity and forbidden-overhead evidence discipline.
- [CPU 0008B](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md)
  and [CPU 0008E1](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md) — Complete;
  atomic numerical-order barriers and shared partition-DAG adoption.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
- Java 26.0.1 Class-File and incubating [Vector](../../../../glossary.md#term-vector) APIs through the existing [CPU](../../../../glossary.md#term-cpu) build.

## Follow-up tasks

- CPU 0008N is the sole Ready frontier after 0008M completes. It owns only measured
  profitable FLOAT32/FLOAT64 Conv2d/Conv3d accumulation-axis SIMD and requires its benchmark-axis
  spike; do not detail or implement it here.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Ready glossary](../../../../glossary.md#term-ready); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [axis glossary](../../../../glossary.md#term-axis); [SIMD glossary](../../../../glossary.md#simd))</span>
- CPU 0008O remains the later Draft stable-reduction numerical spike for softmax, categorical
  loss, and attention.<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [loss glossary](../../../../glossary.md#loss))</span>
- CPU 0008P remains the later Draft deterministic partial-reduction architecture with per-worker
  workspace and combine execution.
- CPU 0009 remains the generated-coverage closure checkpoint and must inventory schema-62 vector
  MSE evidence separately from CPU 0008I's still-missing corrected full scalar-loss performance
  gate.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [schema glossary](../../../../glossary.md#term-schema); [vector glossary](../../../../glossary.md#term-vector); [MSE glossary](../../../../glossary.md#mean-squared-error--mse); [scalar glossary](../../../../glossary.md#term-scalar))</span>

## Architecture impact

Expected impact: None.

Required no-change conclusions:

- Architecture contract/pages/ADRs/tests remain unchanged because ownership, lifecycle,
  dependencies, generated-code authority, and run isolation do not change.<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [run glossary](../../../../glossary.md#term-run))</span>
- [Model](../../../../glossary.md#term-model) [loss](../../../../glossary.md#loss) semantics, `LossKind`, `LossReduction`, `MeanSquaredErrorAttrs`, Tensor/Compiler/
  Training APIs, and capability-query contracts remain unchanged because CPU support and output
  meaning already exist.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Planning, shared Prepare, and Runtime remain unchanged; current projected facts, declarations,
  slot assignment, finalization, cold binding, publication, worker ranges, and run state suffice.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding); [publication glossary](../../../../glossary.md#term-publication); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- `CpuCapabilityProvider`, `CpuLossLowering`, `CpuPreparedExecutable`, [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup), and scalar
  reference semantics remain behaviorally unchanged. CPU-private analysis changes only strategy
  eligibility for the exact form.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- [MSE](../../../../glossary.md#mean-squared-error--mse) `SUM`/[`MEAN`](../../../../glossary.md#term-mean), categorical losses, prior scalar loss forms, other generated families, native
  routes, materialization/tuning policy, and other backends remain behaviorally unchanged.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [backends glossary](../../../../glossary.md#backend))</span>
- Backend-conformance and integration tests remain unchanged because no shared/public behavior
  owner or Engine path changes; CPU-private semantic and generated evidence owns this increment.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [owner glossary](../../../../glossary.md#term-owner); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Module dependencies, plugins, source sets, and root Java 26 toolchain remain unchanged. CPU
  Gradle may forward only guarded evidence properties to the test JVM.<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [toolchain glossary](../../../../glossary.md#term-toolchain); [Gradle glossary](../../../../glossary.md#term-gradle); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- The [CPU](../../../../glossary.md#term-cpu) guide and existing [`CPU portable route`](../../../../glossary.md#cpu-portable-route) glossary entry receive only the narrow current-
  behavior/schema update identified under Affected files. No architecture page, ADR, public API
  guide, new glossary term, or unrelated guide section changes; discovering such a need is a
  stop-and-replan condition.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

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
  remain specialization facts rather than Model or loss-IR semantics.<span data-glossary-navigation="true"> ([fingerprint glossary](../../../../glossary.md#term-fingerprint); [specialization glossary](../../../../glossary.md#term-specialization); [Model glossary](../../../../glossary.md#term-model); [loss glossary](../../../../glossary.md#loss); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- [`VECTOR`](../../../../glossary.md#term-vector) and `PARALLEL_VECTOR` use the same schema-62 class. Parallel range count, worker
  identity, geometry, offsets, and invocation bounds remain cold facts.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Vector chunks use one subtraction followed by self-multiplication. FMA, reduction, widening,
  reassociation, and masked tails remain excluded; the existing scalar formula owns every tail.<span data-glossary-navigation="true"> ([FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [widening glossary](../../../../glossary.md#term-widening); [reassociation glossary](../../../../glossary.md#term-reassociation); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Schema](../../../../glossary.md#term-schema) 62 is selected only for same-typed [contiguous](../../../../glossary.md#term-contiguous) [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [MSE](../../../../glossary.md#mean-squared-error--mse) `NONE`. Every scalar
  loss stays on schema 58, including `SUM`, [`MEAN`](../../../../glossary.md#term-mean), BFLOAT16, mixed types, categorical losses,
  non-contiguous access, and short domains.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>

## Known limitations

- Only same-typed [contiguous](../../../../glossary.md#term-contiguous) [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [MSE](../../../../glossary.md#mean-squared-error--mse) `NONE` gains SIMD. Every reduction, mixed type,
  BFLOAT16, categorical, and non-contiguous form remains scalar.<span data-glossary-navigation="true"> ([SIMD glossary](../../../../glossary.md#simd); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Preferred species and observed performance are target/JVM-specific. This task establishes the
  fixed recorded Java 26 forms, not a hardware-intrinsic or universal speedup promise.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- Scalar tails remain required for short ranges and arbitrary chunk boundaries; there is no
  masked-tail path.
- CPU 0008I's corrected full 792-class by five-fork scalar-loss performance evidence remains
  missing and owned by CPU 0009. The 0008M scalar hash-preservation check and vector MSE evidence
  do not close or relabel that gap.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [fork glossary](../../../../glossary.md#term-fork-meanings); [loss glossary](../../../../glossary.md#loss); [hash glossary](../../../../glossary.md#term-hash); [vector glossary](../../../../glossary.md#term-vector))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Documentation-focused context: `01a0730d-6446-7150-84ad-328833bdf27a`.

- Documentation profiles: General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Planning](../../../../glossary.md#term-planning), [backend](../../../../glossary.md#backend)-guide, and example-format.
- Reviewed contracts and [owners](../../../../glossary.md#term-owner): `AGENTS.md`, `ARCHITECTURE.md`, current architecture index,
  documentation rules/profiles, Planning Guide, this task, CPU master plan, roadmap, CPU guide,
  glossary, all five changed production owners and their Javadocs, all three new test/evidence
  owners, guarded CPU Gradle evidence-property forwarding, relevant package summaries, and the
  public/shared boundaries named under Architecture impact.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Implementation contexts: `01a072d0-c30a-7821-90ec-3294b221b2d1` and continuation
  `01a072ec-6d5d-7892-b82c-130cfd882597`. Documentation context:
  `01a0730d-6446-7150-84ad-328833bdf27a`.
- Retained evidence root: `/private/tmp/synaptik-cpu-0008m-final-Ad4Cnvnl`. Task-start [commit](../../../../glossary.md#term-commit):
  `34b637dbf519712c45d154a88d6770b69b7ff461`; task-start tree [SHA-256](../../../../glossary.md#term-sha-256):
  `04d49430747cd1991005363c8b4b973f2a42b8a16074a0abed7bc002889cee99`.
- Structural evidence passed for exactly 24 schema-62 classes, keys, and hashes: 12 FLOAT32 and
  12 FLOAT64 identities, 24 complete <span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>`javap` dossiers, and an exactly eight-column dossier CSV.
  The 1,626-record structural manifest SHA-256 is<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest))</span>
  `4612618d4c3c594ca52ea19c61c8e5a7966235cfc14b8f7a32d15f3e11c0e563`; command-record SHA-256
  is `2f49f967f19d523bbb50ecc8cdfd5b2e50c0a357225d472a7b70b1f12e592771`; dossier SHA-256 is
  `824e1a741d2dfc74bcfa34954273f87ee41805ea81cfccb64094ff740bb3f9d3`; combined-`javap`
  SHA-256 is `60c0b467286e0d7fabae28eb27ad64bd09ccea4ecb5ef76454808c86ab91bcf6`; and source-snapshot
  SHA-256 is <span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>`edab5873c40eec28d84b9616436155be2b115db9b5da4d036a089d15dd19e7a4`.
  Inspection found exact typed subtraction followed by self-multiplication, direct stores, typed
  scalar tails, shared <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>[`VECTOR`](../../../../glossary.md#term-vector)/`PARALLEL_VECTOR` [identity](../../../../glossary.md#term-identity-meanings), and no forbidden hot overhead.
- Scalar preservation passed for exactly 792 before and 792 after schema-58 classes. Both
  inventory SHA-256 values are
  `ce863c65922a5de76527eeaf48f7d1d969914765c3450508859846d39354fe2b`; key-map SHA-256 is
  `84eabd309ee018d743f35ead5f90a705723f5d379e2f19680c3039d24233b093`; equality-report
  SHA-256 is `86818e9e555c226a3828cb49e9cac85963176aa7ff27aeb720316cfed32bed65`.
- The authoritative performance command completed in 1 minute 36 seconds. All
  `12 * 5 * 9 = 540` retained pairs, 60 fork medians, and 12 aggregates passed with 4,096
  iterations. Pair ratios ranged from <span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>`0.898334532720123x` to `1.050397665425858x`; fork medians
  ranged from `0.919616703782793x` to `1.010812953112698x`; aggregates ranged from
  `0.924263487813383x` to `1.009408829343164x`. Individual timings ranged from 25,521,209 to
  33,990,208 ns. There were zero threshold or timing-floor violations, retry and discard were
  false, and exact equality checks passed. The 107-record performance manifest SHA-256 is
  `f2e104742250c82ffa28521f8ae390f34c9350d20b5192a0b7ecbdf033c20265`; aggregate SHA-256 is
  `2b3933e69aefc9cf1feded4d973b09e7de80e02162f6a22d6a82f46c0d5b5b52`.
- Final validation after evidence repair passed: the exact focused command ran 164 tests with one
  guard skip and zero failures/errors; the guarded structural run activated and passed all 11
  tests; and the full CPU run passed 139 suites and 730 tests with 21 expected opt-in skips and
  zero failures/errors.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard); [run glossary](../../../../glossary.md#term-run))</span>
- Javadoc generation passed after the `CpuPartitionPreparer` class-Javadoc correction, with 53
  pre-existing unrelated warnings. All five changed production owners now document their
  affected contracts accurately.
- Package/public/shared review: existing package summaries remain accurate at their ownership
  level and need no 0008M-specific edit. No public API, Model/Compiler/Training contract,
  capability-query contract, shared Prepare/Runtime contract, architecture page/ADR/test,
  backend-conformance/integration owner, other backend, dependency, materialization policy,
  tuning policy, or root Java 26 toolchain document changes.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance); [owner glossary](../../../../glossary.md#term-owner); [dependency glossary](../../../../glossary.md#term-dependency); [materialization glossary](../../../../glossary.md#materialization); [toolchain glossary](../../../../glossary.md#term-toolchain))</span>
- Documentation validation in context `01a0730d-6446-7150-84ad-328833bdf27a` passed the prior
  979-link/310-fragment check. The final five-file rerun passed local targets and anchors, unique
  effective headings, balanced fences, final newlines, no carriage returns or trailing
  whitespace, exact arithmetic/status/schema checks, <span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>`git diff --check`,
  `git diff --cached --check`, and empty staging. Exact scope is 24 of the authorized maximum 28
  paths: five production, 14 test/evidence/build, and five documentation/planning, with no path
  outside the task inventory and no tracked evidence artifact.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [artifact glossary](../../../../glossary.md#term-artifact))</span>

## Implementation notes

- Production changes are confined to the five authorized owners. CPU analysis removes generic
  loss vector eligibility and admits only the exact MSE predicate; emission uses direct typed
  preferred-species loads/stores and a scalar tail; specialization advances the compatibility
  envelope and selects schema 62 only for the vector form.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [loss glossary](../../../../glossary.md#loss); [vector glossary](../../../../glossary.md#term-vector); [MSE glossary](../../../../glossary.md#mean-squared-error--mse); [scalar tail glossary](../../../../glossary.md#term-scalar-tail); [specialization glossary](../../../../glossary.md#term-specialization); [schema glossary](../../../../glossary.md#term-schema))</span>
- The three new package-private test owners encode the exact 106-case semantic matrix, 24
  structural dossiers, scalar inventory equality, independently compiled clean-Java oracle, and
  12-row/five-fork performance protocol. CPU Gradle forwards only the three explicitly supplied<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [oracle glossary](../../../../glossary.md#term-test-oracle); [fork glossary](../../../../glossary.md#term-fork-meanings); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
  `synaptik.cpu.vectorMse.*` properties.
- Documentation finalization updates only the five authorized Markdown paths. No Java, test,
  build, architecture, API, conformance, integration, package-summary, or evidence file was
  changed by the documentation context.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Completion summary

- Completed changes: implemented and documented the bounded same-typed contiguous FLOAT32/FLOAT64
  vector and parallel-vector MSE <span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [vector glossary](../../../../glossary.md#term-vector); [MSE glossary](../../../../glossary.md#mean-squared-error--mse))</span>`NONE` increment with [scalar](../../../../glossary.md#term-scalar) tails and selective [schema](../../../../glossary.md#term-schema) 62.
- Files changed or created: see the exact 5 + 18 + 5 task inventory; documentation finalization
  changed only the five authorized Markdown paths.
- Tests and validation: exact structural, scalar-preservation, five-fork performance, focused,
  guarded structural, full CPU, Javadoc, Markdown, status, arithmetic, and scope gates passed as
  recorded above.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [fork glossary](../../../../glossary.md#term-fork-meanings); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: completed for the CPU guide, glossary entry, planning records,
  production/test/build contracts, package summaries, and public/shared no-change boundaries.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: narrow [CPU](../../../../glossary.md#term-cpu)-private strategy and schema explanation only.
- Javadoc review: all five affected production-owner contracts are accurate; generation passed
  after the preparer class-Javadoc correction.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Glossary impact: updated only the existing [`CPU portable route`](../../../../glossary.md#cpu-portable-route) entry; no reusable term or
  heading was added.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
