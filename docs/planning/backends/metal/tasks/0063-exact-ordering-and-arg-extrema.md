# Task 0063: Exact Ordering and Arg [Extrema](../../../../glossary.md#saturation-pole-and-extremum)

## Status

Complete — approved plan review and the external P1 correction preceded implementation
`9931d5f8`, documentation `f7800a26`, remediation `8a74b499`, final source/test correction
`86399d53`, and active-document reconciliation `c80d79c0`. External final cumulative review at
`c80d79c0` returned `APPROVE` with zero P0/P1/P2, confirmed the tracked full-width [INT32](../../../../glossary.md#data-type--datatype) XOR key,
clean worktree, tests, documentation, and counts. Native build/sign/package, packaged [Metal](../../../../glossary.md#term-metal),
[conformance](../../../../glossary.md#term-conformance), [CPU](../../../../glossary.md#term-cpu)-free public [Engine](../../../../glossary.md#term-engine), [Javadoc](../../../../glossary.md#term-javadoc), and architecture [validation](../../../../glossary.md#term-validation) pass.

## Change class

Class C — implementation will widen profile-common [Metal](../../../../glossary.md#term-metal) capability, add exact custom-[kernel](../../../../glossary.md#kernel)
execution, complete native multi-output handling, advance private route [identities](../../../../glossary.md#term-identity-meanings), change public
[Engine](../../../../glossary.md#term-engine) behavior, and update authoritative [backend](../../../../glossary.md#backend) documentation. Use one clean serial implementation
context only after independent review of this plan, then an independent cumulative code, evidence,
documentation, and security review.

## Goal

Implement the maximal deterministic current [Metal](../../../../glossary.md#term-metal) domain for `SORT`, `ARGSORT`, `TOP_K`,
`ARG_MAX`, and `ARG_MIN` through one fixed exact custom-program route. Preserve [Model](../../../../glossary.md#term-model)'s represented-
value, [NaN](../../../../glossary.md#term-nan), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), stability, tie, index, [Shape](../../../../glossary.md#shape), output-role, and [gradient](../../../../glossary.md#term-gradient)-[metadata](../../../../glossary.md#term-metadata) contracts
under both numerical profiles. Select the route before resource creation and use no direct-selector
assumption, timing, [autotuning](../../../../glossary.md#autotuning--model-autotuning), [runtime](../../../../glossary.md#term-runtime) choice, retry, fallback, or [host](../../../../glossary.md#term-host) repair.

## [Scope](../../../../glossary.md#term-scope)

### Exact audited 115-kind [registry](../../../../glossary.md#term-registry) boundary

The current registry has exactly five ordering/arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum) kinds in [scope](../../../../glossary.md#term-scope):

| Wire/kind | [Arity](../../../../glossary.md#term-arity) and attribute wire | Exact [Model](../../../../glossary.md#term-model) role | Pre-cutover [Metal](../../../../glossary.md#term-metal) state | Task-0063 decision |
|---|---|---|---|---|
| 94 `SORT` | 1 input/1 output; 31 `SORT` = `[axis, descending]` | stable represented values, same type/[Shape](../../../../glossary.md#shape)/[gradient](../../../../glossary.md#term-gradient) [metadata](../../../../glossary.md#term-metadata) | non-executable; `DIRECT / MD_SORT`; custom `PENDING / CP_AGGREGATE` | exact custom route |
| 95 `ARGSORT` | 1/1; 31 `SORT` | stable logical indices, fixed [INT64](../../../../glossary.md#data-type--datatype)/no-grad | non-executable; `COMPOSED / MC_INDEX64`; custom pending | exact custom route |
| 96 `TOP_K` | 1/2; 32 `TOP_K` = `[axis, k, largest, sorted]` | values slot 0 plus mandatory INT64 indices slot 1 | non-executable; `COMPOSED / MC_INDEX64`; custom pending | exact custom route for positive K |
| 109 `ARG_MAX` | 1/1; 36 `ARG_EXTREMA` = `[axis, keepDimensions, tiePolicy]` | fixed INT64 [logical index](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)/no-grad | non-executable; `COMPOSED / MC_INDEX64`; custom pending | exact custom route |
| 110 `ARG_MIN` | 1/1; 36 `ARG_EXTREMA` | fixed INT64 logical index/no-grad | non-executable; `COMPOSED / MC_INDEX64`; custom pending | exact custom route |

The existing [schema](../../../../glossary.md#term-schema)-15 validators already require [normalized axis](../../../../glossary.md#normalized-axis) words and [Boolean](../../../../glossary.md#data-type--datatype) flag words;
`TOP_K` additionally requires positive K, and arg-extrema tie wire values are exactly
`FIRST_INDEX=1` and `LAST_INDEX=2`. The image already represents variable [node](../../../../glossary.md#node)-output [cardinality](../../../../glossary.md#term-cardinality),
including both ordered `TOP_K` references. The native decoder/execution [owners](../../../../glossary.md#term-owner) currently retain only
the first output and therefore remain incapable of executing that valid two-output record.
`MetalCapabilityProvider` currently has no ordering branch and rejects any output list whose size is
not one, so all five production capabilities are false. Implementation must handle exact two-output
TOP_K cardinality before the retained single-output gate and must not make another multi-output kind
eligible.

No other registry kind joins this task. Existing comparison/extrema/[scan](../../../../glossary.md#term-cumulative-scan) wires retain their prior
bounded capability and executable state: 20..25 `GT/GE/LT/LE/EQ/NE` are
`DIRECT / MD_PRED`, 26/27 `TENSOR_MIN/MAX` and 30 `CLAMP` are `DIRECT / MD_ARITH`, 28/29
`SCALAR_MIN/MAX` are `COMPOSED / MC_SCALAR`, and 31..34
`REDUCTION_MIN/REDUCTION_MAX/CUM_SUM/CUM_PROD` are `DIRECT / MD_REDUCE`; all use available
`CA_0052` custom routes. [Pooling](../../../../glossary.md#term-pooling) wires 97/98 remain `DIRECT / MD_POOL2D` and 99/100 remain
`COMPOSED / MC_WINDOW3D`, all custom-pending; only maximum-pool wires 97 and 99 have winner
selection, and their generated `ARG_MAX` use does not make a pool [occurrence](../../../../glossary.md#term-occurrence) an ordering
occurrence. [Loss](../../../../glossary.md#loss), [normalization](../../../../glossary.md#term-normalization), statistical reduction, random, recurrent, [convolution](../../../../glossary.md#term-channels-first-convolution-layers), attention,
and every other structural-only row retain their current decisions.

### Exact production domain

Admit each [occurrence](../../../../glossary.md#term-occurrence) under both `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator) only when all of the following hold:

- the sole input and every output have fully static, canonical dense, zero-offset, non-view storage;
  input rank is `1..16`, every input dimension is positive, every output has its exact derived
  canonical Shape, and all output storage is fresh and pairwise non-overlapping with every input and
  other output;<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension))</span>
- before capability returns true, checked `long`/[`uint64_t`](../../../../glossary.md#index-mappings-and-cost-dominance) arithmetic proves that the input element
  count, every output element count, every dimension, canonical element stride, slice count,
  selected-axis extent, K, custom thread count, and one-dimensional grid width are each in<span data-glossary-navigation="true"> ([element stride glossary](../../../../glossary.md#element-stride); [slice glossary](../../../../glossary.md#slice))</span>
  `1..UINT32_MAX`, while every derived logical coordinate, rank, compacted position, and linear
  index is in `0..UINT32_MAX` and strictly below its governing count/extent when used for access.
  Validation occurs before any unsigned-32-bit narrowing; a wider metadata carrier does not widen
  the admitted index domain;<span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing))</span>
- byte widths, byte offsets/spans, K/output products, [metadata](../../../../glossary.md#term-metadata) sizes, and every
  `NSUInteger`/buffer conversion additionally pass the existing checked native-size and allocation
  bounds without wrap;
- legal floating input descriptors may be either no-grad or gradient-bearing; integral and Boolean
  inputs retain their Model-legal no-grad metadata; no capability decision inspects values;<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [Boolean glossary](../../../../glossary.md#data-type--datatype); [Model glossary](../../../../glossary.md#term-model))</span>
- `SORT` and `ARGSORT` accept [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), [INT32](../../../../glossary.md#data-type--datatype), [INT64](../../../../glossary.md#data-type--datatype), and [BOOL](../../../../glossary.md#data-type--datatype) for either direction;
  `SORT` returns the same type and [Shape](../../../../glossary.md#shape) with the exact input [gradient](../../../../glossary.md#term-gradient) flag, while `ARGSORT` returns
  the same Shape as INT64/no-grad;
- `TOP_K` accepts the same six types, both `largest` values, both `sorted` values, and
  `1 <= k <= selectedExtent`; values at slot 0 have the input type and gradient flag, indices at
  slot 1 are INT64/no-grad, and both outputs have the one exact Shape obtained by replacing the
  selected dimension with K;
- `ARG_MAX` and `ARG_MIN` accept FLOAT64, FLOAT32, BFLOAT16, INT32, and INT64, both keep-dimension
  forms, and both tie policies; BOOL is invalid, and the result is always INT64/no-grad with the
  selected dimension removed or replaced by extent one exactly as requested; and
- repeated input [identity](../../../../glossary.md#term-identity-meanings) and arbitrary [fan-out](../../../../glossary.md#term-fan-out) are valid. A `TOP_K` node always computes both
  ordered outputs even when only one is a graph target; either or both roles may be identity-unique
  targets in one compilation, requested again in independent compilations or sessions, reused
  internally, or consumed by later admitted custom-program nodes.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [graph glossary](../../../../glossary.md#graph); [target glossary](../../../../glossary.md#term-target); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>

The fixed [dispatch](../../../../glossary.md#term-dispatch) truth table is:

| Family | One-dimensional thread count | Additional indexed quantities |
|---|---:|---|
| `SORT`, `ARGSORT` | exact input [element](../../../../glossary.md#term-element) count | selected-[axis](../../../../glossary.md#term-axis) [extent](../../../../glossary.md#term-extent) and every destination [rank](../../../../glossary.md#term-rank) |
| `TOP_K` | exact input element count | selected-axis extent, K, selected rank, and compacted output position |
| `ARG_MAX`, `ARG_MIN` | exact output element count, including [scalar](../../../../glossary.md#term-scalar) count one | selected-axis [scan](../../../../glossary.md#term-cumulative-scan) extent and input/output [linear](../../../../glossary.md#term-linear-meanings) [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) |

Every count and extent in the table is in `1..UINT32_MAX`; every coordinate, rank, position, and
linear index is unsigned-32-bit representable and strictly below its governing count/extent when
used for access. Java capability/preparation and native image [validation](../../../../glossary.md#term-validation) enforce the same boundary
before pipeline or [buffer](../../../../glossary.md#term-buffer) creation. The [kernel](../../../../glossary.md#kernel) uses no multi-dimensional flattening or 64-bit
dispatch strategy, and an out-of-domain occurrence fails closed without [MPSGraph](../../../../glossary.md#term-mpsgraph) or [host](../../../../glossary.md#term-host) fallback.

This is the maximal exact route inside the current static [canonical](../../../../glossary.md#term-canonical-form) unsigned-32-bit [Metal](../../../../glossary.md#term-metal) custom
[materialization](../../../../glossary.md#materialization) boundary: there is no arbitrary value subset, direction, axis, [carrier](../../../../glossary.md#term-storage-carrier), positive K,
tie policy, or Shape carve-out within that representable domain. Dynamic, scalar, empty,
over-`UINT32_MAX` dispatch/index geometry, arbitrary-[view](../../../../glossary.md#term-view-meanings), unresolved, negative/zero-[stride](../../../../glossary.md#term-stride),
overlapping, aliased, in-place, and K-zero occurrences remain outside that boundary.

### Exact ordering and selection semantics

The route must implement [Model](../../../../glossary.md#term-model)'s total output policy rather than the [host](../../../../glossary.md#term-host) language or [Metal shading
language](../../../../glossary.md#term-metal-shading-language-msl)'s partial floating comparison:

- For `SORT`, `ARGSORT`, and `TOP_K`, all non-NaNs use numerical order. Ascending/smallest places
  negative infinity first and positive infinity last; descending/largest reverses only that non-NaN
  order. Negative zero is below positive zero in ascending/smallest order and above it after the
  direction reversal.<span data-glossary-navigation="true"> ([negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [Negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Every floating NaN is one equal final class after every non-NaN for both directions. NaN sign,
  payload, and signaling/quiet state do not affect placement, but value-producing operations copy
  the exact selected raw representation.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Equal finite, integral, or Boolean values and multiple NaNs use increasing original logical-axis
  index. This index tiebreak is not direction-reversed. INT32/INT64 use signed mathematical order;
  BOOL uses <span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [Boolean glossary](../../../../glossary.md#data-type--datatype); [axis index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>`false < true`.
- `TOP_K` selects exactly the first K pairs in the requested complete stable order. With
  `sorted=true`, output pairs retain that order. With `sorted=false`, the exact same selected set is
  deterministically reordered by increasing original logical-axis index; values and indices never
  separate.
- For both arg-extrema operations, a NaN is preferred to every non-NaN; multiple NaNs tie. Among
  non-NaNs, negative zero is below positive zero, infinities and finite values use numerical order,
  and signed integers use mathematical order. <span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`FIRST_INDEX` or `LAST_INDEX` selects the exact
  logical-axis coordinate among equal extrema.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>

These are exact, profile-common policies. Ordering and arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum) receive no [DAZ](../../../../glossary.md#term-daz), [FTZ](../../../../glossary.md#term-ftz), [epsilon](../../../../glossary.md#epsilon),
[tolerance](../../../../glossary.md#term-tolerance), [reassociation](../../../../glossary.md#term-reassociation), [canonical](../../../../glossary.md#term-canonical-form)-[NaN](../../../../glossary.md#term-nan), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), approximate-index, or final-only relaxation.

### Fixed custom route and native algorithm

Do not authorize any [MPSGraph](../../../../glossary.md#term-mpsgraph) selector. Installed `MPSGraphSortOps.h`
`sortWithTensor:axis:descending:name:` does not guarantee unconditional stability, NaNs-last,
[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero)/[subnormal](../../../../glossary.md#term-subnormal) order, [represented-bit](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) copying, all six carriers, or every admitted [Shape](../../../../glossary.md#shape);
`argSortWithTensor:axis:descending:name:` explicitly returns [INT32](../../../../glossary.md#data-type--datatype) rather than [Model](../../../../glossary.md#term-model)'s [INT64](../../../../glossary.md#data-type--datatype).
Installed `MPSGraphTopKOps.h` [axis](../../../../glossary.md#term-axis) forms `topKWithSourceTensor:axis:k:name:` and
`bottomKWithSourceTensor:axis:k:name:` do not close index type, stable cutoff ties,
[NaN](../../../../glossary.md#term-nan)/signed-zero, exact-bit, output-order, or `sorted=false` contracts. Installed
`MPSGraphReductionOps.h` `reductionArgMaximumWithTensor:axis:name:` and
`reductionArgMinimumWithTensor:axis:name:` expose neither Model's explicit tie policy nor the
required INT64 and floating-special-value guarantees. Historical Task 0034 therefore remains a
truthful blocked direct-selector record; Task 0063 uses its separately named exact-custom unblock,
not a reinterpretation of that evidence.

Add one Task-0063 custom [Metal](../../../../glossary.md#term-metal) source/header whose comparisons are integer-only:

- classify floating NaNs from raw [exponent](../../../../glossary.md#term-exponent)/fraction fields before key comparison;
- for each non-NaN BFLOAT16, binary32/FLOAT32, or binary64/FLOAT64 raw carrier, map a negative word
  to its bitwise complement and a non-negative word by toggling its sign bit. Unsigned key order is
  then exact numerical order with <span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [FLOAT32 glossary](../../../../glossary.md#term-float32); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>`-0 < +0`, including subnormals and [infinities](../../../../glossary.md#nan-infinity-and-signed-zero);
- map INT32/INT64 signed order by toggling the sign bit and comparing unsigned keys; compare BOOL as
  canonical byte zero/one; and<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- compare the original logical-axis index only after semantic equality. NaNs remain one semantic
  class, and descending/largest reverses the non-NaN value relation without reversing this tie key.<span data-glossary-navigation="true"> ([axis index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>

For `SORT`, `ARGSORT`, and `TOP_K`, [dispatch](../../../../glossary.md#term-dispatch) one one-dimensional grid whose width is the exact input
[element](../../../../glossary.md#term-element) count in `1..UINT32_MAX`. Each [kernel](../../../../glossary.md#kernel) reads only the 32-bit
`thread_position_in_grid.x`; it does not reconstruct a wider identifier from `y`/`z`. The thread
derives its [slice](../../../../glossary.md#slice) and source axis [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), computes that source's unique stable [rank](../../../../glossary.md#term-rank) by counting
strict predecessors under `(semantic order, original index)`, and writes only its unique
destination. `SORT` copies raw [carrier](../../../../glossary.md#term-storage-carrier) bits; `ARGSORT` widens the validated non-negative 32-bit
logical coordinate to INT64 only when storing the public result. `TOP_K` writes nothing when rank is
at least K; otherwise one thread writes both paired outputs, using `rank` for sorted output or the
number of earlier source coordinates whose stable rank is below K for unsorted output. Recomputing
ranks for the unsorted form is permitted; no scratch, allocation, [atomic](../../../../glossary.md#resource-transaction-and-rollback), [race](../../../../glossary.md#term-race), timing choice, or
data-dependent route exists.

For arg [extrema](../../../../glossary.md#saturation-pole-and-extremum), dispatch one one-dimensional grid whose width is the exact positive output element
count at most `UINT32_MAX`. [Scan](../../../../glossary.md#term-cumulative-scan) the validated selected-axis [extent](../../../../glossary.md#term-extent) in increasing 32-bit logical
coordinate, maintain one raw best word and INT64 best index under the NaN/value/tie [invariant](../../../../glossary.md#term-invariant), and
publish only the widened index. All [dimension](../../../../glossary.md#dimension), [stride](../../../../glossary.md#term-stride), slice, rank, count, coordinate, and logical
[linear](../../../../glossary.md#term-linear-meanings)-index arithmetic stays in the prevalidated unsigned-32-bit domain; byte addressing retains
the separate checked native-size contract. No index wraps or narrows from an out-of-range carrier.
Every kernel traversal is [half-open](../../../../glossary.md#term-half-open-interval) (`index < extent`), never computes `extent + 1`, and therefore
terminates at an admitted `UINT32_MAX` extent without increment wrap.

A `TOP_K` custom step binds one input, values output, indices output, and [immutable](../../../../glossary.md#term-immutable) [metadata](../../../../glossary.md#term-metadata) in one
[occurrence](../../../../glossary.md#term-occurrence). Extend the internal decoded-[node](../../../../glossary.md#node) and program-step representations from a first-output
field to the bounded ordered output-reference set already carried by [schema](../../../../glossary.md#term-schema) 15. Validate, account,
allocate, bind, store, consume, and publish every output through used/produced-value checks, type/
Shape/[gradient](../../../../glossary.md#term-gradient) [validation](../../../../glossary.md#term-validation), byte/[device](../../../../glossary.md#term-device)/[alias](../../../../glossary.md#aliasing-and-arena) checks, [target](../../../../glossary.md#term-target) mapping, [lifecycle](../../../../glossary.md#lifecycle) [rollback](../../../../glossary.md#term-rollback), and reuse.
Do not broaden nested MPSGraph steps or another multi-output kind. All validation completes before
resource creation or any device write.

Every [partition](../../../../glossary.md#partition) containing one of the five admitted kinds selects `CUSTOM_PROGRAM` only. Candidate
generation must never offer MPSGraph for such an occurrence, including [singleton](../../../../glossary.md#term-singleton) SORT. A custom
program may compose these [kernels](../../../../glossary.md#kernel) with already approved custom or nested-MPSGraph steps while
preserving [topological](../../../../glossary.md#term-topological-order) value order and both `TOP_K` values.

### Complete-domain proof

Authorization is source-structural and algebraic, never a sampled [device](../../../../glossary.md#term-device)-output [inference](../../../../glossary.md#term-inference):

1. Prove the raw floating classifier partitions NaN from every non-NaN encoding. Prove separately
   for each sign partition that complement/sign-bit-toggle is a monotone bijection over all
   non-NaN encodings and joins the partitions at <span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [partition glossary](../../../../glossary.md#partition); [bijection glossary](../../../../glossary.md#term-bijection))</span>`-0 < +0`; direction reversal affects only that
   order. Enumerate all 65,536 BFLOAT16 words as a complete machine check and cover binary32/64
   exponent/fraction partitions algebraically.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [exponent glossary](../../../../glossary.md#term-exponent))</span>
2. Prove sign-bit toggle is a bijection from INT32/INT64 signed order to unsigned-key order and
   exhaust BOOL. Retain adversarial native corpora at signed extrema and zero/one boundaries.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
3. Prove `(semantic class/key, increasing logical index)` is a strict total order, every source has
   one distinct predecessor count in `0..N-1`, and distinct threads therefore have distinct writes.
   Prove raw-copy exactness, both TOP_K selected-set/output-order formulas, and the arg-extrema loop
   invariant for both operations and tie policies.<span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant))</span>
4. Prove every dimension, element count, stride, slice count, selected extent, K, rank, compacted
   position, linear index, thread count, and grid width is checked in a wider carrier and is at most<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension); [element glossary](../../../../glossary.md#term-element); [stride glossary](../../../../glossary.md#term-stride); [slice glossary](../../../../glossary.md#slice); [extent glossary](../../../../glossary.md#term-extent); [rank glossary](../../../../glossary.md#term-rank); [linear glossary](../../../../glossary.md#term-linear-meanings))</span>
   `UINT32_MAX` before narrowing or resource creation; every access index is strictly below its
   governing bound. Prove the one-dimensional kernel never derives a wider logical index and uses
   only half-open traversals that terminate at the exact limit without <span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing); [kernel glossary](../../../../glossary.md#kernel); [logical index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [half-open glossary](../../../../glossary.md#term-half-open-interval))</span>`extent + 1` or increment
   wrap. Exercise exact `UINT32_MAX` acceptance and `UINT32_MAX + 1`, product-overflow,
   stride-overflow, and byte/span rejection in allocation-free Java/native validation tests.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [validation glossary](../../../../glossary.md#term-validation))</span>
5. Cross-check an independently written Java raw-word oracle against real-device kernels for all
   carriers and operation modes. Include NaN signs/payloads/signaling forms, both zeros, subnormals,
   normals, infinities, repeated values, INT extrema, and BOOL. Exhaust all small finite sequences
   over a boundary alphabet to exercise stability, cutoff, unsorted compaction, and ties; use
   representative rank/axis and safely materializable near-boundary cases only as implementation
   corroboration, not scope authorization.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [kernels glossary](../../../../glossary.md#kernel); [operation glossary](../../../../glossary.md#operation); [subnormals glossary](../../../../glossary.md#term-subnormal); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [finite glossary](../../../../glossary.md#term-finite); [axis glossary](../../../../glossary.md#term-axis); [scope glossary](../../../../glossary.md#term-scope))</span>
6. Independently review the exact generated Metal source, fixed entrypoint selection, metadata,
   unsigned-32-bit logical-index closure, integer widths, byte-address derivation, output
   disjointness, absence of floating comparisons and hidden route choices, and the proof-to-source
   correspondence.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [metadata glossary](../../../../glossary.md#term-metadata))</span>

No benchmark or timing result is an acceptance gate or route input. If the implementation cannot
supply this proof for one [carrier](../../../../glossary.md#term-storage-carrier) or form, remove that entire carrier/form from a newly reviewed
plan before production edits rather than admitting a sampled subset.

### [Compiler](../../../../glossary.md#term-compiler) [gradients](../../../../glossary.md#term-gradient) and public callers

[Model](../../../../glossary.md#term-model) and Compiler remain unchanged. Audit `TensorSortExpressions`, `TensorTopKExpressions`,
`TensorArgExtremaExpressions`, their result carriers, [inference](../../../../glossary.md#term-inference), `AutogradPreflight`,
`OrderingGradientRules`, `ReductionGradientRules`, [pooling](../../../../glossary.md#term-pooling) callers, and public [Tensor](../../../../glossary.md#tensor) overloads.
Task 0063 admits forward occurrences with the exact [metadata](../../../../glossary.md#term-metadata) above but claims no generated ordering
backward route:

- floating `SORT` backward constructs matching [INT64](../../../../glossary.md#data-type--datatype) `ARGSORT` and replacement
  `SCATTER_ELEMENTS`;
- `TOP_K` values backward consumes the original INT64 indices and replacement
  `SCATTER_ELEMENTS`; and
- ARGSORT, TOP_K indices, ARG_MIN, and ARG_MAX are non-[differentiable](../../../../glossary.md#derivative-and-chain-rule).

[Metal](../../../../glossary.md#term-metal)'s current `SCATTER_ELEMENTS` domain requires [canonical](../../../../glossary.md#term-canonical-form) [INT32](../../../../glossary.md#data-type--datatype) indices, so both generated
floating backward [graphs](../../../../glossary.md#graph) remain unsupported as complete partitions. Do not widen scatter or claim
pooling gradients in this task. Prove [CPU](../../../../glossary.md#term-cpu)-free public [Engine](../../../../glossary.md#term-engine) forward execution for legal no-grad and
gradient-metadata inputs, both profiles, every family/output role, two-output [publication](../../../../glossary.md#term-publication), nested
consumers, [fan-out](../../../../glossary.md#term-fan-out), retained executable reuse, and independent sessions. With no CPU registered,
[compile](../../../../glossary.md#compile) an allocation-free static [descriptor](../../../../glossary.md#term-descriptor) whose logical count or derived index [extent](../../../../glossary.md#term-extent) is
`UINT32_MAX + 1` and prove Metal declines [ownership](../../../../glossary.md#term-ownership-meanings) before native invocation or input/output
allocation. Generated ordering backward, pooling, K-zero, empty/dynamic/[view](../../../../glossary.md#term-view-meanings)/[alias](../../../../glossary.md#aliasing-and-arena), wrong-role,
malformed, and every other over-limit graph must fail before resources/native invocation.

### Catalog, [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), and [lifecycle](../../../../glossary.md#lifecycle)

- Mark only wires 94, 95, 96, 109, and 110 structurally executable: `88/27 -> 93/22`.
  Production capability becomes `70/45 -> 75/40`.
- Keep the [MPSGraph](../../../../glossary.md#term-mpsgraph) catalog exactly `75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE`: SORT remains
  `DIRECT / MD_SORT`; the other four remain `COMPOSED / MC_INDEX64`. Catalog structure is not
  selector authorization.
- Add `CA_0063`, change exactly those five custom rows from pending to available, and produce custom
  catalog `52 AVAILABLE / 63 PENDING / 0 UNAVAILABLE-WITH-PROOF`.
- Keep schema 15, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, type wires `1..6`, [operation](../../../../glossary.md#operation) wires `1..115`, attribute wires `0..41`, route
  wires `1..3`, the fixed image grammar, and thirteen exports. Advance workload-signature,
  exact-default-policy, candidate-schema, compatibility-schema, route-policy, and codec identities
  together <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [codec glossary](../../../../glossary.md#term-codec))</span>`18 -> 19`; identity 18 fails closed with no alias, migration reader, or compatibility
  shim.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Preserve cold deterministic route selection, stable value/feed/target order, transactional
  finalization, synchronous invocation, direct and internal output liveness, leases, rollback and
  suppression, reuse, concurrency/session isolation, close ordering, trace identity, and the
  profile-free Runtime/Trace boundary.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [synchronous glossary](../../../../glossary.md#term-synchronous); [liveness glossary](../../../../glossary.md#term-liveness); [leases glossary](../../../../glossary.md#term-lease); [rollback glossary](../../../../glossary.md#term-rollback); [concurrency glossary](../../../../glossary.md#term-concurrency); [trace glossary](../../../../glossary.md#trace); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Provable exact domain versus blockers

The exact domain is the complete static [canonical](../../../../glossary.md#term-canonical-form) positive-[rank](../../../../glossary.md#term-rank)/positive-[extent](../../../../glossary.md#term-extent) [matrix](../../../../glossary.md#term-matrix-meanings) whose counts
and extents are representable in `1..UINT32_MAX` and whose derived [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)/indices are
representable in `0..UINT32_MAX` as specified above. Its proof is constructive and independent of
[opaque](../../../../glossary.md#term-opaque) [MPSGraph](../../../../glossary.md#term-mpsgraph) ordering behavior.

Remain explicitly blocked:

- [scalar](../../../../glossary.md#term-scalar), dynamic, expression, zero-extent, arbitrary affine/[view](../../../../glossary.md#term-view-meanings)/overlapping/in-place geometry;
- any input/output element count, dimension, stride, slice count, selected extent, K, rank, compacted
  position, logical linear index, or thread/grid count above <span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [dimension glossary](../../../../glossary.md#dimension); [stride glossary](../../../../glossary.md#term-stride); [slice glossary](../../../../glossary.md#slice); [linear glossary](../../../../glossary.md#term-linear-meanings))</span>`UINT32_MAX`, even when `long`,
  [`uint64_t`](../../../../glossary.md#index-mappings-and-cost-dominance), `NSUInteger`, or the byte span can represent it;
- [Model](../../../../glossary.md#term-model)-valid `TOP_K(k=0)`, because the current schema/native custom materialization contract is
  positive and this task adds no zero-dispatch/publication route;<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [materialization glossary](../../../../glossary.md#materialization); [dispatch glossary](../../../../glossary.md#term-dispatch); [publication glossary](../../../../glossary.md#term-publication))</span>
- BOOL arg extrema, which Model rejects; user comparators, named axes, unstable sorting, partial or
  percentage K, kth-value, search/partition, lexicographic ordering, and hidden outputs;<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [axes glossary](../../../../glossary.md#term-axis); [partition glossary](../../../../glossary.md#partition))</span>
- any direct MPSGraph ordering/selection/arg-extrema selector, selector-result cast/repair, host
  sorting or validation, sampled-value authorization, retry, fallback, timing, autotuning, cache
  winner, or performance claim;<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [validation glossary](../../../../glossary.md#term-validation); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [cache glossary](../../../../glossary.md#term-cache))</span>
- generated SORT/TOP_K backward and maximum-pool forward/backward until their independent complete
  scatter/window routes exist; and
- every non-ordering structural-only [operation](../../../../glossary.md#operation) and resumed Task-0053 production.

## Non-goals

No [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), public [Tensor](../../../../glossary.md#tensor), Config, [Planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Trace](../../../../glossary.md#trace), or [Engine](../../../../glossary.md#term-engine) [API](../../../../glossary.md#term-application-programming-interface-api) change;
no [schema](../../../../glossary.md#term-schema) field, wire, [ABI](../../../../glossary.md#term-application-binary-interface-abi) export, general multi-output [MPSGraph](../../../../glossary.md#term-mpsgraph) support, arbitrary [layout](../../../../glossary.md#layout), dynamic
[binding](../../../../glossary.md#term-binding), empty-domain [kernel](../../../../glossary.md#kernel), 64-bit or multi-dimensional custom [dispatch](../../../../glossary.md#term-dispatch), [pooling](../../../../glossary.md#term-pooling), scatter
[widening](../../../../glossary.md#term-widening), [host](../../../../glossary.md#term-host) calculation, benchmark, or new public option. No direct selector is rehabilitated.
No historical blocked result is rewritten.

## Contracts

- [`ARCHITECTURE.md` — Core lifecycle and Core invariants](../../../../../ARCHITECTURE.md#core-lifecycle)
  — Model owns exact profile-indexed meaning; route choice is cold and Runtime/Trace stay
  profile-free.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Trace glossary](../../../../glossary.md#trace))</span>
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  — ordering, selected values, indices, and ties receive no accelerator relaxation.<span data-glossary-navigation="true"> ([accelerator glossary](../../../../glossary.md#term-accelerator))</span>
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal))</span>
  and [Numerical-profile backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity)<span data-glossary-navigation="true"> ([backend identity glossary](../../../../glossary.md#backend-identity--backendid))</span>
  — Metal owns truthful occurrence capability, fixed routes, native state, and private identities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>.
- [Model sort/argsort](../../../modules/model/tasks/0019c-sort-and-argsort.md),
  [top-K](../../../modules/model/tasks/0019c1-top-k-values-and-indices.md), and
  [shared arg extrema](../../../modules/model/tasks/0018u1-integral-reductions-and-arg-min-normalization.md)<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
  own the exact operation semantics. <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>[Compiler ordering gradients](../../../modules/compiler/tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [gradients glossary](../../../../glossary.md#term-gradient))</span>
  owns formula topology; <span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>[CPU ordering](../../cpu/tasks/0006c-portable-stable-ordering-and-selection.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
  is an exact precedent, not Metal authorization.

If implementation needs a new result freedom, a wider or different [dispatch](../../../../glossary.md#term-dispatch)/index domain,
[opaque](../../../../glossary.md#term-opaque)-selector assumption, [runtime](../../../../glossary.md#term-runtime)/value-dependent route, scratch allocation, shared [lifecycle](../../../../glossary.md#lifecycle)/[API](../../../../glossary.md#term-application-programming-interface-api)
change, [schema](../../../../glossary.md#term-schema)/[ABI](../../../../glossary.md#term-application-binary-interface-abi) field, or wider scatter/[pooling](../../../../glossary.md#term-pooling) [ownership](../../../../glossary.md#term-ownership-meanings), stop for a new independently reviewed
plan.

## Dependencies and integration

- Depends on: Task 0062 Complete through `06c57844`; current Model ordering/top-K/arg-extrema and
  Compiler gradient contracts; Task 0003's existing <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [Compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient))</span>`1..UINT32_MAX` custom thread/grid contract;
  and completed Metal six-carrier transfer, INT64 materialization, schema-15, custom-program,
  catalog, identity, lifecycle, and public Engine foundations.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [carrier glossary](../../../../glossary.md#term-storage-carrier); [INT64 glossary](../../../../glossary.md#data-type--datatype); [materialization glossary](../../../../glossary.md#materialization); [schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Supersedes only the future-route conclusion of historical Task 0034 by choosing its exact custom
  unblock. Task 0034 remains Blocked as a direct-MPSGraph task and its evidence remains immutable.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Conflicts with: every concurrent Metal capability/schema/native/custom-source/multi-output/route/
  package/shared-document scope and resumed Task-0053 production.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None.
- Common base revision: `b369b8aa`.
- Integration order: independent plan review, then one serial proof/kernel/multi-output/capability/
  preparation/execution/test/documentation cutover; no production edit before review.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>
- Integration validation: native build/sign/package/verification, focused Metal, conformance,
  CPU-free public Engine, Javadoc/architecture/docs, exact count/identity/export audits, and
  independent cumulative Class C review.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): Task-0063 coordinator.

## Files and symbols

Expected implementation [owners](../../../../glossary.md#term-owner):

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNegPartitionPreparer`,
  `MetalNegPreparationPlan`, `MetalNativeApi`, `MetalOperationRouteCatalog`, custom route candidate/
  tuning/codec identity owners, and affected package/type Javadocs;<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- native foundation decoded-node/program-step/output validation and binding, plus one Task-0063
  integer-ordering kernel source/header and the native package guide;<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [validation glossary](../../../../glossary.md#term-validation); [binding glossary](../../../../glossary.md#term-binding); [kernel glossary](../../../../glossary.md#kernel))</span>
- focused capability/catalog/schema/malformed/prepared/native/comparator-proof tests, including
  allocation-free <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>`UINT32_MAX`/`UINT32_MAX + 1` [dispatch](../../../../glossary.md#term-dispatch)/index boundary and [overflow](../../../../glossary.md#term-overflow) parity,
  `MetalNegCapabilityPartitionConformanceTest`, and a dedicated [CPU](../../../../glossary.md#term-cpu)-free
  `EngineOrderingMetalIntegrationTest` for observable output roles, over-limit pre-resource
  rejection, and lifecycle; and<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Metal/backend/root architecture and capability documentation, targeted API/glossary status,
  ADR identity claims if current, this brief, master plan, and roadmap.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

The original [planning](../../../../glossary.md#term-planning)-only revision changed exactly this brief, the [Metal](../../../../glossary.md#term-metal) master plan, and the
roadmap. Implementation `9931d5f8` performs the reviewed production, native, and test cutover;
the subsequent documentation revision updates the authorized explanatory and status surfaces.

## Acceptance criteria

1. The five-wire audit and tests retain exact kind, arity, output role/order, attribute, type, Shape,
   layout, metadata, axis, direction, K, tie, index-width, unsigned-32-bit dispatch/index cap,
   byte/span overflow, and profile decisions. Every excluded occurrence rejects before resource
   creation.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [metadata glossary](../../../../glossary.md#term-metadata); [axis glossary](../../../../glossary.md#term-axis); [dispatch glossary](../../../../glossary.md#term-dispatch); [overflow glossary](../../../../glossary.md#term-overflow))</span>
2. The integer-only raw comparator and algorithm proof cover the complete admitted carrier domains;
   BFLOAT16/BOOL exhaustive checks, floating/integer partition proofs, small-sequence exhaustive
   checks, adversarial real-device corpora, and independent source review agree exactly.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [partition glossary](../../../../glossary.md#partition); [device glossary](../../../../glossary.md#term-device))</span>
3. SORT/ARGSORT stability, NaNs-last, signed-zero and exact-copy behavior; TOP_K selected-set and
   both output orders; and arg-extrema NaN preference/ties are exact on every admitted axis/rank.<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [NaN glossary](../../../../glossary.md#term-nan); [rank glossary](../../../../glossary.md#term-rank))</span>
4. Every logical coordinate/index is checked in a wider carrier before narrowing, is unsigned-
   32-bit representable, and is strictly below its governing access bound; every positive
   count/extent/dispatch quantity remains in <span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [narrowing glossary](../../../../glossary.md#term-narrowing); [extent glossary](../../../../glossary.md#term-extent))</span>`1..UINT32_MAX`. Half-open exact-limit termination,
   one-past-limit rejection, and overflow rejection have Java/native parity and occur before
   pipeline, buffer, or other resource creation.<span data-glossary-navigation="true"> ([Half-open glossary](../../../../glossary.md#term-half-open-interval); [buffer glossary](../../../../glossary.md#term-buffer))</span>
5. One TOP_K step validates, materializes, writes, retains, consumes, and publishes both ordered
   outputs. Values and INT64 indices remain paired under identity-unique target subsets, nesting,
   fan-out, reuse, repeated role selection across independent compilations and sessions, rollback,
   and close; malformed second-output state cannot hide.<span data-glossary-navigation="true"> ([INT64 glossary](../../../../glossary.md#data-type--datatype); [identity glossary](../../../../glossary.md#term-identity-meanings); [target glossary](../../../../glossary.md#term-target); [fan-out glossary](../../../../glossary.md#term-fan-out); [rollback glossary](../../../../glossary.md#term-rollback))</span>
6. Every admitted [occurrence](../../../../glossary.md#term-occurrence) has only `CUSTOM_PROGRAM`; direct MPSGraph, host repair, retry,
   fallback, timing, autotuning, multi-dimensional/64-bit dispatch, and value-dependent selection
   are absent.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
7. Counts become `75/40` capability, `93/22` structural, `75/35/5` [MPSGraph](../../../../glossary.md#term-mpsgraph), and `52/63/0` custom;
   schema 15, ABI 5, and thirteen exports remain; all backend-local identities are 19 and identity
   18 rejects.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [backend glossary](../../../../glossary.md#backend))</span>
8. CPU-free public Engine proves both profiles, six ordering/top-K carriers, five arg-extrema
   carriers, all output roles and representative axes/directions/K/forms/ties, exact raw values and
   INT64 indices, input preservation, composition, reuse, and independent sessions with no skip.
   An allocation-free one-past-limit static graph, other excluded domains, and generated
   SORT/TOP_K backward reject without native invocation or resource allocation.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [graph glossary](../../../../glossary.md#graph))</span>
9. Native/package, complete Metal, Metal conformance, dedicated Engine, Javadoc, architecture,
   Markdown/link/diff checks pass; independent cumulative Class C review has no unresolved
   P0/P1/P2 finding.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## [Validation](../../../../glossary.md#term-validation) after implementation

```bash
./native/metal-macos-arm64/build.sh
codesign --force --sign - --identifier io.github.pho001.synaptik.metal.foundation native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/package-local.sh native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/verify-package.sh native/metal-macos-arm64/build/package-v1/macos-arm64
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineOrderingMetalIntegrationTest*'
./gradlew :backends:metal:javadoc :testing:architecture-tests:test
git diff --check
```

No timing or benchmark command is authorized. No full repository build is planned absent an
unexpected shared [API](../../../../glossary.md#term-application-programming-interface-api)/[dependency](../../../../glossary.md#term-dependency)/build change; the complete packaged [Metal](../../../../glossary.md#term-metal) suite, [conformance](../../../../glossary.md#term-conformance),
public [Engine](../../../../glossary.md#term-engine), and architecture gates are stronger for this [backend](../../../../glossary.md#backend)-local cutover.

## Documentation and review impact

The implementation must update every current capability/count/[identity](../../../../glossary.md#term-identity-meanings)/ordering statement and
describe the route as the exact profile-common **unsigned-32-bit-bounded** custom domain, never as
all positive static extents. Public/[backend](../../../../glossary.md#backend)/native/[Javadoc](../../../../glossary.md#term-javadoc) wording must name the
`1..UINT32_MAX` count/[extent](../../../../glossary.md#term-extent) and `0..UINT32_MAX` [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)/index boundary, pre-resource failure,
two-output [publication](../../../../glossary.md#term-publication), direct-selector exclusion, [gradient](../../../../glossary.md#term-gradient) boundary, and remaining blockers.
Preserve Task 0034 as historical direct-selector evidence. Independent plan review precedes every
production edit. Independent final review must inspect the full plan-to-source proof, [kernel](../../../../glossary.md#kernel) integer
semantics, cap parity and boundary negatives, [metadata](../../../../glossary.md#term-metadata) and byte/span [overflow](../../../../glossary.md#term-overflow) safety, multi-output
[lifecycle](../../../../glossary.md#lifecycle), route exclusivity, negative-domain evidence, public [Engine](../../../../glossary.md#term-engine) behavior, docs, and changed-
path [scope](../../../../glossary.md#term-scope). Passing worker execution evidence may be reused unless remediation changes executable
behavior.

## Implementation and proof result

Implementation `9931d5f8` delivers all five wires through the fixed custom-program route.
Remediation `8a74b499` hardens native/raw-boundary and same-sign [INT32](../../../../glossary.md#data-type--datatype) [regression](../../../../glossary.md#term-regression) evidence. Final
source/test correction `86399d53` retains every INT32 magnitude bit with the literal expression
`(word ^ 0x0000000080000000ul) & 0x00000000fffffffful` and restores the Task-0052 custom-only
direct-[MPSGraph](../../../../glossary.md#term-mpsgraph) [guard](../../../../glossary.md#term-guard). The source uses only [unsigned integer](../../../../glossary.md#term-unsigned-integer) raw-word classification and keys:
floating [exponent](../../../../glossary.md#term-exponent)/fraction tests separate every [NaN](../../../../glossary.md#term-nan) encoding, sign complement/sign-bit toggle maps
each non-NaN sign [partition](../../../../glossary.md#partition) monotonically and joins at negative-zero-before-positive-zero,
signed-integer sign-bit toggle maps signed order to unsigned order, and [BOOL](../../../../glossary.md#data-type--datatype) uses its complete
`0/1` domain. The stable tuple is `(NaN class or numerical key, increasing logical index)`;
predecessor counting therefore assigns one distinct [rank](../../../../glossary.md#term-rank) in `0..N-1` to every source [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
and one distinct destination writer. Direction reversal changes only non-NaN key order. TOP_K
uses rank `< K` for its selected set and either rank or the count of earlier selected coordinates
for output position. Arg [extrema](../../../../glossary.md#saturation-pole-and-extremum) maintain the best-so-far [invariant](../../../../glossary.md#term-invariant) with NaN preference and
explicit first/last replacement.

`MetalOrderingComparatorProofTest` enumerates all 65,536 [BFLOAT16](../../../../glossary.md#term-bfloat16) words, exhausts BOOL, checks the
[binary32](../../../../glossary.md#floating-point-precision-and-rounding)/[binary64](../../../../glossary.md#floating-point-precision-and-rounding) exponent/fraction partitions, verifies signed-key boundaries and deterministic
domain [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), and exhausts boundary-alphabet sequences through length four for stable ranks,
cutoffs, compaction, and ties. `MetalOrderingNativeTest` and
`EngineOrderingMetalIntegrationTest` independently compute raw-word expectations and execute all
six ordering/top-K carriers, five arg-extrema carriers, both profiles, directions, largest/smallest,
sorted/compacted forms, both tie policies, signed extrema, zeros, [subnormals](../../../../glossary.md#term-subnormal), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), repeated
values, and signed/[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication)/signaling NaNs on the real [device](../../../../glossary.md#term-device). Exact input and selected raw words,
[INT64](../../../../glossary.md#data-type--datatype) logical indices, [target](../../../../glossary.md#term-target) subsets, nesting, [fan-out](../../../../glossary.md#term-fan-out), retained reuse, and independent sessions
are observed.

Java capability and Java/native image [validation](../../../../glossary.md#term-validation) enforce ranks `1..16` and check every [dimension](../../../../glossary.md#dimension),
product, [canonical](../../../../glossary.md#term-canonical-form) [stride](../../../../glossary.md#term-stride), [slice](../../../../glossary.md#slice)/output count, selected [extent](../../../../glossary.md#term-extent), K, and grid width in wider carriers
before [narrowing](../../../../glossary.md#term-narrowing). The admitted positive values are at most `UINT32_MAX`; derived coordinates and
indices remain at most `UINT32_MAX`, and every access is strictly below its governing count.
The one-dimensional [kernels](../../../../glossary.md#kernel) use only `thread_position_in_grid.x` and [half-open](../../../../glossary.md#term-half-open-interval) loops, so an
exact-limit traversal never computes `extent + 1` or increments a loop variable from
`UINT32_MAX`. Byte offsets multiply into 64-bit `ulong`, while Java/native byte/span validation
remains independent. Allocation-free tests
accept the exact limit and reject one-past/product [overflow](../../../../glossary.md#term-overflow); excluded [Engine](../../../../glossary.md#term-engine) [graphs](../../../../glossary.md#graph) and generated
SORT/TOP_K backward requests fail before preparation.

The [native bridge](../../../../glossary.md#term-native-bridge) was rebuilt, ad-hoc signed with
`io.github.pho001.synaptik.metal.foundation`, packaged, and independently verified. The complete
packaged [Metal](../../../../glossary.md#term-metal) suite, Metal [conformance](../../../../glossary.md#term-conformance) suite, dedicated four-case [CPU](../../../../glossary.md#term-cpu)-free Engine integration,
Metal [Javadoc](../../../../glossary.md#term-javadoc), and architecture tests pass. Counts are exactly `75/40` capability, `93/22`
structural execution, `75/35/5` MPSGraph catalog, and `52/63/0` custom catalog. [Schema](../../../../glossary.md#term-schema) 15, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5,
type wires `1..6`, [operation](../../../../glossary.md#operation) wires `1..115`, attribute wires `0..41`, route wires `1..3`, image
grammar, and thirteen exports are unchanged. All [backend](../../../../glossary.md#backend)-local [identities](../../../../glossary.md#term-identity-meanings) are 19 and [identity](../../../../glossary.md#term-identity-meanings) 18
fails closed. No benchmark or timing result participates in the route.
