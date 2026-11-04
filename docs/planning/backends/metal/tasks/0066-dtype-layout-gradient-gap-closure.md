# Task 0066: [Dtype](../../../../glossary.md#data-type--datatype), [Layout](../../../../glossary.md#layout), and [Gradient](../../../../glossary.md#term-gradient) Gap Closure

## Reader context

This closed descriptor/representation gaps inside existing kinds rather than adding mathematics. Model describes carrier, shape and gradient eligibility; Compiler emits actual backward operations; Metal's provider/preparer and independent native preflight authenticate physical mappings. A logical affine value can share source addressing, while a materialized value has a distinct destination.

Movement preserves bits across mappings; CAST obeys its conversion rule. The six-carrier/36-pair inventory and blocked adjoints below are the selected historical domain. Neither true-kind counts nor a forward gradient flag proves complete present-day derivatives.

- [Current logical and physical authentication](../../../../backend-guide/metal-backend.md#physical-authentication-and-fixed-routes) — Explains the representation distinction driving the cutover.
- [Current casts carriers and derivatives](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains) — Locates subsequent expansion.

## Status

Complete — independently approved implementation
Approved plan `e0ec3d2360b0b7ab1119ce0613a612bf3e18b218` landed through implementation
`0b88f897`, documentation `946fff39`, [alias](../../../../glossary.md#aliasing-and-arena)-safety remediations `b7d2616f`/`07641c51`, and
[scalar](../../../../glossary.md#term-scalar)/window contract remediations `31eb77c5`/`4b426898`. The cutover preserves all named blockers
and fixed [schema](../../../../glossary.md#term-schema)/[ABI](../../../../glossary.md#term-application-binary-interface-abi)/export boundaries while completing the exact selected [occurrence](../../../../glossary.md#term-occurrence), route,
physical-[layout](../../../../glossary.md#layout), generated-[gradient](../../../../glossary.md#term-gradient), package, and documentation [scope](../../../../glossary.md#term-scope). Independent cumulative
quality and security review at `4b426898` returned `APPROVE` with zero P0/P1/P2.

## Change class

Class C — the implementation broadens [occurrence](../../../../glossary.md#term-occurrence) capability inside 33 already-production-true
[operation](../../../../glossary.md#operation) kinds, replaces ten pending movement rows with fixed custom implementations, closes
first-order movement [gradients](../../../../glossary.md#term-gradient) that generate only already-exact operations, and advances every
[backend](../../../../glossary.md#backend)-local [identity](../../../../glossary.md#term-identity-meanings). It adds no operation kind and does not reopen the semantic families
represented by the 32 production-false rows.

## Goal

Close the maximal currently reachable [Metal](../../../../glossary.md#term-metal) [dtype](../../../../glossary.md#data-type--datatype), affine-[layout](../../../../glossary.md#layout), exact-movement, and first-order
[gradient](../../../../glossary.md#term-gradient) gaps after Task 0065. Use integer/bit-defined custom [kernels](../../../../glossary.md#kernel) rather than [opaque](../../../../glossary.md#term-opaque) floating
selectors so all six represented carriers can move without changing raw payloads, all 36 current
`CAST` pairs match the [Model](../../../../glossary.md#term-model) [oracle](../../../../glossary.md#term-test-oracle), [floating classification](../../../../glossary.md#floating-classification) and mixed floating `WHERE` work for
[FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16), [INT64](../../../../glossary.md#data-type--datatype) index roles match [INT32](../../../../glossary.md#data-type--datatype) roles, [scalar](../../../../glossary.md#term-scalar) [Shapes](../../../../glossary.md#shape) compose, and exact
replacement/[view](../../../../glossary.md#term-view-meanings)/window adjoints [run](../../../../glossary.md#term-run) through their generated [graphs](../../../../glossary.md#graph).

Make every selected [occurrence](../../../../glossary.md#term-occurrence) use one deterministic `CUSTOM_PROGRAM` route chosen from authenticated
[metadata](../../../../glossary.md#term-metadata) before resource creation. Preserve the existing external transfer contract and distinguish
logical affine geometry from run-owned dense [materialization](../../../../glossary.md#materialization). Route choice may use no payload, timing,
[autotuning](../../../../glossary.md#autotuning--model-autotuning), [cache](../../../../glossary.md#term-cache) winner, retry, fallback, or [host](../../../../glossary.md#term-host) repair.

Do not duplicate a later full 115-kind semantic audit. This task inventories all 115 rows only far
enough to prove that no production-false row is dtype/layout/gradient infrastructure. The [operation](../../../../glossary.md#operation)-
kind totals therefore remain fixed while occurrence domains, custom catalog state, and [identity](../../../../glossary.md#term-identity-meanings)
change.

## [Scope](../../../../glossary.md#term-scope)

### Exact post-Task-0065 115-kind boundary

The [registry](../../../../glossary.md#term-registry) remains exactly wires `1..115`. Current structural execution is `101 true / 14 false`,
and current production capability is `83 true / 32 false`.

The 14 structurally non-executable rows are exactly:

- wire 35 `SCALED_DOT_PRODUCT_ATTENTION`;
- wires 55 and 64 [`EXP`](../../../../glossary.md#term-expx) and [`SIGMOID`](../../../../glossary.md#term-sigmoid);
- wires 86..93 `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`,
  `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`, `BATCH_NORM_INFERENCE`,
  `BATCH_NORM_TRAINING`, `LAYER_NORM`, `RMS_NORM`, [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax), and `LOG_SOFTMAX`; and
- wires 103..105 `RNN_TANH`, `GRU_RESET_AFTER`, and [`LSTM`](../../../../glossary.md#term-long-short-term-memory-lstm).

The other 18 production-false rows are structurally executable only:

- wires 38 and 50 `TENSOR_POW` and `SCALAR_POW`;
- wires 53, 54, 56..59, 65..68 [`LOG`](../../../../glossary.md#term-log), `LOG1P`, `EXPM1`, [`ERF`](../../../../glossary.md#term-erf), `SQRT`, `RSQRT`,
  [`TANH`](../../../../glossary.md#term-tanh), [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu), `GELU_TANH_APPROXIMATION`, and `SILU`;
- wire 70 `SCATTER_ADD`; and
- wires 111..115 `LOG_SUM_EXP`, [`VARIANCE`](../../../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, and
  `L2_NORM`.

These 32 rows remain false. They require missing transcendental/recursive, aggregate/additive,
[normalization](../../../../glossary.md#term-normalization)/[loss](../../../../glossary.md#loss), attention, or recurrent/[BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) semantics. [Schema](../../../../glossary.md#term-schema) vocabulary, a movement [kernel](../../../../glossary.md#kernel),
or a selected no-work value does not resolve those blockers. Task 0066 changes neither their
catalog entries nor their capability tests.

The selected set is exactly these 33 already-true wires:

```text
6..11,
16..19,
39..45,
51,
69,
71..84
```

That expands to `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, [`CONTIGUOUS`](../../../../glossary.md#term-contiguous),
[`GATHER`](../../../../glossary.md#gather), `ONE_HOT`, `SCATTER_ELEMENTS`, `UNFOLD_AXIS`, `CAST`, `IS_FINITE`, `IS_NAN`,
`IS_INF`, `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`, `WHERE`, `GATHER_ELEMENTS`,
`GATHER_ND`, `SCATTER_ND`, `SELECT`, [`PAD`](../../../../glossary.md#padding), [`SLICE`](../../../../glossary.md#slice), `SLICE_UPDATE`, `CONCAT`, `STACK`,
`TILE`, `FOLD_AXIS`, `UNFOLD2D`, `FOLD2D`, `UNFOLD3D`, and `FOLD3D`.

No arithmetic comparison, arithmetic [elementwise](../../../../glossary.md#term-elementwise), reduction, [scan](../../../../glossary.md#term-cumulative-scan), [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), ordering, loss,
convolution, [pooling](../../../../glossary.md#term-pooling), random, attention, or recurrent row is silently included. Transfer is a
cross-cutting boundary rather than a registry [operation](../../../../glossary.md#operation) and is audited separately below.

### Common static geometry and physical-[layout](../../../../glossary.md#layout) contract

All new occurrences are common to `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator); exact movement and bit-defined
conversion have no profile relaxation. A selected value [Shape](../../../../glossary.md#shape) is fully static, has [rank](../../../../glossary.md#term-rank) `0..16`,
and has strictly positive present extents. Rank zero is one logical [element](../../../../glossary.md#term-element), not an empty [tensor](../../../../glossary.md#tensor).
Every element count, represented-order ordinal, affine address, [stride](../../../../glossary.md#term-stride) product, [storage offset](../../../../glossary.md#term-storage-offset),
[referenced span](../../../../glossary.md#referenced-element-span), byte count, [metadata](../../../../glossary.md#term-metadata) offset, and `NSUInteger` conversion must pass checked Java and
native arithmetic. The positive logical count must be representable by the existing three-dimensional
custom grid. Because Java products are positive signed `long` values, the grid's three unsigned-
32-bit [dimensions](../../../../glossary.md#dimension) cover the admitted count without a new per-dimension `UINT32_MAX` restriction.

The preparer must authenticate two representations independently:

1. the exact resolved logical layout from Compiler inference, including rank-zero, nonzero offsets,
   positive strides, and <span data-glossary-navigation="true"> ([rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`EXPAND` zero strides; and
2. the actual run-owned physical representation. Ordinary affine producers may materialize dense
   represented-order bytes while preserving their exact logical view descriptor. <span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`SELECT` and
   explicit positive-step [`SLICE`](../../../../glossary.md#slice) retain their derived physical view layout and referenced span.
   A target-relative crop preserves Compiler's exact logical layout but independently derives its
   physical view from the input's authenticated physical descriptor. Thus a derivative scalar
   seed's dense-materialized <span data-glossary-navigation="true"> ([target-relative crop glossary](../../../../glossary.md#target-relative-crop); [derivative glossary](../../../../glossary.md#term-derivative); [scalar glossary](../../../../glossary.md#term-scalar); [seed glossary](../../../../glossary.md#term-seed))</span>`EXPAND` may have zero logical strides while its crop has a safe
   positive-stride physical view. Downstream kernels consume the authenticated physical descriptor,
   never infer it from the logical descriptor.<span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel))</span>

An admitted affine read is one of: [canonical](../../../../glossary.md#term-canonical-form) storage; an authenticated external positive-stride,
non-overlapping `MATERIALIZED_LAYOUT` transfer; an authenticated dense-materialized affine value;
or a resolved nonnegative-stride affine [view](../../../../glossary.md#term-view-meanings) produced in the same maximal [partition](../../../../glossary.md#partition). Every selected
read role admits that common set. This both preserves current SELECT/SLICE storage-layout feeds and
closes composition such as `SLICE -> PERMUTE -> CONTIGUOUS`. `RESHAPE` remains limited to [Compiler](../../../../glossary.md#term-compiler)-
resolved [contiguous](../../../../glossary.md#term-contiguous) input layouts with equal element count; Task 0066 does not invent a layout for
the current unresolved noncontiguous-reshape case. View-producing operations retain the exact
Compiler-resolved output layout. Materializing operations write canonical zero-offset physical
output.

Dynamic or unresolved Shapes/layouts, zero extents, negative-stride logical views, arbitrary
positive-stride overlap, and external zero-stride broadcast remain false. Zero-stride logical
layouts are admitted only when their complete [provenance](../../../../glossary.md#provenance) is an authenticated local `EXPAND` followed
by selected exact rank/[axis](../../../../glossary.md#term-axis)/slice/crop view transforms and every separate physical [descriptor](../../../../glossary.md#term-descriptor) stays
safe. Unsafe or overlapping physical layouts, unauthenticated external view [aliases](../../../../glossary.md#aliasing-and-arena), in-place
output, input/output or output/output physical overlap, and checked-size failure also remain false.
Zero extents need a separate zero-byte [Metal](../../../../glossary.md#term-metal) allocation and no-[dispatch](../../../../glossary.md#term-dispatch) [lifecycle](../../../../glossary.md#lifecycle) contract.
Negative-stride [publication](../../../../glossary.md#term-publication) needs a signed physical-address and transfer contract. Neither may be
special-cased as no work here.

### Affine and [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) wires 6..11

Admit all six carriers for all six rows. Preserve exact [Model](../../../../glossary.md#term-model) [dtype](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), and [gradient](../../../../glossary.md#term-gradient) eligibility.
The exact per-kind geometry remains:

- `RESHAPE`: equal element count, Compiler-resolved contiguous input, exact target Shape, canonical
  represented-order output, including scalar-to-singleton and singleton-to-scalar cases;<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [target glossary](../../../../glossary.md#term-target); [canonical glossary](../../../../glossary.md#term-canonical-form); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- `EXPAND`: exact right-aligned source-equal-or-source-one mapping and exact zero-stride logical
  output, including scalar expansion;<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride))</span>
- `PERMUTE`: a complete [permutation](../../../../glossary.md#permutation), including the empty permutation for a [scalar](../../../../glossary.md#term-scalar);
- `EXPAND_DIMS` and `SQUEEZE`: one valid inserted singleton or removed singleton, including the
  scalar boundary; and
- [`CONTIGUOUS`](../../../../glossary.md#term-contiguous): exact logical represented-order materialization of any admitted affine input to
  canonical output, including nonzero-offset and broadcast views.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [views glossary](../../../../glossary.md#term-view-meanings))</span>

A raw-width custom movement [kernel](../../../../glossary.md#kernel) owns each logical output ordinal and copies exactly 1, 2, 4, or
8 bytes without floating [evaluation](../../../../glossary.md#term-evaluation). Newly broadened occurrences and existing [FLOAT32](../../../../glossary.md#term-float32) occurrences
use the same fixed custom route; the implementation leaves no dtype- or Shape-dependent [MPSGraph](../../../../glossary.md#term-mpsgraph)
alternative.

[Compiler](../../../../glossary.md#term-compiler)'s exact adjoints are [identity](../../../../glossary.md#term-identity-meanings) for `CONTIGUOUS`, reverse `RESHAPE`, inverse `PERMUTE`, and
paired `EXPAND_DIMS`/`SQUEEZE`. Those first-order [graphs](../../../../glossary.md#graph) become [CPU](../../../../glossary.md#term-cpu)-free for all three floating
carriers, including scalar boundaries. `EXPAND` forward supports every [carrier](../../../../glossary.md#term-storage-carrier) and legal gradient
[metadata](../../../../glossary.md#term-metadata), but its [adjoint](../../../../glossary.md#term-adjoint) is `SUM_TO_SHAPE`; [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT64](../../../../glossary.md#term-float64) and strict FLOAT32 reduction gaps remain
blocked. Existing qualified [accelerator](../../../../glossary.md#term-accelerator) FLOAT32 reduction behavior is retained, not generalized.

### [Axis](../../../../glossary.md#term-axis) indexing and windows wires 16..19

Admit the following exact domains:

- [`GATHER`](../../../../glossary.md#gather): all six data/output carriers, INT32 or INT64 indices, admitted affine data and index
  reads, static data rank <span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>`1..16`, [scalar](../../../../glossary.md#term-scalar) or higher-[rank](../../../../glossary.md#term-rank) indices, and output rank `0..16`. Every
  represented index must be non-negative and strictly below the selected extent; negative values
  are invalid and never wrap or normalize. The complete index tensor validates through its
  authenticated physical descriptor before output mutation.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [tensor glossary](../../../../glossary.md#tensor))</span>
- `ONE_HOT`: scalar or rank-`1..15` INT32/INT64 admitted affine index reads and canonical BOOL output
  with one trailing positive depth dimension. Every represented index must be non-negative and below
  depth; each row has exactly one true byte and canonical false elsewhere.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [dimension glossary](../../../../glossary.md#dimension))</span>
- `SCATTER_ELEMENTS/NONE`: all six base/update/output carriers, INT32 or INT64 indices, admitted
  affine base/update/index read roles, exact rank/axis geometry, and runtime bounds plus exact-
  destination uniqueness validation through authenticated physical descriptors before output
  mutation. Negative indices are invalid. The output is canonical. No reduction variant is included.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [validation glossary](../../../../glossary.md#term-validation))</span>
- `UNFOLD_AXIS`: every [Model](../../../../glossary.md#term-model)-legal [carrier](../../../../glossary.md#term-storage-carrier) (all six), static input rank `1..15`, output rank one
  greater, positive size and step, exact floor-count geometry, and no historical selector-expansion
  cap. The custom kernel copies raw words and therefore does not require <span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>`size <= 16`.

[Gradient](../../../../glossary.md#term-gradient) eligibility follows Model [descriptors](../../../../glossary.md#term-descriptor) rather than a [FLOAT32](../../../../glossary.md#term-float32)-only [backend](../../../../glossary.md#backend) rule. `GATHER`
adjoints still generate additive `SCATTER_ADD` and therefore remain outside the [CPU](../../../../glossary.md#term-cpu)-free closure.
`SCATTER_ELEMENTS/NONE` first-order gradients close for floating carriers: the base path performs
replacement with typed zeros and the update path gathers using the exact saved indices. Native
prevalidation keeps those indices live and validates them once before any write. `UNFOLD_AXIS`
gradients close only when `step >= size`, so the generated `FOLD_AXIS` performs no accumulation;
overlapping windows remain a fold/reduction blocker.

The [BFLOAT16](../../../../glossary.md#term-bfloat16) `GATHER` proof reuses Task 0040's exact 20-word reverse-[permutation](../../../../glossary.md#permutation) corpus as a
[regression](../../../../glossary.md#term-regression) and must return raw `0xffa6`, not direct-selector [canonical](../../../../glossary.md#term-canonical-form) `0x7fc0`; broader raw [NaN](../../../../glossary.md#term-nan),
[infinity](../../../../glossary.md#term-infinity), zero, [subnormal](../../../../glossary.md#term-subnormal), and ordinary payloads prove the custom route rather than rehabilitate the
failed selector.

### Complete CAST [matrix](../../../../glossary.md#term-matrix-meanings) at wire 39

Admit all `6 x 6 = 36` ordered source/[target](../../../../glossary.md#term-target) pairs over [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), [INT64](../../../../glossary.md#data-type--datatype), [INT32](../../../../glossary.md#data-type--datatype),
and [BOOL](../../../../glossary.md#data-type--datatype). This retains the 19 currently admitted no-[gradient](../../../../glossary.md#term-gradient) pairs and adds exactly 17 pairs:

- five remaining cross-floating pairs: FLOAT64-to-FLOAT32, FLOAT64-to-BFLOAT16,
  FLOAT32-to-FLOAT64, FLOAT32-to-BFLOAT16, and BFLOAT16-to-FLOAT64;
- six floating-to-signed-integral pairs; and
- six signed-integral-to-floating pairs.

Implement the [Model](../../../../glossary.md#term-model) `CastValueConversions` contract with integer bit [decomposition](../../../../glossary.md#term-decomposition) only; Apple [GPU](../../../../glossary.md#term-gpu)
FLOAT64 arithmetic is not assumed. Same-type casts copy [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity). [Finite](../../../../glossary.md#term-finite) integer/floating conversion
to a floating target rounds directly to that target using [round-to-nearest, ties-to-even](../../../../glossary.md#floating-point-precision-and-rounding), including
direct FLOAT64/integer-to-BFLOAT16 conversion. Signed floating zero is preserved; [gradual underflow](../../../../glossary.md#term-gradual-underflow)
may produce a target [subnormal](../../../../glossary.md#term-subnormal) or [signed zero](../../../../glossary.md#term-signed-zero); [overflow](../../../../glossary.md#term-overflow) produces signed [infinity](../../../../glossary.md#term-infinity). Lossy floating
[narrowing](../../../../glossary.md#term-narrowing) maps every [NaN](../../../../glossary.md#term-nan) to the target's positive [canonical](../../../../glossary.md#term-canonical-form) [quiet NaN](../../../../glossary.md#nan-infinity-and-signed-zero). Lossless [widening](../../../../glossary.md#term-widening) preserves
NaN sign, quiet/signaling state, and the complete source fraction by left [alignment](../../../../glossary.md#term-alignment).

Floating-to-integral conversion truncates toward zero, maps NaN to zero, and saturates [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) and
out-of-range finite values. INT64-to-INT32 retains low [two's-complement](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) bits and INT32-to-INT64 sign-
extends. Numeric-to-BOOL is false only when all magnitude bits are zero, so both floating signed
zeros are false and every NaN is true. BOOL converts to exact positive numeric zero or one.

The capability relation is the exact Model relation:

```text
output.requiresGrad = input.requiresGrad && source.isFloating && target.isFloating
```

Thus all nine ordered floating-to-floating pairs admit legal gradient [metadata](../../../../glossary.md#term-metadata), and their generated
[adjoint](../../../../glossary.md#term-adjoint) is [identity](../../../../glossary.md#term-identity-meanings) for equal types or the reverse floating cast. Float-to-integral may legally read
a gradient-eligible input while producing a non-gradient output; it does not create a [derivative](../../../../glossary.md#term-derivative).
Integral and BOOL roles remain non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) by Model contract.

Proof is not a [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)-only [device](../../../../glossary.md#term-device) [oracle](../../../../glossary.md#term-test-oracle). Tests use an independent [scalar](../../../../glossary.md#term-scalar) oracle, exhaust all 65,536
BFLOAT16 words for each distinct target behavior, exhaust BOOL, cover integral [extrema](../../../../glossary.md#saturation-pole-and-extremum) and [rounding](../../../../glossary.md#term-rounding)
boundaries, cover every floating class/sign/[exponent](../../../../glossary.md#term-exponent) transition, and use deterministic partitioned
raw-word corpora for FLOAT32/FLOAT64. NaN sign/quiet/payload, halfway ties, subnormal boundaries,
overflow, [saturation](../../../../glossary.md#term-saturation), and signed zero are mandatory. Production helpers may not compute expected
answers.

### Classification, [BOOL](../../../../glossary.md#data-type--datatype) logic, and WHERE wires 40..45 and 51

`IS_FINITE`, `IS_NAN`, and `IS_INF` accept [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), or [BFLOAT16](../../../../glossary.md#term-bfloat16) admitted affine input and
produce [canonical](../../../../glossary.md#term-canonical-form) BOOL of the same [Shape](../../../../glossary.md#shape), including [scalars](../../../../glossary.md#term-scalar). Classification inspects sign/[exponent](../../../../glossary.md#term-exponent)/
fraction bits and never evaluates, quiets, narrows, or canonicalizes the source.

`LOGICAL_AND`, `LOGICAL_OR`, and `LOGICAL_NOT` accept admitted affine BOOL inputs, including scalar
and right-aligned broadcast Shapes, and produce canonical BOOL. External logical bytes are
completely validated during existing ingress before invocation; authenticated local BOOL [producers](../../../../glossary.md#term-producer)
write only zero or one, so the [operation](../../../../glossary.md#operation) [kernel](../../../../glossary.md#kernel) never interprets an arbitrary nonzero byte as true.
There is no [gradient](../../../../glossary.md#term-gradient) role.

`WHERE` accepts an admitted BOOL condition and every ordered pair of floating branch carriers. The
output [carrier](../../../../glossary.md#term-storage-carrier) is the exact [Model](../../../../glossary.md#term-model) promotion of the two branch types, giving nine ordered branch-type
[signatures](../../../../glossary.md#term-signature). Condition and branches may be scalar or admitted affine values and broadcast by the
exact right-aligned Model rule to a canonical [rank](../../../../glossary.md#term-rank)-`0..16` output. The kernel reads only the selected
branch and applies the same bit-defined cast contract when that branch is narrower than the promoted
output. It does not evaluate an unselected floating value.

`WHERE` gradient [metadata](../../../../glossary.md#term-metadata) remains the OR of the branch roles; condition is never [differentiable](../../../../glossary.md#derivative-and-chain-rule).
The [CPU](../../../../glossary.md#term-cpu)-free first-order domain requires each selected branch Shape to equal the output Shape, so the
generated [graph](../../../../glossary.md#graph) uses only exact BOOL `WHERE`, typed zero splats/`EXPAND`, and optional reverse casts.
A broadcasted condition is allowed because it receives no [cotangent](../../../../glossary.md#cotangent). A broadcasted differentiable
branch generates `SUM_TO_SHAPE` and therefore retains the same BFLOAT16/FLOAT64/strict-FLOAT32
reduction blocker as `EXPAND`.

### Exact movement and replacement wires 69 and 71..84

Keep every current [Model](../../../../glossary.md#term-model) geometry and extend [dtype](../../../../glossary.md#data-type--datatype)/[layout](../../../../glossary.md#layout)/[gradient](../../../../glossary.md#term-gradient) roles as follows:

- `GATHER_ELEMENTS` and `GATHER_ND`: all six data carriers and both signed index carriers remain;
  data and indices may be admitted affine reads. Legal floating gradient metadata is admitted, but
  the generated adjoints use <span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>`SCATTER_ELEMENTS/ADD` or `SCATTER_ND/ADD` and are not [CPU](../../../../glossary.md#term-cpu)-free.
- `SCATTER_ND/NONE`: all six carriers and both signed index carriers; admitted affine
  base/update/index reads; canonical output; complete non-negative runtime bounds and exact-
  destination uniqueness through authenticated physical descriptors before mutation. Negative
  indices are invalid. Floating first-order base/update gradients close using replacement plus<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
  `GATHER_ND` with the exact saved indices.
- `SELECT`: all six carriers, admitted affine input [rank](../../../../glossary.md#term-rank) `1..16`, scalar output allowed, and
  exact derived positive-stride/nonzero-offset view layout. Floating backward closes through
  scalar-capable <span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings))</span>`EXPAND_DIMS` plus `SLICE_UPDATE` into a typed zero.
- explicit [`SLICE`](../../../../glossary.md#slice): all six carriers for positive-step nonempty coordinate entries plus the
  zero-entry identity form at any admitted rank, including scalar. It retains exact derived
  rank-preserving view geometry and floating backward through replacement into a typed zero.
  Negative-step, empty-extent, and mixed unsupported view forms remain false.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [identity glossary](../../../../glossary.md#term-identity-meanings); [extent glossary](../../../../glossary.md#term-extent))</span>
- [target](../../../../glossary.md#term-target)-relative `CropToShapeAttrs` `SLICE`: add backend-neutral Compiler layout inference for a
  fully static positive-count target (including scalar), a same-rank non-negative prefix, and
  statically proved <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>`prefix + target <= base` bounds over any resolved representable base layout.
  It preserves the exact base logical strides, computes the checked prefix-relative logical storage
  offset, and returns the exact resolved logical view. Add a narrow topological<span data-glossary-navigation="true"> ([storage offset glossary](../../../../glossary.md#term-storage-offset); [topological glossary](../../../../glossary.md#term-topological-order))</span>
  `StaticResultLogicalLayoutClosure` branch that retries only this attribute form after its input
  layout closes; explicit slice behavior remains unchanged. Dynamic, empty, unresolved-base, or
  not-yet-proved bounds stay unresolved. Metal authenticates the input's separate physical
  descriptor and derives the crop's checked physical view from it. An authenticated local
  dense-materialized <span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>`EXPAND` and its exact selected view descendants therefore admit the logical
  zero strides required by derivative scalar seeds; unsafe physical layouts, negative logical
  strides, and logical overlaps without that exact local provenance remain false. This is the
  generated crop used by <span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>[`PAD`](../../../../glossary.md#padding) and `CONCAT` adjoints.
- `SLICE_UPDATE`: all six carriers, admitted affine base/update reads, canonical output, both target-
  relative and explicit-coordinate forms, including the zero-entry whole-value replacement at
  scalar or positive rank and the already-supported negative coordinate step when its nonempty
  canonical update geometry is valid. Floating first-order closure is exact for positive-step and
  zero-entry forms. A negative-step occurrence is also CPU-free when the requested target subset
  selects only the base role: <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>`LayoutGradientRules` emits replacement `SLICE_UPDATE` into a typed
  zero and no extraction. Selecting the update role remains blocked because its cotangent requires
  the unadmitted negative-step `SLICE`; selecting both roles remains blocked for the same reason.
- `PAD`: all six carriers, exact same-typed scalar fill bits, admitted affine input, canonical
  output, scalar included, and floating backward through the newly resolved exact crop.
- `CONCAT` and `STACK`: all six carriers, `1..16` ordered admitted affine inputs, canonical output,
  scalar inputs for `STACK`, and floating backward through exact ordered crops/selects.
- `TILE`: all six carriers, admitted affine input, canonical output, scalar included. Forward accepts
  legal gradient metadata, but every positive-rank adjoint uses <span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint))</span>`SUM`; it is not generalized beyond
  existing qualified accelerator FLOAT32 reduction support. Scalar <span data-glossary-navigation="true"> ([accelerator glossary](../../../../glossary.md#term-accelerator))</span>`TILE` has the Compiler identity
  adjoint and is CPU-free.
- `FOLD_AXIS`: [FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16)/[INT64](../../../../glossary.md#data-type--datatype)/[INT32](../../../../glossary.md#data-type--datatype), never [BOOL](../../../../glossary.md#data-type--datatype), [canonical](../../../../glossary.md#term-canonical-form) geometry, and only
  `step >= windowSize`, so writes never accumulate. Floating backward closes through `UNFOLD_AXIS`.
- `UNFOLD2D`, `FOLD2D`, `UNFOLD3D`, and `FOLD3D`: retain the Model's three floating carriers and exact
  typed padding. Forward unfold admits all valid static geometry. Fold remains restricted to
  non-overlap on every spatial axis. Floating backward is CPU-free only for non-overlapping unfold
  geometry; fold backward closes through its exact unfold counterpart.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [spatial axis glossary](../../../../glossary.md#spatial-axes-channels-and-window-sampling))</span>

Every replacement gradient preserves independent base/update selection. Generated typed-zero
[constants](../../../../glossary.md#term-constant) remain request-local splats, and Task 0066's [scalar](../../../../glossary.md#term-scalar)-capable all-[carrier](../../../../glossary.md#term-storage-carrier) `EXPAND` materializes
them. Saved condition/index roles remain live until their exact backward consumers finish. No [kernel](../../../../glossary.md#kernel)
reconstructs an index, condition, branch choice, or forward output from public targets. The proof
covers target subsets in which those saved roles are internal only.

If a generated [graph](../../../../glossary.md#graph) accumulates two [cotangent](../../../../glossary.md#cotangent) paths, performs `SUM`/`SUM_TO_SHAPE`, uses an additive
scatter, or performs an overlapping fold, that arithmetic [operation](../../../../glossary.md#operation) must be independently capable.
Task 0066 proves single-path exact-movement closures and does not infer arithmetic support from a
forward movement row. Higher-order differentiation is outside the current [Compiler](../../../../glossary.md#term-compiler) contract.

### Transfer, [publication](../../../../glossary.md#term-publication), and [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) boundary

Retain the existing all-six-[carrier](../../../../glossary.md#term-storage-carrier) transfer contract: [canonical](../../../../glossary.md#term-canonical-form) [rank](../../../../glossary.md#term-rank)-`0..16` values and positive-
[stride](../../../../glossary.md#term-stride) non-overlapping storage [layouts](../../../../glossary.md#layout) with checked positive spans may upload/download through the
current typed boundary; [BOOL](../../../../glossary.md#data-type--datatype) is canonicalized/validated exactly. The resulting authenticated
external `MATERIALIZED_LAYOUT` state is an admitted affine read for every selected custom
[occurrence](../../../../glossary.md#term-occurrence), not only the existing SELECT/[SLICE](../../../../glossary.md#slice) subset. Broaden authenticated dense-affine
publication from its current positive-rank [FLOAT32](../../../../glossary.md#term-float32)-only remainder to all six carriers, exposing
canonical represented-order bytes for ordinary affine outputs; this publication evidence is bound
to a prepared [producer](../../../../glossary.md#term-producer). `SELECT`, explicit positive-step `SLICE`, and fully-static [target-relative
crop](../../../../glossary.md#target-relative-crop) publications continue to use their authenticated physical spans. Rank-zero ingress and
publication receive direct public [Engine](../../../../glossary.md#term-engine) coverage for every byte width.

Task 0066 adds no zero-stride or arbitrary overlapping external input, no negative-stride transfer,
no zero-byte transfer, and no [alias](../../../../glossary.md#aliasing-and-arena) promise. Local zero-stride `EXPAND` and its authenticated [view](../../../../glossary.md#term-view-meanings)
descendants do not expand the external transfer contract. Transfer failures still occur before
native invocation and do not create a fallback.

### Fixed custom route and native implementation

Add custom reason `CA_0066` and make exactly ten previously pending movement rows custom-available:
wires 6..11 and 16..19. Split the current shared `DIRECT_INDEX_PENDING_MOVE` catalog mapping so
[`GATHER`](../../../../glossary.md#gather), `ONE_HOT`, and `SCATTER_ELEMENTS` receive the Task-0066 available entry while wire 70
`SCATTER_ADD` alone retains the pending additive blocker. Every selected wire in the exact 33-wire
set then has a custom implementation. Candidate generation must select `CUSTOM_PROGRAM` only for
every selected [occurrence](../../../../glossary.md#term-occurrence), including the historical [FLOAT32](../../../../glossary.md#term-float32) subsets; no old direct/composed
[MPSGraph](../../../../glossary.md#term-mpsgraph) candidate remains for these occurrences. Nested MPSGraph steps for unselected capable
operations may still coexist in one custom whole-program resource in stable [topological order](../../../../glossary.md#term-topological-order).
Rename the ten route-specific native [constants](../../../../glossary.md#term-constant) from their historical `MPSGRAPH` spelling to
`CUSTOM` while retaining numeric wires exactly; no compatibility [alias](../../../../glossary.md#aliasing-and-arena) remains.

Add one Task-0066 native source/header to the existing fixed library and cleanly replace the selected
old [kernel](../../../../glossary.md#kernel) dispatches rather than retain [dtype](../../../../glossary.md#data-type--datatype)-dependent aliases. The implementation may share four
boring kernel families: raw affine/movement, validated index/replacement, integer-defined cast, and
bit-defined predicate/WHERE. Ordinary movement/predicate/cast [dispatch](../../../../glossary.md#term-dispatch) gives one thread [ownership](../../../../glossary.md#term-ownership-meanings) of
each dense output ordinal. [View](../../../../glossary.md#term-view-meanings) outputs use one [owner](../../../../glossary.md#term-owner) per represented physical destination;
broadcast logical views are materialized densely rather than racing writes to an aliased address.
Replacement uses the existing ordered stages: complete [validation](../../../../glossary.md#term-validation), base copy, then one writer per
proved-unique exact update destination. No atomic arithmetic, scratch reduction, floating emulation
library, [host](../../../../glossary.md#term-host) repair, or payload-dependent route is authorized.

Java preparation and native [preflight](../../../../glossary.md#term-preflight) independently authenticate [operation](../../../../glossary.md#operation) wire, attribute kind and
count, ordered roles, type [matrix](../../../../glossary.md#term-matrix-meanings), [rank](../../../../glossary.md#term-rank)/[Shape](../../../../glossary.md#shape), exact logical and physical [layout](../../../../glossary.md#layout), [gradient](../../../../glossary.md#term-gradient) flags,
output state, index width, static products, byte spans, resource access, alias exclusion, and kernel
[metadata](../../../../glossary.md#term-metadata). Failure creates no pipeline-visible mutation. Prepared executables remain [immutable](../../../../glossary.md#term-immutable) and
reusable; [run](../../../../glossary.md#term-run)-owned storage, [lease](../../../../glossary.md#term-lease)/close/[rollback](../../../../glossary.md#term-rollback)/suppression semantics, tracing, sessions, and
[concurrency](../../../../glossary.md#term-concurrency) retain their current contracts.

### Catalog, [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), and count deltas

- Production [operation](../../../../glossary.md#operation)-kind capability remains exactly `83 true / 32 false`; all changes are inside
  the exact 33 already-true wires.
- Structural execution remains exactly `101 true / 14 false`.
- The [MPSGraph](../../../../glossary.md#term-mpsgraph) catalog remains exactly `75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE`; structural
  selector evidence is not production authorization.
- The custom catalog changes exactly `60 AVAILABLE / 55 PENDING / 0 UNAVAILABLE-WITH-PROOF` to
  `70 AVAILABLE / 45 PENDING / 0 UNAVAILABLE-WITH-PROOF` by moving wires 6..11 and 16..19. The
  shared index entry is split so wire 70 remains pending and does not participate in this delta.
- Keep schema 15, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, type wires `1..6`, operation wires `1..115`, attribute wires `0..41`, route
  wires `1..3`, generic cardinality/layout grammar, and exactly thirteen exports. No compatibility
  reader or new public enum exists.<span data-glossary-navigation="true"> ([cardinality glossary](../../../../glossary.md#term-cardinality); [layout glossary](../../../../glossary.md#layout))</span>
- Advance workload-signature, exact-default-policy, candidate-schema, compatibility-schema,
  route-policy, and codec identities together <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [codec glossary](../../../../glossary.md#term-codec))</span>`21 -> 22`. Identity 21 and every older identity
  reject with no alias, migration reader, fallback library, or shim.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Preserve stable feed/value/target order, whole-partition finalization, synchronous invocation,
  liveness, run-local storage, trace identity, and the profile-free Runtime/Trace boundary.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [partition glossary](../../../../glossary.md#partition); [synchronous glossary](../../../../glossary.md#term-synchronous); [liveness glossary](../../../../glossary.md#term-liveness); [run glossary](../../../../glossary.md#term-run); [trace glossary](../../../../glossary.md#trace); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Provable closures versus explicit blockers

The maximal selected closure is exactly:

- all six carriers across exact affine/[canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) and raw movement where [Model](../../../../glossary.md#term-model) permits them;
- all 36 cast pairs with exact Model conversion behavior and nine floating [gradient](../../../../glossary.md#term-gradient) pairs;
- [FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16) classification and all nine promoted floating `WHERE` [signatures](../../../../glossary.md#term-signature);
- [scalar](../../../../glossary.md#term-scalar) and affine [BOOL](../../../../glossary.md#data-type--datatype) logical/broadcast roles;
- [INT64](../../../../glossary.md#data-type--datatype) parity for [gather](../../../../glossary.md#gather)/one-hot/replacement index roles;
- all-[carrier](../../../../glossary.md#term-storage-carrier) [axis](../../../../glossary.md#term-axis) gather/scatter-none/unfold movement and integer non-overlap `FOLD_AXIS`;
- legal gradient [metadata](../../../../glossary.md#term-metadata) on selected forward movement rows; and
- [CPU](../../../../glossary.md#term-cpu)-free first-order [graphs](../../../../glossary.md#graph) consisting only of inverse affine moves, casts, same-[shape](../../../../glossary.md#shape) `WHERE`,
  positive-step slice/select/pad/concat/stack, unique replacement scatter/gather, and non-overlap
  unfold/fold, with exact saved-role liveness.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [liveness glossary](../../../../glossary.md#term-liveness))</span>

Remain explicitly blocked:

- the exact 32 production-false [operation](../../../../glossary.md#operation) kinds listed above;
- additive scatter and every scatter reduction other than `NONE`, overlapping folds/windows, and
  gather adjoints that require those additive operations;
- BFLOAT16/FLOAT64/strict-FLOAT32 `SUM` or `SUM_TO_SHAPE`, positive-[rank](../../../../glossary.md#term-rank) `TILE` adjoints,
  broadcasted-branch `WHERE` adjoints, `EXPAND` adjoints, and multi-path cotangent accumulation unless
  each generated arithmetic node is independently capable;<span data-glossary-navigation="true"> ([cotangent glossary](../../../../glossary.md#cotangent); [node glossary](../../../../glossary.md#node))</span>
- zero extents/empty slices, dynamic or unresolved geometry, rank above 16, noncontiguous unresolved
  reshape, negative-stride view production/publication, arbitrary overlapping layouts, unchecked
  overflow, resource aliasing, and external zero-stride input;<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [publication glossary](../../../../glossary.md#term-publication); [layouts glossary](../../../../glossary.md#layout); [overflow glossary](../../../../glossary.md#term-overflow); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- non-floating differentiation, higher-order gradients, floating arithmetic beyond bit-defined
  conversion/classification/selection, and any transcendental, reduction, attention, or recurrent
  assumption; and
- direct MPSGraph execution for the selected set, host calculation/repair, payload selection,
  timing, autotuning, cache winners, retry, or fallback.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [host glossary](../../../../glossary.md#term-host); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [cache glossary](../../../../glossary.md#term-cache))</span>

If implementation needs a new type/operation/attribute/route wire, [schema](../../../../glossary.md#term-schema) field, [ABI](../../../../glossary.md#term-application-binary-interface-abi) export, public
[API](../../../../glossary.md#term-application-programming-interface-api), zero-byte [buffer](../../../../glossary.md#term-buffer), signed-[stride](../../../../glossary.md#term-stride) transfer, arithmetic emulation, additive update, [runtime](../../../../glossary.md#term-runtime) route
choice, or narrowed corpus, stop for a new independently reviewed plan rather than silently changing
this one.

## Non-goals

No [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation) meaning, public [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), or [Compiler](../../../../glossary.md#term-compiler) [gradient](../../../../glossary.md#term-gradient)-formula change; the sole
Compiler production change is [backend](../../../../glossary.md#backend)-neutral fully-static [target-relative crop](../../../../glossary.md#target-relative-crop) inference plus its
narrow post-input-[layout](../../../../glossary.md#layout) closure branch. No [Planning](../../../../glossary.md#term-planning)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace)/[Engine](../../../../glossary.md#term-engine) API change; no
[FLOAT16](../../../../glossary.md#term-float16); no new production kind; no reopening of Task 0053; no general reduction, transcendental,
[normalization](../../../../glossary.md#term-normalization), [loss](../../../../glossary.md#loss), attention, random, or recurrent execution; no empty/dynamic tensors; no
arbitrary [alias](../../../../glossary.md#aliasing-and-arena)/[view](../../../../glossary.md#term-view-meanings) transfer; no benchmark; and no rewrite of historical blocked evidence. A later
current 115-kind completeness audit remains separate and must not count this bounded [occurrence](../../../../glossary.md#term-occurrence)
closure as semantic support for another row.

## Contracts

- [`ARCHITECTURE.md` — Core lifecycle and invariants](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  — Model owns profile-indexed meaning; backend occurrence support is fail-closed and route choice
  is cold.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend); [occurrence glossary](../../../../glossary.md#term-occurrence); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  — exact movement and bit-defined conversion are common-profile; arithmetic gradients retain their
  primitive/aggregate floors.<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient))</span>
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  and [Numerical-profile backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity)<span data-glossary-navigation="true"> ([backend identity glossary](../../../../glossary.md#backend-identity--backendid))</span>
  — truthful occurrence capability, authenticated physical layouts, fixed routes, and no fallback.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>.
- Complete [Metal](../../../../glossary.md#term-metal) [Task 0059](0059-casts-layout-indexing.md) and
  [Task 0060](0060-exact-replacement-fold-and-aggregate-reductions.md) own the current cast/movement
  and replacement/non-overlap foundations this task broadens without changing their historical
  evidence.
- [Compiler](../../../../glossary.md#term-compiler) [layout/window/indexing gradients](../../../modules/compiler/tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)
  own the generated formulas, saved roles, and additive/overlap boundaries.

## Dependencies and integration

- Depends on: Task 0065 Complete through documentation revision
  `c48b94d7fb2dfa6391901ede580cf886d74cc889`; current Model cast/layout/indexing/window semantics;
  current Compiler first-order formulas and logical-layout closure; completed Tasks 0055..0065
  six-carrier, schema-15, custom-program, layout, replacement, identity, lifecycle, package, and
  public Engine foundations.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [layout glossary](../../../../glossary.md#layout); [Compiler glossary](../../../../glossary.md#term-compiler); [carrier glossary](../../../../glossary.md#term-storage-carrier); [schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Preserves: Blocked Tasks 0016..0018, 0026..0027, 0030..0037, 0039, 0047, 0051, and 0053
  under their recorded gates. On implementation it supersedes only Task 0040's future-route
  conclusion by replacing its failed direct BFLOAT16 <span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>[`GATHER`](../../../../glossary.md#gather) selector with this exact custom raw-
  movement route; the `0xffa6 -> 0x7fc0` failure and all historical evidence remain [immutable](../../../../glossary.md#term-immutable).
- Conflicts with: every concurrent Metal capability/schema/native/custom-source/catalog/candidate/
  route/preparation/resource/publication/package/shared-document scope; resumed Task-0053 production;
  and any concurrent Compiler layout/gradient or transfer-contract change. Task 0066 owns only the
  backend-neutral fully-static target-relative crop inference and its narrow post-input-layout
  closure branch, not a gradient-formula change.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [publication glossary](../../../../glossary.md#term-publication); [scope glossary](../../../../glossary.md#term-scope); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend); [target-relative crop glossary](../../../../glossary.md#target-relative-crop))</span>
- Parallel group: None.
- Common base revision: `c48b94d7fb2dfa6391901ede580cf886d74cc889`.
- Integration order: independent plan review, then one serial implementation/proof cutover, package
  and documentation reconciliation, and independent cumulative Class C review. No production edit
  precedes approved planning.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Integration validation: native build/sign/package/verification; complete Metal; focused Compiler
  source-backed generated-graph contracts; conformance; CPU-free packaged-Metal public Engine
  forward/backward/negative evidence; Javadoc/architecture; one full repository test/build pass;
  exact registry/catalog/identity/export audits; documentation/diff; final independent review.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [graph glossary](../../../../glossary.md#graph); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc); [registry glossary](../../../../glossary.md#term-registry))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): Task-0066 coordinator.

## Files and symbols

Implementation [ownership](../../../../glossary.md#term-ownership-meanings):

- [Compiler](../../../../glossary.md#term-compiler) `LayoutInference` [target](../../../../glossary.md#term-target)-relative [slice](../../../../glossary.md#slice) derivation and
  `StaticResultLogicalLayoutClosure`'s narrow post-input-layout crop retry, with backend-neutral
  Javadocs and focused direct plus generated PAD/CONCAT graph tests for initially unresolved
  derivative splats and static/resolved versus dynamic/empty/unresolved cases;<span data-glossary-navigation="true"> ([PAD glossary](../../../../glossary.md#padding); [graph glossary](../../../../glossary.md#graph); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- `CapturedGraphInference`'s matching [layout](../../../../glossary.md#layout)-only deferral, explicitly restricted to `SLICE`, so
  the inference/closure pass can resolve that crop without broadening `SLICE_UPDATE` or changing a
  gradient formula;
- `AutogradPreflight`'s existing 3D window counterpart validation, corrected to compare semantic
  type, Shape, and gradient properties rather than an incidental intermediate layout;<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape))</span>
- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNegPartitionPreparer`,
  `MetalNegPreparationPlan`, `MetalNativeApi`, `MetalOperationRouteCatalog`, custom candidate/tuning/
  codec identities, transfer/publication authentication, and affected package/type Javadocs;<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [publication glossary](../../../../glossary.md#term-publication); [authentication glossary](../../../../glossary.md#term-authentication))</span>
- native foundation [preflight](../../../../glossary.md#term-preflight)/[metadata](../../../../glossary.md#term-metadata)/pipeline/step/[binding](../../../../glossary.md#term-binding) code plus
  `synaptik_task0066_dtype_layout_kernels.h`, with no new export or [schema](../../../../glossary.md#term-schema) field;
- focused capability/catalog/schema/native/malformed/raw-word/index/layout/rank-zero/bounds tests,
  including allocation-free overflow and native-parity rejection;<span data-glossary-navigation="true"> ([rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- current Compiler contract suites plus a dedicated CPU-free Engine Metal integration for all six
  carriers, cast matrix representatives, scalar/view/broadcast composition, saved condition/index
  roles, exact first-order generated graphs, reuse/concurrency/sessions, and early failures; and<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [Metal glossary](../../../../glossary.md#term-metal); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [view glossary](../../../../glossary.md#term-view-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- current Metal/backend/native guides, capability/preparer status, architecture identity claims,
  this brief, master plan, and roadmap.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>

No [Model](../../../../glossary.md#term-model) or Compiler [gradient](../../../../glossary.md#term-gradient)-formula edit occurred. The three Compiler accommodations above are
the complete [backend](../../../../glossary.md#backend)-neutral Compiler source [scope](../../../../glossary.md#term-scope) used by the implementation.

## Acceptance criteria

1. The exact selected set is wires `6..11,16..19,39..45,51,69,71..84` (33 kinds). All other 82
   kinds retain their current occurrence domains, and the named 32 false rows remain false for the
   recorded semantic reasons.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
2. All selected descriptors obey exact Model dtype, role, Shape, layout, output, and gradient
   relations. Static rank-zero works; positive extents and checked spans are mandatory; zero extent,
   unresolved/dynamic, negative-view, alias, and overflow cases reject before resource creation.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [Model glossary](../../../../glossary.md#term-model); [dtype glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [extent glossary](../../../../glossary.md#term-extent); [view glossary](../../../../glossary.md#term-view-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [overflow glossary](../../../../glossary.md#term-overflow))</span>
3. Every one of the 36 cast pairs matches an independent bit oracle. The 17 new pairs, nine floating
   gradient pairs, NaN policies, ties-to-even, direct BFLOAT16 rounding, subnormals, signed zero,
   infinities, saturation, modulo narrowing, and BOOL mapping are covered without hardware-double or
   production-oracle expected values.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [NaN glossary](../../../../glossary.md#term-nan); [ties-to-even glossary](../../../../glossary.md#term-ties-to-even); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [rounding glossary](../../../../glossary.md#term-rounding); [subnormals glossary](../../../../glossary.md#term-subnormal); [signed zero glossary](../../../../glossary.md#term-signed-zero); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [saturation glossary](../../../../glossary.md#term-saturation); [narrowing glossary](../../../../glossary.md#term-narrowing); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
4. All-carrier affine/contiguous movement preserves raw words through scalar, nonzero-offset,
   permutation, singleton, broadcast, and canonicalization chains. Public evidence includes<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [contiguous glossary](../../../../glossary.md#term-contiguous); [scalar glossary](../../../../glossary.md#term-scalar); [permutation glossary](../../../../glossary.md#permutation); [singleton glossary](../../../../glossary.md#term-singleton); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
   `EXPAND -> PERMUTE/SLICE -> consumer`, with safe derived physical descriptors despite exact
   zero-stride logical descendants. Logical versus physical layout is authenticated explicitly; no
   broadcast destination race or unowned physical write exists.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [race glossary](../../../../glossary.md#term-race))</span>
5. FLOAT64/FLOAT32/BFLOAT16 classification is bit-defined. BOOL logical rows and all nine mixed
   floating <span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>`WHERE` signatures obey scalar/affine/broadcast semantics, canonical BOOL, selected-
   branch-only conversion, exact promotion, and branch-gradient roles.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
6. Axis and ND index operations support both signed index carriers over the declared data domains.
   Data, update, and index reads accept authenticated canonical, external <span data-glossary-navigation="true"> ([Axis glossary](../../../../glossary.md#term-axis); [ND glossary](../../../../glossary.md#term-nd))</span>`MATERIALIZED_LAYOUT`,
   dense-materialized, or local affine physical descriptors; public chains feed integer `PERMUTE`,
   [`SLICE`](../../../../glossary.md#slice), and `EXPAND` results directly into each index consumer. Complete physical-address
   prevalidation precedes mutation. Every negative value rejects without wrapping or normalization;
   non-negative bounds, replacement uniqueness, canonical BOOL one-hot, and complete validation-
   before-mutation hold for scalar and positive-rank cases.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [validation glossary](../../../../glossary.md#term-validation); [rank glossary](../../../../glossary.md#term-rank))</span>
7. First-order CPU-free evidence covers floating casts, inverse affine movement, positive-step
   SELECT/SLICE/SLICE_UPDATE, negative-step base-only SLICE_UPDATE target subsets, PAD,
   CONCAT/STACK, unique replacement scatters, same-shape WHERE, and non-overlap window pairs. Exact
   saved condition/index roles remain live for target subsets and are neither recomputed nor
   published solely to make backward work. Focused Compiler tests prove negative-step SLICE_UPDATE
   base-only generation and preserve the update-selected negative-SLICE blocker. Direct Compiler
   tests and actual generated PAD/CONCAT graphs from a scalar seed's dense-materialized <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [target glossary](../../../../glossary.md#term-target); [PAD glossary](../../../../glossary.md#padding); [Compiler glossary](../../../../glossary.md#term-compiler); [graphs glossary](../../../../glossary.md#graph); [seed glossary](../../../../glossary.md#term-seed))</span>`EXPAND`
   prove the fully-static target-relative crop closes after the derivative splat's logical layout
   initially resolves to zero strides. Dynamic/empty/unresolved cases remain false. The Compiler
   rule remains backend-neutral; Metal rejects unsafe physical layout, not the authenticated local
   zero-stride logical crop.<span data-glossary-navigation="true"> ([target-relative crop glossary](../../../../glossary.md#target-relative-crop); [derivative glossary](../../../../glossary.md#term-derivative); [backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal))</span>
8. Negative evidence proves additive gather adjoints, overlapping windows/folds, arithmetic scatter,
   positive-rank TILE reduction, broadcasted differentiable branches, EXPAND reductions,
   multi-path arithmetic accumulation, and all 32 false kinds remain outside this closure.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
9. Every selected [partition](../../../../glossary.md#partition) has only `CUSTOM_PROGRAM`. Exactly wires 6..11 and 16..19 move pending
   to custom-available; no MPSGraph candidate, retry, fallback, timing, autotuning, payload decision,
   or host repair exists.<span data-glossary-navigation="true"> ([autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [host glossary](../../../../glossary.md#term-host))</span>
10. Counts finish at `83/32` capability, `101/14` structural, `75/35/5` [MPSGraph](../../../../glossary.md#term-mpsgraph), and `70/45/0`
    custom. Schema 15, ABI 5, six type wires, 115 operation wires, 42 attribute wires, three route
    wires, and thirteen exports remain; all backend-local identities are 22 and identity 21 rejects.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [operation glossary](../../../../glossary.md#operation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
11. Packaged-native/prepared-Metal evidence proves preflight parity, malformed rejection, no partial
    writes, fixed routes, raw-word results, and checked bounds. CPU-free packaged-Metal public Engine
    proves rank-zero/all-width transfer, forward chains, generated backward graphs, saved roles,
    target subsets, prepared reuse, concurrent/independent sessions, and named early failures with
    no skip or CPU fallback.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [Engine glossary](../../../../glossary.md#term-engine))</span>
12. Native/package, complete Metal, focused Compiler, conformance, dedicated Engine, Javadoc,
    architecture, Markdown/link/diff, and one full repository test/build pass; independent cumulative
    Class C review has no unresolved P0/P1/P2.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## [Validation](../../../../glossary.md#term-validation) after implementation

```bash
./native/metal-macos-arm64/build.sh
codesign --force --sign - --identifier io.github.pho001.synaptik.metal.foundation native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/package-local.sh native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/verify-package.sh native/metal-macos-arm64/build/package-v1/macos-arm64
./gradlew :backends:metal:verifyMetalNativePackage :backends:metal:metalNativeLocalZip \
  -PsynaptikMetalNativePackage="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64"
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test
./gradlew :modules:compiler:test --tests '*Cast*' --tests '*Layout*' --tests '*Indexing*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*DtypeLayoutGradientMetalIntegrationTest*'
./gradlew :backends:metal:javadoc :testing:architecture-tests:test
./gradlew test
./gradlew build
git diff --check
```

No timing or benchmark command was [run](../../../../glossary.md#term-run). Native build/sign/package/verification, the [Gradle](../../../../glossary.md#term-gradle) package
tasks, complete [Metal](../../../../glossary.md#term-metal) tests, focused [Compiler](../../../../glossary.md#term-compiler) contracts, Metal [conformance](../../../../glossary.md#term-conformance), the dedicated [CPU](../../../../glossary.md#term-cpu)-free
public [Engine](../../../../glossary.md#term-engine) integration, Metal [Javadoc](../../../../glossary.md#term-javadoc), architecture tests, `./gradlew test`, `./gradlew build`,
and `git diff --check` all passed for the implementation and documentation cutover.

## Documentation and review impact

Implementation must update every current selected-domain, custom-count, route, [identity](../../../../glossary.md#term-identity-meanings), cast,
[layout](../../../../glossary.md#layout), index, transfer, and [gradient](../../../../glossary.md#term-gradient) statement while preserving the current [operation](../../../../glossary.md#operation)-kind counts
and all semantic blockers. Public/[backend](../../../../glossary.md#backend)/native/[Javadoc](../../../../glossary.md#term-javadoc) wording must distinguish operation-kind
truth from [occurrence](../../../../glossary.md#term-occurrence) breadth, logical [views](../../../../glossary.md#term-view-meanings) from physical [materialization](../../../../glossary.md#materialization), forward gradient
[metadata](../../../../glossary.md#term-metadata) from a [CPU](../../../../glossary.md#term-cpu)-free generated [backward graph](../../../../glossary.md#backward-graph), and authenticated external transferred
materialization from unauthenticated view [aliases](../../../../glossary.md#aliasing-and-arena). Historical tasks remain [immutable](../../../../glossary.md#term-immutable) evidence
records.

Independent plan review verified the exact 115-row partition, 33-wire selection, `+10` custom
catalog arithmetic, 36-pair cast [matrix](../../../../glossary.md#term-matrix-meanings), [Model](../../../../glossary.md#term-model) gradient relations, [Compiler](../../../../glossary.md#term-compiler)-generated operations,
saved roles, [scalar](../../../../glossary.md#term-scalar)/empty distinction, physical-layout safety, transfer boundary, fixed-route
policy, and non-overlap with the future full [registry](../../../../glossary.md#term-registry) audit, then returned `APPROVE` with zero
P0/P1/P2 findings. Independent cumulative quality review rechecked the full capability-to-[kernel](../../../../glossary.md#kernel)
path, including [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) TARGET_SHAPE/[PERMUTATION](../../../../glossary.md#permutation), scalar [GATHER](../../../../glossary.md#gather)/ONE_HOT, and size-17
UNFOLD_AXIS. Independent security review rechecked integer arithmetic, index bounds,
[validation](../../../../glossary.md#term-validation)-before-write, [dispatch](../../../../glossary.md#term-dispatch) geometry, and resource [aliasing](../../../../glossary.md#term-aliasing). Both returned `APPROVE` at
`4b426898` with zero P0/P1/P2.

## [Planning](../../../../glossary.md#term-planning) result

The maximal reachable closure is an [occurrence](../../../../glossary.md#term-occurrence)-domain and fixed-route cutover inside 33 existing
production kinds, not a new semantic family. It plans all six exact movement carriers, all 36 cast
pairs, [floating classification](../../../../glossary.md#floating-classification)/promotion, [BOOL](../../../../glossary.md#data-type--datatype) [scalar](../../../../glossary.md#term-scalar)/broadcast roles, [INT64](../../../../glossary.md#data-type--datatype) indexing parity,
[rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) composition, and exact first-order movement/replacement/non-overlap [graphs](../../../../glossary.md#graph). It preserves
zero-[extent](../../../../glossary.md#term-extent)/signed-[stride](../../../../glossary.md#term-stride)/overlap/additive/reduction/transcendental/attention/recurrent blockers and
all 32 production-false rows.

Implementation will retain capability `83/32`, structural execution `101/14`, and [MPSGraph](../../../../glossary.md#term-mpsgraph) catalog
`75/35/5`; move exactly ten custom rows to finish `70/45/0`; retain [schema](../../../../glossary.md#term-schema) 15, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, and thirteen
exports; and advance every [backend](../../../../glossary.md#backend)-local [identity](../../../../glossary.md#term-identity-meanings) from 21 to 22.

## Implementation result

Implementation `0b88f897` and its reviewed remediations complete exactly wires
`6..11,16..19,39..45,51,69,71..84` through one selected-[occurrence](../../../../glossary.md#term-occurrence) `CUSTOM_PROGRAM` route. Java
preparation and native [preflight](../../../../glossary.md#term-preflight) independently derive and authenticate logical versus physical
[layout](../../../../glossary.md#layout) geometry; native [kernels](../../../../glossary.md#kernel) implement raw movement, validated replacement/indexing, all 36
casts, bit-defined classification/WHERE, [scalar](../../../../glossary.md#term-scalar) values, and non-overlapping window movement. The
public [Engine](../../../../glossary.md#term-engine) integration covers all [carrier](../../../../glossary.md#term-storage-carrier) widths, scalar and affine [publication](../../../../glossary.md#term-publication), saved
condition/index roles, generated backward [graphs](../../../../glossary.md#graph), prepared reuse, independent/concurrent sessions,
and early rejection of additive scatter, direct EXPAND [adjoint](../../../../glossary.md#term-adjoint) reduction, and other named blockers.

The resulting boundary is capability `83/32`, structural execution `101/14`, [MPSGraph](../../../../glossary.md#term-mpsgraph) catalog
`75/35/5`, custom catalog `70/45/0`, [schema](../../../../glossary.md#term-schema) 15, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, thirteen exports, and [backend](../../../../glossary.md#backend)-local [identity](../../../../glossary.md#term-identity-meanings)
22. Identity 21 and every older identity reject. Task 0040's historical direct [BFLOAT16](../../../../glossary.md#term-bfloat16) [GATHER](../../../../glossary.md#gather)
failure remains unchanged; only its future-route conclusion is superseded by the exact custom raw
movement route.

Security remediation rejects input/output and output/output [buffer](../../../../glossary.md#term-buffer) [aliasing](../../../../glossary.md#term-aliasing) before mutation and
proves rejected [targets](../../../../glossary.md#term-target) retain sentinel bytes. Contract remediation aligns Java and native
[validation](../../../../glossary.md#term-validation) for [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) target/[permutation](../../../../glossary.md#permutation) [shapes](../../../../glossary.md#shape), scalar GATHER/ONE_HOT, and arbitrary positive
UNFOLD_AXIS windows that fit the selected [extent](../../../../glossary.md#term-extent); focused real-native regressions exercise each
boundary.
