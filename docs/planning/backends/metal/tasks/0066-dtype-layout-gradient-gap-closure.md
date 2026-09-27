# Task 0066: Dtype, Layout, and Gradient Gap Closure

## Status

Complete — independently approved plan
`e0ec3d2360b0b7ab1119ce0613a612bf3e18b218` was implemented at `0b88f897`. The cutover preserves
all named blockers and fixed schema/ABI/export boundaries while completing the exact selected
occurrence, route, physical-layout, generated-gradient, package, and documentation scope.

## Change class

Class C — the implementation broadens occurrence capability inside 33 already-production-true
operation kinds, replaces ten pending movement rows with fixed custom implementations, closes
first-order movement gradients that generate only already-exact operations, and advances every
backend-local identity. It adds no operation kind and does not reopen the semantic families
represented by the 32 production-false rows.

## Goal

Close the maximal currently reachable Metal dtype, affine-layout, exact-movement, and first-order
gradient gaps after Task 0065. Use integer/bit-defined custom kernels rather than opaque floating
selectors so all six represented carriers can move without changing raw payloads, all 36 current
`CAST` pairs match the Model oracle, floating classification and mixed floating `WHERE` work for
FLOAT64/FLOAT32/BFLOAT16, INT64 index roles match INT32 roles, scalar Shapes compose, and exact
replacement/view/window adjoints run through their generated graphs.

Make every selected occurrence use one deterministic `CUSTOM_PROGRAM` route chosen from authenticated
metadata before resource creation. Preserve the existing external transfer contract and distinguish
logical affine geometry from run-owned dense materialization. Route choice may use no payload, timing,
autotuning, cache winner, retry, fallback, or host repair.

Do not duplicate a later full 115-kind semantic audit. This task inventories all 115 rows only far
enough to prove that no production-false row is dtype/layout/gradient infrastructure. The operation-
kind totals therefore remain fixed while occurrence domains, custom catalog state, and identity
change.

## Scope

### Exact post-Task-0065 115-kind boundary

The registry remains exactly wires `1..115`. Current structural execution is `101 true / 14 false`,
and current production capability is `83 true / 32 false`.

The 14 structurally non-executable rows are exactly:

- wire 35 `SCALED_DOT_PRODUCT_ATTENTION`;
- wires 55 and 64 `EXP` and `SIGMOID`;
- wires 86..93 `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`,
  `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`, `BATCH_NORM_INFERENCE`,
  `BATCH_NORM_TRAINING`, `LAYER_NORM`, `RMS_NORM`, `SOFTMAX`, and `LOG_SOFTMAX`; and
- wires 103..105 `RNN_TANH`, `GRU_RESET_AFTER`, and `LSTM`.

The other 18 production-false rows are structurally executable only:

- wires 38 and 50 `TENSOR_POW` and `SCALAR_POW`;
- wires 53, 54, 56..59, 65..68 `LOG`, `LOG1P`, `EXPM1`, `ERF`, `SQRT`, `RSQRT`,
  `TANH`, `GELU`, `GELU_TANH_APPROXIMATION`, and `SILU`;
- wire 70 `SCATTER_ADD`; and
- wires 111..115 `LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, and
  `L2_NORM`.

These 32 rows remain false. They require missing transcendental/recursive, aggregate/additive,
normalization/loss, attention, or recurrent/BPTT semantics. Schema vocabulary, a movement kernel,
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

That expands to `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, `CONTIGUOUS`,
`GATHER`, `ONE_HOT`, `SCATTER_ELEMENTS`, `UNFOLD_AXIS`, `CAST`, `IS_FINITE`, `IS_NAN`,
`IS_INF`, `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`, `WHERE`, `GATHER_ELEMENTS`,
`GATHER_ND`, `SCATTER_ND`, `SELECT`, `PAD`, `SLICE`, `SLICE_UPDATE`, `CONCAT`, `STACK`,
`TILE`, `FOLD_AXIS`, `UNFOLD2D`, `FOLD2D`, `UNFOLD3D`, and `FOLD3D`.

No arithmetic comparison, arithmetic elementwise, reduction, scan, MATMUL, ordering, loss,
convolution, pooling, random, attention, or recurrent row is silently included. Transfer is a
cross-cutting boundary rather than a registry operation and is audited separately below.

### Common static geometry and physical-layout contract

All new occurrences are common to `STRICT_IEEE` and `ACCELERATOR`; exact movement and bit-defined
conversion have no profile relaxation. A selected value Shape is fully static, has rank `0..16`,
and has strictly positive present extents. Rank zero is one logical element, not an empty tensor.
Every element count, represented-order ordinal, affine address, stride product, storage offset,
referenced span, byte count, metadata offset, and `NSUInteger` conversion must pass checked Java and
native arithmetic. The positive logical count must be representable by the existing three-dimensional
custom grid. Because Java products are positive signed `long` values, the grid's three unsigned-
32-bit dimensions cover the admitted count without a new per-dimension `UINT32_MAX` restriction.

The preparer must authenticate two representations independently:

1. the exact resolved logical layout from Compiler inference, including rank-zero, nonzero offsets,
   positive strides, and `EXPAND` zero strides; and
2. the actual run-owned physical representation. Ordinary affine producers may materialize dense
   represented-order bytes while preserving their exact logical view descriptor. `SELECT` and
   explicit positive-step `SLICE` retain their derived physical view layout and referenced span.
   A target-relative crop preserves Compiler's exact logical layout but independently derives its
   physical view from the input's authenticated physical descriptor. Thus a derivative scalar
   seed's dense-materialized `EXPAND` may have zero logical strides while its crop has a safe
   positive-stride physical view. Downstream kernels consume the authenticated physical descriptor,
   never infer it from the logical descriptor.

An admitted affine read is one of: canonical storage; an authenticated external positive-stride,
non-overlapping `MATERIALIZED_LAYOUT` transfer; an authenticated dense-materialized affine value;
or a resolved nonnegative-stride affine view produced in the same maximal partition. Every selected
read role admits that common set. This both preserves current SELECT/SLICE storage-layout feeds and
closes composition such as `SLICE -> PERMUTE -> CONTIGUOUS`. `RESHAPE` remains limited to Compiler-
resolved contiguous input layouts with equal element count; Task 0066 does not invent a layout for
the current unresolved noncontiguous-reshape case. View-producing operations retain the exact
Compiler-resolved output layout. Materializing operations write canonical zero-offset physical
output.

Dynamic or unresolved Shapes/layouts, zero extents, negative-stride logical views, arbitrary
positive-stride overlap, and external zero-stride broadcast remain false. Zero-stride logical
layouts are admitted only when their complete provenance is an authenticated local `EXPAND` followed
by selected exact rank/axis/slice/crop view transforms and every separate physical descriptor stays
safe. Unsafe or overlapping physical layouts, unauthenticated external view aliases, in-place
output, input/output or output/output physical overlap, and checked-size failure also remain false.
Zero extents need a separate zero-byte Metal allocation and no-dispatch lifecycle contract.
Negative-stride publication needs a signed physical-address and transfer contract. Neither may be
special-cased as no work here.

### Affine and canonicalization wires 6..11

Admit all six carriers for all six rows. Preserve exact Model dtype, Shape, and gradient eligibility.
The exact per-kind geometry remains:

- `RESHAPE`: equal element count, Compiler-resolved contiguous input, exact target Shape, canonical
  represented-order output, including scalar-to-singleton and singleton-to-scalar cases;
- `EXPAND`: exact right-aligned source-equal-or-source-one mapping and exact zero-stride logical
  output, including scalar expansion;
- `PERMUTE`: a complete permutation, including the empty permutation for a scalar;
- `EXPAND_DIMS` and `SQUEEZE`: one valid inserted singleton or removed singleton, including the
  scalar boundary; and
- `CONTIGUOUS`: exact logical represented-order materialization of any admitted affine input to
  canonical output, including nonzero-offset and broadcast views.

A raw-width custom movement kernel owns each logical output ordinal and copies exactly 1, 2, 4, or
8 bytes without floating evaluation. Newly broadened occurrences and existing FLOAT32 occurrences
use the same fixed custom route; the implementation leaves no dtype- or Shape-dependent MPSGraph
alternative.

Compiler's exact adjoints are identity for `CONTIGUOUS`, reverse `RESHAPE`, inverse `PERMUTE`, and
paired `EXPAND_DIMS`/`SQUEEZE`. Those first-order graphs become CPU-free for all three floating
carriers, including scalar boundaries. `EXPAND` forward supports every carrier and legal gradient
metadata, but its adjoint is `SUM_TO_SHAPE`; BFLOAT16/FLOAT64 and strict FLOAT32 reduction gaps remain
blocked. Existing qualified accelerator FLOAT32 reduction behavior is retained, not generalized.

### Axis indexing and windows wires 16..19

Admit the following exact domains:

- `GATHER`: all six data/output carriers, INT32 or INT64 indices, admitted affine data and index
  reads, static data rank `1..16`, scalar or higher-rank indices, and output rank `0..16`. Every
  represented index must be non-negative and strictly below the selected extent; negative values
  are invalid and never wrap or normalize. The complete index tensor validates through its
  authenticated physical descriptor before output mutation.
- `ONE_HOT`: scalar or rank-`1..15` INT32/INT64 admitted affine index reads and canonical BOOL output
  with one trailing positive depth dimension. Every represented index must be non-negative and below
  depth; each row has exactly one true byte and canonical false elsewhere.
- `SCATTER_ELEMENTS/NONE`: all six base/update/output carriers, INT32 or INT64 indices, admitted
  affine base/update/index read roles, exact rank/axis geometry, and runtime bounds plus exact-
  destination uniqueness validation through authenticated physical descriptors before output
  mutation. Negative indices are invalid. The output is canonical. No reduction variant is included.
- `UNFOLD_AXIS`: every Model-legal carrier (all six), static input rank `1..15`, output rank one
  greater, positive size and step, exact floor-count geometry, and no historical selector-expansion
  cap. The custom kernel copies raw words and therefore does not require `size <= 16`.

Gradient eligibility follows Model descriptors rather than a FLOAT32-only backend rule. `GATHER`
adjoints still generate additive `SCATTER_ADD` and therefore remain outside the CPU-free closure.
`SCATTER_ELEMENTS/NONE` first-order gradients close for floating carriers: the base path performs
replacement with typed zeros and the update path gathers using the exact saved indices. Native
prevalidation keeps those indices live and validates them once before any write. `UNFOLD_AXIS`
gradients close only when `step >= size`, so the generated `FOLD_AXIS` performs no accumulation;
overlapping windows remain a fold/reduction blocker.

The BFLOAT16 `GATHER` proof reuses Task 0040's exact 20-word reverse-permutation corpus as a
regression and must return raw `0xffa6`, not direct-selector canonical `0x7fc0`; broader raw NaN,
infinity, zero, subnormal, and ordinary payloads prove the custom route rather than rehabilitate the
failed selector.

### Complete CAST matrix at wire 39

Admit all `6 x 6 = 36` ordered source/target pairs over FLOAT64, FLOAT32, BFLOAT16, INT64, INT32,
and BOOL. This retains the 19 currently admitted no-gradient pairs and adds exactly 17 pairs:

- five remaining cross-floating pairs: FLOAT64-to-FLOAT32, FLOAT64-to-BFLOAT16,
  FLOAT32-to-FLOAT64, FLOAT32-to-BFLOAT16, and BFLOAT16-to-FLOAT64;
- six floating-to-signed-integral pairs; and
- six signed-integral-to-floating pairs.

Implement the Model `CastValueConversions` contract with integer bit decomposition only; Apple GPU
FLOAT64 arithmetic is not assumed. Same-type casts copy raw bits. Finite integer/floating conversion
to a floating target rounds directly to that target using round-to-nearest, ties-to-even, including
direct FLOAT64/integer-to-BFLOAT16 conversion. Signed floating zero is preserved; gradual underflow
may produce a target subnormal or signed zero; overflow produces signed infinity. Lossy floating
narrowing maps every NaN to the target's positive canonical quiet NaN. Lossless widening preserves
NaN sign, quiet/signaling state, and the complete source fraction by left alignment.

Floating-to-integral conversion truncates toward zero, maps NaN to zero, and saturates infinities and
out-of-range finite values. INT64-to-INT32 retains low two's-complement bits and INT32-to-INT64 sign-
extends. Numeric-to-BOOL is false only when all magnitude bits are zero, so both floating signed
zeros are false and every NaN is true. BOOL converts to exact positive numeric zero or one.

The capability relation is the exact Model relation:

```text
output.requiresGrad = input.requiresGrad && source.isFloating && target.isFloating
```

Thus all nine ordered floating-to-floating pairs admit legal gradient metadata, and their generated
adjoint is identity for equal types or the reverse floating cast. Float-to-integral may legally read
a gradient-eligible input while producing a non-gradient output; it does not create a derivative.
Integral and BOOL roles remain non-differentiable by Model contract.

Proof is not a sample-only device oracle. Tests use an independent scalar oracle, exhaust all 65,536
BFLOAT16 words for each distinct target behavior, exhaust BOOL, cover integral extrema and rounding
boundaries, cover every floating class/sign/exponent transition, and use deterministic partitioned
raw-word corpora for FLOAT32/FLOAT64. NaN sign/quiet/payload, halfway ties, subnormal boundaries,
overflow, saturation, and signed zero are mandatory. Production helpers may not compute expected
answers.

### Classification, BOOL logic, and WHERE wires 40..45 and 51

`IS_FINITE`, `IS_NAN`, and `IS_INF` accept FLOAT64, FLOAT32, or BFLOAT16 admitted affine input and
produce canonical BOOL of the same Shape, including scalars. Classification inspects sign/exponent/
fraction bits and never evaluates, quiets, narrows, or canonicalizes the source.

`LOGICAL_AND`, `LOGICAL_OR`, and `LOGICAL_NOT` accept admitted affine BOOL inputs, including scalar
and right-aligned broadcast Shapes, and produce canonical BOOL. External logical bytes are
completely validated during existing ingress before invocation; authenticated local BOOL producers
write only zero or one, so the operation kernel never interprets an arbitrary nonzero byte as true.
There is no gradient role.

`WHERE` accepts an admitted BOOL condition and every ordered pair of floating branch carriers. The
output carrier is the exact Model promotion of the two branch types, giving nine ordered branch-type
signatures. Condition and branches may be scalar or admitted affine values and broadcast by the
exact right-aligned Model rule to a canonical rank-`0..16` output. The kernel reads only the selected
branch and applies the same bit-defined cast contract when that branch is narrower than the promoted
output. It does not evaluate an unselected floating value.

`WHERE` gradient metadata remains the OR of the branch roles; condition is never differentiable.
The CPU-free first-order domain requires each selected branch Shape to equal the output Shape, so the
generated graph uses only exact BOOL `WHERE`, typed zero splats/`EXPAND`, and optional reverse casts.
A broadcasted condition is allowed because it receives no cotangent. A broadcasted differentiable
branch generates `SUM_TO_SHAPE` and therefore retains the same BFLOAT16/FLOAT64/strict-FLOAT32
reduction blocker as `EXPAND`.

### Exact movement and replacement wires 69 and 71..84

Keep every current Model geometry and extend dtype/layout/gradient roles as follows:

- `GATHER_ELEMENTS` and `GATHER_ND`: all six data carriers and both signed index carriers remain;
  data and indices may be admitted affine reads. Legal floating gradient metadata is admitted, but
  the generated adjoints use `SCATTER_ELEMENTS/ADD` or `SCATTER_ND/ADD` and are not CPU-free.
- `SCATTER_ND/NONE`: all six carriers and both signed index carriers; admitted affine
  base/update/index reads; canonical output; complete non-negative runtime bounds and exact-
  destination uniqueness through authenticated physical descriptors before mutation. Negative
  indices are invalid. Floating first-order base/update gradients close using replacement plus
  `GATHER_ND` with the exact saved indices.
- `SELECT`: all six carriers, admitted affine input rank `1..16`, scalar output allowed, and
  exact derived positive-stride/nonzero-offset view layout. Floating backward closes through
  scalar-capable `EXPAND_DIMS` plus `SLICE_UPDATE` into a typed zero.
- explicit `SLICE`: all six carriers for positive-step nonempty coordinate entries plus the
  zero-entry identity form at any admitted rank, including scalar. It retains exact derived
  rank-preserving view geometry and floating backward through replacement into a typed zero.
  Negative-step, empty-extent, and mixed unsupported view forms remain false.
- target-relative `CropToShapeAttrs` `SLICE`: add backend-neutral Compiler layout inference for a
  fully static positive-count target (including scalar), a same-rank non-negative prefix, and
  statically proved `prefix + target <= base` bounds over any resolved representable base layout.
  It preserves the exact base logical strides, computes the checked prefix-relative logical storage
  offset, and returns the exact resolved logical view. Add a narrow topological
  `StaticResultLogicalLayoutClosure` branch that retries only this attribute form after its input
  layout closes; explicit slice behavior remains unchanged. Dynamic, empty, unresolved-base, or
  not-yet-proved bounds stay unresolved. Metal authenticates the input's separate physical
  descriptor and derives the crop's checked physical view from it. An authenticated local
  dense-materialized `EXPAND` and its exact selected view descendants therefore admit the logical
  zero strides required by derivative scalar seeds; unsafe physical layouts, negative logical
  strides, and logical overlaps without that exact local provenance remain false. This is the
  generated crop used by `PAD` and `CONCAT` adjoints.
- `SLICE_UPDATE`: all six carriers, admitted affine base/update reads, canonical output, both target-
  relative and explicit-coordinate forms, including the zero-entry whole-value replacement at
  scalar or positive rank and the already-supported negative coordinate step when its nonempty
  canonical update geometry is valid. Floating first-order closure is exact for positive-step and
  zero-entry forms. A negative-step occurrence is also CPU-free when the requested target subset
  selects only the base role: `LayoutGradientRules` emits replacement `SLICE_UPDATE` into a typed
  zero and no extraction. Selecting the update role remains blocked because its cotangent requires
  the unadmitted negative-step `SLICE`; selecting both roles remains blocked for the same reason.
- `PAD`: all six carriers, exact same-typed scalar fill bits, admitted affine input, canonical
  output, scalar included, and floating backward through the newly resolved exact crop.
- `CONCAT` and `STACK`: all six carriers, `1..16` ordered admitted affine inputs, canonical output,
  scalar inputs for `STACK`, and floating backward through exact ordered crops/selects.
- `TILE`: all six carriers, admitted affine input, canonical output, scalar included. Forward accepts
  legal gradient metadata, but every positive-rank adjoint uses `SUM`; it is not generalized beyond
  existing qualified accelerator FLOAT32 reduction support. Scalar `TILE` has the Compiler identity
  adjoint and is CPU-free.
- `FOLD_AXIS`: FLOAT64/FLOAT32/BFLOAT16/INT64/INT32, never BOOL, canonical geometry, and only
  `step >= windowSize`, so writes never accumulate. Floating backward closes through `UNFOLD_AXIS`.
- `UNFOLD2D`, `FOLD2D`, `UNFOLD3D`, and `FOLD3D`: retain the Model's three floating carriers and exact
  typed padding. Forward unfold admits all valid static geometry. Fold remains restricted to
  non-overlap on every spatial axis. Floating backward is CPU-free only for non-overlapping unfold
  geometry; fold backward closes through its exact unfold counterpart.

Every replacement gradient preserves independent base/update selection. Generated typed-zero
constants remain request-local splats, and Task 0066's scalar-capable all-carrier `EXPAND` materializes
them. Saved condition/index roles remain live until their exact backward consumers finish. No kernel
reconstructs an index, condition, branch choice, or forward output from public targets. The proof
covers target subsets in which those saved roles are internal only.

If a generated graph accumulates two cotangent paths, performs `SUM`/`SUM_TO_SHAPE`, uses an additive
scatter, or performs an overlapping fold, that arithmetic operation must be independently capable.
Task 0066 proves single-path exact-movement closures and does not infer arithmetic support from a
forward movement row. Higher-order differentiation is outside the current Compiler contract.

### Transfer, publication, and rank-zero boundary

Retain the existing all-six-carrier transfer contract: canonical rank-`0..16` values and positive-
stride non-overlapping storage layouts with checked positive spans may upload/download through the
current typed boundary; BOOL is canonicalized/validated exactly. The resulting authenticated
external `MATERIALIZED_LAYOUT` state is an admitted affine read for every selected custom
occurrence, not only the existing SELECT/SLICE subset. Broaden authenticated dense-affine
publication from its current positive-rank FLOAT32-only remainder to all six carriers, exposing
canonical represented-order bytes for ordinary affine outputs; this publication evidence is bound
to a prepared producer. `SELECT`, explicit positive-step `SLICE`, and fully-static target-relative
crop publications continue to use their authenticated physical spans. Rank-zero ingress and
publication receive direct public Engine coverage for every byte width.

Task 0066 adds no zero-stride or arbitrary overlapping external input, no negative-stride transfer,
no zero-byte transfer, and no alias promise. Local zero-stride `EXPAND` and its authenticated view
descendants do not expand the external transfer contract. Transfer failures still occur before
native invocation and do not create a fallback.

### Fixed custom route and native implementation

Add custom reason `CA_0066` and make exactly ten previously pending movement rows custom-available:
wires 6..11 and 16..19. Split the current shared `DIRECT_INDEX_PENDING_MOVE` catalog mapping so
`GATHER`, `ONE_HOT`, and `SCATTER_ELEMENTS` receive the Task-0066 available entry while wire 70
`SCATTER_ADD` alone retains the pending additive blocker. Every selected wire in the exact 33-wire
set then has a custom implementation. Candidate generation must select `CUSTOM_PROGRAM` only for
every selected occurrence, including the historical FLOAT32 subsets; no old direct/composed
MPSGraph candidate remains for these occurrences. Nested MPSGraph steps for unselected capable
operations may still coexist in one custom whole-program resource in stable topological order.
Rename the ten route-specific native constants from their historical `MPSGRAPH` spelling to
`CUSTOM` while retaining numeric wires exactly; no compatibility alias remains.

Add one Task-0066 native source/header to the existing fixed library and cleanly replace the selected
old kernel dispatches rather than retain dtype-dependent aliases. The implementation may share four
boring kernel families: raw affine/movement, validated index/replacement, integer-defined cast, and
bit-defined predicate/WHERE. Ordinary movement/predicate/cast dispatch gives one thread ownership of
each dense output ordinal. View outputs use one owner per represented physical destination;
broadcast logical views are materialized densely rather than racing writes to an aliased address.
Replacement uses the existing ordered stages: complete validation, base copy, then one writer per
proved-unique exact update destination. No atomic arithmetic, scratch reduction, floating emulation
library, host repair, or payload-dependent route is authorized.

Java preparation and native preflight independently authenticate operation wire, attribute kind and
count, ordered roles, type matrix, rank/Shape, exact logical and physical layout, gradient flags,
output state, index width, static products, byte spans, resource access, alias exclusion, and kernel
metadata. Failure creates no pipeline-visible mutation. Prepared executables remain immutable and
reusable; run-owned storage, lease/close/rollback/suppression semantics, tracing, sessions, and
concurrency retain their current contracts.

### Catalog, schema, identity, and count deltas

- Production operation-kind capability remains exactly `83 true / 32 false`; all changes are inside
  the exact 33 already-true wires.
- Structural execution remains exactly `101 true / 14 false`.
- The MPSGraph catalog remains exactly `75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE`; structural
  selector evidence is not production authorization.
- The custom catalog changes exactly `60 AVAILABLE / 55 PENDING / 0 UNAVAILABLE-WITH-PROOF` to
  `70 AVAILABLE / 45 PENDING / 0 UNAVAILABLE-WITH-PROOF` by moving wires 6..11 and 16..19. The
  shared index entry is split so wire 70 remains pending and does not participate in this delta.
- Keep schema 15, ABI 5, type wires `1..6`, operation wires `1..115`, attribute wires `0..41`, route
  wires `1..3`, generic cardinality/layout grammar, and exactly thirteen exports. No compatibility
  reader or new public enum exists.
- Advance workload-signature, exact-default-policy, candidate-schema, compatibility-schema,
  route-policy, and codec identities together `21 -> 22`. Identity 21 and every older identity
  reject with no alias, migration reader, fallback library, or shim.
- Preserve stable feed/value/target order, whole-partition finalization, synchronous invocation,
  liveness, run-local storage, trace identity, and the profile-free Runtime/Trace boundary.

## Provable closures versus explicit blockers

The maximal selected closure is exactly:

- all six carriers across exact affine/canonicalization and raw movement where Model permits them;
- all 36 cast pairs with exact Model conversion behavior and nine floating gradient pairs;
- FLOAT64/FLOAT32/BFLOAT16 classification and all nine promoted floating `WHERE` signatures;
- scalar and affine BOOL logical/broadcast roles;
- INT64 parity for gather/one-hot/replacement index roles;
- all-carrier axis gather/scatter-none/unfold movement and integer non-overlap `FOLD_AXIS`;
- legal gradient metadata on selected forward movement rows; and
- CPU-free first-order graphs consisting only of inverse affine moves, casts, same-shape `WHERE`,
  positive-step slice/select/pad/concat/stack, unique replacement scatter/gather, and non-overlap
  unfold/fold, with exact saved-role liveness.

Remain explicitly blocked:

- the exact 32 production-false operation kinds listed above;
- additive scatter and every scatter reduction other than `NONE`, overlapping folds/windows, and
  gather adjoints that require those additive operations;
- BFLOAT16/FLOAT64/strict-FLOAT32 `SUM` or `SUM_TO_SHAPE`, positive-rank `TILE` adjoints,
  broadcasted-branch `WHERE` adjoints, `EXPAND` adjoints, and multi-path cotangent accumulation unless
  each generated arithmetic node is independently capable;
- zero extents/empty slices, dynamic or unresolved geometry, rank above 16, noncontiguous unresolved
  reshape, negative-stride view production/publication, arbitrary overlapping layouts, unchecked
  overflow, resource aliasing, and external zero-stride input;
- non-floating differentiation, higher-order gradients, floating arithmetic beyond bit-defined
  conversion/classification/selection, and any transcendental, reduction, attention, or recurrent
  assumption; and
- direct MPSGraph execution for the selected set, host calculation/repair, payload selection,
  timing, autotuning, cache winners, retry, or fallback.

If implementation needs a new type/operation/attribute/route wire, schema field, ABI export, public
API, zero-byte buffer, signed-stride transfer, arithmetic emulation, additive update, runtime route
choice, or narrowed corpus, stop for a new independently reviewed plan rather than silently changing
this one.

## Non-goals

No Model operation meaning, public Tensor API, or Compiler gradient-formula change; the sole
Compiler production change is backend-neutral fully-static target-relative crop inference plus its
narrow post-input-layout closure branch. No Planning/Prepare/Runtime/Trace/Engine API change; no
FLOAT16; no new production kind; no reopening of Task 0053; no general reduction, transcendental,
normalization, loss, attention, random, or recurrent execution; no empty/dynamic tensors; no
arbitrary alias/view transfer; no benchmark; and no rewrite of historical blocked evidence. A later
current 115-kind completeness audit remains separate and must not count this bounded occurrence
closure as semantic support for another row.

## Contracts

- [`ARCHITECTURE.md` — Core lifecycle and invariants](../../../../../ARCHITECTURE.md#core-lifecycle)
  — Model owns profile-indexed meaning; backend occurrence support is fail-closed and route choice
  is cold.
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  — exact movement and bit-defined conversion are common-profile; arithmetic gradients retain their
  primitive/aggregate floors.
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  and [Numerical-profile backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity)
  — truthful occurrence capability, authenticated physical layouts, fixed routes, and no fallback.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle).
- Complete Metal [Task 0059](0059-casts-layout-indexing.md) and
  [Task 0060](0060-exact-replacement-fold-and-aggregate-reductions.md) own the current cast/movement
  and replacement/non-overlap foundations this task broadens without changing their historical
  evidence.
- Compiler [layout/window/indexing gradients](../../../modules/compiler/tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)
  own the generated formulas, saved roles, and additive/overlap boundaries.

## Dependencies and integration

- Depends on: Task 0065 Complete through documentation revision
  `c48b94d7fb2dfa6391901ede580cf886d74cc889`; current Model cast/layout/indexing/window semantics;
  current Compiler first-order formulas and logical-layout closure; completed Tasks 0055..0065
  six-carrier, schema-15, custom-program, layout, replacement, identity, lifecycle, package, and
  public Engine foundations.
- Preserves: Blocked Tasks 0016..0018, 0026..0027, 0030..0037, 0039, 0047, 0051, and 0053
  under their recorded gates. On implementation it supersedes only Task 0040's future-route
  conclusion by replacing its failed direct BFLOAT16 `GATHER` selector with this exact custom raw-
  movement route; the `0xffa6 -> 0x7fc0` failure and all historical evidence remain immutable.
- Conflicts with: every concurrent Metal capability/schema/native/custom-source/catalog/candidate/
  route/preparation/resource/publication/package/shared-document scope; resumed Task-0053 production;
  and any concurrent Compiler layout/gradient or transfer-contract change. Task 0066 owns only the
  backend-neutral fully-static target-relative crop inference and its narrow post-input-layout
  closure branch, not a gradient-formula change.
- Parallel group: None.
- Common base revision: `c48b94d7fb2dfa6391901ede580cf886d74cc889`.
- Integration order: independent plan review, then one serial implementation/proof cutover, package
  and documentation reconciliation, and independent cumulative Class C review. No production edit
  precedes approved planning.
- Integration validation: native build/sign/package/verification; complete Metal; focused Compiler
  source-backed generated-graph contracts; conformance; CPU-free packaged-Metal public Engine
  forward/backward/negative evidence; Javadoc/architecture; one full repository test/build pass;
  exact registry/catalog/identity/export audits; documentation/diff; final independent review.
- Shared-document integration owner: Task-0066 coordinator.

## Files and symbols

Implementation ownership:

- Compiler `LayoutInference` target-relative slice derivation and
  `StaticResultLogicalLayoutClosure`'s narrow post-input-layout crop retry, with backend-neutral
  Javadocs and focused direct plus generated PAD/CONCAT graph tests for initially unresolved
  derivative splats and static/resolved versus dynamic/empty/unresolved cases;
- `CapturedGraphInference`'s matching layout-only deferral, explicitly restricted to `SLICE`, so
  the inference/closure pass can resolve that crop without broadening `SLICE_UPDATE` or changing a
  gradient formula;
- `AutogradPreflight`'s existing 3D window counterpart validation, corrected to compare semantic
  type, Shape, and gradient properties rather than an incidental intermediate layout;
- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNegPartitionPreparer`,
  `MetalNegPreparationPlan`, `MetalNativeApi`, `MetalOperationRouteCatalog`, custom candidate/tuning/
  codec identities, transfer/publication authentication, and affected package/type Javadocs;
- native foundation preflight/metadata/pipeline/step/binding code plus
  `synaptik_task0066_dtype_layout_kernels.h`, with no new export or schema field;
- focused capability/catalog/schema/native/malformed/raw-word/index/layout/rank-zero/bounds tests,
  including allocation-free overflow and native-parity rejection;
- current Compiler contract suites plus a dedicated CPU-free Engine Metal integration for all six
  carriers, cast matrix representatives, scalar/view/broadcast composition, saved condition/index
  roles, exact first-order generated graphs, reuse/concurrency/sessions, and early failures; and
- current Metal/backend/native guides, capability/preparer status, architecture identity claims,
  this brief, master plan, and roadmap.

No Model or Compiler gradient-formula edit occurred. The three Compiler accommodations above are
the complete backend-neutral Compiler source scope used by the implementation.

## Acceptance criteria

1. The exact selected set is wires `6..11,16..19,39..45,51,69,71..84` (33 kinds). All other 82
   kinds retain their current occurrence domains, and the named 32 false rows remain false for the
   recorded semantic reasons.
2. All selected descriptors obey exact Model dtype, role, Shape, layout, output, and gradient
   relations. Static rank-zero works; positive extents and checked spans are mandatory; zero extent,
   unresolved/dynamic, negative-view, alias, and overflow cases reject before resource creation.
3. Every one of the 36 cast pairs matches an independent bit oracle. The 17 new pairs, nine floating
   gradient pairs, NaN policies, ties-to-even, direct BFLOAT16 rounding, subnormals, signed zero,
   infinities, saturation, modulo narrowing, and BOOL mapping are covered without hardware-double or
   production-oracle expected values.
4. All-carrier affine/contiguous movement preserves raw words through scalar, nonzero-offset,
   permutation, singleton, broadcast, and canonicalization chains. Public evidence includes
   `EXPAND -> PERMUTE/SLICE -> consumer`, with safe derived physical descriptors despite exact
   zero-stride logical descendants. Logical versus physical layout is authenticated explicitly; no
   broadcast destination race or unowned physical write exists.
5. FLOAT64/FLOAT32/BFLOAT16 classification is bit-defined. BOOL logical rows and all nine mixed
   floating `WHERE` signatures obey scalar/affine/broadcast semantics, canonical BOOL, selected-
   branch-only conversion, exact promotion, and branch-gradient roles.
6. Axis and ND index operations support both signed index carriers over the declared data domains.
   Data, update, and index reads accept authenticated canonical, external `MATERIALIZED_LAYOUT`,
   dense-materialized, or local affine physical descriptors; public chains feed integer `PERMUTE`,
   `SLICE`, and `EXPAND` results directly into each index consumer. Complete physical-address
   prevalidation precedes mutation. Every negative value rejects without wrapping or normalization;
   non-negative bounds, replacement uniqueness, canonical BOOL one-hot, and complete validation-
   before-mutation hold for scalar and positive-rank cases.
7. First-order CPU-free evidence covers floating casts, inverse affine movement, positive-step
   SELECT/SLICE/SLICE_UPDATE, negative-step base-only SLICE_UPDATE target subsets, PAD,
   CONCAT/STACK, unique replacement scatters, same-shape WHERE, and non-overlap window pairs. Exact
   saved condition/index roles remain live for target subsets and are neither recomputed nor
   published solely to make backward work. Focused Compiler tests prove negative-step SLICE_UPDATE
   base-only generation and preserve the update-selected negative-SLICE blocker. Direct Compiler
   tests and actual generated PAD/CONCAT graphs from a scalar seed's dense-materialized `EXPAND`
   prove the fully-static target-relative crop closes after the derivative splat's logical layout
   initially resolves to zero strides. Dynamic/empty/unresolved cases remain false. The Compiler
   rule remains backend-neutral; Metal rejects unsafe physical layout, not the authenticated local
   zero-stride logical crop.
8. Negative evidence proves additive gather adjoints, overlapping windows/folds, arithmetic scatter,
   positive-rank TILE reduction, broadcasted differentiable branches, EXPAND reductions,
   multi-path arithmetic accumulation, and all 32 false kinds remain outside this closure.
9. Every selected partition has only `CUSTOM_PROGRAM`. Exactly wires 6..11 and 16..19 move pending
   to custom-available; no MPSGraph candidate, retry, fallback, timing, autotuning, payload decision,
   or host repair exists.
10. Counts finish at `83/32` capability, `101/14` structural, `75/35/5` MPSGraph, and `70/45/0`
    custom. Schema 15, ABI 5, six type wires, 115 operation wires, 42 attribute wires, three route
    wires, and thirteen exports remain; all backend-local identities are 22 and identity 21 rejects.
11. Packaged-native/prepared-Metal evidence proves preflight parity, malformed rejection, no partial
    writes, fixed routes, raw-word results, and checked bounds. CPU-free packaged-Metal public Engine
    proves rank-zero/all-width transfer, forward chains, generated backward graphs, saved roles,
    target subsets, prepared reuse, concurrent/independent sessions, and named early failures with
    no skip or CPU fallback.
12. Native/package, complete Metal, focused Compiler, conformance, dedicated Engine, Javadoc,
    architecture, Markdown/link/diff, and one full repository test/build pass; independent cumulative
    Class C review has no unresolved P0/P1/P2.

## Validation after implementation

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

No timing or benchmark command was run. Native build/sign/package/verification, the Gradle package
tasks, complete Metal tests, focused Compiler contracts, Metal conformance, the dedicated CPU-free
public Engine integration, Metal Javadoc, architecture tests, `./gradlew test`, `./gradlew build`,
and `git diff --check` all passed for the implementation and documentation cutover.

## Documentation and review impact

Implementation must update every current selected-domain, custom-count, route, identity, cast,
layout, index, transfer, and gradient statement while preserving the current operation-kind counts
and all semantic blockers. Public/backend/native/Javadoc wording must distinguish operation-kind
truth from occurrence breadth, logical views from physical materialization, forward gradient
metadata from a CPU-free generated backward graph, and authenticated external transferred
materialization from unauthenticated view aliases. Historical tasks remain immutable evidence
records.

Independent plan review verified the exact 115-row partition, 33-wire selection, `+10` custom
catalog arithmetic, 36-pair cast matrix, Model gradient relations, Compiler-generated operations,
saved roles, scalar/empty distinction, physical-layout safety, transfer boundary, fixed-route
policy, and non-overlap with the future full registry audit, then returned `APPROVE` with zero
P0/P1/P2 findings. Independent final review must inspect the full plan-to-source proof, integer
conversion arithmetic, exhaustive BFLOAT16 evidence, layout address checks, validation-before-
write, saved-role liveness, route exclusivity, identity invalidation, public Engine behavior,
documentation, and changed-path scope.

## Planning result

The maximal reachable closure is an occurrence-domain and fixed-route cutover inside 33 existing
production kinds, not a new semantic family. It plans all six exact movement carriers, all 36 cast
pairs, floating classification/promotion, BOOL scalar/broadcast roles, INT64 indexing parity,
rank-zero composition, and exact first-order movement/replacement/non-overlap graphs. It preserves
zero-extent/signed-stride/overlap/additive/reduction/transcendental/attention/recurrent blockers and
all 32 production-false rows.

Implementation will retain capability `83/32`, structural execution `101/14`, and MPSGraph catalog
`75/35/5`; move exactly ten custom rows to finish `70/45/0`; retain schema 15, ABI 5, and thirteen
exports; and advance every backend-local identity from 21 to 22.

## Implementation result

Implementation `0b88f897` completes exactly wires
`6..11,16..19,39..45,51,69,71..84` through one selected-occurrence `CUSTOM_PROGRAM` route. Java
preparation and native preflight independently derive and authenticate logical versus physical
layout geometry; native kernels implement raw movement, validated replacement/indexing, all 36
casts, bit-defined classification/WHERE, scalar values, and non-overlapping window movement. The
public Engine integration covers all carrier widths, scalar and affine publication, saved
condition/index roles, generated backward graphs, prepared reuse, independent/concurrent sessions,
and early rejection of additive scatter, direct EXPAND adjoint reduction, and other named blockers.

The resulting boundary is capability `83/32`, structural execution `101/14`, MPSGraph catalog
`75/35/5`, custom catalog `70/45/0`, schema 15, ABI 5, thirteen exports, and backend-local identity
22. Identity 21 and every older identity reject. Task 0040's historical direct BFLOAT16 GATHER
failure remains unchanged; only its future-route conclusion is superseded by the exact custom raw
movement route.
