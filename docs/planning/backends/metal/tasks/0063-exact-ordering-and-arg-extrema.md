# Task 0063: Exact Ordering and Arg Extrema

## Status

Implementation complete; final review pending — the approved plan review and external P1 correction
preceded implementation `9931d5f8`. The implementation bounds every custom dispatch and logical
index to the existing unsigned-32-bit Metal contract, preserves schema 15/ABI 5/thirteen exports,
and advances the exact counts and private identities specified below. Native build/sign/package,
packaged Metal, conformance, CPU-free public Engine, Javadoc, and architecture validation pass.

## Change class

Class C — implementation will widen profile-common Metal capability, add exact custom-kernel
execution, complete native multi-output handling, advance private route identities, change public
Engine behavior, and update authoritative backend documentation. Use one clean serial implementation
context only after independent review of this plan, then an independent cumulative code, evidence,
documentation, and security review.

## Goal

Implement the maximal deterministic current Metal domain for `SORT`, `ARGSORT`, `TOP_K`,
`ARG_MAX`, and `ARG_MIN` through one fixed exact custom-program route. Preserve Model's represented-
value, NaN, signed-zero, stability, tie, index, Shape, output-role, and gradient-metadata contracts
under both numerical profiles. Select the route before resource creation and use no direct-selector
assumption, timing, autotuning, runtime choice, retry, fallback, or host repair.

## Scope

### Exact audited 115-kind registry boundary

The current registry has exactly five ordering/arg-extrema kinds in scope:

| Wire/kind | Arity and attribute wire | Exact Model role | Pre-cutover Metal state | Task-0063 decision |
|---|---|---|---|---|
| 94 `SORT` | 1 input/1 output; 31 `SORT` = `[axis, descending]` | stable represented values, same type/Shape/gradient metadata | non-executable; `DIRECT / MD_SORT`; custom `PENDING / CP_AGGREGATE` | exact custom route |
| 95 `ARGSORT` | 1/1; 31 `SORT` | stable logical indices, fixed INT64/no-grad | non-executable; `COMPOSED / MC_INDEX64`; custom pending | exact custom route |
| 96 `TOP_K` | 1/2; 32 `TOP_K` = `[axis, k, largest, sorted]` | values slot 0 plus mandatory INT64 indices slot 1 | non-executable; `COMPOSED / MC_INDEX64`; custom pending | exact custom route for positive K |
| 109 `ARG_MAX` | 1/1; 36 `ARG_EXTREMA` = `[axis, keepDimensions, tiePolicy]` | fixed INT64 logical index/no-grad | non-executable; `COMPOSED / MC_INDEX64`; custom pending | exact custom route |
| 110 `ARG_MIN` | 1/1; 36 `ARG_EXTREMA` | fixed INT64 logical index/no-grad | non-executable; `COMPOSED / MC_INDEX64`; custom pending | exact custom route |

The existing schema-15 validators already require normalized axis words and Boolean flag words;
`TOP_K` additionally requires positive K, and arg-extrema tie wire values are exactly
`FIRST_INDEX=1` and `LAST_INDEX=2`. The image already represents variable node-output cardinality,
including both ordered `TOP_K` references. The native decoder/execution owners currently retain only
the first output and therefore remain incapable of executing that valid two-output record.
`MetalCapabilityProvider` currently has no ordering branch and rejects any output list whose size is
not one, so all five production capabilities are false. Implementation must handle exact two-output
TOP_K cardinality before the retained single-output gate and must not make another multi-output kind
eligible.

No other registry kind joins this task. Existing comparison/extrema/scan wires retain their prior
bounded capability and executable state: 20..25 `GT/GE/LT/LE/EQ/NE` are
`DIRECT / MD_PRED`, 26/27 `TENSOR_MIN/MAX` and 30 `CLAMP` are `DIRECT / MD_ARITH`, 28/29
`SCALAR_MIN/MAX` are `COMPOSED / MC_SCALAR`, and 31..34
`REDUCTION_MIN/REDUCTION_MAX/CUM_SUM/CUM_PROD` are `DIRECT / MD_REDUCE`; all use available
`CA_0052` custom routes. Pooling wires 97/98 remain `DIRECT / MD_POOL2D` and 99/100 remain
`COMPOSED / MC_WINDOW3D`, all custom-pending; only maximum-pool wires 97 and 99 have winner
selection, and their generated `ARG_MAX` use does not make a pool occurrence an ordering
occurrence. Loss, normalization, statistical reduction, random, recurrent, convolution, attention,
and every other structural-only row retain their current decisions.

### Exact production domain

Admit each occurrence under both `STRICT_IEEE` and `ACCELERATOR` only when all of the following hold:

- the sole input and every output have fully static, canonical dense, zero-offset, non-view storage;
  input rank is `1..16`, every input dimension is positive, every output has its exact derived
  canonical Shape, and all output storage is fresh and pairwise non-overlapping with every input and
  other output;
- before capability returns true, checked `long`/`uint64_t` arithmetic proves that the input element
  count, every output element count, every dimension, canonical element stride, slice count,
  selected-axis extent, K, custom thread count, and one-dimensional grid width are each in
  `1..UINT32_MAX`, while every derived logical coordinate, rank, compacted position, and linear
  index is in `0..UINT32_MAX` and strictly below its governing count/extent when used for access.
  Validation occurs before any unsigned-32-bit narrowing; a wider metadata carrier does not widen
  the admitted index domain;
- byte widths, byte offsets/spans, K/output products, metadata sizes, and every
  `NSUInteger`/buffer conversion additionally pass the existing checked native-size and allocation
  bounds without wrap;
- legal floating input descriptors may be either no-grad or gradient-bearing; integral and Boolean
  inputs retain their Model-legal no-grad metadata; no capability decision inspects values;
- `SORT` and `ARGSORT` accept FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, and BOOL for either direction;
  `SORT` returns the same type and Shape with the exact input gradient flag, while `ARGSORT` returns
  the same Shape as INT64/no-grad;
- `TOP_K` accepts the same six types, both `largest` values, both `sorted` values, and
  `1 <= k <= selectedExtent`; values at slot 0 have the input type and gradient flag, indices at
  slot 1 are INT64/no-grad, and both outputs have the one exact Shape obtained by replacing the
  selected dimension with K;
- `ARG_MAX` and `ARG_MIN` accept FLOAT64, FLOAT32, BFLOAT16, INT32, and INT64, both keep-dimension
  forms, and both tie policies; BOOL is invalid, and the result is always INT64/no-grad with the
  selected dimension removed or replaced by extent one exactly as requested; and
- repeated input identity and arbitrary fan-out are valid. A `TOP_K` node always computes both
  ordered outputs even when only one is a graph target; either or both roles may be published,
  reused internally, repeated as targets, or consumed by later admitted custom-program nodes.

The fixed dispatch truth table is:

| Family | One-dimensional thread count | Additional indexed quantities |
|---|---:|---|
| `SORT`, `ARGSORT` | exact input element count | selected-axis extent and every destination rank |
| `TOP_K` | exact input element count | selected-axis extent, K, selected rank, and compacted output position |
| `ARG_MAX`, `ARG_MIN` | exact output element count, including scalar count one | selected-axis scan extent and input/output linear coordinates |

Every count and extent in the table is in `1..UINT32_MAX`; every coordinate, rank, position, and
linear index is unsigned-32-bit representable and strictly below its governing count/extent when
used for access. Java capability/preparation and native image validation enforce the same boundary
before pipeline or buffer creation. The kernel uses no multi-dimensional flattening or 64-bit
dispatch strategy, and an out-of-domain occurrence fails closed without MPSGraph or host fallback.

This is the maximal exact route inside the current static canonical unsigned-32-bit Metal custom
materialization boundary: there is no arbitrary value subset, direction, axis, carrier, positive K,
tie policy, or Shape carve-out within that representable domain. Dynamic, scalar, empty,
over-`UINT32_MAX` dispatch/index geometry, arbitrary-view, unresolved, negative/zero-stride,
overlapping, aliased, in-place, and K-zero occurrences remain outside that boundary.

### Exact ordering and selection semantics

The route must implement Model's total output policy rather than the host language or Metal shading
language's partial floating comparison:

- For `SORT`, `ARGSORT`, and `TOP_K`, all non-NaNs use numerical order. Ascending/smallest places
  negative infinity first and positive infinity last; descending/largest reverses only that non-NaN
  order. Negative zero is below positive zero in ascending/smallest order and above it after the
  direction reversal.
- Every floating NaN is one equal final class after every non-NaN for both directions. NaN sign,
  payload, and signaling/quiet state do not affect placement, but value-producing operations copy
  the exact selected raw representation.
- Equal finite, integral, or Boolean values and multiple NaNs use increasing original logical-axis
  index. This index tiebreak is not direction-reversed. INT32/INT64 use signed mathematical order;
  BOOL uses `false < true`.
- `TOP_K` selects exactly the first K pairs in the requested complete stable order. With
  `sorted=true`, output pairs retain that order. With `sorted=false`, the exact same selected set is
  deterministically reordered by increasing original logical-axis index; values and indices never
  separate.
- For both arg-extrema operations, a NaN is preferred to every non-NaN; multiple NaNs tie. Among
  non-NaNs, negative zero is below positive zero, infinities and finite values use numerical order,
  and signed integers use mathematical order. `FIRST_INDEX` or `LAST_INDEX` selects the exact
  logical-axis coordinate among equal extrema.

These are exact, profile-common policies. Ordering and arg-extrema receive no DAZ, FTZ, epsilon,
tolerance, reassociation, canonical-NaN, signed-zero, approximate-index, or final-only relaxation.

### Fixed custom route and native algorithm

Do not authorize any MPSGraph selector. Installed `MPSGraphSortOps.h`
`sortWithTensor:axis:descending:name:` does not guarantee unconditional stability, NaNs-last,
signed-zero/subnormal order, represented-bit copying, all six carriers, or every admitted Shape;
`argSortWithTensor:axis:descending:name:` explicitly returns INT32 rather than Model's INT64.
Installed `MPSGraphTopKOps.h` axis forms `topKWithSourceTensor:axis:k:name:` and
`bottomKWithSourceTensor:axis:k:name:` do not close index type, stable cutoff ties,
NaN/signed-zero, exact-bit, output-order, or `sorted=false` contracts. Installed
`MPSGraphReductionOps.h` `reductionArgMaximumWithTensor:axis:name:` and
`reductionArgMinimumWithTensor:axis:name:` expose neither Model's explicit tie policy nor the
required INT64 and floating-special-value guarantees. Historical Task 0034 therefore remains a
truthful blocked direct-selector record; Task 0063 uses its separately named exact-custom unblock,
not a reinterpretation of that evidence.

Add one Task-0063 custom Metal source/header whose comparisons are integer-only:

- classify floating NaNs from raw exponent/fraction fields before key comparison;
- for each non-NaN BFLOAT16, binary32/FLOAT32, or binary64/FLOAT64 raw carrier, map a negative word
  to its bitwise complement and a non-negative word by toggling its sign bit. Unsigned key order is
  then exact numerical order with `-0 < +0`, including subnormals and infinities;
- map INT32/INT64 signed order by toggling the sign bit and comparing unsigned keys; compare BOOL as
  canonical byte zero/one; and
- compare the original logical-axis index only after semantic equality. NaNs remain one semantic
  class, and descending/largest reverses the non-NaN value relation without reversing this tie key.

For `SORT`, `ARGSORT`, and `TOP_K`, dispatch one one-dimensional grid whose width is the exact input
element count in `1..UINT32_MAX`. Each kernel reads only the 32-bit
`thread_position_in_grid.x`; it does not reconstruct a wider identifier from `y`/`z`. The thread
derives its slice and source axis coordinate, computes that source's unique stable rank by counting
strict predecessors under `(semantic order, original index)`, and writes only its unique
destination. `SORT` copies raw carrier bits; `ARGSORT` widens the validated non-negative 32-bit
logical coordinate to INT64 only when storing the public result. `TOP_K` writes nothing when rank is
at least K; otherwise one thread writes both paired outputs, using `rank` for sorted output or the
number of earlier source coordinates whose stable rank is below K for unsorted output. Recomputing
ranks for the unsorted form is permitted; no scratch, allocation, atomic, race, timing choice, or
data-dependent route exists.

For arg extrema, dispatch one one-dimensional grid whose width is the exact positive output element
count at most `UINT32_MAX`. Scan the validated selected-axis extent in increasing 32-bit logical
coordinate, maintain one raw best word and INT64 best index under the NaN/value/tie invariant, and
publish only the widened index. All dimension, stride, slice, rank, count, coordinate, and logical
linear-index arithmetic stays in the prevalidated unsigned-32-bit domain; byte addressing retains
the separate checked native-size contract. No index wraps or narrows from an out-of-range carrier.
Every kernel traversal is half-open (`index < extent`), never computes `extent + 1`, and therefore
terminates at an admitted `UINT32_MAX` extent without increment wrap.

A `TOP_K` custom step binds one input, values output, indices output, and immutable metadata in one
occurrence. Extend the internal decoded-node and program-step representations from a first-output
field to the bounded ordered output-reference set already carried by schema 15. Validate, account,
allocate, bind, store, consume, and publish every output through used/produced-value checks, type/
Shape/gradient validation, byte/device/alias checks, target mapping, lifecycle rollback, and reuse.
Do not broaden nested MPSGraph steps or another multi-output kind. All validation completes before
resource creation or any device write.

Every partition containing one of the five admitted kinds selects `CUSTOM_PROGRAM` only. Candidate
generation must never offer MPSGraph for such an occurrence, including singleton SORT. A custom
program may compose these kernels with already approved custom or nested-MPSGraph steps while
preserving topological value order and both `TOP_K` values.

### Complete-domain proof

Authorization is source-structural and algebraic, never a sampled device-output inference:

1. Prove the raw floating classifier partitions NaN from every non-NaN encoding. Prove separately
   for each sign partition that complement/sign-bit-toggle is a monotone bijection over all
   non-NaN encodings and joins the partitions at `-0 < +0`; direction reversal affects only that
   order. Enumerate all 65,536 BFLOAT16 words as a complete machine check and cover binary32/64
   exponent/fraction partitions algebraically.
2. Prove sign-bit toggle is a bijection from INT32/INT64 signed order to unsigned-key order and
   exhaust BOOL. Retain adversarial native corpora at signed extrema and zero/one boundaries.
3. Prove `(semantic class/key, increasing logical index)` is a strict total order, every source has
   one distinct predecessor count in `0..N-1`, and distinct threads therefore have distinct writes.
   Prove raw-copy exactness, both TOP_K selected-set/output-order formulas, and the arg-extrema loop
   invariant for both operations and tie policies.
4. Prove every dimension, element count, stride, slice count, selected extent, K, rank, compacted
   position, linear index, thread count, and grid width is checked in a wider carrier and is at most
   `UINT32_MAX` before narrowing or resource creation; every access index is strictly below its
   governing bound. Prove the one-dimensional kernel never derives a wider logical index and uses
   only half-open traversals that terminate at the exact limit without `extent + 1` or increment
   wrap. Exercise exact `UINT32_MAX` acceptance and `UINT32_MAX + 1`, product-overflow,
   stride-overflow, and byte/span rejection in allocation-free Java/native validation tests.
5. Cross-check an independently written Java raw-word oracle against real-device kernels for all
   carriers and operation modes. Include NaN signs/payloads/signaling forms, both zeros, subnormals,
   normals, infinities, repeated values, INT extrema, and BOOL. Exhaust all small finite sequences
   over a boundary alphabet to exercise stability, cutoff, unsorted compaction, and ties; use
   representative rank/axis and safely materializable near-boundary cases only as implementation
   corroboration, not scope authorization.
6. Independently review the exact generated Metal source, fixed entrypoint selection, metadata,
   unsigned-32-bit logical-index closure, integer widths, byte-address derivation, output
   disjointness, absence of floating comparisons and hidden route choices, and the proof-to-source
   correspondence.

No benchmark or timing result is an acceptance gate or route input. If the implementation cannot
supply this proof for one carrier or form, remove that entire carrier/form from a newly reviewed
plan before production edits rather than admitting a sampled subset.

### Compiler gradients and public callers

Model and Compiler remain unchanged. Audit `TensorSortExpressions`, `TensorTopKExpressions`,
`TensorArgExtremaExpressions`, their result carriers, inference, `AutogradPreflight`,
`OrderingGradientRules`, `ReductionGradientRules`, pooling callers, and public Tensor overloads.
Task 0063 admits forward occurrences with the exact metadata above but claims no generated ordering
backward route:

- floating `SORT` backward constructs matching INT64 `ARGSORT` and replacement
  `SCATTER_ELEMENTS`;
- `TOP_K` values backward consumes the original INT64 indices and replacement
  `SCATTER_ELEMENTS`; and
- ARGSORT, TOP_K indices, ARG_MIN, and ARG_MAX are non-differentiable.

Metal's current `SCATTER_ELEMENTS` domain requires canonical INT32 indices, so both generated
floating backward graphs remain unsupported as complete partitions. Do not widen scatter or claim
pooling gradients in this task. Prove CPU-free public Engine forward execution for legal no-grad and
gradient-metadata inputs, both profiles, every family/output role, two-output publication, nested
consumers, fan-out, retained executable reuse, and independent sessions. With no CPU registered,
compile an allocation-free static descriptor whose logical count or derived index extent is
`UINT32_MAX + 1` and prove Metal declines ownership before native invocation or input/output
allocation. Generated ordering backward, pooling, K-zero, empty/dynamic/view/alias, wrong-role,
malformed, and every other over-limit graph must fail before resources/native invocation.

### Catalog, schema, identity, and lifecycle

- Mark only wires 94, 95, 96, 109, and 110 structurally executable: `88/27 -> 93/22`.
  Production capability becomes `70/45 -> 75/40`.
- Keep the MPSGraph catalog exactly `75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE`: SORT remains
  `DIRECT / MD_SORT`; the other four remain `COMPOSED / MC_INDEX64`. Catalog structure is not
  selector authorization.
- Add `CA_0063`, change exactly those five custom rows from pending to available, and produce custom
  catalog `52 AVAILABLE / 63 PENDING / 0 UNAVAILABLE-WITH-PROOF`.
- Keep schema 15, ABI 5, type wires `1..6`, operation wires `1..115`, attribute wires `0..41`, route
  wires `1..3`, the fixed image grammar, and thirteen exports. Advance workload-signature,
  exact-default-policy, candidate-schema, compatibility-schema, route-policy, and codec identities
  together `18 -> 19`; identity 18 fails closed with no alias, migration reader, or compatibility
  shim.
- Preserve cold deterministic route selection, stable value/feed/target order, transactional
  finalization, synchronous invocation, direct and internal output liveness, leases, rollback and
  suppression, reuse, concurrency/session isolation, close ordering, trace identity, and the
  profile-free Runtime/Trace boundary.

## Provable exact domain versus blockers

The exact domain is the complete static canonical positive-rank/positive-extent matrix whose counts
and extents are representable in `1..UINT32_MAX` and whose derived coordinates/indices are
representable in `0..UINT32_MAX` as specified above. Its proof is constructive and independent of
opaque MPSGraph ordering behavior.

Remain explicitly blocked:

- scalar, dynamic, expression, zero-extent, arbitrary affine/view/overlapping/in-place geometry;
- any input/output element count, dimension, stride, slice count, selected extent, K, rank, compacted
  position, logical linear index, or thread/grid count above `UINT32_MAX`, even when `long`,
  `uint64_t`, `NSUInteger`, or the byte span can represent it;
- Model-valid `TOP_K(k=0)`, because the current schema/native custom materialization contract is
  positive and this task adds no zero-dispatch/publication route;
- BOOL arg extrema, which Model rejects; user comparators, named axes, unstable sorting, partial or
  percentage K, kth-value, search/partition, lexicographic ordering, and hidden outputs;
- any direct MPSGraph ordering/selection/arg-extrema selector, selector-result cast/repair, host
  sorting or validation, sampled-value authorization, retry, fallback, timing, autotuning, cache
  winner, or performance claim;
- generated SORT/TOP_K backward and maximum-pool forward/backward until their independent complete
  scatter/window routes exist; and
- every non-ordering structural-only operation and resumed Task-0053 production.

## Non-goals

No Model, Compiler, public Tensor, Config, Planning, Prepare, Runtime, Trace, or Engine API change;
no schema field, wire, ABI export, general multi-output MPSGraph support, arbitrary layout, dynamic
binding, empty-domain kernel, 64-bit or multi-dimensional custom dispatch, pooling, scatter
widening, host calculation, benchmark, or new public option. No direct selector is rehabilitated.
No historical blocked result is rewritten.

## Contracts

- [`ARCHITECTURE.md` — Core lifecycle and Core invariants](../../../../../ARCHITECTURE.md#core-lifecycle)
  — Model owns exact profile-indexed meaning; route choice is cold and Runtime/Trace stay
  profile-free.
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  — ordering, selected values, indices, and ties receive no accelerator relaxation.
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  and [Numerical-profile backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity)
  — Metal owns truthful occurrence capability, fixed routes, native state, and private identities.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle).
- [Model sort/argsort](../../../modules/model/tasks/0019c-sort-and-argsort.md),
  [top-K](../../../modules/model/tasks/0019c1-top-k-values-and-indices.md), and
  [shared arg extrema](../../../modules/model/tasks/0018u1-integral-reductions-and-arg-min-normalization.md)
  own the exact operation semantics. [Compiler ordering gradients](../../../modules/compiler/tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)
  owns formula topology; [CPU ordering](../../cpu/tasks/0006c-portable-stable-ordering-and-selection.md)
  is an exact precedent, not Metal authorization.

If implementation needs a new result freedom, a wider or different dispatch/index domain,
opaque-selector assumption, runtime/value-dependent route, scratch allocation, shared lifecycle/API
change, schema/ABI field, or wider scatter/pooling ownership, stop for a new independently reviewed
plan.

## Dependencies and integration

- Depends on: Task 0062 Complete through `06c57844`; current Model ordering/top-K/arg-extrema and
  Compiler gradient contracts; Task 0003's existing `1..UINT32_MAX` custom thread/grid contract;
  and completed Metal six-carrier transfer, INT64 materialization, schema-15, custom-program,
  catalog, identity, lifecycle, and public Engine foundations.
- Supersedes only the future-route conclusion of historical Task 0034 by choosing its exact custom
  unblock. Task 0034 remains Blocked as a direct-MPSGraph task and its evidence remains immutable.
- Conflicts with: every concurrent Metal capability/schema/native/custom-source/multi-output/route/
  package/shared-document scope and resumed Task-0053 production.
- Parallel group: None.
- Common base revision: `b369b8aa`.
- Integration order: independent plan review, then one serial proof/kernel/multi-output/capability/
  preparation/execution/test/documentation cutover; no production edit before review.
- Integration validation: native build/sign/package/verification, focused Metal, conformance,
  CPU-free public Engine, Javadoc/architecture/docs, exact count/identity/export audits, and
  independent cumulative Class C review.
- Shared-document integration owner: Task-0063 coordinator.

## Files and symbols

Expected implementation owners:

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNegPartitionPreparer`,
  `MetalNegPreparationPlan`, `MetalNativeApi`, `MetalOperationRouteCatalog`, custom route candidate/
  tuning/codec identity owners, and affected package/type Javadocs;
- native foundation decoded-node/program-step/output validation and binding, plus one Task-0063
  integer-ordering kernel source/header and the native package guide;
- focused capability/catalog/schema/malformed/prepared/native/comparator-proof tests, including
  allocation-free `UINT32_MAX`/`UINT32_MAX + 1` dispatch/index boundary and overflow parity,
  `MetalNegCapabilityPartitionConformanceTest`, and a dedicated CPU-free
  `EngineOrderingMetalIntegrationTest` for observable output roles, over-limit pre-resource
  rejection, and lifecycle; and
- Metal/backend/root architecture and capability documentation, targeted API/glossary status,
  ADR identity claims if current, this brief, master plan, and roadmap.

The original planning-only revision changed exactly this brief, the Metal master plan, and the
roadmap. Implementation `9931d5f8` performs the reviewed production, native, and test cutover;
the subsequent documentation revision updates the authorized explanatory and status surfaces.

## Acceptance criteria

1. The five-wire audit and tests retain exact kind, arity, output role/order, attribute, type, Shape,
   layout, metadata, axis, direction, K, tie, index-width, unsigned-32-bit dispatch/index cap,
   byte/span overflow, and profile decisions. Every excluded occurrence rejects before resource
   creation.
2. The integer-only raw comparator and algorithm proof cover the complete admitted carrier domains;
   BFLOAT16/BOOL exhaustive checks, floating/integer partition proofs, small-sequence exhaustive
   checks, adversarial real-device corpora, and independent source review agree exactly.
3. SORT/ARGSORT stability, NaNs-last, signed-zero and exact-copy behavior; TOP_K selected-set and
   both output orders; and arg-extrema NaN preference/ties are exact on every admitted axis/rank.
4. Every logical coordinate/index is checked in a wider carrier before narrowing, is unsigned-
   32-bit representable, and is strictly below its governing access bound; every positive
   count/extent/dispatch quantity remains in `1..UINT32_MAX`. Half-open exact-limit termination,
   one-past-limit rejection, and overflow rejection have Java/native parity and occur before
   pipeline, buffer, or other resource creation.
5. One TOP_K step validates, materializes, writes, retains, consumes, and publishes both ordered
   outputs. Values and INT64 indices remain paired under target subsets, nesting, fan-out, reuse,
   repeated targets, sessions, rollback, and close; malformed second-output state cannot hide.
6. Every admitted occurrence has only `CUSTOM_PROGRAM`; direct MPSGraph, host repair, retry,
   fallback, timing, autotuning, multi-dimensional/64-bit dispatch, and value-dependent selection
   are absent.
7. Counts become `75/40` capability, `93/22` structural, `75/35/5` MPSGraph, and `52/63/0` custom;
   schema 15, ABI 5, and thirteen exports remain; all backend-local identities are 19 and identity
   18 rejects.
8. CPU-free public Engine proves both profiles, six ordering/top-K carriers, five arg-extrema
   carriers, all output roles and representative axes/directions/K/forms/ties, exact raw values and
   INT64 indices, input preservation, composition, reuse, and independent sessions with no skip.
   An allocation-free one-past-limit static graph, other excluded domains, and generated
   SORT/TOP_K backward reject without native invocation or resource allocation.
9. Native/package, complete Metal, Metal conformance, dedicated Engine, Javadoc, architecture,
   Markdown/link/diff checks pass; independent cumulative Class C review has no unresolved
   P0/P1/P2 finding.

## Validation after implementation

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
unexpected shared API/dependency/build change; the complete packaged Metal suite, conformance,
public Engine, and architecture gates are stronger for this backend-local cutover.

## Documentation and review impact

The implementation must update every current capability/count/identity/ordering statement and
describe the route as the exact profile-common **unsigned-32-bit-bounded** custom domain, never as
all positive static extents. Public/backend/native/Javadoc wording must name the
`1..UINT32_MAX` count/extent and `0..UINT32_MAX` coordinate/index boundary, pre-resource failure,
two-output publication, direct-selector exclusion, gradient boundary, and remaining blockers.
Preserve Task 0034 as historical direct-selector evidence. Independent plan review precedes every
production edit. Independent final review must inspect the full plan-to-source proof, kernel integer
semantics, cap parity and boundary negatives, metadata and byte/span overflow safety, multi-output
lifecycle, route exclusivity, negative-domain evidence, public Engine behavior, docs, and changed-
path scope. Passing worker execution evidence may be reused unless remediation changes executable
behavior.

## Implementation and proof result

Implementation `9931d5f8` delivers all five wires through the fixed custom-program route. The
source uses only unsigned integer raw-word classification and keys: floating exponent/fraction
tests separate every NaN encoding, sign complement/sign-bit toggle maps each non-NaN sign partition
monotonically and joins at negative-zero-before-positive-zero, signed-integer sign-bit toggle maps
signed order to unsigned order, and BOOL uses its complete `0/1` domain. The stable tuple is
`(NaN class or numerical key, increasing logical index)`; predecessor counting therefore assigns
one distinct rank in `0..N-1` to every source coordinate and one distinct destination writer.
Direction reversal changes only non-NaN key order. TOP_K uses rank `< K` for its selected set and
either rank or the count of earlier selected coordinates for output position. Arg extrema maintain
the best-so-far invariant with NaN preference and explicit first/last replacement.

`MetalOrderingComparatorProofTest` enumerates all 65,536 BFLOAT16 words, exhausts BOOL, checks the
binary32/binary64 exponent/fraction partitions, verifies signed-key boundaries and deterministic
domain samples, and exhausts boundary-alphabet sequences through length four for stable ranks,
cutoffs, compaction, and ties. `MetalOrderingNativeTest` and
`EngineOrderingMetalIntegrationTest` independently compute raw-word expectations and execute all
six ordering/top-K carriers, five arg-extrema carriers, both profiles, directions, largest/smallest,
sorted/compacted forms, both tie policies, signed extrema, zeros, subnormals, infinities, repeated
values, and signed/payload/signaling NaNs on the real device. Exact input and selected raw words,
INT64 logical indices, target subsets, nesting, fan-out, retained reuse, and independent sessions
are observed.

Java capability and Java/native image validation enforce ranks `1..16` and check every dimension,
product, canonical stride, slice/output count, selected extent, K, and grid width in wider carriers
before narrowing. The admitted positive values are at most `UINT32_MAX`; derived coordinates and
indices remain at most `UINT32_MAX`, and every access is strictly below its governing count.
The one-dimensional kernels use only `thread_position_in_grid.x` and half-open loops, so an
exact-limit traversal never computes `extent + 1` or increments a loop variable from
`UINT32_MAX`. Byte offsets multiply into 64-bit `ulong`, while Java/native byte/span validation
remains independent. Allocation-free tests
accept the exact limit and reject one-past/product overflow; excluded Engine graphs and generated
SORT/TOP_K backward requests fail before preparation.

The native bridge was rebuilt, ad-hoc signed with
`io.github.pho001.synaptik.metal.foundation`, packaged, and independently verified. The complete
packaged Metal suite, Metal conformance suite, dedicated three-case CPU-free Engine integration,
Metal Javadoc, and architecture tests pass. Counts are exactly `75/40` capability, `93/22`
structural execution, `75/35/5` MPSGraph catalog, and `52/63/0` custom catalog. Schema 15, ABI 5,
type wires `1..6`, operation wires `1..115`, attribute wires `0..41`, route wires `1..3`, image
grammar, and thirteen exports are unchanged. All backend-local identities are 19 and identity 18
fails closed. No benchmark or timing result participates in the route.
