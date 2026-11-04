# Task 0012: [MPSGraph](../../../../glossary.md#term-mpsgraph) [FLOAT32](../../../../glossary.md#term-float32) [Extrema](../../../../glossary.md#saturation-pole-and-extremum) Reductions

## Reader context

A selector advertised as NaN-propagating did not prove the full minimum/maximum contract.
This gate also tested signed-zero preference and represented subnormal ordering; failures there
stopped implementation despite passing NaN classification.

Reduction axes and Shape described the grouping, while raw input and output words established
the value behavior. An empty-axis identity would preserve bits without arithmetic. The proposed
rank-zero publication was a backend-local result path, not permission to transfer arbitrary
scalars between owners. Model defined the result obligations; Metal could not relax them because
the API had a convenient selector.
The [current exact and numerical reduction domains](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains)
provide the present reading path. The later-frontier statements below belong to this blocked
task's own chronology.

## Status

Blocked

[Metal](../../../../glossary.md#term-metal) 0008 is Complete; 0006, 0007, and 0009–0011 are independently Blocked. The mandatory
real-[device](../../../../glossary.md#term-device) gate below also blocked this task before production changes. Metal 0013 is the sole
authorized Metal frontier.

## Change class

Class C — this extends [Metal](../../../../glossary.md#term-metal) capability, the private typed native [schema](../../../../glossary.md#term-schema), candidate [identity](../../../../glossary.md#term-identity-meanings),
whole-[partition](../../../../glossary.md#partition) [lowering](../../../../glossary.md#lowering), [scalar](../../../../glossary.md#term-scalar) [publication](../../../../glossary.md#term-publication), persistent [MPSGraph](../../../../glossary.md#term-mpsgraph) resources, and public [Engine](../../../../glossary.md#term-engine)
execution while preserving the existing architecture, [ABI](../../../../glossary.md#term-application-binary-interface-abi) exports, storage types, and transfer
contract.

## Goal

Add bounded forward [`FLOAT32`](../../../../glossary.md#term-float32) ordinary `AggregateReductionKind.MIN` and `MAX` through [MPSGraph](../../../../glossary.md#term-mpsgraph)'s
[NaN](../../../../glossary.md#term-nan)-propagating reduction selectors. Support full, single-[axis](../../../../glossary.md#term-axis), and ordered multi-axis [Model](../../../../glossary.md#term-model) forms,
including keep-[dimensions](../../../../glossary.md#dimension), [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) outputs, and the empty-axis [identity](../../../../glossary.md#term-identity-meanings) form. Do not reopen
blocked SUM, [MEAN](../../../../glossary.md#term-mean), or SUM_TO_SHAPE and do not introduce custom [kernels](../../../../glossary.md#kernel).

## Inventory and [scope](../../../../glossary.md#term-scope) decision

The current [Model](../../../../glossary.md#term-model) aggregate inventory is `SUM`, [`MEAN`](../../../../glossary.md#term-mean), `PROD`, `MIN`, `MAX`, `ALL`, `ANY`,
`ARG_MIN`, `ARG_MAX`, `LOG_SUM_EXP`, [`VARIANCE`](../../../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM`.
Ordinary [extrema](../../../../glossary.md#saturation-pole-and-extremum) use [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), `AxisReductionAttrs`, or `MultiAxisReductionAttrs` and
have a complete non-accumulating represented-value contract: [NaN](../../../../glossary.md#term-nan) propagates without a payload or
signaling promise, [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) use numeric order, `MIN` selects [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero), and `MAX` selects
[positive zero](../../../../glossary.md#nan-infinity-and-signed-zero). This is independent of 0007's exact-real SUM/MEAN requirement.

The installed macOS [SDK](../../../../glossary.md#term-sdk) declares
`reductionMinimumPropagateNaNWithTensor:axes:name:` and
`reductionMaximumPropagateNaNWithTensor:axes:name:`. It also declares product and non-propagating
extrema selectors; only the two propagating selectors match the requested NaN policy closely
enough to probe. Product remains separate after 0011 exposed [MPSGraph](../../../../glossary.md#term-mpsgraph) [scalar](../../../../glossary.md#term-scalar) arithmetic,
[subnormal](../../../../glossary.md#term-subnormal), and division failures. `LOG_SUM_EXP`, variance, [standard deviation](../../../../glossary.md#term-standard-deviation), and [L1](../../../../glossary.md#term-l1)/[L2](../../../../glossary.md#term-l2) norms
contain sum/mean or blocked [unary](../../../../glossary.md#term-unary) mathematics in their Model [targets](../../../../glossary.md#term-target), and the installed headers do
not expose a direct independent selector for all five. [Boolean](../../../../glossary.md#data-type--datatype) and arg extrema require [BOOL](../../../../glossary.md#data-type--datatype) or
[INT64](../../../../glossary.md#data-type--datatype) storage and are not part of this numerical [slice](../../../../glossary.md#slice).

## Exact capability domain

Admit one or more existing 0005/0008 [nodes](../../../../glossary.md#node) plus `MIN`/`MAX` occurrences with one input and one
output, identical [`FLOAT32`](../../../../glossary.md#term-float32) type and `requiresGrad`, fully static positive input [Shape](../../../../glossary.md#shape), [rank](../../../../glossary.md#term-rank)
`1..16`, [canonical](../../../../glossary.md#term-canonical-form) dense-[contiguous](../../../../glossary.md#term-contiguous) zero-offset non-[view](../../../../glossary.md#term-view-meanings) [layouts](../../../../glossary.md#layout), exact inferred output Shape, and
checked [element](../../../../glossary.md#term-element)/byte geometry. Ordinary full form reduces [axes](../../../../glossary.md#term-axis) `0..rank-1`; single-axis and
multi-axis forms retain normalized axes and exact keep-[dimensions](../../../../glossary.md#dimension) meaning. Output rank is `0..16`.
An empty multi-axis list is a typed [identity](../../../../glossary.md#term-identity-meanings)/copy and must not invoke an accidental reduction.

Capability and analysis [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) for SUM, [MEAN](../../../../glossary.md#term-mean), SUM_TO_SHAPE, PROD, masked, advanced,
statistical, [Boolean](../../../../glossary.md#data-type--datatype), arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum), and [scan](../../../../glossary.md#term-cumulative-scan) reductions; zero/dynamic input extents; [scalar](../../../../glossary.md#term-scalar) input;
non-FLOAT32 types; wrong attrs, [arity](../../../../glossary.md#term-arity), result [metadata](../../../../glossary.md#term-metadata), or axis sets; arbitrary views; unchecked
geometry; malformed [topology](../../../../glossary.md#term-topology); and any backward/training claim.

## Mandatory exhaustive real-[device](../../../../glossary.md#term-device) probe and [fail-closed](../../../../glossary.md#term-fail-closed) gate

Before production edits, [compile](../../../../glossary.md#compile) and [run](../../../../glossary.md#term-run) a disposable Apple-silicon probe at [MPSGraph](../../../../glossary.md#term-mpsgraph) optimization
levels `0` and `1`, with reduced-[precision](../../../../glossary.md#term-precision) fast math disabled, at least three independently compiled
executables per level, and at least eight executions per case. Use production-style [FLOAT32](../../../../glossary.md#term-float32)
placeholders, the two exact [NaN](../../../../glossary.md#term-nan)-propagating selectors, caller-supplied [canary](../../../../glossary.md#term-canary)-filled direct [targets](../../../../glossary.md#term-target),
feed/target [permutation](../../../../glossary.md#permutation) checks, and raw [identity](../../../../glossary.md#term-identity-meanings) controls.

Exhaust the ordered two-value cross-product of a fixed raw-bit corpus containing both zeros,
positive/negative minimum and ordinary [subnormals](../../../../glossary.md#term-subnormal), minimum normals, adjacent and equal ordinary
[finite](../../../../glossary.md#term-finite) values, maximum finite values, both [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and quiet/signaling NaNs of both signs with
multiple payloads. Also exhaust every permutation of representative three- and four-value domains
covering NaN position, duplicate [extrema](../../../../glossary.md#saturation-pole-and-extremum), opposite zeros, subnormal versus zero, finite/[infinity](../../../../glossary.md#term-infinity),
and mixed signs. Check every case for both MIN and MAX. NaN results require classification only;
every uniquely fixed non-NaN result requires exact [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity).

The probe must additionally prove full, single-[axis](../../../../glossary.md#term-axis), and ordered multi-axis results; both
keep-[dimensions](../../../../glossary.md#dimension) choices; empty-axis identity; [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), and [rank](../../../../glossary.md#term-rank)-sixteen outputs; exact
[Shape](../../../../glossary.md#shape) and represented [layout](../../../../glossary.md#layout); complete direct writes; untouched outside canaries; stable mixed
chains with existing 0005 [nodes](../../../../glossary.md#node); permutations, repetition, and executable independence. If any
selector, NaN propagation, [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), subnormal ordering, infinity, exact-bit, Shape, target,
canary, permutation, repetition, or independence gate fails, remove the probe and mark 0012
Blocked. Do not narrow the corpus, substitute a non-propagating selector, custom [kernel](../../../../glossary.md#kernel), [host](../../../../glossary.md#term-host)/[CPU](../../../../glossary.md#term-cpu)
fallback, [tolerance](../../../../glossary.md#term-tolerance), or [Model](../../../../glossary.md#term-model) relaxation.

## Native [schema](../../../../glossary.md#term-schema), candidates, and [lifecycle](../../../../glossary.md#lifecycle)

Keep native [ABI](../../../../glossary.md#term-application-binary-interface-abi) version 4, statuses, and exactly thirteen exports. Bump the fixed 160-byte [node](../../../../glossary.md#node)
schema from version 2 to version 3; preserve [operation](../../../../glossary.md#operation) wires `1..10`; assign `11` and `12` to
reduction MIN and MAX and attribute wire `4` to normalized reduction [axes](../../../../glossary.md#term-axis). A reduction record has
one input, `second_input = UINT32_MAX`, `attribute_count = 0..16`, `axis = UINT32_MAX`,
`reserved = 0`, ordered normalized axes in `attribute_values`, and zero remaining cells. Java and
native [validation](../../../../glossary.md#term-validation) independently prove axis uniqueness/range and exact output geometry. Full forms
lower to all axes; empty-axis forms lower to [identity](../../../../glossary.md#term-identity-meanings); non-empty selectors reshape only when needed
to realize exact keep-[dimensions](../../../../glossary.md#dimension) [Shape](../../../../glossary.md#shape).

Bump candidate, compatibility/workload, route-policy, and [backend](../../../../glossary.md#backend)-local [codec](../../../../glossary.md#term-codec) versions. Identity
includes typed [Model](../../../../glossary.md#term-model) form, ordered axes, keep-dimensions, [descriptors](../../../../glossary.md#term-descriptor), schema/ABI, [topology](../../../../glossary.md#term-topology), splats,
and live [target](../../../../glossary.md#term-target); old decisions [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation). Existing custom [singleton](../../../../glossary.md#term-singleton) NEG remains exact; any
[partition](../../../../glossary.md#partition) containing [extrema](../../../../glossary.md#saturation-pole-and-extremum) uses [MPSGraph](../../../../glossary.md#term-mpsgraph).

Analysis declares all [buffers](../../../../glossary.md#term-buffer) and workspace before assignment. Finalization transactionally creates
one reusable Shape-specialized executable. Cold [binding](../../../../glossary.md#term-binding) validates representations; hot execution
performs one [synchronous](../../../../glossary.md#term-synchronous) native submission. Add only the backend-local four-byte [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [FLOAT32](../../../../glossary.md#term-float32)
[publication](../../../../glossary.md#term-publication) path; do not widen [rank](../../../../glossary.md#term-rank)-`1..16` [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) transfer. Preserve [rollback](../../../../glossary.md#term-rollback), autorelease-pool,
closure, repeated/concurrent [run](../../../../glossary.md#term-run), and independent-session rules.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects ownership; Metal owns truthful lowering and Runtime receives prepared work.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Metal glossary](../../../../glossary.md#term-metal); [lowering glossary](../../../../glossary.md#lowering); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Concrete backend modules and Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend))</span>
  — Metal owns MPSGraph lowering, native schema, resources, and materialization.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [schema glossary](../../../../glossary.md#term-schema); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span> —
  route and resources are fixed before direct hot execution.
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [automatic differentiation glossary](../../../../glossary.md#autograd))</span>
  — forward capture is consumed without claiming the extrema gradient graph, which needs blocked
  predicate/selection capability.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [gradient glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span> — ordinary extrema attributes, Shapes, and exact
  represented-value semantics remain Model-owned.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [Model glossary](../../../../glossary.md#term-model))</span>

If implementation requires a shared contract, public [API](../../../../glossary.md#term-application-programming-interface-api), new [dependency](../../../../glossary.md#term-dependency), [ABI](../../../../glossary.md#term-application-binary-interface-abi) export/version,
cross-[owner](../../../../glossary.md#term-owner) [scalar](../../../../glossary.md#term-scalar) transfer, or numerical relaxation, stop and mark 0012 Blocked.

## Dependencies and integration

- **Depends on:** Metal 0005 and 0008; current Model ordinary-extrema semantics and Compiler forward
  capture; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016. Not Metal 0006,
  0007, or 0009–0011, and not Model 0026.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Model glossary](../../../../glossary.md#term-model); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- **Conflicts with:** every concurrent Metal capability, preparation, native schema,
  candidate/codec, materialization, or Engine Metal integration change.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization))</span>
- **Parallel group:** None.
- **Common base revision:** N/A; serial frontier prepared from clean `ae3c74c4b97138bd6b0a196ff295654408f518ff`.
- **Integration order:** mandatory probe → schema/geometry → capability/topology →
  candidates/lifecycle/scalar publication → Engine proof → documentation and independent review.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [lifecycle glossary](../../../../glossary.md#lifecycle); [scalar glossary](../../../../glossary.md#term-scalar); [publication glossary](../../../../glossary.md#term-publication))</span>
- **Integration [validation](../../../../glossary.md#term-validation):** native build/export audit; focused Metal, conformance, real Engine,
  Compiler-contract, architecture, Javadoc, Markdown, and whitespace checks.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- **Shared-document integration [owner](../../../../glossary.md#term-owner):** Main planner.

## Files and symbols

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNativeApi`, preparer and candidate/codec
  types — exact extrema domain, schema v3, geometry, topology, and invalidation.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [schema glossary](../../../../glossary.md#term-schema); [topology glossary](../../../../glossary.md#term-topology))</span>
- [Metal](../../../../glossary.md#term-metal) finalization/execution types and `native/metal-macos-arm64/src/synaptik_metal_foundation.m`
  — propagating selectors, rank-zero publication, reusable resource, and unchanged hot lifecycle.<span data-glossary-navigation="true"> ([rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [publication glossary](../../../../glossary.md#term-publication); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Focused Metal/conformance/Compiler/Engine tests and affected Metal/native guides and Javadocs —
  fail-closed capability, schema, lifecycle, and real behavior proofs.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Compiler glossary](../../../../glossary.md#term-compiler); [Engine glossary](../../../../glossary.md#term-engine); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>

## Acceptance criteria

- Capability/preparation admit exactly the bounded MIN/MAX domain and preserve every existing route,
  affine restriction, FLOAT32 storage/publication, and FLOAT32-only cross-owner transfer.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [publication glossary](../../../../glossary.md#term-publication); [owner glossary](../../../../glossary.md#term-owner))</span>
- The disposable probe passes the complete raw corpus, pair cross-product, required permutations,
  exact-bit/special-value, Shape, target, canary, repetition, and independence gates first.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [target glossary](../../../../glossary.md#term-target); [canary glossary](../../../../glossary.md#term-canary))</span>
- [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4 retains thirteen exports; [schema](../../../../glossary.md#term-schema) v3 retains 160-byte records, uses only wires `11`, `12`,
  and attribute `4`, and rejects malformed/stale records and decisions.
- Fake and real tests cover both kinds/forms, axis/Shape boundaries, identity, wrong attrs/types,
  topology, rollback, close, reuse, concurrency, rank-zero publication, and independent sessions.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [identity glossary](../../../../glossary.md#term-identity-meanings); [topology glossary](../../../../glossary.md#term-topology); [rollback glossary](../../../../glossary.md#term-rollback); [concurrency glossary](../../../../glossary.md#term-concurrency); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- A Metal-only public Engine run composes caller-input extrema with existing 0005 nodes, publishes
  direct rank-zero and positive-rank exact outputs, repeats across sessions, and proves no CPU owner.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Engine glossary](../../../../glossary.md#term-engine); [run glossary](../../../../glossary.md#term-run); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [nodes glossary](../../../../glossary.md#node); [rank glossary](../../../../glossary.md#term-rank); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- No excluded reduction, custom kernel, backward/training, transfer widening, relaxed math,
  FLOAT16/BFLOAT16, or performance claim is introduced. Independent Class C review approves.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [widening glossary](../../../../glossary.md#term-widening); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>

## [Validation](../../../../glossary.md#term-validation)

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test --tests '*Metal*Reduction*' --tests '*Metal*MpsGraph*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :modules:compiler:test --tests '*Reduction*' :backends:metal:javadoc :testing:architecture-tests:test
git diff --check
```

Also validate balanced Markdown fences, local links/anchors, exactly one [Ready](../../../../glossary.md#term-ready) [Metal](../../../../glossary.md#term-metal) frontier,
thirteen exports, unchanged contracts/dependencies, no custom [extrema](../../../../glossary.md#saturation-pole-and-extremum) [kernel](../../../../glossary.md#kernel) or [scalar](../../../../glossary.md#term-scalar) cross-[owner](../../../../glossary.md#term-owner)
transfer, and no excluded-reduction or backward claim.

## Documentation and review impact

Update [Metal](../../../../glossary.md#term-metal) Javadocs, package documentation, [backend](../../../../glossary.md#backend)/native guides, targeted public [scope](../../../../glossary.md#term-scope) text, and
glossary status only after evidence passes. [Model](../../../../glossary.md#term-model)/[Tensor](../../../../glossary.md#tensor) semantics and architecture contracts remain
unchanged. Independent Class C review inspects capability truth, numerical gates, [schema](../../../../glossary.md#term-schema) and
compatibility invalidation, [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [lifecycle](../../../../glossary.md#lifecycle), [Engine](../../../../glossary.md#term-engine) evidence, and all exclusions.

## Result

- The mandatory Apple M3 Max probe ran the exact NaN-propagating MIN/MAX selectors at optimization
  levels 0 and 1 with reduced-precision fast math disabled, three independently compiled
  executables per length and level, eight runs per case, and 43,824 total executions.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [precision glossary](../../../../glossary.md#term-precision))</span>
- For `[+0,-0]`, both selectors returned `-0`: MIN passed and MAX failed in all 48 observations.
  For `[-0,+0]`, both returned `+0`: MAX passed and MIN failed in all 48 observations. Three of
  four representative zero permutations were order-dependent.
- For `[+0,+minimum-subnormal]`, both selectors returned the subnormal, so MIN failed; reversing
  the pair made both return <span data-glossary-navigation="true"> ([subnormal glossary](../../../../glossary.md#term-subnormal))</span>`+0`, so MAX failed. The matrix recorded 2,016 MIN mismatches, 2,016
  MAX mismatches, and 2,976 subnormal failures.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- NaN classification, minimum normals, identity controls, direct targets, canaries, permutations,
  repetition, and bindings passed, but they do not remove the signed-zero/subnormal blocker.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [targets glossary](../../../../glossary.md#term-target); [bindings glossary](../../../../glossary.md#term-binding); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Independent review returned `APPROVE-BLOCKER`. The disposable probe and all production, native,
  and test changes were removed. Unblocking requires an exact replacement route or explicit Model
  numerical-contract change; custom kernels and relaxation remain unauthorized here.<span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel))</span>
- Status: Incomplete
- Follow-up required: leave 0012 Blocked until an exact replacement or approved [Model](../../../../glossary.md#term-model) change exists.
