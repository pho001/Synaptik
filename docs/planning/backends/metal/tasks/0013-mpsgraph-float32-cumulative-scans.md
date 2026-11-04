# Task 0013: [MPSGraph](../../../../glossary.md#term-mpsgraph) [FLOAT32](../../../../glossary.md#term-float32) Cumulative Scans

## Reader context

These cumulative scans visit values along a tensor axis; they are not the later general Scan
operation that repeats a body with carried state. Inclusive/exclusive controls whether the
current value contributes before publication, and reverse controls traversal direction. The
sequential oracle below was this route's conservative gate, not a new Model-wide arithmetic rule.

Compiler supplied capture and Metal would have lowered those descriptions into prepared work.
The corrected result matters: the valid axis worked, but arithmetic flushed represented
subnormal values. Identity and target guards isolated that failure without proving scan
correctness.
The [current numerical and cumulative-operation domains](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains)
explain the later implementation separately from this removed probe and its blocked route.

## Status

Blocked

The mandatory exact real-[device](../../../../glossary.md#term-device) probe failed the [FLOAT32](../../../../glossary.md#term-float32) [subnormal](../../../../glossary.md#term-subnormal) gate. Independent review
returned `APPROVE-BLOCKER` and corrected the initial [axis](../../../../glossary.md#term-axis) interpretation. No production, native,
test, or probe change remains; [Metal](../../../../glossary.md#term-metal) 0014 is the independent successor.

## Change class

Class C — this extends [Metal](../../../../glossary.md#term-metal) capability, the private typed native [schema](../../../../glossary.md#term-schema), candidate [identity](../../../../glossary.md#term-identity-meanings),
whole-[partition](../../../../glossary.md#partition) [lowering](../../../../glossary.md#lowering), persistent [MPSGraph](../../../../glossary.md#term-mpsgraph) resources, and public [Engine](../../../../glossary.md#term-engine) execution while
preserving the existing architecture, [ABI](../../../../glossary.md#term-application-binary-interface-abi) exports, [FLOAT32](../../../../glossary.md#term-float32) storage, and transfer contract.

## Goal

Add bounded forward-only, non-[gradient](../../../../glossary.md#term-gradient) [`FLOAT32`](../../../../glossary.md#term-float32) `CumulativeScanKind.CUM_SUM` and `CUM_PROD`
through [MPSGraph](../../../../glossary.md#term-mpsgraph)'s dedicated cumulative selectors. Support every inclusive/exclusive and
forward/reverse combination along one [normalized axis](../../../../glossary.md#normalized-axis) without reopening blocked aggregate
SUM/[MEAN](../../../../glossary.md#term-mean), [scalar](../../../../glossary.md#term-scalar) arithmetic, [BOOL](../../../../glossary.md#data-type--datatype) storage, [extrema](../../../../glossary.md#saturation-pole-and-extremum), or custom [kernels](../../../../glossary.md#kernel).

## Inventory and [scope](../../../../glossary.md#term-scope) decision

The current [Model](../../../../glossary.md#term-model) candidates are first-class and distinct:

- [masked reduction](../../../../glossary.md#masked-reduction) consists only of two-input [axis](../../../../glossary.md#term-axis)-removing floating `SUM`/[`MEAN`](../../../../glossary.md#term-mean) with a
  broadcast BOOL mask; it depends on exact aggregate arithmetic blocked by 0007 and a BOOL
  storage/input path not delivered by blocked 0010;<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- arg [extrema](../../../../glossary.md#saturation-pole-and-extremum) consists of single-axis `ARG_MIN`/`ARG_MAX`, fixed INT64 results, FIRST/LAST logical
  index ties, NaN preference, negative-zero-below-positive-zero ordering, and non-empty selected
  axes. Installed <span data-glossary-navigation="true"> ([logical index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`reductionArgMinimum...`/`reductionArgMaximum...` selectors have no tie-policy
  parameter and expose INT32 symbolic results on this SDK, so exact tie selection, widening, INT64
  storage, and publication require a separate probe and task; and<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [widening glossary](../../../../glossary.md#term-widening); [publication glossary](../../../../glossary.md#term-publication))</span>
- [cumulative scan](../../../../glossary.md#cumulative-scan) consists exactly of `CUM_SUM` and `CUM_PROD` with
  `CumulativeScanAttrs(axis, exclusive, reverse)`. The installed [SDK](../../../../glossary.md#term-sdk) exposes matching
  `cumulativeSumWithTensor:axis:exclusive:reverse:name:` and
  `cumulativeProductWithTensor:axis:exclusive:reverse:name:` selectors.

0013 therefore selects the complete bounded [FLOAT32](../../../../glossary.md#term-float32) cumulative-[scan](../../../../glossary.md#term-cumulative-scan) family. Scans produce values,
not ties or indices and add no tie policy, index width, or [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) storage. Their represented
storage remains FLOAT32. The probe locks [NaN](../../../../glossary.md#term-nan) classification, signed zeros, [subnormals](../../../../glossary.md#term-subnormal), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero),
exact positive-zero/positive-one exclusive [identities](../../../../glossary.md#term-identity-meanings), and logical placement before capability.

## Exact capability domain

Admit one or more [scan](../../../../glossary.md#term-cumulative-scan) occurrences mixed with existing 0005/0008 [nodes](../../../../glossary.md#node) when every scan has one
input and one output, exact `CumulativeScanAttrs`, `requiresGrad == false`, identical [FLOAT32](../../../../glossary.md#term-float32) type
and [Shape](../../../../glossary.md#shape), fully static positive [rank](../../../../glossary.md#term-rank) `1..16`, [canonical](../../../../glossary.md#term-canonical-form) dense-[contiguous](../../../../glossary.md#term-contiguous) zero-offset non-[view](../../../../glossary.md#term-view-meanings)
[layouts](../../../../glossary.md#layout), a valid [normalized axis](../../../../glossary.md#normalized-axis), and checked [element](../../../../glossary.md#term-element)/byte geometry. All four exclusive/reverse
combinations are in [scope](../../../../glossary.md#term-scope). Scan results may feed later supported FLOAT32 nodes and supported nodes
may feed scans; every maximal supported [topology](../../../../glossary.md#term-topology) must lower as one typed [MPSGraph](../../../../glossary.md#term-mpsgraph) program.

Capability and analysis [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) for masked, ordinary, arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum), advanced, statistical,
[Boolean](../../../../glossary.md#data-type--datatype), recurrent, or any other reduction/scan family; [scalar](../../../../glossary.md#term-scalar) or zero/dynamic extents; [gradient](../../../../glossary.md#term-gradient)
eligibility; non-FLOAT32 types; wrong attrs, [arity](../../../../glossary.md#term-arity), Shape, layout, or [axis](../../../../glossary.md#term-axis); unchecked geometry;
malformed topology; and any backward/training claim.

## Mandatory comprehensive real-[device](../../../../glossary.md#term-device) probe and [fail-closed](../../../../glossary.md#term-fail-closed) gate

Before production edits, [compile](../../../../glossary.md#compile) and [run](../../../../glossary.md#term-run) a disposable Apple-silicon probe at [MPSGraph](../../../../glossary.md#term-mpsgraph) optimization
levels `0` and `1`, with reduced-[precision](../../../../glossary.md#term-precision) fast math set to none, at least three independently
compiled executables per [axis](../../../../glossary.md#term-axis) length and level, and at least eight executions per case. Use the two
exact cumulative selectors, production-style [FLOAT32](../../../../glossary.md#term-float32) placeholders, caller-supplied [canary](../../../../glossary.md#term-canary)-filled
direct [targets](../../../../glossary.md#term-target), feed/target [permutation](../../../../glossary.md#permutation) checks, raw input [identity](../../../../glossary.md#term-identity-meanings) controls, and no [host](../../../../glossary.md#term-host) fallback.

For both kinds and all four mode combinations, exhaust the ordered two-value cross-product of a
fixed raw-bit corpus containing both zeros; positive/negative minimum and ordinary [subnormals](../../../../glossary.md#term-subnormal);
minimum normals; adjacent, equal, and opposite-sign ordinary values; positive/negative one and
two; maximum [finite](../../../../glossary.md#term-finite) values; both [infinities](../../../../glossary.md#nan-infinity-and-signed-zero); and quiet/signaling NaNs of both signs with multiple
payloads. Also exhaust every permutation of representative three- and four-value domains covering
[NaN](../../../../glossary.md#term-nan) position, duplicate values, opposite zeros, subnormal/zero, [underflow](../../../../glossary.md#term-underflow)/[overflow](../../../../glossary.md#term-overflow), zero/[infinity](../../../../glossary.md#term-infinity),
opposing infinities, and sign parity.

The conservative [oracle](../../../../glossary.md#term-test-oracle) is one logical-[slice](../../../../glossary.md#slice) sequential FLOAT32 fold in the declared traversal
direction, initialized with represented `+0.0f` for CUM_SUM or `+1.0f` for CUM_PROD, with one
ordinary same-type [IEEE](../../../../glossary.md#term-ieee) addition or multiplication per visited value and one [rounding](../../../../glossary.md#term-rounding) after each
[operation](../../../../glossary.md#operation). Exclusive mode stores before applying the current value; inclusive mode applies then
stores. Require NaN classification only, but require exact [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) for every non-NaN output,
including zeros, subnormals, infinities, and exclusive identities. This deliberately strict route
gate does not relax or rewrite [Model](../../../../glossary.md#term-model) semantics.

Additionally prove axes at the first, middle, and last positions; [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), and
[rank](../../../../glossary.md#term-rank)-sixteen [Shapes](../../../../glossary.md#shape); axis lengths `1`, `2`, `3`, `4`, and representative longer lengths; exact
Shape/[layout](../../../../glossary.md#layout); complete writes; untouched surrounding canaries; chains of both [scan](../../../../glossary.md#term-cumulative-scan) kinds with
existing 0005 [nodes](../../../../glossary.md#node); permutations, repetition, concurrent runs, and executable/session
independence. If any selector, mode, NaN, zero, subnormal, infinity, exact-bit, axis, Shape, target,
canary, [binding](../../../../glossary.md#term-binding), permutation, repetition, [concurrency](../../../../glossary.md#term-concurrency), or independence gate fails, remove the
probe and mark 0013 Blocked. Do not narrow the corpus, modes, or domain; substitute aggregate or
[pointwise](../../../../glossary.md#term-pointwise) decompositions; add a custom [kernel](../../../../glossary.md#kernel)/[CPU](../../../../glossary.md#term-cpu) fallback; introduce [tolerance](../../../../glossary.md#term-tolerance); or relax Model.

## Native [schema](../../../../glossary.md#term-schema), candidates, and [lifecycle](../../../../glossary.md#lifecycle)

Keep native [ABI](../../../../glossary.md#term-application-binary-interface-abi) version 4, statuses, and exactly thirteen exports. Bump the fixed 160-byte [node](../../../../glossary.md#node)
schema from version 2 to version 3; preserve [operation](../../../../glossary.md#operation) wires `1..10`; assign `11` to CUM_SUM and
`12` to CUM_PROD; and assign attribute wire `4` to [cumulative scan](../../../../glossary.md#cumulative-scan). A [scan](../../../../glossary.md#term-cumulative-scan) record has one input,
`second_input = UINT32_MAX`, `attribute_count = 2`, the [normalized axis](../../../../glossary.md#normalized-axis) in [`axis`](../../../../glossary.md#term-axis),
`attribute_values[0] = exclusive ? 1 : 0`, `attribute_values[1] = reverse ? 1 : 0`, `reserved = 0`,
and zero remaining cells. Java and native [validation](../../../../glossary.md#term-validation) independently prove exact same-[Shape](../../../../glossary.md#shape) [FLOAT32](../../../../glossary.md#term-float32)
geometry, axis range, [Boolean](../../../../glossary.md#data-type--datatype) flag encoding, [topology](../../../../glossary.md#term-topology), and output [ownership](../../../../glossary.md#term-ownership-meanings).

Bump workload-[signature](../../../../glossary.md#term-signature), exact-policy, candidate, compatibility, route-policy, and [backend](../../../../glossary.md#backend)-local
[codec](../../../../glossary.md#term-codec) versions from 3 to 4. Candidate/workload [identity](../../../../glossary.md#term-identity-meanings) includes scan kind, normalized axis, both
mode flags, [descriptors](../../../../glossary.md#term-descriptor), schema/ABI, topology, splats, and live [target](../../../../glossary.md#term-target); every old or malformed
decision fails closed. Existing custom [singleton](../../../../glossary.md#term-singleton) NEG remains exact; any [partition](../../../../glossary.md#partition) containing a
scan uses [MPSGraph](../../../../glossary.md#term-mpsgraph).

Analysis declares every [buffer](../../../../glossary.md#term-buffer) and address workspace before assignment. Finalization
transactionally creates one reusable Shape-specialized executable. Cold [binding](../../../../glossary.md#term-binding) validates exact
FLOAT32 representations and fixed address order; hot execution performs one [synchronous](../../../../glossary.md#term-synchronous) native
submission without route, [graph](../../../../glossary.md#graph), operation, slot, map, or type [lookup](../../../../glossary.md#term-lookup). Preserve direct FLOAT32
resident storage, current [canonical](../../../../glossary.md#term-canonical-form)-only [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) transfer, [rollback](../../../../glossary.md#term-rollback), autorelease pools,
close/[run](../../../../glossary.md#term-run) [leases](../../../../glossary.md#term-lease), repeated/concurrent run isolation, and independent sessions. Add no [BOOL](../../../../glossary.md#data-type--datatype),
[INT32](../../../../glossary.md#data-type--datatype), [INT64](../../../../glossary.md#data-type--datatype), [scalar](../../../../glossary.md#term-scalar), [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), or new cross-[owner](../../../../glossary.md#term-owner) storage/[materialization](../../../../glossary.md#materialization) path.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects ownership; Metal owns truthful lowering and Runtime receives prepared work.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Metal glossary](../../../../glossary.md#term-metal); [lowering glossary](../../../../glossary.md#lowering); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Concrete backend modules and Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend))</span>
  — Metal owns MPSGraph lowering, schema, resources, and materialization.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [schema glossary](../../../../glossary.md#term-schema); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span> —
  route and resources are fixed before direct hot execution.
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [automatic differentiation glossary](../../../../glossary.md#autograd))</span>
  — consume current forward capture only; no scan adjoint or training coverage is claimed.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [scan glossary](../../../../glossary.md#term-cumulative-scan); [adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- [Tensor API](../../../../api/tensor-api.md#cumulative-scan-expressions)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span> — preserve scan kind,
  normalized axis, mode, Shape/type, identity, and represented floating semantics.<span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis); [Shape glossary](../../../../glossary.md#shape); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

If implementation requires a shared contract, public [API](../../../../glossary.md#term-application-programming-interface-api), new [dependency](../../../../glossary.md#term-dependency), [ABI](../../../../glossary.md#term-application-binary-interface-abi) export/version,
new [data type](../../../../glossary.md#data-type--datatype)/storage path, aggregate/[pointwise](../../../../glossary.md#term-pointwise) [decomposition](../../../../glossary.md#term-decomposition), custom [kernel](../../../../glossary.md#kernel), or numerical
relaxation, stop and mark 0013 Blocked.

## Dependencies and integration

- **Depends on:** Metal 0005 and 0008; current Model cumulative-scan semantics and Compiler forward
  capture; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016. Not Metal 0006,
  0007, or 0009–0012, and not Model 0026.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Model glossary](../../../../glossary.md#term-model); [scan glossary](../../../../glossary.md#term-cumulative-scan); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- **Conflicts with:** every concurrent Metal capability, preparation, native schema,
  candidate/codec, materialization, or Engine Metal integration change.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization))</span>
- **Parallel group:** None.
- **Common base revision:** N/A; serial frontier verified on clean `main` at `7b34a7a9352d05d8930d4463b286209dff9a6228`.
- **Integration order:** mandatory probe → schema/geometry → capability/topology →
  candidates/lifecycle → Engine proof → documentation and independent review.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- **Integration [validation](../../../../glossary.md#term-validation):** native build/export audit; focused Metal, conformance, real Engine,
  Compiler-contract, architecture, Javadoc, Markdown, and whitespace checks.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- **Shared-document integration [owner](../../../../glossary.md#term-owner):** Main planner.

## Files and symbols

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNativeApi`, preparer, and
  candidate/codec types — exact scan domain, schema v3, topology, and invalidation.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [scan glossary](../../../../glossary.md#term-cumulative-scan); [schema glossary](../../../../glossary.md#term-schema); [topology glossary](../../../../glossary.md#term-topology))</span>
- [Metal](../../../../glossary.md#term-metal) finalization/execution types and `native/metal-macos-arm64/src/synaptik_metal_foundation.m`
  — direct cumulative selectors, reusable resources, and unchanged FLOAT32 hot lifecycle.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Focused Metal/conformance/Compiler/Engine tests and affected Metal/native guides and Javadocs —
  fail-closed capability, schema, lifecycle, and real behavior proofs.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Compiler glossary](../../../../glossary.md#term-compiler); [Engine glossary](../../../../glossary.md#term-engine); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>

## Acceptance criteria

- The disposable probe first passes the complete corpus, both kinds, all modes, exact oracle,
  Shape/axis, target, canary, binding, repetition, concurrency, and independence gates.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [Shape glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis); [target glossary](../../../../glossary.md#term-target); [canary glossary](../../../../glossary.md#term-canary); [binding glossary](../../../../glossary.md#term-binding); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Capability/preparation admit exactly the bounded no-gradient FLOAT32 scan domain and preserve
  every existing route, affine restriction, storage/materialization, and transfer rule.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [FLOAT32 glossary](../../../../glossary.md#term-float32); [scan glossary](../../../../glossary.md#term-cumulative-scan); [materialization glossary](../../../../glossary.md#materialization))</span>
- [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4 retains thirteen exports; [schema](../../../../glossary.md#term-schema) v3 retains 160-byte records, uses only wires `11`, `12`,
  and attribute `4`, and rejects malformed/stale records and decisions.
- Fake and real tests cover both kinds, all modes, axis/Shape boundaries, wrong attrs/types,
  topology, rollback, close, reuse, concurrency, and independent sessions.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- A Metal-only public Engine run composes both scans with existing nodes, publishes exact outputs,
  repeats/concurrently runs across sessions, and proves no CPU owner.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Engine glossary](../../../../glossary.md#term-engine); [run glossary](../../../../glossary.md#term-run); [nodes glossary](../../../../glossary.md#node); [CPU glossary](../../../../glossary.md#term-cpu); [owner glossary](../../../../glossary.md#term-owner))</span>
- No excluded reduction/scan, BOOL/index storage, rank-zero path, custom kernel,
  backward/training, transfer widening, relaxed math, FLOAT16/BFLOAT16, or performance claim is
  introduced. Independent Class C review approves.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [kernel glossary](../../../../glossary.md#kernel); [widening glossary](../../../../glossary.md#term-widening); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>

## [Validation](../../../../glossary.md#term-validation)

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test --tests '*Metal*Scan*' --tests '*Metal*MpsGraph*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :modules:compiler:test --tests '*Scan*' :backends:metal:javadoc :testing:architecture-tests:test
git diff --check
```

Also validate balanced Markdown fences, local links/anchors, exactly one [Ready](../../../../glossary.md#term-ready) [Metal](../../../../glossary.md#term-metal) frontier,
thirteen exports, unchanged contracts/dependencies, no custom [scan](../../../../glossary.md#term-cumulative-scan) [kernel](../../../../glossary.md#kernel) or new storage path, and
no excluded-family or backward claim.

## Documentation and review impact

Update [Metal](../../../../glossary.md#term-metal) Javadocs, package documentation, [backend](../../../../glossary.md#backend)/native guides, targeted public [scope](../../../../glossary.md#term-scope) text,
and glossary status only after evidence passes. [Model](../../../../glossary.md#term-model)/[Tensor](../../../../glossary.md#tensor) semantics and architecture contracts
remain unchanged. Independent Class C review inspects capability truth, probe gates, [schema](../../../../glossary.md#term-schema) and
compatibility invalidation, [lifecycle](../../../../glossary.md#lifecycle), [Engine](../../../../glossary.md#term-engine) evidence, and all exclusions.

## Result

- The Apple M3 Max probe used optimization levels `0` and `1`, reduced-precision fast math none,
  and twelve independently compiled executables: three for each level/axis-length pair at lengths<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision))</span>
  `1` and `2`. Every one of the eight kind/mode combinations ran eight direct executions.
- Inclusive length-one `CUM_SUM` and `CUM_PROD` flushed all six subnormal corpus values to zero;
  for example, raw <span data-glossary-navigation="true"> ([subnormal glossary](../../../../glossary.md#term-subnormal))</span>`0x00000001` became `0x00000000`. Length two produced additional mismatches
  across multiple inclusive/exclusive and forward/reverse modes.
- Strict [host](../../../../glossary.md#term-host) controls preserved `+0.0f + minimum-subnormal` and
  `+1.0f * minimum-subnormal`. Exact exclusive `+0.0f`/`+1.0f` identities, raw identity,
  direct-target canaries, and bindings passed, so the failures are in cumulative arithmetic.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [target glossary](../../../../glossary.md#term-target); [bindings glossary](../../../../glossary.md#term-binding))</span>
- [Axis](../../../../glossary.md#term-axis) is not a blocker: corrected review evidence shows valid [rank](../../../../glossary.md#term-rank)-six axis `5` compiles and runs
  for both cumulative SUM and PRODUCT, while out-of-range axis `6` aborts during compilation as
  required. The earlier contrary axis claim is superseded.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Independent review returned `APPROVE-BLOCKER`. Unblocking requires an exact replacement route or
  an explicit Model numerical-contract change; custom kernels and relaxation remain unauthorized.<span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel))</span>
- Status: Incomplete
- Follow-up required: leave 0013 Blocked until an exact replacement or approved [Model](../../../../glossary.md#term-model) change exists.
