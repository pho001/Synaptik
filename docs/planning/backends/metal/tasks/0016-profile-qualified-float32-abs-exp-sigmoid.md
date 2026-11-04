# Task 0016: Profile-Qualified [FLOAT32](../../../../glossary.md#term-float32) ABS, [EXP](../../../../glossary.md#term-expx), and [SIGMOID](../../../../glossary.md#term-sigmoid)

## Reader context

The historical authorization covered all three operations together. ABS passing did not make
this a partially delivered task: EXP and SIGMOID failed their numerical gates before production
edits, and the disposable experiment was removed. Task 0019 later treated ABS as its own slice.

Profiles, shapes and typed records described admissible programs; the oracle checked actual
FLOAT32 results. A unit in the last place (ULP) measures spacing between adjacent represented
values, but the additional one-ULP gate did not replace Model's obligations for result classes
and nonzero subnormals. Preserved inputs and target guards checked buffer behavior, not numerical
accuracy by themselves.
The [current unary implementation and dedicated oracles](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains)
explain the later routes without retroactively qualifying this blocked experiment.

## Status

Blocked

Readiness was verified before execution: [CPU](../../../../glossary.md#term-cpu) 0017 and [Metal](../../../../glossary.md#term-metal) 0015 were Complete, their independent
Class C reviews returned `APPROVE` with zero findings, and the serial numerical-profile [backend](../../../../glossary.md#backend)
checkpoint passed. The mandatory real-[device](../../../../glossary.md#term-device) gate then blocked this task before production edits.
Historical Metal 0006 remains Blocked; this task neither rewrites that result nor assumes its broad
nine-[operation](../../../../glossary.md#operation) [scope](../../../../glossary.md#term-scope).

## Change class

Class C — this changes profile-qualified [Metal](../../../../glossary.md#term-metal) capability, the typed native [schema](../../../../glossary.md#term-schema), whole-[partition](../../../../glossary.md#partition)
[lowering](../../../../glossary.md#lowering), prepared compatibility, and public [Engine](../../../../glossary.md#term-engine) execution across the native resource boundary.
Fresh [operation](../../../../glossary.md#operation)-specific real-[device](../../../../glossary.md#term-device) evidence and independent review are mandatory.

## Goal

Add exactly `UnaryElementwiseKind.ABS`, [`EXP`](../../../../glossary.md#term-expx), and [`SIGMOID`](../../../../glossary.md#term-sigmoid) for [canonical](../../../../glossary.md#term-canonical-form) [`FLOAT32`](../../../../glossary.md#term-float32) [Metal](../../../../glossary.md#term-metal)
execution under both `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator), without adding an accelerator-only [unary](../../../../glossary.md#term-unary)
relaxation. Preserve every current strict NEG/affine/[`CONTIGUOUS`](../../../../glossary.md#term-contiguous) row and accelerator [tensor](../../../../glossary.md#tensor)-[binary](../../../../glossary.md#term-binary-descriptor-and-metadata)
row. Keep every other unary [operation](../../../../glossary.md#operation) [fail-closed](../../../../glossary.md#term-fail-closed).

## [Scope](../../../../glossary.md#term-scope)

- Treat the three operations as separate gates. `ABS` must be exact for every non-NaN raw input;
  NaN may vary only within the current NaN representation freedom. <span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan))</span>[`EXP`](../../../../glossary.md#term-expx) and [`SIGMOID`](../../../../glossary.md#term-sigmoid) must each
  satisfy their own bounded high-precision oracle, exact special-value classes, signed-zero rules,
  overflow/underflow boundaries, and no-DAZ/no-FTZ controls.<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [oracle glossary](../../../../glossary.md#term-test-oracle); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz))</span>
- Require one [`FLOAT32`](../../../../glossary.md#term-float32) input and output, `NoOperationAttrs.INSTANCE`, fully static positive [rank](../../../../glossary.md#term-rank)
  `1..16`, equal [Shape](../../../../glossary.md#shape) and `requiresGrad`, checked geometry, and canonical dense-contiguous,
  zero-offset non-view descriptors. Unary nodes produce canonical state and may consume only
  canonical state.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [view glossary](../../../../glossary.md#term-view-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor); [Unary glossary](../../../../glossary.md#term-unary))</span>
- Strict partitions may compose the retained NEG/affine/[`CONTIGUOUS`](../../../../glossary.md#term-contiguous) domain with the three new
  unary kinds. Accelerator partitions may compose the retained tensor <span data-glossary-navigation="true"> ([Accelerator glossary](../../../../glossary.md#term-accelerator); [tensor glossary](../../../../glossary.md#tensor))</span>`ADD`/`SUB`/`MUL`/`DIV`
  domain with the three new unary kinds. The task does not make NEG or affine/layout operations
  accelerator-capable and does not make binary arithmetic strict-capable.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- Keep [ABI](../../../../glossary.md#term-application-binary-interface-abi) version 4, statuses, create [signature](../../../../glossary.md#term-signature), and exactly thirteen exports. Advance [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema)
  `5 -> 6`; retain wires `1..11`, append `ABS=12`, `EXP=13`, and `SIGMOID=14`, and retain the
  fixed 160-byte typed record with one input, no attributes, canonical input/output state, and all
  unused fields at their required sentinels or zero.
- Advance workload-signature, exact-policy, candidate, compatibility, route-policy, and codec
  identities <span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>`6 -> 7`. Include the profile and ordered unary topology; version-six, cross-profile,
  malformed, corrupt, foreign-session, and unknown-wire data must reject or miss safely.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>
- Preserve MPSGraph as the sole route for the three additions. The custom singleton route remains
  NEG-only. Preserve transactional finalization, context leases, direct targets, run isolation,
  synchronous submission, reuse, concurrency, close rejection, and canonical-only transfer.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [singleton glossary](../../../../glossary.md#term-singleton); [leases glossary](../../../../glossary.md#term-lease); [targets glossary](../../../../glossary.md#term-target); [run glossary](../../../../glossary.md#term-run); [synchronous glossary](../../../../glossary.md#term-synchronous); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>

### Mandatory [operation](../../../../glossary.md#operation)-by-operation [device](../../../../glossary.md#term-device) gate

Before production edits, [run](../../../../glossary.md#term-run) a disposable [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary)/[MPSGraph](../../../../glossary.md#term-mpsgraph) probe at optimization levels `0`
and `1`, with reduced-[precision](../../../../glossary.md#term-precision) fast math disabled, three independently compiled executables per
operation/level/[Shape](../../../../glossary.md#shape) case, at least eight runs per executable, and two fresh contexts. Exercise
each operation alone and in repeated-[node](../../../../glossary.md#node)/[fan-out](../../../../glossary.md#term-fan-out)/direct-[target](../../../../glossary.md#term-target) forms; permute feeds and targets,
preserve inputs, [guard](../../../../glossary.md#term-guard) targets with canaries, and verify reported Shapes.

Use [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) covering both zeros; positive/negative minimum, ordinary, and maximum [subnormals](../../../../glossary.md#term-subnormal);
minimum normals and adjacent values; ordinary small and large [finite](../../../../glossary.md#term-finite) values; [overflow](../../../../glossary.md#term-overflow)/[underflow](../../../../glossary.md#term-underflow)
threshold neighborhoods; maximum finite values; both [infinities](../../../../glossary.md#nan-infinity-and-signed-zero); and quiet/signaling NaNs of both
signs with multiple payloads. Report results separately for each operation, optimization level,
Shape, context, and semantic bucket.

- `ABS`: require exact magnitude bits for every non-[NaN](../../../../glossary.md#term-nan) input, `-0 -> +0`, infinities to positive
  infinity, and NaN classification preservation.<span data-glossary-navigation="true"> ([positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- [`EXP`](../../../../glossary.md#term-expx): require `-infinity -> +0`, `+infinity -> +infinity`, NaN preservation, no DAZ input
  rewrite, no FTZ of a representable subnormal result, correct overflow/underflow class, and at most
  one unit in the last place for every ordinary finite result against a higher-precision oracle.<span data-glossary-navigation="true"> ([unit in the last place glossary](../../../../glossary.md#ulp-and-numerical-tolerance))</span><span data-glossary-navigation="true"> ([DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- [`SIGMOID`](../../../../glossary.md#term-sigmoid): require `-infinity -> +0`, `+infinity -> +1`, NaN preservation, stable finite-domain
  evaluation, no DAZ/FTZ shortcut, and at most one unit in the last place for every ordinary finite
  result against a higher-precision stable oracle.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>

The one-[ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) measurements are conservative device gates, not new allowed-result sets or a generic
[tolerance](../../../../glossary.md#term-tolerance); every output must still satisfy the operation's existing [Model](../../../../glossary.md#term-model) contract.

All three gates must pass before production changes. Any failure blocks 0016 with no capability,
[schema](../../../../glossary.md#term-schema), native, or test change. Passing one operation does not authorize another operation or a
family-wide approximation policy.

## Non-goals

No `RECIPROCAL`, [`LOG`](../../../../glossary.md#term-log), `SQRT`, `RSQRT`, `RELU`, [`TANH`](../../../../glossary.md#term-tanh), other [unary](../../../../glossary.md#term-unary) kind, [scalar](../../../../glossary.md#term-scalar) arithmetic,
reduction, [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), comparison, [extrema](../../../../glossary.md#saturation-pole-and-extremum), [scan](../../../../glossary.md#term-cumulative-scan), new [data type](../../../../glossary.md#data-type--datatype), [rank](../../../../glossary.md#term-rank) zero, dynamic [Shape](../../../../glossary.md#shape), [view](../../../../glossary.md#term-view-meanings) input,
custom unary [kernel](../../../../glossary.md#kernel), transfer [widening](../../../../glossary.md#term-widening), backward/training claim, packaging, discovery, or
performance claim. In particular, never authorize observed `RELU(NaN) -> +0` or
`TANH(NaN) -> +1`; both are [Model](../../../../glossary.md#term-model)-forbidden gross behavior. A future relaxed unary [operation](../../../../glossary.md#operation) needs
a coordinated Model contract with a named [finite](../../../../glossary.md#term-finite)-domain and special-value envelope plus its own
bounded [oracle](../../../../glossary.md#term-test-oracle). Model 0026 remains Draft.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Model
  owns result sets and profile selection stays cold.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  — unlisted unary operations gain no accelerator relaxation; NaN-to-ordinary gross behavior is
  forbidden.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [accelerator glossary](../../../../glossary.md#term-accelerator); [NaN glossary](../../../../glossary.md#term-nan))</span>
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span> — Metal
  owns truthful capability, lowering, native integration, and resources.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span> —
  routes and resources are fixed before direct hot execution.

If any [operation](../../../../glossary.md#operation) needs relaxed semantics, a shared contract change, a new [ABI](../../../../glossary.md#term-application-binary-interface-abi) export, a custom
[kernel](../../../../glossary.md#kernel), [host](../../../../glossary.md#term-host) rewrite, fallback, or value-dependent [dispatch](../../../../glossary.md#term-dispatch), stop and report the blocker.

## Dependencies and integration

- Depends on: [Metal](../../../../glossary.md#term-metal) 0015; [Model](../../../../glossary.md#term-model) 0027; Config 0006; [Engine](../../../../glossary.md#term-engine) 0018
- Conflicts with: Metal 0017–0018 and every Metal capability, preparation, native schema,
  candidate/codec, materialization, or public Metal Engine integration scope<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None
- Common base revision: N/A
- Integration order: Serial; after 0015 and before Metal 0017–0018
- Integration [validation](../../../../glossary.md#term-validation): profile-qualified Metal [unary](../../../../glossary.md#term-unary) checkpoint
- Shared-document integration [owner](../../../../glossary.md#term-owner): task implementer, finalized by independent Class C review

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, and `MetalNegPreparationPlan` — exact
  two-profile occurrence matrix, topology/state validation, lowering, and retained profile.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [topology glossary](../../../../glossary.md#term-topology); [validation glossary](../../../../glossary.md#term-validation); [lowering glossary](../../../../glossary.md#lowering))</span>
- `MetalMpsGraphProgram`, `MetalNativeApi`, `MetalDeviceContext`, and the native foundation — schema
  six, wires <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>`12..14`, Java [preflight](../../../../glossary.md#term-preflight), [MPSGraph](../../../../glossary.md#term-mpsgraph) selectors, and [ABI](../../../../glossary.md#term-application-binary-interface-abi)-v4 stability.
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec`, and decision
  compatibility — version-seven identities and stale rejection.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- Focused Metal schema/native/execution/candidate tests, Metal backend conformance, and the real
  explicit-composition Metal Engine integration test — per-operation positive and negative proof.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [integration test glossary](../../../../glossary.md#term-integration-test); [operation glossary](../../../../glossary.md#operation))</span>
- Metal package Javadocs, backend/native guides, this brief, Metal master plan, roadmap, focused API
  and glossary status — current capability and exclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Acceptance criteria

1. The disposable `ABS`, [`EXP`](../../../../glossary.md#term-expx), and [`SIGMOID`](../../../../glossary.md#term-sigmoid) gates each pass every numerical and control check
   before production edits; probe source and binary are absent from the repository and <span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>`/tmp`.
2. Both profiles admit exactly those three new unary kinds within the stated domain. Existing
   strict and accelerator matrices remain otherwise unchanged, and every excluded unary row is
   false and rejected before native resource creation.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [accelerator glossary](../../../../glossary.md#term-accelerator); [matrices glossary](../../../../glossary.md#term-matrix-meanings))</span>
3. [Schema](../../../../glossary.md#term-schema) six uses only wires `12..14` for the additions; ABI v4 retains exactly thirteen exports.
   Java and native validation agree on typed records, states, Shapes, and profile legality.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [validation glossary](../../../../glossary.md#term-validation); [Shapes glossary](../../../../glossary.md#shape))</span>
4. Version-seven workload/candidate/compatibility/route/codec identities are deterministic and
   reject version six, cross-profile, malformed, corrupt, stale-session, and unknown data.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>
5. Real backend and CPU-free Engine execution materialize observable operation-specific values,
   reuse a session, open an independent session, and prove close rejection; tests do not reduce the
   claim to compile or not-throw behavior.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [operation glossary](../../../../glossary.md#operation); [compile glossary](../../../../glossary.md#compile))</span>
6. Runtime/Trace stay profile-free; transfer, lifecycle, current routes, public API, and Model
   semantics do not change. Focused validation and independent Class C review pass.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [lifecycle glossary](../../../../glossary.md#lifecycle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test :backends:metal:javadoc
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :testing:architecture-tests:test
git diff --check
```

Inspect the per-[operation](../../../../glossary.md#operation) [oracle](../../../../glossary.md#term-test-oracle) summaries, exact exports/[schema](../../../../glossary.md#term-schema), generated [Metal](../../../../glossary.md#term-metal) Javadocs,
materialized [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), and absence of new [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) profile state. Validate changed Markdown
against the clean baseline for links, anchors, fences, final newlines, and whitespace.

Integration/repository validation: the serial task owns its focused Metal checkpoint; [CI](../../../../glossary.md#term-continuous-integration-ci) owns the
next full repository build unless implementation changes a [dependency](../../../../glossary.md#term-dependency) or shared build contract.

## Documentation and review impact

Update affected [Metal](../../../../glossary.md#term-metal) Javadocs, Metal/native guides, current capability/[API](../../../../glossary.md#term-application-programming-interface-api)/glossary status, this
brief, master plan, and roadmap. Independent Class C review must inspect each [operation](../../../../glossary.md#operation) separately,
profile [topology](../../../../glossary.md#term-topology), no-relaxation proof, gross-error exclusions, [schema](../../../../glossary.md#term-schema)/[identity](../../../../glossary.md#term-identity-meanings) evolution, [lifecycle](../../../../glossary.md#lifecycle),
real [Engine](../../../../glossary.md#term-engine) evidence, [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) absence, and exact [scope](../../../../glossary.md#term-scope).

## Result

The mandatory Apple M3 Max probe ran 324 independently compiled executables for 2,592 executions
across both optimization levels, all three [Shapes](../../../../glossary.md#shape), two fresh contexts, and all three [topology](../../../../glossary.md#term-topology)
forms. `ABS` passed its exact gate completely. [`EXP`](../../../../glossary.md#term-expx) and [`SIGMOID`](../../../../glossary.md#term-sigmoid) reproducibly flushed
representable [subnormal](../../../../glossary.md#term-subnormal) results to [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), including input `-88.0f` (`0xc2b00000`), whose
expected result was `0x0041edc4`, across every level, Shape, context, and form. `SIGMOID` also
produced two-unit-in-the-last-place ordinary [finite](../../../../glossary.md#term-finite) results at `-7.25f` and `-1.125f`, exceeding
the unchanged one-[ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) gate. Input preservation, canaries, reported Shapes, direct [targets](../../../../glossary.md#term-target),
feed/target permutations, and no-[DAZ](../../../../glossary.md#term-daz) controls all passed.

Because the contract requires all three independent gates to pass, no capability, [schema](../../../../glossary.md#term-schema), native,
[identity](../../../../glossary.md#term-identity-meanings), test, or documentation behavior change was made. The disposable probe was removed and
the tree returned to clean revision `9f7ad2d3e3d0fbdd690f200c14887fd4a158b514`. A future restart
of 0016 requires an exact replacement for both failed operations without weakening this contract.

Status: Incomplete
Follow-up required: leave 0016 Blocked unless both `EXP` and `SIGMOID` satisfy every unchanged
gate through an exact replacement route.
