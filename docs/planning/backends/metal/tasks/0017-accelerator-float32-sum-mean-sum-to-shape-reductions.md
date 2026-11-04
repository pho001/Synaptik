# Task 0017: [ACCELERATOR](../../../../glossary.md#term-accelerator) [FLOAT32](../../../../glossary.md#term-float32) SUM/[MEAN](../../../../glossary.md#term-mean)/SUM_TO_SHAPE Reductions

## Reader context

There were two different failures to distinguish: a probe-lifetime defect was corrected, then
the valid numerical probe encountered the signed-zero blocker. Native labels and shape strings
needed to remain alive while Apple APIs used them; fixing that ownership problem did not change
which reduction results Model permitted.

Axes and bound target shapes described the reduction's terms. Actual input words determined
the permitted result set, rather than one preferred CPU summation order. A local scalar result
would be one FLOAT32 value copied to detached host bytes, not a newly supported scalar transfer.
Model owned the subsequent semantic decision; Metal could not make that decision inside a probe.
The [current reduction domains](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains)
provide present navigation while this task remains a record of its original blocked contract.

## Status

Blocked

The corrected mandatory pre-edit Apple M3 probe completed 300 independently created executables
and 2,400 runs after fixing probe-only [ownership](../../../../glossary.md#term-ownership-meanings): [graph](../../../../glossary.md#graph) label/[Shape](../../../../glossary.md#shape) strings remain live for native
use, and every [compile](../../../../glossary.md#compile)/[run](../../../../glossary.md#term-run) cell has its own [autorelease pool](../../../../glossary.md#native-packaging-and-platform-vocabulary). A minimal valid [rank](../../../../glossary.md#term-rank)-16 middle-[axis](../../../../glossary.md#term-axis)
cell (`axis=8`, `keepDimensions=false`) passed before the full [matrix](../../../../glossary.md#term-matrix-meanings). The full [oracle](../../../../glossary.md#term-test-oracle) then found
one existing-contract blocker: [accelerator](../../../../glossary.md#term-accelerator) SUM of declared terms `[-0,-0]` returned [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero),
while the current [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)-tree/per-step-[`FLOAT32`](../../../../glossary.md#term-float32)-[rounding](../../../../glossary.md#term-rounding) plus [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) set permits only [negative
zero](../../../../glossary.md#nan-infinity-and-signed-zero) because zero is not [subnormal](../../../../glossary.md#term-subnormal). No production, test, [schema](../../../../glossary.md#term-schema), capability, or documentation
implementation change was made; disposable probe [artifacts](../../../../glossary.md#term-artifact) were removed. [Model](../../../../glossary.md#term-model) 0028 owns the
separate proposed semantic decision, and [Metal](../../../../glossary.md#term-metal) 0020 is its Draft reduction successor.

## Change class

Class C — this adds profile-qualified reduction capability, native [schema](../../../../glossary.md#term-schema), [scalar](../../../../glossary.md#term-scalar) [publication](../../../../glossary.md#term-publication),
[cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings), prepared [lifecycle](../../../../glossary.md#lifecycle), and public [Engine](../../../../glossary.md#term-engine) execution across the native boundary. Fresh
pre-edit Apple M3 evidence and independent review are mandatory.

## Goal

Under `NumericalProfile.ACCELERATOR` only, add [canonical](../../../../glossary.md#term-canonical-form) [`FLOAT32`](../../../../glossary.md#term-float32) full, single-[axis](../../../../glossary.md#term-axis), and ordered
multi-axis `SUM`/[`MEAN`](../../../../glossary.md#term-mean) plus [binding](../../../../glossary.md#term-binding)-resolved `SUM_TO_SHAPE`. Preserve every declared term and admit
only the [Model](../../../../glossary.md#term-model)-owned [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)-tree/per-step-[rounding](../../../../glossary.md#term-rounding) [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) set. Keep all strict reductions false
and every existing profile-qualified [operation](../../../../glossary.md#operation) unchanged.

## Exact capability and [Shape](../../../../glossary.md#shape) domain

- Admit one [canonical](../../../../glossary.md#term-canonical-form) dense-[contiguous](../../../../glossary.md#term-contiguous) non-[view](../../../../glossary.md#term-view-meanings) [`FLOAT32`](../../../../glossary.md#term-float32) input of static positive [rank](../../../../glossary.md#term-rank) `1..16`,
  checked geometry, one canonical output, and equal `requiresGrad`.
- `SUM` accepts only no attrs, `AxisReductionAttrs`, `MultiAxisReductionAttrs`, or
  `SumToShapeAttrs`; [`MEAN`](../../../../glossary.md#term-mean) accepts only the first three. All masked/other forms remain false.
- Full form reduces all [axes](../../../../glossary.md#term-axis) to rank zero. Single-axis uses one [normalized axis](../../../../glossary.md#normalized-axis) and exact
  `keepDimensions`. Multi-axis preserves ordered distinct normalized axes; empty axes mean an
  identity, not full reduction, and retain `keepDimensions` in [identity](../../../../glossary.md#term-identity-meanings).
- `SUM_TO_SHAPE` right-aligns the bound target: leading axes reduce/disappear, target-one axes
  reduce/remain, and equal axes remain. Equal Shape copies exactly; scalar target reduces all axes;
  any other pair rejects before native creation.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Output rank is `0..16` with positive extents. Rank zero is a Metal-owned result/publication only,
  never a feed or ABS/binary operand; CPU↔Metal transfer remains positive rank <span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [publication glossary](../../../../glossary.md#term-publication); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [CPU glossary](../../../../glossary.md#term-cpu))</span>`1..16`.
- Accelerator reductions may compose with admitted canonical ABS/binary nodes when every
  occurrence qualifies. Outputs are canonical. Strict remains exactly NEG/ABS/affine/<span data-glossary-navigation="true"> ([Accelerator glossary](../../../../glossary.md#term-accelerator); [nodes glossary](../../../../glossary.md#node); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`CONTIGUOUS`
  and no strict reduction reaches native creation.

## Controlling numerical contract

For each output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), derive exactly the declared term multiset from kind, ordered [axes](../../../../glossary.md#term-axis),
`keepDimensions`, and bound [Shapes](../../../../glossary.md#shape). The allowed set contains every result reachable by a
[Model](../../../../glossary.md#term-model)-permitted reordering and full [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) tree over all and only those terms. Each addition rounds
once to [`FLOAT32`](../../../../glossary.md#term-float32); each [subnormal](../../../../glossary.md#term-subnormal) operand independently remains itself or becomes same-[signed zero](../../../../glossary.md#term-signed-zero),
and each [finite](../../../../glossary.md#term-finite) subnormal intermediate/result may remain or flush to either signed zero. Never
drop, duplicate, or invent a term.

`SUM` publishes a permitted root. [`MEAN`](../../../../glossary.md#term-mean) performs the final `FLOAT32` divide of a permitted sum by
the declared positive count under only the same row-scoped [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz). Point domains and equal-Shape
`SUM_TO_SHAPE` perform no arithmetic and preserve input bits. Preserve current [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity),
[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), and exceptional-path rules.

This is not an exact-real shortcut, fixed tree, [tolerance](../../../../glossary.md#term-tolerance), [ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) allowance, or generic fast math.
Never use [host](../../../../glossary.md#term-host) rewrite, value-dependent fallback, route retry, reciprocal substitution, algebraic
[identity](../../../../glossary.md#term-identity-meanings), reduced [precision](../../../../glossary.md#term-precision), cross-[node](../../../../glossary.md#node) [contraction](../../../../glossary.md#contraction-dimension), or special-value repair.

## Mandatory fresh disposable Apple M3 probe

Before production edits, create `/tmp/synaptik-metal-0017-reduction-probe.m`, [compile](../../../../glossary.md#compile) it with
`xcrun clang` against Foundation/[Metal](../../../../glossary.md#term-metal)/[MPSGraph](../../../../glossary.md#term-mpsgraph), [run](../../../../glossary.md#term-run) it on the real M3, and remove source/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata).
Use the intended production selectors and [Shape](../../../../glossary.md#shape) construction at optimization levels `0` and `1`,
reduced-[precision](../../../../glossary.md#term-precision) fast math disabled, three independent executables per cell, at least eight runs
per executable, and two fresh contexts.

Report `SUM` and [`MEAN`](../../../../glossary.md#term-mean) separately for full, first/middle/last single [axis](../../../../glossary.md#term-axis), ordered non-monotonic
multi-axis and empty-axis forms, both `keepDimensions`, [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)/positive-[rank](../../../../glossary.md#term-rank) outputs, Shapes
`[4]`, `[2,3,4]`, and a bounded rank-16 control. Report `SUM_TO_SHAPE` [scalar](../../../../glossary.md#term-scalar), leading-axis,
[target](../../../../glossary.md#term-target)-one, combined, and equal-Shape mappings. Include historical [cancellation](../../../../glossary.md#term-cancellation)
`[1.0e20f,1.0f,1.0f,-1.0e20f]`, tree-sensitive permutations, signed zeros, signed
minimum/ordinary/maximum [subnormals](../../../../glossary.md#term-subnormal), minimum normals/neighbors, [FTZ](../../../../glossary.md#term-ftz)-producing cancellations,
ordinary/maximum [finite](../../../../glossary.md#term-finite) values, [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and signed quiet/signaling NaNs with varied payloads.

An independent raw-bit [oracle](../../../../glossary.md#term-test-oracle) must enumerate every permitted outcome for each small term multiset:
all permutations and full binary trees, per-step [`FLOAT32`](../../../../glossary.md#term-float32) [rounding](../../../../glossary.md#term-rounding), independent [DAZ](../../../../glossary.md#term-daz) operands,
optional FTZ for every finite subnormal intermediate/result, and MEAN's final count divide. Compare
set membership, never one preferred tree; strict [host](../../../../glossary.md#term-host) arithmetic is control-only.

Use guarded direct targets; permute feeds/targets; verify Shape/count, inputs, canaries,
[identity](../../../../glossary.md#term-identity-meanings)/copy, selector isolation, repetition, and context independence. Report totals/mismatches
by [operation](../../../../glossary.md#operation), form, axes, keep-[dimensions](../../../../glossary.md#dimension), Shape, bucket, and level. Any out-of-set value, missing
term, Shape error, or control failure blocks 0017 without production/[schema](../../../../glossary.md#term-schema)/capability edits.

## Native [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), [lifecycle](../../../../glossary.md#lifecycle), and [cache](../../../../glossary.md#term-cache)

- Keep ABI v4, statuses, create signature, and exactly thirteen exports. Evolve fixed 160-byte node
  schema <span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [signature glossary](../../../../glossary.md#term-signature); [node glossary](../../../../glossary.md#node))</span>`6 -> 7`; retain wires `1..12`; append only `SUM=13` and `MEAN=14`.
- Schema 7 carries a typed full/single/multi reduction-form discriminator, ordered normalized axes
  including empty, exact <span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis))</span>`keepDimensions`, and [SUM-to-Shape](../../../../glossary.md#sum-to-shape) [target](../../../../glossary.md#term-target) [rank](../../../../glossary.md#term-rank)/[dimensions](../../../../glossary.md#dimension).
  `SUM_TO_SHAPE` remains `SUM`, not a new [Model](../../../../glossary.md#term-model) kind. Unused fields are zero/exact sentinel.
- Java/native validation agrees on output Shape, reduced axes, target mapping, positive count,
  canonical states, and bytes. Rank zero is legal only for a locally produced one-element
  reduction target; malformed attrs/axes/flags/ranks/Shapes/states, wires <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [canonical glossary](../../../../glossary.md#term-canonical-form); [element glossary](../../../../glossary.md#term-element))</span>`15+`, and
  profile-incompatible programs reject before resource creation.
- Advance workload-signature, exact-policy, candidate, compatibility, route-policy, and codec
  identities <span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>`7 -> 8`. Include profile/schema, kind/form, ordered axes, keep-dimensions, target
  Shape, term/count geometry, descriptors/states/topology, targets, ABI, and splats. Version seven,
  cross-profile, corrupt, malformed, foreign-session, and unknown data reject or miss safely.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [topology glossary](../../../../glossary.md#term-topology))</span>
- Reductions use MPSGraph only; custom singleton NEG remains strict-only. Typed complete candidates,
  safe absent-decision heuristic, and cache/tuning cannot broaden capability or declared terms.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [singleton glossary](../../../../glossary.md#term-singleton); [heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- Analysis declares exact buffers/address workspace; finalization transactionally creates one
  shape-specialized executable; cold binding validates buffers; hot execution makes one
  synchronous submission. Preserve rollback, leases, isolation, reuse, concurrency, direct
  targets, close rejection, and cleanup.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [binding glossary](../../../../glossary.md#term-binding); [synchronous glossary](../../../../glossary.md#term-synchronous); [rollback glossary](../../../../glossary.md#term-rollback); [leases glossary](../../../../glossary.md#term-lease); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Extend only local canonical materialization to download rank-zero FLOAT32 as four detached host
  bytes. Do not change transfer, shared Prepare/Runtime, or Runtime/Trace profile state.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [FLOAT32 glossary](../../../../glossary.md#term-float32); [detached glossary](../../../../glossary.md#detached-model-results); [host glossary](../../../../glossary.md#term-host); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace))</span>

## Non-goals

No strict or [masked reduction](../../../../glossary.md#masked-reduction); other reduction/[scan](../../../../glossary.md#term-cumulative-scan)/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[convolution](../../../../glossary.md#term-channels-first-convolution-layers)/[scalar](../../../../glossary.md#term-scalar) family; custom
reduction [kernel](../../../../glossary.md#kernel); exact-real/[tolerance](../../../../glossary.md#term-tolerance) claim; [host](../../../../glossary.md#term-host) fallback/rewrite; [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) transfer; dynamic,
zero-[extent](../../../../glossary.md#term-extent), or [view](../../../../glossary.md#term-view-meanings) [Shape](../../../../glossary.md#shape); other type; backward/training; packaging/discovery; or performance
claim. [Metal](../../../../glossary.md#term-metal) 0016 and historical 0006–0007/0009–0013 remain Blocked; 0018 and [Model](../../../../glossary.md#term-model) 0026 remain
Draft.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — [Model](../../../../glossary.md#term-model) owns results; profile and route decisions stay cold.
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span> — sole all-term tree, [rounding](../../../../glossary.md#term-rounding), [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz), count-divide, and exclusion rules.
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Concrete backend glossary](../../../../glossary.md#concrete-backend))</span> and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span> — truthful capability, typed candidates, [lowering](../../../../glossary.md#lowering), native state, and [materialization](../../../../glossary.md#materialization).
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>, [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>, and [Numerical-profile lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#numerical-profile-lifecycle) — fixed resources/routes, direct hot execution, and profile-free [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace).

If this needs a shared semantic change, new export, [host](../../../../glossary.md#term-host) rewrite/fallback, reduced [precision](../../../../glossary.md#term-precision),
[rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) transfer [widening](../../../../glossary.md#term-widening), or Runtime profile state, stop and report the blocker.

## Dependencies and integration

- Depends on: [Metal](../../../../glossary.md#term-metal) 0019; [Model](../../../../glossary.md#term-model) 0027; Config 0006; [Engine](../../../../glossary.md#term-engine) 0018; not Metal 0016
- Conflicts with: Metal 0018, any 0016 restart, and every Metal capability/preparation/native
  schema/candidate/codec/materialization/public Engine scope<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None
- Common base revision: `bcb717a6` plus this [planning](../../../../glossary.md#term-planning)-only authorization
- Integration order: Serial; after 0019 and before 0018
- Integration [validation](../../../../glossary.md#term-validation): [accelerator](../../../../glossary.md#term-accelerator) all-term Metal reduction [checkpoint](../../../../glossary.md#term-checkpoint)
- Shared-document integration [owner](../../../../glossary.md#term-owner): task implementer, finalized by independent Class C review

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalNegPreparationPlan` — exact profile/form/[Shape](../../../../glossary.md#shape)/state [matrix](../../../../glossary.md#term-matrix-meanings), [lowering](../../../../glossary.md#lowering), counts, and profile.
- `MetalMpsGraphProgram`, `MetalNativeApi`, `MetalDeviceContext`, native foundation — [schema](../../../../glossary.md#term-schema) 7, wires 13/14, [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [target](../../../../glossary.md#term-target), [preflight](../../../../glossary.md#term-preflight)/selectors, and [ABI](../../../../glossary.md#term-application-binary-interface-abi)-v4 stability.
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec`, and decision — version-eight deterministic [identity](../../../../glossary.md#term-identity-meanings) and stale rejection.
- `MetalBackendRuntime`/`MetalBackendIntegration` — four-byte local [scalar](../../../../glossary.md#term-scalar) [materialization](../../../../glossary.md#materialization) with unchanged positive-[rank](../../../../glossary.md#term-rank) transfer.
- Focused capability/schema/native/prepared/candidate tests, [Metal](../../../../glossary.md#term-metal) [conformance](../../../../glossary.md#term-conformance), and `EngineExplicitCompositionMetalIntegrationTest`; add narrow reduction tests.
- Affected Javadocs/package docs, Metal guide, native README, scoped architecture/[API](../../../../glossary.md#term-application-programming-interface-api)/user/capability/preparer status, glossary, and synchronized [planning](../../../../glossary.md#term-planning).

## Acceptance criteria

1. The fresh M3 probe passes every [operation](../../../../glossary.md#operation)/form/[axis](../../../../glossary.md#term-axis)/keep-[dimensions](../../../../glossary.md#dimension)/[Shape](../../../../glossary.md#shape)/[reassociation](../../../../glossary.md#term-reassociation)-sensitive enumerated-set and control gate before edits; no [artifact](../../../../glossary.md#term-artifact) remains.
2. Strict reductions stay false. [Accelerator](../../../../glossary.md#term-accelerator) adds exactly the declared [canonical](../../../../glossary.md#term-canonical-form) forms; malformed or wrong profile/type/[layout](../../../../glossary.md#layout)/[rank](../../../../glossary.md#term-rank)/Shape forms reject before native creation. Prior capability is unchanged.
3. Every result belongs to the all-term [oracle](../../../../glossary.md#term-test-oracle); [identities](../../../../glossary.md#term-identity-meanings) keep bits and [MEAN](../../../../glossary.md#term-mean) uses the positive count. No exact-real shortcut, [tolerance](../../../../glossary.md#term-tolerance), [host](../../../../glossary.md#term-host) rewrite, value fallback, reduced [precision](../../../../glossary.md#term-precision), or term [loss](../../../../glossary.md#loss) exists.
4. [Schema](../../../../glossary.md#term-schema) 7/Java/native agree; [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4 has thirteen exports. Deterministic version-eight [identity](../../../../glossary.md#term-identity-meanings) rejects version 7, cross-profile, malformed/corrupt/stale/unknown, and changed-Shape data.
5. [Lifecycle](../../../../glossary.md#lifecycle) proof covers all forms/chains, direct [targets](../../../../glossary.md#term-target), [scalar](../../../../glossary.md#term-scalar)/positive-rank [publication](../../../../glossary.md#term-publication), reuse, [concurrency](../../../../glossary.md#term-concurrency), contexts/sessions, [rollback](../../../../glossary.md#term-rollback)/close, and four-byte [materialization](../../../../glossary.md#materialization). [Rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) transfer still rejects before analysis; positive-rank transfer is unchanged.
6. [CPU](../../../../glossary.md#term-cpu)-free [Engine](../../../../glossary.md#term-engine) selects accelerator, executes/materializes oracle-checked every-form results including scalar, reuses and independently opens sessions, and proves close rejection. Strict [Metal](../../../../glossary.md#term-metal)-only reduction fails [ownership](../../../../glossary.md#term-ownership-meanings) selection; no not-throw-only claim.
7. Native/export, Metal/[Javadoc](../../../../glossary.md#term-javadoc), [conformance](../../../../glossary.md#term-conformance), Engine, architecture, full-build, Markdown/status/frontier, and diff checks pass; independent Class C review has no unresolved finding; [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) and [Model](../../../../glossary.md#term-model) semantics remain unchanged.

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
xcrun clang -fobjc-arc -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph /tmp/synaptik-metal-0017-reduction-probe.m -o /tmp/synaptik-metal-0017-reduction-probe
/tmp/synaptik-metal-0017-reduction-probe
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test :backends:metal:javadoc
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :testing:architecture-tests:test
./gradlew build
git diff --check
```

Record probe totals/removal, exports, [schema](../../../../glossary.md#term-schema)/[identity](../../../../glossary.md#term-identity-meanings), test counts/skips, [scalar](../../../../glossary.md#term-scalar) bits, profile
negatives, Markdown/link/anchor/fence/newline/whitespace, and [planning](../../../../glossary.md#term-planning) status/frontier evidence.
The serial [checkpoint](../../../../glossary.md#term-checkpoint) owns one full build; review reruns it only for changed code or stale evidence.

## Documentation and review impact

Update affected Javadocs, [Metal](../../../../glossary.md#term-metal)/native guides, scoped capability, [API](../../../../glossary.md#term-application-programming-interface-api)/user and capability/preparer status, targeted glossary, and synchronized [planning](../../../../glossary.md#term-planning). Independent Class C review inspects [oracle](../../../../glossary.md#term-test-oracle) summaries, every term/count/[Shape](../../../../glossary.md#shape) mapping, strict false [matrix](../../../../glossary.md#term-matrix-meanings), [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings), local-only [rank](../../../../glossary.md#term-rank) zero, [lifecycle](../../../../glossary.md#lifecycle)/[concurrency](../../../../glossary.md#term-concurrency)/cleanup, real [Engine](../../../../glossary.md#term-engine) proof, [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) absence, changed [scope](../../../../glossary.md#term-scope), and probe removal. Reuse worker [validation](../../../../glossary.md#term-validation) unless code changes or evidence is concretely stale.

## Result

The corrected pre-edit gate completed 300 executables and 2,400 runs. Probe-only dangling
autorelease-string [ownership](../../../../glossary.md#term-ownership-meanings) and per-cell pool [lifetime](../../../../glossary.md#term-lifetime) were fixed, and the minimal [rank](../../../../glossary.md#term-rank)-16
`axis=8`, `keepDimensions=false` control passed. The full valid [matrix](../../../../glossary.md#term-matrix-meanings) then reproduced an
old-contract mismatch: SUM of `[-0,-0]` produced [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) outside the permitted set, whose
only result is [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero). Because zero is not [subnormal](../../../../glossary.md#term-subnormal), existing [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) does not admit that
sign change. Task 0017 is Blocked under its unchanged contract. No production, test, [schema](../../../../glossary.md#term-schema),
capability, native, documentation-implementation, or retained probe [artifact](../../../../glossary.md#term-artifact) remains.
