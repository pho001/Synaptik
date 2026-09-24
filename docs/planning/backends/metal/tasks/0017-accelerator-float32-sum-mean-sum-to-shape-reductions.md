# Task 0017: ACCELERATOR FLOAT32 SUM/MEAN/SUM_TO_SHAPE Reductions

## Status

Ready

Readiness verification: Metal 0019 is Complete at implementation `a6d1796d` plus mixed-owner
remediation `bcb717a6`; all validation and independent Class C gates passed. Model 0027, Config
0006, and Engine 0018 are Complete. Metal 0016 remains Blocked without production changes, no
conflicting Metal write is active, and 0018 remains Draft. Clean base `bcb717a6` has ABI v4 with
exactly thirteen exports, node schema 6, and version-seven route/cache identities.

## Change class

Class C — this adds profile-qualified reduction capability, native schema, scalar publication,
cache identity, prepared lifecycle, and public Engine execution across the native boundary. Fresh
pre-edit Apple M3 evidence and independent review are mandatory.

## Goal

Under `NumericalProfile.ACCELERATOR` only, add canonical `FLOAT32` full, single-axis, and ordered
multi-axis `SUM`/`MEAN` plus binding-resolved `SUM_TO_SHAPE`. Preserve every declared term and admit
only the Model-owned binary-tree/per-step-rounding DAZ/FTZ set. Keep all strict reductions false
and every existing profile-qualified operation unchanged.

## Exact capability and Shape domain

- Admit one canonical dense-contiguous non-view `FLOAT32` input of static positive rank `1..16`,
  checked geometry, one canonical output, and equal `requiresGrad`.
- `SUM` accepts only no attrs, `AxisReductionAttrs`, `MultiAxisReductionAttrs`, or
  `SumToShapeAttrs`; `MEAN` accepts only the first three. All masked/other forms remain false.
- Full form reduces all axes to rank zero. Single-axis uses one normalized axis and exact
  `keepDimensions`. Multi-axis preserves ordered distinct normalized axes; empty axes mean an
  identity, not full reduction, and retain `keepDimensions` in identity.
- `SUM_TO_SHAPE` right-aligns the bound target: leading axes reduce/disappear, target-one axes
  reduce/remain, and equal axes remain. Equal Shape copies exactly; scalar target reduces all axes;
  any other pair rejects before native creation.
- Output rank is `0..16` with positive extents. Rank zero is a Metal-owned result/publication only,
  never a feed or ABS/binary operand; CPU↔Metal transfer remains positive rank `1..16`.
- Accelerator reductions may compose with admitted canonical ABS/binary nodes when every
  occurrence qualifies. Outputs are canonical. Strict remains exactly NEG/ABS/affine/`CONTIGUOUS`
  and no strict reduction reaches native creation.

## Controlling numerical contract

For each output coordinate, derive exactly the declared term multiset from kind, ordered axes,
`keepDimensions`, and bound Shapes. The allowed set contains every result reachable by a
Model-permitted reordering and full binary tree over all and only those terms. Each addition rounds
once to `FLOAT32`; each subnormal operand independently remains itself or becomes same-signed zero,
and each finite subnormal intermediate/result may remain or flush to either signed zero. Never
drop, duplicate, or invent a term.

`SUM` publishes a permitted root. `MEAN` performs the final `FLOAT32` divide of a permitted sum by
the declared positive count under only the same row-scoped DAZ/FTZ. Point domains and equal-Shape
`SUM_TO_SHAPE` perform no arithmetic and preserve input bits. Preserve current NaN, infinity,
signed-zero, and exceptional-path rules.

This is not an exact-real shortcut, fixed tree, tolerance, ULP allowance, or generic fast math.
Never use host rewrite, value-dependent fallback, route retry, reciprocal substitution, algebraic
identity, reduced precision, cross-node contraction, or special-value repair.

## Mandatory fresh disposable Apple M3 probe

Before production edits, create `/tmp/synaptik-metal-0017-reduction-probe.m`, compile it with
`xcrun clang` against Foundation/Metal/MPSGraph, run it on the real M3, and remove source/binary.
Use the intended production selectors and Shape construction at optimization levels `0` and `1`,
reduced-precision fast math disabled, three independent executables per cell, at least eight runs
per executable, and two fresh contexts.

Report `SUM` and `MEAN` separately for full, first/middle/last single axis, ordered non-monotonic
multi-axis and empty-axis forms, both `keepDimensions`, rank-zero/positive-rank outputs, Shapes
`[4]`, `[2,3,4]`, and a bounded rank-16 control. Report `SUM_TO_SHAPE` scalar, leading-axis,
target-one, combined, and equal-Shape mappings. Include historical cancellation
`[1.0e20f,1.0f,1.0f,-1.0e20f]`, tree-sensitive permutations, signed zeros, signed
minimum/ordinary/maximum subnormals, minimum normals/neighbors, FTZ-producing cancellations,
ordinary/maximum finite values, infinities, and signed quiet/signaling NaNs with varied payloads.

An independent raw-bit oracle must enumerate every permitted outcome for each small term multiset:
all permutations and full binary trees, per-step `FLOAT32` rounding, independent DAZ operands,
optional FTZ for every finite subnormal intermediate/result, and MEAN's final count divide. Compare
set membership, never one preferred tree; strict host arithmetic is control-only.

Use guarded direct targets; permute feeds/targets; verify Shape/count, inputs, canaries,
identity/copy, selector isolation, repetition, and context independence. Report totals/mismatches
by operation, form, axes, keep-dimensions, Shape, bucket, and level. Any out-of-set value, missing
term, Shape error, or control failure blocks 0017 without production/schema/capability edits.

## Native schema, identity, lifecycle, and cache

- Keep ABI v4, statuses, create signature, and exactly thirteen exports. Evolve fixed 160-byte node
  schema `6 -> 7`; retain wires `1..12`; append only `SUM=13` and `MEAN=14`.
- Schema 7 carries a typed full/single/multi reduction-form discriminator, ordered normalized axes
  including empty, exact `keepDimensions`, and SUM-to-Shape target rank/dimensions.
  `SUM_TO_SHAPE` remains `SUM`, not a new Model kind. Unused fields are zero/exact sentinel.
- Java/native validation agrees on output Shape, reduced axes, target mapping, positive count,
  canonical states, and bytes. Rank zero is legal only for a locally produced one-element
  reduction target; malformed attrs/axes/flags/ranks/Shapes/states, wires `15+`, and
  profile-incompatible programs reject before resource creation.
- Advance workload-signature, exact-policy, candidate, compatibility, route-policy, and codec
  identities `7 -> 8`. Include profile/schema, kind/form, ordered axes, keep-dimensions, target
  Shape, term/count geometry, descriptors/states/topology, targets, ABI, and splats. Version seven,
  cross-profile, corrupt, malformed, foreign-session, and unknown data reject or miss safely.
- Reductions use MPSGraph only; custom singleton NEG remains strict-only. Typed complete candidates,
  safe absent-decision heuristic, and cache/tuning cannot broaden capability or declared terms.
- Analysis declares exact buffers/address workspace; finalization transactionally creates one
  shape-specialized executable; cold binding validates buffers; hot execution makes one
  synchronous submission. Preserve rollback, leases, isolation, reuse, concurrency, direct
  targets, close rejection, and cleanup.
- Extend only local canonical materialization to download rank-zero FLOAT32 as four detached host
  bytes. Do not change transfer, shared Prepare/Runtime, or Runtime/Trace profile state.

## Non-goals

No strict or masked reduction; other reduction/scan/MATMUL/convolution/scalar family; custom
reduction kernel; exact-real/tolerance claim; host fallback/rewrite; rank-zero transfer; dynamic,
zero-extent, or view Shape; other type; backward/training; packaging/discovery; or performance
claim. Metal 0016 and historical 0006–0007/0009–0013 remain Blocked; 0018 and Model 0026 remain
Draft.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Model owns results; profile and route decisions stay cold.
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles) — sole all-term tree, rounding, DAZ/FTZ, count-divide, and exclusion rules.
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules) and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) — truthful capability, typed candidates, lowering, native state, and materialization.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle), [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle), and [Numerical-profile lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#numerical-profile-lifecycle) — fixed resources/routes, direct hot execution, and profile-free Runtime/Trace.

If this needs a shared semantic change, new export, host rewrite/fallback, reduced precision,
rank-zero transfer widening, or Runtime profile state, stop and report the blocker.

## Dependencies and integration

- Depends on: Metal 0019; Model 0027; Config 0006; Engine 0018; not Metal 0016
- Conflicts with: Metal 0018, any 0016 restart, and every Metal capability/preparation/native
  schema/candidate/codec/materialization/public Engine scope
- Parallel group: None
- Common base revision: `bcb717a6` plus this planning-only authorization
- Integration order: Serial; after 0019 and before 0018
- Integration validation: accelerator all-term Metal reduction checkpoint
- Shared-document integration owner: task implementer, finalized by independent Class C review

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalNegPreparationPlan` — exact profile/form/Shape/state matrix, lowering, counts, and profile.
- `MetalMpsGraphProgram`, `MetalNativeApi`, `MetalDeviceContext`, native foundation — schema 7, wires 13/14, rank-zero target, preflight/selectors, and ABI-v4 stability.
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec`, and decision — version-eight deterministic identity and stale rejection.
- `MetalBackendRuntime`/`MetalBackendIntegration` — four-byte local scalar materialization with unchanged positive-rank transfer.
- Focused capability/schema/native/prepared/candidate tests, Metal conformance, and `EngineExplicitCompositionMetalIntegrationTest`; add narrow reduction tests.
- Affected Javadocs/package docs, Metal guide, native README, scoped architecture/API/user/capability/preparer status, glossary, and synchronized planning.

## Acceptance criteria

1. The fresh M3 probe passes every operation/form/axis/keep-dimensions/Shape/reassociation-sensitive enumerated-set and control gate before edits; no artifact remains.
2. Strict reductions stay false. Accelerator adds exactly the declared canonical forms; malformed or wrong profile/type/layout/rank/Shape forms reject before native creation. Prior capability is unchanged.
3. Every result belongs to the all-term oracle; identities keep bits and MEAN uses the positive count. No exact-real shortcut, tolerance, host rewrite, value fallback, reduced precision, or term loss exists.
4. Schema 7/Java/native agree; ABI v4 has thirteen exports. Deterministic version-eight identity rejects version 7, cross-profile, malformed/corrupt/stale/unknown, and changed-Shape data.
5. Lifecycle proof covers all forms/chains, direct targets, scalar/positive-rank publication, reuse, concurrency, contexts/sessions, rollback/close, and four-byte materialization. Rank-zero transfer still rejects before analysis; positive-rank transfer is unchanged.
6. CPU-free Engine selects accelerator, executes/materializes oracle-checked every-form results including scalar, reuses and independently opens sessions, and proves close rejection. Strict Metal-only reduction fails ownership selection; no not-throw-only claim.
7. Native/export, Metal/Javadoc, conformance, Engine, architecture, full-build, Markdown/status/frontier, and diff checks pass; independent Class C review has no unresolved finding; Runtime/Trace and Model semantics remain unchanged.

## Validation

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

Record probe totals/removal, exports, schema/identity, test counts/skips, scalar bits, profile
negatives, Markdown/link/anchor/fence/newline/whitespace, and planning status/frontier evidence.
The serial checkpoint owns one full build; review reruns it only for changed code or stale evidence.

## Documentation and review impact

Update affected Javadocs, Metal/native guides, scoped capability, API/user and capability/preparer status, targeted glossary, and synchronized planning. Independent Class C review inspects oracle summaries, every term/count/Shape mapping, strict false matrix, schema/cache identity, local-only rank zero, lifecycle/concurrency/cleanup, real Engine proof, Runtime/Trace absence, changed scope, and probe removal. Reuse worker validation unless code changes or evidence is concretely stale.

## Result

Empty until execution.
