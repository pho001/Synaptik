# Task 0016: Profile-Qualified FLOAT32 ABS, EXP, and SIGMOID

## Status

Ready

Readiness verification: CPU 0017 and Metal 0015 are Complete. Their independent Class C reviews
returned `APPROVE` with zero findings, and the serial numerical-profile backend checkpoint passed.
Metal 0015 supplies the current ABI-v4, node-schema-5, profile-guarded whole-partition route and
version-six identities. No conflicting Metal write task is active. Historical Metal 0006 remains
Blocked; this successor neither rewrites that result nor assumes its broad nine-operation scope.

## Change class

Class C — this changes profile-qualified Metal capability, the typed native schema, whole-partition
lowering, prepared compatibility, and public Engine execution across the native resource boundary.
Fresh operation-specific real-device evidence and independent review are mandatory.

## Goal

Add exactly `UnaryElementwiseKind.ABS`, `EXP`, and `SIGMOID` for canonical `FLOAT32` Metal
execution under both `STRICT_IEEE` and `ACCELERATOR`, without adding an accelerator-only unary
relaxation. Preserve every current strict NEG/affine/`CONTIGUOUS` row and accelerator tensor-binary
row. Keep every other unary operation fail-closed.

## Scope

- Treat the three operations as separate gates. `ABS` must be exact for every non-NaN raw input;
  NaN may vary only within the current NaN representation freedom. `EXP` and `SIGMOID` must each
  satisfy their own bounded high-precision oracle, exact special-value classes, signed-zero rules,
  overflow/underflow boundaries, and no-DAZ/no-FTZ controls.
- Require one `FLOAT32` input and output, `NoOperationAttrs.INSTANCE`, fully static positive rank
  `1..16`, equal Shape and `requiresGrad`, checked geometry, and canonical dense-contiguous,
  zero-offset non-view descriptors. Unary nodes produce canonical state and may consume only
  canonical state.
- Strict partitions may compose the retained NEG/affine/`CONTIGUOUS` domain with the three new
  unary kinds. Accelerator partitions may compose the retained tensor `ADD`/`SUB`/`MUL`/`DIV`
  domain with the three new unary kinds. The task does not make NEG or affine/layout operations
  accelerator-capable and does not make binary arithmetic strict-capable.
- Keep ABI version 4, statuses, create signature, and exactly thirteen exports. Advance node schema
  `5 -> 6`; retain wires `1..11`, append `ABS=12`, `EXP=13`, and `SIGMOID=14`, and retain the
  fixed 160-byte typed record with one input, no attributes, canonical input/output state, and all
  unused fields at their required sentinels or zero.
- Advance workload-signature, exact-policy, candidate, compatibility, route-policy, and codec
  identities `6 -> 7`. Include the profile and ordered unary topology; version-six, cross-profile,
  malformed, corrupt, foreign-session, and unknown-wire data must reject or miss safely.
- Preserve MPSGraph as the sole route for the three additions. The custom singleton route remains
  NEG-only. Preserve transactional finalization, context leases, direct targets, run isolation,
  synchronous submission, reuse, concurrency, close rejection, and canonical-only transfer.

### Mandatory operation-by-operation device gate

Before production edits, run a disposable Objective-C/MPSGraph probe at optimization levels `0`
and `1`, with reduced-precision fast math disabled, three independently compiled executables per
operation/level/Shape case, at least eight runs per executable, and two fresh contexts. Exercise
each operation alone and in repeated-node/fan-out/direct-target forms; permute feeds and targets,
preserve inputs, guard targets with canaries, and verify reported Shapes.

Use raw bits covering both zeros; positive/negative minimum, ordinary, and maximum subnormals;
minimum normals and adjacent values; ordinary small and large finite values; overflow/underflow
threshold neighborhoods; maximum finite values; both infinities; and quiet/signaling NaNs of both
signs with multiple payloads. Report results separately for each operation, optimization level,
Shape, context, and semantic bucket.

- `ABS`: require exact magnitude bits for every non-NaN input, `-0 -> +0`, infinities to positive
  infinity, and NaN classification preservation.
- `EXP`: require `-infinity -> +0`, `+infinity -> +infinity`, NaN preservation, no DAZ input
  rewrite, no FTZ of a representable subnormal result, correct overflow/underflow class, and at most
  one unit in the last place for every ordinary finite result against a higher-precision oracle.
- `SIGMOID`: require `-infinity -> +0`, `+infinity -> +1`, NaN preservation, stable finite-domain
  evaluation, no DAZ/FTZ shortcut, and at most one unit in the last place for every ordinary finite
  result against a higher-precision stable oracle.

The one-ULP measurements are conservative device gates, not new allowed-result sets or a generic
tolerance; every output must still satisfy the operation's existing Model contract.

All three gates must pass before production changes. Any failure blocks 0016 with no capability,
schema, native, or test change. Passing one operation does not authorize another operation or a
family-wide approximation policy.

## Non-goals

No `RECIPROCAL`, `LOG`, `SQRT`, `RSQRT`, `RELU`, `TANH`, other unary kind, scalar arithmetic,
reduction, MATMUL, comparison, extrema, scan, new data type, rank zero, dynamic Shape, view input,
custom unary kernel, transfer widening, backward/training claim, packaging, discovery, or
performance claim. In particular, never authorize observed `RELU(NaN) -> +0` or
`TANH(NaN) -> +1`; both are Model-forbidden gross behavior. A future relaxed unary operation needs
a coordinated Model contract with a named finite-domain and special-value envelope plus its own
bounded oracle. Model 0026 remains Draft.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Model
  owns result sets and profile selection stays cold.
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  — unlisted unary operations gain no accelerator relaxation; NaN-to-ordinary gross behavior is
  forbidden.
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) — Metal
  owns truthful capability, lowering, native integration, and resources.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) —
  routes and resources are fixed before direct hot execution.

If any operation needs relaxed semantics, a shared contract change, a new ABI export, a custom
kernel, host rewrite, fallback, or value-dependent dispatch, stop and report the blocker.

## Dependencies and integration

- Depends on: Metal 0015; Model 0027; Config 0006; Engine 0018
- Conflicts with: Metal 0017–0018 and every Metal capability, preparation, native schema,
  candidate/codec, materialization, or public Metal Engine integration scope
- Parallel group: None
- Common base revision: N/A
- Integration order: Serial; after 0015 and before Metal 0017–0018
- Integration validation: profile-qualified Metal unary checkpoint
- Shared-document integration owner: task implementer, finalized by independent Class C review

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, and `MetalNegPreparationPlan` — exact
  two-profile occurrence matrix, topology/state validation, lowering, and retained profile.
- `MetalMpsGraphProgram`, `MetalNativeApi`, `MetalDeviceContext`, and the native foundation — schema
  six, wires `12..14`, Java preflight, MPSGraph selectors, and ABI-v4 stability.
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec`, and decision
  compatibility — version-seven identities and stale rejection.
- Focused Metal schema/native/execution/candidate tests, Metal backend conformance, and the real
  explicit-composition Metal Engine integration test — per-operation positive and negative proof.
- Metal package Javadocs, backend/native guides, this brief, Metal master plan, roadmap, focused API
  and glossary status — current capability and exclusions.

## Acceptance criteria

1. The disposable `ABS`, `EXP`, and `SIGMOID` gates each pass every numerical and control check
   before production edits; probe source and binary are absent from the repository and `/tmp`.
2. Both profiles admit exactly those three new unary kinds within the stated domain. Existing
   strict and accelerator matrices remain otherwise unchanged, and every excluded unary row is
   false and rejected before native resource creation.
3. Schema six uses only wires `12..14` for the additions; ABI v4 retains exactly thirteen exports.
   Java and native validation agree on typed records, states, Shapes, and profile legality.
4. Version-seven workload/candidate/compatibility/route/codec identities are deterministic and
   reject version six, cross-profile, malformed, corrupt, stale-session, and unknown data.
5. Real backend and CPU-free Engine execution materialize observable operation-specific values,
   reuse a session, open an independent session, and prove close rejection; tests do not reduce the
   claim to compile or not-throw behavior.
6. Runtime/Trace stay profile-free; transfer, lifecycle, current routes, public API, and Model
   semantics do not change. Focused validation and independent Class C review pass.

## Validation

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

Inspect the per-operation oracle summaries, exact exports/schema, generated Metal Javadocs,
materialized raw bits, and absence of new Runtime/Trace profile state. Validate changed Markdown
against the clean baseline for links, anchors, fences, final newlines, and whitespace.

Integration/repository validation: the serial task owns its focused Metal checkpoint; CI owns the
next full repository build unless implementation changes a dependency or shared build contract.

## Documentation and review impact

Update affected Metal Javadocs, Metal/native guides, current capability/API/glossary status, this
brief, master plan, and roadmap. Independent Class C review must inspect each operation separately,
profile topology, no-relaxation proof, gross-error exclusions, schema/identity evolution, lifecycle,
real Engine evidence, Runtime/Trace absence, and exact scope.

## Result
