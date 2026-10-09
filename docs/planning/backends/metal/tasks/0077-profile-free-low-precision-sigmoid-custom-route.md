# Task 0077: BFLOAT16/FLOAT16 SIGMOID custom route

## Status

Draft — successor to Metal 0076. It is not an authorized implementation frontier until 0076 is
Complete and this brief receives a separate dependency/route/validation review.

## Change class

Class C — independently admitting low-precision numerical capability changes native custom
execution, prepared-route and compatibility behavior, and observable results.

## Goal

Qualify homogeneous BFLOAT16 and FLOAT16 `SIGMOID` separately using typed custom kernels inside
the existing fixed low-containing `CUSTOM_PROGRAM` partition.

## Scope

- Candidate domain: canonical positive-static-rank `1..16` no-gradient unary `SIGMOID`,
  same-shape/same-type low input and output, with checked two-byte spans and native geometry.
- Each element uses its original low storage type, exact stored sign guard, FLOAT32 working
  evaluation of the Model's stable branch formula, and one final narrowing to its original low
  type. No implicit BFLOAT16/FLOAT16 substitution or low MPSGraph step.
- Predeclare separate per-dtype finite metrics/thresholds and underflow gates before testing;
  test signed zeros, infinities, NaN, stored subnormals, saturation and both sign branches
  independently. A failed dtype remains provider-false; do not infer parity from the other.
- Update native source authentication, provider/current v2 ledger, identity, conformance, Engine,
  docs and source-bound 0069 evidence if the native foundation changes.

## Non-goals

No mixed-low, FLOAT64, gradient-bearing SIGMOID, direct MPSGraph low route, hidden casts,
runtime numerical policy, certificates, fallback or performance claim.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants):
  distinct low storage, FLOAT32 working values, one final narrowing, exact guard and fixed custom
  low partition.
- [Foundational contract — Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics),
  especially [Composite inheritance](../../../../architecture/contracts/foundational-modules.md#composite-inheritance)
  and [Model low-precision contract](../../../../architecture/contracts/foundational-modules.md#model-low-precision-contract).
- [Backend contract — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  and [Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity).
- [Runtime/Prepare contract — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle).

## Dependencies and integration

- Depends on: Metal 0076 Complete and the current Model 0032 contract.
- Conflicts with: concurrent Metal low kernels, native source/schema, provider, route, identity,
  shared-contract/documentation or 0069 source-bound evidence edits.
- Parallel group: None.
- Common base revision: N/A (serial); set after 0076 integration.
- Integration order: 0076 completion → this implementation → source-matched native/device
  qualification → independent Class C review → integration closure.
- Integration validation: focused Metal/conformance/Engine/device and 0069 evidence checks, then
  one repository build if this change crosses modules as expected.
- Shared-document integration owner: main coordination context.

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalNativeApi` and route/identity
  fixtures — independently typed admission and preflight.
- `native/metal-macos-arm64/src/synaptik_low_precision_kernels.h` and
  `synaptik_metal_foundation.m` — typed stable-branch kernels, dispatch and source authentication.
- Metal, backend-conformance and Engine tests; current v2 ledger and explanatory/Javadoc updates.

## Acceptance criteria

- Each low type passes its predeclared finite and special-value gates and stays on the fixed
  custom partition route; unsupported occurrences fail before mutation.
- No extra low narrowing occurs at internal named arithmetic sites; one final conversion stores
  the declared output type. FLOAT32 route and historical ledger remain unchanged.
- Source-matched real-device tests, native/package authentication, stale-identity rejection,
  documentation and independent review pass.

## Validation

Finalize exact thresholds, corpus, commands, and source/AIR rebinding steps when promoting this
Draft to Ready after Metal 0076. Do not launch an executor from this Draft.

## Documentation and review impact

Class C requires a separate code and documentation review. Update low capability explanation,
affected Javadocs, v2 ledger interpretation and targeted glossary references; retain Model-owned
semantics and historical v1 evidence.

## Result

Empty until execution.
