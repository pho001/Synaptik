# Task 0075: BFLOAT16/FLOAT16 EXP custom route

## Status

Complete — BFLOAT16/FLOAT16 custom EXP implementation, source-matched device suites,
documentation, 0069 evidence rebinding, repository integration build, and independent Class C
review all passed.

## Change class

Class C — new low-precision Metal capability changes native custom execution, dtype-specific
provider answers, route identity, and user-visible numerical behavior.

## Goal

Admit homogeneous BFLOAT16 and FLOAT16 Model `EXP` occurrences through the existing fixed
`CUSTOM_PROGRAM` low-containing partition route. Both storage types remain distinct; each
element is decoded to FLOAT32 working value, EXP is evaluated there, and each declared low
result is narrowed once to its original type.

## Scope

- Qualify parameterless `UnaryElementwiseKind.EXP` with one canonical positive-static rank
  `1..16` BFLOAT16 or FLOAT16 input and a same-shape/same-type canonical output, initially
  without gradient roles. Checked two-byte element spans, dispatch geometry, resource aliases,
  and Java/native preflight must agree before mutation.
- Add a bounded low custom unary kernel and explicit BFLOAT16/FLOAT16 kernel selections, wire-55
  validation, provider predicates, prepared-route and compatibility identity updates. No MPSGraph
  node may consume or produce a low value, and no hidden tensor cast, host repair, retry, or
  alternate route is permitted.
- Qualify each dtype independently against the high-precision mathematical oracle
  `StrictMath.exp((double) widenedLowInput)`; separately compute the expected storage value by
  rounding that result first to FLOAT32 and then once to the original low dtype, without an
  intervening implicit tensor cast. For an
  ordinary positive normal finite reference/result, use test-only relative error
  `abs(actual-reference)/reference <= 0.0040` for BFLOAT16 and `<= 0.00055` for FLOAT16. These
  bounds allow at most roughly one half low-format rounding ULP plus a small FLOAT32 evaluation
  margin; they are predeclared test gates, not public accuracy guarantees or a runtime policy.
  If device evidence misses a bound, report the concrete inputs/error and stop for a reviewed
  threshold or algorithm decision; do not silently widen it.
- Test exact `EXP(±0)=+1`, `EXP(-infinity)=+0`, `EXP(+infinity)=+infinity`, and NaN class separately.
  Exercise stored low subnormal inputs, low-output underflow and genuine overflow. For a positive
  low-subnormal reference, a nonzero low result must be positive and within one low-subnormal ULP
  of the independently rounded low reference. Require FLOAT16 `EXP(-12)` to stay positive
  subnormal as a deliberately stricter qualification gate for this custom route; the Model's
  FLOAT16 primitive-result FTZ permission would otherwise allow `+0`, so this test is not a
  general framework guarantee. BFLOAT16 subnormal output may be `+0` under its arithmetic-site
  FTZ permission. Let `q` be the target low format's minimum positive subnormal. When the
  high-precision reference rounds to low `+0`, permit one positive `q` only in the declared
  rounding-boundary neighborhood `0.4q <= reference < 0.5q`; require `+0` below `0.4q` (for
  example FLOAT16 `EXP(-20)`). These are test-only underflow gates, not a blanket one-ULP
  allowance. `EXP(12)` must overflow to
  `+infinity` in FLOAT16 but remain finite in BFLOAT16. Test values near each dtype's overflow
  boundary without accepting an incorrect class under the ordinary finite tolerance.
- Update current-provider ledger rows and tests without changing the immutable historical v1
  baseline, and add backend-conformance and public Engine coverage for both low types.

## Non-goals

No mixed BFLOAT16/FLOAT16 operation, implicit BFLOAT16-to-FLOAT16 substitution, low MPSGraph,
gradient-bearing EXP, SIGMOID, runtime tolerance policy, numerical certificate, autotuning, or
performance claim. The FLOAT32 direct MPSGraph route remains owned by Task 0074.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants):
  distinct low storage, FLOAT32 working values, one final narrowing, custom-only low partition,
  independent provider answers, and append-only changed identities.
- [Foundational contract — Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics),
  especially [Family formulas and unary sites](../../../../architecture/contracts/foundational-modules.md#family-formulas-and-unary-sites)
  and [Arithmetic-site DAZ/FTZ](../../../../architecture/contracts/foundational-modules.md#arithmetic-site-dazftz):
  EXP's mathematical and special-value behavior.
- [Backend contract — Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity)
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend):
  occurrence-specific capability and fixed custom low route.
- [Runtime/Prepare contract — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle):
  fixed route and resource declarations precede Runtime.

## Dependencies and integration

- Depends on: 0074A Complete; 0074 Complete, including its final source-matched package,
  independent review and renewed 0069 evidence; 0075A Complete.
- Conflicts with: any concurrent Metal low-kernel/native/provider/schema/identity edit;
  resumed 0053 production; shared authoritative/explanatory-document edits.
- Parallel group: None.
- Common base revision: N/A for serial work. Current repository base is
  `1cab0731b75c035f589a60f8b9922a767211e6c6` plus completed uncommitted 0069/0074
  changes; current native foundation SHA-256 is
  `de9f80b61f81c860fab86c979625a171a6b0ce57507148051ba0ca593fe23014` and low-kernel
  header SHA-256 is `5753d2a80c46fca4aff27cab1ad7b631a30c3eeab4b185b13d9f75a2a3eece65`.
- Integration order: completed 0074 → 0075A contract update and review → re-verify this brief as
  Ready → clean Class C implementation context → independent review/documentation and 0069
  evidence rebind → integrated validation.
- Integration validation: source-matched native package, affected Metal backend/conformance/public
  Engine tests with real device and no relevant skips, architecture/Javadoc/docs checks, 0069
  evidence rebind if its authenticated source changes, and one full repository build at this
  cross-module capability checkpoint.
- Shared-document integration owner: main coordination context; disjoint worker/review scopes
  are assigned before launch.
- Frontier verification (2026-10-09): 0074A, 0074, and 0075A are Complete with independent
  approvals; 0074 has source-matched device tests, 0069 rebinding, documentation, and full build.
  Task 0075A removed the absolute root/backend prohibition without claiming current low support.
  Revised numerical gates have independent read-only approval; no competing Metal capability or
  native writer is active. This is the sole authorized serial Metal frontier. The stopped first
  worker changed no files; a clean implementation context may now resume against this brief.

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalNativeApi`,
  `MetalMpsGraphProgram`, route/identity companions — exact low occurrence and fixed route.
- `native/metal-macos-arm64/src/synaptik_low_precision_kernels.h` and
  `synaptik_metal_foundation.m` — typed low EXP kernel, preflight, and dispatch.
- Focused backend, conformance, and Engine tests; current ledger and affected Metal guide/Javadoc.
  The final documentation pass must also reconcile current-identity statements found by targeted
  search in `docs/glossary.md`, `docs/architecture/current-architecture-plan.md`,
  `docs/architecture/lifecycle.md`, `docs/user-guide/compiling-graphs.md`,
  `docs/backend-guide/partition-preparer.md`, `docs/api/tensor-api.md`, and other *current*
  explanatory references; preserve explicitly historical identity-30 records.

## Acceptance criteria

- Both low dtypes, and only their declared occurrence domain, report true and execute on the
  custom-only low route. FLOAT32 continues on its separate qualified MPSGraph route.
- Source-matched real-device results satisfy dtype-specific finite and special-value tests.
  The low output narrows once, with no full-tensor implicit cast or host repair.
- Java and native reject wrong dtype, geometry, layout, gradient, route, alias, malformed/stale
  image, or unavailable resources before output mutation. Historical v1 remains unchanged;
  current-provider ledger and actual provider answers agree. Changed route/serialized identities
  advance append-only from 31, with old values rejected; ABI 7 and schema 20 remain only if their
  actual wire formats do not change.
- The final current-support update is coordinated across `ARCHITECTURE.md`, the owning
  `backend-execution.md` contract, explanatory documentation, Javadoc, current v2 ledger,
  integration tests, and independent Class C review before this task is marked Complete. Task
  0075A conditionally authorizes the change but does not itself turn either provider answer true.

## Validation

Extend `MetalExpCapabilityTest`, `MetalExpNativeTest`, `MetalExpCapabilityConformanceTest`, and
`EngineExpMetalIntegrationTest` with both low dtypes and the declared negative/boundary matrix.
Worker builds, ad-hoc-signs, packages and verifies the edited native source, runs the affected
Metal/conformance/Engine device suites and Javadoc, and records exact per-dtype worst observed
ordinary-finite errors and special/boundary verdicts. Independently revalidate/rebind 0069
source/AIR evidence against any changed native foundation or low-kernel source. Integration owner
then runs one repository `./gradlew build --max-workers=1` against the source-matched package.
Always run `git diff --check` and validate documentation links.

## Documentation and review impact

Supported backend behavior and durable native route change. Independent Class C review/doc pass
must finalize guide, Javadocs, examples, glossary impact, and conformance evidence. No Model
formula or low-partition architecture change is proposed.

## Result

The first clean implementation context stopped before editing on the former absolute low-`EXP`
prohibition. After 0075A contract approval, the resumed worker implemented distinct
BFLOAT16/FLOAT16 custom kernels, Java/native preflight and fixed route selection, current v2 ledger
and identity 31→32. A preliminary review found scan metadata passed to the pointwise EXP kernel;
the worker fixed that P1 before device validation and added multi-element/sentinel evidence. A
later duplicate catalog entry P2 was also removed. Historical v1 remains unchanged; FLOAT32
retains MPSGraph and explicit-cast composition. Source-matched package verification, Metal 337
tests, conformance 31, integration 92, Metal Javadoc, documentation links, and `git diff --check`
passed with no relevant EXP skip or failure. Worst observed ordinary-finite relative errors were
BFLOAT16 0.0038361508384089 (gate 0.0040) and FLOAT16 0.0004676545351310302 (gate 0.00055);
declared special/underflow/overflow boundaries passed. Independent 0069 proof/source/AIR
rebinding passed without changing its fixed MSL/AIR corpus; final foundation SHA-256 is
`f831032edeecaee60129d1b6560697770a3d5447059d7d9a1c5e1ac6d0f2318f` and built/packaged
dylib SHA-256 is `b511cd16601de9c57618445229b11ae3b6a5b9b0f177b159f16ea1c931f01b1c`.
The final repository `./gradlew build --max-workers=1` passed. Independent Class C code,
documentation, and evidence review returned APPROVE with no remaining P0/P1/P2 findings. No
commit or push was made.

Status: Complete
