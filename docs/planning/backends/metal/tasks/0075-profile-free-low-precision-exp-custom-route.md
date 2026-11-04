# Task 0075: [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) [EXP](../../../../glossary.md#term-expx) custom route

## Status

Complete — [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) custom [EXP](../../../../glossary.md#term-expx) implementation, source-matched [device](../../../../glossary.md#term-device) suites,
documentation, 0069 evidence rebinding, repository integration build, and independent Class C
review all passed.

## Change class

Class C — new low-[precision](../../../../glossary.md#term-precision) [Metal](../../../../glossary.md#term-metal) capability changes native custom execution, [dtype](../../../../glossary.md#data-type--datatype)-specific
provider answers, route [identity](../../../../glossary.md#term-identity-meanings), and user-visible numerical behavior.

## Goal

Admit [homogeneous](../../../../glossary.md#term-homogeneous) [BFLOAT16](../../../../glossary.md#term-bfloat16) and [FLOAT16](../../../../glossary.md#term-float16) [Model](../../../../glossary.md#term-model) [`EXP`](../../../../glossary.md#term-expx) occurrences through the existing fixed
`CUSTOM_PROGRAM` low-containing [partition](../../../../glossary.md#partition) route. Both storage types remain distinct; each
[element](../../../../glossary.md#term-element) is decoded to [FLOAT32](../../../../glossary.md#term-float32) working value, EXP is evaluated there, and each declared low
result is narrowed once to its original type.

## [Scope](../../../../glossary.md#term-scope)

- Qualify parameterless `UnaryElementwiseKind.EXP` with one [canonical](../../../../glossary.md#term-canonical-form) positive-static [rank](../../../../glossary.md#term-rank)
  `1..16` BFLOAT16 or FLOAT16 input and a same-shape/same-type canonical output, initially
  without gradient roles. Checked two-byte element spans, dispatch geometry, resource aliases,
  and Java/native preflight must agree before mutation.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [element glossary](../../../../glossary.md#term-element); [dispatch glossary](../../../../glossary.md#term-dispatch); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- Add a bounded low custom unary kernel and explicit BFLOAT16/FLOAT16 kernel selections, wire-55
  validation, provider predicates, prepared-route and compatibility identity updates. No MPSGraph
  node may consume or produce a low value, and no hidden tensor cast, host repair, retry, or
  alternate route is permitted.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [kernel glossary](../../../../glossary.md#kernel); [validation glossary](../../../../glossary.md#term-validation); [identity glossary](../../../../glossary.md#term-identity-meanings); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [node glossary](../../../../glossary.md#node); [tensor glossary](../../../../glossary.md#tensor); [host glossary](../../../../glossary.md#term-host))</span>
- Qualify each [dtype](../../../../glossary.md#data-type--datatype) independently against the high-[precision](../../../../glossary.md#term-precision) mathematical [oracle](../../../../glossary.md#term-test-oracle)
  `StrictMath.exp((double) widenedLowInput)`; separately compute the expected storage value by
  rounding that result first to FLOAT32 and then once to the original low dtype, without an
  intervening implicit tensor cast. For an
  ordinary positive normal finite reference/result, use test-only relative error<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [FLOAT32 glossary](../../../../glossary.md#term-float32); [finite glossary](../../../../glossary.md#term-finite); [relative error glossary](../../../../glossary.md#term-relative-error))</span>
  `abs(actual-reference)/reference <= 0.0040` for [BFLOAT16](../../../../glossary.md#term-bfloat16) and `<= 0.00055` for FLOAT16. These
  bounds allow at most roughly one half low-format rounding ULP plus a small FLOAT32 evaluation
  margin; they are predeclared test gates, not public accuracy guarantees or a runtime policy.
  If device evidence misses a bound, report the concrete inputs/error and stop for a reviewed
  threshold or algorithm decision; do not silently widen it.<span data-glossary-navigation="true"> ([ULP glossary](../../../../glossary.md#term-unit-in-the-last-place-ulp); [evaluation glossary](../../../../glossary.md#term-evaluation); [runtime glossary](../../../../glossary.md#term-runtime); [device glossary](../../../../glossary.md#term-device))</span>
- Test exact `EXP(±0)=+1`, `EXP(-infinity)=+0`, `EXP(+infinity)=+infinity`, and NaN class separately.
  Exercise stored low subnormal inputs, low-output underflow and genuine overflow. For a positive
  low-subnormal reference, a nonzero low result must be positive and within one low-subnormal ULP
  of the independently rounded low reference. Require FLOAT16 <span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [subnormal glossary](../../../../glossary.md#term-subnormal); [underflow glossary](../../../../glossary.md#term-underflow))</span>`EXP(-12)` to stay positive
  subnormal as a deliberately stricter qualification gate for this custom route; the Model's
  FLOAT16 primitive-result FTZ permission would otherwise allow <span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification); [Model glossary](../../../../glossary.md#term-model); [FTZ glossary](../../../../glossary.md#term-ftz))</span>`+0`, so this test is not a
  general framework guarantee. BFLOAT16 subnormal output may be `+0` under its arithmetic-site
  FTZ permission. Let `q` be the target low format's minimum positive subnormal. When the
  high-precision reference rounds to low <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`+0`, permit one positive `q` only in the declared
  rounding-boundary neighborhood `0.4q <= reference < 0.5q`; require `+0` below `0.4q` (for
  example FLOAT16 `EXP(-20)`). These are test-only underflow gates, not a blanket one-ULP
  allowance. `EXP(12)` must [overflow](../../../../glossary.md#term-overflow) to
  `+infinity` in FLOAT16 but remain finite in BFLOAT16. Test values near each dtype's overflow
  boundary without accepting an incorrect class under the ordinary finite tolerance.<span data-glossary-navigation="true"> ([tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- Update current-provider ledger rows and tests without changing the immutable historical v1
  baseline, and add backend-conformance and public Engine coverage for both low types.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>

## Non-goals

No mixed [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) [operation](../../../../glossary.md#operation), implicit BFLOAT16-to-FLOAT16 substitution, low [MPSGraph](../../../../glossary.md#term-mpsgraph),
[gradient](../../../../glossary.md#term-gradient)-bearing [EXP](../../../../glossary.md#term-expx), [SIGMOID](../../../../glossary.md#term-sigmoid), [runtime](../../../../glossary.md#term-runtime) [tolerance](../../../../glossary.md#term-tolerance) policy, numerical [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions), [autotuning](../../../../glossary.md#autotuning--model-autotuning), or
performance claim. The [FLOAT32](../../../../glossary.md#term-float32) direct MPSGraph route remains owned by Task 0074.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants):
  distinct low storage, FLOAT32 working values, one final narrowing, custom-only low partition,
  independent provider answers, and append-only changed identities.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [narrowing glossary](../../../../glossary.md#term-narrowing); [partition glossary](../../../../glossary.md#partition))</span>
- [Foundational contract — Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([Profile-free numerical semantics glossary](../../../../glossary.md#profile-free-numerical-semantics))</span>,
  especially [Family formulas and unary sites](../../../../architecture/contracts/foundational-modules.md#family-formulas-and-unary-sites)<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>
  and [Arithmetic-site DAZ/FTZ](../../../../architecture/contracts/foundational-modules.md#arithmetic-site-dazftz)<span data-glossary-navigation="true"> ([DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz))</span>:
  EXP's mathematical and special-value behavior.<span data-glossary-navigation="true"> ([EXP glossary](../../../../glossary.md#term-expx))</span>
- [Backend contract — Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [backend capability glossary](../../../../glossary.md#backend-capability); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>:
  occurrence-specific capability and fixed custom low route.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- [Runtime/Prepare contract — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>:
  fixed route and resource declarations precede Runtime.

## Dependencies and integration

- Depends on: 0074A Complete; 0074 Complete, including its final source-matched package,
  independent review and renewed 0069 evidence; 0075A Complete.
- Conflicts with: any concurrent Metal low-kernel/native/provider/schema/identity edit;
  resumed 0053 production; shared authoritative/explanatory-document edits.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [kernel glossary](../../../../glossary.md#kernel); [schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Parallel group: None.
- Common base revision: N/A for serial work. Current repository base is
  `1cab0731b75c035f589a60f8b9922a767211e6c6` plus completed uncommitted 0069/0074
  changes; current native foundation SHA-256 is<span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256))</span>
  `de9f80b61f81c860fab86c979625a171a6b0ce57507148051ba0ca593fe23014` and low-kernel
  header SHA-256 is `5753d2a80c46fca4aff27cab1ad7b631a30c3eeab4b185b13d9f75a2a3eece65`.
- Integration order: completed 0074 → 0075A contract update and review → re-verify this brief as
  Ready → clean Class C implementation context → independent review/documentation and 0069
  evidence rebind → integrated validation.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [rebind glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [validation glossary](../../../../glossary.md#term-validation))</span>
- Integration validation: source-matched native package, affected Metal backend/conformance/public
  Engine tests with real device and no relevant skips, architecture/Javadoc/docs checks, 0069
  evidence rebind if its authenticated source changes, and one full repository build at this
  cross-module capability checkpoint.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [device glossary](../../../../glossary.md#term-device); [Javadoc glossary](../../../../glossary.md#term-javadoc); [module glossary](../../../../glossary.md#term-module); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Shared-document integration owner: main coordination context; disjoint worker/review scopes
  are assigned before launch.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Frontier verification (2026-10-09): 0074A, 0074, and 0075A are Complete with independent
  approvals; 0074 has source-matched device tests, 0069 rebinding, documentation, and full build.
  Task 0075A removed the absolute root/backend prohibition without claiming current low support.
  Revised numerical gates have independent read-only approval; no competing Metal capability or
  native writer is active. This is the sole authorized serial Metal frontier. The stopped first
  worker changed no files; a clean implementation context may now resume against this brief.<span data-glossary-navigation="true"> ([resume glossary](../../../../glossary.md#schema-serialization-and-checkpoint))</span>

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalNativeApi`,
  `MetalMpsGraphProgram`, route/[identity](../../../../glossary.md#term-identity-meanings) companions — exact low [occurrence](../../../../glossary.md#term-occurrence) and fixed route.
- `native/metal-macos-arm64/src/synaptik_low_precision_kernels.h` and
  `synaptik_metal_foundation.m` — typed low [EXP](../../../../glossary.md#term-expx) [kernel](../../../../glossary.md#kernel), [preflight](../../../../glossary.md#term-preflight), and [dispatch](../../../../glossary.md#term-dispatch).
- Focused backend, conformance, and Engine tests; current ledger and affected Metal guide/Javadoc.
  The final documentation pass must also reconcile current-identity statements found by targeted
  search in <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [Metal glossary](../../../../glossary.md#term-metal); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`docs/glossary.md`, `docs/architecture/current-architecture-plan.md`,
  `docs/architecture/lifecycle.md`, `docs/user-guide/compiling-graphs.md`,
  `docs/backend-guide/partition-preparer.md`, `docs/api/tensor-api.md`, and other *current*
  explanatory references; preserve explicitly historical identity-30 records.

## Acceptance criteria

- Both low dtypes, and only their declared occurrence domain, report true and execute on the
  custom-only low route. FLOAT32 continues on its separate qualified MPSGraph route.<span data-glossary-navigation="true"> ([dtypes glossary](../../../../glossary.md#data-type--datatype); [occurrence glossary](../../../../glossary.md#term-occurrence); [FLOAT32 glossary](../../../../glossary.md#term-float32); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- Source-matched real-device results satisfy dtype-specific finite and special-value tests.
  The low output narrows once, with no full-tensor implicit cast or host repair.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [finite glossary](../../../../glossary.md#term-finite); [tensor glossary](../../../../glossary.md#tensor); [host glossary](../../../../glossary.md#term-host))</span>
- Java and native reject wrong dtype, geometry, layout, gradient, route, alias, malformed/stale
  image, or unavailable resources before output mutation. Historical v1 remains unchanged;
  current-provider ledger and actual provider answers agree. Changed route/serialized identities
  advance append-only from 31, with old values rejected; ABI 7 and schema 20 remain only if their
  actual wire formats do not change.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [alias glossary](../../../../glossary.md#aliasing-and-arena); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema))</span>
- The final current-support update is coordinated across `ARCHITECTURE.md`, the owning
  `backend-execution.md` contract, explanatory documentation, Javadoc, current v2 ledger,
  integration tests, and independent Class C review before this task is marked Complete. Task
  0075A conditionally authorizes the change but does not itself turn either provider answer true.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## [Validation](../../../../glossary.md#term-validation)

Extend `MetalExpCapabilityTest`, `MetalExpNativeTest`, `MetalExpCapabilityConformanceTest`, and
`EngineExpMetalIntegrationTest` with both low [dtypes](../../../../glossary.md#data-type--datatype) and the declared negative/boundary [matrix](../../../../glossary.md#term-matrix-meanings).
Worker builds, ad-hoc-signs, packages and verifies the edited native source, runs the affected
[Metal](../../../../glossary.md#term-metal)/[conformance](../../../../glossary.md#term-conformance)/[Engine](../../../../glossary.md#term-engine) [device](../../../../glossary.md#term-device) suites and [Javadoc](../../../../glossary.md#term-javadoc), and records exact per-dtype worst observed
ordinary-[finite](../../../../glossary.md#term-finite) errors and special/boundary verdicts. Independently revalidate/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) 0069
source/[AIR](../../../../glossary.md#term-air) evidence against any changed native foundation or low-[kernel](../../../../glossary.md#kernel) source. Integration [owner](../../../../glossary.md#term-owner)
then runs one repository `./gradlew build --max-workers=1` against the source-matched package.
Always [run](../../../../glossary.md#term-run) `git diff --check` and validate documentation links.

## Documentation and review impact

Supported [backend](../../../../glossary.md#backend) behavior and durable native route change. Independent Class C review/doc pass
must finalize guide, Javadocs, examples, glossary impact, and [conformance](../../../../glossary.md#term-conformance) evidence. No [Model](../../../../glossary.md#term-model)
formula or low-[partition](../../../../glossary.md#partition) architecture change is proposed.

## Result

The first clean implementation context stopped before editing on the former absolute low-[`EXP`](../../../../glossary.md#term-expx)
prohibition. After 0075A contract approval, the resumed worker implemented distinct
[BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) custom [kernels](../../../../glossary.md#kernel), Java/native [preflight](../../../../glossary.md#term-preflight) and fixed route selection, current v2 ledger
and [identity](../../../../glossary.md#term-identity-meanings) 31→32. A preliminary review found [scan](../../../../glossary.md#term-cumulative-scan) [metadata](../../../../glossary.md#term-metadata) passed to the [pointwise](../../../../glossary.md#term-pointwise) EXP [kernel](../../../../glossary.md#kernel);
the worker fixed that P1 before [device](../../../../glossary.md#term-device) [validation](../../../../glossary.md#term-validation) and added multi-[element](../../../../glossary.md#term-element)/sentinel evidence. A
later duplicate catalog entry P2 was also removed. Historical v1 remains unchanged; [FLOAT32](../../../../glossary.md#term-float32)
retains [MPSGraph](../../../../glossary.md#term-mpsgraph) and explicit-cast composition. Source-matched package verification, [Metal](../../../../glossary.md#term-metal) 337
tests, [conformance](../../../../glossary.md#term-conformance) 31, integration 92, Metal [Javadoc](../../../../glossary.md#term-javadoc), documentation links, and `git diff --check`
passed with no relevant EXP skip or failure. Worst observed ordinary-[finite](../../../../glossary.md#term-finite) relative errors were
BFLOAT16 0.0038361508384089 (gate 0.0040) and FLOAT16 0.0004676545351310302 (gate 0.00055);
declared special/[underflow](../../../../glossary.md#term-underflow)/[overflow](../../../../glossary.md#term-overflow) boundaries passed. Independent 0069 proof/source/[AIR](../../../../glossary.md#term-air)
rebinding passed without changing its fixed [MSL](../../../../glossary.md#term-metal-shading-language-msl)/AIR corpus; final foundation [SHA-256](../../../../glossary.md#term-sha-256) is
`f831032edeecaee60129d1b6560697770a3d5447059d7d9a1c5e1ac6d0f2318f` and built/packaged
[dylib](../../../../glossary.md#term-dylib) SHA-256 is `b511cd16601de9c57618445229b11ae3b6a5b9b0f177b159f16ea1c931f01b1c`.
The final repository `./gradlew build --max-workers=1` passed. Independent Class C code,
documentation, and evidence review returned APPROVE with no remaining P0/P1/P2 findings. No
[commit](../../../../glossary.md#term-commit) or push was made.

Status: Complete
