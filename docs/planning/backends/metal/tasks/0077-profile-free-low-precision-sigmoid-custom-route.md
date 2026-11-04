# Task 0077: [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) [SIGMOID](../../../../glossary.md#term-sigmoid) custom route

## Reader context

Low SIGMOID keeps the original carrier and chooses a branch from raw bits. Custom work uses FLOAT32 EXP, addition and division, then one final ties-to-even BFLOAT16/FLOAT16 store. Working intermediates are not separately narrowed low tensors, and the two low formats are not substitutes.

Metal owns the fixed recipe and native writes; provider/current-ledger answers are independently qualified. Observed errors, source hashes and no-gradient limits below are task-era evidence, not new measurements here. Current source shares numerical implementation through a common codec rather than the historical separate low-kernel owner.

- [Current shared numerical source and original carriers](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains) — Shows evolved implementation and typed domains.
- [Current low partition boundaries](../../../../backend-guide/metal-backend.md#physical-authentication-and-fixed-routes) — Explains why FLOAT32 working sites do not enable low MPSGraph.

## Status

Complete — separately qualified [BFLOAT16](../../../../glossary.md#term-bfloat16) and [FLOAT16](../../../../glossary.md#term-float16) [SIGMOID](../../../../glossary.md#term-sigmoid) custom routes passed source-matched
real-[device](../../../../glossary.md#term-device) [validation](../../../../glossary.md#term-validation), integration, documentation, and independent Class C review on 2026-10-09.

## Change class

Class C — independently admitting low-[precision](../../../../glossary.md#term-precision) numerical capability changes native custom
execution, prepared-route and compatibility behavior, and observable results.

## Goal

Qualify [homogeneous](../../../../glossary.md#term-homogeneous) [BFLOAT16](../../../../glossary.md#term-bfloat16) and [FLOAT16](../../../../glossary.md#term-float16) [`SIGMOID`](../../../../glossary.md#term-sigmoid) separately using typed custom [kernels](../../../../glossary.md#kernel) inside
the existing fixed low-containing `CUSTOM_PROGRAM` [partition](../../../../glossary.md#partition).

## [Scope](../../../../glossary.md#term-scope)

- Candidate domain: [canonical](../../../../glossary.md#term-canonical-form) positive-static-[rank](../../../../glossary.md#term-rank) `1..16` no-[gradient](../../../../glossary.md#term-gradient) [unary](../../../../glossary.md#term-unary) [`SIGMOID`](../../../../glossary.md#term-sigmoid),
  same-shape/same-type low input and output, with checked two-byte spans and native geometry.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- Each [element](../../../../glossary.md#term-element) uses its original low storage type and an exact raw-bit sign [guard](../../../../glossary.md#term-guard). Select `x` for
  negative input or `-x` for nonnegative input, evaluate one [`EXP`](../../../../glossary.md#term-expx) in FLOAT32, select the numerator
  (<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32))</span>`EXP` or typed one), then perform one FLOAT32 addition and division before exactly one final
  ties-to-even narrowing to the original low type. This is the Model's two stable branches with
  their named sites, not a direct opaque sigmoid or an algebraic substitute. No implicit
  BFLOAT16/FLOAT16 substitution or low MPSGraph step.<span data-glossary-navigation="true"> ([ties-to-even glossary](../../../../glossary.md#term-ties-to-even); [narrowing glossary](../../../../glossary.md#term-narrowing); [Model glossary](../../../../glossary.md#term-model); [opaque glossary](../../../../glossary.md#term-opaque); [FLOAT16 glossary](../../../../glossary.md#term-float16); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- Qualify ordinary positive normal [finite](../../../../glossary.md#term-finite) outputs against the independent branch-correct
  `StrictMath.exp((double) widenedInput)` reference with test-only [relative error](../../../../glossary.md#term-relative-error)
  `abs(actual-reference)/reference <= 0.0040` for [BFLOAT16](../../../../glossary.md#term-bfloat16) and `<= 0.00055` for FLOAT16. These
  allow roughly half a low-format rounding ULP plus a small FLOAT32 working margin, and are not
  public guarantees. Exact classes and signs override tolerance. If a gate fails, report the
  concrete word/error and stop for a reviewed algorithm or threshold decision; never widen it
  silently.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [ULP glossary](../../../../glossary.md#term-unit-in-the-last-place-ulp); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- Test `SIGMOID(±0)=0.5`, `SIGMOID(-infinity)=+0`, `SIGMOID(+infinity)=1`, and NaN class
  separately. Exercise both sign branches, stored low subnormal inputs, ordinary near-zero,
  saturation, low-output subnormal/underflow and large tensors for each dtype. For a positive
  low-subnormal reference, a nonzero low result must be positive and within one target minimum-
  subnormal ULP of the independently narrowed reference. BFLOAT16 may return <span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [saturation glossary](../../../../glossary.md#term-saturation); [underflow glossary](../../../../glossary.md#term-underflow); [tensors glossary](../../../../glossary.md#tensor); [dtype glossary](../../../../glossary.md#data-type--datatype))</span>`+0` where its
  FLOAT32 working EXP/result site is subnormal and FTZ is allowed. FLOAT16 must retain a nonzero
  subnormal for <span data-glossary-navigation="true"> ([FTZ glossary](../../../../glossary.md#term-ftz))</span>`SIGMOID(-12)` and other cases whose FLOAT32 working value is normal and final
  narrowing is nonzero; `SIGMOID(-20)` must round to `+0`. When the high-precision reference
  rounds to low <span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision))</span>`+0`, permit one positive minimum subnormal only in the declared boundary
  neighborhood `0.4q <= reference < 0.5q`, with `q` the [target](../../../../glossary.md#term-target) minimum [subnormal](../../../../glossary.md#term-subnormal); below `0.4q`
  require `+0`. A finite result must remain in `[0,1]` and cannot use tolerance to excuse NaN,
  wrong infinity, negative zero or impossible saturation.<span data-glossary-navigation="true"> ([infinity glossary](../../../../glossary.md#term-infinity); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Update native source authentication, provider/current v2 ledger, identity, conformance, Engine,
  Javadocs, the root and owning backend contract's current capability/identity statements,
  explanatory docs, and source-bound 0069 evidence if the native foundation changes. Historical
  v1 and the separate FLOAT32 route stay unchanged.<span data-glossary-navigation="true"> ([authentication glossary](../../../../glossary.md#term-authentication); [identity glossary](../../../../glossary.md#term-identity-meanings); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend))</span>

## Non-goals

No mixed-low, [FLOAT64](../../../../glossary.md#term-float64), [gradient](../../../../glossary.md#term-gradient)-bearing [SIGMOID](../../../../glossary.md#term-sigmoid), direct [MPSGraph](../../../../glossary.md#term-mpsgraph) low route, hidden casts,
[runtime](../../../../glossary.md#term-runtime) numerical policy, certificates, [fusion](../../../../glossary.md#term-fusion), fallback, [autotuning](../../../../glossary.md#autotuning--model-autotuning) or performance claim.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants):
  distinct low storage, FLOAT32 working values, one final narrowing, exact guard and fixed custom
  low partition.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [narrowing glossary](../../../../glossary.md#term-narrowing); [guard glossary](../../../../glossary.md#term-guard); [partition glossary](../../../../glossary.md#partition))</span>
- [Foundational contract — Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([Profile-free numerical semantics glossary](../../../../glossary.md#profile-free-numerical-semantics))</span>,
  especially [Family formulas and unary sites](../../../../architecture/contracts/foundational-modules.md#family-formulas-and-unary-sites)<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>,
  [Arithmetic-site DAZ/FTZ](../../../../architecture/contracts/foundational-modules.md#arithmetic-site-dazftz)<span data-glossary-navigation="true"> ([DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz))</span>,
  [Composite inheritance](../../../../architecture/contracts/foundational-modules.md#composite-inheritance)
  and [Model low-precision contract](../../../../architecture/contracts/foundational-modules.md#model-low-precision-contract)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [precision glossary](../../../../glossary.md#term-precision))</span>.
- [Backend contract — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal))</span>
  and [Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity)<span data-glossary-navigation="true"> ([backend capability glossary](../../../../glossary.md#backend-capability); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>.
- [Runtime/Prepare contract — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>.

## Dependencies and integration

- Depends on: [Model](../../../../glossary.md#term-model) 0032 and [Metal](../../../../glossary.md#term-metal) 0069/0074A/0074/0075A/0075/0076 (all Complete).
- Conflicts with: concurrent Metal low kernels, native source/schema, provider, route, identity,
  shared-contract/documentation or 0069 source-bound evidence edits.<span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel); [schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Parallel group: None.
- Common base revision: N/A for serial work; clean implementation base `bcf9d3b4` on `main`.
- Integration order: 0076 completion → this implementation → source-matched native/device
  qualification → independent Class C review → integration closure.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Integration validation: source-matched native build/sign/package/verify, affected Metal,
  conformance, Engine, architecture and Javadoc checks, 0069 source/AIR rebinding if the native
  foundation changes, then one full <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [Javadoc glossary](../../../../glossary.md#term-javadoc); [AIR glossary](../../../../glossary.md#term-air))</span>`./gradlew build --max-workers=1` after executable integration.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main coordination context.
- Frontier verification (2026-10-09): all listed dependencies are Complete at `bcf9d3b4`;
  the only previously authorized Metal frontier 0076 is independently approved and pushed;
  0053 remains historical Blocked and is not a dependency. This brief is Ready at the sole serial
  Metal frontier. The current root/scoped contract permits a new separately qualified low
  occurrence through custom execution and coordinated current-capability/identity updates.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Ready glossary](../../../../glossary.md#term-ready); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalNativeApi`,
  `MetalPointwiseFusionPlanner` and route/identity fixtures — independently typed admission,
  preflight, fixed low route and no-fusion protection.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [admission glossary](../../../../glossary.md#term-admission); [preflight glossary](../../../../glossary.md#term-preflight); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- `native/metal-macos-arm64/src/synaptik_low_precision_kernels.h` and
  `synaptik_metal_foundation.m` — typed stable-branch [kernels](../../../../glossary.md#kernel), [dispatch](../../../../glossary.md#term-dispatch) and source [authentication](../../../../glossary.md#term-authentication).
- [Metal](../../../../glossary.md#term-metal), [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) and [Engine](../../../../glossary.md#term-engine) tests; current v2 ledger and explanatory/[Javadoc](../../../../glossary.md#term-javadoc) updates.

## Acceptance criteria

- Each low type passes its predeclared finite and special-value gates and stays on the fixed
  custom partition route; wrong type/rank/Shape/layout/attributes/gradient/span/alias/route and
  stale image fail before mutation. No low-valued MPSGraph or F32-only route is selected.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [partition glossary](../../../../glossary.md#partition); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [alias glossary](../../../../glossary.md#aliasing-and-arena); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- No extra low narrowing occurs at internal named arithmetic sites; one final conversion stores
  the declared output type. FLOAT32 route, ABI 7/schema 20 (unless actual wire changes), and
  historical ledger remain unchanged. Current v2 records both new true low answers independently.<span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing); [FLOAT32 glossary](../../../../glossary.md#term-float32); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema))</span>
- Source-matched real-device tests, native/package authentication, stale-identity rejection,
  backend conformance, public Engine composition, documentation/Javadoc and independent Class C
  review pass. Changed compatibility identity advances append-only beyond 33.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [authentication glossary](../../../../glossary.md#term-authentication); [identity glossary](../../../../glossary.md#term-identity-meanings); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker builds, ad-hoc signs, packages and verifies the edited native source. [Run](../../../../glossary.md#term-run) real-[device](../../../../glossary.md#term-device)
[Metal](../../../../glossary.md#term-metal) native/capability tests for each type, affected [conformance](../../../../glossary.md#term-conformance) and public [Engine](../../../../glossary.md#term-engine) tests against
the verified absolute [dylib](../../../../glossary.md#term-dylib) without relevant skips, plus architecture, Metal [Javadoc](../../../../glossary.md#term-javadoc) and
`git diff --check`:

```bash
./native/metal-macos-arm64/build.sh
/usr/bin/codesign --force --sign - --timestamp=none --identifier io.github.pho001.synaptik.metal.foundation native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/package-local.sh native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/verify-package.sh native/metal-macos-arm64/build/package-v1/macos-arm64
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test :testing:backend-conformance:test :testing:integration-tests:test --rerun-tasks --max-workers=1
./gradlew :backends:metal:javadoc
git diff --check
```

The integration [owner](../../../../glossary.md#term-owner) verifies and rebinds Task-0069 source/[AIR](../../../../glossary.md#term-air) evidence if the native foundation
changes, then runs the one full repository build against the final package. The independent
documentation/review pass reuses successful executable evidence unless it changes behavior.

## Documentation and review impact

Class C requires a separate code and documentation review. Update the current root/[backend](../../../../glossary.md#backend)
contract capability and [identity](../../../../glossary.md#term-identity-meanings) facts together, low capability guide, affected Javadocs, v2
ledger interpretation, examples/links and targeted glossary references; retain [Model](../../../../glossary.md#term-model)-owned
semantics and historical v1 evidence. No new architecture decision or [ADR](../../../../glossary.md#term-architecture-decision-record-adr) is intended because the
existing contract already specifies independently qualified low custom [admission](../../../../glossary.md#term-admission); stop if this
reading conflicts with source or another applicable contract.

## Result

Implemented separate typed [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) custom [SIGMOID](../../../../glossary.md#term-sigmoid) [kernels](../../../../glossary.md#kernel) with an exact raw-word sign
[guard](../../../../glossary.md#term-guard), one [FLOAT32](../../../../glossary.md#term-float32) [`EXP`](../../../../glossary.md#term-expx), FLOAT32 addition/division, and one final [ties-to-even](../../../../glossary.md#term-ties-to-even) low [narrowing](../../../../glossary.md#term-narrowing).
Capability, Java/native [preflight](../../../../glossary.md#term-preflight), fixed custom routing, no-[fusion](../../../../glossary.md#term-fusion) protection, current v2 ledger,
and coordinated [backend identity](../../../../glossary.md#backend-identity--backendid) 34 were updated. FLOAT32 remains on its [MPSGraph](../../../../glossary.md#term-mpsgraph) step, including
after an explicit low-to-FLOAT32 cast in the same custom [partition](../../../../glossary.md#partition). [ABI](../../../../glossary.md#term-application-binary-interface-abi) 7, [schema](../../../../glossary.md#term-schema) 20, and the
historical ledger remain unchanged; unsupported occurrences still [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

Changed production and test areas: [Metal](../../../../glossary.md#term-metal) provider, preflight, preparer, [identity](../../../../glossary.md#term-identity-meanings)/route/fusion and
Javadocs; native low [kernel](../../../../glossary.md#kernel) and foundation source; Metal capability/native tests, [backend](../../../../glossary.md#backend)
[conformance](../../../../glossary.md#term-conformance), public [Engine](../../../../glossary.md#term-engine) integration, and the current v2 ledger. Updated the root and owning
backend contract, relevant [API](../../../../glossary.md#term-application-programming-interface-api)/architecture/backend/user guides, glossary, native README, and
Task-0069 foundation source [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions). No new architecture decision or [ADR](../../../../glossary.md#term-architecture-decision-record-adr) was required because
the authorized route and numerical sites were already specified by the existing contracts.

[Validation](../../../../glossary.md#term-validation): native build, [ad-hoc signing](../../../../glossary.md#term-ad-hoc-signing), local package and package verification passed. The
source-matched real-[device](../../../../glossary.md#term-device) [run](../../../../glossary.md#term-run) passed 346 Metal tests (10 pre-existing skips), 32 backend
conformance tests (0 skips), and 94 integration tests (1 pre-existing skip), all with zero
failures. Focused SIGMOID native 6/6, capability 1/1, conformance 1/1, and Engine 2/2 had no
skips. Worst ordinary [finite](../../../../glossary.md#term-finite) relative errors were 0.003787 BFLOAT16 versus the predeclared
0.0040 gate and 0.000474 FLOAT16 versus 0.00055. The Task-0069 source certificate, [Lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) proof,
and compiled-[MSL](../../../../glossary.md#term-metal-shading-language-msl)/[AIR](../../../../glossary.md#term-air) audit passed against foundation [SHA-256](../../../../glossary.md#term-sha-256)
`b2c94c90d1a58d768f409bfa70986c40f26ca41e01954e1edff3894d3481f4cd`.
`./gradlew :backends:metal:javadoc` and `./gradlew build --max-workers=1` passed; the latter
reported 89 tasks (5 executed, 84 up-to-date). An independent Class C review returned `APPROVE`
with zero P0/P1/P2 findings and finalized documentation/Javadocs without changing executable
behavior. `git diff --check` passed. Existing unrelated [Javadoc](../../../../glossary.md#term-javadoc) warnings remain.

No unresolved Task-0077 issue or required follow-up. No performance claim is made.

Status: Complete
