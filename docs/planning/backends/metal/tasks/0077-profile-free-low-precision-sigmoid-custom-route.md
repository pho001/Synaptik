# Task 0077: BFLOAT16/FLOAT16 SIGMOID custom route

## Status

Ready — the sole serial Metal frontier verified on 2026-10-09 at clean `bcf9d3b4`:
Model 0032 and Metal 0069/0074–0076 are Complete, Task 0076 passed independent Class C review,
and no other Metal native/capability writer is active.

## Change class

Class C — independently admitting low-precision numerical capability changes native custom
execution, prepared-route and compatibility behavior, and observable results.

## Goal

Qualify homogeneous BFLOAT16 and FLOAT16 `SIGMOID` separately using typed custom kernels inside
the existing fixed low-containing `CUSTOM_PROGRAM` partition.

## Scope

- Candidate domain: canonical positive-static-rank `1..16` no-gradient unary `SIGMOID`,
  same-shape/same-type low input and output, with checked two-byte spans and native geometry.
- Each element uses its original low storage type and an exact raw-bit sign guard. Select `x` for
  negative input or `-x` for nonnegative input, evaluate one `EXP` in FLOAT32, select the numerator
  (`EXP` or typed one), then perform one FLOAT32 addition and division before exactly one final
  ties-to-even narrowing to the original low type. This is the Model's two stable branches with
  their named sites, not a direct opaque sigmoid or an algebraic substitute. No implicit
  BFLOAT16/FLOAT16 substitution or low MPSGraph step.
- Qualify ordinary positive normal finite outputs against the independent branch-correct
  `StrictMath.exp((double) widenedInput)` reference with test-only relative error
  `abs(actual-reference)/reference <= 0.0040` for BFLOAT16 and `<= 0.00055` for FLOAT16. These
  allow roughly half a low-format rounding ULP plus a small FLOAT32 working margin, and are not
  public guarantees. Exact classes and signs override tolerance. If a gate fails, report the
  concrete word/error and stop for a reviewed algorithm or threshold decision; never widen it
  silently.
- Test `SIGMOID(±0)=0.5`, `SIGMOID(-infinity)=+0`, `SIGMOID(+infinity)=1`, and NaN class
  separately. Exercise both sign branches, stored low subnormal inputs, ordinary near-zero,
  saturation, low-output subnormal/underflow and large tensors for each dtype. For a positive
  low-subnormal reference, a nonzero low result must be positive and within one target minimum-
  subnormal ULP of the independently narrowed reference. BFLOAT16 may return `+0` where its
  FLOAT32 working EXP/result site is subnormal and FTZ is allowed. FLOAT16 must retain a nonzero
  subnormal for `SIGMOID(-12)` and other cases whose FLOAT32 working value is normal and final
  narrowing is nonzero; `SIGMOID(-20)` must round to `+0`. When the high-precision reference
  rounds to low `+0`, permit one positive minimum subnormal only in the declared boundary
  neighborhood `0.4q <= reference < 0.5q`, with `q` the target minimum subnormal; below `0.4q`
  require `+0`. A finite result must remain in `[0,1]` and cannot use tolerance to excuse NaN,
  wrong infinity, negative zero or impossible saturation.
- Update native source authentication, provider/current v2 ledger, identity, conformance, Engine,
  Javadocs, the root and owning backend contract's current capability/identity statements,
  explanatory docs, and source-bound 0069 evidence if the native foundation changes. Historical
  v1 and the separate FLOAT32 route stay unchanged.

## Non-goals

No mixed-low, FLOAT64, gradient-bearing SIGMOID, direct MPSGraph low route, hidden casts,
runtime numerical policy, certificates, fusion, fallback, autotuning or performance claim.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants):
  distinct low storage, FLOAT32 working values, one final narrowing, exact guard and fixed custom
  low partition.
- [Foundational contract — Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics),
  especially [Family formulas and unary sites](../../../../architecture/contracts/foundational-modules.md#family-formulas-and-unary-sites),
  [Arithmetic-site DAZ/FTZ](../../../../architecture/contracts/foundational-modules.md#arithmetic-site-dazftz),
  [Composite inheritance](../../../../architecture/contracts/foundational-modules.md#composite-inheritance)
  and [Model low-precision contract](../../../../architecture/contracts/foundational-modules.md#model-low-precision-contract).
- [Backend contract — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  and [Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity).
- [Runtime/Prepare contract — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle).

## Dependencies and integration

- Depends on: Model 0032 and Metal 0069/0074A/0074/0075A/0075/0076 (all Complete).
- Conflicts with: concurrent Metal low kernels, native source/schema, provider, route, identity,
  shared-contract/documentation or 0069 source-bound evidence edits.
- Parallel group: None.
- Common base revision: N/A for serial work; clean implementation base `bcf9d3b4` on `main`.
- Integration order: 0076 completion → this implementation → source-matched native/device
  qualification → independent Class C review → integration closure.
- Integration validation: source-matched native build/sign/package/verify, affected Metal,
  conformance, Engine, architecture and Javadoc checks, 0069 source/AIR rebinding if the native
  foundation changes, then one full `./gradlew build --max-workers=1` after executable integration.
- Shared-document integration owner: main coordination context.
- Frontier verification (2026-10-09): all listed dependencies are Complete at `bcf9d3b4`;
  the only previously authorized Metal frontier 0076 is independently approved and pushed;
  0053 remains historical Blocked and is not a dependency. This brief is Ready at the sole serial
  Metal frontier. The current root/scoped contract permits a new separately qualified low
  occurrence through custom execution and coordinated current-capability/identity updates.

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalNativeApi`,
  `MetalPointwiseFusionPlanner` and route/identity fixtures — independently typed admission,
  preflight, fixed low route and no-fusion protection.
- `native/metal-macos-arm64/src/synaptik_low_precision_kernels.h` and
  `synaptik_metal_foundation.m` — typed stable-branch kernels, dispatch and source authentication.
- Metal, backend-conformance and Engine tests; current v2 ledger and explanatory/Javadoc updates.

## Acceptance criteria

- Each low type passes its predeclared finite and special-value gates and stays on the fixed
  custom partition route; wrong type/rank/Shape/layout/attributes/gradient/span/alias/route and
  stale image fail before mutation. No low-valued MPSGraph or F32-only route is selected.
- No extra low narrowing occurs at internal named arithmetic sites; one final conversion stores
  the declared output type. FLOAT32 route, ABI 7/schema 20 (unless actual wire changes), and
  historical ledger remain unchanged. Current v2 records both new true low answers independently.
- Source-matched real-device tests, native/package authentication, stale-identity rejection,
  backend conformance, public Engine composition, documentation/Javadoc and independent Class C
  review pass. Changed compatibility identity advances append-only beyond 33.

## Validation

Worker builds, ad-hoc signs, packages and verifies the edited native source. Run real-device
Metal native/capability tests for each type, affected conformance and public Engine tests against
the verified absolute dylib without relevant skips, plus architecture, Metal Javadoc and
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

The integration owner verifies and rebinds Task-0069 source/AIR evidence if the native foundation
changes, then runs the one full repository build against the final package. The independent
documentation/review pass reuses successful executable evidence unless it changes behavior.

## Documentation and review impact

Class C requires a separate code and documentation review. Update the current root/backend
contract capability and identity facts together, low capability guide, affected Javadocs, v2
ledger interpretation, examples/links and targeted glossary references; retain Model-owned
semantics and historical v1 evidence. No new architecture decision or ADR is intended because the
existing contract already specifies independently qualified low custom admission; stop if this
reading conflicts with source or another applicable contract.

## Result

Empty until execution.
