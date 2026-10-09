# Task 0074: Profile-free FLOAT32 EXP route

## Status

Complete — FLOAT32-only EXP qualification, 0069 source/AIR evidence rebinding, independent Class C
review/documentation, and repository integration build all passed.

## Change class

Class C — admitting a Metal numerical capability changes a native execution route, prepared
selection, compatibility identity, and backend-visible behavior.

## Goal

Admit one bounded, fixed MPSGraph route for Model `EXP` over canonical FLOAT32 tensors using the
profile-free numerical contract and ordinary route conformance tests. The registered operation
wire 55 is currently capability-false and rejected by native creation; registration alone is not
support.

## Scope

- Candidate occurrence: `UnaryElementwiseKind.EXP`, `NoOperationAttrs`, one FLOAT32 input and
  output of the same fully static positive canonical shape, rank `1..16`, without gradient roles.
  Admit only element counts, four-byte spans and allocation/dispatch geometry that pass the
  existing checked Java and native bounds; oversized or overflowing geometry remains false.
- Use one statically selected direct MPSGraph exponent route at Prepare; retain the current native
  ABI and operation wire if their encodings do not change. Java and native validate the same
  occurrence before resource creation or output mutation. A FLOAT32-only EXP after an explicit
  low-to-FLOAT32 cast may also be an internal MPSGraph boundary step inside a low-containing
  `CUSTOM_PROGRAM` partition; the whole-partition route remains custom and no low-valued EXP is
  admitted.
- Qualify ordinary finite results against `(float) StrictMath.exp((double) input)` with the
  test-only metric `abs(actual - reference) / abs(reference) <= 2e-6` when the reference is
  normal and finite. Exercise scalar-sized, non-power-of-two and larger tensors with the same
  pointwise threshold; no shape-specific relaxation. Test underflow, overflow and subnormal
  boundaries separately from NaN, infinity and signed-zero classes. For a positive subnormal
  reference, a nonzero result must be positive subnormal and within four FLOAT32 subnormal ULPs
  of that reference; `+0` is also permitted by the named arithmetic FTZ permission. When a
  finite input's wider-precision EXP rounds to FLOAT32 `+0`, accept `+0` or a positive subnormal
  no larger than four minimum-subnormal ULPs as a separate test-only underflow threshold. Neither
  underflow rule applies to exact `EXP(-infinity)=+0`. Ordinary finite tolerance cannot excuse
  infinity or NaN.
- Update exact provider answers, negative predicates, route catalog, identity/stale-input checks,
  source-matched native package, backend conformance and public Engine execution evidence. Keep
  the historical ACCELERATOR ledger immutable; update the current-provider snapshot and its test
  so only the reviewed FLOAT32 EXP row differs from the frozen false baseline, while both low
  EXP rows remain false.

## Non-goals

No SIGMOID, BFLOAT16/FLOAT16/FLOAT64 EXP, gradient-bearing EXP, new public configuration,
runtime tolerance engine, numerical certificate, proof kernel, route autotuning, fallback,
fusion, or performance claim. A failed direct-route gate leaves EXP unsupported; a custom
alternative needs a separate decision rather than a silent substitution.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants): Model
  owns the formula; finite tolerances are test-only; native identities reject stale inputs.
- [Foundational contract — Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics), especially
  [Family formulas and unary sites](../../../../architecture/contracts/foundational-modules.md#family-formulas-and-unary-sites),
  [Arithmetic-site DAZ/FTZ](../../../../architecture/contracts/foundational-modules.md#arithmetic-site-dazftz), and
  [Floating arithmetic primitives](../../../../architecture/contracts/foundational-modules.md#floating-arithmetic-primitives): EXP's mathematical, finite and special-value rules.
- [Backend contract — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) and
  [Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity): fixed qualified route, capability and compatibility boundaries.
- [Runtime/Prepare contract — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle): select the route before shared resource declarations; Runtime does not retry or choose it.

The root's current low-type parity statement is qualified by "where declared"; the owning
backend contract explicitly preserves BFLOAT16/FLOAT16 counterparts for the *frozen supported*
ACCELERATOR FLOAT32 occurrences. Metal EXP is false for all three types in that frozen baseline.
Admitting only this new FLOAT32 EXP occurrence therefore preserves that baseline and does not
claim BFLOAT16/FLOAT16 EXP support. The historical ledger's frozen true rows must remain true;
its EXP row and current-provider snapshot must distinguish the newly approved FLOAT32 delta from
the old false baseline rather than infer low support or silently rewrite historical evidence.

## Dependencies and integration

- Depends on: Model 0032 (`Complete`); Metal 0069 Slice-3 review/remediation (`Complete` at
  prior source checkpoint); Metal 0074A contract and independent review (`Complete`).
- Conflicts with: Metal 0069 remediation; a resumed 0053 production route; concurrent Metal
  capability, native source/schema, route, identity, or shared-contract edits.
- Parallel group: None.
- Common base revision: N/A for serial work; current worktree includes the completed, uncommitted
  0069 evidence renewal over `1cab0731b75c035f589a60f8b9922a767211e6c6`.
- Integration order: completed 0069 evidence renewal → clean implementation context → independent
  Class C review/documentation → integration owner closes this brief and shared documents.
- Integration validation: source-matched native package and configured real-device Metal tests,
  affected conformance/Engine/architecture/Javadoc checks, Task-0069 source/AIR evidence rebinding
  if the native foundation source changes, then one repository build because the capability
  crosses backend and testing modules.
- Shared-document integration owner: main coordination context; implementation and review
  contexts receive disjoint document scopes.
- Frontier verification (2026-10-09, renewed after 0074A): Model 0032, Metal 0069 at its
  approved source checkpoint, and Metal 0074A are Complete. The architecture writer and
  independent reviewer are finished; no concurrent Metal capability/native writer is active.
  Partial 0074 code and tests remain unaccepted, with the low-type proxy and failed Java negative
  preflight known. This brief is Ready at the sole serial Metal frontier; historical 0051/0053
  proof gates do not apply to the current profile-free contract. Integration must revalidate and
  rebind 0069 evidence after the native source stabilizes.

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalOperationRouteCatalog`,
  `MetalMpsGraphProgram`, `MetalNativeApi` — provider, lowering, fixed route and schema checks.
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` — wire-55 preflight and direct
  exponent construction, with unchanged ABI if signatures stay fixed.
- Focused Metal native/capability tests, backend conformance (including the frozen-ledger/current-
  answer distinction), public Engine integration tests, and the affected Metal guide/Javadocs/
  identity fixtures.

## Acceptance criteria

- Only the declared occurrence returns true; wrong type, rank, shape, layout, attributes,
  gradient metadata, aliasing, or stale image remains fail-closed in Java and native.
- Both input zeros yield exact `+1`; `-infinity` yields `+0`, `+infinity` remains positive infinity,
  and NaN remains NaN. Ordinary finite values meet a recorded test-only tolerance; no tolerance
  excuses a wrong class, sign, or genuine-overflow boundary.
- A real-device source-matched package executes the static MPSGraph route without skips, preserves
  inputs, and satisfies fail-before-mutation and repeated prepared-run tests. No hidden CPU repair,
  alternate route, per-element host work, runtime numerical policy, or unsupported low-type claim.
- An explicit BFLOAT16/FLOAT16-to-FLOAT32 cast followed by FLOAT32 EXP is accepted inside an
  otherwise low-containing custom partition using an internal FLOAT32 MPSGraph boundary step;
  BFLOAT16/FLOAT16-valued EXP remains provider-false and is rejected before resource mutation.
- Changed compatibility identities advance append-only where needed; old images/candidates reject.
  ABI 7 and schema 20 remain only if their actual wire formats are unchanged.
- Public Engine and backend-conformance tests establish the same support and rejection matrix;
  the historical ledger is unchanged and the current-provider snapshot records only the approved
  FLOAT32 EXP delta. Affected Javadocs, Metal guide, current capability-count explanations
  (distinguished from the frozen baseline in the owning backend contract), planning statuses, and
  independent Class C review agree with source.

## Validation

Worker: build, ad-hoc-sign, package and verify the native library from the edited source. Add
`MetalExp*` backend tests, `MetalExp*` conformance tests and an `EngineExpMetalIntegrationTest`,
then run them against the verified absolute dylib with no device skips:

```bash
./native/metal-macos-arm64/build.sh
/usr/bin/codesign --force --sign - --timestamp=none --identifier io.github.pho001.synaptik.metal.foundation native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/package-local.sh native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/verify-package.sh native/metal-macos-arm64/build/package-v1/macos-arm64
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test :testing:backend-conformance:test :testing:integration-tests:test --rerun-tasks --max-workers=1
./gradlew :backends:metal:javadoc
git diff --check
```

Check the thirteen exports and schema/stale-input rejection in the focused tests; validate
documentation links after the final edits. If native foundation changes, the integration owner
must independently revalidate and rebind Task-0069's pinned source/AIR evidence to the resulting
source before closure; do not treat its earlier digest as current. Reuse unchanged worker evidence;
run `./gradlew build --max-workers=1` once against the verified package after final executable
integration. Do not rerun it merely for a documentation-only review.

## Follow-up

- Plan SIGMOID separately after EXP qualification. The MPSGraph scalar `+0` guard's performance
  measurement remains an independent, nonblocking follow-up.

## Documentation and review impact

Implementation will change supported backend behavior, so a separate clean documentation/review
context must finalize affected Javadocs, Metal guide, examples/links and targeted glossary impact.
No architecture contract is changed by this task; any required parity-rule change needs a
coordinated root/scoped-contract decision before implementation.

## Result

FLOAT32-only provider/Prepare/native EXP, explicit low-to-FLOAT32 cast composition inside the
fixed custom partition, and identity 30→31 are complete. The frozen-baseline low proxy remains
separate from evolving FLOAT32 support. A direct-preparer regression first reproduced then fixed
an inconsistent BFLOAT16/FLOAT16 EXP admission: Prepare now uses the exact provider answer as its
sole occurrence gate. Native build/sign/package/verify, the affected Metal/conformance/Engine
device suites (454 tests, zero failures/errors; new EXP tests without skips), architecture tests,
Metal Javadoc, documentation links, and `git diff --check` passed. The full repository
`./gradlew build --max-workers=1` passed after the final executable fix. Current ledger v2 changes
only FLOAT32 EXP to true; both low rows remain false and historical v1 is unchanged. Independent
0069 source/AIR evidence rebinding and proof passed against native foundation SHA-256
`de9f80b61f81c860fab86c979625a171a6b0ce57507148051ba0ca593fe23014`. Independent Class C
code and documentation reviews ended in APPROVE with no P0/P1/P2 findings. No commit or push was
made.

Status: Complete
