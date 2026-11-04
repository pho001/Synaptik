# Task 0074: Profile-free [FLOAT32](../../../../glossary.md#term-float32) [EXP](../../../../glossary.md#term-expx) route

## Status

Complete — [FLOAT32](../../../../glossary.md#term-float32)-only [EXP](../../../../glossary.md#term-expx) [qualification](../../../../glossary.md#term-qualification), 0069 source/[AIR](../../../../glossary.md#term-air) evidence rebinding, independent Class C
review/documentation, and repository integration build all passed.

## Change class

Class C — admitting a [Metal](../../../../glossary.md#term-metal) numerical capability changes a native execution route, prepared
selection, compatibility [identity](../../../../glossary.md#term-identity-meanings), and [backend](../../../../glossary.md#backend)-visible behavior.

## Goal

Admit one bounded, fixed [MPSGraph](../../../../glossary.md#term-mpsgraph) route for [Model](../../../../glossary.md#term-model) [`EXP`](../../../../glossary.md#term-expx) over [canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) [tensors](../../../../glossary.md#tensor) using the
profile-free numerical contract and ordinary route [conformance](../../../../glossary.md#term-conformance) tests. The registered [operation](../../../../glossary.md#operation)
wire 55 is currently capability-false and rejected by native creation; registration alone is not
support.

## [Scope](../../../../glossary.md#term-scope)

- Candidate [occurrence](../../../../glossary.md#term-occurrence): `UnaryElementwiseKind.EXP`, [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), one FLOAT32 input and
  output of the same fully static positive canonical shape, rank <span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [canonical glossary](../../../../glossary.md#term-canonical-form); [shape glossary](../../../../glossary.md#shape); [rank glossary](../../../../glossary.md#term-rank))</span>`1..16`, without gradient roles.
  Admit only element counts, four-byte spans and allocation/dispatch geometry that pass the
  existing checked Java and native bounds; oversized or overflowing geometry remains false.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [element glossary](../../../../glossary.md#term-element); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Use one statically selected direct MPSGraph exponent route at Prepare; retain the current native
  ABI and operation wire if their encodings do not change. Java and native validate the same
  occurrence before resource creation or output mutation. A FLOAT32-only EXP after an explicit
  low-to-FLOAT32 cast may also be an internal MPSGraph boundary step inside a low-containing<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [exponent glossary](../../../../glossary.md#term-exponent); [Prepare glossary](../../../../glossary.md#prepare); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [operation glossary](../../../../glossary.md#operation); [EXP glossary](../../../../glossary.md#term-expx))</span>
  `CUSTOM_PROGRAM` partition; the whole-partition route remains custom and no low-valued EXP is
  admitted.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>
- Qualify ordinary [finite](../../../../glossary.md#term-finite) results against `(float) StrictMath.exp((double) input)` with the
  test-only metric <span data-glossary-navigation="true"> ([metric glossary](../../../../glossary.md#term-metric))</span>`abs(actual - reference) / abs(reference) <= 2e-6` when the reference is
  normal and finite. Exercise scalar-sized, non-power-of-two and larger tensors with the same
  pointwise threshold; no shape-specific relaxation. Test underflow, overflow and subnormal
  boundaries separately from NaN, infinity and signed-zero classes. For a positive subnormal
  reference, a nonzero result must be positive subnormal and within four FLOAT32 subnormal ULPs
  of that reference; <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [tensors glossary](../../../../glossary.md#tensor); [pointwise glossary](../../../../glossary.md#term-pointwise); [underflow glossary](../../../../glossary.md#term-underflow); [overflow glossary](../../../../glossary.md#term-overflow); [subnormal glossary](../../../../glossary.md#term-subnormal); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`+0` is also permitted by the named arithmetic FTZ permission. When a
  finite input's wider-precision EXP rounds to FLOAT32 <span data-glossary-navigation="true"> ([FTZ glossary](../../../../glossary.md#term-ftz); [precision glossary](../../../../glossary.md#term-precision))</span>`+0`, accept `+0` or a positive subnormal
  no larger than four minimum-subnormal ULPs as a separate test-only underflow threshold. Neither
  underflow rule applies to exact `EXP(-infinity)=+0`. Ordinary finite tolerance cannot excuse
  infinity or NaN.<span data-glossary-navigation="true"> ([tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- Update exact provider answers, negative predicates, route catalog, identity/stale-input checks,
  source-matched native package, backend conformance and public Engine execution evidence. Keep
  the historical ACCELERATOR ledger immutable; update the current-provider snapshot and its test
  so only the reviewed FLOAT32 EXP row differs from the frozen false baseline, while both low
  EXP rows remain false.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [ACCELERATOR glossary](../../../../glossary.md#term-accelerator); [immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>

## Non-goals

No [SIGMOID](../../../../glossary.md#term-sigmoid), [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16)/[FLOAT64](../../../../glossary.md#term-float64) [EXP](../../../../glossary.md#term-expx), [gradient](../../../../glossary.md#term-gradient)-bearing EXP, new public configuration,
[runtime](../../../../glossary.md#term-runtime) [tolerance](../../../../glossary.md#term-tolerance) [engine](../../../../glossary.md#term-engine), numerical [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions), proof [kernel](../../../../glossary.md#kernel), route [autotuning](../../../../glossary.md#autotuning--model-autotuning), fallback,
[fusion](../../../../glossary.md#term-fusion), or performance claim. A failed direct-route gate leaves EXP unsupported; a custom
alternative needs a separate decision rather than a silent substitution.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants): Model
  owns the formula; finite tolerances are test-only; native identities reject stale inputs.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Foundational contract — Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([Profile-free numerical semantics glossary](../../../../glossary.md#profile-free-numerical-semantics))</span>, especially
  [Family formulas and unary sites](../../../../architecture/contracts/foundational-modules.md#family-formulas-and-unary-sites)<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>,
  [Arithmetic-site DAZ/FTZ](../../../../architecture/contracts/foundational-modules.md#arithmetic-site-dazftz)<span data-glossary-navigation="true"> ([DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz))</span>, and
  [Floating arithmetic primitives](../../../../architecture/contracts/foundational-modules.md#floating-arithmetic-primitives): [EXP](../../../../glossary.md#term-expx)'s mathematical, [finite](../../../../glossary.md#term-finite) and special-value rules.
- [Backend contract — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) and
  [Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity)<span data-glossary-navigation="true"> ([backend capability glossary](../../../../glossary.md#backend-capability); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>: fixed qualified route, capability and compatibility boundaries.
- [Runtime/Prepare contract — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>: select the route before shared resource declarations; [Runtime](../../../../glossary.md#term-runtime) does not retry or choose it.

The root's current low-type parity statement is qualified by "where declared"; the owning
[backend](../../../../glossary.md#backend) contract explicitly preserves [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) counterparts for the *frozen supported*
[ACCELERATOR](../../../../glossary.md#term-accelerator) [FLOAT32](../../../../glossary.md#term-float32) occurrences. [Metal](../../../../glossary.md#term-metal) EXP is false for all three types in that frozen baseline.
Admitting only this new FLOAT32 EXP [occurrence](../../../../glossary.md#term-occurrence) therefore preserves that baseline and does not
claim BFLOAT16/FLOAT16 EXP support. The historical ledger's frozen true rows must remain true;
its EXP row and current-provider [snapshot](../../../../glossary.md#term-snapshot) must distinguish the newly approved FLOAT32 delta from
the old false baseline rather than infer low support or silently rewrite historical evidence.

## Dependencies and integration

- Depends on: [Model](../../../../glossary.md#term-model) 0032 (`Complete`); [Metal](../../../../glossary.md#term-metal) 0069 [Slice](../../../../glossary.md#slice)-3 review/remediation (`Complete` at
  prior source checkpoint); Metal 0074A contract and independent review (<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>`Complete`).
- Conflicts with: Metal 0069 remediation; a resumed 0053 production route; concurrent Metal
  capability, native source/schema, route, identity, or shared-contract edits.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Parallel group: None.
- Common base revision: N/A for serial work; current worktree includes the completed, uncommitted
  0069 evidence renewal over `1cab0731b75c035f589a60f8b9922a767211e6c6`.
- Integration order: completed 0069 evidence renewal → clean implementation context → independent
  Class C review/documentation → integration owner closes this brief and shared documents.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Integration validation: source-matched native package and configured real-device Metal tests,
  affected conformance/Engine/architecture/Javadoc checks, Task-0069 source/AIR evidence rebinding
  if the native foundation source changes, then one repository build because the capability
  crosses backend and testing modules.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [device glossary](../../../../glossary.md#term-device); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [Javadoc glossary](../../../../glossary.md#term-javadoc); [AIR glossary](../../../../glossary.md#term-air); [backend glossary](../../../../glossary.md#backend); [modules glossary](../../../../glossary.md#term-module))</span>
- Shared-document integration owner: main coordination context; implementation and review
  contexts receive disjoint document scopes.
- Frontier verification (2026-10-09, renewed after 0074A): Model 0032, Metal 0069 at its
  approved source checkpoint, and Metal 0074A are Complete. The architecture writer and
  independent reviewer are finished; no concurrent Metal capability/native writer is active.
  Partial 0074 code and tests remain unaccepted, with the low-type proxy and failed Java negative
  preflight known. This brief is Ready at the sole serial Metal frontier; historical 0051/0053
  proof gates do not apply to the current profile-free contract. Integration must revalidate and
  rebind 0069 evidence after the native source stabilizes.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [Ready glossary](../../../../glossary.md#term-ready); [rebind glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction))</span>

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalOperationRouteCatalog`,
  `MetalMpsGraphProgram`, `MetalNativeApi` — provider, [lowering](../../../../glossary.md#lowering), fixed route and [schema](../../../../glossary.md#term-schema) checks.
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` — wire-55 preflight and direct
  exponent construction, with unchanged ABI if signatures stay fixed.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [exponent glossary](../../../../glossary.md#term-exponent); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Focused Metal native/capability tests, backend conformance (including the frozen-ledger/current-
  answer distinction), public Engine integration tests, and the affected Metal guide/Javadocs/
  identity fixtures.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Acceptance criteria

- Only the declared occurrence returns true; wrong type, rank, shape, layout, attributes,
  gradient metadata, aliasing, or stale image remains fail-closed in Java and native.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [rank glossary](../../../../glossary.md#term-rank); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata); [aliasing glossary](../../../../glossary.md#term-aliasing); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Both input zeros yield exact `+1`; `-infinity` yields `+0`, `+infinity` remains positive infinity,
  and NaN remains NaN. Ordinary finite values meet a recorded test-only tolerance; no tolerance
  excuses a wrong class, sign, or genuine-overflow boundary.<span data-glossary-navigation="true"> ([positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [finite glossary](../../../../glossary.md#term-finite); [tolerance glossary](../../../../glossary.md#term-tolerance); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- A real-device source-matched package executes the static MPSGraph route without skips, preserves
  inputs, and satisfies fail-before-mutation and repeated prepared-run tests. No hidden CPU repair,
  alternate route, per-element host work, runtime numerical policy, or unsupported low-type claim.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [run glossary](../../../../glossary.md#term-run); [CPU glossary](../../../../glossary.md#term-cpu); [element glossary](../../../../glossary.md#term-element); [host glossary](../../../../glossary.md#term-host); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- An explicit BFLOAT16/FLOAT16-to-FLOAT32 cast followed by FLOAT32 EXP is accepted inside an
  otherwise low-containing custom partition using an internal FLOAT32 MPSGraph boundary step;
  BFLOAT16/FLOAT16-valued EXP remains provider-false and is rejected before resource mutation.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [EXP glossary](../../../../glossary.md#term-expx); [partition glossary](../../../../glossary.md#partition))</span>
- Changed compatibility identities advance append-only where needed; old images/candidates reject.
  ABI 7 and schema 20 remain only if their actual wire formats are unchanged.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema))</span>
- Public Engine and backend-conformance tests establish the same support and rejection matrix;
  the historical ledger is unchanged and the current-provider snapshot records only the approved
  FLOAT32 EXP delta. Affected Javadocs, Metal guide, current capability-count explanations
  (distinguished from the frozen baseline in the owning backend contract), planning statuses, and
  independent Class C review agree with source.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [snapshot glossary](../../../../glossary.md#term-snapshot); [Metal glossary](../../../../glossary.md#term-metal); [planning glossary](../../../../glossary.md#term-planning))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker: build, ad-hoc-sign, package and verify the [native library](../../../../glossary.md#term-native-library) from the edited source. Add
`MetalExp*` [backend](../../../../glossary.md#backend) tests, `MetalExp*` [conformance](../../../../glossary.md#term-conformance) tests and an `EngineExpMetalIntegrationTest`,
then [run](../../../../glossary.md#term-run) them against the verified absolute [dylib](../../../../glossary.md#term-dylib) with no [device](../../../../glossary.md#term-device) skips:

```bash
./native/metal-macos-arm64/build.sh
/usr/bin/codesign --force --sign - --timestamp=none --identifier io.github.pho001.synaptik.metal.foundation native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/package-local.sh native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/verify-package.sh native/metal-macos-arm64/build/package-v1/macos-arm64
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test :testing:backend-conformance:test :testing:integration-tests:test --rerun-tasks --max-workers=1
./gradlew :backends:metal:javadoc
git diff --check
```

Check the thirteen exports and [schema](../../../../glossary.md#term-schema)/stale-input rejection in the focused tests; validate
documentation links after the final edits. If native foundation changes, the integration [owner](../../../../glossary.md#term-owner)
must independently revalidate and [rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) Task-0069's pinned source/[AIR](../../../../glossary.md#term-air) evidence to the resulting
source before closure; do not treat its earlier [digest](../../../../glossary.md#term-digest) as current. Reuse unchanged worker evidence;
run `./gradlew build --max-workers=1` once against the verified package after final executable
integration. Do not rerun it merely for a documentation-only review.

## Follow-up

- Plan [SIGMOID](../../../../glossary.md#term-sigmoid) separately after [EXP](../../../../glossary.md#term-expx) [qualification](../../../../glossary.md#term-qualification). The [MPSGraph](../../../../glossary.md#term-mpsgraph) [scalar](../../../../glossary.md#term-scalar) `+0` guard's performance
  measurement remains an independent, nonblocking follow-up.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard))</span>

## Documentation and review impact

Implementation will change supported [backend](../../../../glossary.md#backend) behavior, so a separate clean documentation/review
context must finalize affected Javadocs, [Metal](../../../../glossary.md#term-metal) guide, examples/links and targeted glossary impact.
No [architecture contract](../../../../glossary.md#architecture-contract) is changed by this task; any required parity-rule change needs a
coordinated root/scoped-contract decision before implementation.

## Result

[FLOAT32](../../../../glossary.md#term-float32)-only provider/[Prepare](../../../../glossary.md#prepare)/native [EXP](../../../../glossary.md#term-expx), explicit low-to-FLOAT32 cast composition inside the
fixed custom [partition](../../../../glossary.md#partition), and [identity](../../../../glossary.md#term-identity-meanings) 30→31 are complete. The frozen-baseline low proxy remains
separate from evolving FLOAT32 support. A direct-preparer [regression](../../../../glossary.md#term-regression) first reproduced then fixed
an inconsistent [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) EXP [admission](../../../../glossary.md#term-admission): Prepare now uses the exact provider answer as its
sole [occurrence](../../../../glossary.md#term-occurrence) gate. Native build/sign/package/verify, the affected [Metal](../../../../glossary.md#term-metal)/[conformance](../../../../glossary.md#term-conformance)/[Engine](../../../../glossary.md#term-engine)
[device](../../../../glossary.md#term-device) suites (454 tests, zero failures/errors; new EXP tests without skips), architecture tests,
Metal [Javadoc](../../../../glossary.md#term-javadoc), documentation links, and `git diff --check` passed. The full repository
`./gradlew build --max-workers=1` passed after the final executable fix. Current ledger v2 changes
only FLOAT32 EXP to true; both low rows remain false and historical v1 is unchanged. Independent
0069 source/[AIR](../../../../glossary.md#term-air) evidence rebinding and proof passed against native foundation [SHA-256](../../../../glossary.md#term-sha-256)
`de9f80b61f81c860fab86c979625a171a6b0ce57507148051ba0ca593fe23014`. Independent Class C
code and documentation reviews ended in APPROVE with no P0/P1/P2 findings. No [commit](../../../../glossary.md#term-commit) or push was
made.

Status: Complete
