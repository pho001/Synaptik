# Task 0032: Profile-free numerical-semantics reset

## Status

Complete — 2026-10-08. The approved profile-free contract, coordinated executable cutover,
source-matched native package, integrated validation, independent Class C review, and this final
documentation reconciliation are present in the main worktree. Validation completion is distinct
from Git publication. Task 0032A is Complete. The frozen base,
integration ownership, original acceptance criteria, and validation packet below are retained
as the execution record.

## Change class

Class C (numerics, Compiler rewrites, public API/capability, native identity, and cross-module
contracts); this planning reconciliation changes no executable behavior.

## Goal

Remove `NumericalProfile`, both values and cold propagation. Keep exact
structural/discrete/cast/raw boundaries after route qualification. Judge finite arithmetic by
family formula, dtype, output rounding and *test-only* backend/route/family/dtype/size tolerances,
never public envelopes or runtime policy.
No interim rename to `RELAXED` or `FAST`; `ACCELERATOR` is a legacy provider-support baseline,
not a performance promise.

## Implemented decision table

| Boundary | Current rule |
|---|---|
| Structure/discrete | Exact kinds/descriptors, dtype/promotion, Shape/layout/axis/contributors, guards, classification, bounds/indices, predicates/masks, ordering, state, saved values, transfers, publication, casts, and raw movement/selection. |
| `WHERE` | Exact Boolean selects a branch, then exact declared cast if promoted; same-type selection preserves bits. Qualify both branches, promotion, zero/NaN/payloads. |
| Comparison/extrema | Stored-value guarantee and NaN/zero/tie/index rules: see exception inventory below. |
| Exact pointwise islands | Preserve family-specific exact represented-value rules for `ABS`/`NEG`/`SIGN`/`FLOOR`/`CEIL`/`RELU`. |
| Floating scatter `MUL` | All contributors once with permitted rounded trees; see exception inventory. |
| Approximate arithmetic | Retain family formula, contributor domain, guards, and specified precision/rounding. Apply only the approved arithmetic-site DAZ/FTZ table below; qualify each supported route. |
| Low precision | Distinct `BFLOAT16`/`FLOAT16` storage; no hidden low narrowing; see exact and arithmetic exceptions below. |

The approved permission applies only to *named floating arithmetic primitive sites* of the
applicable Model family formula, not to an operation's output merely because it is floating:

| Arithmetic-site dtype | Input DAZ | Result FTZ |
|---|---|---|
| `FLOAT32` | Permitted at named primitive inputs | Permitted at named primitive results |
| `BFLOAT16` | Permitted at named primitive inputs | Permitted at named primitive results |
| `FLOAT16` | Permitted at named primitive inputs, including multiplication | Permitted at named primitive results, including multiplication |
| `FLOAT64` | Forbidden | Forbidden |

DAZ maps a subnormal primitive input to same-signed zero without changing stored input bits.
FTZ may map a finite subnormal primitive result to zero subject to the existing family-specific
signed-zero contract; it grants no blanket choice of sign. Neither applies at stored comparison,
extrema/winner selection, raw movement, cast, selection, index, mask, or state sites, or at exact
unary `ABS`/`NEG`/`SIGN`/`FLOOR`/`CEIL`/`RELU` sites. Composites and generated gradients inherit
only their named arithmetic primitives' permissions. Family formulas, contributor domains,
guards, and special-value rules remain intact. Provider observations below are qualification
evidence, not the source of this approved rule.

## Numerical exceptions and stronger guarantees

An *exception* is a scoped arithmetic deviation or stronger exact guarantee, not a Java exception
or runtime registry. At cutover, Model defines allowed results; provider routes qualify against them.

| Case | Scoped rule and example |
|---|---|
| Stored-value boundary | Comparison/extrema use represented subnormals and retain NaN, signed-zero, tie/winner/index rules; declared casts and raw movement/selection stay exact. A raw move of low `0x0001` preserves its bits. |
| Low arithmetic underflow | The approved table permits DAZ/FTZ at named `BFLOAT16` and `FLOAT16` arithmetic primitives, including multiplication, but not casts or raw moves. `BFLOAT16` `0x0001 * 128` may zero through input DAZ despite exact normal `0x0080`; `FLOAT16` `0x0080` is instead subnormal and needs separate qualification. Neither example nor a provider result identifies the backend's zeroing site. |
| Scatter `MUL` | Floating base and every addressed update contribute once, but a rounded tree may yield NaN for `maxFinite * 2 * 0`; integral fixed-width modular multiplication and unaddressed raw cells remain exact. |
| Precision and tests | Qualified homogeneous `BFLOAT16`/`FLOAT16` arithmetic uses `FLOAT32` working values and accumulators without hidden low intermediate narrowing, then one final ties-to-even low narrowing per declared output. This is not a once-rounded exact-mathematical result promise. NaN/infinity, signed zero, and subnormal/underflow stay outside finite tolerance. |

Historical exploratory backend-local observations, not clean-base proof or public semantics: CPU
generated scatter yielded exact `+0` for `maxFinite * 2 * 0`; an overflow-first tree gives NaN.
Verified-package Metal `CUSTOM_PROGRAM` preserved `0x0001` in casts/raw moves for both low
types; minimum-subnormal `* 1` gave `0x0000` for `BFLOAT16`, `0x0001` for `FLOAT16`. Metal
`BFLOAT16` `0x0001 * 128` gave `0x0000`, ruling out final low-result FTZ alone; input DAZ or
earlier loss fits, but the zeroing site is unproved. Carry and revalidate these observations.

Framework comparison, nonnormative: [PyTorch accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
and [JAX accuracy](https://docs.jax.dev/en/latest/accuracy.html) discuss order/precision/underflow;
[JAX scatter](https://docs.jax.dev/en/latest/_autosummary/jax.Array.at.html) applies conflicting
updates in implementation-defined order. [MLX](https://ml-explore.github.io/mlx/build/html/usage/precision.html)
and [TensorFlow](https://www.tensorflow.org/guide/mixed_precision) describe scoped compute precision;
none establishes identical semantics or a universal standard for Synaptik's exact boundaries.

## Scope

- Inventory formula, conversions, work/accumulator dtype and final cast across unary, binary/scalar,
  reduction/scan/statistics/loss, contraction/spatial/attention/recurrent, dropout and gradients.
  Cover `MIN`/`MAX`/`CLAMP`, empty/domain/special, guards and under/overflow.
  Guarded attention's no-eligible/all-negative-infinity/positive-infinity-tie rules differ from
  literal softmax; composites gain no universal zero/subnormal promise. DAZ/FTZ remains arithmetic-
  only; no direct mixed-low route.
- Preserve Compiler's seven guarded identity rewrites, phase/descriptor/output/gradient guards and
  autograd ownership. Tolerance or a provider sample cannot license `x * 0`, `x - x`, reassociation,
  fusion or contraction across observable/special boundaries; unproved legality keeps the old path.
- Remove numerical selectors/branches/wires/trace fields/cold propagation; preserve actual CPU/
  Metal `ACCELERATOR` *true and false* provider answers, not semantic reachability. Project the
  representative, non-exhaustive 508-row v1 ledger to profile-free backend/occurrence rows with
  false answers, query provenance, independent low queries and negative predicates. Lost true rows
  block cutover; the ledger is not numerical certification.
- Retain CPU generation/OpenBLAS, training/autograd, transfers, lifetimes, tuning and eligible
  FLOAT32 MPSGraph/custom routes. Low Metal stays fixed `CUSTOM_PROGRAM`, without retry/fallback.
  Remove *numerical* certification, not source/manifest/package authentication, native application
  binary interface (ABI), decoder/preflight, bounds, ownership or compiled-code checks.

## Non-goals

- No fast-math switch, production tolerance registry/certificate substitute or proof engine,
  determinism promise, MPSGraph removal, Metal fallback, or sample-inferred capability.
- Retain tuning, caches, benchmarks, input `ProfileFingerprint`, cost evidence, non-profile
  identities, and history; singleton Metal `NEG` tuning is separate.

## Contracts

The root and owning scoped contracts now govern the completed profile-free implementation.
Architecture decision record (ADR) 0026 records the accepted choice but does not replace those
contracts. Architecture explanation and tests were reconciled in the integrated change.

- [Root core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [dependency rules](../../../../../ARCHITECTURE.md#dependency-rules): the main-worktree contract
  specifies profile-free semantics.
- [Foundational Model and profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics):
  stored-value comparison/extrema and named arithmetic-site DAZ/FTZ are separate boundaries.
  Also review [profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries).
- [Compiler compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)
  and [profile-free compile identity](../../../../architecture/contracts/compiler-autograd.md#profile-free-compile-identity):
  retain optimization/gradient ownership.
- [Profile-free Engine and Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#profile-free-engine-and-prepare-lifecycle):
  remove cold transport, not Runtime's policy-free boundary.
- [Backend low-precision capability evidence](../../../../architecture/contracts/backend-execution.md#low-precision-capability-evidence):
  retain ledger/routes/source/ABI safety; review [profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity) and [CPU routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes).
- [Training ownership](../../../../architecture/contracts/extensions-training.md#extensionstraining): retain Engine lifecycle and optimizer ownership.

## Dependencies and integration

- Depends on: 0032A (Complete). The user approved the named arithmetic-site DAZ/FTZ rule on
  2026-10-08, closing the former non-task decision gate. Completed 0030/0031 and the current
  frozen-HEAD profile spine remain the clean-base context.
- Conflicts with: concurrent edits to numerical authority, compiler rewrite legality, provider
  predicates, trace DTOs, Metal schema/native source, CPU generated identity, and shared guides.
- Parallel group: None.
- Common base revision: `9fbd0cc36f2f187d1a658671cb55710ff13a5ed7`, the historical
  clean source/test base used for isolated implementation and validation before transfer to main.
- Integration order: authority-first revision and integration-owner acceptance → isolated
  code/test/identity cutover → independent targeted Class C review → isolated full validation →
  byte-identical transfer and source-matched native rebuild in main → final documentation
  reconciliation. Git publication is a separate step from this validation record.
- Integration validation: the architecture, backend-conformance, integration, Javadoc, native,
  and repository build gates recorded under Validation and Result.
- Shared-document integration owner: Codex main coordination context. It assigned exclusive
  contract and final documentation scopes, preserved user dirty edits, and owns ordered
  integration; workers had no overlapping writes.

## Files and symbols

- `modules/model`, `modules/config`, `modules/planning`, `modules/compiler`, `modules/prepare`,
  `modules/engine`: semantics/Javadocs, `NumericalProfile`, `OperationCapabilityQuery`,
  `CompileArtifacts`, `PrepareContext`, Config and `Engine.Builder.numericalProfile(NumericalProfile)`
  removal. Runtime is already profile-free; no removal there.
- `modules/trace`, `backends/cpu`, `backends/metal`, `native/metal-macos-arm64`: DTOs, CPU codegen/OpenBLAS/
  tuning/cache, Metal plans/schema/decoder, authentication, ABI/package gates.
- `extensions/training`, `testing/architecture-tests`, `testing/backend-conformance`, `testing/integration-tests`: optimizer and cross-module tests.

## Historical readiness evidence and limits

The initial 2026-10-08 handoff reported these results from the frozen pre-cutover HEAD;
this final documentation pass did not rerun Java or native tests. `LowPrecisionCapabilityLedgerTest`
passed 6/6. The then-checked-in
representative accelerator rows give CPU `FLOAT32` 122 true/5 false and Metal `FLOAT32` 98
true/29 false. Independently queried `BFLOAT16` and `FLOAT16` columns each give CPU 114 true/5
false and Metal 90 true/29 false, with eight not-applicable controls per backend and low type.
These are provider answers for named occurrences, not an enumerated support domain or numerical
proof. Preserve *both* true and false answers and the query provenance at cutover.

A pre-cutover source-matched native build, fixed-identifier ad-hoc signing, local packaging, and
independent verification passed from that HEAD: ABI 7, schema 19, exactly thirteen exports, and packaged
dylib SHA-256 `30e899d1f663b64b217087073d54077fbcd2e3fae11cd5abc7c2508b6f1bb294`.
Configured real-device `MetalStoredSubnormalProviderNativeTest` passed 2/2 without skips.
Existing 0032A CPU generated FLOAT32 scalar/vector and Metal FLOAT32 `CUSTOM_PROGRAM` witnesses
were route-local. Earlier main-worktree dirty CPU scatter and expanded Metal low tests were not
in that clean base; the integration preserved user edits and carried the intended assertions
through cutover. These historical observations did not, by themselves, prove all-route behavior
or the device's precise low-word zeroing site.

At launch, 0032A, 0030, and 0031 were Complete; the approved table closed the sole recorded
decision gate. This was the one authorized Model frontier, with no declared conflicting write
task. The root/scoped revision was reviewed before code. The completed main-worktree cutover
uses CPU schema 69, Metal schema 20/header 124 and identity 30, and native ABI 7. The earlier
schema-19 certificate is historical and does not certify schema 20; source/package authentication
remains a current safety gate. ADR 0026 is Accepted.

## Historical implementation sequence and checkpoints

The following was the original execution sequence. The Result records its closure; these steps
are not outstanding instructions or a claim of remote publication.

1. **Authority and pre-implementation inventory, integration owner and clean agents:** confirm
   this HEAD and current contract set remain stable; enumerate cold fields, native wires, CPU generated/tuning/
   OpenBLAS and Metal program/tuning/pointwise/manifest identities, ledger/Trace/benchmark fields,
   unqualified routes, and exact/special/finite test cases. Recheck the provider baseline if the
   base changes. The architecture-documentation agent's root and owning scoped-contract revision
   was reviewed and accepted by the integration owner before implementation.
   Current 0032A witnesses do not certify other routes or low arithmetic.
2. **Cutover, coordinated clean implementation context:** land approved Model-owned family semantics,
   Javadocs and guarded Compiler rewrites with counterexamples; remove cold numerical-profile
   transport through Config/Planning/Compiler/Prepare/Engine/Trace and CPU/Metal/native identities,
   schema and routes together. Runtime stays profile-free. Preserve kernels, MPSGraph, transfers,
   Training, safety verifiers and fail-closed unsupported rows. Version changed identities without
   preselected numbers; reject stale inputs/cache hits. Drop only numerical-profile bytes; retain
   distinct low keys, class/checksum checks and other wires. Remove Metal's profile wire atomically
   in Java/native; keep ABI if all thirteen exports are unchanged, otherwise review a bump. The
   integration owner transfers only the complete runnable cutover.
3. **Semantic/safety checkpoint, implementer then integration owner:** qualify supported routes
   with exact structure/discrete/cast/raw and stored-value comparison (subnormal versus `+0` both
   orders, first/last arg ties), scatter contributors/specials/underflow, low work/accumulator and
   final-rounding tests, plus test-only backend/route/family/dtype/size finite tolerances with an
   explicit oracle, metric, threshold and cancellation boundary. Keep NaN/Inf, signed zero,
   empty/domain and subnormal cases outside
   finite tolerance. Prove provider parity, negative predicates, stale identity/schema rejection,
   native ABI/source authentication and fail-before-mutation; run configured real-device Metal.
4. **Final checkpoint, integration owner and independent reviewer:** reconcile accepted authority,
   ADR, architecture explanation, Javadocs, user guide, glossary and shared plans; complete targeted
   Class C review. Reuse successful affected-module tests, run cross-module gates after integration
   and repository-wide validation once. Record limitations; mark `Complete` only after acceptance,
   review and validation pass.

## Acceptance criteria

- Authority, Javadocs and tests agree on the decision table/exception inventory. Every supported
  route qualifies the approved family-by-dtype arithmetic-site DAZ/FTZ table, exact/stored-value,
  scatter special/integral/unaddressed, low underflow/precision, and separate finite-tolerance
  behavior; no production tolerance engine.
- No numerical selector or ignored adapter remains in API, cold artifacts, Trace, backend/native
  plans or active docs. Retained tuning identities remain; CPU/Metal true/false answers match the
  baseline, and independent low queries, negative predicates and stale rejection pass.
- Compiler's seven guarded rewrites preserve special, rounding and saved/state counterexamples;
  tolerance never proves legality. Changed generated algorithms/hot paths alone require optimal-Java
  algorithm/loop/dataflow, bytecode/hidden-overhead and generated-versus-direct performance evidence.
- Structural/source/ABI/route/fail-before-mutation, CPU/Metal, Training/autograd, transfer,
  lifecycle and native-failure gates pass. Remove only numerical certificates/complete-domain
  result proofs; publish authority, ADR, guide, glossary and independent Class C review.

## Validation

The following commands were the recorded implementation and integration validation plan from
repository root. The final documentation pass reused the successful reported executable evidence
and did not rerun Java suites. The Result gives observed counts and skip classifications.

```bash
./gradlew :modules:model:test --tests '*AxisScatterSemanticsTest' --tests '*ScatterNdSemanticsTest'
./gradlew :backends:cpu:test --tests '*CpuStoredSubnormalProviderRegressionTest' --tests '*CpuScatterGeneratedKernelTest'
./gradlew :testing:backend-conformance:test --tests '*LowPrecisionCapabilityLedgerTest'
./gradlew :modules:model:test :modules:config:test :modules:planning:test :modules:compiler:test :modules:prepare:test :modules:trace:test :modules:engine:test :backends:cpu:test :extensions:training:test
```

After native source changes, rebuild from that source, sign, package, and verify before any
configured Metal run. Use the verified package path; check manifest ABI/schema, thirteen export
names/signatures, source match, hash, stale-schema rejection, and no-skips device evidence:

```bash
./native/metal-macos-arm64/build.sh
/usr/bin/codesign --force --sign - --timestamp=none --identifier io.github.pho001.synaptik.metal.foundation native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/package-local.sh native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/verify-package.sh native/metal-macos-arm64/build/package-v1/macos-arm64
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test --tests '*MetalStoredSubnormalProviderNativeTest' --rerun-tasks
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test --rerun-tasks
```

The integration owner ran the post-cutover architecture, backend-conformance, integration,
Javadoc, native, and full-build gates with the verified package configured for Metal where
required. Device-test skips would not count as a pass. The command list remains as the planned
validation recipe; see Result for the completed evidence:

```bash
./gradlew :testing:architecture-tests:test
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :testing:backend-conformance:test :testing:integration-tests:test --rerun-tasks
./gradlew :modules:model:javadoc :modules:config:javadoc :modules:planning:javadoc :modules:compiler:javadoc :modules:prepare:javadoc :modules:trace:javadoc :modules:engine:javadoc :backends:cpu:javadoc :backends:metal:javadoc :extensions:training:javadoc
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew build
git diff --check
```

Audit active profile references, provider negative predicates, identity/cache stale-input tests,
and reported skips. Add generated-bytecode/oracle and generated-versus-direct performance evidence
only if a generated hot-path algorithm changes; no default five-fork matrix.

## Documentation and review impact

This brief records the authority-first revision, isolated executable cutover, transfer to main,
and final reconciliation. The independent documentation pass covered architecture explanation,
active API/backend guides, trace, glossary, examples, and links; executable and Javadoc changes
remained owned by implementation contexts. Final Class C review and integration validation passed.
This atomic cross-module brief exceeds the planning guide's
200-line/15 KB targets to keep its decision exceptions, provider baseline, safety gates, and
execution packet together; splitting those contracts would risk a runnable half-cutover.

## Result

- Completed the Model-owned profile-free semantics and arithmetic-site DAZ/FTZ cutover, seven
  guarded Compiler identities, cold-selector removal, CPU/Metal/native identities, qualified
  routes, and coordinated Javadocs, guides, glossary, and architecture/conformance/integration
  tests. No generated hot-path algorithm changed.
- Source/tests/docs were transferred byte-identically from the validated isolated worktree into
  main before this final documentation reconciliation; existing user dirty edits were preserved.
  Current CPU generator schema 69, Metal schema 20
  (124-byte header), Metal identity 30, and native ABI 7 with thirteen exports reject stale
  pre-cutover inputs. The 254-row v2 capability ledger retains actual historical ACCELERATOR
  true **and** false answers, independent BFLOAT16/FLOAT16 queries, and query provenance; the
  508-row v1 basis is historical evidence, not current numerical certification.
- Main native package was rebuilt from matching source, ad-hoc signed and verified: SHA-256
  `77f683d4331a688423a5a5e14cb462e3d9bafdf1aaab3ae5963c20dfadb3d318`.
  Focused real-device tests passed 7/7 with zero skips.
- Isolated `./gradlew build` with the verified `SYNAPTIK_METAL_TEST_LIBRARY` setting shown above
  ran 3,853 total tests, zero failures, 39 explained skips. Architecture: 9 total, 0 failures,
  0 skips; CPU: 1,044 total, 0 failures,
  28 skips; Metal: 325 total, 0 failures, 10 observer skips; backend conformance: 30 total,
  0 failures, 0 skips; integration: 89 total, 0 failures, one CPU autotune skip, including
  60 Metal integration tests with zero skips. Javadocs passed with 46
  preexisting warnings. These are reused implementation/integration results, not rerun here.
- Independent Class C reviews found P1 OpenBLAS runtime tolerance, fixed by exact ABI smoke V2,
  and P2 stale documentation, corrected before this final reconciliation. No P0/P1 remains.
  The MPSGraph scalar `+0` guard is retained. Its performance effect is unmeasured and is a
  nonblocking follow-up, not missing acceptance evidence.
- Documentation review checked the relevant glossary term/anchor; no new term or glossary edit
  was needed. A read-only Ruby local-link/anchor check passed for all nine final documents;
  `git diff --check` passed, and targeted trailing-whitespace search found none. This records
  pre-publication validation; subsequent commit and push status belongs to Git history.

Status: Complete
