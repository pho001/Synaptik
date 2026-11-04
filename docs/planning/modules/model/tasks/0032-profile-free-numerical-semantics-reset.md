# Task 0032: Profile-free numerical-semantics reset

## Status

Complete — 2026-10-08. The approved profile-free contract, coordinated executable cutover,
source-matched native package, integrated [validation](../../../../glossary.md#term-validation), independent Class C review, and this final
documentation reconciliation are present in the main worktree. Validation completion is distinct
from Git [publication](../../../../glossary.md#term-publication). Task 0032A is Complete. The frozen base,
integration [ownership](../../../../glossary.md#term-ownership-meanings), original acceptance criteria, and validation packet below are retained
as the execution record.

## Change class

Class C (numerics, [Compiler](../../../../glossary.md#term-compiler) rewrites, public [API](../../../../glossary.md#term-application-programming-interface-api)/capability, native [identity](../../../../glossary.md#term-identity-meanings), and cross-[module](../../../../glossary.md#term-module)
contracts); this [planning](../../../../glossary.md#term-planning) reconciliation changes no executable behavior.

## Goal

Remove `NumericalProfile`, both values and cold propagation. Keep exact
structural/discrete/cast/raw boundaries after route [qualification](../../../../glossary.md#term-qualification). Judge [finite](../../../../glossary.md#term-finite) arithmetic by
family formula, [dtype](../../../../glossary.md#data-type--datatype), output [rounding](../../../../glossary.md#term-rounding) and *test-only* [backend](../../../../glossary.md#backend)/route/family/dtype/size tolerances,
never public envelopes or [runtime](../../../../glossary.md#term-runtime) policy.
No interim rename to `RELAXED` or `FAST`; [`ACCELERATOR`](../../../../glossary.md#term-accelerator) is a legacy provider-support baseline,
not a performance promise.

## Implemented decision table

| Boundary | Current rule |
|---|---|
| Structure/discrete | Exact kinds/[descriptors](../../../../glossary.md#term-descriptor), [dtype](../../../../glossary.md#data-type--datatype)/promotion, [Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout)/[axis](../../../../glossary.md#term-axis)/contributors, guards, classification, bounds/indices, predicates/[masks](../../../../glossary.md#term-boolean-mask), ordering, state, saved values, transfers, [publication](../../../../glossary.md#term-publication), casts, and raw movement/selection. |
| `WHERE` | Exact [Boolean](../../../../glossary.md#data-type--datatype) selects a branch, then exact declared cast if promoted; same-type selection preserves bits. Qualify both branches, promotion, zero/[NaN](../../../../glossary.md#term-nan)/payloads. |
| Comparison/[extrema](../../../../glossary.md#saturation-pole-and-extremum) | Stored-value guarantee and NaN/zero/tie/index rules: see exception inventory below. |
| Exact [pointwise](../../../../glossary.md#term-pointwise) islands | Preserve family-specific exact represented-value rules for `ABS`/`NEG`/`SIGN`/`FLOOR`/`CEIL`/`RELU`. |
| Floating scatter `MUL` | All contributors once with permitted rounded trees; see exception inventory. |
| Approximate arithmetic | Retain family formula, contributor domain, guards, and specified [precision](../../../../glossary.md#term-precision)/[rounding](../../../../glossary.md#term-rounding). Apply only the approved arithmetic-site [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) table below; qualify each supported route. |
| Low precision | Distinct [`BFLOAT16`](../../../../glossary.md#term-bfloat16)/[`FLOAT16`](../../../../glossary.md#term-float16) storage; no hidden low [narrowing](../../../../glossary.md#term-narrowing); see exact and arithmetic exceptions below. |

The approved permission applies only to *named floating arithmetic primitive sites* of the
applicable [Model](../../../../glossary.md#term-model) family formula, not to an [operation](../../../../glossary.md#operation)'s output merely because it is floating:

| Arithmetic-site dtype | Input DAZ | Result FTZ |
|---|---|---|
| [`FLOAT32`](../../../../glossary.md#term-float32) | Permitted at named primitive inputs | Permitted at named primitive results |
| `BFLOAT16` | Permitted at named primitive inputs | Permitted at named primitive results |
| `FLOAT16` | Permitted at named primitive inputs, including multiplication | Permitted at named primitive results, including multiplication |
| [`FLOAT64`](../../../../glossary.md#term-float64) | Forbidden | Forbidden |

DAZ maps a [subnormal](../../../../glossary.md#term-subnormal) primitive input to same-[signed zero](../../../../glossary.md#term-signed-zero) without changing stored input bits.
FTZ may map a [finite](../../../../glossary.md#term-finite) subnormal primitive result to zero subject to the existing family-specific
[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) contract; it grants no blanket choice of sign. Neither applies at stored comparison,
extrema/winner selection, raw movement, cast, selection, index, mask, or state sites, or at exact
[unary](../../../../glossary.md#term-unary) `ABS`/`NEG`/`SIGN`/`FLOOR`/`CEIL`/`RELU` sites. Composites and generated [gradients](../../../../glossary.md#term-gradient) inherit
only their named arithmetic primitives' permissions. Family formulas, contributor domains,
guards, and special-value rules remain intact. Provider observations below are [qualification](../../../../glossary.md#term-qualification)
evidence, not the source of this approved rule.

## Numerical exceptions and stronger guarantees

An *exception* is a scoped arithmetic deviation or stronger exact guarantee, not a Java exception
or [runtime](../../../../glossary.md#term-runtime) [registry](../../../../glossary.md#term-registry). At cutover, [Model](../../../../glossary.md#term-model) defines allowed results; provider routes qualify against them.

| Case | Scoped rule and example |
|---|---|
| Stored-value boundary | Comparison/[extrema](../../../../glossary.md#saturation-pole-and-extremum) use represented [subnormals](../../../../glossary.md#term-subnormal) and retain [NaN](../../../../glossary.md#term-nan), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), tie/winner/index rules; declared casts and raw movement/selection stay exact. A raw move of low `0x0001` preserves its bits. |
| Low arithmetic [underflow](../../../../glossary.md#term-underflow) | The approved table permits [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) at named [`BFLOAT16`](../../../../glossary.md#term-bfloat16) and [`FLOAT16`](../../../../glossary.md#term-float16) arithmetic primitives, including multiplication, but not casts or raw moves. `BFLOAT16` `0x0001 * 128` may zero through input DAZ despite exact normal `0x0080`; `FLOAT16` `0x0080` is instead subnormal and needs separate [qualification](../../../../glossary.md#term-qualification). Neither example nor a provider result identifies the [backend](../../../../glossary.md#backend)'s zeroing site. |
| Scatter `MUL` | Floating base and every addressed update contribute once, but a rounded tree may yield NaN for `maxFinite * 2 * 0`; integral fixed-width modular multiplication and unaddressed raw cells remain exact. |
| [Precision](../../../../glossary.md#term-precision) and tests | Qualified [homogeneous](../../../../glossary.md#term-homogeneous) `BFLOAT16`/`FLOAT16` arithmetic uses [`FLOAT32`](../../../../glossary.md#term-float32) working values and accumulators without hidden low intermediate [narrowing](../../../../glossary.md#term-narrowing), then one final [ties-to-even](../../../../glossary.md#term-ties-to-even) low narrowing per declared output. This is not a once-rounded exact-mathematical result promise. NaN/[infinity](../../../../glossary.md#term-infinity), [signed zero](../../../../glossary.md#term-signed-zero), and subnormal/underflow stay outside [finite](../../../../glossary.md#term-finite) [tolerance](../../../../glossary.md#term-tolerance). |

Historical exploratory backend-local observations, not clean-base proof or public semantics: [CPU](../../../../glossary.md#term-cpu)
generated scatter yielded exact `+0` for `maxFinite * 2 * 0`; an [overflow](../../../../glossary.md#term-overflow)-first tree gives NaN.
Verified-package [Metal](../../../../glossary.md#term-metal) `CUSTOM_PROGRAM` preserved `0x0001` in casts/raw moves for both low
types; minimum-subnormal `* 1` gave `0x0000` for `BFLOAT16`, `0x0001` for `FLOAT16`. Metal
`BFLOAT16` `0x0001 * 128` gave `0x0000`, ruling out final low-result FTZ alone; input DAZ or
earlier [loss](../../../../glossary.md#loss) fits, but the zeroing site is unproved. [Carry](../../../../glossary.md#term-carry) and revalidate these observations.

Framework comparison, nonnormative: [PyTorch accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
and [JAX accuracy](https://docs.jax.dev/en/latest/accuracy.html) discuss order/precision/underflow;
[JAX scatter](https://docs.jax.dev/en/latest/_autosummary/jax.Array.at.html) applies conflicting
updates in implementation-defined order. [MLX](https://ml-explore.github.io/mlx/build/html/usage/precision.html)
and [TensorFlow](https://www.tensorflow.org/guide/mixed_precision) describe scoped compute precision;
none establishes identical semantics or a universal standard for Synaptik's exact boundaries.

## [Scope](../../../../glossary.md#term-scope)

- Inventory formula, conversions, work/accumulator dtype and final cast across unary, binary/scalar,
  reduction/scan/statistics/loss, contraction/spatial/attention/recurrent, dropout and gradients.
  Cover <span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type); [dtype glossary](../../../../glossary.md#data-type--datatype); [unary glossary](../../../../glossary.md#term-unary); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [scan glossary](../../../../glossary.md#term-cumulative-scan); [loss glossary](../../../../glossary.md#loss); [contraction glossary](../../../../glossary.md#contraction-dimension); [spatial glossary](../../../../glossary.md#term-spatial-axes); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [gradients glossary](../../../../glossary.md#term-gradient))</span>`MIN`/`MAX`/`CLAMP`, empty/domain/special, guards and under/overflow.
  Guarded attention's no-eligible/all-negative-infinity/positive-infinity-tie rules differ from
  literal softmax; composites gain no universal zero/subnormal promise. DAZ/FTZ remains arithmetic-
  only; no direct mixed-low route.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [infinity glossary](../../../../glossary.md#term-infinity); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [subnormal glossary](../../../../glossary.md#term-subnormal); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz))</span>
- Preserve Compiler's seven guarded identity rewrites, phase/descriptor/output/gradient guards and
  autograd ownership. Tolerance or a provider sample cannot license <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [autograd glossary](../../../../glossary.md#autograd); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Tolerance glossary](../../../../glossary.md#term-tolerance); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>`x * 0`, `x - x`, reassociation,
  fusion or contraction across observable/special boundaries; unproved legality keeps the old path.<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Remove numerical selectors/branches/wires/trace fields/cold propagation; preserve actual CPU/
  Metal <span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal))</span>[`ACCELERATOR`](../../../../glossary.md#term-accelerator) *true and false* provider answers, not semantic reachability. Project the
  representative, non-exhaustive 508-row v1 ledger to profile-free backend/occurrence rows with
  false answers, query provenance, independent low queries and negative predicates. Lost true rows
  block cutover; the ledger is not numerical certification.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [occurrence glossary](../../../../glossary.md#term-occurrence); [provenance glossary](../../../../glossary.md#provenance))</span>
- Retain CPU generation/OpenBLAS, training/autograd, transfers, lifetimes, tuning and eligible
  FLOAT32 MPSGraph/custom routes. Low Metal stays fixed <span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [FLOAT32 glossary](../../../../glossary.md#term-float32); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>`CUSTOM_PROGRAM`, without retry/fallback.
  Remove *numerical* certification, not source/manifest/package authentication, native application
  binary interface (ABI), decoder/preflight, bounds, ownership or compiled-code checks.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#application-binary-interface-and-native-bridge))</span><span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest); [authentication glossary](../../../../glossary.md#term-authentication); [application binary interface glossary](../../../../glossary.md#term-application-binary-interface-abi); [preflight glossary](../../../../glossary.md#term-preflight))</span>

## Non-goals

- No fast-math switch, production tolerance registry/certificate substitute or proof engine,
  determinism promise, MPSGraph removal, Metal fallback, or sample-inferred capability.<span data-glossary-navigation="true"> ([tolerance glossary](../../../../glossary.md#term-tolerance); [registry glossary](../../../../glossary.md#term-registry); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions); [engine glossary](../../../../glossary.md#term-engine); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [Metal glossary](../../../../glossary.md#term-metal); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Retain tuning, caches, benchmarks, input `ProfileFingerprint`, cost evidence, non-profile
  identities, and history; singleton Metal <span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>`NEG` tuning is separate.

## Contracts

The root and owning scoped contracts now govern the completed profile-free implementation.
[Architecture decision record](../../../../glossary.md#term-architecture-decision-record-adr) ([ADR](../../../../glossary.md#adr-ci-and-qualification)) 0026 records the accepted choice but does not replace those
contracts. Architecture explanation and tests were reconciled in the integrated change.

- [Root core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>: the main-worktree contract
  specifies profile-free semantics.
- [Foundational Model and profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [profile-free numerical semantics glossary](../../../../glossary.md#profile-free-numerical-semantics))</span>:
  stored-value comparison/extrema and named arithmetic-site DAZ/FTZ are separate boundaries.
  Also review <span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz))</span>[profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries).
- [Compiler compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [compile glossary](../../../../glossary.md#compile))</span>
  and [profile-free compile identity](../../../../architecture/contracts/compiler-autograd.md#profile-free-compile-identity)<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>:
  retain optimization/gradient ownership.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- [Profile-free Engine and Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#profile-free-engine-and-prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>:
  remove cold transport, not Runtime's policy-free boundary.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Backend low-precision capability evidence](../../../../architecture/contracts/backend-execution.md#low-precision-capability-evidence)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [precision glossary](../../../../glossary.md#term-precision))</span>:
  retain ledger/routes/source/ABI safety; review <span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>[profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity)<span data-glossary-navigation="true"> ([backend capability glossary](../../../../glossary.md#backend-capability))</span> and [CPU routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>.
- [Training ownership](../../../../architecture/contracts/extensions-training.md#extensionstraining): retain [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle) and [optimizer](../../../../glossary.md#term-optimizer) [ownership](../../../../glossary.md#term-ownership-meanings).

## Dependencies and integration

- Depends on: 0032A (Complete). The user approved the named arithmetic-site DAZ/FTZ rule on
  2026-10-08, closing the former non-task decision gate. Completed 0030/0031 and the current
  frozen-HEAD profile spine remain the clean-base context.<span data-glossary-navigation="true"> ([DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz))</span>
- Conflicts with: concurrent edits to numerical authority, compiler rewrite legality, provider
  predicates, trace DTOs, Metal schema/native source, CPU generated identity, and shared guides.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [trace glossary](../../../../glossary.md#trace); [Metal glossary](../../../../glossary.md#term-metal); [schema glossary](../../../../glossary.md#term-schema); [CPU glossary](../../../../glossary.md#term-cpu); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Parallel group: None.
- Common base revision: `9fbd0cc36f2f187d1a658671cb55710ff13a5ed7`, the historical
  clean source/test base used for isolated implementation and validation before transfer to main.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Integration order: authority-first revision and integration-owner acceptance → isolated
  code/test/identity cutover → independent targeted Class C review → isolated full validation →
  byte-identical transfer and source-matched native rebuild in main → final documentation
  reconciliation. Git publication is a separate step from this validation record.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [publication glossary](../../../../glossary.md#term-publication))</span>
- Integration validation: the architecture, backend-conformance, integration, Javadoc, native,
  and repository build gates recorded under Validation and Result.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration owner: Codex main coordination context. It assigned exclusive
  contract and final documentation scopes, preserved user dirty edits, and owns ordered
  integration; workers had no overlapping writes.

## Files and symbols

- `modules/model`, `modules/config`, `modules/planning`, `modules/compiler`, `modules/prepare`,
  `modules/engine`: semantics/Javadocs, `NumericalProfile`, [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery),
  `CompileArtifacts`, [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext), Config and `Engine.Builder.numericalProfile(NumericalProfile)`
  removal. Runtime is already profile-free; no removal there.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- `modules/trace`, `backends/cpu`, `backends/metal`, `native/metal-macos-arm64`: DTOs, CPU codegen/OpenBLAS/
  tuning/cache, Metal plans/schema/decoder, authentication, ABI/package gates.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [cache glossary](../../../../glossary.md#term-cache); [Metal glossary](../../../../glossary.md#term-metal); [schema glossary](../../../../glossary.md#term-schema); [authentication glossary](../../../../glossary.md#term-authentication); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- `extensions/training`, `testing/architecture-tests`, `testing/backend-conformance`, `testing/integration-tests`: [optimizer](../../../../glossary.md#term-optimizer) and cross-[module](../../../../glossary.md#term-module) tests.

## Historical readiness evidence and limits

The initial 2026-10-08 [handoff](../../../../glossary.md#term-handoff) reported these results from the frozen pre-cutover HEAD;
this final documentation pass did not rerun Java or native tests. `LowPrecisionCapabilityLedgerTest`
passed 6/6. The then-checked-in
representative [accelerator](../../../../glossary.md#term-accelerator) rows give [CPU](../../../../glossary.md#term-cpu) [`FLOAT32`](../../../../glossary.md#term-float32) 122 true/5 false and [Metal](../../../../glossary.md#term-metal) `FLOAT32` 98
true/29 false. Independently queried [`BFLOAT16`](../../../../glossary.md#term-bfloat16) and [`FLOAT16`](../../../../glossary.md#term-float16) columns each give CPU 114 true/5
false and Metal 90 true/29 false, with eight not-applicable controls per [backend](../../../../glossary.md#backend) and low type.
These are provider answers for named occurrences, not an enumerated support domain or numerical
proof. Preserve *both* true and false answers and the query [provenance](../../../../glossary.md#provenance) at cutover.

A pre-cutover source-matched native build, fixed-identifier [ad-hoc signing](../../../../glossary.md#term-ad-hoc-signing), local packaging, and
independent verification passed from that HEAD: [ABI](../../../../glossary.md#term-application-binary-interface-abi) 7, [schema](../../../../glossary.md#term-schema) 19, exactly thirteen exports, and packaged
[dylib](../../../../glossary.md#term-dylib) [SHA-256](../../../../glossary.md#term-sha-256) `30e899d1f663b64b217087073d54077fbcd2e3fae11cd5abc7c2508b6f1bb294`.
Configured real-[device](../../../../glossary.md#term-device) `MetalStoredSubnormalProviderNativeTest` passed 2/2 without skips.
Existing 0032A CPU generated FLOAT32 [scalar](../../../../glossary.md#term-scalar)/[vector](../../../../glossary.md#term-vector) and Metal FLOAT32 `CUSTOM_PROGRAM` witnesses
were route-local. Earlier main-worktree dirty CPU scatter and expanded Metal low tests were not
in that clean base; the integration preserved user edits and carried the intended assertions
through cutover. These historical observations did not, by themselves, prove all-route behavior
or the device's precise low-word zeroing site.

At launch, 0032A, 0030, and 0031 were Complete; the approved table closed the sole recorded
decision gate. This was the one authorized [Model](../../../../glossary.md#term-model) frontier, with no declared conflicting write
task. The root/scoped revision was reviewed before code. The completed main-worktree cutover
uses CPU schema 69, Metal schema 20/header 124 and [identity](../../../../glossary.md#term-identity-meanings) 30, and native ABI 7. The earlier
schema-19 [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) is historical and does not certify schema 20; source/package [authentication](../../../../glossary.md#term-authentication)
remains a current safety gate. [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0026 is Accepted.

## Historical implementation sequence and [checkpoints](../../../../glossary.md#term-checkpoint)

The following was the original execution sequence. The Result records its closure; these steps
are not outstanding instructions or a claim of remote [publication](../../../../glossary.md#term-publication).

1. **Authority and pre-implementation inventory, integration [owner](../../../../glossary.md#term-owner) and clean agents:** confirm
   this HEAD and current contract set remain stable; enumerate cold fields, native wires, CPU generated/tuning/
   OpenBLAS and Metal program/tuning/pointwise/manifest identities, ledger/Trace/benchmark fields,
   unqualified routes, and exact/special/finite test cases. Recheck the provider baseline if the
   base changes. The architecture-documentation agent's root and owning scoped-contract revision
   was reviewed and accepted by the integration owner before implementation.
   Current 0032A witnesses do not certify other routes or low arithmetic.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [Metal glossary](../../../../glossary.md#term-metal); [pointwise glossary](../../../../glossary.md#term-pointwise); [manifest glossary](../../../../glossary.md#term-manifest); [Trace glossary](../../../../glossary.md#trace); [finite glossary](../../../../glossary.md#term-finite))</span>
2. **Cutover, coordinated clean implementation context:** land approved Model-owned family semantics,
   Javadocs and guarded Compiler rewrites with counterexamples; remove cold numerical-profile
   transport through Config/Planning/Compiler/Prepare/Engine/Trace and CPU/Metal/native identities,
   schema and routes together. Runtime stays profile-free. Preserve kernels, MPSGraph, transfers,
   Training, safety verifiers and fail-closed unsupported rows. Version changed identities without
   preselected numbers; reject stale inputs/cache hits. Drop only numerical-profile bytes; retain
   distinct low keys, class/checksum checks and other wires. Remove Metal's profile wire atomically
   in Java/native; keep ABI if all thirteen exports are unchanged, otherwise review a bump. The
   integration owner transfers only the complete runnable cutover.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [schema glossary](../../../../glossary.md#term-schema); [Runtime glossary](../../../../glossary.md#term-runtime); [kernels glossary](../../../../glossary.md#kernel); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [cache glossary](../../../../glossary.md#term-cache); [checksum glossary](../../../../glossary.md#term-checksum); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
3. **Semantic/safety checkpoint, implementer then integration owner:** qualify supported routes
   with exact structure/discrete/cast/raw and stored-value comparison (subnormal versus <span data-glossary-navigation="true"> ([subnormal glossary](../../../../glossary.md#term-subnormal))</span>`+0` both
   orders, first/last arg ties), scatter contributors/specials/underflow, low work/accumulator and
   final-rounding tests, plus test-only backend/route/family/dtype/size finite tolerances with an
   explicit oracle, metric, threshold and cancellation boundary. Keep NaN/Inf, signed zero,
   empty/domain and subnormal cases outside
   finite tolerance. Prove provider parity, negative predicates, stale identity/schema rejection,
   native ABI/source authentication and fail-before-mutation; run configured real-device Metal.<span data-glossary-navigation="true"> ([underflow glossary](../../../../glossary.md#term-underflow); [accumulator glossary](../../../../glossary.md#accumulator-type); [rounding glossary](../../../../glossary.md#term-rounding); [backend glossary](../../../../glossary.md#backend); [dtype glossary](../../../../glossary.md#data-type--datatype); [oracle glossary](../../../../glossary.md#term-test-oracle); [metric glossary](../../../../glossary.md#term-metric); [cancellation glossary](../../../../glossary.md#term-cancellation); [NaN glossary](../../../../glossary.md#term-nan); [signed zero glossary](../../../../glossary.md#term-signed-zero); [tolerance glossary](../../../../glossary.md#term-tolerance); [identity glossary](../../../../glossary.md#term-identity-meanings); [authentication glossary](../../../../glossary.md#term-authentication); [run glossary](../../../../glossary.md#term-run); [device glossary](../../../../glossary.md#term-device))</span>
4. **Final checkpoint, integration owner and independent reviewer:** reconcile accepted authority,
   ADR, architecture explanation, Javadocs, user guide, glossary and shared plans; complete targeted
   Class C review. Reuse successful affected-module tests, run cross-module gates after integration
   and repository-wide validation once. Record limitations; mark <span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [validation glossary](../../../../glossary.md#term-validation))</span>`Complete` only after acceptance,
   review and validation pass.

## Acceptance criteria

- Authority, Javadocs and tests agree on the decision table/exception inventory. Every supported
  route qualifies the approved family-by-dtype arithmetic-site DAZ/FTZ table, exact/stored-value,
  scatter special/integral/unaddressed, low underflow/precision, and separate finite-tolerance
  behavior; no production tolerance engine.<span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [underflow glossary](../../../../glossary.md#term-underflow); [precision glossary](../../../../glossary.md#term-precision); [finite glossary](../../../../glossary.md#term-finite); [tolerance glossary](../../../../glossary.md#term-tolerance); [engine glossary](../../../../glossary.md#term-engine))</span>
- No numerical selector or ignored adapter remains in API, cold artifacts, Trace, backend/native
  plans or active docs. Retained tuning identities remain; CPU/Metal true/false answers match the
  baseline, and independent low queries, negative predicates and stale rejection pass.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [artifacts glossary](../../../../glossary.md#term-artifact); [Trace glossary](../../../../glossary.md#trace); [backend glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal))</span>
- Compiler's seven guarded rewrites preserve special, rounding and saved/state counterexamples;
  tolerance never proves legality. Changed generated algorithms/hot paths alone require optimal-Java
  algorithm/loop/dataflow, bytecode/hidden-overhead and generated-versus-direct performance evidence.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [rounding glossary](../../../../glossary.md#term-rounding); [dataflow glossary](../../../../glossary.md#term-dataflow); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Structural/source/ABI/route/fail-before-mutation, CPU/Metal, Training/autograd, transfer,
  lifecycle and native-failure gates pass. Remove only numerical certificates/complete-domain
  result proofs; publish authority, ADR, guide, glossary and independent Class C review.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [autograd glossary](../../../../glossary.md#autograd); [lifecycle glossary](../../../../glossary.md#lifecycle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## [Validation](../../../../glossary.md#term-validation)

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
configured [Metal](../../../../glossary.md#term-metal) [run](../../../../glossary.md#term-run). Use the verified package path; check [manifest](../../../../glossary.md#term-manifest) [ABI](../../../../glossary.md#term-application-binary-interface-abi)/[schema](../../../../glossary.md#term-schema), thirteen export
names/[signatures](../../../../glossary.md#term-signature), source match, [hash](../../../../glossary.md#term-hash), stale-schema rejection, and no-skips [device](../../../../glossary.md#term-device) evidence:

```bash
./native/metal-macos-arm64/build.sh
/usr/bin/codesign --force --sign - --timestamp=none --identifier io.github.pho001.synaptik.metal.foundation native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/package-local.sh native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/verify-package.sh native/metal-macos-arm64/build/package-v1/macos-arm64
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test --tests '*MetalStoredSubnormalProviderNativeTest' --rerun-tasks
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test --rerun-tasks
```

The integration [owner](../../../../glossary.md#term-owner) ran the post-cutover architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), integration,
[Javadoc](../../../../glossary.md#term-javadoc), native, and full-build gates with the verified package configured for Metal where
required. Device-test skips would not count as a pass. The command list remains as the planned
validation recipe; see Result for the completed evidence:

```bash
./gradlew :testing:architecture-tests:test
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :testing:backend-conformance:test :testing:integration-tests:test --rerun-tasks
./gradlew :modules:model:javadoc :modules:config:javadoc :modules:planning:javadoc :modules:compiler:javadoc :modules:prepare:javadoc :modules:trace:javadoc :modules:engine:javadoc :backends:cpu:javadoc :backends:metal:javadoc :extensions:training:javadoc
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew build
git diff --check
```

Audit active profile references, provider negative predicates, [identity](../../../../glossary.md#term-identity-meanings)/[cache](../../../../glossary.md#term-cache) stale-input tests,
and reported skips. Add generated-[bytecode](../../../../glossary.md#term-bytecode)/[oracle](../../../../glossary.md#term-test-oracle) and generated-versus-direct performance evidence
only if a generated hot-path algorithm changes; no default five-[fork](../../../../glossary.md#term-fork-meanings) [matrix](../../../../glossary.md#term-matrix-meanings).

## Documentation and review impact

This brief records the authority-first revision, isolated executable cutover, transfer to main,
and final reconciliation. The independent documentation pass covered architecture explanation,
active [API](../../../../glossary.md#term-application-programming-interface-api)/[backend](../../../../glossary.md#backend) guides, [trace](../../../../glossary.md#trace), glossary, examples, and links; executable and [Javadoc](../../../../glossary.md#term-javadoc) changes
remained owned by implementation contexts. Final Class C review and integration [validation](../../../../glossary.md#term-validation) passed.
This [atomic](../../../../glossary.md#resource-transaction-and-rollback) cross-[module](../../../../glossary.md#term-module) brief exceeds the [planning](../../../../glossary.md#term-planning) guide's
200-line/15 KB [targets](../../../../glossary.md#term-target) to keep its decision exceptions, provider baseline, safety gates, and
execution packet together; splitting those contracts would risk a runnable half-cutover.

## Result

- Completed the Model-owned profile-free semantics and arithmetic-site DAZ/FTZ cutover, seven
  guarded Compiler identities, cold-selector removal, CPU/Metal/native identities, qualified
  routes, and coordinated Javadocs, guides, glossary, and architecture/conformance/integration
  tests. No generated hot-path algorithm changed.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Source/tests/docs were transferred byte-identically from the validated isolated worktree into
  main before this final documentation reconciliation; existing user dirty edits were preserved.
  Current CPU generator schema 69, Metal schema 20
  (124-byte header), Metal identity 30, and native ABI 7 with thirteen exports reject stale
  pre-cutover inputs. The 254-row v2 capability ledger retains actual historical ACCELERATOR
  true <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [capability ledger glossary](../../../../glossary.md#capability-ledger); [ACCELERATOR glossary](../../../../glossary.md#term-accelerator))</span>**and** false answers, independent BFLOAT16/FLOAT16 queries, and query provenance; the
  508-row v1 basis is historical evidence, not current numerical certification.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [provenance glossary](../../../../glossary.md#provenance))</span>
- Main native package was rebuilt from matching source, ad-hoc signed and verified: [SHA-256](../../../../glossary.md#term-sha-256)
  `77f683d4331a688423a5a5e14cb462e3d9bafdf1aaab3ae5963c20dfadb3d318`.
  Focused real-device tests passed 7/7 with zero skips.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- Isolated `./gradlew build` with the verified `SYNAPTIK_METAL_TEST_LIBRARY` setting shown above
  ran 3,853 total tests, zero failures, 39 explained skips. Architecture: 9 total, 0 failures,
  0 skips; CPU: 1,044 total, 0 failures,
  28 skips; Metal: 325 total, 0 failures, 10 observer skips; backend conformance: 30 total,
  0 failures, 0 skips; integration: 89 total, 0 failures, one CPU autotune skip, including
  60 Metal integration tests with zero skips. Javadocs passed with 46
  preexisting warnings. These are reused implementation/integration results, not rerun here.<span data-glossary-navigation="true"> ([observer glossary](../../../../glossary.md#term-observer); [backend glossary](../../../../glossary.md#backend))</span>
- Independent Class C reviews found P1 OpenBLAS runtime tolerance, fixed by exact ABI smoke V2,
  and P2 stale documentation, corrected before this final reconciliation. No P0/P1 remains.
  The MPSGraph scalar <span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [runtime glossary](../../../../glossary.md#term-runtime); [tolerance glossary](../../../../glossary.md#term-tolerance); [smoke glossary](../../../../glossary.md#term-smoke-test); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [scalar glossary](../../../../glossary.md#term-scalar))</span>`+0` guard is retained. Its performance effect is unmeasured and is a
  nonblocking follow-up, not missing acceptance evidence.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard))</span>
- Documentation review checked the relevant glossary term/anchor; no new term or glossary edit
  was needed. A read-only Ruby local-link/anchor check passed for all nine final documents;
  `git diff --check` passed, and targeted trailing-whitespace search found none. This records
  pre-publication validation; subsequent commit and push status belongs to Git history.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [validation glossary](../../../../glossary.md#term-validation); [commit glossary](../../../../glossary.md#term-commit))</span>

Status: Complete
