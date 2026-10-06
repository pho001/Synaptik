# Task 0032: Profile-free numerical-semantics reset

## Status

Draft — not an authorized frontier; current profile-based contracts remain until an approved
architecture decision, the evidence gates below, and a later `Ready` transition.

## Change class

Future Class C (numerics, Compiler rewrites, public API/capability, native format, cross-module
identity); this Draft edit is documentation-only.

## Goal

Remove `NumericalProfile`, both values, and cold propagation. Keep structural/discrete/cast
boundaries exact; judge approximate arithmetic by family formula, dtype, final cast, and *test-only*
backend/family/dtype/size tolerances, never public envelopes or runtime policy. **No interim public
rename** to `RELAXED`, `FAST`, or another profile; `ACCELERATOR` is only the legacy provider-support
baseline, not a performance promise.

## Proposed decision table (not current authority)

| Boundary | Proposed rule | Readiness evidence or open decision |
|---|---|---|
| Structure/discrete | Exact kinds/descriptors, dtype/promotion, Shape/layout/axis/contributors, guards, classification, bounds/indices, predicates/masks, ordering, state, saved values, transfers, publication, casts, and raw movement/selection. | Test exactness and fail-before-mutation separately. |
| `WHERE` | Exact Boolean selects a branch, then exact declared cast if promoted; same-type selection preserves bits. | Test promotion, both branches, zero/NaN/payloads. |
| Comparison/extrema | Use stored represented values, not denormals-are-zero (DAZ) comparison; preserve NaN, signed-zero, ties, winners, and first/last arg indices. | 0032A covers only CPU generated FLOAT32 scalar/vector and Metal FLOAT32 `CUSTOM_PROGRAM`; qualify other routes/dtypes. |
| Exact islands | `ABS`/`NEG`/`SIGN`/`FLOOR`/`CEIL`/`RELU`; scatter floating `MUL` is an order-independent abstract product with one final result-format rounding. | Resolve accelerator aggregate versus `ScatterReduction`/`AxisScatterKind` conflict; test routes. |
| Approximate arithmetic | Retain family formula, input/work/accumulator dtype, contributor domain and final cast; backend rounding may differ. | Evidence-backed test-only tolerances by backend/family/dtype/size; special classes tested separately. |
| Low precision | Distinct `BFLOAT16`/`FLOAT16` storage; homogeneous FLOAT32 work/accumulation, no intermediate low narrowing, one final ties-to-even conversion. Retain arithmetic low-result flush-to-zero (FTZ): a low-subnormal result may publish zero with operation-specific sign, no universal sign promise. Cast/raw movement/selection stay exact. | Route/dtype underflow tests; source encoding/raw comparison cannot prove device FLOAT32 work preserves subnormals. |

## Scope

- Inventory formula, conversions, work/accumulator dtype, and final cast across unary,
  binary/scalar, reduction/scan/statistics/loss, contraction/spatial/attention/recurrent,
  dropout, and generated gradients. Cover `MIN`/`MAX`/`CLAMP`, empty/domain/special cases,
  guards, and overflow/underflow. Guarded attention's no-eligible, all-negative-infinity, and
  positive-infinity-tie rules differ from literal softmax; composites gain no universal
  zero/subnormal promise. DAZ/FTZ is arithmetic-only; no direct mixed-low route.
- Preserve Compiler's seven guarded exact-identity rewrites, phase/descriptor/output/gradient
  guards, and autograd ownership. Neither tolerance nor a provider sample licenses `x * 0`,
  `x - x`, reassociation, fusion, or contraction across observable/special boundaries; retain the
  original path if legality is unproved.
- Remove numerical selectors, branches, wires, trace fields, and cold propagation; preserve actual
  CPU/Metal `ACCELERATOR` *true and false* provider answers, not semantic reachability alone.
  Project the representative, non-exhaustive 508-row v1 ledger to profile-free backend/occurrence
  rows with false answers, query provenance, independent `BFLOAT16`/`FLOAT16` queries, and negative
  predicate tests. Any lost true row blocks cutover; the ledger is not numerical certification.
- Retain CPU generation/OpenBLAS, training/autograd, transfers, lifetimes, tuning, and eligible
  FLOAT32 MPSGraph/custom routes. Low Metal partitions stay fixed `CUSTOM_PROGRAM`, without retry
  or fallback. Remove *numerical* certification, not source/manifest/package authentication, native
  application binary interface (ABI), decoder/preflight, bounds, ownership, or compiled-code checks.

## Non-goals

- No fast-math switch, production tolerance registry/certificate substitute, determinism promise,
  MPSGraph removal, Metal fallback, or sample-inferred capability.
- Retain tuning, caches, benchmarks, input `ProfileFingerprint`, cost evidence, non-profile
  identities, and history; singleton Metal `NEG` tuning is separate.

## Contracts

Current authority conflicts. Approve root and owning scoped-contract changes, an architecture
decision record (ADR), and affected architecture tests before implementation; ambiguity blocks work.

- [Root core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [dependency rules](../../../../../ARCHITECTURE.md#dependency-rules): profiles are normative now.
- [Foundational Model and numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles):
  current comparison DAZ/extrema ties are permitted; stored-value behavior narrows that guarantee.
  Also review [profile transport](../../../../architecture/contracts/foundational-modules.md#numerical-profile-transport).
- [Compiler compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)
  and profile identity: retain optimization/gradient ownership.
- [Runtime/Prepare/Engine profile lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#numerical-profile-lifecycle):
  remove cold transport, not Runtime's policy-free boundary.
- [Backend low-precision capability evidence](../../../../architecture/contracts/backend-execution.md#low-precision-capability-evidence):
  retain ledger/routes/source/ABI safety; review [backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity) and [CPU routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes).
- [Training ownership](../../../../architecture/contracts/extensions-training.md#extensionstraining): retain Engine lifecycle and optimizer ownership.

## Dependencies and integration

- Depends on: coordinated numerical-architecture decision; no task predecessor is declared.
- Conflicts with: concurrent edits to numerical authority, compiler rewrite legality, provider
  predicates, trace DTOs, Metal schema/native source, CPU generated identity, and shared guides.
- Parallel group: None.
- Common base revision: TBD before authorization.
- Integration order: authority decision → atomic contract/code/test/identity cutover → independent
  Class C review; no runnable half-cutover.
- Integration validation: focused and cross-module gates under Validation, then one repository build.
- Shared-document integration owner: TBD before authorization.

## Files and symbols

- `modules/model`, `modules/config`, `modules/planning`, `modules/compiler`, `modules/prepare`,
  `modules/engine`: formulas/Javadocs, `NumericalProfile`, `OperationCapabilityQuery`,
  `CompileArtifacts`, `PrepareContext`, Config, and removal of public
  `Engine.Builder.numericalProfile(NumericalProfile)`; Runtime stays profile-free.
- `modules/trace`, `backends/cpu`, `backends/metal`, `native/metal-macos-arm64`: DTOs, CPU codegen/OpenBLAS/
  tuning/cache, Metal plans/schema/decoder, authentication, ABI/package gates.
- `extensions/training`, `testing/architecture-tests`, `testing/backend-conformance`,
  `testing/integration-tests`: optimizer, architecture, provider, and end-to-end coverage.

## Identity migration gate

At authorization, inventory formats; version changed identities and test stale rejection/cache
miss for CPU generated/tuning/OpenBLAS, Metal program/tuning/pointwise/manifest, and ledger/Trace/
benchmarks. Drop only numerical-profile bytes; keep distinct `BFLOAT16`/`FLOAT16` keys,
class/checksum checks, and non-profile wires unless separately reviewed. Remove the Metal profile
wire atomically in Java/native; retain ABI if all thirteen export signatures stay unchanged,
otherwise review a bump. Reverify package, exports, source hashes, and stale inputs; preselect no
future version number.

## Remaining gates before `Ready`

- Approve per-family exact/special decisions. Resolve `AxisScatterKind` accelerator `MUL` tree
  versus `ScatterReduction` abstract-product/Javadoc with route tests before promising that island.
- Qualify stored-value comparison/extrema beyond 0032A: source lacks explicit FLOAT32 comparison
  DAZ and low `lp_compare`/`lp_extreme` use raw words, but neither proves route conformance. Test
  minimum subnormal versus `+0` in both orders, predicates/result bits, and first/last arg ties
  across routes and dtypes.
- On a verified source-matched package, test low FTZ versus exact casts/raw movement:
  `lp_pack_finite`/`lp_float_cast` encode subnormals, but device FLOAT32 work may flush before
  `lp_narrow`. Check work, final narrowing, saved values, and zero signs by route/dtype.
- Approve finite test tolerances by backend/route, family, dtype, and size, with reference oracle,
  error metric/threshold, cancellation/underflow boundary, and cases. Invent no threshold; assert
  NaN/Inf, signed zero, empty/domain, and exact islands separately.
- Reconcile promoted `WHERE`, guarded attention versus literal softmax, and empty/special cases in
  tests/Javadocs. Audit true/false predicates beyond the ledger, query low types independently,
  and classify numerical versus source/ABI/safety verifiers.

## Acceptance criteria

- Approved authority, tests, and Javadocs implement the decision table, including exact, special,
  empty, DAZ/FTZ, and zero-sign behavior; no production tolerance/proof engine.
- No numerical selector or ignored-profile adapter remains in API, cold artifacts, Trace,
  backend/native plans, or active docs; unrelated measurement/tuning profiles remain and old
  identities fail closed. CPU/Metal true *and false* provider answers match the legacy baseline.
- Compiler's guarded rewrites retain NaN, infinity, signed-zero, rounding, saved/state
  counterexamples; tolerance never proves legality. Specialized CPU bytecode matches optimal clean
  Java's algorithm/hot loop/dataflow/overhead, with semantic/decompilation, hidden-call/allocation,
  and generated-versus-direct performance evidence.
- Profile-free ledger, independent low queries, negative predicates, and structural, source/ABI,
  route, and fail-before-mutation gates pass. Remove only numerical certificates and complete-domain
  allowed-result proof prerequisites.
- CPU/Metal conformance separates finite tolerance from exact/special cases and covers low
  conversion, fixed routes, Training/autograd, transfer, lifecycle, and native failure. Authority,
  docs, and independent Class C review complete the cutover.

## Validation

Before `Ready`, select exact focused and affected-module commands for Model, Config, Planning,
Compiler, Prepare, Trace, Engine, CPU, Metal, and Training. Include architecture, backend-
conformance, and integration tests; Javadoc; one post-integration repository build; and
`git diff --check`. Build, authenticate, package, and verify native source before Metal tests;
use the extracted library to check ABI/exports and stale-schema rejection. Report device skips
and audit active profile references, not completed history.

## Documentation and review impact

Future cutover needs authority, ADR, API/Javadoc, guide, glossary, and plan updates plus independent
Class C review; this edit changes only Draft planning text.

## Result

Empty until the future authorized implementation and independent review complete.
