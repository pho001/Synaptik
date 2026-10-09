# Task 0076: Profile-free FLOAT32 SIGMOID route

## Status

Ready — the serial Metal frontier verified on 2026-10-09 at `c19f16ee`: Model 0032,
Metal 0069, and Metal 0074–0075 are Complete; no other Metal native/capability writer is active.

## Change class

Class C — a new numerical capability changes native lowering, prepared route behavior,
compatibility identity, and user-visible results across the Metal/Prepare boundary.

## Goal

Admit the bounded, no-gradient, canonical FLOAT32 Model `SIGMOID` occurrence through one fixed
MPSGraph route that implements the Model's sign-guarded stable formula.

## Scope

- One `UnaryElementwiseKind.SIGMOID` with `NoOperationAttrs`, equal positive fully static canonical
  input/output shapes of rank `1..16`, FLOAT32 type, no gradient roles, and checked element count,
  four-byte spans, and native dispatch/allocation geometry. Everything outside this domain is
  provider-false and rejected before resource mutation.
- Lower wire 64 as an explicit sign-guarded MPSGraph composition: for nonnegative `x`,
  `1/(1+EXP(-x))`; for negative `x`, `EXP(x)/(1+EXP(x))`. The predicate must classify stored
  finite subnormals and signed zero without MPSGraph floating-comparison DAZ. Both `EXP` sites,
  typed-one additions, and divisions retain the Model's named arithmetic semantics. A direct
  opaque sigmoid selector, unguarded formula, host repair, and route fallback are not substitutes.
- A FLOAT32-only SIGMOID inside a low-containing `CUSTOM_PROGRAM` partition may use an internal
  FLOAT32 MPSGraph boundary step, including after an explicit low-to-FLOAT32 cast. The partition
  route stays custom. BFLOAT16 and FLOAT16 SIGMOID remain independently false.
- Qualify ordinary finite outputs against the branch-correct `StrictMath.exp` reference with a
  predeclared test-only relative threshold `2e-6` for positive normal finite references, plus
  explicit cancellation/near-zero and saturation cases. Test NaN/infinity, `±0`, stored
  subnormals, output underflow, and branch-stability boundaries separately; tolerance cannot
  excuse a wrong class, sign, exact guard, or overflow. If the gate fails, report the input and
  error and stop for a reviewed route/threshold decision rather than silently widening it.
- Update current provider snapshot, negative predicates, route catalog, compatibility identity,
  native source/package authentication, backend conformance, public Engine execution, and
  affected Javadocs/docs. Keep the historical v1 ledger unchanged.

## Non-goals

No low-valued or FLOAT64 SIGMOID, gradient-bearing occurrence, generated backward, direct
`sigmoidWithTensor` substitution, runtime accuracy policy, certificate/proof layer, autotuning,
performance claim, new ABI/export, or new schema field.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants): Model
  formula ownership, exact guards, test-only finite qualification, independent dtype admission,
  and append-only changed identities.
- [Foundational contract — Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics), especially
  [Family formulas and unary sites](../../../../architecture/contracts/foundational-modules.md#family-formulas-and-unary-sites),
  [Arithmetic-site DAZ/FTZ](../../../../architecture/contracts/foundational-modules.md#arithmetic-site-dazftz),
  and [Composite inheritance](../../../../architecture/contracts/foundational-modules.md#composite-inheritance):
  the exact sign guard, stable branch formulas, arithmetic sites, and special values.
- [Backend contract — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  and [Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity):
  occurrence-specific route and compatibility boundaries.
- [Runtime/Prepare contract — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle):
  fixed route and resource declarations before Runtime.

## Dependencies and integration

- Depends on: Model 0032; Metal 0069, 0074A, 0074, 0075A, and 0075 (all Complete).
- Conflicts with: any concurrent Metal provider, native source/schema, route, identity,
  source-bound 0069 evidence, or shared-document writer; resumed historical 0053 production.
- Parallel group: None.
- Common base revision: N/A (serial); clean `c19f16ee` on `main`.
- Integration order: implementation → source-matched device and 0069 evidence rebind → independent
  Class C code/documentation review → main-context integration and status closure.
- Integration validation: native build/sign/package/verify; affected Metal, conformance, Engine,
  architecture and Javadoc checks; 0069 source/AIR rebind if native foundation changes; one full
  repository build after executable integration.
- Shared-document integration owner: main coordination context; no parallel shared-doc edits.

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalMpsGraphProgram`,
  `MetalOperationRouteCatalog`, `MetalNativeApi` — exact admission, wire-64 lowering/validation,
  selected route and identity.
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` — bounded native preflight and
  sign-guarded MPSGraph construction; keep ABI 7/schema 20 if their encodings stay fixed.
- Focused Metal, conformance and public Engine tests; v2 ledger/identity fixtures; affected Metal
  guide, source/Javadoc, and 0069 source-bound evidence.

## Acceptance criteria

- Provider, Java preflight and native creation agree exactly on supported and rejected
  occurrences; wrong type, gradient, rank, shape, layout, attributes, aliasing, span, route and
  stale identity fail before mutation.
- Sign guard is based on stored value (including signed zero and subnormals), not a floating
  comparison that can DAZ. `SIGMOID(±0)` is exactly `0.5`, `SIGMOID(-infinity)` is exactly `+0`,
  `SIGMOID(+infinity)` is exactly `+1`, and NaN remains NaN. Finite outputs satisfy the declared
  test-only metric or a separately specified underflow/saturation gate.
- Real-device source-matched tests exercise repeated prepared runs and input preservation with no
  relevant skips. There is no hidden CPU repair, implicit low conversion, runtime branch
  dispatch, fallback, or direct opaque sigmoid selector.
- Current v2 ledger changes only F32 SIGMOID; low SIGMOID stays false and historical v1 is
  untouched. Changed compatibility versions advance append-only and reject old inputs.
- Affected docs/Javadocs and independent Class C review agree with final code and tests.

## Validation

Worker builds, ad-hoc signs, packages, and verifies the native dylib, then runs focused backend,
conformance, and Engine tests with `SYNAPTIK_METAL_TEST_LIBRARY` pointing to that verified package.
Run affected-module Javadoc and `git diff --check`. The integration owner rebinds Task-0069
source/AIR evidence after the final native edit and runs one `./gradlew build --max-workers=1`
against the verified package. Reuse successful executable evidence in the independent docs/review
pass unless it changes behavior.

## Follow-up

- Plan and separately qualify BFLOAT16 and FLOAT16 SIGMOID custom routes as Task 0077 only after
  this route is complete; neither low answer follows from FLOAT32 admission.

## Documentation and review impact

Update Metal guide/capability explanation, identity and source/Javadoc, and targeted glossary
references. A separate clean Class C review/documentation context is mandatory because supported
behavior and the native/Prepare route change. No architecture decision is intended; stop before
implementation if an applicable contract proves ambiguous.

## Result

Empty until execution.
