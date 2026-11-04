# Task 0076: Profile-free [FLOAT32](../../../../glossary.md#term-float32) [SIGMOID](../../../../glossary.md#term-sigmoid) route

## Reader context

The sign guard chooses the stable SIGMOID branch from stored bits before EXP/add/divide sites execute. It is not a floating comparison allowed to erase a subnormal sign decision. Metal's prepared MPSGraph composition can share one exponential while retaining the selected numerator and rounding sites; a direct opaque sigmoid selector is a different algorithm.

An explicit low-to-FLOAT32 cast changes carrier, not low SIGMOID eligibility. Finite gates below test this route rather than install a runtime tolerance engine. Positive-rank/no-gradient admission and identity are the initial checkpoint, not today's full unary domain.

- [Current SIGMOID original-carrier and scalar coverage](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains) — Connects later typed expansion to stable-branch semantics.
- [Current named-site and qualification boundary](../../../../backend-guide/metal-backend.md#numerical-semantics-and-capability) — Distinguishes formulas from ordinary finite thresholds.

## Status

Complete — the bounded [FLOAT32](../../../../glossary.md#term-float32) route, source-matched [device](../../../../glossary.md#term-device)/[conformance](../../../../glossary.md#term-conformance)/[Engine](../../../../glossary.md#term-engine) [validation](../../../../glossary.md#term-validation),
0069 source rebind, full build, documentation and independent Class C review passed.

## Change class

Class C — a new numerical capability changes native [lowering](../../../../glossary.md#lowering), prepared route behavior,
compatibility [identity](../../../../glossary.md#term-identity-meanings), and user-visible results across the [Metal](../../../../glossary.md#term-metal)/[Prepare](../../../../glossary.md#prepare) boundary.

## Goal

Admit the bounded, no-[gradient](../../../../glossary.md#term-gradient), [canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) [Model](../../../../glossary.md#term-model) [`SIGMOID`](../../../../glossary.md#term-sigmoid) [occurrence](../../../../glossary.md#term-occurrence) through one fixed
[MPSGraph](../../../../glossary.md#term-mpsgraph) route that implements the Model's sign-guarded stable formula.

## [Scope](../../../../glossary.md#term-scope)

- One `UnaryElementwiseKind.SIGMOID` with [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), equal positive fully static canonical
  input/output shapes of rank <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [shapes glossary](../../../../glossary.md#shape); [rank glossary](../../../../glossary.md#term-rank))</span>`1..16`, FLOAT32 type, no gradient roles, and checked element count,
  four-byte spans, and native dispatch/allocation geometry. Everything outside this domain is
  provider-false and rejected before resource mutation.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [element glossary](../../../../glossary.md#term-element); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Lower wire 64 as an explicit sign-guarded [MPSGraph](../../../../glossary.md#term-mpsgraph) composition: for nonnegative `x`,
  `1/(1+EXP(-x))`; for negative `x`, `EXP(x)/(1+EXP(x))`. The predicate must classify stored
  finite subnormals and signed zero without MPSGraph floating-comparison DAZ. An exact guard may
  select the argument for one shared <span data-glossary-navigation="true"> ([subnormals glossary](../../../../glossary.md#term-subnormal); [signed zero glossary](../../../../glossary.md#term-signed-zero); [DAZ glossary](../../../../glossary.md#term-daz); [guard glossary](../../../../glossary.md#term-guard); [argument glossary](../../../../glossary.md#term-argument))</span>[`EXP`](../../../../glossary.md#term-expx) evaluation and the numerator before the typed-one
  addition and division, retaining each branch's named arithmetic semantics. A direct
  opaque sigmoid selector, unguarded formula, host repair, and route fallback are not substitutes.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [opaque glossary](../../../../glossary.md#term-opaque); [host glossary](../../../../glossary.md#term-host))</span>
- A [FLOAT32](../../../../glossary.md#term-float32)-only [SIGMOID](../../../../glossary.md#term-sigmoid) inside a low-containing `CUSTOM_PROGRAM` partition may use an internal
  FLOAT32 MPSGraph boundary step, including after an explicit low-to-FLOAT32 cast. The partition
  route stays custom. BFLOAT16 and FLOAT16 SIGMOID remain independently false.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16))</span>
- Qualify ordinary [finite](../../../../glossary.md#term-finite) outputs against the branch-correct `StrictMath.exp` reference with a
  predeclared test-only relative threshold `2e-6` for positive normal finite references, plus
  explicit cancellation/near-zero and saturation cases. Test NaN/infinity, <span data-glossary-navigation="true"> ([cancellation glossary](../../../../glossary.md#term-cancellation); [saturation glossary](../../../../glossary.md#term-saturation); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity))</span>`±0`, stored
  subnormals, output underflow, and branch-stability boundaries separately; tolerance cannot
  excuse a wrong class, sign, exact guard, or overflow. If the gate fails, report the input and
  error and stop for a reviewed route/threshold decision rather than silently widening it.<span data-glossary-navigation="true"> ([underflow glossary](../../../../glossary.md#term-underflow); [tolerance glossary](../../../../glossary.md#term-tolerance); [overflow glossary](../../../../glossary.md#term-overflow); [widening glossary](../../../../glossary.md#term-widening))</span>
- Update current provider snapshot, negative predicates, route catalog, compatibility identity,
  native source/package authentication, backend conformance, public Engine execution, and
  affected Javadocs/docs. Keep the historical v1 ledger unchanged.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [identity glossary](../../../../glossary.md#term-identity-meanings); [authentication glossary](../../../../glossary.md#term-authentication); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>

## Non-goals

No low-valued or [FLOAT64](../../../../glossary.md#term-float64) [SIGMOID](../../../../glossary.md#term-sigmoid), [gradient](../../../../glossary.md#term-gradient)-bearing [occurrence](../../../../glossary.md#term-occurrence), generated backward, direct
`sigmoidWithTensor` substitution, [runtime](../../../../glossary.md#term-runtime) accuracy policy, [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions)/proof layer, [autotuning](../../../../glossary.md#autotuning--model-autotuning),
performance claim, new [ABI](../../../../glossary.md#term-application-binary-interface-abi)/export, or new [schema](../../../../glossary.md#term-schema) field.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants): Model
  formula ownership, exact guards, test-only finite qualification, independent dtype admission,
  and append-only changed identities.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [finite glossary](../../../../glossary.md#term-finite); [qualification glossary](../../../../glossary.md#term-qualification); [dtype glossary](../../../../glossary.md#data-type--datatype); [admission glossary](../../../../glossary.md#term-admission))</span>
- [Foundational contract — Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([Profile-free numerical semantics glossary](../../../../glossary.md#profile-free-numerical-semantics))</span>, especially
  [Family formulas and unary sites](../../../../architecture/contracts/foundational-modules.md#family-formulas-and-unary-sites)<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>,
  [Arithmetic-site DAZ/FTZ](../../../../architecture/contracts/foundational-modules.md#arithmetic-site-dazftz)<span data-glossary-navigation="true"> ([DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz))</span>,
  and [Composite inheritance](../../../../architecture/contracts/foundational-modules.md#composite-inheritance):
  the exact sign guard, stable branch formulas, arithmetic sites, and special values.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard))</span>
- [Backend contract — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal))</span>
  and [Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity)<span data-glossary-navigation="true"> ([backend capability glossary](../../../../glossary.md#backend-capability); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>:
  occurrence-specific route and compatibility boundaries.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- [Runtime/Prepare contract — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>:
  fixed route and resource declarations before Runtime.

## Dependencies and integration

- Depends on: [Model](../../../../glossary.md#term-model) 0032; [Metal](../../../../glossary.md#term-metal) 0069, 0074A, 0074, 0075A, and 0075 (all Complete).
- Conflicts with: any concurrent Metal provider, native source/schema, route, identity,
  source-bound 0069 evidence, or shared-document writer; resumed historical 0053 production.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Parallel group: None.
- Common base revision: N/A (serial); clean `c19f16ee` on `main`.
- Integration order: implementation → source-matched device and 0069 evidence rebind → independent
  Class C code/documentation review → main-context integration and status closure.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- Integration validation: native build/sign/package/verify; affected Metal, conformance, Engine,
  architecture and Javadoc checks; 0069 source/AIR rebind if native foundation changes; one full
  repository build after executable integration.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [Javadoc glossary](../../../../glossary.md#term-javadoc); [AIR glossary](../../../../glossary.md#term-air))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): main coordination context; no parallel shared-doc edits.

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalMpsGraphProgram`,
  `MetalOperationRouteCatalog`, `MetalNativeApi` — exact admission, wire-64 lowering/validation,
  selected route and identity.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [lowering glossary](../../../../glossary.md#lowering); [validation glossary](../../../../glossary.md#term-validation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` — bounded native preflight and
  sign-guarded MPSGraph construction; keep ABI 7/schema 20 if their encodings stay fixed.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema))</span>
- Focused Metal, conformance and public Engine tests; v2 ledger/identity fixtures; affected Metal
  guide, source/Javadoc, and 0069 source-bound evidence.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Acceptance criteria

- Provider, Java preflight and native creation agree exactly on supported and rejected
  occurrences; wrong type, gradient, rank, shape, layout, attributes, aliasing, span, route and
  stale identity fail before mutation.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [gradient glossary](../../../../glossary.md#term-gradient); [rank glossary](../../../../glossary.md#term-rank); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [aliasing glossary](../../../../glossary.md#term-aliasing); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Sign guard is based on stored value (including signed zero and subnormals), not a floating
  comparison that can DAZ. <span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard); [signed zero glossary](../../../../glossary.md#term-signed-zero); [subnormals glossary](../../../../glossary.md#term-subnormal); [DAZ glossary](../../../../glossary.md#term-daz))</span>`SIGMOID(±0)` is exactly `0.5`, `SIGMOID(-infinity)` is exactly `+0`,
  `SIGMOID(+infinity)` is exactly `+1`, and NaN remains NaN. Finite outputs satisfy the declared
  test-only metric or a separately specified underflow/saturation gate.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [Finite glossary](../../../../glossary.md#term-finite); [metric glossary](../../../../glossary.md#term-metric); [underflow glossary](../../../../glossary.md#term-underflow); [saturation glossary](../../../../glossary.md#term-saturation))</span>
- Real-device source-matched tests exercise repeated prepared runs and input preservation with no
  relevant skips. There is no hidden CPU repair, implicit low conversion, runtime branch
  dispatch, fallback, or direct opaque sigmoid selector.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [CPU glossary](../../../../glossary.md#term-cpu); [runtime glossary](../../../../glossary.md#term-runtime); [dispatch glossary](../../../../glossary.md#term-dispatch); [opaque glossary](../../../../glossary.md#term-opaque); [sigmoid glossary](../../../../glossary.md#term-sigmoid))</span>
- Current v2 ledger changes only F32 SIGMOID; low SIGMOID stays false and historical v1 is
  untouched. Changed compatibility versions advance append-only and reject old inputs.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- Affected docs/Javadocs and independent Class C review agree with final code and tests.

## [Validation](../../../../glossary.md#term-validation)

Worker builds, ad-hoc signs, packages, and verifies the native [dylib](../../../../glossary.md#term-dylib), then runs focused [backend](../../../../glossary.md#backend),
[conformance](../../../../glossary.md#term-conformance), and [Engine](../../../../glossary.md#term-engine) tests with `SYNAPTIK_METAL_TEST_LIBRARY` pointing to that verified package.
[Run](../../../../glossary.md#term-run) affected-[module](../../../../glossary.md#term-module) [Javadoc](../../../../glossary.md#term-javadoc) and `git diff --check`. The integration [owner](../../../../glossary.md#term-owner) rebinds Task-0069
source/[AIR](../../../../glossary.md#term-air) evidence after the final native edit and runs one `./gradlew build --max-workers=1`
against the verified package. Reuse successful executable evidence in the independent docs/review
pass unless it changes behavior.

## Follow-up

- Plan and separately qualify BFLOAT16 and FLOAT16 SIGMOID custom routes as Task 0077 only after
  this route is complete; neither low answer follows from FLOAT32 admission.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [FLOAT32 glossary](../../../../glossary.md#term-float32); [admission glossary](../../../../glossary.md#term-admission))</span>

## Documentation and review impact

Update [Metal](../../../../glossary.md#term-metal) guide/capability explanation, [identity](../../../../glossary.md#term-identity-meanings) and source/[Javadoc](../../../../glossary.md#term-javadoc), and targeted glossary
references. A separate clean Class C review/documentation context is mandatory because supported
behavior and the native/[Prepare](../../../../glossary.md#prepare) route change. No architecture decision is intended; stop before
implementation if an applicable contract proves ambiguous.

## Result

The provider admits only the bounded no-[gradient](../../../../glossary.md#term-gradient) [canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) [occurrence](../../../../glossary.md#term-occurrence). Wire 64 lowers to
one fixed [MPSGraph](../../../../glossary.md#term-mpsgraph) composition with a stored-bit sign [guard](../../../../glossary.md#term-guard) and one [`EXP`](../../../../glossary.md#term-expx) [evaluation](../../../../glossary.md#term-evaluation) per [element](../../../../glossary.md#term-element);
an explicit low-to-FLOAT32 cast may precede it inside a fixed custom [partition](../../../../glossary.md#partition). [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16)
[SIGMOID](../../../../glossary.md#term-sigmoid) remain false. Current v2 ledger and [identity](../../../../glossary.md#term-identity-meanings) advance to 33; historical v1, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 7 and
[schema](../../../../glossary.md#term-schema) 20 are unchanged. Native foundation source [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) is rebound to [SHA-256](../../../../glossary.md#term-sha-256)
`3ffacd1843f13ddc3753899e09d700d86a05162391f90a156d6404ae068a76fd`.

Native build/sign/package/verify, focused [device](../../../../glossary.md#term-device), affected [Metal](../../../../glossary.md#term-metal)/[conformance](../../../../glossary.md#term-conformance)/[Engine](../../../../glossary.md#term-engine)/architecture
tests, [Javadoc](../../../../glossary.md#term-javadoc), both 0069 evidence checks, `git diff --check`, and the full
`./gradlew build --max-workers=1` passed. [JUnit](../../../../glossary.md#term-junit) reports show 342 Metal, 32 conformance and 94
integration tests with zero failures; 10/0/1 previously explained skips and no new SIGMOID skip.
Independent Class C code/documentation review returned APPROVE with zero P0/P1/P2; it finalized
Javadoc, guide/[API](../../../../glossary.md#term-application-programming-interface-api) explanations, glossary references and 529 local Markdown links without editing
executable behavior. No remaining task blocker or performance claim.

Status: Complete
