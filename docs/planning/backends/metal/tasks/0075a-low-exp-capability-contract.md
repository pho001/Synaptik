# Task 0075A: Authorize separately qualified low-[precision](../../../../glossary.md#term-precision) [EXP](../../../../glossary.md#term-expx)

## Reader context

This contract permitted future separately qualified low EXP while leaving then-current provider answers false. It neither uploaded input values nor created a shader. Task 0075 owned the support change after implementation, conformance and review.

The participants are an original-typed provider query, Metal's fixed custom preparation and the current ledger recording its answer. FLOAT32 EXP after an explicit cast is a different occurrence from low EXP. The dated prohibition/permission below is a checkpoint distinction, not a present claim that low EXP is unavailable.

- [Current typed EXP support](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains) — Separates implementation from conditional authorization.
- [Current live versus frozen ledger meaning](../../../../backend-guide/metal-backend.md#numerical-semantics-and-capability) — Explains coexistence of later true answers and preserved history.

## Status

Complete — the conditional future capability is authorized without changing current provider
answers; independent Class C review returned APPROVE with no in-[scope](../../../../glossary.md#term-scope) P0/P1/P2 findings.

## Change class

Class C — authoritative [Metal](../../../../glossary.md#term-metal) capability boundary and native route [ownership](../../../../glossary.md#term-ownership-meanings).

## Goal

Reconcile the root and owning [backend](../../../../glossary.md#backend) contract with the approved Task 0075 direction: [BFLOAT16](../../../../glossary.md#term-bfloat16)
and [FLOAT16](../../../../glossary.md#term-float16) [`EXP`](../../../../glossary.md#term-expx) may be admitted only after separate custom-[kernel](../../../../glossary.md#kernel) [qualification](../../../../glossary.md#term-qualification), without
prematurely claiming that the current provider supports either type.

## [Scope](../../../../glossary.md#term-scope)

- Replace unconditional current low-[`EXP`](../../../../glossary.md#term-expx) prohibition with an explicitly dated Task-0074
  checkpoint plus a conditional rule for Task 0075. The provider and v2 ledger remain false until
  the new low route, conformance, and identity work pass.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [conformance glossary](../../../../glossary.md#term-conformance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Keep homogeneous low types distinct, FLOAT32 working evaluation with one final low narrowing,
  exact special-value rules, a custom step for any low-valued <span data-glossary-navigation="true"> ([homogeneous glossary](../../../../glossary.md#term-homogeneous); [FLOAT32 glossary](../../../../glossary.md#term-float32); [evaluation glossary](../../../../glossary.md#term-evaluation); [narrowing glossary](../../../../glossary.md#term-narrowing))</span>`EXP`, and one fixed
  `CUSTOM_PROGRAM` whole-partition route. No low MPSGraph node, implicit cast, fallback, runtime
  tolerance policy, or certificate is authorized.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [node glossary](../../../../glossary.md#node); [runtime glossary](../../../../glossary.md#term-runtime); [tolerance glossary](../../../../glossary.md#term-tolerance); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions))</span>
- Update the root, one owning scoped contract, and its targeted Runtime/Prepare explanation. ADR
  0027 already records the capability-evolution decision; no new architecture decision or ADR is
  needed unless review finds a substantive deviation. Leave current user-facing support claims
  truthful; Task 0075 will update them after qualification.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [qualification glossary](../../../../glossary.md#term-qualification))</span>

## Non-goals

No provider, native, test, ledger, route, [identity](../../../../glossary.md#term-identity-meanings), [ABI](../../../../glossary.md#term-application-binary-interface-abi), or [schema](../../../../glossary.md#term-schema) change. Do not mark Task 0075
Complete or claim [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) [`EXP`](../../../../glossary.md#term-expx) works before implementation and [device](../../../../glossary.md#term-device) evidence.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants):
  capability evolution, distinct low dtypes, custom low partition and preserved historical v1.<span data-glossary-navigation="true"> ([dtypes glossary](../../../../glossary.md#data-type--datatype); [partition glossary](../../../../glossary.md#partition))</span>
- [Backend contract — Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [backend capability glossary](../../../../glossary.md#backend-capability); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>:
  independently qualified low occurrence and current Metal <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>[`EXP`](../../../../glossary.md#term-expx) clause.
- [Foundational contract — Family formulas and unary sites](../../../../architecture/contracts/foundational-modules.md#family-formulas-and-unary-sites)<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>
  and [Arithmetic-site DAZ/FTZ](../../../../architecture/contracts/foundational-modules.md#arithmetic-site-dazftz)<span data-glossary-navigation="true"> ([DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz))</span>:
  unchanged mathematical and special-value semantics.
- [Runtime/Prepare contract — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>:
  fixed route before Runtime.

## Dependencies and integration

- Depends on: 0074A Complete; 0074 Complete; explicit user approval of a separate low custom
  route and Task 0075 numerical/test direction.
- Conflicts with: Task 0075 implementation; other authoritative [Metal](../../../../glossary.md#term-metal) capability edits.
- Parallel group: None.
- Common base revision: N/A for serial work; base [commit](../../../../glossary.md#term-commit)
  `1cab0731b75c035f589a60f8b9922a767211e6c6` plus completed uncommitted 0074 contract/code.
- Integration order: this contract update → independent Class C review → unblock/launch 0075.
- Integration [validation](../../../../glossary.md#term-validation): links/anchors, targeted glossary, documentation style and
  `git diff --check`; no executable test rerun for docs-only change.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main coordination context.
- Frontier verification (2026-10-09): 0074A and 0074 are Complete with review and build; Task
  0075 worker stopped before code edits on the explicit root/backend prohibition. No competing
  Metal writer is active. This is the sole authorized serial Metal frontier until accepted.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>

## Files and symbols

- `ARCHITECTURE.md` — exact low [`EXP`](../../../../glossary.md#term-expx) sentence only within the root capability [invariant](../../../../glossary.md#term-invariant).
- `docs/architecture/contracts/backend-execution.md` — owning `EXP` [occurrence](../../../../glossary.md#term-occurrence) clause.
- `docs/architecture/runtime-prepare-backend-boundary.md` — current versus conditional route
  explanation. Review targeted guide/glossary impact, but do not claim current support.

## Acceptance criteria

- Root, [backend](../../../../glossary.md#backend) contract, and explanation agree that both low [`EXP`](../../../../glossary.md#term-expx) provider answers are still
  false at the Task-0074 checkpoint and may become true only through Task 0075's independently
  qualified custom route and current-provider ledger/identity updates.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- FLOAT32 direct MPSGraph and explicit low-to-FLOAT32 cast composition remain unchanged;
  low-valued work never runs through MPSGraph.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- Independent Class C review finds no ambiguous premature support claim; all edited links and
  anchors resolve and diff is clean.

## Documentation and review impact

Authority and cross-[module](../../../../glossary.md#term-module) explanation change. A clean contract writer and independent targeted
review context are mandatory. [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0027 remains the accepted decision for capability evolution.

## Result

The clean contract-writing context updated the root, owning [backend](../../../../glossary.md#backend) contract, and [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)
explanation. Both low [`EXP`](../../../../glossary.md#term-expx) answers remain false at the Task-0074 [checkpoint](../../../../glossary.md#term-checkpoint); Task 0075 may admit
each only after a separately qualified custom step, provider/current-v2/[conformance](../../../../glossary.md#term-conformance) updates, and
compatible [identity](../../../../glossary.md#term-identity-meanings) change. Independent review returned APPROVE with no in-[scope](../../../../glossary.md#term-scope) P0/P1/P2.
Links/anchors, targeted glossary, and `git diff --check` passed; no executable tests were rerun
for this docs-only prerequisite. Pre-existing *current* identity-30 references in glossary and
guides are assigned to Task 0075's final documentation pass, not accepted as current facts. No
code, provider, ledger, [ABI](../../../../glossary.md#term-application-binary-interface-abi), or [schema](../../../../glossary.md#term-schema) changed; no [commit](../../../../glossary.md#term-commit) or push was made.

Status: Complete
