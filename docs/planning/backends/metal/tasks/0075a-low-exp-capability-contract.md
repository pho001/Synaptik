# Task 0075A: Authorize separately qualified low-precision EXP

## Status

Complete — the conditional future capability is authorized without changing current provider
answers; independent Class C review returned APPROVE with no in-scope P0/P1/P2 findings.

## Change class

Class C — authoritative Metal capability boundary and native route ownership.

## Goal

Reconcile the root and owning backend contract with the approved Task 0075 direction: BFLOAT16
and FLOAT16 `EXP` may be admitted only after separate custom-kernel qualification, without
prematurely claiming that the current provider supports either type.

## Scope

- Replace unconditional current low-`EXP` prohibition with an explicitly dated Task-0074
  checkpoint plus a conditional rule for Task 0075. The provider and v2 ledger remain false until
  the new low route, conformance, and identity work pass.
- Keep homogeneous low types distinct, FLOAT32 working evaluation with one final low narrowing,
  exact special-value rules, a custom step for any low-valued `EXP`, and one fixed
  `CUSTOM_PROGRAM` whole-partition route. No low MPSGraph node, implicit cast, fallback, runtime
  tolerance policy, or certificate is authorized.
- Update the root, one owning scoped contract, and its targeted Runtime/Prepare explanation. ADR
  0027 already records the capability-evolution decision; no new architecture decision or ADR is
  needed unless review finds a substantive deviation. Leave current user-facing support claims
  truthful; Task 0075 will update them after qualification.

## Non-goals

No provider, native, test, ledger, route, identity, ABI, or schema change. Do not mark Task 0075
Complete or claim BFLOAT16/FLOAT16 `EXP` works before implementation and device evidence.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants):
  capability evolution, distinct low dtypes, custom low partition and preserved historical v1.
- [Backend contract — Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity)
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend):
  independently qualified low occurrence and current Metal `EXP` clause.
- [Foundational contract — Family formulas and unary sites](../../../../architecture/contracts/foundational-modules.md#family-formulas-and-unary-sites)
  and [Arithmetic-site DAZ/FTZ](../../../../architecture/contracts/foundational-modules.md#arithmetic-site-dazftz):
  unchanged mathematical and special-value semantics.
- [Runtime/Prepare contract — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle):
  fixed route before Runtime.

## Dependencies and integration

- Depends on: 0074A Complete; 0074 Complete; explicit user approval of a separate low custom
  route and Task 0075 numerical/test direction.
- Conflicts with: Task 0075 implementation; other authoritative Metal capability edits.
- Parallel group: None.
- Common base revision: N/A for serial work; base commit
  `1cab0731b75c035f589a60f8b9922a767211e6c6` plus completed uncommitted 0074 contract/code.
- Integration order: this contract update → independent Class C review → unblock/launch 0075.
- Integration validation: links/anchors, targeted glossary, documentation style and
  `git diff --check`; no executable test rerun for docs-only change.
- Shared-document integration owner: main coordination context.
- Frontier verification (2026-10-09): 0074A and 0074 are Complete with review and build; Task
  0075 worker stopped before code edits on the explicit root/backend prohibition. No competing
  Metal writer is active. This is the sole authorized serial Metal frontier until accepted.

## Files and symbols

- `ARCHITECTURE.md` — exact low `EXP` sentence only within the root capability invariant.
- `docs/architecture/contracts/backend-execution.md` — owning `EXP` occurrence clause.
- `docs/architecture/runtime-prepare-backend-boundary.md` — current versus conditional route
  explanation. Review targeted guide/glossary impact, but do not claim current support.

## Acceptance criteria

- Root, backend contract, and explanation agree that both low `EXP` provider answers are still
  false at the Task-0074 checkpoint and may become true only through Task 0075's independently
  qualified custom route and current-provider ledger/identity updates.
- FLOAT32 direct MPSGraph and explicit low-to-FLOAT32 cast composition remain unchanged;
  low-valued work never runs through MPSGraph.
- Independent Class C review finds no ambiguous premature support claim; all edited links and
  anchors resolve and diff is clean.

## Documentation and review impact

Authority and cross-module explanation change. A clean contract writer and independent targeted
review context are mandatory. ADR 0027 remains the accepted decision for capability evolution.

## Result

The clean contract-writing context updated the root, owning backend contract, and Runtime/Prepare
explanation. Both low `EXP` answers remain false at the Task-0074 checkpoint; Task 0075 may admit
each only after a separately qualified custom step, provider/current-v2/conformance updates, and
compatible identity change. Independent review returned APPROVE with no in-scope P0/P1/P2.
Links/anchors, targeted glossary, and `git diff --check` passed; no executable tests were rerun
for this docs-only prerequisite. Pre-existing *current* identity-30 references in glossary and
guides are assigned to Task 0075's final documentation pass, not accepted as current facts. No
code, provider, ledger, ABI, or schema changed; no commit or push was made.

Status: Complete
