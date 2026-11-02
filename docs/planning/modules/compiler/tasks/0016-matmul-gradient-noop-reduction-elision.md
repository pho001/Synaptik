# Task Compiler0016: Omit redundant MATMUL-gradient shape reductions

## Status

Complete — MAIN accepted source, validation and targeted documentation review on 2026-10-28.

## Change class

Class C conservatively: a compiler-generated hot-path topology changes. The user explicitly
approved continuing in MAIN after separate workers repeatedly made no progress; this exception
covers execution and targeted review, not architectural or numerical rules.

## Goal and scope

Construct `sumToShape` in MATMUL derivatives only when the rank-specific contribution's exact
Shape differs from the selected operand's Shape. Reuse the existing `normalize` owner, retaining
necessary batch reductions followed by at most one dtype-normalizing cast. Cover all four rank
pairings, both roles, floating types, symbolic equal Shapes and empty contractions.

## Non-goals

Do not change public `Tensor.sumToShape`, arithmetic, layouts, capabilities, saved values,
backend lowering, generated formats, dependencies or API. Do not remove real broadcast reductions.
The remaining Scan-versus-Java performance investigation waits for the approved repair queue.

## Contracts

Start at [the authority index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Read root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and the owning contract's
[`modules/compiler`](../../../../architecture/contracts/compiler-autograd.md#modulescompiler),
[Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
and [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle).
Existing Tensor operations remain the sole derivative algebra; tolerance is not rewrite permission.

## Dependencies and integration

- Depends on: Compiler0015 Complete and CPU0045 source/docs/evidence accepted by MAIN.
- Conflicts with: any writer to MATMUL derivative formulas, affected tests or shared planning.
- Parallel group: None; MAIN executes serially under the user-approved context exception.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus
  `/private/tmp/synaptik-cpu-repairs.8TWk5O/program-base.json` (2489 accepted dirty-tree paths).
- Integration order: Ready verification → source/tests → validation → targeted diff/docs review → acceptance.
- Integration validation: Compiler tests/Javadoc once, then focused actual CPU Scan reverse consumers;
  no benchmark, full-repository campaign, Git commit or push.
- Shared-document integration owner: MAIN.

Ready verification: MAIN checked the completed dependencies, unchanged accepted dirty baseline,
HEAD/origin/index and predecessor evidence, current contracts and absence of an active competing
writer/test/measurement. Compiler0016 is the sole new implementation frontier. Other repair
items remain queued until their own scoped packet is verified.

## Files and symbols

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LinearAlgebraGradientRules.java`:
  rank-case formulas and `normalize` documentation.
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GradientRulesTest.java`:
  direct formula structure, broadcast, role, type and empty-domain regressions.
- CPU `CpuSymbolicScanTimingMethodologyTest`: update current reverse fixture identities after
  actual ClassFile comparison; retain forward/historical hashes and all structural checks.
- Compiler master, CPU master/CPU0045 closeout and roadmap: serial repair ordering/status only.
- Review `docs/api/compile-api.md` MATMUL normalization description and the glossary's existing
  normalization definition; update explanatory wording only if inaccurate.

## Acceptance criteria

1. Exactly equal Shapes introduce no `sumToShape` node, including equal symbolic Dimensions.
2. Real leading/singleton batch broadcast still introduces one reduction for each affected role.
3. Cast follows the necessary reduction and occurs only when the operand dtype differs.
4. Selected-role nulls, repeated operands, rank edits, zero extents and public shape-reduction
   semantics remain unchanged; actual compiled Scan reverse consumers still pass.

## Validation

Run focused `GradientRulesTest`, then `:modules:compiler:test :modules:compiler:javadoc` after
source stabilization. Run CPU `CpuSymbolicScanGeneratedKernelTest` and `CpuGeneralScanExecutionTest`
with all general-Scan/dynamic-batch performance flags false. Reuse successful results for review.
Inspect the task-local diff, Javadoc and planning links; audit accepted baseline/evidence preservation.

## Documentation and review impact

Planning profile for this brief and status updates. Update implementation Javadoc to explain
the exact-Shape guard; review API/glossary text without adding new terminology or architecture.
MAIN performs the targeted review under the explicit user exception and records its limitation.

## Result

- Removed six unconditional rank-case `sumToShape` constructions; the existing exact-Shape
  normalizer alone decides whether a batch reduction is needed, before any dtype cast.
- Updated implementation Javadoc and the explicit-seed structural assertion; added all-four-type
  rank/role/equal-symbolic/empty/broadcast and promoted-cast-order checks. Public Tensor API is unchanged.
- Focused gradient-rule tests passed. Final combined command passed in 19 s: Compiler **402 tests**,
  no failures/errors/skips; Compiler Javadoc generated successfully; actual CPU general/symbolic
  Scan consumers **101 tests**, no failures/errors/skips. All timing opt-ins were false.
- Evidence: `/private/tmp/synaptik-cpu-repairs.8TWk5O/compiler0016-focused.log` and
  `compiler0016-validation.log`. No generated-code format, numerical or performance claim.
- Later full CPU integration exposed old reverse fixture pins. Actual before/after disassembly
  is identical after normalizing the binary name (503/497 reverse instructions); both forward
  ClassFiles are unchanged. The Scan lowerer already elided these copies, but changed source
  graph metadata changes both reverse compatibility keys. Updated only current reverse pins and
  old-discriminant projections, retained all previous checks and added an actual-body no-copy
  assertion. Deterministic identity/cache/specialization repair run: 35 PASS; see
  `compiler0016-identity-review.txt` and `cpu0046-repaired-validation.log`.
- MAIN reviewed the bounded source/test diff and current contracts under the user-approved context
  exception; this was not an independent-context review. Required reductions, rank edits and
  cast order remain; the existing backend already accepts this derivative algebra.
- Documentation impact: implementation Javadoc and planning updated. Existing API MATMUL
  normalization text and glossary already describe conditional broadcast reduction, so no change
  is needed there; no new term, architecture decision, dependency or ADR.

Status: Complete
