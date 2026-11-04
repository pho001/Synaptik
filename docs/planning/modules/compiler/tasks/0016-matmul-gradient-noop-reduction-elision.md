# Task Compiler0016: Omit redundant [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)-[gradient](../../../../glossary.md#term-gradient) [shape](../../../../glossary.md#shape) reductions

## Status

Complete — MAIN accepted source, [validation](../../../../glossary.md#term-validation) and targeted documentation review on 2026-10-28.

## Change class

Class C conservatively: a [compiler](../../../../glossary.md#term-compiler)-generated hot-path [topology](../../../../glossary.md#term-topology) changes. The user explicitly
approved continuing in MAIN after separate workers repeatedly made no progress; this exception
covers execution and targeted review, not architectural or numerical rules.

## Goal and [scope](../../../../glossary.md#term-scope)

Construct `sumToShape` in [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [derivatives](../../../../glossary.md#term-derivative) only when the [rank](../../../../glossary.md#term-rank)-specific contribution's exact
[Shape](../../../../glossary.md#shape) differs from the selected operand's Shape. Reuse the existing `normalize` [owner](../../../../glossary.md#term-owner), retaining
necessary [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) reductions followed by at most one [dtype](../../../../glossary.md#data-type--datatype)-normalizing cast. Cover all four rank
pairings, both roles, floating types, symbolic equal Shapes and empty contractions.

## Non-goals

Do not change public `Tensor.sumToShape`, arithmetic, [layouts](../../../../glossary.md#layout), capabilities, saved values,
[backend](../../../../glossary.md#backend) [lowering](../../../../glossary.md#lowering), generated formats, dependencies or [API](../../../../glossary.md#term-application-programming-interface-api). Do not remove real broadcast reductions.
The remaining [Scan](../../../../glossary.md#term-cumulative-scan)-versus-Java performance investigation waits for the approved repair queue.

## Contracts

Start at [the authority index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Read root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and the owning contract's
[`modules/compiler`](../../../../architecture/contracts/compiler-autograd.md#modulescompiler),
[Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [automatic differentiation glossary](../../../../glossary.md#autograd))</span>
and [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>.
Existing [Tensor](../../../../glossary.md#tensor) operations remain the sole [derivative](../../../../glossary.md#term-derivative) algebra; [tolerance](../../../../glossary.md#term-tolerance) is not rewrite permission.

## Dependencies and integration

- Depends on: Compiler0015 Complete and CPU0045 source/docs/evidence accepted by MAIN.
- Conflicts with: any writer to [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [derivative](../../../../glossary.md#term-derivative) formulas, affected tests or shared [planning](../../../../glossary.md#term-planning).
- Parallel group: None; MAIN executes serially under the user-approved context exception.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus
  `/private/tmp/synaptik-cpu-repairs.8TWk5O/program-base.json` (2489 accepted dirty-tree paths).
- Integration order: [Ready](../../../../glossary.md#term-ready) verification → source/tests → [validation](../../../../glossary.md#term-validation) → targeted diff/docs review → acceptance.
- Integration validation: Compiler tests/Javadoc once, then focused actual CPU Scan reverse consumers;
  no benchmark, full-repository campaign, Git commit or push.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Javadoc glossary](../../../../glossary.md#term-javadoc); [CPU glossary](../../../../glossary.md#term-cpu); [Scan glossary](../../../../glossary.md#term-cumulative-scan); [commit glossary](../../../../glossary.md#term-commit))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN.

Ready verification: MAIN checked the completed dependencies, unchanged accepted dirty baseline,
HEAD/origin/index and predecessor evidence, current contracts and absence of an active competing
writer/test/measurement. Compiler0016 is the sole new implementation frontier. Other repair
items remain queued until their own scoped packet is verified.

## Files and symbols

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LinearAlgebraGradientRules.java`:
  rank-case formulas and <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>`normalize` documentation.
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GradientRulesTest.java`:
  direct formula structure, broadcast, role, type and empty-domain regressions.
- [CPU](../../../../glossary.md#term-cpu) `CpuSymbolicScanTimingMethodologyTest`: update current reverse fixture identities after
  actual ClassFile comparison; retain forward/historical hashes and all structural checks.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings); [hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture))</span>
- [Compiler](../../../../glossary.md#term-compiler) master, CPU master/CPU0045 closeout and roadmap: serial repair ordering/status only.
- Review `docs/api/compile-api.md` MATMUL normalization description and the glossary's existing
  normalization definition; update explanatory wording only if inaccurate.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [normalization glossary](../../../../glossary.md#term-normalization))</span>

## Acceptance criteria

1. Exactly equal [Shapes](../../../../glossary.md#shape) introduce no `sumToShape` [node](../../../../glossary.md#node), including equal symbolic [Dimensions](../../../../glossary.md#dimension).
2. Real leading/[singleton](../../../../glossary.md#term-singleton) [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) broadcast still introduces one reduction for each affected role.
3. Cast follows the necessary reduction and occurs only when the operand [dtype](../../../../glossary.md#data-type--datatype) differs.
4. Selected-role nulls, repeated operands, rank edits, zero extents and public shape-reduction
   semantics remain unchanged; actual compiled Scan reverse consumers still pass.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Scan glossary](../../../../glossary.md#term-cumulative-scan))</span>

## [Validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused `GradientRulesTest`, then `:modules:compiler:test :modules:compiler:javadoc` after
source stabilization. Run [CPU](../../../../glossary.md#term-cpu) `CpuSymbolicScanGeneratedKernelTest` and `CpuGeneralScanExecutionTest`
with all general-[Scan](../../../../glossary.md#term-cumulative-scan)/dynamic-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) performance flags false. Reuse successful results for review.
Inspect the task-local diff, [Javadoc](../../../../glossary.md#term-javadoc) and [planning](../../../../glossary.md#term-planning) links; audit accepted baseline/evidence preservation.

## Documentation and review impact

[Planning](../../../../glossary.md#term-planning) profile for this brief and status updates. Update implementation [Javadoc](../../../../glossary.md#term-javadoc) to explain
the exact-[Shape](../../../../glossary.md#shape) [guard](../../../../glossary.md#term-guard); review [API](../../../../glossary.md#term-application-programming-interface-api)/glossary text without adding new terminology or architecture.
MAIN performs the targeted review under the explicit user exception and records its limitation.

## Result

- Removed six unconditional [rank](../../../../glossary.md#term-rank)-case `sumToShape` constructions; the existing exact-Shape
  normalizer alone decides whether a batch reduction is needed, before any dtype cast.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [dtype glossary](../../../../glossary.md#data-type--datatype))</span>
- Updated implementation Javadoc and the explicit-seed structural assertion; added all-four-type
  rank/role/equal-symbolic/empty/broadcast and promoted-cast-order checks. Public Tensor API is unchanged.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [seed glossary](../../../../glossary.md#term-seed); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Focused [gradient](../../../../glossary.md#term-gradient)-rule tests passed. Final combined command passed in 19 s: [Compiler](../../../../glossary.md#term-compiler) **402 tests**,
  no failures/errors/skips; Compiler Javadoc generated successfully; actual CPU general/symbolic
  Scan consumers <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [symbolic Scan glossary](../../../../glossary.md#symbolic-scan))</span>**101 tests**, no failures/errors/skips. All timing opt-ins were false.
- Evidence: `/private/tmp/synaptik-cpu-repairs.8TWk5O/compiler0016-focused.log` and
  `compiler0016-validation.log`. No generated-code format, numerical or performance claim.
- Later full CPU integration exposed old reverse fixture pins. Actual before/after disassembly
  is identical after normalizing the binary name (503/497 reverse instructions); both forward
  ClassFiles are unchanged. The Scan lowerer already elided these copies, but changed source
  graph metadata changes both reverse compatibility keys. Updated only current reverse pins and
  old-discriminant projections, retained all previous checks and added an actual-body no-copy
  assertion. Deterministic identity/cache/specialization repair run: 35 PASS; see<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [Scan glossary](../../../../glossary.md#term-cumulative-scan); [graph glossary](../../../../glossary.md#graph); [metadata glossary](../../../../glossary.md#term-metadata); [identity glossary](../../../../glossary.md#term-identity-meanings); [cache glossary](../../../../glossary.md#term-cache); [specialization glossary](../../../../glossary.md#term-specialization); [run glossary](../../../../glossary.md#term-run))</span>
  `compiler0016-identity-review.txt` and `cpu0046-repaired-validation.log`.
- MAIN reviewed the bounded source/test diff and current contracts under the user-approved context
  exception; this was not an independent-context review. Required reductions, rank edits and
  cast order remain; the existing backend already accepts this derivative algebra.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Documentation impact: implementation Javadoc and planning updated. Existing API MATMUL
  normalization text and glossary already describe conditional broadcast reduction, so no change
  is needed there; no new term, architecture decision, dependency or ADR.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [normalization glossary](../../../../glossary.md#term-normalization); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

Status: Complete
