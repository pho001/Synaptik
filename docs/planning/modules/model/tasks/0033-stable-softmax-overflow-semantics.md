# Model 0033 — Decide stable softmax overflow semantics

## Status

Complete

## Goal and class

Class C numerical-contract and generated-code change. Resolve review G1: the current CPU
implementation avoids intermediate EXP overflow, while Model's literal wording can require a
different output class. The owner accepted stable finite softmax after the final review report
and authorized the implementation in the follow-up conversation.

## Concrete conflict

For contiguous FLOAT32 `[1000,1000]`, axis 0, the current CPU provider/finite-shift validator
admit execution. `CpuSoftmaxEmitter` subtracts the maximum, so EXP sees `[0,0]` and softmax
returns `[0.5,0.5]`. `SoftmaxKind` and the foundational composite row instead describe EXP
of each unshifted value, a rounded sum and division: EXP overflows to infinity and infinity
divided by infinity is NaN. The alternative-algorithm clause requires special-value preservation;
finite-input qualification alone does not authorize changing that result class.

This was source-confirmed by an independent read-only review. Existing CPU tests use a stable
oracle, not a demonstration that the literal contract authorizes the result. No new numerical
run was performed in that cross-check. CPU 0021's storage/validation fixes and CPU 0022's
activation-site repairs are separate and do not settle G1.

## Accepted decision and scope

Explicitly define stable finite softmax and log-softmax without
incidental unshifted-EXP overflow, retaining precise input NaN/infinity, empty, dtype, accumulator
and publication rules. Preserve the current stable CPU algorithm where conforming; do not
replace useful finite probabilities with NaNs solely to reproduce accidental intermediate overflow.

The owner's earlier response deferred the decision until the review report. That report was
delivered before the later authorization. For an admitted finite slice `[1000,1000]`, SOFTMAX
produces `[0.5,0.5]` and LOG_SOFTMAX produces two values near `-ln(2)`.

Define maximum selection, shifted exponentials, sum, probability/log-probability, working
precision, narrowing, accepted/rejected value domains and exact special cases. Check the current
FLOAT32 and FLOAT16 CPU emitter's double locals against the root's declared working sites and
correct demonstrated mismatches without weakening the root low-precision rule. Review attention
and loss references explicitly; do not generalize this permission to other composites by analogy.

Non-goals: new dtype/layout/provider/route support, public API, numerical profile, production
tolerance registry, broad performance campaign or unrelated attention/loss algorithm rewrite.

## Contracts and implementation packet

Read `ARCHITECTURE.md` headings `Authority, incorporation, and precedence`, `Core invariants`,
`Scope-indexed normative contracts`, `Documentation authority and architecture updates`, and
`Testing requirements`; `docs/architecture/contracts/foundational-modules.md` headings
`modules/model`, `Profile-free numerical semantics` (especially `Floating arithmetic primitives`,
`Aggregate floor`, `Composite inheritance`, `Model low-precision contract`); and
`docs/architecture/contracts/backend-execution.md` heading `CPU backend routes`.
If an applicable contract is ambiguous, stop and report it. Read `SoftmaxKind`, current CPU
emitter/validator/reference and focused tests.
Update root and the one owning scoped contract, family Javadoc, numerical explanation and a
significant ADR in one reviewed change. Add independent contract examples and backend-conformance
regressions for finite EXP overflow/underflow, cancellation-relevant offsets, both kinds and dtypes,
plus NaN/infinity/empty behavior. Review consumers and compatibility identities explicitly.
Keep test-only tolerances separate from class/guard semantics and retain clean Java code shape.

Files and symbols: `SoftmaxKind.java`; `CpuSoftmaxEmitter.java`,
`CpuSoftmaxReferenceKernel.java`, `CpuSoftmaxInputValidator.java` when correction is required;
their focused tests; relevant backend conformance/integration tests; `docs/api/tensor-api.md`;
and targeted numerical/CPU explanation and glossary entries. The implementation context owns
the root, scoped contract, ADR, source/tests and affected API docs. The main coordinator owns
the roadmap and master-plan status, which are separate write scopes.

Acceptance: finite `[1000,1000]` produces finite normalized outputs; offset and underflow cases
show LOG_SOFTMAX need not equal a rounded `log(softmax)`; all four admitted floating dtypes are
covered; Model's special-value/empty rules and CPU rejection-before-write behavior are distinct;
generated and clean Java sites align with the declared working precision; no per-element
allocation or dispatch is introduced. Record the result of attention/loss and CPU/Metal identity
review, including a reasoned no-change conclusion if appropriate.

Worker validation after focused development tests:

```bash
./gradlew :modules:model:test :backends:cpu:test :testing:backend-conformance:test :modules:model:javadoc :backends:cpu:javadoc --max-workers=2 --console=plain
```

The coordinator runs affected integration selectors and one repository checkpoint if executable
changes span Model and CPU; it reuses the worker's successful module evidence. Generated emitter
changes require class-shape inspection against direct Java. An independent Class C review inspects
the final code, contract, tests, Javadoc, explanatory docs, glossary, and links; it does not rerun
successful Java suites without a concrete reason.

## Dependencies and integration

- Depends on: owner decision above; Model 0032 Complete; review G1.
- Conflicts with: Model numerical semantics, softmax code/oracles and shared numerical docs.
- Parallel group: None; contract-first serial work.
- Common base revision: `4e5eb7baca6c16ddb02a05518d569e90aa557b76`.
- Integration order: decision/contract → scoped consumers and conformance → independent review.
- Integration validation: focused Model/CPU/conformance tests and Javadoc, affected integration
  selectors, and one repository-wide `./gradlew test javadoc --max-workers=2 --console=plain`
  checkpoint if executable changes cross module boundaries.
- Shared-document integration owner: main coordinator for roadmap/master-plan status.

Frontier verification (2026-10-13): Model 0032 records `Status: Complete`; the final project
review delivered G1 and the owner then selected stable finite semantics. Model 0033 is the sole
active Model frontier, no conflicting writer is active, and the base above is the clean source
revision before these planning edits. The implementation and coordinator write scopes above are
disjoint; contract and backend changes are serial within this one task.

## Result

Status: Complete. Stable finite SOFTMAX and LOG_SOFTMAX now have explicit Model authority,
explanatory documentation and ADR 0029. CPU generated and direct bodies use the declared
working precision, retain the prior non-finite result boundary and CPU pre-write rejection,
and bind version-two artifact identity. Backend conformance and Model special-case examples
cover the accepted finite and exceptional domains; current generated inventory and dependent
digest fixtures were refreshed without changing other families.

The affected Model, CPU and backend-conformance tests, CPU integration selector, one full
repository `test` checkpoint, and repository `javadoc` checkpoint passed. A bounded one-JVM
paired v2 diagnostic checked raw generated/direct equality for both kinds at FLOAT32 and
FLOAT16; its four observed generated/direct median ratios were 0.999465–1.021311x. This is
local evidence, not a universal performance guarantee. The initial independent Class C review
identified missing performance evidence and special-case examples; both were added and the
independent re-review returned `APPROVE` with no blocking findings. G1 is closed. No unresolved
task-specific follow-up remains; CPU 0023 and Training 0003A are separate Draft work.
