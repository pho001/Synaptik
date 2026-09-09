# Task 0009D: Aggregate, Scan, Ordering, and Fold Direct-Java Migration Parent

## Status

Ready

## Goal

Sequence the static CPU families that are cheaper to maintain as direct, optimized typed Java than
as generated code plus increasingly formal structural evidence. This parent is orchestration only;
it does not reopen completed 0009A--C or D1A.

## Scope

The active family sequence is:

| Child | Status | Dependency | Decision |
|---|---|---|---|
| 0009D1 aggregate | Ready | 0009C, 0009D1A | Migrate selected aggregate routes to direct Java and retire their generated implementation/evidence after replacement. |
| 0009D2 scan | Draft | D1 | Direct typed Java migration for static scan recurrence and range-owned lines. |
| 0009D3 ordering | Draft | D2 | Direct typed Java migration for ordering/index/scratch topology. |
| 0009D4 fold | Draft | D3 | Direct typed Java migration for finite fold/window accumulation. |

Only D1 has a detailed specification. D2--D4 remain named summary tasks until they reach the
frontier. Each uses current clean Java/reference algorithms first and may inspect
`legacy/pre-rewrite` solely for capabilities, observable behavior, algorithm ideas, and tests;
none may copy legacy source, packages, dependencies, runtime coupling, or shortcuts.

## Out of scope

Universal structural verification, a performance ratio gate, a big-bang rewrite, production work
outside the active child, and a permanent generated/direct duplicate route.

## Architecture references and constraints

Preserve `ARCHITECTURE.md`: prepare selects a concrete typed kernel; worker/caller supplies ranges;
Runtime invokes prepared work. Direct kernels are finite typed methods/classes, not one generic
runtime-dispatch loop. Dispatch is cold; element loops have proportionate hot-loop hygiene.

## Acceptance criteria

- D1 is the sole detailed and actionable next task; D2--D4 are sequenced summary-only tasks.
- Each migration removes generated implementation and associated evidence only after replacement
  semantics, invocation/conformance, and no-selected-reference checks pass.
- Every migrated form is direct Java, retained bounded generated code, or an exact rejection; no
  permanent parallel implementation exists without a recorded reason.
- Existing performance evidence and NON_PASSING facts are retained, but new benchmarking does not
  block migration correctness.

## Tests / validation

This parent is planning-only. Validate task/master/roadmap status and dependency consistency,
Markdown links/anchors/fences, terminology, scope, and `git diff --check`. Children define focused
executable and documentation validation.

## Dependencies and follow-up tasks

0009D follows completed 0009C and D1A. D1 → D2 → D3 → D4, then 0009E. No D5 reconciliation gate
is planned: direct route selection and semantic/invocation coverage replace structural promotion.

## Architecture impact

Expected impact: None. This is a non-authoritative CPU implementation/evidence-plan correction.

## Implementation prompt

```text
Execute only the current detailed child. Preserve completed evidence, keep selection cold and loops
typed, use legacy only read-only as allowed evidence, and do not create later detailed specs early.
```

## Local decisions

The old 684-row structural-oracle denominator is historical accounting, not a completion criterion
for direct-Java migrations.

## Known limitations

No fresh performance claim follows from this plan. Final profiling remains optional unless a
concrete correctness regression is found.

## Validation evidence

Planning replan only; no executable behavior changed.

## Implementation notes

None.

## Completion summary

Status: Ready
