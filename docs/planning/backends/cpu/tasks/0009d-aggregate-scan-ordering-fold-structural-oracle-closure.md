# Task 0009D: Aggregate, Scan, Ordering, and Fold Route-Closure Parent

## Status

Ready

## Goal

Sequence the static CPU families through explicit lowest-cost route decisions. Direct typed Java is
used only when it saves implementation and verification work; otherwise a supported generated
route is retained. This parent is orchestration only; it does not reopen completed 0009A--C or D1A.

## Scope

The active family sequence is:

| Child | Status | Dependency | Decision |
|---|---|---|---|
| 0009D1 aggregate | Complete | 0009C, 0009D1A | Retain the complete ordinary aggregate family on its supported generated route; the direct migration attempts are decision research, not implementation. |
| 0009D2 scan | Ready | D1 | Detailed route-decision gate: migrate only if full direct typed Java is genuinely cheaper to implement and verify; otherwise retain generation explicitly. |
| 0009D3 ordering | Draft | D2 | Direct typed Java migration for ordering/index/scratch topology. |
| 0009D4 fold | Draft | D3 | Direct typed Java migration for finite fold/window accumulation. |

Only D2 has a detailed specification. D3--D4 remain named summary tasks until they reach the
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

- D1 is Complete as a generated-route retention decision; D2 is the sole detailed and actionable
  next task, while D3--D4 are sequenced summary-only tasks.
- Each family has an explicit direct-migration or retained-generated-route decision. Migration
  removes generated implementation and associated evidence only after replacement semantics,
  invocation/conformance, and no-selected-reference checks pass.
- Every form is direct Java, retained bounded generated code, or an exact rejection; no permanent
  parallel implementation exists without a recorded reason.
- Existing performance evidence and NON_PASSING facts are retained, but new benchmarking does not
  block migration correctness.

## Tests / validation

This parent is planning-only. Validate task/master/roadmap status and dependency consistency,
Markdown links/anchors/fences, terminology, scope, and `git diff --check`. Children define focused
executable and documentation validation.

## Dependencies and follow-up tasks

0009D follows completed 0009C and D1A. D1 → D2 → D3 → D4, then 0009E. No D5 reconciliation gate
is planned: explicit route decisions and semantic/invocation coverage replace structural promotion.

## Architecture impact

Expected impact: None. This is a non-authoritative CPU implementation/evidence-plan correction.

## Implementation prompt

```text
Execute only the current detailed child. Preserve completed evidence, keep selection cold and loops
typed, use legacy only read-only as allowed evidence, and do not create later detailed specs early.
```

## Local decisions

The old 684-row structural-oracle denominator is historical accounting, not a route-decision
criterion. D1 retains aggregate generation because migration would duplicate exact-state machinery
or add a more complex partial split.

## Known limitations

No fresh performance claim follows from this plan. Final profiling remains optional unless a
concrete correctness regression is found.

## Validation evidence

Planning replan only; no executable behavior changed.

## Implementation notes

None.

## Completion summary

Status: Ready
