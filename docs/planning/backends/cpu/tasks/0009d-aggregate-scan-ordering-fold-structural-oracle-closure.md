# Task 0009D: Aggregate, Scan, Ordering, and Fold Route-Closure Parent

## Status

Complete

## Goal

Sequence the static CPU families through explicit lowest-cost route decisions. Direct typed Java is
used only when it saves implementation and verification work; otherwise a supported generated
route is retained. This parent is orchestration only; it does not reopen completed 0009A--C or D1A.

## Scope

The active family sequence is:

| Child | Status | Dependency | Decision |
|---|---|---|---|
| 0009D1 aggregate | Complete | 0009C, 0009D1A | Retain the complete ordinary aggregate family on its supported generated route; the direct migration attempts are decision research, not implementation. |
| 0009D2 scan | Complete | D1 | Retains the complete generated scan route: replacing typed carrier/layout bodies, BFLOAT16 per-value rounding, range ownership, and guarded segment specialization would increase implementation and verification work. |
| 0009D3 ordering | Complete | D2 | Retains complete generated SORT/ARGSORT/TOP_K: a direct replacement would duplicate the 192-row typed carrier/layout, stable-comparison, logical-index, scratch, cold-binding, and retirement-proof topology. |
| 0009D4 fold | Complete | D3 | Retained the complete 32-row generated FOLD_AXIS/FOLD2D route: direct Java would duplicate typed mappings, carriers, dense/general forms, cold binding, numeric rules, ranges, and replacement evidence. |

Each child uses current clean Java/reference algorithms first and may inspect
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

- D1, D2, D3, and D4 are Complete generated-route retention decisions.
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

0009D follows completed 0009C and D1A. D1 → D2 → D3 → D4, then summary-only 0009E. No D5 reconciliation gate
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
or add a more complex partial split. D2 retains scan generation because its typed dense/general,
carrier, BFLOAT16, range, and guarded segment bodies are already one cohesive generated route;
direct replacement would multiply implementation and retirement-evidence work. D3 retains the
192-row ordering route for the same total-work reason. D4 retains the entire 32-row fold route:
a finite direct replacement would duplicate five typed axis and three typed 2D bodies over four
carrier combinations, dense/general addressing, cold selection/binding, output ranges, exact
mapping/rejection, BFLOAT16 per-add rounding, integral wrap, and retirement proof.

## Known limitations

No fresh performance claim follows from this plan. Final profiling remains optional unless a
concrete correctness regression is found.

## Validation evidence

The D4 implementation context's specified eight-class fold command passed on 2026-09-09: 131
tests, zero failures, zero errors, and zero skips. It covered 3 lowering, 2 IR, 12 generated-fold,
2 semantic-closure, 29 preparer, 16 finalizer, 58 executable, and 9 checkpoint tests.
Documentation-focused context `/root/cpu_0009d4_docs` independently reviewed the complete retained
fold route and final planning diff, then checked links, heading anchors, fences, terminology,
scope/order/status synchronization, exact five-path scope, and `git diff --check`. No executable
behavior changed, so it reused the successful Java evidence and ran neither Java tests nor Javadoc.

## Implementation notes

D1 through D4 retain generation without a performance claim. D4's all-or-nothing decision keeps
the current coherent fold route selected; CPU 0009E is the summary-only next frontier.

## Completion summary

- Completed changes: D1 through D4 are Complete retained-generated-route decisions. D4 retains
  the complete 32-row generated fold family because direct Java would increase total work.
- Files changed or created: This parent and synchronized CPU 0009 planning records only.
- Tests and validation: Reused D4's recorded eight-class 131-test focused CPU evidence with zero
  failures, errors, or skips; this planning-only correction checks synchronization, Markdown
  links/anchors/fences, exact changed-path scope, stale status claims, and `git diff --check`.
- Documentation-agent review: Clean context `/root/cpu_0009d4_docs` finalized the planning
  records using the General and Planning documentation profiles.
- Documentation impact: Planning status records only; public/API Javadoc, guides, glossary,
  architecture/ADR, Gradle, conformance/integration, and other modules remain unchanged because
  Java behavior, contracts, and reusable terminology are unchanged.
- Javadoc review: No change; no Java contract or implementation changed.
- Glossary impact: No change; no reusable terminology changed.
- Unresolved issues: None.
- Follow-up required: CPU 0009E remains summary-only until it is the actionable frontier.

Status: Complete
