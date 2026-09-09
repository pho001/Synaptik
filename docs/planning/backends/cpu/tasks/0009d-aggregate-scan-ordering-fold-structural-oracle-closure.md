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
| 0009D2 scan | Complete | D1 | Retains the complete generated scan route: replacing typed carrier/layout bodies, BFLOAT16 per-value rounding, range ownership, and guarded segment specialization would increase implementation and verification work. |
| 0009D3 ordering | Complete | D2 | Retains complete generated SORT/ARGSORT/TOP_K: a direct replacement would duplicate the 192-row typed carrier/layout, stable-comparison, logical-index, scratch, cold-binding, and retirement-proof topology. |
| 0009D4 fold | Draft | D3 | Direct typed Java migration for finite fold/window accumulation. |

D4 is Draft and the next summary-only frontier, so this parent remains Ready. Create no detailed
D4 specification until it reaches the frontier. Each uses current clean Java/reference algorithms first and may inspect
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

- D1, D2, and D3 are Complete generated-route retention decisions; D4 is Draft and remains the
  next summary-only frontier.
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
or add a more complex partial split. D2 retains scan generation because its typed dense/general,
carrier, BFLOAT16, range, and guarded segment bodies are already one cohesive generated route;
direct replacement would multiply implementation and retirement-evidence work.

## Known limitations

No fresh performance claim follows from this plan. Final profiling remains optional unless a
concrete correctness regression is found.

## Validation evidence

D2's focused 65-test scan/lowering/preparation/generated-route command passed with zero failures,
errors, or skips. Documentation-focused context `/root/cpu_0009d2_docs` independently reviewed
the retained-route evidence and final planning diff, then checked links, headings used as anchors,
fences, terminology, scope/order/status synchronization, and `git diff --check`. No executable
behavior changed, so it did not repeat Java tests.

## Implementation notes

D2 and D3 retain generation without a performance claim. D4 remains Draft and summary-only, so
the parent remains Ready and D4 must not receive a detailed specification yet.

## Completion summary

- Completed changes: D1, D2, and D3 are Complete retained-generated-route decisions. D4 remains
  Draft and summary-only, so this parent remains Ready.
- Files changed or created: This parent and synchronized CPU 0009 planning records only.
- Tests and validation: Reused D3's recorded 66-test focused CPU evidence; this planning-only
  status correction checks task/master/roadmap synchronization, local Markdown links, headings
  used as anchors, fences, exact changed-path scope, stale status claims, and `git diff --check`.
- Documentation-agent review: Clean documentation-focused context finalized the planning records
  using the General and Planning documentation profiles.
- Documentation impact: Planning status records only; public/API Javadoc, guides, glossary,
  architecture/ADR, Gradle, conformance/integration, and other modules remain unchanged because
  Java behavior, contracts, and reusable terminology are unchanged.
- Javadoc review: No change; no Java contract or implementation changed.
- Glossary impact: No change; no reusable terminology changed.
- Unresolved issues: None.
- Follow-up required: Complete or otherwise resolve Draft D4 before completing this parent; do not
  create a detailed D4 specification until it becomes actionable.

Status: Ready
