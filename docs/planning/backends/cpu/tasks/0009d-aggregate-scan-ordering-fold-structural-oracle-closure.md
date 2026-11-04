# Task 0009D: Aggregate, [Scan](../../../../glossary.md#term-cumulative-scan), Ordering, and Fold Route-Closure Parent

## Status

Complete

## Goal

Sequence the static [CPU](../../../../glossary.md#term-cpu) families through explicit lowest-cost route decisions. Direct typed Java is
used only when it saves implementation and verification work; otherwise a supported generated
route is retained. This parent is orchestration only; it does not reopen completed 0009A--C or D1A.

## [Scope](../../../../glossary.md#term-scope)

The active family sequence is:

| Child | Status | [Dependency](../../../../glossary.md#term-dependency) | Decision |
|---|---|---|---|
| 0009D1 aggregate | Complete | 0009C, 0009D1A | Retain the complete ordinary aggregate family on its supported generated route; the direct migration attempts are decision research, not implementation. |
| 0009D2 [scan](../../../../glossary.md#term-cumulative-scan) | Complete | D1 | Retains the complete generated scan route: replacing typed [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout) bodies, [BFLOAT16](../../../../glossary.md#term-bfloat16) per-value [rounding](../../../../glossary.md#term-rounding), range [ownership](../../../../glossary.md#term-ownership-meanings), and guarded segment [specialization](../../../../glossary.md#term-specialization) would increase implementation and verification work. |
| 0009D3 ordering | Complete | D2 | Retains complete generated SORT/ARGSORT/TOP_K: a direct replacement would duplicate the 192-row typed carrier/layout, stable-comparison, logical-index, scratch, cold-[binding](../../../../glossary.md#term-binding), and retirement-proof [topology](../../../../glossary.md#term-topology). |
| 0009D4 fold | Complete | D3 | Retained the complete 32-row generated FOLD_AXIS/FOLD2D route: direct Java would duplicate typed mappings, carriers, dense/general forms, cold binding, numeric rules, ranges, and replacement evidence. |

Each child uses current clean Java/reference algorithms first and may inspect
`legacy/pre-rewrite` solely for capabilities, observable behavior, algorithm ideas, and tests;
none may copy legacy source, packages, dependencies, [runtime](../../../../glossary.md#term-runtime) coupling, or shortcuts.

## Out of [scope](../../../../glossary.md#term-scope)

Universal structural verification, a performance ratio gate, a big-bang rewrite, production work
outside the active child, and a permanent generated/direct duplicate route.

## Architecture references and constraints

Preserve `ARCHITECTURE.md`: [prepare](../../../../glossary.md#prepare) selects a concrete typed [kernel](../../../../glossary.md#kernel); worker/caller supplies ranges;
[Runtime](../../../../glossary.md#term-runtime) invokes prepared work. Direct [kernels](../../../../glossary.md#kernel) are [finite](../../../../glossary.md#term-finite) typed methods/classes, not one generic
runtime-[dispatch](../../../../glossary.md#term-dispatch) loop. Dispatch is cold; [element](../../../../glossary.md#term-element) loops have proportionate hot-loop hygiene.

## Acceptance criteria

- D1, D2, D3, and D4 are Complete generated-route retention decisions.
- Each family has an explicit direct-migration or retained-generated-route decision. Migration
  removes generated implementation and associated evidence only after replacement semantics,
  invocation/conformance, and no-selected-reference checks pass.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- Every form is direct Java, retained bounded generated code, or an exact rejection; no permanent
  parallel implementation exists without a recorded reason.
- Existing performance evidence and NON_PASSING facts are retained, but new benchmarking does not
  block migration correctness.<span data-glossary-navigation="true"> ([benchmarking glossary](../../../../glossary.md#term-benchmarking))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

This parent is [planning](../../../../glossary.md#term-planning)-only. Validate task/master/roadmap status and [dependency](../../../../glossary.md#term-dependency) consistency,
Markdown links/anchors/fences, terminology, [scope](../../../../glossary.md#term-scope), and `git diff --check`. Children define focused
executable and documentation validation.

## Dependencies and follow-up tasks

0009D follows completed 0009C and D1A. D1 → D2 → D3 → D4, then summary-only 0009E. No D5 reconciliation gate
is planned: explicit route decisions and semantic/invocation coverage replace structural promotion.

## Architecture impact

Expected impact: None. This is a non-authoritative [CPU](../../../../glossary.md#term-cpu) implementation/evidence-plan correction.

## Implementation prompt

```text
Execute only the current detailed child. Preserve completed evidence, keep selection cold and loops
typed, use legacy only read-only as allowed evidence, and do not create later detailed specs early.
```

## Local decisions

The old 684-row structural-[oracle](../../../../glossary.md#term-test-oracle) denominator is historical accounting, not a route-decision
criterion. D1 retains aggregate generation because migration would duplicate exact-state machinery
or add a more complex partial split. D2 retains [scan](../../../../glossary.md#term-cumulative-scan) generation because its typed dense/general,
[carrier](../../../../glossary.md#term-storage-carrier), [BFLOAT16](../../../../glossary.md#term-bfloat16), range, and guarded segment bodies are already one cohesive generated route;
direct replacement would multiply implementation and retirement-evidence work. D3 retains the
192-row ordering route for the same total-work reason. D4 retains the entire 32-row fold route:
a [finite](../../../../glossary.md#term-finite) direct replacement would duplicate five typed [axis](../../../../glossary.md#term-axis) and three typed 2D bodies over four
carrier combinations, dense/general addressing, cold selection/[binding](../../../../glossary.md#term-binding), output ranges, exact
mapping/rejection, BFLOAT16 per-add [rounding](../../../../glossary.md#term-rounding), integral wrap, and retirement proof.

## Known limitations

No fresh performance claim follows from this plan. Final profiling remains optional unless a
concrete correctness [regression](../../../../glossary.md#term-regression) is found.

## [Validation](../../../../glossary.md#term-validation) evidence

The D4 implementation context's specified eight-class fold command passed on 2026-09-09: 131
tests, zero failures, zero errors, and zero skips. It covered 3 [lowering](../../../../glossary.md#lowering), 2 [IR](../../../../glossary.md#term-intermediate-representation-ir), 12 generated-fold,
2 semantic-closure, 29 preparer, 16 finalizer, 58 executable, and 9 [checkpoint](../../../../glossary.md#term-checkpoint) tests.
Documentation-focused context `/root/cpu_0009d4_docs` independently reviewed the complete retained
fold route and final [planning](../../../../glossary.md#term-planning) diff, then checked links, heading anchors, fences, terminology,
[scope](../../../../glossary.md#term-scope)/order/status [synchronization](../../../../glossary.md#term-synchronization), exact five-path scope, and `git diff --check`. No executable
behavior changed, so it reused the successful Java evidence and ran neither Java tests nor [Javadoc](../../../../glossary.md#term-javadoc).

## Implementation notes

D1 through D4 retain generation without a performance claim. D4's all-or-nothing decision keeps
the current coherent fold route selected; [CPU](../../../../glossary.md#term-cpu) 0009E is the summary-only next frontier.

## Completion summary

- Completed changes: D1 through D4 are Complete retained-generated-route decisions. D4 retains
  the complete 32-row generated fold family because direct Java would increase total work.
- Files changed or created: This parent and synchronized [CPU](../../../../glossary.md#term-cpu) 0009 [planning](../../../../glossary.md#term-planning) records only.
- Tests and validation: Reused D4's recorded eight-class 131-test focused CPU evidence with zero
  failures, errors, or skips; this planning-only correction checks synchronization, Markdown
  links/anchors/fences, exact changed-path scope, stale status claims, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [synchronization glossary](../../../../glossary.md#term-synchronization); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`.
- Documentation-agent review: Clean context `/root/cpu_0009d4_docs` finalized the planning
  records using the General and Planning documentation profiles.
- Documentation impact: Planning status records only; public/API Javadoc, guides, glossary,
  architecture/ADR, Gradle, conformance/integration, and other modules remain unchanged because
  Java behavior, contracts, and reusable terminology are unchanged.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: No change; no Java contract or implementation changed.
- Glossary impact: No change; no reusable terminology changed.
- Unresolved issues: None.
- Follow-up required: CPU 0009E remains summary-only until it is the actionable frontier.

Status: Complete
