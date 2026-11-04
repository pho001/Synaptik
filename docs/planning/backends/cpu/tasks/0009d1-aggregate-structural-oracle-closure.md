# Task 0009D1: Aggregate Generated-Route Retention Decision

## Reader context

This task records retaining ordinary aggregate generation rather than porting `SUM`, `MEAN`, `PROD`,
extrema, Boolean reductions, and SUM_TO_SHAPE to direct Java. `CpuAggregateEmitter` owns the family
matrix and delegates exact floating arithmetic to `CpuExactSumEmitter` and `CpuExactProductEmitter`.
Their primitive-limb state represents exact numerical accumulation without promoting the
BigInteger-based `CpuScalarReferenceKernel` into production.

Here, the reference kernel supplies independent semantic comparisons, while inventory and structural
evidence describe generated artifacts and implementation shape. Prepare retains selected-kernel
ownership and Runtime invokes prepared work. Packed scratch is private to an invocation, not
persistent model state. The two unsuccessful direct-migration contexts support the effort decision
only; this planning task accepts no executable replacement and makes no speed claim.

Current-topic navigation (not a replacement for the historical record):

- [current selected-domain reduction meanings](../../../../reference/tensor/reductions.md#multi-axis-and-statistical-reduction-expressions) — Separates axes, correction, norms, and numerical semantics from historical CPU algorithm choices.
- [backend ownership versus implementation route](../../../../design/notes/cpu-kernel-strategy.md#strategy) — Separates Planning ownership, CPU route choice, finalization, and Runtime invocation.

## Status

Complete

## Goal

Decide whether the ordinary aggregate family should replace its generated route at this [CPU](../../../../glossary.md#term-cpu) 0009
[checkpoint](../../../../glossary.md#term-checkpoint). The controlling goal is to replace generated families only where doing so saves
implementation and verification time; this task establishes no new performance benefit or gate.

## [Scope](../../../../glossary.md#term-scope)

Inspect the current ordinary aggregate implementation, its clean reference, and the completed D1A
evidence. Retain the complete ordinary aggregate family on its current generated route, including
SUM, [MEAN](../../../../glossary.md#term-mean), PROD, [extrema](../../../../glossary.md#saturation-pole-and-extremum), [Boolean](../../../../glossary.md#data-type--datatype) reductions, and SUM_TO_SHAPE, together with its existing semantic
and inventory evidence.

Two clean implementation attempts, contexts `01a085f7-3211-7763-8851-8c2de1bde597` and
`01a085fc-7908-7b30-9f62-e1628ddcac75`, are research and decision evidence only. Neither is a
successful implementation and neither changes production or test code.

## Out of [scope](../../../../glossary.md#term-scope)

Direct-Java aggregate migration, generated-route retirement, structural-scaffolding retirement,
new aggregate performance evidence, public [API](../../../../glossary.md#term-application-programming-interface-api) or [Javadoc](../../../../glossary.md#term-javadoc) changes, glossary changes, [Gradle](../../../../glossary.md#term-gradle),
architecture documentation, and code or test changes.

## Architecture references and constraints

[`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) remains authoritative. [Prepare](../../../../glossary.md#prepare) retains selected
[kernel](../../../../glossary.md#kernel) [ownership](../../../../glossary.md#term-ownership-meanings), [Runtime](../../../../glossary.md#term-runtime) invokes prepared work, and generated hot paths require proportionate
hygiene review. This local [planning](../../../../glossary.md#term-planning) decision changes neither ownership nor [dependency direction](../../../../glossary.md#java-platform-and-build-tools).

The current generated route is not an unsupported fallback: `CpuAggregateEmitter` owns the
ordinary aggregate [matrix](../../../../glossary.md#term-matrix-meanings) and delegates exact floating SUM/[MEAN](../../../../glossary.md#term-mean)/PROD state transitions to
`CpuExactSumEmitter` and `CpuExactProductEmitter`. Those emitters use allocation-free
primitive-[limb](../../../../glossary.md#exact-superaccumulator-and-limbs) state and the established packed, invocation-private scratch contract. The
`CpuScalarReferenceKernel` instead uses `BigInteger` for independent differential semantics, so
it cannot be promoted to a hot production implementation under the [architecture contract](../../../../glossary.md#architecture-contract).

## Package impact

None. No source, test, resource, [API](../../../../glossary.md#term-application-programming-interface-api), or [module](../../../../glossary.md#term-module) package changes are authorized.

## Affected files and maximum [scope](../../../../glossary.md#term-scope)

This task changes only this [planning](../../../../glossary.md#term-planning) record and synchronized [CPU](../../../../glossary.md#term-cpu) 0009 planning records: parent
0009, parent 0009D, detailed 0009D2, the CPU master plan, and the roadmap.

## Acceptance criteria

- The aggregate family is explicitly retained on its current generated route for this [checkpoint](../../../../glossary.md#term-checkpoint).
- The decision records the two unsuccessful implementation contexts as research evidence, the
  exact-state/scratch rationale, and the reference-kernel <span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>`BigInteger` boundary.
- Completed D1A and all existing aggregate semantic and inventory evidence remain preserved.
- The decision makes no performance claim and adds no benchmark gate.
- [CPU](../../../../glossary.md#term-cpu) 0009D2 becomes the sole detailed [`Ready`](../../../../glossary.md#term-ready) child; D3 and D4 remain summary-only.

## Tests / [validation](../../../../glossary.md#term-validation)

No executable behavior changed. Reuse the pre-decision [CPU](../../../../glossary.md#term-cpu) baseline: 930 tests passed, 28 skipped,
and 0 failed; the focused aggregate baseline also passed. Validate [planning](../../../../glossary.md#term-planning) status/order/link
consistency, Markdown links, anchors and fences, exact changed-path [scope](../../../../glossary.md#term-scope), and `git diff --check`.

## Dependencies and follow-up tasks

Completed 0009A--C and 0009D1A remain historical evidence. [CPU](../../../../glossary.md#term-cpu) 0009D2 [scan](../../../../glossary.md#term-cumulative-scan) is now the next
detailed [`Ready`](../../../../glossary.md#term-ready) task. D3 ordering follows D2; D4 fold follows D3; CPU 0009E follows D4.

## Architecture impact

None. This is a non-authoritative retained-route decision, not an architecture change.

## Local decisions

The aggregate route is retained as a reasoned hybrid decision. `CpuAggregateEmitter` (1,747
lines), `CpuExactSumEmitter` (446 lines), and `CpuExactProductEmitter` (785 lines) collectively
own the exact numerical machinery. Porting that full [matrix](../../../../glossary.md#term-matrix-meanings) would duplicate the most complex
numerical mechanism; a partial direct/generated split would retain generator and evidence
complexity while adding routing complexity. Retention therefore saves implementation and
verification effort at this [checkpoint](../../../../glossary.md#term-checkpoint) without asserting that it is faster.

## Known limitations

No fresh aggregate performance evidence is supplied. Existing aggregate semantics, inventory, and
D1A hygiene evidence remain the applicable evidence for the retained route.

## [Validation](../../../../glossary.md#term-validation) evidence

Documentation/[planning](../../../../glossary.md#term-planning) context `/root` read AGENTS.md, the [architecture contract](../../../../glossary.md#architecture-contract) and current
architecture index, Planning Guide, documentation rules with General and Planning profiles, [CPU](../../../../glossary.md#term-cpu)
0009 parent/D/D1/D1A, CPU master plan, and roadmap. It inspected the aggregate emitter, exact sum
and product emitters, [scalar](../../../../glossary.md#term-scalar) reference [kernel](../../../../glossary.md#kernel), and [scan](../../../../glossary.md#term-cumulative-scan) emitter. It records the two clean failed
implementation contexts as decision research, not implementation success. No executable command
was rerun because no Java, test, resource, or behavior changed; the recorded 930-pass/28-skipped
CPU baseline and focused aggregate baseline predate this planning decision and remain unchanged.

## Implementation notes

No implementation was attempted or accepted by this task. The prior direct-migration and
generated-retirement plan is superseded by the retained-route decision above.

## Completion summary

- Completed changes: Closed D1 as an explicit retained generated-route decision for ordinary
  aggregates and advanced the detailed frontier to D2 scan.
- Files changed or created: This task and synchronized [CPU](../../../../glossary.md#term-cpu) 0009 [planning](../../../../glossary.md#term-planning) records only.
- Tests and validation: Reused the recorded pre-decision 930-pass, 28-skipped, zero-failure CPU
  baseline and focused aggregate baseline; planning static checks are recorded with this change.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Documentation-agent review: This clean documentation/planning context finalized the decision.
- Documentation impact: No public API/Javadoc or explanatory-guide change; no public behavior,
  workflow, or terminology changed.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: No change; no Java contract changed.
- Glossary impact: No change; no reusable term or meaning changed.
- Unresolved issues: None.
- Follow-up required: Execute CPU 0009D2's explicit [scan](../../../../glossary.md#term-cumulative-scan) decision gate.

Status: Complete
