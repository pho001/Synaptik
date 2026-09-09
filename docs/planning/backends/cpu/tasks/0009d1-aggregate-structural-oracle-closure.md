# Task 0009D1: Aggregate Generated-Route Retention Decision

## Status

Complete

## Goal

Decide whether the ordinary aggregate family should replace its generated route at this CPU 0009
checkpoint. The controlling goal is to replace generated families only where doing so saves
implementation and verification time; this task establishes no new performance benefit or gate.

## Scope

Inspect the current ordinary aggregate implementation, its clean reference, and the completed D1A
evidence. Retain the complete ordinary aggregate family on its current generated route, including
SUM, MEAN, PROD, extrema, Boolean reductions, and SUM_TO_SHAPE, together with its existing semantic
and inventory evidence.

Two clean implementation attempts, contexts `01a085f7-3211-7763-8851-8c2de1bde597` and
`01a085fc-7908-7b30-9f62-e1628ddcac75`, are research and decision evidence only. Neither is a
successful implementation and neither changes production or test code.

## Out of scope

Direct-Java aggregate migration, generated-route retirement, structural-scaffolding retirement,
new aggregate performance evidence, public API or Javadoc changes, glossary changes, Gradle,
architecture documentation, and code or test changes.

## Architecture references and constraints

[`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) remains authoritative. Prepare retains selected
kernel ownership, Runtime invokes prepared work, and generated hot paths require proportionate
hygiene review. This local planning decision changes neither ownership nor dependency direction.

The current generated route is not an unsupported fallback: `CpuAggregateEmitter` owns the
ordinary aggregate matrix and delegates exact floating SUM/MEAN/PROD state transitions to
`CpuExactSumEmitter` and `CpuExactProductEmitter`. Those emitters use allocation-free
primitive-limb state and the established packed, invocation-private scratch contract. The
`CpuScalarReferenceKernel` instead uses `BigInteger` for independent differential semantics, so
it cannot be promoted to a hot production implementation under the architecture contract.

## Package impact

None. No source, test, resource, API, or module package changes are authorized.

## Affected files and maximum scope

This task changes only this planning record and synchronized CPU 0009 planning records: parent
0009, parent 0009D, detailed 0009D2, the CPU master plan, and the roadmap.

## Acceptance criteria

- The aggregate family is explicitly retained on its current generated route for this checkpoint.
- The decision records the two unsuccessful implementation contexts as research evidence, the
  exact-state/scratch rationale, and the reference-kernel `BigInteger` boundary.
- Completed D1A and all existing aggregate semantic and inventory evidence remain preserved.
- The decision makes no performance claim and adds no benchmark gate.
- CPU 0009D2 becomes the sole detailed `Ready` child; D3 and D4 remain summary-only.

## Tests / validation

No executable behavior changed. Reuse the pre-decision CPU baseline: 930 tests passed, 28 skipped,
and 0 failed; the focused aggregate baseline also passed. Validate planning status/order/link
consistency, Markdown links, anchors and fences, exact changed-path scope, and `git diff --check`.

## Dependencies and follow-up tasks

Completed 0009A--C and 0009D1A remain historical evidence. CPU 0009D2 scan is now the next
detailed `Ready` task. D3 ordering follows D2; D4 fold follows D3; CPU 0009E follows D4.

## Architecture impact

None. This is a non-authoritative retained-route decision, not an architecture change.

## Local decisions

The aggregate route is retained as a reasoned hybrid decision. `CpuAggregateEmitter` (1,747
lines), `CpuExactSumEmitter` (446 lines), and `CpuExactProductEmitter` (785 lines) collectively
own the exact numerical machinery. Porting that full matrix would duplicate the most complex
numerical mechanism; a partial direct/generated split would retain generator and evidence
complexity while adding routing complexity. Retention therefore saves implementation and
verification effort at this checkpoint without asserting that it is faster.

## Known limitations

No fresh aggregate performance evidence is supplied. Existing aggregate semantics, inventory, and
D1A hygiene evidence remain the applicable evidence for the retained route.

## Validation evidence

Documentation/planning context `/root` read AGENTS.md, the architecture contract and current
architecture index, Planning Guide, documentation rules with General and Planning profiles, CPU
0009 parent/D/D1/D1A, CPU master plan, and roadmap. It inspected the aggregate emitter, exact sum
and product emitters, scalar reference kernel, and scan emitter. It records the two clean failed
implementation contexts as decision research, not implementation success. No executable command
was rerun because no Java, test, resource, or behavior changed; the recorded 930-pass/28-skipped
CPU baseline and focused aggregate baseline predate this planning decision and remain unchanged.

## Implementation notes

No implementation was attempted or accepted by this task. The prior direct-migration and
generated-retirement plan is superseded by the retained-route decision above.

## Completion summary

- Completed changes: Closed D1 as an explicit retained generated-route decision for ordinary
  aggregates and advanced the detailed frontier to D2 scan.
- Files changed or created: This task and synchronized CPU 0009 planning records only.
- Tests and validation: Reused the recorded pre-decision 930-pass, 28-skipped, zero-failure CPU
  baseline and focused aggregate baseline; planning static checks are recorded with this change.
- Documentation-agent review: This clean documentation/planning context finalized the decision.
- Documentation impact: No public API/Javadoc or explanatory-guide change; no public behavior,
  workflow, or terminology changed.
- Javadoc review: No change; no Java contract changed.
- Glossary impact: No change; no reusable term or meaning changed.
- Unresolved issues: None.
- Follow-up required: Execute CPU 0009D2's explicit scan decision gate.

Status: Complete
