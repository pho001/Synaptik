# Task 0009D2: Scan Route Decision and Possible Direct-Java Migration

## Status

Ready

## Goal

Decide the least costly maintainable implementation route for the ordinary cumulative-scan family.
Use finite direct typed Java only if focused inspection establishes that it is genuinely simpler to
implement and verify than retaining the current generated route; otherwise retain generation with
an explicit recorded decision. No performance benefit or new benchmark gate is required.

## Scope

Inspect the current scan lowering, preparation/invocation boundary, `CpuScanEmitter`, semantic and
inventory evidence, and the clean/reference implementation used by current tests. The
`legacy/pre-rewrite` branch is read-only behavioral/test evidence only; do not copy its source,
packages, dependencies, runtime coupling, or shortcuts. Cover current
CUM_SUM and CUM_PROD forms across admitted types, axes, inclusive/exclusive and forward/reverse
modes, carriers, layouts, and caller-owned complete-slice ranges.

The first implementation step is a decision gate. It must compare the full supported matrix,
typed carrier/address routes, BFLOAT16 conversion-after-each-value semantics, integral wrap
semantics, range ownership, and existing generated inventory/evidence. Direct migration is
authorized only when that inspection proves a cohesive finite direct design can replace the full
selected generated route with less implementation and verification work. Otherwise record
retention, preserve the generated route and all evidence, and complete the task without code
changes.

## Out of scope

Aggregate work and D1 evidence; ordering, fold, general reductions, normalization, loss, pooling,
attention, batch normalization, MATMUL, convolution, a universal structural verifier, a required
performance study or benchmark gate, architecture/module-boundary changes, and a permanent
direct/generated duplicate route.

## Architecture references and constraints

[`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) is authoritative. Read it with the [current
architecture index](../../../../architecture/current-architecture-plan.md), [Planning
Guide](../../../planning-guide.md), [CPU master plan](../master-plan.md), parent
[CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), and parent
[CPU 0009D](0009d-aggregate-scan-ordering-fold-structural-oracle-closure.md). Prepare selects a
concrete typed implementation; Runtime invokes prepared work; ranges own complete scan slices;
dispatch, carrier selection, geometry validation, and partitioning stay cold. Element loops must
not allocate, box, reflect, synchronize, use string/map dispatch, or make avoidable
virtual/interface calls. A retained generated route remains supported only with its current
proportionate hygiene, semantic, and inventory evidence.

## Package impact

The decision-only outcome changes planning records only. If the gate authorizes direct migration,
the task must first name the exact existing CPU-private lowering, prepare, executable, generator,
and test packages it changes; no public package, dependency, or architecture change is allowed.

## Affected files and maximum scope

Decision-only outcome: this task, parent 0009D, parent 0009, CPU master plan, and roadmap. A
direct-migration outcome may change only the CPU-private owners and focused tests/resources proved
necessary by the decision, plus those planning records; stop and replan if it crosses a public or
shared boundary or needs more than 18 CPU production/test/resource paths.

## Acceptance criteria

- The decision gate documents the inspected scan matrix and concludes either a full direct route
  with less implementation/verification work or an explicit retained generated route.
- A direct route, if authorized, preserves selected typed carrier/address entries, complete-slice
  range ownership, all current scan semantics and exact rejections, and has no selected generated
  scan route after replacement validation.
- A retained route preserves existing scan semantics, inventory, and hygiene evidence without
  unsupported or performance claims.
- No partial direct/generated split is retained unless the task records a concrete lower-complexity
  reason and exact route/evidence ownership.
- D3 remains summary-only until D2 is complete.

## Tests / validation

Before deciding, run or reuse current focused scan lowering, preparation/invocation, semantic, and
generated-route hygiene evidence, recording exact commands and results. A direct-migration outcome
adds focused route-selection, semantic/canary, invocation/range, and retirement checks, then runs
the CPU module suite once after final executable edits. A retained outcome runs no new benchmark;
it validates the decision record, selected-route references, Markdown links/anchors/fences, status
and order, exact changed-path scope, and `git diff --check`.

## Dependencies and follow-up tasks

CPU 0009D1 is Complete as an aggregate retained-route decision; D2 is the current detailed
frontier. D3 ordering follows only after D2 completes, D4 fold follows D3, and CPU 0009E follows
D4.

## Architecture impact

Expected impact: None. Stop and report any required public, dependency, module-boundary, or
architecture-contract change.

## Implementation prompt

```text
Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, Planning Guide, documentation
rules with General and Planning profiles, CPU master plan, CPU 0009/D/D1, and this task. Implement
CPU 0009D2 exactly. First perform and record the scan route decision gate. Migrate to finite direct
typed Java only if the full current scan matrix is genuinely cheaper to implement and verify than
retained generation; otherwise retain the generated route explicitly. Preserve cold selection,
complete-slice range ownership, semantics, and existing evidence. Do not create D3 details, add a
benchmark gate, change architecture, commit, or push. Hand any executable diff and evidence to a
separate clean documentation-focused context.
```

## Local decisions

None until the route-decision gate is completed.

## Known limitations

No new performance claim follows from either route decision. Existing performance facts remain
historical evidence.

## Validation evidence

Empty until implemented.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
