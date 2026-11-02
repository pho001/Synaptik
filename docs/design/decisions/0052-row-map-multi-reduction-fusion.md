# ADR 0052: Share row maps across multiple terminal CPU reductions

## Status

Accepted — 2026-10-30 user approval; implemented by CPU0073, independently reviewed and
MAIN accepted on 2026-10-30. This ADR records
rationale, not authority; the [architecture root](../../../ARCHITECTURE.md) and its incorporated
[CPU mechanism contract](../../architecture/contracts/backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)
own the exact approved packet. Implementation evidence is retrospective and recorded in the
[CPU0073 result](../../planning/backends/cpu/tasks/0073-row-map-multi-reduction-fusion.md#result);
performance remains unmeasured.

## Context

Existing operation emission shares pointwise, MATMUL and represented-SUM mathematics, but
separate whole-domain stages can repeatedly traverse rows and materialize a common producer.
The bounded backward-code comparison motivates sharing that producer with multiple terminal
outputs. A row map means pure ordered local work per row; a terminal consumes rows and cannot
feed that same pass. The small comparison is not general correctness or timing evidence.

## Decision drivers

- Preserve represented arithmetic, saved/public uses, aliases, activity and resource lifetimes.
- Reuse current operation math through both ordinary and Scan callers, without a second executor.
- Bound cold proofs and live locals; retain fallbacks and literal noneligible artifacts.
- Distinguish exact ordinary SUM from Scan's represented algorithm and avoid capability inflation.

## Options considered

### Keep separate complete traversals

This preserves current code and resources but leaves common row work/staging intact. It remains
the fallback for ineligible groups, not the selected solution for proved eligible regions.

### Add an RNN-specific backward kernel or runtime coordinator

This could target the observed case, but duplicates semantics or introduces execution ownership
and would not solve ordinary dense-gradient sharing. It is rejected.

### Derive one bounded CPU row region using current math owners

This shares a proved row map across row outputs and multiple terminals. It needs explicit
legality/resource proof, generation-local lifetime ownership and selective artifact identities.
It is selected; exact names/signatures/budgets live only in the owning contract.

## Decision and rationale

Select the common cold IR/proof/emission mechanism in the CPU owner. Ordinary graphs lower to
one full-body work item; Scan wraps generation inside its existing time/activity loop. The
region is physical lowering of an unchanged semantic DAG, not a new Model operation, Compiler
graph, scheduler or Runtime/Engine/Prepare protocol. Existing math owners retain algorithms;
ordinary exact SUM and its required producer storage/workspace remain separate.

For example, a dense gradient may share D between row projections and `X.T@D`, while preserving
D for separate exact `db=SUM_R(D)`. Scan may share D with represented terminal bias SUM but must
perform the shared carry ADD only after row reduction. A terminal feeding row work cannot fuse
in that pass. This addresses general sharing without a model-specific shortcut or new numerics.

## Consequences

### Positive

Eligible groups can avoid repeated row work and unobserved intermediate traffic in both callers.
No new semantic operation, dtype/layout capability, executor or numerical policy is required.
These are design opportunities, not measured speedups or guaranteed no-copy behavior.

### Negative and risks

Complete uses, dependency convexity, named axes, physical aliases and inclusive lifetimes make
cold admission more demanding. Bounded features/live locals deliberately reject some groups.
Saved or external uses can retain materialization. Shared generation seams must preserve old
fallback byte/local order; selected groups cannot reuse incompatible unfused artifacts.

### Migration, testing and follow-up

The owning contract reserves only selected ordinary projection 72 and selective ordinary/Scan
row-fusion-v1 suffixes. Global envelope 71 and Scan projection 71 remain; noneligible controls
retain literal keys/ClassFiles/resources. Both callers qualified before eligible traversal glue
retired; fallback math, temporal/history ownership and required ordinary D stores remain.
CPU0073 provides durable actual-AD, empty/range/activity/alias/resource tests and specialized clean
Java/generated-ClassFile inspection. Its implemented selection requires a computed row producer
shared by a terminal and another required sink, not direct-input/movement sharing alone.
Independent code/tests/Javadoc review closed all findings; MAIN's 42-test join passed. The task
distinguishes the failed full-module run from passing repairs and approved unchanged-outcome
transfer. Finite retention evidence authenticates 356 pre-edit payloads and seven selected reverse
retirements; the missing initial case-level diagnostic is not reconstructed. These results establish
bounded implementation, not speed or universal artifact preservation.
All old timing budgets are closed; measurements require
a fresh explicit finite grant. Across-time carry promotion, SIMD/parallel and Job reuse remain
outside this decision.

## Related documentation

- [CPU0073 brief](../../planning/backends/cpu/tasks/0073-row-map-multi-reduction-fusion.md)
- [Approved detailed proposal](../../planning/backends/cpu/row-map-multi-reduction-fusion-plan.md)
- [Scan semantic owner](../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)
- [Scan explanation](../../architecture/general-differentiable-scan.md#accepted-row-map-fusion-design-not-execution-availability)
- [CPU contributor guide](../../backend-guide/cpu-backend.md#accepted-row-map-and-multiple-terminal-fusion-design)
