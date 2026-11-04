# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0052: Share row maps across multiple terminal [CPU](../../glossary.md#term-cpu) reductions


## Editorial reading context

CPU lowering can compute a row-local producer once and feed several terminal reductions. A terminal consumes row contributions and cannot feed that same row pass. This physical grouping must preserve required stores and each operation's arithmetic; it is not a new public Tensor operation. Compiler still constructs gradients, and Runtime still invokes the already-prepared work.

Use the [row-region lowering boundary](../../backend-guide/cpu-backend.md#accepted-row-map-and-multiple-terminal-fusion-design) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-10-30 user approval; implemented by CPU0073, independently reviewed and
MAIN accepted on 2026-10-30. This [ADR](../../glossary.md#term-architecture-decision-record-adr) records
rationale, not authority; the [architecture root](../../../ARCHITECTURE.md) and its incorporated
[CPU mechanism contract](../../architecture/contracts/backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu))</span>
own the exact approved packet. Implementation evidence is retrospective and recorded in the
[CPU0073 result](../../planning/backends/cpu/tasks/0073-row-map-multi-reduction-fusion.md#result);
performance remains unmeasured.

## Context

Existing [operation](../../glossary.md#operation) emission shares [pointwise](../../glossary.md#term-pointwise), [MATMUL](../../glossary.md#matrix-multiplication--matmul) and represented-SUM mathematics, but
separate whole-domain stages can repeatedly traverse rows and materialize a common [producer](../../glossary.md#term-producer).
The bounded backward-code comparison motivates sharing that producer with multiple terminal
outputs. A row map means pure ordered local work per row; a terminal consumes rows and cannot
feed that same pass. The small comparison is not general correctness or timing evidence.

## Decision drivers

- Preserve represented arithmetic, saved/public uses, [aliases](../../glossary.md#aliasing-and-arena), activity and resource lifetimes.
- Reuse current [operation](../../glossary.md#operation) math through both ordinary and [Scan](../../glossary.md#general-differentiable-scan) callers, without a second executor.
- Bound cold proofs and live locals; retain fallbacks and literal noneligible [artifacts](../../glossary.md#term-artifact).
- Distinguish exact ordinary SUM from Scan's represented algorithm and avoid capability inflation.

## Options considered

### Keep separate complete traversals

This preserves current code and resources but leaves common row work/staging intact. It remains
the fallback for ineligible groups, not the selected solution for proved eligible regions.

### Add an [RNN](../../glossary.md#term-recurrent-neural-network-rnn)-specific backward [kernel](../../glossary.md#kernel) or [runtime](../../glossary.md#term-runtime) coordinator

This could [target](../../glossary.md#term-target) the observed case, but duplicates semantics or introduces execution [ownership](../../glossary.md#term-ownership-meanings)
and would not solve ordinary dense-[gradient](../../glossary.md#term-gradient) sharing. It is rejected.

### Derive one bounded [CPU](../../glossary.md#term-cpu) row region using current math [owners](../../glossary.md#term-owner)

This shares a proved row map across row outputs and multiple terminals. It needs explicit
legality/resource proof, generation-local [lifetime](../../glossary.md#term-lifetime) [ownership](../../glossary.md#term-ownership-meanings) and selective [artifact](../../glossary.md#term-artifact) [identities](../../glossary.md#term-identity-meanings).
It is selected; exact names/[signatures](../../glossary.md#term-signature)/budgets live only in the owning contract.

## Decision and rationale

Select the common cold [IR](../../glossary.md#term-intermediate-representation-ir)/proof/emission mechanism in the [CPU](../../glossary.md#term-cpu) [owner](../../glossary.md#term-owner). Ordinary [graphs](../../glossary.md#graph) lower to
one full-body work item; [Scan](../../glossary.md#general-differentiable-scan) wraps generation inside its existing time/activity loop. The
region is physical [lowering](../../glossary.md#lowering) of an unchanged semantic [DAG](../../glossary.md#term-directed-acyclic-graph-dag), not a new [Model](../../glossary.md#term-model) [operation](../../glossary.md#operation), [Compiler](../../glossary.md#term-compiler)
graph, scheduler or [Runtime](../../glossary.md#term-runtime)/[Engine](../../glossary.md#term-engine)/[Prepare](../../glossary.md#prepare) protocol. Existing math owners retain algorithms;
ordinary exact SUM and its required [producer](../../glossary.md#term-producer) storage/workspace remain separate.

For example, a dense [gradient](../../glossary.md#term-gradient) may share D between row projections and `X.T@D`, while preserving
D for separate exact `db=SUM_R(D)`. Scan may share D with represented terminal [bias](../../glossary.md#term-bias) SUM but must
perform the shared [carry](../../glossary.md#term-carry) ADD only after row reduction. A terminal feeding row work cannot fuse
in that pass. This addresses general sharing without a model-specific shortcut or new numerics.

## Consequences

### Positive

Eligible groups can avoid repeated row work and unobserved intermediate traffic in both callers.
No new semantic [operation](../../glossary.md#operation), [dtype](../../glossary.md#data-type--datatype)/[layout](../../glossary.md#layout) capability, executor or numerical policy is required.
These are design opportunities, not measured speedups or guaranteed no-copy behavior.

### Negative and risks

Complete uses, [dependency](../../glossary.md#term-dependency) convexity, named [axes](../../glossary.md#term-axis), physical [aliases](../../glossary.md#aliasing-and-arena) and inclusive lifetimes make
cold [admission](../../glossary.md#term-admission) more demanding. Bounded [features](../../glossary.md#term-feature)/live locals deliberately reject some groups.
Saved or external uses can retain [materialization](../../glossary.md#materialization). Shared generation seams must preserve old
fallback byte/local order; selected groups cannot reuse incompatible unfused [artifacts](../../glossary.md#term-artifact).

### Migration, testing and follow-up

The owning contract reserves only selected ordinary projection 72 and selective ordinary/[Scan](../../glossary.md#general-differentiable-scan)
row-[fusion](../../glossary.md#term-fusion)-v1 suffixes. Global envelope 71 and Scan projection 71 remain; noneligible controls
retain literal keys/ClassFiles/resources. Both callers qualified before eligible traversal glue
retired; fallback math, temporal/history [ownership](../../glossary.md#term-ownership-meanings) and required ordinary D stores remain.
CPU0073 provides durable actual-[AD](../../glossary.md#autograd), empty/range/activity/[alias](../../glossary.md#aliasing-and-arena)/resource tests and specialized clean
Java/generated-ClassFile inspection. Its implemented selection requires a computed row [producer](../../glossary.md#term-producer)
shared by a terminal and another required sink, not direct-input/movement sharing alone.
Independent code/tests/[Javadoc](../../glossary.md#term-javadoc) review closed all findings; MAIN's 42-test join passed. The task
distinguishes the failed full-[module](../../glossary.md#term-module) run from passing repairs and approved unchanged-outcome
transfer. Finite retention evidence authenticates 356 pre-edit payloads and seven selected reverse
retirements; the missing initial case-level diagnostic is not reconstructed. These results establish
bounded implementation, not speed or universal [artifact](../../glossary.md#term-artifact) preservation.
All old timing budgets are closed; measurements require
a fresh explicit finite grant. Across-time [carry](../../glossary.md#term-carry) promotion, [SIMD](../../glossary.md#simd)/parallel and Job reuse remain
outside this decision.

## Related documentation

- [CPU0073 brief](../../planning/backends/cpu/tasks/0073-row-map-multi-reduction-fusion.md)
- [Approved detailed proposal](../../planning/backends/cpu/row-map-multi-reduction-fusion-plan.md)
- [Scan semantic owner](../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([Scan glossary](../../glossary.md#general-differentiable-scan); [owner glossary](../../glossary.md#term-owner))</span>
- [Scan explanation](../../architecture/general-differentiable-scan.md#accepted-row-map-fusion-design-not-execution-availability)
- [CPU contributor guide](../../backend-guide/cpu-backend.md#accepted-row-map-and-multiple-terminal-fusion-design)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu))</span>
