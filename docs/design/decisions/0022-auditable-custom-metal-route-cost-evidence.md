# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0022: Auditable Custom [Metal](../../glossary.md#term-metal) Route Cost Evidence


## Editorial reading context

Backend authors compare only candidates already qualified for identical semantics and domain. Dispatch count counts submitted compute work; temporary bytes count route-owned transient storage, excluding steady inputs and outputs. A source ledger can establish those quantities only when the route owns every relevant action. A local timing report cannot fill missing observations for an opaque vendor implementation.

Use the [evidence versus route-selection authority](../../architecture/performance-evidence-and-tuning.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

### Mathematical reading aid: strict dominance

Let $d_A,d_B$ be nonnegative integer compute-dispatch counts and $b_A,b_B$ be
nonnegative integer transient-byte counts for qualified candidates $A$ and $B$ over
the same complete invocation domain. The rule below means

$$
(d_A\le d_B)\land(b_A\le b_B)\land((d_A<d_B)\lor(b_A<b_B)).
$$

For example, hypothetical $A=(2,1024)$ and $B=(3,1024)$, in dispatches and bytes,
make $A$ strictly dominant. Comparing $(2,2048)$ with $(3,1024)$ does not: one
improves dispatches while worsening bytes. These exact integer comparisons are not
a weighted timing estimate and add no new measured evidence.

## Status

Accepted — 2026-09-25

## Context

[Metal](../../glossary.md#term-metal) [operation](../../glossary.md#operation) [qualification](../../glossary.md#term-qualification) considers only candidates that already have identical proven
semantics over the complete authorized domain and have passed their bounded numerical [regression](../../glossary.md#term-regression)
gate. Earlier wording treated local [target](../../glossary.md#term-target)-[device](../../glossary.md#term-device) hot duration as a route-selection fact alongside
[dispatch](../../glossary.md#term-dispatch) count and temporary bytes. A developer workstation cannot provide controlled comparative
timing: background load, thermal state, frequency policy, and process placement are uncontrolled.
Its measurements remain useful diagnostics, but they are not qualification, route-selection, or
tuning-[identity](../../glossary.md#term-identity-meanings) authority.

An explicit custom route is different from an [opaque](../../glossary.md#term-opaque) framework route when its retained source owns
every command encoder dispatch and every resource in one [synchronous](../../glossary.md#term-synchronous) hot invocation. In that closed
case, a source and resource ledger can establish dispatch count and route-owned temporary bytes
exactly. Opaque framework routes still require supported observation of their actual internals;
[graph](../../glossary.md#graph)-[node](../../glossary.md#node) counts, command-[buffer](../../glossary.md#term-buffer) counts, inferred [fusion](../../glossary.md#term-fusion), estimates, resident memory, and guesses
are not substitutes.

## Decision drivers

- Keep [opaque](../../glossary.md#term-opaque) framework internals [fail-closed](../../glossary.md#term-fail-closed).
- Use only exact, independently auditable structural facts for uncontrolled-environment decisions.
- Treat local-[device](../../glossary.md#term-device) timing as diagnostic history, never decision authority.
- Avoid a meaningless comparison when only one proved candidate survives.
- Keep route choice cold and [immutable](../../glossary.md#term-immutable) during execution.
- Add no fallback, retry, input-dependent choice, or benchmark [matrix](../../glossary.md#term-review-matrix).

## Decision

A single complete-domain, numerical-pass survivor needs no comparative cost gate. It may become the
fixed route after its ordinary [lifecycle](../../glossary.md#lifecycle)/resource review because there is no alternative to [rank](../../glossary.md#term-rank).

When multiple candidates survive with identical proven semantics and domain, a timing-free decision
is allowed only by **strict [structural dominance](../../glossary.md#term-structural-dominance-index-mappings-and-cost-dominance)**. Candidate A strictly dominates candidate B when
A has no more compute dispatches, no more route-owned temporary bytes, and is strictly better in at
least one of those two [dimensions](../../glossary.md#dimension). Every compared fact must be exact:

- an explicit custom route may use an auditable source-bound declaration only when retained source
  owns every command encoder dispatch and every resource used by the synchronous hot invocation;<span data-glossary-navigation="true"> ([dispatch glossary](../../glossary.md#term-dispatch); [synchronous glossary](../../glossary.md#term-synchronous))</span>
- the declaration enumerates every dispatch and transient resource, excludes steady prepared inputs
  and outputs, and identifies the exact source; and
- an opaque route, including MPSGraph, requires supported observation of actual framework-internal
  dispatches and peak transient resources and may never receive inferred values.<span data-glossary-navigation="true"> ([opaque glossary](../../glossary.md#term-opaque); [MPSGraph glossary](../../glossary.md#term-mpsgraph))</span>

Strict dominance may eliminate a dominated candidate and select the sole remaining candidate. It
does not create a weighted score or permit trading more dispatches for fewer bytes. If multiple
[nondominated](../../glossary.md#compiler-transformations-and-dominance) candidates remain, route selection stays pending until an explicitly authorized,
controlled comparative environment and protocol exist. An uncontrolled local timing result cannot
close that gate.

Local [device](../../glossary.md#term-device) timing may still be retained as a diagnostic report tied to its exact workload,
environment, source, and raw [samples](../../glossary.md#term-sample-machine-learning-and-hyperparameters). It never qualifies a candidate, selects a route, changes
tuning [identity](../../glossary.md#term-identity-meanings), populates a production decision, or breaks a structural tie. Production fixes any
authorized route during cold preparation; [Runtime](../../glossary.md#term-runtime) performs no timing, search, retry, fallback, or
[matrix](../../glossary.md#term-matrix-meanings).

## Options considered

### Require comparative timing even for one survivor

There is no competing route to [rank](../../glossary.md#term-rank), so this adds noise without decision information. Rejected.

### Let uncontrolled local timing [rank](../../glossary.md#term-rank) survivors

The environment cannot support a [qualification](../../glossary.md#term-qualification) claim. Retaining the [samples](../../glossary.md#term-sample-machine-learning-and-hyperparameters) as diagnostics while
withholding route authority is the truthful boundary. Rejected as a selection rule.

### Infer [opaque](../../glossary.md#term-opaque) [MPSGraph](../../glossary.md#term-mpsgraph) internals

Framework [graph](../../glossary.md#graph) structure does not establish actual [dispatch](../../glossary.md#term-dispatch) or transient allocation. Rejected.

### Select by strict [structural dominance](../../glossary.md#term-structural-dominance-index-mappings-and-cost-dominance), otherwise wait

This uses exact comparable facts, admits single-survivor closure, and leaves true tradeoffs pending
a controlled environment. Accepted.

## Consequences

- Single-survivor rows close without a comparative cost [run](../../glossary.md#term-run).
- Multiple identical-semantics/domain survivors may close without timing only when one strictly
  structurally dominates every other survivor.
- [Nondominated](../../glossary.md#compiler-transformations-and-dominance) alternatives remain pending a controlled environment; local timing is diagnostic.
- [Opaque](../../glossary.md#term-opaque) [MPSGraph](../../glossary.md#term-mpsgraph) candidates remain domain- or evidence-blocked and receive no inferred facts.
- Route declarations are review evidence tied to exact source identity, not runtime state or tuning
  identity.<span data-glossary-navigation="true"> ([identity glossary](../../glossary.md#term-identity-meanings))</span>
- Task 0052's raw local timing remains historical diagnostic evidence. Fused CLAMP is justified
  solely by its exact one dispatch and zero route-owned temporary bytes versus composed CLAMP's two
  dispatches and 4,194,304 route-owned bytes.<span data-glossary-navigation="true"> ([Fused glossary](../../glossary.md#specialization-and-fusion); [dispatch glossary](../../glossary.md#term-dispatch))</span>
- No public [API](../../glossary.md#term-application-programming-interface-api), native export, [ABI](../../glossary.md#term-application-binary-interface-abi) version, [Runtime](../../glossary.md#term-runtime) contract, or tuning-[cache](../../glossary.md#term-cache) behavior changes.

## Related documents

- [Architecture root](../../../ARCHITECTURE.md)
- [Backend execution contract](../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)<span data-glossary-navigation="true"> ([Backend glossary](../../glossary.md#backend))</span>
- [Performance evidence and tuning](../../architecture/performance-evidence-and-tuning.md)
- [ADR 0021](0021-total-recursive-accelerator-numerical-floor.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>
