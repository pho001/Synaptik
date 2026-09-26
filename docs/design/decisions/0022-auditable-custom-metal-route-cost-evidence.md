# ADR 0022: Auditable Custom Metal Route Cost Evidence

## Status

Accepted — 2026-09-25

## Context

Metal operation qualification considers only candidates that already have identical proven
semantics over the complete authorized domain and have passed their bounded numerical regression
gate. Earlier wording treated local target-device hot duration as a route-selection fact alongside
dispatch count and temporary bytes. A developer workstation cannot provide controlled comparative
timing: background load, thermal state, frequency policy, and process placement are uncontrolled.
Its measurements remain useful diagnostics, but they are not qualification, route-selection, or
tuning-identity authority.

An explicit custom route is different from an opaque framework route when its retained source owns
every command encoder dispatch and every resource in one synchronous hot invocation. In that closed
case, a source and resource ledger can establish dispatch count and route-owned temporary bytes
exactly. Opaque framework routes still require supported observation of their actual internals;
graph-node counts, command-buffer counts, inferred fusion, estimates, resident memory, and guesses
are not substitutes.

## Decision drivers

- Keep opaque framework internals fail-closed.
- Use only exact, independently auditable structural facts for uncontrolled-environment decisions.
- Treat local-device timing as diagnostic history, never decision authority.
- Avoid a meaningless comparison when only one proved candidate survives.
- Keep route choice cold and immutable during execution.
- Add no fallback, retry, input-dependent choice, or benchmark matrix.

## Decision

A single complete-domain, numerical-pass survivor needs no comparative cost gate. It may become the
fixed route after its ordinary lifecycle/resource review because there is no alternative to rank.

When multiple candidates survive with identical proven semantics and domain, a timing-free decision
is allowed only by **strict structural dominance**. Candidate A strictly dominates candidate B when
A has no more compute dispatches, no more route-owned temporary bytes, and is strictly better in at
least one of those two dimensions. Every compared fact must be exact:

- an explicit custom route may use an auditable source-bound declaration only when retained source
  owns every command encoder dispatch and every resource used by the synchronous hot invocation;
- the declaration enumerates every dispatch and transient resource, excludes steady prepared inputs
  and outputs, and identifies the exact source; and
- an opaque route, including MPSGraph, requires supported observation of actual framework-internal
  dispatches and peak transient resources and may never receive inferred values.

Strict dominance may eliminate a dominated candidate and select the sole remaining candidate. It
does not create a weighted score or permit trading more dispatches for fewer bytes. If multiple
nondominated candidates remain, route selection stays pending until an explicitly authorized,
controlled comparative environment and protocol exist. An uncontrolled local timing result cannot
close that gate.

Local device timing may still be retained as a diagnostic report tied to its exact workload,
environment, source, and raw samples. It never qualifies a candidate, selects a route, changes
tuning identity, populates a production decision, or breaks a structural tie. Production fixes any
authorized route during cold preparation; Runtime performs no timing, search, retry, fallback, or
matrix.

## Options considered

### Require comparative timing even for one survivor

There is no competing route to rank, so this adds noise without decision information. Rejected.

### Let uncontrolled local timing rank survivors

The environment cannot support a qualification claim. Retaining the samples as diagnostics while
withholding route authority is the truthful boundary. Rejected as a selection rule.

### Infer opaque MPSGraph internals

Framework graph structure does not establish actual dispatch or transient allocation. Rejected.

### Select by strict structural dominance, otherwise wait

This uses exact comparable facts, admits single-survivor closure, and leaves true tradeoffs pending
a controlled environment. Accepted.

## Consequences

- Single-survivor rows close without a comparative cost run.
- Multiple identical-semantics/domain survivors may close without timing only when one strictly
  structurally dominates every other survivor.
- Nondominated alternatives remain pending a controlled environment; local timing is diagnostic.
- Opaque MPSGraph candidates remain domain- or evidence-blocked and receive no inferred facts.
- Route declarations are review evidence tied to exact source identity, not runtime state or tuning
  identity.
- Task 0052's raw local timing remains historical diagnostic evidence. Fused CLAMP is justified
  solely by its exact one dispatch and zero route-owned temporary bytes versus composed CLAMP's two
  dispatches and 4,194,304 route-owned bytes.
- No public API, native export, ABI version, Runtime contract, or tuning-cache behavior changes.

## Related documents

- [Architecture root](../../../ARCHITECTURE.md)
- [Backend execution contract](../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
- [Performance evidence and tuning](../../architecture/performance-evidence-and-tuning.md)
- [ADR 0021](0021-total-recursive-accelerator-numerical-floor.md)
