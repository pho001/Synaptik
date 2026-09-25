# ADR 0022: Auditable Custom Metal Route Cost Evidence

## Status

Accepted — 2026-09-25

## Context

Metal operation qualification compares only candidates that already have a complete-domain proof
and bounded numerical pass. The existing rule required hot duration, actual compute-dispatch count,
and peak temporary bytes. Installed Command Line Tools can measure a custom route on the target
Apple GPU but cannot expose framework-internal MPSGraph dispatches or transient resources.

An explicit custom route is different from an opaque framework route when its retained source owns
every command encoder dispatch and every resource in one synchronous hot invocation. In that closed
case the source and resource ledger can state dispatch count and route-owned temporary bytes exactly;
there is no hidden framework execution to infer.

## Decision drivers

- Keep opaque framework internals fail-closed.
- Admit evidence that is exact and independently auditable for closed custom routes.
- Measure runtime on the actual target device.
- Keep route choice cold and immutable during execution.
- Add no fallback, retry, input-dependent choice, or benchmark matrix.

## Decision

For an explicit custom Metal route whose retained exact source owns every command encoder dispatch
and every resource used by the synchronous hot invocation, qualification may accept an auditable
route declaration as the compute-dispatch count and route-owned temporary-byte fact. The declaration
must enumerate every dispatch and every transient resource, exclude steady prepared inputs and
outputs, and identify the exact measured source.

This allowance does not apply when the hot invocation calls MPSGraph or another opaque framework,
uses an unowned encoder, or depends on an unowned resource. Such candidates still require supported
observation of actual framework-internal dispatches and peak transient resources. Graph-node counts,
command-buffer counts, inferred fusion, framework estimates, process resident memory, and custom
resource guesses are not substitutes.

Hot duration is measured on the named target hardware after all routes are prepared. The
qualification protocol remains a bounded cold decision. Production fixes one selected route during
backend preparation; Runtime performs no timing, candidate search, retry, fallback, or matrix.

## Options considered

### Require full-Xcode trace data for every route

This blocks closed custom routes even when all relevant dispatch and resource facts are explicit in
retained source. Rejected.

### Infer opaque MPSGraph internals

Framework graph structure does not establish actual dispatch or transient allocation. Rejected.

### Accept auditable declarations only for closed custom routes

This preserves fail-closed opaque routes while using exact source-owned facts. Accepted.

## Consequences

- Custom-only Gate-3 adjudication may proceed when every survivor satisfies the closed-source rule.
- Opaque MPSGraph candidates remain domain- or evidence-blocked and receive no inferred facts.
- Route declarations become review evidence tied to the measured source identity, not runtime state.
- No public API, native export, ABI version, Runtime contract, or tuning-cache behavior changes.

## Related documents

- [Architecture root](../../../ARCHITECTURE.md)
- [Backend execution contract](../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
- [Performance evidence and tuning](../../architecture/performance-evidence-and-tuning.md)
- [ADR 0021](0021-total-recursive-accelerator-numerical-floor.md)
