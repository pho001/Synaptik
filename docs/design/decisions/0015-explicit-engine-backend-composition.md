# ADR 0015: Explicit Engine Backend Composition

## Status

Accepted — 2026-09-23

## Context

Synaptik has an Engine-owned CPU composition and independently executable Metal Prepare/Runtime
building blocks. The current public convenience `Engine.standard()` deliberately constructs one
fresh CPU integration, while `AdvancedEngine.takeOwnership(...)` accepts one CPU integration.
Neither surface can explicitly compose more than one available backend, and Metal has no public
Engine lifecycle adapter.

Adding composition without a complete contract would create several failure modes: duplicate
backend identities could make Planning ownership ambiguous; a live or global registry could move
selection into Runtime; partially constructed native integrations could leak; and a mixed-owner
plan could be assembled before cross-owner representations and transfers have an owner.

This ADR records the selected architecture. The
[Runtime, Prepare, and Engine contract](../../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition)
remains authoritative.

## Decision drivers

- Preserve explicit, deterministic compile-time ownership and the Runtime hot-path boundary.
- Keep concrete backend modules independent of Engine.
- Make native and partial-construction ownership mechanically unambiguous.
- Preserve the current ordinary CPU convenience without implying discovery.
- Admit a complete single-owner CPU or Metal vertical slice before cross-owner transfer semantics
  exist.
- Make dependency, API shape, duplicate identity, rollback, and closure rules automatable.

## Options considered

### Engine builder with concrete ownership overloads

`Engine.builder()` returns an `AutoCloseable` construction owner with concrete
`takeOwnership(CpuBackendIntegration)` and `takeOwnership(MetalBackendIntegration)` overloads.
Engine adapts those integrations into a private ordered registry. Backend modules publish their own
integrations and configuration but implement no Engine-owned interface.

### Shared generic backend integration interface

A common interface would make registration syntactically uniform, but no current inward module can
own the complete capability, Prepare, Runtime representation, host materialization, tuning, and
close surface without becoming a new service-locator-shaped abstraction. Placing it in Engine
would also reverse the required backend-to-Engine dependency direction.

### Engine configuration that discovers or opens every backend

A central Engine configuration could accept library paths and provider names, then discover and
open backends itself. This would duplicate backend configuration policy in Engine, blur native
rollback ownership, and encourage classpath, environment, reflection, or `ServiceLoader`
discovery.

### Process-global registry or Runtime lookup

A global registry would avoid passing a composition object through construction, but it would make
tests and lifetimes process-coupled and permit late backend lookup after compilation or
preparation. Runtime selection and fallback would violate the prepared-schedule model.

## Decision

Synaptik selects the Engine builder with concrete ownership overloads.

The planned public shape is:

```java
try (Engine.Builder builder = Engine.builder()) {
    builder.takeOwnership(CpuBackendIntegration.open());
    builder.takeOwnership(MetalBackendIntegration.open(metalConfiguration));
    try (Engine engine = builder.build()) {
        // compile, prepare, and run
    }
}
```

For each non-null `takeOwnership` argument, ownership transfers at method entry. Registration
captures one equal `BackendId` across the integration, capability provider, and immutable
availability snapshot. Equal duplicate IDs are rejected without replacing the first entry; the
newly transferred integration is closed. Registration order is retained as deterministic
compile-time input. Availability is captured once at registration and never refreshed within that
Engine.

Before successful `build()`, the builder owns accepted integrations. A successful build transfers
the complete registry atomically to Engine. Registration failure preserves its primary failure and
closes only the newly transferred candidate; earlier registrations remain owned by the builder.
Build failure preserves its primary failure, terminally closes the builder, and closes every
accepted integration in reverse registration order. Cleanup suppresses later distinct close
failures in encounter order and skips self-suppression. Engine closes results, prepared handles,
and integrations in that order, with each owner class using its existing reverse-publication or
reverse-registration order.

Planning selects a `BackendId` from the frozen providers and availability snapshots. Cold Prepare
routes planned partitions by that identity. The first implementation accepts only a non-empty
plan whose partitions all have one equal registered owner. Missing, empty, or mixed-owner plans
fail before backend analysis. Cross-owner execution remains blocked until a separate contract
owns representation creation, transfer capability and direction, shared declarations, schedule
ordering, and transactional rollback.

Successful preparation captures the selected Engine adapter directly in the outward prepared
handle. Engine run uses it for caller-host-storage ingress, and the outward result uses it for
backend-owned host materialization. The registry map is not queried again; Engine keeps the
integration open until results and prepared handles close.

Public `prepareTuned(...)` stays CPU-only. A complete CPU-owned plan may tune even when Metal is
also registered. A Metal-owned plan fails with `IllegalStateException` before representative
borrowing or trial work, while missing, empty, and mixed ownership retain ordinary preparation
rejection. Allowed safe-heuristic fallback stays inside the selected CPU entry and never changes
owner or prepares Metal.

`MetalBackendConfiguration` and `MetalBackendIntegration` belong to `backends/metal`. Metal
validates and snapshots its configuration and opens native ownership before Engine receives the
integration. Engine stores no duplicate Metal path or device policy. `Engine.standard()` continues
to open one fresh default CPU integration through the same ownership path.

Runtime receives only the completed inward `PreparedExecution`; the direct adapter remains in
outward Engine handles. Neither Runtime nor a cold-bound hot invocation can access the builder,
registry, adapter, providers, availability snapshots, configurations, backend IDs, reflection,
`ServiceLoader`, or a global lookup.

## Rationale

Concrete overloads make every supported built-in backend an explicit Engine composition decision
without forcing backend modules to depend outward on Engine. Entry-time transfer gives every
failure point exactly one cleanup owner. Freezing availability once gives one Engine a stable
Planning input, while rebuilding an Engine remains the explicit refresh operation.

The single-owner gate is intentionally narrower than the registry. It lets the same explicit
composition select a complete CPU-owned or Metal-owned graph without pretending that merely
having both integrations defines data movement between them. Rejecting mixed ownership before
analysis prevents partial preparation-time resource acquisition and accidental fallback.

## Consequences

### Positive

- Composition, ordering, ownership, and cleanup are local and testable.
- Duplicate backend identities cannot produce ambiguous routing.
- Metal keeps its native configuration and rollback policy.
- Existing compile-time Planning and Prepare/Runtime boundaries remain intact.
- The runtime hot path retains direct prepared references and no lookup overhead.

### Negative and risks

- Adding a built-in backend requires an Engine adapter and public builder overload.
- The concrete public overloads make CPU and Metal API-visible Engine dependencies; this is
  deliberate construction-surface coupling rather than a backend-neutral plugin SPI.
- Availability is intentionally stale for one Engine lifetime; callers rebuild to refresh it.
- A registry containing CPU and Metal can still reject a graph at Prepare when Planning assigns
  more than one owner.
- The first Metal Engine slice also needs Metal-owned host ingress, host materialization, complete
  schedule contribution, and conformance evidence; capability alone is insufficient.
- Metal's backend-local route candidates do not make public model autotuning generic;
  `prepareTuned(...)` remains CPU-only for this slice.
- The existing CPU-only advanced surface is not made generic by this decision.

### Migration, testing, and follow-up

The implementation must extend the public Engine API-shape test with `builder`, `Builder`, its two
ownership overloads, `build`, and `close`; add lifecycle tests for entry-time transfer, duplicate
IDs, snapshot timing, partial-construction rollback, reverse close, suppressed failures, and
post-build builder closure; and update architecture tests for the intentional API-visible
Engine-to-CPU/Metal edges while retaining the prohibition on every backend production dependency
on Engine. Prepare/integration tests must prove one registered CPU owner and one registered Metal
owner independently, direct selected-adapter ingress/materialization without a registry lookup,
and pre-analysis rejection for missing, empty, and mixed owner sets. Tuning tests must cover a
CPU-owned plan with Metal also registered, pre-trial Metal rejection, and CPU-only fallback.
Hot-path validation must continue to prove that Runtime executes direct prepared references
without registry or adapter lookup.

A separate architecture decision is required before mixed-owner execution. It must define the
transfer contract named above before any implementation relaxes the fail-closed gate.

## Related documentation

- [Authoritative Runtime, Prepare, and Engine contract](../../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition)
- [Authoritative Metal backend contract](../../architecture/contracts/backend-execution.md#metal-backend)
- [Lifecycle explanation](../../architecture/lifecycle.md#explicit-backend-composition)
- [Module boundaries](../../architecture/module-boundaries.md#modulesengine)
- [Public API status](../../api/public-api.md#planned-explicit-backend-composition)
- [ADR 0006: No runtime service locator](0006-no-runtime-service-locator.md)
- [Engine task 0014](../../planning/modules/engine/tasks/0014-explicit-backend-composition-architecture.md)
