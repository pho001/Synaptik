# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0015: Explicit [Engine](../../glossary.md#term-engine) [Backend Composition](../../glossary.md#backend-composition)


## Editorial reading context

The application calls takeOwnership with freshly opened integrations. The builder must now clean them up until build successfully transfers that responsibility to Engine. That transfer is about native-resource lifetime, unlike Planning assigning a backend to computation. Closing a failed builder must not require the application to guess which registrations succeeded.

Use the [builder-to-Engine ownership transition](../../architecture/lifecycle.md#explicit-backend-composition) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-09-23

## Context

Before this decision, Synaptik had an [Engine](../../glossary.md#term-engine)-owned [CPU](../../glossary.md#term-cpu) composition and independently executable
[Metal](../../glossary.md#term-metal) [Prepare](../../glossary.md#prepare)/[Runtime](../../glossary.md#term-runtime) building blocks. The public convenience `Engine.standard()` constructed one
fresh CPU integration, while `AdvancedEngine.takeOwnership(...)` accepted one CPU integration.
Neither surface could explicitly compose more than one available [backend](../../glossary.md#backend), and Metal had no public
Engine [lifecycle](../../glossary.md#lifecycle) adapter.

Adding composition without a complete contract would create several failure modes: duplicate
backend [identities](../../glossary.md#term-identity-meanings) could make [Planning](../../glossary.md#term-planning) [ownership](../../glossary.md#term-ownership-meanings) ambiguous; a live or global [registry](../../glossary.md#term-registry) could move
selection into Runtime; partially constructed native integrations could leak; and a mixed-[owner](../../glossary.md#term-owner)
plan could be assembled before cross-owner representations and transfers have an owner.

This [ADR](../../glossary.md#term-architecture-decision-record-adr) records the selected architecture. The
[Runtime, Prepare, and Engine contract](../../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition)
remains authoritative.

## Decision drivers

- Preserve explicit, deterministic [compile](../../glossary.md#compile)-time [ownership](../../glossary.md#term-ownership-meanings) and the [Runtime](../../glossary.md#term-runtime) hot-path boundary.
- Keep [concrete backend](../../glossary.md#concrete-backend) [modules](../../glossary.md#term-module) independent of [Engine](../../glossary.md#term-engine).
- Make native and partial-construction ownership mechanically unambiguous.
- Preserve the current ordinary [CPU](../../glossary.md#term-cpu) convenience without implying discovery.
- Admit a complete single-owner CPU or Metal vertical slice before cross-owner transfer semantics
  exist.<span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner); [Metal glossary](../../glossary.md#term-metal); [slice glossary](../../glossary.md#slice))</span>
- Make [dependency](../../glossary.md#term-dependency), [API](../../glossary.md#term-application-programming-interface-api) [shape](../../glossary.md#shape), duplicate [identity](../../glossary.md#term-identity-meanings), [rollback](../../glossary.md#term-rollback), and closure rules automatable.

## Options considered

### [Engine](../../glossary.md#term-engine) builder with concrete [ownership](../../glossary.md#term-ownership-meanings) overloads

`Engine.builder()` returns an `AutoCloseable` construction [owner](../../glossary.md#term-owner) with concrete
`takeOwnership(CpuBackendIntegration)` and `takeOwnership(MetalBackendIntegration)` overloads.
Engine adapts those integrations into a private ordered [registry](../../glossary.md#term-registry). [Backend](../../glossary.md#backend) [modules](../../glossary.md#term-module) publish their own
integrations and configuration but implement no Engine-owned interface.

### Shared generic [backend](../../glossary.md#backend) integration interface

A common interface would make registration syntactically uniform, but no current inward [module](../../glossary.md#term-module) can
own the complete capability, [Prepare](../../glossary.md#prepare), [Runtime](../../glossary.md#term-runtime) representation, [host](../../glossary.md#term-host) [materialization](../../glossary.md#materialization), tuning, and
close surface without becoming a new service-locator-shaped abstraction. Placing it in [Engine](../../glossary.md#term-engine)
would also reverse the required backend-to-Engine [dependency direction](../../glossary.md#java-platform-and-build-tools).

### [Engine](../../glossary.md#term-engine) configuration that discovers or opens every [backend](../../glossary.md#backend)

A central Engine configuration could accept library paths and provider names, then discover and
open backends itself. This would duplicate backend configuration policy in Engine, blur native
[rollback](../../glossary.md#term-rollback) [ownership](../../glossary.md#term-ownership-meanings), and encourage [classpath](../../glossary.md#term-classpath), environment, reflection, or `ServiceLoader`
discovery.

### Process-global [registry](../../glossary.md#term-registry) or [Runtime](../../glossary.md#term-runtime) [lookup](../../glossary.md#term-lookup)

A global registry would avoid passing a composition object through construction, but it would make
tests and lifetimes process-coupled and permit late [backend](../../glossary.md#backend) lookup after [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) or
preparation. Runtime selection and fallback would violate the prepared-schedule [model](../../glossary.md#term-model).

## Decision

Synaptik selects the [Engine](../../glossary.md#term-engine) builder with concrete [ownership](../../glossary.md#term-ownership-meanings) overloads.

The current public [shape](../../glossary.md#shape) is:

```java
try (Engine.Builder builder = Engine.builder()) {
    builder.takeOwnership(CpuBackendIntegration.open());
    builder.takeOwnership(MetalBackendIntegration.open(metalConfiguration));
    try (Engine engine = builder.build()) {
        // compile, prepare, and run
    }
}
```

For each non-null `takeOwnership` [argument](../../glossary.md#term-argument), ownership transfers at method entry. Registration
[captures](../../glossary.md#term-capture-meanings) one equal [`BackendId`](../../glossary.md#backend-identity--backendid) across the integration, capability provider, and [immutable](../../glossary.md#term-immutable)
availability [snapshot](../../glossary.md#term-snapshot). Equal duplicate IDs are rejected without replacing the first entry; the
newly transferred integration is closed. Registration order is retained as deterministic
[compile](../../glossary.md#compile)-time input. Availability is captured once at registration and never refreshed within that
Engine.

Before successful `build()`, the builder owns accepted integrations. A successful build transfers
the complete [registry](../../glossary.md#term-registry) atomically to Engine. Registration failure preserves its primary failure and
closes only the newly transferred candidate; earlier registrations remain owned by the builder.
Build failure preserves its primary failure, terminally closes the builder, and closes every
accepted integration in reverse registration order. Cleanup suppresses later distinct close
failures in encounter order and skips self-suppression. Engine closes results, prepared handles,
and integrations in that order, with each [owner](../../glossary.md#term-owner) class using its existing reverse-[publication](../../glossary.md#term-publication) or
reverse-registration order.

[Planning](../../glossary.md#term-planning) selects a `BackendId` from the frozen providers and availability snapshots. Cold [Prepare](../../glossary.md#prepare)
routes planned partitions by that [identity](../../glossary.md#term-identity-meanings). The first implementation accepts only a non-empty
plan whose partitions all have one equal registered owner. Missing, empty, or mixed-owner plans
fail before [backend](../../glossary.md#backend) analysis. Cross-owner execution remains blocked until a separate contract
owns representation creation, transfer capability and direction, shared declarations, schedule
ordering, and transactional [rollback](../../glossary.md#term-rollback).

Successful preparation captures the selected Engine adapter directly in the outward prepared
handle. Engine [run](../../glossary.md#term-run) uses it for caller-[host](../../glossary.md#term-host)-storage ingress, and the outward result uses it for
backend-owned host [materialization](../../glossary.md#materialization). The registry map is not queried again; Engine keeps the
integration open until results and prepared handles close.

Public `prepareTuned(...)` stays [CPU](../../glossary.md#term-cpu)-only. A complete CPU-owned plan may tune even when [Metal](../../glossary.md#term-metal) is
also registered. A Metal-owned plan fails with `IllegalStateException` before representative
[borrowing](../../glossary.md#term-borrowing) or trial work, while missing, empty, and mixed ownership retain ordinary preparation
rejection. Allowed safe-[heuristic](../../glossary.md#term-heuristic) fallback stays inside the selected CPU entry and never changes
owner or prepares Metal.

`MetalBackendConfiguration` and `MetalBackendIntegration` belong to `backends/metal`. Metal
validates and snapshots its configuration and opens native ownership before Engine receives the
integration. Engine stores no duplicate Metal path or [device](../../glossary.md#term-device) policy. `Engine.standard()` continues
to open one fresh default CPU integration through the same ownership path.

[Runtime](../../glossary.md#term-runtime) receives only the completed inward [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution); the direct adapter remains in
outward Engine handles. Neither Runtime nor a cold-bound hot invocation can access the builder,
registry, adapter, providers, availability snapshots, configurations, backend IDs, reflection,
`ServiceLoader`, or a global [lookup](../../glossary.md#term-lookup).

## Rationale

Concrete overloads make every supported built-in [backend](../../glossary.md#backend) an explicit [Engine](../../glossary.md#term-engine) composition decision
without forcing backend [modules](../../glossary.md#term-module) to depend outward on Engine. Entry-time transfer gives every
failure point exactly one cleanup [owner](../../glossary.md#term-owner). Freezing availability once gives one Engine a stable
[Planning](../../glossary.md#term-planning) input, while rebuilding an Engine remains the explicit refresh [operation](../../glossary.md#operation).

The single-owner gate is intentionally narrower than the [registry](../../glossary.md#term-registry). It lets the same explicit
composition select a complete [CPU](../../glossary.md#term-cpu)-owned or [Metal](../../glossary.md#term-metal)-owned [graph](../../glossary.md#graph) without pretending that merely
having both integrations defines data movement between them. Rejecting mixed [ownership](../../glossary.md#term-ownership-meanings) before
analysis prevents partial preparation-time resource acquisition and accidental fallback.

## Consequences

### Positive

- Composition, ordering, [ownership](../../glossary.md#term-ownership-meanings), and cleanup are local and testable.
- Duplicate [backend](../../glossary.md#backend) [identities](../../glossary.md#term-identity-meanings) cannot produce ambiguous routing.
- [Metal](../../glossary.md#term-metal) keeps its native configuration and [rollback](../../glossary.md#term-rollback) policy.
- Existing [compile](../../glossary.md#compile)-time [Planning](../../glossary.md#term-planning) and [Prepare](../../glossary.md#prepare)/[Runtime](../../glossary.md#term-runtime) boundaries remain intact.
- The runtime [hot path](../../glossary.md#term-hot-path) retains direct prepared references and no [lookup](../../glossary.md#term-lookup) overhead.

### Negative and risks

- Adding a built-in [backend](../../glossary.md#backend) requires an [Engine](../../glossary.md#term-engine) adapter and public builder overload.
- The concrete public overloads make CPU and Metal API-visible Engine dependencies; this is
  deliberate construction-surface coupling rather than a backend-neutral plugin SPI.<span data-glossary-navigation="true"> ([API glossary](../../glossary.md#term-application-programming-interface-api); [plugin glossary](../../glossary.md#term-plugin); [SPI glossary](../../glossary.md#term-service-provider-interface-spi))</span>
- Availability is intentionally stale for one Engine [lifetime](../../glossary.md#term-lifetime); callers rebuild to refresh it.
- A registry containing CPU and Metal can still reject a graph at Prepare when Planning assigns
  more than one owner.<span data-glossary-navigation="true"> ([registry glossary](../../glossary.md#term-registry); [graph glossary](../../glossary.md#graph); [Prepare glossary](../../glossary.md#prepare); [Planning glossary](../../glossary.md#term-planning); [owner glossary](../../glossary.md#term-owner))</span>
- The current Metal Engine slice includes Metal-owned host ingress, host materialization, complete
  schedule contribution, and conformance evidence; capability alone remains insufficient.<span data-glossary-navigation="true"> ([host glossary](../../glossary.md#term-host); [materialization glossary](../../glossary.md#materialization); [conformance glossary](../../glossary.md#term-conformance))</span>
- [Metal](../../glossary.md#term-metal)'s backend-local route candidates do not make public [model autotuning](../../glossary.md#autotuning--model-autotuning) generic;
  `prepareTuned(...)` remains [CPU](../../glossary.md#term-cpu)-only for this [slice](../../glossary.md#slice).
- The existing CPU-only advanced surface is not made generic by this decision.

### Implementation evidence and follow-up

[Engine](../../glossary.md#term-engine) task 0015 implemented the public builder, its two [ownership](../../glossary.md#term-ownership-meanings) overloads, [atomic](../../glossary.md#resource-transaction-and-rollback) build, and
[close lifecycle](../../glossary.md#ownership-borrowing-and-lease). [API](../../glossary.md#term-application-programming-interface-api)-[shape](../../glossary.md#shape) and [lifecycle](../../glossary.md#lifecycle) tests cover entry-time transfer, duplicate IDs, [snapshot](../../glossary.md#term-snapshot)
timing, registration and construction [rollback](../../glossary.md#term-rollback), reverse close, distinct suppression,
self-suppression avoidance, and post-build builder closure. Architecture tests lock the
intentional API-visible Engine-to-[CPU](../../glossary.md#term-cpu)/[Metal](../../glossary.md#term-metal) edges while retaining the prohibition on every
[backend](../../glossary.md#backend) production [dependency](../../glossary.md#term-dependency) on Engine.

Preparation and integration tests prove registered CPU and Metal [owners](../../glossary.md#term-owner) independently, direct
selected-adapter ingress/[materialization](../../glossary.md#materialization) without [registry](../../glossary.md#term-registry) [lookup](../../glossary.md#term-lookup), and pre-analysis rejection for
missing, empty, and mixed [owner](../../glossary.md#term-owner) sets. Tuning tests cover a CPU-owned plan with Metal also
registered, pre-trial Metal rejection, and CPU-only fallback. [Runtime](../../glossary.md#term-runtime) continues to execute direct
prepared references without registry or adapter lookup.

A separate architecture decision is required before mixed-owner execution. It must define the
transfer contract named above before any implementation relaxes the [fail-closed](../../glossary.md#term-fail-closed) gate.

## Related documentation

- [Authoritative Runtime, Prepare, and Engine contract](../../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [Prepare glossary](../../glossary.md#prepare); [Engine glossary](../../glossary.md#term-engine))</span>
- [Authoritative Metal backend contract](../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../glossary.md#term-metal); [backend glossary](../../glossary.md#backend))</span>
- [Lifecycle explanation](../../architecture/lifecycle.md#explicit-backend-composition)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Module boundaries](../../architecture/module-boundaries.md#modulesengine)<span data-glossary-navigation="true"> ([Module glossary](../../glossary.md#term-module))</span>
- [Public API status](../../api/public-api.md#current-explicit-cpu-and-metal-composition)<span data-glossary-navigation="true"> ([API glossary](../../glossary.md#term-application-programming-interface-api))</span>
- [ADR 0006: No runtime service locator](0006-no-runtime-service-locator.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr); [service locator glossary](../../glossary.md#term-service-locator))</span>
- [Engine task 0014](../../planning/modules/engine/tasks/0014-explicit-backend-composition-architecture.md)
- [Engine task 0015 implementation](../../planning/modules/engine/tasks/0015-cpu-metal-single-owner-composition.md)
