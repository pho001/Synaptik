# Task 0041: Prepared Route and Invocation Trace

## Status

Blocked

Blocked only on Trace 0003. No Metal implementation is authorized until its additive IDs/DTOs are
Complete and 0041 is rebased and promoted to Ready. Trace 0003 is the sole authorized frontier.

## Change class

Class C — this adds a supported Metal observer API, concurrent producer stream, Prepare-finalizer
and hot-invocation instrumentation, failure isolation, and public Engine lifecycle observability.
Use one clean implementation context and an independent targeted Class C review/documentation
context. The native ABI and Engine production API remain unchanged.

## Goal

Make Metal route/profile/no-cache, finalization, and native-invocation outcomes observable as typed
Trace events while preserving backend semantics, ownership, failures, privacy, and disabled cost.

## Scope

### Exact public Metal API

Add public functional interface `io.github.pho001.synaptik.backend.metal.MetalTraceObserver` with
exactly one abstract method:

```java
void onEvent(TraceEvent<? extends TracePayload> event);
```

Add exactly this overload while preserving the existing overload unchanged:

```java
public static MetalBackendIntegration open(
        MetalBackendConfiguration configuration,
        MetalTraceObserver observer)
```

Both arguments are non-null and retained after validation. The observer remains caller-owned after
integration transfer and is never closed; callbacks may be concurrent, so it must be thread-safe.
Existing `open(configuration)` selects the no-trace path and creates no producer work or payload.
`MetalBackendConfiguration` remains the exact one-component native-library-path record.

### Producer stream

Add package-private `MetalTraceProducer` owned by one traced `MetalBackendIntegration` and retained
by its runtime/prepared resources. One integration is one producer-defined trace stream:

- backend and device correlations are fixed stream-local `TraceBackendId(0)` and
  `TraceDeviceId(0)`; no name/path/device claim is encoded;
- `TraceEventId`, `TracePreparedUnitId`, and `TraceInvocationId` each use an independent,
  thread-safe, non-negative sequence beginning at zero;
- every event uses `System.nanoTime()` and its own stream-local event ID;
- ID exhaustion, DTO/event construction failure, or observer `RuntimeException` atomically disables
  later tracing and never changes backend success, failure, rollback, or outward exception;
- Java `Error` follows ordinary fatal-error propagation and is not converted into diagnostics;
- callbacks are not globally serialized. IDs are unique, a preparation outcome precedes any
  invocation for its unit, and concurrent invocation completion order is otherwise unspecified.

### PREPARE outcomes

Allocate one prepared-unit ID only after analysis fixes the route and before finalization. Retain
that ID plus profile, neutral route, and `NOT_QUERIED` in the immutable plan and executable.

While the producer remains enabled, `MetalNegPartitionFinalizer` emits exactly one
`TraceEvent<BackendPreparationOutcome>` at `TracePhase.PREPARE`: `INFO` after executable/resource
construction succeeds, or `ERROR` when finalization fails after a route was selected. Map:

- `STRICT_IEEE`/`ACCELERATOR` to the trace-owned profile enum;
- `CUSTOM_SINGLE_NEG` to `CUSTOM_KERNEL`;
- `MPSGRAPH` to `GRAPH_EXECUTABLE`;
- every current path to `TraceCacheStatus.NOT_QUERIED` because Metal performs no cache lookup.

Supplied decisions still report `NOT_QUERIED`: Metal authenticates but does not know whether an
outer owner used a cache. Pre-route analysis rejection emits nothing; cleanup/open/close are out of scope.

### RUN outcomes and native status mapping

While the producer remains enabled, each route-specific bound invocation allocates one invocation
ID and emits exactly one `TraceEvent<BackendInvocationOutcome>` at `TracePhase.RUN`: `INFO` on
native success or `ERROR` on failure. No start, transfer, allocation, binding, materialization,
publication, or close event is added.

Map native status `0` to `SUCCESS`; `1..12` respectively to `INVALID_ARGUMENT`,
`DEVICE_UNAVAILABLE`, `COMMAND_QUEUE_UNAVAILABLE`, `ALLOCATION_FAILED`, `RANGE_OUT_OF_BOUNDS`,
`COPY_FAILED`, `INTERNAL_ERROR`, `UNSUPPORTED_SHAPE`, `COMPILATION_FAILED`,
`INCOMPATIBLE_RESOURCE`, `EXECUTION_FAILED`, and `COMPILATION_FAILED`. Map any other signed code to
`UNKNOWN` while retaining the exact code. A Java-side failure without a returned native status uses
an empty `nativeStatus`.

For MPSGraph status `RANGE_OUT_OF_BOUNDS`, emit the typed failure before the existing Java rescan
translates it to the public indexing exception. Never include the rescanned index, ordinal, extent,
exception message, `Throwable`, or suppressed failure in the event.

### Redaction and behavior preservation

Events contain only the Trace 0003 components. Never expose the native library path, session nonce,
FFM segment, address/handle, Tensor/storage value, scalar constant, shape, byte extent, workload or
candidate fingerprint, thread identity, exception text/stack, free-form string, or generic map.
Tracing never changes route selection, tuning compatibility, resource ownership, native call count,
result, failure type/message/suppression, retry/fallback behavior, or cleanup order.

## Non-goals

- Trace API changes beyond consuming completed Trace 0003; no shared emitter/sink/consumer/state.
- Engine production changes, CPU parity, CLI/TUI implementation, event persistence, formatting,
  filtering, serialization, logging, metrics, durations, or benchmarks.
- Compile events, integration open/close events, analysis rejections, transfers, buffer lifecycle,
  materialization, publication, resource close, or cache hit/miss events.
- Native Objective-C/C source, ABI version, symbols, signatures, status codes, node schema,
  candidate/codec identities, capability, numerics, or device discovery.

## Contracts

- [`ARCHITECTURE.md` — Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle) preserves
  compile/prepare/run ordering.
- [Foundational modules — `modules/trace`](../../../../architecture/contracts/foundational-modules.md#modulestrace)
  owns DTOs while Metal owns producer behavior.
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  assigns Metal trace contributions and native state to Metal without an Engine dependency.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  fixes route selection before finalization and executable construction after slot assignment.
- [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) keeps
  invocation prepared and synchronous; tracing remains passive observation.

Stop if implementation needs Engine production changes, a new shared Trace behavior API, a native
ABI/schema/capability change, a generic map/string payload, or observer failure that changes work.

## Dependencies and integration

- Depends on: Trace 0003 Complete; Metal 0001–0005/0008/0014–0015/0019–0025 Complete; Engine 0017–0018.
- Conflicts with: Trace 0003 until integrated; every concurrent Metal integration/preparation/
  executable/native-failure/hot-path/API scope and shared Trace/Metal/API/roadmap documents.
- Parallel group: None.
- Common base revision: N/A until Trace 0003 is integrated.
- Integration order: complete Trace 0003, rebase, promote 0041 to Ready, then implement serially.
- Integration validation: focused Metal trace tests, public Engine Metal integration trace scenario,
  affected architecture checks, Javadocs, Markdown/status/diff, then independent Class C review.
- Shared-document integration owner: Metal 0041 executor after unblocking.

## Files and symbols

- `MetalTraceObserver`, `MetalBackendIntegration.open(configuration, observer)` — public surface.
- `MetalTraceProducer` — package-private stream, IDs, clock, mapping, containment, disabled state.
- Runtime, analysis inputs, preparer/plan/finalizer, and prepared executable — propagate facts and
  emit only finalization/native-invocation outcomes.
- Focused Metal tests, the public Engine integration test, Javadocs/guides, task/master/roadmap;
  native README remains unchanged.

## Acceptance criteria

1. The exact observer interface and open overload exist; existing open/configuration/API behavior is
   unchanged, and the no-trace path performs no trace allocation, clock read, ID work, or callback.
2. A traced prepared unit emits one typed PREPARE outcome with exact profile, neutral route,
   `NOT_QUERIED`, correlation IDs, phase, level, and success/failure status.
3. Each native invocation emits one typed RUN outcome with a unique invocation ID and exact
   known/unknown status mapping; MPSGraph range failure is captured before outward rescan mapping.
4. Observer/runtime trace failures disable later trace without changing backend outcomes; repeated
   and concurrent session runs retain unique stream-local IDs and documented ordering only.
5. Payloads expose none of the prohibited data and use no string/map/Throwable escape hatch.
6. Public Engine composition, `InferenceSession.run`, repeated execution, materialization, failure
   text/suppression, ownership, close, route, and native call counts remain unchanged.
7. No Engine production, CPU, native source/ABI/export/schema/status/capability, dependency, or build
   change occurs.
8. Deterministic injected-native tests and an actual public Engine lifecycle collector scenario pass;
   Javadocs/docs/status/diff checks and independent Class C review approve.

## Validation

Worker validation after unblocking:

```bash
./gradlew :backends:metal:test --tests '*MetalTrace*'
./gradlew :testing:integration-tests:test \
  --tests 'io.github.pho001.synaptik.testing.integration.EngineExplicitCompositionMetalIntegrationTest'
./gradlew :testing:architecture-tests:test \
  --tests 'io.github.pho001.synaptik.testing.architecture.EngineCompositionContractTest'
./gradlew :backends:metal:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/architecture/tracing.md docs/api/public-api.md docs/backend-guide/metal-backend.md \
  docs/glossary.md docs/planning/backends/metal/tasks/0041-prepared-route-and-invocation-trace.md \
  docs/planning/backends/metal/master-plan.md docs/planning/roadmap.md
git diff --check
```

The integration scenario must drive `Engine.compile`, `Engine.session`, and `InferenceSession.run`
with the observer collector; direct backend/DTO tests are insufficient. Reuse the configured Metal
library convention; add no disposable device probe or public testing hook.

## Documentation and review impact

- Document observer ownership/concurrency/containment, stream-local IDs, disabled cost, emitted
  fields, redaction, and unchanged ABI/lifecycle. Independent Class C review covers public API,
  concurrency, hot path, native failure translation, and Engine-observable behavior.

## Result

Empty while Blocked.
