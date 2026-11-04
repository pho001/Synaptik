# Task 0041: Prepared Route and Invocation [Trace](../../../../glossary.md#trace)

## Status

Complete

Implemented at `ba16d942`, remediated at `386705ca`, and finalized after independent Class C
rereview returned `APPROVE` with zero findings. [Trace](../../../../glossary.md#trace) 0003 completed its final independent
public-[API](../../../../glossary.md#term-application-programming-interface-api)/documentation review with `APPROVE` after remediation at
`0e796a82f18fbaaa1d5210638f41e0785e22094c`; this task was implemented from that exact clean common
base. No task is [Ready](../../../../glossary.md#term-ready).

## Change class

Class C — this adds a supported [Metal](../../../../glossary.md#term-metal) [observer](../../../../glossary.md#term-observer) [API](../../../../glossary.md#term-application-programming-interface-api), concurrent [producer](../../../../glossary.md#term-producer) stream, [Prepare](../../../../glossary.md#prepare)-finalizer
and hot-invocation instrumentation, failure isolation, and public [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle) observability.
Use one clean implementation context and an independent targeted Class C review/documentation
context. The native [ABI](../../../../glossary.md#term-application-binary-interface-abi) and Engine production API remain unchanged.

## Goal

Make [Metal](../../../../glossary.md#term-metal) route/profile/no-[cache](../../../../glossary.md#term-cache), finalization, and native-invocation outcomes observable as typed
[Trace](../../../../glossary.md#trace) events while preserving [backend](../../../../glossary.md#backend) semantics, [ownership](../../../../glossary.md#term-ownership-meanings), failures, privacy, and disabled cost.

## [Scope](../../../../glossary.md#term-scope)

### Exact public [Metal](../../../../glossary.md#term-metal) [API](../../../../glossary.md#term-application-programming-interface-api)

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

Both [arguments](../../../../glossary.md#term-argument) are non-null and retained after [validation](../../../../glossary.md#term-validation). The [observer](../../../../glossary.md#term-observer) remains caller-owned after
integration transfer and is never closed; callbacks may be concurrent, so it must be [thread-safe](../../../../glossary.md#term-thread-safe).
Existing `open(configuration)` selects the no-[trace](../../../../glossary.md#trace) path and creates no [producer](../../../../glossary.md#term-producer) work or [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication).
`MetalBackendConfiguration` remains the exact one-component native-library-path record.

### [Producer](../../../../glossary.md#term-producer) stream

Add package-private `MetalTraceProducer` owned by one traced `MetalBackendIntegration` and retained
by its [runtime](../../../../glossary.md#term-runtime)/prepared resources. One integration is one producer-defined [trace](../../../../glossary.md#trace) stream:

- [backend](../../../../glossary.md#backend) and [device](../../../../glossary.md#term-device) correlations are fixed stream-local `TraceBackendId(0)` and
  `TraceDeviceId(0)`; no name/path/device claim is encoded;
- [`TraceEventId`](../../../../glossary.md#trace-event-identity--traceeventid), [`TracePreparedUnitId`](../../../../glossary.md#trace-prepared-unit-correlation-identity--tracepreparedunitid), and [`TraceInvocationId`](../../../../glossary.md#trace-invocation-correlation-identity--traceinvocationid) each use an independent,
  thread-safe, non-negative sequence beginning at zero;<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe))</span>
- every event uses `System.nanoTime()` and its own stream-local event ID;
- ID exhaustion, [DTO](../../../../glossary.md#data-transfer-object--dto)/event construction failure, or [observer](../../../../glossary.md#term-observer) `RuntimeException` atomically disables
  later tracing and never changes backend success, failure, rollback, or outward exception;
- observer `Error` propagates from success reporting. During failure reporting, the existing
  backend/finalization/run failure remains primary and receives a distinct acyclic observer<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
  `Error` as a suppressed failure without disabling tracing or losing prior [rollback](../../../../glossary.md#term-rollback) suppression;
- callbacks are not globally serialized. IDs are unique, a preparation outcome precedes any
  invocation for its unit, and concurrent invocation completion order is otherwise unspecified.

### [PREPARE](../../../../glossary.md#prepare) outcomes

Allocate one prepared-unit ID only after analysis fixes the route and before finalization. Retain
that ID plus profile, neutral route, and `NOT_QUERIED` in the [immutable](../../../../glossary.md#term-immutable) plan and executable.

While the [producer](../../../../glossary.md#term-producer) remains enabled, `MetalNegPartitionFinalizer` emits exactly one
`TraceEvent<BackendPreparationOutcome>` at `TracePhase.PREPARE`: `INFO` after executable/resource
construction succeeds, or `ERROR` when finalization fails after a route was selected. Map:

- `STRICT_IEEE`/[`ACCELERATOR`](../../../../glossary.md#term-accelerator) to the [trace](../../../../glossary.md#trace)-owned profile enum;
- `CUSTOM_SINGLE_NEG` to `CUSTOM_KERNEL`;
- [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph) to `GRAPH_EXECUTABLE`;
- every current path to `TraceCacheStatus.NOT_QUERIED` because [Metal](../../../../glossary.md#term-metal) performs no [cache](../../../../glossary.md#term-cache) [lookup](../../../../glossary.md#term-lookup).

Supplied decisions still report `NOT_QUERIED`: Metal authenticates but does not know whether an
outer [owner](../../../../glossary.md#term-owner) used a cache. Pre-route analysis rejection emits nothing; cleanup/open/close are out of [scope](../../../../glossary.md#term-scope).

### [RUN](../../../../glossary.md#term-run) outcomes and native status mapping

While the [producer](../../../../glossary.md#term-producer) remains enabled, each route-specific [bound invocation](../../../../glossary.md#bound-invocation--boundinvocation) allocates one invocation
ID and emits exactly one `TraceEvent<BackendInvocationOutcome>` at `TracePhase.RUN`: `INFO` on
native success or `ERROR` on failure. No start, transfer, allocation, [binding](../../../../glossary.md#term-binding), [materialization](../../../../glossary.md#materialization),
[publication](../../../../glossary.md#term-publication), or close event is added.

Map native status `0` to `SUCCESS`; `1..12` respectively to `INVALID_ARGUMENT`,
`DEVICE_UNAVAILABLE`, `COMMAND_QUEUE_UNAVAILABLE`, `ALLOCATION_FAILED`, `RANGE_OUT_OF_BOUNDS`,
`COPY_FAILED`, `INTERNAL_ERROR`, `UNSUPPORTED_SHAPE`, `COMPILATION_FAILED`,
`INCOMPATIBLE_RESOURCE`, `EXECUTION_FAILED`, and `COMPILATION_FAILED`. Map any other signed code to
`UNKNOWN` while retaining the exact code. A Java-side failure without a returned native status uses
an empty `nativeStatus`.

For [MPSGraph](../../../../glossary.md#term-mpsgraph) status `RANGE_OUT_OF_BOUNDS`, emit the typed failure before the existing Java rescan
translates it to the public indexing exception. Never include the rescanned index, ordinal, [extent](../../../../glossary.md#term-extent),
exception message, `Throwable`, or suppressed failure in the event.

### Redaction and behavior preservation

Events contain only the [Trace](../../../../glossary.md#trace) 0003 components. Never expose the [native library](../../../../glossary.md#term-native-library) path, session nonce,
[FFM](../../../../glossary.md#term-ffm) segment, address/handle, [Tensor](../../../../glossary.md#tensor)/storage value, [scalar](../../../../glossary.md#term-scalar) [constant](../../../../glossary.md#term-constant), [shape](../../../../glossary.md#shape), byte [extent](../../../../glossary.md#term-extent), workload or
candidate [fingerprint](../../../../glossary.md#term-fingerprint), thread [identity](../../../../glossary.md#term-identity-meanings), exception text/stack, free-form string, or generic map.
Tracing never changes route selection, tuning compatibility, resource [ownership](../../../../glossary.md#term-ownership), native call count,
result, retry/fallback behavior, or cleanup order. It does not replace a failure or change its
type/message/native status; failure reporting only appends a distinct acyclic [observer](../../../../glossary.md#term-observer) `Error`
after any existing [rollback](../../../../glossary.md#term-rollback) suppression.

## Non-goals

- [Trace](../../../../glossary.md#trace) [API](../../../../glossary.md#term-application-programming-interface-api) changes beyond consuming completed Trace 0003; no shared emitter/sink/consumer/state.
- Engine production changes, CPU parity, CLI/TUI implementation, event persistence, formatting,
  filtering, serialization, logging, metrics, durations, or benchmarks.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [CLI glossary](../../../../glossary.md#term-cli); [persistence glossary](../../../../glossary.md#term-persistence); [serialization glossary](../../../../glossary.md#term-serialization); [metrics glossary](../../../../glossary.md#machine-learning-and-hyperparameters))</span>
- Compile events, integration open/close events, analysis rejections, transfers, buffer lifecycle,
  materialization, publication, resource close, or cache hit/miss events.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [buffer glossary](../../../../glossary.md#term-buffer); [lifecycle glossary](../../../../glossary.md#lifecycle); [materialization glossary](../../../../glossary.md#materialization); [publication glossary](../../../../glossary.md#term-publication); [cache hit glossary](../../../../glossary.md#cache-and-memoization))</span>
- Native Objective-C/C source, ABI version, symbols, signatures, status codes, node schema,
  candidate/codec identities, capability, numerics, or device discovery.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Objective-C glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [signatures glossary](../../../../glossary.md#term-signature); [node glossary](../../../../glossary.md#node); [schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [device glossary](../../../../glossary.md#term-device))</span>

## Contracts

- [`ARCHITECTURE.md` — Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span> preserves
  compile/prepare/run ordering.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run))</span>
- [Foundational modules — `modules/trace`](../../../../architecture/contracts/foundational-modules.md#modulestrace)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  owns DTOs while Metal owns producer behavior.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [producer glossary](../../../../glossary.md#term-producer))</span>
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>
  assigns Metal trace contributions and native state to Metal without an Engine dependency.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  fixes route selection before finalization and executable construction after slot assignment.
- [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) keeps
  invocation prepared and synchronous; tracing remains passive observation.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous))</span>

Stop if implementation needs [Engine](../../../../glossary.md#term-engine) production changes, a new shared [Trace](../../../../glossary.md#trace) behavior [API](../../../../glossary.md#term-application-programming-interface-api), a native
[ABI](../../../../glossary.md#term-application-binary-interface-abi)/[schema](../../../../glossary.md#term-schema)/capability change, a generic map/string [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), or [observer](../../../../glossary.md#term-observer) failure that changes work.

## Dependencies and integration

- Depends on: [Trace](../../../../glossary.md#trace) 0003 Complete and independently approved after remediation at
  `0e796a82f18fbaaa1d5210638f41e0785e22094c`; Metal 0001–0005/0008/0014–0015/0019–0025
  Complete; Engine 0017–0018.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- Conflicts with: every concurrent Metal integration/preparation/executable/native-failure/
  hot-path/API scope and shared Trace/Metal/API/roadmap documents.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None.
- Common base revision: `0e796a82f18fbaaa1d5210638f41e0785e22094c`.
- Integration order: implementation completed serially from the exact common base; independent
  Class C review follows.
- Integration validation: focused Metal trace tests, public Engine Metal integration trace scenario,
  affected architecture checks, Javadocs, Markdown/status/diff, then independent Class C review.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): [Metal](../../../../glossary.md#term-metal) 0041 executor.

## Files and symbols

- [`MetalTraceObserver`](../../../../glossary.md#metal-trace-observer--metaltraceobserver), `MetalBackendIntegration.open(configuration, observer)` — public surface.
- `MetalTraceProducer` — package-private stream, IDs, clock, mapping, containment, disabled state.
- Runtime, analysis inputs, preparer/plan/finalizer, and prepared executable — propagate facts and
  emit only finalization/native-invocation outcomes.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>
- Focused Metal tests, the public Engine integration test, Javadocs/guides, task/master/roadmap;
  native README remains unchanged.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Engine glossary](../../../../glossary.md#term-engine); [integration test glossary](../../../../glossary.md#term-integration-test))</span>

## Acceptance criteria

1. The exact observer interface and open overload exist; existing open/configuration/API behavior is
   unchanged, and the no-trace path performs no trace allocation, clock read, ID work, or callback.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [callback glossary](../../../../glossary.md#term-callback))</span>
2. A traced prepared unit emits one typed [PREPARE](../../../../glossary.md#prepare) outcome with exact profile, neutral route,
   `NOT_QUERIED`, correlation IDs, phase, level, and success/failure status.
3. Each native invocation emits one typed RUN outcome with a unique invocation ID and exact
   known/unknown status mapping; MPSGraph range failure is captured before outward rescan mapping.<span data-glossary-navigation="true"> ([RUN glossary](../../../../glossary.md#term-run); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
4. [Runtime](../../../../glossary.md#term-runtime) [trace](../../../../glossary.md#trace) failures disable later trace without changing [backend](../../../../glossary.md#backend) outcomes. An [observer](../../../../glossary.md#term-observer)
   `Error` propagates from success reporting but is suppressed on the already-primary failure
   during PREPARE or either RUN path, including MPSGraph pre-rescan reporting; repeated and
   concurrent session runs retain unique stream-local IDs and documented ordering only.
5. Payloads expose none of the prohibited data and use no string/map/Throwable escape hatch.
6. Public [Engine](../../../../glossary.md#term-engine) composition, `InferenceSession.run`, repeated execution, materialization, ownership,
   close, route, and native call counts remain unchanged. Failure type/message/native status and
   earlier suppression remain intact; only the defined observer <span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`Error` suppression is added.
7. No Engine production, CPU, native source/ABI/export/schema/status/capability, dependency, or build
   change occurs.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [dependency glossary](../../../../glossary.md#term-dependency))</span>
8. Deterministic injected-native tests and an actual public Engine lifecycle collector scenario pass;
   Javadocs/docs/status/diff checks and independent Class C review approve.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## [Validation](../../../../glossary.md#term-validation)

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
with the [observer](../../../../glossary.md#term-observer) collector; direct [backend](../../../../glossary.md#backend)/[DTO](../../../../glossary.md#data-transfer-object--dto) tests are insufficient. Reuse the configured [Metal](../../../../glossary.md#term-metal)
library convention; add no disposable [device](../../../../glossary.md#term-device) probe or public testing hook.

## Documentation and review impact

- Document observer ownership/concurrency/containment, stream-local IDs, disabled cost, emitted
  fields, redaction, and unchanged ABI/lifecycle. Independent Class C review covers public API,
  concurrency, hot path, native failure translation, and Engine-observable behavior.<span data-glossary-navigation="true"> ([observer glossary](../../../../glossary.md#term-observer); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [lifecycle glossary](../../../../glossary.md#lifecycle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [hot path glossary](../../../../glossary.md#term-hot-path); [Engine glossary](../../../../glossary.md#term-engine))</span>

## Result

Implemented the exact public [`MetalTraceObserver`](../../../../glossary.md#metal-trace-observer--metaltraceobserver) and two-[argument](../../../../glossary.md#term-argument)
`MetalBackendIntegration.open(configuration, observer)` surface. The existing one-argument open
retains its [producer](../../../../glossary.md#term-producer)-free path. One package-private integration-owned producer now allocates
independent [atomic](../../../../glossary.md#resource-transaction-and-rollback) event/prepared-unit/invocation IDs, reads `System.nanoTime()`, maps the exact
native status vocabulary, and contains [runtime](../../../../glossary.md#term-runtime) [trace](../../../../glossary.md#trace) failures by permanently disabling later
events. [Callback](../../../../glossary.md#term-callback) `Error` propagates from success reporting. During failure reporting, [PREPARE](../../../../glossary.md#prepare) and
both route-specific [RUN](../../../../glossary.md#term-run) paths preserve the [backend](../../../../glossary.md#backend) failure as primary and append a distinct
acyclic callback `Error` after any existing [rollback](../../../../glossary.md#term-rollback) suppression without disabling tracing.
[Immutable](../../../../glossary.md#term-immutable) prepared facts flow through the selected plan and route-specific executable without an
[Engine](../../../../glossary.md#term-engine) production, [dependency](../../../../glossary.md#term-dependency)/build, capability, native source, [ABI](../../../../glossary.md#term-application-binary-interface-abi)/export, or [schema](../../../../glossary.md#term-schema) change.

[Metal](../../../../glossary.md#term-metal) finalization emits only its final `PREPARE` success/failure outcome, with truthful
`NOT_QUERIED`; each custom or [MPSGraph](../../../../glossary.md#term-mpsgraph) native invocation emits only one `RUN` outcome while tracing
is enabled. MPSGraph range failure is recorded before the existing Java bounds rescan. Payloads
contain only the authorized Trace 0003 components. Deterministic fake-native coverage proves
successful and failed finalization, malformed native finalization with rollback suppression,
custom and MPSGraph run failure precedence, range-status [capture](../../../../glossary.md#term-capture-meanings) and [observer](../../../../glossary.md#term-observer) suppression before
outward Java bounds translation, observer-failure containment and disabled silence, every exact
status mapping, sequence exhaustion, concurrent callback/ID behavior, and rejection of self or
cyclic suppression.

[Validation](../../../../glossary.md#term-validation) passed: nine focused Metal trace tests; all 14 actual public Engine Metal integration
tests against a freshly built existing [dylib](../../../../glossary.md#term-dylib), including [compile](../../../../glossary.md#compile)/session/repeated-run collection and
bounds failure; the focused Engine composition [architecture test](../../../../glossary.md#term-architecture-test); Metal Javadocs; seven affected
Markdown files; and `git diff --check`. The one required full repository build passed with 87
actionable tasks. Review-block remediation then revalidated the focused Metal
producer/prepared-execution tests and the exact public Engine [lifecycle](../../../../glossary.md#lifecycle) collector scenario; the
already-passing full repository build was intentionally not repeated.

Implementation `ba16d942` plus failure-precedence remediation `386705ca` passed final independent
Class C rereview with `APPROVE` and zero findings.
