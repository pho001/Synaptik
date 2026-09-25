# Task 0003: Backend Preparation and Invocation Diagnostic DTOs

## Status

Ready

Readiness is verified from Trace 0001–0002 Complete, the current `PREPARE`/`RUN` lifecycle,
completed profile contracts, stable Metal route/finalization/invocation facts, and the existing
native-status taxonomy. This is the sole authorized repository frontier. It defines data only;
Metal 0041 remains blocked until this task is Complete.

## Change class

Class B — additive public Trace identifiers and payload DTOs establish a supported diagnostic API.
No dependency, lifecycle ownership, emitter, consumer, runtime state, or backend behavior changes.
Use one clean implementation context and one independent public-API/documentation review context.

## Goal

Add the smallest trace-owned vocabulary needed to describe one backend prepared-unit finalization
outcome and each native invocation outcome without importing producer types or exposing
backend-specific names. Preserve the existing event envelope and keep `modules/trace` a JDK-only,
immutable DTO dependency leaf.

## Scope

### Trace-local identities

Add these public one-component records under `io.github.pho001.synaptik.trace.id`:

- `TraceBackendId(long value)`;
- `TraceDeviceId(long value)`;
- `TracePreparedUnitId(long value)`; and
- `TraceInvocationId(long value)`.

Each ID rejects negative values with `IllegalArgumentException("value must be non-negative")`,
accepts `0..Long.MAX_VALUE`, reserves no sentinel, remains nominally distinct, and makes uniqueness
meaningful only inside its producer-defined trace stream. It has no allocator, source string,
namespace, UUID, clock, registry, or producer-object reference.

### Closed payload vocabulary

Add these public enums under `io.github.pho001.synaptik.trace.payload`, with exactly the listed
constants and no fields or methods:

- `TraceOutcomeStatus`: `SUCCEEDED`, `FAILED`;
- `TraceNumericalProfile`: `STRICT_IEEE`, `ACCELERATOR`;
- `TraceRouteKind`: `CUSTOM_KERNEL`, `GRAPH_EXECUTABLE`;
- `TraceCacheStatus`: `NOT_QUERIED`;
- `TraceNativeStatusKind`: `SUCCESS`, `INVALID_ARGUMENT`, `DEVICE_UNAVAILABLE`,
  `COMMAND_QUEUE_UNAVAILABLE`, `ALLOCATION_FAILED`, `RANGE_OUT_OF_BOUNDS`, `COPY_FAILED`,
  `INTERNAL_ERROR`, `UNSUPPORTED_SHAPE`, `COMPILATION_FAILED`, `INCOMPATIBLE_RESOURCE`,
  `EXECUTION_FAILED`, `UNKNOWN`.

`TraceRouteKind` describes only the two stable execution mechanisms required by the first producer;
it is not a universal backend route registry. `TraceCacheStatus.NOT_QUERIED` is intentionally the
only current cache fact and must never be interpreted as a miss.

Add immutable public record `TraceNativeStatus(TraceNativeStatusKind kind, int code)`. It rejects a
null kind. `SUCCESS` requires code `0`, code `0` requires `SUCCESS`, and every nonzero signed code is
retained exactly. Codes are backend-stream-local; the record defines no global numeric mapping.

Add public record `BackendPreparationOutcome` implementing `TracePayload`, with exactly these
components in order:

1. `TraceBackendId backendId`
2. `TraceDeviceId deviceId`
3. `TracePreparedUnitId preparedUnitId`
4. `TraceOutcomeStatus status`
5. `TraceNumericalProfile profile`
6. `TraceRouteKind route`
7. `TraceCacheStatus cacheStatus`
8. `Optional<TraceNativeStatus> nativeStatus`

Add public record `BackendInvocationOutcome` implementing `TracePayload`, with exactly these
components in order:

1. `TraceBackendId backendId`
2. `TraceDeviceId deviceId`
3. `TracePreparedUnitId preparedUnitId`
4. `TraceInvocationId invocationId`
5. `TraceOutcomeStatus status`
6. `TraceNumericalProfile profile`
7. `TraceRouteKind route`
8. `Optional<TraceNativeStatus> nativeStatus`

Both payloads null-check components in declaration order, retain immutable references, and add no
other public API or implemented interface. `SUCCEEDED` requires a present `SUCCESS` native status;
`FAILED` permits an empty native status for a failure before a status return and otherwise requires
a non-success status. No payload contains producer objects, free-form text, a generic map, a
`Throwable`, path, pointer, handle, tensor value, shape, byte extent, fingerprint, or secret.

## Non-goals

- Changing `TraceEvent`, `TraceEventId`, `TracePhase`, `TraceLevel`, `TracePayload`, or existing IDs.
- An emitter, observer, sink, consumer, allocator, clock, counter, storage, filtering, logging,
  serialization, schema registry, visitor, or mutable state.
- Compile payloads, transfer/materialization/publication events, integration open/close events,
  partition/schedule/run IDs, generic attributes, or `TraceAttributes`.
- Metal names such as MPSGraph, NEG, ABI symbols, or Metal status-code assignments in shared API.
- Importing Model, Config, Planning, Compiler, Prepare, Runtime, Engine, backend-contract, or a
  concrete backend; changing Gradle or any other module.

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
  keeps the scoped Trace contract authoritative.
- [Foundational modules — `modules/trace`](../../../../architecture/contracts/foundational-modules.md#modulestrace)
  permits only typed DTOs/IDs and forbids producer dependencies, business logic, and runtime state.
- [Backend execution — Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
  keeps tracing passive and non-selecting.
- [Tracing explanation](../../../../architecture/tracing.md) defines producer-owned identity,
  monotonic time, emission, and `PREPARE`/`RUN` classification.

Stop if implementation requires a producer import, emission API, mutable state, generic attribute
map, architecture/dependency change, or modification of an existing Trace public type.

## Dependencies and integration

- Depends on: Trace 0001 and 0002 Complete; stable current Metal profile, route, finalization,
  invocation, and native-status facts.
- Conflicts with: every concurrent Trace ID/payload/API task and Metal 0041 until this task is
  integrated.
- Parallel group: None.
- Common base revision: `790454309df60b44e805c8552549e935611fd4c2`.
- Integration order: Trace 0003 first; rebase and unblock Metal 0041 only after 0003 is Complete.
- Integration validation: `:modules:trace:test`, `:modules:trace:javadoc`, Markdown/link validation,
  public-shape/import checks, and `git diff --check`.
- Shared-document integration owner: Trace 0003 executor.

## Files and symbols

- `modules/trace/.../trace/id/` — the four exact ID records above.
- `modules/trace/.../trace/payload/` — package documentation, five enums, `TraceNativeStatus`, and
  the two exact payload records above.
- `modules/trace/src/test/...` — focused ID/API-shape and payload/invariant tests.
- After Java stabilizes: affected Javadocs, `docs/architecture/tracing.md`,
  `docs/api/public-api.md`, targeted glossary entries, this brief, Trace master plan, and roadmap.

## Acceptance criteria

1. All named public symbols, packages, enum constants, record components/order, validation, and
   nominal distinctions match this brief exactly.
2. Production imports are JDK-only; the two outcome records are the only new `TracePayload`
   implementations.
3. Existing Trace public types and semantics are byte-for-source unchanged.
4. Automated tests reject negative IDs, nulls, invalid success/code and outcome/status
   combinations, public-shape drift, extra interfaces/methods, mutable containers, producer
   imports, strings/maps, and non-DTO state.
5. Native codes remain exact signed integers scoped by backend stream; neutral kinds do not publish
   Metal ABI code assignments.
6. No emitter, observer, consumer, state, dependency, other module, or serialization behavior is
   introduced.
7. Focused tests, final Trace suite, Javadoc, documentation/status synchronization, Markdown, exact
   scope, and diff checks pass; independent public-API/documentation review approves.

## Validation

Worker validation:

```bash
./gradlew :modules:trace:test
./gradlew :modules:trace:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/architecture/tracing.md docs/api/public-api.md docs/glossary.md \
  docs/planning/modules/trace/tasks/0003-backend-preparation-and-invocation-diagnostic-dtos.md \
  docs/planning/modules/trace/master-plan.md docs/planning/roadmap.md
git diff --check
```

Integration/repository validation: no full build is required because this is an additive JDK-only
DTO task with no dependency edge; the independent review reuses successful executable evidence
unless Java changes.

## Follow-up

- After completion, rebase and set Metal 0041 Ready without broadening its scope.
- Broader compile/prepare/run/backend payload families, typed attributes, and serialization remain
  Draft work under Trace 0004–0008.

## Documentation and review impact

- Update only current Trace API/explanation/glossary status and planning documents after code lands.
- Independent public-API/documentation review is mandatory because this adds durable public types.

## Result

Empty until execution.
