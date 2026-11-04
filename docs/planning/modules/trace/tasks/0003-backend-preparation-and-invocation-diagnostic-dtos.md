# Task 0003: [Backend](../../../../glossary.md#backend) Preparation and Invocation Diagnostic DTOs

## Status

Complete

Implemented from clean [planning](../../../../glossary.md#term-planning) HEAD `c734f0bb52cd35b94cd794cc580326dca09ec62b`.
The bounded [JDK](../../../../glossary.md#term-java-development-kit-jdk)-only ID and [DTO](../../../../glossary.md#data-transfer-object--dto) surface, documentation, and executable [validation](../../../../glossary.md#term-validation) are complete.
[Metal](../../../../glossary.md#term-metal) 0041 was rebased and promoted to [Ready](../../../../glossary.md#term-ready) in a separate planning-only [commit](../../../../glossary.md#term-commit).

## Change class

Class B — additive public [Trace](../../../../glossary.md#trace) identifiers and [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) DTOs establish a supported diagnostic [API](../../../../glossary.md#term-application-programming-interface-api).
No [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings), emitter, consumer, [runtime](../../../../glossary.md#term-runtime) state, or [backend](../../../../glossary.md#backend) behavior changes.
Use one clean implementation context and one independent public-API/documentation review context.

## Goal

Add the smallest [trace](../../../../glossary.md#trace)-owned vocabulary needed to describe one [backend](../../../../glossary.md#backend) prepared-unit finalization
outcome and each native invocation outcome without importing [producer](../../../../glossary.md#term-producer) types or exposing
backend-specific names. Preserve the existing event envelope and keep `modules/trace` a [JDK](../../../../glossary.md#term-java-development-kit-jdk)-only,
[immutable](../../../../glossary.md#term-immutable) [DTO](../../../../glossary.md#data-transfer-object--dto) [dependency leaf](../../../../glossary.md#term-dependency-leaf).

## [Scope](../../../../glossary.md#term-scope)

### [Trace](../../../../glossary.md#trace)-local [identities](../../../../glossary.md#term-identity-meanings)

Add these public one-component records under `io.github.pho001.synaptik.trace.id`:

- `TraceBackendId(long value)`;
- `TraceDeviceId(long value)`;
- `TracePreparedUnitId(long value)`; and
- `TraceInvocationId(long value)`.

Each ID rejects negative values with `IllegalArgumentException("value must be non-negative")`,
accepts `0..Long.MAX_VALUE`, reserves no sentinel, remains nominally distinct, and makes uniqueness
meaningful only inside its [producer](../../../../glossary.md#term-producer)-defined trace stream. It has no allocator, source string,
namespace, UUID, clock, [registry](../../../../glossary.md#term-registry), or producer-object reference.

### Closed [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) vocabulary

Add these public enums under `io.github.pho001.synaptik.trace.payload`, with exactly the listed
[constants](../../../../glossary.md#term-constant) and no fields or methods:

- `TraceOutcomeStatus`: `SUCCEEDED`, `FAILED`;
- `TraceNumericalProfile`: `STRICT_IEEE`, [`ACCELERATOR`](../../../../glossary.md#term-accelerator);
- `TraceRouteKind`: `CUSTOM_KERNEL`, `GRAPH_EXECUTABLE`;
- `TraceCacheStatus`: `NOT_QUERIED`;
- `TraceNativeStatusKind`: `SUCCESS`, `INVALID_ARGUMENT`, `DEVICE_UNAVAILABLE`,
  `COMMAND_QUEUE_UNAVAILABLE`, `ALLOCATION_FAILED`, `RANGE_OUT_OF_BOUNDS`, `COPY_FAILED`,
  `INTERNAL_ERROR`, `UNSUPPORTED_SHAPE`, `COMPILATION_FAILED`, `INCOMPATIBLE_RESOURCE`,
  `EXECUTION_FAILED`, `UNKNOWN`.

`TraceRouteKind` describes only the two stable execution mechanisms required by the first [producer](../../../../glossary.md#term-producer);
it is not a universal [backend route](../../../../glossary.md#backend-route) [registry](../../../../glossary.md#term-registry). `TraceCacheStatus.NOT_QUERIED` is intentionally the
only current [cache](../../../../glossary.md#term-cache) fact and must never be interpreted as a miss.

Add [immutable](../../../../glossary.md#term-immutable) public record `TraceNativeStatus(TraceNativeStatusKind kind, int code)`. It rejects a
null kind. `SUCCESS` requires code `0`, code `0` requires `SUCCESS`, and every nonzero signed code is
retained exactly. Codes are [backend](../../../../glossary.md#backend)-stream-local; the record defines no global numeric mapping.

Add public record `BackendPreparationOutcome` implementing [`TracePayload`](../../../../glossary.md#trace-payload--tracepayload), with exactly these
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
other public [API](../../../../glossary.md#term-application-programming-interface-api) or implemented interface. `SUCCEEDED` requires a present `SUCCESS` native status;
`FAILED` permits an empty native status for a failure before a status return and otherwise requires
a non-success status. No payload contains producer objects, free-form text, a generic map, a
`Throwable`, path, pointer, handle, [tensor](../../../../glossary.md#tensor) value, [shape](../../../../glossary.md#shape), byte [extent](../../../../glossary.md#term-extent), [fingerprint](../../../../glossary.md#term-fingerprint), or secret.

## Non-goals

- Changing `TraceEvent`, [`TraceEventId`](../../../../glossary.md#trace-event-identity--traceeventid), [`TracePhase`](../../../../glossary.md#trace-phase--tracephase), [`TraceLevel`](../../../../glossary.md#trace-level--tracelevel), [`TracePayload`](../../../../glossary.md#trace-payload--tracepayload), or existing IDs.
- An emitter, observer, sink, consumer, allocator, clock, counter, storage, filtering, logging,
  serialization, schema registry, visitor, or mutable state.<span data-glossary-navigation="true"> ([observer glossary](../../../../glossary.md#term-observer); [serialization glossary](../../../../glossary.md#term-serialization); [schema glossary](../../../../glossary.md#term-schema); [registry glossary](../../../../glossary.md#term-registry); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Compile payloads, transfer/materialization/publication events, integration open/close events,
  partition/schedule/run IDs, generic attributes, or <span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [materialization glossary](../../../../glossary.md#materialization); [publication glossary](../../../../glossary.md#term-publication); [partition glossary](../../../../glossary.md#partition); [run glossary](../../../../glossary.md#term-run))</span>`TraceAttributes`.
- [Metal](../../../../glossary.md#term-metal) names such as [MPSGraph](../../../../glossary.md#term-mpsgraph), NEG, [ABI](../../../../glossary.md#term-application-binary-interface-abi) symbols, or Metal status-code assignments in shared [API](../../../../glossary.md#term-application-programming-interface-api).
- Importing Model, Config, Planning, Compiler, Prepare, Runtime, Engine, backend-contract, or a
  concrete backend; changing Gradle or any other module.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [concrete backend glossary](../../../../glossary.md#concrete-backend); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
  keeps the scoped Trace contract authoritative.
- [Foundational modules — `modules/trace`](../../../../architecture/contracts/foundational-modules.md#modulestrace)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  permits only typed DTOs/IDs and forbids producer dependencies, business logic, and runtime state.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Backend execution — Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>
  keeps tracing passive and non-selecting.
- [Tracing explanation](../../../../architecture/tracing.md) defines producer-owned identity,
  monotonic time, emission, and <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>[`PREPARE`](../../../../glossary.md#prepare)/[`RUN`](../../../../glossary.md#term-run) classification.

Stop if implementation requires a [producer](../../../../glossary.md#term-producer) import, emission [API](../../../../glossary.md#term-application-programming-interface-api), [mutable](../../../../glossary.md#term-mutable) state, generic attribute
map, architecture/[dependency](../../../../glossary.md#term-dependency) change, or modification of an existing [Trace](../../../../glossary.md#trace) public type.

## Dependencies and integration

- Depends on: Trace 0001 and 0002 Complete; stable current Metal profile, route, finalization,
  invocation, and native-status facts.
- Conflicts with: every concurrent Trace ID/payload/API task and Metal 0041 until this task is
  integrated.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Parallel group: None.
- Common base revision: `c734f0bb52cd35b94cd794cc580326dca09ec62b`.
- Integration order: [Trace](../../../../glossary.md#trace) 0003 first; rebase and unblock [Metal](../../../../glossary.md#term-metal) 0041 only after 0003 is Complete.
- Integration [validation](../../../../glossary.md#term-validation): `:modules:trace:test`, `:modules:trace:javadoc`, Markdown/link validation,
  public-shape/import checks, and <span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>`git diff --check`.
- Shared-document integration [owner](../../../../glossary.md#term-owner): Trace 0003 executor.

## Files and symbols

- `modules/trace/.../trace/id/` — the four exact ID records above.
- `modules/trace/.../trace/payload/` — package documentation, five enums, [`TraceNativeStatus`](../../../../glossary.md#trace-native-status--tracenativestatus), and
  the two exact payload records above.
- `modules/trace/src/test/...` — focused ID/[API](../../../../glossary.md#term-application-programming-interface-api)-[shape](../../../../glossary.md#shape) and [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication)/[invariant](../../../../glossary.md#term-invariant) tests.
- After Java stabilizes: affected Javadocs, `docs/architecture/tracing.md`,
  `docs/api/public-api.md`, targeted glossary entries, this brief, [Trace](../../../../glossary.md#trace) master plan, and roadmap.

## Acceptance criteria

1. All named public symbols, packages, enum constants, record components/order, validation, and
   nominal distinctions match this brief exactly.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [validation glossary](../../../../glossary.md#term-validation); [nominal glossary](../../../../glossary.md#term-nominal))</span>
2. Production imports are [JDK](../../../../glossary.md#term-java-development-kit-jdk)-only; the two outcome records are the only new [`TracePayload`](../../../../glossary.md#trace-payload--tracepayload)
   implementations.
3. Existing [Trace](../../../../glossary.md#trace) public types and semantics are byte-for-source unchanged.
4. Automated tests reject negative IDs, nulls, invalid success/code and outcome/status
   combinations, public-shape drift, extra interfaces/methods, mutable containers, producer
   imports, strings/maps, and non-DTO state.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [mutable glossary](../../../../glossary.md#term-mutable); [producer glossary](../../../../glossary.md#term-producer); [DTO glossary](../../../../glossary.md#data-transfer-object--dto))</span>
5. Native codes remain exact signed integers scoped by backend stream; neutral kinds do not publish
   Metal ABI code assignments.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
6. No emitter, observer, consumer, state, dependency, other module, or serialization behavior is
   introduced.<span data-glossary-navigation="true"> ([observer glossary](../../../../glossary.md#term-observer); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [serialization glossary](../../../../glossary.md#term-serialization))</span>
7. Focused tests, final Trace suite, Javadoc, documentation/status synchronization, Markdown, exact
   scope, and diff checks pass; independent public-API/documentation review approves.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [synchronization glossary](../../../../glossary.md#term-synchronization); [scope glossary](../../../../glossary.md#term-scope); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## [Validation](../../../../glossary.md#term-validation)

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

Integration/repository validation: no full build is required because this is an additive [JDK](../../../../glossary.md#term-java-development-kit-jdk)-only
[DTO](../../../../glossary.md#data-transfer-object--dto) task with no [dependency edge](../../../../glossary.md#term-dependency-edge); the independent review reuses successful executable evidence
unless Java changes.

## Follow-up

- [Metal](../../../../glossary.md#term-metal) 0041 was rebased and set [Ready](../../../../glossary.md#term-ready) without broadening its [scope](../../../../glossary.md#term-scope).
- Broader compile/prepare/run/backend payload families, typed attributes, and serialization remain
  Draft work under Trace 0004–0008.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [backend glossary](../../../../glossary.md#backend); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [serialization glossary](../../../../glossary.md#term-serialization); [Trace glossary](../../../../glossary.md#trace))</span>

## Documentation and review impact

- Update only current [Trace](../../../../glossary.md#trace) [API](../../../../glossary.md#term-application-programming-interface-api)/explanation/glossary status and [planning](../../../../glossary.md#term-planning) documents after code lands.
- Independent public-API/documentation review is mandatory because this adds durable public types.

## Result

Implemented the four [nominal](../../../../glossary.md#term-nominal) [backend](../../../../glossary.md#backend)-execution correlation IDs, five exact closed enums,
[`TraceNativeStatus`](../../../../glossary.md#trace-native-status--tracenativestatus), and the two exact outcome [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) records. The implementation preserves every
pre-existing public [Trace](../../../../glossary.md#trace) type source unchanged, adds no [dependency](../../../../glossary.md#term-dependency) or behavior [owner](../../../../glossary.md#term-owner), and keeps
`modules:trace` a [JDK](../../../../glossary.md#term-java-development-kit-jdk)-only [immutable](../../../../glossary.md#term-immutable) [DTO](../../../../glossary.md#data-transfer-object--dto) [leaf](../../../../glossary.md#term-leaf). Focused and full Trace tests, Trace [Javadoc](../../../../glossary.md#term-javadoc),
targeted Markdown/link [validation](../../../../glossary.md#term-validation), exact [scope](../../../../glossary.md#term-scope) inspection, and `git diff --check` passed.
Independent public-[API](../../../../glossary.md#term-application-programming-interface-api)/documentation review follows separately as requested before any [Metal](../../../../glossary.md#term-metal)
implementation.
