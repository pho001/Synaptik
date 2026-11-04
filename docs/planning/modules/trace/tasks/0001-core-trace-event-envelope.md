# Task 0001: Core [Trace Event Envelope](../../../../glossary.md#trace-event-envelope)

## Status

Complete

## Goal

Replace the `modules/trace` placeholder with the smallest typed diagnostic event foundation that
all later [compile](../../../../glossary.md#compile), [prepare](../../../../glossary.md#prepare), [run](../../../../glossary.md#term-run), and [backend](../../../../glossary.md#backend) [trace](../../../../glossary.md#trace) [producers](../../../../glossary.md#term-producer) can share without importing their
domain models. Add a caller-supplied trace-event [identity](../../../../glossary.md#term-identity-meanings), [lifecycle](../../../../glossary.md#lifecycle) phase, diagnostic level,
typed-[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) marker, and [immutable](../../../../glossary.md#term-immutable) generic event envelope.

[Mental model](../../../../glossary.md#term-mental-model):

```text
producer-owned facts
  -> producer translates them into a TracePayload
  -> TraceEvent adds identity, lifecycle phase, level, and monotonic ordering time
  -> diagnostic consumers inspect typed trace DTOs
```

This task establishes only the envelope. It does not define concrete lifecycle payloads,
correlation identifiers beyond the event itself, [serialization](../../../../glossary.md#term-serialization), filtering, storage, or emission.

## [Scope](../../../../glossary.md#term-scope)

- Delete the temporary `TraceModule` marker after the real public [trace](../../../../glossary.md#trace) [API](../../../../glossary.md#term-application-programming-interface-api) exists.
- Add public record [`TraceEventId`](../../../../glossary.md#trace-event-identity--traceeventid) with exactly one `long value` component.
  - Reject a negative value with `IllegalArgumentException` and exact message
    `value must be non-negative`.
  - Accept zero through `Long.MAX_VALUE`.
  - Retain ordinary record value semantics and expose an explicitly documented `value()` accessor.
  - Allocate no [identity](../../../../glossary.md#term-identity-meanings) and promise uniqueness only within the [producer](../../../../glossary.md#term-producer)-defined trace stream.
- Add public enum [`TracePhase`](../../../../glossary.md#trace-phase--tracephase) with exactly [`COMPILE`](../../../../glossary.md#compile), [`PREPARE`](../../../../glossary.md#prepare), and [`RUN`](../../../../glossary.md#term-run) in that order.
  - `COMPILE` covers capture, validation, transformation, ownership, partitioning, logical memory,
    and publication planning diagnostics.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning))</span>
  - `PREPARE` covers backend preparation, selected routes, prepared partitions/units, prepared
    memory, and prepared schedules.
  - `RUN` covers invocation, execution, transfers, materialization, step boundaries, and
    publication.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
  - [Backend](../../../../glossary.md#backend) is not a phase. Backend payloads use the [lifecycle](../../../../glossary.md#lifecycle) phase in which the fact occurs.
- Add public enum [`TraceLevel`](../../../../glossary.md#trace-level--tracelevel) with exactly `TRACE`, `DEBUG`, `INFO`, `WARN`, and `ERROR` in that
  order from most detailed to most severe.
  - The enum classifies an event only. It does not define filtering, thresholds, sinks, logging,
    failure handling, or process exit behavior.
- Add public method-free [marker interface](../../../../glossary.md#term-marker-interface) [`TracePayload`](../../../../glossary.md#trace-payload--tracepayload).
  - Implementations are immutable typed diagnostic DTOs, not producer-domain objects or business
    logic.
  - Keep the marker open so later trace-owned payload families can be added without a premature
    permits list or central registry.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry))</span>
  - Add no [serialization](../../../../glossary.md#term-serialization) parent, visitor, type tag, attribute map, or default method.
- Add public generic record `TraceEvent<T extends TracePayload>` with exactly these components in
  order:
  1. `TraceEventId id`
  2. `TracePhase phase`
  3. `TraceLevel level`
  4. `long monotonicNanos`
  5. `T payload`
- `TraceEvent` must:
  - null-check `id`, `phase`, `level`, and [`payload`](../../../../glossary.md#artifact-envelope-and-atomic-publication) in that order with exact messages equal to the
    component names;
  - accept and retain every `long monotonicNanos` value unchanged, including negative values;
  - retain the exact [immutable](../../../../glossary.md#term-immutable) component references;
  - expose explicitly documented component accessors and ordinary record value semantics; and
  - add no event source, thread, wall-clock instant, duration, parent ID, correlation map, tags,
    attributes, sequence allocator, clock, sink, filtering, or emission behavior.
- Define `monotonicNanos` as a producer-supplied monotonic-clock reading expressed in nanoseconds.
  Only differences interpreted within the producer's documented clock domain are meaningful. It
  is not an epoch timestamp and the record does not enforce ordering across events.<span data-glossary-navigation="true"> ([epoch glossary](../../../../glossary.md#term-time-epoch))</span>
- Add focused tests for exact public shapes, enum order, marker-interface shape, validation order
  and messages, accepted boundaries, reference retention, timestamp preservation, generic payload
  typing, and record equality/hash behavior.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [hash glossary](../../../../glossary.md#term-hash))</span>
- Finalize all new Javadocs and update the focused tracing explanation, public API status, glossary,
  trace master plan, and roadmap in the same overall change.

## Out of [scope](../../../../glossary.md#term-scope)

- [trace](../../../../glossary.md#trace)-local [node](../../../../glossary.md#node), value, [tensor](../../../../glossary.md#tensor), [partition](../../../../glossary.md#partition), [backend](../../../../glossary.md#backend), [device](../../../../glossary.md#term-device), unit, schedule, or [run](../../../../glossary.md#term-run) identifiers
- concrete [compile](../../../../glossary.md#compile), [prepare](../../../../glossary.md#prepare), run, or backend [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) records
- `TraceAttributes`, attribute value variants, maps, tags, [metadata](../../../../glossary.md#term-metadata) bags, or string [dispatch](../../../../glossary.md#term-dispatch)
- Java native serialization, JSON, binary encoding, schema registry, version fields, migration,
  compatibility policy, or persistence<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [JSON glossary](../../../../glossary.md#term-json); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [schema glossary](../../../../glossary.md#term-schema); [registry glossary](../../../../glossary.md#term-registry); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- event allocation, global sequence generation, clock access, timestamp normalization, wall-clock
  time, duration calculation, buffering, storage, sinks, listeners, publishers, subscriptions,
  filtering, sampling, or logging-framework integration<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>
- graph traversal, compiler capture, planning, preparation, runtime state, backend execution, or
  any producer business logic<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [producer glossary](../../../../glossary.md#term-producer))</span>
- imports or dependencies on model, config, planning, compiler, runtime, prepare, engine, backend
  contract, concrete backends, extensions, or external libraries<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [engine glossary](../../../../glossary.md#term-engine))</span>
- Gradle, dependency, Java-version, architecture-contract, ADR, architecture-test, another-module,
  backend-conformance, or integration-test changes<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- a detailed task 0002 or any later trace task specification
- unrelated refactoring or documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Tracing explanation](../../../../architecture/tracing.md)
- [Typed trace DTO ADR](../../../../design/decisions/0003-typed-trace-dtos.md)<span data-glossary-navigation="true"> ([Typed trace DTO glossary](../../../../glossary.md#typed-trace-dto); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Trace master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>
- [Public API status](../../../../api/public-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/trace` owns typed diagnostic DTOs only and remains a [JDK](../../../../glossary.md#term-java-development-kit-jdk)-only [dependency leaf](../../../../glossary.md#term-dependency-leaf).
- Trace contracts must not import producer-domain types or make trace responsible for graph
  traversal, business logic, execution, or runtime state.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [producer glossary](../../../../glossary.md#term-producer); [graph glossary](../../../../glossary.md#graph); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Typed payloads are the primary [model](../../../../glossary.md#term-model). This task adds no string map or generic untyped [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication).
- Producers translate local facts and identities into trace-owned DTOs. This task does not move
  producer facts or ownership into trace.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Trace records are observational metadata. Their construction must not mutate or control the
  producing lifecycle.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `ARCHITECTURE.md` remains unchanged. Stop if the implementation needs a new dependency,
  ownership rule, producer integration, or architecture-test change.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.trace` — deliberate small public root for contracts shared by every
  trace producer and consumer<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>

No subpackage is added in this task. Later correlation identifiers, concrete [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) families,
and typed attributes use the `id`, `payload`, and `attribute` subpackages selected by the [trace](../../../../glossary.md#trace)
master plan.

Type placement:

- `io.github.pho001.synaptik.trace.TraceEventId` — [identity](../../../../glossary.md#term-identity-meanings) of the common event envelope
- `io.github.pho001.synaptik.trace.TracePhase` — common [lifecycle](../../../../glossary.md#lifecycle) discriminator
- `io.github.pho001.synaptik.trace.TraceLevel` — common diagnostic detail/severity discriminator
- `io.github.pho001.synaptik.trace.TracePayload` — common generic payload bound
- `io.github.pho001.synaptik.trace.TraceEvent` — common typed envelope

Test types mirror the production root package.

## Affected files

Production — exactly six paths:

- delete `modules/trace/src/main/java/io/github/pho001/synaptik/trace/TraceModule.java`
- add `modules/trace/src/main/java/io/github/pho001/synaptik/trace/TraceEventId.java`
- add `modules/trace/src/main/java/io/github/pho001/synaptik/trace/TracePhase.java`
- add `modules/trace/src/main/java/io/github/pho001/synaptik/trace/TraceLevel.java`
- add `modules/trace/src/main/java/io/github/pho001/synaptik/trace/TracePayload.java`
- add `modules/trace/src/main/java/io/github/pho001/synaptik/trace/TraceEvent.java`

Tests — exactly three paths:

- delete `modules/trace/src/test/java/io/github/pho001/synaptik/trace/.gitkeep`
- add `modules/trace/src/test/java/io/github/pho001/synaptik/trace/TraceEventIdTest.java`
- add `modules/trace/src/test/java/io/github/pho001/synaptik/trace/TraceEventTest.java`

Documentation and [planning](../../../../glossary.md#term-planning) — exactly six paths:

- `docs/architecture/tracing.md`
- `docs/api/public-api.md`
- `docs/glossary.md`
- add and finalize this task
- `docs/planning/modules/trace/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: `AGENTS.md`, `ARCHITECTURE.md`, the typed-[trace](../../../../glossary.md#trace) [ADR](../../../../glossary.md#term-architecture-decision-record-adr), other focused
architecture pages, architecture tests, [Gradle](../../../../glossary.md#term-gradle), other [modules](../../../../glossary.md#term-module), [conformance](../../../../glossary.md#term-conformance) tests, integration
tests, and legacy evidence.

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact fifteen paths above. Stop if implementation requires another production type,
test, documentation page, [dependency](../../../../glossary.md#term-dependency), package, [module](../../../../glossary.md#term-module), architecture change, or build edit.

## Acceptance criteria

- `TraceModule` and the test placeholder are removed after real contracts/tests replace them.
- [`TraceEventId`](../../../../glossary.md#trace-event-identity--traceeventid) is a public one-component record with exact validation, boundary acceptance,
  accessor contract, and value semantics.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- [`TracePhase`](../../../../glossary.md#trace-phase--tracephase) and [`TraceLevel`](../../../../glossary.md#trace-level--tracelevel) contain exactly the specified constants in the specified order and
  add no project fields, methods, nested types, aliases, producer logic, or policy.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [producer glossary](../../../../glossary.md#term-producer))</span>
- [`TracePayload`](../../../../glossary.md#trace-payload--tracepayload) is a public open method-free [marker interface](../../../../glossary.md#term-marker-interface) with no parent interface.
- `TraceEvent` is a public generic five-component record with the exact type bound, component
  order, validation order/messages, reference retention, timestamp semantics, accessors, and
  record value behavior.
- The complete production [module](../../../../glossary.md#term-module) imports only [JDK](../../../../glossary.md#term-java-development-kit-jdk) classes and its own [trace](../../../../glossary.md#trace) contracts.
- No type implements `Serializable`, selects an encoding, exposes a map, allocates identity, reads
  a clock, emits an event, or imports a producer layer.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Focused tests cover every specified [invariant](../../../../glossary.md#term-invariant) and fail if the public envelope [shape](../../../../glossary.md#shape) drifts.
- Javadocs explain purpose, units, nullability, ownership, value semantics, limitations, every
  parameter/result, and expected failure without promising concrete payload schemas or emission.<span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [parameter glossary](../../../../glossary.md#term-parameter); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [schemas glossary](../../../../glossary.md#term-schema))</span>
- The tracing explanation distinguishes the now-current envelope from still-conceptual payload
  families and explains why backend is not a lifecycle phase.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Public [API](../../../../glossary.md#term-application-programming-interface-api) status and glossary describe only the implemented foundation as current.
- Trace master plan and roadmap identify task 0001 as the sole Ready frontier and keep later trace
  tasks Draft without detailed specifications.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- A separate clean-context documentation-focused pass finalizes Javadocs and affected docs after
  Java tests pass.
- Exact fifteen-path [scope](../../../../glossary.md#term-scope), Markdown, final newlines, trailing whitespace, and `git diff --check`
  pass; no architecture, dependency, build, or cross-module path changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [module path glossary](../../../../glossary.md#term-module-path))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) focused tests as needed. After executable Java stabilizes, run exactly
one final [trace](../../../../glossary.md#trace) [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:trace:test
```

Record test and suite counts from the XML reports. Do not run the repository-wide suite: this is a
small [JDK](../../../../glossary.md#term-java-development-kit-jdk)-only single-module task, changes no [dependency](../../../../glossary.md#term-dependency) declaration or architecture rule, and the
trace milestone will own its repository [checkpoint](../../../../glossary.md#term-checkpoint).

Hand the actual diff and exact Java evidence to a separate clean-context documentation-focused
agent in the same overall change. That pass reads the final source/tests and selected General plus
[API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc) profiles, finalizes all new Javadocs and the six affected documentation/[planning](../../../../glossary.md#term-planning)
files, reviews glossary impact, and runs:

```bash
./gradlew :modules:trace:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass reuses the successful Java evidence unless it changes executable Java or
records a concrete reason to rerun. Also inspect imports and public type [shapes](../../../../glossary.md#shape) through the focused
tests, confirm generated Javadoc contains all five contracts, confirm exactly fifteen authorized
paths, and confirm 0001 Complete/0002 Draft/no detailed 0002 task before completion.

## Dependencies

- Completed selected `modules/model` milestone, which satisfies the roadmap entry condition.
- Existing authoritative [trace](../../../../glossary.md#trace) [ownership](../../../../glossary.md#term-ownership-meanings)/[dependency](../../../../glossary.md#term-dependency) contract and accepted typed-trace [DTO](../../../../glossary.md#data-transfer-object--dto) [ADR](../../../../glossary.md#term-architecture-decision-record-adr).
- Java 26 repository baseline and existing [JDK](../../../../glossary.md#term-java-development-kit-jdk)-only trace [module](../../../../glossary.md#term-module).

## Follow-up tasks

- Task 0002: trace-local correlation identifiers. It remains Draft without a detailed
  specification.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>
- Later Draft tasks add typed attributes, compile/prepare/run/backend payload families, and final
  serialization/schema validation in dependency order.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [backend glossary](../../../../glossary.md#backend); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [serialization glossary](../../../../glossary.md#term-serialization); [schema glossary](../../../../glossary.md#term-schema); [validation glossary](../../../../glossary.md#term-validation); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Architecture impact

Expected impact: None.

The task implements the already-authoritative [trace](../../../../glossary.md#trace) [DTO](../../../../glossary.md#data-transfer-object--dto) boundary. It changes no [ownership](../../../../glossary.md#term-ownership-meanings) or
[dependency](../../../../glossary.md#term-dependency) rule. The focused tracing explanation changes implementation-status wording only.

## Implementation prompt

Use this prompt in one separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, focused tracing/module/dependency architecture documents, the
typed-trace ADR, documentation/planning rules and profiles, roadmap, trace master plan, task 0001,
the current trace placeholder/module build, Public API, glossary, and Java 26 root configuration.

Implement docs/planning/modules/trace/tasks/0001-core-trace-event-envelope.md exactly inside its
fifteen authorized paths. Replace only the trace placeholder with TraceEventId, TracePhase,
TraceLevel, TracePayload, TraceEvent, and focused tests. Preserve the JDK-only DTO leaf, exact API,
validation, timestamp, and no-serialization/no-emission boundaries. Stop on architecture,
dependency, package, affected-file, or maximum-scope conflict. Do not implement later trace tasks.

Run focused tests while developing and exactly one final :modules:trace:test after executable Java
stabilizes. Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That pass must inspect final source/tests,
finalize Javadocs, tracing explanation, Public API, glossary, task/master/roadmap status, run trace
Javadoc plus documentation/scope checks, and reuse successful Java evidence unless executable
behavior changes. Mark 0001 Complete only after both passes and every criterion succeed. Leave
0002 and later tasks Draft without detailed specifications.
```

## Local decisions

- Keep the first task to the envelope foundation. Concrete payload families depend on producer
  contracts that are not yet stable and would make this task speculative.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [producer glossary](../../../../glossary.md#term-producer))</span>
- Use [lifecycle](../../../../glossary.md#lifecycle) phases [`COMPILE`](../../../../glossary.md#compile), [`PREPARE`](../../../../glossary.md#prepare), and [`RUN`](../../../../glossary.md#term-run). Backend is an emitting owner/payload family;
  treating it as a phase would lose whether a backend fact occurred during prepare or run.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [owner glossary](../../../../glossary.md#term-owner))</span>
- Use [`TRACE`](../../../../glossary.md#trace), `DEBUG`, `INFO`, `WARN`, and `ERROR` as classification only, with no filtering or
  logging-framework contract.
- Keep [`TracePayload`](../../../../glossary.md#trace-payload--tracepayload) open and method-free for incremental trace-owned DTO additions. A registry or
  sealed hierarchy is unnecessary before concrete payload families exist.<span data-glossary-navigation="true"> ([DTO glossary](../../../../glossary.md#data-transfer-object--dto); [registry glossary](../../../../glossary.md#term-registry); [sealed glossary](../../../../glossary.md#term-sealed))</span>
- Accept every timestamp bit pattern. Monotonic clocks have arbitrary origins and may wrap; only
  producer-domain differences are meaningful.
- Defer serialization format and compatibility policy until the concrete schemas exist. Do not
  equate “serializable DTO” in the architecture goal with Java native serialization.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [schemas glossary](../../../../glossary.md#term-schema))</span>
- Use caller-assigned event identity. Allocation and clock ownership belong to producers, not the
  dependency-leaf DTO module.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency glossary](../../../../glossary.md#term-dependency); [leaf glossary](../../../../glossary.md#term-leaf); [module glossary](../../../../glossary.md#term-module))</span>

## Known limitations

- The open marker cannot enforce that third-party implementations are immutable or trace-owned;
  documented ownership and later schema validation provide that boundary without a premature
  closed permits list.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [trace glossary](../../../../glossary.md#trace); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- Events from different clock domains cannot be ordered through `monotonicNanos` alone.
- The foundation cannot yet be encoded, persisted, emitted, filtered, or correlated to model/
  compiler/runtime identities; later tasks own those capabilities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Enum evolution and external [schema](../../../../glossary.md#term-schema) compatibility remain unsettled until [serialization](../../../../glossary.md#term-serialization) [planning](../../../../glossary.md#term-planning).

## [Validation](../../../../glossary.md#term-validation) evidence

- The implementation context `/root/implement_trace_0001` ran the development command
  `./gradlew :modules:trace:test --tests io.github.pho001.synaptik.trace.TraceEventTest`; it passed
  all eight selected tests.
- After executable Java stabilized, that implementation context ran exactly one final
  `./gradlew :modules:trace:test`; [Gradle](../../../../glossary.md#term-gradle) reported `BUILD SUCCESSFUL` in 913 ms. XML reports contain
  two suites: `TraceEventTest` has 8 tests and `TraceEventIdTest` has 4 tests, for 12 tests total
  with zero skipped, failures, or errors.
- The separate clean documentation context
  `/root/implement_trace_0001/trace_0001_docs` used General style with API/Javadoc as the primary
  profile, Architecture for <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`tracing.md`, and Planning for the task, master plan, and roadmap. It
  read the architecture trace contract, focused boundary and tracing explanations, typed-trace
  ADR, documentation/planning rules, final source/tests, public API status, glossary, and root plus
  trace Gradle configuration before finalizing the diff.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [trace glossary](../../../../glossary.md#trace); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- The documentation context changed no executable behavior and therefore reused the successful
  Java evidence as required. It ran `./gradlew :modules:trace:javadoc`; Gradle reported
  `BUILD SUCCESSFUL` in 1 s with two executed tasks, and generated output contains `TraceEvent`,
  [`TraceEventId`](../../../../glossary.md#trace-event-identity--traceeventid), [`TraceLevel`](../../../../glossary.md#trace-level--tracelevel), [`TracePayload`](../../../../glossary.md#trace-payload--tracepayload), and [`TracePhase`](../../../../glossary.md#trace-phase--tracephase) pages.
- `python3 /tmp/validate_synaptik_markdown.py` passed 214 Markdown files, 3,664 local links, 211
  local anchors, 2,694 fence markers, final newlines, and trailing whitespace. `git diff --check`
  also passed.
- The final [scope](../../../../glossary.md#term-scope) command
  `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u` and
  `git status --short` confirmed all fifteen authorized task paths: six production, three test,
  and six documentation/planning paths. They also reported seven unrelated pre-existing
  model/closure-audit paths in the shared tree; those changes were preserved. Existing model
  updates within the authorized shared Public API and roadmap files were also preserved.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Source and focused-test inspection confirms the exact public record, enum, marker, generic bound,
  validation, timestamp, reference-retention, and ordinary record-value shapes. Production imports
  are limited to <span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape))</span>`java.util.Objects` and same-package trace contracts; the trace Gradle project has
  no module-specific dependency declaration.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Task status inspection confirms task 0001 is Complete, task 0002 and every later trace row remain
  Draft, and `docs/planning/modules/trace/tasks/` contains no detailed task 0002 specification.
- No changes were needed in `ARCHITECTURE.md`, the current architecture index, module boundaries,
  dependency rules, ADR 0003, architecture tests, Gradle, backend conformance, integration tests,
  Compile API, Runtime API, or another module. The task implements an existing DTO-leaf rule and
  changes no dependency, build, compiler/runtime/backend behavior, or cross-module contract.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [DTO glossary](../../../../glossary.md#data-transfer-object--dto); [leaf glossary](../../../../glossary.md#term-leaf); [compiler glossary](../../../../glossary.md#term-compiler))</span>

## Implementation notes

- Deleted the placeholder production marker and test `.gitkeep`, then added the five public event
  contracts and two focused test classes in the planned root package.
- Null [validation](../../../../glossary.md#term-validation) uses component order `id`, `phase`, `level`, then [`payload`](../../../../glossary.md#artifact-envelope-and-atomic-publication); timestamp values are
  retained without validation, and event ID accepts exactly the non-negative `long` domain.
- The documentation pass clarified shallow record immutability and the payload implementation
  obligation without changing Java behavior. It also separated the current envelope from planned
  payload, correlation, attribute, serialization, sink, and emission capabilities.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization))</span>

## Completion summary

- Completed changes: replaced the trace placeholder with the caller-supplied event identity,
  lifecycle phase, diagnostic level, open typed-payload marker, generic event envelope, and focused
  exact-shape/behavior tests.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [identity glossary](../../../../glossary.md#term-identity-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [shape glossary](../../../../glossary.md#shape))</span>
- Files changed or created: exactly the six production, three test, and six documentation/planning
  paths listed under Affected files.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: the implementation context passed 8 focused development tests and one
  final 12-test/two-suite trace module run; the documentation context passed trace Javadoc,
  repository Markdown, whitespace, generated-page, package/import, task-status, and exact-scope
  checks without rerunning Java tests.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context
  `/root/implement_trace_0001/trace_0001_docs` independently finalized all five public Javadocs,
  the tracing explanation, Public API status, glossary, task, trace master plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Documentation impact: the current envelope and producer ownership/timestamp boundaries are
  documented; concrete payload families and event processing remain explicitly planned.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Javadoc review: all five public contracts and every public constructor, accessor, and enum
  constant document purpose, ownership, units where applicable, nullability, results, failures,
  value semantics, and unsupported behavior without promising serialization or emission.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [nullability glossary](../../../../glossary.md#term-nullability); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Glossary impact: updated the implementation-status boundary and added current definitions for the
  trace envelope, event identity, level, payload marker, and lifecycle phase.
- Unresolved issues: None.
- Follow-up required: None. Task 0002 remains the next Draft frontier without a detailed task.

Status: Complete
