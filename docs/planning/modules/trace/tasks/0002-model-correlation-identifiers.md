# Task 0002: Model Correlation Identifiers

## Status

Complete

## Goal

Add three trace-local identifier domains for correlating diagnostic events with stable model
concepts without importing model types or retaining producer objects. A producer may translate a
model node, logical value, or public Tensor identity into `TraceNodeId`, `TraceValueId`, or
`TraceTensorId`; trace consumers then correlate typed payloads using trace-owned values.

Mental model:

```text
producer-owned NodeId / ValueId / TensorId
          producer-defined translation
                     ↓
TraceNodeId / TraceValueId / TraceTensorId
                     ↓
later typed TracePayload records
```

This task defines correlation values only. It adds no payload, mapping table, translator,
allocator, registry, graph traversal, or dependency on `modules/model`.

## Scope

- Add package `io.github.pho001.synaptik.trace.id` with package documentation defining the
  producer-translation and trace-stream scope boundary.
- Add public record `TraceNodeId` with exactly one `long value` component.
  - It identifies a computation occurrence for diagnostic correlation.
  - It does not identify operation semantics, an output value, a producer object, or a runtime
    unit.
- Add public record `TraceValueId` with exactly one `long value` component.
  - It identifies logical graph data for diagnostic correlation.
  - It does not identify a node, public Tensor, storage location, buffer, or runtime slot.
- Add public record `TraceTensorId` with exactly one `long value` component.
  - It identifies public Tensor state for diagnostic correlation.
  - It does not identify a graph node/value, storage address, device allocation, or runtime
    residency.
- Each record must:
  - be final through record semantics and add no other component, field, constructor, public
    method, nested type, or implemented interface;
  - reject a negative value with `IllegalArgumentException` and exact message
    `value must be non-negative`;
  - accept zero through `Long.MAX_VALUE`, reserve no sentinel, and retain the exact primitive;
  - expose an explicitly documented `value()` accessor;
  - preserve ordinary record equality, hashing, and diagnostic `toString()` semantics; and
  - allocate nothing and make no process-wide or cross-stream uniqueness promise.
- Define every ID as trace-local within the producer-defined trace stream or correlation domain.
  The producer owns allocation, uniqueness, lifetime, and the mapping from its local identity.
- Do not require a trace-local numeric value to equal the producer identifier's numeric value.
  Translation may preserve or remap values as long as the producer maintains its documented
  correlation mapping.
- Keep the three types nominally distinct. Add no common `TraceId`, generic identifier, string
  namespace, kind tag, source field, graph ID, union, factory, converter, or registry.
- Add one focused test class covering exact API shape, package, validation/message, boundaries,
  nominal separation, lack of serialization/interfaces/nested types, and record value behavior for
  all three records.
- Finalize Javadocs and update the focused tracing explanation, Public API status, glossary, trace
  master plan, and roadmap in the same overall change.

## Out of scope

- `TracePartitionId`, `TraceBackendId`, `TraceDeviceId`, `TraceUnitId`, `TraceScheduleId`,
  `TraceRunId`, or any other correlation domain
- changing `TraceEventId`, `TraceEvent`, `TracePhase`, `TraceLevel`, or `TracePayload`
- concrete compile, prepare, run, or backend payloads
- importing, accepting, returning, wrapping, or reflecting on model `NodeId`, `ValueId`,
  `TensorId`, `Tensor`, graph records, or any producer-domain type
- a translator API, mapping context, identity allocator, registry, cache, global counter, factory,
  builder, parser, formatter, random/UUID/time identity, or reset/test hook
- storing source module, graph, session, stream, run, parent, or namespace metadata in an ID
- Java native serialization, external encoding, schema/version policy, attributes, emission,
  filtering, sinks, logging, traversal, execution, or runtime state
- dependencies, Gradle, Java version, architecture-contract, ADR, architecture-test, another
  module, conformance, or integration changes
- a detailed task 0003 or later specification
- unrelated refactoring or documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Tracing explanation](../../../../architecture/tracing.md)
- [Typed trace DTO ADR](../../../../design/decisions/0003-typed-trace-dtos.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Trace master plan](../master-plan.md)
- [Task 0001](0001-core-trace-event-envelope.md)
- [Public API status](../../../../api/public-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/trace` remains a JDK-only DTO dependency leaf.
- Trace-local IDs prevent diagnostic correlation from importing or exposing model and later-layer
  identifiers. Producers translate; trace never traverses or introspects producer state.
- IDs carry correlation identity only. They add no business meaning, ownership decision, storage,
  execution state, or lifecycle behavior.
- The task implements the already-authoritative trace-local-ID boundary and changes no ownership
  or dependency rule.
- Stop if implementation needs a model import, another module, mapping service, shared registry,
  architecture change, or dependency edit.

## Package impact

Existing package retained:

- `io.github.pho001.synaptik.trace` — current shared event envelope and discriminators; unchanged

Package added:

- `io.github.pho001.synaptik.trace.id` — public trace-local correlation values translated from
  producer identities, with no producer dependency

Type placement:

- `io.github.pho001.synaptik.trace.id.TraceNodeId` — node-occurrence diagnostic correlation
- `io.github.pho001.synaptik.trace.id.TraceValueId` — logical-value diagnostic correlation
- `io.github.pho001.synaptik.trace.id.TraceTensorId` — public-Tensor diagnostic correlation

The focused test mirrors `io.github.pho001.synaptik.trace.id`. No helper package is added.

## Affected files

Production — exactly four paths:

- add `modules/trace/src/main/java/io/github/pho001/synaptik/trace/id/package-info.java`
- add `modules/trace/src/main/java/io/github/pho001/synaptik/trace/id/TraceNodeId.java`
- add `modules/trace/src/main/java/io/github/pho001/synaptik/trace/id/TraceValueId.java`
- add `modules/trace/src/main/java/io/github/pho001/synaptik/trace/id/TraceTensorId.java`

Tests — exactly one path:

- add `modules/trace/src/test/java/io/github/pho001/synaptik/trace/id/TraceModelCorrelationIdTest.java`

Documentation and planning — exactly six paths:

- `docs/architecture/tracing.md`
- `docs/api/public-api.md`
- `docs/glossary.md`
- add and finalize this task
- `docs/planning/modules/trace/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: task-0001 source/tests/Javadocs, model identity source/tests,
`AGENTS.md`, `ARCHITECTURE.md`, the typed-trace ADR, other architecture documents/tests, Gradle,
other modules, backend conformance, integration tests, and legacy evidence.

## Maximum scope

At most the exact eleven paths above. Stop if implementation requires another production type,
test, document, package, dependency, module, architecture change, or build edit.

## Acceptance criteria

- The three records exist in the exact `trace.id` package with exactly one `long value` component
  each and no additional public API or implemented interface.
- All records follow the exact non-negative validation, message, boundary acceptance, accessor,
  and record-value contracts.
- Nominal typing prevents node, value, and tensor correlation IDs from comparing equal or being
  substituted without an explicit caller conversion.
- Package and type Javadocs explain trace-local scope, producer translation, allocation ownership,
  exact meaning, distinctions, and unsupported global/cross-stream guarantees.
- No record stores or imports a model/producer ID, promises numeric preservation, implements
  serialization, or allocates identity.
- Production imports are JDK-only; the preferred implementation needs no import at all.
- The focused test validates all stated contracts for all three types and fails on public-shape
  drift.
- Tracing documentation marks only these three model-correlation IDs current and keeps partition,
  backend, unit, concrete payload, and serialization contracts planned.
- Public API status and glossary distinguish trace-local correlations from producer identities.
- Trace master plan and roadmap identify 0002 as Complete and retain 0003+ as Draft without a
  detailed task-0003 specification or a prematurely selected next frontier.
- A separate clean-context documentation-focused pass finalizes Javadocs and affected docs after
  Java tests pass.
- Exact eleven-path scope, Markdown, final newlines, trailing whitespace, and `git diff --check`
  pass; no task-0001 Java/test, architecture, dependency, build, or cross-module path changes.

## Tests / validation

Run focused tests while developing. After executable Java stabilizes, run exactly one final trace
module suite:

```bash
./gradlew :modules:trace:test
```

Record test/suite counts from XML reports. Do not run repository-wide tests: this is a small
JDK-only additive DTO task with no dependency or architecture-rule change, and the trace milestone
will own its repository checkpoint.

Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That pass independently inspects final
source/tests and task-0001 contracts, applies the General and API/Javadoc profiles, finalizes all
new Javadocs and the six affected documentation/planning paths, records no-change conclusions,
and runs:

```bash
./gradlew :modules:trace:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass reuses the successful Java evidence unless executable Java changes or it
records a concrete reason to rerun. Inspect generated pages for the package and all three IDs;
confirm the focused automated API-shape/import checks; confirm exactly eleven task paths; and
confirm 0002 Complete/0003 Draft/no detailed 0003 task before completion.

## Dependencies

- Completed task 0001 core trace event envelope.
- Completed model identity contracts as stable source concepts only; this task adds no model
  dependency.

## Follow-up tasks

- Reassess the ordered frontier after 0002. Task 0003 typed trace attributes remains Draft without
  a detailed specification.
- Partition, backend, and prepared-unit trace identifiers remain with later payload tasks after
  their producer contracts stabilize.
- If exact payload schemas remain premature, record an explicit roadmap interleave with the owning
  project area rather than inventing trace schemas.

## Architecture impact

Expected impact: None.

The task implements the accepted trace-local translation boundary and adds no dependency or
ownership change. Focused architecture documentation changes implementation-status wording only.

## Implementation prompt

Use this prompt in one separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, focused tracing/module/dependency architecture documents, the
typed-trace ADR, documentation/planning rules and profiles, roadmap, trace master plan, completed
task 0001, task 0002, current trace event source/tests/Javadocs, model NodeId/ValueId/TensorId as
read-only source concepts, Public API, glossary, and Java 26 root configuration.

Implement docs/planning/modules/trace/tasks/0002-model-correlation-identifiers.md exactly inside
its eleven authorized paths. Add only package documentation plus TraceNodeId, TraceValueId,
TraceTensorId and one focused test. Preserve exact non-negative/value semantics, producer-defined
trace-local scope, nominal separation, and the no-import/no-mapping/no-allocation/no-serialization
boundaries. Do not change task-0001 Java or tests, add later identity domains, or implement task
0003. Stop on architecture, dependency, package, affected-file, or maximum-scope conflict.

Run focused tests while developing and exactly one final :modules:trace:test after executable Java
stabilizes. Then hand the actual diff and Java evidence to a separate clean-context documentation
agent in the same overall change. That pass finalizes Javadocs, tracing explanation, Public API,
glossary, task/master/roadmap status, runs trace Javadoc and documentation/scope checks, and reuses
successful Java evidence unless executable behavior changes. Mark 0002 Complete only after both
passes succeed. Leave 0003+ Draft without detailed specifications.
```

## Local decisions

- Limit 0002 to node, logical-value, and public-Tensor correlations because their source concepts
  are stable after the model milestone. Avoid guessing later partition/backend/unit domains.
- Use three nominal record types rather than one generic ID or kind tag so consumers cannot
  accidentally mix identity domains.
- Use non-negative `long` values consistently with the current event/model identity foundations,
  but do not require numeric equality with any producer-owned ID.
- Let producers own mapping and uniqueness within their documented trace stream; trace adds no
  allocator, context, or registry.
- Add package documentation because this is the first trace subpackage and its translation
  boundary is reusable across all three public types.
- Keep the records independent of `TracePayload`; payload families will refer to them later.

## Known limitations

- The records alone do not reveal which producer stream or graph assigned a value. Consumers must
  interpret them within the enclosing trace stream contract.
- No mapping DTO/API exists, so producers must maintain translation consistently when they begin
  emitting payloads.
- Partition, backend, prepared-unit, run, and device correlations remain unavailable until their
  owning contracts stabilize.
- No serialization/version compatibility is promised yet.

## Validation evidence

- The implementation context `/root/implement_trace_0002` ran
  `./gradlew :modules:trace:test --tests
  io.github.pho001.synaptik.trace.id.TraceModelCorrelationIdTest`; all 4 focused tests passed.
  After executable Java stabilized, that context ran exactly one final
  `./gradlew :modules:trace:test`; XML reports contained 16 tests across 3 suites with zero
  failures, errors, or skips.
- The separate clean documentation context
  `/root/implement_trace_0002/trace_0002_docs` applied General style with API/Javadoc as the
  primary profile, Architecture for `tracing.md`, and Planning for the task, master plan, and
  roadmap. It inspected the authoritative architecture contract, focused trace/boundary
  explanations, typed-trace ADR, documentation/planning rules, task 0001, final trace source/tests,
  model `NodeId`/`ValueId`/`TensorId`, Public API status, glossary, and Java 26 root/module build
  configuration.
- That documentation context changed no executable Java behavior and therefore reused the
  successful Java-test evidence. It ran `./gradlew :modules:trace:javadoc`; Gradle reported
  `BUILD SUCCESSFUL` in 991 ms with one executed and one up-to-date task. Generated output contains
  the `trace.id` package page and `TraceNodeId`, `TraceValueId`, and `TraceTensorId` pages with
  rendered record-component, constructor, accessor, ownership, failure, and limitation contracts.
- `python3 /tmp/validate_synaptik_markdown.py` passed 215 Markdown files, 3,686 local links, 214
  local anchors, 2,702 fence markers, final newlines, and trailing whitespace. `git diff --check`
  passed.
- The final scope command
  `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u` and
  `git status --short` confirmed exactly eleven task paths: four production, one test, and six
  documentation/planning paths. No unrelated working-tree path was present.
- Source and focused-test inspection confirms exactly one `long value` component, the sole public
  constructor, explicitly documented `value()`, exact negative-value message, complete
  non-negative domain, nominal separation, ordinary record behavior, no interfaces or nested
  types, and no Java serialization. All four production files contain no import declaration and
  no model or producer type.
- Status inspection confirms task 0002 and its master-plan row are Complete, task 0003 and every
  later trace row remain Draft, no row is Ready, and the tasks directory contains no detailed
  task-0003 specification.
- Task-0001 source, tests, and Javadocs remain accurate and unchanged because task 0002 adds an
  independent subpackage without changing the event envelope. Model `NodeId`, `ValueId`, and
  `TensorId` remain unchanged read-only source concepts because trace values translate rather than
  import or wrap them.
- No changes were needed in `ARCHITECTURE.md`, ADR 0003, the architecture index, module-boundary or
  dependency-rule explanations, or architecture tests: this task implements the existing
  trace-local DTO boundary without changing ownership or dependencies. Gradle, dependencies, and
  the Java 26 baseline remain unchanged because the records are JDK-only. Other modules, backend
  conformance, and integration tests remain unchanged because no producer, backend, or end-to-end
  behavior was added. Serialization, event emission, attributes, concrete payloads, and later
  correlation domains remain planned and unimplemented.

## Implementation notes

- Added the first trace subpackage with package-level ownership and translation documentation,
  then implemented three independent one-`long` record domains with the shared exact validation
  contract.
- The records intentionally repeat their tiny validation bodies so no common identifier API,
  interface, converter, allocator, or mapping abstraction enters the public surface.
- The documentation pass retained the already-complete production/package Javadocs after
  independently confirming their parameter, result, failure, value-semantics, ownership, and
  limitation contracts. It changed no executable Java behavior.

## Completion summary

- Completed changes: added trace-local node, logical-value, and public-Tensor correlation records,
  package documentation, and one focused exact-shape/behavior test.
- Files changed or created: exactly the four production, one test, and six
  documentation/planning paths listed under Affected files.
- Tests and validation: the implementation context passed the focused 4-test command and one
  final 16-test/three-suite trace module run; the documentation context passed final trace
  Javadoc, repository Markdown, generated-page, exact-scope, status, and whitespace checks without
  rerunning Java tests.
- Documentation-agent review: clean context
  `/root/implement_trace_0002/trace_0002_docs` independently reviewed and finalized all affected
  Javadocs, tracing explanation, Public API status, glossary, task, trace master plan, and roadmap.
- Documentation impact: current model-correlation domains and producer-owned translation are
  documented; later IDs, attributes, payloads, serialization, and emission remain explicitly
  planned.
- Javadoc review: package and three public records document trace-local scope, distinctions,
  producer allocation/uniqueness/lifetime/mapping ownership, numeric remapping, ordinary record
  semantics, inputs, results, failures, and unsupported behavior.
- Glossary impact: added current trace-local correlation terminology and distinct node, value, and
  Tensor correlation definitions.
- Unresolved issues: None.
- Follow-up required: None. Task 0003 remains Draft without a detailed specification.

Status: Complete
