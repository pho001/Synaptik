# Trace Master Plan

## Goal

Define typed, serializable diagnostic DTOs shared by compile, prepare, run, and backend trace producers.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- trace envelopes and phases
- typed payload families
- trace-local identifiers
- typed backend attribute escape hatch

## Out of scope

- graph traversal
- business logic
- runtime state
- backend execution

## Module invariants

- Trace remains a DTO-only dependency leaf.
- `Map<String,String>` is never the primary trace model.
- Producers translate local state into trace-local types.

## Allowed dependencies

- JDK standard library only.

## Forbidden dependencies

- model, planning, compiler, runtime, prepare, engine, and concrete backend modules

## Package structure

```text
io.github.pho001.synaptik.trace/
  <root>       shared public event envelope, event identity, lifecycle phase, level, and payload marker
  id/          current model-correlation identifiers and later trace-local identity domains
  payload/     later typed compile, prepare, run, and backend diagnostic DTO families
  attribute/   later typed backend-specific attribute escape hatch
```

The root package is deliberately small and contains only contracts needed by every producer and
consumer. Concrete payload families do not accumulate in the root package. No package may expose
or import producer-domain types.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Core trace event envelope](tasks/0001-core-trace-event-envelope.md) | Complete | Model milestone complete | Replaced the placeholder with the caller-supplied event identity, lifecycle phase, diagnostic level, open typed-payload marker, and immutable generic event envelope. |
| 0002 | [Model correlation identifiers](tasks/0002-model-correlation-identifiers.md) | Complete | 0001, completed model milestone | Added trace-local node, value, and tensor identities for stable model correlations without importing or duplicating producer objects. |
| 0003 | Typed trace attributes | Draft | 0001 | Add the constrained typed backend-specific attribute escape hatch without making a string map the primary model. |
| 0004 | Compile payload family | Draft | 0001–0002 | Define typed capture, transformation, ownership, partition, logical-memory, and publication diagnostic payloads after compiler/planning facts stabilize. |
| 0005 | Prepare payload family | Draft | 0001–0003 | Define typed preparation, route, prepared-memory, partition, unit, and schedule diagnostics after prepare contracts stabilize. |
| 0006 | Run payload family | Draft | 0001–0003 | Define typed invocation, execution, transfer, materialization, step, and publication diagnostics after runtime contracts stabilize. |
| 0007 | Backend payload family | Draft | 0001–0003 | Define typed availability, capability, route, kernel, storage, and backend-detail diagnostics without backend implementation dependencies. |
| 0008 | Serialization and schema validation | Draft | 0001–0007 | Select and validate a stable external encoding only after all shared DTO families are concrete. |


## Milestones

- Core envelope and identifiers
- Lifecycle payload families
- Serialization and schema validation

## Current status

In progress but deliberately interleaved after the stable foundation. Tasks 0001 and 0002 are
Complete and supply the common event envelope plus three model-correlation domains. No trace task
is Ready. Backend-contract tasks 0001–0003 now provide identity, class, and supplied availability
facts, and task 0004 completes its hard-requirement vocabulary. The backend-contract project area
and capability checkpoint are Complete. The latest explicit interleave has config tasks 0001 and
0002 Complete and planning task 0001 Complete before config scoring work resumes. Config task
0003 is now Complete with one optional soft coarse device-class preference.
Planning 0001 defines the operation-capability query/provider boundary, and Planning 0002 is now
Complete with internal per-query hard eligibility. It adds no trace DTO,
structured rejection reason, public matrix, score, ownership event, or compiler event. Trace task
0003 and later rows remain Draft without detailed specifications until the broader config,
planning, compiler, prepare, and runtime producer facts are stable.

## Open questions

- The external serialization format and compatibility/versioning policy remain intentionally open
  until task 0008; task 0001 adds no Java-native serialization promise.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Lifecycle phase classifies `COMPILE`, `PREPARE`, or `RUN`. Backend is a payload family and
  producer role, not a fourth lifecycle phase.
- The initial payload marker is open and method-free so later trace-owned DTO families can be
  added incrementally without a central registry or premature permits list.
- Event identity and monotonic time are supplied by producers. Trace owns neither an allocator nor
  a clock.
- The event record is shallowly immutable. Its open payload bound documents an immutable DTO
  obligation but cannot enforce payload implementation immutability at runtime.
- Task 0002 introduces only node, value, and tensor correlation IDs because their source concepts
  are stable. Partition, backend, and prepared-unit IDs remain with the later payload tasks that
  can validate their actual producer domains.
- Trace-local correlation values are assigned within a producer-defined trace stream. They are not
  direct references to, or required numeric copies of, producer-owned identifiers.
- After task 0002, roadmap execution interleaves `modules/backend-contract` rather than designing
  typed backend attributes or payloads without a concrete producer vocabulary. This pauses no
  completed contract and adds no trace dependency on backend-contract.
- Backend-contract task 0001 supplies `BackendId` and `BackendDeviceId`, but trace-local backend
  and device correlations remain deferred to the later payload work that can define their actual
  producer domains. Trace continues to import no backend-contract type.
- Backend-contract classification, supplied availability, and declarative requirement work do not
  make trace consume those producer types. Later producers will translate selected facts into
  trace-owned identities and payloads after their complete schemas are concrete.
- The next interleave opened only planning task 0001 before config scoring configuration. A boolean
  capability answer does not stabilize trace rejection diagnostics or backend payload schemas, so
  trace task 0003 and later rows remain Draft and trace keeps no planning dependency.
- Config task 0003 is Complete after that interleave. Its declarative class preference
  creates no trace producer schema, score, rejection diagnostic, or ownership event.
- Planning task 0002 remains intentionally boolean/filtering-only at the diagnostic boundary: its
  result retains eligible backend identities and no rejection reason. It therefore does not
  stabilize typed capability-rejection or no-match trace payloads.

## Risks

- Allowing producer-domain types or unstructured string maps into shared trace contracts.
- Treating caller-supplied monotonic timestamps as wall-clock instants or globally comparable
  values.
- Selecting a serialization mechanism before the shared payload schemas exist.
- Defining partition, backend, or prepared-unit identity before their owning module contracts are
  concrete.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).

Task 0001 passed 12 focused tests across two suites, final trace Javadoc generation, repository
Markdown validation, and exact fifteen-path scope validation. The implementation introduced no
dependency, architecture, build, backend, or cross-module behavior change.

Task 0002 passed 4 focused tests and one final 16-test/three-suite trace module run. Its separate
documentation pass finalized the correlation Javadocs and explanations and passed trace Javadoc,
repository Markdown, exact eleven-path, status, and whitespace validation without rerunning Java
tests.

After task 0002, reassess the frontier rather than forcing speculative lifecycle payload schemas.
If producer contracts are still absent, the roadmap may explicitly interleave their owning
project areas before returning to the corresponding trace payload rows.

That reassessment selected the backend-contract interleave. Backend-contract task 0001 is now
Complete with identity-only producer vocabulary, tasks 0002–0003 add class and supplied
availability facts, and task 0004 completes the hard-requirement closure vocabulary and selected
backend-contract checkpoint. Trace tasks 0003–0008 remain ordered Draft work and return to
planning only when their directly relevant producer facts are stable; no detailed task-0003
specification exists.

The following reassessment selected config task 0001 rather than prematurely resuming trace. Its
single optional hard-intent contract is now current, but it does not stabilize scoring, compiler,
prepare, runtime, or backend payload schemas. Config task 0001's repository checkpoint passed.
Config task 0002 is now Complete, but its declarative mode and optimization values still create no
compiler or trace producer schema. Config task 0003 is now Complete, and no trace task 0003
specification has been created.

The following reassessment selected planning task 0001 before config 0003. That task is now
Complete after stabilizing the typed capability question and passing its single final repository
suite. It deliberately defers structured rejection diagnostics and device-level capability,
creates no trace producer schema, and does not justify a trace task 0003 specification. The likely
frontier returned to config 0003 rather than trace. Config task 0003 is now Complete after focused
validation and an independent documentation pass; it creates no trace producer schema. Trace task
0003 remains Draft without a detailed specification. A later reassessment selected Planning task
0002 alone for internal hard eligibility; its no-reason result still creates no trace schema.
Config 0004 is the likely next area after that task, subject to another reassessment, while every
trace task remains Draft. Planning task 0002 is now Complete and no trace or global task is Ready.
