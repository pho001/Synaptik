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
  id/          current model and backend-execution trace-local identity domains
  payload/     current backend preparation/invocation outcomes and later typed DTO families
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
| 0003 | [Backend preparation and invocation diagnostic DTOs](tasks/0003-backend-preparation-and-invocation-diagnostic-dtos.md) | Complete | 0001–0002; stable Metal prepare/run producer facts | Added four trace-local producer IDs and the closed neutral profile/route/cache/native-status preparation and invocation outcome DTOs required by Metal 0041, without emission behavior. |
| 0004 | Compile payload family | Draft | 0001–0002 | Define typed capture, transformation, ownership, partition, logical-memory, and publication diagnostic payloads after compiler/planning facts stabilize. |
| 0005 | Broader prepare payload family | Draft | 0003 | Extend beyond the bounded prepared-unit finalization outcome only after additional prepare producer contracts stabilize. |
| 0006 | Broader run payload family | Draft | 0003 | Extend beyond the bounded native-invocation outcome only after additional Runtime producer contracts stabilize. |
| 0007 | Broader backend payloads and typed attributes | Draft | 0003 | Define later availability, capability, kernel, storage, and constrained backend-detail DTOs without a generic string map or backend dependency. |
| 0008 | Serialization and schema validation | Draft | 0001–0007 | Select and validate a stable external encoding only after all shared DTO families are concrete. |

## Dependency DAG and authorized frontier

`0001 (Complete) -> 0002 (Complete) -> 0003 (Complete) -> Metal 0041 (Blocked pending separate promotion)`

Tasks 0004–0008 remain Draft and are not parallel frontiers. No Trace task is Ready. Trace 0003 is
complete; Metal 0041 remains blocked until its separate planning-only rebase and Ready promotion.

## Integration ownership and shared documents

- Integration owner: Trace 0003 executor through its completion commit.
- Shared documents: Trace 0003 owns current Trace API/explanation/glossary status, this master plan,
  and the roadmap through its completion commit.
- Common base revision: `c734f0bb52cd35b94cd794cc580326dca09ec62b`.
- Integration validation: Trace module tests/Javadocs, public-shape/import checks, Markdown/link
  validation, exact scope, and `git diff --check`.


## Milestones

- Core envelope and identifiers
- Lifecycle payload families
- Serialization and schema validation

## Current status

Tasks 0001–0003 are Complete. Task 0003 added only trace-local backend/device/prepared-unit/
invocation IDs, closed neutral profile/route/cache/native-status vocabulary, and two immutable
outcome payloads. It preserved every pre-existing Trace public source file and added no emitter,
observer, consumer, allocator, clock, mutable state, producer dependency, generic map, or
serialization behavior. No Trace task is Ready.

Metal 0041 is the immediate coordinated successor and remains Blocked until its separate
planning-only rebase and Ready promotion. Broader compile, prepare, run, backend-detail,
typed-attribute, and serialization work remains Draft under 0004–0008.

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
- Task 0002 introduced only node, value, and tensor correlation IDs. Task 0003 now adds the four
  producer-stream identity domains whose Metal producer semantics are stable: backend, device,
  prepared unit, and invocation.
- Trace-local correlation values are assigned within a producer-defined trace stream. They are not
  direct references to, or required numeric copies of, producer-owned identifiers.
- Task 0003 uses only closed trace-owned neutral vocabulary. It does not import Backend Contract,
  Config, Prepare, Runtime, Engine, or Metal types and does not publish MPSGraph, NEG, native symbol,
  or backend-local numeric-code names as shared concepts.
- `TraceCacheStatus.NOT_QUERIED` is the only current cache fact. It is not a cache miss, and Trace
  does not infer hits, misses, or selection policy.
- `TraceRouteKind.CUSTOM_KERNEL` and `GRAPH_EXECUTABLE` describe the current producer mechanisms;
  they are not a universal registry into which every later backend route must fit.
- Benchmark reports, workload tuning caches, model-plan results, planning cost profiles, and
  runtime profiling remain distinct. Only passive observations from stable producers become trace
  payloads; Trace never runs searches or changes production settings.

## Risks

- Allowing producer-domain types, backend-specific names, or unstructured string maps into shared
  trace contracts.
- Treating caller-supplied monotonic timestamps as wall-clock instants or globally comparable
  values.
- Selecting a serialization mechanism before the shared payload schemas exist.
- Mistaking stream-local backend/device IDs for a global backend registry, or `NOT_QUERIED` for a
  cache miss.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).

Task 0001 passed 12 focused tests across two suites, final trace Javadoc generation, repository
Markdown validation, and exact fifteen-path scope validation. The implementation introduced no
dependency, architecture, build, backend, or cross-module behavior change.

Task 0002 passed 4 focused tests and one final 16-test/three-suite trace module run. Its separate
documentation pass finalized the correlation Javadocs and explanations and passed trace Javadoc,
repository Markdown, exact eleven-path, status, and whitespace validation without rerunning Java
tests.

Task 0003 completed from clean planning revision
`c734f0bb52cd35b94cd794cc580326dca09ec62b`. The current lifecycle fixes route and profile during
Prepare, constructs the executable during finalization, invokes the prepared route during Run,
and retains a closed native-status taxonomy. The implementation added only the matching neutral
JDK-only DTO vocabulary and documentation; focused/full Trace tests, Javadoc, Markdown, scope, and
diff validation passed. Metal 0041 remains blocked pending its separate planning-only rebase and
Ready promotion.
