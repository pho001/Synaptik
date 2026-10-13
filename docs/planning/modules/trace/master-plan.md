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
| 0003 | [Backend preparation and invocation diagnostic DTOs](tasks/0003-backend-preparation-and-invocation-diagnostic-dtos.md) | Complete | 0001–0002; stable Metal prepare/run producer facts | Added producer IDs and neutral outcome DTOs. Model 0032 later removed numerical-profile fields; route/cache/native-status facts remain. |
| 0004 | Compile payload family | Draft | 0001–0002 | Define typed capture, transformation, ownership, partition, logical-memory, and publication diagnostic payloads after compiler/planning facts stabilize. |
| 0005 | Broader prepare payload family | Draft | 0003 | Extend beyond the bounded prepared-unit finalization outcome only after additional prepare producer contracts stabilize. |
| 0006 | Broader run payload family | Draft | 0003 | Extend beyond the bounded native-invocation outcome only after additional Runtime producer contracts stabilize. |
| 0007 | Broader backend payloads and typed attributes | Draft | 0003 | Define later availability, capability, kernel, storage, and constrained backend-detail DTOs without a generic string map or backend dependency. |
| 0008 | Serialization and schema validation | Draft | 0001–0007 | Select and validate a stable external encoding only after all shared DTO families are concrete. |

## Dependency DAG and authorized frontier

`0001 (Complete) -> 0002 (Complete) -> 0003 (Complete) -> Metal 0041 (Complete)`

Tasks 0004–0008 remain Draft. No Trace task is Ready. Each new payload family requires a concrete
producer; serialization does not become a prerequisite for current in-memory tracing. Model 0032's
profile-free cutover is Complete. The roadmap owns cross-area frontier status.

## Integration ownership and shared documents

- No active Trace implementation group. The next Ready brief must record its own base and owner.
- Project-review planning reconciliation is owned by the coordinator; completed Trace 0003's
  launch metadata remains historical evidence in its task brief.


## Milestones

- Core envelope and identifiers
- Lifecycle payload families
- Serialization and schema validation

## Current status

Tasks 0001–0003 are Complete. Current code contains the event envelope, trace-local IDs,
preparation/invocation outcome DTOs, and low-precision route/dtype facts. Numerical-profile and
certificate metadata are absent after Model 0032. Trace remains JDK-only passive data: no emitter,
clock, allocator, mutable producer state, numerical validator, or serialization codec.
Broader payload families and external encoding remain Draft under 0004–0008.

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

Historical Trace 0003 and Metal 0041 evidence remains in their task briefs. Current lifecycle
fixes the route during Prepare and invokes that route during Run; no numerical profile travels
with these trace facts.
