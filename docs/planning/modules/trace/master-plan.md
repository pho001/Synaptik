# [Trace](../../../glossary.md#trace) Master Plan

## Editorial reading key: recording a fact does not own the work

Trace supplies data-transfer objects: records that carry diagnostic facts between the component
observing work and the component displaying or collecting those facts. The observing component
(for example, a backend integration) constructs the event and supplies its identifiers and clock
reading. Trace does not execute the work, own its memory or close its resources.

“Dependency leaf” here means Trace imports no other Synaptik module; it is unrelated to a Tensor
leaf, which has no producing expression. The plan's integration owner is the person coordinating
edits, not a resource-lifetime owner. Read [tracing participants and correlation](../../../architecture/tracing.md#mental-model)
for that separation and [diagnostic observation points](../../../developer-guide/debugging-trace.md#where-to-observe)
for the public opt-in path. Historical task statuses and receipts below are unchanged.


## Goal

Define typed, serializable diagnostic DTOs shared by [compile](../../../glossary.md#compile), [prepare](../../../glossary.md#prepare), [run](../../../glossary.md#term-run), and [backend](../../../glossary.md#backend) [trace](../../../glossary.md#trace) [producers](../../../glossary.md#term-producer).

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../glossary.md#architecture-contract))</span>
- [Module boundaries](../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../glossary.md#term-module))</span>
- [Dependency rules](../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../glossary.md#term-dependency))</span>

## [Scope](../../../glossary.md#term-scope)

- [trace](../../../glossary.md#trace) envelopes and phases
- typed [payload](../../../glossary.md#artifact-envelope-and-atomic-publication) families
- trace-local identifiers
- typed [backend](../../../glossary.md#backend) attribute escape hatch

## Out of [scope](../../../glossary.md#term-scope)

- [graph](../../../glossary.md#graph) traversal
- business logic
- [runtime](../../../glossary.md#term-runtime) state
- [backend](../../../glossary.md#backend) execution

## [Module](../../../glossary.md#term-module) invariants

- [Trace](../../../glossary.md#trace) remains a [DTO](../../../glossary.md#data-transfer-object--dto)-only [dependency leaf](../../../glossary.md#term-dependency-leaf).
- `Map<String,String>` is never the primary trace [model](../../../glossary.md#term-model).
- [Producers](../../../glossary.md#term-producer) translate local state into trace-local types.

## Allowed dependencies

- [JDK](../../../glossary.md#term-java-development-kit-jdk) standard library only.

## Forbidden dependencies

- [model](../../../glossary.md#term-model), [planning](../../../glossary.md#term-planning), [compiler](../../../glossary.md#term-compiler), [runtime](../../../glossary.md#term-runtime), [prepare](../../../glossary.md#prepare), [engine](../../../glossary.md#term-engine), and [concrete backend](../../../glossary.md#concrete-backend) [modules](../../../glossary.md#term-module)

## Package structure

```text
io.github.pho001.synaptik.trace/
  <root>       shared public event envelope, event identity, lifecycle phase, level, and payload marker
  id/          current model and backend-execution trace-local identity domains
  payload/     current backend preparation/invocation outcomes and later typed DTO families
  attribute/   later typed backend-specific attribute escape hatch
```

The root package is deliberately small and contains only contracts needed by every [producer](../../../glossary.md#term-producer) and
consumer. Concrete [payload](../../../glossary.md#artifact-envelope-and-atomic-publication) families do not accumulate in the root package. No package may expose
or import producer-domain types.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Core trace event envelope](tasks/0001-core-trace-event-envelope.md)<span data-glossary-navigation="true"> ([trace event envelope glossary](../../../glossary.md#trace-event-envelope))</span> | Complete | [Model](../../../glossary.md#term-model) milestone complete | Replaced the placeholder with the caller-supplied event [identity](../../../glossary.md#term-identity-meanings), [lifecycle](../../../glossary.md#lifecycle) phase, diagnostic level, open typed-[payload](../../../glossary.md#artifact-envelope-and-atomic-publication) marker, and [immutable](../../../glossary.md#term-immutable) generic event envelope. |
| 0002 | [Model correlation identifiers](tasks/0002-model-correlation-identifiers.md) | Complete | 0001, completed model milestone | Added [trace](../../../glossary.md#trace)-local [node](../../../glossary.md#node), value, and [tensor](../../../glossary.md#tensor) identities for stable model correlations without importing or duplicating [producer](../../../glossary.md#term-producer) objects. |
| 0003 | [Backend preparation and invocation diagnostic DTOs](tasks/0003-backend-preparation-and-invocation-diagnostic-dtos.md) | Complete | 0001–0002; stable [Metal](../../../glossary.md#term-metal) [prepare](../../../glossary.md#prepare)/[run](../../../glossary.md#term-run) producer facts | Added producer IDs and neutral outcome DTOs. Model 0032 later removed numerical-profile fields; route/[cache](../../../glossary.md#term-cache)/native-status facts remain. |
| 0004 | [Compile](../../../glossary.md#compile) payload family | Draft | 0001–0002 | Define typed [capture](../../../glossary.md#term-capture-meanings), transformation, [ownership](../../../glossary.md#term-ownership-meanings), [partition](../../../glossary.md#partition), logical-memory, and [publication](../../../glossary.md#term-publication) diagnostic payloads after [compiler](../../../glossary.md#term-compiler)/[planning](../../../glossary.md#term-planning) facts stabilize. |
| 0005 | Broader prepare payload family | Draft | 0003 | Extend beyond the bounded prepared-unit finalization outcome only after additional prepare producer contracts stabilize. |
| 0006 | Broader run payload family | Draft | 0003 | Extend beyond the bounded native-invocation outcome only after additional [Runtime](../../../glossary.md#term-runtime) producer contracts stabilize. |
| 0007 | Broader [backend](../../../glossary.md#backend) payloads and typed attributes | Draft | 0003 | Define later availability, capability, [kernel](../../../glossary.md#kernel), storage, and constrained backend-detail DTOs without a generic string map or backend [dependency](../../../glossary.md#term-dependency). |
| 0008 | [Serialization](../../../glossary.md#term-serialization) and [schema](../../../glossary.md#term-schema) [validation](../../../glossary.md#term-validation) | Draft | 0001–0007 | Select and validate a stable external encoding only after all shared [DTO](../../../glossary.md#data-transfer-object--dto) families are concrete. |

## [Dependency](../../../glossary.md#term-dependency) [DAG](../../../glossary.md#term-directed-acyclic-graph-dag) and [authorized frontier](../../../glossary.md#term-authorized-frontier)

`0001 (Complete) -> 0002 (Complete) -> 0003 (Complete) -> Metal 0041 (Complete)`

Tasks 0004–0008 remain Draft. No [Trace](../../../glossary.md#trace) task is [Ready](../../../glossary.md#term-ready). Each new [payload](../../../glossary.md#artifact-envelope-and-atomic-publication) family requires a concrete
[producer](../../../glossary.md#term-producer); [serialization](../../../glossary.md#term-serialization) does not become a prerequisite for current in-memory tracing. [Model](../../../glossary.md#term-model) 0032's
profile-free cutover is Complete. The roadmap owns cross-area frontier status.

## Integration [ownership](../../../glossary.md#term-ownership-meanings) and shared documents

- No active [Trace](../../../glossary.md#trace) implementation group. The next [Ready](../../../glossary.md#term-ready) brief must record its own base and [owner](../../../glossary.md#term-owner).
- Project-review planning reconciliation is owned by the coordinator; completed Trace 0003's
  launch metadata remains historical evidence in its task brief.<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning); [metadata glossary](../../../glossary.md#term-metadata))</span>


## Milestones

- Core envelope and identifiers
- [Lifecycle](../../../glossary.md#lifecycle) [payload](../../../glossary.md#artifact-envelope-and-atomic-publication) families
- [Serialization](../../../glossary.md#term-serialization) and [schema](../../../glossary.md#term-schema) [validation](../../../glossary.md#term-validation)

## Current status

Tasks 0001–0003 are Complete. Current code contains the event envelope, [trace](../../../glossary.md#trace)-local IDs,
preparation/invocation outcome DTOs, and low-[precision](../../../glossary.md#term-precision) route/[dtype](../../../glossary.md#data-type--datatype) facts. Numerical-profile and
[certificate](../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) [metadata](../../../glossary.md#term-metadata) are absent after [Model](../../../glossary.md#term-model) 0032. Trace remains [JDK](../../../glossary.md#term-java-development-kit-jdk)-only passive data: no emitter,
clock, allocator, [mutable](../../../glossary.md#term-mutable) [producer](../../../glossary.md#term-producer) state, numerical validator, or [serialization](../../../glossary.md#term-serialization) [codec](../../../glossary.md#term-codec).
Broader [payload](../../../glossary.md#artifact-envelope-and-atomic-publication) families and external encoding remain Draft under 0004–0008.

## Open questions

- The external serialization format and compatibility/versioning policy remain intentionally open
  until task 0008; task 0001 adds no Java-native serialization promise.<span data-glossary-navigation="true"> ([serialization glossary](../../../glossary.md#term-serialization))</span>

## Decisions made

- The implementation must follow the current [architecture contract](../../../glossary.md#architecture-contract).
- Legacy code is capability evidence only; new implementation is written from scratch.
- [Lifecycle](../../../glossary.md#lifecycle) phase classifies [`COMPILE`](../../../glossary.md#compile), [`PREPARE`](../../../glossary.md#prepare), or [`RUN`](../../../glossary.md#term-run). Backend is a payload family and
  producer role, not a fourth lifecycle phase.<span data-glossary-navigation="true"> ([Backend glossary](../../../glossary.md#backend); [payload glossary](../../../glossary.md#artifact-envelope-and-atomic-publication); [producer glossary](../../../glossary.md#term-producer))</span>
- The initial payload marker is open and method-free so later trace-owned DTO families can be
  added incrementally without a central registry or premature permits list.<span data-glossary-navigation="true"> ([trace glossary](../../../glossary.md#trace); [DTO glossary](../../../glossary.md#data-transfer-object--dto); [registry glossary](../../../glossary.md#term-registry))</span>
- Event identity and monotonic time are supplied by producers. Trace owns neither an allocator nor
  a clock.<span data-glossary-navigation="true"> ([identity glossary](../../../glossary.md#term-identity-meanings))</span>
- The event record is shallowly immutable. Its open payload bound documents an immutable DTO
  obligation but cannot enforce payload implementation immutability at runtime.<span data-glossary-navigation="true"> ([immutable glossary](../../../glossary.md#term-immutable); [runtime glossary](../../../glossary.md#term-runtime))</span>
- Task 0002 introduced only node, value, and tensor correlation IDs. Task 0003 now adds the four
  producer-stream identity domains whose Metal producer semantics are stable: backend, device,
  prepared unit, and invocation.<span data-glossary-navigation="true"> ([node glossary](../../../glossary.md#node); [tensor glossary](../../../glossary.md#tensor); [Metal glossary](../../../glossary.md#term-metal); [device glossary](../../../glossary.md#term-device))</span>
- Trace-local correlation values are assigned within a producer-defined trace stream. They are not
  direct references to, or required numeric copies of, producer-owned identifiers.
- Task 0003 uses only closed trace-owned neutral vocabulary. It does not import Backend Contract,
  Config, Prepare, Runtime, Engine, or Metal types and does not publish MPSGraph, NEG, native symbol,
  or backend-local numeric-code names as shared concepts.<span data-glossary-navigation="true"> ([Engine glossary](../../../glossary.md#term-engine); [MPSGraph glossary](../../../glossary.md#term-mpsgraph); [symbol glossary](../../../glossary.md#term-symbol))</span>
- `TraceCacheStatus.NOT_QUERIED` is the only current cache fact. It is not a cache miss, and Trace
  does not infer hits, misses, or selection policy.<span data-glossary-navigation="true"> ([cache glossary](../../../glossary.md#term-cache); [cache miss glossary](../../../glossary.md#cache-and-memoization))</span>
- `TraceRouteKind.CUSTOM_KERNEL` and `GRAPH_EXECUTABLE` describe the current producer mechanisms;
  they are not a universal registry into which every later backend route must fit.<span data-glossary-navigation="true"> ([backend route glossary](../../../glossary.md#backend-route))</span>
- Benchmark reports, workload tuning caches, model-plan results, planning cost profiles, and
  runtime profiling remain distinct. Only passive observations from stable producers become trace
  payloads; Trace never runs searches or changes production settings.<span data-glossary-navigation="true"> ([model glossary](../../../glossary.md#term-model); [planning glossary](../../../glossary.md#term-planning); [runtime profiling glossary](../../../glossary.md#runtime-profiling))</span>

## Risks

- Allowing producer-domain types, backend-specific names, or unstructured string maps into shared
  trace contracts.<span data-glossary-navigation="true"> ([producer glossary](../../../glossary.md#term-producer); [trace glossary](../../../glossary.md#trace))</span>
- Treating caller-supplied monotonic timestamps as wall-clock instants or globally comparable
  values.
- Selecting a [serialization](../../../glossary.md#term-serialization) mechanism before the shared [payload](../../../glossary.md#artifact-envelope-and-atomic-publication) [schemas](../../../glossary.md#term-schema) exist.
- Mistaking stream-local [backend](../../../glossary.md#backend)/[device](../../../glossary.md#term-device) IDs for a global backend [registry](../../../glossary.md#term-registry), or `NOT_QUERIED` for a
  cache miss.<span data-glossary-navigation="true"> ([cache miss glossary](../../../glossary.md#cache-and-memoization))</span>

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span>.

Task 0001 passed 12 focused tests across two suites, final [trace](../../../glossary.md#trace) [Javadoc](../../../glossary.md#term-javadoc) generation, repository
Markdown [validation](../../../glossary.md#term-validation), and exact fifteen-path [scope](../../../glossary.md#term-scope) validation. The implementation introduced no
[dependency](../../../glossary.md#term-dependency), architecture, build, [backend](../../../glossary.md#backend), or cross-[module](../../../glossary.md#term-module) behavior change.

Task 0002 passed 4 focused tests and one final 16-test/three-suite trace module [run](../../../glossary.md#term-run). Its separate
documentation pass finalized the correlation Javadocs and explanations and passed trace Javadoc,
repository Markdown, exact eleven-path, status, and whitespace validation without rerunning Java
tests.

Historical Trace 0003 and [Metal](../../../glossary.md#term-metal) 0041 evidence remains in their task briefs. Current [lifecycle](../../../glossary.md#lifecycle)
fixes the route during [Prepare](../../../glossary.md#prepare) and invokes that route during Run; no numerical profile travels
with these trace facts.
