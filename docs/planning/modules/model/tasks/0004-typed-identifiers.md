# Task 0004: Typed Identifiers

**Editorial reading key (later context, not a status update).**

A TensorId names public Tensor state; NodeId names one calculation occurrence inside a graph; ValueId names one described input or result there. Equal numbers do not make these roles interchangeable, and a graph value is not the array holding its eventual values. Constructors validate identifier values; later factory/compiler callers allocate identities in their own contexts. Semantic ownership of these small types does not mean Model owns the graph-building process or the data's lifetime.

Focused current reading: [Current identifier scopes](../../../../reference/tensor/identifiers.md#typed-identifiers). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Define small [immutable](../../../../glossary.md#term-immutable) [`TensorId`](../../../../glossary.md#tensorid), [`NodeId`](../../../../glossary.md#nodeid), and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) value types that prevent accidental interchange of [tensor](../../../../glossary.md#tensor), [graph](../../../../glossary.md#graph)-[node](../../../../glossary.md#node), and graph-value [identities](../../../../glossary.md#term-identity-meanings) without introducing ID generation, graph construction, tensor behavior, [serialization](../../../../glossary.md#term-serialization), [runtime](../../../../glossary.md#term-runtime) state, or [backend](../../../../glossary.md#backend) coupling.

## [Scope](../../../../glossary.md#term-scope)

- Define [`TensorId`](../../../../glossary.md#tensorid) in the [tensor](../../../../glossary.md#tensor) package.
- Define [`NodeId`](../../../../glossary.md#nodeid) and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) in the [graph](../../../../glossary.md#graph) package.
- Represent every identifier as an [immutable](../../../../glossary.md#term-immutable) non-negative `long` value.
- Provide [structural equality](../../../../glossary.md#term-structural-equality), hashing, a typed value accessor, and diagnostic text through Java record semantics.
- Reject negative values without reserving numeric sentinels.
- Document identifier scope, [ownership](../../../../glossary.md#term-ownership-meanings), immutability, and the separation between [identity](../../../../glossary.md#term-identity-meanings) values and later allocation policy.
- Add focused unit tests and public [API](../../../../glossary.md#term-application-programming-interface-api) documentation.

## Out of [scope](../../../../glossary.md#term-scope)

- identifier generators, counters, registries, pools, factories, or service locators
- enforcing uniqueness across [tensors](../../../../glossary.md#tensor), [graphs](../../../../glossary.md#graph), processes, serialized [artifacts](../../../../glossary.md#term-artifact), or executions
- `OperationId`, `GraphId`, [trace](../../../../glossary.md#trace)-local IDs, [backend](../../../../glossary.md#backend) IDs, [partition](../../../../glossary.md#partition) IDs, or [runtime](../../../../glossary.md#term-runtime) IDs
- parsing, formatting contracts, UUIDs, [serialization](../../../../glossary.md#term-serialization), [persistence](../../../../glossary.md#term-persistence), or distributed [identity](../../../../glossary.md#term-identity-meanings)
- `Tensor`, `TensorDescriptor`, graph values, graph [nodes](../../../../glossary.md#node), compiled graphs, operations, [publication](../../../../glossary.md#term-publication) [bindings](../../../../glossary.md#term-binding), or [provenance](../../../../glossary.md#provenance)
- [compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), runtime, [prepare](../../../../glossary.md#prepare), [engine](../../../../glossary.md#term-engine), backend, or tracing behavior
- changing existing [data type](../../../../glossary.md#data-type--datatype), [shape](../../../../glossary.md#shape), or [layout](../../../../glossary.md#layout) contracts
- [Gradle](../../../../glossary.md#term-gradle), Java baseline, [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), or architecture changes

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially the allowed [`TensorId`](../../../../glossary.md#tensorid), [`NodeId`](../../../../glossary.md#nodeid), and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) [model](../../../../glossary.md#term-model) contracts
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md), especially the [tensor](../../../../glossary.md#tensor) and [graph](../../../../glossary.md#graph) package [ownership](../../../../glossary.md#term-ownership-meanings)

## Legacy evidence

The read-only legacy implementation used raw non-negative `int` [node](../../../../glossary.md#node) IDs in `graph.model.CompiledNode` and propagated them through lists and [backend](../../../../glossary.md#backend)/[runtime](../../../../glossary.md#term-runtime) [descriptors](../../../../glossary.md#term-descriptor). Public [`Tensor`](../../../../glossary.md#tensor) objects did not expose a distinct typed tensor [identity](../../../../glossary.md#term-identity-meanings), and [graph](../../../../glossary.md#graph) values did not have a separate typed value identity.

This demonstrates the need for stable graph correlation but not the legacy representation. Raw integers allowed unrelated identifier roles to be interchanged and backend/runtime code to inherit [compile](../../../../glossary.md#compile)-node identity directly. The new [model](../../../../glossary.md#term-model) introduces type separation and `long` capacity without copying legacy graph, backend, or runtime coupling. No legacy evidence establishes an [operation](../../../../glossary.md#operation) identity distinct from a graph node.

## Architecture constraints

- Production packages remain below `io.github.pho001.synaptik.*`.
- Identifiers are [backend](../../../../glossary.md#backend)-independent [model](../../../../glossary.md#term-model) values with no live service or [mutable](../../../../glossary.md#term-mutable) global state.
- [`TensorId`](../../../../glossary.md#tensorid) represents public [tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) facts but does not generate or own tensors.
- [`NodeId`](../../../../glossary.md#nodeid) identifies a [node](../../../../glossary.md#node) [occurrence](../../../../glossary.md#term-occurrence) within an owning [graph](../../../../glossary.md#graph) context; it does not identify [operation](../../../../glossary.md#operation) semantics.
- [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) identifies a [graph value](../../../../glossary.md#graph-value) within an owning graph context and remains distinct from `NodeId`, including for future multi-output nodes.
- Equal numeric values in different identifier types are never equal.
- Graph-local identifiers may reuse the same numeric value in different graphs; callers must interpret them through the owning graph container.
- [Trace](../../../../glossary.md#trace) DTOs use trace-local identifiers and must not import these model IDs.
- If implementation requires a generator, graph [scope](../../../../glossary.md#term-scope) object, architecture change, or [dependency](../../../../glossary.md#term-dependency) addition, stop and report the issue.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.graph`

No package exists physically until its first type is added by this task; both packages are already assigned by the [model](../../../../glossary.md#term-model) master plan.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorId` — [identity](../../../../glossary.md#term-identity-meanings) value for public [tensor](../../../../glossary.md#tensor) state.
- `io.github.pho001.synaptik.model.graph.NodeId` — [graph](../../../../glossary.md#graph)-local identity value for a [node](../../../../glossary.md#node) [occurrence](../../../../glossary.md#term-occurrence).
- `io.github.pho001.synaptik.model.graph.ValueId` — graph-local identity value for an input, intermediate, or output [graph value](../../../../glossary.md#graph-value).

Test placement:

- `TensorIdTest` lives in `model.tensor`.
- `GraphIdentifierTest` lives in `model.graph` and verifies both graph identifier types and their separation.

## Required contracts

Create three public records:

```text
TensorId(long value)
NodeId(long value)
ValueId(long value)
```

For each identifier:

- zero is valid;
- positive values through `Long.MAX_VALUE` are valid;
- negative values fail with `IllegalArgumentException`;
- the [canonical](../../../../glossary.md#term-canonical-form) constructor stores the exact validated value;
- `value()` returns that value;
- record equality and hashing remain structural within the concrete record type; and
- diagnostic text identifies the concrete record type and numeric value but is not a [serialization](../../../../glossary.md#term-serialization) format.

Use an explicit documented canonical constructor and an explicitly documented `value()` accessor so the public [Javadoc](../../../../glossary.md#term-javadoc) covers [parameters](../../../../glossary.md#term-parameter), result semantics, valid range, [scope](../../../../glossary.md#term-scope), and failures according to `AGENTS.md`. Do not add a shared identifier interface merely to deduplicate three small contracts.

No type may expose `next()`, `random()`, `parse()`, an invalid sentinel, [mutable](../../../../glossary.md#term-mutable) state, or cross-type conversion. Identifier allocation belongs to later [tensor](../../../../glossary.md#tensor), [graph](../../../../glossary.md#graph)-construction, and [compiler](../../../../glossary.md#term-compiler) tasks.

## Affected files

Expected production files:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorId.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/NodeId.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/ValueId.java`

Expected test files:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorIdTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/graph/GraphIdentifierTest.java`

Expected documentation/[planning](../../../../glossary.md#term-planning) updates:

- `docs/api/tensor-api.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

The capability baseline already records the identifier decision. If implementation evidence requires changing it, stop and explain the discrepancy before editing that baseline.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- three production Java files;
- two test Java files; and
- the four documentation/[planning](../../../../glossary.md#term-planning) files listed above.

Do not add abstraction or generation files. If another source file is required, stop and propose a follow-up task.

## Acceptance criteria

- Exactly [`TensorId`](../../../../glossary.md#tensorid), [`NodeId`](../../../../glossary.md#nodeid), and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) are introduced.
- Every identifier is an [immutable](../../../../glossary.md#term-immutable) public record over one non-negative `long` component named `value`.
- Zero and `Long.MAX_VALUE` are accepted; negative values are rejected.
- The three types cannot be assigned or compared as the same Java type.
- Equality and hashing are structural within each identifier type.
- No generator, global counter, sentinel, [registry](../../../../glossary.md#term-registry), parser, [serialization](../../../../glossary.md#term-serialization) contract, [graph](../../../../glossary.md#graph) object, [tensor](../../../../glossary.md#tensor) behavior, or [operation](../../../../glossary.md#operation) [identity](../../../../glossary.md#term-identity-meanings) is introduced.
- `OperationId` remains deferred because no separate identity contract is established.
- Public [Javadoc](../../../../glossary.md#term-javadoc) explains semantic [scope](../../../../glossary.md#term-scope), the numeric value, non-negativity, immutability, allocation-policy separation, results, and failures.
- Tests cover boundaries, accessors, equality, hashing, type separation, record status, and diagnostic text without treating text as serialization.
- The Tensor [API](../../../../glossary.md#term-application-programming-interface-api) reference documents identifier packages and scopes.
- Production code has no imports and no [dependency](../../../../glossary.md#term-dependency) outside the [JDK](../../../../glossary.md#term-java-development-kit-jdk) language itself.
- No existing Java contract, [Gradle](../../../../glossary.md#term-gradle) file, architecture document, or other [module](../../../../glossary.md#term-module) is changed.
- Task, master-plan row, and roadmap row have matching final statuses.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run):

```bash
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
```

Also run:

```bash
git diff --check
```

Manually verify:

- exactly three production and two test files were added;
- record components use primitive `long` and reject negative values;
- production files contain no imports;
- no generation, sentinel, conversion, parsing, [graph](../../../../glossary.md#graph), [tensor](../../../../glossary.md#tensor), [runtime](../../../../glossary.md#term-runtime), [backend](../../../../glossary.md#backend), or [trace](../../../../glossary.md#trace) behavior appears;
- packages match the [model](../../../../glossary.md#term-model) package map;
- generated [Javadoc](../../../../glossary.md#term-javadoc) contains all three public records and their explicit constructor/accessor documentation; and
- task, master plan, roadmap, and [API](../../../../glossary.md#term-application-programming-interface-api) documentation agree on status and identifier [scope](../../../../glossary.md#term-scope).

## Dependencies

- Package migrations 0003A–0003C — complete.
- The [model](../../../../glossary.md#term-model) package map — defined.

## Follow-up tasks

- Task 0008: [graph](../../../../glossary.md#graph) values and [nodes](../../../../glossary.md#node) consume [`NodeId`](../../../../glossary.md#nodeid) and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) and define graph-local uniqueness.
- Task 0011: public [`Tensor`](../../../../glossary.md#tensor) consumes [`TensorId`](../../../../glossary.md#tensorid) and defines assignment/[lifecycle](../../../../glossary.md#lifecycle) behavior.
- Task 0012: `TensorFactory` may own the default tensor-ID allocation policy.
- `modules/trace` later translates [model](../../../../glossary.md#term-model) identifiers into [trace](../../../../glossary.md#trace)-local identifiers without importing model into the trace [leaf](../../../../glossary.md#term-leaf).

Do not create detailed specifications for these follow-ups until task 0004 is complete and the frontier advances.

## Architecture impact

Expected impact: None.

The architecture already assigns [`TensorId`](../../../../glossary.md#tensorid), [`NodeId`](../../../../glossary.md#nodeid), and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) to `modules/model`. Package placement and non-negative `long` representation are local implementation decisions. `OperationId` remains deferred rather than silently extending the [architecture contract](../../../../glossary.md#architecture-contract).

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are working in the Synaptik repository.

Read first:
- AGENTS.md
- ARCHITECTURE.md
- docs/planning/planning-guide.md
- docs/planning/modules/model/capabilities.md
- docs/planning/modules/model/master-plan.md
- docs/planning/modules/model/tasks/0004-typed-identifiers.md

Implement task 0004 exactly as specified.

Create only TensorId, NodeId, ValueId and their two focused test files in the planned tensor and graph packages. Use immutable public records over validated non-negative long values. Do not implement ID generation, OperationId, graph or tensor behavior, serialization, parsing, sentinels, cross-type conversion, compiler, runtime, backend, trace behavior, dependencies, or changes to existing Java contracts, Gradle, ARCHITECTURE.md, or focused architecture documentation.

Add complete Javadoc for every public record, canonical constructor, and accessor. Run every validation command from the task specification.

At the end, update this task file with local decisions, known limitations, validation evidence, implementation notes, completion summary, and final status. Synchronize the task status in the model master plan and implementation roadmap.
```

## Local decisions

- Each identifier is a separate public record rather than implementing a shared interface; Java type separation is the primary safety property and no cross-domain polymorphism is required.
- Explicit [canonical](../../../../glossary.md#term-canonical-form) constructors and accessors preserve normal record equality, hashing, and diagnostic behavior while providing complete public [Javadoc](../../../../glossary.md#term-javadoc).
- Zero is a normal first identifier value. Negative numbers are rejected instead of reserving invalid sentinels.
- [Graph](../../../../glossary.md#graph) [scope](../../../../glossary.md#term-scope) is documented rather than embedded as a second ID component; graph containers will own uniqueness and contextual interpretation.

## Known limitations

- The records do not allocate or guarantee uniqueness. [Tensor](../../../../glossary.md#tensor), [graph](../../../../glossary.md#graph)-construction, and [compiler](../../../../glossary.md#term-compiler) tasks must define allocation within their lifecycles.
- Two graph-local IDs of the same type and numeric value compare equal even when obtained from different graph containers; callers must not compare them outside the owning graph context.
- No parsing, [persistence](../../../../glossary.md#term-persistence), [serialization](../../../../glossary.md#term-serialization), distributed [identity](../../../../glossary.md#term-identity-meanings), or compatibility format is defined.
- `OperationId` remains intentionally absent until a separate identity [lifecycle](../../../../glossary.md#lifecycle) is demonstrated.

## [Validation](../../../../glossary.md#term-validation) evidence

- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew :modules:model:test` — passed with 64 tests, zero failures, zero errors, and zero skipped tests; eight tests are new for task 0004.
- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew :modules:model:javadoc` — passed and generated public documentation for [`TensorId`](../../../../glossary.md#tensorid), [`NodeId`](../../../../glossary.md#nodeid), and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) in their planned packages.
- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew test` — passed for the complete multi-[module](../../../../glossary.md#term-module) repository.
- `git diff --check` — passed after implementation and [planning](../../../../glossary.md#term-planning) [synchronization](../../../../glossary.md#term-synchronization).
- Manual [scope](../../../../glossary.md#term-scope) review confirmed exactly three production and two test files were added.
- Manual source review confirmed production files contain no imports, generation, sentinels, parsing, conversion, [runtime](../../../../glossary.md#term-runtime), [backend](../../../../glossary.md#backend), [trace](../../../../glossary.md#trace), or [graph](../../../../glossary.md#graph)/[tensor](../../../../glossary.md#tensor) behavior.
- Manual package review confirmed `TensorId` belongs to `model.tensor` and `NodeId`/`ValueId` belong to `model.graph`.
- No existing Java contract, [Gradle](../../../../glossary.md#term-gradle) file, architecture document, focused architecture document, or other module was changed.
- Gradle emitted a non-fatal filesystem-watching warning in the sandbox; every requested task completed successfully.

## Implementation notes

- Added three public records with validated non-negative `long` values, explicit documented [canonical](../../../../glossary.md#term-canonical-form) constructors, and explicit documented accessors.
- Added boundary, [validation](../../../../glossary.md#term-validation), record, equality, hashing, type-separation, and diagnostic-text tests.
- Extended the [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) reference with identifier [ownership](../../../../glossary.md#term-ownership-meanings), [graph](../../../../glossary.md#graph) [scope](../../../../glossary.md#term-scope), allocation-policy separation, [publication](../../../../glossary.md#term-publication) [binding](../../../../glossary.md#term-binding), and the reason `OperationId` remains deferred.
- Synchronized task, master-plan, and roadmap status and advanced the frontier to task 0005.

## Completion summary

- Completed changes: Implemented typed [tensor](../../../../glossary.md#tensor), [graph](../../../../glossary.md#graph)-[node](../../../../glossary.md#node), and graph-value [identity](../../../../glossary.md#term-identity-meanings) values without allocation policy or behavior leakage.
- Files changed or created: Three production records, two unit-test classes, the Tensor [API](../../../../glossary.md#term-application-programming-interface-api) reference, this task, the [model](../../../../glossary.md#term-model) capability baseline, the model master plan, and the roadmap.
- Tests and [validation](../../../../glossary.md#term-validation): Model tests, model [Javadoc](../../../../glossary.md#term-javadoc), full repository tests, diff checks, [dependency](../../../../glossary.md#term-dependency) checks, package checks, [scope](../../../../glossary.md#term-scope) review, and forbidden-behavior review passed.
- Documentation impact: Documented identifier scopes and future [publication](../../../../glossary.md#term-publication) [binding](../../../../glossary.md#term-binding); no architecture documentation update was required.
- Javadoc review: Every public record, [canonical](../../../../glossary.md#term-canonical-form) constructor, component, and explicit accessor documents semantics, valid range, scope, results, and failures.
- Unresolved issues: None.
- Follow-up required: Plan task 0005 before introducing [operation](../../../../glossary.md#operation) semantic contracts.

Status: Complete
