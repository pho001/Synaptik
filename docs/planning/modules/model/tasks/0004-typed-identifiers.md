# Task 0004: Typed Identifiers

## Status

Complete

## Goal

Define small immutable `TensorId`, `NodeId`, and `ValueId` value types that prevent accidental interchange of tensor, graph-node, and graph-value identities without introducing ID generation, graph construction, tensor behavior, serialization, runtime state, or backend coupling.

## Scope

- Define `TensorId` in the tensor package.
- Define `NodeId` and `ValueId` in the graph package.
- Represent every identifier as an immutable non-negative `long` value.
- Provide structural equality, hashing, a typed value accessor, and diagnostic text through Java record semantics.
- Reject negative values without reserving numeric sentinels.
- Document identifier scope, ownership, immutability, and the separation between identity values and later allocation policy.
- Add focused unit tests and public API documentation.

## Out of scope

- identifier generators, counters, registries, pools, factories, or service locators
- enforcing uniqueness across tensors, graphs, processes, serialized artifacts, or executions
- `OperationId`, `GraphId`, trace-local IDs, backend IDs, partition IDs, or runtime IDs
- parsing, formatting contracts, UUIDs, serialization, persistence, or distributed identity
- `Tensor`, `TensorDescriptor`, graph values, graph nodes, compiled graphs, operations, publication bindings, or provenance
- compiler, planning, runtime, prepare, engine, backend, or tracing behavior
- changing existing data type, shape, or layout contracts
- Gradle, Java baseline, module dependency, or architecture changes

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially the allowed `TensorId`, `NodeId`, and `ValueId` model contracts
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md), especially the tensor and graph package ownership

## Legacy evidence

The read-only legacy implementation used raw non-negative `int` node IDs in `graph.model.CompiledNode` and propagated them through lists and backend/runtime descriptors. Public `Tensor` objects did not expose a distinct typed tensor identity, and graph values did not have a separate typed value identity.

This demonstrates the need for stable graph correlation but not the legacy representation. Raw integers allowed unrelated identifier roles to be interchanged and backend/runtime code to inherit compile-node identity directly. The new model introduces type separation and `long` capacity without copying legacy graph, backend, or runtime coupling. No legacy evidence establishes an operation identity distinct from a graph node.

## Architecture constraints

- Production packages remain below `io.github.pho001.synaptik.*`.
- Identifiers are backend-independent model values with no live service or mutable global state.
- `TensorId` represents public tensor identity facts but does not generate or own tensors.
- `NodeId` identifies a node occurrence within an owning graph context; it does not identify operation semantics.
- `ValueId` identifies a graph value within an owning graph context and remains distinct from `NodeId`, including for future multi-output nodes.
- Equal numeric values in different identifier types are never equal.
- Graph-local identifiers may reuse the same numeric value in different graphs; callers must interpret them through the owning graph container.
- Trace DTOs use trace-local identifiers and must not import these model IDs.
- If implementation requires a generator, graph scope object, architecture change, or dependency addition, stop and report the issue.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.graph`

No package exists physically until its first type is added by this task; both packages are already assigned by the model master plan.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorId` — identity value for public tensor state.
- `io.github.pho001.synaptik.model.graph.NodeId` — graph-local identity value for a node occurrence.
- `io.github.pho001.synaptik.model.graph.ValueId` — graph-local identity value for an input, intermediate, or output graph value.

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
- the canonical constructor stores the exact validated value;
- `value()` returns that value;
- record equality and hashing remain structural within the concrete record type; and
- diagnostic text identifies the concrete record type and numeric value but is not a serialization format.

Use an explicit documented canonical constructor and an explicitly documented `value()` accessor so the public Javadoc covers parameters, result semantics, valid range, scope, and failures according to `AGENTS.md`. Do not add a shared identifier interface merely to deduplicate three small contracts.

No type may expose `next()`, `random()`, `parse()`, an invalid sentinel, mutable state, or cross-type conversion. Identifier allocation belongs to later tensor, graph-construction, and compiler tasks.

## Affected files

Expected production files:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorId.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/NodeId.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/ValueId.java`

Expected test files:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorIdTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/graph/GraphIdentifierTest.java`

Expected documentation/planning updates:

- `docs/api/tensor-api.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

The capability baseline already records the identifier decision. If implementation evidence requires changing it, stop and explain the discrepancy before editing that baseline.

## Maximum scope

This task may create or modify at most:

- three production Java files;
- two test Java files; and
- the four documentation/planning files listed above.

Do not add abstraction or generation files. If another source file is required, stop and propose a follow-up task.

## Acceptance criteria

- Exactly `TensorId`, `NodeId`, and `ValueId` are introduced.
- Every identifier is an immutable public record over one non-negative `long` component named `value`.
- Zero and `Long.MAX_VALUE` are accepted; negative values are rejected.
- The three types cannot be assigned or compared as the same Java type.
- Equality and hashing are structural within each identifier type.
- No generator, global counter, sentinel, registry, parser, serialization contract, graph object, tensor behavior, or operation identity is introduced.
- `OperationId` remains deferred because no separate identity contract is established.
- Public Javadoc explains semantic scope, the numeric value, non-negativity, immutability, allocation-policy separation, results, and failures.
- Tests cover boundaries, accessors, equality, hashing, type separation, record status, and diagnostic text without treating text as serialization.
- The Tensor API reference documents identifier packages and scopes.
- Production code has no imports and no dependency outside the JDK language itself.
- No existing Java contract, Gradle file, architecture document, or other module is changed.
- Task, master-plan row, and roadmap row have matching final statuses.

## Tests / validation

Run:

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
- no generation, sentinel, conversion, parsing, graph, tensor, runtime, backend, or trace behavior appears;
- packages match the model package map;
- generated Javadoc contains all three public records and their explicit constructor/accessor documentation; and
- task, master plan, roadmap, and API documentation agree on status and identifier scope.

## Dependencies

- Package migrations 0003A–0003C — complete.
- The model package map — defined.

## Follow-up tasks

- Task 0008: graph values and nodes consume `NodeId` and `ValueId` and define graph-local uniqueness.
- Task 0011: public `Tensor` consumes `TensorId` and defines assignment/lifecycle behavior.
- Task 0012: `TensorFactory` may own the default tensor-ID allocation policy.
- `modules/trace` later translates model identifiers into trace-local identifiers without importing model into the trace leaf.

Do not create detailed specifications for these follow-ups until task 0004 is complete and the frontier advances.

## Architecture impact

Expected impact: None.

The architecture already assigns `TensorId`, `NodeId`, and `ValueId` to `modules/model`. Package placement and non-negative `long` representation are local implementation decisions. `OperationId` remains deferred rather than silently extending the architecture contract.

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
- Explicit canonical constructors and accessors preserve normal record equality, hashing, and diagnostic behavior while providing complete public Javadoc.
- Zero is a normal first identifier value. Negative numbers are rejected instead of reserving invalid sentinels.
- Graph scope is documented rather than embedded as a second ID component; graph containers will own uniqueness and contextual interpretation.

## Known limitations

- The records do not allocate or guarantee uniqueness. Tensor, graph-construction, and compiler tasks must define allocation within their lifecycles.
- Two graph-local IDs of the same type and numeric value compare equal even when obtained from different graph containers; callers must not compare them outside the owning graph context.
- No parsing, persistence, serialization, distributed identity, or compatibility format is defined.
- `OperationId` remains intentionally absent until a separate identity lifecycle is demonstrated.

## Validation evidence

- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew :modules:model:test` — passed with 64 tests, zero failures, zero errors, and zero skipped tests; eight tests are new for task 0004.
- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew :modules:model:javadoc` — passed and generated public documentation for `TensorId`, `NodeId`, and `ValueId` in their planned packages.
- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew test` — passed for the complete multi-module repository.
- `git diff --check` — passed after implementation and planning synchronization.
- Manual scope review confirmed exactly three production and two test files were added.
- Manual source review confirmed production files contain no imports, generation, sentinels, parsing, conversion, runtime, backend, trace, or graph/tensor behavior.
- Manual package review confirmed `TensorId` belongs to `model.tensor` and `NodeId`/`ValueId` belong to `model.graph`.
- No existing Java contract, Gradle file, architecture document, focused architecture document, or other module was changed.
- Gradle emitted a non-fatal filesystem-watching warning in the sandbox; every requested task completed successfully.

## Implementation notes

- Added three public records with validated non-negative `long` values, explicit documented canonical constructors, and explicit documented accessors.
- Added boundary, validation, record, equality, hashing, type-separation, and diagnostic-text tests.
- Extended the Tensor API reference with identifier ownership, graph scope, allocation-policy separation, publication binding, and the reason `OperationId` remains deferred.
- Synchronized task, master-plan, and roadmap status and advanced the frontier to task 0005.

## Completion summary

- Completed changes: Implemented typed tensor, graph-node, and graph-value identity values without allocation policy or behavior leakage.
- Files changed or created: Three production records, two unit-test classes, the Tensor API reference, this task, the model capability baseline, the model master plan, and the roadmap.
- Tests and validation: Model tests, model Javadoc, full repository tests, diff checks, dependency checks, package checks, scope review, and forbidden-behavior review passed.
- Documentation impact: Documented identifier scopes and future publication binding; no architecture documentation update was required.
- Javadoc review: Every public record, canonical constructor, component, and explicit accessor documents semantics, valid range, scope, results, and failures.
- Unresolved issues: None.
- Follow-up required: Plan task 0005 before introducing operation semantic contracts.

Status: Complete
