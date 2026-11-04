# Task 0006: [Operation](../../../../glossary.md#operation) [Model](../../../../glossary.md#term-model)

**Editorial reading key (later context, not a status update).**

The constructor receives an immutable kind and immutable attributes and retains those exact references: this boundary does not copy tensor values or even receive a Tensor. An Operation is reusable semantic description; a producer or CompiledNode identifies one occurrence with particular ordered inputs. Model's responsibility for that description does not extend to choosing a backend or owning result buffers. The absence of family compatibility checks is the original stage recorded here; the linked current signature contract documents a later layer of validation.

Focused current reading: [Current operation foundation](../../../../reference/tensor/operation-foundation.md#operation-semantic-foundation). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Define the minimal [immutable](../../../../glossary.md#term-immutable), [backend](../../../../glossary.md#backend)-independent [`Operation`](../../../../glossary.md#operation) [descriptor](../../../../glossary.md#term-descriptor) that pairs one
non-null [`OperationKind`](../../../../api/tensor-api.md#operation-semantic-foundation)<span data-glossary-navigation="true"> ([OperationKind glossary](../../../../glossary.md#operationkind))</span> with one
non-null typed [`OperationAttrs`](../../../../glossary.md#operationattrs) value, without adding operation families, [graph](../../../../glossary.md#graph) [occurrence](../../../../glossary.md#term-occurrence)
[identity](../../../../glossary.md#term-identity-meanings), [inference](../../../../glossary.md#term-inference), execution, [compiler](../../../../glossary.md#term-compiler) policy, or backend [metadata](../../../../glossary.md#term-metadata).

## [Scope](../../../../glossary.md#term-scope)

- Add one public [`Operation`](../../../../glossary.md#operation) value type in the existing `model.operation` package.
- Store exactly one [`OperationKind`](../../../../glossary.md#operationkind) and one [`OperationAttrs`](../../../../glossary.md#operationattrs) reference.
- Reject a null kind or null attributes value at construction.
- Expose the two stored values through typed record accessors.
- Provide [structural equality](../../../../glossary.md#term-structural-equality), hashing, and diagnostic text through record value semantics.
- Add one focused unit-test class using only test-local [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) kinds and attributes.
- Update the public [API](../../../../glossary.md#term-application-programming-interface-api) explanation, glossary, and [planning](../../../../glossary.md#term-planning) status after implementation.

## Out of [scope](../../../../glossary.md#term-scope)

- concrete mathematical, [layout](../../../../glossary.md#layout), [compiler](../../../../glossary.md#term-compiler)-generated, [`FUSED`](../../../../glossary.md#specialization-and-fusion), `UNKNOWN`, or sentinel kinds
- production family-specific attribute records or [validation](../../../../glossary.md#term-validation) of family-specific [parameters](../../../../glossary.md#term-parameter)
- a production kind-to-attribute compatibility registry, lookup table, map, reflection-based
  family matcher, runtime discovery mechanism, or service; test reflection may inspect record shape<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- convenience constructors or factories, including an implicit no-attributes factory
- `OperationId`, [`NodeId`](../../../../glossary.md#nodeid), [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), [`TensorId`](../../../../glossary.md#tensorid), or any other [identity](../../../../glossary.md#term-identity-meanings) or [occurrence](../../../../glossary.md#term-occurrence) field
- input or output [tensors](../../../../glossary.md#tensor), tensor [descriptors](../../../../glossary.md#term-descriptor), [graph](../../../../glossary.md#graph) values, graph [nodes](../../../../glossary.md#node), or graph behavior
- input/output arity, result count, shape inference, data type inference, broadcasting, or layout
  inference<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span><span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [inference glossary](../../../../glossary.md#term-inference); [data type glossary](../../../../glossary.md#data-type--datatype); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- execution, [lowering](../../../../glossary.md#lowering), [kernels](../../../../glossary.md#kernel), [autograd](../../../../glossary.md#autograd), differentiation policy, or compiler transformations
- [backend](../../../../glossary.md#backend) support, [backend ownership](../../../../glossary.md#backend-ownership), fallback, capability declarations, or [device](../../../../glossary.md#term-device) facts
- cost, [fusion](../../../../glossary.md#term-fusion) eligibility, [materialization](../../../../glossary.md#materialization) decisions, or [kernel](../../../../glossary.md#kernel) routes
- storage, [runtime](../../../../glossary.md#term-runtime), compiler, [planning](../../../../glossary.md#term-planning), [prepare](../../../../glossary.md#prepare), or [mutable](../../../../glossary.md#term-mutable) state
- changes to existing Java contracts, [Gradle](../../../../glossary.md#term-gradle), `ARCHITECTURE.md`, focused architecture
  documentation, the capability baseline, or another module<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially the [model](../../../../glossary.md#term-model) [ownership](../../../../glossary.md#term-ownership-meanings) and
  [`Operation`](../../../../glossary.md#operation) invariants
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Model capability baseline](../capabilities.md), especially the operation foundation baseline
- [Model master plan](../master-plan.md), especially `model.operation` ownership
- [Task 0005](0005-operation-semantic-foundation.md), which defines the component contracts

## Legacy evidence

The read-only `legacy/pre-rewrite` branch represented an [operation](../../../../glossary.md#operation) through an
`operations.Operation` interface implemented by many concrete classes. Its nested `OpType` enum
identified a primitive, while concrete classes stored [parameters](../../../../glossary.md#term-parameter) such as a reduction [dimension](../../../../glossary.md#dimension) or
`keepDims`. The interface also exposed [arity](../../../../glossary.md#term-arity) classification, semantic family, cost, result kind,
[fusion](../../../../glossary.md#term-fusion) eligibility, and expression text. Representative tests checked those [metadata](../../../../glossary.md#term-metadata) values but
did not establish [structural equality](../../../../glossary.md#term-structural-equality) for a minimal kind-and-attributes [descriptor](../../../../glossary.md#term-descriptor).

The useful evidence is limited to the need to [carry](../../../../glossary.md#term-carry) stable semantic [identity](../../../../glossary.md#term-identity-meanings) together with
[immutable](../../../../glossary.md#term-immutable) semantic parameters. This task does not copy the legacy interface hierarchy, monolithic
enum, concrete classes, nullable broadcast plans, metadata categories, class names, or package
structure. Kind identity and typed attributes come only from the new contracts completed by task
0005; [planning](../../../../glossary.md#term-planning) and [backend](../../../../glossary.md#backend) concerns remain outside the [model](../../../../glossary.md#term-model) descriptor.

## Architecture constraints

- Production packages remain below `io.github.pho001.synaptik.*`.
- [`Operation`](../../../../glossary.md#operation) lives in `io.github.pho001.synaptik.model.operation` with its component contracts.
- Production code uses only the Java language and JDK standard library and adds no project-module
  dependency.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- The [descriptor](../../../../glossary.md#term-descriptor) is [immutable](../../../../glossary.md#term-immutable) and [backend](../../../../glossary.md#backend)-independent. Its complete stored state is `kind` plus
  `attrs`.
- `Operation` owns semantic description only. It does not identify a graph occurrence; a later
  graph node owns <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [node glossary](../../../../glossary.md#node))</span>[`NodeId`](../../../../glossary.md#nodeid) and applies operation semantics to [graph](../../../../glossary.md#graph) values.
- [Runtime](../../../../glossary.md#term-runtime) hot paths must not consume `Operation`; this [model](../../../../glossary.md#term-model) type is [compile](../../../../glossary.md#compile)-time semantic state.
- The constructor may validate component nullness because both component roles are known. It must
  not invent a family contract to decide whether a particular attributes type matches a kind.
- Family attribute constructors remain responsible for their own value ranges, normalization,
  defensive copies, immutability, equality, and hashing.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- [`OperationKind`](../../../../glossary.md#operationkind) implementations remain responsible for their documented stable, non-null,
  non-blank names and typed equality. `Operation` must not call `name()` merely to duplicate that
  contract during construction.
- If implementation requires another production type, concrete kind or attributes value,
  registry, dependency, identity, inference rule, or architecture change, stop and report the
  issue.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [identity glossary](../../../../glossary.md#term-identity-meanings); [inference glossary](../../../../glossary.md#term-inference))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation` — already owns backend-independent operation
  semantics and immutable attribute contracts.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation); [immutable glossary](../../../../glossary.md#term-immutable))</span>

Packages added or changed:

- No package is added. The existing `model.operation` package gains one public [descriptor](../../../../glossary.md#term-descriptor).

Type placement:

- `io.github.pho001.synaptik.model.operation.Operation` — pairs the kind and typed attributes that
  jointly describe backend-independent computation semantics.

Test placement:

- `io.github.pho001.synaptik.model.operation.OperationTest` — mirrors the production package and
  uses private test-local kind enums and attribute records so the task adds no production family.

## Required contract

Implement [`Operation`](../../../../glossary.md#operation) as a public record with exactly these components:

```java
public record Operation(OperationKind kind, OperationAttrs attrs) { ... }
```

A record is the smallest Java form that makes the two components final and supplies equality,
hashing, and diagnostic rendering from the complete declared state. It also keeps the [descriptor](../../../../glossary.md#term-descriptor)
open to future typed [`OperationKind`](../../../../glossary.md#operationkind) and [`OperationAttrs`](../../../../glossary.md#operationattrs) implementations without inheritance,
registration, or a production reflection-based mechanism.

The [canonical](../../../../glossary.md#term-canonical-form) constructor must use explicit `Objects.requireNonNull` checks with component-specific
messages for `kind` and `attrs`. Null does not mean “no [parameters](../../../../glossary.md#term-parameter)”; callers use
`NoOperationAttrs.INSTANCE`. The constructor stores both validated references unchanged. It does
not normalize, copy, wrap, or replace them because their existing contracts require [immutable](../../../../glossary.md#term-immutable)
values and their typed [identity](../../../../glossary.md#term-identity-meanings) is semantically significant.

Explicit `kind()` and `attrs()` accessors must document and return the exact stored, non-null
objects. No `of`, one-[argument](../../../../glossary.md#term-argument) constructor, or no-attributes factory is needed. Requiring callers
to supply attributes prevents a convenience [API](../../../../glossary.md#term-application-programming-interface-api) from silently assigning
`NoOperationAttrs.INSTANCE` to a future kind that requires parameters.

Do not override `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), or `toString`. Record-generated equality and hashing compare
both typed components structurally. Therefore equal kind and attributes values produce equal
operations and equal [hashes](../../../../glossary.md#term-hash), while equal diagnostic kind names from different kind types do not
collapse typed identity. Record-generated text exposes the `kind` and `attrs` component labels and
their diagnostic values. It is inspection text only, not a [serialization](../../../../glossary.md#term-serialization) format, parser contract,
[dispatch](../../../../glossary.md#term-dispatch) key, or substitute for typed accessors.

The generic descriptor can validate only that both component references are present and implement
their declared interfaces. Without family-specific contracts it cannot validate kind-to-attributes
compatibility, parameter ranges, [arity](../../../../glossary.md#term-arity), input/output relationships, [shape](../../../../glossary.md#shape) or [data type](../../../../glossary.md#data-type--datatype) semantics,
or the [runtime](../../../../glossary.md#term-runtime) immutability of arbitrary custom implementations. Do not emulate those missing
contracts in production with class-name checks, maps, registries, reflection-based matching, or
hard-coded implementation types. Reflection remains permitted in tests and manual [validation](../../../../glossary.md#term-validation) only
to inspect the public record shape and declared state.

## Affected files

Expected production file:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/Operation.java`

Expected test file:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) updates:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- one production Java file;
- one test Java file; and
- the five documentation and [planning](../../../../glossary.md#term-planning) files listed above.

Do not modify the task-0005 contracts or tests. If another source, test, build, architecture,
capability, or documentation file is required, stop and propose a separately reviewed follow-up
instead of expanding this task.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- The public record Javadoc must define an operation as the immutable pairing of semantic kind and
  typed attributes, explain its compile-time/model role, distinguish it from a graph node and all
  identifier types, and state the backend/runtime/compiler boundaries.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [immutable glossary](../../../../glossary.md#term-immutable); [compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Record-component `@param` tags must document non-nullness, semantic ownership, unchanged
  reference storage, and the use of <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`NoOperationAttrs.INSTANCE` for a parameterless kind.
- The canonical constructor Javadoc must document both inputs, the absence of copying or
  normalization, and <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [normalization glossary](../../../../glossary.md#term-normalization))</span>`@throws NullPointerException` separately for null `kind` and null `attrs`.
- The explicit `kind()` and `attrs()` Javadocs must each include `@return`, non-nullness, exact
  stored-object identity, and what the returned value means.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span>
- The type Javadoc must describe record-generated structural equality and hashing over both
  components and classify <span data-glossary-navigation="true"> ([structural equality glossary](../../../../glossary.md#term-structural-equality))</span>`toString()` as diagnostic, non-[serialization](../../../../glossary.md#term-serialization) text.
- Javadoc must state that compatibility and family parameter validation are not performed by this
  descriptor and belong to future family-specific contracts.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Javadoc must be meaningful contract documentation, not a restatement of signatures. No public
  constructor, component, or accessor may be left with only generated or implicit documentation.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>

## Acceptance criteria

- [`Operation`](../../../../glossary.md#operation) is a public record with exactly the `OperationKind kind` and `OperationAttrs attrs`
  components and no additional instance state.
- Construction preserves the exact non-null kind and attributes references supplied by the caller.
- Null kind and null attributes are independently rejected with `NullPointerException`; neither is
  defaulted or treated as absence.
- `kind()` and `attrs()` expose the stored non-null values with complete [Javadoc](../../../../glossary.md#term-javadoc).
- Equal component values yield equal operations and equal hash codes; a changed kind or attributes
  value yields an unequal operation.<span data-glossary-navigation="true"> ([hash codes glossary](../../../../glossary.md#hash-codes-and-content-digests))</span>
- Kind values from distinct concrete types remain distinct even when `name()` returns equal text.
- `NoOperationAttrs.INSTANCE` constructs a valid parameterless descriptor and retains singleton
  identity through <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [singleton glossary](../../../../glossary.md#term-singleton); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>`attrs()`.
- A private test-local [immutable](../../../../glossary.md#term-immutable) record implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) constructs a valid typed
  descriptor and participates in structural equality and diagnostic text.<span data-glossary-navigation="true"> ([structural equality glossary](../../../../glossary.md#term-structural-equality))</span>
- Diagnostic text identifies `Operation`, the `kind` component, the `attrs` component, and their
  representative values without becoming an exact serialization-format assertion.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization))</span>
- Tests cover construction, both null failures, accessor identity and values, structural equality
  and hashing, typed kind separation, canonical no-attributes use, typed test-local attributes,
  and diagnostic text.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- No production concrete kind, family attributes, factory, registry, map, reflection-based family
  matching or runtime mechanism, identity, inference, graph, execution, autograd, backend, cost,
  fusion, storage, or runtime contract is introduced. Test reflection is limited to structural
  verification of the <span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [runtime glossary](../../../../glossary.md#term-runtime); [inference glossary](../../../../glossary.md#term-inference); [graph glossary](../../../../glossary.md#graph); [autograd glossary](../../../../glossary.md#autograd); [backend glossary](../../../../glossary.md#backend); [fusion glossary](../../../../glossary.md#term-fusion))</span>`Operation` record.
- All Javadoc requirements in this task are satisfied and generated Javadoc includes the record,
  canonical constructor, and both accessors.
- A separate documentation-focused agent or thread with clean context independently reviews the
  final implementation and tests, finalizes Javadoc, updates the Tensor API and glossary, and
  records a reasoned review of explanatory documentation, terminology, and glossary impact in the
  same overall change.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Task, master-plan row, and roadmap row have matching final status.
- No existing Java source, test, Gradle file, architecture document, capability baseline, other
  module, or unrelated documentation is changed.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) after implementation and after the documentation-focused pass:

```bash
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Manually verify:

- the diff contains only one production file, one focused test file, and the five allowed
  documentation/planning files;<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- reflection reports [`Operation`](../../../../glossary.md#operation) as a record with exactly two components in the specified order
  and types, with no additional declared instance field;
- focused tests use only private test-local kinds and attributes and introduce no production
  operation family;
- construction performs only component null checks and introduces no family matching mechanism;
- generated Javadoc documents the type, canonical constructor, both record components/accessors,
  null failures, equality/hash semantics, diagnostic text, and cross-layer exclusions;<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [canonical glossary](../../../../glossary.md#term-canonical-form); [hash glossary](../../../../glossary.md#term-hash))</span>
- `docs/api/tensor-api.md` presents `Operation = OperationKind + OperationAttrs` as implemented,
  updates its conceptual example and planned-contract list, and does not imply family-specific or
  executable support;
- `docs/glossary.md` updates the implementation-status convention, `Operation` definition, and
  kind/attributes/operation distinction without changing architecture authority;
- local Markdown links and anchors in all five changed documentation/planning files resolve, code
  fences are balanced, and no changed file has trailing whitespace;
- the separate clean-context documentation pass followed
  `docs/developer-guide/documentation-rules.md`, applied the API/Javadoc and general style profiles,
  inspected source and tests, and recorded its commands, findings, changes, reasoned Javadoc
  review, and any limitations; and<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- task, master plan, roadmap, API documentation, and glossary agree on implemented versus planned
  scope and final status.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Dependencies

- Task 0005 is complete and provides [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), and
  `NoOperationAttrs.INSTANCE`.
- The existing `model.operation` package ownership and dependency boundary are defined by the
  model master plan.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency glossary](../../../../glossary.md#term-dependency); [model glossary](../../../../glossary.md#term-model))</span>

## Follow-up tasks

- Task 0008 will compose [`Operation`](../../../../glossary.md#operation) into an immutable graph node after task 0007 provides tensor
  descriptors; it owns graph occurrence identity and input/output relationships.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [descriptors glossary](../../../../glossary.md#term-descriptor); [occurrence glossary](../../../../glossary.md#term-occurrence); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Task 0013 will use `Operation` in minimal public [tensor](../../../../glossary.md#tensor) [provenance](../../../../glossary.md#provenance) without turning `Tensor` into
  compiled IR.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Tasks 0014–0022 will introduce concrete public operation kinds and typed family attributes
  progressively.
- Task 0023 will introduce compiler-generated semantic operations without moving autograd rules
  into the model.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [model glossary](../../../../glossary.md#term-model))</span>

Do not create a detailed task specification for task 0007 or any later task as part of this work.

## Architecture impact

Expected impact: None.

The architecture already assigns [`Operation`](../../../../glossary.md#operation) and operation semantics to `modules/model`. This task
adds one value [descriptor](../../../../glossary.md#term-descriptor) inside the existing package and changes no [module](../../../../glossary.md#term-module) boundary, [dependency
direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle) rule, or [backend](../../../../glossary.md#backend) contract. Architecture documentation and architecture tests
therefore require no update. If implementation reveals otherwise, stop and report the conflicting
rule and required decision before editing architecture files.

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are a clean-context implementation agent working in the Synaptik repository.

Read first and in full:
- AGENTS.md
- ARCHITECTURE.md
- docs/developer-guide/documentation-rules.md
- docs/planning/planning-guide.md
- docs/planning/modules/model/capabilities.md
- docs/planning/modules/model/master-plan.md
- docs/planning/modules/model/tasks/0005-operation-semantic-foundation.md
- docs/planning/modules/model/tasks/0006-operation-model.md
- docs/api/tensor-api.md
- docs/glossary.md
- the existing production and test files under
  modules/model/src/{main,test}/java/io/github/pho001/synaptik/model/operation

Implement task 0006 exactly as specified. Create only Operation.java and OperationTest.java for
code and tests. Operation must be a public record with exactly OperationKind kind and
OperationAttrs attrs, explicit null validation, documented accessors, and record-generated
equals/hashCode/toString. Do not add a factory, convenience constructor, concrete production kind,
family attributes, registry, map, production reflection-based family matching or runtime discovery,
identity, tensors or descriptors, arity, inference, graph behavior, execution, autograd, backend
support or ownership, costs, fusion, routes, storage, runtime/compiler/planning state, dependencies,
or unrelated refactors. OperationTest and manual validation may use reflection only to verify the
record shape, component order and types, and absence of additional instance state.

Do not modify existing Java contracts or tests, Gradle, ARCHITECTURE.md, focused architecture
documentation, capabilities.md, another module, or unrelated documentation. Stop and report if a
required change exceeds the task's affected-file or maximum-scope list, if kind-to-attributes
compatibility appears to require a new family contract, or if any architecture uncertainty or
conflict appears.

Add complete Javadoc required by the task. Use only private test-local kind enums and a private
test-local attributes record. Run every validation command and manual check in the task.

After code implementation and initial validation, hand the resulting diff to a separate
documentation-focused agent or thread with a clean context in the same overall change. The handoff
must include this task, the diff, affected API behavior, architecture constraints, expected
Tensor API and glossary updates, all Javadoc requirements, and validation commands. That agent
must read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md, General style,
API and Javadoc style, the task, final source/tests, docs/api/tensor-api.md, and docs/glossary.md;
then independently finalize Operation Javadoc, the API explanation, glossary/status distinctions,
links, anchors, terminology, examples, and formatting. It must record a reasoned Javadoc review,
including why existing OperationKind, OperationAttrs, and NoOperationAttrs Javadocs remain aligned
without modification or why an out-of-scope discrepancy requires stopping.

At the end, update only this task file, the model master plan, and the roadmap for planning status.
Record local decisions, known limitations, exact validation evidence including the documentation
agent identity and results, implementation notes, and the canonical completion summary. Do not mark
task 0006 Complete until all acceptance criteria, both validation passes, documentation changes,
and status synchronization are complete.
```

## Local decisions

- [`Operation`](../../../../glossary.md#operation) is a two-component record because kind and typed attributes are its entire semantic
  state; a class hierarchy or generic pairing would add machinery without a current family
  contract.
- The component names are `kind` and `attrs`, matching the completed task-0005 vocabulary and
  producing concise accessors without aliases.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- The canonical constructor checks only null component references and retains their identity. It
  does not call <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>`kind.name()` or inspect the attributes implementation because those contracts and
  family validation belong to their owning types.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- There is no convenience factory. Explicitly supplying `NoOperationAttrs.INSTANCE` keeps absence
  non-null and avoids guessing whether a future kind requires parameters.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>
- Record-generated equality, hashing, and text are sufficient. Equality remains typed through the
  component values; diagnostic text is intentionally not a wire format.<span data-glossary-navigation="true"> ([wire format glossary](../../../../glossary.md#term-wire-format))</span>

## Known limitations

- The descriptor cannot prove at runtime that an arbitrary open-interface implementation is
  immutable or obeys structural equality and stable-name requirements; the component contracts
  and their focused implementations own those obligations.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable); [structural equality glossary](../../../../glossary.md#term-structural-equality))</span>
- Until family-specific contracts exist, [`Operation`](../../../../glossary.md#operation) cannot reject a structurally valid but
  semantically mismatched kind-and-attributes pair.
- No production kind or family attributes value exists yet, so task 0006 demonstrates the
  descriptor only with test-local values and `NoOperationAttrs.INSTANCE`.
- The descriptor provides no graph occurrence, inference, execution, autograd, backend, planning,
  or runtime behavior.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence); [inference glossary](../../../../glossary.md#term-inference); [autograd glossary](../../../../glossary.md#autograd); [backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean [planning](../../../../glossary.md#term-planning) context `/root/plan_model_0006` read the required architecture, documentation,
  planning, capability, task-0005, API, glossary, current operation source/test, and read-only
  legacy evidence before defining the contract.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `git status --short` confirmed that planning changed only this new task, the model master plan,
  and the implementation roadmap. Manual scope review confirmed no Java, test, Gradle,
  architecture, capability-baseline, other-module, or unrelated documentation change.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [scope glossary](../../../../glossary.md#term-scope); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- A targeted local Markdown path-and-heading check resolved every relative link and anchor in all
  three changed planning documents.
- Fence counts were even in every changed document, a trailing-whitespace scan returned no
  matches, and `git diff --check` passed for tracked changes. A no-index whitespace check of this
  new task returned no diagnostic.
- Status review confirmed task 0005 remains `Complete` and task 0006 is [`Ready`](../../../../glossary.md#term-ready) in this task, the
  master-plan row/current status/notes, and the roadmap frontier/table. Task order and dependencies
  are unchanged, and no task-0007 specification was created.
- At planning completion, implementation had not started; the implementation and documentation
  evidence below was appended before the task became `Complete`.
- Clean implementation context `/root/implement_model_0006` ran
  `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; the model suite contains 78 passing tests,
  including seven focused `OperationTest` tests with zero failures, errors, or skipped tests.
- The implementation context ran `./gradlew :modules:model:javadoc` and `./gradlew test` — both
  `BUILD SUCCESSFUL`; `git diff --check` passed with no output.
- Implementation inspection with `javap -p -c` confirmed exactly two private final fields in record
  component order, two component null checks, direct reference-returning accessors, and generated
  record equality, hashing, and diagnostic text. Exactly `Operation.java` and `OperationTest.java`
  were added during the code phase.
- Clean documentation context `/root/review_model_0006_docs` applied General style, API and Javadoc
  style, and Example format. It independently reviewed the architecture and planning inputs, the
  complete `model.operation` production/test package, generated Javadoc, test reports, and the
  actual diff.
- The documentation context finalized only [`Operation`](../../../../glossary.md#operation) [Javadoc](../../../../glossary.md#term-javadoc), `docs/api/tensor-api.md`, and
  `docs/glossary.md`. It confirmed with reasoned no-change conclusions that [`OperationKind`](../../../../glossary.md#operationkind),
  [`OperationAttrs`](../../../../glossary.md#operationattrs), and [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) Javadocs remain accurate because their contracts did not
  change.
- The documentation context ran `./gradlew :modules:model:javadoc` and
  `./gradlew :modules:model:test` — both `BUILD SUCCESSFUL`; all 78 model tests passed. It validated
  48 local Markdown links and anchors, code fences, trailing whitespace, current-versus-planned
  terminology, generated Javadoc, record bytecode, and <span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>`git diff --check` with no findings.
- The final coordinating context reran `./gradlew :modules:model:test`,
  `./gradlew :modules:model:javadoc`, and `./gradlew test` after documentation and planning
  synchronization — all reported <span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>`BUILD SUCCESSFUL`; a final `git diff --check` passed with no
  output.
- Final test-report inspection recorded 78 tests with zero failures, errors, or skipped tests.
  `javap -p -c` reconfirmed exactly two private final component fields, only the two required null
  checks and direct assignments in the constructor, direct accessors, and generated record
  equality, hashing, and text.
- Final scope review confirmed only one new production file, one new test file, Tensor API,
  glossary, this task, model master plan, and implementation roadmap changed. Existing Java
  contracts and tests, <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>`ARCHITECTURE.md`, focused architecture documentation, the capability
  baseline, Gradle files, other modules, and unrelated documentation remain unchanged.

## Implementation notes

- Added [`Operation`](../../../../glossary.md#operation) as a public two-component record over `OperationKind kind` and
  `OperationAttrs attrs` in the existing `model.operation` package.
- The compact [canonical](../../../../glossary.md#term-canonical-form) constructor uses component-specific `Objects.requireNonNull` checks and
  retains both validated references unchanged. Explicit accessors document and return those exact
  references.
- Record-generated `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString` cover the complete declared state; no
  factory, convenience constructor, additional state, family matcher, or runtime mechanism was
  added.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- Added one focused `OperationTest` using only private test-local enum kinds and a private record
  attribute value. It covers record shape, reference identity, both null failures, structural
  equality and hashing, typed kind separation, <span data-glossary-navigation="true"> ([reference identity glossary](../../../../glossary.md#term-reference-identity); [structural equality glossary](../../../../glossary.md#term-structural-equality))</span>`NoOperationAttrs.INSTANCE`, typed attributes, and
  diagnostic text.
- Updated the [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) example and glossary to classify `Operation` as implemented while keeping
  concrete operation families, family-specific attributes, and graph/executable integration
  planned.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>

## Completion summary

- Completed changes: Implemented the minimal [immutable](../../../../glossary.md#term-immutable) [backend](../../../../glossary.md#backend)-independent [`Operation`](../../../../glossary.md#operation) descriptor
  over the task-0005 kind and attribute contracts.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Files changed or created: `Operation.java`, `OperationTest.java`, Tensor API, glossary, this task,
  model master plan, and implementation roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: All 78 model tests, generated model Javadoc, complete repository tests,
  reflection/bytecode shape checks, documentation checks, scope checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [shape glossary](../../../../glossary.md#shape); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`
  passed.
- Documentation-agent review: `/root/review_model_0006_docs` independently completed the required
  clean-context documentation pass in the same overall change.
- Documentation impact: The API and glossary now explain the implemented descriptor, exact
  reference ownership, record value semantics, missing family compatibility validation, and
  current-versus-planned boundaries.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Javadoc review: Complete for the record, canonical constructor, components/accessors, failure
  modes, equality/hash/text semantics, and cross-layer exclusions. Existing task-0005 contract
  Javadocs remain accurate without modification.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [hash glossary](../../../../glossary.md#term-hash))</span>
- Glossary impact: `Operation` is now implemented; concrete kinds, family attributes, and graph
  integration remain planned.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0006. Plan task 0007 separately before implementation.

Status: Complete
