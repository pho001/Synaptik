# Task 0005: [Operation](../../../../glossary.md#operation) Semantic Foundation

## Status

Complete

## Goal

Define the minimal extensible contracts for [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) kinds and [immutable](../../../../glossary.md#term-immutable) typed operation attributes without introducing concrete mathematical operations, a monolithic operation inventory, [graph](../../../../glossary.md#graph) behavior, [compiler](../../../../glossary.md#term-compiler) policy, [autograd](../../../../glossary.md#autograd), [fusion](../../../../glossary.md#term-fusion), cost estimates, or backend support [metadata](../../../../glossary.md#term-metadata).

## [Scope](../../../../glossary.md#term-scope)

- Define a public [`OperationKind`](../../../../glossary.md#operationkind) semantic discriminator contract.
- Define a public [`OperationAttrs`](../../../../glossary.md#operationattrs) marker for [immutable](../../../../glossary.md#term-immutable) typed attribute values.
- Define a [singleton](../../../../glossary.md#term-singleton) [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) value for operations with no semantic [parameters](../../../../glossary.md#term-parameter).
- Specify stability, immutability, equality, diagnostic-name, and [ownership](../../../../glossary.md#term-ownership-meanings) expectations for future kind and attribute implementations.
- Add focused tests and public [API](../../../../glossary.md#term-application-programming-interface-api) documentation.

## Out of [scope](../../../../glossary.md#term-scope)

- the [`Operation`](../../../../glossary.md#operation) [descriptor](../../../../glossary.md#term-descriptor) itself, which belongs to task 0006
- concrete kinds such as add, reduction, reshape, [convolution](../../../../glossary.md#term-channels-first-convolution-layers), [loss](../../../../glossary.md#loss), or [compiler](../../../../glossary.md#term-compiler)-generated operations
- family-specific attribute records, [axes](../../../../glossary.md#term-axis), [scalars](../../../../glossary.md#term-scalar), [padding](../../../../glossary.md#padding), windows, reductions, or [shape](../../../../glossary.md#shape) [parameters](../../../../glossary.md#term-parameter)
- a complete operation enum or one Java type per legacy operation
- [arity](../../../../glossary.md#term-arity) [validation](../../../../glossary.md#term-validation), input/output descriptors, shape [inference](../../../../glossary.md#term-inference), [data type](../../../../glossary.md#data-type--datatype) inference, [broadcasting](../../../../glossary.md#broadcasting), or [graph](../../../../glossary.md#graph) construction
- operation execution, [kernels](../../../../glossary.md#kernel), [lowering](../../../../glossary.md#lowering), [backend capability](../../../../glossary.md#backend-capability), [backend ownership](../../../../glossary.md#backend-ownership), or fallback
- cost classification, [fusion](../../../../glossary.md#term-fusion) eligibility, [materialization](../../../../glossary.md#materialization) policy, or result storage
- [autograd](../../../../glossary.md#autograd) rules, differentiability policy, or [backward graph](../../../../glossary.md#backward-graph) construction
- [`FUSED`](../../../../glossary.md#specialization-and-fusion), `UNKNOWN`, invalid sentinels, string-keyed attribute maps, or reflective discovery
- changes to existing Java contracts, [Gradle](../../../../glossary.md#term-gradle), architecture documentation, or another [module](../../../../glossary.md#term-module)

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially the [`Operation`](../../../../glossary.md#operation), [`OperationAttrs`](../../../../glossary.md#operationattrs), and forbidden [backend](../../../../glossary.md#backend)-support rules
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>, especially the operation foundation and public operation inventory
- [Model master plan](../master-plan.md), especially `model.operation` [ownership](../../../../glossary.md#term-ownership-meanings)

## Legacy evidence

The read-only legacy `operations.Operation` interface combined a monolithic `OpType` enum with execution-category, semantic-family, computational-cost, result-kind, [fusion](../../../../glossary.md#term-fusion)-eligibility, and diagnostic-expression methods. Concrete [operation](../../../../glossary.md#operation) classes also carried semantic [parameters](../../../../glossary.md#term-parameter) directly, while raw enum values included [`FUSED`](../../../../glossary.md#specialization-and-fusion) and `UNKNOWN` sentinels.

The useful evidence is that [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) needs a stable semantic kind and [immutable](../../../../glossary.md#term-immutable) operation parameters. The new design must not copy legacy coupling:

- computational cost belongs to [planning](../../../../glossary.md#term-planning) profiles and [backend capability](../../../../glossary.md#backend-capability) data;
- concrete fusion belongs to [backend](../../../../glossary.md#backend) [prepare](../../../../glossary.md#prepare);
- backend support never belongs to `Operation` or its kind;
- `UNKNOWN` is not a supported semantic operation;
- family parameters use typed immutable values rather than untyped maps; and
- the complete kind inventory is introduced progressively with the operation-family tasks.

## Architecture constraints

- Production packages remain below `io.github.pho001.synaptik.*`.
- All new contracts live in `io.github.pho001.synaptik.model.operation`.
- Production code uses only the Java language and has no project-[module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency).
- [Operation](../../../../glossary.md#operation) kinds express semantic [identity](../../../../glossary.md#term-identity-meanings) only and expose no [backend](../../../../glossary.md#backend), [planning](../../../../glossary.md#term-planning), [runtime](../../../../glossary.md#term-runtime), storage, cost, [fusion](../../../../glossary.md#term-fusion), or execution facts.
- Operation attributes express [immutable](../../../../glossary.md#term-immutable) semantic [parameters](../../../../glossary.md#term-parameter) only and expose no live [tensor](../../../../glossary.md#tensor), service, backend, runtime, or [mutable](../../../../glossary.md#term-mutable) collection state.
- Future family enums or immutable values may implement the open contracts without reflective registration.
- If implementation requires a concrete operation kind, family attribute, `Operation`, dependency, or architecture change, stop and report the issue.

## Package impact

Package added:

- `io.github.pho001.synaptik.model.operation` owns [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) semantics and [immutable](../../../../glossary.md#term-immutable) attribute contracts.

Type placement:

- `io.github.pho001.synaptik.model.operation.OperationKind` — open semantic-kind contract implemented by future typed family vocabularies.
- `io.github.pho001.synaptik.model.operation.OperationAttrs` — open marker contract implemented by future immutable typed attribute records.
- `io.github.pho001.synaptik.model.operation.NoOperationAttrs` — [canonical](../../../../glossary.md#term-canonical-form) [singleton](../../../../glossary.md#term-singleton) for kinds with no semantic [parameters](../../../../glossary.md#term-parameter).

Test placement:

- `OperationKindTest` validates the semantic-name and concrete-type separation contract with test-local [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) enums.
- `OperationAttrsTest` validates typed structural attributes with a test-local record and the production empty singleton.

## Required contracts

### [`OperationKind`](../../../../glossary.md#operationkind)

Create a public interface with one method:

```java
String name();
```

The contract requires implementations to be [immutable](../../../../glossary.md#term-immutable) values with stable [structural equality](../../../../glossary.md#term-structural-equality). `name()` returns a non-null, non-blank semantic name suitable for diagnostics and deterministic [model](../../../../glossary.md#term-model) inspection. It is not a [serialization](../../../../glossary.md#term-serialization), [backend](../../../../glossary.md#backend) [dispatch](../../../../glossary.md#term-dispatch), [kernel](../../../../glossary.md#kernel) route, or reflective class name. Future enum implementations satisfy the method through `Enum.name()`; other implementations must provide equivalent stable behavior.

Kinds with the same textual name but different concrete kind types are not implicitly equivalent. Semantic equivalence uses the typed kind value, not a global string [registry](../../../../glossary.md#term-registry).

### [`OperationAttrs`](../../../../glossary.md#operationattrs)

Create a public [marker interface](../../../../glossary.md#term-marker-interface) with no methods. Implementations must:

- be [immutable](../../../../glossary.md#term-immutable) values;
- defensively isolate [mutable](../../../../glossary.md#term-mutable) constructor inputs;
- provide [structural equality](../../../../glossary.md#term-structural-equality) and hashing;
- use typed fields and accessors rather than string-keyed maps; and
- contain semantic [parameters](../../../../glossary.md#term-parameter) only.

The marker deliberately does not prescribe family fields before their focused tasks. It must not expose [backend](../../../../glossary.md#backend) support, [device](../../../../glossary.md#term-device) handles, [runtime](../../../../glossary.md#term-runtime) state, [compiler](../../../../glossary.md#term-compiler) services, or mutable [tensors](../../../../glossary.md#tensor).

### [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs)

Create a public enum implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with exactly one value:

```text
INSTANCE
```

It represents the complete [immutable](../../../../glossary.md#term-immutable) attribute value for an [operation](../../../../glossary.md#operation) kind with no semantic [parameters](../../../../glossary.md#term-parameter). Later `Operation` construction uses this object rather than `null` or an empty map.

Do not add operation families, [arity](../../../../glossary.md#term-arity) classes, costs, result kinds, [fusion](../../../../glossary.md#term-fusion) flags, sentinels, registries, factories, or concrete mathematical kinds in this task.

## Affected files

Expected production files:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/NoOperationAttrs.java`

Expected test files:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationKindTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationAttrsTest.java`

Expected documentation/[planning](../../../../glossary.md#term-planning) updates:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

The capability baseline already records the selected foundation. If implementation evidence requires changing it, stop and report the discrepancy before editing that baseline.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- three production Java files;
- two test Java files; and
- the five documentation/[planning](../../../../glossary.md#term-planning) files listed above.

Do not add [`Operation`](../../../../glossary.md#operation), family types, utilities, registries, or abstraction layers. If another source file is required, stop and propose a follow-up task.

## Acceptance criteria

- [`OperationKind`](../../../../glossary.md#operationkind) is a public one-method interface with complete semantic-name documentation.
- [`OperationAttrs`](../../../../glossary.md#operationattrs) is a public zero-method marker with explicit immutability and semantic-only requirements.
- [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) is a public single-value enum implementing `OperationAttrs`.
- Future enum kinds can implement `OperationKind` through their existing `name()` method.
- Future typed [immutable](../../../../glossary.md#term-immutable) records can implement `OperationAttrs` without maps or reflection.
- No concrete mathematical kind, family attribute, [`Operation`](../../../../glossary.md#operation), generator, [registry](../../../../glossary.md#term-registry), parser, or service is introduced.
- No [backend](../../../../glossary.md#backend) support, cost, [fusion](../../../../glossary.md#term-fusion), [kernel](../../../../glossary.md#kernel), [device](../../../../glossary.md#term-device), storage, [runtime](../../../../glossary.md#term-runtime), [compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), [autograd](../../../../glossary.md#autograd), or [materialization](../../../../glossary.md#materialization) vocabulary appears in the public [API](../../../../glossary.md#term-application-programming-interface-api) except documentation that explicitly forbids it.
- [`FUSED`](../../../../glossary.md#specialization-and-fusion), `UNKNOWN`, and null-as-empty attributes are not modeled.
- Public types, methods, and enum [constants](../../../../glossary.md#term-constant) have detailed [Javadoc](../../../../glossary.md#term-javadoc) covering semantics, results, immutability, [nullability](../../../../glossary.md#term-nullability), intended implementations, and exclusions.
- After implementation, a separate documentation-focused agent or thread with clean context independently reviews and finalizes the public Javadoc, [Tensor](../../../../glossary.md#tensor) API explanation, and glossary impact in the same overall change, updating the glossary for new or changed project-specific terms when applicable.
- Tests cover interface [shape](../../../../glossary.md#shape), enum compatibility, stable names, concrete-type separation, typed structural attributes, marker membership, [singleton](../../../../glossary.md#term-singleton) [identity](../../../../glossary.md#term-identity-meanings), and diagnostic text.
- Production code contains no imports.
- No existing Java source, [Gradle](../../../../glossary.md#term-gradle) file, architecture document, or other [module](../../../../glossary.md#term-module) is changed.
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
- [`OperationKind`](../../../../glossary.md#operationkind) declares only `name()` and [`OperationAttrs`](../../../../glossary.md#operationattrs) declares no methods;
- [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) contains exactly `INSTANCE`;
- production files contain no imports;
- no legacy [planning](../../../../glossary.md#term-planning)/[backend](../../../../glossary.md#backend) [metadata](../../../../glossary.md#term-metadata) or concrete [operation](../../../../glossary.md#operation) inventory was copied;
- generated [Javadoc](../../../../glossary.md#term-javadoc) contains all three public contracts;
- a separate clean-context documentation pass followed `docs/developer-guide/documentation-rules.md`, finalized affected Javadoc and explanatory documentation, updated the glossary for new or changed project-specific terms when applicable, and recorded its evidence in this task; and
- task, master plan, roadmap, [API](../../../../glossary.md#term-application-programming-interface-api) documentation, and glossary agree on status and [scope](../../../../glossary.md#term-scope).

## Dependencies

- No hard implementation-task [dependency](../../../../glossary.md#term-dependency).
- Package migrations and the ordered preceding [model](../../../../glossary.md#term-model) tasks are complete.
- The `model.operation` package [ownership](../../../../glossary.md#term-ownership-meanings) is defined by the master plan.

## Follow-up tasks

- Task 0006: define the [immutable](../../../../glossary.md#term-immutable) [backend](../../../../glossary.md#backend)-independent [`Operation`](../../../../glossary.md#operation) [descriptor](../../../../glossary.md#term-descriptor) over [`OperationKind`](../../../../glossary.md#operationkind) and [`OperationAttrs`](../../../../glossary.md#operationattrs).
- Tasks 0014–0022: introduce concrete public operation kinds and typed family attributes progressively.
- Task 0023: introduce [compiler](../../../../glossary.md#term-compiler)-generated semantic kinds without adding [autograd](../../../../glossary.md#autograd) rules to [model](../../../../glossary.md#term-model).
- Backend-contract and [planning](../../../../glossary.md#term-planning) tasks later consume semantic kinds through their own declarative capability and profile contracts.

Do not create detailed specifications for these follow-ups until task 0005 is complete and the frontier advances.

## Architecture impact

Expected impact: None.

The architecture already assigns [operation](../../../../glossary.md#operation) semantics and [`OperationAttrs`](../../../../glossary.md#operationattrs) to `modules/model`. The open kind and attribute interfaces, typed empty [singleton](../../../../glossary.md#term-singleton), and progressive inventory are local implementation decisions that preserve the existing boundary.

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are working in the Synaptik repository.

Read first:
- AGENTS.md
- ARCHITECTURE.md
- docs/developer-guide/documentation-rules.md
- docs/planning/planning-guide.md
- docs/planning/modules/model/capabilities.md
- docs/planning/modules/model/master-plan.md
- docs/planning/modules/model/tasks/0005-operation-semantic-foundation.md

Implement task 0005 exactly as specified.

Create only OperationKind, OperationAttrs, NoOperationAttrs and their two focused tests in io.github.pho001.synaptik.model.operation. Do not implement Operation, concrete operation kinds, family attributes, maps, registries, costs, fusion, result kinds, backend support, execution, compiler, autograd, graph behavior, dependencies, or changes to existing Java contracts, Gradle, ARCHITECTURE.md, focused architecture documentation, or another module.

Add complete Javadoc for every public contract, method, and enum constant. Run every validation command from the task specification.

After implementation and code validation, hand the resulting diff to a separate documentation-focused agent or thread with clean context. The documentation agent must independently review and finalize all affected Javadoc, docs/api/tensor-api.md, and docs/glossary.md; update the glossary for new or changed project-specific terms when applicable, or record a reasoned no-change conclusion; then run and record applicable documentation validation. This is a separate context working in the same overall branch and change, not a later documentation task or separate commit.

At the end, update this task file with local decisions, known limitations, validation evidence including the documentation-agent pass, implementation notes, completion summary, and final status. Synchronize the task status in the model master plan and implementation roadmap. Do not mark task 0005 Complete until the documentation pass has finished and its evidence is recorded.
```

## Local decisions

- [`OperationKind`](../../../../glossary.md#operationkind) remains an open one-method interface. Enum-based family vocabularies inherit
  `Enum.name()` directly; the foundation adds no adapter, [registry](../../../../glossary.md#term-registry), or duplicate name field.
- Name validity, kind immutability, and typed equality are documented implementation obligations.
  The interface does not add runtime validation because it neither constructs nor wraps future
  kind values.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [validation glossary](../../../../glossary.md#term-validation))</span>
- [`OperationAttrs`](../../../../glossary.md#operationattrs) remains a zero-method marker. Test-local immutable records demonstrate typed
  structural attributes without committing production family fields before their focused tasks.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) uses a one-value enum so the empty attribute value has canonical identity,
  stable equality and hashing, and deterministic diagnostic text without allocation.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Two test-local enum types deliberately use the same constant name to prove that identical name
  text does not collapse distinct typed kind values.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>

## Known limitations

- The open interfaces cannot enforce non-blank names, immutability, defensive copying, or
  structural equality at runtime; future implementations must satisfy these documented contracts
  and their own focused tests.<span data-glossary-navigation="true"> ([defensive copying glossary](../../../../glossary.md#memory-management-atomic-counters-and-complexity); [structural equality glossary](../../../../glossary.md#term-structural-equality); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- No production mathematical kind or family-specific attribute value exists yet.
- [`Operation`](../../../../glossary.md#operation) is not implemented, so the model does not yet pair a kind with attributes or validate
  that an attribute type belongs to a kind.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- This foundation provides semantic vocabulary only. It performs no graph construction, shape or
  data type inference, execution, backend capability analysis, lowering, fusion, or cost modeling.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [shape glossary](../../../../glossary.md#shape); [data type glossary](../../../../glossary.md#data-type--datatype); [inference glossary](../../../../glossary.md#term-inference); [backend capability glossary](../../../../glossary.md#backend-capability); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean implementation context `/root/implement_model_0005` ran
  `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; the model suite contains 71 passing tests,
  including seven new focused tests with zero failures, errors, or skipped tests.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- The implementation context ran `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated
  documentation contains [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), `name()`, and
  `INSTANCE`.
- The implementation context ran `./gradlew test` — `BUILD SUCCESSFUL` for the complete repository.
- The implementation context ran `git diff --check` — passed with no output.
- Manual implementation checks confirmed exactly three new production and two new test files,
  `OperationKind` with only `name()`, zero declared methods on `OperationAttrs`, exactly one
  `NoOperationAttrs` constant, no production imports, and no copied legacy metadata or concrete
  operation inventory.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [metadata glossary](../../../../glossary.md#term-metadata); [operation glossary](../../../../glossary.md#operation))</span>
- Clean documentation context `/root/review_model_0005_docs` applied the API and Javadoc profile,
  General style, and Example format. It independently reviewed all five Java/test files, finalized
  the operation-foundation explanation in <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`docs/api/tensor-api.md`, and updated the status and
  distinctions of the three new contracts in `docs/glossary.md`.
- The documentation context concluded that the submitted Javadocs already fully covered semantics,
  result nullability, immutability, equality, intended implementations, and exclusions, so no
  Javadoc text change was necessary. It changed no Java declaration or behavior.<span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability))</span>
- The documentation context ran `./gradlew :modules:model:javadoc` and
  `./gradlew :modules:model:test` — both `BUILD SUCCESSFUL`; all seven new tests passed.
- The documentation context validated 45 local Markdown links and anchors, code fences, trailing
  whitespace, current-versus-planned terminology, generated Javadoc coverage, and
  `git diff --check` — all passed.
- The final coordinating context reran `./gradlew :modules:model:test`,
  `./gradlew :modules:model:javadoc`, and `./gradlew test` after documentation and planning
  synchronization — all reported <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>`BUILD SUCCESSFUL`; a final `git diff --check` passed with no
  output.
- Final `javap` inspection confirmed that `OperationKind` exposes only `String name()`,
  `OperationAttrs` exposes no methods, and `NoOperationAttrs` declares only the `INSTANCE` enum
  constant. A production-source import scan returned no matches.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Final scope review confirmed that only the three production files, two test files, Tensor API,
  glossary, this task, model master plan, and implementation roadmap changed. <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [Tensor glossary](../../../../glossary.md#tensor))</span>`ARCHITECTURE.md`,
  focused architecture documentation, Gradle files, existing Java contracts, and other modules
  remain unchanged.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Implementation notes

- Added [`OperationKind`](../../../../glossary.md#operationkind) as the minimal semantic-name contract and documented why its name is
  diagnostic rather than a serialization or dispatch key.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Added [`OperationAttrs`](../../../../glossary.md#operationattrs) as the typed [immutable](../../../../glossary.md#term-immutable)-attribute role without prescribing family fields.
- Added `NoOperationAttrs.INSTANCE` as the explicit non-null representation of no semantic
  parameters.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>
- Added focused tests using only test-local sample enums and a record. Tests cover reflection shape,
  inherited enum names, concrete-type separation, structural equality and hashing, marker
  membership, singleton identity, and diagnostic text.<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [shape glossary](../../../../glossary.md#shape); [structural equality glossary](../../../../glossary.md#term-structural-equality); [singleton glossary](../../../../glossary.md#term-singleton); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Updated the Tensor API and glossary to classify the three foundation contracts as implemented
  while keeping <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>[`Operation`](../../../../glossary.md#operation), concrete kinds, and family-specific attributes planned.

## Completion summary

- Completed changes: Implemented the minimal open operation-kind and immutable-attribute
  foundation without concrete operation semantics or cross-layer behavior.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Files changed or created: Three production Java files, two focused test files, Tensor API,
  glossary, this task specification, model master plan, and implementation roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Model tests, generated Javadoc, complete repository tests, scope checks,
  Markdown checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: `/root/review_model_0005_docs` independently completed the required
  clean-context pass in the same overall change.
- Documentation impact: Tensor API now explains the implemented foundation with a clearly labeled
  test-local example and preserves the planned status of [`Operation`](../../../../glossary.md#operation) and concrete families.
- [Javadoc](../../../../glossary.md#term-javadoc) review: Complete for every new public type, `OperationKind.name()`, and
  `NoOperationAttrs.INSTANCE`; no post-review correction was required.
- Glossary impact: [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), and [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) now describe current
  contracts and are distinguished from the planned `Operation` [descriptor](../../../../glossary.md#term-descriptor).
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0005. Plan task 0006 separately before implementation.

Status: Complete
