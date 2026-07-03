# Task 0005: Operation Semantic Foundation

## Status

Complete

## Goal

Define the minimal extensible contracts for backend-independent operation kinds and immutable typed operation attributes without introducing concrete mathematical operations, a monolithic operation inventory, graph behavior, compiler policy, autograd, fusion, cost estimates, or backend support metadata.

## Scope

- Define a public `OperationKind` semantic discriminator contract.
- Define a public `OperationAttrs` marker for immutable typed attribute values.
- Define a singleton `NoOperationAttrs` value for operations with no semantic parameters.
- Specify stability, immutability, equality, diagnostic-name, and ownership expectations for future kind and attribute implementations.
- Add focused tests and public API documentation.

## Out of scope

- the `Operation` descriptor itself, which belongs to task 0006
- concrete kinds such as add, reduction, reshape, convolution, loss, or compiler-generated operations
- family-specific attribute records, axes, scalars, padding, windows, reductions, or shape parameters
- a complete operation enum or one Java type per legacy operation
- arity validation, input/output descriptors, shape inference, data type inference, broadcasting, or graph construction
- operation execution, kernels, lowering, backend capability, backend ownership, or fallback
- cost classification, fusion eligibility, materialization policy, or result storage
- autograd rules, differentiability policy, or backward graph construction
- `FUSED`, `UNKNOWN`, invalid sentinels, string-keyed attribute maps, or reflective discovery
- changes to existing Java contracts, Gradle, architecture documentation, or another module

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially the `Operation`, `OperationAttrs`, and forbidden backend-support rules
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Model capability baseline](../capabilities.md), especially the operation foundation and public operation inventory
- [Model master plan](../master-plan.md), especially `model.operation` ownership

## Legacy evidence

The read-only legacy `operations.Operation` interface combined a monolithic `OpType` enum with execution-category, semantic-family, computational-cost, result-kind, fusion-eligibility, and diagnostic-expression methods. Concrete operation classes also carried semantic parameters directly, while raw enum values included `FUSED` and `UNKNOWN` sentinels.

The useful evidence is that compilation needs a stable semantic kind and immutable operation parameters. The new design must not copy legacy coupling:

- computational cost belongs to planning profiles and backend capability data;
- concrete fusion belongs to backend prepare;
- backend support never belongs to `Operation` or its kind;
- `UNKNOWN` is not a supported semantic operation;
- family parameters use typed immutable values rather than untyped maps; and
- the complete kind inventory is introduced progressively with the operation-family tasks.

## Architecture constraints

- Production packages remain below `io.github.pho001.synaptik.*`.
- All new contracts live in `io.github.pho001.synaptik.model.operation`.
- Production code uses only the Java language and has no project-module dependency.
- Operation kinds express semantic identity only and expose no backend, planning, runtime, storage, cost, fusion, or execution facts.
- Operation attributes express immutable semantic parameters only and expose no live tensor, service, backend, runtime, or mutable collection state.
- Future family enums or immutable values may implement the open contracts without reflective registration.
- If implementation requires a concrete operation kind, family attribute, `Operation`, dependency, or architecture change, stop and report the issue.

## Package impact

Package added:

- `io.github.pho001.synaptik.model.operation` owns backend-independent operation semantics and immutable attribute contracts.

Type placement:

- `io.github.pho001.synaptik.model.operation.OperationKind` — open semantic-kind contract implemented by future typed family vocabularies.
- `io.github.pho001.synaptik.model.operation.OperationAttrs` — open marker contract implemented by future immutable typed attribute records.
- `io.github.pho001.synaptik.model.operation.NoOperationAttrs` — canonical singleton for kinds with no semantic parameters.

Test placement:

- `OperationKindTest` validates the semantic-name and concrete-type separation contract with test-local sample enums.
- `OperationAttrsTest` validates typed structural attributes with a test-local record and the production empty singleton.

## Required contracts

### `OperationKind`

Create a public interface with one method:

```java
String name();
```

The contract requires implementations to be immutable values with stable structural equality. `name()` returns a non-null, non-blank semantic name suitable for diagnostics and deterministic model inspection. It is not a serialization, backend dispatch, kernel route, or reflective class name. Future enum implementations satisfy the method through `Enum.name()`; other implementations must provide equivalent stable behavior.

Kinds with the same textual name but different concrete kind types are not implicitly equivalent. Semantic equivalence uses the typed kind value, not a global string registry.

### `OperationAttrs`

Create a public marker interface with no methods. Implementations must:

- be immutable values;
- defensively isolate mutable constructor inputs;
- provide structural equality and hashing;
- use typed fields and accessors rather than string-keyed maps; and
- contain semantic parameters only.

The marker deliberately does not prescribe family fields before their focused tasks. It must not expose backend support, device handles, runtime state, compiler services, or mutable tensors.

### `NoOperationAttrs`

Create a public enum implementing `OperationAttrs` with exactly one value:

```text
INSTANCE
```

It represents the complete immutable attribute value for an operation kind with no semantic parameters. Later `Operation` construction uses this object rather than `null` or an empty map.

Do not add operation families, arity classes, costs, result kinds, fusion flags, sentinels, registries, factories, or concrete mathematical kinds in this task.

## Affected files

Expected production files:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/NoOperationAttrs.java`

Expected test files:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationKindTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationAttrsTest.java`

Expected documentation/planning updates:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

The capability baseline already records the selected foundation. If implementation evidence requires changing it, stop and report the discrepancy before editing that baseline.

## Maximum scope

This task may create or modify at most:

- three production Java files;
- two test Java files; and
- the five documentation/planning files listed above.

Do not add `Operation`, family types, utilities, registries, or abstraction layers. If another source file is required, stop and propose a follow-up task.

## Acceptance criteria

- `OperationKind` is a public one-method interface with complete semantic-name documentation.
- `OperationAttrs` is a public zero-method marker with explicit immutability and semantic-only requirements.
- `NoOperationAttrs` is a public single-value enum implementing `OperationAttrs`.
- Future enum kinds can implement `OperationKind` through their existing `name()` method.
- Future typed immutable records can implement `OperationAttrs` without maps or reflection.
- No concrete mathematical kind, family attribute, `Operation`, generator, registry, parser, or service is introduced.
- No backend support, cost, fusion, kernel, device, storage, runtime, compiler, planning, autograd, or materialization vocabulary appears in the public API except documentation that explicitly forbids it.
- `FUSED`, `UNKNOWN`, and null-as-empty attributes are not modeled.
- Public types, methods, and enum constants have detailed Javadoc covering semantics, results, immutability, nullability, intended implementations, and exclusions.
- After implementation, a separate documentation-focused agent or thread with clean context independently reviews and finalizes the public Javadoc, Tensor API explanation, and glossary impact in the same overall change, updating the glossary for new or changed project-specific terms when applicable.
- Tests cover interface shape, enum compatibility, stable names, concrete-type separation, typed structural attributes, marker membership, singleton identity, and diagnostic text.
- Production code contains no imports.
- No existing Java source, Gradle file, architecture document, or other module is changed.
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
- `OperationKind` declares only `name()` and `OperationAttrs` declares no methods;
- `NoOperationAttrs` contains exactly `INSTANCE`;
- production files contain no imports;
- no legacy planning/backend metadata or concrete operation inventory was copied;
- generated Javadoc contains all three public contracts;
- a separate clean-context documentation pass followed `docs/developer-guide/documentation-rules.md`, finalized affected Javadoc and explanatory documentation, updated the glossary for new or changed project-specific terms when applicable, and recorded its evidence in this task; and
- task, master plan, roadmap, API documentation, and glossary agree on status and scope.

## Dependencies

- No hard implementation-task dependency.
- Package migrations and the ordered preceding model tasks are complete.
- The `model.operation` package ownership is defined by the master plan.

## Follow-up tasks

- Task 0006: define the immutable backend-independent `Operation` descriptor over `OperationKind` and `OperationAttrs`.
- Tasks 0014–0022: introduce concrete public operation kinds and typed family attributes progressively.
- Task 0023: introduce compiler-generated semantic kinds without adding autograd rules to model.
- Backend-contract and planning tasks later consume semantic kinds through their own declarative capability and profile contracts.

Do not create detailed specifications for these follow-ups until task 0005 is complete and the frontier advances.

## Architecture impact

Expected impact: None.

The architecture already assigns operation semantics and `OperationAttrs` to `modules/model`. The open kind and attribute interfaces, typed empty singleton, and progressive inventory are local implementation decisions that preserve the existing boundary.

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

- `OperationKind` remains an open one-method interface. Enum-based family vocabularies inherit
  `Enum.name()` directly; the foundation adds no adapter, registry, or duplicate name field.
- Name validity, kind immutability, and typed equality are documented implementation obligations.
  The interface does not add runtime validation because it neither constructs nor wraps future
  kind values.
- `OperationAttrs` remains a zero-method marker. Test-local immutable records demonstrate typed
  structural attributes without committing production family fields before their focused tasks.
- `NoOperationAttrs` uses a one-value enum so the empty attribute value has canonical identity,
  stable equality and hashing, and deterministic diagnostic text without allocation.
- Two test-local enum types deliberately use the same constant name to prove that identical name
  text does not collapse distinct typed kind values.

## Known limitations

- The open interfaces cannot enforce non-blank names, immutability, defensive copying, or
  structural equality at runtime; future implementations must satisfy these documented contracts
  and their own focused tests.
- No production mathematical kind or family-specific attribute value exists yet.
- `Operation` is not implemented, so the model does not yet pair a kind with attributes or validate
  that an attribute type belongs to a kind.
- This foundation provides semantic vocabulary only. It performs no graph construction, shape or
  data type inference, execution, backend capability analysis, lowering, fusion, or cost modeling.

## Validation evidence

- Clean implementation context `/root/implement_model_0005` ran
  `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; the model suite contains 71 passing tests,
  including seven new focused tests with zero failures, errors, or skipped tests.
- The implementation context ran `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated
  documentation contains `OperationKind`, `OperationAttrs`, `NoOperationAttrs`, `name()`, and
  `INSTANCE`.
- The implementation context ran `./gradlew test` — `BUILD SUCCESSFUL` for the complete repository.
- The implementation context ran `git diff --check` — passed with no output.
- Manual implementation checks confirmed exactly three new production and two new test files,
  `OperationKind` with only `name()`, zero declared methods on `OperationAttrs`, exactly one
  `NoOperationAttrs` constant, no production imports, and no copied legacy metadata or concrete
  operation inventory.
- Clean documentation context `/root/review_model_0005_docs` applied the API and Javadoc profile,
  General style, and Example format. It independently reviewed all five Java/test files, finalized
  the operation-foundation explanation in `docs/api/tensor-api.md`, and updated the status and
  distinctions of the three new contracts in `docs/glossary.md`.
- The documentation context concluded that the submitted Javadocs already fully covered semantics,
  result nullability, immutability, equality, intended implementations, and exclusions, so no
  Javadoc text change was necessary. It changed no Java declaration or behavior.
- The documentation context ran `./gradlew :modules:model:javadoc` and
  `./gradlew :modules:model:test` — both `BUILD SUCCESSFUL`; all seven new tests passed.
- The documentation context validated 45 local Markdown links and anchors, code fences, trailing
  whitespace, current-versus-planned terminology, generated Javadoc coverage, and
  `git diff --check` — all passed.
- The final coordinating context reran `./gradlew :modules:model:test`,
  `./gradlew :modules:model:javadoc`, and `./gradlew test` after documentation and planning
  synchronization — all reported `BUILD SUCCESSFUL`; a final `git diff --check` passed with no
  output.
- Final `javap` inspection confirmed that `OperationKind` exposes only `String name()`,
  `OperationAttrs` exposes no methods, and `NoOperationAttrs` declares only the `INSTANCE` enum
  constant. A production-source import scan returned no matches.
- Final scope review confirmed that only the three production files, two test files, Tensor API,
  glossary, this task, model master plan, and implementation roadmap changed. `ARCHITECTURE.md`,
  focused architecture documentation, Gradle files, existing Java contracts, and other modules
  remain unchanged.

## Implementation notes

- Added `OperationKind` as the minimal semantic-name contract and documented why its name is
  diagnostic rather than a serialization or dispatch key.
- Added `OperationAttrs` as the typed immutable-attribute role without prescribing family fields.
- Added `NoOperationAttrs.INSTANCE` as the explicit non-null representation of no semantic
  parameters.
- Added focused tests using only test-local sample enums and a record. Tests cover reflection shape,
  inherited enum names, concrete-type separation, structural equality and hashing, marker
  membership, singleton identity, and diagnostic text.
- Updated the Tensor API and glossary to classify the three foundation contracts as implemented
  while keeping `Operation`, concrete kinds, and family-specific attributes planned.

## Completion summary

- Completed changes: Implemented the minimal open operation-kind and immutable-attribute
  foundation without concrete operation semantics or cross-layer behavior.
- Files changed or created: Three production Java files, two focused test files, Tensor API,
  glossary, this task specification, model master plan, and implementation roadmap.
- Tests and validation: Model tests, generated Javadoc, complete repository tests, scope checks,
  Markdown checks, and `git diff --check` passed.
- Documentation-agent review: `/root/review_model_0005_docs` independently completed the required
  clean-context pass in the same overall change.
- Documentation impact: Tensor API now explains the implemented foundation with a clearly labeled
  test-local example and preserves the planned status of `Operation` and concrete families.
- Javadoc review: Complete for every new public type, `OperationKind.name()`, and
  `NoOperationAttrs.INSTANCE`; no post-review correction was required.
- Glossary impact: `OperationKind`, `OperationAttrs`, and `NoOperationAttrs` now describe current
  contracts and are distinguished from the planned `Operation` descriptor.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0005. Plan task 0006 separately before implementation.

Status: Complete
