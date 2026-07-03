# Task 0003A: Data Type Package Migration

## Status

Complete

## Goal

Move the completed data type contracts and their tests from the flat module root package into `io.github.pho001.synaptik.model.datatype` without changing public behavior, supported data types, validation, conversion semantics, or module dependencies.

## Scope

- Move `DataType`, `DataTypeCategory`, `DataTypePromotion`, and `BFloat16Bits` into the planned `model.datatype` package.
- Move their unit tests into the matching test package.
- Update package declarations, imports, Javadoc links, and public API documentation affected by the new fully qualified names.
- Preserve all existing public signatures other than their package qualification.
- Preserve the existing test inventory and behavioral assertions.

## Out of scope

- adding, removing, or renaming data types
- changing promotion, differentiability, byte-width, or BFLOAT16 conversion behavior
- adding compatibility shims in the old root package
- moving shape, dimension, or layout contracts
- introducing typed identifiers, operations, tensors, storage, graph contracts, or backend mappings
- changing module dependencies, Gradle files, Java versions, or preview-feature configuration
- changing `ARCHITECTURE.md` or focused architecture documentation

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially the `modules/model` boundary and Java namespace
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Planning guide](../../../planning-guide.md), especially package structure planning
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md), especially the package map
- [Task 0001](0001-data-type-model.md), which defines the completed behavior being moved

## Architecture constraints

- All production packages remain below `io.github.pho001.synaptik.*`.
- The migration stays entirely inside `modules/model` and the explicitly listed documentation files.
- The data type package remains backend-, runtime-, storage-, compiler-, and planning-independent.
- Production imports remain limited to the JDK and contracts in `model.datatype`.
- The legacy branch is read-only capability evidence and is not a package-layout source.
- If the migration requires an architecture or Gradle change, stop and report the issue.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.model` no longer owns `DataType`, `DataTypeCategory`, `DataTypePromotion`, or `BFloat16Bits` after this migration.

Package added:

- `io.github.pho001.synaptik.model.datatype` owns backend-independent data type metadata, floating promotion, and BFLOAT16 bit conversion.

Type placement:

- `io.github.pho001.synaptik.model.datatype.DataType` — public data type vocabulary and metadata.
- `io.github.pho001.synaptik.model.datatype.DataTypeCategory` — public category vocabulary used by `DataType`.
- `io.github.pho001.synaptik.model.datatype.DataTypePromotion` — public backend-independent floating promotion rules.
- `io.github.pho001.synaptik.model.datatype.BFloat16Bits` — public host-independent scalar BFLOAT16 bit conversion.

Test placement:

- Tests move to `io.github.pho001.synaptik.model.datatype` so they mirror the production package and continue to validate the same public contracts.

## Affected files

Expected production moves:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/DataType.java` to `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataType.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/DataTypeCategory.java` to `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataTypeCategory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/DataTypePromotion.java` to `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataTypePromotion.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/BFloat16Bits.java` to `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/BFloat16Bits.java`

Expected test moves:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/DataTypeTest.java` to `modules/model/src/test/java/io/github/pho001/synaptik/model/datatype/DataTypeTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/DataTypePromotionTest.java` to `modules/model/src/test/java/io/github/pho001/synaptik/model/datatype/DataTypePromotionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/BFloat16BitsTest.java` to `modules/model/src/test/java/io/github/pho001/synaptik/model/datatype/BFloat16BitsTest.java`

Expected documentation/planning updates:

- `docs/api/tensor-api.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may move and update at most:

- four existing production Java files;
- three existing test Java files; and
- the four documentation/planning files listed above.

The seven Java changes are an atomic package migration of one already completed concept. Do not add production or test types. If another Java file requires a semantic change rather than an import update, stop and propose a follow-up task.

## Acceptance criteria

- All four completed data type contracts use `io.github.pho001.synaptik.model.datatype`.
- All three corresponding tests use the matching `model.datatype` package.
- No data type contract or compatibility wrapper remains in the module root package.
- Public behavior, enum constants, metadata, promotion rules, BFLOAT16 conversion, signatures, equality, failures, and test assertions are unchanged.
- Every affected Javadoc remains complete and resolves links under the new package.
- `docs/api/tensor-api.md` identifies the public data type package without changing documented semantics.
- Production imports remain limited to `java.*` and the new local data type package.
- No shape, dimension, layout, storage, tensor, operation, graph, runtime, compiler, planning, prepare, engine, or backend behavior is changed.
- No Gradle file, `ARCHITECTURE.md`, focused architecture document, or other module is changed.
- The task, model master-plan row, and roadmap row have matching final statuses.

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

- the old root package contains none of the four migrated production types;
- exactly four production and three test files moved;
- Git recognizes the changes as moves where possible;
- no behavioral Java statement changed except package declarations, imports, and Javadoc references;
- no production dependency outside the JDK and `modules/model` was introduced;
- generated Javadoc contains the new `model.datatype` package and all four public contracts; and
- task, master-plan, roadmap, and API documentation agree on the package and status.

## Dependencies

- Task 0001: DataType model — complete.
- The model package map in the master plan — defined.

## Follow-up tasks

- Task 0003B: Shape package migration.
- Task 0005: Operation semantic foundation consumes the migrated data type contracts.
- Task 0010: Host storage abstraction consumes the migrated data type contracts.

Do not create the detailed task 0003B specification until this task is complete.

## Architecture impact

Expected impact: None.

This task changes Java package organization within the existing `modules/model` ownership boundary. It does not change module ownership, dependency direction, or model semantics. If implementation requires an architecture change, stop and report the issue.

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
- docs/planning/modules/model/tasks/0003a-data-type-package-migration.md

Implement task 0003A exactly as specified.

Move only the completed data type contracts and their tests into io.github.pho001.synaptik.model.datatype. Preserve behavior and public signatures other than package qualification. Do not add compatibility wrappers, new data types, storage, operations, tensors, graph contracts, dependencies, or behavioral changes. Do not modify shape, dimension, layout, Gradle, ARCHITECTURE.md, focused architecture documentation, or another module.

Review and preserve complete Javadoc for every moved public Java contract. Run every validation command from the task specification.

At the end, update this task file with local decisions, known limitations, validation evidence, implementation notes, completion summary, and final status. Synchronize the task status in the model master plan and implementation roadmap.
```

## Local decisions

- All four production contracts moved together so their intra-package references remain unchanged and no transitional import state is published.
- No compatibility types remain in `io.github.pho001.synaptik.model`; the rewrite has no released API that requires package forwarding shims.
- Tests mirror the new production package. Their assertions and test method bodies remain unchanged.

## Known limitations

- This task migrates only data type contracts. Shape and layout contracts remain in the root model package until tasks 0003B and 0003C.
- Source consumers must import the new fully qualified package. Binary and source compatibility with the unreleased root-package location is intentionally not preserved.

## Validation evidence

- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew :modules:model:test` — passed with 56 tests, zero failures, zero errors, and zero skipped tests.
- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew :modules:model:javadoc` — passed; generated output contains `io/github/pho001/synaptik/model/datatype/DataType.html` and the other migrated public contracts.
- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew test` — passed for the complete multi-module repository.
- `git diff --check` — passed after the implementation and planning updates.
- Manual scope review confirmed exactly four production and three test files moved, with no added Java type or compatibility wrapper.
- Manual source review confirmed that Java changes are limited to package declarations and paths; production imports remain limited to `java.util.Objects`.
- Manual architecture review confirmed no changes to Gradle, `ARCHITECTURE.md`, focused architecture documentation, other modules, or model behavior.
- Gradle emitted a non-fatal filesystem-watching warning in the sandbox; all requested tasks completed successfully.

## Implementation notes

- Moved `DataType`, `DataTypeCategory`, `DataTypePromotion`, and `BFloat16Bits` into `io.github.pho001.synaptik.model.datatype`.
- Moved the three existing tests into the matching test package without changing their assertions.
- Updated the Tensor API reference to publish the new data type package.
- Synchronized the task, model master plan, and roadmap status and advanced the planning frontier to task 0003B.

## Completion summary

- Completed changes: Migrated the completed data type model into its planned cohesive package without semantic changes.
- Files changed or created: Four production moves, three test moves, the Tensor API reference, this task, the model master plan, and the roadmap.
- Tests and validation: Model tests, model Javadoc, the full repository test suite, diff checks, source-scope review, dependency review, and documentation review passed.
- Documentation impact: The public API reference now identifies `io.github.pho001.synaptik.model.datatype`; no architecture documentation change was required.
- Javadoc review: Existing detailed Javadoc remains accurate and resolves under the new package; no behavioral contract changed.
- Unresolved issues: None.
- Follow-up required: Plan and implement task 0003B, then task 0003C.

Status: Complete
