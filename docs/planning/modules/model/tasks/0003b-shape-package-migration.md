# Task 0003B: Shape Package Migration

## Status

Complete

## Goal

Move the completed dimension, shape, and local broadcasting contracts into `io.github.pho001.synaptik.model.shape`, update the existing layout model to import those contracts, and preserve all behavior and public signatures other than package qualification.

## Scope

- Move `Dimension`, `StaticDimension`, `DynamicDimension`, `Shape`, and `ShapeBroadcast` into the planned shape package.
- Move their three unit-test classes into the matching test package.
- Add shape imports to the existing layout production and test files that remain in the root package until task 0003C.
- Update affected Javadoc links and public API documentation.
- Preserve all shape, dimension, axis, element-count, broadcasting, validation, equality, and diagnostic behavior.

## Out of scope

- changing dimension or shape semantics
- adding bounds, constraints, runtime bindings, reshape inference, or graph-wide shape inference
- moving or changing layout contracts beyond required import statements
- adding root-package compatibility wrappers
- moving data type contracts or introducing new model contracts
- changing dependencies, Gradle files, Java configuration, architecture documentation, or another module

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially the `modules/model` boundary and Java namespace
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Planning guide](../../../planning-guide.md), especially package structure planning
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md), especially the package map
- [Task 0002](0002-shape-and-dimension-model.md), which defines the completed behavior being moved
- [Task 0003](0003-layout-descriptor-model.md), whose layout implementation consumes `Shape`

## Architecture constraints

- All production packages remain below `io.github.pho001.synaptik.*`.
- The migration stays inside `modules/model` and the explicitly listed documentation files.
- The shape package remains backend-, storage-, compiler-, planning-, runtime-, and execution-independent.
- `model.shape` uses only the JDK and its own contracts.
- Root-package layout contracts may depend on `model.shape` during this intermediate migration; `model.shape` must not depend on layout.
- If the migration requires behavior, architecture, Gradle, or dependency changes, stop and report the issue.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.model` no longer owns dimension, shape, or local broadcasting contracts after this migration.

Package added:

- `io.github.pho001.synaptik.model.shape` owns dimension values, immutable shapes, axis normalization, checked element counts, and local broadcasting.

Type placement:

- `io.github.pho001.synaptik.model.shape.Dimension` — public sealed dimension contract.
- `io.github.pho001.synaptik.model.shape.StaticDimension` — public known-size dimension value.
- `io.github.pho001.synaptik.model.shape.DynamicDimension` — public symbolic dimension value.
- `io.github.pho001.synaptik.model.shape.Shape` — public immutable shape value.
- `io.github.pho001.synaptik.model.shape.ShapeBroadcast` — public deterministic local broadcasting utility.

Test placement:

- `DimensionTest`, `ShapeTest`, and `ShapeBroadcastTest` move to the matching `model.shape` test package.
- Layout tests remain in the root model test package and import public shape contracts until task 0003C.

## Affected files

Expected production moves:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/Dimension.java` to `modules/model/src/main/java/io/github/pho001/synaptik/model/shape/Dimension.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/StaticDimension.java` to `modules/model/src/main/java/io/github/pho001/synaptik/model/shape/StaticDimension.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/DynamicDimension.java` to `modules/model/src/main/java/io/github/pho001/synaptik/model/shape/DynamicDimension.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/Shape.java` to `modules/model/src/main/java/io/github/pho001/synaptik/model/shape/Shape.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/ShapeBroadcast.java` to `modules/model/src/main/java/io/github/pho001/synaptik/model/shape/ShapeBroadcast.java`

Expected test moves:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/DimensionTest.java` to `modules/model/src/test/java/io/github/pho001/synaptik/model/shape/DimensionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/ShapeTest.java` to `modules/model/src/test/java/io/github/pho001/synaptik/model/shape/ShapeTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/ShapeBroadcastTest.java` to `modules/model/src/test/java/io/github/pho001/synaptik/model/shape/ShapeBroadcastTest.java`

Expected import-only Java updates:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/LayoutDescriptor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/LayoutGeometry.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/LayoutDescriptorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/LayoutGeometryTest.java`

Expected documentation/planning updates:

- `docs/api/tensor-api.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may move or update at most:

- five shape production files and three shape test files;
- two layout production files and two layout test files for imports only; and
- the four documentation/planning files listed above.

The temporary layout imports are required to keep every sequential frontier buildable. Do not change layout statements, tests, or semantics. If another Java file needs more than an import or Javadoc reference update, stop and revise the plan.

## Acceptance criteria

- All five completed shape contracts use `io.github.pho001.synaptik.model.shape`.
- The three shape test classes use the matching package.
- No dimension, shape, broadcasting contract, or compatibility wrapper remains in the root package.
- Layout production and test files compile through explicit imports without changing behavior.
- Shape, dimension, broadcasting, element-count, axis, equality, and validation behavior is unchanged.
- Public Javadoc remains complete and resolves under the new package.
- The Tensor API reference identifies the public shape package.
- `model.shape` imports only JDK classes and its own contracts.
- No data type, layout behavior, storage, graph, compiler, planning, runtime, backend, Gradle, or architecture contract changes.
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

- exactly five production and three test contracts moved into `model.shape`;
- exactly four layout files received import-only changes;
- no old root-package shape type or compatibility wrapper remains;
- test method bodies and production behavior statements are unchanged;
- generated Javadoc contains the new shape package and all public contracts;
- package dependencies follow `layout -> shape`, never `shape -> layout`; and
- task, master plan, roadmap, and API documentation agree on package and status.

## Dependencies

- Task 0002: Shape and dimension model — complete.
- Task 0003A: Data type package migration — complete by ordered frontier.
- The model package map in the master plan — defined.

## Follow-up tasks

- Task 0003C: Layout package migration moves the remaining layout contracts and tests into `model.layout`.
- Task 0007: Tensor descriptor composes the migrated data type, shape, and layout values.
- Task 0017: Layout and view operations consume the migrated shape and layout contracts.

Do not create the detailed task 0003C specification until this task is complete.

## Architecture impact

Expected impact: None.

This task changes package organization within the existing model module and makes the already planned `layout -> shape` package direction explicit. It does not change module ownership or architecture dependencies.

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
- docs/planning/modules/model/tasks/0003b-shape-package-migration.md

Implement task 0003B exactly as specified.

Move only the completed dimension, shape, broadcasting contracts and their tests into io.github.pho001.synaptik.model.shape. Add only the required shape imports to existing layout production and test files. Preserve all behavior and public signatures other than package qualification. Do not add compatibility wrappers or change data types, layout behavior, storage, operations, tensors, graph contracts, dependencies, Gradle, ARCHITECTURE.md, focused architecture documentation, or another module.

Review and preserve complete Javadoc for every moved public contract. Run every validation command from the task specification.

At the end, update this task file with local decisions, known limitations, validation evidence, implementation notes, completion summary, and final status. Synchronize the task status in the model master plan and implementation roadmap.
```

## Local decisions

- The five shape contracts moved together so sealed-type permits clauses, package-local collaboration, and Javadoc links remain direct.
- Existing layout contracts received explicit imports instead of temporary root-package wrappers, preserving the intended `layout -> shape` direction.
- Shape tests moved with their production contracts; layout tests remained in place with import-only updates until task 0003C.

## Known limitations

- Layout contracts and tests remain in the root model package until task 0003C.
- Source consumers must use the new shape package; compatibility shims for the unreleased root-package names are intentionally absent.

## Validation evidence

- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew :modules:model:test` — passed with 56 tests, zero failures, zero errors, and zero skipped tests.
- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew :modules:model:javadoc` — passed and generated all five public contracts under `io/github/pho001/synaptik/model/shape`.
- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew test` — passed for the complete multi-module repository.
- `git diff --check` — passed after implementation and planning synchronization.
- Manual scope review confirmed five production and three test moves plus import-only changes in two layout production and two layout test files.
- Manual dependency review confirmed `model.shape` imports only `java.*`; layout imports shape and shape does not import layout.
- Manual behavior review confirmed production and test method bodies are unchanged.
- No Gradle, architecture, other-module, data type, or layout-behavior change was made.
- Gradle emitted a non-fatal filesystem-watching warning in the sandbox; every requested task completed successfully.

## Implementation notes

- Moved the sealed dimension hierarchy, `Shape`, and `ShapeBroadcast` into `io.github.pho001.synaptik.model.shape`.
- Moved all three shape test classes into the corresponding test package.
- Added explicit public shape imports to the four layout files that consume shape contracts.
- Updated the Tensor API reference and advanced the planning frontier to task 0003C.

## Completion summary

- Completed changes: Migrated the complete shape foundation into its cohesive package and preserved a buildable intermediate layout dependency.
- Files changed or created: Five production moves, three test moves, four import-only layout updates, the Tensor API reference, this task, the model master plan, and the roadmap.
- Tests and validation: Model tests, model Javadoc, full repository tests, diff checks, package-dependency checks, scope review, and behavior review passed.
- Documentation impact: The public API reference now identifies `io.github.pho001.synaptik.model.shape`; no architecture documentation change was required.
- Javadoc review: Existing detailed Javadoc remains accurate and resolves in the new package.
- Unresolved issues: None.
- Follow-up required: Plan and implement task 0003C.

Status: Complete
