# Task 0003B: [Shape](../../../../glossary.md#shape) Package Migration

**Editorial reading key (later context, not a status update).**

This migration moves shape descriptions and their local algebra, not tensor values. Layout code is the consumer of Shape: it needs axis extents to describe geometry, while Shape itself does not need storage positions. Thus the stated layout-to-shape dependency describes code responsibility, not resource ownership. Updating imports preserves this direction and does not create runtime dimension binding or graph-wide inference.

Focused current reading: [Current Shape contracts](../../../../reference/tensor/shapes.md#shapes-and-dimensions); [Current layout consumer](../../../../reference/tensor/layouts.md#resolved-layouts). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Move the completed [dimension](../../../../glossary.md#dimension), [shape](../../../../glossary.md#shape), and local [broadcasting](../../../../glossary.md#broadcasting) contracts into `io.github.pho001.synaptik.model.shape`, update the existing [layout](../../../../glossary.md#layout) [model](../../../../glossary.md#term-model) to import those contracts, and preserve all behavior and public [signatures](../../../../glossary.md#term-signature) other than package [qualification](../../../../glossary.md#term-qualification).

## [Scope](../../../../glossary.md#term-scope)

- Move [`Dimension`](../../../../glossary.md#dimension), `StaticDimension`, `DynamicDimension`, [`Shape`](../../../../glossary.md#shape), and `ShapeBroadcast` into the planned shape package.
- Move their three unit-test classes into the matching test package.
- Add shape imports to the existing [layout](../../../../glossary.md#layout) production and test files that remain in the root package until task 0003C.
- Update affected [Javadoc](../../../../glossary.md#term-javadoc) links and public [API](../../../../glossary.md#term-application-programming-interface-api) documentation.
- Preserve all shape, dimension, [axis](../../../../glossary.md#term-axis), [element](../../../../glossary.md#term-element)-count, [broadcasting](../../../../glossary.md#broadcasting), [validation](../../../../glossary.md#term-validation), equality, and diagnostic behavior.

## Out of [scope](../../../../glossary.md#term-scope)

- changing [dimension](../../../../glossary.md#dimension) or [shape](../../../../glossary.md#shape) semantics
- adding bounds, constraints, [runtime](../../../../glossary.md#term-runtime) [bindings](../../../../glossary.md#term-binding), reshape [inference](../../../../glossary.md#term-inference), or [graph](../../../../glossary.md#graph)-wide shape inference
- moving or changing [layout](../../../../glossary.md#layout) contracts beyond required import statements
- adding root-package compatibility wrappers
- moving [data type](../../../../glossary.md#data-type--datatype) contracts or introducing new [model](../../../../glossary.md#term-model) contracts
- changing dependencies, [Gradle](../../../../glossary.md#term-gradle) files, Java configuration, architecture documentation, or another [module](../../../../glossary.md#term-module)

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially the `modules/model` boundary and Java namespace
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Planning guide](../../../planning-guide.md), especially package structure [planning](../../../../glossary.md#term-planning)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md), especially the package map
- [Task 0002](0002-shape-and-dimension-model.md), which defines the completed behavior being moved
- [Task 0003](0003-layout-descriptor-model.md), whose [layout](../../../../glossary.md#layout) implementation consumes [`Shape`](../../../../glossary.md#shape)

## Architecture constraints

- All production packages remain below `io.github.pho001.synaptik.*`.
- The migration stays inside `modules/model` and the explicitly listed documentation files.
- The [shape](../../../../glossary.md#shape) package remains [backend](../../../../glossary.md#backend)-, storage-, [compiler](../../../../glossary.md#term-compiler)-, [planning](../../../../glossary.md#term-planning)-, [runtime](../../../../glossary.md#term-runtime)-, and execution-independent.
- `model.shape` uses only the [JDK](../../../../glossary.md#term-java-development-kit-jdk) and its own contracts.
- Root-package layout contracts may depend on `model.shape` during this intermediate migration; `model.shape` must not depend on [layout](../../../../glossary.md#layout).
- If the migration requires behavior, architecture, [Gradle](../../../../glossary.md#term-gradle), or [dependency](../../../../glossary.md#term-dependency) changes, stop and report the issue.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.model` no longer owns [dimension](../../../../glossary.md#dimension), [shape](../../../../glossary.md#shape), or local [broadcasting](../../../../glossary.md#broadcasting) contracts after this migration.

Package added:

- `io.github.pho001.synaptik.model.shape` owns dimension values, [immutable](../../../../glossary.md#term-immutable) shapes, [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization), checked [element](../../../../glossary.md#term-element) counts, and local broadcasting.

Type placement:

- `io.github.pho001.synaptik.model.shape.Dimension` — public [sealed](../../../../glossary.md#term-sealed) dimension contract.
- `io.github.pho001.synaptik.model.shape.StaticDimension` — public known-size dimension value.
- `io.github.pho001.synaptik.model.shape.DynamicDimension` — public symbolic dimension value.
- `io.github.pho001.synaptik.model.shape.Shape` — public immutable shape value.
- `io.github.pho001.synaptik.model.shape.ShapeBroadcast` — public deterministic local broadcasting utility.

Test placement:

- `DimensionTest`, `ShapeTest`, and `ShapeBroadcastTest` move to the matching `model.shape` test package.
- [Layout](../../../../glossary.md#layout) tests remain in the root [model](../../../../glossary.md#term-model) test package and import public shape contracts until task 0003C.

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

Expected documentation/[planning](../../../../glossary.md#term-planning) updates:

- `docs/api/tensor-api.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may move or update at most:

- five [shape](../../../../glossary.md#shape) production files and three shape test files;
- two [layout](../../../../glossary.md#layout) production files and two layout test files for imports only; and
- the four documentation/[planning](../../../../glossary.md#term-planning) files listed above.

The temporary layout imports are required to keep every sequential frontier buildable. Do not change layout statements, tests, or semantics. If another Java file needs more than an import or [Javadoc](../../../../glossary.md#term-javadoc) reference update, stop and revise the plan.

## Acceptance criteria

- All five completed [shape](../../../../glossary.md#shape) contracts use `io.github.pho001.synaptik.model.shape`.
- The three shape test classes use the matching package.
- No [dimension](../../../../glossary.md#dimension), shape, [broadcasting](../../../../glossary.md#broadcasting) contract, or compatibility wrapper remains in the root package.
- [Layout](../../../../glossary.md#layout) production and test files [compile](../../../../glossary.md#compile) through explicit imports without changing behavior.
- Shape, dimension, broadcasting, [element](../../../../glossary.md#term-element)-count, [axis](../../../../glossary.md#term-axis), equality, and [validation](../../../../glossary.md#term-validation) behavior is unchanged.
- Public [Javadoc](../../../../glossary.md#term-javadoc) remains complete and resolves under the new package.
- The [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) reference identifies the public shape package.
- `model.shape` imports only [JDK](../../../../glossary.md#term-java-development-kit-jdk) classes and its own contracts.
- No [data type](../../../../glossary.md#data-type--datatype), layout behavior, storage, [graph](../../../../glossary.md#graph), [compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), [runtime](../../../../glossary.md#term-runtime), [backend](../../../../glossary.md#backend), [Gradle](../../../../glossary.md#term-gradle), or [architecture contract](../../../../glossary.md#architecture-contract) changes.
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

- exactly five production and three test contracts moved into `model.shape`;
- exactly four [layout](../../../../glossary.md#layout) files received import-only changes;
- no old root-package [shape](../../../../glossary.md#shape) type or compatibility wrapper remains;
- test method bodies and production behavior statements are unchanged;
- generated [Javadoc](../../../../glossary.md#term-javadoc) contains the new shape package and all public contracts;
- package dependencies follow `layout -> shape`, never `shape -> layout`; and
- task, master plan, roadmap, and [API](../../../../glossary.md#term-application-programming-interface-api) documentation agree on package and status.

## Dependencies

- Task 0002: [Shape](../../../../glossary.md#shape) and [dimension](../../../../glossary.md#dimension) [model](../../../../glossary.md#term-model) — complete.
- Task 0003A: [Data type](../../../../glossary.md#data-type--datatype) package migration — complete by ordered frontier.
- The model package map in the master plan — defined.

## Follow-up tasks

- Task 0003C: [Layout](../../../../glossary.md#layout) package migration moves the remaining layout contracts and tests into `model.layout`.
- Task 0007: [Tensor descriptor](../../../../glossary.md#tensor-descriptor) composes the migrated [data type](../../../../glossary.md#data-type--datatype), [shape](../../../../glossary.md#shape), and layout values.
- Task 0017: Layout and [view](../../../../glossary.md#term-view-meanings) operations consume the migrated shape and layout contracts.

Do not create the detailed task 0003C specification until this task is complete.

## Architecture impact

Expected impact: None.

This task changes package organization within the existing [model module](../../../../glossary.md#model-module-buffer-and-scan-distinctions) and makes the already planned `layout -> shape` package direction explicit. It does not change [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings) or architecture dependencies.

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

- The five [shape](../../../../glossary.md#shape) contracts moved together so [sealed](../../../../glossary.md#term-sealed)-type permits clauses, package-local collaboration, and [Javadoc](../../../../glossary.md#term-javadoc) links remain direct.
- Existing [layout](../../../../glossary.md#layout) contracts received explicit imports instead of temporary root-package wrappers, preserving the intended `layout -> shape` direction.
- Shape tests moved with their production contracts; layout tests remained in place with import-only updates until task 0003C.

## Known limitations

- [Layout](../../../../glossary.md#layout) contracts and tests remain in the root [model](../../../../glossary.md#term-model) package until task 0003C.
- Source consumers must use the new [shape](../../../../glossary.md#shape) package; compatibility shims for the unreleased root-package names are intentionally absent.

## [Validation](../../../../glossary.md#term-validation) evidence

- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew :modules:model:test` — passed with 56 tests, zero failures, zero errors, and zero skipped tests.
- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew :modules:model:javadoc` — passed and generated all five public contracts under `io/github/pho001/synaptik/model/shape`.
- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew test` — passed for the complete multi-[module](../../../../glossary.md#term-module) repository.
- `git diff --check` — passed after implementation and [planning](../../../../glossary.md#term-planning) [synchronization](../../../../glossary.md#term-synchronization).
- Manual [scope](../../../../glossary.md#term-scope) review confirmed five production and three test moves plus import-only changes in two [layout](../../../../glossary.md#layout) production and two layout test files.
- Manual [dependency](../../../../glossary.md#term-dependency) review confirmed `model.shape` imports only `java.*`; layout imports [shape](../../../../glossary.md#shape) and shape does not import layout.
- Manual behavior review confirmed production and test method bodies are unchanged.
- No [Gradle](../../../../glossary.md#term-gradle), architecture, other-module, [data type](../../../../glossary.md#data-type--datatype), or layout-behavior change was made.
- Gradle emitted a non-fatal filesystem-watching warning in the sandbox; every requested task completed successfully.

## Implementation notes

- Moved the [sealed](../../../../glossary.md#term-sealed) [dimension](../../../../glossary.md#dimension) hierarchy, [`Shape`](../../../../glossary.md#shape), and `ShapeBroadcast` into `io.github.pho001.synaptik.model.shape`.
- Moved all three shape test classes into the corresponding test package.
- Added explicit public shape imports to the four [layout](../../../../glossary.md#layout) files that consume shape contracts.
- Updated the [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) reference and advanced the [planning](../../../../glossary.md#term-planning) frontier to task 0003C.

## Completion summary

- Completed changes: Migrated the complete [shape](../../../../glossary.md#shape) foundation into its cohesive package and preserved a buildable intermediate [layout](../../../../glossary.md#layout) [dependency](../../../../glossary.md#term-dependency).
- Files changed or created: Five production moves, three test moves, four import-only layout updates, the [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) reference, this task, the [model](../../../../glossary.md#term-model) master plan, and the roadmap.
- Tests and [validation](../../../../glossary.md#term-validation): Model tests, model [Javadoc](../../../../glossary.md#term-javadoc), full repository tests, diff checks, package-dependency checks, [scope](../../../../glossary.md#term-scope) review, and behavior review passed.
- Documentation impact: The public API reference now identifies `io.github.pho001.synaptik.model.shape`; no architecture documentation change was required.
- Javadoc review: Existing detailed Javadoc remains accurate and resolves in the new package.
- Unresolved issues: None.
- Follow-up required: Plan and implement task 0003C.

Status: Complete
