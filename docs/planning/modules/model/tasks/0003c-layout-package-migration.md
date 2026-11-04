# Task 0003C: [Layout](../../../../glossary.md#layout) Package Migration

**Editorial reading key (later context, not a status update).**

This task moves the types that describe resolved element geometry. Application or internal callers supply a Shape and geometry; the layout package checks the description, but does not allocate or close memory. Its package-private helper is implementation structure, not a new public capability. The move keeps the shape dependency and the original non-negative-stride restriction; the current reference is a navigation destination, not a replacement completion record.

Focused current reading: [Current layout package and geometry](../../../../reference/tensor/layouts.md#resolved-layouts). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Move the completed [layout](../../../../glossary.md#layout) taxonomy, [immutable](../../../../glossary.md#term-immutable) [descriptor](../../../../glossary.md#term-descriptor), package-private geometry helper, and their tests into `io.github.pho001.synaptik.model.layout` without changing logical geometry, [validation](../../../../glossary.md#term-validation), classification, span calculation, or public [signatures](../../../../glossary.md#term-signature) other than package [qualification](../../../../glossary.md#term-qualification).

## [Scope](../../../../glossary.md#term-scope)

- Move `LayoutKind`, `LayoutDescriptor`, and package-private `LayoutGeometry` into the planned [layout](../../../../glossary.md#layout) package.
- Move both layout test classes into the matching package so `LayoutGeometryTest` retains package-private access.
- Preserve the existing [dependency](../../../../glossary.md#term-dependency) from layout to public [shape](../../../../glossary.md#shape) contracts.
- Update affected [Javadoc](../../../../glossary.md#term-javadoc) links and public [API](../../../../glossary.md#term-application-programming-interface-api) documentation.
- Complete the migration of all current production contracts out of the flat [model](../../../../glossary.md#term-model) root package.

## Out of [scope](../../../../glossary.md#term-scope)

- changing [layout](../../../../glossary.md#layout) kinds, [strides](../../../../glossary.md#term-stride), offsets, [view](../../../../glossary.md#term-view-meanings) [metadata](../../../../glossary.md#term-metadata), span arithmetic, or [validation](../../../../glossary.md#term-validation)
- adding negative strides, dynamic numeric layouts, storage, [materialization](../../../../glossary.md#materialization), or layout operations
- adding root-package compatibility wrappers
- moving [data type](../../../../glossary.md#data-type--datatype) or [shape](../../../../glossary.md#shape) contracts again
- introducing identifiers, [tensors](../../../../glossary.md#tensor), operations, [graph](../../../../glossary.md#graph) contracts, [runtime](../../../../glossary.md#term-runtime), [compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), or [backend](../../../../glossary.md#backend) behavior
- changing [Gradle](../../../../glossary.md#term-gradle), Java configuration, [module](../../../../glossary.md#term-module) dependencies, `ARCHITECTURE.md`, or focused architecture documentation

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially the `modules/model` boundary and Java namespace
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Planning guide](../../../planning-guide.md), especially package structure [planning](../../../../glossary.md#term-planning)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md), especially the package map
- [Task 0003](0003-layout-descriptor-model.md), which defines the completed behavior being moved
- [Task 0003B](0003b-shape-package-migration.md), which establishes the imported [shape](../../../../glossary.md#shape) package

## Architecture constraints

- All production packages remain below `io.github.pho001.synaptik.*`.
- The migration stays inside `modules/model` and the explicitly listed documentation files.
- `model.layout` may depend on public `model.shape` contracts and the [JDK](../../../../glossary.md#term-java-development-kit-jdk) only.
- `LayoutGeometry` remains package-private and is not promoted into public [API](../../../../glossary.md#term-application-programming-interface-api).
- [Layout](../../../../glossary.md#layout) remains independent of storage, [materialization](../../../../glossary.md#materialization) policy, [compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), [runtime](../../../../glossary.md#term-runtime), and [backends](../../../../glossary.md#backend).
- If the migration requires behavior, architecture, [Gradle](../../../../glossary.md#term-gradle), or [dependency](../../../../glossary.md#term-dependency) changes, stop and report the issue.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.model` no longer owns [layout](../../../../glossary.md#layout) contracts after this migration and contains no current production type.

Package added:

- `io.github.pho001.synaptik.model.layout` owns [backend](../../../../glossary.md#backend)-independent resolved logical layout geometry.

Type placement:

- `io.github.pho001.synaptik.model.layout.LayoutKind` — public geometry classification.
- `io.github.pho001.synaptik.model.layout.LayoutDescriptor` — public [immutable](../../../../glossary.md#term-immutable) resolved-layout value.
- `io.github.pho001.synaptik.model.layout.LayoutGeometry` — package-private checked geometry implementation used by the [descriptor](../../../../glossary.md#term-descriptor).

Test placement:

- `LayoutDescriptorTest` and `LayoutGeometryTest` move to `model.layout`; the latter remains a white-box package test for the package-private helper.

## Affected files

Expected production moves:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/LayoutKind.java` to `modules/model/src/main/java/io/github/pho001/synaptik/model/layout/LayoutKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/LayoutDescriptor.java` to `modules/model/src/main/java/io/github/pho001/synaptik/model/layout/LayoutDescriptor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/LayoutGeometry.java` to `modules/model/src/main/java/io/github/pho001/synaptik/model/layout/LayoutGeometry.java`

Expected test moves:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/LayoutDescriptorTest.java` to `modules/model/src/test/java/io/github/pho001/synaptik/model/layout/LayoutDescriptorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/LayoutGeometryTest.java` to `modules/model/src/test/java/io/github/pho001/synaptik/model/layout/LayoutGeometryTest.java`

Expected documentation/[planning](../../../../glossary.md#term-planning) updates:

- `docs/api/tensor-api.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may move and update at most:

- three existing production Java files;
- two existing test Java files; and
- the four documentation/[planning](../../../../glossary.md#term-planning) files listed above.

Do not add Java types or modify method bodies. If a consumer outside these files requires an update, stop and revise the task before expanding scope.

## Acceptance criteria

- All three completed [layout](../../../../glossary.md#layout) types use `io.github.pho001.synaptik.model.layout`.
- Both layout tests use the matching test package and retain all assertions.
- `LayoutGeometry` remains package-private and accessible to its package test.
- No production type or compatibility wrapper remains directly in `io.github.pho001.synaptik.model`.
- Layout imports `model.shape.Shape`; [shape](../../../../glossary.md#shape) does not import layout.
- Layout kinds, [canonical](../../../../glossary.md#term-canonical-form) [strides](../../../../glossary.md#term-stride), classification, offsets, [view](../../../../glossary.md#term-view-meanings) [metadata](../../../../glossary.md#term-metadata), span arithmetic, [validation](../../../../glossary.md#term-validation), equality, hashing, and diagnostics are unchanged.
- Public and package-private [Javadoc](../../../../glossary.md#term-javadoc) remains complete and resolves in the new package.
- The [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) reference identifies the public layout package.
- No [data type](../../../../glossary.md#data-type--datatype), shape, storage, [operation](../../../../glossary.md#operation), [graph](../../../../glossary.md#graph), [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), [backend](../../../../glossary.md#backend), [Gradle](../../../../glossary.md#term-gradle), or architecture behavior changes.
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

- exactly three production and two test files moved;
- no production Java file remains in the flat [model](../../../../glossary.md#term-model) root package;
- no method or test body changed;
- generated [Javadoc](../../../../glossary.md#term-javadoc) contains `model.layout.LayoutKind` and `model.layout.LayoutDescriptor`;
- `LayoutGeometry` is not public;
- imports preserve the acyclic `layout -> shape` [dependency](../../../../glossary.md#term-dependency); and
- task, master plan, roadmap, and [API](../../../../glossary.md#term-application-programming-interface-api) documentation agree on package and status.

## Dependencies

- Task 0003: [Layout](../../../../glossary.md#layout) [descriptor](../../../../glossary.md#term-descriptor) [model](../../../../glossary.md#term-model) — complete.
- Task 0003B: [Shape](../../../../glossary.md#shape) package migration — complete.
- The model package map in the master plan — defined.

## Follow-up tasks

- Task 0004: Typed identifiers introduces new contracts in their owning `model.tensor` and `model.graph` packages.
- Task 0007: [Tensor descriptor](../../../../glossary.md#tensor-descriptor) composes the migrated [data type](../../../../glossary.md#data-type--datatype), [shape](../../../../glossary.md#shape), and [layout](../../../../glossary.md#layout) values.
- Task 0017: Layout and [view](../../../../glossary.md#term-view-meanings) operations consume the migrated contracts.

Do not create the detailed task 0004 specification until this task is complete.

## Architecture impact

Expected impact: None.

This task completes the planned package organization for existing [model](../../../../glossary.md#term-model) foundations without changing [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), or behavior.

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
- docs/planning/modules/model/tasks/0003c-layout-package-migration.md

Implement task 0003C exactly as specified.

Move only the completed layout contracts, package-private helper, and tests into io.github.pho001.synaptik.model.layout. Preserve all behavior and public signatures other than package qualification. Keep LayoutGeometry package-private. Do not add compatibility wrappers or change data types, shape behavior, layout behavior, storage, operations, tensors, graph contracts, dependencies, Gradle, ARCHITECTURE.md, focused architecture documentation, or another module.

Review and preserve complete Javadoc for every moved public contract and the package-private helper. Run every validation command from the task specification.

At the end, update this task file with local decisions, known limitations, validation evidence, implementation notes, completion summary, and final status. Synchronize the task status in the model master plan and implementation roadmap.
```

## Local decisions

- All three [layout](../../../../glossary.md#layout) production types moved together so package-private `LayoutGeometry` remains directly available to `LayoutDescriptor`.
- Both tests moved into the matching layout package, preserving white-box coverage of the package-private helper without [widening](../../../../glossary.md#term-widening) visibility.
- The existing explicit import of `model.shape.Shape` was retained, making the intended package direction visible.

## Known limitations

- Numeric [layouts](../../../../glossary.md#layout) still require fully static [shapes](../../../../glossary.md#shape), and negative [strides](../../../../glossary.md#term-stride) remain unsupported as defined by task 0003.
- Source consumers must use the new layout package; compatibility wrappers for unreleased root-package names are intentionally absent.

## [Validation](../../../../glossary.md#term-validation) evidence

- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew :modules:model:test` — passed with 56 tests, zero failures, zero errors, and zero skipped tests.
- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew :modules:model:javadoc` — passed and generated `LayoutKind` and `LayoutDescriptor` under `io/github/pho001/synaptik/model/layout`.
- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew test` — passed for the complete multi-[module](../../../../glossary.md#term-module) repository.
- `git diff --check` — passed after implementation and final [planning](../../../../glossary.md#term-planning) [synchronization](../../../../glossary.md#term-synchronization).
- Manual [scope](../../../../glossary.md#term-scope) review confirmed exactly three production and two test files moved, with no method or test body changes.
- Manual visibility review confirmed `LayoutGeometry` remains package-private and its package test still compiles.
- Manual package review confirmed no production Java file remains directly in the flat [model](../../../../glossary.md#term-model) root and [layout](../../../../glossary.md#layout) depends only on [shape](../../../../glossary.md#shape) plus [JDK](../../../../glossary.md#term-java-development-kit-jdk) classes.
- No [Gradle](../../../../glossary.md#term-gradle), architecture, other-module, [data type](../../../../glossary.md#data-type--datatype), shape behavior, or layout behavior change was made.
- Gradle emitted a non-fatal filesystem-watching warning in the sandbox; every requested task completed successfully.

## Implementation notes

- Moved the complete [layout](../../../../glossary.md#layout) [model](../../../../glossary.md#term-model) into `io.github.pho001.synaptik.model.layout`.
- Moved both layout tests into the matching test package.
- Updated the [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) reference with the public layout package.
- Completed the foundational package migration and advanced the [planning](../../../../glossary.md#term-planning) frontier to task 0004.

## Completion summary

- Completed changes: Migrated the complete [layout](../../../../glossary.md#layout) foundation into its cohesive package without semantic or visibility changes.
- Files changed or created: Three production moves, two test moves, the [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) reference, this task, the [model](../../../../glossary.md#term-model) master plan, and the roadmap.
- Tests and [validation](../../../../glossary.md#term-validation): Model tests, model [Javadoc](../../../../glossary.md#term-javadoc), full repository tests, diff checks, visibility checks, package checks, [scope](../../../../glossary.md#term-scope) review, and behavior review passed.
- Documentation impact: The public API reference now identifies `io.github.pho001.synaptik.model.layout`; no architecture documentation change was required.
- Javadoc review: Public Javadoc and the documented package-private helper contract remain accurate after the move.
- Unresolved issues: None.
- Follow-up required: Plan task 0004 before introducing typed identifiers.

Status: Complete
