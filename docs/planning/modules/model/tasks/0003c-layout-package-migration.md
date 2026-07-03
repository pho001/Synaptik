# Task 0003C: Layout Package Migration

## Status

Complete

## Goal

Move the completed layout taxonomy, immutable descriptor, package-private geometry helper, and their tests into `io.github.pho001.synaptik.model.layout` without changing logical geometry, validation, classification, span calculation, or public signatures other than package qualification.

## Scope

- Move `LayoutKind`, `LayoutDescriptor`, and package-private `LayoutGeometry` into the planned layout package.
- Move both layout test classes into the matching package so `LayoutGeometryTest` retains package-private access.
- Preserve the existing dependency from layout to public shape contracts.
- Update affected Javadoc links and public API documentation.
- Complete the migration of all current production contracts out of the flat model root package.

## Out of scope

- changing layout kinds, strides, offsets, view metadata, span arithmetic, or validation
- adding negative strides, dynamic numeric layouts, storage, materialization, or layout operations
- adding root-package compatibility wrappers
- moving data type or shape contracts again
- introducing identifiers, tensors, operations, graph contracts, runtime, compiler, planning, or backend behavior
- changing Gradle, Java configuration, module dependencies, `ARCHITECTURE.md`, or focused architecture documentation

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially the `modules/model` boundary and Java namespace
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Planning guide](../../../planning-guide.md), especially package structure planning
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md), especially the package map
- [Task 0003](0003-layout-descriptor-model.md), which defines the completed behavior being moved
- [Task 0003B](0003b-shape-package-migration.md), which establishes the imported shape package

## Architecture constraints

- All production packages remain below `io.github.pho001.synaptik.*`.
- The migration stays inside `modules/model` and the explicitly listed documentation files.
- `model.layout` may depend on public `model.shape` contracts and the JDK only.
- `LayoutGeometry` remains package-private and is not promoted into public API.
- Layout remains independent of storage, materialization policy, compiler, planning, runtime, and backends.
- If the migration requires behavior, architecture, Gradle, or dependency changes, stop and report the issue.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.model` no longer owns layout contracts after this migration and contains no current production type.

Package added:

- `io.github.pho001.synaptik.model.layout` owns backend-independent resolved logical layout geometry.

Type placement:

- `io.github.pho001.synaptik.model.layout.LayoutKind` — public geometry classification.
- `io.github.pho001.synaptik.model.layout.LayoutDescriptor` — public immutable resolved-layout value.
- `io.github.pho001.synaptik.model.layout.LayoutGeometry` — package-private checked geometry implementation used by the descriptor.

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

Expected documentation/planning updates:

- `docs/api/tensor-api.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may move and update at most:

- three existing production Java files;
- two existing test Java files; and
- the four documentation/planning files listed above.

Do not add Java types or modify method bodies. If a consumer outside these files requires an update, stop and revise the task before expanding scope.

## Acceptance criteria

- All three completed layout types use `io.github.pho001.synaptik.model.layout`.
- Both layout tests use the matching test package and retain all assertions.
- `LayoutGeometry` remains package-private and accessible to its package test.
- No production type or compatibility wrapper remains directly in `io.github.pho001.synaptik.model`.
- Layout imports `model.shape.Shape`; shape does not import layout.
- Layout kinds, canonical strides, classification, offsets, view metadata, span arithmetic, validation, equality, hashing, and diagnostics are unchanged.
- Public and package-private Javadoc remains complete and resolves in the new package.
- The Tensor API reference identifies the public layout package.
- No data type, shape, storage, operation, graph, compiler, runtime, backend, Gradle, or architecture behavior changes.
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

- exactly three production and two test files moved;
- no production Java file remains in the flat model root package;
- no method or test body changed;
- generated Javadoc contains `model.layout.LayoutKind` and `model.layout.LayoutDescriptor`;
- `LayoutGeometry` is not public;
- imports preserve the acyclic `layout -> shape` dependency; and
- task, master plan, roadmap, and API documentation agree on package and status.

## Dependencies

- Task 0003: Layout descriptor model — complete.
- Task 0003B: Shape package migration — complete.
- The model package map in the master plan — defined.

## Follow-up tasks

- Task 0004: Typed identifiers introduces new contracts in their owning `model.tensor` and `model.graph` packages.
- Task 0007: Tensor descriptor composes the migrated data type, shape, and layout values.
- Task 0017: Layout and view operations consume the migrated contracts.

Do not create the detailed task 0004 specification until this task is complete.

## Architecture impact

Expected impact: None.

This task completes the planned package organization for existing model foundations without changing module ownership, dependency direction, or behavior.

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

- All three layout production types moved together so package-private `LayoutGeometry` remains directly available to `LayoutDescriptor`.
- Both tests moved into the matching layout package, preserving white-box coverage of the package-private helper without widening visibility.
- The existing explicit import of `model.shape.Shape` was retained, making the intended package direction visible.

## Known limitations

- Numeric layouts still require fully static shapes, and negative strides remain unsupported as defined by task 0003.
- Source consumers must use the new layout package; compatibility wrappers for unreleased root-package names are intentionally absent.

## Validation evidence

- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew :modules:model:test` — passed with 56 tests, zero failures, zero errors, and zero skipped tests.
- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew :modules:model:javadoc` — passed and generated `LayoutKind` and `LayoutDescriptor` under `io/github/pho001/synaptik/model/layout`.
- `GRADLE_USER_HOME=/tmp/synaptik-gradle-home ./gradlew test` — passed for the complete multi-module repository.
- `git diff --check` — passed after implementation and final planning synchronization.
- Manual scope review confirmed exactly three production and two test files moved, with no method or test body changes.
- Manual visibility review confirmed `LayoutGeometry` remains package-private and its package test still compiles.
- Manual package review confirmed no production Java file remains directly in the flat model root and layout depends only on shape plus JDK classes.
- No Gradle, architecture, other-module, data type, shape behavior, or layout behavior change was made.
- Gradle emitted a non-fatal filesystem-watching warning in the sandbox; every requested task completed successfully.

## Implementation notes

- Moved the complete layout model into `io.github.pho001.synaptik.model.layout`.
- Moved both layout tests into the matching test package.
- Updated the Tensor API reference with the public layout package.
- Completed the foundational package migration and advanced the planning frontier to task 0004.

## Completion summary

- Completed changes: Migrated the complete layout foundation into its cohesive package without semantic or visibility changes.
- Files changed or created: Three production moves, two test moves, the Tensor API reference, this task, the model master plan, and the roadmap.
- Tests and validation: Model tests, model Javadoc, full repository tests, diff checks, visibility checks, package checks, scope review, and behavior review passed.
- Documentation impact: The public API reference now identifies `io.github.pho001.synaptik.model.layout`; no architecture documentation change was required.
- Javadoc review: Public Javadoc and the documented package-private helper contract remain accurate after the move.
- Unresolved issues: None.
- Follow-up required: Plan task 0004 before introducing typed identifiers.

Status: Complete
