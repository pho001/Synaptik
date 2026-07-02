# Task 0002: Shape and Dimension Model

## Status

Complete

## Goal

Implement immutable, backend-independent dimension and shape value models for `modules/model`. Represent static and symbolic dynamic dimensions, rank-0 scalars, zero-sized dimensions, checked static element counts, axis normalization, and deterministic broadcast-result shapes without introducing layout, storage, graph inference, runtime, or backend behavior.

## Scope

- Define a sealed `Dimension` contract with explicit static and symbolic dynamic variants.
- Define non-negative static dimensions using `long` sizes.
- Define named dynamic dimensions with stable value equality.
- Define immutable `Shape` values containing zero or more dimensions.
- Use rank zero as the canonical scalar shape.
- Support zero-sized static dimensions and therefore empty tensor shapes.
- Expose rank, dimension lookup, negative-axis normalization, static-shape extraction, and checked known element count.
- Define right-aligned, NumPy-style broadcast-result shape calculation.
- Support conservative broadcasting for symbolic dimensions without creating a constraint solver.
- Add focused unit tests and document the public shape contract in the tensor API reference.

## Out of scope

- strides, storage offsets, contiguity, aliasing, or other layout metadata
- effective broadcast strides or gradient-reduction axes
- reshape requests, `-1` inference, expand, squeeze, permute, or transpose operations
- tensor storage allocation or Java array-size limits
- tensor indexing or coordinate-to-flat-index conversion
- graph-wide shape inference, symbolic constraint solving, or runtime dimension binding
- operation-specific shape inference
- compiler, planning, prepare, runtime, engine, or backend integration
- mutable shapes or mutable dimensions
- Gradle or dependency changes

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially `modules/model` ownership and forbidden dependencies
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Model capability baseline](../capabilities.md), especially the shape and dimension baseline
- [Model master plan](../master-plan.md)
- [Planning guide](../../../planning-guide.md)

## Legacy evidence

The read-only legacy implementation provides capability evidence through `tensor.layout.TensorShape`, `tensor.layout.TensorLayoutTransform`, `tensor.layout.BroadcastPlanner`, `TensorShapeValidationTest`, `BroadcastPlannerTest`, `BroadcastContractMatrixTest`, and `TransformOpsTest`.

Relevant legacy behavior includes:

- right-aligned broadcasting with singleton dimensions;
- positive and negative axis normalization;
- checked element-count multiplication;
- reshape support for one inferred `-1` dimension; and
- rejection of incompatible broadcast and reshape shapes.

The legacy representation also normalized rank-0 scalar shapes to `[1]`, rejected zero dimensions, used `int` dimensions, and rejected element counts greater than `Integer.MAX_VALUE`. Those representation limits are not copied into the new model. Reshape behavior remains selected capability evidence but is implemented later with layout/view operations.

## Architecture constraints

- All production packages use `io.github.pho001.synaptik.*`.
- Dimension, shape, and deterministic shape algebra belong to `modules/model` and remain backend-independent.
- Production code may use only the JDK and local `modules/model` types.
- `ShapeBroadcast` may combine already-known dimension semantics but must not perform graph traversal, operation inference, runtime binding, or backend capability checks.
- No shape API may expose strides, storage, device residency, physical buffers, kernel selection, runtime state, or prepared execution.
- Dynamic dimensions are symbolic model values, not mutable runtime variables.
- The legacy branch is read-only evidence. Do not copy its package structure or implementation.
- If implementation requires a change to `ARCHITECTURE.md`, stop and report the conflict instead of changing the contract.

## Required contracts

### Dimension hierarchy

Provide these public model types:

```text
Dimension
  StaticDimension
  DynamicDimension
```

`Dimension` is a sealed interface permitting exactly `StaticDimension` and `DynamicDimension`. It exposes enough typed inspection to determine whether a dimension is static or dynamic without backend or runtime context.

`StaticDimension` is an immutable value with a `long size`:

- size must be in `[0, Long.MAX_VALUE]`;
- zero is valid and represents an empty axis;
- negative sizes are rejected with `IllegalArgumentException`; and
- equality and hashing use the numeric size.

`DynamicDimension` is an immutable symbolic value with a `String symbol`:

- symbol must be non-null and non-blank;
- leading and trailing Unicode whitespace is removed using `String.strip()`;
- the stripped symbol is the stored canonical value;
- null fails with `NullPointerException`;
- blank input fails with `IllegalArgumentException`; and
- equality and hashing use the canonical symbol.

The dimension contract exposes:

- `isStatic()` and `isDynamic()` category queries;
- `staticSize()` returning an `OptionalLong`; and
- `dynamicSymbol()` returning an `Optional<String>`.

Do not encode dynamic dimensions through negative numeric sentinels. In particular, `-1` is not a valid canonical dimension.

### Shape value

`Shape` is an immutable final value in `io.github.pho001.synaptik.model`.

Required creation paths:

- `Shape.scalar()` creates the canonical rank-0 scalar shape;
- `Shape.of(long... sizes)` creates a fully static shape; and
- `Shape.ofDimensions(Dimension... dimensions)` creates a static/dynamic shape.

Creation defensively copies caller-owned arrays and rejects null arrays or null dimensions. `Shape.dimensions()` returns an immutable view or copy that cannot mutate the shape.

Required semantics:

- scalar shape has zero dimensions, rank `0`, and known element count `1`;
- a shape such as `[2, 0, 3]` is valid and has known element count `0`;
- rank is the number of dimensions;
- `dimension(int axis)` accepts positive and negative axes;
- `normalizeAxis(int axis)` maps `[-rank, rank - 1]` to `[0, rank - 1]`;
- every axis lookup on a scalar shape fails because rank zero has no axes;
- invalid axes fail with `IndexOutOfBoundsException`;
- `isFullyStatic()` reports whether all dimensions are static;
- `knownElementCount()` returns `OptionalLong.empty()` when any dimension is dynamic;
- a fully static zero-sized shape returns `OptionalLong.of(0)` without overflow;
- multiplication overflow for a non-empty fully static shape fails with `ArithmeticException`; and
- extraction to `long[]` succeeds only for fully static shapes and otherwise fails with `IllegalStateException`.

`Shape` equality and hashing are structural and order-sensitive. Its string representation must distinguish scalar, static, zero-sized, and symbolic shapes sufficiently for diagnostics, without becoming a serialization contract.

### Broadcast-result shape

`ShapeBroadcast.broadcast(Shape left, Shape right)` returns the immutable result of right-aligned broadcasting.

For each aligned dimension pair:

- equal dimensions produce that dimension;
- static size `1` broadcasts to the other dimension;
- a missing leading dimension behaves as static size `1`;
- static `0` with static `1` produces static `0`;
- two equal dynamic symbols produce that dynamic dimension;
- dynamic with static `1` produces the dynamic dimension; and
- all other pairs are rejected with `IllegalArgumentException` because compatibility cannot be proven locally.

Consequences that must be tested:

```text
[]        with [2, 3]    -> [2, 3]
[3]       with [2, 1, 3] -> [2, 1, 3]
[2, 1, 3] with [1, 4, 3] -> [2, 4, 3]
[0, 3]    with [1, 3]    -> [0, 3]
[N, 1]    with [N, 4]    -> [N, 4]
```

Incompatible static dimensions, different dynamic symbols, and a dynamic dimension paired with a non-singleton static size are rejected. Symbolic constraint creation belongs to compiler shape inference, not this utility.

`ShapeBroadcast` must not calculate strides, layout, materialization, gradient reduction, or backend execution information.

## Affected files

Expected production files:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/Dimension.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/StaticDimension.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/DynamicDimension.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/Shape.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/ShapeBroadcast.java`

Expected test files:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/DimensionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/ShapeTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/ShapeBroadcastTest.java`

Expected documentation/planning updates:

- `docs/api/tensor-api.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

The capability baseline already records the selected scalar, zero-sized, numeric-width, and dynamic-dimension decisions. If implementation evidence requires changing them, stop and explain the discrepancy before editing that baseline.

## Maximum scope

This task may create or modify at most:

- eight Java production/test files in `modules/model`; and
- the four documentation/planning files listed above.

Do not modify existing data type contracts, Gradle files, `ARCHITECTURE.md`, focused architecture documents, or another module. If more files are necessary, stop and propose a follow-up task.

## Acceptance criteria

- `Dimension` is sealed and permits exactly the immutable static and dynamic variants.
- Static dimensions accept every non-negative `long`, including zero, and reject negative values.
- Dynamic dimensions canonicalize non-blank symbols and reject null or blank symbols with the specified exception types.
- Dimension inspection returns consistent static/dynamic flags, optional size, and optional symbol.
- `Shape.scalar()` is rank zero with known element count one and no valid axis.
- Shapes are immutable and defensively isolate caller-owned arrays and returned collections/arrays.
- Static shapes support zero-sized dimensions and checked `long` element counts.
- Dynamic shapes report an unknown element count without inventing a numeric sentinel.
- Element-count overflow is detected without rejecting a shape merely because a later zero dimension makes its product zero.
- Positive and negative axis normalization covers every valid axis and rejects every out-of-range axis.
- Static extraction returns the exact ordered sizes and rejects dynamic shapes.
- Broadcast calculation covers scalar, rank-mismatched, singleton, zero-sized, and equal-symbol dynamic cases.
- Broadcast calculation rejects incompatible static sizes and unprovable symbolic combinations.
- No production class imports another Synaptik module or exposes layout, storage, compiler, runtime, or backend concepts.
- Unit tests cover value equality/hashing, constructor validation, immutability, element-count overflow, axis behavior, and all documented broadcast cases.
- All public types, constructors, record components, and methods have detailed Javadoc covering semantics, parameters, results, nullability, constraints, immutability, and expected failures according to `AGENTS.md`.
- `docs/api/tensor-api.md` documents the implemented scalar, static, dynamic, zero-sized, axis, element-count, and broadcasting contracts without claiming layout or backend support.
- No Gradle file or architecture contract is changed.

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

- production imports remain limited to `java.*` and local model types;
- exactly five production and three test files are added;
- no layout, storage, compiler, runtime, engine, or backend dependency is introduced;
- caller-owned arrays cannot mutate constructed values;
- the task, master-plan row, and roadmap row have matching final statuses; and
- documentation and Javadoc describe the new model rather than the legacy implementation.

## Dependencies

- No hard implementation dependency on task 0001.
- Task 0001 is complete because project work follows the ordered model frontier.
- Requires the repository skeleton and model capability baseline already present.

## Follow-up tasks

- Task 0003: Layout descriptor model consumes static shape sizes when deriving contiguous strides and describing views.
- Task 0007: Tensor descriptor model composes `DataType`, `Shape`, and `LayoutDescriptor`.
- Task 0017: Layout and view operations own reshape `-1` inference, expand, squeeze, permute, and other operation-level shape transformations.
- Compiler planning must later define graph-wide symbolic shape constraints and operation-specific inference.

Do not create detailed specifications for these follow-ups until task 0002 is complete and the planning frontier advances.

## Architecture impact

Expected impact: None.

This task implements an explicitly allowed `modules/model` responsibility. Rank-0 scalars, zero-sized dimensions, `long` sizes, and symbolic dimensions are local model-design decisions within the existing boundary. If implementation requires architecture changes, stop and report the issue.

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are working in the Synaptik repository.

Read first:
- AGENTS.md
- ARCHITECTURE.md
- docs/planning/planning-guide.md
- docs/planning/modules/model/capabilities.md
- docs/planning/modules/model/tasks/0002-shape-and-dimension-model.md

Implement task 0002 exactly as specified.

Keep the change inside modules/model and the explicitly listed documentation files. Do not implement layout descriptors, strides, storage, reshape operations, graph-wide shape inference, compiler behavior, runtime behavior, or backend behavior. Do not modify existing data type contracts, Gradle files, or ARCHITECTURE.md. Do not copy legacy source code.

Add complete Javadoc for every affected public Java contract. Run every validation command from the task specification.

At the end, update this task file with local decisions, known limitations, validation evidence, implementation notes, completion summary, and final status. Synchronize the task status in the model master plan and implementation roadmap.
```

## Local decisions

- `Dimension` is a sealed interface with default typed inspection methods. `StaticDimension` and `DynamicDimension` are public immutable records and the only permitted variants.
- Dynamic symbols are canonicalized once in the record constructor with `String.strip()`; their canonical value defines equality and hashing.
- `Shape` stores dimensions in a JDK immutable list, returns that safe list directly, and defensively isolates both primitive and dimension arrays supplied to factories.
- The scalar shape is a shared immutable rank-0 instance returned by all empty creation paths.
- `knownElementCount()` first rejects dynamic shapes, then detects any zero dimension before checked multiplication. This preserves the defined unknown-versus-zero distinction and avoids irrelevant overflow.
- Axis normalization uses a `long` intermediate so extreme negative `int` inputs cannot overflow during normalization.
- `ShapeBroadcast` uses a shared immutable singleton dimension for missing leading axes and returns existing immutable dimensions instead of cloning them.

## Known limitations

- Dynamic dimensions carry names only; bounds, equivalence constraints, and runtime bindings are deferred to compiler/runtime planning.
- Broadcasting intentionally rejects different symbols and dynamic-versus-non-singleton-static pairs even when a future constraint solver might prove them compatible.
- The logical model permits sizes beyond Java array limits; storage tasks must validate their own physical limits.
- Reshape inference, strides, layouts, coordinate indexing, and materialization are outside this task.

## Validation evidence

- `./gradlew :modules:model:test` — passed. 39 model tests executed with no failures, errors, or skips; 24 tests are new for task 0002: 6 `DimensionTest`, 9 `ShapeTest`, and 9 `ShapeBroadcastTest`.
- `./gradlew :modules:model:javadoc` — passed. Javadoc for all public dimension and shape contracts generated without errors.
- `./gradlew test` — passed. The complete repository test lifecycle completed successfully with 36 actionable tasks in the final run.
- `git diff --check` — passed before closure; repeated after the final planning update.
- Markdown link validation — passed for all 73 repository Markdown files.
- Manual dependency review — new production imports are limited to `java.util` and local model types; no project-module dependency was added.
- Manual scope review — exactly five production files and three test files were added.
- Manual immutability review — tests verify defensive handling of input arrays, copied static output arrays, and unmodifiable dimension lists.
- Data type, Gradle, and architecture review — no existing data type contract, Gradle file, `ARCHITECTURE.md`, or focused architecture document changed.

## Implementation notes

- Added sealed static and symbolic dimension values with optional typed inspection.
- Added immutable scalar, static, dynamic, and zero-sized shapes with checked `long` element counts and positive/negative axis normalization.
- Added conservative right-aligned broadcasting for scalar, rank-mismatched, singleton, zero-sized, and equal-symbol dynamic shapes.
- Added rejection coverage for invalid sizes, symbols, axes, overflow, incompatible static shapes, and unprovable symbolic broadcasts.
- Expanded the public tensor API reference with implemented dimension, shape, element-count, axis, and broadcasting behavior.

## Completion summary

- Completed changes: Implemented the immutable shape and dimension foundation and deterministic local broadcasting.
- Files changed or created: Five production classes, three unit-test classes, the tensor API documentation, the capability baseline established during planning, and synchronized planning documents.
- Tests and validation: All required Gradle commands, Javadoc generation, diff checks, link checks, dependency review, scope review, and immutability checks passed.
- Documentation impact: Documented the public shape model and retained the architecture boundary between shape values, layout, compiler inference, storage, and backends.
- Javadoc review: Every public type, record component, constructor, and explicit method documents semantics, parameters, results, constraints, immutability, and failures where applicable.
- Unresolved issues: None.
- Follow-up required: None. Task 0003 is the next planned frontier.

Status: Complete
