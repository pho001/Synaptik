# Task 0002: [Shape](../../../../glossary.md#shape) and [Dimension](../../../../glossary.md#dimension) [Model](../../../../glossary.md#term-model)

**Editorial reading key (later context, not a status update).**

An application or an expression helper supplies ordered extents to Shape; Shape describes which coordinates exist without storing elements. Rank counts axes, not values: a scalar has no axes but one logical value, while an extent of zero makes a tensor empty. Local broadcasting compares axes from the right and permits the singleton repetitions selected here; it neither repeats bytes nor binds a named dimension at runtime. The symbolic forms in the current reference are later context, not additions to this task's two-variant contract.

**Shape reading example.** For a fully static shape with rank $r\in\mathbb{Z}_{\ge0}$ and extents $d_0,\ldots,d_{r-1}\in\mathbb{Z}_{\ge0}$, the logical element count is $n=\prod_{i=0}^{r-1}d_i$. The empty product for $r=0$ is $1$, explaining the scalar; for shape $[2,0,3]$, $n=0$. These are logical counts, not memory capacities. The original contract below determines unknown counts and checked overflow.

Focused current reading: [Current dimension model](../../../../reference/tensor/shapes.md#shapes-and-dimensions); [Current local broadcasting reference](../../../../reference/tensor/shapes.md#broadcasting). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Implement [immutable](../../../../glossary.md#term-immutable), [backend](../../../../glossary.md#backend)-independent [dimension](../../../../glossary.md#dimension) and [shape](../../../../glossary.md#shape) value models for `modules/model`. Represent static and symbolic dynamic dimensions, [rank](../../../../glossary.md#term-rank)-0 [scalars](../../../../glossary.md#term-scalar), zero-sized dimensions, checked static [element](../../../../glossary.md#term-element) counts, [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization), and deterministic broadcast-result shapes without introducing [layout](../../../../glossary.md#layout), storage, [graph](../../../../glossary.md#graph) [inference](../../../../glossary.md#term-inference), [runtime](../../../../glossary.md#term-runtime), or backend behavior.

## [Scope](../../../../glossary.md#term-scope)

- Define a [sealed](../../../../glossary.md#term-sealed) [`Dimension`](../../../../glossary.md#dimension) contract with explicit static and symbolic dynamic variants.
- Define non-negative static dimensions using `long` sizes.
- Define named dynamic dimensions with stable value equality.
- Define [immutable](../../../../glossary.md#term-immutable) [`Shape`](../../../../glossary.md#shape) values containing zero or more dimensions.
- Use [rank](../../../../glossary.md#term-rank) zero as the [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) shape.
- Support zero-sized static dimensions and therefore empty [tensor](../../../../glossary.md#tensor) shapes.
- Expose rank, dimension [lookup](../../../../glossary.md#term-lookup), negative-[axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization), static-shape extraction, and checked known [element](../../../../glossary.md#term-element) count.
- Define right-aligned, NumPy-style broadcast-result shape calculation.
- Support conservative [broadcasting](../../../../glossary.md#broadcasting) for symbolic dimensions without creating a constraint solver.
- Add focused unit tests and document the public shape contract in the tensor [API](../../../../glossary.md#term-application-programming-interface-api) reference.

## Out of [scope](../../../../glossary.md#term-scope)

- [strides](../../../../glossary.md#term-stride), storage offsets, [contiguity](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), [aliasing](../../../../glossary.md#term-aliasing), or other [layout](../../../../glossary.md#layout) [metadata](../../../../glossary.md#term-metadata)
- effective broadcast strides or [gradient](../../../../glossary.md#term-gradient)-reduction [axes](../../../../glossary.md#term-axis)
- reshape requests, `-1` [inference](../../../../glossary.md#term-inference), expand, squeeze, permute, or [transpose](../../../../glossary.md#term-transpose) operations
- [tensor](../../../../glossary.md#tensor) storage allocation or Java array-size limits
- tensor indexing or [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)-to-flat-index conversion
- [graph](../../../../glossary.md#graph)-wide [shape](../../../../glossary.md#shape) inference, symbolic constraint solving, or [runtime](../../../../glossary.md#term-runtime) [dimension](../../../../glossary.md#dimension) [binding](../../../../glossary.md#term-binding)
- [operation](../../../../glossary.md#operation)-specific shape inference
- [compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), [prepare](../../../../glossary.md#prepare), runtime, [engine](../../../../glossary.md#term-engine), or [backend](../../../../glossary.md#backend) integration
- [mutable](../../../../glossary.md#term-mutable) shapes or mutable dimensions
- [Gradle](../../../../glossary.md#term-gradle) or [dependency](../../../../glossary.md#term-dependency) changes

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially `modules/model` [ownership](../../../../glossary.md#term-ownership-meanings) and forbidden dependencies
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>, especially the [shape](../../../../glossary.md#shape) and [dimension](../../../../glossary.md#dimension) baseline
- [Model master plan](../master-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>

## Legacy evidence

The read-only legacy implementation provides capability evidence through `tensor.layout.TensorShape`, `tensor.layout.TensorLayoutTransform`, `tensor.layout.BroadcastPlanner`, `TensorShapeValidationTest`, `BroadcastPlannerTest`, `BroadcastContractMatrixTest`, and `TransformOpsTest`.

Relevant legacy behavior includes:

- right-aligned [broadcasting](../../../../glossary.md#broadcasting) with [singleton](../../../../glossary.md#term-singleton) [dimensions](../../../../glossary.md#dimension);
- positive and negative [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization);
- checked [element](../../../../glossary.md#term-element)-count multiplication;
- reshape support for one inferred `-1` dimension; and
- rejection of incompatible broadcast and reshape [shapes](../../../../glossary.md#shape).

The legacy representation also normalized [rank](../../../../glossary.md#term-rank)-0 [scalar](../../../../glossary.md#term-scalar) shapes to `[1]`, rejected zero dimensions, used `int` dimensions, and rejected element counts greater than `Integer.MAX_VALUE`. Those representation limits are not copied into the new [model](../../../../glossary.md#term-model). Reshape behavior remains selected capability evidence but is implemented later with [layout](../../../../glossary.md#layout)/[view](../../../../glossary.md#term-view-meanings) operations.

## Architecture constraints

- All production packages use `io.github.pho001.synaptik.*`.
- [Dimension](../../../../glossary.md#dimension), [shape](../../../../glossary.md#shape), and deterministic shape algebra belong to `modules/model` and remain [backend](../../../../glossary.md#backend)-independent.
- Production code may use only the [JDK](../../../../glossary.md#term-java-development-kit-jdk) and local `modules/model` types.
- `ShapeBroadcast` may combine already-known dimension semantics but must not perform [graph](../../../../glossary.md#graph) traversal, [operation](../../../../glossary.md#operation) [inference](../../../../glossary.md#term-inference), [runtime](../../../../glossary.md#term-runtime) [binding](../../../../glossary.md#term-binding), or [backend capability](../../../../glossary.md#backend-capability) checks.
- No shape [API](../../../../glossary.md#term-application-programming-interface-api) may expose [strides](../../../../glossary.md#term-stride), storage, [device](../../../../glossary.md#term-device) [residency](../../../../glossary.md#residency), physical [buffers](../../../../glossary.md#term-buffer), [kernel](../../../../glossary.md#kernel) selection, runtime state, or [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution).
- Dynamic dimensions are symbolic [model](../../../../glossary.md#term-model) values, not [mutable](../../../../glossary.md#term-mutable) runtime variables.
- The legacy branch is read-only evidence. Do not copy its package structure or implementation.
- If implementation requires a change to `ARCHITECTURE.md`, stop and report the conflict instead of changing the contract.

## Required contracts

### [Dimension](../../../../glossary.md#dimension) hierarchy

Provide these public [model](../../../../glossary.md#term-model) types:

```text
Dimension
  StaticDimension
  DynamicDimension
```

`Dimension` is a [sealed](../../../../glossary.md#term-sealed) interface permitting exactly `StaticDimension` and `DynamicDimension`. It exposes enough typed inspection to determine whether a dimension is static or dynamic without [backend](../../../../glossary.md#backend) or [runtime](../../../../glossary.md#term-runtime) context.

`StaticDimension` is an [immutable](../../../../glossary.md#term-immutable) value with a `long size`:

- size must be in `[0, Long.MAX_VALUE]`;
- zero is valid and represents an empty [axis](../../../../glossary.md#term-axis);
- negative sizes are rejected with `IllegalArgumentException`; and
- equality and hashing use the numeric size.

`DynamicDimension` is an immutable symbolic value with a `String symbol`:

- [symbol](../../../../glossary.md#term-symbol) must be non-null and non-blank;
- leading and trailing [Unicode](../../../../glossary.md#term-unicode) whitespace is removed using `String.strip()`;
- the stripped symbol is the stored [canonical](../../../../glossary.md#term-canonical-form) value;
- null fails with `NullPointerException`;
- blank input fails with `IllegalArgumentException`; and
- equality and hashing use the canonical symbol.

The dimension contract exposes:

- `isStatic()` and `isDynamic()` category queries;
- `staticSize()` returning an `OptionalLong`; and
- `dynamicSymbol()` returning an `Optional<String>`.

Do not encode dynamic dimensions through negative numeric sentinels. In particular, `-1` is not a valid canonical dimension.

### [Shape](../../../../glossary.md#shape) value

`Shape` is an [immutable](../../../../glossary.md#term-immutable) final value in `io.github.pho001.synaptik.model`.

Required creation paths:

- `Shape.scalar()` creates the [canonical](../../../../glossary.md#term-canonical-form) [rank](../../../../glossary.md#term-rank)-0 [scalar](../../../../glossary.md#term-scalar) shape;
- `Shape.of(long... sizes)` creates a fully static shape; and
- `Shape.ofDimensions(Dimension... dimensions)` creates a static/dynamic shape.

Creation defensively copies caller-owned arrays and rejects null arrays or null [dimensions](../../../../glossary.md#dimension). `Shape.dimensions()` returns an immutable [view](../../../../glossary.md#term-view-meanings) or copy that cannot mutate the shape.

Required semantics:

- scalar shape has zero dimensions, rank `0`, and known [element](../../../../glossary.md#term-element) count `1`;
- a shape such as `[2, 0, 3]` is valid and has known element count `0`;
- rank is the number of dimensions;
- `dimension(int axis)` accepts positive and negative [axes](../../../../glossary.md#term-axis);
- `normalizeAxis(int axis)` maps `[-rank, rank - 1]` to `[0, rank - 1]`;
- every axis [lookup](../../../../glossary.md#term-lookup) on a scalar shape fails because rank zero has no axes;
- invalid axes fail with `IndexOutOfBoundsException`;
- `isFullyStatic()` reports whether all dimensions are static;
- `knownElementCount()` returns `OptionalLong.empty()` when any dimension is dynamic;
- a fully static zero-sized shape returns `OptionalLong.of(0)` without [overflow](../../../../glossary.md#term-overflow);
- multiplication overflow for a non-empty fully static shape fails with `ArithmeticException`; and
- extraction to `long[]` succeeds only for fully static shapes and otherwise fails with `IllegalStateException`.

`Shape` equality and hashing are structural and order-sensitive. Its string representation must distinguish scalar, static, zero-sized, and symbolic shapes sufficiently for diagnostics, without becoming a [serialization](../../../../glossary.md#term-serialization) contract.

### Broadcast-result [shape](../../../../glossary.md#shape)

`ShapeBroadcast.broadcast(Shape left, Shape right)` returns the [immutable](../../../../glossary.md#term-immutable) result of right-aligned [broadcasting](../../../../glossary.md#broadcasting).

For each aligned [dimension](../../../../glossary.md#dimension) pair:

- equal dimensions produce that dimension;
- static size `1` broadcasts to the other dimension;
- a missing [leading dimension](../../../../glossary.md#term-leading-dimension) behaves as static size `1`;
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

Incompatible static dimensions, different dynamic symbols, and a dynamic dimension paired with a non-[singleton](../../../../glossary.md#term-singleton) static size are rejected. Symbolic constraint creation belongs to [compiler](../../../../glossary.md#term-compiler) shape [inference](../../../../glossary.md#term-inference), not this utility.

`ShapeBroadcast` must not calculate [strides](../../../../glossary.md#term-stride), [layout](../../../../glossary.md#layout), [materialization](../../../../glossary.md#materialization), [gradient](../../../../glossary.md#term-gradient) reduction, or [backend](../../../../glossary.md#backend) execution information.

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

Expected documentation/[planning](../../../../glossary.md#term-planning) updates:

- `docs/api/tensor-api.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

The capability baseline already records the selected [scalar](../../../../glossary.md#term-scalar), zero-sized, numeric-width, and dynamic-[dimension](../../../../glossary.md#dimension) decisions. If implementation evidence requires changing them, stop and explain the discrepancy before editing that baseline.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- eight Java production/test files in `modules/model`; and
- the four documentation/[planning](../../../../glossary.md#term-planning) files listed above.

Do not modify existing [data type](../../../../glossary.md#data-type--datatype) contracts, [Gradle](../../../../glossary.md#term-gradle) files, `ARCHITECTURE.md`, focused architecture documents, or another [module](../../../../glossary.md#term-module). If more files are necessary, stop and propose a follow-up task.

## Acceptance criteria

- [`Dimension`](../../../../glossary.md#dimension) is [sealed](../../../../glossary.md#term-sealed) and permits exactly the [immutable](../../../../glossary.md#term-immutable) static and dynamic variants.
- Static dimensions accept every non-negative `long`, including zero, and reject negative values.
- Dynamic dimensions canonicalize non-blank symbols and reject null or blank symbols with the specified exception types.
- Dimension inspection returns consistent static/dynamic flags, optional size, and optional [symbol](../../../../glossary.md#term-symbol).
- `Shape.scalar()` is [rank](../../../../glossary.md#term-rank) zero with known [element](../../../../glossary.md#term-element) count one and no valid [axis](../../../../glossary.md#term-axis).
- [Shapes](../../../../glossary.md#shape) are immutable and defensively isolate caller-owned arrays and returned collections/arrays.
- Static shapes support zero-sized dimensions and checked `long` element counts.
- Dynamic shapes report an unknown element count without inventing a numeric sentinel.
- Element-count [overflow](../../../../glossary.md#term-overflow) is detected without rejecting a shape merely because a later zero dimension makes its product zero.
- Positive and negative axis [normalization](../../../../glossary.md#term-normalization) covers every valid axis and rejects every out-of-range axis.
- Static extraction returns the exact ordered sizes and rejects dynamic shapes.
- Broadcast calculation covers [scalar](../../../../glossary.md#term-scalar), rank-mismatched, [singleton](../../../../glossary.md#term-singleton), zero-sized, and equal-symbol dynamic cases.
- Broadcast calculation rejects incompatible static sizes and unprovable symbolic combinations.
- No production class imports another Synaptik [module](../../../../glossary.md#term-module) or exposes [layout](../../../../glossary.md#layout), storage, [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), or [backend](../../../../glossary.md#backend) concepts.
- Unit tests cover value equality/hashing, constructor [validation](../../../../glossary.md#term-validation), immutability, element-count overflow, axis behavior, and all documented broadcast cases.
- All public types, constructors, record components, and methods have detailed [Javadoc](../../../../glossary.md#term-javadoc) covering semantics, [parameters](../../../../glossary.md#term-parameter), results, [nullability](../../../../glossary.md#term-nullability), constraints, immutability, and expected failures according to `AGENTS.md`.
- `docs/api/tensor-api.md` documents the implemented scalar, static, dynamic, zero-sized, axis, element-count, and [broadcasting](../../../../glossary.md#broadcasting) contracts without claiming layout or backend support.
- No [Gradle](../../../../glossary.md#term-gradle) file or [architecture contract](../../../../glossary.md#architecture-contract) is changed.

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

- production imports remain limited to `java.*` and local [model](../../../../glossary.md#term-model) types;
- exactly five production and three test files are added;
- no [layout](../../../../glossary.md#layout), storage, [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), [engine](../../../../glossary.md#term-engine), or [backend](../../../../glossary.md#backend) [dependency](../../../../glossary.md#term-dependency) is introduced;
- caller-owned arrays cannot mutate constructed values;
- the task, master-plan row, and roadmap row have matching final statuses; and
- documentation and [Javadoc](../../../../glossary.md#term-javadoc) describe the new model rather than the legacy implementation.

## Dependencies

- No hard implementation [dependency](../../../../glossary.md#term-dependency) on task 0001.
- Task 0001 is complete because project work follows the ordered [model](../../../../glossary.md#term-model) frontier.
- Requires the repository skeleton and model capability baseline already present.

## Follow-up tasks

- Task 0003: [Layout](../../../../glossary.md#layout) [descriptor](../../../../glossary.md#term-descriptor) [model](../../../../glossary.md#term-model) consumes static [shape](../../../../glossary.md#shape) sizes when deriving [contiguous](../../../../glossary.md#term-contiguous) [strides](../../../../glossary.md#term-stride) and describing [views](../../../../glossary.md#term-view-meanings).
- Task 0003B: Shape package migration moves the completed [dimension](../../../../glossary.md#dimension) and shape contracts into the package defined by the model master plan without changing behavior.
- Task 0007: [Tensor descriptor](../../../../glossary.md#tensor-descriptor) model composes [`DataType`](../../../../glossary.md#data-type--datatype), `Shape`, and `LayoutDescriptor`.
- Task 0017: Layout and view operations own reshape `-1` [inference](../../../../glossary.md#term-inference), expand, squeeze, permute, and other [operation](../../../../glossary.md#operation)-level shape transformations.
- [Compiler](../../../../glossary.md#term-compiler) [planning](../../../../glossary.md#term-planning) must later define [graph](../../../../glossary.md#graph)-wide symbolic shape constraints and operation-specific inference.

Do not create detailed specifications for these follow-ups until task 0002 is complete and the planning frontier advances.

## Architecture impact

Expected impact: None.

This task implements an explicitly allowed `modules/model` responsibility. [Rank](../../../../glossary.md#term-rank)-0 [scalars](../../../../glossary.md#term-scalar), zero-sized [dimensions](../../../../glossary.md#dimension), `long` sizes, and symbolic dimensions are local [model](../../../../glossary.md#term-model)-design decisions within the existing boundary. If implementation requires architecture changes, stop and report the issue.

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

- [`Dimension`](../../../../glossary.md#dimension) is a [sealed](../../../../glossary.md#term-sealed) interface with default typed inspection methods. `StaticDimension` and `DynamicDimension` are public [immutable](../../../../glossary.md#term-immutable) records and the only permitted variants.
- Dynamic symbols are canonicalized once in the record constructor with `String.strip()`; their [canonical](../../../../glossary.md#term-canonical-form) value defines equality and hashing.
- [`Shape`](../../../../glossary.md#shape) stores dimensions in a [JDK](../../../../glossary.md#term-java-development-kit-jdk) immutable list, returns that safe list directly, and defensively isolates both primitive and dimension arrays supplied to factories.
- The [scalar](../../../../glossary.md#term-scalar) shape is a shared immutable [rank](../../../../glossary.md#term-rank)-0 instance returned by all empty creation paths.
- `knownElementCount()` first rejects dynamic shapes, then detects any zero dimension before checked multiplication. This preserves the defined unknown-versus-zero distinction and avoids irrelevant [overflow](../../../../glossary.md#term-overflow).
- [Axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization) uses a `long` intermediate so extreme negative `int` inputs cannot overflow during normalization.
- `ShapeBroadcast` uses a shared immutable [singleton](../../../../glossary.md#term-singleton) dimension for missing leading axes and returns existing immutable dimensions instead of cloning them.

## Known limitations

- Dynamic [dimensions](../../../../glossary.md#dimension) [carry](../../../../glossary.md#term-carry) names only; bounds, equivalence constraints, and [runtime](../../../../glossary.md#term-runtime) [bindings](../../../../glossary.md#term-binding) are deferred to [compiler](../../../../glossary.md#term-compiler)/runtime [planning](../../../../glossary.md#term-planning).
- [Broadcasting](../../../../glossary.md#broadcasting) intentionally rejects different symbols and dynamic-versus-non-[singleton](../../../../glossary.md#term-singleton)-static pairs even when a future constraint solver might prove them compatible.
- The logical [model](../../../../glossary.md#term-model) permits sizes beyond Java array limits; storage tasks must validate their own physical limits.
- Reshape [inference](../../../../glossary.md#term-inference), [strides](../../../../glossary.md#term-stride), [layouts](../../../../glossary.md#layout), [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) indexing, and [materialization](../../../../glossary.md#materialization) are outside this task.

## [Validation](../../../../glossary.md#term-validation) evidence

- `./gradlew :modules:model:test` — passed. 39 [model](../../../../glossary.md#term-model) tests executed with no failures, errors, or skips; 24 tests are new for task 0002: 6 `DimensionTest`, 9 `ShapeTest`, and 9 `ShapeBroadcastTest`.
- `./gradlew :modules:model:javadoc` — passed. [Javadoc](../../../../glossary.md#term-javadoc) for all public [dimension](../../../../glossary.md#dimension) and [shape](../../../../glossary.md#shape) contracts generated without errors.
- `./gradlew test` — passed. The complete repository test [lifecycle](../../../../glossary.md#lifecycle) completed successfully with 36 actionable tasks in the final [run](../../../../glossary.md#term-run).
- `git diff --check` — passed before closure; repeated after the final [planning](../../../../glossary.md#term-planning) update.
- Markdown link validation — passed for all 73 repository Markdown files.
- Manual [dependency](../../../../glossary.md#term-dependency) review — new production imports are limited to `java.util` and local model types; no project-[module](../../../../glossary.md#term-module) dependency was added.
- Manual [scope](../../../../glossary.md#term-scope) review — exactly five production files and three test files were added.
- Manual immutability review — tests verify defensive handling of input arrays, copied static output arrays, and unmodifiable dimension lists.
- [Data type](../../../../glossary.md#data-type--datatype), [Gradle](../../../../glossary.md#term-gradle), and architecture review — no existing data type contract, Gradle file, `ARCHITECTURE.md`, or focused architecture document changed.

## Implementation notes

- Added [sealed](../../../../glossary.md#term-sealed) static and symbolic [dimension](../../../../glossary.md#dimension) values with optional typed inspection.
- Added [immutable](../../../../glossary.md#term-immutable) [scalar](../../../../glossary.md#term-scalar), static, dynamic, and zero-sized [shapes](../../../../glossary.md#shape) with checked `long` [element](../../../../glossary.md#term-element) counts and positive/negative [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization).
- Added conservative right-aligned [broadcasting](../../../../glossary.md#broadcasting) for scalar, [rank](../../../../glossary.md#term-rank)-mismatched, [singleton](../../../../glossary.md#term-singleton), zero-sized, and equal-[symbol](../../../../glossary.md#term-symbol) dynamic shapes.
- Added rejection coverage for invalid sizes, symbols, axes, [overflow](../../../../glossary.md#term-overflow), incompatible static shapes, and unprovable symbolic broadcasts.
- Expanded the public [tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) reference with implemented dimension, shape, element-count, axis, and broadcasting behavior.

## Completion summary

- Completed changes: Implemented the [immutable](../../../../glossary.md#term-immutable) [shape](../../../../glossary.md#shape) and [dimension](../../../../glossary.md#dimension) foundation and deterministic local [broadcasting](../../../../glossary.md#broadcasting).
- Files changed or created: Five production classes, three unit-test classes, the [tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) documentation, the capability baseline established during [planning](../../../../glossary.md#term-planning), and synchronized planning documents.
- Tests and [validation](../../../../glossary.md#term-validation): All required [Gradle](../../../../glossary.md#term-gradle) commands, [Javadoc](../../../../glossary.md#term-javadoc) generation, diff checks, link checks, [dependency](../../../../glossary.md#term-dependency) review, [scope](../../../../glossary.md#term-scope) review, and immutability checks passed.
- Documentation impact: Documented the public shape [model](../../../../glossary.md#term-model) and retained the architecture boundary between shape values, [layout](../../../../glossary.md#layout), [compiler](../../../../glossary.md#term-compiler) [inference](../../../../glossary.md#term-inference), storage, and [backends](../../../../glossary.md#backend).
- Javadoc review: Every public type, record component, constructor, and explicit method documents semantics, [parameters](../../../../glossary.md#term-parameter), results, constraints, immutability, and failures where applicable.
- Unresolved issues: None.
- Follow-up required: None. Task 0003 is the next planned frontier.

Status: Complete
