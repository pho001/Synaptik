# Task 0003: Layout Descriptor Model

## Status

Complete

## Goal

Implement an immutable, backend-independent descriptor for resolved logical tensor layouts in `modules/model`. Represent row-major contiguous, offset-contiguous, general strided, and zero-stride broadcast layouts with checked `long` element strides, storage offset, alias/view metadata, and referenced element span without introducing storage ownership, materialization policy, operations, runtime, or backend behavior.

## Scope

- Define stable backend-independent layout kinds.
- Derive canonical row-major element strides for fully static shapes.
- Represent explicit non-negative element strides and storage offset.
- Record independently whether a layout aliases another tensor's storage.
- Classify resolved layouts as dense contiguous, dense with offset, strided, or broadcast zero-stride.
- Support scalar and zero-sized static shapes.
- Calculate a checked referenced element span for later storage-bound validation.
- Expose immutable defensive copies and positive/negative stride-axis lookup.
- Add focused unit tests and document the public layout contract in the tensor API reference.

## Out of scope

- numeric layout resolution for dynamic shapes
- symbolic strides or symbolic storage offsets
- negative strides or reverse views
- tensor storage allocation, ownership, lifetime, or mutation
- physical device buffers, byte addresses, alignment, or backend ABI descriptors
- reshape, expand, permute, transpose, slice, select, or contiguous operations
- deciding whether a layout must be materialized
- planning costs, backend capabilities, lowering, kernels, or runtime execution
- graph-wide shape/layout inference
- modifying existing data type, dimension, or shape contracts
- Gradle or dependency changes

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially `modules/model` and `modules/planning` responsibilities
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Model capability baseline](../capabilities.md), especially the layout baseline
- [Task 0002](0002-shape-and-dimension-model.md), which defines `Shape`
- [Model master plan](../master-plan.md)
- [Planning guide](../../../planning-guide.md)

## Legacy evidence

The read-only legacy implementation provides capability evidence through `tensor.TensorMetadata`, `tensor.layout.TensorShape`, `planning.descriptor.LayoutClass`, `backend.cpu.nativecpu.layout.TensorPhysicalView`, `BroadcastPlannerTest`, `TensorPhysicalViewTest`, and `StridedLayoutPlanningTest`.

Relevant legacy behavior includes:

- row-major contiguous element strides;
- non-zero storage offsets for selected/sliced views;
- non-contiguous strides for permuted views;
- zero strides for expanded/broadcast views;
- defensive copying of shape and stride arrays;
- rejection of negative strides in supported native views; and
- separation between layout description and later materialization decisions.

Legacy runtime and backend descriptors also mix data type, storage family, byte length, backend-specific layout classes, and execution policy. Those responsibilities must not be copied into the new model descriptor.

## Architecture constraints

- All production packages use `io.github.pho001.synaptik.*`.
- `LayoutDescriptor` contains logical element geometry only and remains backend- and storage-independent.
- Production code may use only the JDK and local `modules/model` contracts.
- Layout classification is a deterministic description of supplied shape/stride facts, not a materialization or kernel-selection decision.
- Planning may later consume layout facts to derive logical materialization requirements; the model must not expose `requiresMaterialization()` or equivalent policy.
- Resolved numeric layout creation requires a fully static `Shape`. Dynamic layout resolution belongs to later compiler/runtime contracts.
- Element strides and storage offsets use `long` and must be non-negative.
- A zero stride represents broadcast only when it repeats a dimension with static size greater than one. Canonical empty-tensor strides may contain zero without becoming broadcast layout.
- The legacy branch is read-only evidence. Do not copy its package structure or backend/runtime coupling.
- If implementation requires a change to `ARCHITECTURE.md`, stop and report the conflict instead of changing the contract.

## Required contracts

### Layout kinds

Create a public `LayoutKind` enum with exactly:

- `DENSE_CONTIGUOUS` — canonical row-major strides and zero storage offset;
- `DENSE_WITH_OFFSET` — canonical row-major strides and a non-zero storage offset;
- `STRIDED` — resolved non-canonical non-broadcast strides; and
- `BROADCAST_ZERO_STRIDE` — at least one non-singleton logical dimension is repeated through stride zero.

Layout kind describes geometry only. Whether a descriptor is a view is recorded separately.

### Layout geometry

Create a package-private, stateless `LayoutGeometry` implementation helper used by `LayoutDescriptor`. It owns only deterministic checked calculations:

- canonical row-major element strides for a fully static `Shape`;
- classification from shape, strides, and storage offset;
- detection of raw zero strides and broadcast zero strides; and
- referenced element span.

Canonical row-major stride calculation proceeds from the final axis toward the first:

```text
Shape[]        -> []
Shape[2, 3]    -> [3, 1]
Shape[2, 1, 3] -> [3, 3, 1]
Shape[2, 0, 4] -> [0, 4, 1]
```

Only products needed to produce an actual stride are calculated. Arithmetic overflow fails with `ArithmeticException`. Fully static shapes with element counts larger than `long` may still have representable strides when no required stride product overflows.

Classification order is significant:

1. exact canonical strides produce `DENSE_CONTIGUOUS` or `DENSE_WITH_OFFSET` according to offset;
2. otherwise, stride zero on any dimension with size greater than one produces `BROADCAST_ZERO_STRIDE`;
3. all other valid layouts produce `STRIDED`.

This ordering ensures canonical zero-element layouts such as `Shape[2, 0, 4]` with `[0, 4, 1]` remain dense rather than being mislabeled as broadcast.

Referenced element span is the minimum storage element count needed to make the greatest referenced element index valid:

```text
shape containing any zero-sized dimension -> 0
scalar Shape[] at offset 0                 -> 1
offset 0, shape [2, 3], strides [3, 1]     -> 6
offset 3, shape [3], strides [1]           -> 6
offset 0, shape [2, 3], strides [0, 1]     -> 3
```

For a non-empty shape, calculate:

```text
storageOffset + sum((dimensionSize - 1) * stride) + 1
```

All multiplication and addition use checked `long` arithmetic and fail with `ArithmeticException` on overflow. A shape containing a zero-sized dimension has span zero regardless of offset because it references no element. Rank-0 scalar shape references one element.

### Layout descriptor

Create an immutable final `LayoutDescriptor` in `io.github.pho001.synaptik.model`.

Required creation paths:

- `LayoutDescriptor.contiguous(Shape shape)` creates a non-view canonical row-major descriptor with offset zero; and
- `LayoutDescriptor.of(Shape shape, long[] strides, long storageOffset, boolean view)` creates and classifies an explicit resolved descriptor.

Both creation paths require a non-null fully static shape. A dynamic shape fails with `IllegalArgumentException`. The explicit factory also requires:

- non-null strides;
- stride count equal to shape rank;
- every stride non-negative;
- non-negative storage offset; and
- `view == true` when classification is `BROADCAST_ZERO_STRIDE`.

The descriptor stores:

- rank;
- immutable element strides;
- non-negative storage offset in elements;
- derived `LayoutKind`;
- explicit view/alias flag; and
- checked referenced element span.

Required inspection includes:

- `rank()`;
- `kind()`;
- `strides()` returning a defensive `long[]` copy;
- `stride(int axis)` with positive and negative axes;
- `storageOffset()`;
- `isView()`;
- `isContiguous()` for both dense kinds;
- `hasStorageOffset()`;
- `hasZeroStride()` as a raw stride fact;
- `isBroadcast()` from the derived kind; and
- `referencedElementSpan()`.

Invalid axes fail with `IndexOutOfBoundsException`, including every axis for scalar rank zero. Descriptor equality and hashing are structural across all stored fields. Diagnostic text must include kind, rank, strides, offset, view flag, and span without becoming a serialization contract.

Resolved descriptors deliberately do not retain `Shape`. Task 0007 will pair shape and layout in `TensorDescriptor`, avoiding duplicate shape ownership while keeping layout values independently immutable.

## Affected files

Expected production files:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/LayoutKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/LayoutGeometry.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/LayoutDescriptor.java`

Expected test files:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/LayoutGeometryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/LayoutDescriptorTest.java`

Expected documentation/planning updates:

- `docs/api/tensor-api.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

The capability baseline already records the selected layout kinds, resolved-static boundary, and planning/materialization separation. If implementation evidence requires changing them, stop and explain the discrepancy before editing that baseline.

## Maximum scope

This task may create or modify at most:

- five Java production/test files in `modules/model`; and
- the four documentation/planning files listed above.

Do not modify existing data type, dimension, or shape contracts, Gradle files, `ARCHITECTURE.md`, focused architecture documents, or another module. If more files are necessary, stop and propose a follow-up task.

## Acceptance criteria

- `LayoutKind` contains exactly the four required backend-independent values.
- Canonical strides are correct for scalar, ordinary, singleton, and zero-sized static shapes.
- Canonical stride calculation detects every required `long` overflow without unnecessarily multiplying beyond the first-axis stride.
- Explicit descriptors reject null, dynamic, rank-mismatched, negative-stride, negative-offset, and non-view broadcast inputs.
- Layout classification follows the required order and covers dense, offset-dense, strided, broadcast, scalar, and canonical zero-element cases.
- `view` remains independent of dense/strided geometry, except that broadcast repetition requires a view.
- Referenced element span is correct for dense, offset, strided, broadcast, scalar, and empty layouts and detects checked arithmetic overflow.
- Descriptor arrays are defensively copied on input and output.
- Positive and negative stride-axis lookup covers every valid axis and rejects every invalid axis.
- Equality and hashing include kind, rank, strides, offset, view flag, and span.
- No API exposes byte-level, storage-owner, device, runtime, backend, materialization-policy, or operation information.
- Unit tests cover every layout kind, constructor/factory validation, overflow, zero-sized shapes, scalar behavior, immutability, equality/hashing, and diagnostic text.
- All public types, enum constants, factories, and methods have detailed Javadoc covering semantics, parameters, results, units, nullability, ownership, immutability, and expected failures according to `AGENTS.md`. The package-private helper documents its implementation contract.
- `docs/api/tensor-api.md` documents resolved layout kinds, strides, element offset/span, view metadata, dynamic-shape limitation, and the separation from materialization policy.
- No existing model contract, Gradle file, or architecture document is changed.

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
- exactly three production and two test files are added;
- existing data type, dimension, and shape source files are unchanged;
- no storage, compiler, planning, runtime, prepare, engine, or backend dependency is introduced;
- the descriptor cannot be mutated through supplied or returned arrays;
- the task, master-plan row, and roadmap row have matching final statuses; and
- documentation and Javadoc describe the new model rather than legacy backend/runtime descriptors.

## Dependencies

- Task 0002: Shape and dimension model — complete.

## Follow-up tasks

- Task 0003C: Layout package migration moves the completed layout contracts into the package defined by the model master plan without changing behavior.
- Task 0007: Tensor descriptor model pairs `DataType`, `Shape`, and `LayoutDescriptor` and validates their rank compatibility.
- Task 0017: Layout and view operations create descriptors for reshape, expand, permute, slice, select, and contiguous semantics.
- `modules/planning` later derives logical materialization requirements from layout facts and backend capabilities.
- Concrete backend prepare implementations later translate logical element layouts into backend storage and executable routes.

Do not create detailed specifications for these follow-ups until task 0003 is complete and the planning frontier advances.

## Architecture impact

Expected impact: None.

This task implements the explicitly allowed `LayoutDescriptor` model responsibility. Resolved-static layout geometry and explicit view metadata are local design decisions within the existing boundary. If implementation requires architecture changes, stop and report the issue.

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are working in the Synaptik repository.

Read first:
- AGENTS.md
- ARCHITECTURE.md
- docs/planning/planning-guide.md
- docs/planning/modules/model/capabilities.md
- docs/planning/modules/model/tasks/0003-layout-descriptor-model.md

Implement task 0003 exactly as specified.

Keep the change inside modules/model and the explicitly listed documentation files. Do not implement storage, layout operations, materialization policy, graph-wide inference, compiler behavior, planning behavior, runtime behavior, or backend behavior. Do not modify existing data type, dimension, or shape contracts, Gradle files, or ARCHITECTURE.md. Do not copy legacy source code.

Add complete Javadoc for every affected public Java contract and document the package-private geometry helper. Run every validation command from the task specification.

At the end, update this task file with local decisions, known limitations, validation evidence, implementation notes, completion summary, and final status. Synchronize the task status in the model master plan and implementation roadmap.
```

## Local decisions

- `LayoutDescriptor` validates and classifies caller-supplied geometry before storing the already
  copied stride array, so no public or package-private path retains caller-owned mutable state.
- Canonical stride classification is evaluated before broadcast detection exactly as specified;
  therefore canonical empty shapes remain dense even when their raw stride list contains zero.
- Axis normalization is implemented locally because a resolved descriptor intentionally does not
  retain its source `Shape`.
- The package-private helper extracts static sizes through the existing `Shape` contract and uses
  `Math.multiplyExact` and `Math.addExact` for every required geometric product and sum.

## Known limitations

- Numeric descriptors require fully static shapes; symbolic strides and later binding of dynamic
  dimensions are not represented.
- Strides and offsets are non-negative. Negative-stride reverse views are not supported.
- The descriptor records geometry and explicit view metadata only. It cannot validate a span against
  storage capacity because storage is outside this task and model contract.
- Layout operations and materialization decisions remain assigned to their follow-up tasks.

## Validation evidence

- `./gradlew :modules:model:test` — passed with 56 tests, including 17 layout tests.
- `./gradlew :modules:model:javadoc` — passed without Javadoc errors.
- `./gradlew test` — passed for the full multi-module project.
- `git diff --check` — passed.
- Manual scope review confirmed exactly three new production and two new test files, production
  imports limited to `java.*` and local model types, and no changes to existing data type,
  dimension, or shape sources, Gradle files, architecture documents, or other modules.
- Manual immutability review and tests confirmed defensive copying on both input and output stride
  arrays.

## Implementation notes

- Added the four-value `LayoutKind` geometry taxonomy.
- Added package-private `LayoutGeometry` for canonical strides, ordered classification, raw and
  broadcast zero-stride detection, and checked referenced-span calculation.
- Added immutable `LayoutDescriptor` factories and inspection methods, structural equality and
  hashing, and diagnostic text.
- Added tests for all layout kinds, scalar and empty layouts, offset and strided spans, validation,
  overflow, axis normalization, array isolation, equality, hashing, and diagnostics.
- Extended the tensor API reference with the resolved-static layout contract and its separation
  from storage and materialization policy.

## Completion summary

Implemented the resolved layout descriptor foundation inside `modules/model` without adding storage,
operations, compiler, planning, runtime, or backend behavior. All acceptance criteria and requested
validation passed. The model master plan and implementation roadmap mark task 0003 complete. The
subsequently planned task 0003A is the current package-migration frontier; task 0003C will move these
layout contracts before task 0004 introduces new model types.

Status: Complete
