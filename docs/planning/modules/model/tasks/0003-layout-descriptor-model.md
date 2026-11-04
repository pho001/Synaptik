# Task 0003: [Layout](../../../../glossary.md#layout) [Descriptor](../../../../glossary.md#term-descriptor) [Model](../../../../glossary.md#term-model)

## Status

Complete

## Goal

Implement an [immutable](../../../../glossary.md#term-immutable), [backend](../../../../glossary.md#backend)-independent [descriptor](../../../../glossary.md#term-descriptor) for resolved logical [tensor](../../../../glossary.md#tensor) [layouts](../../../../glossary.md#layout) in `modules/model`. Represent [row-major](../../../../glossary.md#term-row-major) [contiguous](../../../../glossary.md#term-contiguous), offset-contiguous, general strided, and zero-[stride](../../../../glossary.md#term-stride) broadcast layouts with checked `long` [element](../../../../glossary.md#term-element) strides, [storage offset](../../../../glossary.md#term-storage-offset), [alias](../../../../glossary.md#aliasing-and-arena)/[view](../../../../glossary.md#term-view-meanings) [metadata](../../../../glossary.md#term-metadata), and [referenced element span](../../../../glossary.md#referenced-element-span) without introducing storage [ownership](../../../../glossary.md#term-ownership-meanings), [materialization](../../../../glossary.md#materialization) policy, operations, [runtime](../../../../glossary.md#term-runtime), or backend behavior.

## [Scope](../../../../glossary.md#term-scope)

- Define stable [backend](../../../../glossary.md#backend)-independent [layout](../../../../glossary.md#layout) kinds.
- Derive [canonical](../../../../glossary.md#term-canonical-form) [row-major](../../../../glossary.md#term-row-major) [element](../../../../glossary.md#term-element) [strides](../../../../glossary.md#term-stride) for fully static [shapes](../../../../glossary.md#shape).
- Represent explicit non-negative element strides and [storage offset](../../../../glossary.md#term-storage-offset).
- Record independently whether a layout [aliases](../../../../glossary.md#aliasing-and-arena) another [tensor](../../../../glossary.md#tensor)'s storage.
- Classify resolved layouts as dense [contiguous](../../../../glossary.md#term-contiguous), dense with offset, strided, or broadcast zero-stride.
- Support [scalar](../../../../glossary.md#term-scalar) and zero-sized static shapes.
- Calculate a checked [referenced element span](../../../../glossary.md#referenced-element-span) for later storage-bound [validation](../../../../glossary.md#term-validation).
- Expose [immutable](../../../../glossary.md#term-immutable) defensive copies and positive/negative stride-[axis](../../../../glossary.md#term-axis) [lookup](../../../../glossary.md#term-lookup).
- Add focused unit tests and document the public layout contract in the tensor [API](../../../../glossary.md#term-application-programming-interface-api) reference.

## Out of [scope](../../../../glossary.md#term-scope)

- numeric [layout](../../../../glossary.md#layout) resolution for dynamic [shapes](../../../../glossary.md#shape)
- symbolic [strides](../../../../glossary.md#term-stride) or symbolic storage offsets
- negative strides or reverse [views](../../../../glossary.md#term-view-meanings)
- [tensor](../../../../glossary.md#tensor) storage allocation, [ownership](../../../../glossary.md#term-ownership-meanings), [lifetime](../../../../glossary.md#term-lifetime), or mutation
- physical [device](../../../../glossary.md#term-device) [buffers](../../../../glossary.md#term-buffer), byte addresses, [alignment](../../../../glossary.md#term-alignment), or [backend](../../../../glossary.md#backend) [ABI](../../../../glossary.md#term-application-binary-interface-abi) [descriptors](../../../../glossary.md#term-descriptor)
- reshape, expand, permute, [transpose](../../../../glossary.md#term-transpose), [slice](../../../../glossary.md#slice), select, or [contiguous](../../../../glossary.md#term-contiguous) operations
- deciding whether a layout must be materialized
- [planning](../../../../glossary.md#term-planning) costs, backend capabilities, [lowering](../../../../glossary.md#lowering), [kernels](../../../../glossary.md#kernel), or [runtime](../../../../glossary.md#term-runtime) execution
- [graph](../../../../glossary.md#graph)-wide shape/layout [inference](../../../../glossary.md#term-inference)
- modifying existing [data type](../../../../glossary.md#data-type--datatype), [dimension](../../../../glossary.md#dimension), or shape contracts
- [Gradle](../../../../glossary.md#term-gradle) or [dependency](../../../../glossary.md#term-dependency) changes

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially `modules/model` and `modules/planning` responsibilities
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>, especially the [layout](../../../../glossary.md#layout) baseline
- [Task 0002](0002-shape-and-dimension-model.md), which defines [`Shape`](../../../../glossary.md#shape)
- [Model master plan](../master-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>

## Legacy evidence

The read-only legacy implementation provides capability evidence through `tensor.TensorMetadata`, `tensor.layout.TensorShape`, `planning.descriptor.LayoutClass`, `backend.cpu.nativecpu.layout.TensorPhysicalView`, `BroadcastPlannerTest`, `TensorPhysicalViewTest`, and `StridedLayoutPlanningTest`.

Relevant legacy behavior includes:

- [row-major](../../../../glossary.md#term-row-major) [contiguous](../../../../glossary.md#term-contiguous) [element](../../../../glossary.md#term-element) [strides](../../../../glossary.md#term-stride);
- non-zero storage offsets for selected/sliced [views](../../../../glossary.md#term-view-meanings);
- non-contiguous strides for permuted views;
- zero strides for expanded/broadcast views;
- [defensive copying](../../../../glossary.md#memory-management-atomic-counters-and-complexity) of [shape](../../../../glossary.md#shape) and stride arrays;
- rejection of negative strides in supported native views; and
- separation between [layout](../../../../glossary.md#layout) description and later [materialization](../../../../glossary.md#materialization) decisions.

Legacy [runtime](../../../../glossary.md#term-runtime) and [backend](../../../../glossary.md#backend) [descriptors](../../../../glossary.md#term-descriptor) also mix [data type](../../../../glossary.md#data-type--datatype), storage family, byte length, backend-specific layout classes, and execution policy. Those responsibilities must not be copied into the new [model](../../../../glossary.md#term-model) descriptor.

## Architecture constraints

- All production packages use `io.github.pho001.synaptik.*`.
- `LayoutDescriptor` contains logical [element](../../../../glossary.md#term-element) geometry only and remains [backend](../../../../glossary.md#backend)- and storage-independent.
- Production code may use only the [JDK](../../../../glossary.md#term-java-development-kit-jdk) and local `modules/model` contracts.
- [Layout](../../../../glossary.md#layout) classification is a deterministic description of supplied [shape](../../../../glossary.md#shape)/[stride](../../../../glossary.md#term-stride) facts, not a [materialization](../../../../glossary.md#materialization) or [kernel](../../../../glossary.md#kernel)-selection decision.
- [Planning](../../../../glossary.md#term-planning) may later consume layout facts to derive logical materialization requirements; the [model](../../../../glossary.md#term-model) must not expose `requiresMaterialization()` or equivalent policy.
- Resolved numeric layout creation requires a fully static `Shape`. Dynamic layout resolution belongs to later [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime) contracts.
- Element strides and storage offsets use `long` and must be non-negative.
- A zero stride represents broadcast only when it repeats a [dimension](../../../../glossary.md#dimension) with static size greater than one. [Canonical](../../../../glossary.md#term-canonical-form) empty-[tensor](../../../../glossary.md#tensor) strides may contain zero without becoming broadcast layout.
- The legacy branch is read-only evidence. Do not copy its package structure or backend/runtime coupling.
- If implementation requires a change to `ARCHITECTURE.md`, stop and report the conflict instead of changing the contract.

## Required contracts

### [Layout](../../../../glossary.md#layout) kinds

Create a public `LayoutKind` enum with exactly:

- `DENSE_CONTIGUOUS` — [canonical](../../../../glossary.md#term-canonical-form) [row-major](../../../../glossary.md#term-row-major) [strides](../../../../glossary.md#term-stride) and zero [storage offset](../../../../glossary.md#term-storage-offset);
- `DENSE_WITH_OFFSET` — canonical row-major strides and a non-zero storage offset;
- `STRIDED` — resolved non-canonical non-broadcast strides; and
- `BROADCAST_ZERO_STRIDE` — at least one non-[singleton](../../../../glossary.md#term-singleton) logical [dimension](../../../../glossary.md#dimension) is repeated through stride zero.

Layout kind describes geometry only. Whether a [descriptor](../../../../glossary.md#term-descriptor) is a [view](../../../../glossary.md#term-view-meanings) is recorded separately.

### [Layout](../../../../glossary.md#layout) geometry

Create a package-private, stateless `LayoutGeometry` implementation helper used by `LayoutDescriptor`. It owns only deterministic checked calculations:

- [canonical](../../../../glossary.md#term-canonical-form) [row-major](../../../../glossary.md#term-row-major) [element](../../../../glossary.md#term-element) [strides](../../../../glossary.md#term-stride) for a fully static [`Shape`](../../../../glossary.md#shape);
- classification from shape, strides, and [storage offset](../../../../glossary.md#term-storage-offset);
- detection of raw zero strides and broadcast zero strides; and
- [referenced element span](../../../../glossary.md#referenced-element-span).

Canonical row-major stride calculation proceeds from the final [axis](../../../../glossary.md#term-axis) toward the first:

```text
Shape[]        -> []
Shape[2, 3]    -> [3, 1]
Shape[2, 1, 3] -> [3, 3, 1]
Shape[2, 0, 4] -> [0, 4, 1]
```

Only products needed to produce an actual stride are calculated. Arithmetic [overflow](../../../../glossary.md#term-overflow) fails with `ArithmeticException`. Fully static shapes with element counts larger than `long` may still have representable strides when no required stride product overflows.

Classification order is significant:

1. exact canonical strides produce `DENSE_CONTIGUOUS` or `DENSE_WITH_OFFSET` according to offset;
2. otherwise, stride zero on any [dimension](../../../../glossary.md#dimension) with size greater than one produces `BROADCAST_ZERO_STRIDE`;
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

All multiplication and addition use checked `long` arithmetic and fail with `ArithmeticException` on overflow. A shape containing a zero-sized dimension has span zero regardless of offset because it references no element. [Rank](../../../../glossary.md#term-rank)-0 [scalar](../../../../glossary.md#term-scalar) shape references one element.

### [Layout](../../../../glossary.md#layout) [descriptor](../../../../glossary.md#term-descriptor)

Create an [immutable](../../../../glossary.md#term-immutable) final `LayoutDescriptor` in `io.github.pho001.synaptik.model`.

Required creation paths:

- `LayoutDescriptor.contiguous(Shape shape)` creates a non-[view](../../../../glossary.md#term-view-meanings) [canonical](../../../../glossary.md#term-canonical-form) [row-major](../../../../glossary.md#term-row-major) descriptor with offset zero; and
- `LayoutDescriptor.of(Shape shape, long[] strides, long storageOffset, boolean view)` creates and classifies an explicit resolved descriptor.

Both creation paths require a non-null fully static [shape](../../../../glossary.md#shape). A dynamic shape fails with `IllegalArgumentException`. The explicit factory also requires:

- non-null [strides](../../../../glossary.md#term-stride);
- stride count equal to shape [rank](../../../../glossary.md#term-rank);
- every stride non-negative;
- non-negative [storage offset](../../../../glossary.md#term-storage-offset); and
- `view == true` when classification is `BROADCAST_ZERO_STRIDE`.

The descriptor stores:

- rank;
- immutable [element](../../../../glossary.md#term-element) strides;
- non-negative storage offset in elements;
- derived `LayoutKind`;
- explicit view/[alias](../../../../glossary.md#aliasing-and-arena) flag; and
- checked [referenced element span](../../../../glossary.md#referenced-element-span).

Required inspection includes:

- `rank()`;
- `kind()`;
- `strides()` returning a defensive `long[]` copy;
- `stride(int axis)` with positive and negative [axes](../../../../glossary.md#term-axis);
- `storageOffset()`;
- `isView()`;
- `isContiguous()` for both dense kinds;
- `hasStorageOffset()`;
- `hasZeroStride()` as a raw stride fact;
- `isBroadcast()` from the derived kind; and
- `referencedElementSpan()`.

Invalid axes fail with `IndexOutOfBoundsException`, including every axis for [scalar](../../../../glossary.md#term-scalar) rank zero. Descriptor equality and hashing are structural across all stored fields. Diagnostic text must include kind, rank, strides, offset, view flag, and span without becoming a [serialization](../../../../glossary.md#term-serialization) contract.

Resolved descriptors deliberately do not retain `Shape`. Task 0007 will pair shape and layout in `TensorDescriptor`, avoiding duplicate shape [ownership](../../../../glossary.md#term-ownership-meanings) while keeping layout values independently immutable.

## Affected files

Expected production files:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/LayoutKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/LayoutGeometry.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/LayoutDescriptor.java`

Expected test files:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/LayoutGeometryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/LayoutDescriptorTest.java`

Expected documentation/[planning](../../../../glossary.md#term-planning) updates:

- `docs/api/tensor-api.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

The capability baseline already records the selected [layout](../../../../glossary.md#layout) kinds, resolved-static boundary, and planning/[materialization](../../../../glossary.md#materialization) separation. If implementation evidence requires changing them, stop and explain the discrepancy before editing that baseline.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- five Java production/test files in `modules/model`; and
- the four documentation/[planning](../../../../glossary.md#term-planning) files listed above.

Do not modify existing [data type](../../../../glossary.md#data-type--datatype), [dimension](../../../../glossary.md#dimension), or [shape](../../../../glossary.md#shape) contracts, [Gradle](../../../../glossary.md#term-gradle) files, `ARCHITECTURE.md`, focused architecture documents, or another [module](../../../../glossary.md#term-module). If more files are necessary, stop and propose a follow-up task.

## Acceptance criteria

- `LayoutKind` contains exactly the four required [backend](../../../../glossary.md#backend)-independent values.
- [Canonical](../../../../glossary.md#term-canonical-form) [strides](../../../../glossary.md#term-stride) are correct for [scalar](../../../../glossary.md#term-scalar), ordinary, [singleton](../../../../glossary.md#term-singleton), and zero-sized static [shapes](../../../../glossary.md#shape).
- Canonical stride calculation detects every required `long` [overflow](../../../../glossary.md#term-overflow) without unnecessarily multiplying beyond the first-[axis](../../../../glossary.md#term-axis) stride.
- Explicit [descriptors](../../../../glossary.md#term-descriptor) reject null, dynamic, [rank](../../../../glossary.md#term-rank)-mismatched, negative-stride, negative-offset, and non-[view](../../../../glossary.md#term-view-meanings) broadcast inputs.
- [Layout](../../../../glossary.md#layout) classification follows the required order and covers dense, offset-dense, strided, broadcast, scalar, and canonical zero-[element](../../../../glossary.md#term-element) cases.
- `view` remains independent of dense/strided geometry, except that broadcast repetition requires a view.
- [Referenced element span](../../../../glossary.md#referenced-element-span) is correct for dense, offset, strided, broadcast, scalar, and empty layouts and detects checked arithmetic overflow.
- Descriptor arrays are defensively copied on input and output.
- Positive and negative stride-axis [lookup](../../../../glossary.md#term-lookup) covers every valid axis and rejects every invalid axis.
- Equality and hashing include kind, rank, strides, offset, view flag, and span.
- No [API](../../../../glossary.md#term-application-programming-interface-api) exposes byte-level, storage-[owner](../../../../glossary.md#term-owner), [device](../../../../glossary.md#term-device), [runtime](../../../../glossary.md#term-runtime), backend, [materialization](../../../../glossary.md#materialization)-policy, or [operation](../../../../glossary.md#operation) information.
- Unit tests cover every layout kind, constructor/factory [validation](../../../../glossary.md#term-validation), overflow, zero-sized shapes, scalar behavior, immutability, equality/hashing, and diagnostic text.
- All public types, enum [constants](../../../../glossary.md#term-constant), factories, and methods have detailed [Javadoc](../../../../glossary.md#term-javadoc) covering semantics, [parameters](../../../../glossary.md#term-parameter), results, units, [nullability](../../../../glossary.md#term-nullability), [ownership](../../../../glossary.md#term-ownership-meanings), immutability, and expected failures according to `AGENTS.md`. The package-private helper documents its implementation contract.
- `docs/api/tensor-api.md` documents resolved layout kinds, strides, element offset/span, view [metadata](../../../../glossary.md#term-metadata), dynamic-shape limitation, and the separation from materialization policy.
- No existing [model](../../../../glossary.md#term-model) contract, [Gradle](../../../../glossary.md#term-gradle) file, or architecture document is changed.

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
- exactly three production and two test files are added;
- existing [data type](../../../../glossary.md#data-type--datatype), [dimension](../../../../glossary.md#dimension), and [shape](../../../../glossary.md#shape) source files are unchanged;
- no storage, [compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), [runtime](../../../../glossary.md#term-runtime), [prepare](../../../../glossary.md#prepare), [engine](../../../../glossary.md#term-engine), or [backend](../../../../glossary.md#backend) [dependency](../../../../glossary.md#term-dependency) is introduced;
- the [descriptor](../../../../glossary.md#term-descriptor) cannot be mutated through supplied or returned arrays;
- the task, master-plan row, and roadmap row have matching final statuses; and
- documentation and [Javadoc](../../../../glossary.md#term-javadoc) describe the new model rather than legacy backend/runtime descriptors.

## Dependencies

- Task 0002: [Shape](../../../../glossary.md#shape) and [dimension](../../../../glossary.md#dimension) [model](../../../../glossary.md#term-model) — complete.

## Follow-up tasks

- Task 0003C: [Layout](../../../../glossary.md#layout) package migration moves the completed layout contracts into the package defined by the [model](../../../../glossary.md#term-model) master plan without changing behavior.
- Task 0007: [Tensor descriptor](../../../../glossary.md#tensor-descriptor) model pairs [`DataType`](../../../../glossary.md#data-type--datatype), [`Shape`](../../../../glossary.md#shape), and `LayoutDescriptor` and validates their [rank](../../../../glossary.md#term-rank) compatibility.
- Task 0017: Layout and [view](../../../../glossary.md#term-view-meanings) operations create [descriptors](../../../../glossary.md#term-descriptor) for reshape, expand, permute, [slice](../../../../glossary.md#slice), select, and [contiguous](../../../../glossary.md#term-contiguous) semantics.
- `modules/planning` later derives logical [materialization](../../../../glossary.md#materialization) requirements from layout facts and [backend](../../../../glossary.md#backend) capabilities.
- [Concrete backend](../../../../glossary.md#concrete-backend) [prepare](../../../../glossary.md#prepare) implementations later translate logical [element](../../../../glossary.md#term-element) layouts into backend storage and executable routes.

Do not create detailed specifications for these follow-ups until task 0003 is complete and the [planning](../../../../glossary.md#term-planning) frontier advances.

## Architecture impact

Expected impact: None.

This task implements the explicitly allowed `LayoutDescriptor` [model](../../../../glossary.md#term-model) responsibility. Resolved-static [layout](../../../../glossary.md#layout) geometry and explicit [view](../../../../glossary.md#term-view-meanings) [metadata](../../../../glossary.md#term-metadata) are local design decisions within the existing boundary. If implementation requires architecture changes, stop and report the issue.

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
  copied stride array, so no public or package-private path retains caller-owned mutable state.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Canonical stride classification is evaluated before broadcast detection exactly as specified;
  therefore canonical empty shapes remain dense even when their raw stride list contains zero.<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Axis normalization is implemented locally because a resolved descriptor intentionally does not
  retain its source <span data-glossary-navigation="true"> ([Axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>[`Shape`](../../../../glossary.md#shape).
- The package-private helper extracts static sizes through the existing `Shape` contract and uses
  `Math.multiplyExact` and `Math.addExact` for every required geometric product and sum.

## Known limitations

- Numeric descriptors require fully static shapes; symbolic strides and later binding of dynamic
  dimensions are not represented.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [shapes glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding); [dimensions glossary](../../../../glossary.md#dimension))</span>
- [Strides](../../../../glossary.md#term-stride) and offsets are non-negative. Negative-stride reverse [views](../../../../glossary.md#term-view-meanings) are not supported.
- The descriptor records geometry and explicit view metadata only. It cannot validate a span against
  storage capacity because storage is outside this task and model contract.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [model glossary](../../../../glossary.md#term-model))</span>
- [Layout](../../../../glossary.md#layout) operations and [materialization](../../../../glossary.md#materialization) decisions remain assigned to their follow-up tasks.

## [Validation](../../../../glossary.md#term-validation) evidence

- `./gradlew :modules:model:test` — passed with 56 tests, including 17 [layout](../../../../glossary.md#layout) tests.
- `./gradlew :modules:model:javadoc` — passed without [Javadoc](../../../../glossary.md#term-javadoc) errors.
- `./gradlew test` — passed for the full multi-[module](../../../../glossary.md#term-module) project.
- `git diff --check` — passed.
- Manual scope review confirmed exactly three new production and two new test files, production
  imports limited to <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`java.*` and local model types, and no changes to existing data type,
  dimension, or shape sources, Gradle files, architecture documents, or other modules.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [data type glossary](../../../../glossary.md#data-type--datatype); [dimension glossary](../../../../glossary.md#dimension); [shape glossary](../../../../glossary.md#shape); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Manual immutability review and tests confirmed defensive copying on both input and output stride
  arrays.<span data-glossary-navigation="true"> ([defensive copying glossary](../../../../glossary.md#memory-management-atomic-counters-and-complexity); [stride glossary](../../../../glossary.md#term-stride))</span>

## Implementation notes

- Added the four-value `LayoutKind` geometry taxonomy.
- Added package-private `LayoutGeometry` for canonical strides, ordered classification, raw and
  broadcast zero-stride detection, and checked referenced-span calculation.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [strides glossary](../../../../glossary.md#term-stride))</span>
- Added [immutable](../../../../glossary.md#term-immutable) `LayoutDescriptor` factories and inspection methods, structural equality and
  hashing, and diagnostic text.<span data-glossary-navigation="true"> ([structural equality glossary](../../../../glossary.md#term-structural-equality))</span>
- Added tests for all layout kinds, scalar and empty layouts, offset and strided spans, validation,
  overflow, axis normalization, array isolation, equality, hashing, and diagnostics.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [validation glossary](../../../../glossary.md#term-validation); [overflow glossary](../../../../glossary.md#term-overflow); [axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Extended the tensor API reference with the resolved-static layout contract and its separation
  from storage and materialization policy.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [materialization glossary](../../../../glossary.md#materialization))</span>

## Completion summary

Implemented the resolved [layout](../../../../glossary.md#layout) [descriptor](../../../../glossary.md#term-descriptor) foundation inside `modules/model` without adding storage,
operations, [compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), [runtime](../../../../glossary.md#term-runtime), or [backend](../../../../glossary.md#backend) behavior. All acceptance criteria and requested
[validation](../../../../glossary.md#term-validation) passed. The [model](../../../../glossary.md#term-model) master plan and implementation roadmap mark task 0003 complete. The
subsequently planned task 0003A is the current package-migration frontier; task 0003C will move these
layout contracts before task 0004 introduces new model types.

Status: Complete
