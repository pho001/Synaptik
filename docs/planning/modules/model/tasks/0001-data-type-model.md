# Task 0001: [DataType](../../../../glossary.md#data-type--datatype) [Model](../../../../glossary.md#term-model)

## Status

Complete

## Goal

Implement the [backend](../../../../glossary.md#backend)-independent [data type](../../../../glossary.md#data-type--datatype) foundation for `modules/model`. Define the six initial Synaptik data types, their stable category and width [metadata](../../../../glossary.md#term-metadata), floating data type promotion, and deterministic [BFLOAT16](../../../../glossary.md#term-bfloat16) bit conversion without introducing storage, [tensor](../../../../glossary.md#tensor), [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), or backend behavior.

## [Scope](../../../../glossary.md#term-scope)

- Define the [data type](../../../../glossary.md#data-type--datatype) categories `FLOATING`, `INTEGRAL`, and [`BOOLEAN`](../../../../glossary.md#data-type--datatype).
- Define exactly these initial data types: [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`INT32`](../../../../glossary.md#data-type--datatype), [`INT64`](../../../../glossary.md#data-type--datatype), and [`BOOL`](../../../../glossary.md#data-type--datatype).
- Expose [immutable](../../../../glossary.md#term-immutable) category, logical bit width, storage byte width, and differentiability [metadata](../../../../glossary.md#term-metadata).
- Expose category predicates and the default floating data type (`FLOAT32`).
- Implement floating-only promotion using `BFLOAT16 < FLOAT32 < FLOAT64`.
- Implement [backend](../../../../glossary.md#backend)-neutral conversion between Java `float` values and raw BFLOAT16 `short` bits.
- Remove the `ModelModule` placeholder when the first real public [model](../../../../glossary.md#term-model) [API](../../../../glossary.md#term-application-programming-interface-api) is introduced.
- Add focused unit tests and update the data type section of the [tensor](../../../../glossary.md#tensor) API documentation.

## Out of [scope](../../../../glossary.md#term-scope)

- [tensor](../../../../glossary.md#tensor) or [shape](../../../../glossary.md#shape) types
- array, memory-segment, [device](../../../../glossary.md#term-device), or other storage implementations
- [scalar](../../../../glossary.md#term-scalar)/array read and write APIs
- cast operations or cast execution
- promotion between floating, integral, and [boolean](../../../../glossary.md#data-type--datatype) categories
- arithmetic, comparison, reduction, or other [operation](../../../../glossary.md#operation) semantics
- [graph](../../../../glossary.md#graph) [inference](../../../../glossary.md#term-inference) or graph [validation](../../../../glossary.md#term-validation)
- [autograd](../../../../glossary.md#autograd) rules
- [compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), [prepare](../../../../glossary.md#prepare), [runtime](../../../../glossary.md#term-runtime), [engine](../../../../glossary.md#term-engine), or [backend](../../../../glossary.md#backend) integration
- [`FLOAT16`](../../../../glossary.md#term-float16), unsigned, quantized, complex, string, or sparse data types
- [Gradle](../../../../glossary.md#term-gradle) or [dependency](../../../../glossary.md#term-dependency) changes

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially `modules/model` [ownership](../../../../glossary.md#term-ownership-meanings) and forbidden dependencies
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>, especially the [DataType](../../../../glossary.md#data-type--datatype) baseline
- [Model master plan](../master-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>

## Architecture constraints

- All production packages use `io.github.pho001.synaptik.*`.
- [Data type](../../../../glossary.md#data-type--datatype) contracts belong to `modules/model` and remain [backend](../../../../glossary.md#backend)-independent.
- Production code may use only the [JDK](../../../../glossary.md#term-java-development-kit-jdk) and must not introduce a project-[module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency).
- No data type [API](../../../../glossary.md#term-application-programming-interface-api) may expose backend support, [backend identity](../../../../glossary.md#backend-identity--backendid), [device](../../../../glossary.md#term-device) [residency](../../../../glossary.md#residency), physical storage, [kernel](../../../../glossary.md#kernel) selection, [runtime](../../../../glossary.md#term-runtime) state, or [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution).
- Do not add `supportedBackends()` or an equivalent [backend capability](../../../../glossary.md#backend-capability) API.
- [BFLOAT16](../../../../glossary.md#term-bfloat16) conversion is a value-format utility; it must not allocate or own [tensor](../../../../glossary.md#tensor) storage.
- The legacy branch is read-only capability evidence. Do not copy its package structure or source implementation.
- If implementation requires a change to `ARCHITECTURE.md`, stop and report the conflict instead of changing the contract.

## Required contracts

### [DataType](../../../../glossary.md#data-type--datatype) category

Provide an [immutable](../../../../glossary.md#term-immutable) public category type with exactly:

- `FLOATING`
- `INTEGRAL`
- [`BOOLEAN`](../../../../glossary.md#data-type--datatype)

### [DataType](../../../../glossary.md#data-type--datatype) [metadata](../../../../glossary.md#term-metadata)

Each [data type](../../../../glossary.md#data-type--datatype) exposes the following stable metadata:

| DataType | Category | Bit width | Storage bytes | [Differentiable](../../../../glossary.md#derivative-and-chain-rule) |
|---|---:|---:|---:|---:|
| [`FLOAT64`](../../../../glossary.md#term-float64) | `FLOATING` | 64 | 8 | yes |
| [`FLOAT32`](../../../../glossary.md#term-float32) | `FLOATING` | 32 | 4 | yes |
| [`BFLOAT16`](../../../../glossary.md#term-bfloat16) | `FLOATING` | 16 | 2 | yes |
| [`INT32`](../../../../glossary.md#data-type--datatype) | `INTEGRAL` | 32 | 4 | no |
| [`INT64`](../../../../glossary.md#data-type--datatype) | `INTEGRAL` | 64 | 8 | no |
| [`BOOL`](../../../../glossary.md#data-type--datatype) | [`BOOLEAN`](../../../../glossary.md#data-type--datatype) | 8 | 1 | no |

The [API](../../../../glossary.md#term-application-programming-interface-api) must support category predicates equivalent to `isFloating()`, `isIntegral()`, and `isBoolean()`, plus an explicit differentiability query. `FLOAT32` is the initial default floating data type.

Do not expose Java array classes, memory segments, native carriers, [device](../../../../glossary.md#term-device) formats, or [backend](../../../../glossary.md#backend)-specific data type codes in this task. Those mappings belong to later storage and backend work.

### Floating promotion

Floating promotion is symmetric and follows:

```text
BFLOAT16 < FLOAT32 < FLOAT64
```

Promotion of two equal floating data types returns that [data type](../../../../glossary.md#data-type--datatype). The public promotion entry point is `DataTypePromotion.promoteFloating(DataType left, DataType right)`. It throws `NullPointerException` when either input is null and `IllegalArgumentException` when either input is integral or [boolean](../../../../glossary.md#data-type--datatype). It does not define implicit cross-category conversion.

### [BFLOAT16](../../../../glossary.md#term-bfloat16) conversion

Provide conversions equivalent to:

```text
float -> raw BFLOAT16 short bits
raw BFLOAT16 short bits -> float
```

The conversion from `float` must use [round-to-nearest, ties-to-even](../../../../glossary.md#floating-point-precision-and-rounding) for [finite](../../../../glossary.md#term-finite) values. [Signed zero](../../../../glossary.md#term-signed-zero) and [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) must be preserved. [NaN](../../../../glossary.md#term-nan) input must produce the [canonical](../../../../glossary.md#term-canonical-form) quiet BFLOAT16 NaN bit pattern `0x7FC0`; converting BFLOAT16 NaN bits back must yield a Java NaN. The [scalar](../../../../glossary.md#term-scalar) entry points are `BFloat16Bits.fromFloat(float value)` and `BFloat16Bits.toFloat(short bits)`.

The helper must operate on scalar values only. Bulk conversion belongs to a later storage task.

## Affected files

Expected production files:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/DataTypeCategory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/DataType.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/DataTypePromotion.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/BFloat16Bits.java`
- remove `modules/model/src/main/java/io/github/pho001/synaptik/model/ModelModule.java`

Expected test files:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/DataTypeTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/DataTypePromotionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/BFloat16BitsTest.java`

Expected documentation/[planning](../../../../glossary.md#term-planning) updates:

- `docs/api/tensor-api.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

If implementation evidence requires changing the capability baseline, stop and explain the discrepancy before editing it.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create, modify, or remove at most:

- eight Java production/test files in `modules/model` in total; and
- the four documentation/[planning](../../../../glossary.md#term-planning) files listed above.

Do not modify [Gradle](../../../../glossary.md#term-gradle) files, `ARCHITECTURE.md`, focused architecture documents, or another [module](../../../../glossary.md#term-module). If more files are necessary, stop and propose a follow-up task.

## Acceptance criteria

- The six required [data type](../../../../glossary.md#data-type--datatype) [constants](../../../../glossary.md#term-constant) exist and no additional data type is introduced.
- Every data type reports the exact category, bit width, storage byte width, and differentiability listed in this specification.
- Category predicates are mutually consistent for all data type constants.
- The default floating data type is [`FLOAT32`](../../../../glossary.md#term-float32).
- Floating promotion is exhaustive, symmetric, and [idempotent](../../../../glossary.md#term-idempotent) across the three floating data types.
- Floating promotion rejects null inputs with `NullPointerException` and integral or [boolean](../../../../glossary.md#data-type--datatype) inputs with `IllegalArgumentException`.
- [BFLOAT16](../../../../glossary.md#term-bfloat16) conversion correctly covers ordinary [finite](../../../../glossary.md#term-finite) values, values requiring [rounding](../../../../glossary.md#term-rounding), exact ties, [signed zero](../../../../glossary.md#term-signed-zero), positive/[negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero), and [NaN](../../../../glossary.md#term-nan).
- No production class imports another Synaptik [module](../../../../glossary.md#term-module) or exposes [backend](../../../../glossary.md#backend)/[runtime](../../../../glossary.md#term-runtime)/storage concepts.
- The placeholder `ModelModule` class is removed.
- Unit tests cover every enum constant and all promotion pairs.
- All new public types, constants, constructors, and methods have detailed [Javadoc](../../../../glossary.md#term-javadoc) that documents semantics, [parameters](../../../../glossary.md#term-parameter), return values, [nullability](../../../../glossary.md#term-nullability), constraints, and expected failures according to `AGENTS.md`.
- `docs/api/tensor-api.md` documents the initial data type set, categories, promotion boundary, differentiability, and BFLOAT16 representation without claiming backend support.
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

- production imports remain limited to `java.*` and the local [model](../../../../glossary.md#term-model) package;
- no [backend](../../../../glossary.md#backend)/[runtime](../../../../glossary.md#term-runtime)/storage vocabulary appears in public [data type](../../../../glossary.md#data-type--datatype) contracts;
- the task, master-plan row, and roadmap row have matching final statuses; and
- documentation and [Javadoc](../../../../glossary.md#term-javadoc) describe the implemented behavior rather than the legacy implementation.

## Dependencies

- No implementation task [dependency](../../../../glossary.md#term-dependency).
- Requires the repository skeleton and [model](../../../../glossary.md#term-model) capability baseline already present.

## Follow-up tasks

- Task 0002: [Shape](../../../../glossary.md#shape) and [dimension](../../../../glossary.md#dimension) [model](../../../../glossary.md#term-model).
- Task 0003A: [Data type](../../../../glossary.md#data-type--datatype) package migration moves these completed contracts into the package defined by the model master plan without changing behavior.
- Task 0005: [Operation](../../../../glossary.md#operation) semantic foundation consumes data type categories and promotion semantics.
- Task 0010: [Host storage](../../../../glossary.md#host-storage) abstraction consumes data type width and [BFLOAT16](../../../../glossary.md#term-bfloat16) representation.

Do not create detailed specifications for these follow-ups until task 0001 is complete and the [planning](../../../../glossary.md#term-planning) frontier advances.

## Architecture impact

Expected impact: None.

This task implements an explicitly allowed `modules/model` responsibility. If it requires architecture changes, stop and report the issue.

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are working in the Synaptik repository.

Read first:
- AGENTS.md
- ARCHITECTURE.md
- docs/planning/planning-guide.md
- docs/planning/modules/model/capabilities.md
- docs/planning/modules/model/tasks/0001-data-type-model.md

Implement task 0001 exactly as specified.

Keep the change inside modules/model and the explicitly listed documentation files. Do not implement Tensor, storage, cast operations, compiler behavior, runtime behavior, or backend behavior. Do not modify Gradle files or ARCHITECTURE.md. Do not copy legacy source code.

Add complete Javadoc for every affected public Java contract. Run every validation command from the task specification.

At the end, update this task file with local decisions, known limitations, validation evidence, implementation notes, completion summary, and final status. Synchronize the task status in the model master plan and implementation roadmap.
```

## Local decisions

- [`DataType`](../../../../glossary.md#data-type--datatype), `DataTypeCategory`, `DataTypePromotion`, and `BFloat16Bits` were initially created as small public contracts in the root `io.github.pho001.synaptik.model` package. The later package-structure review supersedes that placement through task 0003A without changing their behavior.
- `DataType` stores category, bit width, byte width, and differentiability directly in each enum [constant](../../../../glossary.md#term-constant). `byteWidth()` is derived once during enum initialization because every initial [data type](../../../../glossary.md#data-type--datatype) is byte-aligned.
- The default is exposed through `DataType.defaultFloating()` instead of a [mutable](../../../../glossary.md#term-mutable) or [backend](../../../../glossary.md#backend)-configured default.
- `DataTypePromotion` supports floating promotion only. Null operands fail with `NullPointerException`; non-floating operands fail with `IllegalArgumentException`.
- `BFloat16Bits` is a [scalar](../../../../glossary.md#term-scalar), allocation-free utility. It uses [round-to-nearest](../../../../glossary.md#floating-point-precision-and-rounding) with ties to even and canonicalizes every Java [NaN](../../../../glossary.md#term-nan) to raw [BFLOAT16](../../../../glossary.md#term-bfloat16) bits `0x7FC0`.
- Java [carrier](../../../../glossary.md#term-storage-carrier) and array classes are intentionally absent from `DataType`; storage mappings remain deferred to task 0010.

## Known limitations

- Only the six data types in the initial capability baseline are represented.
- Cross-category and integral promotion are intentionally unsupported; a later explicit cast [operation](../../../../glossary.md#operation) owns conversion semantics.
- [BFLOAT16](../../../../glossary.md#term-bfloat16) bulk conversion and storage [ownership](../../../../glossary.md#term-ownership-meanings) are not included in this task.
- [Backend](../../../../glossary.md#backend) support and physical [data type](../../../../glossary.md#data-type--datatype) mappings remain outside the [model](../../../../glossary.md#term-model) contract.

## [Validation](../../../../glossary.md#term-validation) evidence

- `./gradlew :modules:model:test` — passed. 15 tests executed: 6 `DataTypeTest`, 3 `DataTypePromotionTest`, and 6 `BFloat16BitsTest`; no failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — passed. Public [API](../../../../glossary.md#term-application-programming-interface-api) [Javadoc](../../../../glossary.md#term-javadoc) generated without errors.
- `./gradlew test` — passed. The complete repository test [lifecycle](../../../../glossary.md#lifecycle) completed successfully with 36 actionable tasks in the final [run](../../../../glossary.md#term-run).
- `git diff --check` — passed before task closure; repeated after the final [planning](../../../../glossary.md#term-planning) update.
- Markdown link validation — passed for all 72 repository Markdown files.
- Manual [dependency](../../../../glossary.md#term-dependency) review — production code imports only `java.util.Objects`; no project-[module](../../../../glossary.md#term-module), [backend](../../../../glossary.md#backend), [runtime](../../../../glossary.md#term-runtime), [prepare](../../../../glossary.md#prepare), [engine](../../../../glossary.md#term-engine), or storage implementation dependency was added.
- Manual [scope](../../../../glossary.md#term-scope) review — exactly eight Java changes: four production files created, three test files created, and the placeholder `ModelModule` removed.
- [Gradle](../../../../glossary.md#term-gradle) and architecture review — no Gradle file, `ARCHITECTURE.md`, or focused architecture document changed.

## Implementation notes

- Added [immutable](../../../../glossary.md#term-immutable) category and [data type](../../../../glossary.md#data-type--datatype) [metadata](../../../../glossary.md#term-metadata) contracts for the six initial data types.
- Added exhaustive floating promotion behavior and rejection tests for null, integral, and [boolean](../../../../glossary.md#data-type--datatype) operands.
- Added [BFLOAT16](../../../../glossary.md#term-bfloat16) conversion coverage for exact values, both [rounding](../../../../glossary.md#term-rounding) directions, [ties-to-even](../../../../glossary.md#term-ties-to-even), [signed zero](../../../../glossary.md#term-signed-zero), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), [subnormal](../../../../glossary.md#term-subnormal) round trips, and [NaN](../../../../glossary.md#term-nan) [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).
- Removed `ModelModule` because the [module](../../../../glossary.md#term-module) now exposes real public [model](../../../../glossary.md#term-model) contracts.
- Replaced the [tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) placeholder with the implemented data type, promotion, and BFLOAT16 documentation.

## Completion summary

- Completed changes: Implemented the [backend](../../../../glossary.md#backend)-independent [`DataType`](../../../../glossary.md#data-type--datatype) foundation, floating promotion, and [BFLOAT16](../../../../glossary.md#term-bfloat16) [scalar](../../../../glossary.md#term-scalar) conversion.
- Files changed or created: Four production classes, three unit-test classes, one removed placeholder, the [tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) documentation, and synchronized [planning](../../../../glossary.md#term-planning) documents.
- Tests and [validation](../../../../glossary.md#term-validation): All required [Gradle](../../../../glossary.md#term-gradle) commands, [Javadoc](../../../../glossary.md#term-javadoc) generation, diff checks, link checks, [dependency](../../../../glossary.md#term-dependency) review, and [scope](../../../../glossary.md#term-scope) review passed.
- Documentation impact: Added the initial public [data type](../../../../glossary.md#data-type--datatype) API reference; no architecture documentation change was required.
- Javadoc review: Every public type, enum [constant](../../../../glossary.md#term-constant), and method documents semantics, [parameters](../../../../glossary.md#term-parameter), results, constraints, and failures where applicable.
- Unresolved issues: None.
- Follow-up required: None. Task 0002 is the next planned frontier.

Status: Complete
