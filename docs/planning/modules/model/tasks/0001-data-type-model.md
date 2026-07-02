# Task 0001: DataType Model

## Status

Ready

## Goal

Implement the backend-independent data type foundation for `modules/model`. Define the six initial Synaptik data types, their stable category and width metadata, floating data type promotion, and deterministic BFLOAT16 bit conversion without introducing storage, tensor, compiler, runtime, or backend behavior.

## Scope

- Define the data type categories `FLOATING`, `INTEGRAL`, and `BOOLEAN`.
- Define exactly these initial data types: `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, and `BOOL`.
- Expose immutable category, logical bit width, storage byte width, and differentiability metadata.
- Expose category predicates and the default floating data type (`FLOAT32`).
- Implement floating-only promotion using `BFLOAT16 < FLOAT32 < FLOAT64`.
- Implement backend-neutral conversion between Java `float` values and raw BFLOAT16 `short` bits.
- Remove the `ModelModule` placeholder when the first real public model API is introduced.
- Add focused unit tests and update the data type section of the tensor API documentation.

## Out of scope

- tensor or shape types
- array, memory-segment, device, or other storage implementations
- scalar/array read and write APIs
- cast operations or cast execution
- promotion between floating, integral, and boolean categories
- arithmetic, comparison, reduction, or other operation semantics
- graph inference or graph validation
- autograd rules
- compiler, planning, prepare, runtime, engine, or backend integration
- `FLOAT16`, unsigned, quantized, complex, string, or sparse data types
- Gradle or dependency changes

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially `modules/model` ownership and forbidden dependencies
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Model capability baseline](../capabilities.md), especially the DataType baseline
- [Model master plan](../master-plan.md)
- [Planning guide](../../../planning-guide.md)

## Architecture constraints

- All production packages use `io.github.pho001.synaptik.*`.
- Data type contracts belong to `modules/model` and remain backend-independent.
- Production code may use only the JDK and must not introduce a project-module dependency.
- No data type API may expose backend support, backend identity, device residency, physical storage, kernel selection, runtime state, or prepared execution.
- Do not add `supportedBackends()` or an equivalent backend capability API.
- BFLOAT16 conversion is a value-format utility; it must not allocate or own tensor storage.
- The legacy branch is read-only capability evidence. Do not copy its package structure or source implementation.
- If implementation requires a change to `ARCHITECTURE.md`, stop and report the conflict instead of changing the contract.

## Required contracts

### DataType category

Provide an immutable public category type with exactly:

- `FLOATING`
- `INTEGRAL`
- `BOOLEAN`

### DataType metadata

Each data type exposes the following stable metadata:

| DataType | Category | Bit width | Storage bytes | Differentiable |
|---|---:|---:|---:|---:|
| `FLOAT64` | `FLOATING` | 64 | 8 | yes |
| `FLOAT32` | `FLOATING` | 32 | 4 | yes |
| `BFLOAT16` | `FLOATING` | 16 | 2 | yes |
| `INT32` | `INTEGRAL` | 32 | 4 | no |
| `INT64` | `INTEGRAL` | 64 | 8 | no |
| `BOOL` | `BOOLEAN` | 8 | 1 | no |

The API must support category predicates equivalent to `isFloating()`, `isIntegral()`, and `isBoolean()`, plus an explicit differentiability query. `FLOAT32` is the initial default floating data type.

Do not expose Java array classes, memory segments, native carriers, device formats, or backend-specific data type codes in this task. Those mappings belong to later storage and backend work.

### Floating promotion

Floating promotion is symmetric and follows:

```text
BFLOAT16 < FLOAT32 < FLOAT64
```

Promotion of two equal floating data types returns that data type. The public promotion entry point is `DataTypePromotion.promoteFloating(DataType left, DataType right)`. It throws `NullPointerException` when either input is null and `IllegalArgumentException` when either input is integral or boolean. It does not define implicit cross-category conversion.

### BFLOAT16 conversion

Provide conversions equivalent to:

```text
float -> raw BFLOAT16 short bits
raw BFLOAT16 short bits -> float
```

The conversion from `float` must use round-to-nearest, ties-to-even for finite values. Signed zero and infinities must be preserved. NaN input must produce the canonical quiet BFLOAT16 NaN bit pattern `0x7FC0`; converting BFLOAT16 NaN bits back must yield a Java NaN. The scalar entry points are `BFloat16Bits.fromFloat(float value)` and `BFloat16Bits.toFloat(short bits)`.

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

Expected documentation/planning updates:

- `docs/api/tensor-api.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

If implementation evidence requires changing the capability baseline, stop and explain the discrepancy before editing it.

## Maximum scope

This task may create, modify, or remove at most:

- eight Java production/test files in `modules/model` in total; and
- the four documentation/planning files listed above.

Do not modify Gradle files, `ARCHITECTURE.md`, focused architecture documents, or another module. If more files are necessary, stop and propose a follow-up task.

## Acceptance criteria

- The six required data type constants exist and no additional data type is introduced.
- Every data type reports the exact category, bit width, storage byte width, and differentiability listed in this specification.
- Category predicates are mutually consistent for all data type constants.
- The default floating data type is `FLOAT32`.
- Floating promotion is exhaustive, symmetric, and idempotent across the three floating data types.
- Floating promotion rejects null inputs with `NullPointerException` and integral or boolean inputs with `IllegalArgumentException`.
- BFLOAT16 conversion correctly covers ordinary finite values, values requiring rounding, exact ties, signed zero, positive/negative infinity, and NaN.
- No production class imports another Synaptik module or exposes backend/runtime/storage concepts.
- The placeholder `ModelModule` class is removed.
- Unit tests cover every enum constant and all promotion pairs.
- All new public types, constants, constructors, and methods have detailed Javadoc that documents semantics, parameters, return values, nullability, constraints, and expected failures according to `AGENTS.md`.
- `docs/api/tensor-api.md` documents the initial data type set, categories, promotion boundary, differentiability, and BFLOAT16 representation without claiming backend support.
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

- production imports remain limited to `java.*` and the local model package;
- no backend/runtime/storage vocabulary appears in public data type contracts;
- the task, master-plan row, and roadmap row have matching final statuses; and
- documentation and Javadoc describe the implemented behavior rather than the legacy implementation.

## Dependencies

- No implementation task dependency.
- Requires the repository skeleton and model capability baseline already present.

## Follow-up tasks

- Task 0002: Shape and dimension model.
- Task 0005: Operation taxonomy and attribute foundation consumes data type categories and promotion semantics.
- Task 0011: Host storage abstraction consumes data type width and BFLOAT16 representation.

Do not create detailed specifications for these follow-ups until task 0001 is complete and the planning frontier advances.

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

Empty until implemented.

## Known limitations

Empty until implemented.

## Validation evidence

Empty until implemented.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
