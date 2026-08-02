# OpenBLAS Provider Master Plan

## Goal

Provide a low-level OpenBLAS leaf for library loading, symbol binding, GEMM calls, and thread control.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- native library resolution
- symbol binding
- GEMM invocation
- OpenBLAS thread control

## Out of scope

- configuration interpretation
- fallback and backend ownership
- prepared execution
- Tensor API and residency

## Module invariants

- The provider remains a low-level leaf.
- Dependency direction is CPU backend to OpenBLAS provider, never the reverse.

## Allowed dependencies

- JDK standard library and required native interop APIs.

## Forbidden dependencies

- model Tensor API
- planning, compiler, runtime, prepare, engine, and concrete backend dependencies

## Package structure

```text
io.github.pho001.synaptik.backend.provider.openblas/
  <root>  small public library-lifetime and later low-level GEMM/thread-control surface, with
          package-private JDK FFM loading, symbol binding, and native handles
```

The root package is deliberate: task 0001 opens one small public lifetime boundary, and task 0002
adds only its inseparable low-level calls plus one package-private field-free invocation helper.
Native implementation state remains package-private. The package must not grow into a generic
native framework, registry, manager, configuration package, or CPU route owner.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Library loading and required symbol binding](tasks/0001-library-loading-and-required-symbol-binding.md) | Complete | Prepare 0003; JDK 26 FFM and OpenBLAS C ABI evidence | Load one caller-specified library, bind the exact FLOAT32/FLOAT64 GEMM and get/set thread-count symbols, and own their closeable lookup lifetime without invocation or policy. |
| 0002 | [FLOAT32/FLOAT64 row-major GEMM invocation](tasks/0002-float32-float64-row-major-gemm-invocation.md) | Complete | 0001 | Add validated dense row-major no-transpose `MemorySegment` GEMM calls over the task-0001 lifetime without Tensor, route, allocation, or fallback behavior. |
| 0003 | Thread control and native provider checkpoint | Draft | 0001–0002 | Add low-level thread-count query/control and explicit compatible-library validation, then close the selected provider capability milestone. |

## Milestones

- Library and symbols
- GEMM contract
- Thread control and native validation

## Current status

In progress after completion of
[task 0002](tasks/0002-float32-float64-row-major-gemm-invocation.md). The provider now has explicit
caller-directed loading, complete required-symbol binding, caller-owned lookup lifetime, and
validated FLOAT32/FLOAT64 dense row-major no-transpose GEMM invocation. Task 0003 remains a Draft
row without a detailed specification and owns thread control plus the explicitly supplied native
provider checkpoint. No CPU task is Ready.

## Open questions

- The later thread-control task must account for OpenBLAS thread count as process/library-global
  native state and define its checkpoint environment without moving thread choice into the
  provider.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- CPU owns OpenBLAS route configuration and candidate generation, including thread choices. This
  provider exposes only low-level calls and thread control; it owns no tuning cache, workload
  signature, objective, budget, or candidate policy.
- Task 0001 accepts only a caller-specified library name or absolute path. It does not read config,
  environment variables, system properties, probe platform filenames, or decide fallback.
- Task 0001 requires `cblas_sgemm`, `cblas_dgemm`, `openblas_set_num_threads`, and
  `openblas_get_num_threads` under the standard 32-bit-`blasint` C ABI. It selects no minimum
  OpenBLAS version, ILP64 support, or optional BFLOAT16 symbols.
- One caller-owned public library handle encapsulates the shared FFM arena and package-private
  bound handles. No eager singleton, global cache, exposed native address, registry, manager, or
  service locator is planned.
- Task 0002 adds exactly `sgemm` and `dgemm` to that public lifetime owner for dense row-major
  no-transpose matrices. It derives leading dimensions internally and leaves batching,
  broadcasting, transpose/layout conversion, packing, loops, route choice, and fallback to CPU.
- One package-private field-free invocation helper owns ordered segment validation and exact typed
  handle calls. The existing bindings/native-access path remains the sole deterministic test seam.
- Task 0002 validates completely before output-empty no-op, invokes positive-output `k == 0`
  GEMM for `beta * C`, forwards all scalar bits unchanged, rejects C/input overlap, and adds no
  close-race coordination or native numerical guarantee.

## Risks

- Allowing CPU backend policy or fallback logic into the provider.
- Treating caller-directed native loading as platform discovery or hiding a default candidate
  search in the provider.
- Exposing native handles or failing to close a partial lookup after required-symbol binding
  fails.
- Letting a low-level GEMM call grow offsets, transpose/layout normalization, batching, packing,
  allocation, numerical policy, or CPU-owned route behavior.
- Treating separate Java library handles as separate OpenBLAS thread-control state.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
