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

The root package is deliberate: task 0001 opens only one small public lifetime boundary and keeps
its native implementation package-private. It must not grow into a generic native framework,
registry, manager, configuration package, or CPU route owner.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Library loading and required symbol binding](tasks/0001-library-loading-and-required-symbol-binding.md) | Complete | Prepare 0003; JDK 26 FFM and OpenBLAS C ABI evidence | Load one caller-specified library, bind the exact FLOAT32/FLOAT64 GEMM and get/set thread-count symbols, and own their closeable lookup lifetime without invocation or policy. |
| 0002 | FLOAT32/FLOAT64 row-major GEMM invocation | Draft | 0001 | Add validated low-level `MemorySegment` GEMM calls over the task-0001 lifetime without Tensor, route, allocation, or fallback behavior. |
| 0003 | Thread control and native provider checkpoint | Draft | 0001–0002 | Add low-level thread-count query/control and explicit compatible-library validation, then close the selected provider capability milestone. |

## Milestones

- Library and symbols
- GEMM contract
- Thread control and native validation

## Current status

In progress after completion of
[task 0001](tasks/0001-library-loading-and-required-symbol-binding.md), the only detailed OpenBLAS
provider task. The provider now has explicit caller-directed loading, complete required-symbol
binding, and caller-owned lookup lifetime. Task 0002 is the next ordered Draft row; tasks
0002–0003 have no detailed specifications.

## Open questions

- The later GEMM task must fix exact row-major validation, `MemorySegment` range/alignment rules,
  and call failure semantics before becoming Ready.
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

## Risks

- Allowing CPU backend policy or fallback logic into the provider.
- Treating caller-directed native loading as platform discovery or hiding a default candidate
  search in the provider.
- Exposing native handles or failing to close a partial lookup after required-symbol binding
  fails.
- Treating separate Java library handles as separate OpenBLAS thread-control state.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
