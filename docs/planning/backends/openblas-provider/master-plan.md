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
| 0003 | [Thread control and native provider checkpoint](tasks/0003-thread-control-and-native-provider-checkpoint.md) | Complete | 0001–0002 | Add low-level thread-count query/control and explicit compatible-library validation, then close the selected provider capability milestone. |

## Milestones

- Library and symbols
- GEMM contract
- Thread control and native validation

## Current status

Complete. Tasks 0001–0003 provide explicit caller-directed loading, complete required-symbol
binding, caller-owned lookup lifetime, validated FLOAT32/FLOAT64 dense row-major no-transpose GEMM
invocation, and direct positive thread-count query/control over the already-bound handles. The
ordinary provider suite passed 5 suites and 50 tests. The isolated native checkpoint passed
against the caller-supplied arm64 OpenBLAS 0.3.33 library, verified shared observation through two
owners and the fixed SGEMM/DGEMM cases, and restored the original thread count of 16. The ordered
repository/architecture capability checkpoint then passed with 54 actionable tasks (2 executed,
52 up-to-date). Documentation, exact twelve-path scope, surface, package, dependency, history,
status, later-specification, and whitespace gates passed. The selected provider milestone and
project area are Complete. CPU is the next Draft planning frontier; no CPU task is Ready and no
CPU task specification exists.

## Open questions

- None for the current provider milestone. Any future symbol, precision, layout, or lifecycle
  capability requires a separately justified task.

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
- Task 0003 exposes only `threadCount()` and `setThreadCount(int)` on the existing lifetime owner.
  As a conservative integration rule it treats owners of the same loaded binary as potentially
  sharing library/process state, without claiming coordination across independently loaded copies
  or arbitrary native consumers. It stores no per-owner value, performs no close-time
  restoration, and leaves thread choice and coordination to CPU.
- Task 0003's real-native checkpoint is a test-source command accepting one explicit absolute
  compatible-library path. It runs with explicitly enabled JDK native access, verifies shared
  observation through two owners and one fixed SGEMM/DGEMM case per precision, and restores the
  captured count in `finally`; production and ordinary tests perform no discovery or native load.

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
