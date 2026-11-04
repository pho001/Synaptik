# [OpenBLAS](../../../glossary.md#term-openblas) Provider Master Plan

Current source: `modules/backend/openblas-provider`; [Gradle](../../../glossary.md#term-gradle) [identity](../../../glossary.md#term-identity-meanings) remains
`:backends:openblas-provider`. Historical [planning](../../../glossary.md#term-planning)/report locations remain unchanged; see the
[path migration table](../../../developer-guide/repository-layout.md#backend-path-migration-and-historical-navigation).

## Goal

Provide a low-level [OpenBLAS](../../../glossary.md#term-openblas) [leaf](../../../glossary.md#term-leaf) for library loading, required and optional [symbol](../../../glossary.md#term-symbol) [binding](../../../glossary.md#term-binding),
direct supported [GEMM](../../../glossary.md#general-matrix-multiplication--gemm) calls, and thread control.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../glossary.md#architecture-contract))</span>
- [Module boundaries](../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../glossary.md#term-module))</span>
- [Dependency rules](../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../glossary.md#term-dependency))</span>

## [Scope](../../../glossary.md#term-scope)

- [native library](../../../glossary.md#term-native-library) resolution
- [symbol](../../../glossary.md#term-symbol) [binding](../../../glossary.md#term-binding)
- [GEMM](../../../glossary.md#general-matrix-multiplication--gemm) invocation
- optional, versioned direct BFLOAT16-input/BFLOAT16-output GEMM capability when the exact loaded
  binary exports and satisfies a proved ABI<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../glossary.md#term-bfloat16); [binary glossary](../../../glossary.md#term-binary-descriptor-and-metadata); [ABI glossary](../../../glossary.md#term-application-binary-interface-abi))</span>
- [OpenBLAS](../../../glossary.md#term-openblas) thread control

## Out of [scope](../../../glossary.md#term-scope)

- configuration interpretation
- fallback and [backend ownership](../../../glossary.md#backend-ownership)
- [prepared execution](../../../glossary.md#prepared-execution--preparedexecution)
- [Tensor](../../../glossary.md#tensor) [API](../../../glossary.md#term-application-programming-interface-api) and [residency](../../../glossary.md#residency)

## [Module](../../../glossary.md#term-module) invariants

- The provider remains a low-level [leaf](../../../glossary.md#term-leaf).
- [Dependency direction](../../../glossary.md#java-platform-and-build-tools) is [CPU](../../../glossary.md#term-cpu) [backend](../../../glossary.md#backend) to [OpenBLAS](../../../glossary.md#term-openblas) provider, never the reverse.
- The completed mandatory baseline remains FLOAT32/FLOAT64 only. No 16-bit capability is inferred
  from library presence, version, header declarations, architecture, storage width, related
  symbols, or logical <span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../glossary.md#term-float32); [FLOAT64 glossary](../../../glossary.md#term-float64))</span>[`DataType`](../../../glossary.md#data-type--datatype) availability.
- Direct BFLOAT16 output is optional per loaded binary. Its absence or failed optional binding
  never invalidates the required FLOAT32/FLOAT64 and thread-control surface.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../glossary.md#term-bfloat16); [binary glossary](../../../glossary.md#term-binary-descriptor-and-metadata); [binding glossary](../../../glossary.md#term-binding))</span>

## Allowed dependencies

- [JDK](../../../glossary.md#term-java-development-kit-jdk) standard library and required native interop APIs.

## Forbidden dependencies

- [model](../../../glossary.md#term-model) [Tensor](../../../glossary.md#tensor) [API](../../../glossary.md#term-application-programming-interface-api)
- [planning](../../../glossary.md#term-planning), [compiler](../../../glossary.md#term-compiler), [runtime](../../../glossary.md#term-runtime), [prepare](../../../glossary.md#prepare), [engine](../../../glossary.md#term-engine), and [concrete backend](../../../glossary.md#concrete-backend) dependencies

## Package structure

```text
io.github.pho001.synaptik.backend.provider.openblas/
  <root>  small public library-lifetime and later low-level GEMM/thread-control surface, with
          package-private JDK FFM loading, symbol binding, and native handles
```

The root package is deliberate: task 0001 opens one small public [lifetime](../../../glossary.md#term-lifetime) boundary, and task 0002
adds only its inseparable low-level calls plus one package-private field-free invocation helper.
Native implementation state remains package-private. The package must not grow into a generic
native framework, [registry](../../../glossary.md#term-registry), manager, configuration package, or [CPU](../../../glossary.md#term-cpu) route [owner](../../../glossary.md#term-owner).

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Library loading and required symbol binding](tasks/0001-library-loading-and-required-symbol-binding.md)<span data-glossary-navigation="true"> ([symbol glossary](../../../glossary.md#term-symbol); [binding glossary](../../../glossary.md#term-binding))</span> | Complete | [Prepare](../../../glossary.md#prepare) 0003; [JDK](../../../glossary.md#term-java-development-kit-jdk) 26 [FFM](../../../glossary.md#term-ffm) and [OpenBLAS](../../../glossary.md#term-openblas) C [ABI](../../../glossary.md#term-application-binary-interface-abi) evidence | Load one caller-specified library, bind the exact [FLOAT32](../../../glossary.md#term-float32)/[FLOAT64](../../../glossary.md#term-float64) [GEMM](../../../glossary.md#general-matrix-multiplication--gemm) and get/set thread-count symbols, and own their closeable [lookup](../../../glossary.md#term-lookup) [lifetime](../../../glossary.md#term-lifetime) without invocation or policy. |
| 0002 | [FLOAT32/FLOAT64 row-major GEMM invocation](tasks/0002-float32-float64-row-major-gemm-invocation.md) | Complete | 0001 | Add validated dense [row-major](../../../glossary.md#term-row-major) no-[transpose](../../../glossary.md#term-transpose) [`MemorySegment`](../../../glossary.md#term-memorysegment) GEMM calls over the task-0001 lifetime without [Tensor](../../../glossary.md#tensor), route, allocation, or fallback behavior. |
| 0003 | [Thread control and native provider checkpoint](tasks/0003-thread-control-and-native-provider-checkpoint.md)<span data-glossary-navigation="true"> ([checkpoint glossary](../../../glossary.md#term-checkpoint))</span> | Complete | 0001–0002 | Add low-level thread-count query/control and explicit compatible-library [validation](../../../glossary.md#term-validation), then close the original provider capability milestone. |
| 0004 | [Optional direct BFLOAT16-output GEMM capability](tasks/0004-optional-direct-bfloat16-output-gemm-capability.md) | Blocked | 0001–0003; pinned OpenBLAS direct-BGEMM ABI and semantic proof | Pinned OpenBLAS v0.3.31 does not establish an exported `cblas_bgemm` C ABI and narrows [BFLOAT16](../../../glossary.md#term-bfloat16) output between sufficiently large [contraction](../../../glossary.md#contraction-dimension) panels, so no capability was exposed; retry only with evidence that resolves both gates. |

## Milestones

- Library and symbols
- [GEMM](../../../glossary.md#general-matrix-multiplication--gemm) contract
- Thread control and native [validation](../../../glossary.md#term-validation)
- Optional direct [BFLOAT16](../../../glossary.md#term-bfloat16)-output [ABI](../../../glossary.md#term-application-binary-interface-abi) proof and invocation

## Current status

The mandatory provider baseline is Complete; optional task 0004 is Blocked and deferred as a
capability side branch. Tasks 0001–0003 remain Complete and provide explicit caller-directed loading, complete required-[symbol](../../../glossary.md#term-symbol)
[binding](../../../glossary.md#term-binding), caller-owned [lookup](../../../glossary.md#term-lookup) [lifetime](../../../glossary.md#term-lifetime), validated [FLOAT32](../../../glossary.md#term-float32)/[FLOAT64](../../../glossary.md#term-float64) dense [row-major](../../../glossary.md#term-row-major) no-[transpose](../../../glossary.md#term-transpose) [GEMM](../../../glossary.md#general-matrix-multiplication--gemm)
invocation, and direct positive thread-count query/control over the already-bound handles. The
ordinary provider suite passed 5 suites and 50 tests. The isolated native [checkpoint](../../../glossary.md#term-checkpoint) passed
against the caller-supplied [arm64](../../../glossary.md#native-packaging-and-platform-vocabulary) [OpenBLAS](../../../glossary.md#term-openblas) 0.3.33 library, verified shared observation through two
[owners](../../../glossary.md#term-owner) and the fixed [SGEMM](../../../glossary.md#general-matrix-multiplication--gemm)/[DGEMM](../../../glossary.md#general-matrix-multiplication--gemm) cases, and restored the original thread count of 16. The ordered
repository/architecture capability checkpoint then passed with 54 actionable tasks (2 executed,
52 up-to-date). Documentation, exact twelve-path [scope](../../../glossary.md#term-scope), surface, package, [dependency](../../../glossary.md#term-dependency), history,
status, later-specification, and whitespace gates passed. That original provider milestone remains
closed.

Detailed task 0004 is the blocked/deferred optional provider side branch. Pinned OpenBLAS v0.3.31
does not establish `cblas_bgemm` in its official shared-library C export lists, and its driver/[kernel](../../../glossary.md#kernel) path narrows
[BFLOAT16](../../../glossary.md#term-bfloat16) output between sufficiently large [contraction](../../../glossary.md#contraction-dimension) panels rather than accumulating the full
contraction in FLOAT32 before one final BFLOAT16 conversion. The local 0.3.34 installation
independently confirms only optional absence: its header declares `cblas_bgemm`, but its [binary](../../../glossary.md#term-binary-descriptor-and-metadata)
exports `cblas_sbgemm` and not `cblas_bgemm`. No optional capability was exposed, and the completed
FLOAT32/FLOAT64/thread baseline remains usable. [CPU](../../../glossary.md#term-cpu) 0010D1 remains a dependent, non-executable
blocked/deferred optional side branch. It does not precede or block FLOAT32/FLOAT64 CPU 0010E,
[Prepare](../../../glossary.md#prepare) 0004, or tools/tuning 0001. Provider 0004 may [resume](../../../glossary.md#schema-serialization-and-checkpoint) only when both proof gates have new
concrete evidence.

## Open questions

- Whether pinned OpenBLAS implementation sources prove FLOAT32 accumulation followed by one
  direct BFLOAT16 result narrowing for every implementation reachable through the exported<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../glossary.md#term-openblas); [FLOAT32 glossary](../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../glossary.md#term-bfloat16); [narrowing glossary](../../../glossary.md#term-narrowing))</span>
  `cblas_bgemm` [ABI](../../../glossary.md#term-application-binary-interface-abi). Task 0004 must stop before exposure if this remains ambiguous.
- Whether 32-bit CBLAS enums and 16-bit by-value BFLOAT16 scalars can be proved for every supported
  macOS/Linux/Windows AArch64/x86-64 ABI. Any unproved target remains capability-absent rather than
  being inferred from the current machine.<span data-glossary-navigation="true"> ([CBLAS glossary](../../../glossary.md#term-cblas); [scalars glossary](../../../glossary.md#term-scalar); [AArch64 glossary](../../../glossary.md#term-aarch64); [x86-64 glossary](../../../glossary.md#term-x86-64); [target glossary](../../../glossary.md#term-target))</span>

## Decisions made

- The implementation must follow the current [architecture contract](../../../glossary.md#architecture-contract).
- Legacy code is capability evidence only; new implementation is written from scratch.
- CPU owns OpenBLAS route configuration and candidate generation, including thread choices. This
  provider exposes only low-level calls and thread control; it owns no tuning cache, workload
  signature, objective, budget, or candidate policy.<span data-glossary-navigation="true"> ([CPU glossary](../../../glossary.md#term-cpu); [OpenBLAS glossary](../../../glossary.md#term-openblas); [cache glossary](../../../glossary.md#term-cache); [signature glossary](../../../glossary.md#term-signature); [objective glossary](../../../glossary.md#term-objective))</span>
- Task 0001 accepts only a caller-specified library name or absolute path. It does not read config,
  environment variables, system properties, probe platform filenames, or decide fallback.
- Task 0001 continues to require `cblas_sgemm`, `cblas_dgemm`, `openblas_set_num_threads`, and
  `openblas_get_num_threads` under the standard 32-bit-`blasint` C ABI. It selects no minimum
  OpenBLAS version or ILP64 support. Task 0004 may probe its one optional symbol only after this
  mandatory set binds completely.<span data-glossary-navigation="true"> ([ABI glossary](../../../glossary.md#term-application-binary-interface-abi); [ILP64 glossary](../../../glossary.md#term-ilp64); [symbol glossary](../../../glossary.md#term-symbol))</span>
- `cblas_bgemm` is treated as a build-optional, versioned OpenBLAS extension rather than
  standardized CBLAS. Structural presence comes only from exact optional lookup/binding. Header,
  version, architecture, build-option text, and <span data-glossary-navigation="true"> ([CBLAS glossary](../../../glossary.md#term-cblas); [lookup glossary](../../../glossary.md#term-lookup); [binding glossary](../../../glossary.md#term-binding))</span>`cblas_sbgemm` presence are not substitutes.
- The task-0004 route is direct [BFLOAT16](../../../glossary.md#term-bfloat16) input and output only. It does not call `cblas_sbgemm`,
  widen input matrices, stage a FLOAT32 output, or convert a result buffer.<span data-glossary-navigation="true"> ([matrices glossary](../../../glossary.md#term-matrix-meanings); [FLOAT32 glossary](../../../glossary.md#term-float32); [buffer glossary](../../../glossary.md#term-buffer))</span>
- CPU 0010D1 owns exact installed-binary numerical qualification, route eligibility, fallback,
  materialization, fingerprint compatibility, and execution after provider 0004. The provider
  capability by itself is not a CPU qualification credential.<span data-glossary-navigation="true"> ([binary glossary](../../../glossary.md#term-binary-descriptor-and-metadata); [qualification glossary](../../../glossary.md#term-qualification); [materialization glossary](../../../glossary.md#materialization); [fingerprint glossary](../../../glossary.md#term-fingerprint))</span>
- The failed optional BFLOAT16 proof does not change Model or portable CPU BFLOAT16 support and
  does not gate existing FLOAT32/FLOAT64 OpenBLAS development or model-autotuning work.<span data-glossary-navigation="true"> ([Model glossary](../../../glossary.md#term-model); [FLOAT64 glossary](../../../glossary.md#term-float64); [autotuning glossary](../../../glossary.md#autotuning--model-autotuning))</span>
- Model FLOAT16 is implemented, but that does not create an OpenBLAS FLOAT16 route. A future
  native low-type route still needs its own ABI and numerical qualification; stale Model 0026
  planning is not its prerequisite.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../glossary.md#term-float16); [planning glossary](../../../glossary.md#term-planning))</span>
- One caller-owned public library handle encapsulates the shared FFM arena and package-private
  bound handles. No eager singleton, global cache, exposed native address, registry, manager, or
  service locator is planned.<span data-glossary-navigation="true"> ([FFM glossary](../../../glossary.md#term-ffm); [arena glossary](../../../glossary.md#term-arena); [eager glossary](../../../glossary.md#term-eager-evaluation); [singleton glossary](../../../glossary.md#term-singleton); [registry glossary](../../../glossary.md#term-registry); [service locator glossary](../../../glossary.md#term-service-locator))</span>
- Task 0002 adds exactly `sgemm` and `dgemm` to that public lifetime owner for dense row-major
  no-transpose matrices. It derives leading dimensions internally and leaves batching,
  broadcasting, transpose/layout conversion, packing, loops, route choice, and fallback to CPU.<span data-glossary-navigation="true"> ([lifetime glossary](../../../glossary.md#term-lifetime); [owner glossary](../../../glossary.md#term-owner); [row-major glossary](../../../glossary.md#term-row-major); [transpose glossary](../../../glossary.md#term-transpose); [dimensions glossary](../../../glossary.md#dimension); [broadcasting glossary](../../../glossary.md#broadcasting); [layout glossary](../../../glossary.md#layout))</span>
- One package-private field-free invocation helper owns ordered segment validation and exact typed
  handle calls. The existing bindings/native-access path remains the sole deterministic test seam.<span data-glossary-navigation="true"> ([validation glossary](../../../glossary.md#term-validation))</span>
- Task 0002 validates completely before output-empty no-op, invokes positive-output `k == 0`
  GEMM for <span data-glossary-navigation="true"> ([GEMM glossary](../../../glossary.md#general-matrix-multiplication--gemm))</span>`beta * C`, forwards all scalar bits unchanged, rejects C/input overlap, and adds no
  close-race coordination or native numerical guarantee.<span data-glossary-navigation="true"> ([scalar glossary](../../../glossary.md#term-scalar); [race glossary](../../../glossary.md#term-race))</span>
- Task 0003 exposes only `threadCount()` and `setThreadCount(int)` on the existing lifetime owner.
  As a conservative integration rule it treats owners of the same loaded binary as potentially
  sharing library/process state, without claiming coordination across independently loaded copies
  or arbitrary native consumers. It stores no per-owner value, performs no close-time
  restoration, and leaves thread choice and coordination to CPU.
- Task 0003's real-native checkpoint is a test-source command accepting one explicit absolute
  compatible-library path. It runs with explicitly enabled JDK native access, verifies shared
  observation through two owners and one fixed SGEMM/DGEMM case per precision, and restores the
  captured count in <span data-glossary-navigation="true"> ([checkpoint glossary](../../../glossary.md#term-checkpoint); [JDK glossary](../../../glossary.md#term-java-development-kit-jdk); [SGEMM glossary](../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../glossary.md#general-matrix-multiplication--gemm); [precision glossary](../../../glossary.md#term-precision))</span>`finally`; production and ordinary tests perform no discovery or native load.

## Risks

- Allowing [CPU](../../../glossary.md#term-cpu) [backend](../../../glossary.md#backend) policy or fallback logic into the provider.
- Treating caller-directed native loading as platform discovery or hiding a default candidate
  search in the provider.
- Exposing native handles or failing to close a partial lookup after required-symbol binding
  fails.<span data-glossary-navigation="true"> ([lookup glossary](../../../glossary.md#term-lookup); [symbol glossary](../../../glossary.md#term-symbol); [binding glossary](../../../glossary.md#term-binding))</span>
- Letting a low-level GEMM call grow offsets, transpose/layout normalization, batching, packing,
  allocation, numerical policy, or CPU-owned route behavior.<span data-glossary-navigation="true"> ([GEMM glossary](../../../glossary.md#general-matrix-multiplication--gemm); [transpose glossary](../../../glossary.md#term-transpose); [layout glossary](../../../glossary.md#layout); [normalization glossary](../../../glossary.md#term-normalization))</span>
- Treating separate Java library handles as separate [OpenBLAS](../../../glossary.md#term-openblas) thread-control state.
- Treating optional or version-specific 16-bit symbols as broad baseline support, or letting the
  provider advertise/select a route from [`DataType`](../../../glossary.md#data-type--datatype) availability alone.
- Treating a declaring installed header or `cblas_sbgemm` export as evidence that the loaded
  binary exports a compatible direct BFLOAT16-output ABI.<span data-glossary-navigation="true"> ([binary glossary](../../../glossary.md#term-binary-descriptor-and-metadata); [ABI glossary](../../../glossary.md#term-application-binary-interface-abi))</span>
- Publishing the optional call before proving enum, integer, scalar, accumulation, and narrowing
  semantics across supported host ABIs.<span data-glossary-navigation="true"> ([scalar glossary](../../../glossary.md#term-scalar); [narrowing glossary](../../../glossary.md#term-narrowing); [host glossary](../../../glossary.md#term-host))</span>
- Hiding [BFLOAT16](../../../glossary.md#term-bfloat16)/[FLOAT32](../../../glossary.md#term-float32) conversion or FLOAT32 result staging behind a [nominal](../../../glossary.md#term-nominal) direct capability.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span>.
