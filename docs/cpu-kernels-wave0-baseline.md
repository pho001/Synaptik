# CPU Kernels Wave 0 Baseline

This document freezes the pre-rewrite state of `backend.cpu.kernels` before removing native CPU transition layers. It is intentionally descriptive, not a target architecture.

Wave 104 closure note: this is now historical baseline evidence. The listed native CPU plan/facts/per-op executor
classes were removed from runtime ownership. Current non-BLAS `MemorySegment` execution lives in storage-specialized
loops under the existing CPU kernel families, while OpenBLAS `MemorySegment` matmul remains a provider route under
`linalg.matmul`.

## Scope

Wave 0 covers six representative operations:

- `ADD`
- `WHERE`
- `SUM`
- `CAST`
- `CONTIGUOUS`
- `MATMUL`

The goal is to make later cleanup measurable: every later wave should shrink this map or move ownership inward without creating a compatibility facade.

## Current Owner Map

| Op | Kernel entrypoint | Java array owner | Native or MemorySegment owner | Prepare-time owner | Notes |
|---|---|---|---|---|---|
| `ADD` | `backend.cpu.kernels.elementwise.binary.CpuAddKernel` | `CpuAddKernel` owns dense direct array loops through `StorageAwareBinaryElementwiseKernel`. | `StorageAwareBinaryElementwiseKernel` owns array, `MemorySegment`, dense, broadcast/indexed, and native fallback routing for migrated numeric binary ops. | `ElementwiseDispatchPlanner` owns array dispatch hints; storage binding is passed through `CpuKernelCall`. | Numeric binary hot paths are now storage-aware family-owned loops, not the old shared binary executor path. |
| `WHERE` | `backend.cpu.kernels.elementwise.where.CpuWhereKernel` | `WhereExecutor` validates inputs and falls back to `ElementwiseLoops.runWhere(...)`. | `NativeCpuElementwiseExecutor.tryRunWhere(...)` owns the condition-array/native-output route. | `BroadcastPlanResolver.resolveWhere(...)`, `PreparedInputPlanner`, and `NativeCpuPlanResolver`. | `WHERE` is the representative mixed-input case: condition is `BOOL`, branches/output are floating tensors. |
| `SUM` | `backend.cpu.kernels.reduction.CpuSumKernel` | `SumLikeReductionExecutor` falls back to `SumLoops` and applies `SumLikeReduction` finalization. | `NativeCpuReductionExecutor.tryRunSumLike(...)` owns current native reduction path. | `ReductionPlanner` and `NativeCpuPlanResolver`. | Later waves should keep reduction accumulation policy explicit and avoid per-element storage accessors. |
| `CAST` | `backend.cpu.kernels.layout.CpuCastKernel` | `CpuCastKernel.cast(...)` loops over logical flat indexes and writes typed arrays directly. | `NativeCpuCastExecutor.tryRunCast(...)` owns current native materialization route. | `CpuTypeContractResolver`, `PreparedInputPlanner`, and `NativeCpuPlanResolver`. | `CAST` is materialization/type conversion, not compute. The future storage contract should expose that distinction directly. |
| `CONTIGUOUS` | `backend.cpu.kernels.layout.CpuContiguousKernel` | `LayoutExecutor.contiguous(...)` owns array fallback. | `NativeCpuContiguousExecutor.tryRunContiguous(...)` owns current native layout materialization route. | `PreparedInputPlanner`, `CpuLayoutPlan`, and `NativeCpuPlanResolver`. | `CONTIGUOUS` is layout materialization. It should not need a parallel native executor once storage views are explicit. |
| `MATMUL` | `backend.cpu.kernels.linalg.CpuMatMulKernel` | `PreparedMatMulExecutableFactory` selects `F32JavaMatMulExecutable`, `F64JavaMatMulExecutable`, or `BF16JavaMatMulExecutable`; Java implementations keep vector/parallel tiling in the dtype matmul packages. | OpenBLAS array-copy routes use `F32BlasMatMulExecutable`/`F64BlasMatMulExecutable`/`BF16BlasMatMulExecutable`; MemorySegment routes use `F32NativeBlasMatMulExecutable`/`F64NativeBlasMatMulExecutable`/`BF16NativeBlasMatMulExecutable`. | `MatMulPlanner` chooses `JAVA_DIRECT`, `OPENBLAS_ARRAY_COPYING`, or `OPENBLAS_NATIVE_SEGMENT`; `CpuPlanAssembler` stores the prepared executable. | Unlike non-BLAS native executors, OpenBLAS is a real provider boundary. Later cleanup should keep the provider route but place it under matmul ownership. |

## Execution Path Versus Reporting

Execution path classes participate in compile, prepare, or per-op execution:

- `backend.cpu.CPUBackend`
- `backend.cpu.CpuNodePreparer`
- `backend.cpu.prepare.CpuExecutionPlanner`
- `backend.cpu.prepare.CpuPlanAssembler`
- `backend.cpu.plan.CpuNodeExecutionPlan`
- `backend.cpu.kernels.CpuKernelContext`
- `backend.cpu.kernels.*` family kernels, executors, loops, planners, and prepared executable classes
- `backend.cpu.nativecpu.NativeCpuPlanResolver`
- `backend.cpu.nativecpu.PreparedNativeCpuPlan`
- `backend.cpu.nativecpu.NativeCpuElementwiseExecutor`
- `backend.cpu.nativecpu.NativeCpuReductionExecutor`
- `backend.cpu.nativecpu.NativeCpuBoolMaskExecutor`
- `backend.cpu.nativecpu.NativeCpuCompareExecutor`
- `backend.cpu.nativecpu.NativeCpuCastExecutor`
- `backend.cpu.nativecpu.NativeCpuContiguousExecutor`
- `backend.cpu.nativecpu.NativeCpuViewExecutor`
- `backend.provider.blas.openblas.OpenBlasArrayGemm`, `OpenBlasSegmentGemm`, and `OpenBlasRuntime`

Reporting or evidence classes are not per-element executors, but some still influence planning and trace output:

- `NativeCpuKernelFacts` is metadata used by coverage, parity, and executor trace evidence.
- `NativeCpuCoverageMatrix` is used by `NativeCpuPlanResolver` and `CpuPartitionLowerer`; it is not docs-only today.
- `NativeCpuParityMatrix` is used by CPU lowering and trace reporting; it is not docs-only today.
- Benchmark report renderers and gates under `tuning.benchmark.report` consume traces and should not become runtime owners.

This distinction matters because later waves should remove execution dependencies first. Metadata/reporting can survive temporarily only when it remains truthful and does not select a hidden runtime route.

## Current Native Import Gate

The current allowlist is intentionally narrow. New imports from `backend.cpu.kernels` into `backend.cpu.nativecpu` must be treated as debt unless they are part of a planned removal wave.

```java
Map.ofEntries(
        Map.entry("CpuNodeExecutionPlan.java", Set.of("PreparedNativeCpuPlan")),
        Map.entry("plan/CpuPlanAssembler.java", Set.of("NativeCpuPlanResolver", "PreparedNativeCpuPlan")),
        Map.entry("elementwise/binary/StorageAwareBinaryElementwiseKernel.java", Set.of("CpuNativeTraceSupport")),
        Map.entry("elementwise/unary/ElementwiseUnaryExecutor.java", Set.of("NativeCpuElementwiseExecutor")),
        Map.entry("elementwise/where/WhereExecutor.java", Set.of("NativeCpuElementwiseExecutor")),
        Map.entry("elementwise/compare/CompareExecutor.java", Set.of("NativeCpuCompareExecutor")),
        Map.entry("elementwise/logical/LogicalExecutor.java", Set.of("NativeCpuBoolMaskExecutor")),
        Map.entry("reduction/SumLikeReductionExecutor.java", Set.of("NativeCpuReductionExecutor")),
        Map.entry("reduction/MinMaxReduceExecutor.java", Set.of("NativeCpuReductionExecutor")),
        Map.entry("reduction/BoolReduceExecutor.java", Set.of("NativeCpuBoolMaskExecutor")),
        Map.entry("layout/CpuCastKernel.java", Set.of("NativeCpuCastExecutor")),
        Map.entry("layout/CpuContiguousKernel.java", Set.of("NativeCpuContiguousExecutor")),
        Map.entry("layout/CpuAliasViewKernel.java", Set.of("NativeCpuViewExecutor")),
        Map.entry("layout/CpuNoopKernel.java", Set.of("NativeCpuViewExecutor")),
        Map.entry("layout/CpuReshapeLikeKernel.java", Set.of("NativeCpuViewExecutor"))
)
```

Why this gate exists: Wave 0 cannot remove the native stack yet, but it must prevent the stack from spreading while the rewrite is prepared.

## Root Package Baseline

`backend.cpu.kernels` root is reserved for shared CPU execution infrastructure:

- dtype and compute contract records/enums
- `CpuKernel`
- `CpuKernelContext`
- `CpuNodeExecutionPlan`
- `CpuNodeWorkspace`
- `CpuThreadPool`

Concrete op entrypoints must stay in family packages such as `elementwise`, `reduction`, `layout`, `linalg`, `index`, `nn`, or `fused`.

Example rule:

```java
assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/kernels/CpuAddKernel.java")));
```

Why: root-package concrete kernels make ownership harder to see and encourage shared helper layers that become accidental facades.

## Minimal Benchmark Sanity

Run these slices before and after each cleanup wave that touches the relevant family. Do not commit local calibration or profile artifacts produced by these checks.

| Slice | Purpose | Current handle |
|---|---|---|
| dense F32 add | Preserve cheap elementwise Java hot path, vector width, and parallel threshold behavior. | `debug.HotPathAnalysisTest#printRepresentativeInferenceHotPaths` covers representative cheap elementwise F32. |
| dense F64 add | Preserve F64 direct/vector path, especially because F64 profiles can have different stage order. | Existing F64 debug profile comparisons such as `debug.AbcF64StageOrderHotspotBenchmarkTest`. |
| BF16 add | Preserve BF16 storage with F32-promoted compute/continuation behavior. | Existing BF16 debug profile comparisons such as `debug.AbcBf16CheapContiguousWidthBenchmarkTest`. |
| F32 sum | Preserve reduction vector/parallel policy and accumulation behavior. | `debug.HotPathAnalysisTest#printRepresentativeInferenceHotPaths` includes `analysis_reduction_sum`. |
| F32 matmul Java | Preserve non-BLAS tiled Java route and worker scheduling. | `debug.HotPathAnalysisTest#printRepresentativeInferenceHotPaths` includes `analysis_matmul_attention_like`; force `BlasConfig.disabled()` when isolating Java matmul. |
| F32 OpenBLAS array copy | Preserve provider route that copies Java arrays into BLAS-compatible calls. | `debug.NativeOpenBlasSegmentGemmBenchmarkTest#benchmarkJavaArrayOpenBlasArrayCopyAndNativeSegmentRoutes`. |
| F32 OpenBLAS segment | Preserve direct `MemorySegment` provider route and copy-byte evidence. | `debug.NativeOpenBlasSegmentGemmBenchmarkTest#benchmarkJavaArrayOpenBlasArrayCopyAndNativeSegmentRoutes`. |

Command examples:

```bash
./gradlew classes
./gradlew test --tests CpuKernelFamilyArchitectureTest
./gradlew test --tests debug.HotPathAnalysisTest
SYNAPTIK_BENCHMARK_NATIVE_OPENBLAS_SEGMENT=true ./gradlew test --tests debug.NativeOpenBlasSegmentGemmBenchmarkTest
```

## Wave 0 Validation

- Architecture baseline: `./gradlew test --tests CpuKernelFamilyArchitectureTest`
- Compile baseline: `./gradlew classes`
- The historical `NativeCpuPlanResolverTest` no longer exists. Current replacement coverage is
  `./gradlew test --tests backend.cpu.nativecpu.NativeCpuPartitionSelectionTest` plus
  `CpuKernelFamilyArchitectureTest` and `SourceTreeHygieneTest` removal guards.
- Benchmark sanity: run the slices above without staging generated profile/calibration artifacts.
