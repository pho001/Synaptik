# Codebase Concerns

**Analysis Date:** 2026-04-29

## Tech Debt

**Tensor remains a mutable god object and public facade:**
- Issue: `src/main/java/tensor/Tensor.java` is 2,226 lines and owns construction, mutable storage, dtype conversion, graph links, autograd entry points, compile/prepare/compute shortcuts, layout ops, elementwise ops, reductions, losses, linalg, pooling, and normalization. The class explicitly exposes mutable typed backing arrays and unsafe shape/stride accessors.
- Files: `src/main/java/tensor/Tensor.java`, `src/main/java/tensor/TensorInternalAccess.java`, `src/main/java/tensor/TensorStorage.java`
- Impact: Small feature changes can accidentally cross storage, graph, autograd, and runtime boundaries. Aliasing and stale storage bugs are easy to introduce because `Tensor` is mutable, not thread-safe, and can share storage across views.
- Fix approach: Keep new operation behavior in focused packages under `src/main/java/tensor/ops/` and use `TensorInternalAccess` only for graph/runtime plumbing. Move additional public surface out of `Tensor.java` into narrow helper/facade classes before adding more operation families.

**Generated/profile artifacts are tracked as source:**
- Issue: `git ls-files profiles` reports 271 tracked calibration/profile artifacts totaling about 24.8 MB. Many files under `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/calibration/schema-v2/runs/` are machine-specific run outputs, and `.idea/*` project files are tracked.
- Files: `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/calibration/schema-v2/runs/`, `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/reports/calibration-f64-forward-backward.json`, `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/reports/calibration-f32-forward-backward.json`, `.idea/compiler.xml`, `.idea/gradle.xml`
- Impact: Repository size and diffs grow with local calibration runs. Machine-specific data can bias defaults and makes review noisy. IDE files encode local project state outside Gradle.
- Fix approach: Keep only stable seed profiles or fixtures in source control. Move run history, reports, and local best profiles to `build/`, a fixture subdirectory with explicit test ownership, or release artifacts. Extend hygiene checks to reject accidental `profiles/platform/**/runs/**` additions.

**Repository hygiene checks do not cover root-level generated artifacts:**
- Issue: Current hygiene tasks scan `src/` and `test/`, but ignored generated files are present at the repository root and local metadata directories are present in the worktree.
- Files: `build.gradle`, `src/test/java/SourceTreeHygieneTest.java`, `OptimizedFusedOperation_207109404218375.class`, `OptimizedFusedOperation_207204834517375.class`, `SumCalculator.class`, `fusedOperationClass.class`, `.gradle-userhome/`, `out/`
- Impact: Root artifacts can mask classpath issues in ad hoc runs, confuse codebase scans, and normalize generated files living beside source files.
- Fix approach: Add a root-level generated artifact check for committed and uncommitted files, with explicit allowlists for Gradle outputs. Prefer `build/` for all generated class dumps.

**Default test task includes benchmark and calibration workloads:**
- Issue: `src/test/java/debug/` contains many `@Test` classes that run benchmark/calibration sessions. `DebugMeasurementPolicies` uses 30 warmup iterations, 100 measurement iterations, and 3 repeats for standard debug tests. `./gradlew test` was still running after roughly five minutes and had to be killed.
- Files: `src/test/java/debug/DebugMeasurementPolicies.java`, `src/test/java/debug/AbcLongMeasurementAutotuneTest.java`, `src/test/java/debug/AttentionMatMulFamilyCalibrationTest.java`, `src/test/java/debug/FusedNonCheapFamilyCalibrationTest.java`, `build.gradle`
- Impact: Fast correctness feedback is mixed with performance exploration. CI or local verification can become slow, noisy, or hardware-dependent.
- Fix approach: Move debug benchmark tests behind a separate Gradle task or JUnit tag. Keep `test` focused on deterministic unit/integration coverage and create explicit `benchmarkTest` or `calibrationTest` tasks for measurement workloads.

**JSON persistence uses regex/string parsing instead of a JSON library:**
- Issue: Profile and tuning stores hand-roll JSON writing and parsing with regex/key lookup. Unknown or malformed fields silently fall back to defaults in several paths.
- Files: `src/main/java/config/profile/ExecutionProfileIO.java`, `src/main/java/config/profile/PlatformRuntimeProfileIO.java`, `src/main/java/tuning/store/JsonFileTuningHistoryStore.java`, `src/main/java/tuning/store/JsonFileBestProfileStore.java`
- Impact: Nested data, escaped strings, schema drift, or duplicate keys can load incorrectly without hard failure. Tuning results can appear valid while using fallback runtime settings.
- Fix approach: Introduce a small JSON dependency or a strict internal parser with schema version validation. Make production profile loading report invalid fields separately from intentionally missing optional fields.

**Memory binding supports only FLOAT64 and FLOAT32 runtime slot reuse:**
- Issue: `RuntimeMemoryBinder.bindTypedStorage(...)` is a no-op for `BFLOAT16`, `INT32`, and `BOOL` even though the memory planner tracks region slots and binding metadata for all graph values.
- Files: `src/main/java/graph/execution/RuntimeMemoryBinder.java`, `src/main/java/graph/optimizer/memory/MemoryPlanner.java`, `src/main/java/graph/optimizer/memory/MemoryPlan.java`
- Impact: BF16-heavy, index-heavy, and bool-mask workloads do not receive the same runtime memory reuse benefits as float workloads. Optimization summaries can overstate practical memory reuse for unsupported dtypes.
- Fix approach: Add typed slot pools for `short[]`, `int[]`, and `byte[]`, and expand runtime binding tests to cover BF16, INT32, and BOOL region values.

**OpenCL backend is effectively a placeholder:**
- Issue: OpenCL has a registry with only `NOOP` and no native bridge/runtime implementation comparable to Metal or CUDA.
- Files: `src/main/java/backend/opencl/registry/OpenClKernelRegistry.java`, `src/main/java/backend/opencl/kernels/OpenClNoopKernel.java`, `src/main/java/config/runtime/AcceleratorConfig.java`
- Impact: Runtime configuration exposes OpenCL policy, but real compute coverage is absent. Users can enable an accelerator target that cannot execute meaningful work.
- Fix approach: Either document and enforce OpenCL as unavailable/experimental, or add a real bridge, capability checks, lowering, and tests before exposing it as a selectable backend.

## Known Bugs

**Prepared executions are not globally invalidated after semantic graph mutation:**
- Symptoms: A `PreparedExecution` can be reused after tensor graph topology, dtype, shape, layout, backend intent, or runtime assumptions change. Documentation states no single public stale-check guard exists.
- Files: `src/main/java/graph/execution/PreparedExecution.java`, `src/main/java/graph/CompiledGraph.java`, `docs/compute-flow.md`
- Trigger: Compile and prepare a graph, mutate the underlying semantic tensor graph or operation metadata, then execute the old prepared artifact.
- Workaround: Recompile and prepare after any graph contract change. Treat prepared executions as bound to compile-time shape, dtype, topology, layout, operation, backend, and runtime assumptions.

**Prepared conv2d GEMM can hard-fail when OpenBLAS becomes unavailable:**
- Symptoms: Matmul BLAS calls generally fall back to Java kernels, but prepared conv2d GEMM hints that require `OPENBLAS_FFM` throw when the bridge is unavailable.
- Files: `src/main/java/backend/cpu/kernels/nn/Conv2dGemmBackend.java`, `src/main/java/backend/cpu/kernels/linalg/matmul/blas/MatMulBlasBackend.java`, `src/main/java/backend/blas/OpenBlasFfmBridge.java`, `docs/native-bridges-and-blas.md`, `docs/troubleshooting.md`
- Trigger: Prepare a conv2d GEMM plan with `BlasProvider.OPENBLAS_FFM`, then run where the OpenBLAS library cannot be loaded or required symbols are missing.
- Workaround: Use `BlasConfig.disabled()` or a profile that does not require OpenBLAS for conv2d unless the bridge is installed and stable.

**Default `./gradlew test` can run too long for fast verification:**
- Symptoms: The test command reached the `:test` task and continued for several minutes with no completion while benchmark-style tests were part of the default test source set.
- Files: `build.gradle`, `src/test/java/debug/`, `src/test/java/debug/DebugMeasurementPolicies.java`
- Trigger: Run `./gradlew test` on the full repository.
- Workaround: Use narrower test filters for code changes until benchmark/debug tests are split into a separate task.

## Security Considerations

**Native library loading trusts local system properties and environment variables:**
- Risk: The process loads native libraries from `openblas.lib`, `OPENBLAS_LIB`, `synaptik.metal.mps.lib`, `SYNAPTIK_METAL_MPS_LIB`, `synaptik.cuda.graph.lib`, and `SYNAPTIK_CUDA_GRAPH_LIB`. Gradle also enables `--enable-native-access=ALL-UNNAMED`.
- Files: `build.gradle`, `src/main/java/backend/blas/OpenBlasFfmBridge.java`, `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`, `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`
- Current mitigation: Library paths are local opt-in configuration and failures become unavailable bridge records or fallback paths in most cases.
- Recommendations: Treat native path configuration as trusted-local only. For packaged use, restrict native lookup directories, validate expected library names/signatures, and avoid enabling unrestricted native access for unrelated modules.

**CLI and tuning stores write to caller-supplied paths:**
- Risk: Calibration and tuning commands can write profile, report, and history files to arbitrary local paths supplied by command-line options or API builders.
- Files: `src/main/java/synaptik/app/TuningCli.java`, `src/main/java/tuning/calibration/run/CalibrationCommand.java`, `src/main/java/config/profile/ExecutionProfileIO.java`, `src/main/java/config/profile/PlatformRuntimeProfileIO.java`, `src/main/java/tuning/store/JsonFileTuningHistoryStore.java`
- Current mitigation: This is a local developer tool with no network-exposed surface detected.
- Recommendations: For multi-user or service contexts, restrict output roots, reject path traversal outside configured workspace roots, and write atomically.

**Calibration artifacts can expose hardware fingerprints:**
- Risk: Tracked profile paths and JSON metadata include OS, architecture, vendor, core count, and run timing data.
- Files: `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/`, `src/main/java/tuning/store/HardwareFingerprint.java`, `src/main/java/config/profile/PlatformRuntimeProfileIO.java`
- Current mitigation: No credentials were detected in inspected files.
- Recommendations: Keep hardware fingerprints out of public source unless intentionally published as benchmark data. Store local calibration history in ignored directories.

## Performance Bottlenecks

**Metal and CUDA legacy paths copy through Java arrays:**
- Problem: CUDA execution allocates native memory from Java arrays and copies outputs back. Metal keeps a buffer path, but it falls back to the tensor-array bridge or CPU fallback when buffer binding is unavailable.
- Files: `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`, `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`, `src/main/java/backend/metal/exec/PreparedMetalExecutable.java`, `docs/metal-backend.md`, `docs/compute-flow.md`
- Cause: Current accelerator ABI is built around FLOAT32 graph execution and external input/output arrays, with buffer binding only implemented for Metal and only for compatible residency/layout contracts.
- Improvement path: Prefer buffer-resident execution for accelerator regions, add CUDA buffer binding, and make fallback reasons visible in performance reports so CPU replay does not look like GPU execution.

**Metal native ABI is rank-limited and FLOAT32-centered:**
- Problem: Native compile/execute symbols are `_f32`, Java capabilities allow FLOAT32 compute/output and BOOL only in predicate roles, and native shape handling stores up to four dimensions.
- Files: `src/main/java/backend/metal/MetalMpsCapabilities.java`, `src/main/java/backend/metal/lowering/MetalPartitionSupport.java`, `src/main/native/apple/synaptik_apple_mps_stub.m`
- Cause: Java and Objective-C bridge contracts encode a narrow dtype/rank subset.
- Improvement path: Add explicit capability/version structs from native code, expand dtype and rank support deliberately, and keep planner rejection messages aligned with native ABI limits.

**CUDA backend has Java FFM bridge code but no native implementation in the repository:**
- Problem: CUDA bridge and tests exist, but `find src/main/native scripts` shows only the Apple MPS shim script and Objective-C source. Docs mark CUDA native build instructions as not present.
- Files: `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`, `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java`, `src/test/java/backend/cuda/bridge/CudaFfmBridgeTest.java`, `scripts/build-metal-mps-shim.sh`, `docs/testing.md`, `docs/troubleshooting.md`
- Cause: CUDA ABI consumers are present without a checked-in native shim/build path.
- Improvement path: Add CUDA native source, build scripts, capability tests, and CI coverage, or gate CUDA as unavailable until the native package exists.

**Tuning history append rewrites the whole file:**
- Problem: `JsonFileTuningHistoryStore.append(...)` reads every existing line, appends one entry in memory, then rewrites the whole file.
- Files: `src/main/java/tuning/store/JsonFileTuningHistoryStore.java`
- Cause: Append is implemented with `Files.readAllLines(...)` plus `Files.write(...)` instead of `StandardOpenOption.APPEND`.
- Improvement path: Use append-only writes with file locking or atomic temp-file replacement, and cap or rotate history files for long calibration sessions.

**Large monolithic hot-path implementations increase optimization risk:**
- Problem: Several runtime-critical files are very large: `ScaledDotProductAttentionExecutor.java` is 1,892 lines, `ElementwiseLoops.java` is 1,382 lines, `Conv2dGemmBackend.java` is 1,359 lines, and `MemoryPlanner.java` is 977 lines.
- Files: `src/main/java/backend/cpu/kernels/linalg/ScaledDotProductAttentionExecutor.java`, `src/main/java/backend/cpu/kernels/elementwise/ElementwiseLoops.java`, `src/main/java/backend/cpu/kernels/nn/Conv2dGemmBackend.java`, `src/main/java/graph/optimizer/memory/MemoryPlanner.java`
- Cause: Dtype dispatch, shape handling, vector/scalar loops, backward support, trace publication, and fallback logic live together.
- Improvement path: Split by dtype, execution mode, and planning vs execution responsibilities while preserving existing tests. Add microbenchmarks only outside the default `test` task.

## Fragile Areas

**Native bridge ABI boundaries:**
- Files: `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`, `src/main/native/apple/synaptik_apple_mps_stub.m`, `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`, `src/main/java/backend/blas/OpenBlasFfmBridge.java`
- Why fragile: Java FFM method signatures, native symbol names, dtype codes, rank fields, and status codes must match exactly. Missing symbols often become fallback paths, which can hide regressions unless tests require native execution.
- Safe modification: Change Java and native ABI together, add capability/version checks, update status-code tests, and run `./gradlew metalTest` when touching Metal. Require explicit tests for both native success and fallback failure reasons.
- Test coverage: Java-side bridge and buffer-binding tests exist, but CUDA native runtime coverage is incomplete because native CUDA sources/build scripts are absent.

**ASM fused kernel generation and dynamic classes:**
- Files: `src/main/java/backend/cpu/fused/asm/AsmPreparedFusedExecutableFactory.java`, `src/main/java/backend/cpu/fused/codegen/FusedClassEmitter.java`, `src/main/java/backend/cpu/fused/codegen/FusedAsmSupport.java`, `src/main/java/backend/cpu/fused/codegen/FusedVectorOps.java`
- Why fragile: Bytecode generation, vector widths, dtype specialization, cache keys, and fallback to interpreted execution must stay consistent. Root-level generated `.class` files show this area can leak artifacts into the workspace.
- Safe modification: Keep generated classes under `build/`, validate cache keys when adding operation attributes, and run focused fused execution tests plus source hygiene checks.
- Test coverage: `src/test/java/FusedExecutionModesTest.java` and related fused tests exist, but generated artifact hygiene does not cover the repository root.

**Memory planning and runtime binding:**
- Files: `src/main/java/graph/optimizer/memory/MemoryPlanner.java`, `src/main/java/graph/optimizer/memory/MemoryPlan.java`, `src/main/java/graph/execution/RuntimeMemoryBinder.java`, `src/test/java/graph/optimizer/memory/MemoryPlannerRegionViewTest.java`, `src/test/java/graph/execution/RuntimeMemoryBinderTest.java`
- Why fragile: Correctness depends on lifetimes, saved forward values, gradient targets, view aliases, region materialization, and dtype-specific runtime storage binding all agreeing.
- Safe modification: Add tests for any new operation family that can alias, materialize, use workspaces, or cross forward/backward boundaries. Verify both memory-plan metadata and actual runtime tensor storage.
- Test coverage: There are targeted memory planner and binder tests, but BF16/INT32/BOOL runtime binding remains uncovered because binding is currently disabled for those dtypes.

**Source package architecture is enforced by broad text-scanning tests:**
- Files: `src/test/java/SourceTreeHygieneTest.java`, `src/test/java/CpuKernelFamilyArchitectureTest.java`, `src/test/java/LowercasePackageNamingTest.java`
- Why fragile: Many architectural constraints are encoded as string scans over source files. This protects against regressions but can fail unexpectedly when package names, comments, or documentation strings include legacy names.
- Safe modification: When moving packages, update these tests in the same change. Prefer AST/package-level checks for new constraints instead of broad substring scans.
- Test coverage: Strong for package boundaries, weaker for root-level generated artifacts and tracked calibration outputs.

**Prepared execution side effects on source tensors:**
- Files: `src/main/java/graph/execution/PreparedExecution.java`, `src/main/java/graph/execution/ExecutionState.java`, `src/main/java/tensor/Tensor.java`
- Why fragile: Each execution creates a fresh `ExecutionState`, but output storage and gradients are synchronized back to source tensors. The class documentation states concurrent calls against shared source tensors or backend workspaces are not supported.
- Safe modification: Treat `PreparedExecution` as reusable only for repeated single-threaded runs over the same graph contract. Add explicit synchronization or immutable source snapshots before supporting concurrent execution.
- Test coverage: Tests cover execution isolation and residency, but not a general concurrent execution contract.

## Scaling Limits

**Tensor and shape sizes use `int` products in many paths:**
- Current capacity: Tensor flat sizes, shape dimensions, and native ABI dimensions are mostly represented as `int`; `Tensor.calculateSize(...)` multiplies dimensions with `int`.
- Limit: Large tensors can overflow element counts before allocation or native dispatch, producing wrong sizes or validation mismatches.
- Scaling path: Introduce checked `long` element-count helpers and use them consistently across `TensorMetadata`, storage allocation, memory planning, native byte-size calculations, and tests.

**Native accelerator shape ABI supports only rank 1-4:**
- Current capacity: Metal native helper `SynaptikShapeFromDims(...)` accepts ranks 1 through 4; CUDA bridge marshals `dim0` through `dim3`.
- Limit: Higher-rank tensors must stay on CPU, be flattened/lowered before native dispatch, or fail native planning.
- Scaling path: Encode shape arrays with explicit rank and dimension pointer/length in native ABI, then update lowerers and capability checks.

**Default test heap is fixed at 2g unless overridden:**
- Current capacity: Gradle `Test` tasks use `maxHeapSize = '2g'` by default.
- Limit: Large graph, calibration, or benchmark tests can fail locally with heap pressure or hide memory regressions by relying on a large heap.
- Scaling path: Split heavy tests out of `test`, set memory expectations per task, and track allocation-sensitive workloads with dedicated performance tasks.

## Dependencies at Risk

**Java 25 plus incubator Vector API:**
- Risk: The build requires Java toolchain 25 and `jdk.incubator.vector`, so compiler/runtime availability is narrower and APIs can change.
- Impact: Contributors without JDK 25 cannot build. Vector API changes can break CPU kernel code or Gradle flags.
- Migration plan: Pin and document a tested JDK distribution, keep scalar fallbacks covered, and isolate vector-specific code under focused packages such as `src/main/java/backend/cpu/kernels/elementwise/`.

**ASM bytecode generation:**
- Risk: ASM 9.6 is used for fused class generation and must track classfile/JDK compatibility.
- Impact: JDK upgrades can break generated bytecode verification or runtime loading.
- Migration plan: Add bytecode verification tests for generated fused classes and update ASM before moving beyond currently tested Java versions.

**Optional native BLAS/MPS/CUDA libraries:**
- Risk: OpenBLAS, Metal MPSGraph, and CUDA availability depends on local libraries and symbols outside Gradle dependency management.
- Impact: Performance and even correctness behavior can differ by machine when profiles require native paths.
- Migration plan: Package native shims, add capability discovery reports, and keep Java fallback profiles as portable defaults.

## Missing Critical Features

**CUDA native shim and build workflow:**
- Problem: CUDA Java bridge and preparer code exist without checked-in native source or build script.
- Blocks: Real CUDA execution, native CUDA CI, and reliable CUDA performance claims.

**OpenCL real execution backend:**
- Problem: OpenCL exposes only a legacy `NOOP` kernel registry.
- Blocks: Meaningful OpenCL offload and any cross-vendor accelerator story based on OpenCL.

**Strict persisted-profile schema validation:**
- Problem: Profile parsers accept partial/malformed JSON by falling back to defaults.
- Blocks: Reliable profile migration, reproducible calibration, and clear user diagnostics when a profile is stale or corrupt.

**Separated fast, native, and benchmark test tasks:**
- Problem: Correctness tests, optional-native tests, and benchmark/calibration tests share the default test source set.
- Blocks: Fast CI feedback and predictable local verification.

## Test Coverage Gaps

**Native CUDA execution:**
- What's not tested: End-to-end CUDA native compile and execute against a checked-in native shim.
- Files: `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`, `src/test/java/backend/cuda/bridge/CudaFfmBridgeTest.java`, `docs/testing.md`
- Risk: CUDA Java lowering/preparation can appear complete while native execution remains unavailable.
- Priority: High

**Runtime memory binding for non-float dtypes:**
- What's not tested: Actual reusable runtime slot binding for BF16, INT32, and BOOL tensors.
- Files: `src/main/java/graph/execution/RuntimeMemoryBinder.java`, `src/test/java/graph/execution/RuntimeMemoryBinderTest.java`
- Risk: Memory planner changes can claim reuse that runtime does not perform for these dtypes.
- Priority: Medium

**Prepared execution invalidation:**
- What's not tested: A stale prepared execution rejecting or detecting graph contract changes after prepare.
- Files: `src/main/java/graph/execution/PreparedExecution.java`, `src/main/java/graph/CompiledGraph.java`, `docs/compute-flow.md`
- Risk: Wrong kernels, shape metadata, or backend assumptions can be used after mutation.
- Priority: High

**Root and generated artifact hygiene:**
- What's not tested: Root-level `.class` files, tracked profile run outputs, `.gradle-userhome/`, and IDE metadata.
- Files: `src/test/java/SourceTreeHygieneTest.java`, `build.gradle`, `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/`, `.idea/`
- Risk: Generated and machine-local artifacts continue to enter the workspace or repository.
- Priority: Medium

**Default test duration regression:**
- What's not tested: A time budget or task split proving `./gradlew test` remains a fast deterministic correctness suite.
- Files: `build.gradle`, `src/test/java/debug/`
- Risk: New benchmark/calibration tests make default verification slower and less reliable.
- Priority: Medium

---

*Concerns audit: 2026-04-29*
