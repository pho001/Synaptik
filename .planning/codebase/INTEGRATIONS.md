# External Integrations

**Analysis Date:** 2026-04-29

## APIs & External Services

**Hosted APIs:**
- Not detected. The repository contains no HTTP client, REST endpoint, GraphQL, cloud SDK, payment SDK, or third-party hosted service integration in `src/main/java`.

**Native Math/Accelerator Libraries:**
- OpenBLAS - Optional CPU BLAS provider for matmul and GEMM-lowered conv2d.
  - SDK/Client: Java FFM direct CBLAS bridge in `src/main/java/backend/blas/OpenBlasFfmBridge.java`.
  - Auth: Not applicable.
  - Symbols: `cblas_sgemm`, `cblas_dgemm`, optional `cblas_sbgemm`.
  - Lookup: `openblas.lib` system property, `OPENBLAS_LIB` environment variable, then library name `openblas`.
  - Runtime selector: `BlasProvider.OPENBLAS_FFM` in `src/main/java/backend/blas/BlasProvider.java` and `BlasConfig` in `src/main/java/config/runtime/BlasConfig.java`.
  - Dispatch gates: dtype/work/shape/contiguity checks in `src/main/java/backend/cpu/kernels/linalg/matmul/plan/MatMulPlanner.java` and conv2d checks in `src/main/java/backend/cpu/kernels/nn/conv2d/plan/Conv2dPlanner.java`.
  - Fallback: `src/main/java/backend/cpu/kernels/linalg/matmul/blas/MatMulBlasBackend.java` returns false on unavailable/failed BLAS so Java CPU kernels can execute.
- Apple Metal/MPSGraph - Optional macOS accelerator path for selected FLOAT32 graph regions.
  - SDK/Client: Java FFM bridge in `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`.
  - Native shim: Objective-C source in `src/main/native/apple/synaptik_apple_mps_stub.m`.
  - Build: `scripts/build-metal-mps-shim.sh` invoked by `buildMetalMpsShim` in `build.gradle`; output is `build/native/apple/libsynaptik_apple_mps.dylib`.
  - Auth: Not applicable.
  - Lookup: `synaptik.metal.mps.lib` system property, `SYNAPTIK_METAL_MPS_LIB` environment variable, then library name `synaptik_apple_mps`.
  - Symbols: `synaptik_apple_mps_available`, `synaptik_apple_mps_create_context`, `synaptik_apple_mps_compile_partition_f32`, `synaptik_apple_mps_execute_partition_f32`, buffer symbols such as `synaptik_apple_mps_create_buffer`, `synaptik_apple_mps_read_buffer`, and `synaptik_apple_mps_execute_partition_f32_buffers`.
  - Capability boundary: FLOAT32 compute/output and BOOL predicate inputs are enforced by `src/main/java/backend/metal/MetalMpsCapabilities.java`.
  - Fallback: `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` uses CPU fallback steps when the bridge/context/executable/buffer path is unavailable.
- CUDA graph shim - Optional native CUDA graph execution ABI.
  - SDK/Client: Java FFM bridge in `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`.
  - Auth: Not applicable.
  - Lookup: `synaptik.cuda.graph.lib` system property, `SYNAPTIK_CUDA_GRAPH_LIB` environment variable, then library name `synaptik_cuda_graph`.
  - Symbols: `synaptik_cuda_graph_available`, `synaptik_cuda_graph_create_context`, `synaptik_cuda_graph_compile_partition_f32`, `synaptik_cuda_graph_execute_partition_f32`, `synaptik_cuda_graph_destroy_context`, and `synaptik_cuda_graph_destroy_executable`.
  - Capability boundary: FLOAT32/BOOL external inputs and FLOAT32 outputs are enforced by `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`.
  - Fallback: `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` replays CPU fallback steps when native CUDA execution is unavailable.
- OpenCL - Legacy backend facade with only a NOOP kernel registered.
  - SDK/Client: No external OpenCL SDK binding detected in the repository.
  - Auth: Not applicable.
  - Implementation: `src/main/java/backend/opencl/OpenClBackend.java`, `src/main/java/backend/opencl/registry/OpenClKernelRegistry.java`, and `src/main/java/backend/opencl/kernels/OpenClNoopKernel.java`.

**Benchmark/Calibration Tooling:**
- Gradle application CLI - `src/main/java/synaptik/app/TuningCli.java` runs calibration, autotune, and benchmark flows.
  - SDK/Client: Local Java CLI through `./gradlew run --args="..."`.
  - Auth: Not applicable.
  - Persistence root: defaults to `profiles`, override via `--profile-root`/`--output-root`.
- Numerics harness - `src/main/java/numerics/NumericsCli.java` reads `numerics.*` JVM properties for local numerical drift comparisons.
  - SDK/Client: Java main class, not wired as the Gradle application main.
  - Auth: Not applicable.
- Etalon benchmark harness - `src/main/java/tuning/etalon/FrameworkEtalonCli.java` reads `etalon.suite`, `etalon.preset`, and `etalon.outDir`.
  - SDK/Client: Java main class, not wired as the Gradle application main.
  - Auth: Not applicable.

## Data Storage

**Databases:**
- Not detected. No JDBC, JPA, embedded database, ORM, or database connection configuration exists in `build.gradle` or `src/main/java`.
  - Connection: Not applicable.
  - Client: Not applicable.

**File Storage:**
- Local filesystem only.
- Runtime profiles are read/written by `src/main/java/config/profile/ExecutionProfileIO.java` and `src/main/java/config/profile/PlatformRuntimeProfileIO.java`.
- Calibration artifact paths are managed by `src/main/java/tuning/calibration/store/CalibrationArtifactLayout.java` and `src/main/java/tuning/calibration/store/PlatformCalibrationPaths.java`.
- Best profile and tuning history paths are managed by `src/main/java/tuning/store/JsonFileBestProfileStore.java`, `src/main/java/tuning/store/JsonFileTuningHistoryStore.java`, and `src/main/java/synaptik/app/TuningCli.java`.
- Benchmark report output is written by `src/main/java/tuning/store/JsonFileBenchmarkReportStore.java`.
- Default persisted platform artifacts exist under `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/...`.
- Build/test/benchmark generated outputs are written under `build/`, including `build/native/apple/libsynaptik_apple_mps.dylib` and etalon reports such as `build/tuning-etalon`.

**Caching:**
- In-memory native executable cache for Metal MPS compiled partitions: `EXECUTABLE_CACHE` in `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`.
- Shared native bridge contexts for Metal and CUDA: `SHARED_CONTEXT` in `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java` and `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`.
- Packed linear weight and runtime cache classes exist under `src/main/java/backend/cpu/kernels/linalg/matmul/common/PackedLinearWeightCache.java` and `src/main/java/backend/cpu/kernels/linalg/ScaledDotProductAttentionRuntimeCache.java`.
- External cache services: None detected.

## Authentication & Identity

**Auth Provider:**
- Not detected.
  - Implementation: No authentication, OAuth, session, token, identity-provider, or user account code exists in `src/main/java`.

**Runtime Identity:**
- Local hardware/runtime identity is captured for tuning artifact scoping in `src/main/java/tuning/store/HardwareFingerprint.java`.
- Hardware fingerprint fields are JVM/system facts: `os.name`, `os.arch`, `java.vm.name`, `java.vendor`, and `Runtime.getRuntime().availableProcessors()`.
- Platform ids are derived by `src/main/java/tuning/calibration/store/PlatformCalibrationPaths.java`.

## Monitoring & Observability

**Error Tracking:**
- None. No Sentry, OpenTelemetry, metrics backend, logging service, or hosted error tracker is configured in `build.gradle` or `src/main/java`.

**Logs:**
- Console/stdout/stderr only.
- CLI flows print progress and reports from `src/main/java/synaptik/app/TuningCli.java`.
- BLAS diagnostics print to `System.err` when debug is enabled in `src/main/java/backend/cpu/kernels/linalg/matmul/blas/MatMulBlasBackend.java`.
- Native bridge availability/fallback reasons are represented as records and diagnostics in `src/main/java/backend/metal/bridge/MetalMpsBridgeContext.java`, `src/main/java/backend/cuda/bridge/CudaBridgeContext.java`, and executable classes under `src/main/java/backend/metal/bridge` and `src/main/java/backend/cuda/bridge`.
- Prepared execution traces include backend/native timing fields in classes under `src/main/java/graph/execution/trace` and Metal stats in `src/main/java/backend/metal/bridge/MetalMpsBridgeExecutionStats.java`.

## CI/CD & Deployment

**Hosting:**
- Not detected. No deployment target, container file, server runtime, or hosted platform config exists in the repository.

**CI Pipeline:**
- Not detected. No GitHub Actions, GitLab CI, CircleCI, Jenkinsfile, or similar CI config was found in the repository root or immediate config files.

**Local Build/Verification:**
- Compile: `./gradlew classes`.
- Test: `./gradlew test`.
- Run CLI: `./gradlew run`.
- Optional native Metal build: `./gradlew buildMetalMpsShim`.
- Optional native lifecycle: `./gradlew nativeBuild`.
- Optional Metal/MPS test slice: `./gradlew metalTest`.

## Environment Configuration

**Required env vars:**
- None for the default Java CPU build/test/run path.
- Optional native library lookup variables:
- `OPENBLAS_LIB`: fallback OpenBLAS path/name for `src/main/java/backend/blas/OpenBlasFfmBridge.java`.
- `SYNAPTIK_METAL_MPS_LIB`: fallback Metal/MPS shim path/name for `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`.
- `SYNAPTIK_CUDA_GRAPH_LIB`: fallback CUDA graph shim path/name for `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`.

**System properties:**
- `synaptik.testMaxHeap`: Gradle test heap override, read in `build.gradle`.
- `openblas.lib`: explicit OpenBLAS lookup override in `src/main/java/backend/blas/OpenBlasFfmBridge.java`.
- `synaptik.metal.mps.lib`: explicit Metal/MPS lookup override in `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`.
- `synaptik.cuda.graph.lib`: explicit CUDA graph lookup override in `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`.
- `cg.cpu.blas.provider`, `cg.cpu.blas.matmulMinWork`, `cg.cpu.blas.debug`, `cg.cpu.blas.f32RequireMgeK`, `cg.cpu.blas.f32MaxNOverK`: BLAS runtime toggles in `src/main/java/backend/blas/BlasRuntime.java`.
- `cg.math.forceExactTranscendentals`: math implementation toggle in `src/main/java/utils/FastTranscendentals.java`.
- `cg.optimizer.enableMemoryReuse`: memory optimizer toggle in `src/main/java/graph/optimizer/memory/MemoryOptimizerRule.java`.
- `cg.optimizer.ar.disable*`: algebraic rewrite toggles in `src/main/java/graph/optimizer/rewrite/AlgebraicRewrite.java`.
- `numerics.*`: numerics harness options in `src/main/java/numerics/NumericsCli.java` and `src/main/java/numerics/NumericsHarness.java`.
- `etalon.*`: etalon benchmark options in `src/main/java/tuning/etalon/FrameworkEtalonCli.java`.

**Secrets location:**
- Not applicable. No `.env` files, credential files, secret files, cloud credentials, or package manager token files were detected.
- `.gitignore` excludes local generated artifacts such as `build/`, `.gradle/`, `config/optimizer-profile*.json`, and `todo/`, but not an application secret store.

## Webhooks & Callbacks

**Incoming:**
- None. No server, route, listener, webhook endpoint, socket server, or HTTP framework is detected.

**Outgoing:**
- None. No HTTP client, webhook sender, event bus bridge, queue client, or hosted callback integration is detected.

**Native callbacks:**
- Not detected. OpenBLAS, Metal/MPS, and CUDA integrations use Java FFM downcalls from Java into native code; no FFM upcall/callback bridge is present.

---

*Integration audit: 2026-04-29*
