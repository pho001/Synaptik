# Technology Stack

**Analysis Date:** 2026-04-29

## Languages

**Primary:**
- Java 25 - Main tensor framework, graph optimizer, backend runtime, tuning CLI, calibration, and tests under `src/main/java` and `src/test/java`. The Gradle toolchain is set in `build.gradle`.

**Secondary:**
- Objective-C - Optional macOS Metal/MPS native shim in `src/main/native/apple/synaptik_apple_mps_stub.m`.
- Shell/Zsh - Native shim build automation in `scripts/build-metal-mps-shim.sh`.
- Markdown - Project documentation under `docs/`, `README.md`, and package docs such as `src/main/java/tensor/README.md`.
- JSON/JSONL - Runtime profile, calibration, autotune, and benchmark artifacts written by `config.profile.*`, `tuning.store.*`, and `tuning.calibration.store.*`.

## Runtime

**Environment:**
- JVM using Java 25 toolchain from `build.gradle`.
- Java modules required at compile/test/run time: `jdk.incubator.vector`, configured in `build.gradle`.
- Native access required for Java FFM bridges: `--enable-native-access=ALL-UNNAMED`, configured for `Test`, `JavaExec`, and application defaults in `build.gradle`.
- Java Foreign Function and Memory API is used directly through `java.lang.foreign.*` in `src/main/java/backend/blas/OpenBlasFfmBridge.java`, `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`, and `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`.

**Package Manager:**
- Gradle wrapper 9.4.1 from `gradle/wrapper/gradle-wrapper.properties`.
- Plugins: `java` and `application` in `build.gradle`; Foojay toolchain resolver `org.gradle.toolchains.foojay-resolver-convention` 0.8.0 in `settings.gradle`.
- Repositories: Maven Central in `build.gradle`.
- Lockfile: Not detected. No `gradle.lockfile` or dependency lock file exists in the repository.

## Frameworks

**Core:**
- Plain Java application/runtime - `src/main/java/synaptik/app/TuningCli.java` is the Gradle application entry point.
- Java Vector API (`jdk.incubator.vector`) - CPU vector paths use imports such as `jdk.incubator.vector.FloatVector`, `DoubleVector`, `VectorSpecies`, and `VectorOperators` under `src/main/java/backend/cpu/kernels/**`.
- Java FFM (`java.lang.foreign`) - Native OpenBLAS, Metal/MPS, and CUDA shim bridges are implemented in `src/main/java/backend/blas/OpenBlasFfmBridge.java`, `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`, and `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`.
- ASM 9.6 - Generated fused JVM bytecode support via `org.ow2.asm:asm:9.6` and `org.ow2.asm:asm-commons:9.6` in `build.gradle`; fused execution policy is centered on `FusedPrimaryBackend.ASM` in `src/main/java/config/runtime/FusedExecutionPolicy.java`.

**Testing:**
- JUnit Jupiter 5.11.2 - Test dependency in `build.gradle`.
- JUnit Platform Launcher - `testRuntimeOnly 'org.junit.platform:junit-platform-launcher'` in `build.gradle`.
- Gradle `Test` tasks call `useJUnitPlatform()` and inherit the Vector API/native-access JVM args from `build.gradle`.
- Optional Metal/MPS test slice is a custom `metalTest` Gradle task in `build.gradle`.

**Build/Dev:**
- Gradle application plugin - Main class `synaptik.app.TuningCli` in `build.gradle`.
- Custom Gradle verification tasks - `verifySourceTreeClean` and `cleanSourceArtifacts` in `build.gradle`.
- Optional native build tasks - `buildMetalMpsShim`, `nativeBuild`, and `metalTest` in `build.gradle`.
- macOS native compiler/toolchain - `scripts/build-metal-mps-shim.sh` invokes `clang` with `Foundation`, `Metal`, `MetalPerformanceShaders`, and `MetalPerformanceShadersGraph`.

## Key Dependencies

**Critical:**
- `org.ow2.asm:asm:9.6` - Bytecode generation dependency for fused ASM execution paths, configured in `build.gradle`.
- `org.ow2.asm:asm-commons:9.6` - ASM helper dependency for generated fused execution support, configured in `build.gradle`.
- `jdk.incubator.vector` - Required module for CPU vectorized kernels under `src/main/java/backend/cpu/kernels/**`.
- Java FFM API - Required for optional native bridge classes under `src/main/java/backend/blas`, `src/main/java/backend/metal/bridge`, and `src/main/java/backend/cuda/bridge`.

**Infrastructure:**
- OpenBLAS CBLAS symbols - Optional CPU matmul/conv2d GEMM provider called by `src/main/java/backend/blas/OpenBlasFfmBridge.java`.
- Apple Metal/MPSGraph frameworks - Optional macOS accelerator path implemented by `src/main/native/apple/synaptik_apple_mps_stub.m` and called by `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`.
- CUDA graph shim ABI - Optional native CUDA graph bridge expected by `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`; no CUDA native source or build task is present in the repository.
- Local filesystem persistence - Calibration, tuning, profile, and benchmark JSON artifacts are handled by classes such as `src/main/java/tuning/store/JsonFileBestProfileStore.java`, `src/main/java/config/profile/PlatformRuntimeProfileIO.java`, and `src/main/java/tuning/calibration/store/CalibrationArtifactLayout.java`.

## Configuration

**Environment:**
- Build/test JVM args are centralized in `runtimeJvmArgs` inside `build.gradle`.
- Test heap defaults to `2g`; override with `-Dsynaptik.testMaxHeap=<size>`, read by `build.gradle`.
- Optional OpenBLAS lookup order: `-Dopenblas.lib=<path>`, `OPENBLAS_LIB`, then library name `openblas`, implemented in `src/main/java/backend/blas/OpenBlasFfmBridge.java`.
- Optional Metal/MPS lookup order: `-Dsynaptik.metal.mps.lib=<path>`, `SYNAPTIK_METAL_MPS_LIB`, then library name `synaptik_apple_mps`, implemented in `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`.
- Optional CUDA graph lookup order: `-Dsynaptik.cuda.graph.lib=<path>`, `SYNAPTIK_CUDA_GRAPH_LIB`, then library name `synaptik_cuda_graph`, implemented in `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`.
- Runtime BLAS property keys are defined in `src/main/java/backend/blas/BlasRuntime.java`: `cg.cpu.blas.provider`, `cg.cpu.blas.matmulMinWork`, `cg.cpu.blas.debug`, `cg.cpu.blas.f32RequireMgeK`, and `cg.cpu.blas.f32MaxNOverK`.
- Runtime optimizer/math flags include `cg.math.forceExactTranscendentals` in `src/main/java/utils/FastTranscendentals.java`, `cg.optimizer.enableMemoryReuse` in `src/main/java/graph/optimizer/memory/MemoryOptimizerRule.java`, and many `cg.optimizer.ar.disable*` rewrite flags in `src/main/java/graph/optimizer/rewrite/AlgebraicRewrite.java`.
- Diagnostic CLIs use system properties such as `numerics.*` in `src/main/java/numerics/NumericsCli.java` and `etalon.*` in `src/main/java/tuning/etalon/FrameworkEtalonCli.java`.

**Build:**
- `build.gradle`: plugins, dependencies, Java 25 toolchain, JVM args, test heap, application main class, source hygiene tasks, and native/Metal tasks.
- `settings.gradle`: root project name `Synaptik` and Foojay resolver plugin.
- `gradle.properties`: contains only a commented `org.gradle.java.home` example.
- `gradle/wrapper/gradle-wrapper.properties`: Gradle wrapper distribution 9.4.1.
- `config/optimizer-profile.json`: Local optimizer profile artifact exists but is ignored by `.gitignore`; it records runtime/tuning values and is not the canonical Gradle build config.
- `.env` files: Not detected.

## Platform Requirements

**Development:**
- Use `./gradlew` or `gradlew.bat` from the repository root.
- JDK 25-capable environment is required by `build.gradle`.
- `jdk.incubator.vector` must be available for compile, test, and run tasks.
- Native FFM bridge use requires `--enable-native-access=ALL-UNNAMED`, already supplied by Gradle tasks in `build.gradle`.
- macOS plus `clang` and Apple frameworks are required for `./gradlew buildMetalMpsShim`, `./gradlew nativeBuild`, and `./gradlew metalTest`; the shell build script is `scripts/build-metal-mps-shim.sh`.
- OpenBLAS tests and BLAS-backed runtime paths require a local OpenBLAS library exposing CBLAS symbols used by `src/main/java/backend/blas/OpenBlasFfmBridge.java`.
- CUDA bridge tests and CUDA-backed execution require an external `synaptik_cuda_graph` native library matching `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`; no repository build task creates it.

**Production:**
- Deployment target is a Java application/library runtime, not a web service. The application main class is `synaptik.app.TuningCli` in `build.gradle`.
- CPU is the complete execution backend according to `README.md`; Metal, CUDA, and OpenCL packages exist as optional/scaffolded accelerator surfaces under `src/main/java/backend/metal`, `src/main/java/backend/cuda`, and `src/main/java/backend/opencl`.
- Runtime profile artifacts live under `profiles/platform/<platform-id>/...` and are read/written by `src/main/java/synaptik/app/TuningCli.java`, `src/main/java/tuning/calibration/store/CalibrationArtifactLayout.java`, and `src/main/java/tuning/store/JsonFileBestProfileStore.java`.

---

*Stack analysis: 2026-04-29*
