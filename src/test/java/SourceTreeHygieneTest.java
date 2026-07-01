import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SourceTreeHygieneTest {

    @Test
    void legacyArchitecturePackagesAreRemoved() throws IOException {
        List<Path> legacy = List.of(
                Path.of("src/main/java/graph/execution"),
                Path.of("src/main/java/graph/compile/descriptor"),
                Path.of("src/main/java/graph/compile/intent"),
                Path.of("src/main/java/graph/compile/planning"),
                Path.of("src/main/java/backend/prepare"),
                Path.of("src/main/java/backend/runtime"),
                Path.of("src/main/java/backend/memory"),
                Path.of("src/main/java/backend/blas"),
                Path.of("src/main/java/backend/ComputeEngine.java"),
                Path.of("src/test/java/backend/accelerator/buffer"),
                Path.of("src/test/java/backend/cpu/nativecpu/NativeCpuStorageTest.java")
        );
        List<String> offenders = legacy.stream()
                .flatMap(path -> {
                    try {
                        return javaFilesUnder(path).stream();
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })
                .sorted()
                .toList();
        assertTrue(offenders.isEmpty(), () -> "Legacy architecture Java sources remain: " + offenders);
    }

    @Test
    void openBlasProviderSourcesDoNotImportHigherLayersOrConcreteBackends() throws IOException {
        Path root = Path.of("src/main/java/backend/provider/blas/openblas");
        List<String> projectForbidden = List.of(
                "import config.", "import graph.", "import planning.", "import prepare.",
                "import runtime.", "import trace.", "import tensor.",
                "import backend.cpu.", "import backend.cpu1.", "import backend.metal.",
                "import backend.cuda.", "import backend.opencl."
        );
        List<String> offenders;
        try (Stream<Path> paths = Files.walk(root)) {
            offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .flatMap(path -> {
                        try {
                            return Files.readAllLines(path).stream()
                                    .map(String::trim)
                                    .filter(line -> line.startsWith("import "))
                                    .filter(line -> projectForbidden.stream().anyMatch(line::startsWith)
                                            || (!line.startsWith("import java.")
                                            && !line.startsWith("import static java.")))
                                    .map(line -> path + ": " + line);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sorted()
                    .toList();
        }
        assertTrue(offenders.isEmpty(), () -> "OpenBLAS provider ownership violations: " + offenders);
    }

    @Test
    void planningTmpScratchIsIgnored() throws IOException {
        String gitignore = Files.readString(Path.of(".gitignore"));
        assertTrue(gitignore.contains(".planning/tmp/"), ".planning/tmp/ must stay ignored for local verification scratch files.");

        String trackedScratch = gitOutput("ls-files", ".planning/tmp");
        assertTrue(trackedScratch.isBlank(), () -> ".planning/tmp/ scratch files must not be tracked: " + trackedScratch);

        String ignoredScratch = gitOutput("check-ignore", "--no-index", ".planning/tmp/phase-05-scratch.txt");
        assertTrue(ignoredScratch.contains(".planning/tmp/phase-05-scratch.txt"),
                () -> ".planning/tmp/ scratch files should be ignored, got: " + ignoredScratch);
    }

    @Test
    void rootGeneratedClassArtifactsAreIgnored() throws IOException {
        String gitignore = Files.readString(Path.of(".gitignore"));
        assertTrue(gitignore.contains("/*.class"), "Root generated .class files must stay ignored.");
        assertTrue(gitignore.contains("**/*.class"), "Nested generated .class files must stay ignored.");
    }

    @Test
    void cudaNativeBuildOutputsStayUntracked() throws IOException {
        String gitignore = Files.readString(Path.of(".gitignore"));
        assertTrue(gitignore.contains("build/"), "Gradle build output, including build/native/cuda/, must stay ignored.");

        String trackedCudaNative = gitOutput("ls-files", "build/native/cuda");
        assertTrue(trackedCudaNative.isBlank(), () -> "CUDA native build outputs must not be tracked: " + trackedCudaNative);

        List<String> trackedCudaNativePaths = gitOutput("ls-files").lines()
                .filter(line -> line.contains("build/native/cuda/"))
                .sorted()
                .toList();
        assertTrue(trackedCudaNativePaths.isEmpty(),
                () -> "CUDA native build outputs must not appear in tracked paths: " + trackedCudaNativePaths);
    }

    @Test
    void trackedLocalTuningArtifactsStayExplicit() throws IOException {
        String trackedProfiles = gitOutput("ls-files", "profiles/platform");
        List<String> profileFiles = trackedProfiles.lines()
                .filter(line -> line.contains("/tuning/abc/"))
                .sorted()
                .toList();
        assertTrue(profileFiles.isEmpty() || profileFiles.stream().allMatch(line -> line.startsWith("profiles/platform/")),
                () -> "Tracked profiles/platform/.../tuning/abc/* files are explicit canonical fixtures; "
                        + "do not stage local profile tuning changes accidentally: " + profileFiles);
    }

    @Test
    void sourceTreeDoesNotContainCompiledOrTempArtifacts() throws IOException {
        List<Path> roots = List.of(Path.of("src"), Path.of("test"));
        try (Stream<Path> paths = roots.stream()
                .filter(Files::exists)
                .flatMap(root -> {
                    try {
                        return Files.walk(root);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .map(Path::normalize)
                    .filter(path -> {
                        String name = path.getFileName().toString();
                        return name.endsWith(".class")
                                || name.endsWith(".java.txt")
                                || name.endsWith(".java.bak")
                                || name.endsWith(".java.orig")
                                || name.endsWith(".java.tmp")
                                || name.equals(".DS_Store")
                                || name.startsWith(".tmp")
                                || name.contains(".tmp")
                                || name.endsWith("~");
                    })
                    .map(Path::toString)
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "Source tree contains generated artifacts: " + offenders);
        }
    }

    @Test
    void prepareOrchestrationDoesNotRebuildOptimizerArtifacts() throws IOException {
        Path root = Path.of("src/main/java/prepare/orchestration");
        try (Stream<Path> paths = Files.walk(root)) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> !path.getFileName().toString().equals("SourceTreeHygieneTest.java"))
                    .flatMap(path -> {
                        try {
                            return Files.readAllLines(path).stream()
                                    .filter(line -> line.contains("planning.memory.MemoryPlanner")
                                            || line.contains("planning.partition.execution.PartitionExecutionPlanner")
                                            || line.contains("planning.partition.execution.PartitionExecutionPlanningContext"))
                                    .map(line -> path + ": " + line.trim());
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "prepare orchestration rebuilds optimizer artifacts: " + offenders);
        }
    }

    @Test
    void runtimeMemoryBinderDoesNotUseGlobalMigrationGuards() throws IOException {
        Path binder = Path.of("src/main/java/runtime/residency/RuntimeMemoryBinder.java");
        String source = Files.readString(binder);
        assertTrue(!source.contains("containsPhase12BinderExcludedFamily"), "RuntimeMemoryBinder must not disable binding for a whole graph.");
        assertTrue(!source.contains("skipRuntimeBinding"), "RuntimeMemoryBinder skip policy must be explicit and named.");
        assertTrue(!source.contains("Phase 12"), "RuntimeMemoryBinder must not keep migration-era guard comments.");
        assertTrue(!source.contains("MAX_POOL2D"), "RuntimeMemoryBinder must consume memory-plan binding policy, not hardcode workspace-sensitive families.");
    }

    @Test
    void planningPartitionPackageDoesNotOwnConcreteCpuBackendCode() throws IOException {
        assertPlanningPartitionBackendPackageAbsent("cpu", "CPU backend partition code belongs under backend.cpu.partition");
    }

    @Test
    void planningPartitionPackageDoesNotOwnConcreteCudaBackendCode() throws IOException {
        assertPlanningPartitionBackendPackageAbsent("cuda", "CUDA backend partition code belongs under backend.cuda.lowering");
    }

    @Test
    void planningPartitionPackageDoesNotOwnConcreteAppleBackendCode() throws IOException {
        assertPlanningPartitionBackendPackageAbsent("apple", "Metal backend partition code belongs under backend.metal.lowering");
    }

    @Test
    void planningPartitionPackageDoesNotOwnAcceleratorDagModelCode() throws IOException {
        assertPlanningPartitionBackendPackageAbsent("model", "Accelerator DAG model belongs under backend.accelerator.dag");
    }

    @Test
    void graphOptimizerRulesPackageIsRemoved() throws IOException {
        List<String> offenders = javaFilesUnder(Path.of("src/main/java/graph/optimizer/rules"));
        assertTrue(offenders.isEmpty(), () -> "optimizer stage adapters belong in their domain packages: " + offenders);
    }

    @Test
    void graphOptimizerDoesNotOwnCompilePlanningPackages() throws IOException {
        List<Path> legacyDirs = List.of(
                Path.of("src/main/java/graph/optimizer/partition"),
                Path.of("src/main/java/graph/optimizer/partition"),
                Path.of("src/main/java/graph/optimizer/memory"),
                Path.of("src/main/java/graph/optimizer/intent"),
                Path.of("src/main/java/graph/optimizer/cleanup"),
                Path.of("src/main/java/graph/optimizer/cf"),
                Path.of("src/main/java/graph/optimizer/cse"),
                Path.of("src/main/java/graph/optimizer/dce")
        );
        List<String> offenders = legacyDirs.stream()
                .filter(Files::exists)
                .map(Path::toString)
                .sorted()
                .toList();
        assertTrue(offenders.isEmpty(), () -> "graph.optimizer must stay graph-rewrite-only: " + offenders);
    }

    @Test
    void graphOptimizerDoesNotOwnCompilePlanningValueReferences() {
        List<Path> legacyFiles = List.of(
                Path.of("src/main/java/graph/optimizer/GraphValueRef.java"),
                Path.of("src/main/java/graph/optimizer/GraphValueKind.java")
        );
        List<String> offenders = legacyFiles.stream()
                .filter(Files::exists)
                .map(Path::toString)
                .sorted()
                .toList();
        assertTrue(offenders.isEmpty(), () -> "compile-planning value references belong under planning.value: " + offenders);
    }

    @Test
    void planningPartitionPackageDoesNotImportConcreteBackendImplementations() throws IOException {
        Path root = Path.of("src/main/java/planning/partition");
        try (Stream<Path> paths = Files.walk(root)) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> !path.getFileName().toString().equals("SourceTreeHygieneTest.java"))
                    .flatMap(path -> {
                        try {
                            return Files.readAllLines(path).stream()
                                    .filter(line -> line.contains("backend.metal.lowering")
                                            || line.contains("backend.cuda.lowering")
                                            || line.contains("backend.cpu.partition"))
                                    .map(line -> path + ": " + line.trim());
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "planning.partition imports concrete backend implementations: " + offenders);
        }
    }

    @Test
    void graphPackageDoesNotOwnCpuFusedExecutableRuntime() throws IOException {
        List<String> offenders = javaFilesUnder(Path.of("src/main/java/graph/fused"));
        assertTrue(offenders.isEmpty(), () -> "CPU fused executable runtime belongs under backend.cpu.fused: " + offenders);
    }

    @Test
    void legacyCpuFusedPackageDirectoriesAreRemoved() {
        List<Path> legacyDirs = List.of(
                Path.of("src/main/java/graph/fused"),
                Path.of("src/main/java/graph/codegen"),
                Path.of("src/main/java/graph/optimizer/fusion"),
                Path.of("src/main/java/operations/fused")
        );
        List<String> offenders = legacyDirs.stream()
                .filter(Files::exists)
                .map(Path::toString)
                .sorted()
                .toList();
        assertTrue(offenders.isEmpty(), () -> "Legacy CPU fused package directories remain: " + offenders);
    }

    @Test
    void sourceDoesNotImportLegacyGraphFusedRuntimePackage() throws IOException {
        List<Path> roots = List.of(Path.of("src/main/java"), Path.of("src/test/java"));
        try (Stream<Path> paths = roots.stream()
                .filter(Files::exists)
                .flatMap(root -> {
                    try {
                        return Files.walk(root);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> !path.getFileName().toString().equals("SourceTreeHygieneTest.java"))
                    .flatMap(path -> {
                        try {
                            return Files.readAllLines(path).stream()
                                    .filter(line -> line.contains("graph.fused"))
                                    .map(line -> path + ": " + line.trim());
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "Legacy graph.fused runtime references remain: " + offenders);
        }
    }

    @Test
    void gpuFusionDoesNotImportCpuFusedInternals() throws IOException {
        List<Path> roots = List.of(
                Path.of("src/main/java/backend/accelerator"),
                Path.of("src/main/java/backend/metal"),
                Path.of("src/main/java/backend/cuda")
        );
        try (Stream<Path> paths = roots.stream()
                .filter(Files::exists)
                .flatMap(root -> {
                    try {
                        return Files.walk(root);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .flatMap(path -> {
                        try {
                            return Files.readAllLines(path).stream()
                                    .filter(line -> line.contains("import backend.cpu.fused"))
                                    .map(line -> path + ": " + line.trim());
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "GPU fusion packages import CPU fused internals: " + offenders);
        }
    }

    @Test
    void phaseNineteenAcceleratorPackagesDoNotImportCpuFusedInternals() throws IOException {
        List<String> offenders = linesContainingAny(
                List.of(
                        Path.of("src/main/java/backend/accelerator"),
                        Path.of("src/main/java/backend/metal"),
                        Path.of("src/main/java/backend/cuda")
                ),
                List.of("import backend.cpu.fused")
        );
        assertTrue(offenders.isEmpty(), () -> "Phase 19 GPU partition lowering must not import CPU fused internals: " + offenders);
    }

    @Test
    void phaseNineteenPublicTensorApiDoesNotExposeDeviceResidency() throws IOException {
        List<String> offenders = Files.readString(Path.of("src/main/java/tensor/Tensor.java")).lines()
                .map(String::trim)
                .filter(line -> line.startsWith("public "))
                .filter(line -> line.contains("("))
                .filter(line -> line.contains("DeviceHandle")
                        || line.contains("GpuHandle")
                        || line.contains("residentOnDevice"))
                .sorted()
                .toList();
        assertTrue(offenders.isEmpty(), () -> "Public Tensor API must stay logical, not expose device residency: " + offenders);
    }

    @Test
    void tensorNodeDoesNotOwnBackendIntent() throws IOException {
        String node = Files.readString(Path.of("src/main/java/tensor/TensorNode.java"));
        String access = Files.readString(Path.of("src/main/java/tensor/TensorInternalAccess.java"));
        String tensor = Files.readString(Path.of("src/main/java/tensor/Tensor.java"));
        assertTrue(!node.contains("backend.contract.ComputeBackend"), "TensorNode must not import backend.contract.ComputeBackend.");
        assertTrue(!node.contains("backendIntent"), "Backend intent belongs to compile planning, not TensorNode.");
        assertTrue(!access.contains("setBackendIntent"), "TensorInternalAccess must not expose backend intent mutation.");
        assertTrue(!access.contains("backendIntent("), "TensorInternalAccess must not expose backend intent reads.");
        assertTrue(!tensor.contains("setBackendIntentInternal"), "Tensor must not own backend intent mutation.");
        assertTrue(!tensor.contains("backendIntentInternal"), "Tensor must not own backend intent reads.");
    }

    @Test
    void tensorDocsDoNotDescribeBackendIntentAsTensorState() throws IOException {
        List<String> offenders = sourceLinesContaining(
                List.of(Path.of("src/main/java/tensor/README.md"), Path.of("src/main/java/tensor/API.md")),
                List.of("TensorNode internal backend intent", "backend intent for compile/optimizer use")
        );
        assertTrue(offenders.isEmpty(),
                () -> "Tensor docs must describe backend intent as compile-local planning state, not Tensor/TensorNode state: "
                        + offenders);
    }

    @Test
    void phaseNineteenTensorArrayBridgeIsNotMarkedNativeBufferCoverage() throws IOException {
        String source = Files.readString(Path.of("src/main/java/tuning/benchmark/report/GpuCoverageSummary.java"));
        assertTrue(source.contains("case \"BUFFER_BINDING\" -> coverage.bufferBindingStepCount++;"),
                "BUFFER_BINDING steps must be counted separately as native buffer coverage.");
        assertTrue(source.contains("case \"TENSOR_ARRAY\" -> coverage.tensorArrayStepCount++;"),
                "TENSOR_ARRAY bridge steps must stay separately visible.");

        int methodStart = source.indexOf("public int nativeBufferStepCount()");
        assertTrue(methodStart >= 0, "GpuCoverageSummary.BackendCoverage must expose nativeBufferStepCount().");
        int methodEnd = source.indexOf("\n        }\n", methodStart);
        assertTrue(methodEnd > methodStart, "nativeBufferStepCount() method body must be readable.");
        String methodBody = source.substring(methodStart, methodEnd);
        assertTrue(methodBody.contains("return bufferBindingStepCount;"),
                "nativeBufferStepCount() must return bufferBindingStepCount.");
        assertTrue(!methodBody.contains("tensorArrayStepCount"),
                "nativeBufferStepCount() must not count tensor-array bridge execution.");
    }

    @Test
    void phaseTwentyCoverageGatesDoNotDependOnTimingOnlyMetrics() throws IOException {
        String source = Files.readString(Path.of("src/main/java/tuning/benchmark/report/GpuCoverageRegressionGate.java"));
        List<String> timingOnlyFields = List.of("median", "medianNs", "durationNs");
        List<String> offenders = timingOnlyFields.stream()
                .filter(source::contains)
                .sorted()
                .toList();
        assertTrue(offenders.isEmpty(),
                () -> "Phase 20 coverage gates must use trace/report residency evidence, not timing-only fields: " + offenders);
    }

    @Test
    void phaseTwentyHotPathTargetsRemainSourceOfTruth() throws IOException {
        String source = Files.readString(Path.of("src/main/java/tuning/benchmark/report/GpuHotPathCoverageTargets.java"));
        List<String> requiredTargets = List.of(
                "transformer_block_hot_path",
                "mlp_classifier_small",
                "conv2d_resnet_3x3",
                "layer_norm_small"
        );
        List<String> missing = requiredTargets.stream()
                .filter(target -> !source.contains(target))
                .toList();
        assertTrue(missing.isEmpty(), () -> "Phase 20 hot-path target registry drifted: " + missing);
    }

    @Test
    void phaseTwentyLocalTuningArtifactsRemainNonCanonical() throws IOException {
        String stagedProfiles = gitOutput("diff", "--cached", "--name-only", "--", "profiles/platform");
        List<String> stagedLocalTuningArtifacts = stagedProfiles.lines()
                .filter(line -> line.contains("/tuning/abc/"))
                .sorted()
                .toList();
        assertTrue(stagedLocalTuningArtifacts.isEmpty(),
                () -> "profiles/platform/.../tuning/abc/* are local tuning artifacts, not Phase 20 closure evidence: "
                        + stagedLocalTuningArtifacts);
    }

    @Test
    void acceleratorPackagesRejectCpuFusedOperationType() throws IOException {
        String detector = Files.readString(Path.of("src/main/java/backend/accelerator/lowering/GpuCompoundPatternDetector.java"));

        assertTrue(detector.contains("Operation.OpType.FUSED"));
        assertTrue(detector.contains("CPU_FUSED_OPERATION_UNSUPPORTED"));
    }

    @Test
    void graphPackageDoesNotOwnCpuFusedCodegen() throws IOException {
        List<String> offenders = javaFilesUnder(Path.of("src/main/java/graph/codegen"));
        assertTrue(offenders.isEmpty(), () -> "CPU fused ASM emission belongs under backend.cpu.fused.asm.emit: " + offenders);
    }

    @Test
    void sourceDoesNotImportLegacyGraphCodegenPackage() throws IOException {
        List<Path> roots = List.of(Path.of("src/main/java"), Path.of("src/test/java"));
        try (Stream<Path> paths = roots.stream()
                .filter(Files::exists)
                .flatMap(root -> {
                    try {
                        return Files.walk(root);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> !path.getFileName().toString().equals("SourceTreeHygieneTest.java"))
                    .flatMap(path -> {
                        try {
                            return Files.readAllLines(path).stream()
                                    .filter(line -> line.contains("graph.codegen"))
                                    .map(line -> path + ": " + line.trim());
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "Legacy graph.codegen references remain: " + offenders);
        }
    }

    @Test
    void tuningProductionPackagesDoNotUseLegacySessionOrReportBuckets() throws IOException {
        List<String> legacyDirs = List.of(
                        Path.of("src/main/java/tuning/session"),
                        Path.of("src/main/java/tuning/report")
                ).stream()
                .filter(Files::exists)
                .flatMap(path -> {
                    try {
                        return javaFilesUnder(path).stream();
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })
                .sorted()
                .toList();
        assertTrue(legacyDirs.isEmpty(), () -> "Legacy tuning package directories still contain Java files: " + legacyDirs);

        List<String> offenders = linesContainingAny(
                List.of(Path.of("src/main/java")),
                List.of("package tuning.session;", "import tuning.session.", "package tuning.report;", "import tuning.report.")
        );
        assertTrue(offenders.isEmpty(), () -> "Production code still references legacy tuning packages: " + offenders);
    }

    @Test
    void calibrationDoesNotExposeRemovedFamiliesOrMigrationFallbacks() throws IOException {
        List<String> offenders = sourceLinesContaining(
                List.of(Path.of("src/main/java/tuning"), Path.of("README.md")),
                List.of(
                        "PlatformCalibrationFamily",
                        "FUSED_ARITHMETIC",
                        "FUSED_THRESHOLDS",
                        "CONV2D_GEMM_DISPATCH_F64",
                        "CONV2D_GEMM_DISPATCH_F32",
                        "CONV2D_GEMM_DISPATCH_BF16",
                        "NUMERICS",
                        "ACCELERATOR_METAL_SELECTION",
                        "build/platform-calibration",
                        "CalibrationFamilyTarget",
                        "PlatformCalibrationRunner",
                        "runtime.blas.minWork",
                        "cpu.materialization.cheap",
                        "cpu.materialization.where",
                        "numericsStep",
                        "PlatformRuntimeProfileMutators.blasThreads"
                )
        );
        assertTrue(offenders.isEmpty(), () -> "Removed calibration surfaces remain: " + offenders);
    }

    @Test
    void graphPackageDoesNotOwnCpuFusedOptimizationPolicy() throws IOException {
        List<String> offenders = javaFilesUnder(Path.of("src/main/java/graph/optimizer/fusion"));
        assertTrue(offenders.isEmpty(), () -> "CPU fused planning policy belongs under backend.cpu.fused.plan: " + offenders);
    }

    @Test
    void cpuFusedPackageUsesTargetArchitecturePackages() throws IOException {
        assertTrue(javaFilesUnder(Path.of("src/main/java/backend/cpu/fused/codegen")).isEmpty(),
                "backend.cpu.fused.codegen must not contain production sources");
        assertTrue(javaFilesUnder(Path.of("src/main/java/backend/cpu/fused/optimize")).isEmpty(),
                "backend.cpu.fused.optimize must not contain production sources");

        List<String> forbiddenSources = javaFilesUnder(Path.of("src/main/java/backend/cpu/fused")).stream()
                .filter(path -> path.endsWith("FusedExecutionBackend.java")
                        || path.endsWith("FusedExecutionBackendResolver.java")
                        || path.endsWith("AsmFusedExecutionBackend.java")
                        || path.endsWith("FusedKernelGeneratorRouter.java")
                        || path.endsWith("FusedOperationFactory.java")
                        || path.endsWith("LoweredFusedOperationBuilder.java")
                        || path.endsWith("FusedAsmSupport.java")
                        || path.endsWith("Helper.java")
                        || path.endsWith("Adapter.java"))
                .sorted()
                .toList();
        assertTrue(forbiddenSources.isEmpty(), () -> "CPU fused package still has removed transition or junk-drawer classes: " + forbiddenSources);

        List<String> forbiddenReferences = sourceLinesContaining(
                List.of(Path.of("src/main/java"), Path.of("src/test/java")),
                List.of(
                        "backend.cpu.fused.codegen",
                        "backend/cpu/fused/codegen",
                        "backend.cpu.fused.optimize",
                        "FusedExecutionBackendResolver",
                        "FusedExecutionBackend",
                        "AsmFusedExecutionBackend",
                        "FusedKernelGeneratorRouter",
                        "FusedOperationFactory",
                        "LoweredFusedOperationBuilder",
                        "FusedAsmSupport"
                )
        );
        assertTrue(forbiddenReferences.isEmpty(), () -> "Legacy CPU fused references remain: " + forbiddenReferences);

        List<String> plannerAsmImports = sourceLinesContaining(
                List.of(Path.of("src/main/java/backend/cpu/kernels/plan")),
                List.of("backend.cpu.fused.asm")
        );
        assertTrue(plannerAsmImports.isEmpty(), () -> "CPU planning policy must not import ASM specialization internals: " + plannerAsmImports);
    }

    @Test
    void cpuFusedGeneratedBytecodeDoesNotReadRuntimeApproximationBooleans() throws IOException {
        List<String> offenders = sourceLinesContaining(
                List.of(Path.of("src/main/java/backend/cpu/fused/asm/emit")),
                List.of(
                        "useFastExpApprox",
                        "useFastTanhApprox",
                        "FusedExecutionOptions",
                        "(FZ)F",
                        "(DZ)D",
                        "(Ljava/lang/Object;IZ)"
                )
        );
        assertTrue(offenders.isEmpty(),
                () -> "CPU fused ASM must specialize EXP/TANH approximation at prepare/generation time: " + offenders);
    }

    @Test
    void cpuFusedRuntimeOpsDoNotOwnGeneratedHotPathSemantics() throws IOException {
        String scalarOps = "Fused" + "Scalar" + "Ops";
        String broadcastOps = "Fused" + "Broadcast" + "Vector" + "Ops";
        String storageOps = "Fused" + "Storage" + "Ops";
        String gatherHelper = "Fused" + "Vector" + "Gather" + "Helper";
        String broadcastHelper = "Fused" + "Broadcast" + "Helper";
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/fused/runtime/" + scalarOps + ".java")),
                scalarOps + " must not be restored as a scalar math helper layer");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/fused/runtime/" + broadcastOps + ".java")),
                broadcastOps + " must not hide broadcast/gather vector helper loops");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/fused/runtime/" + storageOps + ".java")),
                storageOps + " must not hide storage/vector helper loops");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/fused/runtime/" + gatherHelper + ".java")),
                gatherHelper + " must not hide generated vector gather bytecode behind a runtime helper");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/fused/runtime/" + broadcastHelper + ".java")),
                broadcastHelper + " must not hide generated vector broadcast bytecode behind a runtime helper");

        List<String> hotPathOffenders = sourceLinesContaining(
                List.of(Path.of("src/main/java/backend/cpu/fused/asm"), Path.of("src/main/java/backend/cpu/fused/plan")),
                List.of(
                        "backend/cpu/fused/runtime/" + scalarOps,
                        "backend/cpu/fused/runtime/" + broadcastOps,
                        "backend/cpu/fused/runtime/" + storageOps,
                        "backend/cpu/fused/runtime/" + gatherHelper,
                        "backend/cpu/fused/runtime/" + broadcastHelper,
                        "loadVectorBF16Array",
                        "storeVectorBF16Array",
                        "loadMaskF32Array",
                        "loadMaskF64Array",
                        "storeMaskF32Array",
                        "storeMaskF64Array"
                )
        );
        assertTrue(hotPathOffenders.isEmpty(),
                () -> "Generated CPU fused hot path must not call removed runtime operation helpers: " + hotPathOffenders);
    }

    @Test
    void cpuFusedVectorOpsRemainAllocationFreePrimitives() throws IOException {
        Path vectorOps = Path.of("src/main/java/backend/cpu/fused/runtime/FusedVectorOps.java");
        String source = Files.readString(vectorOps);
        List<String> forbidden = List.of(
                "new float[",
                "new double[",
                "intoArray(lanes",
                "fromArray(species, lanes",
                "mapUnary",
                "mapBinary",
                "Math.pow",
                "Math.exp",
                "Math.log",
                "Math.tanh",
                "quantizeBF16",
                "powBF16",
                "sigmoidF32",
                "BF16"
        );
        List<String> offenders = forbidden.stream()
                .filter(source::contains)
                .toList();
        assertTrue(offenders.isEmpty(),
                () -> "FusedVectorOps must stay a small allocation-free Vector API primitive library: " + offenders);
    }

    @Test
    void cpuFusedVectorSpeciesSelectionHasSingleOwner() throws IOException {
        List<String> speciesConstants = List.of("SPECIES_64", "SPECIES_128", "SPECIES_256", "SPECIES_512");
        List<String> offenders = javaFilesUnder(Path.of("src/main/java/backend/cpu/fused")).stream()
                .filter(path -> !path.endsWith("backend/cpu/fused/runtime/FusedVectorSpecies.java"))
                .flatMap(path -> {
                    try {
                        String source = Files.readString(Path.of(path));
                        return speciesConstants.stream()
                                .filter(source::contains)
                                .map(constant -> path + ": " + constant);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })
                .sorted()
                .toList();
        assertTrue(offenders.isEmpty(),
                () -> "CPU fused fixed Vector API species selection must be owned by FusedVectorSpecies: " + offenders);
    }

    @Test
    void cpuFusedPackageUsesExplicitStorageContractNames() throws IOException {
        List<String> offenders = sourceLinesContaining(
                List.of(Path.of("src/main/java/backend/cpu/fused"), Path.of("src/test/java/backend/cpu/fused")),
                List.of("CPU_NATIVE", "CPU_OFF_HEAP", "OFF_HEAP")
        );
        assertTrue(offenders.isEmpty(),
                () -> "CPU fused storage contracts must use CPU_JAVA_ARRAY and CPU_MEMORY_SEGMENT only: " + offenders);
    }

    @Test
    void cpuFusedMemorySegmentPathDoesNotBindGeneratedKernelsToTensorArrays() throws IOException {
        String inputEmitter = Files.readString(Path.of("src/main/java/backend/cpu/fused/asm/emit/FusedInputBindingEmitter.java"));
        String outputEmitter = Files.readString(Path.of("src/main/java/backend/cpu/fused/asm/emit/FusedOutputBindingEmitter.java"));
        String vectorEmitter = Files.readString(Path.of("src/main/java/backend/cpu/fused/asm/emit/FusedVectorMethodEmitter.java"));
        String runtimeCalls = Files.readString(Path.of("src/main/java/backend/cpu/fused/asm/emit/FusedRuntimeCalls.java"));
        String preparer = Files.readString(Path.of("src/main/java/backend/cpu/fused/exec/FusedExecutablePreparer.java"));

        assertTrue(inputEmitter.contains("emitGetNativeInputSegmentCall"),
                "segment fused input binding must use the explicit MemorySegment runtime call");
        assertTrue(outputEmitter.contains("emitGetNativeOutputSegmentCall"),
                "segment fused output binding must use the explicit MemorySegment runtime call");
        assertTrue(runtimeCalls.contains("backend/cpu/fused/exec/FusedNativeSegmentBindings"),
                "generated segment kernels must call the fused-owned MemorySegment binding boundary");
        assertFalse(runtimeCalls.contains("fusedNativeInputSegment"),
                "generated segment kernels must not call MemorySegment accessors on CpuKernelContext");
        assertFalse(runtimeCalls.contains("fusedNativeOutputSegment"),
                "generated segment kernels must not call MemorySegment accessors on CpuKernelContext");
        assertTrue(vectorEmitter.contains("FusedVectorGuard.supportsAllocationFreeVectorPath(context.numericContract(), plan)"),
                "segment fused vector support must use the allocation-free vector guard shared with dispatch planning");
        assertTrue(preparer.contains("refusing Java-array interpreter fallback"),
                "segment fused ASM failures must not use the array-bound interpreted fallback");
    }

    @Test
    void sourceDoesNotImportLegacyGraphFusionPackage() throws IOException {
        List<Path> roots = List.of(Path.of("src/main/java"), Path.of("src/test/java"));
        try (Stream<Path> paths = roots.stream()
                .filter(Files::exists)
                .flatMap(root -> {
                    try {
                        return Files.walk(root);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> !path.getFileName().toString().equals("SourceTreeHygieneTest.java"))
                    .flatMap(path -> {
                        try {
                            return Files.readAllLines(path).stream()
                                    .filter(line -> line.contains("graph.optimizer.fusion"))
                                    .map(line -> path + ": " + line.trim());
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "Legacy graph.optimizer.fusion references remain: " + offenders);
        }
    }

    @Test
    void operationsPackageDoesNotOwnCpuFusedPlanDescriptors() throws IOException {
        List<String> offenders = javaFilesUnder(Path.of("src/main/java/operations/fused"));
        assertTrue(offenders.isEmpty(), () -> "CPU fused plan descriptors belong under backend.cpu.fused.plan: " + offenders);
    }

    @Test
    void sourceDoesNotImportLegacyOperationsFusedPackage() throws IOException {
        List<Path> roots = List.of(Path.of("src/main/java"), Path.of("src/test/java"));
        try (Stream<Path> paths = roots.stream()
                .filter(Files::exists)
                .flatMap(root -> {
                    try {
                        return Files.walk(root);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> !path.getFileName().toString().equals("SourceTreeHygieneTest.java"))
                    .flatMap(path -> {
                        try {
                            return Files.readAllLines(path).stream()
                                    .filter(line -> line.contains("operations.fused"))
                                    .map(line -> path + ": " + line.trim());
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "Legacy operations.fused references remain: " + offenders);
        }
    }

    @Test
    void tensorOpsDoNotDependOnOptimizerRewriteHelpers() throws IOException {
        Path root = Path.of("src/main/java/tensor/ops");
        try (Stream<Path> paths = Files.walk(root)) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .flatMap(path -> {
                        try {
                            return Files.readAllLines(path).stream()
                                    .filter(line -> line.contains("graph.optimizer.rewrite"))
                                    .map(line -> path + ": " + line.trim());
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "tensor ops must not depend on optimizer rewrite helpers: " + offenders);
        }
    }

    @Test
    void tensorOpsDoNotImportBackendOrCompileIntent() throws IOException {
        List<String> offenders = linesContainingAny(
                Path.of("src/main/java/tensor/ops"),
                List.of("import backend.", "import planning.intent.")
        );
        assertTrue(offenders.isEmpty(), () -> "tensor.ops must stay semantic and backend-neutral: " + offenders);
    }

    @Test
    void graphOptimizerDoesNotDependOnCompileIntent() throws IOException {
        List<String> offenders = linesContainingAny(
                Path.of("src/main/java/graph/optimizer"),
                List.of("import graph.compile.", "BackendIntentPlan")
        );
        assertTrue(offenders.isEmpty(),
                () -> "graph.optimizer must expose graph rewrites, not consume compile-owned backend intent: "
                        + offenders);
    }

    @Test
    void graphPackageDoesNotImportCpuImplementationDetails() throws IOException {
        List<String> offenders = sourceLinesContaining(
                List.of(Path.of("src/main/java/graph")),
                List.of(
                        "import backend.cpu.fused.plan.FusedOperation",
                        "import tensor.dtype.TensorDTypeOps"
                )
        );
        assertTrue(offenders.isEmpty(),
                () -> "graph package must stay backend-neutral and must not import CPU fused/kernel implementation details: "
                        + offenders);
    }

    @Test
    void backendIntentPlanDoesNotUseGlobalRecordingSideChannel() throws IOException {
        List<String> offenders = sourceLinesContaining(
                List.of(Path.of("src/main/java"), Path.of("src/test/java")),
                List.of("BackendIntentPlan.record", "recordedBackend", "RECORDED_INTENTS", "WeakHashMap")
        );
        assertTrue(offenders.isEmpty(),
                () -> "Backend intent must be passed as a compile-local plan, not recorded through global state: "
                        + offenders);
    }

    @Test
    void tensorOpsDoNotContainGenericSupportHelperAdapterOrV2Classes() throws IOException {
        List<String> offenders;
        try (Stream<Path> paths = Files.walk(Path.of("src/main/java/tensor/ops"))) {
            offenders = paths
                    .filter(Files::isRegularFile)
                    .map(Path::getFileName)
                    .map(Path::toString)
                    .filter(name -> name.endsWith("Support.java")
                            || name.endsWith("Helper.java")
                            || name.endsWith("Adapter.java")
                            || name.endsWith("Bridge.java")
                            || name.contains("V2"))
                    .sorted()
                    .toList();
        }
        assertTrue(offenders.isEmpty(),
                () -> "tensor.ops should use concrete operation classes or narrow domain names, not generic cleanup classes: "
                        + offenders);
    }

    @Test
    void tensorPackageDoesNotContainGenericSupportHelperAdapterOrV2Classes() throws IOException {
        List<String> offenders;
        try (Stream<Path> paths = Files.walk(Path.of("src/main/java/tensor"))) {
            offenders = paths
                    .filter(Files::isRegularFile)
                    .map(Path::getFileName)
                    .map(Path::toString)
                    .filter(name -> name.endsWith("Support.java")
                            || name.endsWith("Helper.java")
                            || name.endsWith("Adapter.java")
                            || name.endsWith("Bridge.java")
                            || name.contains("V2"))
                    .sorted()
                    .toList();
        }
        assertTrue(offenders.isEmpty(),
                () -> "tensor package should use domain names, not generic support/helper/adapter/bridge/V2 classes: "
                        + offenders);
    }

    @Test
    void publicTensorDoesNotImportConcreteOperationBuilders() throws IOException {
        List<String> offenders = Files.readString(Path.of("src/main/java/tensor/Tensor.java")).lines()
                .map(String::trim)
                .filter(line -> line.startsWith("import tensor.ops."))
                .sorted()
                .toList();
        assertTrue(offenders.isEmpty(), () -> "Tensor instance methods must delegate through TensorOps: " + offenders);
    }

    @Test
    void tensorOpsFacadeDoesNotOwnBackendCompileOrMutationPolicy() throws IOException {
        String source = Files.readString(Path.of("src/main/java/tensor/TensorOps.java"));
        List<String> forbidden = List.of(
                "import backend.",
                "import graph.compile.",
                "TensorInternalAccess.setGradient",
                "TensorInternalAccess.replaceStorage",
                "TensorInternalAccess.markStorageModified",
                ".setGradient(",
                ".getStorage()",
                ".markStorageModified()"
        );
        List<String> offenders = forbidden.stream()
                .filter(source::contains)
                .sorted()
                .toList();
        assertTrue(offenders.isEmpty(), () -> "TensorOps must remain a thin operation facade: " + offenders);
    }

    @Test
    void tensorAutogradRulesDoNotUseRunnableOrDirectGradientMutation() throws IOException {
        List<String> offenders = linesContainingAny(
                List.of(Path.of("src/main/java/tensor/ops"), Path.of("src/main/java/tensor/internal")),
                List.of("Runnable", "setBackwardFunction", "TensorInternalAccess.setGradient(", "accumulateGradient(")
        );
        assertTrue(offenders.isEmpty(), () -> "Tensor gradient rules must use GradientRule and GradientContext.accumulate: " + offenders);
    }

    @Test
    void publicTensorDoesNotExposeRawStorageAccessors() throws IOException {
        String source = Files.readString(Path.of("src/main/java/tensor/Tensor.java"));
        List<String> removedMethods = List.of(
                "public TensorStorage getStorage(",
                "public void markStorageModified(",
                "public float[] getFloat32Data(",
                "public double[] getFloat64Data(",
                "public short[] getBFloat16Data(",
                "public int[] getInt32Data(",
                "public long[] getInt64Data(",
                "public byte[] getBoolData(",
                "public double[] getData(",
                "public void markDataViewStale("
        );
        List<String> offenders = removedMethods.stream()
                .filter(source::contains)
                .sorted()
                .toList();
        assertTrue(offenders.isEmpty(), () -> "Tensor public API must not expose raw mutable storage accessors: " + offenders);
    }

    @Test
    void publicTensorDoesNotExposeRedundantShapeUtilityWrappers() throws IOException {
        String source = Files.readString(Path.of("src/main/java/tensor/Tensor.java"));
        List<String> removedMethods = List.of(
                "public int calculateSize(",
                "public int[] computeStrides(int[]",
                "public int[] computeStrides()"
        );
        List<String> offenders = removedMethods.stream()
                .filter(source::contains)
                .sorted()
                .toList();
        assertTrue(offenders.isEmpty(),
                () -> "Tensor public API should use existing shape/metadata APIs instead of redundant utility wrappers: "
                        + offenders);
    }

    @Test
    void productionDoesNotCallRawPublicTensorStorageAccessors() throws IOException {
        List<String> offenders = sourceLinesContaining(
                List.of(Path.of("src/main/java")),
                List.of(".getStorage()", ".getFloat32Data()", ".getFloat64Data()", ".getBFloat16Data()",
                        ".getInt32Data()", ".getInt64Data()", ".getBoolData()", ".getData()",
                        ".markDataViewStale()", "\"getFloat32Data\"", "\"getFloat64Data\"", "\"getBFloat16Data\"",
                        "\"getInt32Data\"", "\"getInt64Data\"", "\"getBoolData\"")
        ).stream()
                .filter(line -> !line.startsWith("src/main/java/tensor/Tensor.java:"))
                .sorted()
                .toList();
        assertTrue(offenders.isEmpty(), () -> "Production code must use TensorInternalAccess or logical copy APIs instead of raw public Tensor storage access: " + offenders);
    }

    @Test
    void testsDoNotCallRawPublicTensorStorageAccessors() throws IOException {
        List<String> offenders = sourceLinesContaining(
                List.of(Path.of("src/test/java")),
                List.of(".getStorage()", ".getFloat32Data()", ".getFloat64Data()", ".getBFloat16Data()",
                        ".getInt32Data()", ".getInt64Data()", ".getBoolData()", ".getData()",
                        ".markDataViewStale()")
        );
        assertTrue(offenders.isEmpty(), () -> "Tests should use logical copy APIs or TensorInternalAccess for explicit storage assertions: " + offenders);
    }

    @Test
    void defaultPartitionPlannerDoesNotOwnCpuMixedUnitPolicy() throws IOException {
        Path planner = Path.of("src/main/java/planning/partition/execution/PartitionExecutionPlanner.java");
        String source = Files.readString(planner);
        assertTrue(!source.contains("buildMixedCpuUnits"), "CPU mixed-unit policy belongs in CpuPartitionExecutionPlanningPolicy.");
        assertTrue(!source.contains("fused-subchain"), "CPU fused-subchain policy belongs outside PartitionExecutionPlanner.");
        assertTrue(!source.contains("isSubchainFusable"), "CPU subchain fusion checks belong outside PartitionExecutionPlanner.");
        assertTrue(!source.contains("consumesUnitOutput"), "CPU subchain traversal belongs outside PartitionExecutionPlanner.");
    }

    @Test
    void partitionPlannerDoesNotOwnBackendLoweringPolicy() throws IOException {
        Path partitionRoot = Path.of("src/main/java/planning/partition/execution");
        assertTrue(!Files.exists(partitionRoot.resolve("PartitionOptimizationPolicy.java")),
                "Partition planner should not keep an extra policy abstraction layer.");
        assertTrue(!Files.exists(partitionRoot.resolve("GenericGpuPartitionOptimizationPolicy.java")),
                "Generic GPU policy should be expressed as backend-neutral structural partition planning.");
        List<String> offenders = linesContainingAny(
                partitionRoot,
                List.of(
                        "LoweringFamily.",
                        "BackendCapabilities",
                        "ComputeBackend.GPU_METAL",
                        "ComputeBackend.GPU_CUDA",
                        "METAL_GRAPH",
                        "CUDA_GRAPH"
                )
        );
        assertTrue(offenders.isEmpty(), () -> "Partition planner must stay backend-lowering neutral: " + offenders);
    }

    @Test
    void prepareDoesNotGloballySkipLoweringForLegacyFusedGraphs() throws IOException {
        Path builder = Path.of("src/main/java/prepare/orchestration/PreparedExecutionBuilder.java");
        String source = Files.readString(builder);
        assertTrue(!source.contains("containsLegacyFusedGraph"), "prepare must not globally suppress lowered-partition publication for legacy fused nodes.");
        assertTrue(!source.contains("OpType.FUSED"), "legacy fused nodes must not be a prepare-layer global lowering gate.");
    }

    @Test
    void partitionExecutionRuleAdapterIsRemoved() {
        Path rule = Path.of("src/main/java/planning/partition/PartitionOptimizationRule.java");
        assertTrue(!Files.exists(rule), "Partition optimization is a compile-planning service, not an optimizer rule adapter.");
    }

    @Test
    void cpuNodePreparerDoesNotInlineLoweredFusedDescriptorSynthesis() throws IOException {
        Path preparer = Path.of("src/main/java/backend/cpu/prepare/CpuNodePreparer.java");
        String source = Files.readString(preparer);
        assertTrue(!source.contains("synthesizeFusedPreparation"), "Lowered fused descriptor construction belongs under backend.cpu.fused.plan.");
        assertTrue(!source.contains("FusedOperationBuilder"), "CpuNodePreparer should consume backend CPU fused plan preparation, not build descriptors inline.");
    }

    @Test
    void preparedExecutionBuilderDoesNotOwnCompileArtifactRecovery() throws IOException {
        Path builder = Path.of("src/main/java/prepare/orchestration/PreparedExecutionBuilder.java");
        String source = Files.readString(builder);
        assertTrue(!source.contains("MemoryPlanner"), "prepare must consume compile artifacts instead of rebuilding memory plans.");
        assertTrue(!source.contains("PartitionExecutionPlanner"), "prepare must consume compile artifacts instead of rebuilding planned partitions.");
        assertTrue(source.contains("loweringInput"), "prepare must rely on CompileArtifacts lowering input contract.");
    }

    @Test
    void partitionExecutionRoleDoesNotLeakIntoGraphRuntimeContract() throws IOException {
        List<String> offenders = sourceLinesContaining(
                List.of(
                        Path.of("src/main/java/runtime/execution/PreparedStepMetadata.java"),
                        Path.of("src/main/java/backend/ComputeEngine.java"),
                        Path.of("src/main/java/prepare/orchestration/PreparedExecutionBuilder.java"),
                        Path.of("src/test/java/testsupport/MetadataArtifacts.java")
                ),
                List.of("PartitionExecutionRole", "partitionRole()")
        );
        assertTrue(offenders.isEmpty(),
                () -> "Partition roles belong to prepare.context and must not be graph runtime metadata/dispatch contract: "
                        + offenders);
    }

    @Test
    void compiledNodesDoNotExposePublicationTensorBindings() throws IOException {
        List<String> offenders = sourceLinesContaining(
                List.of(Path.of("src/main/java")),
                List.of("publicationTensor(")
        );
        assertTrue(offenders.isEmpty(), () -> "Publication tensors belong in PublicationPlan, not CompiledNode: " + offenders);
    }

    @Test
    void compileArtifactsDoesNotExposeMutableFinalGraphOrTrace() throws IOException {
        String source = Files.readString(Path.of("src/main/java/graph/compile/CompileArtifacts.java"));
        assertTrue(!source.contains("finalGraph("), "CompileArtifacts must not expose mutable Tensor finalGraph.");
        assertTrue(!source.contains("partitionPlanningTrace"), "CompileArtifacts must not own diagnostic partition trace.");
    }

    @Test
    void compileSessionOwnsWorkflowWithoutStaticStages() throws IOException {
        String source = Files.readString(Path.of("src/main/java/graph/compile/session/CompileSession.java"));
        assertTrue(source.contains("captureForwardGraph()"), "CompileSession should expose the compile flow as named methods.");
        assertTrue(source.contains("optimizeGraph("), "CompileSession should keep optimizer flow explicit.");
        assertTrue(source.contains("snapshotProgram("), "CompileSession should keep immutable snapshot creation explicit.");
        assertTrue(source.contains("planBackendOwnership("), "CompileSession should own backend ownership planning.");
        assertTrue(source.contains("planExecutablePartitionsAndMemory("), "CompileSession should own executable partition and memory planning.");
        assertTrue(source.contains("buildPublicationPlan("), "CompileSession should own publication plan assembly.");
        List<String> removedStages = List.of(
                "ForwardGraphCapture.java",
                "OptimizerSnapshotStage.java",
                "CompiledProgramSnapshotStage.java",
                "BackendOwnershipPlanningStage.java",
                "PartitionAndMemoryPlanningStage.java",
                "PublicationPlanBuilder.java"
        );
        List<String> existingStages = removedStages.stream()
                .map(name -> Path.of("src/main/java/graph/compile/session", name))
                .filter(Files::exists)
                .map(Path::toString)
                .toList();
        assertTrue(existingStages.isEmpty(), () -> "CompileSession should not be split through static stage/pass-through classes: " + existingStages);
    }

    @Test
    void graphPackageDoesNotContainSupportHelperOrAdapterJunkDrawers() throws IOException {
        List<String> offenders;
        try (var paths = Files.walk(Path.of("src/main/java/graph"))) {
            offenders = paths
                    .filter(Files::isRegularFile)
                    .map(Path::getFileName)
                    .map(Path::toString)
                    .filter(name -> name.endsWith("Support.java")
                            || name.endsWith("Helper.java")
                            || name.endsWith("Adapter.java"))
                    .sorted()
                    .toList();
        }
        assertTrue(offenders.isEmpty(), () -> "graph package should use domain names, not support/helper/adapter junk drawers: " + offenders);
    }

    @Test
    void maxPartitionPartitionPlannerIsSingleAlgorithmWithSeedOrdering() throws IOException {
        assertTrue(Files.exists(Path.of("src/main/java/planning/partition/MaxPartitionPlanner.java")),
                "Max-partition planning should use one concrete planner.");
        assertTrue(!Files.exists(Path.of("src/main/java/planning/partition/GreedyMaxPartitionPlanner.java")),
                "Node-order max-partition planning must not keep a duplicate planner class.");
        assertTrue(!Files.exists(Path.of("src/main/java/planning/partition/AnchorBasedPartitionPlanner.java")),
                "Anchor-first max-partition planning must not keep a duplicate planner class.");
        String source = Files.readString(Path.of("src/main/java/planning/partition/MaxPartitionPlanner.java"));
        assertTrue(source.contains("enum SeedOrdering"), "The max-partition planner should express mode differences through seed ordering.");
        assertTrue(source.contains(".partitionPriority("), "Backend-specific partition priority must come from backend capability.");
        assertTrue(!source.contains("Operation.OpType"), "Max-partition traversal must not hardcode backend operation-family priorities.");
    }

    @Test
    void partitionPlanningRequestUsesBackendCapabilityAndNeutralCostPreset() throws IOException {
        String source = Files.readString(Path.of("src/main/java/planning/partition/PartitionPlanningRequest.java"));
        assertTrue(source.contains("BackendPartitionCapability"), "Partition planning must consume backend capability directly.");
        assertTrue(source.contains("StaticCostPreset"), "Partition planning cost input must stay backend-neutral.");
        assertTrue(!source.contains("PartitionLegalityAdapter"), "Partition planning must not depend on adapter contracts.");
        assertTrue(!source.contains("TransferCostPreset"), "Partition planning request must not own profile cost preset details.");
        assertTrue(!source.contains("transferCostPreset"), "Partition planning request must not expose profile cost preset fields.");
    }

    @Test
    void genericTransferCostConfigDoesNotUseMetalTransferModel() throws IOException {
        List<String> offenders = linesContainingAny(
                        Path.of("src/main/java"),
                        List.of("config.optimizer.MetalTransferModel", "MetalTransferModel", "metalTransferModel")
                )
                .stream()
                .filter(line -> !line.contains("\"partitionMetalTransferModel\""))
                .sorted()
                .toList();
        assertTrue(offenders.isEmpty(), () -> "Generic compile/profile cost config must use TransferCostPreset: " + offenders);

        String profileIo = Files.readString(Path.of("src/main/java/config/profile/ExecutionProfileIO.java"));
        String writer = profileIo.substring(
                profileIo.indexOf("public static String toJson"),
                profileIo.indexOf("private static String jsonStringArray")
        );
        assertTrue(profileIo.contains("partitionMetalTransferModel"),
                "ExecutionProfileIO should keep a read-only migration fallback for old profiles.");
        assertTrue(writer.contains("partitionTransferCostPreset"),
                "ExecutionProfileIO must write the backend-neutral transfer cost key.");
        assertTrue(!writer.contains("partitionMetalTransferModel"),
                "ExecutionProfileIO must not write the old Metal-specific transfer key.");
    }

    @Test
    void productionPartitionContractsDoNotUseLegalityAdapters() throws IOException {
        List<String> offenders = sourceLinesContaining(
                List.of(Path.of("src/main/java/backend/partition"), Path.of("src/main/java/planning/partition")),
                List.of("PartitionLegalityAdapter", "legalityAdapter")
        );
        assertTrue(offenders.isEmpty(), () -> "Partition ownership must use backend capabilities, not adapters: " + offenders);
    }

    @Test
    void compiledProgramStaysExecutableValueOnly() throws IOException {
        String source = Files.readString(Path.of("src/main/java/graph/compile/CompiledProgram.java"));
        List<String> offenders = List.of("tensor.Tensor", "PublicationPlan", "CompileTrace", "PartitionCompileTrace")
                .stream()
                .filter(source::contains)
                .toList();
        assertTrue(offenders.isEmpty(), () -> "CompiledProgram must not own publication or diagnostics: " + offenders);
    }

    @Test
    void compiledGraphDoesNotExposeMutableCompileOrTrainingConvenience() throws IOException {
        String source = Files.readString(Path.of("src/main/java/graph/CompiledGraph.java"));
        assertTrue(!source.contains("public void compile("), "CompiledGraph must be an immutable compile result.");
        assertTrue(!source.contains("public void execute("),
                "One-shot execution convenience belongs on PreparedExecution, not CompiledGraph.");
        assertTrue(!source.contains("public RunTrace executeTraced("),
                "Traced execution convenience belongs on PreparedExecution, not CompiledGraph.");
        assertTrue(!source.contains("executePrepared("),
                "Prepared plan execution belongs on PreparedExecution, not CompiledGraph.");
        assertTrue(!source.contains("executeOptimizerStep("),
                "Optimizer-step convenience belongs on PreparedExecution, not CompiledGraph.");
        assertTrue(!source.contains("zeroGrad("), "Gradient mutation convenience must stay out of CompiledGraph.");
        assertTrue(!source.contains("getRootTensor("), "CompiledGraph must not expose mutable root Tensor state.");
        assertTrue(!source.contains("public List<CompiledNode> compiledNodes("),
                "CompiledGraph must expose compiled nodes through the read-only program accessor.");
        assertTrue(!source.contains("public CompileArtifacts compileArtifacts("),
                "CompiledGraph must not expose broad compile artifacts.");
        assertTrue(!source.contains("getCompiledGraphAsList("),
                "Mutable optimized Tensor graph must not be exposed as public compile facade state.");
        assertTrue(source.contains("public CompiledProgram program()"),
                "CompiledGraph should keep a narrow read-only program inspector.");
        assertTrue(source.contains("public PublicationPlan publication()"),
                "CompiledGraph should keep a narrow read-only publication inspector.");
    }

    @Test
    void preparedExecutionDelegatesPerRunStateToExecutionRun() throws IOException {
        String source = Files.readString(Path.of("src/main/java/runtime/execution/PreparedExecution.java"));
        assertTrue(source.contains("new ExecutionRun("), "PreparedExecution should delegate one-run state ownership.");
        assertTrue(!source.contains("ExecutionState.create("), "ExecutionRun must own per-run execution state creation.");
        assertTrue(!source.contains("ExecutionPublisher."), "ExecutionRun must own runtime publication orchestration.");
        assertTrue(!source.contains("RuntimeMemoryBinder.bind("), "ExecutionRun must own runtime memory binding.");
    }

    @Test
    void preparedExecutionDoesNotOwnBackendTraceAttributeDetails() throws IOException {
        String preparedExecution = Files.readString(Path.of("src/main/java/runtime/execution/PreparedExecution.java"));
        String stepTracer = Files.readString(Path.of("src/main/java/runtime/runner/StepExecutionTracer.java"));
        assertTrue(stepTracer.contains("traceContribution("),
                "StepExecutionTracer should consume backend-owned trace contribution from prepared artifacts.");
        assertTrue(!stepTracer.contains("BackendRunTraceContributors"),
                "StepExecutionTracer must not route through a broad backend contributor registry.");
        assertTrue(!preparedExecution.contains("BackendRunTraceContributors"),
                "PreparedExecution should delegate step trace assembly to StepExecutionTracer.");
        assertTrue(!preparedExecution.contains("\"metalBridgeAvailable\""), "Metal trace attributes belong in backend-owned trace contribution.");
        assertTrue(!preparedExecution.contains("\"cudaBridgeAvailable\""), "CUDA trace attributes belong in backend-owned trace contribution.");
        assertTrue(!preparedExecution.contains("\"nativeCpuPartitionId\""), "Native CPU trace attributes belong in the CPU trace contributor.");
        assertTrue(!preparedExecution.contains("\"acceleratorBufferMode\""), "Accelerator buffer trace attributes belong in the accelerator trace contributor.");
    }

    @Test
    void topLevelTraceDoesNotImportConcreteBackendDetails() throws IOException {
        List<String> offenders = linesContainingAny(
                Path.of("src/main/java/trace"),
                List.of(
                        "import backend.cpu.",
                        "import backend.provider.blas.openblas.",
                        "import backend.metal.",
                        "import backend.cuda."
                )
        );
        assertTrue(offenders.isEmpty(), () -> "top-level trace must consume backend-owned trace contributions: " + offenders);
    }

    @Test
    void runtimeWorkspaceStoreDoesNotImportConcreteBackendDetails() throws IOException {
        String source = Files.readString(Path.of("src/main/java/runtime/execution/RuntimeWorkspaceStore.java"));
        assertTrue(!source.contains("import backend.cpu."), "RuntimeWorkspaceStore must store backend workspaces opaquely.");
        assertTrue(!source.contains("import backend.provider.blas.openblas."),
                "RuntimeWorkspaceStore must not know OpenBLAS provider state.");
        assertTrue(!source.contains("import backend.metal."), "RuntimeWorkspaceStore must not know Metal runtime state.");
        assertTrue(!source.contains("import backend.cuda."), "RuntimeWorkspaceStore must not know CUDA runtime state.");
    }

    @Test
    void graphPackageDoesNotWriteToStdoutOrStderr() throws IOException {
        List<String> offenders = linesContainingAny(
                Path.of("src/main/java/graph"),
                List.of("System.out", "System.err")
        );
        assertTrue(offenders.isEmpty(), () -> "graph package must not write directly to stdout/stderr: " + offenders);
    }

    @Test
    void runtimeExecutionDoesNotImportConcreteAcceleratorBackends() throws IOException {
        List<String> offenders = linesContainingAny(
                Path.of("src/main/java/runtime/execution"),
                List.of("import backend.metal.", "import backend.cuda.")
        );
        assertTrue(offenders.isEmpty(), () -> "runtime.execution must consume backend-neutral accelerator contracts: " + offenders);
    }

    @Test
    void backendRootContainsOnlyFacadeFiles() throws IOException {
        Set<String> allowedRootFiles = Set.of(
                "ApproxMode.java",
                "ComputeEngine.java"
        );
        try (Stream<Path> paths = Files.list(Path.of("src/main/java/backend"))) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .map(path -> path.getFileName().toString())
                    .filter(name -> !allowedRootFiles.contains(name))
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "Root backend package must not gain concrete helpers or wrappers: " + offenders);
        }
    }

    @Test
    void backendLoweringPackageDoesNotImportConcreteBackends() throws IOException {
        List<String> offenders = linesContainingAny(
                Path.of("src/main/java/backend/lowering"),
                List.of(
                        "import backend.cpu.",
                        "import backend.metal.",
                        "import backend.cuda.",
                        "import backend.metal.",
                        "import backend.kernels."
                )
        );
        assertTrue(offenders.isEmpty(), () -> "backend.lowering must stay backend-neutral: " + offenders);
    }

    @Test
    void backendPartitionPackageDoesNotImportExecutionOrKernelDetails() throws IOException {
        List<String> offenders = linesContainingAny(
                Path.of("src/main/java/backend/partition"),
                List.of(
                        "import backend.kernels.",
                        "import backend.cpu.kernels.",
                        "import backend.cuda.kernels.",
                        "import backend.opencl.kernels.",
                        "import backend.metal.bridge.",
                        "import backend.metal.bridge.",
                        "import backend.cuda.bridge.",
                        "import backend.metal.exec.",
                        "import backend.metal.exec.",
                        "import backend.cuda.exec."
                )
        );
        assertTrue(offenders.isEmpty(), () -> "backend.partition must compose descriptors, not own execution policy: " + offenders);
    }

    @Test
    void prepareOrchestrationDoesNotOwnConcreteBackendPreparers() throws IOException {
        List<String> offenders = javaFilesUnder(Path.of("src/main/java/prepare/orchestration")).stream()
                .map(path -> Path.of(path).getFileName().toString())
                .filter(name -> name.endsWith("NodePreparer.java"))
                .sorted()
                .toList();
        assertTrue(offenders.isEmpty(), () -> "Concrete backend preparers belong under backend.<target>.prepare: " + offenders);
    }

    @Test
    void backendRegistryPackageIsRemoved() throws IOException {
        List<String> offenders = javaFilesUnder(Path.of("src/main/java/backend/registry"));
        assertTrue(offenders.isEmpty(), () -> "Backend-specific registries belong under backend.<target>.registry: " + offenders);
    }

    @Test
    void cudaAndOpenClKernelsLiveUnderBackendRoots() throws IOException {
        List<String> offenders = javaFilesUnderRoots(List.of(
                Path.of("src/main/java/backend/kernels/cuda"),
                Path.of("src/main/java/backend/kernels/opencl")
        ));
        assertTrue(offenders.isEmpty(), () -> "CUDA/OpenCL kernels belong under backend.<target>.kernels: " + offenders);
    }

    @Test
    void cpuKernelsLiveUnderCpuBackendRoot() throws IOException {
        List<String> oldRootFiles = javaFilesUnder(Path.of("src/main/java/backend/" + "kernels/cpu"));
        assertTrue(oldRootFiles.isEmpty(), () -> "CPU kernels belong under backend.cpu.kernels: " + oldRootFiles);

        List<String> oldPackageReferences = linesContainingAny(
                List.of(Path.of("src/main/java"), Path.of("src/test/java")),
                List.of("backend." + "kernels.cpu", "backend/" + "kernels/cpu")
        );
        assertTrue(oldPackageReferences.isEmpty(), () -> "Legacy CPU kernel package references remain: " + oldPackageReferences);
    }

    @Test
    void elementwiseArrayKernelsDoNotRestoreDtypePackages() throws IOException {
        List<String> legacyPackageSuffixes = List.of(
                "binary.f32",
                "binary.f64",
                "binary.bf16",
                "unary.f32",
                "unary.f64",
                "unary.bf16"
        );
        List<String> restoredDirs = List.of(Path.of("src/main/java"), Path.of("src/test/java")).stream()
                .flatMap(root -> legacyPackageSuffixes.stream()
                        .map(suffix -> root.resolve("backend/cpu/kernels/elementwise/" + suffix.replace('.', '/'))))
                .filter(Files::exists)
                .map(Path::toString)
                .sorted()
                .toList();
        assertTrue(restoredDirs.isEmpty(),
                () -> "Elementwise array kernels must not restore dtype package directories: " + restoredDirs);

        List<String> legacyPackageReferences = legacyPackageSuffixes.stream()
                .flatMap(suffix -> Stream.of(
                        "backend.cpu.kernels.elementwise." + suffix,
                        "backend/cpu/kernels/elementwise/" + suffix.replace('.', '/')))
                .toList();
        List<String> offenders = linesContainingAny(
                List.of(Path.of("src/main/java"), Path.of("src/test/java")),
                legacyPackageReferences
        );
        assertTrue(offenders.isEmpty(),
                () -> "Elementwise array kernels must not reference old dtype packages: " + offenders);
    }

    @Test
    void elementwiseBinaryAndUnaryRootsDoNotOwnStorageSpecificLoops() throws IOException {
        List<String> rootLoopFiles = javaFilesUnderRoots(List.of(
                        Path.of("src/main/java/backend/cpu/kernels/elementwise/binary"),
                        Path.of("src/main/java/backend/cpu/kernels/elementwise/unary")
                )).stream()
                .filter(path -> path.endsWith("Loops.java"))
                .sorted()
                .toList();
        assertTrue(rootLoopFiles.isEmpty(),
                () -> "Binary/unary root packages must own kernels and dispatch only; storage loop packages must not return: "
                        + rootLoopFiles);

        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/binary/memorysegmentloops")),
                "Binary MemorySegment execution must not live behind a separate route package.");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/unary/memorysegmentloops")),
                "Unary MemorySegment execution must not live behind a separate route package.");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/binary/arrayloops")),
                "Binary Java-array hot loops must live in their operation kernels, not a separate route package.");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/unary/arrayloops")),
                "Unary Java-array hot loops must live in their operation kernels, not a separate route package.");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/binary/array")),
                "Binary Java-array execution must not use the old binary.array package.");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/unary/array")),
                "Unary Java-array execution must not use the old unary.array package.");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/binary/segment")),
                "Binary MemorySegment execution must not use the old binary.segment package.");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/unary/segment")),
                "Unary MemorySegment execution must not use the old unary.segment package.");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/binary/AddArrayLoops.java")),
                "ADD array behavior must not return as a one-off route class.");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/binary/AddSegmentLoops.java")),
                "ADD segment behavior must not return as a one-off route class.");
    }

    @Test
    void cpuNativeRuntimeExecutorStackDoesNotReturn() throws IOException {
        List<String> deletedNativeStack = List.of(
                "NativeCpuPlanResolver",
                "PreparedNativeCpuPlan",
                "PreparedNativeCpuRoute",
                "PreparedNativeCpuInputPolicy",
                "NativeCpuElementwiseExecutor",
                "NativeCpuReductionExecutor",
                "NativeCpuCompareExecutor",
                "NativeCpuBoolMaskExecutor",
                "NativeCpuCastExecutor",
                "NativeCpuContiguousExecutor",
                "NativeCpuViewExecutor",
                "NativeCpuKernelFacts",
                "NativeCpuKernelFact",
                "NativeCpuCoverageMatrix",
                "NativeCpuParityMatrix",
                "NativeCpuCoverageEntry",
                "NativeCpuParityEntry"
        );
        List<String> restoredFiles = deletedNativeStack.stream()
                .map(className -> Path.of("src/main/java/backend/cpu/nativecpu/" + className + ".java"))
                .filter(Files::exists)
                .map(Path::toString)
                .sorted()
                .toList();
        assertTrue(restoredFiles.isEmpty(),
                () -> "Removed native CPU planner/executor/facts classes must not be restored: " + restoredFiles);

        List<String> sourceReferences = sourceLinesContaining(
                List.of(Path.of("src/main/java")),
                deletedNativeStack
        );
        assertTrue(sourceReferences.isEmpty(),
                () -> "Runtime code must not depend on the removed native CPU executor stack: " + sourceReferences);
    }

    @Test
    void genericBackendSelectionDoesNotLiveUnderAcceleratorPackage() throws IOException {
        List<String> offenders = javaFilesUnder(Path.of("src/main/java/backend/accelerator/select")).stream()
                .map(path -> Path.of(path).getFileName().toString())
                .filter(name -> name.startsWith("BackendSelection") || name.equals("DefaultBackendSelectionPolicy.java"))
                .sorted()
                .toList();
        assertTrue(offenders.isEmpty(), () -> "Generic backend selection belongs under backend.select: " + offenders);
    }

    @Test
    void metalAndCudaPackagesExposeExpectedCoreShape() {
        List<Path> requiredDirs = List.of(
                Path.of("src/main/java/backend/metal/bridge"),
                Path.of("src/main/java/backend/metal/exec"),
                Path.of("src/main/java/backend/metal/lowering"),
                Path.of("src/main/java/backend/metal/prepare"),
                Path.of("src/main/java/backend/cuda/bridge"),
                Path.of("src/main/java/backend/cuda/exec"),
                Path.of("src/main/java/backend/cuda/lowering"),
                Path.of("src/main/java/backend/cuda/prepare"),
                Path.of("src/main/java/backend/cuda/registry"),
                Path.of("src/main/java/backend/cuda/kernels")
        );
        List<String> missing = requiredDirs.stream()
                .filter(path -> !Files.isDirectory(path))
                .map(Path::toString)
                .sorted()
                .toList();
        assertTrue(missing.isEmpty(), () -> "Metal/CUDA backend package shape is incomplete: " + missing);
    }

    @Test
    void metalAndCudaBridgePackagesDoNotImportPrepareInternals() throws IOException {
        List<String> offenders = linesContainingAny(
                List.of(
                        Path.of("src/main/java/backend/metal/bridge"),
                        Path.of("src/main/java/backend/cuda/bridge")
                ),
                List.of("import prepare.")
        );
        assertTrue(offenders.isEmpty(), () -> "Bridge packages must not depend on generic prepare internals: " + offenders);
    }

    @Test
    void appleNamedBackendTreesAreRemoved() {
        List<String> offenders = javaFilesUnderRoots(List.of(
                        Path.of("src/main/java/backend/apple"),
                        Path.of("src/test/java/backend/apple")
                ));
        assertTrue(offenders.isEmpty(), () -> "Apple-named backend trees must not exist after the Metal rename: " + offenders);
    }

    @Test
    void backendSourceDoesNotUseRemovedAppleMigrationNames() throws IOException {
        List<String> offenders = linesContainingAny(
                List.of(Path.of("src/main/java"), Path.of("src/test/java")),
                List.of(
                        "Apple" + "Gpu",
                        "Apple" + "Partition",
                        "Apple" + "Mps",
                        "backend." + "apple",
                        "APPLE" + "_GRAPH_PARTITION",
                        "APPLE" + "_FUSED_ELEMENTWISE_GRAPH",
                        "apple" + "LoweredPartitionForAnchor",
                        "apple" + "PartitionForAnchor",
                        "apple" + "PartitionsByAnchor",
                        "synaptik." + "apple.mps.lib",
                        "SYNAPTIK_" + "APPLE_MPS_LIB"
                )
        );
        assertTrue(offenders.isEmpty(), () -> "Removed Apple migration names remain in Java backend/test source: " + offenders);
    }

    @Test
    void graphAutotuneCandidatePackageDoesNotImportRuntimeOrBackendConfig() throws IOException {
        List<String> offenders = linesContainingAny(
                Path.of("src/main/java/tuning/candidate/graph"),
                List.of(
                        "import config.runtime.",
                        "import config.backend."
                )
        );
        assertTrue(offenders.isEmpty(), () -> "Graph autotune candidates must not import runtime/backend config: " + offenders);
    }

    @Test
    void productionAutotuneDoesNotReferenceStageOrderCandidateSpace() throws IOException {
        Path tuningCli = Path.of("src/main/java/synaptik/app/TuningCli.java");
        String source = Files.readString(tuningCli);
        assertTrue(!Files.exists(Path.of("src/main/java/tuning/candidate/ProfileMutators.java")), "Old mixed ProfileMutators surface must not remain.");
        assertTrue(!source.contains("ProfileMutators"), "Production CLI autotune must not use explicit profile mutators.");
        assertTrue(!source.contains("stageCandidateSpace"), "Production CLI autotune must not use stage candidate space.");
        assertTrue(!source.contains("constrainedStageOrderSpace"), "Production CLI autotune must not use stage-order mutators.");
        assertTrue(!source.contains("stageOrders("), "Production CLI autotune must not use stage-order mutators.");
        assertTrue(!source.contains("benchmark-stage-space"), "Removed stage-space CLI command must not remain.");
    }

    @Test
    void benchmarkCliDoesNotPersistMeasurementResults() throws IOException {
        Path tuningCli = Path.of("src/main/java/synaptik/app/TuningCli.java");
        String source = Files.readString(tuningCli);
        assertTrue(!source.contains("JsonFileBenchmarkReportStore"), "Benchmark CLI must not persist benchmark reports.");
        assertTrue(!source.contains("JsonFileTuningHistoryStore"), "Benchmark CLI must not append tuning history.");
        assertTrue(!source.contains("saveBenchmark("), "Benchmark CLI must not save benchmark reports.");
        assertTrue(!source.contains(".append("), "Benchmark CLI must not append benchmark output to persistent history.");
    }

    private static void assertPlanningPartitionBackendPackageAbsent(String packageName, String message) throws IOException {
        List<Path> roots = List.of(Path.of("src/main/java"), Path.of("src/test/java"));
        try (Stream<Path> paths = roots.stream()
                .filter(Files::exists)
                .flatMap(root -> {
                    try {
                        return Files.walk(root);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> path.toString().contains("planning/partition/" + packageName))
                    .map(Path::toString)
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> message + ": " + offenders);
        }
    }

    private static List<String> javaFilesUnder(Path root) throws IOException {
        if (!Files.exists(root)) {
            return List.of();
        }
        try (Stream<Path> paths = Files.walk(root)) {
            return paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .map(Path::toString)
                    .sorted()
                    .toList();
        }
    }

    private static List<String> javaFilesUnderRoots(List<Path> roots) {
        return roots.stream()
                .filter(Files::exists)
                .flatMap(root -> {
                    try {
                        return Files.walk(root);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })
                .filter(Files::isRegularFile)
                .filter(path -> path.toString().endsWith(".java"))
                .map(Path::toString)
                .sorted()
                .toList();
    }

    private static List<String> linesContainingAny(Path root, List<String> patterns) throws IOException {
        if (!Files.exists(root)) {
            return List.of();
        }
        try (Stream<Path> paths = Files.walk(root)) {
            return paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> !path.getFileName().toString().equals("SourceTreeHygieneTest.java"))
                    .flatMap(path -> {
                        try {
                            return Files.readAllLines(path).stream()
                                    .filter(line -> patterns.stream().anyMatch(line::contains))
                                    .map(line -> path + ": " + line.trim());
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sorted()
                    .toList();
        }
    }

    private static List<String> linesContainingAny(List<Path> roots, List<String> patterns) throws IOException {
        try (Stream<Path> paths = roots.stream()
                .filter(Files::exists)
                .flatMap(root -> {
                    try {
                        return Files.walk(root);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })) {
            return paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> !path.getFileName().toString().equals("SourceTreeHygieneTest.java"))
                    .flatMap(path -> {
                        try {
                            return Files.readAllLines(path).stream()
                                    .filter(line -> patterns.stream().anyMatch(line::contains))
                                    .map(line -> path + ": " + line.trim());
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sorted()
                    .toList();
        }
    }

    private static List<String> sourceLinesContaining(List<Path> roots, List<String> patterns) throws IOException {
        try (Stream<Path> paths = roots.stream()
                .filter(Files::exists)
                .flatMap(root -> {
                    try {
                        return Files.isRegularFile(root) ? Stream.of(root) : Files.walk(root);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })) {
            return paths
                    .filter(Files::isRegularFile)
                    .filter(path -> {
                        String value = path.toString();
                        return value.endsWith(".java") || value.endsWith(".md");
                    })
                    .filter(path -> !path.getFileName().toString().equals("SourceTreeHygieneTest.java"))
                    .flatMap(path -> {
                        try {
                            return Files.readAllLines(path).stream()
                                    .filter(line -> patterns.stream().anyMatch(line::contains))
                                    .map(line -> path + ": " + line.trim());
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sorted()
                    .toList();
        }
    }

    private static String gitOutput(String... args) throws IOException {
        List<String> command = new java.util.ArrayList<>();
        command.add("git");
        command.addAll(List.of(args));
        Process process = new ProcessBuilder(command)
                .redirectErrorStream(true)
                .start();
        try {
            String output = new String(process.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
            int exitCode = process.waitFor();
            assertTrue(exitCode == 0, () -> String.join(" ", command) + " failed: " + output);
            return output.trim();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while running " + String.join(" ", command), e);
        }
    }
}
