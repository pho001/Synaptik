import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class SourceTreeHygieneTest {

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
    void backendPrepareDoesNotRebuildOptimizerArtifacts() throws IOException {
        Path root = Path.of("src/main/java/backend/prepare");
        try (Stream<Path> paths = Files.walk(root)) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> !path.getFileName().toString().equals("SourceTreeHygieneTest.java"))
                    .flatMap(path -> {
                        try {
                            return Files.readAllLines(path).stream()
                                    .filter(line -> line.contains("graph.optimizer.memory.MemoryPlanner")
                                            || line.contains("graph.optimizer.region.DefaultRegionOptimizer")
                                            || line.contains("graph.optimizer.region.RegionOptimizationContext"))
                                    .map(line -> path + ": " + line.trim());
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "backend.prepare rebuilds optimizer artifacts: " + offenders);
        }
    }

    @Test
    void runtimeMemoryBinderDoesNotUseGlobalMigrationGuards() throws IOException {
        Path binder = Path.of("src/main/java/graph/execution/RuntimeMemoryBinder.java");
        String source = Files.readString(binder);
        assertTrue(!source.contains("containsPhase12BinderExcludedFamily"), "RuntimeMemoryBinder must not disable binding for a whole graph.");
        assertTrue(!source.contains("skipRuntimeBinding"), "RuntimeMemoryBinder skip policy must be explicit and named.");
        assertTrue(!source.contains("Phase 12"), "RuntimeMemoryBinder must not keep migration-era guard comments.");
        assertTrue(!source.contains("MAX_POOL2D"), "RuntimeMemoryBinder must consume memory-plan binding policy, not hardcode workspace-sensitive families.");
    }

    @Test
    void graphPartitionPackageDoesNotOwnConcreteCpuBackendCode() throws IOException {
        assertGraphPartitionBackendPackageAbsent("cpu", "CPU backend partition code belongs under backend.cpu.partition");
    }

    @Test
    void graphPartitionPackageDoesNotOwnConcreteCudaBackendCode() throws IOException {
        assertGraphPartitionBackendPackageAbsent("cuda", "CUDA backend partition code belongs under backend.cuda.lowering");
    }

    @Test
    void graphPartitionPackageDoesNotOwnConcreteAppleBackendCode() throws IOException {
        assertGraphPartitionBackendPackageAbsent("apple", "Metal backend partition code belongs under backend.metal.lowering");
    }

    @Test
    void graphPartitionPackageDoesNotOwnAcceleratorDagModelCode() throws IOException {
        assertGraphPartitionBackendPackageAbsent("model", "Accelerator DAG model belongs under backend.accelerator.dag");
    }

    @Test
    void graphOptimizerRulesPackageIsRemoved() throws IOException {
        List<String> offenders = javaFilesUnder(Path.of("src/main/java/graph/optimizer/rules"));
        assertTrue(offenders.isEmpty(), () -> "optimizer stage adapters belong in their domain packages: " + offenders);
    }

    @Test
    void graphPartitionPackageDoesNotImportConcreteBackendImplementations() throws IOException {
        Path root = Path.of("src/main/java/graph/optimizer/partition");
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
            assertTrue(offenders.isEmpty(), () -> "graph.optimizer.partition imports concrete backend implementations: " + offenders);
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
        assertTrue(offenders.isEmpty(), () -> "Phase 19 GPU region lowering must not import CPU fused internals: " + offenders);
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
        assertTrue(offenders.isEmpty(), () -> "CPU fused codegen belongs under backend.cpu.fused.codegen: " + offenders);
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
        assertTrue(offenders.isEmpty(), () -> "CPU fused optimization policy belongs under backend.cpu.fused.optimize: " + offenders);
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
    void defaultRegionOptimizerDoesNotOwnCpuMixedUnitPolicy() throws IOException {
        Path optimizer = Path.of("src/main/java/graph/optimizer/region/DefaultRegionOptimizer.java");
        String source = Files.readString(optimizer);
        assertTrue(!source.contains("buildMixedCpuUnits"), "CPU mixed-unit policy belongs in CpuRegionOptimizationPolicy.");
        assertTrue(!source.contains("fused-subchain"), "CPU fused-subchain policy belongs outside DefaultRegionOptimizer.");
        assertTrue(!source.contains("isSubchainFusable"), "CPU subchain fusion checks belong outside DefaultRegionOptimizer.");
        assertTrue(!source.contains("consumesUnitOutput"), "CPU subchain traversal belongs outside DefaultRegionOptimizer.");
    }

    @Test
    void regionOptimizerDoesNotOwnBackendLoweringPolicy() throws IOException {
        Path regionRoot = Path.of("src/main/java/graph/optimizer/region");
        assertTrue(!Files.exists(regionRoot.resolve("RegionOptimizationPolicy.java")),
                "Region optimizer should not keep an extra policy abstraction layer.");
        assertTrue(!Files.exists(regionRoot.resolve("GenericGpuRegionOptimizationPolicy.java")),
                "Generic GPU policy should be expressed as backend-neutral structural region planning.");
        List<String> offenders = linesContainingAny(
                regionRoot,
                List.of(
                        "LoweringFamily.",
                        "BackendCapabilities",
                        "ComputeBackend.GPU_METAL",
                        "ComputeBackend.GPU_CUDA",
                        "METAL_GRAPH",
                        "CUDA_GRAPH"
                )
        );
        assertTrue(offenders.isEmpty(), () -> "Region optimizer must stay backend-lowering neutral: " + offenders);
    }

    @Test
    void prepareDoesNotGloballySkipLoweringForLegacyFusedGraphs() throws IOException {
        Path builder = Path.of("src/main/java/backend/prepare/PreparedExecutionBuilder.java");
        String source = Files.readString(builder);
        assertTrue(!source.contains("containsLegacyFusedGraph"), "prepare must not globally suppress lowered-region publication for legacy fused nodes.");
        assertTrue(!source.contains("OpType.FUSED"), "legacy fused nodes must not be a prepare-layer global lowering gate.");
    }

    @Test
    void regionOptimizationRuleDoesNotKeepLegacyGraphMutationFallback() throws IOException {
        Path rule = Path.of("src/main/java/graph/optimizer/region/RegionOptimizationRule.java");
        String source = Files.readString(rule);
        assertTrue(!source.contains("applyLegacyGraphFusion"), "FUSE must consume partition state instead of running legacy graph-mutating fusion.");
        assertTrue(!source.contains("TensorInternalAccess"), "FUSE must not mutate tensor operation/input structure directly.");
        assertTrue(!source.contains("FusedOperationFactory"), "FUSED descriptors are backend CPU plan artifacts, not graph optimizer output.");
    }

    @Test
    void cpuNodePreparerDoesNotInlineLoweredFusedDescriptorSynthesis() throws IOException {
        Path preparer = Path.of("src/main/java/backend/cpu/prepare/CpuNodePreparer.java");
        String source = Files.readString(preparer);
        assertTrue(!source.contains("synthesizeFusedPreparation"), "Lowered fused descriptor construction belongs under backend.cpu.fused.plan.");
        assertTrue(!source.contains("FusedOperationFactory"), "CpuNodePreparer should consume backend CPU fused plan preparation, not build descriptors inline.");
    }

    @Test
    void preparedExecutionBuilderDoesNotOwnCompileArtifactRecovery() throws IOException {
        Path builder = Path.of("src/main/java/backend/prepare/PreparedExecutionBuilder.java");
        String source = Files.readString(builder);
        assertTrue(!source.contains("MemoryPlanner"), "prepare must consume compile artifacts instead of rebuilding memory plans.");
        assertTrue(!source.contains("DefaultRegionOptimizer"), "prepare must consume compile artifacts instead of rebuilding optimized regions.");
        assertTrue(source.contains("loweringInput"), "prepare must rely on CompileArtifacts lowering input contract.");
    }

    @Test
    void preparedExecutionDoesNotOwnBackendTraceAttributeDetails() throws IOException {
        String preparedExecution = Files.readString(Path.of("src/main/java/graph/execution/PreparedExecution.java"));
        String stepTracer = Files.readString(Path.of("src/main/java/graph/execution/StepExecutionTracer.java"));
        assertTrue(stepTracer.contains("BackendRunTraceContributors"),
                "StepExecutionTracer should delegate backend-specific trace attributes to contributors.");
        assertTrue(!preparedExecution.contains("BackendRunTraceContributors"),
                "PreparedExecution should delegate step trace assembly to StepExecutionTracer.");
        assertTrue(!preparedExecution.contains("\"metalBridgeAvailable\""), "Metal trace attributes belong in backend-owned trace contribution.");
        assertTrue(!preparedExecution.contains("\"cudaBridgeAvailable\""), "CUDA trace attributes belong in backend-owned trace contribution.");
        assertTrue(!preparedExecution.contains("\"nativeCpuRegionId\""), "Native CPU trace attributes belong in the CPU trace contributor.");
        assertTrue(!preparedExecution.contains("\"acceleratorBufferMode\""), "Accelerator buffer trace attributes belong in the accelerator trace contributor.");
    }

    @Test
    void graphExecutionDoesNotImportConcreteAcceleratorBackends() throws IOException {
        List<String> offenders = linesContainingAny(
                Path.of("src/main/java/graph/execution"),
                List.of("import backend.metal.", "import backend.cuda.")
        );
        assertTrue(offenders.isEmpty(), () -> "graph.execution must consume backend-neutral accelerator contracts: " + offenders);
    }

    @Test
    void backendRootContainsOnlyFacadeFiles() throws IOException {
        Set<String> allowedRootFiles = Set.of(
                "ApproxMode.java",
                "ComputeBackend.java",
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
    void backendPrepareDoesNotOwnConcreteBackendPreparers() throws IOException {
        List<String> offenders = javaFilesUnder(Path.of("src/main/java/backend/prepare")).stream()
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
                List.of("import backend.prepare.")
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
                        "Apple" + "Region",
                        "Apple" + "Mps",
                        "backend." + "apple",
                        "APPLE" + "_GRAPH_REGION",
                        "APPLE" + "_FUSED_ELEMENTWISE_GRAPH",
                        "apple" + "LoweredRegionForAnchor",
                        "apple" + "RegionForAnchor",
                        "apple" + "RegionsByAnchor",
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

    private static void assertGraphPartitionBackendPackageAbsent(String packageName, String message) throws IOException {
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
                    .filter(path -> path.toString().contains("graph/optimizer/partition/" + packageName))
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
