import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelRegistry;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.nativecpu.CpuNativeStorageSupport;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.io.IOException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CpuKernelFamilyArchitectureTest {
    @Test
    void registryResolvesKernelsFromExpectedFamilyPackages() {
        assertPackage(Operation.OpType.ADD, "backend.cpu.kernels.elementwise.binary");
        assertPackage(Operation.OpType.WHERE, "backend.cpu.kernels.elementwise.where");
        assertPackage(Operation.OpType.NOOP, "backend.cpu.kernels.layout");
        assertPackage(Operation.OpType.GATHER, "backend.cpu.kernels.index");
        assertPackage(Operation.OpType.SUM, "backend.cpu.kernels.reduction");
        assertPackage(Operation.OpType.MATMUL, "backend.cpu.kernels.linalg");
        assertPackage(Operation.OpType.CONV2D, "backend.cpu.kernels.nn");
        assertPackage(Operation.OpType.LAYER_NORM, "backend.cpu.kernels.nn");
        assertPackage(Operation.OpType.RMS_NORM, "backend.cpu.kernels.nn");
        assertPackage(Operation.OpType.UNFOLD_AXIS, "backend.cpu.kernels.layout");
        assertPackage(Operation.OpType.UNFOLD2D, "backend.cpu.kernels.layout");
        assertPackage(Operation.OpType.FOLD2D, "backend.cpu.kernels.layout");
        assertPackage(Operation.OpType.FUSED, "backend.cpu.kernels.fused");
    }

    @Test
    void registryRejectsNonFinalLegacyBackwardDescriptors() {
        for (Operation.OpType opType : nonFinalLegacyBackwardOpTypes()) {
            IllegalStateException error = assertThrows(IllegalStateException.class, () -> CpuKernelRegistry.resolve(opType));
            assertTrue(error.getMessage().contains("legacy backward op type " + opType));
        }
    }

    @Test
    void cpuRootPackageOnlyContainsFinalKernelContractTypes() throws IOException {
        Path root = Path.of("src/main/java/backend/cpu/kernels");
        Set<String> allowed = Set.of(
                "CpuKernel.java",
                "CpuKernelCall.java",
                "CpuKernelRegistry.java",
                "CpuKernelResult.java",
                "CpuStorageAwareKernel.java",
                "TypedCpuKernel.java"
        );

        try (Stream<Path> files = Files.list(root)) {
            List<String> unexpected = files
                    .filter(Files::isRegularFile)
                    .map(path -> path.getFileName().toString())
                    .filter(name -> name.endsWith(".java"))
                    .filter(name -> !allowed.contains(name))
                    .sorted()
                    .toList();
            assertTrue(unexpected.isEmpty(), () ->
                    "backend.cpu.kernels root may contain only final kernel boundary types: " + unexpected);
        }
        assertTrue(Files.exists(Path.of("src/main/java/tensor/dtype/TensorDTypeOps.java")),
                "Generic dtype helpers must live under tensor.dtype, not backend.cpu.kernels.");
    }

    @Test
    void cpuKernelPublicApiDoesNotExposeDTypeSpecificForwardMethods() {
        Set<String> publicDeclaredMethods = new TreeSet<>();
        for (Method method : CpuKernel.class.getDeclaredMethods()) {
            if (Modifier.isPublic(method.getModifiers())) {
                publicDeclaredMethods.add(method.getName());
            }
        }

        assertEquals(Set.of("costClass", "execute"), publicDeclaredMethods,
                "CpuKernel public API must remain the single execute(call) contract plus cost metadata.");
        List<String> dtypeSpecificMethods = publicDeclaredMethods.stream()
                .filter(name -> name.startsWith("forward"))
                .sorted()
                .toList();
        assertTrue(dtypeSpecificMethods.isEmpty(),
                () -> "CpuKernel must not expose dtype-specific forward* methods: " + dtypeSpecificMethods);
    }

    @Test
    void cpuKernelCallCarriesPreparedPlanAndStorageViews() {
        assertEquals(8, CpuKernelCall.class.getRecordComponents().length,
                "CpuKernelCall must be the complete executor-to-kernel boundary.");
        Set<String> componentNames = Stream.of(CpuKernelCall.class.getRecordComponents())
                .map(java.lang.reflect.RecordComponent::getName)
                .collect(java.util.stream.Collectors.toCollection(TreeSet::new));
        assertEquals(Set.of(
                        "context",
                        "inputTensors",
                        "inputs",
                        "operation",
                        "output",
                        "outputTensor",
                        "plan",
                        "workspace"
                ),
                componentNames,
                "CpuKernelCall must carry tensors, storage views, immutable plan, context, and workspace.");
    }

    @Test
    void cpuBackendUsesExecutionLayerForKernelInvocation() throws IOException {
        String backend = Files.readString(Path.of("src/main/java/backend/cpu/CpuBackend.java"));
        assertTrue(backend.contains("CpuKernelExecutor"),
                "CpuBackend must route kernel invocation through backend.cpu.execution.CpuKernelExecutor.");
        assertFalse(backend.contains("new CpuKernelCall"),
                "CpuBackend must not assemble CpuKernelCall directly.");
        assertFalse(backend.contains("new CpuKernelContext"),
                "CpuBackend must not assemble CpuKernelContext directly.");
        assertFalse(backend.contains("CpuStridedElementWise.forward"),
                "Strided execution routing belongs in CpuKernelExecutor, not CpuBackend.");
    }

    @Test
    void movedInfrastructureTypesDoNotRemainUnderKernelPackage() throws IOException {
        Map<String, String> movedTypes = Map.ofEntries(
                Map.entry("CpuDTypeOps", "tensor.dtype"),
                Map.entry("CpuKernelContext", "backend.cpu.execution"),
                Map.entry("CpuNodeWorkspace", "backend.cpu.execution"),
                Map.entry("CpuThreadPool", "backend.cpu.execution"),
                Map.entry("CpuNativeStorageSupport", "backend.cpu.nativecpu"),
                Map.entry("CpuNativeTraceSupport", "backend.cpu.nativecpu"),
                Map.entry("CpuAccumulateDType", "backend.cpu.plan"),
                Map.entry("CpuComputeDType", "backend.cpu.plan"),
                Map.entry("CpuExecutionBackend", "backend.cpu.plan"),
                Map.entry("CpuExecutionMode", "backend.cpu.plan"),
                Map.entry("CpuKernelCostClass", "backend.cpu.plan"),
                Map.entry("CpuNodeExecutionPlan", "backend.cpu.plan"),
                Map.entry("ResolvedCpuComputeContract", "backend.cpu.plan")
        );

        Path root = Path.of("src/main/java/backend/cpu/kernels");
        try (Stream<Path> paths = Files.walk(root)) {
            List<String> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .flatMap(path -> movedTypes.keySet().stream()
                            .filter(type -> path.getFileName().toString().equals(type + ".java")
                                    || contains(path, "import backend.cpu.kernels." + type + ";")
                                    || contains(path, "backend.cpu.kernels." + type))
                            .map(type -> root.relativize(path).toString().replace('\\', '/')
                                    + " references legacy backend.cpu.kernels." + type
                                    + " (new owner: " + movedTypes.get(type) + ")"))
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(),
                    () -> "Moved CPU infrastructure types must not remain under backend.cpu.kernels: " + offenders);
        }
    }

    @Test
    void generatedFusedBytecodeDoesNotReferenceMovedDTypeHelper() throws IOException {
        Path fusedAsmRoot = Path.of("src/main/java/backend/cpu/fused/asm");
        try (Stream<Path> paths = Files.walk(fusedAsmRoot)) {
            List<Path> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> contains(path, "backend/cpu/kernels/TensorDTypeOps")
                            || contains(path, "backend.cpu.kernels.TensorDTypeOps"))
                    .toList();
            assertTrue(offenders.isEmpty(),
                    () -> "Generated fused bytecode must reference tensor/dtype/TensorDTypeOps: " + offenders);
        }
    }

    @Test
    void cpuKernelRegistryLivesAtKernelBoundaryWithoutOldResolverPackage() throws IOException {
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/CpuKernelRegistry.java")),
                "CpuKernelRegistry must live at backend.cpu.kernels boundary.");
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/registry/CpuKernelResolver.java")),
                "Old backend.cpu.registry.CpuKernelResolver must not remain as a compatibility facade.");
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/registry")),
                "backend.cpu.registry package must not remain after registry cutover.");
    }

    @Test
    void cpuExecutionRuntimeHelpersLiveOutsideKernelPackage() throws IOException {
        assertCpuExecutionRootType("CpuKernelContext.java");
        assertCpuExecutionRootType("CpuNodeWorkspace.java");
        assertCpuExecutionRootType("CpuThreadPool.java");
    }

    @Test
    void cpuRootPackageDoesNotContainConcreteKernelEntrypoints() throws IOException {
        Path root = Path.of("src/main/java/backend/cpu/kernels");
        try (Stream<Path> files = Files.list(root)) {
            List<String> offenders = files
                    .filter(Files::isRegularFile)
                    .map(path -> path.getFileName().toString())
                    .filter(name -> name.startsWith("Cpu") && name.endsWith("Kernel.java"))
                    .filter(name -> !Set.of("CpuKernel.java", "CpuStorageAwareKernel.java").contains(name))
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "Concrete Cpu*Kernel entrypoints must live in family packages: " + offenders);
        }
    }

    @Test
    void gradKernelsAreOwnedBySourceFamilies() {
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/grad")),
                "Grad kernels must not live in a root dumping package.");
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/grad/CpuMinGradKernel.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/reduction/grad/CpuReduceMinGradKernel.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/reduction/CpuCrossEntropyLossIndicesGradKernel.java")));
        assertFalse(hasConcreteConvBackwardKernel());
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/nn/CpuMaxPool2dBackwardInputKernel.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/nn/CpuAvgPool2dBackwardInputKernel.java")));
    }

    @Test
    void nonFinalLegacyBackwardKernelFilesDoNotRemainUnderCpuKernels() {
        for (String relativePath : List.of(
                "elementwise/grad/CpuMinGradKernel.java",
                "elementwise/grad/CpuMaxGradKernel.java",
                "reduction/grad/CpuReduceMinGradKernel.java",
                "reduction/grad/CpuReduceMaxGradKernel.java",
                "reduction/CpuSoftmaxGradKernel.java",
                "reduction/CpuLogSoftmaxGradKernel.java",
                "reduction/CpuCrossEntropyLossIndicesGradKernel.java",
                "index/CpuGatherGradKernel.java",
                "index/CpuGatherAxisGradKernel.java",
                "index/CpuGatherNdGradKernel.java",
                "index/CpuTakeAlongAxisGradKernel.java",
                "layout/CpuSliceGradKernel.java",
                "linalg/CpuScaledDotProductAttentionBackwardKernel.java"
        )) {
            assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels", relativePath)),
                    relativePath + " must not remain as direct CPU legacy backward kernel support.");
        }
    }

    @Test
    void cpuPlanningTypesLiveOutsideKernelPackage() throws IOException {
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/prepare/CpuExecutionPlanner.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/prepare/CpuPlanAssembler.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/prepare/CpuOperationPlanResolver.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/prepare/CpuPlanningPolicy.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/prepare/CpuTypeContractResolver.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/prepare/CpuComputeContractResolver.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/plan/PreparedTypeContract.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/plan/ResolvedCpuOperationPlans.java")));
        for (String fileName : List.of(
                "CpuAccumulateDType.java",
                "CpuComputeDType.java",
                "CpuExecutionBackend.java",
                "CpuExecutionMode.java",
                "CpuKernelCostClass.java",
                "CpuNodeExecutionPlan.java",
                "ResolvedCpuComputeContract.java"
        )) {
            assertPlanRootType(fileName);
        }
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/prepare/elementwise/ElementwiseDispatchPlanner.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/plan/elementwise/ResolvedDispatchHints.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/plan/layout/ResolvedBroadcastPlan.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/plan/layout/ResolvedWhereBroadcastPlan.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/prepare/layout/BroadcastPlanResolver.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/prepare/layout/PreparedInputPlanner.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/prepare/layout/PreparedInputPolicy.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/plan/layout/PreparedInputsResult.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/prepare/elementwise/StridedPathEligibility.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/plan/layout/StridedLayoutDecision.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/prepare/reduction/ReductionPlanner.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/plan/reduction/ResolvedReductionHints.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/prepare/linalg/matmul/MatMulPlanner.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/plan/linalg/matmul/MatMulExecutionRoute.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/plan/linalg/matmul/ResolvedMatMulHints.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/prepare/linalg/attention/ScaledDotProductAttentionPlanner.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/plan/linalg/attention/ResolvedAttentionHints.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/plan/linalg/attention/ResolvedScaledDotProductAttentionPlan.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/prepare/fused/FusedDispatchPlanner.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/plan/fused/PreparedFusedDispatch.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/CpuStridedElementWise.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedBooleanLoops.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedNumericInputs.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedNumericLoops.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedWhereLoops.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedScalarLoops.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedRank2Loops.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedElementWiseSemantics.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedVectorSupport.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/ElementwiseLayoutPlan.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/ElementwiseOffsetCursor.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/unary/support/CpuPowSupport.java")));

        try (Stream<Path> paths = Files.walk(Path.of("src/main/java/backend/cpu/kernels"))) {
            List<Path> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> path.toString().contains("/plan/")
                            || contains(path, "package backend.cpu.kernels.plan")
                            || contains(path, "import backend.cpu.kernels.plan.")
                            || contains(path, "import backend.cpu.kernels.elementwise.plan.")
                            || contains(path, "import backend.cpu.kernels.layout.plan.")
                            || contains(path, "import backend.cpu.kernels.reduction.plan.")
                            || contains(path, "import backend.cpu.kernels.linalg.matmul.plan.")
                            || contains(path, "import backend.cpu.kernels.linalg.attention.plan.")
                            || contains(path, "import backend.cpu.kernels.nn.conv2d.plan.")
                            || contains(path, "import backend.cpu.kernels.fused.plan."))
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "CPU planning packages must not live under backend.cpu.kernels: " + offenders);
        }
    }

    @Test
    void nativeCpuPolicyAndTraceHelpersLiveOutsideKernelPackage() throws IOException {
        assertNativeCpuRootType("CpuNativeStorageSupport.java");
        assertNativeCpuRootType("CpuNativeTraceSupport.java");
    }

    @Test
    void waveZeroCriticalOpOwnersHaveDocumentedEntrypoints() {
        List<String> requiredPaths = List.of(
                "src/main/java/backend/cpu/kernels/elementwise/binary/CpuAddKernel.java",
                "src/main/java/backend/cpu/kernels/elementwise/ElementwiseRangeLoop.java",
                "src/main/java/backend/cpu/kernels/elementwise/where/CpuWhereKernel.java",
                "src/main/java/backend/cpu/kernels/reduction/CpuSumKernel.java",
                "src/main/java/backend/cpu/kernels/reduction/StorageAwareSumLikeReductionKernel.java",
                "src/main/java/backend/cpu/kernels/reduction/SumLoops.java",
                "src/main/java/backend/cpu/kernels/layout/CpuCastKernel.java",
                "src/main/java/backend/cpu/kernels/layout/CpuContiguousKernel.java",
                "src/main/java/backend/cpu/kernels/layout/CpuAliasLayoutKernel.java",
                "src/main/java/backend/cpu/kernels/layout/CpuLayoutNativeViewSupport.java",
                "src/main/java/backend/cpu/kernels/layout/CpuLayoutOutputStorageDeferredKernel.java",
                "src/main/java/backend/cpu/kernels/linalg/CpuMatMulKernel.java",
                "src/main/java/backend/cpu/prepare/linalg/matmul/MatMulPlanner.java",
                "src/main/java/backend/cpu/provider/linalg/matmul/MatMulProviderExecutableFactory.java",
                "src/main/java/backend/cpu/kernels/linalg/matmul/f32/F32JavaMatMulExecutable.java",
                "src/main/java/backend/cpu/kernels/linalg/matmul/f32/F32BlasMatMulExecutable.java",
                "src/main/java/backend/cpu/kernels/linalg/matmul/f32/F32NativeBlasMatMulExecutable.java",
                "src/main/java/backend/cpu/kernels/elementwise/ElementwiseNativeSupport.java",
                "src/main/java/backend/cpu/kernels/elementwise/unary/StorageAwareScalarUnaryElementwiseKernel.java",
                "src/main/java/backend/cpu/kernels/elementwise/compare/StorageAwareCompareElementwiseKernel.java",
                "src/main/java/backend/cpu/kernels/elementwise/logical/StorageAwareLogicalBinaryElementwiseKernel.java",
                "src/main/java/backend/cpu/kernels/elementwise/logical/StorageAwareLogicalUnaryElementwiseKernel.java",
                "src/main/java/backend/cpu/kernels/reduction/StorageAwareReductionKernel.java"
        );

        List<String> missing = requiredPaths.stream()
                .filter(path -> !Files.exists(Path.of(path)))
                .sorted()
                .toList();
        assertTrue(missing.isEmpty(), () -> "Wave 0 owner map paths drifted: " + missing);
    }

    @Test
    void waveZeroNativeCpuImportsFromKernelPackageRemainExplicitlyAllowlisted() throws IOException {
        Map<String, Set<String>> expected = Map.ofEntries(
                Map.entry("elementwise/binary/StorageAwareBinaryElementwiseKernel.java", Set.of("CpuNativeTraceSupport")),
                Map.entry("elementwise/compare/StorageAwareCompareElementwiseKernel.java", Set.of("CpuNativeTraceSupport")),
                Map.entry("elementwise/logical/StorageAwareLogicalBinaryElementwiseKernel.java", Set.of("CpuNativeTraceSupport")),
                Map.entry("elementwise/logical/StorageAwareLogicalUnaryElementwiseKernel.java", Set.of("CpuNativeTraceSupport")),
                Map.entry("elementwise/unary/StorageAwareScalarUnaryElementwiseKernel.java", Set.of("CpuNativeTraceSupport")),
                Map.entry("elementwise/unary/StorageAwareUnaryElementwiseKernel.java", Set.of("CpuNativeTraceSupport")),
                Map.entry("elementwise/where/CpuWhereKernel.java", Set.of("CpuNativeTraceSupport")),
                Map.entry("layout/CpuCastKernel.java", Set.of("CpuNativeTraceSupport")),
                Map.entry("layout/CpuContiguousKernel.java", Set.of("CpuNativeTraceSupport")),
                Map.entry("layout/CpuLayoutNativeViewSupport.java", Set.of("CpuNativeTraceSupport")),
                Map.entry("reduction/StorageAwareReductionKernel.java", Set.of(
                        "CpuNativeTraceSupport",
                        "layout.NativeCpuStorageFamily",
                        "layout.NativeSegmentStridedKernels",
                        "layout.NativeSegmentView",
                        "layout.TensorPhysicalView"
                ))
        );

        assertEquals(expected, nativeCpuImportsUnder(Path.of("src/main/java/backend/cpu/kernels")),
                "Native CPU dependencies from kernels must remain explicit and restricted to storage ownership helpers.");
    }

    @Test
    void waveFourLayoutRuntimeOwnershipLivesInLayoutKernels() throws IOException {
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/layout/LayoutExecutor.java")),
                "Wave 4 layout runtime ownership must not remain in the shared LayoutExecutor.");

        String aliasBase = Files.readString(Path.of("src/main/java/backend/cpu/kernels/layout/CpuAliasLayoutKernel.java"));
        assertFalse(aliasBase.contains("extends TypedCpuKernel"),
                "CpuAliasLayoutKernel must own the CpuKernelCall execute boundary directly.");
        assertTrue(aliasBase.contains("implements CpuStorageAwareKernel, CpuLayoutOutputStorageDeferredKernel"),
                "CpuAliasLayoutKernel must consume storage-aware calls while deferring output storage identity.");

        String cast = Files.readString(Path.of("src/main/java/backend/cpu/kernels/layout/CpuCastKernel.java"));
        assertFalse(cast.contains("extends TypedCpuKernel"),
                "CpuCastKernel must own the CpuKernelCall execute boundary directly.");
        assertTrue(cast.contains("implements CpuStorageAwareKernel"),
                "CpuCastKernel must own storage-aware CAST materialization.");

        String contiguous = Files.readString(Path.of("src/main/java/backend/cpu/kernels/layout/CpuContiguousKernel.java"));
        assertFalse(contiguous.contains("extends TypedCpuKernel"),
                "CpuContiguousKernel must own the CpuKernelCall execute boundary directly.");
        assertTrue(contiguous.contains("implements CpuStorageAwareKernel"),
                "CpuContiguousKernel must own storage-aware CONTIGUOUS materialization.");

        for (String aliasKernel : List.of(
                "CpuAliasViewKernel.java",
                "CpuExpandKernel.java",
                "CpuNoopKernel.java",
                "CpuPermuteKernel.java"
        )) {
            String source = Files.readString(Path.of("src/main/java/backend/cpu/kernels/layout/" + aliasKernel));
            assertTrue(source.contains("extends CpuAliasLayoutKernel"),
                    aliasKernel + " must use the deferred-output alias layout boundary.");
        }
        String reshape = Files.readString(Path.of("src/main/java/backend/cpu/kernels/layout/CpuReshapeLikeKernel.java"));
        assertFalse(reshape.contains("extends TypedCpuKernel"),
                "CpuReshapeLikeKernel must own the CpuKernelCall execute boundary directly.");
        assertTrue(reshape.contains("implements CpuStorageAwareKernel, CpuLayoutOutputStorageDeferredKernel"),
                "CpuReshapeLikeKernel must consume storage-aware calls while deferring output storage identity.");
        assertTrue(reshape.contains("CpuLayoutOutputStorageDeferredKernel"),
                "CpuReshapeLikeKernel must defer output storage binding until it chooses alias vs materialization.");
    }

    @Test
    void onlyFusedKernelMayExtendLegacyTypedCpuKernel() throws IOException {
        Set<String> typedKernelExtenders = new TreeSet<>();
        try (Stream<Path> paths = Files.walk(Path.of("src/main/java/backend/cpu/kernels"))) {
            for (Path path : paths.filter(path -> path.toString().endsWith(".java")).toList()) {
                if (Files.readString(path).contains("extends TypedCpuKernel")) {
                    typedKernelExtenders.add(path.toString());
                }
            }
        }

        assertEquals(Set.of("src/main/java/backend/cpu/kernels/fused/CpuFusedKernel.java"), typedKernelExtenders,
                "Legacy TypedCpuKernel extension must remain isolated to fused kernels.");
    }

    @Test
    void waveZeroBaselineDocumentCapturesOpsAndBenchmarkSanity() throws IOException {
        String doc = Files.readString(Path.of("docs/cpu-kernels-wave0-baseline.md"));
        List<String> requiredMarkers = List.of(
                "`ADD`",
                "`WHERE`",
                "`SUM`",
                "`CAST`",
                "`CONTIGUOUS`",
                "`MATMUL`",
                "dense F32 add",
                "dense F64 add",
                "BF16 add",
                "F32 sum",
                "F32 matmul Java",
                "F32 OpenBLAS array copy",
                "F32 OpenBLAS segment",
                "NativeCpuElementwiseExecutor",
                "PreparedMatMulExecutableFactory"
        );
        List<String> missing = requiredMarkers.stream()
                .filter(marker -> !doc.contains(marker))
                .toList();
        assertTrue(missing.isEmpty(), () -> "Wave 0 baseline document is missing required markers: " + missing);
    }

    @Test
    void waveFiveMatmulOpenBlasRoutingIsProviderOwned() throws IOException {
        Path providerFactoryPath = Path.of("src/main/java/backend/cpu/provider/linalg/matmul/MatMulProviderExecutableFactory.java");
        assertTrue(Files.exists(providerFactoryPath),
                "Matmul provider routing must live outside backend.cpu.kernels under the CPU provider package.");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/kernels/linalg/matmul/provider/MatMulProviderExecutableFactory.java")),
                "The old kernel provider package must not remain as a compatibility facade.");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/kernels/linalg/matmul/exec/PreparedMatMulExecutableFactory.java")),
                "The old generic prepared factory should not remain as a compatibility facade.");

        String providerFactory = Files.readString(providerFactoryPath);
        assertTrue(providerFactory.contains("package backend.cpu.provider.linalg.matmul;"),
                "Matmul provider factory must declare the CPU provider package.");
        assertTrue(providerFactory.contains("case OPENBLAS_NATIVE_SEGMENT"),
                "OpenBLAS memory-segment routing must stay explicit in the matmul provider factory.");
        assertTrue(providerFactory.contains("case OPENBLAS_ARRAY_COPYING"),
                "OpenBLAS array-copy routing must stay explicit in the matmul provider factory.");
        assertTrue(providerFactory.contains("case JAVA_DIRECT"),
                "Java matmul must remain an explicit array route.");

        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/nativecpu/NativeCpuPlanResolver.java")),
                "Matmul provider ownership must not keep the generic NativeCpuPlanResolver alive.");

        try (Stream<Path> paths = Files.walk(Path.of("src/main/java/backend/cpu/kernels"))) {
            List<Path> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> path.toString().contains("/provider/")
                            || declaresKernelProviderPackage(path)
                            || importsKernelProviderPackage(path))
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "Provider packages must not live under backend.cpu.kernels: " + offenders);
        }
    }

    @Test
    void waveThreeElementwiseRuntimeOwnershipLivesInFamilyExecutors() throws IOException {
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/nativecpu/NativeCpuElementwiseExecutor.java")),
                "Elementwise CPU_NATIVE runtime ownership belongs to family executors, not a standalone native executor.");
        for (String opKernel : List.of(
                "CpuAddKernel",
                "CpuSubKernel",
                "CpuMulKernel",
                "CpuDivKernel",
                "CpuMinKernel",
                "CpuMaxKernel",
                "CpuPowTensorKernel"
        )) {
            String source = Files.readString(Path.of("src/main/java/backend/cpu/kernels/elementwise/binary/" + opKernel + ".java"));
            assertFalse(source.contains("ElementwiseLoops"),
                    opKernel + " must not route through the legacy ElementwiseLoops path.");
            assertFalse(source.contains("extends TypedCpuKernel"),
                    opKernel + " must own the CpuKernelCall execute boundary directly.");
            assertFalse(source.contains("implements " + "Binary" + "ElementwiseKernel"),
                    opKernel + " must not remain on the legacy binary elementwise interface.");
        }
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/binary", "Binary" + "ElementwiseKernel.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/binary", "Elementwise" + "Binary" + "Executor.java")));
        for (String opKernel : List.of(
                "CpuNegKernel",
                "CpuAbsKernel",
                "CpuReluKernel",
                "CpuInvKernel",
                "CpuSqrtKernel",
                "CpuSignKernel",
                "CpuFloorKernel",
                "CpuCeilKernel",
                "CpuLogKernel",
                "CpuExpKernel",
                "CpuTanhKernel",
                "CpuSigmoidKernel",
                "CpuErfKernel",
                "CpuFastExpKernel",
                "CpuFastTanhKernel"
        )) {
            String source = Files.readString(Path.of("src/main/java/backend/cpu/kernels/elementwise/unary/" + opKernel + ".java"));
            assertFalse(source.contains("ElementwiseLoops"),
                    opKernel + " must not route through the legacy ElementwiseLoops path.");
            assertFalse(source.contains("extends TypedCpuKernel"),
                    opKernel + " must own the CpuKernelCall execute boundary directly.");
            assertTrue(source.contains("extends StorageAwareUnaryElementwiseKernel"),
                    opKernel + " must use the storage-aware unary family boundary.");
            assertFalse(source.contains("implements UnaryElementwiseKernel"),
                    opKernel + " must not remain on the legacy unary elementwise interface.");
        }
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/unary", "Unary" + "ElementwiseKernel.java")));
        for (String opKernel : List.of(
                "CpuMulScalarKernel",
                "CpuPowKernel",
                "CpuClampMinKernel",
                "CpuClampMaxKernel"
        )) {
            String source = Files.readString(Path.of("src/main/java/backend/cpu/kernels/elementwise/unary/" + opKernel + ".java"));
            assertFalse(source.contains("ElementwiseLoops"),
                    opKernel + " must not route through the legacy ElementwiseLoops path.");
            assertFalse(source.contains("extends TypedCpuKernel"),
                    opKernel + " must own the CpuKernelCall execute boundary directly.");
            assertTrue(source.contains("extends StorageAwareScalarUnaryElementwiseKernel"),
                    opKernel + " must use the storage-aware scalar unary family boundary.");
            assertFalse(source.contains("implements ScalarUnaryElementwiseKernel"),
                    opKernel + " must not remain on the legacy scalar unary elementwise interface.");
        }
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/unary", "Scalar" + "Unary" + "ElementwiseKernel.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/unary", "Elementwise" + "Unary" + "Executor.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/unary/StorageAwareScalarUnaryElementwiseKernel.java")),
                "Migrated scalar unary kernels must use the storage-aware scalar unary family boundary.");
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/unary/StorageAwareUnaryElementwiseKernel.java")),
                "Migrated unary kernels must use the storage-aware unary family boundary.");
        for (String opKernel : List.of(
                "CpuGreaterThanKernel",
                "CpuGreaterOrEqualKernel",
                "CpuLessThanKernel",
                "CpuLessOrEqualKernel",
                "CpuEqualToKernel",
                "CpuNotEqualToKernel"
        )) {
            String source = Files.readString(Path.of("src/main/java/backend/cpu/kernels/elementwise/compare/" + opKernel + ".java"));
            assertFalse(source.contains("ElementwiseLoops"),
                    opKernel + " must not route through the legacy ElementwiseLoops path.");
            assertFalse(source.contains("extends TypedCpuKernel"),
                    opKernel + " must own the CpuKernelCall execute boundary directly.");
            assertTrue(source.contains("extends StorageAwareCompareElementwiseKernel"),
                    opKernel + " must use the storage-aware compare family boundary.");
            assertFalse(source.contains("implements CompareElementwiseKernel"),
                    opKernel + " must not remain on the legacy compare elementwise interface.");
        }
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/compare", "CompareElementwiseKernel.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/compare", "CompareExecutor.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/compare", "CompareStorageLoops.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/compare/StorageAwareCompareElementwiseKernel.java")),
                "Migrated compare kernels must use the storage-aware compare family boundary.");
        for (String opKernel : List.of(
                "CpuLogicalAndKernel",
                "CpuLogicalOrKernel"
        )) {
            String source = Files.readString(Path.of("src/main/java/backend/cpu/kernels/elementwise/logical/" + opKernel + ".java"));
            assertFalse(source.contains("ElementwiseLoops"),
                    opKernel + " must not route through the legacy ElementwiseLoops path.");
            assertFalse(source.contains("extends TypedCpuKernel"),
                    opKernel + " must own the CpuKernelCall execute boundary directly.");
            assertTrue(source.contains("extends StorageAwareLogicalBinaryElementwiseKernel"),
                    opKernel + " must use the storage-aware logical binary family boundary.");
            assertFalse(source.contains("implements LogicalBinaryElementwiseKernel"),
                    opKernel + " must not remain on the legacy logical binary elementwise interface.");
        }
        String logicalNotSource = Files.readString(Path.of("src/main/java/backend/cpu/kernels/elementwise/logical/CpuLogicalNotKernel.java"));
        assertFalse(logicalNotSource.contains("ElementwiseLoops"),
                "CpuLogicalNotKernel must not route through the legacy ElementwiseLoops path.");
        assertFalse(logicalNotSource.contains("extends TypedCpuKernel"),
                "CpuLogicalNotKernel must own the CpuKernelCall execute boundary directly.");
        assertTrue(logicalNotSource.contains("extends StorageAwareLogicalUnaryElementwiseKernel"),
                "CpuLogicalNotKernel must use the storage-aware logical unary family boundary.");
        assertFalse(logicalNotSource.contains("implements LogicalUnaryElementwiseKernel"),
                "CpuLogicalNotKernel must not remain on the legacy logical unary elementwise interface.");
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/logical/LogicalExecutor.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/logical/LogicalBoolStorageLoops.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/logical/LogicalBinaryElementwiseKernel.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/logical/LogicalUnaryElementwiseKernel.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/logical/StorageAwareLogicalBinaryElementwiseKernel.java")),
                "Migrated logical binary kernels must use the storage-aware logical binary family boundary.");
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/logical/StorageAwareLogicalUnaryElementwiseKernel.java")),
                "Migrated logical unary kernels must use the storage-aware logical unary family boundary.");
        String whereSource = Files.readString(Path.of("src/main/java/backend/cpu/kernels/elementwise/where/CpuWhereKernel.java"));
        assertFalse(whereSource.contains("ElementwiseLoops"),
                "CpuWhereKernel must not route through the legacy ElementwiseLoops path.");
        assertFalse(whereSource.contains("extends TypedCpuKernel"),
                "CpuWhereKernel must own the CpuKernelCall execute boundary directly.");
        assertTrue(whereSource.contains("implements CpuStorageAwareKernel"),
                "CpuWhereKernel must use the storage-aware WHERE boundary.");
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/where/WhereExecutor.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/where/WhereStorageLoops.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/where/WhereElementwiseKernel.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/ElementwiseLoops.java")));
        try (Stream<Path> paths = Files.walk(Path.of("src/main/java/backend/cpu/kernels/elementwise"))) {
            List<Path> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> contains(path, "NativeCpuElementwiseExecutor"))
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "Elementwise kernels must not reference NativeCpuElementwiseExecutor: " + offenders);
        }
    }

    @Test
    void storageAwareSimpleReductionRuntimeOwnershipLivesInFamilyKernels() throws IOException {
        Map<String, String> expectedBases = Map.of(
                "CpuSumKernel", "StorageAwareSumLikeReductionKernel",
                "CpuMeanKernel", "StorageAwareSumLikeReductionKernel",
                "CpuReduceMinKernel", "StorageAwareMinMaxReductionKernel",
                "CpuReduceMaxKernel", "StorageAwareMinMaxReductionKernel",
                "CpuReduceAllKernel", "StorageAwareBoolReductionKernel",
                "CpuReduceAnyKernel", "StorageAwareBoolReductionKernel"
        );
        for (Map.Entry<String, String> entry : expectedBases.entrySet()) {
            String source = Files.readString(Path.of("src/main/java/backend/cpu/kernels/reduction/" + entry.getKey() + ".java"));
            assertFalse(source.contains("extends TypedCpuKernel"),
                    entry.getKey() + " must own the CpuKernelCall execute boundary through the storage-aware reduction base.");
            assertTrue(source.contains("extends " + entry.getValue()),
                    entry.getKey() + " must use the storage-aware reduction family boundary.");
            assertFalse(source.contains("ReductionStorageLoops"),
                    entry.getKey() + " must not route through the old reduction storage loop.");
        }
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/reduction/StorageAwareReductionKernel.java")),
                "Migrated simple reductions must use a storage-aware reduction boundary.");
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/reduction/ReductionStorageLoops.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/reduction/SumLikeReductionExecutor.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/reduction/MinMaxReduceExecutor.java")));
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/reduction/BoolReduceExecutor.java")));
    }

    @Test
    void reductionSoftmaxAndLossEntrypointsOwnStorageAwareBoundary() {
        for (Operation.OpType opType : List.of(
                Operation.OpType.SOFTMAX,
                Operation.OpType.LOG_SOFTMAX,
                Operation.OpType.NLL_LOSS,
                Operation.OpType.CROSS_ENTROPY_LOSS,
                Operation.OpType.CROSS_ENTROPY_LOSS_INDICES
        )) {
            CpuKernel kernel = CpuKernelRegistry.resolve(opType);
            assertEquals("backend.cpu.kernels.reduction", kernel.getClass().getPackageName(),
                    opType + " must be owned by the reduction kernel family.");
            assertTrue(kernel instanceof CpuStorageAwareKernel,
                    opType + " must consume the CpuKernelCall boundary directly.");
            assertFalse(kernel instanceof TypedCpuKernel,
                    opType + " must not route through the legacy TypedCpuKernel tensor-array executor.");
        }
    }

    @Test
    void reductionSpecialCaseEntrypointsOwnStorageAwareBoundaryWithoutNativeRegionPromotion() {
        for (Operation.OpType opType : List.of(
                Operation.OpType.REDUCE_PROD,
                Operation.OpType.CUMSUM,
                Operation.OpType.ARGMAX
        )) {
            CpuKernel kernel = CpuKernelRegistry.resolve(opType);
            assertEquals("backend.cpu.kernels.reduction", kernel.getClass().getPackageName(),
                    opType + " must be owned by the reduction kernel family.");
            assertTrue(kernel instanceof CpuStorageAwareKernel,
                    opType + " must consume CpuKernelCall storage views directly.");
            assertFalse(kernel instanceof TypedCpuKernel,
                    opType + " must not route through the legacy TypedCpuKernel tensor-array executor.");
            for (DataType dtype : DataType.values()) {
                assertFalse(CpuNativeStorageSupport.nativeRegionSupported(opType, dtype),
                        opType + " must not be promoted to CPU_NATIVE region support in the reduction special-case slice.");
            }
        }
    }

    @Test
    void waveFourNonElementwiseNativeExecutorsAreDeleted() throws IOException {
        List<String> deletedExecutors = List.of(
                "NativeCpuReductionExecutor",
                "NativeCpuCompareExecutor",
                "NativeCpuBoolMaskExecutor",
                "NativeCpuCastExecutor",
                "NativeCpuContiguousExecutor",
                "NativeCpuViewExecutor"
        );
        for (String executor : deletedExecutors) {
            assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/nativecpu/" + executor + ".java")),
                    executor + " must not exist after Wave 4 runtime ownership moves to kernel storage owners.");
        }
        try (Stream<Path> paths = Files.walk(Path.of("src/main/java/backend/cpu/kernels"))) {
            List<Path> offenders = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> deletedExecutors.stream().anyMatch(executor -> contains(path, executor)))
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "CPU kernels must not reference deleted Wave 4 native executors: " + offenders);
        }
    }

    @Test
    void waveSixNativePlanStackIsNotInPrepareOrRuntimePath() throws IOException {
        List<String> deletedPlanClasses = List.of(
                "NativeCpuPlanResolver",
                "PreparedNativeCpuPlan",
                "PreparedNativeCpuRoute",
                "PreparedNativeCpuInputPolicy"
        );
        for (String className : deletedPlanClasses) {
            assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/nativecpu/" + className + ".java")),
                    className + " must not exist after Wave 6.");
        }

        List<Path> runtimeRoots = List.of(
                Path.of("src/main/java/backend/cpu/kernels"),
                Path.of("src/main/java/backend/cpu/lowering"),
                Path.of("src/main/java/backend/cpu/prepare"),
                Path.of("src/main/java/backend/cpu/CpuStepTraceContributor.java"),
                Path.of("src/main/java/backend/cpu/nativecpu/PreparedNativeCpuRegionExecutable.java")
        );
        for (Path root : runtimeRoots) {
            List<Path> offenders = javaSourceFiles(root).stream()
                    .filter(path -> deletedPlanClasses.stream().anyMatch(className -> contains(path, className)))
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "Wave 6 plan stack leaked into runtime path " + root + ": " + offenders);
        }

        List<String> deletedFactsRuntimeImports = List.of(
                "NativeCpuKernelFacts",
                "NativeCpuKernelFact",
                "NativeCpuCoverageMatrix",
                "NativeCpuParityMatrix",
                "NativeCpuCoverageEntry",
                "NativeCpuParityEntry"
        );
        for (String className : deletedFactsRuntimeImports) {
            assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/nativecpu/" + className + ".java")),
                    className + " must not exist after Wave 6.");
        }
        for (Path root : runtimeRoots) {
            List<Path> offenders = javaSourceFiles(root).stream()
                    .filter(path -> deletedFactsRuntimeImports.stream().anyMatch(className -> contains(path, className)))
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "Wave 6 facts/parity stack leaked into runtime path " + root + ": " + offenders);
        }
    }

    @Test
    void waveSevenFusedMemorySegmentBindingIsNotOwnedByCpuKernelContext() throws IOException {
        String context = Files.readString(Path.of("src/main/java/backend/cpu/execution/CpuKernelContext.java"));
        assertFalse(context.contains("java.lang.foreign.MemorySegment"),
                "CpuKernelContext must not import or expose MemorySegment for fused special cases.");
        assertFalse(context.contains("bindFusedNativeSegments"),
                "Fused MemorySegment binding lifecycle belongs under backend.cpu.fused.exec.");
        assertFalse(context.contains("fusedNativeInputSegment"),
                "Fused MemorySegment input access belongs under backend.cpu.fused.exec.");
        assertFalse(context.contains("fusedNativeOutputSegment"),
                "Fused MemorySegment output access belongs under backend.cpu.fused.exec.");
        assertFalse(context.contains("fusedNativeOutputStorage"),
                "Fused MemorySegment output storage access belongs under backend.cpu.fused.exec.");
        assertFalse(context.contains("publishFusedNativeOutput"),
                "Fused MemorySegment output publication belongs under backend.cpu.fused.exec.");
        assertFalse(context.contains("clearFusedNativeBindings"),
                "Fused MemorySegment binding cleanup belongs under backend.cpu.fused.exec.");

        String bindings = Files.readString(Path.of("src/main/java/backend/cpu/fused/exec/FusedNativeSegmentBindings.java"));
        assertTrue(bindings.contains("java.lang.foreign.MemorySegment"),
                "Fused MemorySegment ownership must stay explicit in backend.cpu.fused.exec.");
        assertTrue(bindings.contains("static FusedNativeSegmentBindings bind"),
                "Fused MemorySegment binding lifecycle must have a fused-owned entrypoint.");
    }

    @Test
    void waveEightAutoNativePolicyExcludesSegmentScalarKernelsWithoutProof() {
        assertTrue(CpuNativeStorageSupport.nativeRegionSupported(Operation.OpType.ADD, DataType.FLOAT32),
                "ADD has a correct native segment implementation for explicit CPU_NATIVE runs.");
        assertFalse(CpuNativeStorageSupport.autoNativeRegionEligible(Operation.OpType.ADD, DataType.FLOAT32),
                "AUTO must not select segment scalar elementwise kernels without benchmark promotion.");
        assertFalse(CpuNativeStorageSupport.autoNativeRegionEligible(Operation.OpType.SUM, DataType.FLOAT32),
                "AUTO must not select segment scalar reductions without benchmark promotion.");
        assertFalse(CpuNativeStorageSupport.autoNativeRegionEligible(Operation.OpType.CAST, DataType.BFLOAT16),
                "AUTO must not select segment scalar layout/materialization kernels without benchmark promotion.");

        assertTrue(CpuNativeStorageSupport.autoNativeRegionEligible(Operation.OpType.MATMUL, DataType.FLOAT32),
                "AUTO may select measured provider-backed MemorySegment matmul routes.");
        assertTrue(CpuNativeStorageSupport.autoNativeRegionEligible(Operation.OpType.RESHAPE, DataType.FLOAT32),
                "AUTO may preserve native storage through metadata-only view aliases.");
    }

    @Test
    void indexReadKernelsOwnStorageAwareBoundaryWithoutNativeRegionPromotion() {
        for (Operation.OpType opType : List.of(
                Operation.OpType.GATHER,
                Operation.OpType.GATHER_AXIS,
                Operation.OpType.GATHER_ND,
                Operation.OpType.TAKE_ALONG_AXIS
        )) {
            CpuKernel kernel = CpuKernelRegistry.resolve(opType);
            assertTrue(kernel instanceof CpuStorageAwareKernel,
                    opType + " must consume CpuKernelCall storage views directly.");
            assertFalse(kernel instanceof TypedCpuKernel,
                    opType + " must not route through the legacy TypedCpuKernel tensor-array executor.");
            assertFalse(CpuNativeStorageSupport.nativeRegionSupported(opType, DataType.FLOAT32),
                    opType + " must not be promoted to CPU_NATIVE region support in the index read slice.");
        }
    }

    @Test
    void indexWriteScatterKernelsOwnStorageAwareBoundaryWithoutNativeRegionPromotion() {
        for (Operation.OpType opType : List.of(
                Operation.OpType.SCATTER_ADD,
                Operation.OpType.SCATTER_AXIS_ADD,
                Operation.OpType.SCATTER_ELEMENTS,
                Operation.OpType.SCATTER_ND,
                Operation.OpType.SLICE_SCATTER_ADD
        )) {
            CpuKernel kernel = CpuKernelRegistry.resolve(opType);
            assertTrue(kernel instanceof CpuStorageAwareKernel,
                    opType + " must consume CpuKernelCall storage views directly.");
            assertFalse(kernel instanceof TypedCpuKernel,
                    opType + " must not route through the legacy TypedCpuKernel tensor-array executor.");
            assertFalse(CpuNativeStorageSupport.nativeRegionSupported(opType, DataType.FLOAT32),
                    opType + " must not be promoted to CPU_NATIVE region support in the index write scatter slice.");
        }
    }

    @Test
    void layoutMaterializationKernelsOwnStorageAwareBoundaryWithoutNativeRegionPromotion() {
        for (Operation.OpType opType : List.of(
                Operation.OpType.CONCAT,
                Operation.OpType.PAD,
                Operation.OpType.TILE,
                Operation.OpType.UNFOLD_AXIS,
                Operation.OpType.UNFOLD2D,
                Operation.OpType.FOLD2D
        )) {
            CpuKernel kernel = CpuKernelRegistry.resolve(opType);
            assertTrue(kernel instanceof CpuStorageAwareKernel,
                    opType + " must consume CpuKernelCall storage views directly.");
            assertFalse(kernel instanceof TypedCpuKernel,
                    opType + " must not route through the legacy TypedCpuKernel tensor-array executor.");
            for (DataType dtype : DataType.values()) {
                assertFalse(CpuNativeStorageSupport.nativeRegionSupported(opType, dtype),
                        opType + " must not be promoted to CPU_NATIVE region support in the layout materialization slice.");
            }
        }
    }

    @Test
    void linalgEntrypointsOwnStorageAwareBoundary() {
        for (Operation.OpType opType : List.of(
                Operation.OpType.MATMUL,
                Operation.OpType.LINEAR,
                Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION,
                Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS
        )) {
            CpuKernel kernel = CpuKernelRegistry.resolve(opType);
            assertEquals("backend.cpu.kernels.linalg", kernel.getClass().getPackageName(),
                    opType + " must be owned by the linalg kernel family.");
            assertTrue(kernel instanceof CpuStorageAwareKernel,
                    opType + " must consume the CpuKernelCall boundary directly.");
            assertFalse(kernel instanceof TypedCpuKernel,
                    opType + " must not route through the legacy TypedCpuKernel tensor-array executor.");
        }
        for (DataType dtype : DataType.values()) {
            assertFalse(CpuNativeStorageSupport.nativeRegionSupported(Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION, dtype),
                    "attention direct kernel must not be promoted to CPU_NATIVE region support in this entrypoint slice.");
            assertFalse(CpuNativeStorageSupport.nativeRegionSupported(Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS, dtype),
                    "attention weights export must not be promoted to CPU_NATIVE region support in this entrypoint slice.");
        }
    }

    @Test
    void nnForwardEntrypointsOwnStorageAwareBoundary() throws IOException {
        for (Operation.OpType opType : List.of(
                Operation.OpType.CONV2D,
                Operation.OpType.MAX_POOL2D,
                Operation.OpType.AVG_POOL2D,
                Operation.OpType.LAYER_NORM,
                Operation.OpType.RMS_NORM
        )) {
            CpuKernel kernel = CpuKernelRegistry.resolve(opType);
            assertEquals("backend.cpu.kernels.nn", kernel.getClass().getPackageName(),
                    opType + " must be owned by the nn kernel family.");
            assertTrue(kernel instanceof CpuStorageAwareKernel,
                    opType + " must consume the CpuKernelCall boundary directly.");
            assertFalse(kernel instanceof TypedCpuKernel,
                    opType + " must not route through the legacy TypedCpuKernel tensor-array executor.");
            for (DataType dtype : DataType.values()) {
                assertFalse(CpuNativeStorageSupport.nativeRegionSupported(opType, dtype),
                        opType + " must not be promoted to CPU_NATIVE region support in the nn forward slice.");
            }
        }
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/nn/Conv2dExecutor.java")),
                "Conv2dExecutor must not remain as a pass-through forward wrapper.");
        assertFalse(Files.exists(Path.of("src/main/java/backend/cpu/kernels/nn/Pool2dExecutor.java")),
                "Pool2dExecutor must not remain as a pass-through forward wrapper.");
    }

    private static void assertPackage(Operation.OpType opType, String expectedPackage) {
        CpuKernel kernel = CpuKernelRegistry.resolve(opType);
        assertEquals(expectedPackage, kernel.getClass().getPackageName(), () ->
                "Kernel for " + opType + " should live in " + expectedPackage + " but was " + kernel.getClass().getPackageName());
    }

    private static List<Operation.OpType> nonFinalLegacyBackwardOpTypes() {
        return List.of(
                Operation.OpType.MIN_GRAD,
                Operation.OpType.MAX_GRAD,
                Operation.OpType.REDUCE_MIN_GRAD,
                Operation.OpType.REDUCE_MAX_GRAD,
                Operation.OpType.SOFTMAX_GRAD,
                Operation.OpType.LOG_SOFTMAX_GRAD,
                Operation.OpType.GATHER_GRAD,
                Operation.OpType.GATHER_AXIS_GRAD,
                Operation.OpType.GATHER_ND_GRAD,
                Operation.OpType.TAKE_ALONG_AXIS_GRAD,
                Operation.OpType.SLICE_GRAD,
                Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD,
                Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD
        );
    }

    private static Map<String, Set<String>> nativeCpuImportsUnder(Path root) throws IOException {
        Map<String, Set<String>> imports = new TreeMap<>();
        try (Stream<Path> paths = Files.walk(root)) {
            for (Path path : paths
                    .filter(Files::isRegularFile)
                    .filter(file -> file.toString().endsWith(".java"))
                    .toList()) {
                List<String> nativeImports = Files.readAllLines(path).stream()
                        .map(String::trim)
                        .filter(line -> line.startsWith("import backend.cpu.nativecpu."))
                        .map(CpuKernelFamilyArchitectureTest::importedSimpleName)
                        .sorted()
                        .toList();
                if (!nativeImports.isEmpty()) {
                    String relativePath = root.relativize(path).toString().replace('\\', '/');
                    imports.put(relativePath, new TreeSet<>(nativeImports));
                }
            }
        }
        return imports;
    }

    private static String importedSimpleName(String importLine) {
        String prefix = "import backend.cpu.nativecpu.";
        String suffix = importLine.substring(prefix.length());
        return suffix.endsWith(";") ? suffix.substring(0, suffix.length() - 1) : suffix;
    }

    private static void assertPlanRootType(String fileName) throws IOException {
        Path newPath = Path.of("src/main/java/backend/cpu/plan", fileName);
        Path oldPath = Path.of("src/main/java/backend/cpu/kernels", fileName);
        assertTrue(Files.exists(newPath), fileName + " must live under backend.cpu.plan.");
        assertTrue(Files.readString(newPath).contains("package backend.cpu.plan;"),
                fileName + " must declare package backend.cpu.plan.");
        assertFalse(Files.exists(oldPath), fileName + " must not remain under backend.cpu.kernels.");
    }

    private static void assertNativeCpuRootType(String fileName) throws IOException {
        Path newPath = Path.of("src/main/java/backend/cpu/nativecpu", fileName);
        Path oldPath = Path.of("src/main/java/backend/cpu/kernels", fileName);
        assertTrue(Files.exists(newPath), fileName + " must live under backend.cpu.nativecpu.");
        assertTrue(Files.readString(newPath).contains("package backend.cpu.nativecpu;"),
                fileName + " must declare package backend.cpu.nativecpu.");
        assertFalse(Files.exists(oldPath), fileName + " must not remain under backend.cpu.kernels.");
    }

    private static void assertCpuExecutionRootType(String fileName) throws IOException {
        Path newPath = Path.of("src/main/java/backend/cpu/execution", fileName);
        Path oldPath = Path.of("src/main/java/backend/cpu/kernels", fileName);
        assertTrue(Files.exists(newPath), fileName + " must live under backend.cpu.execution.");
        assertTrue(Files.readString(newPath).contains("package backend.cpu.execution;"),
                fileName + " must declare package backend.cpu.execution.");
        assertFalse(Files.exists(oldPath), fileName + " must not remain under backend.cpu.kernels.");
    }

    private static boolean hasConcreteConvBackwardKernel() {
        Path nnRoot = Path.of("src/main/java/backend/cpu/kernels/nn");
        if (!Files.isDirectory(nnRoot)) {
            return false;
        }
        try (Stream<Path> paths = Files.list(nnRoot)) {
            return paths
                    .map(path -> path.getFileName().toString())
                    .anyMatch(name -> name.startsWith("CpuConv2d")
                            && name.contains("Backward")
                            && name.endsWith("Kernel.java"));
        } catch (IOException e) {
            throw new AssertionError("Unable to inspect CPU NN kernel package", e);
        }
    }

    private static boolean contains(Path path, String needle) {
        try {
            return Files.readString(path).contains(needle);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static boolean declaresKernelProviderPackage(Path path) {
        return containsMatchingLine(path, line -> line.startsWith("package backend.cpu.kernels.")
                && line.contains(".provider"));
    }

    private static boolean importsKernelProviderPackage(Path path) {
        return containsMatchingLine(path, line -> line.startsWith("import backend.cpu.kernels.")
                && line.contains(".provider."));
    }

    private static boolean containsMatchingLine(Path path, java.util.function.Predicate<String> predicate) {
        try {
            return Files.readAllLines(path).stream()
                    .map(String::trim)
                    .anyMatch(predicate);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static List<Path> javaSourceFiles(Path root) throws IOException {
        if (Files.isRegularFile(root)) {
            return root.toString().endsWith(".java") ? List.of(root) : List.of();
        }
        try (Stream<Path> paths = Files.walk(root)) {
            return paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .toList();
        }
    }
}
