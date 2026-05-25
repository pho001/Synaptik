import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuNativeStorageSupport;
import backend.cpu.registry.CpuKernelResolver;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.io.IOException;
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
        assertPackage(Operation.OpType.FUSED, "backend.cpu.kernels.fused");
    }

    @Test
    void cpuRootPackageOnlyContainsSharedInfrastructure() throws IOException {
        Path root = Path.of("src/main/java/backend/cpu/kernels");
        Set<String> allowed = Set.of(
                "CpuAccumulateDType.java",
                "CpuComputeDType.java",
                "CpuDTypeOps.java",
                "CpuExecutionBackend.java",
                "CpuExecutionMode.java",
                "CpuKernel.java",
                "CpuKernelContext.java",
                "CpuKernelCostClass.java",
                "CpuNativeStorageSupport.java",
                "CpuNativeTraceSupport.java",
                "CpuNodeExecutionPlan.java",
                "CpuNodeWorkspace.java",
                "CpuThreadPool.java",
                "ResolvedCpuComputeContract.java"
        );

        try (Stream<Path> files = Files.list(root)) {
            List<String> unexpected = files
                    .filter(Files::isRegularFile)
                    .map(path -> path.getFileName().toString())
                    .filter(name -> name.endsWith(".java"))
                    .filter(name -> !allowed.contains(name))
                    .sorted()
                    .toList();
            assertTrue(unexpected.isEmpty(), () -> "Unexpected non-infra files in backend.cpu.kernels root: " + unexpected);
        }
    }

    @Test
    void cpuRootPackageDoesNotContainConcreteKernelEntrypoints() throws IOException {
        Path root = Path.of("src/main/java/backend/cpu/kernels");
        try (Stream<Path> files = Files.list(root)) {
            List<String> offenders = files
                    .filter(Files::isRegularFile)
                    .map(path -> path.getFileName().toString())
                    .filter(name -> name.startsWith("Cpu") && name.endsWith("Kernel.java") && !name.equals("CpuKernel.java"))
                    .sorted()
                    .toList();
            assertTrue(offenders.isEmpty(), () -> "Concrete Cpu*Kernel entrypoints must live in family packages: " + offenders);
        }
    }

    @Test
    void familySpecificPlanningAndSupportHelpersLiveOutsideCpuRoot() {
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/plan/CpuExecutionPlanner.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/plan/CpuPlanAssembler.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/plan/CpuOperationPlanResolver.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/plan/ResolvedCpuOperationPlans.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/plan/ElementwiseDispatchPlanner.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/CpuStridedElementWise.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedBooleanLoops.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedNumericInputs.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedNumericLoops.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedWhereLoops.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedScalarLoops.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedRank2Loops.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedElementWiseSemantics.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedVectorSupport.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/strided/StridedOffsetCursor.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/fused/plan/FusedDispatchPlanner.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/fused/plan/PreparedFusedDispatch.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/elementwise/unary/support/CpuPowSupport.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/layout/PreparedInputPlanner.java")));
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/layout/PreparedInputPolicy.java")));
    }

    @Test
    void waveZeroCriticalOpOwnersHaveDocumentedEntrypoints() {
        List<String> requiredPaths = List.of(
                "src/main/java/backend/cpu/kernels/elementwise/binary/CpuAddKernel.java",
                "src/main/java/backend/cpu/kernels/elementwise/binary/ElementwiseBinaryExecutor.java",
                "src/main/java/backend/cpu/kernels/elementwise/ElementwiseLoops.java",
                "src/main/java/backend/cpu/kernels/elementwise/binary/array/AddArrayLoops.java",
                "src/main/java/backend/cpu/kernels/elementwise/binary/array/AddF32.java",
                "src/main/java/backend/cpu/kernels/elementwise/binary/array/AddF64.java",
                "src/main/java/backend/cpu/kernels/elementwise/binary/array/AddBF16.java",
                "src/main/java/backend/cpu/kernels/elementwise/where/CpuWhereKernel.java",
                "src/main/java/backend/cpu/kernels/elementwise/where/WhereExecutor.java",
                "src/main/java/backend/cpu/kernels/reduction/CpuSumKernel.java",
                "src/main/java/backend/cpu/kernels/reduction/SumLikeReductionExecutor.java",
                "src/main/java/backend/cpu/kernels/reduction/SumLoops.java",
                "src/main/java/backend/cpu/kernels/layout/CpuCastKernel.java",
                "src/main/java/backend/cpu/kernels/layout/CpuContiguousKernel.java",
                "src/main/java/backend/cpu/kernels/layout/LayoutExecutor.java",
                "src/main/java/backend/cpu/kernels/linalg/CpuMatMulKernel.java",
                "src/main/java/backend/cpu/kernels/linalg/matmul/plan/MatMulPlanner.java",
                "src/main/java/backend/cpu/kernels/linalg/matmul/provider/MatMulProviderExecutableFactory.java",
                "src/main/java/backend/cpu/kernels/linalg/matmul/f32/F32JavaMatMulExecutable.java",
                "src/main/java/backend/cpu/kernels/linalg/matmul/f32/F32BlasMatMulExecutable.java",
                "src/main/java/backend/cpu/kernels/linalg/matmul/f32/F32NativeBlasMatMulExecutable.java",
                "src/main/java/backend/cpu/kernels/elementwise/ElementwiseNativeSupport.java",
                "src/main/java/backend/cpu/kernels/elementwise/binary/segment/AddSegmentLoops.java",
                "src/main/java/backend/cpu/kernels/elementwise/binary/segment/BinarySegmentLoops.java",
                "src/main/java/backend/cpu/kernels/elementwise/unary/segment/UnarySegmentLoops.java",
                "src/main/java/backend/cpu/kernels/elementwise/where/WhereStorageLoops.java",
                "src/main/java/backend/cpu/kernels/elementwise/compare/CompareStorageLoops.java",
                "src/main/java/backend/cpu/kernels/elementwise/logical/LogicalBoolStorageLoops.java",
                "src/main/java/backend/cpu/kernels/reduction/ReductionStorageLoops.java",
                "src/main/java/backend/cpu/kernels/layout/LayoutExecutor.java"
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
                Map.entry("CpuNativeTraceSupport.java", Set.of("NativeCpuTraceState")),
                Map.entry("reduction/ReductionStorageLoops.java", Set.of(
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
        assertTrue(Files.exists(Path.of("src/main/java/backend/cpu/kernels/linalg/matmul/provider/MatMulProviderExecutableFactory.java")),
                "Matmul provider routing must live under the linalg matmul provider package.");
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/kernels/linalg/matmul/exec/PreparedMatMulExecutableFactory.java")),
                "The old generic prepared factory should not remain as a compatibility facade.");

        String providerFactory = Files.readString(Path.of("src/main/java/backend/cpu/kernels/linalg/matmul/provider/MatMulProviderExecutableFactory.java"));
        assertTrue(providerFactory.contains("case OPENBLAS_NATIVE_SEGMENT"),
                "OpenBLAS memory-segment routing must stay explicit in the matmul provider factory.");
        assertTrue(providerFactory.contains("case OPENBLAS_ARRAY_COPYING"),
                "OpenBLAS array-copy routing must stay explicit in the matmul provider factory.");
        assertTrue(providerFactory.contains("case JAVA_DIRECT"),
                "Java matmul must remain an explicit array route.");

        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/nativecpu/NativeCpuPlanResolver.java")),
                "Matmul provider ownership must not keep the generic NativeCpuPlanResolver alive.");
    }

    @Test
    void waveThreeElementwiseRuntimeOwnershipMovedToSegmentLoops() throws IOException {
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/nativecpu/NativeCpuElementwiseExecutor.java")),
                "Elementwise CPU_NATIVE runtime ownership belongs to segment loops, not a standalone native executor.");
        assertTrue(Files.readString(Path.of("src/main/java/backend/cpu/kernels/elementwise/binary/ElementwiseBinaryExecutor.java"))
                        .contains("AddSegmentLoops.execute"),
                "Binary executor must route ADD through the ADD segment loop before the generic native elementwise executor.");
        assertTrue(Files.readString(Path.of("src/main/java/backend/cpu/kernels/elementwise/binary/ElementwiseBinaryExecutor.java"))
                        .contains("BinarySegmentLoops.execute"),
                "Non-ADD binary elementwise runtime ownership must live in BinarySegmentLoops.");
        assertTrue(Files.readString(Path.of("src/main/java/backend/cpu/kernels/elementwise/unary/ElementwiseUnaryExecutor.java"))
                        .contains("UnarySegmentLoops.execute"),
                "Unary elementwise runtime ownership must live in UnarySegmentLoops.");
        assertTrue(Files.readString(Path.of("src/main/java/backend/cpu/kernels/elementwise/where/WhereExecutor.java"))
                        .contains("WhereStorageLoops.execute"),
                "WHERE runtime ownership must live in WhereStorageLoops.");
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
        String context = Files.readString(Path.of("src/main/java/backend/cpu/kernels/CpuKernelContext.java"));
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

    private static void assertPackage(Operation.OpType opType, String expectedPackage) {
        CpuKernel kernel = CpuKernelResolver.resolve(opType);
        assertEquals(expectedPackage, kernel.getClass().getPackageName(), () ->
                "Kernel for " + opType + " should live in " + expectedPackage + " but was " + kernel.getClass().getPackageName());
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

    private static boolean contains(Path path, String needle) {
        try {
            return Files.readString(path).contains(needle);
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
