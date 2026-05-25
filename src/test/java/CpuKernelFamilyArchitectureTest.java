import backend.cpu.kernels.CpuKernel;
import backend.cpu.registry.CpuKernelResolver;
import operations.Operation;
import org.junit.jupiter.api.Test;

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
                "src/main/java/backend/cpu/kernels/elementwise/binary/f32/AddF32.java",
                "src/main/java/backend/cpu/kernels/elementwise/binary/f64/AddF64.java",
                "src/main/java/backend/cpu/kernels/elementwise/binary/bf16/AddBF16.java",
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
                "src/main/java/backend/cpu/kernels/linalg/matmul/exec/PreparedMatMulExecutableFactory.java",
                "src/main/java/backend/cpu/kernels/linalg/matmul/f32/F32JavaMatMulExecutable.java",
                "src/main/java/backend/cpu/kernels/linalg/matmul/f32/F32BlasMatMulExecutable.java",
                "src/main/java/backend/cpu/kernels/linalg/matmul/f32/F32NativeBlasMatMulExecutable.java",
                "src/main/java/backend/cpu/kernels/elementwise/ElementwiseNativeSupport.java",
                "src/main/java/backend/cpu/kernels/elementwise/binary/BinaryStorageLoops.java",
                "src/main/java/backend/cpu/kernels/elementwise/unary/UnaryStorageLoops.java",
                "src/main/java/backend/cpu/kernels/elementwise/where/WhereStorageLoops.java",
                "src/main/java/backend/cpu/nativecpu/NativeCpuReductionExecutor.java",
                "src/main/java/backend/cpu/nativecpu/NativeCpuCastExecutor.java",
                "src/main/java/backend/cpu/nativecpu/NativeCpuContiguousExecutor.java"
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
                Map.entry("CpuNodeExecutionPlan.java", Set.of("PreparedNativeCpuPlan")),
                Map.entry("plan/CpuPlanAssembler.java", Set.of("NativeCpuPlanResolver", "PreparedNativeCpuPlan")),
                Map.entry("elementwise/ElementwiseNativeSupport.java", Set.of("NativeCpuKernelFact", "NativeCpuTraceState")),
                Map.entry("elementwise/binary/AddStorageLoops.java", Set.of("NativeCpuKernelFact", "NativeCpuKernelFacts")),
                Map.entry("elementwise/binary/BinaryStorageLoops.java", Set.of("NativeCpuKernelFact", "NativeCpuKernelFacts")),
                Map.entry("elementwise/unary/UnaryStorageLoops.java", Set.of("NativeCpuKernelFact", "NativeCpuKernelFacts")),
                Map.entry("elementwise/where/WhereStorageLoops.java", Set.of("NativeCpuKernelFact", "NativeCpuKernelFacts")),
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
        );

        assertEquals(expected, nativeCpuImportsUnder(Path.of("src/main/java/backend/cpu/kernels")),
                "Wave 0 baseline must make native CPU dependencies explicit before the rewrite removes them.");
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
    void waveThreeElementwiseRuntimeOwnershipMovedToStorageLoops() throws IOException {
        assertTrue(!Files.exists(Path.of("src/main/java/backend/cpu/nativecpu/NativeCpuElementwiseExecutor.java")),
                "Elementwise CPU_NATIVE runtime ownership belongs to storage loops, not a standalone native executor.");
        assertTrue(Files.readString(Path.of("src/main/java/backend/cpu/kernels/elementwise/binary/ElementwiseBinaryExecutor.java"))
                        .contains("AddStorageLoops.execute"),
                "Binary executor must route ADD through the ADD storage loop before the generic native elementwise executor.");
        assertTrue(Files.readString(Path.of("src/main/java/backend/cpu/kernels/elementwise/binary/ElementwiseBinaryExecutor.java"))
                        .contains("BinaryStorageLoops.execute"),
                "Non-ADD binary elementwise runtime ownership must live in BinaryStorageLoops.");
        assertTrue(Files.readString(Path.of("src/main/java/backend/cpu/kernels/elementwise/unary/ElementwiseUnaryExecutor.java"))
                        .contains("UnaryStorageLoops.execute"),
                "Unary elementwise runtime ownership must live in UnaryStorageLoops.");
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
}
