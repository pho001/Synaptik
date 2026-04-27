import backend.cpu.kernels.CpuKernel;
import backend.cpu.registry.CpuKernelResolver;
import operations.Operation;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
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

    private static void assertPackage(Operation.OpType opType, String expectedPackage) {
        CpuKernel kernel = CpuKernelResolver.resolve(opType);
        assertEquals(expectedPackage, kernel.getClass().getPackageName(), () ->
                "Kernel for " + opType + " should live in " + expectedPackage + " but was " + kernel.getClass().getPackageName());
    }
}
