import backend.kernels.cpu.CpuKernel;
import backend.registry.CpuKernelResolver;
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
        assertPackage(Operation.OpType.ADD, "backend.kernels.cpu.elementwise.binary");
        assertPackage(Operation.OpType.WHERE, "backend.kernels.cpu.elementwise.where");
        assertPackage(Operation.OpType.NOOP, "backend.kernels.cpu.layout");
        assertPackage(Operation.OpType.GATHER, "backend.kernels.cpu.index");
        assertPackage(Operation.OpType.SUM, "backend.kernels.cpu.reduction");
        assertPackage(Operation.OpType.MATMUL, "backend.kernels.cpu.linalg");
        assertPackage(Operation.OpType.CONV2D, "backend.kernels.cpu.nn");
        assertPackage(Operation.OpType.FUSED, "backend.kernels.cpu.fused");
    }

    @Test
    void cpuRootPackageOnlyContainsSharedInfrastructure() throws IOException {
        Path root = Path.of("src/main/java/backend/kernels/cpu");
        Set<String> allowed = Set.of(
                "CpuAccumulateDType.java",
                "CpuComputeDType.java",
                "CpuDTypeOps.java",
                "CpuExecutionBackend.java",
                "CpuExecutionMode.java",
                "CpuExecutionPlanner.java",
                "CpuKernel.java",
                "CpuKernelContext.java",
                "CpuKernelCostClass.java",
                "CpuNodeExecutionPlan.java",
                "CpuNodeWorkspace.java",
                "CpuStridedElementWise.java",
                "CpuThreadPool.java",
                "ResolvedBroadcastPlan.java",
                "ResolvedCpuComputeContract.java",
                "ResolvedDispatchHints.java",
                "ResolvedMatMulHints.java",
                "ResolvedReductionHints.java",
                "ResolvedWhereBroadcastPlan.java"
        );

        try (Stream<Path> files = Files.list(root)) {
            List<String> unexpected = files
                    .filter(Files::isRegularFile)
                    .map(path -> path.getFileName().toString())
                    .filter(name -> !allowed.contains(name))
                    .sorted()
                    .toList();
            assertTrue(unexpected.isEmpty(), () -> "Unexpected non-infra files in backend.kernels.cpu root: " + unexpected);
        }
    }

    @Test
    void cpuRootPackageDoesNotContainConcreteKernelEntrypoints() throws IOException {
        Path root = Path.of("src/main/java/backend/kernels/cpu");
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

    private static void assertPackage(Operation.OpType opType, String expectedPackage) {
        CpuKernel kernel = CpuKernelResolver.resolve(opType);
        assertEquals(expectedPackage, kernel.getClass().getPackageName(), () ->
                "Kernel for " + opType + " should live in " + expectedPackage + " but was " + kernel.getClass().getPackageName());
    }
}
