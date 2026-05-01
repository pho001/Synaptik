import org.junit.jupiter.api.Test;
import tuning.benchmark.report.GpuCoverageGatePolicy;
import tuning.benchmark.report.GpuCoverageRegressionGate;
import tuning.benchmark.report.GpuCoverageSummary;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GpuCoverageRegressionGateTest {
    @Test
    void failsWhenGpuCoverageIsLost() {
        var summary = summary("GPU_METAL", coverage(0.25d, 1, 1, 0, 0, 1, 0));
        var policy = GpuCoverageGatePolicy.nativeBufferTarget("GPU_METAL", 0.5d, 3);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> GpuCoverageRegressionGate.requirePass(summary, policy)
        );

        assertTrue(failure.getMessage().contains("lost GPU coverage"));
    }

    @Test
    void failsWhenUnexpectedCpuMaterializationAppears() {
        var summary = summary("GPU_METAL", coverage(0.75d, 3, 1, 0, 0, 0, 0));
        var policy = GpuCoverageGatePolicy.nativeBufferTarget("GPU_METAL", 0.5d, 3);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> GpuCoverageRegressionGate.requirePass(summary, policy)
        );

        assertTrue(failure.getMessage().contains("unexpected CPU materialization"));
    }

    @Test
    void failsWhenTensorArrayFallbackIsHidden() {
        var summary = summary("GPU_METAL", coverage(0.75d, 3, 0, 1, 1, 0, 0));
        var policy = GpuCoverageGatePolicy.nativeBufferTarget("GPU_METAL", 0.5d, 3);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> GpuCoverageRegressionGate.requirePass(summary, policy)
        );

        assertTrue(failure.getMessage().contains("hidden tensor-array fallback"));
    }

    @Test
    void failsWhenDeviceHandoffBudgetIsExceeded() {
        var summary = summary("GPU_METAL", coverage(0.75d, 3, 0, 0, 0, 3, 1));
        var policy = GpuCoverageGatePolicy.nativeBufferTarget("GPU_METAL", 0.5d, 3);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> GpuCoverageRegressionGate.requirePass(summary, policy)
        );

        assertTrue(failure.getMessage().contains("unexpected device handoff"));
    }

    @Test
    void passesWhenCoverageMatchesPolicy() {
        var summary = summary("GPU_CUDA", coverage(0.75d, 3, 0, 0, 0, 1, 1));
        var policy = GpuCoverageGatePolicy.nativeBufferTarget("GPU_CUDA", 0.5d, 3);

        assertDoesNotThrow(() -> GpuCoverageRegressionGate.requirePass(summary, policy));
    }

    @Test
    void failsWhenCoverageSummaryIsMissing() {
        var summary = new GpuCoverageSummary(Map.of());
        var policy = GpuCoverageGatePolicy.nativeBufferTarget("GPU_METAL", 0.5d, 3);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> GpuCoverageRegressionGate.requirePass(summary, policy)
        );

        assertTrue(failure.getMessage().contains("missing coverage summary"));
    }

    private static GpuCoverageSummary summary(String backend, GpuCoverageSummary.BackendCoverage coverage) {
        return new GpuCoverageSummary(Map.of(backend, coverage));
    }

    private static GpuCoverageSummary.BackendCoverage coverage(
            double ratio,
            int maxSelectedRegionLength,
            int cpuMaterializationCount,
            int tensorArrayStepCount,
            int fallbackCount,
            int deviceHandoffCount,
            int bufferBindingStepCount
    ) {
        return new GpuCoverageSummary.BackendCoverage(
                4,
                3,
                ratio,
                1,
                maxSelectedRegionLength,
                maxSelectedRegionLength,
                0,
                Map.of(),
                bufferBindingStepCount,
                tensorArrayStepCount,
                0,
                fallbackCount,
                cpuMaterializationCount,
                cpuMaterializationCount == 0 ? Map.of() : Map.of("CPU_CONSUMER", cpuMaterializationCount),
                4096L * cpuMaterializationCount,
                250_000L * cpuMaterializationCount,
                325_000L,
                deviceHandoffCount,
                Map.of("DEVICE_OWNED", 1),
                Map.of(),
                0,
                List.of(),
                List.of(),
                0,
                List.of(),
                List.of("BUFFER_BINDING_AVAILABLE"),
                List.of("using native buffer bindings")
        );
    }
}
