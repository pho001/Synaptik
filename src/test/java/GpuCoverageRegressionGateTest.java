import org.junit.jupiter.api.Test;
import tuning.benchmark.report.GpuCoverageGatePolicy;
import tuning.benchmark.report.GpuCoverageRegressionGate;
import tuning.benchmark.report.GpuCoverageSummary;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
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

    @Test
    void phaseTwentyFailsWhenMultiOpRegionCoverageIsLost() {
        var summary = summary("GPU_METAL", coverage(0.75d, 3, 0, 0, 0, 1, 1, 0, 3, 1, 0));
        var policy = GpuCoverageGatePolicy.hotPathTarget("GPU_METAL", 0.5d, 3, 1, 3, 1);

        var result = GpuCoverageRegressionGate.evaluate(summary, policy);

        assertTrue(result.failures().contains("lost multi-op GPU region coverage"));
    }

    @Test
    void phaseTwentyFailsWhenLoweredPrimitiveCoverageIsLost() {
        var summary = summary("GPU_METAL", coverage(0.75d, 3, 0, 0, 0, 1, 1, 1, 2, 1, 0));
        var policy = GpuCoverageGatePolicy.hotPathTarget("GPU_METAL", 0.5d, 3, 1, 3, 1);

        var result = GpuCoverageRegressionGate.evaluate(summary, policy);

        assertTrue(result.failures().contains("lost lowered primitive coverage"));
    }

    @Test
    void phaseTwentyFailsWhenFusedSubpatternCoverageIsLost() {
        var summary = summary("GPU_METAL", coverage(0.75d, 3, 0, 0, 0, 1, 1, 1, 3, 0, 0));
        var policy = GpuCoverageGatePolicy.hotPathTarget("GPU_METAL", 0.5d, 3, 1, 3, 1);

        var result = GpuCoverageRegressionGate.evaluate(summary, policy);

        assertTrue(result.failures().contains("lost fused subpattern coverage"));
    }

    @Test
    void phaseTwentyFailureReasonsAreStableAndSpecific() {
        var summary = summary("GPU_METAL", coverage(0.25d, 1, 1, 1, 2, 3, 0, 0, 0, 0, 1));
        var policy = GpuCoverageGatePolicy.hotPathTarget("GPU_METAL", 0.5d, 3, 1, 1, 1);

        var result = GpuCoverageRegressionGate.evaluate(summary, policy);

        assertEquals(List.of(
                "lost GPU coverage",
                "lost GPU coverage",
                "lost multi-op GPU region coverage",
                "lost lowered primitive coverage",
                "lost fused subpattern coverage",
                "unexpected CPU materialization",
                "lost GPU coverage",
                "unexpected CPU fallback",
                "hidden tensor-array fallback",
                "lost native buffer binding",
                "unexpected device handoff"
        ), result.failures());

        var missing = GpuCoverageRegressionGate.evaluate(new GpuCoverageSummary(Map.of()), policy);
        assertEquals(List.of("missing coverage summary"), missing.failures());
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
        return coverage(
                ratio,
                maxSelectedRegionLength,
                cpuMaterializationCount,
                tensorArrayStepCount,
                fallbackCount,
                deviceHandoffCount,
                bufferBindingStepCount,
                1,
                maxSelectedRegionLength,
                0,
                0
        );
    }

    private static GpuCoverageSummary.BackendCoverage coverage(
            double ratio,
            int maxSelectedRegionLength,
            int cpuMaterializationCount,
            int tensorArrayStepCount,
            int fallbackCount,
            int deviceHandoffCount,
            int bufferBindingStepCount,
            int multiOpGpuRegionCount,
            int loweredPrimitiveCount,
            int gpuFusedSubpatternCount,
            int cpuFallbackStepCount
    ) {
        return new GpuCoverageSummary.BackendCoverage(
                4,
                3,
                ratio,
                1,
                multiOpGpuRegionCount,
                maxSelectedRegionLength,
                maxSelectedRegionLength,
                loweredPrimitiveCount,
                0,
                Map.of(),
                bufferBindingStepCount,
                tensorArrayStepCount,
                cpuFallbackStepCount,
                fallbackCount,
                cpuMaterializationCount,
                cpuMaterializationCount == 0 ? Map.of() : Map.of("CPU_CONSUMER", cpuMaterializationCount),
                4096L * cpuMaterializationCount,
                250_000L * cpuMaterializationCount,
                325_000L,
                deviceHandoffCount,
                Map.of("DEVICE_OWNED", 1),
                Map.of(),
                gpuFusedSubpatternCount,
                gpuFusedSubpatternCount == 0 ? List.of() : List.of("ELEMENTWISE_CHAIN"),
                gpuFusedSubpatternCount == 0 ? List.of() : List.of("[1, 2]"),
                gpuFusedSubpatternCount,
                gpuFusedSubpatternCount == 0 ? List.of() : List.of("SUPPORTED_PATTERN"),
                List.of("BUFFER_BINDING_AVAILABLE"),
                List.of("using native buffer bindings")
        );
    }
}
