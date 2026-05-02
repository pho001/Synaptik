import org.junit.jupiter.api.Test;
import tuning.benchmark.report.GpuCoverageGatePolicy;
import tuning.benchmark.report.GpuCoverageHotPathExpectation;
import tuning.benchmark.report.GpuCoverageRegressionGate;
import tuning.benchmark.report.GpuCoverageSummary;
import tuning.benchmark.report.GpuHotPathCoverageTargets;

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

    @Test
    void phaseTwentyFiveSupportedSdpaPolicyFailsHiddenCpuOrTensorArrayExit() {
        var policy = new GpuCoverageGatePolicy(
                "GPU_METAL",
                0.1d,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                true
        );
        var summary = summary("GPU_METAL", coverage(0.5d, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1));

        var result = GpuCoverageRegressionGate.evaluate(summary, policy);

        assertTrue(result.failures().contains("unexpected CPU fallback"));
        assertTrue(result.failures().contains("hidden tensor-array fallback"));
        assertTrue(result.failures().contains("lost native buffer binding"));
    }

    @Test
    void phaseTwentyEightSupportedTargetPoliciesRejectHiddenFallbackModes() {
        for (String workload : List.of(
                "reduction_chain_small",
                "layer_norm_small",
                "rms_norm_small",
                "transformer_block_hot_path",
                "mlp_classifier_small"
        )) {
            GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.defaultExpectations()
                    .stream()
                    .filter(item -> item.workloadName().equals(workload))
                    .findFirst()
                    .orElseThrow();

            var hiddenTensorArray = summary("GPU_METAL", coverage(0.75d, 3, 0, 1, 1, 1, 0, 1, 3, 1, 0));
            var tensorArrayResult = GpuCoverageRegressionGate.evaluate(hiddenTensorArray, expectation.policy());
            assertTrue(tensorArrayResult.failures().contains("hidden tensor-array fallback"), workload);
            assertTrue(tensorArrayResult.failures().contains("lost native buffer binding"), workload);

            var cpuFallback = summary("GPU_METAL", coverage(0.75d, 3, 0, 0, 1, 1, 1, 1, 3, 1, 1));
            var cpuFallbackResult = GpuCoverageRegressionGate.evaluate(cpuFallback, expectation.policy());
            assertTrue(cpuFallbackResult.failures().contains("unexpected CPU fallback"), workload);

            var cpuMaterialization = summary("GPU_METAL", coverage(0.75d, 3, 1, 0, 0, 1, 1, 1, 3, 1, 0));
            var materializationResult = GpuCoverageRegressionGate.evaluate(cpuMaterialization, expectation.policy());
            assertTrue(materializationResult.failures().contains("unexpected CPU materialization"), workload);
        }
    }

    @Test
    void phaseTwentyEightSupportedTargetPoliciesRejectLostPrimitiveOrRegionEvidence() {
        GpuCoverageHotPathExpectation reduction = GpuHotPathCoverageTargets.defaultExpectations()
                .stream()
                .filter(item -> item.workloadName().equals("reduction_chain_small"))
                .findFirst()
                .orElseThrow();
        var summary = summary("GPU_METAL", coverage(0.75d, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0));

        var result = GpuCoverageRegressionGate.evaluate(summary, reduction.policy());

        assertTrue(result.failures().contains("lost GPU coverage"));
        assertTrue(result.failures().contains("lost lowered primitive coverage"));
    }

    @Test
    void phaseTwentyEightMissingTargetCoverageNamesWorkloadAndBackend() {
        GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.defaultExpectations().getFirst();

        var results = GpuCoverageRegressionGate.evaluateTargets(
                new tuning.benchmark.report.BenchmarkSuiteReport(null, List.of()),
                List.of(expectation)
        );

        assertEquals(1, results.size());
        assertTrue(results.getFirst().failures().getFirst().contains(expectation.workloadName()));
        assertTrue(results.getFirst().failures().getFirst().contains(expectation.backend()));
    }

    @Test
    void phaseTwentyEightVisibleBlockerFailureNamesWorkloadAndBackend() {
        GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.defaultExpectations()
                .stream()
                .filter(item -> item.workloadName().equals("bool_compare_where_small"))
                .findFirst()
                .orElseThrow();
        var report = new tuning.benchmark.report.BenchmarkSuiteReport(
                null,
                List.of(tuning.benchmark.report.BenchmarkReport.of(
                        "bool_compare_where_small",
                        List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                                tuning.benchmark.BenchmarkEntry.candidate("candidate", profile()),
                                tuning.validate.ValidationResult.skipped(),
                                new tuning.measure.MeasurementResult(
                                        tuning.measure.MeasurementPolicy.defaults(),
                                        GpuCoverageSummaryTest.traceFor("GPU_METAL", backend.ComputeBackend.GPU_METAL),
                                        new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                                )
                        ))
                ))
        );

        var results = GpuCoverageRegressionGate.evaluateTargets(report, List.of(expectation));

        assertEquals(1, results.size());
        assertTrue(results.getFirst().failures().stream()
                .anyMatch(failure -> failure.contains("bool_compare_where_small") && failure.contains("GPU_METAL")));
    }

    private static config.profile.ExecutionProfile profile() {
        return new config.profile.ExecutionProfile(
                "phase28-gate-profile",
                "phase28-gate-profile",
                tensor.DataType.FLOAT32,
                backend.runtime.ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                config.profile.WorkloadProfile.transformerHotPathDefaults()
        );
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
                0,
                0L,
                Map.of(),
                Map.of(),
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
