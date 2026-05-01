import backend.ComputeBackend;
import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import graph.execution.trace.ExecutionTrace;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.benchmark.report.BenchmarkReport;
import tuning.benchmark.report.BenchmarkSuiteReport;
import tuning.benchmark.report.GpuCoverageGap;
import tuning.benchmark.report.GpuCoverageGapCategory;
import tuning.benchmark.report.GpuCoverageGapTriage;
import tuning.measure.MeasurementPolicy;
import tuning.measure.MeasurementResult;
import tuning.measure.MeasurementStatistics;
import tuning.validate.ValidationResult;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GpuCoverageGapTriageTest {
    @Test
    void ranksRejectedCandidateAndMaterializationReasonsBySeverity() {
        BenchmarkReport report = BenchmarkReport.of(
                "transformer_block_hot_path",
                List.of(candidate("metal-triage", GpuCoverageSummaryTest.traceFor("GPU_METAL", ComputeBackend.GPU_METAL)))
        );

        List<GpuCoverageGap> gaps = GpuCoverageGapTriage.fromReport(report);

        assertFalse(gaps.isEmpty());
        assertEquals(GpuCoverageGapCategory.DEVICE_HANDOFF, gaps.getFirst().category());
        assertEquals(1400, gaps.getFirst().severityScore());
        assertTrue(gaps.stream().anyMatch(gap -> gap.category() == GpuCoverageGapCategory.CPU_MATERIALIZATION
                && gap.reason().equals("CPU_CONSUMER")
                && gap.severityScore() == 1000));
        assertTrue(gaps.stream().anyMatch(gap -> gap.category() == GpuCoverageGapCategory.REJECTED_CANDIDATE
                && gap.reason().equals("unsupported-layout")
                && gap.severityScore() == 500));
    }

    @Test
    void keepsTensorArrayFallbackSeparateFromNativeBufferCoverage() {
        GpuCoverageGapCategory[] categories = GpuCoverageGapCategory.values();
        assertTrue(List.of(categories).contains(GpuCoverageGapCategory.REJECTED_CANDIDATE));
        assertTrue(List.of(categories).contains(GpuCoverageGapCategory.CPU_MATERIALIZATION));
        assertTrue(List.of(categories).contains(GpuCoverageGapCategory.TENSOR_ARRAY_FALLBACK));
        assertTrue(List.of(categories).contains(GpuCoverageGapCategory.CPU_FALLBACK));
        assertTrue(List.of(categories).contains(GpuCoverageGapCategory.DEVICE_HANDOFF));
        assertTrue(List.of(categories).contains(GpuCoverageGapCategory.LOW_REGION_LENGTH));
        assertTrue(List.of(categories).contains(GpuCoverageGapCategory.LOW_GPU_COVERAGE));
        assertTrue(List.of(categories).contains(GpuCoverageGapCategory.STORAGE_RESIDENCY));

        tuning.benchmark.report.GpuCoverageSummary.BackendCoverage coverage =
                new tuning.benchmark.report.GpuCoverageSummary.BackendCoverage(
                        4,
                        2,
                        0.5d,
                        1,
                        2,
                        2.0d,
                        0,
                        Map.of(),
                        1,
                        1,
                        1,
                        2,
                        0,
                        Map.of(),
                        0L,
                        0L,
                        0L,
                        1,
                        Map.of("HOST_READABLE", 1),
                        List.of("TENSOR_ARRAY_BRIDGE"),
                        List.of("bridge")
                );
        assertEquals(1, coverage.tensorArrayStepCount());
        assertEquals(1, coverage.cpuFallbackStepCount());
        assertEquals(2, coverage.fallbackCount());
    }

    @Test
    void classifiesGapsIntoRequirementFamilies() {
        assertEquals("GPUDAG", GpuCoverageGapTriage.requirementFamilyFor(
                GpuCoverageGapCategory.REJECTED_CANDIDATE, "unsupported-layout"));
        assertEquals("GPUSTORAGE", GpuCoverageGapTriage.requirementFamilyFor(
                GpuCoverageGapCategory.CPU_MATERIALIZATION, "CPU_CONSUMER"));
        assertEquals("GPUNORM", GpuCoverageGapTriage.requirementFamilyFor(
                GpuCoverageGapCategory.REJECTED_CANDIDATE, "softmax reduction unsupported"));
        assertEquals("GPUFUSEX", GpuCoverageGapTriage.requirementFamilyFor(
                GpuCoverageGapCategory.REJECTED_CANDIDATE, "bias activation epilogue"));
        assertEquals("GPUMULTI", GpuCoverageGapTriage.requirementFamilyFor(
                GpuCoverageGapCategory.TENSOR_ARRAY_FALLBACK, "tensor-array fallback"));
        assertEquals("GPUHARDEN", GpuCoverageGapTriage.requirementFamilyFor(
                GpuCoverageGapCategory.DEVICE_HANDOFF, "device handoff"));
    }

    @Test
    void summarizesSuiteGapsByBackendAndWorkload() {
        BenchmarkReport metal = BenchmarkReport.of(
                "transformer_block_hot_path",
                List.of(candidate("metal-triage", GpuCoverageSummaryTest.traceFor("GPU_METAL", ComputeBackend.GPU_METAL)))
        );
        BenchmarkReport cuda = BenchmarkReport.of(
                "mlp_classifier_small",
                List.of(candidate("cuda-triage", GpuCoverageSummaryTest.traceFor("GPU_CUDA", ComputeBackend.GPU_CUDA)))
        );
        BenchmarkSuiteReport suite = new BenchmarkSuiteReport(null, List.of(metal, cuda));

        List<GpuCoverageGap> gaps = GpuCoverageGapTriage.fromSuite(suite);

        assertTrue(gaps.stream().anyMatch(gap -> gap.workloadName().equals("transformer_block_hot_path")
                && gap.backend().equals("GPU_METAL")));
        assertTrue(gaps.stream().anyMatch(gap -> gap.workloadName().equals("mlp_classifier_small")
                && gap.backend().equals("GPU_CUDA")));
        assertEquals(3, GpuCoverageGapTriage.topGaps(suite, 3).size());
    }

    static BenchmarkCandidateReport candidate(String name, ExecutionTrace trace) {
        ExecutionProfile profile = new ExecutionProfile(
                name + "-profile",
                name,
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        return BenchmarkCandidateReport.success(
                BenchmarkEntry.candidate(name, profile),
                ValidationResult.skipped(),
                new MeasurementResult(
                        MeasurementPolicy.defaults(),
                        trace,
                        new MeasurementStatistics(1.0d, 1.0d, 1.0d)
                )
        );
    }
}
