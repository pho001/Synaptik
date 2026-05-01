import backend.ComputeBackend;
import backend.accelerator.lowering.GpuCompoundRegionSummary;
import backend.accelerator.lowering.GpuLoweredPrimitiveManifest;
import backend.accelerator.lowering.GpuLoweredRegionCandidateSpan;
import backend.accelerator.lowering.GpuLoweredRegionManifest;
import backend.accelerator.lowering.GpuLoweredRegionOriginalOp;
import backend.memory.CpuMaterializationReason;
import backend.memory.StorageResidency;
import backend.runtime.ExecutionMode;
import graph.execution.trace.BackendSelectionDecisionTrace;
import graph.execution.trace.BackendSelectionTrace;
import graph.execution.trace.CompileTrace;
import graph.execution.trace.CpuMaterializationTrace;
import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.ExecutionTrace;
import graph.execution.trace.PartitionCompileTrace;
import graph.execution.trace.PrepareTrace;
import graph.execution.trace.RunTrace;
import graph.execution.trace.StepExecutionMetadata;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.benchmark.report.GpuCoverageBaseline;
import tuning.benchmark.report.GpuCoverageComparison;
import tuning.benchmark.report.GpuCoverageSummary;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GpuCoverageSummaryTest {
    @Test
    void summarizesMetalCoverageFromSyntheticTrace() {
        GpuCoverageSummary summary = GpuCoverageSummary.fromTrace(traceFor("GPU_METAL", ComputeBackend.GPU_METAL));
        GpuCoverageSummary.BackendCoverage coverage = summary.backends().get("GPU_METAL");

        assertEquals(2, coverage.totalStepCount());
        assertEquals(1, coverage.acceleratorStepCount());
        assertEquals(0.5d, coverage.gpuCoverageRatio(), 1e-9);
        assertEquals(1, coverage.bufferBindingStepCount());
        assertEquals(0, coverage.tensorArrayStepCount());
        assertEquals(0, coverage.cpuFallbackStepCount());
        assertEquals(0, coverage.fallbackCount());
        assertEquals(Map.of("DEVICE_OWNED", 1), coverage.storageResidencyCounts());
        assertTrue(coverage.reasonCodes().contains("BUFFER_BINDING_AVAILABLE"));
    }

    @Test
    void summarizesCudaCoverageFromSyntheticTrace() {
        GpuCoverageSummary summary = GpuCoverageSummary.fromTrace(traceFor("GPU_CUDA", ComputeBackend.GPU_CUDA));
        GpuCoverageSummary.BackendCoverage coverage = summary.backends().get("GPU_CUDA");

        assertEquals(2, coverage.totalStepCount());
        assertEquals(1, coverage.acceleratorStepCount());
        assertEquals(0.5d, coverage.gpuCoverageRatio(), 1e-9);
        assertEquals(1, coverage.bufferBindingStepCount());
        assertEquals(325_000L, coverage.copyDurationNs());
        assertTrue(coverage.fallbackReasons().contains("using native buffer bindings"));
    }

    @Test
    void countsRejectedCandidateReasonsAndSelectedRegionLengths() {
        GpuCoverageSummary summary = GpuCoverageSummary.fromTrace(traceFor("GPU_METAL", ComputeBackend.GPU_METAL));
        GpuCoverageSummary.BackendCoverage coverage = summary.backends().get("GPU_METAL");

        assertEquals(1, coverage.selectedRegionCount());
        assertEquals(3, coverage.maxSelectedRegionLength());
        assertEquals(3.0d, coverage.averageSelectedRegionLength(), 1e-9);
        assertEquals(1, coverage.rejectedCandidateCount());
        assertEquals(Map.of("unsupported-layout", 1), coverage.rejectedCandidateReasonCounts());
    }

    @Test
    void coverageSummaryIgnoresManifestWhenCountingSelectedRegions() {
        GpuLoweredRegionManifest manifest = new GpuLoweredRegionManifest(
                "gpu-metal-region-10",
                ComputeBackend.GPU_METAL,
                10,
                List.of(10, 11, 12),
                List.of(1),
                List.of(12),
                99,
                List.of(new GpuLoweredRegionOriginalOp(
                        10,
                        "LOG_SOFTMAX",
                        List.of(1),
                        List.of(12),
                        DataType.FLOAT32,
                        List.of(2, 3),
                        List.of("p0"),
                        List.of()
                )),
                List.of(new GpuLoweredPrimitiveManifest(
                        "p0",
                        "SOFTMAX",
                        List.of(10),
                        List.of("external:0"),
                        "node:0",
                        DataType.FLOAT32,
                        List.of(2, 3),
                        List.of()
                )),
                List.of(),
                List.of(),
                GpuCompoundRegionSummary.none(ComputeBackend.GPU_METAL, List.of(10, 11, 12)),
                List.of(),
                GpuLoweredRegionCandidateSpan.none(List.of(10, 11, 12)),
                Map.of()
        );
        ExecutionTrace trace = traceWithManifest("GPU_METAL", ComputeBackend.GPU_METAL, manifest);

        GpuCoverageSummary.BackendCoverage coverage = GpuCoverageSummary.fromTrace(trace).backends().get("GPU_METAL");

        assertEquals(1, coverage.selectedRegionCount());
        assertEquals(3, coverage.maxSelectedRegionLength());
        assertEquals(3.0d, coverage.averageSelectedRegionLength(), 1e-9);
    }

    @Test
    void countsCpuMaterializationReasonsAndDeviceHandoffs() {
        GpuCoverageSummary summary = GpuCoverageSummary.fromTrace(traceFor("GPU_METAL", ComputeBackend.GPU_METAL));
        GpuCoverageSummary.BackendCoverage coverage = summary.backends().get("GPU_METAL");

        assertEquals(1, coverage.cpuMaterializationCount());
        assertEquals(Map.of("CPU_CONSUMER", 1), coverage.cpuMaterializationReasonCounts());
        assertEquals(4096L, coverage.cpuMaterializationBytes());
        assertEquals(250_000L, coverage.cpuMaterializationDurationNs());
        assertEquals(2, coverage.deviceHandoffCount());
    }

    @Test
    void comparesCoverageAgainstBaselineWithoutTimingThresholds() {
        GpuCoverageBaseline baseline = new GpuCoverageBaseline("v1.1", "GPU_METAL", 1, 2, 1, 2);
        GpuCoverageSummary.BackendCoverage current = new GpuCoverageSummary.BackendCoverage(
                4,
                3,
                0.75d,
                1,
                3,
                3.0d,
                0,
                Map.of(),
                3,
                0,
                0,
                0,
                1,
                Map.of("CPU_CONSUMER", 1),
                4096L,
                250_000L,
                325_000L,
                1,
                Map.of("DEVICE_OWNED", 3),
                List.of("BUFFER_BINDING_AVAILABLE"),
                List.of("using native buffer bindings")
        );

        GpuCoverageComparison comparison = GpuCoverageComparison.compare(baseline, current);

        assertTrue(comparison.passes());
        assertEquals("v1.1", comparison.baselineName());
        assertEquals("GPU_METAL", comparison.backend());
        assertTrue(comparison.improvements().contains("longer selected region"));
        assertTrue(comparison.improvements().contains("fewer CPU materializations"));
        assertTrue(comparison.improvements().contains("fewer fallbacks"));
        assertTrue(comparison.improvements().contains("fewer device handoffs"));
        assertTrue(comparison.regressions().isEmpty());
    }

    static ExecutionTrace traceFor(String backendName, ComputeBackend backend) {
        ExecutionStepTrace gpuStep = new ExecutionStepTrace(
                0,
                backendName.toLowerCase() + "_linear",
                "LINEAR",
                List.of(16, 16),
                DataType.FLOAT32,
                backendName,
                "PreparedAcceleratorExecutable",
                2_000_000L,
                new StepExecutionMetadata(
                        "node",
                        Map.ofEntries(
                                Map.entry("acceleratorBufferBackend", backendName),
                                Map.entry("acceleratorBufferExecutionPath", "BUFFER_BINDING"),
                                Map.entry("acceleratorBufferReasonCode", "BUFFER_BINDING_AVAILABLE"),
                                Map.entry("acceleratorBufferReason", "using native buffer bindings"),
                                Map.entry("acceleratorJavaToNativeCopyNs", 100_000L),
                                Map.entry("acceleratorNativeToJavaCopyNs", 200_000L),
                                Map.entry("acceleratorNativeDeviceCopyNs", 25_000L),
                                Map.entry("storageResidency", "DEVICE_OWNED")
                        ),
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null
                )
        );
        ExecutionStepTrace cpuStep = new ExecutionStepTrace(
                1,
                "cpu_consumer",
                "ADD",
                List.of(16, 16),
                DataType.FLOAT32,
                "CPU",
                "CpuElementWiseKernel",
                1_000_000L,
                StepExecutionMetadata.none()
        );
        BackendSelectionTrace selection = new BackendSelectionTrace(
                2,
                1,
                1,
                List.of(
                        new BackendSelectionDecisionTrace(
                                10,
                                List.of(10, 11, 12),
                                List.of(backend),
                                true,
                                backend,
                                "selected",
                                4096L
                        ),
                        new BackendSelectionDecisionTrace(
                                20,
                                List.of(20),
                                List.of(backend),
                                false,
                                null,
                                "unsupported-layout",
                                1024L
                        )
                )
        );
        CpuMaterializationTrace materialization = new CpuMaterializationTrace(
                12,
                CpuMaterializationReason.CPU_CONSUMER,
                backendName,
                StorageResidency.DEVICE_OWNED,
                4096L,
                250_000L,
                true,
                "CPU consumer requested readable storage"
        );
        return new ExecutionTrace(
                new CompileTrace(true, 1L, 0, 0, false, PartitionCompileTrace.empty()),
                new PrepareTrace(true, 1L, 0, 0, selection),
                new RunTrace(ExecutionMode.FORWARD, 3_000_000L, List.of(gpuStep, cpuStep), List.of(materialization))
        );
    }

    private static ExecutionTrace traceWithManifest(
            String backendName,
            ComputeBackend backend,
            GpuLoweredRegionManifest manifest
    ) {
        ExecutionTrace trace = traceFor(backendName, backend);
        BackendSelectionTrace selection = new BackendSelectionTrace(
                1,
                1,
                0,
                List.of(new BackendSelectionDecisionTrace(
                        10,
                        List.of(10, 11, 12),
                        List.of(backend),
                        true,
                        backend,
                        "selected",
                        4096L,
                        null,
                        List.of(),
                        manifest
                ))
        );
        return new ExecutionTrace(
                trace.compile(),
                new PrepareTrace(true, 1L, 0, 0, selection),
                trace.run()
        );
    }
}
