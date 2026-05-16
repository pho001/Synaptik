import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.execution.trace.CompileTrace;
import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.ExecutionTrace;
import graph.execution.trace.NativeCpuMemoryTrace;
import graph.execution.trace.PrepareTrace;
import graph.execution.trace.RunTrace;
import graph.execution.trace.StepExecutionMetadata;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.benchmark.report.BenchmarkReport;
import tuning.benchmark.report.NativeCpuNonBlasBenchmarkGate;
import tuning.measure.MeasurementPolicy;
import tuning.measure.MeasurementResult;
import tuning.measure.MeasurementStatistics;
import tuning.validate.ValidationResult;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeCpuNonBlasBenchmarkGateTest {
    @Test
    void rejectsAutoNativeRegionWithSegmentScalarLocalKernel() {
        BenchmarkReport report = report(
                "auto-native-slow",
                CpuStorageProfile.AUTO,
                Map.ofEntries(
                        Map.entry("nativeCpuRegionDecision", "SELECTED"),
                        Map.entry("nativeCpuRegionRoute", "NATIVE"),
                        Map.entry("nativeCpuRegionProviderNodes", List.of(1)),
                        Map.entry("nativeCpuRegionLocalKernelNodes", List.of(2)),
                        Map.entry("nativeCpuRegionSegmentScalarNodes", List.of(2)),
                        Map.entry("nativeCpuRegionPhysicalKernels", List.of("OPENBLAS_NATIVE_SEGMENT", "SEGMENT_SCALAR"))
                )
        );

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> NativeCpuNonBlasBenchmarkGate.requirePass(report)
        );

        assertTrue(failure.getMessage().contains("AUTO native CPU region selected slow segment scalar kernels"));
        assertTrue(failure.getMessage().contains("auto-native-slow"));
    }

    @Test
    void allowsExplicitCpuNativeDiagnosticRegionWithSegmentScalarLocalKernel() {
        BenchmarkReport report = report(
                "forced-native-slow",
                CpuStorageProfile.CPU_NATIVE,
                Map.ofEntries(
                        Map.entry("nativeCpuRegionDecision", "SELECTED"),
                        Map.entry("nativeCpuRegionRoute", "NATIVE"),
                        Map.entry("nativeCpuRegionProviderNodes", List.of(1)),
                        Map.entry("nativeCpuRegionLocalKernelNodes", List.of(2)),
                        Map.entry("nativeCpuRegionSegmentScalarNodes", List.of(2)),
                        Map.entry("nativeCpuRegionPhysicalKernels", List.of("OPENBLAS_NATIVE_SEGMENT", "SEGMENT_SCALAR"))
                )
        );

        assertDoesNotThrow(() -> NativeCpuNonBlasBenchmarkGate.requirePass(report));
    }

    @Test
    void allowsAutoProviderOnlyOrRejectedRegionEvidence() {
        BenchmarkReport providerOnly = report(
                "auto-native-provider-only",
                CpuStorageProfile.AUTO,
                Map.ofEntries(
                        Map.entry("nativeCpuRegionDecision", "SELECTED"),
                        Map.entry("nativeCpuRegionRoute", "NATIVE"),
                        Map.entry("nativeCpuRegionProviderNodes", List.of(1)),
                        Map.entry("nativeCpuRegionLocalKernelNodes", List.of()),
                        Map.entry("nativeCpuRegionPhysicalKernels", List.of("OPENBLAS_NATIVE_SEGMENT"))
                )
        );
        BenchmarkReport rejectedSlow = report(
                "auto-native-rejected-slow",
                CpuStorageProfile.AUTO,
                Map.ofEntries(
                        Map.entry("nativeCpuRegionDecision", "REJECTED"),
                        Map.entry("nativeCpuRegionRoute", "CPU_ARRAY"),
                        Map.entry("nativeCpuRegionReason", "native-cpu-region-auto-rejected-slow-op:relu"),
                        Map.entry("nativeCpuRegionFallbackReason", "native-cpu-region-auto-rejected-slow-op:relu"),
                        Map.entry("nativeCpuRegionPhysicalKernels", List.of("OPENBLAS_NATIVE_SEGMENT", "SEGMENT_SCALAR"))
                )
        );

        assertDoesNotThrow(() -> NativeCpuNonBlasBenchmarkGate.requirePass(providerOnly));
        assertDoesNotThrow(() -> NativeCpuNonBlasBenchmarkGate.requirePass(rejectedSlow));
    }

    private static BenchmarkReport report(String name, CpuStorageProfile cpuStorageProfile, Map<String, Object> attrs) {
        ExecutionProfile profile = new ExecutionProfile(
                name,
                name,
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                CompileConfig.inference(),
                RuntimeConfig.inferenceDefaults().withCpuStorageProfile(cpuStorageProfile),
                WorkloadProfile.none()
        );
        ExecutionStepTrace step = new ExecutionStepTrace(
                0,
                name + "-step",
                "MATMUL",
                List.of(2, 2),
                DataType.FLOAT32,
                "CPU",
                "PreparedNativeCpuRegionExecutable",
                100L,
                new StepExecutionMetadata("node", attrs, null, null, null, null, null, null, null)
        );
        return BenchmarkReport.of(
                name,
                List.of(BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate(name, profile),
                        ValidationResult.skipped(),
                        new MeasurementResult(
                                MeasurementPolicy.defaults(),
                                new ExecutionTrace(
                                        CompileTrace.skipped(),
                                        PrepareTrace.skipped(),
                                        new RunTrace(
                                                ExecutionMode.FORWARD,
                                                100L,
                                                List.of(step),
                                                List.of(),
                                                List.of(),
                                                NativeCpuMemoryTrace.empty(),
                                                List.of()
                                        )
                                ),
                                new MeasurementStatistics(1.0, 1.0, 1.0)
                        )
                ))
        );
    }
}
