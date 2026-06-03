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
                        Map.entry("nativeCpuLayoutClassCounts", Map.of("DENSE_CONTIGUOUS", 2)),
                        Map.entry("nativeCpuRegionSegmentKernelFamilies", List.of("PROVIDER", "SEGMENT_DENSE_SCALAR")),
                        Map.entry("nativeCpuRegionPhysicalKernels", List.of("OPENBLAS_NATIVE_SEGMENT", "SEGMENT_SCALAR"))
                )
        );

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> NativeCpuNonBlasBenchmarkGate.requirePass(report)
        );

        assertTrue(failure.getMessage().contains("AUTO native CPU region selected slow segment scalar kernels"));
        assertTrue(failure.getMessage().contains("auto-native-slow"));
        assertTrue(failure.getMessage().contains("SEGMENT_DENSE_SCALAR"));
        assertTrue(failure.getMessage().contains("DENSE_CONTIGUOUS"));
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
    void rejectsAutoNativeRegionWithNonEligibleNodeEvenWithoutScalarFamily() {
        BenchmarkReport report = report(
                "auto-native-non-eligible",
                CpuStorageProfile.AUTO,
                Map.ofEntries(
                        Map.entry("nativeCpuRegionDecision", "SELECTED"),
                        Map.entry("nativeCpuRegionRoute", "NATIVE"),
                        Map.entry("nativeCpuRegionProviderNodes", List.of(1)),
                        Map.entry("nativeCpuRegionLocalKernelNodes", List.of(2)),
                        Map.entry("nativeCpuRegionAutoEligible", List.of(true, false)),
                        Map.entry("nativeCpuLayoutClassCounts", Map.of("DENSE_CONTIGUOUS", 1, "OFFSET_CONTIGUOUS", 1)),
                        Map.entry("nativeCpuRegionResultResidencies", List.of(List.of("CPU_NATIVE"), List.of("CPU_NATIVE"))),
                        Map.entry("nativeCpuRegionPhysicalKernels", List.of("OPENBLAS_NATIVE_SEGMENT", "CUSTOM_NATIVE"))
                )
        );

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> NativeCpuNonBlasBenchmarkGate.requirePass(report)
        );

        assertTrue(failure.getMessage().contains("AUTO native CPU region selected non-auto-eligible nodes"));
        assertTrue(failure.getMessage().contains("auto-native-non-eligible"));
        assertTrue(failure.getMessage().contains("autoEligible=[true, false]"));
        assertTrue(failure.getMessage().contains("OFFSET_CONTIGUOUS"));
    }

    @Test
    void allowsAutoSlowNativeRegionOnlyWithMeasuredWinProof() {
        BenchmarkReport report = report(
                "auto-native-measured-win",
                CpuStorageProfile.AUTO,
                Map.ofEntries(
                        Map.entry("nativeCpuRegionDecision", "SELECTED"),
                        Map.entry("nativeCpuRegionRoute", "NATIVE"),
                        Map.entry("nativeCpuRegionProviderNodes", List.of(1)),
                        Map.entry("nativeCpuRegionLocalKernelNodes", List.of(2)),
                        Map.entry("nativeCpuRegionSegmentScalarNodes", List.of(2)),
                        Map.entry("nativeCpuRegionAutoEligible", List.of(true, false)),
                        Map.entry("nativeCpuRegionSegmentKernelFamilies", List.of("PROVIDER", "SEGMENT_DENSE_SCALAR")),
                        Map.entry("nativeCpuLayoutClassCounts", Map.of("DENSE_CONTIGUOUS", 2)),
                        Map.entry("nativeCpuRegionMeasuredWin", true),
                        Map.entry("nativeCpuRegionNativeMedianMs", 0.90d),
                        Map.entry("nativeCpuRegionArrayMedianMs", 1.00d),
                        Map.entry("nativeCpuRegionMeasuredWinThreshold", 0.95d)
                )
        );

        assertDoesNotThrow(() -> NativeCpuNonBlasBenchmarkGate.requirePass(report));
    }

    @Test
    void allowsAutoMeasuredWinProofWithStringValuesAndDefaultThreshold() {
        BenchmarkReport report = report(
                "auto-native-measured-win-strings",
                CpuStorageProfile.AUTO,
                Map.ofEntries(
                        Map.entry("nativeCpuRegionDecision", "SELECTED"),
                        Map.entry("nativeCpuRegionRoute", "NATIVE"),
                        Map.entry("nativeCpuRegionLocalKernelNodes", List.of(2)),
                        Map.entry("nativeCpuRegionSegmentScalarNodes", List.of(2)),
                        Map.entry("nativeCpuRegionSegmentKernelFamilies", List.of("SEGMENT_STRIDED_SCALAR")),
                        Map.entry("nativeCpuRegionMeasuredWin", "true"),
                        Map.entry("nativeCpuRegionNativeMedianMs", "0.94"),
                        Map.entry("nativeCpuRegionArrayMedianMs", "1.00")
                )
        );

        assertDoesNotThrow(() -> NativeCpuNonBlasBenchmarkGate.requirePass(report));
    }

    @Test
    void rejectsAutoSlowNativeRegionWhenMeasuredWinFlagLacksNumericProof() {
        BenchmarkReport report = report(
                "auto-native-stale-proof",
                CpuStorageProfile.AUTO,
                Map.ofEntries(
                        Map.entry("nativeCpuRegionDecision", "SELECTED"),
                        Map.entry("nativeCpuRegionRoute", "NATIVE"),
                        Map.entry("nativeCpuRegionProviderNodes", List.of(1)),
                        Map.entry("nativeCpuRegionLocalKernelNodes", List.of(2)),
                        Map.entry("nativeCpuRegionSegmentScalarNodes", List.of(2)),
                        Map.entry("nativeCpuRegionSegmentKernelFamilies", List.of("PROVIDER", "SEGMENT_DENSE_SCALAR")),
                        Map.entry("nativeCpuLayoutClassCounts", Map.of("DENSE_CONTIGUOUS", 2)),
                        Map.entry("nativeCpuRegionMeasuredWin", true),
                        Map.entry("nativeCpuRegionNativeMedianMs", 0.99d),
                        Map.entry("nativeCpuRegionArrayMedianMs", 1.00d),
                        Map.entry("nativeCpuRegionMeasuredWinThreshold", 0.95d)
                )
        );

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> NativeCpuNonBlasBenchmarkGate.requirePass(report)
        );

        assertTrue(failure.getMessage().contains("AUTO native CPU region selected slow segment scalar kernels"));
        assertTrue(failure.getMessage().contains("measuredWinProof={enabled=true"));
        assertTrue(failure.getMessage().contains("nativeMedianMs=0.99"));
        assertTrue(failure.getMessage().contains("arrayMedianMs=1.0"));
    }

    @Test
    void rejectsAutoSlowNativeRegionEvenWhenWholeWorkloadBeatsBaselineWithoutRegionProof() {
        BenchmarkReport report = reportWithBaseline(
                "auto-native-workload-win-without-region-proof",
                Map.ofEntries(
                        Map.entry("nativeCpuRegionDecision", "SELECTED"),
                        Map.entry("nativeCpuRegionRoute", "NATIVE"),
                        Map.entry("nativeCpuRegionProviderNodes", List.of(1)),
                        Map.entry("nativeCpuRegionLocalKernelNodes", List.of(2)),
                        Map.entry("nativeCpuRegionSegmentScalarNodes", List.of(2)),
                        Map.entry("nativeCpuRegionSegmentKernelFamilies", List.of("PROVIDER", "SEGMENT_DENSE_SCALAR")),
                        Map.entry("nativeCpuLayoutClassCounts", Map.of("DENSE_CONTIGUOUS", 2))
                ),
                1.00d,
                0.90d
        );

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> NativeCpuNonBlasBenchmarkGate.requirePass(report)
        );

        assertTrue(failure.getMessage().contains("AUTO native CPU region selected slow segment scalar kernels"));
        assertTrue(failure.getMessage().contains("measuredWinProof={enabled=[]"));
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
                        Map.entry("nativeCpuRegionAutoEligible", List.of(true)),
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

    private static BenchmarkReport reportWithBaseline(
            String name,
            Map<String, Object> candidateAttrs,
            double baselineMedianMs,
            double candidateMedianMs
    ) {
        ExecutionProfile baselineProfile = profile(name + "-baseline", CpuStorageProfile.CPU_ARRAY);
        ExecutionProfile candidateProfile = profile(name, CpuStorageProfile.AUTO);
        return BenchmarkReport.of(
                name,
                List.of(
                        BenchmarkCandidateReport.success(
                                BenchmarkEntry.baseline(name + "-baseline", baselineProfile),
                                ValidationResult.skipped(),
                                measurement(step(0, name + "-baseline-step", "CpuMatMulKernel", Map.of()), baselineMedianMs)
                        ),
                        BenchmarkCandidateReport.success(
                                BenchmarkEntry.candidate(name, candidateProfile),
                                ValidationResult.skipped(),
                                measurement(step(0, name + "-step", "CpuNativeStorageTrace", candidateAttrs), candidateMedianMs)
                        )
                )
        );
    }

    private static ExecutionProfile profile(String name, CpuStorageProfile cpuStorageProfile) {
        return new ExecutionProfile(
                name,
                name,
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                CompileConfig.inference(),
                RuntimeConfig.inferenceDefaults().withCpuStorageProfile(cpuStorageProfile),
                WorkloadProfile.none()
        );
    }

    private static ExecutionStepTrace step(String name, Map<String, Object> attrs) {
        return step(0, name + "-step", "CpuNativeStorageTrace", attrs);
    }

    private static ExecutionStepTrace step(int index, String name, String kernel, Map<String, Object> attrs) {
        return new ExecutionStepTrace(
                index,
                name,
                "MATMUL",
                List.of(2, 2),
                DataType.FLOAT32,
                "CPU",
                kernel,
                100L,
                new StepExecutionMetadata("node", attrs, null, null, null, null, null, null, null)
        );
    }

    private static MeasurementResult measurement(ExecutionStepTrace step, double medianMs) {
        return new MeasurementResult(
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
                new MeasurementStatistics(medianMs, medianMs, medianMs)
        );
    }

    private static BenchmarkReport report(String name, CpuStorageProfile cpuStorageProfile, Map<String, Object> attrs) {
        ExecutionProfile profile = profile(name, cpuStorageProfile);
        ExecutionStepTrace step = step(name, attrs);
        return BenchmarkReport.of(
                name,
                List.of(BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate(name, profile),
                        ValidationResult.skipped(),
                        measurement(step, 1.0d)
                ))
        );
    }
}
