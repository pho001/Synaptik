package tuning.benchmark.report;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Deterministic GPU coverage gap derivation from benchmark coverage summaries.
 */
public final class GpuCoverageGapTriage {
    private static final Comparator<GpuCoverageGap> GAP_ORDER = Comparator
            .comparingInt(GpuCoverageGap::severityScore).reversed()
            .thenComparing(GpuCoverageGap::workloadName)
            .thenComparing(GpuCoverageGap::backend)
            .thenComparing(gap -> gap.category().name())
            .thenComparing(GpuCoverageGap::reason);

    private GpuCoverageGapTriage() {
    }

    public static List<GpuCoverageGap> fromReport(BenchmarkReport report) {
        if (report == null) {
            return List.of();
        }
        List<GpuCoverageGap> gaps = new ArrayList<>();
        for (BenchmarkCandidateReport candidate : report.candidates()) {
            if (candidate == null || !candidate.success() || candidate.measurement() == null) {
                continue;
            }
            GpuCoverageSummary summary = GpuCoverageSummary.fromTrace(candidate.measurement().trace());
            for (var entry : summary.backends().entrySet()) {
                addCoverageGaps(
                        gaps,
                        report.workloadName(),
                        candidate.entry() == null ? "" : candidate.entry().name(),
                        entry.getKey(),
                        entry.getValue()
                );
            }
        }
        return gaps.stream().sorted(GAP_ORDER).toList();
    }

    public static List<GpuCoverageGap> fromSuite(BenchmarkSuiteReport report) {
        if (report == null) {
            return List.of();
        }
        List<GpuCoverageGap> gaps = new ArrayList<>();
        for (BenchmarkReport workloadReport : report.workloadReports()) {
            gaps.addAll(fromReport(workloadReport));
        }
        return gaps.stream().sorted(GAP_ORDER).toList();
    }

    public static List<GpuCoverageGap> topGaps(BenchmarkSuiteReport report, int limit) {
        if (limit <= 0) {
            return List.of();
        }
        return fromSuite(report).stream().limit(limit).toList();
    }

    public static String requirementFamilyFor(GpuCoverageGapCategory category, String reason) {
        String normalizedReason = reason == null ? "" : reason.toLowerCase(Locale.ROOT);
        if (containsAny(normalizedReason, "dtype", "storage", "residency", "bool", "int32", "bfloat",
                "materialization", "consumer")) {
            return "GPUSTORAGE";
        }
        if (containsAny(normalizedReason, "norm", "reduction", "softmax", "loss", "conv")) {
            return "GPUNORM";
        }
        if (containsAny(normalizedReason, "fuse", "activation", "bias", "elementwise", "epilogue")) {
            return "GPUFUSEX";
        }
        return switch (category == null ? GpuCoverageGapCategory.REJECTED_CANDIDATE : category) {
            case CPU_MATERIALIZATION, STORAGE_RESIDENCY -> "GPUSTORAGE";
            case TENSOR_ARRAY_FALLBACK, CPU_FALLBACK, LOW_REGION_LENGTH -> "GPUMULTI";
            case DEVICE_HANDOFF, LOW_GPU_COVERAGE -> "GPUHARDEN";
            case REJECTED_CANDIDATE -> "GPUDAG";
        };
    }

    private static void addCoverageGaps(
            List<GpuCoverageGap> gaps,
            String workloadName,
            String candidateName,
            String backend,
            GpuCoverageSummary.BackendCoverage coverage
    ) {
        for (Map.Entry<String, Integer> entry : coverage.rejectedCandidateReasonCounts().entrySet()) {
            addGap(gaps, workloadName, candidateName, backend, GpuCoverageGapCategory.REJECTED_CANDIDATE,
                    entry.getKey(), entry.getValue(), coverage);
        }
        for (Map.Entry<String, Integer> entry : coverage.cpuMaterializationReasonCounts().entrySet()) {
            addGap(gaps, workloadName, candidateName, backend, GpuCoverageGapCategory.CPU_MATERIALIZATION,
                    entry.getKey(), entry.getValue(), coverage);
        }
        if (coverage.tensorArrayStepCount() > 0) {
            addGap(gaps, workloadName, candidateName, backend, GpuCoverageGapCategory.TENSOR_ARRAY_FALLBACK,
                    "tensor-array fallback", coverage.tensorArrayStepCount(), coverage);
        }
        if (coverage.cpuFallbackStepCount() > 0) {
            addGap(gaps, workloadName, candidateName, backend, GpuCoverageGapCategory.CPU_FALLBACK,
                    "cpu fallback", coverage.cpuFallbackStepCount(), coverage);
        }
        if (coverage.deviceHandoffCount() > 0) {
            addGap(gaps, workloadName, candidateName, backend, GpuCoverageGapCategory.DEVICE_HANDOFF,
                    "device handoff", coverage.deviceHandoffCount(), coverage);
        }
        if (coverage.maxSelectedRegionLength() < 3) {
            addGap(gaps, workloadName, candidateName, backend, GpuCoverageGapCategory.LOW_REGION_LENGTH,
                    "short selected region", 1, coverage);
        }
        if (coverage.gpuCoverageRatio() < 0.5d) {
            addGap(gaps, workloadName, candidateName, backend, GpuCoverageGapCategory.LOW_GPU_COVERAGE,
                    "low gpu coverage ratio", 1, coverage);
        }
        for (Map.Entry<String, Integer> entry : coverage.storageResidencyCounts().entrySet()) {
            if (!"DEVICE_OWNED".equals(entry.getKey())) {
                addGap(gaps, workloadName, candidateName, backend, GpuCoverageGapCategory.STORAGE_RESIDENCY,
                        entry.getKey(), entry.getValue(), coverage);
            }
        }
    }

    private static void addGap(
            List<GpuCoverageGap> gaps,
            String workloadName,
            String candidateName,
            String backend,
            GpuCoverageGapCategory category,
            String reason,
            int count,
            GpuCoverageSummary.BackendCoverage coverage
    ) {
        gaps.add(new GpuCoverageGap(
                workloadName,
                candidateName,
                backend,
                category,
                reason,
                count,
                severityScore(category, count),
                coverage.maxSelectedRegionLength(),
                coverage.cpuMaterializationCount(),
                coverage.fallbackCount(),
                coverage.deviceHandoffCount(),
                requirementFamilyFor(category, reason)
        ));
    }

    private static int severityScore(GpuCoverageGapCategory category, int count) {
        return switch (category) {
            case CPU_MATERIALIZATION -> 1000 * count;
            case TENSOR_ARRAY_FALLBACK -> 900 * count;
            case CPU_FALLBACK -> 850 * count;
            case DEVICE_HANDOFF -> 700 * count;
            case REJECTED_CANDIDATE -> 500 * count;
            case LOW_REGION_LENGTH -> 300;
            case LOW_GPU_COVERAGE -> 250;
            case STORAGE_RESIDENCY -> 200 * count;
        };
    }

    private static boolean containsAny(String value, String... tokens) {
        for (String token : tokens) {
            if (value.contains(token)) {
                return true;
            }
        }
        return false;
    }
}
