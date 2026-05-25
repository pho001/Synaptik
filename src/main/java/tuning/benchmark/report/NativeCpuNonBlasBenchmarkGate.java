package tuning.benchmark.report;

import config.runtime.CpuStorageProfile;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Fail-fast evaluator for native CPU non-BLAS performance evidence.
 *
 * <p>{@code CPU_NATIVE} remains an explicit diagnostic/forced storage mode and may use
 * correctness-first segment scalar kernels. {@code AUTO} must not select those slow native
 * non-BLAS kernels without a future region-level benchmark proof encoded in the planner.</p>
 */
public final class NativeCpuNonBlasBenchmarkGate {
    private NativeCpuNonBlasBenchmarkGate() {
    }

    public static List<String> evaluate(BenchmarkReport report) {
        if (report == null) {
            return List.of("missing native CPU non-BLAS benchmark report");
        }
        ArrayList<String> failures = new ArrayList<>();
        for (BenchmarkCandidateReport candidate : report.candidates()) {
            if (candidate == null || !candidate.success() || candidate.measurement() == null
                    || candidate.entry() == null || candidate.entry().profile() == null
                    || candidate.entry().profile().runtime() == null
                    || candidate.measurement().trace() == null
                    || candidate.measurement().trace().run() == null) {
                continue;
            }
            CpuStorageProfile cpuStorageProfile = candidate.entry().profile().runtime().cpuStorageProfile();
            if (cpuStorageProfile != CpuStorageProfile.AUTO) {
                continue;
            }
            for (var step : candidate.measurement().trace().run().steps()) {
                if (step == null || step.metadata() == null || step.metadata().attributes() == null) {
                    continue;
                }
                Map<String, Object> attrs = step.metadata().attributes();
                if (!"SELECTED".equals(String.valueOf(attrs.getOrDefault("nativeCpuRegionDecision", "")))) {
                    continue;
                }
                int segmentScalarNodes = segmentScalarNodeCount(attrs);
                boolean measuredWinProof = NativeCpuRegionMeasuredWinEvidence.proven(attrs);
                if (segmentScalarNodes > 0 && !measuredWinProof) {
                    failures.add("AUTO native CPU region selected slow segment scalar kernels for "
                            + candidate.entry().name() + " count=" + segmentScalarNodes
                            + " kernels=" + evidence(attrs, "nativeCpuRegionSegmentKernelFamilies")
                            + " layouts=" + evidence(attrs, "nativeCpuLayoutClassCounts")
                            + " nodes=" + evidence(attrs, "nativeCpuRegionSegmentScalarNodes")
                            + " measuredWinProof=" + NativeCpuRegionMeasuredWinEvidence.describe(attrs));
                }
                int nonEligibleNodes = regionNonAutoEligibleNodeCount(attrs);
                if (nonEligibleNodes > 0 && !measuredWinProof) {
                    failures.add("AUTO native CPU region selected non-auto-eligible nodes for "
                            + candidate.entry().name() + " count=" + nonEligibleNodes
                            + " autoEligible=" + evidence(attrs, "nativeCpuRegionAutoEligible")
                            + " layouts=" + evidence(attrs, "nativeCpuLayoutClassCounts")
                            + " resultResidencies=" + evidence(attrs, "nativeCpuRegionResultResidencies")
                            + " measuredWinProof=" + NativeCpuRegionMeasuredWinEvidence.describe(attrs));
                }
            }
        }
        return List.copyOf(failures);
    }

    public static void requirePass(BenchmarkReport report) {
        List<String> failures = evaluate(report);
        if (!failures.isEmpty()) {
            throw new IllegalStateException(String.join("; ", failures));
        }
    }

    private static int segmentScalarNodeCount(Map<String, Object> attrs) {
        int explicit = collectionSize(attrs.get("nativeCpuRegionSegmentScalarNodes"));
        if (explicit > 0) {
            return explicit;
        }
        Object kernels = attrs.get("nativeCpuRegionPhysicalKernels");
        int count = 0;
        if (kernels instanceof Collection<?> collection) {
            for (Object kernel : collection) {
                String value = String.valueOf(kernel);
                if ("SEGMENT_SCALAR".equals(value)
                        || "SEGMENT_DENSE_SCALAR".equals(value)
                        || "SEGMENT_STRIDED_SCALAR".equals(value)) {
                    count++;
                }
            }
        }
        if (count > 0) {
            return count;
        }
        Object families = attrs.get("nativeCpuRegionSegmentKernelFamilies");
        if (families instanceof Collection<?> familyCollection) {
            for (Object family : familyCollection) {
                String value = String.valueOf(family);
                if ("SEGMENT_DENSE_SCALAR".equals(value) || "SEGMENT_STRIDED_SCALAR".equals(value)) {
                    count++;
                }
            }
        }
        return count;
    }

    private static int collectionSize(Object value) {
        return value instanceof Collection<?> collection ? collection.size() : 0;
    }

    private static int regionNonAutoEligibleNodeCount(Map<String, Object> attrs) {
        Object values = attrs.get("nativeCpuRegionAutoEligible");
        if (!(values instanceof Collection<?> collection)) {
            return 0;
        }
        int count = 0;
        for (Object value : collection) {
            if (Boolean.FALSE.equals(value) || "false".equalsIgnoreCase(String.valueOf(value))) {
                count++;
            }
        }
        return count;
    }

    private static String evidence(Map<String, Object> attrs, String key) {
        Object value = attrs == null ? null : attrs.get(key);
        if (value == null) {
            return "[]";
        }
        String text = String.valueOf(value);
        return text.isBlank() ? "[]" : text;
    }
}
