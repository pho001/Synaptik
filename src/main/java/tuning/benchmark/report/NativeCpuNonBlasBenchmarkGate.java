package tuning.benchmark.report;

import config.runtime.CpuStorageProfile;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Fail-fast evaluator for native CPU non-BLAS performance parity evidence.
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
                if (segmentScalarNodes > 0) {
                    failures.add("AUTO native CPU region selected slow segment scalar kernels for "
                            + candidate.entry().name() + " count=" + segmentScalarNodes);
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
        if (!(kernels instanceof Collection<?> collection)) {
            return 0;
        }
        int count = 0;
        for (Object kernel : collection) {
            if ("SEGMENT_SCALAR".equals(String.valueOf(kernel))) {
                count++;
            }
        }
        return count;
    }

    private static int collectionSize(Object value) {
        return value instanceof Collection<?> collection ? collection.size() : 0;
    }
}
