package tuning.benchmark.report;

import java.util.ArrayList;
import java.util.List;

/**
 * Deterministic comparison between a coverage baseline and current coverage.
 *
 * <p>The comparison intentionally ignores raw latency and gates only on selected region length, CPU materialization
 * boundaries, fallback count, and device handoffs.</p>
 */
public record GpuCoverageComparison(
        String baselineName,
        String backend,
        boolean passes,
        int baselineMaxSelectedRegionLength,
        int currentMaxSelectedRegionLength,
        int baselineCpuMaterializationCount,
        int currentCpuMaterializationCount,
        int baselineFallbackCount,
        int currentFallbackCount,
        int baselineDeviceHandoffCount,
        int currentDeviceHandoffCount,
        List<String> improvements,
        List<String> regressions
) {
    public GpuCoverageComparison {
        baselineName = baselineName == null || baselineName.isBlank() ? "baseline" : baselineName;
        backend = backend == null ? "" : backend;
        improvements = improvements == null ? List.of() : List.copyOf(improvements);
        regressions = regressions == null ? List.of() : List.copyOf(regressions);
    }

    /**
     * Compares current coverage to a stable baseline.
     *
     * @param baseline baseline contract
     * @param current current backend coverage
     * @return comparison result
     */
    public static GpuCoverageComparison compare(
            GpuCoverageBaseline baseline,
            GpuCoverageSummary.BackendCoverage current
    ) {
        if (baseline == null) {
            baseline = new GpuCoverageBaseline("baseline", "", 0, 0, 0, 0);
        }
        int currentMaxSelectedRegionLength = current == null ? 0 : current.maxSelectedRegionLength();
        int currentCpuMaterializationCount = current == null ? Integer.MAX_VALUE : current.cpuMaterializationCount();
        int currentFallbackCount = current == null ? Integer.MAX_VALUE : current.fallbackCount();
        int currentDeviceHandoffCount = current == null ? Integer.MAX_VALUE : current.deviceHandoffCount();

        List<String> improvements = new ArrayList<>();
        List<String> regressions = new ArrayList<>();

        compareHigher(
                currentMaxSelectedRegionLength,
                baseline.maxSelectedRegionLength(),
                "longer selected region",
                "shorter selected region",
                improvements,
                regressions
        );
        compareLower(
                currentCpuMaterializationCount,
                baseline.cpuMaterializationCount(),
                "fewer CPU materializations",
                "more CPU materializations",
                improvements,
                regressions
        );
        compareLower(
                currentFallbackCount,
                baseline.fallbackCount(),
                "fewer fallbacks",
                "more fallbacks",
                improvements,
                regressions
        );
        compareLower(
                currentDeviceHandoffCount,
                baseline.deviceHandoffCount(),
                "fewer device handoffs",
                "more device handoffs",
                improvements,
                regressions
        );

        boolean passes = currentMaxSelectedRegionLength >= baseline.maxSelectedRegionLength()
                && currentCpuMaterializationCount <= baseline.cpuMaterializationCount()
                && currentFallbackCount <= baseline.fallbackCount()
                && currentDeviceHandoffCount <= baseline.deviceHandoffCount();

        return new GpuCoverageComparison(
                baseline.baselineName(),
                baseline.backend(),
                passes,
                baseline.maxSelectedRegionLength(),
                currentMaxSelectedRegionLength,
                baseline.cpuMaterializationCount(),
                currentCpuMaterializationCount,
                baseline.fallbackCount(),
                currentFallbackCount,
                baseline.deviceHandoffCount(),
                currentDeviceHandoffCount,
                improvements,
                regressions
        );
    }

    private static void compareHigher(
            int current,
            int baseline,
            String improvement,
            String regression,
            List<String> improvements,
            List<String> regressions
    ) {
        if (current > baseline) {
            improvements.add(improvement);
        } else if (current < baseline) {
            regressions.add(regression);
        }
    }

    private static void compareLower(
            int current,
            int baseline,
            String improvement,
            String regression,
            List<String> improvements,
            List<String> regressions
    ) {
        if (current < baseline) {
            improvements.add(improvement);
        } else if (current > baseline) {
            regressions.add(regression);
        }
    }
}
