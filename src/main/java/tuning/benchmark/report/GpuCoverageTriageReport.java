package tuning.benchmark.report;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Aggregated GPU coverage triage output for v1.3 planning and regression gates.
 *
 * @param hotPathTargets checked hot-path targets
 * @param topGaps highest-ranked coverage gaps
 * @param gapCountsByCategory top-gap counts by category
 * @param gapCountsByRequirementFamily top-gap counts by downstream family
 * @param downstreamPhaseTargets target families and workloads by downstream phase
 */
public record GpuCoverageTriageReport(
        List<GpuHotPathCoverageTarget> hotPathTargets,
        List<GpuCoverageGap> topGaps,
        Map<GpuCoverageGapCategory, Integer> gapCountsByCategory,
        Map<String, Integer> gapCountsByRequirementFamily,
        Map<Integer, List<String>> downstreamPhaseTargets
) {
    public GpuCoverageTriageReport {
        hotPathTargets = hotPathTargets == null ? List.of() : List.copyOf(hotPathTargets);
        topGaps = topGaps == null ? List.of() : List.copyOf(topGaps);
        gapCountsByCategory = copyMap(gapCountsByCategory);
        gapCountsByRequirementFamily = copyMap(gapCountsByRequirementFamily);
        downstreamPhaseTargets = copyListMap(downstreamPhaseTargets);
    }

    public static GpuCoverageTriageReport fromSuite(BenchmarkSuiteReport suiteReport) {
        return fromSuite(suiteReport, 10);
    }

    public static GpuCoverageTriageReport fromSuite(BenchmarkSuiteReport suiteReport, int topGapLimit) {
        List<GpuHotPathCoverageTarget> targets = GpuHotPathCoverageTargets.defaults();
        List<GpuCoverageGap> gaps = GpuCoverageGapTriage.topGaps(suiteReport, topGapLimit);
        LinkedHashMap<GpuCoverageGapCategory, Integer> countsByCategory = new LinkedHashMap<>();
        LinkedHashMap<String, Integer> countsByFamily = new LinkedHashMap<>();
        for (GpuCoverageGap gap : gaps) {
            countsByCategory.merge(gap.category(), 1, Integer::sum);
            countsByFamily.merge(gap.requirementFamily(), 1, Integer::sum);
        }
        return new GpuCoverageTriageReport(
                targets,
                gaps,
                countsByCategory,
                countsByFamily,
                downstreamTargets(targets)
        );
    }

    private static Map<Integer, List<String>> downstreamTargets(List<GpuHotPathCoverageTarget> targets) {
        LinkedHashMap<Integer, List<String>> out = new LinkedHashMap<>();
        out.put(15, new ArrayList<>(List.of("GPUDAG")));
        out.put(16, new ArrayList<>(List.of("GPUSTORAGE")));
        out.put(17, new ArrayList<>(List.of("GPUNORM")));
        out.put(18, new ArrayList<>(List.of("GPUFUSEX")));
        out.put(19, new ArrayList<>(List.of("GPUMULTI")));
        out.put(20, new ArrayList<>(List.of("GPUHARDEN")));
        for (GpuHotPathCoverageTarget target : targets) {
            for (String family : target.requirementFamilies()) {
                int phase = phaseForFamily(family);
                if (phase == 0) {
                    continue;
                }
                addUnique(out.computeIfAbsent(phase, ignored -> new ArrayList<>()),
                        family + ": " + target.workloadName());
            }
            if (target.ownerPhase() > 0) {
                addUnique(out.computeIfAbsent(target.ownerPhase(), ignored -> new ArrayList<>()),
                        "owner: " + target.workloadName());
            }
        }
        return out;
    }

    private static int phaseForFamily(String family) {
        return switch (family == null ? "" : family) {
            case "GPUDAG" -> 15;
            case "GPUSTORAGE" -> 16;
            case "GPUNORM" -> 17;
            case "GPUFUSEX" -> 18;
            case "GPUMULTI" -> 19;
            case "GPUHARDEN" -> 20;
            default -> 0;
        };
    }

    private static void addUnique(List<String> values, String value) {
        if (value != null && !value.isBlank() && !values.contains(value)) {
            values.add(value);
        }
    }

    private static <K, V> Map<K, V> copyMap(Map<K, V> source) {
        if (source == null || source.isEmpty()) {
            return Map.of();
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(source));
    }

    private static Map<Integer, List<String>> copyListMap(Map<Integer, List<String>> source) {
        if (source == null || source.isEmpty()) {
            return Map.of();
        }
        LinkedHashMap<Integer, List<String>> out = new LinkedHashMap<>();
        for (Map.Entry<Integer, List<String>> entry : source.entrySet()) {
            out.put(entry.getKey(), entry.getValue() == null ? List.of() : List.copyOf(entry.getValue()));
        }
        return Collections.unmodifiableMap(out);
    }
}
