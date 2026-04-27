package tuning.benchmark.report;

import tuning.benchmark.BenchmarkEntryRole;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public record BenchmarkReportDiff(
        String workloadName,
        OffsetDateTime previousCreatedAt,
        OffsetDateTime currentCreatedAt,
        String previousBestCandidate,
        String currentBestCandidate,
        double previousBestMedianMs,
        double currentBestMedianMs,
        double bestSpeedupVsPrevious,
        List<BenchmarkCandidateDiff> candidateDiffs
) {
    public BenchmarkReportDiff {
        workloadName = workloadName == null ? "" : workloadName;
        candidateDiffs = candidateDiffs == null ? List.of() : List.copyOf(candidateDiffs);
    }

    public static BenchmarkReportDiff compare(BenchmarkReport previous, BenchmarkReport current) {
        if (previous == null || current == null) {
            throw new IllegalArgumentException("previous and current reports cannot be null");
        }
        Map<String, BenchmarkCandidateReport> previousByName = new LinkedHashMap<>();
        for (BenchmarkCandidateReport candidate : previous.candidates()) {
            previousByName.put(candidate.entry().name(), candidate);
        }
        Map<String, BenchmarkCandidateReport> currentByName = new LinkedHashMap<>();
        for (BenchmarkCandidateReport candidate : current.candidates()) {
            currentByName.put(candidate.entry().name(), candidate);
        }

        java.util.LinkedHashSet<String> names = new java.util.LinkedHashSet<>();
        names.addAll(previousByName.keySet());
        names.addAll(currentByName.keySet());

        List<BenchmarkCandidateDiff> diffs = new ArrayList<>(names.size());
        for (String name : names) {
            BenchmarkCandidateReport before = previousByName.get(name);
            BenchmarkCandidateReport after = currentByName.get(name);
            double previousMedian = median(before);
            double currentMedian = median(after);
            diffs.add(new BenchmarkCandidateDiff(
                    name,
                    after != null ? after.entry().role() : before != null ? before.entry().role() : BenchmarkEntryRole.CANDIDATE,
                    before != null && before.success(),
                    after != null && after.success(),
                    previousMedian,
                    currentMedian,
                    Double.isFinite(previousMedian) && Double.isFinite(currentMedian) ? currentMedian - previousMedian : Double.NaN,
                    Double.isFinite(previousMedian) && Double.isFinite(currentMedian) && currentMedian > 0.0
                            ? previousMedian / currentMedian
                            : Double.NaN
            ));
        }

        double previousBest = previous.bestCandidate().map(c -> c.measurement().steadyStateStats().medianMs()).orElse(Double.NaN);
        double currentBest = current.bestCandidate().map(c -> c.measurement().steadyStateStats().medianMs()).orElse(Double.NaN);
        return new BenchmarkReportDiff(
                current.workloadName(),
                previous.createdAt(),
                current.createdAt(),
                previous.bestCandidateName(),
                current.bestCandidateName(),
                previousBest,
                currentBest,
                Double.isFinite(previousBest) && Double.isFinite(currentBest) && currentBest > 0.0
                        ? previousBest / currentBest
                        : Double.NaN,
                diffs
        );
    }

    private static double median(BenchmarkCandidateReport report) {
        return report != null && report.measurement() != null
                ? report.measurement().steadyStateStats().medianMs()
                : Double.NaN;
    }
}
