package tuning.search;

import tuning.report.BenchmarkCandidateReport;

public final class MedianSteadyStateScoreModel implements CandidateScoreModel {
    @Override
    public double score(BenchmarkCandidateReport report) {
        if (report == null || !report.success() || report.measurement() == null) {
            return Double.POSITIVE_INFINITY;
        }
        return report.measurement().steadyStateStats().medianMs();
    }
}
