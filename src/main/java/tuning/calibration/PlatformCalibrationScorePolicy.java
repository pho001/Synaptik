package tuning.calibration;

import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.benchmark.report.BenchmarkSuiteCandidateSummary;
import tuning.benchmark.report.BenchmarkSuiteReport;

import java.util.ArrayList;
import java.util.List;

/**
 * Scores runtime-profile candidates from a calibration benchmark suite.
 *
 * <p>Lower scores are better. Implementations must be deterministic for a given
 * report because the calibration session uses the minimum valid score to select
 * the runtime profile promoted to the next step. Policies should return
 * {@link PlatformCalibrationScore#invalid(String)} instead of throwing for
 * candidate-local failures so other candidates can still be considered.</p>
 */
public interface PlatformCalibrationScorePolicy {
    /**
     * Scores one candidate across the supplied suite report.
     *
     * @param candidateName candidate id as it appears in benchmark reports
     * @param report suite report for the current calibration step
     * @return valid or invalid score; lower valid scores win
     */
    PlatformCalibrationScore score(String candidateName, BenchmarkSuiteReport report);

    /**
     * @return stable metric name for reports
     */
    String metricName();

    /**
     * Scores by arithmetic mean of successful median latencies across workloads.
     *
     * @return score policy where lower average median milliseconds win
     */
    static PlatformCalibrationScorePolicy averageMedianMs() {
        return new PlatformCalibrationScorePolicy() {
            @Override
            public PlatformCalibrationScore score(String candidateName, BenchmarkSuiteReport report) {
                if (candidateName == null || candidateName.isBlank() || report == null) {
                    return PlatformCalibrationScore.invalid("missing candidate or report");
                }
                List<BenchmarkCandidateReport> reports = report.candidateReports(candidateName);
                if (reports.isEmpty()) {
                    return PlatformCalibrationScore.invalid("candidate missing from suite report");
                }
                List<Double> medians = new ArrayList<>();
                for (BenchmarkCandidateReport candidateReport : reports) {
                    if (!candidateReport.success() || candidateReport.measurement() == null) {
                        return PlatformCalibrationScore.invalid("candidate failed at least one workload");
                    }
                    medians.add(candidateReport.measurement().steadyStateStats().medianMs());
                }
                if (medians.isEmpty()) {
                    return PlatformCalibrationScore.invalid("candidate produced no measurements");
                }
                double sum = 0.0d;
                double worst = Double.NEGATIVE_INFINITY;
                for (double median : medians) {
                    sum += median;
                    worst = Math.max(worst, median);
                }
                double avg = sum / medians.size();
                return new PlatformCalibrationScore(
                        true,
                        avg,
                        avg,
                        worst,
                        0.0d,
                        "averageMedianMs"
                );
            }

            @Override
            public String metricName() {
                return "averageMedianMs";
            }
        };
    }

    /**
     * Scores by geometric mean plus a penalty for the worst workload bucket.
     *
     * @param alpha multiplier applied to the worst-bucket median latency
     * @return score policy where lower penalized scores win
     */
    static PlatformCalibrationScorePolicy weightedGeometricMeanWithWorstBucketPenalty(double alpha) {
        return new PlatformCalibrationScorePolicy() {
            @Override
            public PlatformCalibrationScore score(String candidateName, BenchmarkSuiteReport report) {
                if (candidateName == null || candidateName.isBlank() || report == null) {
                    return PlatformCalibrationScore.invalid("missing candidate or report");
                }
                List<BenchmarkCandidateReport> reports = report.candidateReports(candidateName);
                if (reports.isEmpty()) {
                    return PlatformCalibrationScore.invalid("candidate missing from suite report");
                }
                double worst = Double.NEGATIVE_INFINITY;
                double logSum = 0.0d;
                int count = 0;
                for (BenchmarkCandidateReport candidateReport : reports) {
                    if (!candidateReport.success() || candidateReport.measurement() == null) {
                        return PlatformCalibrationScore.invalid("candidate failed at least one workload");
                    }
                    double median = candidateReport.measurement().steadyStateStats().medianMs();
                    if (!(median > 0.0d) || !Double.isFinite(median)) {
                        return PlatformCalibrationScore.invalid("invalid median measurement");
                    }
                    worst = Math.max(worst, median);
                    logSum += Math.log(median);
                    count++;
                }
                if (count == 0) {
                    return PlatformCalibrationScore.invalid("candidate produced no measurements");
                }
                double geom = Math.exp(logSum / count);
                return new PlatformCalibrationScore(
                        true,
                        geom + alpha * worst,
                        geom,
                        worst,
                        0.0d,
                        "weightedGeometricMeanWithWorstBucketPenalty(alpha=" + alpha + ")"
                );
            }

            @Override
            public String metricName() {
                return "weightedGeometricMeanWithWorstBucketPenalty";
            }
        };
    }
}
