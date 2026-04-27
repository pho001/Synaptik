package tuning.calibration;

import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.benchmark.report.BenchmarkSuiteCandidateSummary;
import tuning.benchmark.report.BenchmarkSuiteReport;

import java.util.ArrayList;
import java.util.List;

public interface PlatformCalibrationScorePolicy {
    PlatformCalibrationScore score(String candidateName, BenchmarkSuiteReport report);

    String metricName();

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
