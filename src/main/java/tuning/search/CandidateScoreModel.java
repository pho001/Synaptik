package tuning.search;

import tuning.benchmark.report.BenchmarkCandidateReport;

public interface CandidateScoreModel {
    double score(BenchmarkCandidateReport report);
}
