package tuning.search;

import tuning.report.BenchmarkCandidateReport;

public interface CandidateScoreModel {
    double score(BenchmarkCandidateReport report);
}
