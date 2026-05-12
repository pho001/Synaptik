import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.candidate.Candidate;
import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.benchmark.report.BenchmarkReport;
import tuning.benchmark.report.BenchmarkSuiteReport;
import tuning.benchmark.report.BenchmarkSuiteReportDiff;
import tuning.benchmark.report.JsonBenchmarkSuiteReportDiffRenderer;
import tuning.autotune.report.JsonTuningResultDiffRenderer;
import tuning.benchmark.report.TextBenchmarkSuiteReportDiffRenderer;
import tuning.autotune.report.TextTuningResultDiffRenderer;
import tuning.autotune.report.TuningResultDiff;
import tuning.autotune.report.TuningSummary;
import tuning.autotune.TuningResult;

import java.time.OffsetDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class ReportingDiffRendererTest {
    @Test
    void benchmarkSuiteDiffRenderersProduceStructuredOutput() {
        BenchmarkSuiteReport previous = new BenchmarkSuiteReport(
                OffsetDateTime.parse("2026-04-04T10:00:00Z"),
                List.of(report("workload_a", "candidate", 2.0))
        );
        BenchmarkSuiteReport current = new BenchmarkSuiteReport(
                OffsetDateTime.parse("2026-04-04T11:00:00Z"),
                List.of(report("workload_a", "candidate", 1.0))
        );

        BenchmarkSuiteReportDiff diff = BenchmarkSuiteReportDiff.compare(previous, current);
        String text = TextBenchmarkSuiteReportDiffRenderer.render(diff);
        String json = JsonBenchmarkSuiteReportDiffRenderer.render(diff);

        assertTrue(text.contains("Benchmark Suite Diff"));
        assertTrue(text.contains("workload_a"));
        assertTrue(json.contains("\"currentBest\": \"candidate\""));
    }

    @Test
    void tuningResultDiffRenderersProduceStructuredOutput() {
        TuningResult previous = tuningResult("candidate_a", 2.0);
        TuningResult current = tuningResult("candidate_b", 1.0);

        TuningResultDiff diff = TuningResultDiff.compare(previous, current);
        String text = TextTuningResultDiffRenderer.render(diff);
        String json = JsonTuningResultDiffRenderer.render(diff);

        assertTrue(text.contains("Tuning Result Diff"));
        assertTrue(text.contains("currentBestProfile=candidate_b"));
        assertTrue(json.contains("\"bestSpeedupVsPrevious\": 2.000000"));
    }

    private static BenchmarkReport report(String workloadName, String candidateName, double medianMs) {
        Candidate candidate = new Candidate(candidateName, profile(candidateName));
        BenchmarkCandidateReport candidateReport = BenchmarkCandidateReport.success(
                candidate,
                tuning.validate.ValidationResult.skipped(),
                new tuning.measure.MeasurementResult(
                        tuning.measure.MeasurementPolicy.defaults(),
                        new graph.execution.trace.ExecutionTrace(null, null, graph.execution.trace.RunTrace.empty(ExecutionMode.FORWARD)),
                        new tuning.measure.MeasurementStatistics(medianMs, medianMs, medianMs)
                )
        );
        return new BenchmarkReport(workloadName, OffsetDateTime.now(), List.of(candidateReport), candidateName);
    }

    private static TuningResult tuningResult(String candidateName, double medianMs) {
        Candidate candidate = new Candidate(candidateName, profile(candidateName));
        BenchmarkCandidateReport finalist = BenchmarkCandidateReport.success(
                candidate,
                tuning.validate.ValidationResult.skipped(),
                new tuning.measure.MeasurementResult(
                        tuning.measure.MeasurementPolicy.defaults(),
                        new graph.execution.trace.ExecutionTrace(null, null, graph.execution.trace.RunTrace.empty(ExecutionMode.FORWARD)),
                        new tuning.measure.MeasurementStatistics(medianMs, medianMs, medianMs)
                )
        );
        return new TuningResult(
                candidate.profile(),
                List.of(finalist),
                "summary",
                new TuningSummary("search", 1, 1, 1, 1, 0, medianMs),
                false
        );
    }

    private static ExecutionProfile profile(String name) {
        return new ExecutionProfile(
                name,
                name,
                DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }
}
