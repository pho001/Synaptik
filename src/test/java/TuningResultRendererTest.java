import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.autotune.report.JsonTuningResultRenderer;
import tuning.autotune.report.TextTuningResultRenderer;
import tuning.autotune.report.TuningSummary;
import tuning.autotune.TuningResult;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class TuningResultRendererTest {
    @Test
    void textAndJsonRenderersProduceStructuredOutput() {
        var candidate = new tuning.candidate.Candidate("cand", profile("cand"));
        var finalist = BenchmarkCandidateReport.success(
                candidate,
                tuning.validate.ValidationResult.skipped(),
                new tuning.measure.MeasurementResult(
                        tuning.measure.MeasurementPolicy.defaults(),
                        new graph.execution.trace.ExecutionTrace(null, null, graph.execution.trace.RunTrace.empty(ExecutionMode.FORWARD)),
                        new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                )
        );
        var result = new TuningResult(
                candidate.profile(),
                List.of(finalist),
                "ok",
                new TuningSummary("BranchAndBoundSearchStrategy", 4, 4, 3, 1, 4, 1.0),
                true
        );

        String text = TextTuningResultRenderer.render(result);
        String json = JsonTuningResultRenderer.render(result);

        assertTrue(text.contains("Tuning Result"));
        assertTrue(text.contains("strategy=BranchAndBoundSearchStrategy"));
        assertTrue(text.contains("Finalists"));
        assertTrue(json.contains("\"strategy\": \"BranchAndBoundSearchStrategy\""));
        assertTrue(json.contains("\"bestProfile\": \"cand\""));
    }

    private static ExecutionProfile profile(String name) {
        return new ExecutionProfile(
                name,
                name,
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }
}
