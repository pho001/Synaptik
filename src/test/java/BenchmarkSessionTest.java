import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tuning.candidate.Candidate;
import tuning.report.BenchmarkReport;
import tuning.report.JsonBenchmarkReportRenderer;
import tuning.report.TextBenchmarkReportRenderer;
import tuning.session.BenchmarkRequest;
import tuning.session.BenchmarkSession;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;
import tuning.workload.WorkloadMetadata;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class BenchmarkSessionTest {
    @Test
    void benchmarkSessionMeasuresSimpleTensorWorkload() {
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "simple_add_mul",
                WorkloadKind.GENERIC,
                environment -> {
                    Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "a", DataType.FLOAT64);
                    Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{4}, null, "b", DataType.FLOAT64);
                    return a.add(b).mul(a);
                },
                environment -> tuning.validate.ValidationReference.none(),
                environment -> new WorkloadMetadata("simple_add_mul", WorkloadKind.GENERIC, Map.of("size", 4))
        );

        ExecutionProfile profile = new ExecutionProfile(
                "bench-default",
                "bench-default",
                DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        Candidate candidate = new Candidate("baseline", profile);

        BenchmarkRequest request = new BenchmarkRequest(
                workload,
                List.of(candidate),
                new tuning.measure.MeasurementPolicy(1, 2, 1, true, true, true, true, false),
                tuning.validate.ValidationPolicy.disabled(),
                tuning.report.ReportPolicy.defaults()
        );

        BenchmarkReport report = BenchmarkSession.create(request).run();

        assertEquals("simple_add_mul", report.workloadName());
        assertEquals(1, report.candidates().size());
        assertEquals("baseline", report.bestCandidateName());
        assertTrue(report.candidates().getFirst().success());
        assertTrue(report.candidates().getFirst().measurement().trace().compile().measured());
        assertTrue(report.candidates().getFirst().measurement().trace().prepare().measured());
        assertTrue(report.candidates().getFirst().measurement().trace().run().durationNs() >= 0L);
        assertTrue(report.candidates().getFirst().measurement().steadyStateStats().medianMs() >= 0.0d);
    }

    @Test
    void textRendererProducesReadableSummary() {
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "renderer_workload",
                WorkloadKind.GENERIC,
                environment -> Tensor.scalar(1.0).add(Tensor.scalar(2.0))
        );

        Candidate candidate = new Candidate(
                "renderer-candidate",
                new ExecutionProfile(
                        "renderer-profile",
                        "renderer-candidate",
                        DataType.FLOAT64,
                        ExecutionMode.FORWARD,
                        config.optimizer.OptimizerConfig.noOptimization(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );

        BenchmarkReport report = BenchmarkSession.create(new BenchmarkRequest(
                workload,
                List.of(candidate),
                new tuning.measure.MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                tuning.validate.ValidationPolicy.disabled(),
                tuning.report.ReportPolicy.defaults()
        )).run();

        String rendered = TextBenchmarkReportRenderer.render(report);
        assertTrue(rendered.contains("Benchmark Report"));
        assertTrue(rendered.contains("Summary"));
        assertTrue(rendered.contains("Candidates"));
        assertTrue(rendered.contains("workload=renderer_workload"));
        assertTrue(rendered.contains("bestCandidate=renderer-candidate"));
        assertTrue(rendered.contains("steadyStateMedianMs="));
        assertFalse(rendered.isBlank());
    }

    @Test
    void jsonRendererProducesStructuredOutput() {
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "json_workload",
                WorkloadKind.GENERIC,
                environment -> Tensor.scalar(1.0).add(Tensor.scalar(2.0))
        );

        Candidate candidate = new Candidate(
                "json-candidate",
                new ExecutionProfile(
                        "json-profile",
                        "json-candidate",
                        DataType.FLOAT64,
                        ExecutionMode.FORWARD,
                        config.optimizer.OptimizerConfig.noOptimization(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );

        BenchmarkReport report = BenchmarkSession.create(new BenchmarkRequest(
                workload,
                List.of(candidate),
                new tuning.measure.MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                tuning.validate.ValidationPolicy.disabled(),
                tuning.report.ReportPolicy.defaults()
        )).run();

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"workloadName\": \"json_workload\""));
        assertTrue(json.contains("\"bestCandidateName\": \"json-candidate\""));
        assertTrue(json.contains("\"candidates\": ["));
        assertTrue(json.contains("\"timing\": {"));
    }
}
