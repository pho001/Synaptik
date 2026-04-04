import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tuning.candidate.Candidate;
import tuning.report.BenchmarkReport;
import tuning.report.BenchmarkBaselineKind;
import tuning.report.JsonBenchmarkReportRenderer;
import tuning.report.TextBenchmarkReportRenderer;
import tuning.session.BaselinePolicy;
import tuning.session.BenchmarkRequest;
import tuning.session.BenchmarkSession;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;
import tuning.workload.WorkloadMetadata;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
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
        assertEquals(3, report.candidates().size());
        assertTrue(report.bestCandidateName().equals("baseline")
                || report.bestCandidateName().equals("BASELINE_NO_OPT")
                || report.bestCandidateName().equals("BASELINE_NO_OPT_CONSERVATIVE_RUNTIME"));
        assertTrue(report.candidates().getFirst().success());
        assertTrue(report.candidates().getFirst().measurement().trace().compile().measured());
        assertTrue(report.candidates().getFirst().measurement().trace().prepare().measured());
        assertTrue(report.candidates().getFirst().measurement().trace().run().durationNs() >= 0L);
        assertTrue(report.candidates().getFirst().measurement().steadyStateStats().medianMs() >= 0.0d);
    }

    @Test
    void benchmarkSessionAddsBothBaselineVariantsAndReportsSpeedups() {
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "baseline_workload",
                WorkloadKind.GENERIC,
                environment -> Tensor.scalar(1.0).add(Tensor.scalar(2.0)).mul(Tensor.scalar(3.0))
        );

        Candidate candidate = new Candidate(
                "optimized",
                new ExecutionProfile(
                        "optimized-profile",
                        "optimized",
                        DataType.FLOAT64,
                        ExecutionMode.FORWARD,
                        config.optimizer.OptimizerConfig.inferenceDefaults(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );

        BenchmarkReport report = BenchmarkSession.create(new BenchmarkRequest(
                workload,
                List.of(candidate),
                new tuning.measure.MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                tuning.validate.ValidationPolicy.disabled(),
                tuning.report.ReportPolicy.defaults(),
                BaselinePolicy.defaults()
        )).run();

        assertTrue(report.baselineNoOpt().isPresent());
        assertTrue(report.baselineNoOptConservativeRuntime().isPresent());
        assertTrue(report.candidates().stream().anyMatch(r -> r.baselineKind() == BenchmarkBaselineKind.NO_OPT));
        assertTrue(report.candidates().stream().anyMatch(r -> r.baselineKind() == BenchmarkBaselineKind.NO_OPT_CONSERVATIVE_RUNTIME));
        assertTrue(Double.isFinite(report.speedupVsNoOpt(report.candidates().stream().filter(r -> r.baselineKind() == BenchmarkBaselineKind.NONE).findFirst().orElseThrow())));
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
        assertTrue(rendered.contains("bestCandidate="));
        assertTrue(rendered.contains("BASELINE_NO_OPT"));
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
        assertTrue(json.contains("\"bestCandidateName\":"));
        assertTrue(json.contains("\"baselineKind\": \"NO_OPT\""));
        assertTrue(json.contains("\"candidates\": ["));
        assertTrue(json.contains("\"timing\": {"));
        assertTrue(json.contains("\"speedup\": {"));
    }

    @Test
    void renderersHandleMissingBaselineMeasurementsAndNonFiniteSpeedups() {
        Candidate candidate = new Candidate(
                "candidate",
                new ExecutionProfile(
                        "candidate-profile",
                        "candidate",
                        DataType.FLOAT64,
                        ExecutionMode.FORWARD,
                        config.optimizer.OptimizerConfig.noOptimization(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );

        BenchmarkReport report = BenchmarkReport.of(
                "manual_report",
                List.of(
                        tuning.report.BenchmarkCandidateReport.failure(
                                new Candidate("BASELINE_NO_OPT", candidate.profile()),
                                tuning.validate.ValidationResult.failure("boom"),
                                "boom",
                                BenchmarkBaselineKind.NO_OPT
                        ),
                        tuning.report.BenchmarkCandidateReport.success(
                                candidate,
                                tuning.validate.ValidationResult.skipped(),
                                new tuning.measure.MeasurementResult(
                                        tuning.measure.MeasurementPolicy.defaults(),
                                        new graph.execution.trace.ExecutionTrace(
                                                new graph.execution.trace.CompileTrace(true, 1L, 0, 0, false),
                                                new graph.execution.trace.PrepareTrace(true, 1L, 0, 0),
                                                graph.execution.trace.RunTrace.empty(ExecutionMode.FORWARD)
                                        ),
                                        new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                                )
                        )
                )
        );

        assertDoesNotThrow(() -> TextBenchmarkReportRenderer.render(report));
        String json = assertDoesNotThrow(() -> JsonBenchmarkReportRenderer.render(report));
        assertTrue(json.contains("\"vsNoOpt\": null"));
    }
}
