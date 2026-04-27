import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tuning.measure.MeasurementPolicy;
import tuning.benchmark.report.BenchmarkSuiteReport;
import tuning.benchmark.report.JsonBenchmarkSuiteReportRenderer;
import tuning.benchmark.report.TextBenchmarkSuiteReportRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkSuiteRequest;
import tuning.benchmark.BenchmarkSuiteSession;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadCatalog;
import tuning.workload.WorkloadKind;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class BenchmarkSuiteSessionTest {
    @Test
    void benchmarkSuiteRunsSelectedWorkloadsFromCatalog() {
        WorkloadCatalog catalog = new WorkloadCatalog()
                .register(new TensorRootWorkloadSpec(
                        "add_only",
                        WorkloadKind.GENERIC,
                        environment -> Tensor.scalar(1.0).add(Tensor.scalar(2.0))
                ))
                .register(new TensorRootWorkloadSpec(
                        "mul_only",
                        WorkloadKind.GENERIC,
                        environment -> Tensor.scalar(3.0).mul(Tensor.scalar(4.0))
                ));

        BenchmarkEntry baseline = BenchmarkEntry.baseline(
                "suite-baseline",
                new ExecutionProfile(
                        "suite-profile",
                        "suite-baseline",
                        DataType.FLOAT64,
                        ExecutionMode.FORWARD,
                        config.optimizer.OptimizerConfig.noOptimization(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );

        BenchmarkSuiteRequest request = catalog.benchmarkSuiteRequest(
                List.of("add_only", "mul_only"),
                List.of(baseline),
                new MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                tuning.validate.ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        BenchmarkSuiteReport report = BenchmarkSuiteSession.create(request).run();

        assertEquals(2, report.workloadReports().size());
        assertEquals("add_only", report.workloadReports().get(0).workloadName());
        assertEquals("mul_only", report.workloadReports().get(1).workloadName());
        assertEquals(2, report.totalCandidateCount());
        assertFalse(report.overallBestCandidate().isPresent());
        assertFalse(report.candidateSummaries().isEmpty());
        assertFalse(report.hotspots(5).isEmpty());
        assertTrue(report.workloadReports().get(0).bestCandidateName().isBlank());
        assertTrue(report.workloadReports().get(1).bestCandidateName().isBlank());
    }

    @Test
    void suiteRendererPrintsPerWorkloadSections() {
        WorkloadCatalog catalog = new WorkloadCatalog()
                .register(new TensorRootWorkloadSpec(
                        "single_case",
                        WorkloadKind.GENERIC,
                        environment -> Tensor.scalar(2.0).mul(Tensor.scalar(5.0))
                ));

        BenchmarkEntry candidate = BenchmarkEntry.candidate(
                "render-suite",
                new ExecutionProfile(
                        "render-suite-profile",
                        "render-suite",
                        DataType.FLOAT64,
                        ExecutionMode.FORWARD,
                        config.optimizer.OptimizerConfig.noOptimization(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );

        BenchmarkSuiteReport report = BenchmarkSuiteSession.create(
                catalog.benchmarkSuiteRequest(
                        List.of("single_case"),
                        List.of(candidate),
                        new MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                        tuning.validate.ValidationPolicy.disabled(),
                        tuning.reporting.ReportPolicy.defaults()
                )
        ).run();

        String rendered = TextBenchmarkSuiteReportRenderer.render(report);
        assertTrue(rendered.contains("Benchmark Suite Report"));
        assertTrue(rendered.contains("Summary"));
        assertTrue(rendered.contains("Workloads"));
        assertTrue(rendered.contains("Candidate Summaries"));
        assertTrue(rendered.contains("Suite Hotspots"));
        assertTrue(rendered.contains("=== single_case ==="));
        assertTrue(rendered.contains("bestCandidate="));
        assertTrue(rendered.contains("render-suite"));
    }

    @Test
    void suiteJsonRendererProducesStructuredOutput() {
        WorkloadCatalog catalog = new WorkloadCatalog()
                .register(new TensorRootWorkloadSpec(
                        "json_case",
                        WorkloadKind.GENERIC,
                        environment -> Tensor.scalar(2.0).mul(Tensor.scalar(5.0))
                ));

        BenchmarkEntry candidate = BenchmarkEntry.candidate(
                "json-suite",
                new ExecutionProfile(
                        "json-suite-profile",
                        "json-suite",
                        DataType.FLOAT64,
                        ExecutionMode.FORWARD,
                        config.optimizer.OptimizerConfig.noOptimization(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );

        BenchmarkSuiteReport report = BenchmarkSuiteSession.create(
                catalog.benchmarkSuiteRequest(
                        List.of("json_case"),
                        List.of(candidate),
                        new MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                        tuning.validate.ValidationPolicy.disabled(),
                        tuning.reporting.ReportPolicy.defaults()
                )
        ).run();

        String json = JsonBenchmarkSuiteReportRenderer.render(report);
        assertTrue(json.contains("\"totalCandidates\": 1"));
        assertTrue(json.contains("\"candidateSummaries\": ["));
        assertTrue(json.contains("\"hotspots\": ["));
        assertTrue(json.contains("\"workloads\": ["));
        assertTrue(json.contains("\"workloadName\": \"json_case\""));
    }

    @Test
    void suiteRunsWithoutBaselineWhenEntriesContainNoBaseline() {
        WorkloadCatalog catalog = new WorkloadCatalog()
                .register(new TensorRootWorkloadSpec(
                        "no_baseline_case",
                        WorkloadKind.GENERIC,
                        environment -> Tensor.scalar(2.0).mul(Tensor.scalar(5.0))
                ));

        BenchmarkEntry candidate = BenchmarkEntry.candidate(
                "no-baseline-candidate",
                new ExecutionProfile(
                        "no-baseline-profile",
                        "no-baseline-candidate",
                        DataType.FLOAT64,
                        ExecutionMode.FORWARD,
                        config.optimizer.OptimizerConfig.noOptimization(),
                        config.runtime.RuntimeConfig.inferenceDefaults(),
                        WorkloadProfile.none()
                )
        );

        BenchmarkSuiteReport report = BenchmarkSuiteSession.create(new BenchmarkSuiteRequest(
                List.of(catalog.require("no_baseline_case")),
                List.of(candidate),
                new MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                tuning.validate.ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        )).run();

        assertEquals(1, report.workloadReports().getFirst().candidates().size());
        assertFalse(report.workloadReports().getFirst().baseline().isPresent());
    }
}
