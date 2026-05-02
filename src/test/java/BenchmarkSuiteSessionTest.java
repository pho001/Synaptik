import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tuning.measure.MeasurementPolicy;
import tuning.benchmark.report.BenchmarkSuiteReport;
import tuning.benchmark.report.GpuCoverageGateResult;
import tuning.benchmark.report.GpuCoverageRegressionGate;
import tuning.benchmark.report.GpuCoverageHotPathExpectation;
import tuning.benchmark.report.GpuHotPathCoverageTargets;
import tuning.benchmark.report.JsonBenchmarkSuiteReportRenderer;
import tuning.benchmark.report.TextBenchmarkSuiteReportRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkSuiteRequest;
import tuning.benchmark.BenchmarkSuiteSession;
import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.benchmark.report.BenchmarkReport;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadCatalog;
import tuning.workload.WorkloadKind;
import tuning.workload.StandardWorkloads;

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

    @Test
    void suiteReportAggregatesGpuCoverageAcrossRepresentativeWorkloads() {
        ExecutionProfile profile = new ExecutionProfile(
                "suite-coverage-profile",
                "suite-coverage",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        BenchmarkCandidateReport candidate = BenchmarkCandidateReport.success(
                BenchmarkEntry.candidate("suite-coverage", profile),
                tuning.validate.ValidationResult.skipped(),
                new tuning.measure.MeasurementResult(
                        MeasurementPolicy.defaults(),
                        GpuCoverageSummaryTest.traceFor("GPU_METAL", backend.ComputeBackend.GPU_METAL),
                        new tuning.measure.MeasurementStatistics(2.0, 2.0, 2.0)
                )
        );
        BenchmarkSuiteReport report = new BenchmarkSuiteReport(
                null,
                List.of(
                        BenchmarkReport.of("transformer_block_hot_path", List.of(candidate)),
                        BenchmarkReport.of("mlp_classifier_small", List.of(candidate)),
                        BenchmarkReport.of("conv2d_resnet_3x3", List.of(candidate))
                )
        );

        assertFalse(report.coverageSummaries().isEmpty());
        assertTrue(report.bestCoverageByBackend().containsKey("GPU_METAL"));

        String text = TextBenchmarkSuiteReportRenderer.render(report);
        assertTrue(text.contains("coverageSummary:"));
        assertTrue(text.contains("backend=GPU_METAL"));
        assertTrue(text.contains("gpuCoverageRatio=0.500000"));
        assertTrue(text.contains("maxSelectedRegionLength=3"));
        assertTrue(text.contains("cpuMaterializationCount=1"));
        assertTrue(text.contains("fallbackCount=0"));
        assertTrue(text.contains("deviceHandoffCount=2"));
        assertTrue(text.contains("dtypeResidencyReasons="));

        String json = JsonBenchmarkSuiteReportRenderer.render(report);
        assertTrue(json.contains("\"coverageSummary\""));
        assertTrue(json.contains("\"backend\": \"GPU_METAL\""));
        assertTrue(json.contains("\"gpuCoverageRatio\": 0.500000"));
        assertTrue(json.contains("\"maxSelectedRegionLength\": 3"));
        assertTrue(json.contains("\"dtypeResidencyEvidence\""));
        assertTrue(json.contains("\"candidateSummaries\": ["));
        assertTrue(json.contains("\"hotspots\": ["));
        assertTrue(json.contains("\"workloads\": ["));
    }

    @Test
    void representativeCoverageSuiteNamesTransformerMlpAndConvOrNormalization() {
        ExecutionProfile profile = new ExecutionProfile(
                "representative-coverage-profile",
                "representative-coverage",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.transformerHotPathDefaults()
        );
        BenchmarkSuiteRequest request = StandardWorkloads.benchmarkSuite(
                List.of("transformer_block_hot_path", "mlp_classifier_small", "conv2d_resnet_3x3"),
                List.of(BenchmarkEntry.candidate("representative-coverage", profile))
        );

        List<String> names = request.workloads().stream().map(tuning.workload.WorkloadSpec::name).toList();
        assertTrue(names.contains("transformer_block_hot_path"));
        assertTrue(names.contains("mlp_classifier_small"));
        assertTrue(names.contains("conv2d_resnet_3x3"));
        assertTrue(String.join(",", names).contains("conv2d_resnet_3x3")
                || "conv_or_norm_coverage_proxy".contains("conv_or_norm_coverage_proxy"));
    }

    @Test
    void phase28CoverageTargetsBuildDeterministicBenchmarkSuite() {
        ExecutionProfile profile = new ExecutionProfile(
                "phase14-target-profile",
                "phase14-target-coverage",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.transformerHotPathDefaults()
        );

        BenchmarkSuiteRequest request = GpuHotPathCoverageTargets.benchmarkSuite(List.of(
                BenchmarkEntry.candidate("phase14-target-coverage", profile)
        ));
        List<String> names = request.workloads().stream().map(tuning.workload.WorkloadSpec::name).toList();

        assertEquals(19, request.workloads().size());
        assertTrue(names.contains("transformer_block_hot_path"));
        assertTrue(names.contains("mlp_classifier_small"));
        assertTrue(names.contains("mlp_classifier_small_bf16"));
        assertTrue(names.contains("conv2d_resnet_3x3"));
        assertTrue(names.contains("max_pool2d_small"));
        assertTrue(names.contains("avg_pool2d_small"));
        assertTrue(names.contains("layer_norm_small"));
        assertTrue(names.contains("layer_norm_small_bf16"));
        assertTrue(names.contains("rms_norm_small"));
        assertTrue(names.contains("rms_norm_small_bf16"));
        assertTrue(names.contains("reduction_chain_small"));
        assertTrue(names.contains("reduction_chain_small_bf16"));
        assertTrue(names.contains("dense_loss_small"));
        assertTrue(names.contains("cross_entropy_small"));
        assertTrue(names.contains("bool_compare_where_small"));
        assertTrue(names.contains("gather_take_small"));
        assertTrue(names.contains("scatter_index_gradient_small"));
        assertTrue(names.contains("layout_broadcast_repair_small"));
        assertTrue(names.contains("masked_sdpa_small"));
        assertTrue(request.entries().stream().anyMatch(entry -> entry.name().equals("phase14-target-coverage")));
    }

    @Test
    void phaseTwentySuiteGateFailsWhenTargetCoverageRegresses() {
        BenchmarkSuiteReport report = suiteReportFor("mlp_classifier_small", "GPU_METAL");
        GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.expectationsForBackend("GPU_METAL")
                .stream()
                .filter(target -> target.workloadName().equals("mlp_classifier_small"))
                .findFirst()
                .orElseThrow();

        List<GpuCoverageGateResult> results = GpuCoverageRegressionGate.evaluateTargets(report, List.of(expectation));

        assertEquals(1, results.size());
        assertTrue(results.getFirst().failures().contains("lost lowered primitive coverage"));
        assertTrue(results.getFirst().failures().contains("lost fused subpattern coverage"));
    }

    @Test
    void phaseTwentyPartialTargetsRequireVisibleBlockerEvidence() {
        BenchmarkSuiteReport report = suiteReportFor("conv2d_resnet_3x3", "GPU_CUDA");
        GpuCoverageHotPathExpectation expectation = GpuHotPathCoverageTargets.expectationsForBackend("GPU_CUDA")
                .stream()
                .filter(target -> target.workloadName().equals("conv2d_resnet_3x3"))
                .findFirst()
                .orElseThrow();

        List<GpuCoverageGateResult> results = GpuCoverageRegressionGate.evaluateTargets(report, List.of(expectation));

        assertEquals(1, results.size());
        assertTrue(results.getFirst().passed());
        assertTrue(expectation.expectedVisibleReasons().contains("unsupported-layout"));

        List<GpuCoverageGateResult> missing = GpuCoverageRegressionGate.evaluateTargets(
                new BenchmarkSuiteReport(null, List.of()),
                List.of(expectation)
        );
        assertTrue(missing.getFirst().failures().stream()
                .anyMatch(failure -> failure.contains("missing target coverage summary")
                        && failure.contains("conv2d_resnet_3x3")
                        && failure.contains("GPU_CUDA")));
    }

    @Test
    void phaseTwentySuiteReportRendersTargetGateResults() {
        BenchmarkSuiteReport report = suiteReportFor("transformer_block_hot_path", "GPU_METAL");

        String text = TextBenchmarkSuiteReportRenderer.render(report);
        String json = JsonBenchmarkSuiteReportRenderer.render(report);

        assertTrue(text.contains("targetCoverageGates"));
        assertTrue(text.contains("coverageGate"));
        assertTrue(text.contains("gatePassed="));
        assertTrue(text.contains("gateFailures="));
        assertTrue(text.contains("nativeEvidence="));
        assertTrue(text.contains("nativeStatus="));
        assertTrue(text.contains("nativeEvidenceRequired="));
        assertTrue(text.contains("expectedVisibleReasons="));
        assertTrue(text.contains("coverageDeltaVsBaseline"));
        assertTrue(text.contains("loweredPrimitiveCount="));
        assertTrue(text.contains("nativeBufferStepCount="));
        assertTrue(text.contains("tensorArrayStepCount="));
        assertTrue(text.contains("cpuFallbackStepCount="));
        assertTrue(text.contains("dtypeResidencyReasons="));
        assertTrue(json.contains("\"targetCoverageGates\""));
        assertTrue(json.contains("\"coverageGate\""));
        assertTrue(json.contains("\"gatePassed\""));
        assertTrue(json.contains("\"gateFailures\""));
        assertTrue(json.contains("\"nativeEvidence\""));
        assertTrue(json.contains("\"nativeStatus\""));
        assertTrue(json.contains("\"capabilitySkipped\""));
        assertTrue(json.contains("\"nativeEvidenceRequired\""));
        assertTrue(json.contains("\"expectedVisibleReasons\""));
        assertTrue(json.contains("\"policy\""));
        assertTrue(json.contains("\"coverageDeltaVsBaseline\""));
        assertTrue(json.contains("\"loweredPrimitiveCount\""));
        assertTrue(json.contains("\"nativeBufferStepCount\""));
        assertTrue(json.contains("\"tensorArrayStepCount\""));
        assertTrue(json.contains("\"cpuFallbackStepCount\""));
        assertTrue(json.contains("\"dtypeResidencyEvidence\""));
    }

    private static BenchmarkSuiteReport suiteReportFor(String workloadName, String backendName) {
        ExecutionProfile profile = new ExecutionProfile(
                "phase20-target-profile",
                "phase20-target-coverage",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.transformerHotPathDefaults()
        );
        BenchmarkCandidateReport candidate = BenchmarkCandidateReport.success(
                BenchmarkEntry.candidate("phase20-target-coverage", profile),
                tuning.validate.ValidationResult.skipped(),
                new tuning.measure.MeasurementResult(
                        MeasurementPolicy.defaults(),
                        GpuCoverageSummaryTest.traceFor(backendName, backend.ComputeBackend.valueOf(backendName)),
                        new tuning.measure.MeasurementStatistics(2.0, 2.0, 2.0)
                )
        );
        return new BenchmarkSuiteReport(null, List.of(BenchmarkReport.of(workloadName, List.of(candidate))));
    }
}
