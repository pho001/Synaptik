import backend.ComputeBackend;
import backend.accelerator.lowering.GpuCompoundRegionSummary;
import backend.accelerator.lowering.GpuLoweredPrimitiveManifest;
import backend.accelerator.lowering.GpuLoweredRegionCandidateSpan;
import backend.accelerator.lowering.GpuLoweredRegionManifest;
import backend.accelerator.lowering.GpuLoweredRegionOriginalOp;
import backend.accelerator.lowering.GpuLoweredRegionRejection;
import backend.accelerator.lowering.GpuLoweredRegionValueAssumption;
import backend.accelerator.lowering.GpuLoweringUnsupportedReason;
import backend.memory.CpuMaterializationReason;
import backend.memory.StorageResidency;
import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import graph.optimizer.partition.cost.AcceleratorPartitionScoreModel;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tuning.benchmark.report.BenchmarkReport;
import tuning.benchmark.report.GpuCoverageGatePolicy;
import tuning.benchmark.report.GpuCoverageRegressionGate;
import tuning.benchmark.report.GpuCoverageSummary;
import tuning.benchmark.report.JsonBenchmarkReportRenderer;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.store.PersistencePolicy;
import tuning.workload.StandardWorkloads;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;
import tuning.workload.WorkloadMetadata;

import java.lang.reflect.RecordComponent;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
        BenchmarkEntry candidate = BenchmarkEntry.baseline("baseline", profile);

        BenchmarkRequest request = new BenchmarkRequest(
                workload,
                List.of(candidate),
                new tuning.measure.MeasurementPolicy(1, 2, 1, true, true, true, true, false),
                tuning.validate.ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        BenchmarkReport report = BenchmarkSession.create(request).run();

        assertEquals("simple_add_mul", report.workloadName());
        assertEquals(1, report.candidates().size());
        assertTrue(report.bestCandidateName().isBlank());
        assertTrue(report.candidates().getFirst().success());
        assertTrue(report.candidates().getFirst().measurement().trace().compile().measured());
        assertTrue(report.candidates().getFirst().measurement().trace().prepare().measured());
        assertTrue(report.candidates().getFirst().measurement().trace().run().durationNs() >= 0L);
        assertTrue(report.candidates().getFirst().measurement().steadyStateStats().medianMs() >= 0.0d);
    }

    @Test
    void benchmarkSessionRunsWithoutProfilePersistencePolicy() {
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "profile_read_only_benchmark",
                WorkloadKind.GENERIC,
                environment -> Tensor.scalar(1.0).add(Tensor.scalar(2.0))
        );

        ExecutionProfile profile = new ExecutionProfile(
                "profile-read-only",
                "profile-read-only",
                DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        BenchmarkRequest request = new BenchmarkRequest(
                workload,
                List.of(BenchmarkEntry.candidate("profile-read-only", profile)),
                new tuning.measure.MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                tuning.validate.ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        assertTrue(Arrays.stream(BenchmarkRequest.class.getRecordComponents())
                .map(RecordComponent::getType)
                .noneMatch(type -> type == PersistencePolicy.class || type.getName().startsWith("tuning.store.")));

        BenchmarkReport report = BenchmarkSession.create(request).run();

        assertEquals("profile_read_only_benchmark", report.workloadName());
        assertEquals(1, report.candidates().size());
        assertTrue(report.candidates().getFirst().success());
    }

    @Test
    void benchmarkSessionUsesExplicitBaselineAndReportsSpeedup() {
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "baseline_workload",
                WorkloadKind.GENERIC,
                environment -> Tensor.scalar(1.0).add(Tensor.scalar(2.0)).mul(Tensor.scalar(3.0))
        );

        BenchmarkEntry baseline = BenchmarkEntry.baseline(
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

        BenchmarkEntry tuned = BenchmarkEntry.candidate("candidate", baseline.profile());

        BenchmarkReport report = BenchmarkSession.create(new BenchmarkRequest(
                workload,
                List.of(baseline, tuned),
                new tuning.measure.MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                tuning.validate.ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        )).run();

        assertTrue(report.baseline().isPresent());
        assertTrue(report.candidates().stream().anyMatch(r -> r.baseline()));
        assertTrue(Double.isFinite(report.speedupVsBaseline(report.candidates().stream().filter(r -> !r.baseline()).findFirst().orElseThrow())));
    }

    @Test
    void textRendererProducesReadableSummary() {
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "renderer_workload",
                WorkloadKind.GENERIC,
                environment -> Tensor.scalar(1.0).add(Tensor.scalar(2.0))
        );

        BenchmarkEntry candidate = BenchmarkEntry.candidate(
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
                tuning.reporting.ReportPolicy.defaults()
        )).run();

        String rendered = TextBenchmarkReportRenderer.render(report);
        assertTrue(rendered.contains("Benchmark Report"));
        assertTrue(rendered.contains("Summary"));
        assertTrue(rendered.contains("Candidates"));
        assertTrue(rendered.contains("workload=renderer_workload"));
        assertTrue(rendered.contains("bestCandidate="));
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

        BenchmarkEntry candidate = BenchmarkEntry.candidate(
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
                tuning.reporting.ReportPolicy.defaults()
        )).run();

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"workloadName\": \"json_workload\""));
        assertTrue(json.contains("\"bestCandidateName\":"));
        assertTrue(json.contains("\"role\": \"CANDIDATE\""));
        assertTrue(json.contains("\"candidates\": ["));
        assertTrue(json.contains("\"timing\": {"));
        assertTrue(json.contains("\"speedup\": {"));
    }

    @Test
    void renderersHandleMissingBaselineMeasurementsAndNonFiniteSpeedups() {
        var profile = new ExecutionProfile(
                "candidate-profile",
                "candidate",
                DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        BenchmarkEntry baseline = BenchmarkEntry.baseline("baseline", profile);
        BenchmarkEntry candidate = BenchmarkEntry.candidate("candidate", profile);

        BenchmarkReport report = BenchmarkReport.of(
                "manual_report",
                List.of(
                        tuning.benchmark.report.BenchmarkCandidateReport.failure(
                                baseline,
                                tuning.validate.ValidationResult.failure("boom"),
                                "boom"
                        ),
                        tuning.benchmark.report.BenchmarkCandidateReport.success(
                                candidate,
                                tuning.validate.ValidationResult.skipped(),
                                new tuning.measure.MeasurementResult(
                                        tuning.measure.MeasurementPolicy.defaults(),
                                                new graph.execution.trace.ExecutionTrace(
                                                new graph.execution.trace.CompileTrace(true, 1L, 0, 0, false, graph.execution.trace.PartitionCompileTrace.empty()),
                                                new graph.execution.trace.PrepareTrace(true, 1L, 0, 0, graph.execution.trace.BackendSelectionTrace.empty()),
                                                graph.execution.trace.RunTrace.empty(ExecutionMode.FORWARD)
                                        ),
                                        new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                                )
                        )
                )
        );

        assertDoesNotThrow(() -> TextBenchmarkReportRenderer.render(report));
        String json = assertDoesNotThrow(() -> JsonBenchmarkReportRenderer.render(report));
        assertTrue(json.contains("\"vsBaseline\": null"));
    }

    @Test
    void renderersExposeBackendSelectionCostDiagnostics() {
        var profile = new ExecutionProfile(
                "cost-candidate-profile",
                "cost-candidate",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var summary = new AcceleratorPartitionScoreModel.MaterializationCostSummary(
                "CONSERVATIVE",
                2,
                3072L,
                8192L,
                1024L,
                250.0d,
                7780.0d,
                "accepted-static-profitable",
                "BUFFER_BINDING",
                "DENSE_PHYSICAL"
        );
        var finalists = List.of(
                new graph.execution.trace.PartitionDecisionTrace.CandidateCostTrace(
                        List.of(7, 8),
                        "rejected-materialization-cost",
                        -12.5d,
                        4,
                        8192L,
                        256L,
                        "CONSERVATIVE"
                ),
                new graph.execution.trace.PartitionDecisionTrace.CandidateCostTrace(
                        List.of(9),
                        "not-selected-lower-score",
                        120.0d,
                        1,
                        2048L,
                        512L,
                        "MEASURED"
                )
        );
        var selection = new graph.execution.trace.BackendSelectionTrace(
                3,
                1,
                2,
                List.of(new graph.execution.trace.BackendSelectionDecisionTrace(
                                4,
                                List.of(4, 5, 6),
                                List.of(ComputeBackend.GPU_METAL),
                                true,
                                ComputeBackend.GPU_METAL,
                                "selected",
                                8192L,
                                summary,
                                finalists
                        ),
                        new graph.execution.trace.BackendSelectionDecisionTrace(
                                12,
                                List.of(12),
                                List.of(ComputeBackend.GPU_METAL),
                                false,
                                null,
                                "estimated-work-below-minimum",
                                64L,
                                new AcceleratorPartitionScoreModel.MaterializationCostSummary(
                                        "CONSERVATIVE",
                                        1,
                                        512L,
                                        64L,
                                        0L,
                                        250.0d,
                                        -125.0d,
                                        "rejected-materialization-cost",
                                        "BUFFER_BINDING",
                                        "DENSE_PHYSICAL"
                                ),
                                List.of()
                        ))
        );

        BenchmarkReport report = BenchmarkReport.of(
                "cost_report",
                List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate("cost-candidate", profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                new graph.execution.trace.ExecutionTrace(
                                        new graph.execution.trace.CompileTrace(
                                                true,
                                                1L,
                                                0,
                                                0,
                                                false,
                                                graph.execution.trace.PartitionCompileTrace.empty()
                                        ),
                                        new graph.execution.trace.PrepareTrace(true, 1L, 0, 0, selection),
                                        graph.execution.trace.RunTrace.empty(ExecutionMode.FORWARD)
                                ),
                                new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                        )
                ))
        );

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("backendSelectionCost:"));
        assertTrue(text.contains("selectedBackend=GPU_METAL"));
        assertTrue(text.contains("preset=CONSERVATIVE"));
        assertTrue(text.contains("finalScore=7780.000000"));
        assertTrue(text.contains("boundaryCount=2"));
        assertTrue(text.contains("estimatedTransferBytes=3072"));
        assertTrue(text.contains("estimatedComputeWork=8192"));
        assertTrue(text.contains("reason=selected"));
        assertTrue(text.contains("rejectedFinalists:"));
        assertTrue(text.contains("reason=rejected-materialization-cost"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"backendSelectionCost\":"));
        assertTrue(json.contains("\"selected\": ["));
        assertTrue(json.contains("\"rejectedFinalists\": ["));
        assertTrue(json.contains("\"nodeIds\": [4, 5, 6]"));
        assertTrue(json.contains("\"selectedBackend\": \"GPU_METAL\""));
        assertTrue(json.contains("\"reason\": \"selected\""));
        assertTrue(json.contains("\"finalScore\": 7780.000000"));
        assertTrue(json.contains("\"boundaryCount\": 2"));
        assertTrue(json.contains("\"estimatedTransferBytes\": 3072"));
        assertTrue(json.contains("\"estimatedComputeWork\": 8192"));
        assertTrue(json.contains("\"preset\": \"CONSERVATIVE\""));
        assertTrue(json.contains("\"reason\": \"rejected-materialization-cost\""));
        assertTrue(json.contains("\"nodeIds\": [12]"));
        assertTrue(json.contains("\"reason\": \"estimated-work-below-minimum\""));
    }

    @Test
    void benchmarkTextReportRendersGpuLoweredRegionManifest() {
        BenchmarkReport report = reportWithGpuLoweredRegionManifest();

        String text = TextBenchmarkReportRenderer.render(report);

        assertTrue(text.contains("GPU Lowered Region"));
        assertTrue(text.contains("Original Ops"));
        assertTrue(text.contains("Lowered Primitives"));
        assertTrue(text.contains("Value Assumptions"));
        assertTrue(text.contains("Fused Subpatterns"));
        assertTrue(text.contains("Rejections"));
        assertTrue(text.contains("regionId: gpu-metal-region-4"));
        assertTrue(text.contains("selectedRegionLength: 3"));
        assertTrue(text.contains("LOG_SOFTMAX"));
        assertTrue(text.contains("SOFTMAX"));
    }

    @Test
    void benchmarkJsonReportRendersGpuLoweredRegionManifest() {
        BenchmarkReport report = reportWithGpuLoweredRegionManifest();

        String json = JsonBenchmarkReportRenderer.render(report);

        assertTrue(json.contains("\"gpuLoweredRegionManifest\":"));
        assertTrue(json.contains("\"regionId\": \"gpu-metal-region-4\""));
        assertTrue(json.contains("\"backend\": \"GPU_METAL\""));
        assertTrue(json.contains("\"selectedRegionLength\": 3"));
        assertTrue(json.contains("\"originalOps\":"));
        assertTrue(json.contains("\"loweredPrimitives\":"));
        assertTrue(json.contains("\"valueAssumptions\":"));
        assertTrue(json.contains("\"fusedSubpatterns\":"));
        assertTrue(json.contains("\"rejections\":"));
        assertTrue(json.contains("\"candidateSpan\":"));
    }

    @Test
    void benchmarkReportsRenderDTypeResidencyEvidence() {
        BenchmarkReport report = reportWithGpuLoweredRegionManifest();

        String text = TextBenchmarkReportRenderer.render(report);
        String json = JsonBenchmarkReportRenderer.render(report);

        assertTrue(text.contains("DType Residency Evidence"));
        assertTrue(text.contains("dtypeResidency"));
        assertTrue(text.contains("UNSUPPORTED_DTYPE"));
        assertTrue(text.contains("backend=GPU_METAL"));
        assertTrue(text.contains("backend=GPU_CUDA"));
        assertTrue(text.contains("dtype=BFLOAT16"));
        assertTrue(text.contains("dtype=INT32"));
        assertTrue(text.contains("dtype=BOOL"));
        assertTrue(json.contains("\"dtypeResidencyEvidence\""));
        assertTrue(json.contains("dtypeResidency"));
        assertTrue(json.contains("UNSUPPORTED_DTYPE"));
        assertTrue(json.contains("backend=GPU_METAL"));
        assertTrue(json.contains("backend=GPU_CUDA"));
        assertTrue(json.contains("dtype=BFLOAT16"));
        assertTrue(json.contains("dtype=INT32"));
        assertTrue(json.contains("dtype=BOOL"));
    }

    @Test
    void benchmarkReportsRenderPhaseSeventeenNormAndLossEvidence() {
        BenchmarkReport report = reportWithGpuLoweredRegionManifest();

        String text = TextBenchmarkReportRenderer.render(report);
        String json = JsonBenchmarkReportRenderer.render(report);

        assertTrue(text.contains("LOG_SOFTMAX"));
        assertTrue(text.contains("SOFTMAX"));
        assertTrue(text.contains("family=NORMALIZATION"));
        assertTrue(text.contains("family=LOSS_ADJACENT"));
        assertTrue(text.contains("UNSUPPORTED_DTYPE"));
        assertTrue(text.contains("target=layer_norm_small"));
        assertTrue(text.contains("target=transformer_block_hot_path"));
        assertTrue(json.contains("LOG_SOFTMAX"));
        assertTrue(json.contains("SOFTMAX"));
        assertTrue(json.contains("family=NORMALIZATION"));
        assertTrue(json.contains("family=LOSS_ADJACENT"));
        assertTrue(json.contains("UNSUPPORTED_DTYPE"));
        assertTrue(json.contains("target=layer_norm_small"));
        assertTrue(json.contains("target=transformer_block_hot_path"));
    }

    @Test
    void renderersExposeAcceleratorEvidenceContract() {
        var profile = new ExecutionProfile(
                "accelerator-evidence-profile",
                "accelerator-evidence",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var step = new graph.execution.trace.ExecutionStepTrace(
                0,
                "metal_linear",
                "LINEAR",
                List.of(16, 16),
                DataType.FLOAT32,
                "GPU_METAL",
                "PreparedMetalExecutable",
                2_000_000L,
                new graph.execution.trace.StepExecutionMetadata(
                        "node",
                        Map.ofEntries(
                                Map.entry("acceleratorBufferBackend", "GPU_METAL"),
                                Map.entry("acceleratorBufferMode", "AUTO"),
                                Map.entry("acceleratorBufferExecutionPath", "BUFFER_BINDING"),
                                Map.entry("acceleratorBufferReasonCode", "BUFFER_BINDING_AVAILABLE"),
                                Map.entry("acceleratorBufferReason", "using native buffer bindings"),
                                Map.entry("acceleratorBufferPreparedInputUsed", true),
                                Map.entry("metalJavaToNativeCopyNs", 100_000L),
                                Map.entry("metalNativeToJavaCopyNs", 0L),
                                Map.entry("metalNativeDeviceCopyNs", 25_000L),
                                Map.entry("storageResidency", "DEVICE_OWNED"),
                                Map.entry("storageCpuCurrent", false),
                                Map.entry("storageDeviceCurrent", true)
                        ),
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null
                )
        );

        BenchmarkReport report = BenchmarkReport.of(
                "accelerator_evidence_report",
                List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate("accelerator-evidence", profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                new graph.execution.trace.ExecutionTrace(
                                        new graph.execution.trace.CompileTrace(true, 1L, 0, 0, false, graph.execution.trace.PartitionCompileTrace.empty()),
                                        new graph.execution.trace.PrepareTrace(true, 1L, 0, 0, graph.execution.trace.BackendSelectionTrace.empty()),
                                        new graph.execution.trace.RunTrace(
                                                ExecutionMode.FORWARD,
                                                2_000_000L,
                                                List.of(step)
                                        )
                                ),
                                new tuning.measure.MeasurementStatistics(2.0, 2.0, 2.0)
                        )
                ))
        );

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("accelerator:"));
        assertTrue(text.contains("backend=GPU_METAL"));
        assertTrue(text.contains("bufferBindingSteps=1"));
        assertTrue(text.contains("preparedInputSteps=1"));
        assertTrue(text.contains("reasonCodes=[BUFFER_BINDING_AVAILABLE]"));
        assertTrue(text.contains("fallbackReasons=[using native buffer bindings]"));
        assertTrue(text.contains("javaToNativeMs=0.100000"));
        assertTrue(text.contains("nativeDeviceCopyMs=0.025000"));
        assertTrue(text.contains("storageResidency=DEVICE_OWNED"));
        assertTrue(text.contains("DEVICE_OWNED"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"accelerator\":"));
        assertTrue(json.contains("\"GPU_METAL\":"));
        assertTrue(json.contains("\"bufferBindingSteps\": 1"));
        assertTrue(json.contains("\"preparedInputSteps\": 1"));
        assertTrue(json.contains("\"reasonCodes\": [\"BUFFER_BINDING_AVAILABLE\"]"));
        assertTrue(json.contains("\"fallbackReasons\": [\"using native buffer bindings\"]"));
        assertTrue(json.contains("\"javaToNativeCopyNs\": 100000"));
        assertTrue(json.contains("\"nativeDeviceCopyNs\": 25000"));
        assertTrue(json.contains("\"storageResidency\": \"DEVICE_OWNED\""));
        assertTrue(json.contains("\"storageCpuCurrent\": false"));
        assertTrue(json.contains("\"storageDeviceCurrent\": true"));
    }

    @Test
    void benchmarkSessionReportsCudaAcceleratorEvidenceContract() {
        var profile = new ExecutionProfile(
                "cuda-accelerator-evidence-profile",
                "cuda-accelerator-evidence",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var step = new graph.execution.trace.ExecutionStepTrace(
                0,
                "cuda_linear",
                "LINEAR",
                List.of(16, 16),
                DataType.FLOAT32,
                "GPU_CUDA",
                "PreparedCudaExecutable",
                2_000_000L,
                new graph.execution.trace.StepExecutionMetadata(
                        "node",
                        Map.ofEntries(
                                Map.entry("acceleratorBufferBackend", "GPU_CUDA"),
                                Map.entry("acceleratorBufferMode", "AUTO"),
                                Map.entry("acceleratorBufferExecutionPath", "BUFFER_BINDING"),
                                Map.entry("acceleratorBufferReasonCode", "BUFFER_BINDING_AVAILABLE"),
                                Map.entry("acceleratorBufferReason", "using native CUDA buffer bindings"),
                                Map.entry("acceleratorBufferPreparedInputUsed", true),
                                Map.entry("acceleratorInputBytes", 2048L),
                                Map.entry("acceleratorOutputBytes", 1024L),
                                Map.entry("acceleratorJavaToNativeCopyNs", 100_000L),
                                Map.entry("acceleratorNativeToJavaCopyNs", 0L),
                                Map.entry("acceleratorNativeDeviceCopyNs", 25_000L),
                                Map.entry("storageResidency", "DEVICE_OWNED"),
                                Map.entry("storageCpuCurrent", false),
                                Map.entry("storageDeviceCurrent", true)
                        ),
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null
                )
        );

        BenchmarkReport report = BenchmarkReport.of(
                "cuda_accelerator_evidence_report",
                List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate("cuda-accelerator-evidence", profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                new graph.execution.trace.ExecutionTrace(
                                        new graph.execution.trace.CompileTrace(true, 1L, 0, 0, false, graph.execution.trace.PartitionCompileTrace.empty()),
                                        new graph.execution.trace.PrepareTrace(true, 1L, 0, 0, graph.execution.trace.BackendSelectionTrace.empty()),
                                        new graph.execution.trace.RunTrace(
                                                ExecutionMode.FORWARD,
                                                2_000_000L,
                                                List.of(step)
                                        )
                                ),
                                new tuning.measure.MeasurementStatistics(2.0, 2.0, 2.0)
                        )
                ))
        );

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("backend=GPU_CUDA"));
        assertTrue(text.contains("bufferBindingSteps=1"));
        assertTrue(text.contains("preparedInputSteps=1"));
        assertTrue(text.contains("reasonCodes=[BUFFER_BINDING_AVAILABLE]"));
        assertTrue(text.contains("fallbackReasons=[using native CUDA buffer bindings]"));
        assertTrue(text.contains("bytes=2048->1024"));
        assertTrue(text.contains("javaToNativeMs=0.100000"));
        assertTrue(text.contains("nativeDeviceCopyMs=0.025000"));
        assertTrue(text.contains("storageResidency=DEVICE_OWNED"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"GPU_CUDA\""));
        assertTrue(json.contains("\"bufferBindingSteps\": 1"));
        assertTrue(json.contains("\"preparedInputSteps\": 1"));
        assertTrue(json.contains("\"reasonCodes\": [\"BUFFER_BINDING_AVAILABLE\"]"));
        assertTrue(json.contains("\"fallbackReasons\": [\"using native CUDA buffer bindings\"]"));
        assertTrue(json.contains("\"inputBytes\": 2048"));
        assertTrue(json.contains("\"outputBytes\": 1024"));
        assertTrue(json.contains("\"javaToNativeCopyNs\": 100000"));
        assertTrue(json.contains("\"nativeDeviceCopyNs\": 25000"));
        assertTrue(json.contains("\"storageResidency\": \"DEVICE_OWNED\""));
    }

    @Test
    void benchmarkSessionReportsClosureTransformerBlockTraceContract() {
        var profile = new ExecutionProfile(
                "accelerator-closure-transformer-block",
                "accelerator-closure-transformer-block",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                config.optimizer.OptimizerConfig.trainingDefaults(),
                config.runtime.RuntimeConfig.trainingDefaults(),
                StandardWorkloads.transformerHotPathDefaults()
        );
        BenchmarkReport report = BenchmarkSession.create(new BenchmarkRequest(
                StandardWorkloads.transformerBlockHotPath("accelerator_closure_transformer_block"),
                List.of(BenchmarkEntry.candidate("accelerator-closure", profile)),
                new tuning.measure.MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                tuning.validate.ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        )).run();

        assertEquals("accelerator_closure_transformer_block", report.workloadName());
        assertTrue(report.candidates().getFirst().success());

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("cpuMaterializationCount="));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"trace\": {"));
    }

    @Test
    void renderersExposeMetalBridgeTransferDiagnostics() {
        var profile = new ExecutionProfile(
                "metal-candidate-profile",
                "metal-candidate",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var step = new graph.execution.trace.ExecutionStepTrace(
                0,
                "metal_linear",
                "LINEAR",
                List.of(16, 16),
                DataType.FLOAT32,
                "GPU_METAL",
                "PreparedMetalExecutable",
                2_000_000L,
                new graph.execution.trace.StepExecutionMetadata(
                        "node",
                        Map.of(
                                "metalBridgeAvailable", true,
                                "metalExecutionPath", "TENSOR_ARRAY_COPY",
                                "metalSupportsBufferBindings", false,
                                "metalUsedCpuFallback", false,
                                "metalInputBytes", 2048L,
                                "metalOutputBytes", 1024L,
                                "metalJavaToNativeCopyNs", 100_000L,
                                "metalNativeExecuteNs", 1_500_000L,
                                "metalNativeToJavaCopyNs", 200_000L
                        ),
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null
                )
        );

        BenchmarkReport report = BenchmarkReport.of(
                "metal_report",
                List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate("metal-candidate", profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                new graph.execution.trace.ExecutionTrace(
                                        new graph.execution.trace.CompileTrace(true, 1L, 0, 0, false, graph.execution.trace.PartitionCompileTrace.empty()),
                                        new graph.execution.trace.PrepareTrace(true, 1L, 0, 0, graph.execution.trace.BackendSelectionTrace.empty()),
                                        new graph.execution.trace.RunTrace(
                                                ExecutionMode.FORWARD,
                                                2_000_000L,
                                                List.of(step),
                                                List.of(new graph.execution.trace.CpuMaterializationTrace(
                                                        42,
                                                        CpuMaterializationReason.GRAPH_OUTPUT,
                                                        "GPU_METAL",
                                                        StorageResidency.DEVICE_OWNED,
                                                        4096L,
                                                        250_000L,
                                                        true,
                                                        "device value synchronized to CPU storage"
                                                ))
                                        )
                                ),
                                new tuning.measure.MeasurementStatistics(2.0, 2.0, 2.0)
                        )
                ))
        );

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("metalPath=TENSOR_ARRAY_COPY"));
        assertTrue(text.contains("metalFallback=false"));
        assertTrue(text.contains("metalBytes=2048->1024"));
        assertTrue(text.contains("metalNativeMs=1.500000"));
        assertTrue(text.contains("cpuMaterializationCount=1"));
        assertTrue(text.contains("nodeId=42 reason=GRAPH_OUTPUT from=GPU_METAL residency=DEVICE_OWNED bytes=4096"));
        assertTrue(text.contains("durationMs=0.250000 completed=true"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"metalExecutionPath\": \"TENSOR_ARRAY_COPY\""));
        assertTrue(json.contains("\"metalSupportsBufferBindings\": false"));
        assertTrue(json.contains("\"metalInputBytes\": 2048"));
        assertTrue(json.contains("\"metalNativeExecuteNs\": 1500000"));
        assertTrue(json.contains("\"cpuMaterializationCount\": 1"));
        assertTrue(json.contains("\"reason\": \"GRAPH_OUTPUT\""));
        assertTrue(json.contains("\"materializedFrom\": \"GPU_METAL\""));
        assertTrue(json.contains("\"sourceResidency\": \"DEVICE_OWNED\""));
        assertTrue(json.contains("\"durationNs\": 250000"));
    }

    @Test
    void renderersExposeGpuCoverageContract() {
        var profile = new ExecutionProfile(
                "gpu-coverage-profile",
                "gpu-coverage",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        BenchmarkReport report = BenchmarkReport.of(
                "gpu_coverage_report",
                List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate("gpu-coverage", profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                GpuCoverageSummaryTest.traceFor("GPU_METAL", ComputeBackend.GPU_METAL),
                                new tuning.measure.MeasurementStatistics(2.0, 2.0, 2.0)
                        )
                ))
        );

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("coverage:"));
        assertTrue(text.contains("backend=GPU_METAL"));
        assertTrue(text.contains("gpuCoverageRatio=0.500000"));
        assertTrue(text.contains("selectedRegionCount=1"));
        assertTrue(text.contains("maxSelectedRegionLength=3"));
        assertTrue(text.contains("averageSelectedRegionLength=3.000000"));
        assertTrue(text.contains("rejectedCandidateReasonCounts={unsupported-layout=1}"));
        assertTrue(text.contains("fallbackCount=0"));
        assertTrue(text.contains("tensorArrayStepCount=0"));
        assertTrue(text.contains("cpuFallbackStepCount=0"));
        assertTrue(text.contains("cpuMaterializationReasonCounts={CPU_CONSUMER=1}"));
        assertTrue(text.contains("deviceHandoffCount=2"));
        assertTrue(text.contains("storageResidencyCounts={DEVICE_OWNED=1}"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"coverage\""));
        assertTrue(json.contains("\"gpuCoverageRatio\": 0.500000"));
        assertTrue(json.contains("\"selectedRegionCount\": 1"));
        assertTrue(json.contains("\"maxSelectedRegionLength\": 3"));
        assertTrue(json.contains("\"averageSelectedRegionLength\": 3.000000"));
        assertTrue(json.contains("\"rejectedCandidateReasonCounts\": {\"unsupported-layout\": 1}"));
        assertTrue(json.contains("\"fallbackCount\": 0"));
        assertTrue(json.contains("\"tensorArrayStepCount\": 0"));
        assertTrue(json.contains("\"cpuFallbackStepCount\": 0"));
        assertTrue(json.contains("\"cpuMaterializationReasonCounts\": {\"CPU_CONSUMER\": 1}"));
        assertTrue(json.contains("\"copyDurationNs\": 325000"));
        assertTrue(json.contains("\"deviceHandoffCount\": 2"));
        assertTrue(json.contains("\"storageResidencyCounts\": {\"DEVICE_OWNED\": 1}"));
    }

    @Test
    void renderersExposeCudaGpuCoverageContract() {
        var profile = new ExecutionProfile(
                "cuda-gpu-coverage-profile",
                "cuda-gpu-coverage",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        BenchmarkReport report = BenchmarkReport.of(
                "cuda_gpu_coverage_report",
                List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate("cuda-gpu-coverage", profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                GpuCoverageSummaryTest.traceFor("GPU_CUDA", ComputeBackend.GPU_CUDA),
                                new tuning.measure.MeasurementStatistics(2.0, 2.0, 2.0)
                        )
                ))
        );

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("backend=GPU_CUDA"));
        assertTrue(text.contains("gpuCoverageRatio=0.500000"));
        assertTrue(text.contains("selectedRegionCount=1"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"GPU_CUDA\""));
        assertTrue(json.contains("\"gpuCoverageRatio\": 0.500000"));
        assertTrue(json.contains("\"selectedRegionCount\": 1"));
    }

    @Test
    void coverageGateRejectsHiddenCpuExitInBenchmarkTrace() {
        GpuCoverageSummary summary = GpuCoverageSummary.fromTrace(
                GpuCoverageSummaryTest.traceFor("GPU_METAL", ComputeBackend.GPU_METAL)
        );
        GpuCoverageGatePolicy policy = GpuCoverageGatePolicy.nativeBufferTarget("GPU_METAL", 0.5d, 3);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> GpuCoverageRegressionGate.requirePass(summary, policy)
        );

        assertTrue(failure.getMessage().contains("unexpected CPU materialization"));
    }

    @Test
    void coverageGateChecksPortableCudaBenchmarkTrace() {
        GpuCoverageSummary summary = GpuCoverageSummary.fromTrace(
                GpuCoverageSummaryTest.traceFor("GPU_CUDA", ComputeBackend.GPU_CUDA)
        );
        GpuCoverageGatePolicy policy = GpuCoverageGatePolicy.nativeBufferTarget("GPU_CUDA", 0.5d, 3);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> GpuCoverageRegressionGate.requirePass(summary, policy)
        );

        assertTrue(failure.getMessage().contains("unexpected CPU materialization"));
    }

    private static BenchmarkReport reportWithGpuLoweredRegionManifest() {
        var profile = new ExecutionProfile(
                "gpu-lowered-region-profile",
                "gpu-lowered-region",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var summary = new AcceleratorPartitionScoreModel.MaterializationCostSummary(
                "CONSERVATIVE",
                1,
                1024L,
                4096L,
                512L,
                250.0d,
                2048.0d,
                "accepted-static-profitable",
                "BUFFER_BINDING",
                "DENSE_PHYSICAL"
        );
        var selection = new graph.execution.trace.BackendSelectionTrace(
                1,
                1,
                0,
                List.of(new graph.execution.trace.BackendSelectionDecisionTrace(
                        4,
                        List.of(4, 5, 6),
                        List.of(ComputeBackend.GPU_METAL),
                        true,
                        ComputeBackend.GPU_METAL,
                        "selected",
                        4096L,
                        summary,
                        List.of(),
                        sampleGpuManifest()
                ))
        );

        return BenchmarkReport.of(
                "gpu_lowered_region_report",
                List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate("gpu-lowered-region", profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                new graph.execution.trace.ExecutionTrace(
                                        new graph.execution.trace.CompileTrace(
                                                true,
                                                1L,
                                                0,
                                                0,
                                                false,
                                                graph.execution.trace.PartitionCompileTrace.empty()
                                        ),
                                        new graph.execution.trace.PrepareTrace(true, 1L, 0, 0, selection),
                                        graph.execution.trace.RunTrace.empty(ExecutionMode.FORWARD)
                                ),
                                new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                        )
                ))
        );
    }

    private static GpuLoweredRegionManifest sampleGpuManifest() {
        return new GpuLoweredRegionManifest(
                "gpu-metal-region-4",
                ComputeBackend.GPU_METAL,
                4,
                List.of(4, 5, 6),
                List.of(1, 2),
                List.of(6),
                3,
                List.of(new GpuLoweredRegionOriginalOp(
                        4,
                        "LOG_SOFTMAX",
                        List.of(3),
                        List.of(4),
                        DataType.FLOAT32,
                        List.of(2, 3),
                        List.of("p0", "p1"),
                        List.of()
                )),
                List.of(new GpuLoweredPrimitiveManifest(
                                "p0",
                                "SOFTMAX",
                                List.of(4),
                                List.of("external:0"),
                                "node:0",
                                DataType.FLOAT32,
                                List.of(2, 3),
                                List.of()
                        ),
                        new GpuLoweredPrimitiveManifest(
                                "p1",
                                "LOG",
                                List.of(4),
                                List.of("p0"),
                                "node:1",
                                DataType.FLOAT32,
                                List.of(2, 3),
                                List.of()
                        )),
                List.of(new GpuLoweredRegionValueAssumption(
                        1,
                        "input",
                        DataType.FLOAT32,
                        2,
                        List.of(2, 3),
                        "CONTIGUOUS",
                        true,
                        false,
                        0L
                )),
                List.of(new GpuLoweredRegionValueAssumption(
                        6,
                        "output",
                        DataType.FLOAT32,
                        2,
                        List.of(2, 3),
                        "CONTIGUOUS",
                        true,
                        false,
                        0L
                )),
                GpuCompoundRegionSummary.supported(
                        ComputeBackend.GPU_METAL,
                        backend.accelerator.lowering.GpuCompoundPatternType.ELEMENTWISE_CHAIN,
                        List.of(4, 5, 6),
                        List.of(1, 2),
                        List.of(6),
                        List.of("SOFTMAX", "LOG"),
                        List.of("LOG"),
                        "benchmark manifest fixture"
                ),
                List.of(
                        new GpuLoweredRegionRejection(
                                "primitive",
                                4,
                                "p0",
                                "",
                                GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE,
                                "dtypeResidency backend=GPU_CUDA role=compute dtype=INT32 unsupported"
                        ),
                        new GpuLoweredRegionRejection(
                                "planner.normalization",
                                90,
                                "",
                                "",
                                GpuLoweringUnsupportedReason.DEFERRED_FUSED_REGION,
                                "REDUCTION_ADJACENT: DEFERRED_FUSED_REGION: operation LAYER_NORM is not supported by GPU_METAL lowering family=NORMALIZATION status=fallback note=normalization requires compound reduction-adjacent GPU region execution; target=layer_norm_small"
                        ),
                        new GpuLoweredRegionRejection(
                                "planner.loss",
                                91,
                                "",
                                "",
                                GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE,
                                "UNSUPPORTED_DTYPE: operation CROSS_ENTROPY_LOSS_INDICES is not supported by GPU_METAL lowering family=LOSS_ADJACENT status=unsupported note=index-target loss uses INT32 targets outside the current accelerator DAG dtype contract; target=transformer_block_hot_path"
                        )
                ),
                GpuLoweredRegionCandidateSpan.none(List.of(4, 5, 6)),
                Map.of(
                        "dagNodeCount", "2",
                        "dtypeResidency.input.1", "backend=GPU_METAL role=externalInput dtype=BOOL residentRepresentable=true",
                        "dtypeResidency.compute.4", "backend=GPU_METAL role=compute dtype=BFLOAT16 unsupported"
                )
        );
    }
}
