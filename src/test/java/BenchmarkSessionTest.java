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
import config.compile.BackendPlanningFailurePolicy;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.BFloat16TrainingPolicy;
import config.runtime.CpuStorageProfile;
import config.runtime.DeviceTransferPolicy;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.NativeCpuMemoryConfig;
import config.runtime.NativeMemoryPoolPolicy;
import config.runtime.RuntimeConfig;
import graph.optimizer.partition.cost.AcceleratorPartitionScoreModel;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tuning.benchmark.report.BenchmarkReport;
import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.benchmark.report.GpuCoverageGatePolicy;
import tuning.benchmark.report.GpuCoverageNativeEvidence;
import tuning.benchmark.report.GpuCoverageRegressionGate;
import tuning.benchmark.report.GpuCoverageSummary;
import tuning.benchmark.report.JsonBenchmarkReportRenderer;
import tuning.benchmark.report.NativeDeviceBridgeBenchmarkGate;
import tuning.benchmark.report.Bf16PerformanceBenchmarkGate;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.Bf16PerformanceBenchmark;
import tuning.benchmark.NativeDeviceBridgeBenchmark;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.store.PersistencePolicy;
import tuning.workload.StandardWorkloads;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;
import tuning.workload.WorkloadMetadata;
import graph.execution.PublicationPolicy;
import graph.execution.trace.NativeCpuMemoryTrace;
import tuning.measure.MeasurementExecutionMode;
import tuning.measure.MeasurementPolicy;

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
    void bf16PerformanceBenchmarkDefinesTruthEvidenceProfiles() {
        List<BenchmarkEntry> entries = Bf16PerformanceBenchmark.entries();

        assertEquals(5, entries.size());
        assertEquals(Bf16PerformanceBenchmark.F32_MLP_BASELINE, entries.get(0).name());
        assertEquals(Bf16PerformanceBenchmark.BF16_PROMOTED_MLP, entries.get(1).name());
        assertEquals(Bf16PerformanceBenchmark.BF16_SBGEMM_CONTINUATION, entries.get(2).name());
        assertEquals(Bf16PerformanceBenchmark.BF16_BGEMM_OUTPUT, entries.get(3).name());
        assertEquals(Bf16PerformanceBenchmark.BF16_TRAINING_POLICY, entries.get(4).name());

        assertEquals(DataType.FLOAT32, entries.get(0).profile().dataType());
        assertEquals(DataType.BFLOAT16, entries.get(1).profile().dataType());
        assertEquals(backend.blas.BlasProvider.NONE, entries.get(1).profile().runtime().blas().provider());
        assertEquals(backend.blas.BlasProvider.OPENBLAS_FFM, entries.get(2).profile().runtime().blas().provider());
        assertEquals(backend.blas.BlasProvider.OPENBLAS_FFM, entries.get(3).profile().runtime().blas().provider());
        assertEquals(ExecutionMode.FORWARD_BACKWARD, entries.get(4).profile().mode());
        assertEquals(CpuStorageProfile.CPU_NATIVE, entries.get(4).profile().runtime().cpuStorageProfile());
        assertEquals(BFloat16TrainingPolicy.ACTIVATIONS_ONLY, entries.get(4).profile().runtime().bfloat16TrainingPolicy());
    }

    @Test
    void nativeDeviceBridgeBenchmarkDefinesThreeTransferProfiles() {
        List<BenchmarkEntry> entries = NativeDeviceBridgeBenchmark.entries();

        assertEquals(3, entries.size());
        assertEquals(NativeDeviceBridgeBenchmark.CPU_ARRAY_METAL, entries.get(0).name());
        assertEquals(NativeDeviceBridgeBenchmark.CPU_NATIVE_ARRAY_BRIDGE_METAL, entries.get(1).name());
        assertEquals(NativeDeviceBridgeBenchmark.CPU_NATIVE_DIRECT_METAL, entries.get(2).name());

        ExecutionProfile array = entries.get(0).profile();
        ExecutionProfile bridge = entries.get(1).profile();
        ExecutionProfile direct = entries.get(2).profile();

        assertEquals(CpuStorageProfile.CPU_ARRAY, array.runtime().cpuStorageProfile());
        assertEquals(DeviceTransferPolicy.ALLOW_ARRAY_BRIDGE, array.runtime().deviceTransferPolicy());
        assertEquals(CpuStorageProfile.CPU_NATIVE, bridge.runtime().cpuStorageProfile());
        assertEquals(DeviceTransferPolicy.ALLOW_ARRAY_BRIDGE, bridge.runtime().deviceTransferPolicy());
        assertEquals(CpuStorageProfile.CPU_NATIVE, direct.runtime().cpuStorageProfile());
        assertEquals(NativeCpuFailurePolicy.REQUIRE_NATIVE, direct.runtime().nativeCpuFailurePolicy());
        assertEquals(DeviceTransferPolicy.REQUIRE_DIRECT, direct.runtime().deviceTransferPolicy());
        assertEquals(
                AcceleratorBufferBindingMode.REQUIRE,
                direct.runtime().accelerator().metal().buffer().bindingMode()
        );
        assertEquals(
                BackendPlanningFailurePolicy.REQUIRE_ACCELERATOR_REGION,
                direct.compile().backendPlanning().failurePolicy()
        );
    }

    @Test
    void nativeDeviceBridgeBenchmarkSessionRendersAndGatesTransferEvidence() {
        BenchmarkRequest request = NativeDeviceBridgeBenchmark.request(
                32,
                32,
                32,
                new MeasurementPolicy(0, 1, 1, true, true, true, true, true)
        );

        BenchmarkReport report = BenchmarkSession.create(
                request,
                (candidate, workload, policy) -> nativeDeviceBridgeMeasurement(candidate.name(), policy),
                (candidate, workloadSpec, workload, policy) -> tuning.validate.ValidationResult.skipped()
        ).run();

        NativeDeviceBridgeBenchmarkGate.requirePass(report);

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("cpu-array-metal"));
        assertTrue(text.contains("cpu-native-array-bridge-metal"));
        assertTrue(text.contains("cpu-native-direct-metal"));
        assertTrue(text.contains("steadyStateP90Ms=1.400000"));
        assertTrue(text.contains("kind=CPU_ARRAY_TO_DEVICE_COPY"));
        assertTrue(text.contains("kind=NATIVE_TO_ARRAY_TO_DEVICE_BRIDGE"));
        assertTrue(text.contains("kind=NATIVE_SEGMENT_TO_DEVICE_COPY"));
        assertTrue(text.contains("javaArrayBytes=0"));
        assertTrue(text.contains("fallbackReason=native-device-direct-transfer-unavailable"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"name\": \"cpu-native-direct-metal\""));
        assertTrue(json.contains("\"p90Ms\": 1.4"));
        assertTrue(json.contains("\"transferKind\": \"CPU_ARRAY_TO_DEVICE_COPY\""));
        assertTrue(json.contains("\"transferKind\": \"NATIVE_TO_ARRAY_TO_DEVICE_BRIDGE\""));
        assertTrue(json.contains("\"transferKind\": \"NATIVE_SEGMENT_TO_DEVICE_COPY\""));
        assertTrue(json.contains("\"javaArrayBytes\": 0"));
        assertTrue(json.contains("\"directTransferSupported\": true"));
        assertTrue(json.contains("\"fallbackReason\": \"native-device-direct-transfer-unavailable\""));
    }

    @Test
    void nativeDeviceBridgeBenchmarkGateRejectsHiddenDirectFallback() {
        BenchmarkReport report = BenchmarkReport.of(
                NativeDeviceBridgeBenchmark.WORKLOAD_NAME,
                List.of(
                        nativeDeviceBridgeCandidate(
                                NativeDeviceBridgeBenchmark.CPU_ARRAY_METAL,
                                graph.execution.trace.HostDeviceTransferKind.CPU_ARRAY_TO_DEVICE_COPY
                        ),
                        nativeDeviceBridgeCandidate(
                                NativeDeviceBridgeBenchmark.CPU_NATIVE_ARRAY_BRIDGE_METAL,
                                graph.execution.trace.HostDeviceTransferKind.NATIVE_TO_ARRAY_TO_DEVICE_BRIDGE
                        ),
                        nativeDeviceBridgeCandidate(
                                NativeDeviceBridgeBenchmark.CPU_NATIVE_DIRECT_METAL,
                                graph.execution.trace.HostDeviceTransferKind.NATIVE_TO_ARRAY_TO_DEVICE_BRIDGE
                        )
                )
        );

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> NativeDeviceBridgeBenchmarkGate.requirePass(report)
        );
        assertTrue(failure.getMessage().contains("missing transfer route NATIVE_SEGMENT_TO_DEVICE_COPY"));
    }

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
                config.compile.CompileConfig.inference(),
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
    void benchmarkSessionReportsNativeMemoryPoolPolicyVariants() {
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "native_memory_pool_variants",
                WorkloadKind.GENERIC,
                environment -> {
                    Tensor left = new Tensor(
                            new float[]{1f, -2f, 3f, -4f, 5f, -6f, 7f, -8f},
                            new int[]{8},
                            null,
                            "left",
                            DataType.FLOAT32
                    );
                    Tensor right = new Tensor(
                            new float[]{8f, 7f, -6f, -5f, 4f, 3f, -2f, -1f},
                            new int[]{8},
                            null,
                            "right",
                            DataType.FLOAT32
                    );
                    return left.add(right).relu();
                },
                environment -> tuning.validate.ValidationReference.none(),
                environment -> new WorkloadMetadata("native_memory_pool_variants", WorkloadKind.GENERIC, Map.of("dtype", "f32"))
        );

        BenchmarkReport report = BenchmarkSession.create(new BenchmarkRequest(
                workload,
                List.of(
                        BenchmarkEntry.baseline(
                                "pool-disabled",
                                nativeMemoryProfile("pool-disabled", NativeCpuMemoryConfig.disabled())
                        ),
                        BenchmarkEntry.candidate(
                                "pool-per-execution",
                                nativeMemoryProfile("pool-per-execution", NativeCpuMemoryConfig.perExecution(4096L))
                        ),
                        BenchmarkEntry.candidate(
                                "pool-per-prepared",
                                nativeMemoryProfile("pool-per-prepared", NativeCpuMemoryConfig.perPreparedExecution(4096L))
                        )
                ),
                new tuning.measure.MeasurementPolicy(
                        1,
                        1,
                        1,
                        true,
                        true,
                        true,
                        true,
                        true,
                        PublicationPolicy.NONE
                ),
                tuning.validate.ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        )).run();

        assertEquals(3, report.candidates().size());
        assertTrue(report.candidates().stream().allMatch(BenchmarkCandidateReport::success));

        NativeCpuMemoryTrace disabled = nativeCpuMemoryTrace(report, "pool-disabled");
        NativeCpuMemoryTrace perExecution = nativeCpuMemoryTrace(report, "pool-per-execution");
        NativeCpuMemoryTrace perPrepared = nativeCpuMemoryTrace(report, "pool-per-prepared");

        assertEquals(NativeMemoryPoolPolicy.DISABLED.name(), disabled.effectivePoolPolicy());
        assertEquals(0L, disabled.poolHitCount());
        assertEquals(0L, disabled.poolMissCount());
        assertTrue(disabled.peakLiveBytes() > 0L);

        assertEquals(NativeMemoryPoolPolicy.PER_EXECUTION.name(), perExecution.effectivePoolPolicy());
        assertTrue(perExecution.poolMissCount() > 0L);
        assertTrue(perExecution.peakLiveBytes() > 0L);

        assertEquals(NativeMemoryPoolPolicy.PER_PREPARED_EXECUTION.name(), perPrepared.effectivePoolPolicy());
        assertTrue(perPrepared.poolHitCount() > 0L);
        assertTrue(perPrepared.peakLiveBytes() > 0L);
        assertEquals(0L, perPrepared.retainedBytes());
        assertTrue(Double.isFinite(findReport(report, "pool-per-prepared").measurement().steadyStateStats().medianMs()));

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("effectivePoolPolicy=PER_PREPARED_EXECUTION"));
        assertTrue(text.contains("poolHitCount=" + perPrepared.poolHitCount()));
        assertTrue(text.contains("peakLiveBytes=" + perPrepared.peakLiveBytes()));
        assertTrue(text.contains("retainedBytes=0"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"effectivePoolPolicy\": \"PER_PREPARED_EXECUTION\""));
        assertTrue(json.contains("\"poolHitCount\": " + perPrepared.poolHitCount()));
        assertTrue(json.contains("\"peakLiveBytes\": " + perPrepared.peakLiveBytes()));
        assertTrue(json.contains("\"retainedBytes\": 0"));
    }

    @Test
    void benchmarkSessionMeasuresNativeTrainingOptimizerStepEvidence() {
        TensorRootWorkloadSpec workload = trainingOptimizerWorkload("native_training_optimizer_evidence");

        BenchmarkReport report = BenchmarkSession.create(new BenchmarkRequest(
                workload,
                List.of(
                        BenchmarkEntry.baseline(
                                "cpu-array-training-sgd",
                                trainingOptimizerProfile("cpu-array-training-sgd", CpuStorageProfile.CPU_ARRAY)
                        ),
                        BenchmarkEntry.candidate(
                                "cpu-native-training-sgd",
                                trainingOptimizerProfile("cpu-native-training-sgd", CpuStorageProfile.CPU_NATIVE)
                        )
                ),
                new tuning.measure.MeasurementPolicy(
                        0,
                        1,
                        1,
                        true,
                        true,
                        true,
                        true,
                        true,
                        PublicationPolicy.OUTPUT_ONLY,
                        MeasurementExecutionMode.OPTIMIZER_STEP_SGD
                ),
                tuning.validate.ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        )).run();

        assertTrue(report.candidates().stream().allMatch(BenchmarkCandidateReport::success));
        var arrayTrace = findReport(report, "cpu-array-training-sgd").measurement().trace().run().nativeOptimizers();
        var nativeTrace = findReport(report, "cpu-native-training-sgd").measurement().trace().run().nativeOptimizers();
        assertTrue(arrayTrace.stream().anyMatch(trace -> "CPU_ARRAY".equals(trace.route())));
        assertTrue(nativeTrace.stream().anyMatch(trace -> "CPU_NATIVE".equals(trace.route())));
        assertTrue(nativeTrace.stream().allMatch(trace -> "SKIPPED".equals(trace.gradientPublication())));
        assertTrue(nativeTrace.stream().allMatch(trace -> "publication-policy-output-only".equals(trace.publicationSkippedReason())));

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("optimizerUpdate=optimizer=SgdOptimizer route=CPU_NATIVE"));
        assertTrue(text.contains("gradientPublication=SKIPPED"));
        assertTrue(text.contains("publicationSkippedReason=publication-policy-output-only"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"optimizer\": \"SgdOptimizer\""));
        assertTrue(json.contains("\"route\": \"CPU_NATIVE\""));
        assertTrue(json.contains("\"gradientPublication\": \"SKIPPED\""));
        assertTrue(json.contains("\"publicationSkippedReason\": \"publication-policy-output-only\""));
    }

    @Test
    void benchmarkSessionMeasuresNativeAdamOptimizerStepEvidence() {
        TensorRootWorkloadSpec workload = trainingOptimizerWorkload("native_training_adam_evidence");

        BenchmarkReport report = BenchmarkSession.create(new BenchmarkRequest(
                workload,
                List.of(BenchmarkEntry.candidate(
                        "cpu-native-training-adam",
                        trainingOptimizerProfile("cpu-native-training-adam", CpuStorageProfile.CPU_NATIVE)
                )),
                new tuning.measure.MeasurementPolicy(
                        0,
                        1,
                        1,
                        true,
                        true,
                        true,
                        true,
                        true,
                        PublicationPolicy.OUTPUT_ONLY,
                        MeasurementExecutionMode.OPTIMIZER_STEP_ADAM
                ),
                tuning.validate.ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        )).run();

        assertTrue(report.candidates().stream().allMatch(BenchmarkCandidateReport::success));
        var nativeTrace = findReport(report, "cpu-native-training-adam").measurement().trace().run().nativeOptimizers();
        assertTrue(nativeTrace.stream().anyMatch(trace -> "AdamOptimizer".equals(trace.optimizer())
                && "CPU_NATIVE".equals(trace.route())
                && "CPU_NATIVE".equals(trace.optimizerStateStorage())));

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("optimizerUpdate=optimizer=AdamOptimizer route=CPU_NATIVE"));
        assertTrue(text.contains("optimizerStateStorage=CPU_NATIVE"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"optimizer\": \"AdamOptimizer\""));
        assertTrue(json.contains("\"optimizerStateStorage\": \"CPU_NATIVE\""));
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
                config.compile.CompileConfig.inference(),
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
                        config.compile.CompileConfig.inference(),
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
                        config.compile.CompileConfig.noGraphOptimizationBaseline(),
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
                        config.compile.CompileConfig.noGraphOptimizationBaseline(),
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
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
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
    void renderersExposeNativeOptimizerTrace() {
        var profile = new ExecutionProfile(
                "native-optimizer-profile",
                "native-optimizer",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                config.compile.CompileConfig.training(),
                config.runtime.RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );
        var report = BenchmarkReport.of(
                "native_optimizer_report",
                List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate("native-optimizer", profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                new graph.execution.trace.ExecutionTrace(
                                        graph.execution.trace.CompileTrace.skipped(),
                                        graph.execution.trace.PrepareTrace.skipped(),
                                        new graph.execution.trace.RunTrace(
                                                ExecutionMode.FORWARD_BACKWARD,
                                                1L,
                                                List.of(),
                                                List.of(),
                                                graph.execution.trace.NativeCpuMemoryTrace.empty(),
                                                List.of(new graph.execution.trace.NativeOptimizerTrace(
                                                        "SgdOptimizer",
                                                        "CPU_NATIVE",
                                                        DataType.FLOAT32,
                                                        7,
                                                        9,
                                                        128,
                                                        "",
                                                        "OUTPUT_ONLY",
                                                        "SKIPPED",
                                                        "NONE",
                                                        "",
                                                        "FALLBACK_TO_ARRAY",
                                                        "CPU_NATIVE",
                                                        "CPU_NATIVE",
                                                        "CPU_NATIVE",
                                                        "CPU_NATIVE",
                                                        "publication-policy-output-only"
                                                ))
                                        )
                                ),
                                new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                        )
                ))
        );

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("nativeOptimizerSummary=updateCount=1 nativeCount=1"));
        assertTrue(text.contains("optimizerUpdate=optimizer=SgdOptimizer route=CPU_NATIVE"));
        assertTrue(text.contains("publicationPolicy=OUTPUT_ONLY"));
        assertTrue(text.contains("gradientPublication=SKIPPED"));
        assertTrue(text.contains("nativeCpuFailurePolicy=FALLBACK_TO_ARRAY"));
        assertTrue(text.contains("parameterResidencyAfter=CPU_NATIVE"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"nativeOptimizers\": [{\"optimizer\": \"SgdOptimizer\", \"route\": \"CPU_NATIVE\""));
        assertTrue(json.contains("\"elementCount\": 128"));
        assertTrue(json.contains("\"publicationPolicy\": \"OUTPUT_ONLY\""));
        assertTrue(json.contains("\"gradientPublication\": \"SKIPPED\""));
        assertTrue(json.contains("\"nativeCpuFailurePolicy\": \"FALLBACK_TO_ARRAY\""));
        assertTrue(json.contains("\"parameterResidencyAfter\": \"CPU_NATIVE\""));
    }

    @Test
    void renderersExposeBackendSelectionCostDiagnostics() {
        var profile = new ExecutionProfile(
                "cost-candidate-profile",
                "cost-candidate",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var summary = new AcceleratorPartitionScoreModel.MaterializationCostSummary(
                "CONSERVATIVE",
                2,
                3072L,
                1536L,
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
                        4096L,
                        256L,
                        "CONSERVATIVE"
                ),
                new graph.execution.trace.PartitionDecisionTrace.CandidateCostTrace(
                        List.of(9),
                        "not-selected-lower-score",
                        120.0d,
                        1,
                        2048L,
                        0L,
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
                                        256L,
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
        var optimizerTrace = new graph.optimizer.state.OptimizerTrace(
                List.of("cleanup-cost iteration=1 reason=cleanup-improved"),
                List.of(graph.optimizer.cost.CostScore.of(
                        "GraphCleanupCostModel",
                        "optimizer-cleanup-graph",
                        List.of(graph.optimizer.cost.CostComponent.lowerIsBetter(
                                "weightedOperationCost",
                                12.0d,
                                "lexicographic cleanup priority"
                        ))
                ).explain("cleanup-improved", graph.optimizer.cost.CostComparison.IMPROVED))
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
                                                graph.execution.trace.PartitionCompileTrace.empty(),
                                                optimizerTrace
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
        assertTrue(text.contains("layoutFallbackBytes=1536"));
        assertTrue(text.contains("estimatedComputeWork=8192"));
        assertTrue(text.contains("reason=selected"));
        assertTrue(text.contains("cost: model=AcceleratorPartitionCostModel"));
        assertTrue(text.contains("input=accelerator-partition-materialization"));
        assertTrue(text.contains("reason=accepted-static-profitable"));
        assertTrue(text.contains("finalScore=7780.000000 HIGHER_IS_BETTER"));
        assertTrue(text.contains("optimizerCost:"));
        assertTrue(text.contains("model=GraphCleanupCostModel"));
        assertTrue(text.contains("input=optimizer-cleanup-graph"));
        assertTrue(text.contains("reason=cleanup-improved"));
        assertTrue(text.contains("rejectedFinalists:"));
        assertTrue(text.contains("reason=rejected-materialization-cost"));
        assertTrue(text.contains("input=accelerator-partition-finalist"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"backendSelectionCost\":"));
        assertTrue(json.contains("\"optimizerCost\":"));
        assertTrue(json.contains("\"events\": [\"cleanup-cost iteration=1 reason=cleanup-improved\"]"));
        assertTrue(json.contains("\"model\": \"GraphCleanupCostModel\""));
        assertTrue(json.contains("\"input_kind\": \"optimizer-cleanup-graph\""));
        assertTrue(json.contains("\"comparison\": \"IMPROVED\""));
        assertTrue(json.contains("\"selected\": ["));
        assertTrue(json.contains("\"rejectedFinalists\": ["));
        assertTrue(json.contains("\"nodeIds\": [4, 5, 6]"));
        assertTrue(json.contains("\"selectedBackend\": \"GPU_METAL\""));
        assertTrue(json.contains("\"reason\": \"selected\""));
        assertTrue(json.contains("\"finalScore\": 7780.000000"));
        assertTrue(json.contains("\"boundaryCount\": 2"));
        assertTrue(json.contains("\"estimatedTransferBytes\": 3072"));
        assertTrue(json.contains("\"layoutFallbackBytes\": 1536"));
        assertTrue(json.contains("\"estimatedComputeWork\": 8192"));
        assertTrue(json.contains("\"preset\": \"CONSERVATIVE\""));
        assertTrue(json.contains("\"cost_explanation\":"));
        assertTrue(json.contains("\"model\": \"AcceleratorPartitionCostModel\""));
        assertTrue(json.contains("\"input_kind\": \"accelerator-partition-materialization\""));
        assertTrue(json.contains("\"reason\": \"accepted-static-profitable\""));
        assertTrue(json.contains("\"name\": \"finalScore\""));
        assertTrue(json.contains("\"direction\": \"HIGHER_IS_BETTER\""));
        assertTrue(json.contains("\"reason\": \"rejected-materialization-cost\""));
        assertTrue(json.contains("\"input_kind\": \"accelerator-partition-finalist\""));
        assertTrue(json.contains("\"nodeIds\": [12]"));
        assertTrue(json.contains("\"reason\": \"estimated-work-below-minimum\""));
    }

    @Test
    void renderersExposeMetalRouteCostDiagnosticsFromStepMetadata() {
        var profile = new ExecutionProfile(
                "metal-route-cost-profile",
                "metal-route-cost",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var attrs = Map.<String, Object>ofEntries(
                Map.entry("metalBridgeAvailable", true),
                Map.entry("metalExecutionPath", "BUFFER_BINDING"),
                Map.entry("metalUsedCpuFallback", false),
                Map.entry("metalInputBytes", 1024L),
                Map.entry("metalOutputBytes", 1024L),
                Map.entry("metalJavaToNativeCopyNs", 10L),
                Map.entry("metalNativeExecuteNs", 20L),
                Map.entry("metalNativeToJavaCopyNs", 30L),
                Map.entry("metalRouteCostModel", "MetalBackendRouteCostModel"),
                Map.entry("metalRouteCostInputKind", "metal-prepared-execution-route"),
                Map.entry("metalRouteCostReason", "MPS_GRAPH_SELECTED"),
                Map.entry("metalRouteCostComparison", "INCOMPARABLE"),
                Map.entry("metalRouteCostTopContributors", List.of("estimatedRouteCost=20.000000 LOWER_IS_BETTER")),
                Map.entry("metalRouteCostComponents", List.of("estimatedRouteCost=20.000000 LOWER_IS_BETTER"))
        );
        var step = new graph.execution.trace.ExecutionStepTrace(
                0,
                "metal-step",
                "MATMUL",
                List.of(16, 16),
                DataType.FLOAT32,
                "GPU_METAL",
                "PreparedMetalExecutable",
                20L,
                new graph.execution.trace.StepExecutionMetadata("node", attrs, null, null, null, null, null, null, null)
        );
        BenchmarkReport report = BenchmarkReport.of(
                "metal_route_cost_report",
                List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate("metal-route-cost", profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                new graph.execution.trace.ExecutionTrace(
                                        graph.execution.trace.CompileTrace.skipped(),
                                        graph.execution.trace.PrepareTrace.skipped(),
                                        new graph.execution.trace.RunTrace(ExecutionMode.FORWARD, 20L, List.of(step))
                                ),
                                new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                        )
                ))
        );

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("metalRouteCost=MetalBackendRouteCostModel/MPS_GRAPH_SELECTED"));
        assertTrue(text.contains("metalRouteCost: model=MetalBackendRouteCostModel"));
        assertTrue(text.contains("input=metal-prepared-execution-route"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"metalRouteCostExplanation\":"));
        assertTrue(json.contains("\"model\": \"MetalBackendRouteCostModel\""));
        assertTrue(json.contains("\"input_kind\": \"metal-prepared-execution-route\""));
        assertTrue(json.contains("\"reason\": \"MPS_GRAPH_SELECTED\""));
        assertTrue(json.contains("\"top_contributors\": [\"estimatedRouteCost=20.000000 LOWER_IS_BETTER\"]"));
    }

    @Test
    void renderersExposeMatmulBfloat16OpenBlasRouteEvidence() {
        var profile = new ExecutionProfile(
                "bf16-openblas-route-profile",
                "bf16-openblas-route",
                DataType.BFLOAT16,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var matMul = new graph.execution.trace.MatMulTraceMetadata(
                true,
                false,
                "OPENBLAS_FFM",
                "cblas_sbgemm",
                "OPENBLAS_ARRAY_COPYING",
                "OPENBLAS_ARRAY_COPYING",
                "CPU_ARRAY",
                "FALLBACK_TO_ARRAY",
                "CPU_ARRAY",
                "CPU_ARRAY",
                "forced native route unsupported shape",
                true,
                true,
                true,
                false,
                "SBGEMM",
                "PROMOTED_F32",
                "F32_PROMOTED",
                "F32",
                4096L,
                8192L,
                -1L,
                "AUTO_UNCONTROLLED",
                "",
                true,
                8,
                8,
                4,
                4,
                8192L,
                "F32_4X2"
        );
        var step = new graph.execution.trace.ExecutionStepTrace(
                0,
                "bf16_matmul",
                "MATMUL",
                List.of(32, 32),
                DataType.BFLOAT16,
                "CPU",
                "BF16BlasMatMulExecutable",
                100L,
                new graph.execution.trace.StepExecutionMetadata(
                        "node",
                        Map.of(),
                        null,
                        null,
                        null,
                        null,
                        matMul,
                        null,
                        null
                )
        );
        var promotedNonBlasStep = new graph.execution.trace.ExecutionStepTrace(
                1,
                "bf16_relu",
                "RELU",
                List.of(32, 32),
                DataType.BFLOAT16,
                "CPU",
                "NativeCpuElementwiseExecutor",
                20L,
                new graph.execution.trace.StepExecutionMetadata(
                        "node",
                        Map.of(
                                "cpuStorageProfile", "CPU_NATIVE",
                                "requestedCpuStorage", "CPU_NATIVE",
                                "actualCpuStorage", "CPU_NATIVE",
                                "nativeCpuKernelStatus", "NATIVE_CORRECT_BUT_SLOW",
                                "nativeCpuKernelFamily", "SEGMENT_SCALAR",
                                "storagePrecision", "BF16",
                                "computePrecision", "F32_PROMOTED"
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
        var promotedRegionStep = new graph.execution.trace.ExecutionStepTrace(
                2,
                "bf16_native_region",
                "MATMUL",
                List.of(32, 32),
                DataType.BFLOAT16,
                "CPU",
                "PreparedNativeCpuRegionExecutable",
                40L,
                new graph.execution.trace.StepExecutionMetadata(
                        "node",
                        Map.of(
                                "nativeCpuRegionDecision", "SELECTED",
                                "nativeCpuRegionRoute", "NATIVE",
                                "nativeCpuRegionBf16PromotedNodes", List.of(22, 23),
                                "nativeCpuRegionBf16PromotedSegmentScalarNodes", List.of(22, 23),
                                "nativeCpuRegionBf16StoragePrecision", "BF16",
                                "nativeCpuRegionBf16ComputePrecision", "F32_PROMOTED"
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
                "bf16_openblas_route_report",
                List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate("bf16-openblas-route", profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                new graph.execution.trace.ExecutionTrace(
                                        graph.execution.trace.CompileTrace.skipped(),
                                        graph.execution.trace.PrepareTrace.skipped(),
                                        new graph.execution.trace.RunTrace(
                                                ExecutionMode.FORWARD,
                                                100L,
                                                List.of(step, promotedNonBlasStep, promotedRegionStep),
                                                List.of(),
                                                List.of(),
                                                graph.execution.trace.NativeCpuMemoryTrace.empty(),
                                                List.of(
                                                        new graph.execution.trace.NativeOptimizerTrace(
                                                                "SgdOptimizer",
                                                                "CPU_ARRAY",
                                                                DataType.BFLOAT16,
                                                                10,
                                                                11,
                                                                1024,
                                                                "native-sgd-ineligible:bf16-policy-ACTIVATIONS_ONLY",
                                                                "OUTPUT_ONLY",
                                                                "SKIPPED",
                                                                "NONE",
                                                                "ACTIVATIONS_ONLY",
                                                                "FALLBACK_TO_ARRAY",
                                                                "CPU_ARRAY",
                                                                "CPU_ARRAY",
                                                                "CPU_ARRAY",
                                                                "CPU_ARRAY",
                                                                "publication-policy-output-only"
                                                        ),
                                                        new graph.execution.trace.NativeOptimizerTrace(
                                                                "SgdOptimizer",
                                                                "CPU_ARRAY",
                                                                DataType.BFLOAT16,
                                                                10,
                                                                11,
                                                                1024,
                                                                "native-sgd-ineligible:bf16-master-not-implemented",
                                                                "OUTPUT_ONLY",
                                                                "SKIPPED",
                                                                "NONE",
                                                                "PARAMS_WITH_F32_MASTER",
                                                                "FALLBACK_TO_ARRAY",
                                                                "CPU_ARRAY",
                                                                "CPU_ARRAY",
                                                                "CPU_ARRAY",
                                                                "CPU_ARRAY",
                                                                "publication-policy-output-only"
                                                        )
                                                )
                                        )
                                ),
                                new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                        )
                ))
        );

        Bf16PerformanceBenchmarkGate.requirePass(report);

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("bf16PerformanceSummary=matMulStepCount=1"));
        assertTrue(text.contains("sbgemmContinuationCount=1"));
        assertTrue(text.contains("promotedF32Count=1"));
        assertTrue(text.contains("promotedNonBlasStepCount=1"));
        assertTrue(text.contains("promotedNonBlasRegionNodeCount=2"));
        assertTrue(text.contains("promotedNonBlasSegmentScalarCount=2"));
        assertTrue(text.contains("promotedNonBlasArrayFallbackCount=0"));
        assertTrue(text.contains("optimizerArrayFallbackCount=2"));
        assertTrue(text.contains("optimizerNativeCount=0"));
        assertTrue(text.contains("activationsOnlyPolicyCount=1"));
        assertTrue(text.contains("f32MasterPolicyCount=1"));
        assertTrue(text.contains("experimentalPolicyCount=0"));
        assertTrue(text.contains("native-sgd-ineligible:bf16-policy-ACTIVATIONS_ONLY"));
        assertTrue(text.contains("native-sgd-ineligible:bf16-master-not-implemented"));
        assertTrue(text.contains("openblasSbgemmAvailable=true"));
        assertTrue(text.contains("openblasBgemmAvailable=false"));
        assertTrue(text.contains("bf16ContinuationRoute=SBGEMM"));
        assertTrue(text.contains("bf16OutputRoute=PROMOTED_F32"));
        assertTrue(text.contains("bf16ComputePrecision=F32_PROMOTED"));
        assertTrue(text.contains("bf16OutputPrecision=F32"));
        assertTrue(text.contains("cpuStorageProfile=CPU_ARRAY"));
        assertTrue(text.contains("nativeCpuFailurePolicy=FALLBACK_TO_ARRAY"));
        assertTrue(text.contains("requestedCpuStorage=CPU_ARRAY"));
        assertTrue(text.contains("actualCpuStorage=CPU_ARRAY"));
        assertTrue(text.contains("nativeCpuFallbackReason=forced native route unsupported shape"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"bf16Performance\": {\"matMulStepCount\": 1"));
        assertTrue(json.contains("\"sbgemmContinuationCount\": 1"));
        assertTrue(json.contains("\"promotedF32Count\": 1"));
        assertTrue(json.contains("\"promotedNonBlasStepCount\": 1"));
        assertTrue(json.contains("\"promotedNonBlasRegionNodeCount\": 2"));
        assertTrue(json.contains("\"promotedNonBlasSegmentScalarCount\": 2"));
        assertTrue(json.contains("\"promotedNonBlasArrayFallbackCount\": 0"));
        assertTrue(json.contains("\"optimizerArrayFallbackCount\": 2"));
        assertTrue(json.contains("\"optimizerNativeCount\": 0"));
        assertTrue(json.contains("\"activationsOnlyPolicyCount\": 1"));
        assertTrue(json.contains("\"f32MasterPolicyCount\": 1"));
        assertTrue(json.contains("\"experimentalPolicyCount\": 0"));
        assertTrue(json.contains("\"native-sgd-ineligible:bf16-policy-ACTIVATIONS_ONLY\""));
        assertTrue(json.contains("\"native-sgd-ineligible:bf16-master-not-implemented\""));
        assertTrue(json.contains("\"openblasSbgemmAvailable\": true"));
        assertTrue(json.contains("\"openblasBgemmAvailable\": false"));
        assertTrue(json.contains("\"bf16ContinuationRoute\": \"SBGEMM\""));
        assertTrue(json.contains("\"bf16OutputRoute\": \"PROMOTED_F32\""));
        assertTrue(json.contains("\"bf16ComputePrecision\": \"F32_PROMOTED\""));
        assertTrue(json.contains("\"bf16OutputPrecision\": \"F32\""));
        assertTrue(json.contains("\"cpuStorageProfile\": \"CPU_ARRAY\""));
        assertTrue(json.contains("\"nativeCpuFailurePolicy\": \"FALLBACK_TO_ARRAY\""));
        assertTrue(json.contains("\"requestedCpuStorage\": \"CPU_ARRAY\""));
        assertTrue(json.contains("\"actualCpuStorage\": \"CPU_ARRAY\""));
        assertTrue(json.contains("\"nativeCpuFallbackReason\": \"forced native route unsupported shape\""));
    }

    @Test
    void bf16PerformanceGateRejectsSbgemmOverclaimedAsNativeBf16Output() {
        var profile = new ExecutionProfile(
                "bf16-overclaim-profile",
                "bf16-overclaim",
                DataType.BFLOAT16,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var matMul = new graph.execution.trace.MatMulTraceMetadata(
                true,
                false,
                "OPENBLAS_FFM",
                "cblas_sbgemm",
                "OPENBLAS_ARRAY_COPYING",
                "OPENBLAS_ARRAY_COPYING",
                "CPU_ARRAY",
                "FALLBACK_TO_ARRAY",
                "CPU_ARRAY",
                "CPU_ARRAY",
                "",
                true,
                true,
                true,
                false,
                "SBGEMM",
                "BGEMM",
                "BF16_OUTPUT",
                "BF16",
                4096L,
                4096L,
                -1L,
                "AUTO_UNCONTROLLED",
                "",
                true,
                8,
                8,
                4,
                4,
                8192L,
                "F32_4X2"
        );
        var step = new graph.execution.trace.ExecutionStepTrace(
                0,
                "bf16_overclaim",
                "MATMUL",
                List.of(32, 32),
                DataType.BFLOAT16,
                "CPU",
                "BF16BlasMatMulExecutable",
                100L,
                new graph.execution.trace.StepExecutionMetadata(
                        "node",
                        Map.of(),
                        null,
                        null,
                        null,
                        null,
                        matMul,
                        null,
                        null
                )
        );
        BenchmarkReport report = BenchmarkReport.of(
                "bf16_overclaim_report",
                List.of(BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate("bf16-overclaim", profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                new graph.execution.trace.ExecutionTrace(
                                        graph.execution.trace.CompileTrace.skipped(),
                                        graph.execution.trace.PrepareTrace.skipped(),
                                        new graph.execution.trace.RunTrace(ExecutionMode.FORWARD, 100L, List.of(step))
                                ),
                                new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                        )
                ))
        );

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> Bf16PerformanceBenchmarkGate.requirePass(report)
        );
        assertTrue(failure.getMessage().contains("sbgemm overclaimed as BF16 output route"));
    }

    @Test
    void bf16PerformanceGateRejectsNonBlasPromotedStepWithoutPrecisionContract() {
        var profile = new ExecutionProfile(
                "bf16-non-blas-contract-profile",
                "bf16-non-blas-contract",
                DataType.BFLOAT16,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var step = new graph.execution.trace.ExecutionStepTrace(
                0,
                "bf16_bad_relu",
                "RELU",
                List.of(2, 2),
                DataType.BFLOAT16,
                "CPU",
                "NativeCpuElementwiseExecutor",
                100L,
                new graph.execution.trace.StepExecutionMetadata(
                        "node",
                        Map.of(
                                "actualCpuStorage", "CPU_NATIVE",
                                "nativeCpuKernelStatus", "NATIVE_CORRECT_BUT_SLOW",
                                "nativeCpuKernelFamily", "SEGMENT_SCALAR",
                                "storagePrecision", "BF16"
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
                "bf16_non_blas_contract_report",
                List.of(BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate("bf16-non-blas-contract", profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                new graph.execution.trace.ExecutionTrace(
                                        graph.execution.trace.CompileTrace.skipped(),
                                        graph.execution.trace.PrepareTrace.skipped(),
                                        new graph.execution.trace.RunTrace(ExecutionMode.FORWARD, 100L, List.of(step))
                                ),
                                new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                        )
                ))
        );

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> Bf16PerformanceBenchmarkGate.requirePass(report)
        );

        assertTrue(failure.getMessage().contains("BF16 non-BLAS promoted step missing BF16/F32_PROMOTED"));
    }

    @Test
    void renderersExposeNativeCpuNonBlasPolicyEvidenceFromStepAttributes() {
        var profile = new ExecutionProfile(
                "native-cpu-non-blas-profile",
                "native-cpu-non-blas",
                DataType.BFLOAT16,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var attrs = Map.<String, Object>of(
                "cpuStorageProfile", "CPU_NATIVE",
                "nativeCpuFailurePolicy", "FALLBACK_TO_ARRAY",
                "requestedCpuStorage", "CPU_NATIVE",
                "actualCpuStorage", "CPU_NATIVE",
                "nativeCpuKernelStatus", "NATIVE_CORRECT_BUT_SLOW",
                "nativeCpuKernelFamily", "SEGMENT_SCALAR",
                "nativeCpuFallbackReason", "",
                "storagePrecision", "BF16",
                "computePrecision", "F32_PROMOTED"
        );
        var step = new graph.execution.trace.ExecutionStepTrace(
                0,
                "bf16_promoted_add",
                "ADD",
                List.of(2, 2),
                DataType.BFLOAT16,
                "CPU",
                "NativeCpuElementwiseExecutor",
                100L,
                new graph.execution.trace.StepExecutionMetadata(
                        "node",
                        attrs,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null
                )
        );
        var fallbackAttrs = Map.<String, Object>of(
                "cpuStorageProfile", "CPU_NATIVE",
                "nativeCpuFailurePolicy", "FALLBACK_TO_ARRAY",
                "requestedCpuStorage", "CPU_NATIVE",
                "actualCpuStorage", "CPU_ARRAY",
                "nativeCpuKernelStatus", "NATIVE_UNSUPPORTED",
                "nativeCpuKernelFamily", "ARRAY_ONLY",
                "nativeCpuFallbackReason", "native-kernel-unsupported:erf"
        );
        var fallbackStep = new graph.execution.trace.ExecutionStepTrace(
                1,
                "unsupported_erf",
                "ERF",
                List.of(2, 2),
                DataType.FLOAT32,
                "CPU",
                "CpuErfKernel",
                200L,
                new graph.execution.trace.StepExecutionMetadata(
                        "node",
                        fallbackAttrs,
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
                "native_cpu_non_blas_report",
                List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate("native-cpu-non-blas", profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                new graph.execution.trace.ExecutionTrace(
                                        graph.execution.trace.CompileTrace.skipped(),
                                        graph.execution.trace.PrepareTrace.skipped(),
                                        new graph.execution.trace.RunTrace(ExecutionMode.FORWARD, 100L, List.of(step, fallbackStep))
                                ),
                                new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                        )
                ))
        );

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("cpuStorageProfile=CPU_NATIVE"));
        assertTrue(text.contains("nativeCpuFailurePolicy=FALLBACK_TO_ARRAY"));
        assertTrue(text.contains("nativeCpuKernelStatus=NATIVE_CORRECT_BUT_SLOW"));
        assertTrue(text.contains("nativeCpuKernelFamily=SEGMENT_SCALAR"));
        assertTrue(text.contains("computePrecision=F32_PROMOTED"));
        assertTrue(text.contains("nativeCpuSummary=nativeKernelCount=1 arrayKernelCount=1 fallbackCount=1"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"cpuStorageProfile\": \"CPU_NATIVE\""));
        assertTrue(json.contains("\"nativeCpuFailurePolicy\": \"FALLBACK_TO_ARRAY\""));
        assertTrue(json.contains("\"nativeCpuKernelStatus\": \"NATIVE_CORRECT_BUT_SLOW\""));
        assertTrue(json.contains("\"nativeCpuKernelFamily\": \"SEGMENT_SCALAR\""));
        assertTrue(json.contains("\"computePrecision\": \"F32_PROMOTED\""));
        assertTrue(json.contains("\"nativeCpu\": {\"nativeKernelCount\": 1, \"arrayKernelCount\": 1, \"fallbackCount\": 1}"));
    }

    @Test
    void renderersAggregateNativeCpuRegionEvidenceFromStepAttributes() {
        var profile = new ExecutionProfile(
                "native-cpu-region-profile",
                "native-cpu-region",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var selectedAttrs = Map.<String, Object>ofEntries(
                Map.entry("nativeCpuRegionDecision", "SELECTED"),
                Map.entry("nativeCpuRegionRoute", "NATIVE"),
                Map.entry("nativeCpuRegionReason", "selected"),
                Map.entry("nativeCpuRegionFallbackReason", ""),
                Map.entry("nativeCpuRegionProviderNodes", List.of(2)),
                Map.entry("nativeCpuRegionLocalKernelNodes", List.of(3)),
                Map.entry("nativeCpuRegionSegmentScalarNodes", List.of(3)),
                Map.entry("nativeCpuRegionPhysicalKernels", List.of("OPENBLAS_NATIVE_SEGMENT", "SEGMENT_SCALAR")),
                Map.entry("nativeCpuRegionSegmentKernelFamilies", List.of("PROVIDER", "SEGMENT_DENSE_SCALAR")),
                Map.entry("nativeCpuLayoutClassCounts", Map.of("DENSE_CONTIGUOUS", 2)),
                Map.entry("nativeCpuParityStoragePaths", List.of(
                        List.of("CPU_ARRAY_DENSE", "CPU_ARRAY_STRIDED", "CPU_NATIVE_REGION_PROVIDER"),
                        List.of("CPU_ARRAY_DENSE", "CPU_ARRAY_STRIDED", "CPU_NATIVE_REGION_DENSE")
                )),
                Map.entry("nativeCpuParityLayoutCapabilities", List.of(
                        List.of("DENSE"),
                        List.of("DENSE", "OFFSET_CONTIGUOUS")
                )),
                Map.entry("nativeCpuParityResultResidencies", List.of(
                        List.of("CPU_NATIVE"),
                        List.of("CPU_NATIVE")
                )),
                Map.entry("nativeCpuParityAutoEligible", List.of(true, false)),
                Map.entry("nativeCpuRegionMeasuredWin", true),
                Map.entry("nativeCpuRegionNativeMedianMs", 0.90d),
                Map.entry("nativeCpuRegionArrayMedianMs", 1.00d),
                Map.entry("nativeCpuRegionMeasuredWinThreshold", 0.95d),
                Map.entry("nativeCpuStridedNodeCount", 0),
                Map.entry("nativeCpuStridedMaterializationCount", 0),
                Map.entry("nativeCpuStridedFallbackReasons", List.of()),
                Map.entry("nativeCpuRegionOutputs", List.of(3))
        );
        var providerOnlyAttrs = Map.<String, Object>ofEntries(
                Map.entry("nativeCpuRegionDecision", "SELECTED"),
                Map.entry("nativeCpuRegionRoute", "NATIVE"),
                Map.entry("nativeCpuRegionReason", "selected"),
                Map.entry("nativeCpuRegionFallbackReason", ""),
                Map.entry("nativeCpuRegionProviderNodes", List.of(5)),
                Map.entry("nativeCpuRegionLocalKernelNodes", List.of()),
                Map.entry("nativeCpuRegionSegmentScalarNodes", List.of()),
                Map.entry("nativeCpuRegionPhysicalKernels", List.of("OPENBLAS_NATIVE_SEGMENT")),
                Map.entry("nativeCpuRegionSegmentKernelFamilies", List.of("PROVIDER")),
                Map.entry("nativeCpuLayoutClassCounts", Map.of("DENSE_CONTIGUOUS", 1)),
                Map.entry("nativeCpuParityAutoEligible", List.of(true)),
                Map.entry("nativeCpuStridedNodeCount", 0),
                Map.entry("nativeCpuStridedMaterializationCount", 0),
                Map.entry("nativeCpuStridedFallbackReasons", List.of()),
                Map.entry("nativeCpuRegionOutputs", List.of(5))
        );
        var scalarAttrs = Map.<String, Object>ofEntries(
                Map.entry("nativeCpuRegionDecision", "SELECTED"),
                Map.entry("nativeCpuRegionRoute", "NATIVE"),
                Map.entry("nativeCpuRegionReason", "selected"),
                Map.entry("nativeCpuRegionFallbackReason", ""),
                Map.entry("nativeCpuRegionProviderNodes", List.of()),
                Map.entry("nativeCpuRegionLocalKernelNodes", List.of(6)),
                Map.entry("nativeCpuRegionSegmentScalarNodes", List.of(6)),
                Map.entry("nativeCpuRegionPhysicalKernels", List.of("SEGMENT_SCALAR")),
                Map.entry("nativeCpuRegionSegmentKernelFamilies", List.of("SEGMENT_STRIDED_SCALAR")),
                Map.entry("nativeCpuLayoutClassCounts", Map.of("STRIDED_VIEW", 1)),
                Map.entry("nativeCpuParityAutoEligible", List.of(false)),
                Map.entry("nativeCpuStridedNodeCount", 1),
                Map.entry("nativeCpuStridedMaterializationCount", 0),
                Map.entry("nativeCpuStridedFallbackReasons", List.of()),
                Map.entry("nativeCpuRegionOutputs", List.of(6))
        );
        var parallelAttrs = Map.<String, Object>ofEntries(
                Map.entry("nativeCpuRegionDecision", "SELECTED"),
                Map.entry("nativeCpuRegionRoute", "NATIVE"),
                Map.entry("nativeCpuRegionReason", "selected"),
                Map.entry("nativeCpuRegionFallbackReason", ""),
                Map.entry("nativeCpuRegionProviderNodes", List.of()),
                Map.entry("nativeCpuRegionLocalKernelNodes", List.of(7)),
                Map.entry("nativeCpuRegionSegmentScalarNodes", List.of()),
                Map.entry("nativeCpuRegionPhysicalKernels", List.of("SEGMENT_PARALLEL")),
                Map.entry("nativeCpuRegionSegmentKernelFamilies", List.of("SEGMENT_PARALLEL")),
                Map.entry("nativeCpuLayoutClassCounts", Map.of("DENSE_CONTIGUOUS", 1)),
                Map.entry("nativeCpuParityAutoEligible", List.of(true)),
                Map.entry("nativeCpuStridedNodeCount", 0),
                Map.entry("nativeCpuStridedMaterializationCount", 0),
                Map.entry("nativeCpuStridedFallbackReasons", List.of()),
                Map.entry("nativeCpuRegionOutputs", List.of(7))
        );
        var fusedAttrs = Map.<String, Object>ofEntries(
                Map.entry("nativeCpuRegionDecision", "SELECTED"),
                Map.entry("nativeCpuRegionRoute", "NATIVE"),
                Map.entry("nativeCpuRegionReason", "selected"),
                Map.entry("nativeCpuRegionFallbackReason", ""),
                Map.entry("nativeCpuRegionProviderNodes", List.of()),
                Map.entry("nativeCpuRegionLocalKernelNodes", List.of(8)),
                Map.entry("nativeCpuRegionSegmentScalarNodes", List.of()),
                Map.entry("nativeCpuRegionPhysicalKernels", List.of("SEGMENT_FUSED")),
                Map.entry("nativeCpuRegionSegmentKernelFamilies", List.of("SEGMENT_FUSED")),
                Map.entry("nativeCpuLayoutClassCounts", Map.of("DENSE_CONTIGUOUS", 1)),
                Map.entry("nativeCpuParityAutoEligible", List.of(true)),
                Map.entry("nativeCpuStridedNodeCount", 0),
                Map.entry("nativeCpuStridedMaterializationCount", 0),
                Map.entry("nativeCpuStridedFallbackReasons", List.of()),
                Map.entry("nativeCpuRegionOutputs", List.of(8))
        );
        var providerFallbackAttrs = Map.<String, Object>ofEntries(
                Map.entry("nativeCpuRegionDecision", "REJECTED"),
                Map.entry("nativeCpuRegionRoute", "CPU_ARRAY"),
                Map.entry("nativeCpuRegionReason", "native-cpu-region-provider-fallback:matmul"),
                Map.entry("nativeCpuRegionFallbackReason", "native-cpu-region-provider-fallback:matmul"),
                Map.entry("nativeCpuRegionProviderNodes", List.of(9)),
                Map.entry("nativeCpuLayoutClassCounts", Map.of("DENSE_CONTIGUOUS", 1)),
                Map.entry("nativeCpuStridedNodeCount", 0),
                Map.entry("nativeCpuStridedMaterializationCount", 0),
                Map.entry("nativeCpuStridedFallbackReasons", List.of()),
                Map.entry("nativeCpuRegionOutputs", List.of(9))
        );
        var stridedRejectedAttrs = Map.<String, Object>ofEntries(
                Map.entry("nativeCpuRegionDecision", "REJECTED"),
                Map.entry("nativeCpuRegionRoute", "CPU_ARRAY"),
                Map.entry("nativeCpuRegionReason", "native-layout-unsupported:negative-stride"),
                Map.entry("nativeCpuRegionFallbackReason", "native-layout-unsupported:negative-stride"),
                Map.entry("nativeCpuLayoutClassCounts", Map.of("NEGATIVE_STRIDE", 1)),
                Map.entry("nativeCpuStridedNodeCount", 1),
                Map.entry("nativeCpuStridedMaterializationCount", 1),
                Map.entry("nativeCpuStridedFallbackReasons", List.of("native-layout-unsupported:negative-stride")),
                Map.entry("nativeCpuRegionOutputs", List.of(10))
        );
        var rejectedAttrs = Map.<String, Object>ofEntries(
                Map.entry("nativeCpuRegionDecision", "REJECTED"),
                Map.entry("nativeCpuRegionRoute", "CPU_ARRAY"),
                Map.entry("nativeCpuRegionReason", "native-cpu-region-provider-unavailable:matmul"),
                Map.entry("nativeCpuRegionFallbackReason", "native-cpu-region-provider-unavailable:matmul"),
                Map.entry("nativeCpuLayoutClassCounts", Map.of("DENSE_CONTIGUOUS", 1)),
                Map.entry("nativeCpuRegionMeasuredWin", true),
                Map.entry("nativeCpuRegionNativeMedianMs", 0.99d),
                Map.entry("nativeCpuRegionArrayMedianMs", 1.00d),
                Map.entry("nativeCpuRegionMeasuredWinThreshold", 0.95d),
                Map.entry("nativeCpuStridedNodeCount", 0),
                Map.entry("nativeCpuStridedMaterializationCount", 0),
                Map.entry("nativeCpuStridedFallbackReasons", List.of()),
                Map.entry("nativeCpuRegionOutputs", List.of(4))
        );
        BenchmarkReport report = BenchmarkReport.of(
                "native_cpu_region_report",
                List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate("native-cpu-region", profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                new graph.execution.trace.ExecutionTrace(
                                        graph.execution.trace.CompileTrace.skipped(),
                                        graph.execution.trace.PrepareTrace.skipped(),
                                        new graph.execution.trace.RunTrace(
                                                ExecutionMode.FORWARD,
                                                100L,
                                                List.of(
                                                        nativeRegionStep(0, "native_provider_local", "PreparedNativeCpuRegionExecutable", selectedAttrs),
                                                        nativeRegionStep(1, "native_provider_only", "PreparedNativeCpuRegionExecutable", providerOnlyAttrs),
                                                        nativeRegionStep(2, "native_segment_scalar", "PreparedNativeCpuRegionExecutable", scalarAttrs),
                                                        nativeRegionStep(3, "native_segment_parallel", "PreparedNativeCpuRegionExecutable", parallelAttrs),
                                                        nativeRegionStep(4, "native_segment_fused", "PreparedNativeCpuRegionExecutable", fusedAttrs),
                                                        nativeRegionStep(5, "native_provider_array_fallback", "CpuMatMulKernel", providerFallbackAttrs),
                                                        nativeRegionStep(6, "array_strided", "CpuStridedKernel", stridedRejectedAttrs),
                                                        nativeRegionStep(7, "array_dense", "CpuMatMulKernel", rejectedAttrs)
                                                )
                                        )
                                ),
                                new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                        )
                ))
        );

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("nativeCpuRegionSummary=selectedRegionCount=5 rejectedRegionCount=3 nativeRouteCount=5 fallbackCount=3"));
        assertTrue(text.contains("measuredWinClaimCount=2 measuredWinProofCount=1"));
        assertTrue(text.contains("providerNodeCount=3 localKernelNodeCount=4 segmentScalarNodeCount=2 stridedNodeCount=2 stridedMaterializationCount=1"));
        assertTrue(text.contains("benchmarkRowCounts="));
        assertTrue(text.contains("native provider + local native=1"));
        assertTrue(text.contains("native provider only=1"));
        assertTrue(text.contains("native segment scalar=1"));
        assertTrue(text.contains("native segment parallel=1"));
        assertTrue(text.contains("native segment fused=1"));
        assertTrue(text.contains("native provider + array fallback=1"));
        assertTrue(text.contains("array strided=1"));
        assertTrue(text.contains("array dense=1"));
        assertTrue(text.contains("layoutClassCounts="));
        assertTrue(text.contains("DENSE_CONTIGUOUS=7"));
        assertTrue(text.contains("STRIDED_VIEW=1"));
        assertTrue(text.contains("NEGATIVE_STRIDE=1"));
        assertTrue(text.contains("boundaryOutputCount=8"));
        assertTrue(text.contains("parityStoragePathCounts="));
        assertTrue(text.contains("CPU_NATIVE_REGION_PROVIDER=1"));
        assertTrue(text.contains("CPU_NATIVE_REGION_DENSE=1"));
        assertTrue(text.contains("parityLayoutCapabilityCounts="));
        assertTrue(text.contains("OFFSET_CONTIGUOUS=1"));
        assertTrue(text.contains("parityResultResidencyCounts={CPU_NATIVE=2}"));
        assertTrue(text.contains("parityAutoEligibleNodeCount=4"));
        assertTrue(text.contains("fallbackReasons=[native-cpu-region-provider-fallback:matmul, native-layout-unsupported:negative-stride, native-cpu-region-provider-unavailable:matmul]"));
        assertTrue(text.contains("stridedFallbackReasons=[native-layout-unsupported:negative-stride]"));
        assertTrue(text.contains("rejectionReasons=[native-cpu-region-provider-fallback:matmul, native-layout-unsupported:negative-stride, native-cpu-region-provider-unavailable:matmul]"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"nativeCpuRegion\": {\"selectedRegionCount\": 5, \"rejectedRegionCount\": 3, \"nativeRouteCount\": 5, \"fallbackCount\": 3"));
        assertTrue(json.contains("\"measuredWinClaimCount\": 2"));
        assertTrue(json.contains("\"measuredWinProofCount\": 1"));
        assertTrue(json.contains("\"providerNodeCount\": 3"));
        assertTrue(json.contains("\"localKernelNodeCount\": 4"));
        assertTrue(json.contains("\"segmentScalarNodeCount\": 2"));
        assertTrue(json.contains("\"stridedNodeCount\": 2"));
        assertTrue(json.contains("\"stridedMaterializationCount\": 1"));
        assertTrue(json.contains("\"benchmarkRowCounts\": {"));
        assertTrue(json.contains("\"native provider + local native\": 1"));
        assertTrue(json.contains("\"native provider only\": 1"));
        assertTrue(json.contains("\"native segment scalar\": 1"));
        assertTrue(json.contains("\"native segment parallel\": 1"));
        assertTrue(json.contains("\"native segment fused\": 1"));
        assertTrue(json.contains("\"native provider + array fallback\": 1"));
        assertTrue(json.contains("\"array strided\": 1"));
        assertTrue(json.contains("\"array dense\": 1"));
        assertTrue(json.contains("\"DENSE_CONTIGUOUS\": 7"));
        assertTrue(json.contains("\"STRIDED_VIEW\": 1"));
        assertTrue(json.contains("\"NEGATIVE_STRIDE\": 1"));
        assertTrue(json.contains("\"parityStoragePathCounts\": {"));
        assertTrue(json.contains("\"CPU_NATIVE_REGION_PROVIDER\": 1"));
        assertTrue(json.contains("\"CPU_NATIVE_REGION_DENSE\": 1"));
        assertTrue(json.contains("\"parityLayoutCapabilityCounts\": {"));
        assertTrue(json.contains("\"OFFSET_CONTIGUOUS\": 1"));
        assertTrue(json.contains("\"parityResultResidencyCounts\": {\"CPU_NATIVE\": 2}"));
        assertTrue(json.contains("\"parityAutoEligibleNodeCount\": 4"));
        assertTrue(json.contains("\"boundaryOutputCount\": 8"));
        assertTrue(json.contains("\"stridedFallbackReasons\": [\"native-layout-unsupported:negative-stride\"]"));
        assertTrue(json.contains("\"rejectionReasons\": [\"native-cpu-region-provider-fallback:matmul\", \"native-layout-unsupported:negative-stride\", \"native-cpu-region-provider-unavailable:matmul\"]"));
    }

    @Test
    void renderersAggregateRuntimeCopyEvidenceFromMaterializationsAndMatmulMetadata() {
        var profile = new ExecutionProfile(
                "runtime-copy-evidence-profile",
                "runtime-copy-evidence",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var matMul = new graph.execution.trace.MatMulTraceMetadata(
                true,
                false,
                "OPENBLAS_FFM",
                "cblas_sgemm",
                "OPENBLAS_NATIVE_SEGMENT",
                "OPENBLAS_ARRAY_COPYING",
                "CPU_NATIVE",
                "FALLBACK_TO_ARRAY",
                "CPU_NATIVE",
                "CPU_ARRAY",
                "native-symbol-unavailable:cblas_sgemm",
                true,
                false,
                false,
                false,
                "",
                "",
                "",
                "",
                8192L,
                4096L,
                1024L,
                "AUTO_UNCONTROLLED",
                "native-symbol-unavailable:cblas_sgemm",
                true,
                16,
                16,
                8,
                4,
                131_072L,
                "F32_8X4"
        );
        var step = new graph.execution.trace.ExecutionStepTrace(
                0,
                "runtime_copy_matmul",
                "MATMUL",
                List.of(32, 32),
                DataType.FLOAT32,
                "CPU",
                "F32BlasMatMulExecutable",
                100L,
                new graph.execution.trace.StepExecutionMetadata(
                        "node",
                        Map.of(),
                        null,
                        null,
                        null,
                        null,
                        matMul,
                        null,
                        null
                )
        );
        var materialization = new graph.execution.trace.CpuMaterializationTrace(
                42,
                CpuMaterializationReason.GRAPH_OUTPUT,
                "GPU_METAL",
                StorageResidency.DEVICE_OWNED,
                4096L,
                250_000L,
                true,
                "device value synchronized to CPU storage"
        );
        BenchmarkReport report = BenchmarkReport.of(
                "runtime_copy_evidence_report",
                List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate("runtime-copy-evidence", profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                new graph.execution.trace.ExecutionTrace(
                                        graph.execution.trace.CompileTrace.skipped(),
                                        graph.execution.trace.PrepareTrace.skipped(),
                                        new graph.execution.trace.RunTrace(
                                                ExecutionMode.FORWARD,
                                                100L,
                                                List.of(step),
                                                List.of(materialization),
                                                List.of(
                                                        new graph.execution.trace.HostDeviceTransferTrace(
                                                                42,
                                                                "GPU_METAL",
                                                                DataType.FLOAT32,
                                                                StorageResidency.CPU_NATIVE,
                                                                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                                                                graph.execution.trace.HostDeviceTransferKind.NATIVE_TO_ARRAY_TO_DEVICE_BRIDGE,
                                                                4096L,
                                                                4096L,
                                                                4096L,
                                                                4096L,
                                                                125_000L,
                                                                false,
                                                                false,
                                                                true,
                                                                "native-device-direct-transfer-unavailable",
                                                                "metal shared input buffer upload"
                                                        ),
                                                        new graph.execution.trace.HostDeviceTransferTrace(
                                                                43,
                                                                "GPU_METAL",
                                                                DataType.FLOAT32,
                                                                StorageResidency.CPU_NATIVE,
                                                                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                                                                graph.execution.trace.HostDeviceTransferKind.NATIVE_SEGMENT_TO_DEVICE_COPY,
                                                                2048L,
                                                                0L,
                                                                2048L,
                                                                2048L,
                                                                50_000L,
                                                                false,
                                                                true,
                                                                true,
                                                                "",
                                                                "metal native segment input buffer upload"
                                                        )
                                                ),
                                                new graph.execution.trace.NativeCpuMemoryTrace(
                                                        3L,
                                                        0L,
                                                        0L,
                                                        0L,
                                                        12_288L,
                                                        12_288L,
                                                        12_288L,
                                                        12_288L,
                                                        0L
                                                ),
                                                List.of()
                                        )
                                ),
                                new tuning.measure.MeasurementStatistics(1.0, 1.0, 1.0)
                        )
                ))
        );

        assertFalse(matMul.nativeCpuFallbackReason().isBlank());
        assertEquals(CpuMaterializationReason.GRAPH_OUTPUT, materialization.reason());
        assertEquals(StorageResidency.DEVICE_OWNED, materialization.sourceResidency());
        assertEquals(4096L, materialization.bytes());
        assertTrue(materialization.completed());

        String text = TextBenchmarkReportRenderer.render(report);
        assertTrue(text.contains("runtimeCopySummary=cpuMaterializationBytes=4096"));
        assertTrue(text.contains("cpuMaterializationDurationNs=250000"));
        assertTrue(text.contains("matMulCopyInBytes=8192"));
        assertTrue(text.contains("matMulCopyOutBytes=4096"));
        assertTrue(text.contains("matMulNativeTempBytes=1024"));
        assertTrue(text.contains("nativeCpuMemory=allocationCount=3"));
        assertTrue(text.contains("hostDeviceTransferSummary=transferCount=2 bytes=6144 javaArrayBytes=4096 nativeBytes=6144 deviceBytes=6144 fallbackCount=1"));
        assertTrue(text.contains("kind=NATIVE_TO_ARRAY_TO_DEVICE_BRIDGE"));
        assertTrue(text.contains("kind=NATIVE_SEGMENT_TO_DEVICE_COPY"));
        assertTrue(text.contains("javaArrayBytes=0"));
        assertTrue(text.contains("fallbackReason=native-device-direct-transfer-unavailable"));
        assertTrue(text.contains("requestedBytes=12288"));
        assertTrue(text.contains("peakLiveBytes=12288"));
        assertTrue(text.contains("poolHitCount=0"));
        assertTrue(text.contains("wastedBytes=0"));
        assertTrue(text.contains("nodeId=42 reason=GRAPH_OUTPUT from=GPU_METAL residency=DEVICE_OWNED bytes=4096"));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"runtimeCopy\": {\"cpuMaterializationBytes\": 4096, \"cpuMaterializationDurationNs\": 250000, \"matMulCopyInBytes\": 8192, \"matMulCopyOutBytes\": 4096, \"matMulNativeTempBytes\": 1024}"));
        assertTrue(json.contains("\"hostDeviceTransfer\": {\"transferCount\": 2, \"bytes\": 6144, \"javaArrayBytes\": 4096, \"nativeBytes\": 6144, \"deviceBytes\": 6144, \"fallbackCount\": 1}"));
        assertTrue(json.contains("\"transferKind\": \"NATIVE_TO_ARRAY_TO_DEVICE_BRIDGE\""));
        assertTrue(json.contains("\"transferKind\": \"NATIVE_SEGMENT_TO_DEVICE_COPY\""));
        assertTrue(json.contains("\"javaArrayBytes\": 0"));
        assertTrue(json.contains("\"fallbackReason\": \"native-device-direct-transfer-unavailable\""));
        assertTrue(json.contains("\"nativeCpuMemory\": {\"allocationCount\": 3, \"releaseCount\": 0, \"retainCount\": 0, \"allocationFailureCount\": 0, \"requestedPoolPolicy\": \"\", \"effectivePoolPolicy\": \"\", \"requestedBytes\": 12288, \"allocatedBytes\": 12288, \"currentLiveBytes\": 12288, \"peakLiveBytes\": 12288, \"retainedBytes\": 0, \"poolHitCount\": 0, \"poolMissCount\": 0, \"pooledBytes\": 0, \"reusedBytes\": 0, \"discardedBytes\": 0, \"wastedBytes\": 0}"));
        assertTrue(json.contains("\"nativeCpuFallbackReason\": \"native-symbol-unavailable:cblas_sgemm\""));
        assertTrue(json.contains("\"materializedFrom\": \"GPU_METAL\""));
        assertTrue(json.contains("\"sourceResidency\": \"DEVICE_OWNED\""));
        assertTrue(json.contains("\"completed\": true"));
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
        assertTrue(text.contains("UNSUPPORTED_INDEX_SEMANTICS"));
        assertTrue(text.contains("target=layer_norm_small"));
        assertTrue(text.contains("target=transformer_block_hot_path"));
        assertTrue(json.contains("LOG_SOFTMAX"));
        assertTrue(json.contains("SOFTMAX"));
        assertTrue(json.contains("family=NORMALIZATION"));
        assertTrue(json.contains("family=LOSS_ADJACENT"));
        assertTrue(json.contains("UNSUPPORTED_INDEX_SEMANTICS"));
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
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
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
                                Map.entry("metalNativeCopyStrategy", "MPSGRAPH_RESULT_COPY"),
                                Map.entry("metalOutputBufferWriteStatus", "COPY_REQUIRED"),
                                Map.entry("metalExecutionRoute", "MPS_GRAPH"),
                                Map.entry("metalRouteRejectedReasonCodes", List.of("CUSTOM_KERNEL_UNAVAILABLE")),
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
        assertTrue(text.contains("nativeCopyStrategies=[MPSGRAPH_RESULT_COPY]"));
        assertTrue(text.contains("outputBufferWriteStatuses=[COPY_REQUIRED]"));
        assertTrue(text.contains("executionRouteCounts={MPS_GRAPH=1}"));
        assertTrue(text.contains("rejectedRouteReasonCounts={CUSTOM_KERNEL_UNAVAILABLE=1}"));
        assertTrue(text.contains("routerEvidence:"));
        assertTrue(text.contains("backendRouteCounts={MPS_GRAPH=1}"));
        assertTrue(text.contains("outputBufferWriteStatusCounts={COPY_REQUIRED=1}"));
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
        assertTrue(json.contains("\"nativeCopyStrategies\": [\"MPSGRAPH_RESULT_COPY\"]"));
        assertTrue(json.contains("\"outputBufferWriteStatuses\": [\"COPY_REQUIRED\"]"));
        assertTrue(json.contains("\"executionRouteCounts\": {\"MPS_GRAPH\": 1}"));
        assertTrue(json.contains("\"rejectedRouteReasonCounts\": {\"CUSTOM_KERNEL_UNAVAILABLE\": 1}"));
        assertTrue(json.contains("\"routerEvidence\":"));
        assertTrue(json.contains("\"backendRouteCounts\": {\"MPS_GRAPH\": 1}"));
        assertTrue(json.contains("\"outputBufferWriteStatusCounts\": {\"COPY_REQUIRED\": 1}"));
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
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
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
        assertTrue(text.contains("routerEvidence:"));
        assertTrue(text.contains("acceleratorPathCounts={BUFFER_BINDING=1}"));
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
        assertTrue(json.contains("\"routerEvidence\":"));
        assertTrue(json.contains("\"acceleratorPathCounts\": {\"BUFFER_BINDING\": 1}"));
        assertTrue(json.contains("\"inputBytes\": 2048"));
        assertTrue(json.contains("\"outputBytes\": 1024"));
        assertTrue(json.contains("\"javaToNativeCopyNs\": 100000"));
        assertTrue(json.contains("\"nativeDeviceCopyNs\": 25000"));
        assertTrue(json.contains("\"storageResidency\": \"DEVICE_OWNED\""));
    }

    private static tuning.measure.MeasurementResult nativeDeviceBridgeMeasurement(
            String candidateName,
            MeasurementPolicy policy
    ) {
        graph.execution.trace.HostDeviceTransferKind kind = switch (candidateName) {
            case NativeDeviceBridgeBenchmark.CPU_ARRAY_METAL ->
                    graph.execution.trace.HostDeviceTransferKind.CPU_ARRAY_TO_DEVICE_COPY;
            case NativeDeviceBridgeBenchmark.CPU_NATIVE_ARRAY_BRIDGE_METAL ->
                    graph.execution.trace.HostDeviceTransferKind.NATIVE_TO_ARRAY_TO_DEVICE_BRIDGE;
            case NativeDeviceBridgeBenchmark.CPU_NATIVE_DIRECT_METAL ->
                    graph.execution.trace.HostDeviceTransferKind.NATIVE_SEGMENT_TO_DEVICE_COPY;
            default -> throw new IllegalArgumentException("unexpected native device bridge candidate " + candidateName);
        };
        return new tuning.measure.MeasurementResult(
                policy,
                new graph.execution.trace.ExecutionTrace(
                        graph.execution.trace.CompileTrace.skipped(),
                        graph.execution.trace.PrepareTrace.skipped(),
                        new graph.execution.trace.RunTrace(
                                ExecutionMode.FORWARD,
                                100_000L,
                                List.of(),
                                List.of(),
                                List.of(nativeDeviceBridgeTransfer(kind)),
                                graph.execution.trace.NativeCpuMemoryTrace.empty(),
                                List.of()
                        )
                ),
                new tuning.measure.MeasurementStatistics(1.2d, 1.0d, 1.4d)
        );
    }

    private static BenchmarkCandidateReport nativeDeviceBridgeCandidate(
            String name,
            graph.execution.trace.HostDeviceTransferKind kind
    ) {
        return BenchmarkCandidateReport.success(
                BenchmarkEntry.candidate(name, NativeDeviceBridgeBenchmark.entries().stream()
                        .filter(entry -> entry.name().equals(name))
                        .findFirst()
                        .orElseThrow()
                        .profile()),
                tuning.validate.ValidationResult.skipped(),
                new tuning.measure.MeasurementResult(
                        NativeDeviceBridgeBenchmark.measurementPolicy(),
                        new graph.execution.trace.ExecutionTrace(
                                graph.execution.trace.CompileTrace.skipped(),
                                graph.execution.trace.PrepareTrace.skipped(),
                                new graph.execution.trace.RunTrace(
                                        ExecutionMode.FORWARD,
                                        100_000L,
                                        List.of(),
                                        List.of(),
                                        List.of(nativeDeviceBridgeTransfer(kind)),
                                        graph.execution.trace.NativeCpuMemoryTrace.empty(),
                                        List.of()
                                )
                        ),
                        new tuning.measure.MeasurementStatistics(1.2d, 1.0d, 1.4d)
                )
        );
    }

    private static graph.execution.trace.HostDeviceTransferTrace nativeDeviceBridgeTransfer(
            graph.execution.trace.HostDeviceTransferKind kind
    ) {
        long bytes = 4096L;
        StorageResidency source = kind == graph.execution.trace.HostDeviceTransferKind.CPU_ARRAY_TO_DEVICE_COPY
                ? StorageResidency.CPU_ARRAY
                : StorageResidency.CPU_NATIVE;
        long javaArrayBytes = kind == graph.execution.trace.HostDeviceTransferKind.NATIVE_SEGMENT_TO_DEVICE_COPY
                ? 0L
                : bytes;
        long nativeBytes = kind == graph.execution.trace.HostDeviceTransferKind.CPU_ARRAY_TO_DEVICE_COPY
                ? 0L
                : bytes;
        boolean directSupported = kind != graph.execution.trace.HostDeviceTransferKind.NATIVE_TO_ARRAY_TO_DEVICE_BRIDGE;
        String fallbackReason = kind == graph.execution.trace.HostDeviceTransferKind.NATIVE_TO_ARRAY_TO_DEVICE_BRIDGE
                ? "native-device-direct-transfer-unavailable"
                : "";
        return new graph.execution.trace.HostDeviceTransferTrace(
                101,
                ComputeBackend.GPU_METAL.name(),
                DataType.FLOAT32,
                source,
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                kind,
                bytes,
                javaArrayBytes,
                nativeBytes,
                bytes,
                25_000L,
                false,
                directSupported,
                true,
                fallbackReason,
                "metal native device bridge benchmark transfer"
        );
    }

    @Test
    void benchmarkSessionReportsClosureTransformerBlockTraceContract() {
        var profile = new ExecutionProfile(
                "accelerator-closure-transformer-block",
                "accelerator-closure-transformer-block",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                config.compile.CompileConfig.training(),
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
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
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
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
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
        assertTrue(text.contains("multiOpGpuRegionCount="));
        assertTrue(text.contains("maxSelectedRegionLength=3"));
        assertTrue(text.contains("averageSelectedRegionLength=3.000000"));
        assertTrue(text.contains("loweredPrimitiveCount="));
        assertTrue(text.contains("rejectedCandidateReasonCounts={unsupported-layout=1}"));
        assertTrue(text.contains("nativeBufferStepCount=1"));
        assertTrue(text.contains("fallbackCount=0"));
        assertTrue(text.contains("tensorArrayStepCount=0"));
        assertTrue(text.contains("cpuFallbackStepCount=0"));
        assertTrue(text.contains("cpuMaterializationReasonCounts={CPU_CONSUMER=1}"));
        assertTrue(text.contains("deviceHandoffCount=2"));
        assertTrue(text.contains("storageResidencyCounts={DEVICE_OWNED=1}"));
        assertTrue(text.contains("dtypeResidencyReasons="));

        String json = JsonBenchmarkReportRenderer.render(report);
        assertTrue(json.contains("\"coverage\""));
        assertTrue(json.contains("\"gpuCoverageRatio\": 0.500000"));
        assertTrue(json.contains("\"selectedRegionCount\": 1"));
        assertTrue(json.contains("\"multiOpGpuRegionCount\""));
        assertTrue(json.contains("\"maxSelectedRegionLength\": 3"));
        assertTrue(json.contains("\"averageSelectedRegionLength\": 3.000000"));
        assertTrue(json.contains("\"loweredPrimitiveCount\""));
        assertTrue(json.contains("\"rejectedCandidateReasonCounts\": {\"unsupported-layout\": 1}"));
        assertTrue(json.contains("\"nativeBufferStepCount\": 1"));
        assertTrue(json.contains("\"fallbackCount\": 0"));
        assertTrue(json.contains("\"tensorArrayStepCount\": 0"));
        assertTrue(json.contains("\"cpuFallbackStepCount\": 0"));
        assertTrue(json.contains("\"cpuMaterializationReasonCounts\": {\"CPU_CONSUMER\": 1}"));
        assertTrue(json.contains("\"copyDurationNs\": 325000"));
        assertTrue(json.contains("\"deviceHandoffCount\": 2"));
        assertTrue(json.contains("\"storageResidencyCounts\": {\"DEVICE_OWNED\": 1}"));
        assertTrue(json.contains("\"dtypeResidencyEvidence\""));
    }

    @Test
    void phaseNineteenBenchmarkReportRendersMultiOpGpuRegionEvidence() {
        BenchmarkReport report = gpuCoverageBenchmarkReport("phase19_multi_op_gpu_region_report");

        String text = TextBenchmarkReportRenderer.render(report);
        String json = JsonBenchmarkReportRenderer.render(report);

        assertTrue(text.contains("multiOpGpuRegionCount="));
        assertTrue(text.contains("maxSelectedRegionLength="));
        assertTrue(text.contains("loweredPrimitiveCount="));
        assertTrue(text.contains("gpuFusedSubpatternCount="));
        assertTrue(text.contains("cpuMaterializationReasonCounts="));
        assertTrue(text.contains("deviceHandoffCount="));
        assertTrue(text.contains("tensorArrayStepCount="));
        assertTrue(text.contains("nativeBufferStepCount="));
        assertTrue(json.contains("\"multiOpGpuRegionCount\""));
        assertTrue(json.contains("\"maxSelectedRegionLength\""));
        assertTrue(json.contains("\"loweredPrimitiveCount\""));
        assertTrue(json.contains("\"gpuFusedSubpatternCount\""));
        assertTrue(json.contains("\"cpuMaterializationReasonCounts\""));
        assertTrue(json.contains("\"deviceHandoffCount\""));
        assertTrue(json.contains("\"tensorArrayStepCount\""));
        assertTrue(json.contains("\"nativeBufferStepCount\""));
    }

    @Test
    void phaseNineteenBenchmarkJsonDistinguishesTensorArrayFromNativeBuffer() {
        BenchmarkReport report = gpuCoverageBenchmarkReport("phase19_gpu_region_runtime_path_report");

        String json = JsonBenchmarkReportRenderer.render(report);

        assertTrue(json.contains("\"nativeBufferStepCount\": 1"));
        assertTrue(json.contains("\"tensorArrayStepCount\": 0"));
    }

    @Test
    void phaseTwentyBenchmarkReportRendersCoverageGateResult() {
        BenchmarkReport report = gpuCoverageBenchmarkReport("phase20_coverage_gate_report");

        String text = TextBenchmarkReportRenderer.render(report);

        assertTrue(text.contains("coverageGate"));
        assertTrue(text.contains("gatePassed="));
        assertTrue(text.contains("gateFailures="));
        assertTrue(text.contains("nativeEvidence"));
        assertTrue(text.contains("nativeStatus="));
    }

    @Test
    void phaseTwentyBenchmarkJsonRendersCoverageGateResult() {
        BenchmarkReport report = gpuCoverageBenchmarkReport("phase20_coverage_gate_json_report");

        String json = JsonBenchmarkReportRenderer.render(report);

        assertTrue(json.contains("\"coverageGate\""));
        assertTrue(json.contains("\"gatePassed\""));
        assertTrue(json.contains("\"gateFailures\""));
        assertTrue(json.contains("\"nativeEvidence\""));
        assertTrue(json.contains("\"nativeStatus\""));
    }

    @Test
    void phaseTwentyNativeEvidenceCanRenderCapabilitySkippedStatus() {
        GpuCoverageNativeEvidence evidence = GpuCoverageNativeEvidence.capabilitySkipped(
                "GPU_CUDA",
                "CUDA device unavailable"
        );

        assertEquals("GPU_CUDA", evidence.backend());
        assertEquals("capabilitySkipped", evidence.nativeStatus());
        assertEquals("CUDA device unavailable", evidence.detail());
    }

    @Test
    void phaseTwentyEightCoverageBaselineRendersDeterministicDeltaWithoutTiming() {
        var baseline = tuning.benchmark.report.GpuCoverageBaseline.v14Closure("GPU_METAL");
        var current = GpuCoverageSummary.fromTrace(
                GpuCoverageSummaryTest.traceFor("GPU_METAL", backend.ComputeBackend.GPU_METAL)
        ).backends().get("GPU_METAL");

        var comparison = tuning.benchmark.report.GpuCoverageComparison.compare(baseline, current);

        assertEquals("v1.4-pre-closure", comparison.baselineName());
        assertEquals("GPU_METAL", comparison.backend());
        assertTrue(comparison.currentMaxSelectedRegionLength() >= comparison.baselineMaxSelectedRegionLength());
        assertTrue(comparison.currentFallbackCount() <= comparison.baselineFallbackCount());
    }

    @Test
    void benchmarkReportRendersGpuFusedSubpatternMetadata() {
        BenchmarkReport report = gpuCoverageBenchmarkReport("gpu_fused_subpattern_text_report");

        String text = TextBenchmarkReportRenderer.render(report);

        assertTrue(text.contains("gpuFusedSubpatternCount="));
        assertTrue(text.contains("gpuFusedSubpatternTypes="));
        assertTrue(text.contains("gpuFusedSubpatternOriginalNodeIds="));
        assertTrue(text.contains("gpuFusedSubpatternLoweredPrimitiveCount="));
        assertTrue(text.contains("gpuFusedSubpatternReasons="));
        assertTrue(text.contains("cpuMaterializationCount=1"));
    }

    @Test
    void benchmarkJsonRendersGpuFusedSubpatternMetadata() {
        BenchmarkReport report = gpuCoverageBenchmarkReport("gpu_fused_subpattern_json_report");

        String json = JsonBenchmarkReportRenderer.render(report);

        assertTrue(json.contains("\"gpuFusedSubpatternCount\""));
        assertTrue(json.contains("\"gpuFusedSubpatternTypes\""));
        assertTrue(json.contains("\"gpuFusedSubpatternOriginalNodeIds\""));
        assertTrue(json.contains("\"gpuFusedSubpatternLoweredPrimitiveCount\""));
        assertTrue(json.contains("\"gpuFusedSubpatternReasons\""));
        assertTrue(json.contains("\"acceleratorBufferReasonCode\""));
        assertTrue(json.contains("\"cpuMaterializationCount\": 1"));
    }

    @Test
    void renderersExposeCudaGpuCoverageContract() {
        var profile = new ExecutionProfile(
                "cuda-gpu-coverage-profile",
                "cuda-gpu-coverage",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
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

    private static BenchmarkReport gpuCoverageBenchmarkReport(String workloadName) {
        var profile = new ExecutionProfile(
                workloadName + "-profile",
                workloadName,
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        return BenchmarkReport.of(
                workloadName,
                List.of(tuning.benchmark.report.BenchmarkCandidateReport.success(
                        BenchmarkEntry.candidate(workloadName, profile),
                        tuning.validate.ValidationResult.skipped(),
                        new tuning.measure.MeasurementResult(
                                tuning.measure.MeasurementPolicy.defaults(),
                                GpuCoverageSummaryTest.traceFor("GPU_METAL", ComputeBackend.GPU_METAL),
                                new tuning.measure.MeasurementStatistics(2.0, 2.0, 2.0)
                        )
                ))
        );
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
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var summary = new AcceleratorPartitionScoreModel.MaterializationCostSummary(
                "CONSERVATIVE",
                1,
                1024L,
                128L,
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

    private static ExecutionProfile nativeMemoryProfile(String name, NativeCpuMemoryConfig nativeCpuMemory) {
        RuntimeConfig runtime = RuntimeConfig.inferenceDefaults()
                .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE)
                .withNativeCpuFailurePolicy(NativeCpuFailurePolicy.FALLBACK_TO_ARRAY)
                .withNativeCpuMemory(nativeCpuMemory);
        return new ExecutionProfile(
                name + "-profile",
                name,
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline()
                        .withSemanticCanonicalization(config.compile.SemanticCanonicalizationConfig.disabled())
                        .withRegionOptimization(config.compile.RegionOptimizationConfig.disabled()),
                runtime,
                WorkloadProfile.none()
        );
    }

    private static ExecutionProfile trainingOptimizerProfile(String name, CpuStorageProfile cpuStorageProfile) {
        RuntimeConfig runtime = RuntimeConfig.trainingDefaults()
                .withCpuStorageProfile(cpuStorageProfile)
                .withNativeCpuFailurePolicy(NativeCpuFailurePolicy.FALLBACK_TO_ARRAY)
                .withNativeCpuMemory(NativeCpuMemoryConfig.perPreparedExecution(8192L));
        return new ExecutionProfile(
                name + "-profile",
                name,
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                config.compile.CompileConfig.training(),
                runtime,
                WorkloadProfile.none()
        );
    }

    private static TensorRootWorkloadSpec trainingOptimizerWorkload(String name) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.MLP_CLASSIFICATION,
                environment -> {
                    Tensor input = new Tensor(
                            new float[]{
                                    0.10f, -0.20f, 0.30f,
                                    0.40f, 0.50f, -0.60f
                            },
                            new int[]{2, 3},
                            null,
                            "TRAIN_X",
                            DataType.FLOAT32
                    );
                    Tensor w1 = new Tensor(
                            new float[]{
                                    0.10f, -0.20f, 0.30f, -0.40f,
                                    0.50f, 0.60f, -0.70f, 0.80f,
                                    -0.10f, 0.20f, 0.30f, -0.50f
                            },
                            new int[]{3, 4},
                            null,
                            "TRAIN_W1",
                            DataType.FLOAT32
                    );
                    Tensor b1 = new Tensor(
                            new float[]{0.01f, -0.02f, 0.03f, -0.04f},
                            new int[]{4},
                            null,
                            "TRAIN_B1",
                            DataType.FLOAT32
                    );
                    Tensor w2 = new Tensor(
                            new float[]{
                                    0.20f, -0.30f,
                                    0.40f, 0.10f,
                                    -0.50f, 0.60f,
                                    0.70f, -0.20f
                            },
                            new int[]{4, 2},
                            null,
                            "TRAIN_W2",
                            DataType.FLOAT32
                    );
                    Tensor b2 = new Tensor(
                            new float[]{0.05f, -0.06f},
                            new int[]{2},
                            null,
                            "TRAIN_B2",
                            DataType.FLOAT32
                    );
                    w1.setTrainableParameter(true);
                    b1.setTrainableParameter(true);
                    w2.setTrainableParameter(true);
                    b2.setTrainableParameter(true);
                    return input.linear(w1, b1).relu().linear(w2, b2).sum();
                },
                environment -> tuning.validate.ValidationReference.none(),
                environment -> new WorkloadMetadata(
                        name,
                        WorkloadKind.MLP_CLASSIFICATION,
                        Map.of(
                                "batch", 2,
                                "inputFeatures", 3,
                                "hidden", 4,
                                "classes", 2,
                                "optimizerEvidence", true
                        )
                )
        );
    }

    private static NativeCpuMemoryTrace nativeCpuMemoryTrace(BenchmarkReport report, String candidateName) {
        return findReport(report, candidateName).measurement().trace().run().nativeCpuMemory();
    }

    private static BenchmarkCandidateReport findReport(BenchmarkReport report, String candidateName) {
        return report.candidates().stream()
                .filter(candidate -> candidate.entry().name().equals(candidateName))
                .findFirst()
                .orElseThrow();
    }

    private static graph.execution.trace.ExecutionStepTrace nativeRegionStep(
            int index,
            String name,
            String kernel,
            Map<String, Object> attrs
    ) {
        return new graph.execution.trace.ExecutionStepTrace(
                index,
                name,
                "MATMUL",
                List.of(2, 2),
                DataType.FLOAT32,
                "CPU",
                kernel,
                100L,
                new graph.execution.trace.StepExecutionMetadata(
                        "node",
                        attrs,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null
                )
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
                                GpuLoweringUnsupportedReason.UNSUPPORTED_LAYOUT,
                                "UNSUPPORTED_LAYOUT: GPU_METAL normalization inputs require dense layout family=NORMALIZATION target=layer_norm_small"
                        ),
                        new GpuLoweredRegionRejection(
                                "planner.loss",
                                91,
                                "",
                                "",
                                GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                                "UNSUPPORTED_INDEX_SEMANTICS: GPU_METAL index-target loss target out of range: 17 for classes=16 family=LOSS_ADJACENT target=transformer_block_hot_path"
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
