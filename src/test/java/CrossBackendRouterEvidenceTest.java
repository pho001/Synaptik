import backend.ComputeBackend;
import backend.accelerator.lowering.GpuCompoundRegionSummary;
import backend.accelerator.lowering.GpuLoweredPrimitiveManifest;
import backend.accelerator.lowering.GpuLoweredRegionCandidateSpan;
import backend.accelerator.lowering.GpuLoweredRegionManifest;
import backend.memory.CpuMaterializationReason;
import backend.memory.StorageResidency;
import backend.runtime.ExecutionMode;
import graph.execution.trace.BackendSelectionDecisionTrace;
import graph.execution.trace.BackendSelectionTrace;
import graph.execution.trace.CompileTrace;
import graph.execution.trace.CpuMaterializationTrace;
import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.ExecutionTrace;
import graph.execution.trace.PartitionCompileTrace;
import graph.execution.trace.PrepareTrace;
import graph.execution.trace.RunTrace;
import graph.execution.trace.StepExecutionMetadata;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.benchmark.report.BenchmarkReport;
import tuning.benchmark.report.BenchmarkSuiteReport;
import tuning.benchmark.report.CrossBackendRouterEvidence;
import tuning.benchmark.report.CrossBackendRouterGatePolicy;
import tuning.benchmark.report.CrossBackendRouterRegressionGate;
import tuning.benchmark.report.CrossBackendRouterWorkloadExpectation;
import tuning.measure.MeasurementPolicy;
import tuning.measure.MeasurementResult;
import tuning.measure.MeasurementStatistics;
import tuning.validate.ValidationResult;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CrossBackendRouterEvidenceTest {
    @Test
    void summarizesMetalMpsGraphCopyRequiredEvidence() {
        ExecutionTrace trace = trace(
                "GPU_METAL",
                ComputeBackend.GPU_METAL,
                Map.ofEntries(
                        Map.entry("acceleratorBufferExecutionPath", "BUFFER_BINDING"),
                        Map.entry("acceleratorBufferReasonCode", "BUFFER_BINDING_AVAILABLE"),
                        Map.entry("acceleratorBufferReason", "using native buffer bindings"),
                        Map.entry("metalExecutionRoute", "MPS_GRAPH"),
                        Map.entry("metalNativeCopyStrategy", "MPSGRAPH_RESULT_COPY"),
                        Map.entry("metalOutputBufferWriteStatus", "COPY_REQUIRED"),
                        Map.entry("metalRouteRejectedReasonCodes", List.of("CUSTOM_KERNEL_UNSUPPORTED")),
                        Map.entry("storageResidency", "DEVICE_OWNED"),
                        Map.entry("gpuLayoutMaterializationCount", 1),
                        Map.entry("gpuLayoutMaterializationBytes", 2048L),
                        Map.entry("gpuLayoutTransformKind", "DENSE_GPU_MATERIALIZATION"),
                        Map.entry("gpuLayoutTransformTargetLayoutClass", "DENSE_CONTIGUOUS")
                ),
                List.of(),
                true
        );

        CrossBackendRouterEvidence.BackendEvidence evidence = CrossBackendRouterEvidence.fromTrace(trace)
                .backends()
                .get("GPU_METAL");

        assertEquals(1, evidence.bufferBindingStepCount());
        assertEquals(2, evidence.maxSelectedRegionLength());
        assertEquals(2, evidence.loweredPrimitiveCount());
        assertEquals(Map.of("BUFFER_BINDING", 1), evidence.acceleratorPathCounts());
        assertEquals(Map.of("MPS_GRAPH", 1), evidence.backendRouteCounts());
        assertEquals(Map.of("MPSGRAPH_RESULT_COPY", 1), evidence.nativeCopyStrategyCounts());
        assertEquals(Map.of("COPY_REQUIRED", 1), evidence.outputBufferWriteStatusCounts());
        assertEquals(Map.of("CUSTOM_KERNEL_UNSUPPORTED", 1), evidence.rejectedRouteReasonCounts());
        assertEquals(1, evidence.gpuLayoutMaterializationCount());
    }

    @Test
    void customMetalKernelTrueWriteRoutePassesStrictEvidenceGate() {
        ExecutionTrace trace = trace(
                "GPU_METAL",
                ComputeBackend.GPU_METAL,
                Map.of(
                        "acceleratorBufferExecutionPath", "BUFFER_BINDING",
                        "acceleratorBufferReasonCode", "BUFFER_BINDING_AVAILABLE",
                        "metalExecutionRoute", "CUSTOM_KERNEL",
                        "metalNativeCopyStrategy", "TRUE_OUTPUT_BUFFER_WRITE",
                        "metalOutputBufferWriteStatus", "PROVEN_TRUE_WRITE",
                        "storageResidency", "DEVICE_OWNED"
                ),
                List.of(),
                true
        );
        CrossBackendRouterGatePolicy policy = CrossBackendRouterGatePolicy.nativeHotPath("GPU_METAL")
                .withRequiredRoutes(Set.of("CUSTOM_KERNEL"))
                .withNativeCopyStrategies(Set.of("TRUE_OUTPUT_BUFFER_WRITE"), Set.of("TRUE_OUTPUT_BUFFER_WRITE"))
                .withOutputBufferWriteStatuses(Set.of("PROVEN_TRUE_WRITE"), Set.of("PROVEN_TRUE_WRITE"));

        var result = CrossBackendRouterRegressionGate.evaluate(CrossBackendRouterEvidence.fromTrace(trace), policy);

        assertTrue(result.passed(), result.failures().toString());
    }

    @Test
    void cudaCapabilityMissingCanBeRequiredAsExplicitFallbackEvidence() {
        ExecutionTrace trace = trace(
                "GPU_CUDA",
                ComputeBackend.GPU_CUDA,
                Map.of(
                        "acceleratorBufferExecutionPath", "CPU_FALLBACK",
                        "acceleratorBufferReasonCode", "CAPABILITY_MISSING",
                        "acceleratorBufferReason", "CAPABILITY_MISSING: CUDA SDPA native route unavailable",
                        "cudaExecutionPath", "CPU_FALLBACK",
                        "cudaFallbackReason", "CAPABILITY_MISSING: CUDA SDPA native route unavailable"
                ),
                List.of(),
                false
        );
        CrossBackendRouterGatePolicy policy = new CrossBackendRouterGatePolicy(
                "GPU_CUDA",
                0,
                1,
                0,
                0,
                1,
                false,
                0,
                0,
                Set.of("CPU_FALLBACK"),
                Set.of("CAPABILITY_MISSING"),
                Set.of(),
                Set.of(),
                Set.of(),
                Set.of(),
                true
        );

        var result = CrossBackendRouterRegressionGate.evaluate(CrossBackendRouterEvidence.fromTrace(trace), policy);

        assertTrue(result.passed(), result.failures().toString());
        assertFalse(result.evidence().hasSupportClaim());
    }

    @Test
    void representativeGateFailsHiddenTensorArrayReplayAndCpuMaterialization() {
        ExecutionTrace trace = trace(
                "GPU_METAL",
                ComputeBackend.GPU_METAL,
                Map.of(
                        "acceleratorBufferExecutionPath", "TENSOR_ARRAY",
                        "acceleratorBufferReasonCode", "NATIVE_BUFFER_ABI_UNAVAILABLE",
                        "acceleratorBufferReason", "legacy tensor-array replay"
                ),
                List.of(new CpuMaterializationTrace(
                        2,
                        CpuMaterializationReason.CPU_CONSUMER,
                        "GPU_METAL",
                        StorageResidency.DEVICE_OWNED,
                        4096L,
                        100L,
                        true,
                        "CPU consumer requested readable storage"
                )),
                true
        );

        var result = CrossBackendRouterRegressionGate.evaluate(
                CrossBackendRouterEvidence.fromTrace(trace),
                CrossBackendRouterGatePolicy.nativeHotPath("GPU_METAL")
        );

        assertTrue(result.failures().contains("hidden tensor-array replay"));
        assertTrue(result.failures().contains("unexpected CPU materialization"));
        assertTrue(result.failures().contains("lost native buffer binding"));
    }

    @Test
    void mpsGraphCannotOverclaimTrueOutputBufferWrite() {
        ExecutionTrace trace = trace(
                "GPU_METAL",
                ComputeBackend.GPU_METAL,
                Map.of(
                        "acceleratorBufferExecutionPath", "BUFFER_BINDING",
                        "acceleratorBufferReasonCode", "BUFFER_BINDING_AVAILABLE",
                        "metalExecutionRoute", "MPS_GRAPH",
                        "metalNativeCopyStrategy", "TRUE_OUTPUT_BUFFER_WRITE",
                        "metalOutputBufferWriteStatus", "PROVEN_TRUE_WRITE"
                ),
                List.of(),
                true
        );
        CrossBackendRouterGatePolicy policy = CrossBackendRouterGatePolicy.nativeHotPath("GPU_METAL")
                .withRequiredRoutes(Set.of("MPS_GRAPH"));

        var result = CrossBackendRouterRegressionGate.evaluate(CrossBackendRouterEvidence.fromTrace(trace), policy);

        assertTrue(result.failures().contains("unsupported route overclaim"));
    }

    @Test
    void suiteTargetFailuresNameWorkloadAndBackend() {
        BenchmarkSuiteReport report = suiteReport(
                "transformer_block_hot_path",
                trace(
                        "GPU_METAL",
                        ComputeBackend.GPU_METAL,
                        Map.of(
                                "acceleratorBufferExecutionPath", "TENSOR_ARRAY",
                                "acceleratorBufferReasonCode", "NATIVE_BUFFER_ABI_UNAVAILABLE"
                        ),
                        List.of(),
                        true
                )
        );
        CrossBackendRouterWorkloadExpectation expectation = new CrossBackendRouterWorkloadExpectation(
                "transformer_block_hot_path",
                CrossBackendRouterGatePolicy.nativeHotPath("GPU_METAL")
        );

        var results = CrossBackendRouterRegressionGate.evaluateTargets(report, List.of(expectation));

        assertEquals(1, results.size());
        assertTrue(results.getFirst().failures().stream()
                .anyMatch(failure -> failure.contains("workload=transformer_block_hot_path")
                        && failure.contains("backend=GPU_METAL")
                        && failure.contains("hidden tensor-array replay")));
    }

    private static ExecutionTrace trace(
            String backendName,
            ComputeBackend backend,
            Map<String, Object> attrs,
            List<CpuMaterializationTrace> materializations,
            boolean selected
    ) {
        LinkedHashMap<String, Object> fullAttrs = new LinkedHashMap<>();
        fullAttrs.put("acceleratorBufferBackend", backendName);
        fullAttrs.putAll(attrs);
        ExecutionStepTrace step = new ExecutionStepTrace(
                0,
                backendName.toLowerCase() + "_router_step",
                "LINEAR",
                List.of(2, 2),
                DataType.FLOAT32,
                backendName,
                "PreparedAcceleratorExecutable",
                1L,
                new StepExecutionMetadata(
                        "node",
                        fullAttrs,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null
                )
        );
        BackendSelectionDecisionTrace decision = selected
                ? new BackendSelectionDecisionTrace(
                        1,
                        List.of(1, 2),
                        List.of(backend),
                        true,
                        backend,
                        "selected",
                        256L,
                        null,
                        List.of(),
                        manifest(backend)
                )
                : new BackendSelectionDecisionTrace(
                        1,
                        List.of(1),
                        List.of(backend),
                        false,
                        null,
                        String.valueOf(attrs.getOrDefault("acceleratorBufferReason", "explicit fallback")),
                        256L
                );
        return new ExecutionTrace(
                new CompileTrace(true, 1L, 0, 0, false, PartitionCompileTrace.empty()),
                new PrepareTrace(true, 1L, 0, 0, new BackendSelectionTrace(1, selected ? 1 : 0, selected ? 0 : 1, List.of(decision))),
                new RunTrace(ExecutionMode.FORWARD, 1L, List.of(step), materializations)
        );
    }

    private static GpuLoweredRegionManifest manifest(ComputeBackend backend) {
        return new GpuLoweredRegionManifest(
                "router-evidence-region",
                backend,
                1,
                List.of(1, 2),
                List.of(0),
                List.of(2),
                2,
                List.of(),
                List.of(
                        primitive("p0", "MATMUL", 1),
                        primitive("p1", "RELU", 2)
                ),
                List.of(),
                List.of(),
                GpuCompoundRegionSummary.none(backend, List.of(1, 2)),
                List.of(),
                GpuLoweredRegionCandidateSpan.none(List.of(1, 2)),
                Map.of("dtypeResidency.compute.1", "backend=" + backend + " role=compute dtype=FLOAT32 supported")
        );
    }

    private static GpuLoweredPrimitiveManifest primitive(String id, String type, int nodeId) {
        return new GpuLoweredPrimitiveManifest(
                id,
                type,
                List.of(nodeId),
                List.of("external:0"),
                "node:" + nodeId,
                DataType.FLOAT32,
                List.of(2, 2),
                List.of()
        );
    }

    private static BenchmarkSuiteReport suiteReport(String workloadName, ExecutionTrace trace) {
        config.profile.ExecutionProfile profile = new config.profile.ExecutionProfile(
                "router-evidence-profile",
                "router-evidence-candidate",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                config.profile.WorkloadProfile.transformerHotPathDefaults()
        );
        return new BenchmarkSuiteReport(
                null,
                List.of(BenchmarkReport.of(
                        workloadName,
                        List.of(BenchmarkCandidateReport.success(
                                BenchmarkEntry.candidate("router-evidence-candidate", profile),
                                ValidationResult.skipped(),
                                new MeasurementResult(
                                        MeasurementPolicy.defaults(),
                                        trace,
                                        new MeasurementStatistics(1.0d, 1.0d, 1.0d)
                                )
                        ))
                ))
        );
    }
}
