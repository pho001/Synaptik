import backend.accelerator.buffer.AcceleratorBufferDecision;
import backend.accelerator.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.cuda.lowering.CudaGpuRegionLegalityAdapter;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionMode;
import backend.ComputeBackend;
import backend.runtime.ExecutionContext;
import config.runtime.AcceleratorBufferBindingMode;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.execution.PreparedExecution;
import graph.execution.PreparedNodeExecution;
import graph.optimizer.partition.PartitionTarget;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.List;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;

public class CompiledGraphTraceTest {
    @Test
    void gpuLoweringCoverageSelectionTraceNamesSupportedLogSoftmaxRegion() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "traceLogSoftmaxInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f}, new int[]{3, 3}, null, "traceLogSoftmaxWeight", DataType.FLOAT32);
        Tensor matmul = input.matmul(weight);
        Tensor out = matmul.logSoftmax(1);
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution prepared = compiled.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        int matmulNodeId = nodeId(compiled, operations.Operation.OpType.MATMUL);
        int logSoftmaxNodeId = nodeId(compiled, operations.Operation.OpType.LOG_SOFTMAX);

        var selected = prepared.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected() && decision.selectedBackend() == ComputeBackend.GPU_METAL)
                .filter(decision -> decision.nodeIds().contains(matmulNodeId) && decision.nodeIds().contains(logSoftmaxNodeId))
                .findFirst()
                .orElseThrow();
        List<String> selectedOpNames = selected.nodeIds().stream()
                .map(nodeId -> compiledNode(compiled, nodeId).operation().opType().name())
                .toList();

        assertEquals("selected", selected.reason());
        assertTrue(selectedOpNames.contains("MATMUL") || selectedOpNames.contains("LINEAR"));
        assertTrue(selectedOpNames.contains("LOG_SOFTMAX"));
    }

    @Test
    void gpuLoweringCoverageRejectionTraceNamesUnsupportedReduction() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "traceReductionInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution prepared = compiled.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        int sumNodeId = nodeId(compiled, operations.Operation.OpType.SUM);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(compiledNode(compiled, sumNodeId), null);

        assertFalse(prepared.prepareTrace().backendSelection().decisions().stream()
                .anyMatch(decision -> decision.selected()
                        && decision.selectedBackend() == ComputeBackend.GPU_CUDA
                        && decision.nodeIds().contains(sumNodeId)));
        assertTrue(reason.contains("UNSUPPORTED_OPERATION"));
    }

    @Test
    void compiledGraphExposesCompilePrepareAndRunTrace() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).mul(a);

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        var runTrace = compiled.executeTraced(config.runtime.RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertTrue(compiled.compileTrace().measured());
        assertTrue(compiled.compileTrace().totalNodeCount() > 0);
        assertTrue(compiled.compileTrace().partitionPlanning() != null);
        assertTrue(runTrace.durationNs() >= 0L);
        assertTrue(runTrace.steps().size() > 0);
        assertEquals("FORWARD", runTrace.mode().name());
    }

    @Test
    void fusedHotPathPublishesPrepareAndRunTraceMetadata() {
        int size = 4096;
        float[] av = new float[size];
        float[] bv = new float[size];
        float[] cv = new float[size];
        for (int i = 0; i < size; i++) {
            av[i] = i * 0.01f;
            bv[i] = 1.0f + i * 0.02f;
            cv[i] = -0.5f + i * 0.03f;
        }
        Tensor a = new Tensor(av, new int[]{size}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(bv, new int[]{size}, null, "b", DataType.FLOAT32);
        Tensor c = new Tensor(cv, new int[]{size}, null, "c", DataType.FLOAT32);
        Tensor out = a.add(b).mul(c).relu().exp();

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        var prepared = compiled.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        var runTrace = prepared.executeTraced(ExecutionMode.FORWARD);

        assertTrue(prepared.prepareTrace().measured());
        assertTrue(prepared.prepareTrace().durationNs() >= 0L);
        assertTrue(runTrace.durationNs() >= 0L);
        var fusedStep = runTrace.steps().stream()
                .filter(step -> step.metadata().fused() != null)
                .findFirst()
                .orElseThrow();
        assertTrue(fusedStep.metadata().fused().fusedNodeCount() > 1);
        assertTrue(!fusedStep.metadata().fused().executionBackend().isBlank());
        assertTrue(fusedStep.metadata().fused().schedulerSignature() != null
                && !fusedStep.metadata().fused().schedulerSignature().isBlank());
    }

    @Test
    void autotuneSessionCanRunWithoutCompiledGraphBackReference() {
        Tensor out = Tensor.scalar(2.0).add(Tensor.scalar(3.0));

        ExecutionProfile profile = new ExecutionProfile(
                "delegate",
                "delegate",
                DataType.FLOAT64,
                ExecutionMode.FORWARD,
                OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        tuning.autotune.TuningResult result = tuning.autotune.AutotuneSession.create(new tuning.autotune.AutotuneRequest(
                new tuning.workload.TensorRootWorkloadSpec(
                        "delegate_workload",
                        tuning.workload.WorkloadKind.GENERIC,
                        environment -> Tensor.scalar(2.0).add(Tensor.scalar(3.0)),
                        environment -> tuning.validate.ValidationReference.snapshot(
                                tuning.validate.TensorSnapshot.capture("out", Tensor.scalar(5.0)),
                                java.util.Map.of(),
                                java.util.List.of()
                        ),
                        environment -> tuning.workload.WorkloadMetadata.of("delegate_workload", tuning.workload.WorkloadKind.GENERIC)
                ),
                new tuning.candidate.ListCandidateSpace(java.util.List.of(new tuning.candidate.Candidate("delegate", profile))),
                new tuning.measure.MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                tuning.validate.ValidationPolicy.defaults(),
                new tuning.search.SearchPolicy(4, 1, 1, false),
                tuning.store.PersistencePolicy.disabled()
        )).run();

        assertTrue(result.bestProfile() != null);
    }

    @Test
    void applePartitionTraceCapturesLargestStructuralDagCandidateWhenLowererRejectsIt() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor abs = matmul.abs();
        Tensor out = relu.add(abs);

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(abs, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, OptimizerConfig.noOptimization());
        var decisions = compiled.compileTrace().partitionPlanning().decisions();

        assertTrue(decisions.stream().anyMatch(decision ->
                decision.structuralNodeIds().size() >= 4
                        && decision.opTypes().contains("RELU")
                        && decision.opTypes().contains("ABS")
                        && decision.opTypes().contains("ADD")
        ));
        assertTrue(decisions.stream().allMatch(decision -> decision.exploredCandidates() >= 0));
        assertTrue(decisions.stream().allMatch(decision -> !decision.searchBudgetHit()));
        assertTrue(decisions.stream().allMatch(decision -> decision.reason() != null && !decision.reason().isBlank()));
    }

    @Test
    void cpuOnlyGraphUsesCpuPartitionTargetInAutoMode() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).mul(a);

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());

        assertEquals(PartitionTarget.CPU, compiled.compileTrace().partitionPlanning().target());
        assertTrue(compiled.compileTrace().partitionPlanning().decisions().size() > 0);
    }

    @Test
    void autoPartitionTargetPrefersGpuOverCpuWhenGpuNodesExist() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());

        assertEquals(PartitionTarget.GPU_METAL, compiled.compileTrace().partitionPlanning().target());
    }

    @Test
    void acceleratorTraceAttributesIncludeBufferReasonAndStorageResidency() {
        Tensor a = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "traceA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{3f, 4f}, new int[]{2}, null, "traceB", DataType.FLOAT32);
        Tensor out = a.add(b);
        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, OptimizerConfig.noOptimization());
        CompiledNode outputNode = compiled.compileArtifacts().compiledNodes().stream()
                .filter(node -> node.semanticTensor() == out || node.sourceTensor() == out)
                .findFirst()
                .orElseThrow();
        SyntheticAcceleratorExecutable executable = new SyntheticAcceleratorExecutable(outputNode.id());
        CompiledNodeExecutionMetadata metadata = new CompiledNodeExecutionMetadata(
                ComputeBackend.GPU_CUDA,
                null,
                null,
                null,
                null,
                executable,
                null,
                List.of(),
                backend.accelerator.exec.PartitionExecutionRole.NONE
        );
        PreparedNodeExecution step = new PreparedNodeExecution(outputNode, metadata);
        PreparedExecution prepared = new PreparedExecution(
                config.runtime.RuntimeConfig.inferenceDefaults(),
                false,
                List.of(step),
                List.of(step),
                List.of(),
                compiled.compileArtifacts().compiledNodes(),
                java.util.Map.of(),
                out,
                outputNode,
                null,
                null,
                graph.execution.trace.PrepareTrace.skipped()
        );

        var trace = prepared.executeTraced(ExecutionMode.FORWARD);
        var attrs = trace.steps().getFirst().metadata().attributes();

        assertEquals("GPU_CUDA", attrs.get("acceleratorBufferBackend"));
        assertEquals("BUFFER_BINDING_AVAILABLE", attrs.get("acceleratorBufferReasonCode"));
        assertEquals("BUFFER_BINDING", attrs.get("acceleratorBufferExecutionPath"));
        assertEquals("CPU_ARRAY", attrs.get("storageResidency"));
    }

    @Test
    void gpuCompoundElementwiseTraceContainsPatternAndDagNodeTypes() {
        Tensor a = new Tensor(new float[]{1f, -2f, 3f, -4f}, new int[]{4}, null, "traceChainA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.5f, 1f, -1f, 2f}, new int[]{4}, null, "traceChainB", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor out = relu.exp();
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution prepared = compiled.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        int addNodeId = nodeId(compiled, operations.Operation.OpType.ADD);
        int reluNodeId = nodeId(compiled, operations.Operation.OpType.RELU);
        var trace = prepared.executeTraced(ExecutionMode.FORWARD);
        var gpuStep = trace.steps().stream()
                .filter(step -> "GPU_CUDA".equals(step.backend()))
                .findFirst()
                .orElseThrow();
        var attrs = gpuStep.metadata().attributes();

        assertEquals("ELEMENTWISE_CHAIN", attrs.get("gpuCompoundPattern"));
        assertEquals(true, attrs.get("gpuCompoundSupported"));
        assertEquals("SUPPORTED", attrs.get("gpuCompoundReason"));
        assertTrue(((List<?>) attrs.get("gpuCompoundOrderedNodeIds")).containsAll(List.of(addNodeId, reluNodeId)));
        assertTrue(((List<?>) attrs.get("gpuCompoundDagNodeTypes")).containsAll(List.of("ADD", "RELU", "EXP")));
        if ("BUFFER_BINDING".equals(attrs.get("acceleratorBufferExecutionPath"))) {
            assertFalse(trace.cpuMaterializations().stream().anyMatch(materialization ->
                    (materialization.nodeId() == addNodeId || materialization.nodeId() == reluNodeId)
                            && materialization.reason() == CpuMaterializationReason.CPU_CONSUMER));
        }
    }

    private record SyntheticAcceleratorExecutable(int nodeId) implements PreparedAcceleratorExecutable {
        @Override
        public ComputeBackend backend() {
            return ComputeBackend.GPU_CUDA;
        }

        @Override
        public void execute(ExecutionContext context) {
            context.markCpuCurrent(nodeId, "synthetic accelerator trace output");
        }

        @Override
        public AcceleratorBufferDecision lastAcceleratorBufferDecision() {
            return new AcceleratorBufferDecision(
                    ComputeBackend.GPU_CUDA,
                    AcceleratorBufferBindingMode.AUTO,
                    AcceleratorBufferExecutionPath.BUFFER_BINDING,
                    true,
                    false,
                    AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                    "synthetic accelerator buffer trace",
                    List.of(),
                    List.of()
            );
        }
    }

    private static int nodeId(graph.CompiledGraph compiled, operations.Operation.OpType opType) {
        return compiled.compileArtifacts().compiledNodes().stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static CompiledNode compiledNode(graph.CompiledGraph compiled, int nodeId) {
        return compiled.compileArtifacts().compiledNodes().stream()
                .filter(node -> node.id() == nodeId)
                .findFirst()
                .orElseThrow();
    }
}
