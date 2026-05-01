package backend.accelerator.lowering;

import backend.ComputeBackend;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorPostOpType;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.optimizer.partition.PartitionPlanningContext;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AcceleratorSubgraphLowererTest {
    @Test
    void linearBiasReluProducesLinearBiasActivationSummary() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "linearInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "linearWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "linearBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        PartitionPlanningContext context = planningContext(out);
        int linearNodeId = nodeId(context, Operation.OpType.LINEAR);
        int reluNodeId = nodeId(context, Operation.OpType.RELU);
        CompiledNode linearNode = context.compiledNode(linearNodeId);

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                new AcceleratorSubgraphSpec(
                        linearNodeId,
                        List.of(linearNodeId, reluNodeId),
                        List.of(
                                new AcceleratorSubgraphOp(linearNodeId, Operation.OpType.LINEAR),
                                new AcceleratorSubgraphOp(reluNodeId, Operation.OpType.RELU)
                        ),
                        linearNode.inputIds(),
                        List.of(reluNodeId)
                ),
                context
        );

        assertNotNull(result);
        assertEquals(GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION, result.compoundSummary().patternType());
        assertTrue(result.compoundSummary().supported());
        assertTrue(result.dagSpec().nodes().stream().anyMatch(node -> node.type() == AcceleratorDagNodeType.LINEAR));
        assertTrue(result.dagSpec().nodes().stream().anyMatch(node -> node.type() == AcceleratorDagNodeType.RELU));
        assertNotNull(result.matMulSpec());
        assertTrue(result.matMulSpec().biasInputNodeId() >= 0);
        assertTrue(result.matMulSpec().postOps().stream().anyMatch(postOp -> postOp.type() == AcceleratorPostOpType.RELU));
    }

    @Test
    void logSoftmaxLowersAsSoftmaxThenLog() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "logSoftmaxInput", DataType.FLOAT32);
        Tensor out = input.logSoftmax(1);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.LOG_SOFTMAX));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(spec(node), context);

        assertEquals(2, result.dagSpec().nodes().size());
        assertEquals(AcceleratorDagNodeType.SOFTMAX, result.dagSpec().nodes().get(0).type());
        assertEquals(AcceleratorDagNodeType.LOG, result.dagSpec().nodes().get(1).type());
        assertEquals(List.of(node.id()), result.dagSpec().outputNodeIds());
    }

    @Test
    void logSoftmaxKeepsOriginalCompiledNodeAsOutput() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "logSoftmaxOutputInput", DataType.FLOAT32);
        Tensor out = input.logSoftmax(1);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.LOG_SOFTMAX));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(spec(node), context);

        assertEquals(node.id(), result.computeNodeId());
        assertEquals(node.id(), result.dagSpec().nodes().get(1).nodeId());
        assertEquals(List.of(1), result.dagSpec().outputNodeIndices());
        assertEquals(List.of(node.id()), result.dagSpec().outputNodeIds());
    }

    @Test
    void logSoftmaxManifestMapsOneOriginalOpToTwoPrimitives() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "logSoftmaxManifestInput", DataType.FLOAT32);
        Tensor out = input.logSoftmax(1);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.LOG_SOFTMAX));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(ComputeBackend.GPU_METAL, spec(node), context);

        assertNotNull(result);
        assertEquals(ComputeBackend.GPU_METAL, result.manifest().backend());
        assertEquals("LOG_SOFTMAX", result.manifest().originalOps().getFirst().opType());
        assertEquals(List.of("p0", "p1"), result.manifest().originalOps().getFirst().loweredPrimitiveIds());
        List<String> primitiveTypes = result.manifest().loweredPrimitives().stream()
                .map(GpuLoweredPrimitiveManifest::primitiveType)
                .toList();
        assertTrue(primitiveTypes.contains("SOFTMAX"));
        assertTrue(primitiveTypes.contains("LOG"));
        assertTrue(result.manifest().loweredPrimitives().stream()
                .allMatch(primitive -> primitive.sourceOriginalNodeIds().contains(node.id())));
    }

    @Test
    void linearBiasReluManifestIncludesFusedSummaryAndAssumptions() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "linearManifestInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "linearManifestWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "linearManifestBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        PartitionPlanningContext context = planningContext(out);
        int linearNodeId = nodeId(context, Operation.OpType.LINEAR);
        int reluNodeId = nodeId(context, Operation.OpType.RELU);
        CompiledNode linearNode = context.compiledNode(linearNodeId);

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                new AcceleratorSubgraphSpec(
                        linearNodeId,
                        List.of(linearNodeId, reluNodeId),
                        List.of(
                                new AcceleratorSubgraphOp(linearNodeId, Operation.OpType.LINEAR),
                                new AcceleratorSubgraphOp(reluNodeId, Operation.OpType.RELU)
                        ),
                        linearNode.inputIds(),
                        List.of(reluNodeId)
                ),
                context
        );

        assertNotNull(result);
        assertEquals(GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION, result.manifest().fusedSummary().patternType());
        assertTrue(!result.manifest().inputAssumptions().isEmpty());
        assertTrue(!result.manifest().outputAssumptions().isEmpty());
        assertTrue(result.manifest().inputAssumptions().stream()
                .anyMatch(assumption -> "CONTIGUOUS".equals(assumption.layout())));
    }

    @Test
    void manifestRegionIdBackendAndLengthAreStable() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "stableManifestInput", DataType.FLOAT32);
        Tensor out = input.logSoftmax(1);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.LOG_SOFTMAX));

        AcceleratorSubgraphLoweringResult result = new AcceleratorSubgraphLowerer().tryLower(ComputeBackend.GPU_CUDA, spec(node), context);

        assertNotNull(result);
        assertEquals("gpu-gpu_cuda-region-" + node.id(), result.manifest().regionId());
        assertEquals(ComputeBackend.GPU_CUDA, result.manifest().backend());
        assertEquals(1, result.manifest().selectedRegionLength());
        assertEquals("2", result.manifest().backendExtensions().get("dagNodeCount"));
    }

    @Test
    void manifestRecordsDTypeResidencyAssumptions() {
        Tensor condition = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{4}, null, "dtypeResidencyMask", DataType.BOOL);
        Tensor trueBranch = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "dtypeResidencyTrue", DataType.FLOAT32);
        Tensor falseBranch = new Tensor(new float[]{-1f, -2f, -3f, -4f}, new int[]{4}, null, "dtypeResidencyFalse", DataType.FLOAT32);
        Tensor where = Tensor.where(condition, trueBranch, falseBranch);
        Tensor bf16 = new Tensor(new float[]{1f, -2f, 3f, -4f}, new int[]{4}, null, "dtypeResidencyBf16", DataType.FLOAT32);
        bf16.setDataType(DataType.BFLOAT16);
        Tensor relu = bf16.relu();
        PartitionPlanningContext boolContext = planningContext(where);
        PartitionPlanningContext bf16Context = planningContext(relu);
        CompiledNode whereNode = boolContext.compiledNode(nodeId(boolContext, Operation.OpType.WHERE));
        CompiledNode reluNode = bf16Context.compiledNode(nodeId(bf16Context, Operation.OpType.RELU));

        AcceleratorSubgraphLoweringResult boolResult = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                spec(whereNode),
                boolContext
        );
        AcceleratorSubgraphLoweringResult bf16Result = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_CUDA,
                spec(reluNode),
                bf16Context
        );

        assertNotNull(boolResult);
        assertNotNull(bf16Result);
        String boolEvidence = String.join("\n", boolResult.manifest().backendExtensions().values());
        String bf16Rendered = GpuLoweredRegionManifestRenderer.renderCompact(bf16Result.manifest());
        assertTrue(boolResult.manifest().backendExtensions().keySet().stream()
                .anyMatch(key -> key.startsWith("dtypeResidency.")));
        assertTrue(boolEvidence.contains("backend=GPU_METAL"));
        assertTrue(boolEvidence.contains("dtype=BOOL"));
        assertTrue(bf16Rendered.contains("dtypeResidency"));
        assertTrue(bf16Rendered.contains("UNSUPPORTED_DTYPE"));
        assertTrue(bf16Rendered.contains("backend=GPU_CUDA"));
        assertTrue(bf16Rendered.contains("dtype=BFLOAT16"));
    }

    @Test
    void sumReductionStillRejectsWhenNoAcceleratorDagTypeExists() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "sumInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        PartitionPlanningContext context = planningContext(out);
        CompiledNode node = context.compiledNode(nodeId(context, Operation.OpType.SUM));

        assertNull(new AcceleratorSubgraphLowerer().tryLower(spec(node), context));
    }

    private static AcceleratorSubgraphSpec spec(CompiledNode node) {
        return new AcceleratorSubgraphSpec(
                node.id(),
                List.of(node.id()),
                List.of(new AcceleratorSubgraphOp(node.id(), node.operation().opType())),
                node.inputIds(),
                List.of(node.id())
        );
    }

    private static PartitionPlanningContext planningContext(Tensor out) {
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(out.topologicalSort());
        return new PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                consumers(compiledNodes)
        );
    }

    private static java.util.Map<Integer, java.util.List<CompiledNode>> consumers(List<CompiledNode> graph) {
        java.util.Map<Integer, java.util.List<CompiledNode>> consumers = new java.util.HashMap<>();
        for (CompiledNode node : graph) {
            consumers.computeIfAbsent(node.id(), ignored -> new java.util.ArrayList<>());
        }
        for (CompiledNode node : graph) {
            for (int inputId : node.inputIds()) {
                consumers.computeIfAbsent(inputId, ignored -> new java.util.ArrayList<>()).add(node);
            }
        }
        return consumers;
    }

    private static int nodeId(PartitionPlanningContext context, Operation.OpType opType) {
        return context.compiledNodes().stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }
}
