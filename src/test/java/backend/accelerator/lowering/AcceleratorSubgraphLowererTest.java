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
