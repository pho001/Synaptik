package graph.execution;

import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.CompiledNode;
import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.memory.RegionMemoryBindingKind;
import graph.optimizer.region.RegionValueRef;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.options.Pool2dOptions;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RuntimeMemoryBinderTest {
    @Test
    void workspaceSensitiveNodesDoNotDisableBindingForIndependentRegionValues() {
        Tensor image = new Tensor(new float[]{
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f,
                13f, 14f, 15f, 16f
        }, new int[]{1, 1, 4, 4}, null, "image", DataType.FLOAT32);
        Tensor pooled = image.maxPool2d(Pool2dOptions.square(2));
        Tensor pooledSum = pooled.sum();

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor relu = a.add(b).relu();
        Tensor reluSum = relu.sum();
        Tensor out = pooledSum.add(reluSum);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution prepared = compiled.prepare(RuntimeConfig.inferenceDefaults());
        MemoryPlan memoryPlan = compiled.compileArtifacts().memoryPlan();
        assertNotNull(memoryPlan);

        List<CompiledNode> nodes = compiled.compileArtifacts().compiledNodes();
        Map<Integer, CompiledNodeExecutionMetadata> metadata = new HashMap<>();
        for (PreparedNodeExecution step : prepared.executionSteps()) {
            metadata.put(step.compiledNode().id(), step.metadata());
        }
        ExecutionState state = ExecutionState.create(nodes, metadata, compiled.compileArtifacts().forwardOutputNode().id());

        CompiledNode maxPoolNode = nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == Operation.OpType.MAX_POOL2D)
                .findFirst()
                .orElseThrow();
        assertTrue(!memoryPlan.runtimeBindingPolicyOf(maxPoolNode.semanticTensor()).regionBindingAllowed());
        float[] maxPoolStorageBefore = state.runtimeTensorForNodeId(maxPoolNode.id()).getFloat32Data();

        Map<Integer, float[]> storageBefore = new HashMap<>();
        for (CompiledNode node : nodes) {
            if (node.dataType() == DataType.FLOAT32) {
                storageBefore.put(node.id(), state.runtimeTensorForNodeId(node.id()).getFloat32Data());
            }
        }

        RuntimeMemoryBinder.bind(memoryPlan, nodes, state);

        assertSame(maxPoolStorageBefore, state.runtimeTensorForNodeId(maxPoolNode.id()).getFloat32Data());
        for (CompiledNode node : nodes) {
            if (node.id() != maxPoolNode.id()
                    && node.operation() != null
                    && node.dataType() == DataType.FLOAT32
                    && hasSingleUseRegionBinding(memoryPlan, node)) {
                assertSame(storageBefore.get(node.id()), state.runtimeTensorForNodeId(node.id()).getFloat32Data());
            }
        }

        prepared.execute(ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{6 + 8 + 14 + 16 + 6 + 8 + 10 + 12}, out.toDoubleArrayCopy(), 1e-6);
    }

    private static boolean hasSingleUseRegionBinding(MemoryPlan memoryPlan, CompiledNode node) {
        RegionValueRef valueRef = memoryPlan.regionValueRefOf(node.semanticTensor());
        if (valueRef == null) {
            return false;
        }
        if (memoryPlan.regionMemoryBindingOf(valueRef).kind() == RegionMemoryBindingKind.NONE) {
            return false;
        }
        Integer slotId = memoryPlan.regionSlotIdOf(valueRef);
        return slotId != null
                && memoryPlan.regionSlotSize(slotId) == node.flatDataSize()
                && memoryPlan.regionSlotUseCount(slotId) < 2;
    }
}
