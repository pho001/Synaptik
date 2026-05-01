package graph.execution;

import backend.ComputeBackend;
import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.CompiledNode;
import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.memory.MemoryPlanSummary;
import graph.optimizer.memory.MemoryPlannerPolicy;
import graph.optimizer.memory.RegionMemoryBinding;
import graph.optimizer.memory.RegionMemoryBindingKind;
import graph.optimizer.memory.StructuralMemoryView;
import graph.optimizer.region.RegionValueRef;
import operations.Operation;
import operations.layout.noop;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.options.Pool2dOptions;

import java.util.IdentityHashMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RuntimeMemoryBinderTest {
    @Test
    void bindsReusableBFLOAT16RegionSlots() {
        RuntimeBindingFixture fixture = runtimeBindingFixture(DataType.BFLOAT16, new TestOperation(Operation.OpType.ADD));

        short[] firstBefore = fixture.runtimeTensor("first").getBFloat16Data();
        short[] secondBefore = fixture.runtimeTensor("second").getBFloat16Data();
        assertNotSame(firstBefore, secondBefore);

        RuntimeMemoryBinder.bind(fixture.memoryPlan(), fixture.nodes(), fixture.state());

        assertSame(fixture.runtimeTensor("first").getBFloat16Data(), fixture.runtimeTensor("second").getBFloat16Data());
    }

    @Test
    void bindsReusableINT32RegionSlots() {
        RuntimeBindingFixture fixture = runtimeBindingFixture(DataType.INT32, new TestOperation(Operation.OpType.ADD));

        int[] firstBefore = fixture.runtimeTensor("first").getInt32Data();
        int[] secondBefore = fixture.runtimeTensor("second").getInt32Data();
        assertNotSame(firstBefore, secondBefore);

        RuntimeMemoryBinder.bind(fixture.memoryPlan(), fixture.nodes(), fixture.state());

        assertSame(fixture.runtimeTensor("first").getInt32Data(), fixture.runtimeTensor("second").getInt32Data());
    }

    @Test
    void bindsReusableBOOLRegionSlots() {
        RuntimeBindingFixture fixture = runtimeBindingFixture(DataType.BOOL, new TestOperation(Operation.OpType.LOGICAL_OR));

        byte[] firstBefore = fixture.runtimeTensor("first").getBoolData();
        byte[] secondBefore = fixture.runtimeTensor("second").getBoolData();
        assertNotSame(firstBefore, secondBefore);

        RuntimeMemoryBinder.bind(fixture.memoryPlan(), fixture.nodes(), fixture.state());

        assertSame(fixture.runtimeTensor("first").getBoolData(), fixture.runtimeTensor("second").getBoolData());
    }

    @Test
    void typedSlotBindingPreservesAliasViewSkip() {
        Tensor input = inputTensor(DataType.BFLOAT16, "input");
        Tensor alias = new Tensor(new int[]{4}, List.of(input), new noop(), "alias", DataType.BFLOAT16);
        Tensor peer = new Tensor(new int[]{4}, List.of(input), new TestOperation(Operation.OpType.ADD), "peer", DataType.BFLOAT16);
        Tensor root = new Tensor(new int[]{4}, List.of(alias, peer), new TestOperation(Operation.OpType.ADD), "root", DataType.BFLOAT16);
        List<CompiledNode> nodes = CompiledNode.snapshot(root.topologicalSort());
        ExecutionState state = ExecutionState.create(nodes, Map.of(), nodes.getLast().id());
        MemoryPlan memoryPlan = memoryPlanFor(nodes, List.of("alias", "peer"), DataType.BFLOAT16);

        RuntimeMemoryBinder.bind(memoryPlan, nodes, state);

        assertSame(runtimeTensor(nodes, state, "input").getBFloat16Data(), runtimeTensor(nodes, state, "alias").getBFloat16Data());
        assertNotSame(runtimeTensor(nodes, state, "alias").getBFloat16Data(), runtimeTensor(nodes, state, "peer").getBFloat16Data());
    }

    @Test
    void typedSlotBindingPreservesFloat32AndFloat64Behavior() {
        RuntimeBindingFixture f32 = runtimeBindingFixture(DataType.FLOAT32, new TestOperation(Operation.OpType.ADD));
        RuntimeMemoryBinder.bind(f32.memoryPlan(), f32.nodes(), f32.state());
        assertSame(f32.runtimeTensor("first").getFloat32Data(), f32.runtimeTensor("second").getFloat32Data());

        RuntimeBindingFixture f64 = runtimeBindingFixture(DataType.FLOAT64, new TestOperation(Operation.OpType.ADD));
        RuntimeMemoryBinder.bind(f64.memoryPlan(), f64.nodes(), f64.state());
        assertSame(f64.runtimeTensor("first").getFloat64Data(), f64.runtimeTensor("second").getFloat64Data());
    }

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

    private static RuntimeBindingFixture runtimeBindingFixture(DataType dataType, Operation operation) {
        Tensor input = inputTensor(dataType, "input");
        Tensor first = new Tensor(new int[]{4}, List.of(input), operation, "first", dataType);
        Tensor second = new Tensor(new int[]{4}, List.of(input), operation, "second", dataType);
        Tensor root = new Tensor(new int[]{4}, List.of(first, second), operation, "root", dataType);
        List<CompiledNode> nodes = CompiledNode.snapshot(root.topologicalSort());
        ExecutionState state = ExecutionState.create(nodes, Map.of(), nodes.getLast().id());
        MemoryPlan memoryPlan = memoryPlanFor(nodes, List.of("first", "second"), dataType);
        return new RuntimeBindingFixture(nodes, state, memoryPlan);
    }

    private static Tensor inputTensor(DataType dataType, String label) {
        return switch (dataType) {
            case FLOAT64 -> new Tensor(new double[]{1d, 2d, 3d, 4d}, new int[]{4}, null, label, dataType);
            case FLOAT32 -> new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, label, dataType);
            case BFLOAT16 -> new Tensor(new short[]{1, 2, 3, 4}, new int[]{4}, null, label, dataType);
            case INT32 -> new Tensor(new int[]{1, 2, 3, 4}, new int[]{4}, null, label, dataType);
            case BOOL -> new Tensor(new byte[]{1, 0, 1, 0}, new int[]{4}, null, label, dataType);
        };
    }

    private static MemoryPlan memoryPlanFor(List<CompiledNode> nodes, List<String> labels, DataType dataType) {
        Map<RegionValueRef, RegionMemoryBinding> regionMemoryBindings = new HashMap<>();
        Map<RegionValueRef, Integer> regionSlotByValueRef = new HashMap<>();
        Map<Tensor, RegionValueRef> tensorToRegionValueRef = new IdentityHashMap<>();
        int slotId = 7;
        for (String label : labels) {
            CompiledNode node = nodeByLabel(nodes, label);
            RegionValueRef valueRef = RegionValueRef.ofNode(node.id());
            regionMemoryBindings.put(valueRef, new RegionMemoryBinding(
                    valueRef,
                    RegionMemoryBindingKind.MATERIALIZED,
                    slotId,
                    dataType,
                    dataType,
                    true
            ));
            regionSlotByValueRef.put(valueRef, slotId);
            tensorToRegionValueRef.put(node.semanticTensor(), valueRef);
        }
        return new MemoryPlan(
                Map.of(),
                Map.of(),
                Map.of(),
                Map.of(),
                MemoryPlannerPolicy.defaults(),
                emptySummary(),
                StructuralMemoryView.empty(),
                Map.of(),
                Map.of(),
                regionMemoryBindings,
                regionSlotByValueRef,
                Map.of(slotId, 4),
                tensorToRegionValueRef,
                List.of(),
                Map.of()
        );
    }

    private static MemoryPlanSummary emptySummary() {
        return new MemoryPlanSummary(0, 0, 0, 0, 0.0d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0d);
    }

    private static Tensor runtimeTensor(List<CompiledNode> nodes, ExecutionState state, String label) {
        return state.runtimeTensorForNodeId(nodeByLabel(nodes, label).id());
    }

    private static CompiledNode nodeByLabel(List<CompiledNode> nodes, String label) {
        return nodes.stream()
                .filter(node -> label.equals(node.label()))
                .findFirst()
                .orElseThrow();
    }

    private record RuntimeBindingFixture(
            List<CompiledNode> nodes,
            ExecutionState state,
            MemoryPlan memoryPlan
    ) {
        Tensor runtimeTensor(String label) {
            return RuntimeMemoryBinderTest.runtimeTensor(nodes, state, label);
        }
    }

    private record TestOperation(Operation.OpType opType) implements Operation {
        @Override
        public String getExpression() {
            return opType.name();
        }
    }
}
