package graph.execution;

import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;

import backend.contract.ComputeBackend;
import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.residency.RuntimeMemoryBinder;
import graph.execution.state.ExecutionState;
import graph.execution.state.RuntimeStorageKind;
import graph.execution.state.RuntimeStorageSlotKey;
import graph.execution.state.RuntimeStorageSlotScope;
import graph.compile.planning.memory.MemoryPlan;
import graph.compile.planning.memory.MemoryPlanSummary;
import graph.compile.planning.memory.MemoryPlannerPolicy;
import graph.compile.planning.memory.RegionMemoryBinding;
import graph.compile.planning.memory.RegionMemoryBindingKind;
import graph.compile.planning.memory.RegionMemoryPlan;
import graph.compile.planning.memory.RuntimeBindingPlan;
import graph.compile.planning.memory.StructuralMemoryView;
import graph.compile.planning.memory.TensorMemoryPlan;
import graph.compile.planning.value.GraphValueRef;
import operations.Operation;
import operations.layout.noop;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.options.Pool2dOptions;
import tensor.storage.NativeTensorStorage;

import java.util.IdentityHashMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RuntimeMemoryBinderTest {
    @Test
    void bindsReusableBFLOAT16RegionSlots() {
        RuntimeBindingFixture fixture = runtimeBindingFixture(DataType.BFLOAT16, new TestOperation(Operation.OpType.ADD));

        short[] firstBefore = TensorInternalAccess.bfloat16Data(fixture.runtimeTensor("first"));
        short[] secondBefore = TensorInternalAccess.bfloat16Data(fixture.runtimeTensor("second"));
        assertNotSame(firstBefore, secondBefore);

        RuntimeMemoryBinder.bind(fixture.memoryPlan(), fixture.nodes(), fixture.state());

        assertSame(TensorInternalAccess.bfloat16Data(fixture.runtimeTensor("first")), TensorInternalAccess.bfloat16Data(fixture.runtimeTensor("second")));
    }

    @Test
    void bindsReusableINT32RegionSlots() {
        RuntimeBindingFixture fixture = runtimeBindingFixture(DataType.INT32, new TestOperation(Operation.OpType.ADD));

        int[] firstBefore = TensorInternalAccess.int32Data(fixture.runtimeTensor("first"));
        int[] secondBefore = TensorInternalAccess.int32Data(fixture.runtimeTensor("second"));
        assertNotSame(firstBefore, secondBefore);

        RuntimeMemoryBinder.bind(fixture.memoryPlan(), fixture.nodes(), fixture.state());

        assertSame(TensorInternalAccess.int32Data(fixture.runtimeTensor("first")), TensorInternalAccess.int32Data(fixture.runtimeTensor("second")));
    }

    @Test
    void bindsReusableBOOLRegionSlots() {
        RuntimeBindingFixture fixture = runtimeBindingFixture(DataType.BOOL, new TestOperation(Operation.OpType.LOGICAL_OR));

        byte[] firstBefore = TensorInternalAccess.boolData(fixture.runtimeTensor("first"));
        byte[] secondBefore = TensorInternalAccess.boolData(fixture.runtimeTensor("second"));
        assertNotSame(firstBefore, secondBefore);

        RuntimeMemoryBinder.bind(fixture.memoryPlan(), fixture.nodes(), fixture.state());

        assertSame(TensorInternalAccess.boolData(fixture.runtimeTensor("first")), TensorInternalAccess.boolData(fixture.runtimeTensor("second")));
    }

    @Test
    void typedSlotBindingPreservesAliasViewSkip() {
        Tensor input = inputTensor(DataType.BFLOAT16, "input");
        Tensor alias = new Tensor(new int[]{4}, List.of(input), new noop(), "alias", DataType.BFLOAT16);
        Tensor peer = new Tensor(new int[]{4}, List.of(input), new TestOperation(Operation.OpType.ADD), "peer", DataType.BFLOAT16);
        Tensor root = new Tensor(new int[]{4}, List.of(alias, peer), new TestOperation(Operation.OpType.ADD), "root", DataType.BFLOAT16);
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(root.topologicalSort(), BackendIntentPlan.empty());
        ExecutionState state = ExecutionState.create(
                nodes,
                CompiledTensorDescriptorBuilder.build(nodes),
                Map.of(),
                nodes.getLast().id(),
                testsupport.PublicationPlans.forRoot(root, nodes, nodes.getLast().id())
        );
        MemoryPlan memoryPlan = memoryPlanFor(nodes, List.of("alias", "peer"), DataType.BFLOAT16);

        RuntimeMemoryBinder.bind(memoryPlan, nodes, state);

        assertSame(TensorInternalAccess.bfloat16Data(runtimeTensor(nodes, state, "input")), TensorInternalAccess.bfloat16Data(runtimeTensor(nodes, state, "alias")));
        assertNotSame(TensorInternalAccess.bfloat16Data(runtimeTensor(nodes, state, "alias")), TensorInternalAccess.bfloat16Data(runtimeTensor(nodes, state, "peer")));
    }

    @Test
    void typedSlotBindingPreservesSliceAliasViewSkip() {
        Tensor input = new Tensor(
                new float[]{1f, 2f, 3f, 4f},
                new int[]{2, 2},
                null,
                "input",
                DataType.FLOAT32
        );
        Tensor slice = input.slice(new int[]{0, 0}, new int[]{2, 1}, new int[]{0, 1}, new int[]{1, 1});
        Tensor peer = new Tensor(new int[]{2}, List.of(input), new TestOperation(Operation.OpType.ADD), "peer", DataType.FLOAT32);
        Tensor root = new Tensor(new int[]{2}, List.of(slice, peer), new TestOperation(Operation.OpType.ADD), "root", DataType.FLOAT32);
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(root.topologicalSort(), BackendIntentPlan.empty());
        ExecutionState state = ExecutionState.create(
                nodes,
                CompiledTensorDescriptorBuilder.build(nodes),
                Map.of(),
                nodes.getLast().id(),
                testsupport.PublicationPlans.forRoot(root, nodes, nodes.getLast().id())
        );
        MemoryPlan memoryPlan = memoryPlanFor(nodes, List.of("peer", slice.getLabel()), DataType.FLOAT32);

        RuntimeMemoryBinder.bind(memoryPlan, nodes, state);

        assertSame(TensorInternalAccess.float32Data(runtimeTensor(nodes, state, "input")), TensorInternalAccess.float32Data(runtimeTensor(nodes, state, slice.getLabel())));
        assertNotSame(TensorInternalAccess.float32Data(runtimeTensor(nodes, state, slice.getLabel())), TensorInternalAccess.float32Data(runtimeTensor(nodes, state, "peer")));
    }

    @Test
    void typedSlotBindingPreservesFloat32AndFloat64Behavior() {
        RuntimeBindingFixture f32 = runtimeBindingFixture(DataType.FLOAT32, new TestOperation(Operation.OpType.ADD));
        RuntimeMemoryBinder.bind(f32.memoryPlan(), f32.nodes(), f32.state());
        assertSame(TensorInternalAccess.float32Data(f32.runtimeTensor("first")), TensorInternalAccess.float32Data(f32.runtimeTensor("second")));

        RuntimeBindingFixture f64 = runtimeBindingFixture(DataType.FLOAT64, new TestOperation(Operation.OpType.ADD));
        RuntimeMemoryBinder.bind(f64.memoryPlan(), f64.nodes(), f64.state());
        assertSame(TensorInternalAccess.float64Data(f64.runtimeTensor("first")), TensorInternalAccess.float64Data(f64.runtimeTensor("second")));
    }

    @Test
    void registersPlannedRegionSlotAndReusesItForNativeOutputReservation() {
        RuntimeBindingFixture fixture = runtimeBindingFixture(DataType.FLOAT32, new TestOperation(Operation.OpType.ADD));
        int firstNodeId = nodeByLabel(fixture.nodes(), "first").id();
        int secondNodeId = nodeByLabel(fixture.nodes(), "second").id();

        RuntimeMemoryBinder.bind(fixture.memoryPlan(), fixture.nodes(), fixture.state());

        RuntimeStorageSlotKey firstKey = fixture.state().runtimeStorageSlotKeyForNodeId(firstNodeId);
        RuntimeStorageSlotKey secondKey = fixture.state().runtimeStorageSlotKeyForNodeId(secondNodeId);
        assertEquals(RuntimeStorageKind.JAVA_ARRAY, firstKey.kind());
        assertEquals(RuntimeStorageSlotScope.REGION_SLOT, firstKey.scope());
        assertEquals(firstKey, secondKey);

        NativeTensorStorage firstStorage = fixture.state().requireNativeOutputStorage(
                firstNodeId,
                DataType.FLOAT32,
                4,
                "first-native-output"
        );
        NativeTensorStorage secondStorage = fixture.state().requireNativeOutputStorage(
                secondNodeId,
                DataType.FLOAT32,
                4,
                "second-native-output"
        );

        assertSame(firstStorage, secondStorage);
        assertSame(firstStorage, fixture.state().nativeStorageForNodeId(firstNodeId));
        assertSame(secondStorage, fixture.state().nativeStorageForNodeId(secondNodeId));
        assertFalse(fixture.state().residencyForNodeId(firstNodeId).nativeCurrent());
        assertFalse(fixture.state().residencyForNodeId(secondNodeId).nativeCurrent());
        fixture.state().closeResources();
    }

    @Test
    void nativeOutputReservationFallsBackToPerNodeSlotWithoutPlanMetadata() {
        RuntimeBindingFixture fixture = runtimeBindingFixture(DataType.FLOAT32, new TestOperation(Operation.OpType.ADD));
        int firstNodeId = nodeByLabel(fixture.nodes(), "first").id();

        NativeTensorStorage firstReservation = fixture.state().requireNativeOutputStorage(
                firstNodeId,
                DataType.FLOAT32,
                4,
                "first-native-output"
        );
        NativeTensorStorage secondReservation = fixture.state().requireNativeOutputStorage(
                firstNodeId,
                DataType.FLOAT32,
                4,
                "first-native-output-repeat"
        );

        assertSame(firstReservation, secondReservation);
        assertSame(firstReservation, fixture.state().nativeStorageForNodeId(firstNodeId));
        assertFalse(fixture.state().residencyForNodeId(firstNodeId).nativeCurrent());
        fixture.state().closeResources();
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

        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference());
        PreparedExecution prepared = compiled.prepare(RuntimeConfig.inferenceDefaults());
        MemoryPlan memoryPlan = compiled.program().memoryPlan();
        assertNotNull(memoryPlan);

        List<CompiledNode> nodes = compiled.program().compiledNodes();
        Map<Integer, CompiledNodeExecutionMetadata> metadata = new HashMap<>();
        for (PreparedExecutionStep step : prepared.executionSteps()) {
            metadata.put(step.compiledNode().id(), step.metadata());
        }
        ExecutionState state = ExecutionState.create(
                nodes,
                compiled.program().descriptorIndex(),
                metadata,
                compiled.program().forwardOutputNode().id(),
                compiled.publication()
        );

        CompiledNode maxPoolNode = nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == Operation.OpType.MAX_POOL2D)
                .findFirst()
                .orElseThrow();
        assertTrue(!memoryPlan.runtimeBindingPolicyOfNodeId(maxPoolNode.id()).regionBindingAllowed());
        float[] maxPoolStorageBefore = TensorInternalAccess.float32Data(state.runtimeTensorForNodeId(maxPoolNode.id()));

        Map<Integer, float[]> storageBefore = new HashMap<>();
        for (CompiledNode node : nodes) {
            if (node.dataType() == DataType.FLOAT32) {
                storageBefore.put(node.id(), TensorInternalAccess.float32Data(state.runtimeTensorForNodeId(node.id())));
            }
        }

        RuntimeMemoryBinder.bind(memoryPlan, nodes, state);

        assertSame(maxPoolStorageBefore, TensorInternalAccess.float32Data(state.runtimeTensorForNodeId(maxPoolNode.id())));
        for (CompiledNode node : nodes) {
            if (node.id() != maxPoolNode.id()
                    && node.operation() != null
                    && node.dataType() == DataType.FLOAT32
                    && hasSingleUseRegionBinding(memoryPlan, node)) {
                assertSame(storageBefore.get(node.id()), TensorInternalAccess.float32Data(state.runtimeTensorForNodeId(node.id())));
            }
        }

        prepared.execute(ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{6 + 8 + 14 + 16 + 6 + 8 + 10 + 12}, out.toDoubleArrayCopy(), 1e-6);
    }

    private static boolean hasSingleUseRegionBinding(MemoryPlan memoryPlan, CompiledNode node) {
        GraphValueRef valueRef = memoryPlan.graphValueRefOfNodeId(node.id());
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
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(root.topologicalSort(), BackendIntentPlan.empty());
        ExecutionState state = ExecutionState.create(
                nodes,
                CompiledTensorDescriptorBuilder.build(nodes),
                Map.of(),
                nodes.getLast().id(),
                testsupport.PublicationPlans.forRoot(root, nodes, nodes.getLast().id())
        );
        MemoryPlan memoryPlan = memoryPlanFor(nodes, List.of("first", "second"), dataType);
        return new RuntimeBindingFixture(nodes, state, memoryPlan);
    }

    private static Tensor inputTensor(DataType dataType, String label) {
        return switch (dataType) {
            case FLOAT64 -> new Tensor(new double[]{1d, 2d, 3d, 4d}, new int[]{4}, null, label, dataType);
            case FLOAT32 -> new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, label, dataType);
            case BFLOAT16 -> new Tensor(new short[]{1, 2, 3, 4}, new int[]{4}, null, label, dataType);
            case INT32 -> new Tensor(new int[]{1, 2, 3, 4}, new int[]{4}, null, label, dataType);
            case INT64 -> new Tensor(new long[]{1L, 2L, 3L, 4L}, new int[]{4}, null, label, dataType);
            case BOOL -> new Tensor(new byte[]{1, 0, 1, 0}, new int[]{4}, null, label, dataType);
        };
    }

    private static MemoryPlan memoryPlanFor(List<CompiledNode> nodes, List<String> labels, DataType dataType) {
        Map<GraphValueRef, RegionMemoryBinding> regionMemoryBindings = new HashMap<>();
        Map<GraphValueRef, Integer> regionSlotByValueRef = new HashMap<>();
        Map<Integer, GraphValueRef> nodeIdToGraphValueRef = new HashMap<>();
        int slotId = 7;
        int slotSize = 0;
        for (String label : labels) {
            CompiledNode node = nodeByLabel(nodes, label);
            slotSize = Math.max(slotSize, node.flatDataSize());
            GraphValueRef valueRef = GraphValueRef.node(node.id());
            regionMemoryBindings.put(valueRef, new RegionMemoryBinding(
                    valueRef,
                    RegionMemoryBindingKind.MATERIALIZED,
                    slotId,
                    dataType,
                    dataType,
                    true
            ));
            regionSlotByValueRef.put(valueRef, slotId);
            nodeIdToGraphValueRef.put(node.id(), valueRef);
        }
        return new MemoryPlan(
                TensorMemoryPlan.empty(),
                new RegionMemoryPlan(
                        StructuralMemoryView.empty(),
                        Map.of(),
                        Map.of(),
                        regionMemoryBindings,
                        regionSlotByValueRef,
                        Map.of(slotId, slotSize),
                        Map.of(),
                        nodeIdToGraphValueRef,
                        List.of()
                ),
                RuntimeBindingPlan.empty(),
                MemoryPlannerPolicy.defaults(),
                emptySummary()
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
        public OpArityClass arityClass() {
            return OpArityClass.ELEMENT_WISE;
        }

        @Override
        public boolean isFusable() {
            return true;
        }

        @Override
        public OpSemanticFamily semanticFamily() {
            return opType == Operation.OpType.LOGICAL_OR ? OpSemanticFamily.LOGICAL : OpSemanticFamily.ARITHMETIC;
        }

        @Override
        public OpComputationalCost computationalCost() {
            return OpComputationalCost.CHEAP;
        }

        @Override
        public OpControlTrait controlTrait() {
            return opType == Operation.OpType.LOGICAL_OR ? OpControlTrait.BOOL_LOGIC : OpControlTrait.NONE;
        }

        @Override
        public OpResultKind resultKind() {
            return opType == Operation.OpType.LOGICAL_OR ? OpResultKind.BOOLEAN : OpResultKind.NUMERIC;
        }

        @Override
        public String getExpression() {
            return opType.name();
        }
    }
}
