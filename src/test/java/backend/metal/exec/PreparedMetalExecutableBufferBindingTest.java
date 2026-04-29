package backend.metal.exec;

import backend.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.buffer.AcceleratorBufferRequest;
import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorDagValueRef;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import backend.accelerator.exec.AcceleratorPreparedInputSite;
import backend.accelerator.exec.PreparedAcceleratorExecutionSupport;
import backend.accelerator.exec.ResolvedAcceleratorInputs;
import backend.accelerator.lowering.AcceleratorSubgraphLoweringResult;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.plan.CpuLayoutPlan;
import backend.cpu.kernels.elementwise.strided.StridedLayoutDecision;
import backend.memory.DeviceBufferBinding;
import backend.memory.StorageResidency;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.bridge.MetalMpsBridgeExecutable;
import backend.metal.bridge.MetalMpsBridgeExecutionPath;
import backend.metal.bridge.MetalMpsBridgeExecutionStats;
import backend.metal.bridge.MetalMpsGraphBridge;
import backend.metal.buffer.MetalAcceleratorBufferBinder;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferAllocator;
import backend.metal.buffer.MetalBufferBinding;
import backend.metal.buffer.MetalBufferHandle;
import backend.metal.lowering.MetalPartitionPlan;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.execution.ExecutionState;
import graph.execution.PreparedNodeExecution;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PreparedMetalExecutableBufferBindingTest {
    @Test
    void usesBufferBindingPathWhenBridgeAndAllBindingsAreAvailable() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(fixture.inputNode().id(), MetalBufferAccess.READ, 8),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );
        MetalBufferBinding outputBinding = binding(fixture.outputNode().id(), MetalBufferAccess.READ_WRITE, 8);
        fixture.state().reserveDeviceBufferBinding(fixture.outputNode().id(), outputBinding);
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
        assertEquals("using native buffer bindings", executable.lastBufferBindingDecision());
        assertEquals(outputBinding, fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
        assertTrue(fixture.state().requiresCpuMaterialization(fixture.outputNode().id()));
    }

    @Test
    void bufferOffUsesTensorArrayWithoutAllocatorPreflight() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(
                fixture.inputNode(),
                fixture.outputNode(),
                bridge,
                List.of(),
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.OFF, true, 0)
                )
        );
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(fixture.inputNode().id(), MetalBufferAccess.READ, 8),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );

        executable.execute(fixture.context());

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(1, bridge.tensorExecutions);
        assertEquals(0, bridge.bufferAllocations);
        assertEquals(AcceleratorBufferExecutionPath.TENSOR_ARRAY, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferBindingMode.OFF, executable.lastAcceleratorBufferDecision().mode());
        assertTrue(executable.lastBufferBindingDecision().contains("buffer bindings disabled"));
    }

    @Test
    void bufferRequireFailsWhenBridgeDoesNotSupportBufferBindings() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(false);
        PreparedMetalExecutable executable = executable(
                fixture.inputNode(),
                fixture.outputNode(),
                bridge,
                List.of(),
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(fixture.context()));

        assertTrue(failure.getMessage().contains("BACKEND_BUFFER_NOT_IMPLEMENTED"));
        assertEquals(0, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, executable.lastAcceleratorBufferDecision().path());
    }

    @Test
    void preparedInputUploadIsExecutionLocalAndDoesNotPromoteSemanticInputBinding() {
        Fixture fixture = nonContiguousInputFixture();
        FakeBridge bridge = new FakeBridge(true);
        MetalAcceleratorBufferBinder binder = new MetalAcceleratorBufferBinder(bridge, bridge.createContext());
        Tensor semanticInput = fixture.context().runtimeTensorForNodeId(fixture.inputNode().id());
        Tensor preparedInput = new Tensor(new float[]{1f, 3f, 2f, 4f}, fixture.inputNode().shape(), null, "prepared", DataType.FLOAT32);
        ResolvedAcceleratorInputs resolved = new ResolvedAcceleratorInputs(
                List.of(fixture.inputNode().id()),
                List.of(semanticInput),
                List.of(preparedInput),
                List.of(true),
                List.of(new AcceleratorPreparedInputSite(
                        fixture.inputNode().id(),
                        fixture.outputNode().id(),
                        0,
                        true
                ))
        );
        AcceleratorBufferRequest request = new AcceleratorBufferRequest(
                ComputeBackend.GPU_METAL,
                fixture.outputNode().flatDataSize(),
                List.of(fixture.inputNode().id()),
                List.of(DataType.FLOAT32),
                List.of(fixture.outputNode().id()),
                List.of(DataType.FLOAT32),
                false
        );

        var decision = binder.decide(request, resolved, AcceleratorBufferConfig.defaults(), fixture.context());
        var bindings = binder.resolve(request, resolved, decision, fixture.context());

        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, decision.path());
        assertTrue(decision.preparedInputUsed());
        assertEquals(1, bindings.inputs().size());
        assertEquals(fixture.inputNode().id(), bindings.inputs().getFirst().nodeId());
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.inputNode().id()));
        assertEquals(2, bridge.bufferAllocations);
    }

    @Test
    void bufferBindingPathDoesNotRequireTensorArrayLayoutCompatibility() {
        Fixture fixture = nonContiguousInputFixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(fixture.inputNode().id(), MetalBufferAccess.READ, 16, fixture.inputNode().shape(), 4),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );
        MetalBufferBinding outputBinding = binding(
                fixture.outputNode().id(),
                MetalBufferAccess.READ_WRITE,
                16,
                fixture.outputNode().shape(),
                4
        );
        fixture.state().reserveDeviceBufferBinding(fixture.outputNode().id(), outputBinding);

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
        assertEquals("using native buffer bindings", executable.lastBufferBindingDecision());
        assertEquals(outputBinding, fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
    }

    @Test
    void bufferBindingOutputWithPrivateStorageBecomesDeviceOwned() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(fixture.inputNode().id(), MetalBufferAccess.READ, 8),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );
        MetalBufferBinding outputBinding = binding(
                fixture.outputNode().id(),
                MetalBufferAccess.READ_WRITE,
                8,
                fixture.outputNode().shape(),
                2,
                "private"
        );
        fixture.state().reserveDeviceBufferBinding(fixture.outputNode().id(), outputBinding);

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
        assertEquals(outputBinding, fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
        assertTrue(fixture.state().requiresCpuMaterialization(fixture.outputNode().id()));
    }

    @Test
    void allocatesMissingOutputBindingWhenBufferBridgeIsAvailable() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(fixture.inputNode().id(), MetalBufferAccess.READ, 8),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
        assertEquals("using native buffer bindings", executable.lastBufferBindingDecision());
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
        assertTrue(fixture.state().requiresCpuMaterialization(fixture.outputNode().id()));
    }

    @Test
    void fallsBackBeforeReservingBufferForNonContiguousOutputTensor() {
        Fixture fixture = nonContiguousOutputFixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        assertFalse(fixture.context().runtimeTensorForNodeId(fixture.outputNode().id()).isContiguous());

        executable.execute(fixture.context());

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.CPU_FALLBACK, executable.lastExecutionStats().executionPath());
        assertTrue(executable.lastBufferBindingDecision().contains("output tensor layout is not contiguous/zero-offset"));
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
    }

    @Test
    void doesNotAllocateOutputBufferWhenInputBufferPreflightFails() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                new NonMetalBinding(fixture.inputNode().id(), 8),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "non-metal shared input"
        );

        executable.execute(fixture.context());

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(1, bridge.tensorExecutions);
        assertEquals(0, bridge.bufferAllocations);
        assertTrue(executable.lastBufferBindingDecision().contains("external input"));
        assertTrue(executable.lastBufferBindingDecision().contains("binding is not Metal-compatible"));
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
    }

    @Test
    void adjacentMetalExecutionsReuseIntermediateBufferWithoutCpuMaterialization() {
        TwoStageFixture fixture = twoStageFixture();
        FakeBridge bridge = new FakeBridge(true);
        PreparedMetalExecutable first = executable(fixture.inputNode(), fixture.middleNode(), bridge);
        PreparedMetalExecutable second = executable(fixture.middleNode(), fixture.outputNode(), bridge);

        first.execute(fixture.context());
        MetalBufferBinding middleBinding = (MetalBufferBinding) fixture.state()
                .deviceBufferBindingForNodeId(fixture.middleNode().id());
        assertEquals(MetalBufferAccess.READ_WRITE, middleBinding.access());
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.middleNode().id()).residency());
        assertTrue(fixture.state().requiresCpuMaterialization(fixture.middleNode().id()));

        second.execute(fixture.context());

        assertEquals(2, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(middleBinding, bridge.lastBufferInputs.getFirst());
        assertTrue(fixture.state().cpuMaterializationTraces().isEmpty());
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
    }

    @Test
    void usesTensorArrayPathWhenBridgeDoesNotSupportBufferBindings() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(false);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(fixture.inputNode().id(), MetalBufferAccess.READ, 8),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );
        fixture.state().reserveDeviceBufferBinding(
                fixture.outputNode().id(),
                binding(fixture.outputNode().id(), MetalBufferAccess.WRITE, 8)
        );

        executable.execute(fixture.context());

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(1, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.TENSOR_ARRAY_COPY, executable.lastExecutionStats().executionPath());
        assertTrue(executable.lastBufferBindingDecision().contains("bridge does not support buffer bindings"));
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
    }

    @Test
    void bufferBindingExecutionFailureFallsBackWithoutPromotingOutput() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(true, false, true);
        PreparedMetalExecutable executable = executable(fixture, bridge);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                binding(fixture.inputNode().id(), MetalBufferAccess.READ, 8),
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "input shared buffer"
        );
        fixture.state().reserveDeviceBufferBinding(
                fixture.outputNode().id(),
                binding(fixture.outputNode().id(), MetalBufferAccess.WRITE, 8)
        );

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.CPU_FALLBACK, executable.lastExecutionStats().executionPath());
        assertTrue(executable.lastExecutionStats().fallbackReason().contains("buffer binding execution failed"));
        assertTrue(executable.lastBufferBindingDecision().contains("buffer binding execution failed"));
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
    }

    @Test
    void tensorArrayExecutionFailureFallsBackWithTraceReason() {
        Fixture fixture = fixture();
        FakeBridge bridge = new FakeBridge(false, true, false);
        PreparedMetalExecutable executable = executable(fixture, bridge);

        executable.execute(fixture.context());

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(1, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.CPU_FALLBACK, executable.lastExecutionStats().executionPath());
        assertTrue(executable.lastExecutionStats().fallbackReason().contains("tensor-array bridge execution failed"));
        assertTrue(executable.lastBufferBindingDecision().contains("bridge does not support buffer bindings"));
    }

    @Test
    void cpuFallbackPublishesEveryInternalStepAsCpuCurrent() {
        TwoStageFixture fixture = twoStageFixture();
        FakeBridge bridge = new FakeBridge(false, true, false);
        PreparedMetalExecutable executable = executable(
                fixture.inputNode(),
                fixture.outputNode(),
                bridge,
                List.of(
                        new PreparedAcceleratorExecutionSupport.CpuFallbackStep(
                                fixture.middleNode(),
                                fixture.metadata().get(fixture.middleNode().id())
                        ),
                        new PreparedAcceleratorExecutionSupport.CpuFallbackStep(
                                fixture.outputNode(),
                                fixture.metadata().get(fixture.outputNode().id())
                        )
                )
        );

        executable.execute(fixture.context());

        assertEquals(MetalMpsBridgeExecutionPath.CPU_FALLBACK, executable.lastExecutionStats().executionPath());
        assertEquals(StorageResidency.CPU_ARRAY, fixture.state().residencyForNodeId(fixture.middleNode().id()).residency());
        assertTrue(fixture.state().residencyForNodeId(fixture.middleNode().id()).cpuCurrent());
        assertEquals(StorageResidency.CPU_ARRAY, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
        assertTrue(fixture.state().residencyForNodeId(fixture.outputNode().id()).cpuCurrent());
        fixture.context().requireCpuReadable(fixture.middleNode().id(), backend.memory.CpuMaterializationReason.CPU_CONSUMER);
        fixture.context().requireCpuReadable(fixture.outputNode().id(), backend.memory.CpuMaterializationReason.CPU_CONSUMER);
    }

    private static PreparedMetalExecutable executable(Fixture fixture, MetalMpsGraphBridge bridge) {
        return executable(fixture.inputNode(), fixture.outputNode(), bridge);
    }

    private static PreparedMetalExecutable executable(
            CompiledNode inputNode,
            CompiledNode outputNode,
            MetalMpsGraphBridge bridge
    ) {
        return executable(inputNode, outputNode, bridge, List.of());
    }

    private static PreparedMetalExecutable executable(
            CompiledNode inputNode,
            CompiledNode outputNode,
            MetalMpsGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps
    ) {
        return executable(inputNode, outputNode, bridge, cpuFallbackSteps, AcceleratorBackendConfig.defaults());
    }

    private static PreparedMetalExecutable executable(
            CompiledNode inputNode,
            CompiledNode outputNode,
            MetalMpsGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps,
            AcceleratorBackendConfig backendConfig
    ) {
        return new PreparedMetalExecutable(
                plan(inputNode, outputNode),
                backend.lowering.LoweringFamily.METAL_GRAPH_REGION,
                bridge,
                cpuFallbackSteps,
                backendConfig
        );
    }

    private static Fixture fixture() {
        Tensor input = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "input", DataType.FLOAT32);
        Tensor output = input.relu();
        return fixture(input, output);
    }

    private static TwoStageFixture twoStageFixture() {
        Tensor input = new Tensor(new float[]{1f, -2f}, new int[]{2}, null, "input", DataType.FLOAT32);
        Tensor middle = input.relu();
        Tensor output = middle.relu();
        CompiledGraph compiled = CompiledGraph.compile(output, OptimizerConfig.noOptimization());
        List<CompiledNode> nodes = compiled.compileArtifacts().compiledNodes();
        List<CompiledNode> reluNodes = nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == Operation.OpType.RELU)
                .sorted(java.util.Comparator.comparingInt(CompiledNode::id))
                .toList();
        CompiledNode middleNode = reluNodes.get(0);
        CompiledNode outputNode = reluNodes.get(1);
        CompiledNode inputNode = nodes.stream()
                .filter(node -> node.id() == middleNode.inputIds().getFirst())
                .findFirst()
                .orElseThrow();
        Map<Integer, CompiledNodeExecutionMetadata> metadata = new HashMap<>();
        for (PreparedNodeExecution step : compiled.prepare(RuntimeConfig.inferenceDefaults()).executionSteps()) {
            metadata.put(step.compiledNode().id(), step.metadata());
        }
        ExecutionState state = ExecutionState.create(nodes, metadata, compiled.compileArtifacts().forwardOutputNode().id());
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(),
                ExecutionMode.FORWARD,
                metadata,
                state
        );
        return new TwoStageFixture(inputNode, middleNode, outputNode, state, context, metadata);
    }

    private static Fixture nonContiguousInputFixture() {
        Tensor base = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "base", DataType.FLOAT32);
        Tensor input = base.permute(1, 0);
        Tensor output = input.relu();
        return fixture(input, output);
    }

    private static Fixture nonContiguousOutputFixture() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "input", DataType.FLOAT32);
        Tensor output = input.permute(1, 0);
        List<CompiledNode> nodes = CompiledNode.snapshot(List.of(input, output));
        CompiledNode inputNode = nodes.get(0);
        CompiledNode outputNode = nodes.get(1);
        Map<Integer, CompiledNodeExecutionMetadata> metadata = new HashMap<>();
        ExecutionState state = ExecutionState.create(nodes, metadata, outputNode.id());
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(),
                ExecutionMode.FORWARD,
                metadata,
                state
        );
        return new Fixture(inputNode, outputNode, state, context);
    }

    private static Fixture fixture(Tensor input, Tensor output) {
        CompiledGraph compiled = CompiledGraph.compile(output, OptimizerConfig.noOptimization());
        List<CompiledNode> nodes = compiled.compileArtifacts().compiledNodes();
        CompiledNode outputNode = nodes.stream()
                .filter(node -> node.semanticTensor() == output
                        || (node.operation() != null && node.operation().opType() == Operation.OpType.RELU))
                .findFirst()
                .orElseThrow();
        int inputNodeId = outputNode.inputIds().getFirst();
        CompiledNode inputNode = nodes.stream()
                .filter(node -> node.id() == inputNodeId)
                .findFirst()
                .orElseThrow();
        Map<Integer, CompiledNodeExecutionMetadata> metadata = new HashMap<>();
        for (PreparedNodeExecution step : compiled.prepare(RuntimeConfig.inferenceDefaults()).executionSteps()) {
            metadata.put(step.compiledNode().id(), step.metadata());
        }
        ExecutionState state = ExecutionState.create(nodes, metadata, compiled.compileArtifacts().forwardOutputNode().id());
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(),
                ExecutionMode.FORWARD,
                metadata,
                state
        );
        return new Fixture(inputNode, outputNode, state, context);
    }

    private static CpuNodeExecutionPlan cpuPlan() {
        CpuLayoutPlan layoutPlan = new CpuLayoutPlan(
                StridedLayoutDecision.NONE,
                DataType.FLOAT32,
                0,
                null,
                null,
                List.of(),
                List.of()
        );
        return new CpuNodeExecutionPlan(layoutPlan, null, false, 1, 0, null, null, null, null, null, null);
    }

    private static MetalPartitionPlan plan(CompiledNode inputNode, CompiledNode outputNode) {
        AcceleratorDagInput input = new AcceleratorDagInput(
                inputNode.id(),
                shapeList(inputNode.shape()),
                inputNode.dataType()
        );
        AcceleratorDagNode node = new AcceleratorDagNode(
                outputNode.id(),
                AcceleratorDagNodeType.RELU,
                AcceleratorDagValueRef.externalInput(0),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                0,
                1,
                2,
                1,
                1,
                1
        );
        AcceleratorDagSpec dag = new AcceleratorDagSpec(List.of(input), List.of(node), List.of(0), List.of(outputNode.id()));
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                outputNode.id(),
                List.of(outputNode.id()),
                List.of(new AcceleratorSubgraphOp(outputNode.id(), Operation.OpType.RELU)),
                List.of(inputNode.id()),
                List.of(outputNode.id())
        );
        return new MetalPartitionPlan(
                outputNode.id(),
                subgraph,
                new AcceleratorSubgraphLoweringResult(outputNode.id(), null, dag, outputNode.flatDataSize())
        );
    }

    private static MetalBufferBinding binding(int nodeId, MetalBufferAccess access, long bytes) {
        return binding(nodeId, access, bytes, new int[]{2}, 2);
    }

    private static MetalBufferBinding binding(
            int nodeId,
            MetalBufferAccess access,
            long bytes,
            int[] shape,
            long elementCount
    ) {
        return binding(nodeId, access, bytes, shape, elementCount, "shared");
    }

    private static MetalBufferBinding binding(
            int nodeId,
            MetalBufferAccess access,
            long bytes,
            int[] shape,
            long elementCount,
            String storageMode
    ) {
        return new MetalBufferBinding(
                nodeId,
                DataType.FLOAT32,
                shape,
                elementCount,
                new MetalBufferHandle(MemorySegment.ofAddress(nodeId + 1L), bytes, storageMode, "test", false),
                access
        );
    }

    private static List<Integer> shapeList(int[] shape) {
        return java.util.Arrays.stream(shape).boxed().toList();
    }

    private record Fixture(
            CompiledNode inputNode,
            CompiledNode outputNode,
            ExecutionState state,
            ExecutionContext context
    ) {
    }

    private record TwoStageFixture(
            CompiledNode inputNode,
            CompiledNode middleNode,
            CompiledNode outputNode,
            ExecutionState state,
            ExecutionContext context,
            Map<Integer, CompiledNodeExecutionMetadata> metadata
    ) {
    }

    private record NonMetalBinding(int nodeId, long logicalByteLength) implements DeviceBufferBinding {
        @Override
        public String backendId() {
            return "GPU_TEST";
        }

        @Override
        public boolean available() {
            return true;
        }

        @Override
        public String describe() {
            return "non-metal nodeId=" + nodeId;
        }
    }

    private static final class FakeBridge implements MetalMpsGraphBridge {
        private final boolean supportsBufferBindings;
        private final boolean failTensorExecution;
        private final boolean failBufferExecution;
        private int tensorExecutions;
        private int bufferExecutions;
        private int bufferAllocations;
        private List<MetalBufferBinding> lastBufferInputs = List.of();

        private FakeBridge(boolean supportsBufferBindings) {
            this(supportsBufferBindings, false, false);
        }

        private FakeBridge(boolean supportsBufferBindings, boolean failTensorExecution, boolean failBufferExecution) {
            this.supportsBufferBindings = supportsBufferBindings;
            this.failTensorExecution = failTensorExecution;
            this.failBufferExecution = failBufferExecution;
        }

        @Override
        public boolean isAvailable() {
            return true;
        }

        @Override
        public String unavailableReason() {
            return "";
        }

        @Override
        public MetalMpsBridgeContext createContext() {
            return new MetalMpsBridgeContext(true, MemorySegment.ofAddress(1), "");
        }

        @Override
        public MetalMpsBridgeExecutable compile(MetalMpsBridgeContext bridgeContext, MetalPartitionPlan plan) {
            return new MetalMpsBridgeExecutable(
                    true,
                    MemorySegment.ofAddress(2),
                    "",
                    false,
                    plan.externalInputNodeIds(),
                    plan.lowering().dagSpec().externalInputs().stream().map(AcceleratorDagInput::dataType).toList(),
                    plan.producedOutputNodeIds(),
                    plan.producedOutputNodeIds().stream().map(ignored -> DataType.FLOAT32).toList(),
                    plan.lowering().dagSpec().outputNodeIndices()
            );
        }

        @Override
        public boolean supportsBufferBindings() {
            return supportsBufferBindings;
        }

        @Override
        public MetalBufferAllocator createBufferAllocator(MetalMpsBridgeContext bridgeContext) {
            if (!supportsBufferBindings) {
                return MetalBufferAllocator.unavailable("fake bridge buffer bindings disabled");
            }
            return MetalBufferAllocator.available(new MetalBufferAllocator.NativeAccess() {
                @Override
                public MetalBufferHandle createBuffer(long byteLength, int storageMode, MemorySegment initialData, long initialDataBytes) {
                    bufferAllocations++;
                    return new MetalBufferHandle(MemorySegment.ofAddress(1000L + byteLength), byteLength, "shared", "test", true);
                }

                @Override
                public void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength) {
                }

                @Override
                public void destroyBuffer(MetalBufferHandle handle) {
                }
            });
        }

        @Override
        public MetalMpsBridgeExecutionStats execute(
                MetalMpsBridgeContext bridgeContext,
                MetalMpsBridgeExecutable executable,
                List<Tensor> externalInputs,
                List<Tensor> outputs
        ) {
            tensorExecutions++;
            if (failTensorExecution) {
                throw new UnsupportedOperationException("synthetic tensor bridge failure");
            }
            return new MetalMpsBridgeExecutionStats(
                    false,
                    "",
                    MetalMpsBridgeExecutionPath.TENSOR_ARRAY_COPY,
                    externalInputs.size(),
                    outputs.size(),
                    8,
                    8,
                    1,
                    1,
                    1,
                    0,
                    1,
                    4
            );
        }

        @Override
        public MetalMpsBridgeExecutionStats executeBuffers(
                MetalMpsBridgeContext bridgeContext,
                MetalMpsBridgeExecutable executable,
                List<MetalBufferBinding> externalInputs,
                List<MetalBufferBinding> outputs
        ) {
            bufferExecutions++;
            lastBufferInputs = List.copyOf(externalInputs);
            if (failBufferExecution) {
                throw new UnsupportedOperationException("synthetic buffer bridge failure");
            }
            return new MetalMpsBridgeExecutionStats(
                    false,
                    "",
                    MetalMpsBridgeExecutionPath.BUFFER_BINDING,
                    externalInputs.size(),
                    outputs.size(),
                    8,
                    8,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1
            );
        }
    }
}
