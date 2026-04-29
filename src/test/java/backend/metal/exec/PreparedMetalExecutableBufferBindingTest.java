package backend.metal.exec;

import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorDagValueRef;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import backend.accelerator.exec.PreparedAcceleratorExecutionSupport;
import backend.accelerator.lowering.AcceleratorSubgraphLoweringResult;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.plan.CpuLayoutPlan;
import backend.cpu.kernels.elementwise.strided.StridedLayoutDecision;
import backend.memory.StorageResidency;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.bridge.MetalMpsBridgeExecutable;
import backend.metal.bridge.MetalMpsBridgeExecutionPath;
import backend.metal.bridge.MetalMpsBridgeExecutionStats;
import backend.metal.bridge.MetalMpsGraphBridge;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferBinding;
import backend.metal.buffer.MetalBufferHandle;
import backend.metal.lowering.MetalPartitionPlan;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
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
import static org.junit.jupiter.api.Assertions.assertNull;
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
        MetalBufferBinding outputBinding = binding(fixture.outputNode().id(), MetalBufferAccess.WRITE, 8);
        fixture.state().reserveDeviceBufferBinding(fixture.outputNode().id(), outputBinding);
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, executable.lastExecutionStats().executionPath());
        assertEquals("using native buffer bindings", executable.lastBufferBindingDecision());
        assertEquals(outputBinding, fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
        assertEquals(StorageResidency.HOST_SHARED_DEVICE_BUFFER, fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
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
                MetalBufferAccess.WRITE,
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
    void usesTensorArrayPathWhenBufferOutputBindingIsMissing() {
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

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(1, bridge.tensorExecutions);
        assertEquals(MetalMpsBridgeExecutionPath.TENSOR_ARRAY_COPY, executable.lastExecutionStats().executionPath());
        assertTrue(executable.lastBufferBindingDecision().contains("output nodeId=" + fixture.outputNode().id()));
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

    private static PreparedMetalExecutable executable(Fixture fixture, MetalMpsGraphBridge bridge) {
        return new PreparedMetalExecutable(
                plan(fixture.inputNode(), fixture.outputNode()),
                backend.lowering.LoweringFamily.METAL_GRAPH_REGION,
                fixture.outputNode(),
                cpuPlan(),
                bridge,
                List.<PreparedAcceleratorExecutionSupport.CpuFallbackStep>of()
        );
    }

    private static Fixture fixture() {
        Tensor input = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "input", DataType.FLOAT32);
        Tensor output = input.relu();
        return fixture(input, output);
    }

    private static Fixture nonContiguousInputFixture() {
        Tensor base = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "base", DataType.FLOAT32);
        Tensor input = base.permute(1, 0);
        Tensor output = input.relu();
        return fixture(input, output);
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
        return new MetalBufferBinding(
                nodeId,
                DataType.FLOAT32,
                shape,
                elementCount,
                new MetalBufferHandle(MemorySegment.ofAddress(nodeId + 1L), bytes, "shared", "test", false),
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

    private static final class FakeBridge implements MetalMpsGraphBridge {
        private final boolean supportsBufferBindings;
        private int tensorExecutions;
        private int bufferExecutions;

        private FakeBridge(boolean supportsBufferBindings) {
            this.supportsBufferBindings = supportsBufferBindings;
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
                    plan.producedOutputNodeIds(),
                    plan.lowering().dagSpec().outputNodeIndices()
            );
        }

        @Override
        public boolean supportsBufferBindings() {
            return supportsBufferBindings;
        }

        @Override
        public MetalMpsBridgeExecutionStats execute(
                MetalMpsBridgeContext bridgeContext,
                MetalMpsBridgeExecutable executable,
                List<Tensor> externalInputs,
                List<Tensor> outputs
        ) {
            tensorExecutions++;
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
                    1
            );
        }
    }
}
