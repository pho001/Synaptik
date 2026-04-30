package backend.cuda.exec;

import backend.accelerator.buffer.AcceleratorBufferAccessMode;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorDagValueRef;
import backend.accelerator.exec.PreparedAcceleratorExecutionSupport;
import backend.cuda.bridge.CudaBridgeContext;
import backend.cuda.bridge.CudaBridgeExecutable;
import backend.cuda.bridge.CudaGraphBridge;
import backend.cuda.buffer.CudaBufferAccess;
import backend.cuda.buffer.CudaBufferAllocator;
import backend.cuda.buffer.CudaBufferBinding;
import backend.cuda.buffer.CudaBufferHandle;
import backend.lowering.LoweringFamily;
import backend.memory.CpuMaterializationReason;
import backend.memory.DeviceBufferBinding;
import backend.memory.StorageResidency;
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
import graph.execution.trace.CpuMaterializationTrace;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PreparedCudaExecutableBufferPolicyTest {
    @Test
    void cudaBufferPathExecutesWithoutTensorArrayBridge() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true);
        PreparedCudaExecutable executable = executable(fixture, bridge, AcceleratorBackendConfig.defaults());

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                executable.lastAcceleratorBufferDecision().reasonCode());
        assertEquals(fixture.inputNode().id(), bridge.lastBufferInputs.getFirst().nodeId());
        assertEquals(fixture.outputNode().id(), bridge.lastBufferOutputs.getFirst().nodeId());
        assertNotNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
        assertEquals(StorageResidency.DEVICE_OWNED,
                fixture.state().residencyForNodeId(fixture.outputNode().id()).residency());
        assertTrue(fixture.state().requiresCpuMaterialization(fixture.outputNode().id()));
    }

    @Test
    void cudaBufferPathMaterializesGraphOutputThroughExecutionState() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true);
        PreparedCudaExecutable executable = executable(fixture, bridge, AcceleratorBackendConfig.defaults());

        executable.execute(fixture.context());
        fixture.state().requireCpuReadable(fixture.outputNode().id(), CpuMaterializationReason.GRAPH_OUTPUT);

        assertArrayEquals(new float[]{0f, 3f}, fixture.context()
                .runtimeTensorForNodeId(fixture.outputNode().id())
                .getFloat32Data(), 0f);
        assertNull(fixture.state().deviceBufferBindingForNodeId(fixture.outputNode().id()));
        List<CpuMaterializationTrace> traces = fixture.state().cpuMaterializationTraces();
        assertEquals(1, traces.size());
        CpuMaterializationTrace trace = traces.getFirst();
        assertEquals(fixture.outputNode().id(), trace.nodeId());
        assertEquals(CpuMaterializationReason.GRAPH_OUTPUT, trace.reason());
        assertEquals("GPU_CUDA", trace.materializedFrom());
        assertEquals(StorageResidency.DEVICE_OWNED, trace.sourceResidency());
        assertTrue(trace.bytes() > 0);
        assertTrue(trace.completed());
    }

    @Test
    void cudaBufferFailureFallsBackInAutoMode() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true, true);
        PreparedCudaExecutable executable = executable(fixture, bridge, AcceleratorBackendConfig.defaults());

        executable.execute(fixture.context());

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.CPU_FALLBACK, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.NATIVE_BUFFER_EXECUTION_FAILED,
                executable.lastAcceleratorBufferDecision().reasonCode());
        assertTrue(executable.lastAcceleratorBufferDecision().reason()
                .startsWith("CUDA buffer binding execution failed:"));
        assertArrayEquals(new float[]{0f, 3f}, fixture.context()
                .runtimeTensorForNodeId(fixture.outputNode().id())
                .getFloat32Data(), 0f);
        assertFalse(fixture.state().requiresCpuMaterialization(fixture.outputNode().id()));
    }

    @Test
    void cudaBufferFailureThrowsInRequireMode() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true, true);
        PreparedCudaExecutable executable = executable(
                fixture,
                bridge,
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(fixture.context()));

        assertEquals(1, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.NATIVE_BUFFER_EXECUTION_FAILED,
                executable.lastAcceleratorBufferDecision().reasonCode());
        assertTrue(failure.getMessage().contains("NATIVE_BUFFER_EXECUTION_FAILED"));
        assertTrue(failure.getMessage().contains("CUDA buffer binding execution failed:"));
    }

    @Test
    void adjacentCudaRegionsReuseDeviceBufferBinding() {
        TwoStageFixture fixture = twoStageFixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true);
        PreparedCudaExecutable first = executable(
                fixture.inputNode(),
                fixture.middleNode(),
                fixture.metadata(),
                bridge,
                AcceleratorBackendConfig.defaults()
        );
        PreparedCudaExecutable second = executable(
                fixture.middleNode(),
                fixture.outputNode(),
                fixture.metadata(),
                bridge,
                AcceleratorBackendConfig.defaults()
        );

        first.execute(fixture.context());
        CudaBufferBinding middleBinding = (CudaBufferBinding) fixture.state()
                .deviceBufferBindingForNodeId(fixture.middleNode().id());
        second.execute(fixture.context());

        assertEquals(2, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(middleBinding.nativeHandleIdentity(), bridge.lastBufferInputs.getFirst().nativeHandleIdentity());
        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, second.lastAcceleratorBufferDecision().path());
        assertTrue(fixture.state().cpuMaterializationTraces().isEmpty());
    }

    @Test
    void adjacentCudaRegionRejectsDifferentBackendBinding() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                new NonCudaBinding(fixture.inputNode().id(), layout(fixture.inputNode())),
                StorageResidency.DEVICE_OWNED,
                "foreign device binding"
        );
        PreparedCudaExecutable executable = executable(
                fixture,
                bridge,
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(fixture.context()));

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.INPUT_BINDING_UNAVAILABLE,
                executable.lastAcceleratorBufferDecision().reasonCode());
        assertTrue(failure.getMessage().contains("INPUT_BINDING_UNAVAILABLE"));
    }

    @Test
    void adjacentCudaRegionRejectsMismatchedLayoutBinding() {
        Fixture fixture = fixture();
        FakeCudaBridge bridge = new FakeCudaBridge(true);
        fixture.state().attachDeviceBufferBinding(
                fixture.inputNode().id(),
                new CudaBufferBinding(
                        fixture.inputNode().id(),
                        AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{1}, new int[]{1}, 0, 1),
                        new CudaBufferHandle(MemorySegment.ofAddress(50_000), Float.BYTES, false),
                        CudaBufferAccess.READ_WRITE
                ),
                StorageResidency.DEVICE_OWNED,
                "mismatched CUDA binding"
        );
        PreparedCudaExecutable executable = executable(
                fixture,
                bridge,
                AcceleratorBackendConfig.defaults().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        );

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> executable.execute(fixture.context()));

        assertEquals(0, bridge.bufferExecutions);
        assertEquals(0, bridge.tensorExecutions);
        assertEquals(AcceleratorBufferExecutionPath.UNAVAILABLE, executable.lastAcceleratorBufferDecision().path());
        assertEquals(AcceleratorBufferReasonCode.INPUT_BINDING_UNAVAILABLE,
                executable.lastAcceleratorBufferDecision().reasonCode());
        assertTrue(failure.getMessage().contains("INPUT_BINDING_UNAVAILABLE"));
        assertTrue(executable.lastAcceleratorBufferDecision().reason().contains("does not match expected shape"));
    }

    private static PreparedCudaExecutable executable(
            Fixture fixture,
            CudaGraphBridge bridge,
            AcceleratorBackendConfig backendConfig
    ) {
        return executable(fixture.inputNode(), fixture.outputNode(), fixture.metadata(), bridge, backendConfig);
    }

    private static PreparedCudaExecutable executable(
            CompiledNode inputNode,
            CompiledNode outputNode,
            Map<Integer, CompiledNodeExecutionMetadata> metadata,
            CudaGraphBridge bridge,
            AcceleratorBackendConfig backendConfig
    ) {
        return new PreparedCudaExecutable(
                dag(inputNode, outputNode),
                LoweringFamily.CUDA_GRAPH_REGION,
                bridge,
                List.of(new PreparedAcceleratorExecutionSupport.CpuFallbackStep(
                        outputNode,
                        metadata.get(outputNode.id())
                )),
                backendConfig
        );
    }

    private static Fixture fixture() {
        Tensor input = new Tensor(new float[]{-2f, 3f}, new int[]{2}, null, "input", DataType.FLOAT32);
        Tensor output = input.relu();
        CompiledGraph compiled = CompiledGraph.compile(output, OptimizerConfig.noOptimization());
        List<CompiledNode> nodes = compiled.compileArtifacts().compiledNodes();
        CompiledNode outputNode = nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == Operation.OpType.RELU)
                .findFirst()
                .orElseThrow();
        CompiledNode inputNode = nodes.stream()
                .filter(node -> node.id() == outputNode.inputIds().getFirst())
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
        return new Fixture(inputNode, outputNode, metadata, state, context);
    }

    private static TwoStageFixture twoStageFixture() {
        Tensor input = new Tensor(new float[]{-2f, 3f}, new int[]{2}, null, "input", DataType.FLOAT32);
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
        return new TwoStageFixture(inputNode, middleNode, outputNode, metadata, state, context);
    }

    private static AcceleratorDagSpec dag(CompiledNode inputNode, CompiledNode outputNode) {
        AcceleratorDagInput input = new AcceleratorDagInput(
                inputNode.id(),
                Arrays.stream(inputNode.shape()).boxed().toList(),
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
        return new AcceleratorDagSpec(List.of(input), List.of(node), List.of(0), List.of(outputNode.id()));
    }

    private record Fixture(
            CompiledNode inputNode,
            CompiledNode outputNode,
            Map<Integer, CompiledNodeExecutionMetadata> metadata,
            ExecutionState state,
            ExecutionContext context
    ) {
    }

    private record TwoStageFixture(
            CompiledNode inputNode,
            CompiledNode middleNode,
            CompiledNode outputNode,
            Map<Integer, CompiledNodeExecutionMetadata> metadata,
            ExecutionState state,
            ExecutionContext context
    ) {
    }

    private record NonCudaBinding(int nodeId, AcceleratorBufferLayout layout) implements DeviceBufferBinding {
        @Override
        public String backendId() {
            return "GPU_TEST";
        }

        @Override
        public AcceleratorBufferAccessMode accessMode() {
            return AcceleratorBufferAccessMode.READ_WRITE;
        }

        @Override
        public String nativeHandleIdentity() {
            return "test-non-cuda-" + nodeId;
        }

        @Override
        public boolean available() {
            return true;
        }

        @Override
        public String describe() {
            return "NonCudaBinding{nodeId=" + nodeId + '}';
        }
    }

    private static AcceleratorBufferLayout layout(CompiledNode node) {
        return AcceleratorBufferLayout.of(
                node.dataType(),
                node.shape(),
                node.strides(),
                node.storageOffset(),
                node.flatDataSize()
        );
    }

    private static final class FakeCudaBridge implements CudaGraphBridge {
        private final boolean supportsBufferBindings;
        private final boolean failBufferExecution;
        private final Map<Long, float[]> buffers = new HashMap<>();
        private long nextHandle = 10_000L;
        private int tensorExecutions;
        private int bufferExecutions;
        private List<CudaBufferBinding> lastBufferInputs = List.of();
        private List<CudaBufferBinding> lastBufferOutputs = List.of();

        private FakeCudaBridge(boolean supportsBufferBindings) {
            this(supportsBufferBindings, false);
        }

        private FakeCudaBridge(boolean supportsBufferBindings, boolean failBufferExecution) {
            this.supportsBufferBindings = supportsBufferBindings;
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
        public CudaBridgeContext createContext() {
            return new CudaBridgeContext(true, MemorySegment.ofAddress(1), "");
        }

        @Override
        public CudaBridgeExecutable compile(CudaBridgeContext bridgeContext, AcceleratorDagSpec dagSpec) {
            return new CudaBridgeExecutable(
                    true,
                    MemorySegment.ofAddress(2),
                    "",
                    false,
                    dagSpec.externalInputs().stream().map(AcceleratorDagInput::nodeId).toList(),
                    dagSpec.externalInputs().stream().map(AcceleratorDagInput::dataType).toList(),
                    dagSpec.outputNodeIds(),
                    dagSpec.outputNodeIds().stream().map(ignored -> DataType.FLOAT32).toList()
            );
        }

        @Override
        public boolean supportsBufferBindings() {
            return supportsBufferBindings;
        }

        @Override
        public CudaBufferAllocator createBufferAllocator(CudaBridgeContext bridgeContext) {
            if (!supportsBufferBindings) {
                return CudaBufferAllocator.unavailable("fake CUDA buffer bindings disabled");
            }
            return CudaBufferAllocator.available(new CudaBufferAllocator.NativeAccess() {
                @Override
                public CudaBufferHandle createBuffer(long byteLength, MemorySegment initialData, long initialDataBytes) {
                    long address = nextHandle++;
                    float[] data = new float[Math.toIntExact(byteLength / Float.BYTES)];
                    if (initialData != null && !initialData.equals(MemorySegment.NULL) && initialDataBytes > 0) {
                        MemorySegment source = initialData.reinterpret(initialDataBytes);
                        for (int i = 0; i < data.length; i++) {
                            data[i] = source.get(JAVA_FLOAT, (long) i * Float.BYTES);
                        }
                    }
                    buffers.put(address, data);
                    return new CudaBufferHandle(MemorySegment.ofAddress(address), byteLength, true);
                }

                @Override
                public void readBuffer(CudaBufferHandle handle, MemorySegment destination, long byteLength) {
                    float[] data = buffers.get(handle.handle().address());
                    if (data == null) {
                        throw new IllegalStateException("unknown fake CUDA handle");
                    }
                    for (int i = 0; i < data.length; i++) {
                        destination.set(JAVA_FLOAT, (long) i * Float.BYTES, data[i]);
                    }
                }

                @Override
                public void destroyBuffer(CudaBufferHandle handle) {
                    buffers.remove(handle.handle().address());
                }
            });
        }

        @Override
        public void executeBuffers(
                CudaBridgeContext bridgeContext,
                CudaBridgeExecutable executable,
                List<CudaBufferBinding> inputs,
                List<CudaBufferBinding> outputs
        ) {
            bufferExecutions++;
            lastBufferInputs = List.copyOf(inputs);
            lastBufferOutputs = List.copyOf(outputs);
            if (failBufferExecution) {
                throw new UnsupportedOperationException("synthetic CUDA buffer failure");
            }
            float[] input = buffers.get(inputs.getFirst().handle().handle().address());
            float[] output = buffers.get(outputs.getFirst().handle().handle().address());
            for (int i = 0; i < output.length; i++) {
                output[i] = Math.max(0f, input[i]);
            }
        }

        @Override
        public void execute(
                CudaBridgeContext bridgeContext,
                CudaBridgeExecutable executable,
                List<Tensor> externalInputs,
                List<Tensor> outputs
        ) {
            tensorExecutions++;
            throw new AssertionError("tensor-array bridge should not execute in CUDA buffer tests");
        }
    }
}
