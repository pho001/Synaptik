package backend.metal.bridge;

import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorDagValueRef;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.lowering.AcceleratorSubgraphLoweringResult;
import backend.memory.CpuMaterializationReason;
import backend.metal.buffer.MetalBufferAllocator;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferBinding;
import backend.metal.lowering.MetalPartitionPlan;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MetalMpsFfmBridgeTest {
    @Test
    void bridgeReportsAvailabilityAndProducesContextWithoutThrowing() {
        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();

        assertNotNull(bridge.unavailableReason());
        MetalMpsBridgeContext bridgeContext = bridge.createContext();
        assertNotNull(bridgeContext);
        if (!bridge.isAvailable()) {
            assertFalse(bridge.unavailableReason().isBlank());
            assertFalse(bridgeContext.available());
        } else {
            assertTrue(bridge.isAvailable());
        }
    }

    @Test
    void explicitShimLibraryLoadsWhenConfigured() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();

        assertNotNull(bridge.unavailableReason());
        if (bridge.isAvailable()) {
            assertTrue(bridge.createContext().available());
        } else {
            assertFalse(bridge.unavailableReason().isBlank());
        }
    }

    @Test
    void explicitShimLibrarySupportsBufferAllocatorRoundtrip() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        Tensor source = new Tensor(new float[]{1.5f, -2.0f}, new int[]{2}, null, "source", DataType.FLOAT32);
        MetalBufferBinding binding = allocator.createInputBinding(7, source);
        Tensor destination = new Tensor(new float[]{0.0f, 0.0f}, new int[]{2}, null, "destination", DataType.FLOAT32);

        allocator.readToCpu(binding, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);
        allocator.destroy(binding.handle());

        assertArrayEquals(new float[]{1.5f, -2.0f}, destination.getFloat32Data(), 0.0f);
    }

    @Test
    void explicitShimExecuteBuffersOverwritesCallerOutputBuffer() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalMpsBridgeExecutable executable = bridge.compile(context, reluPlan());
        assumeTrue(executable.available(), executable.reason());
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        MetalBufferBinding input = null;
        MetalBufferBinding output = null;
        try {
            Tensor source = new Tensor(new float[]{1.0f, -2.0f}, new int[]{2}, null, "source", DataType.FLOAT32);
            input = allocator.createInputBinding(1, source);
            Tensor sentinel = new Tensor(new float[]{123.0f, -456.0f}, new int[]{2}, null, "sentinel", DataType.FLOAT32);
            MetalBufferBinding sentinelBinding = allocator.createInputBinding(9, sentinel);
            output = new MetalBufferBinding(
                    9,
                    AcceleratorBufferLayout.fromTensor(sentinel),
                    sentinelBinding.handle(),
                    MetalBufferAccess.READ_WRITE
            );

            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(context, executable, List.of(input), List.of(output));
            Tensor destination = new Tensor(new float[]{0.0f, 0.0f}, new int[]{2}, null, "destination", DataType.FLOAT32);
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            assertTrue(stats.nativeDeviceCopyNs() >= 0L);
            assertArrayEquals(new float[]{1.0f, 0.0f}, destination.getFloat32Data(), 0.0f);
        } finally {
            if (input != null) {
                allocator.destroy(input.handle());
            }
            if (output != null) {
                allocator.destroy(output.handle());
            }
        }
    }

    @Test
    void explicitShimAdjacentExecutablesReuseIntermediateBuffer() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalMpsBridgeExecutable firstExecutable = bridge.compile(
                context,
                unaryPlan(1, 9, AcceleratorDagNodeType.RELU, Operation.OpType.RELU)
        );
        MetalMpsBridgeExecutable secondExecutable = bridge.compile(
                context,
                unaryPlan(9, 10, AcceleratorDagNodeType.NEG, Operation.OpType.NEG)
        );
        assumeTrue(firstExecutable.available(), firstExecutable.reason());
        assumeTrue(secondExecutable.available(), secondExecutable.reason());
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        MetalBufferBinding input = null;
        MetalBufferBinding intermediate = null;
        MetalBufferBinding output = null;
        try {
            input = allocator.createInputBinding(
                    1,
                    new Tensor(new float[]{1.0f, -2.0f}, new int[]{2}, null, "source", DataType.FLOAT32)
            );
            intermediate = allocator.createOutputBinding(9, denseF32Layout(new int[]{2}));
            output = allocator.createOutputBinding(10, denseF32Layout(new int[]{2}));

            MetalMpsBridgeExecutionStats firstStats = bridge.executeBuffers(
                    context,
                    firstExecutable,
                    List.of(input),
                    List.of(intermediate)
            );
            MetalMpsBridgeExecutionStats secondStats = bridge.executeBuffers(
                    context,
                    secondExecutable,
                    List.of(intermediate),
                    List.of(output)
            );
            Tensor destination = new Tensor(new float[]{0.0f, 0.0f}, new int[]{2}, null, "destination", DataType.FLOAT32);
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, firstStats.executionPath());
            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, secondStats.executionPath());
            assertEquals(0L, firstStats.nativeToJavaCopyNs());
            assertEquals(0L, secondStats.nativeToJavaCopyNs());
            assertArrayEquals(new float[]{-1.0f, -0.0f}, destination.getFloat32Data(), 0.0f);
        } finally {
            if (input != null) {
                allocator.destroy(input.handle());
            }
            if (intermediate != null) {
                allocator.destroy(intermediate.handle());
            }
            if (output != null) {
                allocator.destroy(output.handle());
            }
        }
    }

    @Test
    void bufferBindingValidationRejectsMismatchedInputNodeId() {
        MetalMpsBridgeExecutable executable = executableDescriptor(1, 2);
        MetalBufferBinding wrongInput = binding(99, MetalBufferAccess.READ);
        MetalBufferBinding output = binding(2, MetalBufferAccess.WRITE);

        UnsupportedOperationException failure = assertThrows(UnsupportedOperationException.class,
                () -> MetalMpsFfmBridge.validateBufferBindings(executable, List.of(wrongInput), List.of(output)));

        assertEquals("Metal buffer input 0 nodeId 99 does not match executable nodeId 1.", failure.getMessage());
    }

    @Test
    void bufferBindingValidationRejectsMismatchedOutputNodeId() {
        MetalMpsBridgeExecutable executable = executableDescriptor(1, 2);
        MetalBufferBinding input = binding(1, MetalBufferAccess.READ);
        MetalBufferBinding wrongOutput = binding(99, MetalBufferAccess.WRITE);

        UnsupportedOperationException failure = assertThrows(UnsupportedOperationException.class,
                () -> MetalMpsFfmBridge.validateBufferBindings(executable, List.of(input), List.of(wrongOutput)));

        assertEquals("Metal buffer output 0 nodeId 99 does not match executable nodeId 2.", failure.getMessage());
    }

    private static AcceleratorBufferLayout denseF32Layout(int[] shape) {
        long elements = Arrays.stream(shape).asLongStream().reduce(1L, Math::multiplyExact);
        return AcceleratorBufferLayout.of(DataType.FLOAT32, shape, TensorMetadata.computeStrides(shape), 0, elements);
    }

    private static MetalBufferBinding binding(int nodeId, MetalBufferAccess access) {
        return new MetalBufferBinding(
                nodeId,
                denseF32Layout(new int[]{2}),
                new backend.metal.buffer.MetalBufferHandle(
                        java.lang.foreign.MemorySegment.ofAddress(nodeId + 1L),
                        2L * Float.BYTES,
                        "shared",
                        "test",
                        false
                ),
                access
        );
    }

    private static MetalMpsBridgeExecutable executableDescriptor(int inputNodeId, int outputNodeId) {
        return new MetalMpsBridgeExecutable(
                true,
                java.lang.foreign.MemorySegment.ofAddress(100),
                "",
                false,
                List.of(inputNodeId),
                List.of(DataType.FLOAT32),
                List.of(outputNodeId),
                List.of(DataType.FLOAT32),
                List.of(0)
        );
    }

    private static MetalPartitionPlan reluPlan() {
        return unaryPlan(1, 9, AcceleratorDagNodeType.RELU, Operation.OpType.RELU);
    }

    private static MetalPartitionPlan unaryPlan(
            int inputNodeId,
            int outputNodeId,
            AcceleratorDagNodeType nodeType,
            Operation.OpType opType
    ) {
        AcceleratorDagInput input = new AcceleratorDagInput(inputNodeId, List.of(2), DataType.FLOAT32);
        AcceleratorDagNode node = new AcceleratorDagNode(
                outputNodeId,
                nodeType,
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
        AcceleratorDagSpec dag = new AcceleratorDagSpec(List.of(input), List.of(node), List.of(0), List.of(outputNodeId));
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                outputNodeId,
                List.of(outputNodeId),
                List.of(new AcceleratorSubgraphOp(outputNodeId, opType)),
                List.of(inputNodeId),
                List.of(outputNodeId)
        );
        return new MetalPartitionPlan(
                outputNodeId,
                subgraph,
                new AcceleratorSubgraphLoweringResult(outputNodeId, null, dag, 2)
        );
    }
}
