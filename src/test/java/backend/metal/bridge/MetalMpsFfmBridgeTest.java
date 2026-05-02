package backend.metal.bridge;

import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorDagValueRef;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import backend.accelerator.buffer.AcceleratorLayoutAbiV2Support;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.lowering.AcceleratorSubgraphLowerer;
import backend.accelerator.lowering.AcceleratorSubgraphLoweringResult;
import backend.accelerator.lowering.AcceleratorSubgraphSignature;
import backend.memory.CpuMaterializationReason;
import backend.metal.buffer.MetalBufferAllocator;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferBinding;
import backend.metal.lowering.MetalPartitionPlan;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.optimizer.partition.PartitionPlanningContext;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;
import tensor.options.AttentionOptions;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
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
    void capabilitiesReportLayoutAbiV2StateWithoutThrowing() {
        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();

        MetalMpsBridgeCapabilities capabilities = bridge.capabilities();

        assertNotNull(capabilities);
        assertNotNull(capabilities.reason());
        assertEquals(bridge.supportsBufferBindings(), capabilities.bufferExecutionSupported());
        assertTrue(capabilities.layoutAbiV2Version() >= 0);
        assertTrue(capabilities.dtypeAbiV3Version() >= 0);
        if (capabilities.layoutAbiV2Version() < AcceleratorLayoutAbiV2Support.REQUIRED_VERSION) {
            assertFalse(capabilities.layoutAbiV2Supported());
        }
        if (capabilities.dtypeAbiV3Version() < MetalDTypeAbiV3Support.REQUIRED_VERSION) {
            assertFalse(capabilities.dtypeAbiV3Supported());
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
    void explicitShimExecuteBuffersSupportsKeepDimsReduction() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalMpsBridgeExecutable executable = bridge.compile(
                context,
                reductionPlan(
                        1,
                        9,
                        AcceleratorDagNodeType.SUM,
                        Operation.OpType.SUM,
                        1,
                        true,
                        new int[]{2, 3},
                        new int[]{2, 1}
                )
        );
        assumeTrue(executable.available(), executable.reason());
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        MetalBufferBinding input = null;
        MetalBufferBinding output = null;
        try {
            input = allocator.createInputBinding(
                    1,
                    new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "reductionSource", DataType.FLOAT32)
            );
            output = allocator.createOutputBinding(9, denseF32Layout(new int[]{2, 1}));

            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(context, executable, List.of(input), List.of(output));
            Tensor destination = new Tensor(new float[]{0.0f, 0.0f}, new int[]{2, 1}, null, "reductionDestination", DataType.FLOAT32);
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            assertArrayEquals(new float[]{6f, 15f}, destination.getFloat32Data(), 0.0f);
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
    void explicitShimExecuteBuffersSupportsLayerNormSubdag() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor source = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "layerNormSource", DataType.FLOAT32);
        Tensor gammaTensor = new Tensor(new float[]{1f, 1f, 1f}, new int[]{3}, null, "layerNormGamma", DataType.FLOAT32);
        Tensor betaTensor = new Tensor(new float[]{0f, 0f, 0f}, new int[]{3}, null, "layerNormBeta", DataType.FLOAT32);
        Tensor out = source.layerNorm(gammaTensor, betaTensor, 1.0e-5);
        PartitionPlanningContext planningContext = planningContext(out);
        CompiledNode layerNormNode = planningContext.compiledNode(nodeId(planningContext, Operation.OpType.LAYER_NORM));
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                layerNormNode.id(),
                List.of(layerNormNode.id()),
                List.of(new AcceleratorSubgraphOp(layerNormNode.id(), Operation.OpType.LAYER_NORM)),
                layerNormNode.inputIds(),
                List.of(layerNormNode.id())
        );
        AcceleratorSubgraphLoweringResult lowering = new AcceleratorSubgraphLowerer().tryLower(subgraph, planningContext);
        assertNotNull(lowering);

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalMpsBridgeExecutable executable = bridge.compile(
                context,
                new MetalPartitionPlan(layerNormNode.id(), subgraph, lowering)
        );
        assumeTrue(executable.available(), executable.reason());
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        MetalBufferBinding input = null;
        MetalBufferBinding gamma = null;
        MetalBufferBinding beta = null;
        MetalBufferBinding output = null;
        try {
            input = allocator.createInputBinding(layerNormNode.inputIds().get(0), source);
            gamma = allocator.createInputBinding(layerNormNode.inputIds().get(1), gammaTensor);
            beta = allocator.createInputBinding(layerNormNode.inputIds().get(2), betaTensor);
            output = allocator.createOutputBinding(layerNormNode.id(), denseF32Layout(new int[]{2, 3}));

            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(
                    context,
                    executable,
                    List.of(input, gamma, beta),
                    List.of(output)
            );
            Tensor destination = new Tensor(new float[6], new int[]{2, 3}, null, "layerNormDestination", DataType.FLOAT32);
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            assertArrayEquals(new float[]{
                    -1.2247356f, 0.0f, 1.2247356f,
                    -1.2247356f, 0.0f, 1.2247356f
            }, destination.getFloat32Data(), 1.0e-4f);
        } finally {
            if (input != null) {
                allocator.destroy(input.handle());
            }
            if (gamma != null) {
                allocator.destroy(gamma.handle());
            }
            if (beta != null) {
                allocator.destroy(beta.handle());
            }
            if (output != null) {
                allocator.destroy(output.handle());
            }
        }
    }

    @Test
    void explicitShimExecuteBuffersSupportsUnmaskedSdpaScaleParity() {
        assertNativeSdpaParity(
                new float[]{1f, 0f, 0f, 1f},
                new float[]{1f, 0f, 0f, 1f},
                new float[]{10f, 1f, 1f, 10f},
                new int[]{1, 2, 2},
                AttentionOptions.defaults().withScale(0.5)
        );
    }

    @Test
    void explicitShimExecuteBuffersSupportsUnmaskedSdpaDefaultScaleRank4Parity() {
        assertNativeSdpaParity(
                new float[]{1f, 0f, 0f, 1f},
                new float[]{1f, 0f, 0f, 1f},
                new float[]{10f, 1f, 1f, 10f},
                new int[]{1, 1, 2, 2},
                AttentionOptions.defaults()
        );
    }

    @Test
    void executableSignatureIncludesSdpaScaleBits() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "sdpaSigQ", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "sdpaSigK", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "sdpaSigV", DataType.FLOAT32);

        AcceleratorSubgraphSignature halfScale = sdpaSignature(q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(0.5)));
        AcceleratorSubgraphSignature unitScale = sdpaSignature(q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(1.0)));

        assertFalse(halfScale.equals(unitScale));
    }

    private static void assertNativeSdpaParity(float[] queryValues, float[] keyValues, float[] valueValues, int[] shape, AttentionOptions options) {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor expectedQ = new Tensor(queryValues.clone(), shape, null, "expectedSdpaQ", DataType.FLOAT32);
        Tensor expectedK = new Tensor(keyValues.clone(), shape, null, "expectedSdpaK", DataType.FLOAT32);
        Tensor expectedV = new Tensor(valueValues.clone(), shape, null, "expectedSdpaV", DataType.FLOAT32);
        Tensor expected = expectedQ.scaledDotProductAttention(expectedK, expectedV, options);
        expected.compute();

        Tensor q = new Tensor(queryValues.clone(), shape, null, "sdpaQ", DataType.FLOAT32);
        Tensor k = new Tensor(keyValues.clone(), shape, null, "sdpaK", DataType.FLOAT32);
        Tensor v = new Tensor(valueValues.clone(), shape, null, "sdpaV", DataType.FLOAT32);
        Tensor out = q.scaledDotProductAttention(k, v, options);
        PartitionPlanningContext planningContext = planningContext(out);
        CompiledNode sdpaNode = planningContext.compiledNode(nodeId(planningContext, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION));
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                sdpaNode.id(),
                List.of(sdpaNode.id()),
                List.of(new AcceleratorSubgraphOp(sdpaNode.id(), Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                sdpaNode.inputIds(),
                List.of(sdpaNode.id())
        );
        AcceleratorSubgraphLoweringResult lowering = new AcceleratorSubgraphLowerer().tryLower(subgraph, planningContext);
        assertNotNull(lowering);

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalMpsBridgeExecutable executable = bridge.compile(
                context,
                new MetalPartitionPlan(sdpaNode.id(), subgraph, lowering)
        );
        assumeTrue(executable.available(), executable.reason());
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        MetalBufferBinding query = null;
        MetalBufferBinding key = null;
        MetalBufferBinding value = null;
        MetalBufferBinding output = null;
        try {
            query = allocator.createInputBinding(sdpaNode.inputIds().get(0), q);
            key = allocator.createInputBinding(sdpaNode.inputIds().get(1), k);
            value = allocator.createInputBinding(sdpaNode.inputIds().get(2), v);
            output = allocator.createOutputBinding(sdpaNode.id(), denseF32Layout(shape));

            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(
                    context,
                    executable,
                    List.of(query, key, value),
                    List.of(output)
            );
            Tensor destination = new Tensor(new float[expected.getFlatDataSize()], shape, null, "sdpaDestination", DataType.FLOAT32);
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            assertArrayEquals(expected.getFloat32Data(), destination.getFloat32Data(), 1.0e-4f);
        } finally {
            if (query != null) {
                allocator.destroy(query.handle());
            }
            if (key != null) {
                allocator.destroy(key.handle());
            }
            if (value != null) {
                allocator.destroy(value.handle());
            }
            if (output != null) {
                allocator.destroy(output.handle());
            }
        }
    }

    private static AcceleratorSubgraphSignature sdpaSignature(Tensor out) {
        PartitionPlanningContext planningContext = planningContext(out);
        CompiledNode sdpaNode = planningContext.compiledNode(nodeId(planningContext, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION));
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                sdpaNode.id(),
                List.of(sdpaNode.id()),
                List.of(new AcceleratorSubgraphOp(sdpaNode.id(), Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                sdpaNode.inputIds(),
                List.of(sdpaNode.id())
        );
        AcceleratorSubgraphLoweringResult lowering = new AcceleratorSubgraphLowerer().tryLower(subgraph, planningContext);
        assertNotNull(lowering);
        return AcceleratorSubgraphSignature.from(new MetalPartitionPlan(sdpaNode.id(), subgraph, lowering));
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
    void explicitShimMaterializesPermutedLayoutToDenseBuffer() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        assumeTrue(bridge.supportsLayoutMaterialization());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        MetalBufferBinding input = null;
        MetalBufferBinding destination = null;
        try {
            Tensor base = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "layoutBase", DataType.FLOAT32);
            input = allocator.createInputBinding(1, base);
            AcceleratorBufferLayout permutedLayout = AcceleratorBufferLayout.of(
                    DataType.FLOAT32,
                    new int[]{3, 2},
                    new int[]{1, 3},
                    0,
                    6
            );
            MetalBufferBinding sourceView = new MetalBufferBinding(
                    1,
                    permutedLayout,
                    input.handle(),
                    MetalBufferAccess.READ
            );
            AcceleratorBufferLayout denseTarget = AcceleratorBufferLayout.of(
                    DataType.FLOAT32,
                    new int[]{3, 2},
                    new int[]{2, 1},
                    0,
                    6
            );
            destination = allocator.createOutputBinding(2, denseTarget);

            bridge.materializeLayout(context, sourceView, destination);

            Tensor actual = new Tensor(new float[6], new int[]{3, 2}, null, "layoutDense", DataType.FLOAT32);
            allocator.readToCpu(destination, actual, CpuMaterializationReason.PUBLIC_DATA_ACCESS);
            assertArrayEquals(new float[]{1f, 4f, 2f, 5f, 3f, 6f}, actual.getFloat32Data(), 0.0f);
        } finally {
            if (input != null) {
                allocator.destroy(input.handle());
            }
            if (destination != null) {
                allocator.destroy(destination.handle());
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

    @Test
    void bufferBindingValidationAcceptsPolicyApprovedLogicalViewMetadataWithoutNativeAbiChange() {
        MetalMpsBridgeExecutable executable = executableDescriptor(1, 2);
        MetalBufferBinding input = binding(1, MetalBufferAccess.READ);
        AcceleratorBufferLayout logicalView = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2, 2},
                new int[]{1, 2},
                0,
                4
        );
        MetalBufferBinding output = binding(2, logicalView, MetalBufferAccess.WRITE);

        // native layout ABI not required: Java supplies dense physical buffers and handles logical materialization
        assertDoesNotThrow(() -> MetalMpsFfmBridge.validateBufferBindings(executable, List.of(input), List.of(output)));
    }

    @Test
    void bufferBindingValidationStillRejectsUnsupportedOutputDtype() {
        MetalMpsBridgeExecutable executable = executableDescriptor(1, 2, DataType.FLOAT64);
        MetalBufferBinding input = binding(1, MetalBufferAccess.READ);
        MetalBufferBinding output = binding(
                2,
                AcceleratorBufferLayout.of(DataType.FLOAT64, new int[]{2}, new int[]{1}, 0, 2),
                MetalBufferAccess.WRITE
        );

        UnsupportedOperationException failure = assertThrows(UnsupportedOperationException.class,
                () -> MetalMpsFfmBridge.validateBufferBindings(executable, List.of(input), List.of(output)));

        assertEquals("Metal buffer outputs support FLOAT32 only; got FLOAT64.", failure.getMessage());
    }

    private static AcceleratorBufferLayout denseF32Layout(int[] shape) {
        long elements = Arrays.stream(shape).asLongStream().reduce(1L, Math::multiplyExact);
        return AcceleratorBufferLayout.of(DataType.FLOAT32, shape, TensorMetadata.computeStrides(shape), 0, elements);
    }

    private static MetalBufferBinding binding(int nodeId, MetalBufferAccess access) {
        return binding(nodeId, denseF32Layout(new int[]{2}), access);
    }

    private static MetalBufferBinding binding(int nodeId, AcceleratorBufferLayout layout, MetalBufferAccess access) {
        return new MetalBufferBinding(
                nodeId,
                layout,
                new backend.metal.buffer.MetalBufferHandle(
                        java.lang.foreign.MemorySegment.ofAddress(nodeId + 1L),
                        layout.logicalByteLength(),
                        "shared",
                        "test",
                        false
                ),
                access
        );
    }

    private static MetalMpsBridgeExecutable executableDescriptor(int inputNodeId, int outputNodeId) {
        return executableDescriptor(inputNodeId, outputNodeId, DataType.FLOAT32);
    }

    private static MetalMpsBridgeExecutable executableDescriptor(int inputNodeId, int outputNodeId, DataType outputDType) {
        return new MetalMpsBridgeExecutable(
                true,
                java.lang.foreign.MemorySegment.ofAddress(100),
                "",
                false,
                List.of(inputNodeId),
                List.of(DataType.FLOAT32),
                List.of(outputNodeId),
                List.of(outputDType),
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

    private static MetalPartitionPlan reductionPlan(
            int inputNodeId,
            int outputNodeId,
            AcceleratorDagNodeType nodeType,
            Operation.OpType opType,
            int axis,
            boolean keepDims,
            int[] inputShape,
            int[] outputShape
    ) {
        AcceleratorDagInput input = new AcceleratorDagInput(inputNodeId, Arrays.stream(inputShape).boxed().toList(), DataType.FLOAT32);
        AcceleratorDagNode node = new AcceleratorDagNode(
                outputNodeId,
                nodeType,
                AcceleratorDagValueRef.externalInput(0),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                encodeReductionMode(axis, keepDims),
                outputShape.length,
                outputShape[0],
                outputShape.length >= 2 ? outputShape[1] : 1,
                outputShape.length >= 3 ? outputShape[2] : 1,
                outputShape.length >= 4 ? outputShape[3] : 1
        );
        AcceleratorDagSpec dag = new AcceleratorDagSpec(List.of(input), List.of(node), List.of(0), List.of(outputNodeId));
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                outputNodeId,
                List.of(outputNodeId),
                List.of(new AcceleratorSubgraphOp(outputNodeId, opType)),
                List.of(inputNodeId),
                List.of(outputNodeId)
        );
        long estimatedWork = Arrays.stream(inputShape).asLongStream().reduce(1L, Math::multiplyExact);
        return new MetalPartitionPlan(
                outputNodeId,
                subgraph,
                new AcceleratorSubgraphLoweringResult(outputNodeId, null, dag, estimatedWork)
        );
    }

    private static int encodeReductionMode(int axis, boolean keepDims) {
        return (axis & 0xFFFF) | (keepDims ? 1 << 16 : 0);
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
