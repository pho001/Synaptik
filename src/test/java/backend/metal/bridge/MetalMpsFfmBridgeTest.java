package backend.metal.bridge;

import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;

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
import backend.cpu.kernels.CpuDTypeOps;
import backend.memory.CpuMaterializationReason;
import backend.metal.buffer.MetalBufferAllocator;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferBinding;
import backend.metal.kernel.MetalCustomKernelExecutable;
import backend.metal.lowering.MetalPartitionPlan;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.optimizer.partition.PartitionPlanningContext;
import operations.Operation;
import operations.layout.sliceGrad;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;
import tensor.TensorPrimitiveBuilder;
import tensor.options.AttentionOptions;
import tensor.options.Conv2dOptions;
import tensor.options.Pool2dOptions;

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
    private static final float BF16_EXACT_STORAGE_TOLERANCE = 0.0f;
    private static final float BF16_MATMUL_REDUCTION_TOLERANCE = 0.5f;
    private static final float BF16_NORM_SOFTMAX_TOLERANCE = 0.025f;

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
            if (stats.nativeCopyStrategy() == MetalNativeCopyStrategy.TRUE_OUTPUT_BUFFER_WRITE) {
                assertEquals(MetalNativeCopyStrategy.TRUE_OUTPUT_BUFFER_WRITE, stats.nativeCopyStrategy());
                assertTrue(stats.outputBufferWriteProven());
                assertEquals(0L, stats.nativeDeviceCopyNs());
            } else {
                assertEquals(MetalNativeCopyStrategy.MPSGRAPH_RESULT_COPY, stats.nativeCopyStrategy());
                assertFalse(stats.outputBufferWriteProven());
                assertTrue(stats.nativeDeviceCopyNs() >= 0L);
            }
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
    void explicitShimOutputBufferWriteProbeRunsWithoutResultCopy() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        assumeTrue(bridge.supportsOutputBufferWriteProbe());
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
            float[] sentinelValues = new float[]{123.0f, -456.0f};
            Tensor sentinel = new Tensor(sentinelValues.clone(), new int[]{2}, null, "sentinel", DataType.FLOAT32);
            MetalBufferBinding sentinelBinding = allocator.createInputBinding(9, sentinel);
            output = new MetalBufferBinding(
                    9,
                    AcceleratorBufferLayout.fromTensor(sentinel),
                    sentinelBinding.handle(),
                    MetalBufferAccess.READ_WRITE
            );

            MetalMpsBridgeExecutionStats stats = bridge.probeOutputBufferWriteWithoutResultCopy(context, executable, List.of(input), List.of(output));
            Tensor destination = new Tensor(new float[]{0.0f, 0.0f}, new int[]{2}, null, "destination", DataType.FLOAT32);
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            assertEquals(MetalNativeCopyStrategy.UNKNOWN_OR_UNPROVEN, stats.nativeCopyStrategy());
            assertFalse(stats.outputBufferWriteProven());
            assertEquals(0L, stats.nativeDeviceCopyNs());
            boolean wroteExpected = Arrays.equals(new float[]{1.0f, 0.0f}, destination.getFloat32Data());
            boolean keptSentinel = Arrays.equals(sentinelValues, destination.getFloat32Data());
            assertTrue(
                    wroteExpected || keptSentinel,
                    "MPSGraph output-buffer probe produced neither expected direct-write values nor the original sentinel."
            );

            Tensor proofSentinel = new Tensor(sentinelValues.clone(), new int[]{2}, null, "proofSentinel", DataType.FLOAT32);
            MetalBufferBinding proofSentinelBinding = allocator.createInputBinding(9, proofSentinel);
            MetalBufferBinding proofOutput = new MetalBufferBinding(
                    9,
                    AcceleratorBufferLayout.fromTensor(proofSentinel),
                    proofSentinelBinding.handle(),
                    MetalBufferAccess.READ_WRITE
            );
            MetalOutputBufferWriteProbeResult proof = bridge.probeOutputBufferWriteContract(
                    context,
                    executable,
                    List.of(input),
                    List.of(proofOutput)
            );
            System.out.println("METAL_OUTPUT_BUFFER_WRITE_PROBE status=" + proof.status()
                    + " detail=" + proof.detail());
            assertTrue(
                    proof.status() == MetalOutputBufferWriteProbeStatus.MATCHES_COPIED_RESULT
                            || proof.status() == MetalOutputBufferWriteProbeStatus.UNCHANGED_SENTINEL
                            || proof.status() == MetalOutputBufferWriteProbeStatus.MISMATCHED_RESULT,
                    "Unexpected proof status: " + proof
            );
            allocator.destroy(proofOutput.handle());
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
    void explicitShimCustomReluKernelWritesCallerOutputBuffer() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        MetalMpsFfmCustomKernelBridge customBridge = new MetalMpsFfmCustomKernelBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        assumeTrue(customBridge.capabilities().available(), customBridge.capabilities().reason());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalCustomKernelExecutable executable = customBridge.compile(reluPlan());
        assumeTrue(executable.available(), executable.reason());
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        MetalBufferBinding input = null;
        MetalBufferBinding output = null;
        try {
            Tensor source = new Tensor(new float[]{-3.0f, 2.5f}, new int[]{2}, null, "source", DataType.FLOAT32);
            input = allocator.createInputBinding(1, source);
            output = allocator.createOutputBinding(9, denseF32Layout(new int[]{2}));

            MetalMpsBridgeExecutionStats stats = customBridge.executeBuffers(context, executable, List.of(input), List.of(output));
            Tensor destination = new Tensor(new float[]{0.0f, 0.0f}, new int[]{2}, null, "destination", DataType.FLOAT32);
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.CUSTOM_KERNEL, stats.executionPath());
            assertEquals(MetalNativeCopyStrategy.TRUE_OUTPUT_BUFFER_WRITE, stats.nativeCopyStrategy());
            assertTrue(stats.outputBufferWriteProven());
            assertArrayEquals(new float[]{0.0f, 2.5f}, destination.getFloat32Data(), 0.0f);
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
    void explicitShimExecuteBuffersSupportsMinMaxAndScalarPowMpsGraphMappings() {
        Tensor minLeft = new Tensor(new float[]{1f, 4f, 9f, 16f}, new int[]{4}, null, "nativeMinLeft", DataType.FLOAT32);
        Tensor minRight = new Tensor(new float[]{2f, 3f, 10f, 8f}, new int[]{4}, null, "nativeMinRight", DataType.FLOAT32);
        Tensor minOut = minLeft.min(minRight);
        Tensor minDestination = executeF32LoweredNode(
                minOut,
                Operation.OpType.MIN,
                List.of(minLeft, minRight),
                new int[]{4}
        );

        Tensor maxLeft = new Tensor(new float[]{1f, 4f, 9f, 16f}, new int[]{4}, null, "nativeMaxLeft", DataType.FLOAT32);
        Tensor maxRight = new Tensor(new float[]{2f, 3f, 10f, 8f}, new int[]{4}, null, "nativeMaxRight", DataType.FLOAT32);
        Tensor maxOut = maxLeft.max(maxRight);
        Tensor maxDestination = executeF32LoweredNode(
                maxOut,
                Operation.OpType.MAX,
                List.of(maxLeft, maxRight),
                new int[]{4}
        );

        Tensor powInput = new Tensor(new float[]{1f, 4f, 9f, 16f}, new int[]{4}, null, "nativePowInput", DataType.FLOAT32);
        Tensor powOut = powInput.pow(1.5);
        Tensor powDestination = executeF32LoweredNode(
                powOut,
                Operation.OpType.POW,
                List.of(powInput),
                new int[]{4}
        );

        assertArrayEquals(new float[]{1f, 3f, 9f, 8f}, minDestination.getFloat32Data(), 0.0f);
        assertArrayEquals(new float[]{2f, 4f, 10f, 16f}, maxDestination.getFloat32Data(), 0.0f);
        assertArrayEquals(new float[]{1f, 8f, 27f, 64f}, powDestination.getFloat32Data(), 1.0e-4f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsUnaryMathParityMpsGraphMappings() {
        Tensor input = new Tensor(new float[]{-1.25f, -0.0f, 0.25f, 2.75f}, new int[]{4}, null, "nativeUnaryMathInput", DataType.FLOAT32);

        Tensor erfDestination = executeF32LoweredNode(input.erf(), Operation.OpType.ERF, List.of(input), new int[]{4});
        Tensor floorDestination = executeF32LoweredNode(input.floor(), Operation.OpType.FLOOR, List.of(input), new int[]{4});
        Tensor ceilDestination = executeF32LoweredNode(input.ceil(), Operation.OpType.CEIL, List.of(input), new int[]{4});
        Tensor signDestination = executeF32LoweredNode(input.sign(), Operation.OpType.SIGN, List.of(input), new int[]{4});

        assertArrayEquals(new float[]{
                utils.SpecialFunctions.erf(-1.25f),
                utils.SpecialFunctions.erf(-0.0f),
                utils.SpecialFunctions.erf(0.25f),
                utils.SpecialFunctions.erf(2.75f)
        }, erfDestination.getFloat32Data(), 1.0e-5f);
        assertArrayEquals(new float[]{-2f, -0.0f, 0f, 2f}, floorDestination.getFloat32Data(), 0.0f);
        assertArrayEquals(new float[]{-1f, -0.0f, 1f, 3f}, ceilDestination.getFloat32Data(), 0.0f);
        assertArrayEquals(new float[]{-1f, -0.0f, 1f, 1f}, signDestination.getFloat32Data(), 0.0f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsBfloat16UnaryMathParityMpsGraphMappings() {
        Tensor input = bf16Tensor(new float[]{-1.25f, -0.0f, 0.25f, 2.75f}, new int[]{4}, "nativeBf16UnaryMathInput");

        Tensor erfDestination = executeBf16LoweredNode(input.erf(), Operation.OpType.ERF, List.of(input), new int[]{4});
        Tensor floorDestination = executeBf16LoweredNode(input.floor(), Operation.OpType.FLOOR, List.of(input), new int[]{4});
        Tensor ceilDestination = executeBf16LoweredNode(input.ceil(), Operation.OpType.CEIL, List.of(input), new int[]{4});
        Tensor signDestination = executeBf16LoweredNode(input.sign(), Operation.OpType.SIGN, List.of(input), new int[]{4});

        assertBf16Close(new float[]{
                utils.SpecialFunctions.erf(-1.25f),
                utils.SpecialFunctions.erf(-0.0f),
                utils.SpecialFunctions.erf(0.25f),
                utils.SpecialFunctions.erf(2.75f)
        }, erfDestination, 0.01f);
        assertBf16Close(new float[]{-2f, -0.0f, 0f, 2f}, floorDestination, BF16_EXACT_STORAGE_TOLERANCE);
        assertBf16Close(new float[]{-1f, -0.0f, 1f, 3f}, ceilDestination, BF16_EXACT_STORAGE_TOLERANCE);
        assertBf16Close(new float[]{-1f, -0.0f, 1f, 1f}, signDestination, BF16_EXACT_STORAGE_TOLERANCE);
    }

    @Test
    void explicitShimExecuteBuffersSupportsBfloat16Relu() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        assumeTrue(bridge.capabilities().dtypeAbiV3Supported());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalMpsBridgeExecutable executable = bridge.compile(
                context,
                unaryPlan(1, 9, AcceleratorDagNodeType.RELU, Operation.OpType.RELU, DataType.BFLOAT16)
        );
        assumeTrue(executable.available(), executable.reason());
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        MetalBufferBinding input = null;
        MetalBufferBinding output = null;
        try {
            input = allocator.createInputBinding(1, bf16Tensor(new float[]{-1.0f, 2.0f}, new int[]{2}, "bf16ReluInput"));
            output = allocator.createOutputBinding(9, denseLayout(DataType.BFLOAT16, new int[]{2}));

            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(context, executable, List.of(input), List.of(output));
            Tensor destination = bf16Tensor(new float[]{0.0f, 0.0f}, new int[]{2}, "bf16ReluDestination");
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            assertBf16Close(new float[]{0.0f, 2.0f}, destination, BF16_EXACT_STORAGE_TOLERANCE);
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
    void explicitShimExecuteBuffersSupportsFloatBfloat16Cast() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        assumeTrue(bridge.capabilities().dtypeAbiV3Supported());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalMpsBridgeExecutable f32ToBf16 = bridge.compile(
                context,
                unaryPlan(1, 9, AcceleratorDagNodeType.CAST, Operation.OpType.CAST, 3,
                        new int[]{2}, new int[]{2}, DataType.FLOAT32, DataType.BFLOAT16)
        );
        MetalMpsBridgeExecutable bf16ToF32 = bridge.compile(
                context,
                unaryPlan(2, 10, AcceleratorDagNodeType.CAST, Operation.OpType.CAST, 1,
                        new int[]{2}, new int[]{2}, DataType.BFLOAT16, DataType.FLOAT32)
        );
        assumeTrue(f32ToBf16.available(), f32ToBf16.reason());
        assumeTrue(bf16ToF32.available(), bf16ToF32.reason());
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        MetalBufferBinding f32Input = null;
        MetalBufferBinding bf16Output = null;
        MetalBufferBinding bf16Input = null;
        MetalBufferBinding f32Output = null;
        try {
            f32Input = allocator.createInputBinding(
                    1,
                    new Tensor(new float[]{1.5f, -2.25f}, new int[]{2}, null, "castF32Input", DataType.FLOAT32)
            );
            bf16Output = allocator.createOutputBinding(9, denseLayout(DataType.BFLOAT16, new int[]{2}));
            MetalMpsBridgeExecutionStats firstStats = bridge.executeBuffers(context, f32ToBf16, List.of(f32Input), List.of(bf16Output));
            Tensor bf16Destination = bf16Tensor(new float[]{0.0f, 0.0f}, new int[]{2}, "castBf16Destination");
            allocator.readToCpu(bf16Output, bf16Destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);
            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, firstStats.executionPath());
            assertBf16Close(new float[]{1.5f, -2.25f}, bf16Destination, BF16_EXACT_STORAGE_TOLERANCE);

            bf16Input = allocator.createInputBinding(2, bf16Tensor(new float[]{1.5f, -2.25f}, new int[]{2}, "castBf16Input"));
            f32Output = allocator.createOutputBinding(10, denseF32Layout(new int[]{2}));
            MetalMpsBridgeExecutionStats secondStats = bridge.executeBuffers(context, bf16ToF32, List.of(bf16Input), List.of(f32Output));
            Tensor f32Destination = new Tensor(new float[]{0.0f, 0.0f}, new int[]{2}, null, "castF32Destination", DataType.FLOAT32);
            allocator.readToCpu(f32Output, f32Destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);
            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, secondStats.executionPath());
            assertArrayEquals(new float[]{1.5f, -2.25f}, f32Destination.getFloat32Data(), 0.0f);
        } finally {
            if (f32Input != null) allocator.destroy(f32Input.handle());
            if (bf16Output != null) allocator.destroy(bf16Output.handle());
            if (bf16Input != null) allocator.destroy(bf16Input.handle());
            if (f32Output != null) allocator.destroy(f32Output.handle());
        }
    }

    @Test
    void explicitShimExecuteBuffersSupportsBoolCompareOutput() {
        Tensor destination = executeBoolLoweredPlan(
                binaryPlan(
                        1,
                        2,
                        9,
                        AcceleratorDagNodeType.GT,
                        Operation.OpType.GT,
                        new int[]{2, 2},
                        new int[]{2, 2},
                        new int[]{2, 2},
                        DataType.FLOAT32,
                        DataType.BOOL
                ),
                List.of(
                        new Tensor(new float[]{1f, 3f, 2f, 4f}, new int[]{2, 2}, null, "boolGtLeft", DataType.FLOAT32),
                        new Tensor(new float[]{2f, 2f, 2f, 4f}, new int[]{2, 2}, null, "boolGtRight", DataType.FLOAT32)
                ),
                new int[]{2, 2}
        );

        assertArrayEquals(new byte[]{0, 1, 0, 0}, destination.getBoolData());
    }

    @Test
    void explicitShimExecuteBuffersSupportsBoolLogicalAndOutput() {
        Tensor destination = executeBoolLoweredPlan(
                binaryPlan(
                        1,
                        2,
                        9,
                        AcceleratorDagNodeType.LOGICAL_AND,
                        Operation.OpType.LOGICAL_AND,
                        new int[]{2, 2},
                        new int[]{2, 2},
                        new int[]{2, 2},
                        DataType.BOOL,
                        DataType.BOOL
                ),
                List.of(
                        new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "boolAndLeft", DataType.BOOL),
                        new Tensor(new byte[]{1, 1, 0, 0}, new int[]{2, 2}, null, "boolAndRight", DataType.BOOL)
                ),
                new int[]{2, 2}
        );

        assertArrayEquals(new byte[]{1, 0, 0, 0}, destination.getBoolData());
    }

    @Test
    void explicitShimExecuteBuffersSupportsBoolAnyReductionOutput() {
        Tensor destination = executeBoolLoweredPlan(
                reductionPlan(
                        1,
                        9,
                        AcceleratorDagNodeType.REDUCE_ANY,
                        Operation.OpType.REDUCE_ANY,
                        1,
                        true,
                        new int[]{2, 3},
                        new int[]{2, 1},
                        DataType.BOOL
                ),
                List.of(new Tensor(new byte[]{0, 1, 0, 0, 0, 0}, new int[]{2, 3}, null, "boolAnyInput", DataType.BOOL)),
                new int[]{2, 1}
        );

        assertArrayEquals(new byte[]{1, 0}, destination.getBoolData());
    }

    @Test
    void explicitShimExecuteBuffersSupportsGatherWithInt32Indices() {
        Tensor destination = executeIndexLoweredPlan(
                indexPlan(
                        1,
                        2,
                        9,
                        AcceleratorDagNodeType.GATHER,
                        Operation.OpType.GATHER,
                        1,
                        new int[]{2, 3},
                        new int[]{2},
                        new int[]{2}
                ),
                new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "gatherValueInput", DataType.FLOAT32),
                new Tensor(new int[]{2, 0}, new int[]{2}, null, "gatherIndexInput", DataType.INT32),
                new int[]{2}
        );

        assertArrayEquals(new float[]{3f, 4f}, destination.getFloat32Data(), 0.0f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsRankThreeGatherWithInt32Indices() {
        Tensor destination = executeIndexLoweredPlan(
                indexPlan(
                        1,
                        2,
                        9,
                        AcceleratorDagNodeType.GATHER,
                        Operation.OpType.GATHER,
                        2,
                        new int[]{2, 2, 3},
                        new int[]{2, 2},
                        new int[]{2, 2}
                ),
                new Tensor(new float[]{
                        1f, 2f, 3f,
                        4f, 5f, 6f,
                        7f, 8f, 9f,
                        10f, 11f, 12f
                }, new int[]{2, 2, 3}, null, "rank3GatherValueInput", DataType.FLOAT32),
                new Tensor(new int[]{2, 0, 1, 1}, new int[]{2, 2}, null, "rank3GatherIndexInput", DataType.INT32),
                new int[]{2, 2}
        );

        assertArrayEquals(new float[]{3f, 4f, 8f, 11f}, destination.getFloat32Data(), 0.0f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsTakeAlongAxisWithInt32Indices() {
        Tensor destination = executeIndexLoweredPlan(
                indexPlan(
                        1,
                        2,
                        9,
                        AcceleratorDagNodeType.TAKE_ALONG_AXIS,
                        Operation.OpType.TAKE_ALONG_AXIS,
                        1,
                        new int[]{2, 3},
                        new int[]{2, 2},
                        new int[]{2, 2}
                ),
                new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "takeValueInput", DataType.FLOAT32),
                new Tensor(new int[]{2, 1, 0, 0}, new int[]{2, 2}, null, "takeIndexInput", DataType.INT32),
                new int[]{2, 2}
        );

        assertArrayEquals(new float[]{3f, 2f, 4f, 4f}, destination.getFloat32Data(), 0.0f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsAxisZeroTakeAlongAxisWithInt32Indices() {
        Tensor destination = executeIndexLoweredPlan(
                indexPlan(
                        1,
                        2,
                        9,
                        AcceleratorDagNodeType.TAKE_ALONG_AXIS,
                        Operation.OpType.TAKE_ALONG_AXIS,
                        0,
                        new int[]{3, 2},
                        new int[]{2, 2},
                        new int[]{2, 2}
                ),
                new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "takeAxisZeroValueInput", DataType.FLOAT32),
                new Tensor(new int[]{2, 0, 1, 1}, new int[]{2, 2}, null, "takeAxisZeroIndexInput", DataType.INT32),
                new int[]{2, 2}
        );

        assertArrayEquals(new float[]{5f, 2f, 3f, 4f}, destination.getFloat32Data(), 0.0f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsExpandAndSelectShapeOps() {
        Tensor expectedExpand = new Tensor(new float[]{2f, 4f, 6f}, new int[]{1, 3}, null, "expectedExpandInput", DataType.FLOAT32)
                .expand(2, 3);
        expectedExpand.compute();
        Tensor expandInput = new Tensor(new float[]{2f, 4f, 6f}, new int[]{1, 3}, null, "expandInput", DataType.FLOAT32);
        Tensor expanded = expandInput.expand(2, 3);

        Tensor expandDestination = executeF32LoweredNode(
                expanded,
                Operation.OpType.EXPAND,
                List.of(expandInput),
                new int[]{2, 3}
        );

        Tensor expectedSelect = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, null, "expectedSelectInput", DataType.FLOAT32).select(0, 1);
        expectedSelect.compute();
        Tensor selectInput = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, null, "selectInput", DataType.FLOAT32);
        Tensor selected = selectInput.select(0, 1);

        Tensor selectDestination = executeF32LoweredNode(
                selected,
                Operation.OpType.SELECT,
                List.of(selectInput),
                new int[]{3}
        );

        assertArrayEquals(new float[]{2f, 4f, 6f, 2f, 4f, 6f}, expandDestination.getFloat32Data(), 0.0f);
        assertArrayEquals(new float[]{4f, 5f, 6f}, selectDestination.getFloat32Data(), 0.0f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsGatherAxisAndStaticLayoutOps() {
        Tensor gatherInput = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, null, "metal68NativeGatherInput", DataType.FLOAT32);
        Tensor gatherIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metal68NativeGatherIndices", DataType.INT32);
        Tensor gathered = gatherInput.gatherAxis(gatherIndices, 1);

        Tensor gatherDestination = executeF32LoweredNode(
                gathered,
                Operation.OpType.GATHER_AXIS,
                List.of(gatherInput, gatherIndices),
                new int[]{2, 2}
        );

        Tensor sliceInput = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, null, "metal68NativeSliceInput", DataType.FLOAT32);
        Tensor sliced = sliceInput.slice(new int[]{0, 1}, new int[]{2, 3}, new int[]{0, 1}, new int[]{1, 1});
        Tensor sliceDestination = executeF32LoweredNode(
                sliced,
                Operation.OpType.SLICE,
                List.of(sliceInput),
                new int[]{2, 2}
        );

        Tensor padInput = new Tensor(new float[]{
                1f, 2f,
                3f, 4f
        }, new int[]{2, 2}, null, "metal68NativePadInput", DataType.FLOAT32);
        Tensor padded = padInput.pad(new int[]{1, 0}, new int[]{0, 1}, -1.0);
        Tensor padDestination = executeF32LoweredNode(
                padded,
                Operation.OpType.PAD,
                List.of(padInput),
                new int[]{3, 3}
        );

        Tensor tileInput = new Tensor(new float[]{1f, 2f, 3f}, new int[]{1, 3}, null, "metal68NativeTileInput", DataType.FLOAT32);
        Tensor tiled = tileInput.tile(2, 1);
        Tensor tileDestination = executeF32LoweredNode(
                tiled,
                Operation.OpType.TILE,
                List.of(tileInput),
                new int[]{2, 3}
        );

        Tensor concatLeft = new Tensor(new float[]{
                1f, 2f,
                3f, 4f
        }, new int[]{2, 2}, null, "metal68NativeConcatLeft", DataType.FLOAT32);
        Tensor concatRight = new Tensor(new float[]{
                5f, 6f,
                7f, 8f
        }, new int[]{2, 2}, null, "metal68NativeConcatRight", DataType.FLOAT32);
        Tensor concatenated = Tensor.concat(1, concatLeft, concatRight);
        Tensor concatDestination = executeF32LoweredNode(
                concatenated,
                Operation.OpType.CONCAT,
                List.of(concatLeft, concatRight),
                new int[]{2, 4}
        );

        assertArrayEquals(new float[]{3f, 1f, 6f, 4f}, gatherDestination.getFloat32Data(), 0.0f);
        assertArrayEquals(new float[]{2f, 3f, 5f, 6f}, sliceDestination.getFloat32Data(), 0.0f);
        assertArrayEquals(new float[]{
                -1f, -1f, -1f,
                1f, 2f, -1f,
                3f, 4f, -1f
        }, padDestination.getFloat32Data(), 0.0f);
        assertArrayEquals(new float[]{1f, 2f, 3f, 1f, 2f, 3f}, tileDestination.getFloat32Data(), 0.0f);
        assertArrayEquals(new float[]{1f, 2f, 5f, 6f, 3f, 4f, 7f, 8f}, concatDestination.getFloat32Data(), 0.0f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsGatherNdWithBatchDims() {
        Tensor data = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f,
                10f, 11f, 12f
        }, new int[]{2, 2, 3}, null, "metalGatherNdData", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[]{1, 0}, new int[]{2, 1, 1}, null, "metalGatherNdIndices", DataType.INT32);
        Tensor gathered = data.gatherNd(indices, 1);

        Tensor destination = executeF32LoweredNode(
                gathered,
                Operation.OpType.GATHER_ND,
                List.of(data, indices),
                new int[]{2, 1, 3}
        );

        assertArrayEquals(new float[]{4f, 5f, 6f, 7f, 8f, 9f}, destination.getFloat32Data(), 0.0f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsBfloat16GatherNd() {
        Tensor data = bf16Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f,
                10f, 11f, 12f
        }, new int[]{2, 2, 3}, "metalBf16GatherNdData");
        Tensor indices = new Tensor(new int[]{1, 0}, new int[]{2, 1, 1}, null, "metalBf16GatherNdIndices", DataType.INT32);
        Tensor gathered = data.gatherNd(indices, 1);

        Tensor destination = executeBf16LoweredNode(
                gathered,
                Operation.OpType.GATHER_ND,
                List.of(data, indices),
                new int[]{2, 1, 3}
        );

        assertArrayEquals(new float[]{4f, 5f, 6f, 7f, 8f, 9f}, bf16Floats(destination), 0.0f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsSliceGradAsZeroPad() {
        Tensor outGrad = new Tensor(new float[]{10f, 20f, 30f, 40f}, new int[]{2, 2}, null, "metal72NativeSliceGradOutGrad", DataType.FLOAT32);
        Tensor sliceGrad = TensorPrimitiveBuilder.unaryNoGrad(
                outGrad,
                new int[]{2, 4},
                new sliceGrad(new int[]{0, 1}, new int[]{0, 1}, new int[]{1, 1}, new int[]{2, 4}),
                "metal72NativeSliceGrad",
                DataType.FLOAT32
        );
        Tensor destination = executeF32LoweredNode(
                sliceGrad,
                Operation.OpType.SLICE_GRAD,
                List.of(outGrad),
                new int[]{2, 4}
        );

        assertArrayEquals(new float[]{
                0f, 10f, 20f, 0f,
                0f, 30f, 40f, 0f
        }, destination.getFloat32Data(), 0.0f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsBfloat16SliceGradAsZeroPad() {
        Tensor outGrad = bf16Tensor(new float[]{10f, 20f, 30f, 40f}, new int[]{2, 2}, "metal72NativeBf16SliceGradOutGrad");
        Tensor sliceGrad = TensorPrimitiveBuilder.unaryNoGrad(
                outGrad,
                new int[]{2, 4},
                new sliceGrad(new int[]{0, 1}, new int[]{0, 1}, new int[]{1, 1}, new int[]{2, 4}),
                "metal72NativeBf16SliceGrad",
                DataType.BFLOAT16
        );
        Tensor destination = executeBf16LoweredNode(
                sliceGrad,
                Operation.OpType.SLICE_GRAD,
                List.of(outGrad),
                new int[]{2, 4}
        );

        assertBf16Close(new float[]{
                0f, 10f, 20f, 0f,
                0f, 30f, 40f, 0f
        }, destination, BF16_EXACT_STORAGE_TOLERANCE);
    }

    @Test
    void explicitShimExecuteBuffersSupportsBoolLayoutShapeOps() {
        Tensor reshapeInput = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "boolReshapeInput", DataType.BOOL);
        Tensor reshaped = reshapeInput.reshape(4);
        Tensor reshapeDestination = executeBoolLoweredNode(
                reshaped,
                Operation.OpType.RESHAPE,
                List.of(reshapeInput),
                new int[]{4}
        );

        Tensor permuteInput = new Tensor(new byte[]{1, 0, 1, 0, 1, 0}, new int[]{2, 3}, null, "boolPermuteInput", DataType.BOOL);
        Tensor permuted = permuteInput.permute(1, 0);
        Tensor permuteDestination = executeBoolLoweredNode(
                permuted,
                Operation.OpType.PERMUTE,
                List.of(permuteInput),
                new int[]{3, 2}
        );

        Tensor expandInput = new Tensor(new byte[]{1, 0, 1}, new int[]{1, 3}, null, "boolExpandInput", DataType.BOOL);
        Tensor expanded = expandInput.expand(2, 3);
        Tensor expandDestination = executeBoolLoweredNode(
                expanded,
                Operation.OpType.EXPAND,
                List.of(expandInput),
                new int[]{2, 3}
        );

        Tensor expandDimsInput = new Tensor(new byte[]{1, 0, 0, 1}, new int[]{2, 2}, null, "boolExpandDimsInput", DataType.BOOL);
        Tensor expandDims = expandDimsInput.expandDims(0);
        Tensor expandDimsDestination = executeBoolLoweredNode(
                expandDims,
                Operation.OpType.EXPAND_DIMS,
                List.of(expandDimsInput),
                new int[]{1, 2, 2}
        );

        Tensor squeezeInput = new Tensor(new byte[]{1, 0, 0, 1}, new int[]{1, 2, 2}, null, "boolSqueezeInput", DataType.BOOL);
        Tensor squeezed = squeezeInput.squeeze(0);
        Tensor squeezeDestination = executeBoolLoweredNode(
                squeezed,
                Operation.OpType.SQUEEZE,
                List.of(squeezeInput),
                new int[]{2, 2}
        );

        Tensor selectInput = new Tensor(new byte[]{1, 1, 0, 0, 1, 0}, new int[]{2, 3}, null, "boolSelectInput", DataType.BOOL);
        Tensor selected = selectInput.select(0, 1);
        Tensor selectDestination = executeBoolLoweredNode(
                selected,
                Operation.OpType.SELECT,
                List.of(selectInput),
                new int[]{3}
        );

        Tensor contiguousInput = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{2, 2}, null, "boolContiguousInput", DataType.BOOL);
        Tensor contiguous = contiguousInput.contiguous();
        Tensor contiguousDestination = executeBoolLoweredNode(
                contiguous,
                Operation.OpType.CONTIGUOUS,
                List.of(contiguousInput),
                new int[]{2, 2}
        );

        Tensor noopInput = new Tensor(new byte[]{0, 1, 1, 0}, new int[]{2, 2}, null, "boolNoopInput", DataType.BOOL);
        Tensor noop = TensorPrimitiveBuilder.unary(
                noopInput,
                noopInput.getShape(),
                new operations.layout.noop(),
                "boolNoop",
                DataType.BOOL
        );
        Tensor noopDestination = executeBoolLoweredNode(
                noop,
                Operation.OpType.NOOP,
                List.of(noopInput),
                new int[]{2, 2}
        );

        assertArrayEquals(new byte[]{1, 0, 1, 0}, reshapeDestination.getBoolData());
        assertArrayEquals(new byte[]{1, 0, 0, 1, 1, 0}, permuteDestination.getBoolData());
        assertArrayEquals(new byte[]{1, 0, 1, 1, 0, 1}, expandDestination.getBoolData());
        assertArrayEquals(new byte[]{1, 0, 0, 1}, expandDimsDestination.getBoolData());
        assertArrayEquals(new byte[]{1, 0, 0, 1}, squeezeDestination.getBoolData());
        assertArrayEquals(new byte[]{0, 1, 0}, selectDestination.getBoolData());
        assertArrayEquals(new byte[]{1, 0, 1, 1}, contiguousDestination.getBoolData());
        assertArrayEquals(new byte[]{0, 1, 1, 0}, noopDestination.getBoolData());
    }

    @Test
    void explicitShimExecuteBuffersSupportsScatterAddWithInt32Indices() {
        Tensor expectedBase = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, null, "expectedScatterBase", DataType.FLOAT32);
        Tensor expectedIndices = new Tensor(new int[]{1, 0}, new int[]{2}, null, "expectedScatterIndices", DataType.INT32);
        Tensor expectedSrc = new Tensor(new float[]{10f, 20f}, new int[]{2}, null, "expectedScatterSrc", DataType.FLOAT32);
        Tensor expected = expectedBase.scatterAdd(expectedIndices, expectedSrc, 1);
        expected.compute();

        Tensor base = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, null, "scatterBase", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[]{1, 0}, new int[]{2}, null, "scatterIndices", DataType.INT32);
        Tensor src = new Tensor(new float[]{10f, 20f}, new int[]{2}, null, "scatterSrc", DataType.FLOAT32);
        Tensor out = base.scatterAdd(indices, src, 1);

        Tensor destination = executeF32LoweredNode(
                out,
                Operation.OpType.SCATTER_ADD,
                List.of(base, indices, src),
                new int[]{2, 3}
        );

        assertArrayEquals(expected.getFloat32Data(), destination.getFloat32Data(), 0.0f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsScatterElementsWithInt32Indices() {
        Tensor expectedBase = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, null, "expectedScatterElementsBase", DataType.FLOAT32);
        Tensor expectedIndices = new Tensor(new int[]{2, 0, 1, 2}, new int[]{2, 2}, null, "expectedScatterElementsIndices", DataType.INT32);
        Tensor expectedUpdates = new Tensor(new float[]{10f, 20f, 30f, 40f}, new int[]{2, 2}, null, "expectedScatterElementsUpdates", DataType.FLOAT32);
        Tensor expected = expectedBase.scatterElements(expectedIndices, expectedUpdates, 1, operations.index.ScatterReduction.ADD);
        expected.compute();

        Tensor base = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, null, "scatterElementsBase", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[]{2, 0, 1, 2}, new int[]{2, 2}, null, "scatterElementsIndices", DataType.INT32);
        Tensor updates = new Tensor(new float[]{10f, 20f, 30f, 40f}, new int[]{2, 2}, null, "scatterElementsUpdates", DataType.FLOAT32);
        Tensor out = base.scatterElements(indices, updates, 1, operations.index.ScatterReduction.ADD);

        Tensor destination = executeF32LoweredNode(
                out,
                Operation.OpType.SCATTER_ELEMENTS,
                List.of(base, indices, updates),
                new int[]{2, 3}
        );

        assertArrayEquals(expected.getFloat32Data(), destination.getFloat32Data(), 0.0f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsScatterNdWithInt32Indices() {
        Tensor expectedBase = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, null, "expectedScatterNdBase", DataType.FLOAT32);
        Tensor expectedIndices = new Tensor(new int[]{0, 1, 1, 2}, new int[]{2, 2}, null, "expectedScatterNdIndices", DataType.INT32);
        Tensor expectedUpdates = new Tensor(new float[]{10f, 40f}, new int[]{2}, null, "expectedScatterNdUpdates", DataType.FLOAT32);
        Tensor expected = expectedBase.scatterNd(expectedIndices, expectedUpdates, operations.index.ScatterReduction.MAX);
        expected.compute();

        Tensor base = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, null, "scatterNdBase", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[]{0, 1, 1, 2}, new int[]{2, 2}, null, "scatterNdIndices", DataType.INT32);
        Tensor updates = new Tensor(new float[]{10f, 40f}, new int[]{2}, null, "scatterNdUpdates", DataType.FLOAT32);
        Tensor out = base.scatterNd(indices, updates, operations.index.ScatterReduction.MAX);

        Tensor destination = executeF32LoweredNode(
                out,
                Operation.OpType.SCATTER_ND,
                List.of(base, indices, updates),
                new int[]{2, 3}
        );

        assertArrayEquals(expected.getFloat32Data(), destination.getFloat32Data(), 0.0f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsBfloat16ScatterAddWithInt32Indices() {
        Tensor base = bf16Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, "bf16ScatterBase");
        Tensor indices = new Tensor(new int[]{1, 0}, new int[]{2}, null, "bf16ScatterIndices", DataType.INT32);
        Tensor src = bf16Tensor(new float[]{10f, 20f}, new int[]{2}, "bf16ScatterSrc");
        Tensor expected = base.scatterAdd(indices, src, 1);
        expected.compute();
        Tensor out = base.scatterAdd(indices, src, 1);

        Tensor destination = executeBf16LoweredNode(
                out,
                Operation.OpType.SCATTER_ADD,
                List.of(base, indices, src),
                new int[]{2, 3}
        );

        assertBf16Close(bf16Floats(expected), destination, BF16_MATMUL_REDUCTION_TOLERANCE);
    }

    @Test
    void explicitShimExecuteBuffersSupportsIndexGradientScatterAdd() {
        Tensor gatherIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "gatherGradIndices", DataType.INT32);
        Tensor gatherOutGrad = new Tensor(new float[]{3f, 5f}, new int[]{2}, null, "gatherGradOutGrad", DataType.FLOAT32);
        Tensor gatherGrad = TensorPrimitiveBuilder.binaryNoGrad(
                gatherIndices,
                gatherOutGrad,
                new int[]{2, 3},
                new operations.index.gatherGrad(1),
                "gatherGrad",
                DataType.FLOAT32
        );

        Tensor gatherDestination = executeF32LoweredNode(
                gatherGrad,
                Operation.OpType.GATHER_GRAD,
                List.of(gatherIndices, gatherOutGrad),
                new int[]{2, 3}
        );

        Tensor takeIndices = new Tensor(new int[]{
                1, 1, 0,
                2, 0, 2
        }, new int[]{2, 3}, null, "takeGradIndices", DataType.INT32);
        Tensor takeOutGrad = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, null, "takeGradOutGrad", DataType.FLOAT32);
        Tensor takeGrad = TensorPrimitiveBuilder.binaryNoGrad(
                takeIndices,
                takeOutGrad,
                new int[]{2, 3},
                new operations.index.takeAlongAxisGrad(1),
                "takeAlongAxisGrad",
                DataType.FLOAT32
        );

        Tensor takeDestination = executeF32LoweredNode(
                takeGrad,
                Operation.OpType.TAKE_ALONG_AXIS_GRAD,
                List.of(takeIndices, takeOutGrad),
                new int[]{2, 3}
        );

        assertArrayEquals(new float[]{
                0f, 0f, 3f,
                5f, 0f, 0f
        }, gatherDestination.getFloat32Data(), 0.0f);
        assertArrayEquals(new float[]{
                3f, 3f, 0f,
                5f, 0f, 10f
        }, takeDestination.getFloat32Data(), 0.0f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsBfloat16GatherGrad() {
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "bf16GatherGradIndices", DataType.INT32);
        Tensor outGrad = bf16Tensor(new float[]{3f, 5f}, new int[]{2}, "bf16GatherGradOutGrad");
        Tensor gatherGrad = TensorPrimitiveBuilder.binaryNoGrad(
                indices,
                outGrad,
                new int[]{2, 3},
                new operations.index.gatherGrad(1),
                "bf16GatherGrad",
                DataType.BFLOAT16
        );

        Tensor destination = executeBf16LoweredNode(
                gatherGrad,
                Operation.OpType.GATHER_GRAD,
                List.of(indices, outGrad),
                new int[]{2, 3}
        );

        assertBf16Close(new float[]{
                0f, 0f, 3f,
                5f, 0f, 0f
        }, destination, BF16_EXACT_STORAGE_TOLERANCE);
    }

    @Test
    void explicitShimExecuteBuffersSupportsFloat32Conv2dNoBias() {
        Tensor destination = executeConv2dLoweredPlan(
                conv2dPlan(
                        1,
                        2,
                        -1,
                        9,
                        new int[]{1, 1, 3, 3},
                        new int[]{1, 1, 2, 2},
                        new int[]{1, 1, 2, 2},
                        1,
                        1,
                        0,
                        0
                ),
                List.of(
                        new Tensor(new float[]{
                                1f, 2f, 3f,
                                4f, 5f, 6f,
                                7f, 8f, 9f
                        }, new int[]{1, 1, 3, 3}, null, "conv2dNoBiasInput", DataType.FLOAT32),
                        new Tensor(new float[]{
                                1f, 0f,
                                0f, 1f
                        }, new int[]{1, 1, 2, 2}, null, "conv2dNoBiasWeight", DataType.FLOAT32)
                ),
                new int[]{1, 1, 2, 2}
        );

        assertArrayEquals(new float[]{6f, 8f, 12f, 14f}, destination.getFloat32Data(), 1.0e-5f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsFloat32Conv2dBiasStridePadding() {
        Tensor destination = executeConv2dLoweredPlan(
                conv2dPlan(
                        1,
                        2,
                        3,
                        9,
                        new int[]{1, 1, 4, 4},
                        new int[]{1, 1, 2, 2},
                        new int[]{1, 1, 3, 3},
                        2,
                        2,
                        1,
                        1
                ),
                List.of(
                        new Tensor(new float[]{
                                1f, 2f, 3f, 4f,
                                5f, 6f, 7f, 8f,
                                9f, 10f, 11f, 12f,
                                13f, 14f, 15f, 16f
                        }, new int[]{1, 1, 4, 4}, null, "conv2dBiasInput", DataType.FLOAT32),
                        new Tensor(new float[]{
                                1f, 1f,
                                1f, 1f
                        }, new int[]{1, 1, 2, 2}, null, "conv2dBiasWeight", DataType.FLOAT32),
                        new Tensor(new float[]{0.5f}, new int[]{1}, null, "conv2dBias", DataType.FLOAT32)
                ),
                new int[]{1, 1, 3, 3}
        );

        assertArrayEquals(new float[]{
                1.5f, 5.5f, 4.5f,
                14.5f, 34.5f, 20.5f,
                13.5f, 29.5f, 16.5f
        }, destination.getFloat32Data(), 1.0e-5f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsFloat32Conv2dBackwardInputAndWeight() {
        Conv2dOptions options = Conv2dOptions.defaults();

        Tensor expectedWeight = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 1, 2, 2}, null, "expectedConvBackwardWeightInput", DataType.FLOAT32);
        Tensor expectedOutGrad = new Tensor(new float[]{
                1f, 2f,
                3f, 4f
        }, new int[]{1, 1, 2, 2}, null, "expectedConvBackwardOutGrad", DataType.FLOAT32);
        Tensor expectedInputGrad = TensorPrimitiveBuilder.binaryNoGrad(
                expectedWeight,
                expectedOutGrad,
                new int[]{1, 1, 3, 3},
                new operations.nn.conv.conv2dBackwardInput(options, new int[]{1, 1, 3, 3}),
                "expectedConv2dBackwardInput",
                DataType.FLOAT32
        );
        expectedInputGrad.compute();

        Tensor weight = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 1, 2, 2}, null, "convBackwardWeightInput", DataType.FLOAT32);
        Tensor outGradForInput = new Tensor(new float[]{
                1f, 2f,
                3f, 4f
        }, new int[]{1, 1, 2, 2}, null, "convBackwardOutGradInput", DataType.FLOAT32);
        Tensor inputGrad = TensorPrimitiveBuilder.binaryNoGrad(
                weight,
                outGradForInput,
                new int[]{1, 1, 3, 3},
                new operations.nn.conv.conv2dBackwardInput(options, new int[]{1, 1, 3, 3}),
                "conv2dBackwardInput",
                DataType.FLOAT32
        );

        Tensor inputGradDestination = executeF32LoweredNode(
                inputGrad,
                Operation.OpType.CONV2D_BACKWARD_INPUT,
                List.of(weight, outGradForInput),
                new int[]{1, 1, 3, 3}
        );

        Tensor expectedInput = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f
        }, new int[]{1, 1, 3, 3}, null, "expectedConvBackwardInputSource", DataType.FLOAT32);
        Tensor expectedWeightGrad = TensorPrimitiveBuilder.binaryNoGrad(
                expectedInput,
                expectedOutGrad,
                new int[]{1, 1, 2, 2},
                new operations.nn.conv.conv2dBackwardWeight(options, new int[]{1, 1, 2, 2}),
                "expectedConv2dBackwardWeight",
                DataType.FLOAT32
        );
        expectedWeightGrad.compute();

        Tensor input = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f
        }, new int[]{1, 1, 3, 3}, null, "convBackwardInputSource", DataType.FLOAT32);
        Tensor outGradForWeight = new Tensor(new float[]{
                1f, 2f,
                3f, 4f
        }, new int[]{1, 1, 2, 2}, null, "convBackwardOutGradWeight", DataType.FLOAT32);
        Tensor weightGrad = TensorPrimitiveBuilder.binaryNoGrad(
                input,
                outGradForWeight,
                new int[]{1, 1, 2, 2},
                new operations.nn.conv.conv2dBackwardWeight(options, new int[]{1, 1, 2, 2}),
                "conv2dBackwardWeight",
                DataType.FLOAT32
        );

        Tensor weightGradDestination = executeF32LoweredNode(
                weightGrad,
                Operation.OpType.CONV2D_BACKWARD_WEIGHT,
                List.of(input, outGradForWeight),
                new int[]{1, 1, 2, 2}
        );

        assertArrayEquals(expectedInputGrad.getFloat32Data(), inputGradDestination.getFloat32Data(), 1.0e-5f);
        assertArrayEquals(expectedWeightGrad.getFloat32Data(), weightGradDestination.getFloat32Data(), 1.0e-5f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsFloat32AvgPool2dBackwardInput() {
        Pool2dOptions options = new Pool2dOptions(2, 2, 1, 1, 0, 0, false);
        Tensor expectedOutGrad = new Tensor(new float[]{
                1f, 2f,
                3f, 4f
        }, new int[]{1, 1, 2, 2}, null, "expectedAvgPoolBackwardOutGrad", DataType.FLOAT32);
        Tensor expected = TensorPrimitiveBuilder.unaryNoGrad(
                expectedOutGrad,
                new int[]{1, 1, 3, 3},
                new operations.nn.pool.avgPool2dBackwardInput(options, new int[]{1, 1, 3, 3}),
                "expectedAvgPool2dBackwardInput",
                DataType.FLOAT32
        );
        expected.compute();

        Tensor outGrad = new Tensor(new float[]{
                1f, 2f,
                3f, 4f
        }, new int[]{1, 1, 2, 2}, null, "avgPoolBackwardOutGrad", DataType.FLOAT32);
        Tensor grad = TensorPrimitiveBuilder.unaryNoGrad(
                outGrad,
                new int[]{1, 1, 3, 3},
                new operations.nn.pool.avgPool2dBackwardInput(options, new int[]{1, 1, 3, 3}),
                "avgPool2dBackwardInput",
                DataType.FLOAT32
        );

        Tensor destination = executeF32LoweredNode(
                grad,
                Operation.OpType.AVG_POOL2D_BACKWARD_INPUT,
                List.of(outGrad),
                new int[]{1, 1, 3, 3}
        );

        assertArrayEquals(expected.getFloat32Data(), destination.getFloat32Data(), 1.0e-5f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsFloat32MaxPool2dBackwardInput() {
        Pool2dOptions options = new Pool2dOptions(2, 2, 1, 1, 0, 0, false);
        Tensor source = new Tensor(new float[]{
                1f, 5f, 2f,
                4f, 5f, 3f,
                7f, 6f, 8f
        }, new int[]{1, 1, 3, 3}, null, "maxPoolBackwardSource", DataType.FLOAT32);
        Tensor outGrad = new Tensor(new float[]{
                1f, 2f,
                3f, 4f
        }, new int[]{1, 1, 2, 2}, null, "maxPoolBackwardOutGrad", DataType.FLOAT32);
        Tensor expected = TensorPrimitiveBuilder.binaryNoGrad(
                outGrad,
                source,
                new int[]{1, 1, 3, 3},
                new operations.nn.pool.maxPool2dBackwardInput(options, new int[]{1, 1, 3, 3}),
                "expectedMaxPool2dBackwardInput",
                DataType.FLOAT32
        );
        expected.compute();

        Tensor grad = TensorPrimitiveBuilder.binaryNoGrad(
                outGrad,
                source,
                new int[]{1, 1, 3, 3},
                new operations.nn.pool.maxPool2dBackwardInput(options, new int[]{1, 1, 3, 3}),
                "maxPool2dBackwardInput",
                DataType.FLOAT32
        );

        Tensor destination = executeF32LoweredNode(
                grad,
                Operation.OpType.MAX_POOL2D_BACKWARD_INPUT,
                List.of(outGrad, source),
                new int[]{1, 1, 3, 3}
        );

        assertArrayEquals(expected.getFloat32Data(), destination.getFloat32Data(), 1.0e-5f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsFloat32CrossEntropyLossFromIndices() {
        Tensor logits = new Tensor(new float[]{
                1f, 2f, 3f,
                1f, 0f, -1f
        }, new int[]{2, 3}, null, "ceIndexLogits", DataType.FLOAT32);
        Tensor targets = new Tensor(new int[]{2, 0}, new int[]{2}, null, "ceIndexTargets", DataType.INT32);
        Tensor expected = logits.crossEntropyLossFromIndices(targets, 1);
        expected.compute();

        Tensor nativeLoss = TensorPrimitiveBuilder.naryNoGrad(
                new int[]{1},
                List.of(logits, targets),
                new operations.loss.crossEntropyLossIndices(1, tensor.loss.LossReduction.MEAN, null),
                "ceIndexNative",
                DataType.FLOAT32
        );

        Tensor destination = executeF32LoweredNode(
                nativeLoss,
                Operation.OpType.CROSS_ENTROPY_LOSS_INDICES,
                List.of(logits, targets),
                new int[]{1}
        );

        assertArrayEquals(expected.getFloat32Data(), destination.getFloat32Data(), 1.0e-5f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsBfloat16CrossEntropyLossFromIndices() {
        Tensor logits = bf16Tensor(new float[]{
                1f, 2f, 3f,
                1f, 0f, -1f
        }, new int[]{2, 3}, "bf16CeIndexLogits");
        Tensor targets = new Tensor(new int[]{2, 0}, new int[]{2}, null, "bf16CeIndexTargets", DataType.INT32);
        Tensor expected = logits.crossEntropyLossFromIndices(targets, 1);
        expected.compute();

        Tensor nativeLoss = TensorPrimitiveBuilder.naryNoGrad(
                new int[]{1},
                List.of(logits, targets),
                new operations.loss.crossEntropyLossIndices(1, tensor.loss.LossReduction.MEAN, null),
                "bf16CeIndexNative",
                DataType.BFLOAT16
        );

        Tensor destination = executeBf16LoweredNode(
                nativeLoss,
                Operation.OpType.CROSS_ENTROPY_LOSS_INDICES,
                List.of(logits, targets),
                new int[]{1}
        );

        assertBf16Close(bf16Floats(expected), destination, BF16_NORM_SOFTMAX_TOLERANCE);
    }

    @Test
    void explicitShimExecuteBuffersSupportsFloat32CrossEntropyLossFromIndicesGrad() {
        Tensor logits = new Tensor(new float[]{
                1f, 2f, 3f,
                1f, 0f, -1f
        }, new int[]{2, 3}, null, "ceIndexGradLogits", DataType.FLOAT32);
        Tensor targets = new Tensor(new int[]{2, 0}, new int[]{2}, null, "ceIndexGradTargets", DataType.INT32);
        Tensor sampleScale = new Tensor(new float[]{0.5f, 0.5f}, new int[]{2}, null, "ceIndexGradScale", DataType.FLOAT32);
        Tensor expected = TensorPrimitiveBuilder.ternaryNoGrad(
                logits,
                targets,
                sampleScale,
                new int[]{2, 3},
                new operations.loss.crossEntropyLossIndicesGrad(1),
                "expectedCeIndexGrad",
                DataType.FLOAT32
        );
        expected.compute();
        Tensor grad = TensorPrimitiveBuilder.ternaryNoGrad(
                logits,
                targets,
                sampleScale,
                new int[]{2, 3},
                new operations.loss.crossEntropyLossIndicesGrad(1),
                "ceIndexGrad",
                DataType.FLOAT32
        );

        Tensor destination = executeF32LoweredNode(
                grad,
                Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD,
                List.of(logits, targets, sampleScale),
                new int[]{2, 3}
        );

        assertArrayEquals(expected.getFloat32Data(), destination.getFloat32Data(), 1.0e-5f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsBfloat16CrossEntropyLossFromIndicesGrad() {
        Tensor logits = bf16Tensor(new float[]{
                1f, 2f, 3f,
                1f, 0f, -1f
        }, new int[]{2, 3}, "bf16CeIndexGradLogits");
        Tensor targets = new Tensor(new int[]{2, 0}, new int[]{2}, null, "bf16CeIndexGradTargets", DataType.INT32);
        Tensor sampleScale = bf16Tensor(new float[]{0.5f, 0.5f}, new int[]{2}, "bf16CeIndexGradScale");
        Tensor expected = TensorPrimitiveBuilder.ternaryNoGrad(
                logits,
                targets,
                sampleScale,
                new int[]{2, 3},
                new operations.loss.crossEntropyLossIndicesGrad(1),
                "expectedBf16CeIndexGrad",
                DataType.BFLOAT16
        );
        expected.compute();
        Tensor grad = TensorPrimitiveBuilder.ternaryNoGrad(
                logits,
                targets,
                sampleScale,
                new int[]{2, 3},
                new operations.loss.crossEntropyLossIndicesGrad(1),
                "bf16CeIndexGrad",
                DataType.BFLOAT16
        );

        Tensor destination = executeBf16LoweredNode(
                grad,
                Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD,
                List.of(logits, targets, sampleScale),
                new int[]{2, 3}
        );

        assertBf16Close(bf16Floats(expected), destination, BF16_NORM_SOFTMAX_TOLERANCE);
    }

    @Test
    void explicitShimExecuteBuffersSupportsFloat32MaxPool2d() {
        Tensor destination = executeConv2dLoweredPlan(
                pool2dPlan(
                        1,
                        9,
                        AcceleratorDagNodeType.MAX_POOL2D,
                        Operation.OpType.MAX_POOL2D,
                        new int[]{1, 1, 4, 4},
                        new int[]{1, 1, 2, 2},
                        2,
                        2,
                        2,
                        2,
                        0,
                        0,
                        false
                ),
                List.of(new Tensor(new float[]{
                        1f, 2f, 3f, 4f,
                        5f, 6f, 7f, 8f,
                        9f, 10f, 11f, 12f,
                        13f, 14f, 15f, 16f
                }, new int[]{1, 1, 4, 4}, null, "maxPoolInput", DataType.FLOAT32)),
                new int[]{1, 1, 2, 2}
        );

        assertArrayEquals(new float[]{6f, 8f, 14f, 16f}, destination.getFloat32Data(), 1.0e-5f);
    }

    @Test
    void explicitShimExecuteBuffersSupportsBfloat16Conv2dAndMaxPool2d() {
        Tensor input = bf16Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f
        }, new int[]{1, 1, 3, 3}, "bf16ConvInput");
        Tensor weight = bf16Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 1, 2, 2}, "bf16ConvWeight");
        Tensor expectedConv = input.conv2d(weight, Conv2dOptions.defaults());
        expectedConv.compute();
        Tensor conv = input.conv2d(weight, Conv2dOptions.defaults());

        Tensor convDestination = executeBf16LoweredNode(
                conv,
                Operation.OpType.CONV2D,
                List.of(input, weight),
                new int[]{1, 1, 2, 2}
        );

        Tensor poolInput = bf16Tensor(new float[]{
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f,
                13f, 14f, 15f, 16f
        }, new int[]{1, 1, 4, 4}, "bf16PoolInput");
        Tensor expectedPool = poolInput.maxPool2d(Pool2dOptions.square(2));
        expectedPool.compute();
        Tensor pool = poolInput.maxPool2d(Pool2dOptions.square(2));

        Tensor poolDestination = executeBf16LoweredNode(
                pool,
                Operation.OpType.MAX_POOL2D,
                List.of(poolInput),
                new int[]{1, 1, 2, 2}
        );

        assertBf16Close(bf16Floats(expectedConv), convDestination, BF16_MATMUL_REDUCTION_TOLERANCE);
        assertBf16Close(bf16Floats(expectedPool), poolDestination, BF16_EXACT_STORAGE_TOLERANCE);
    }

    @Test
    void explicitShimExecuteBuffersSupportsFloat32AvgPool2d() {
        Tensor destination = executeConv2dLoweredPlan(
                pool2dPlan(
                        1,
                        9,
                        AcceleratorDagNodeType.AVG_POOL2D,
                        Operation.OpType.AVG_POOL2D,
                        new int[]{1, 1, 4, 4},
                        new int[]{1, 1, 2, 2},
                        2,
                        2,
                        2,
                        2,
                        0,
                        0,
                        false
                ),
                List.of(new Tensor(new float[]{
                        1f, 2f, 3f, 4f,
                        5f, 6f, 7f, 8f,
                        9f, 10f, 11f, 12f,
                        13f, 14f, 15f, 16f
                }, new int[]{1, 1, 4, 4}, null, "avgPoolInput", DataType.FLOAT32)),
                new int[]{1, 1, 2, 2}
        );

        assertArrayEquals(new float[]{3.5f, 5.5f, 11.5f, 13.5f}, destination.getFloat32Data(), 1.0e-5f);
    }

    @Test
    void explicitShimBfloat16BufferRoundTripsRawStorageExactly() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        assumeTrue(bridge.capabilities().dtypeAbiV3Supported());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        MetalBufferBinding input = null;
        try {
            float[] values = new float[]{-1.5f, -0.125f, 0.0f, 1.25f, 8.0f};
            input = allocator.createInputBinding(1, bf16Tensor(values, new int[]{5}, "bf16RawSource"));
            Tensor destination = bf16Tensor(new float[values.length], new int[]{5}, "bf16RawDestination");

            allocator.readToCpu(input, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertBf16RawBitsEqual(values, destination);
            assertBf16Close(values, destination, BF16_EXACT_STORAGE_TOLERANCE);
        } finally {
            if (input != null) {
                allocator.destroy(input.handle());
            }
        }
    }

    @Test
    void explicitShimExecuteBuffersSupportsBfloat16Matmul() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        assumeTrue(bridge.capabilities().dtypeAbiV3Supported());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalMpsBridgeExecutable executable = bridge.compile(
                context,
                binaryPlan(
                        1,
                        2,
                        9,
                        AcceleratorDagNodeType.MATMUL,
                        Operation.OpType.MATMUL,
                        new int[]{2, 2},
                        new int[]{2, 2},
                        new int[]{2, 2},
                        DataType.BFLOAT16
                )
        );
        assumeTrue(executable.available(), executable.reason());
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        MetalBufferBinding left = null;
        MetalBufferBinding right = null;
        MetalBufferBinding output = null;
        try {
            left = allocator.createInputBinding(1, bf16Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, "bf16MatmulLeft"));
            right = allocator.createInputBinding(2, bf16Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, "bf16MatmulRight"));
            output = allocator.createOutputBinding(9, denseLayout(DataType.BFLOAT16, new int[]{2, 2}));

            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(context, executable, List.of(left, right), List.of(output));
            Tensor destination = bf16Tensor(new float[]{0f, 0f, 0f, 0f}, new int[]{2, 2}, "bf16MatmulDestination");
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            assertBf16Close(new float[]{19f, 22f, 43f, 50f}, destination, BF16_MATMUL_REDUCTION_TOLERANCE);
        } finally {
            if (left != null) {
                allocator.destroy(left.handle());
            }
            if (right != null) {
                allocator.destroy(right.handle());
            }
            if (output != null) {
                allocator.destroy(output.handle());
            }
        }
    }

    @Test
    void explicitShimExecuteBuffersSupportsBfloat16ReductionTolerance() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        assumeTrue(bridge.capabilities().dtypeAbiV3Supported());
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
                        new int[]{2, 1},
                        DataType.BFLOAT16
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
                    bf16Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, "bf16ReductionSource")
            );
            output = allocator.createOutputBinding(9, denseLayout(DataType.BFLOAT16, new int[]{2, 1}));

            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(context, executable, List.of(input), List.of(output));
            Tensor destination = bf16Tensor(new float[]{0.0f, 0.0f}, new int[]{2, 1}, "bf16ReductionDestination");
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            assertBf16Close(new float[]{6f, 15f}, destination, BF16_MATMUL_REDUCTION_TOLERANCE);
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
    void explicitShimExecuteBuffersSupportsReduceProdArgMaxAndCumSum() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        assumeTrue(bridge.capabilities().dtypeAbiV3Supported());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        executeF32Unary(
                bridge,
                context,
                allocator,
                unaryPlan(1, 9, AcceleratorDagNodeType.REDUCE_PROD, Operation.OpType.REDUCE_PROD,
                        encodeReductionMode(1, true), new int[]{2, 3}, new int[]{2, 1}, DataType.FLOAT32, DataType.FLOAT32),
                new float[]{2f, 3f, 4f, 5f, 6f, 7f},
                new int[]{2, 3},
                new float[]{24f, 210f},
                new int[]{2, 1},
                "reduceProd"
        );
        executeInt32Unary(
                bridge,
                context,
                allocator,
                unaryPlan(1, 9, AcceleratorDagNodeType.ARGMAX, Operation.OpType.ARGMAX,
                        encodeReductionMode(1, true), new int[]{2, 3}, new int[]{2, 1}, DataType.FLOAT32, DataType.INT32),
                new float[]{1f, 5f, 5f, 4f, 9f, 2f},
                new int[]{2, 3},
                new int[]{1, 1},
                new int[]{2, 1},
                "argMax"
        );
        executeF32Unary(
                bridge,
                context,
                allocator,
                unaryPlan(1, 9, AcceleratorDagNodeType.CUMSUM, Operation.OpType.CUMSUM,
                        encodeCumSumMode(1, true, true), new int[]{2, 3}, new int[]{2, 3}, DataType.FLOAT32, DataType.FLOAT32),
                new float[]{1f, 2f, 3f, 4f, 5f, 6f},
                new int[]{2, 3},
                new float[]{5f, 3f, 0f, 11f, 6f, 0f},
                new int[]{2, 3},
                "cumSum"
        );
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
    void explicitShimExecuteBuffersSupportsBfloat16LayerNormTolerance() {
        Tensor expectedSource = bf16Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, "expectedBf16LayerNormSource");
        Tensor expectedGamma = bf16Tensor(new float[]{1f, 1f, 1f}, new int[]{3}, "expectedBf16LayerNormGamma");
        Tensor expectedBeta = bf16Tensor(new float[]{0f, 0f, 0f}, new int[]{3}, "expectedBf16LayerNormBeta");
        Tensor expected = expectedSource.layerNorm(expectedGamma, expectedBeta, 1.0e-5);
        expected.compute();

        Tensor source = bf16Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, "bf16LayerNormSource");
        Tensor gamma = bf16Tensor(new float[]{1f, 1f, 1f}, new int[]{3}, "bf16LayerNormGamma");
        Tensor beta = bf16Tensor(new float[]{0f, 0f, 0f}, new int[]{3}, "bf16LayerNormBeta");
        Tensor out = source.layerNorm(gamma, beta, 1.0e-5);

        Tensor destination = executeBf16LoweredNode(
                out,
                Operation.OpType.LAYER_NORM,
                List.of(source, gamma, beta),
                new int[]{2, 3}
        );

        assertBf16Close(bf16Floats(expected), destination, BF16_NORM_SOFTMAX_TOLERANCE);
    }

    @Test
    void explicitShimExecuteBuffersSupportsBfloat16SoftmaxTolerance() {
        Tensor expectedSource = bf16Tensor(new float[]{-1f, 0f, 1f, 2f, 1f, -2f}, new int[]{2, 3}, "expectedBf16SoftmaxSource");
        Tensor expected = expectedSource.softmax(1);
        expected.compute();

        Tensor source = bf16Tensor(new float[]{-1f, 0f, 1f, 2f, 1f, -2f}, new int[]{2, 3}, "bf16SoftmaxSource");
        Tensor out = specialSoftmax(source, 1);

        Tensor destination = executeBf16LoweredNode(
                out,
                Operation.OpType.SOFTMAX,
                List.of(source),
                new int[]{2, 3}
        );

        assertBf16Close(bf16Floats(expected), destination, BF16_NORM_SOFTMAX_TOLERANCE);
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
    void explicitShimExecuteBuffersSupportsBfloat16SdpaParity() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        int[] shape = new int[]{1, 2, 2};
        Tensor expectedQ = bf16Tensor(new float[]{1f, 0f, 0f, 1f}, shape, "expectedBf16SdpaQ");
        Tensor expectedK = bf16Tensor(new float[]{1f, 0f, 0f, 1f}, shape, "expectedBf16SdpaK");
        Tensor expectedV = bf16Tensor(new float[]{10f, 1f, 1f, 10f}, shape, "expectedBf16SdpaV");
        Tensor expected = expectedQ.scaledDotProductAttention(
                expectedK,
                expectedV,
                AttentionOptions.defaults().withScale(0.5)
        );
        expected.compute();

        Tensor q = bf16Tensor(new float[]{1f, 0f, 0f, 1f}, shape, "bf16SdpaQ");
        Tensor k = bf16Tensor(new float[]{1f, 0f, 0f, 1f}, shape, "bf16SdpaK");
        Tensor v = bf16Tensor(new float[]{10f, 1f, 1f, 10f}, shape, "bf16SdpaV");
        Tensor out = specialSdpa(q, k, v, null, AttentionOptions.defaults().withScale(0.5));
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
            output = allocator.createOutputBinding(sdpaNode.id(), denseLayout(DataType.BFLOAT16, shape));

            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(
                    context,
                    executable,
                    List.of(query, key, value),
                    List.of(output)
            );
            Tensor destination = bf16Tensor(new float[expected.getFlatDataSize()], shape, "bf16SdpaDestination");
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            assertBf16Close(bf16Floats(expected), destination, BF16_NORM_SOFTMAX_TOLERANCE);
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

    @Test
    void executableSignatureIncludesSdpaScaleBits() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "sdpaSigQ", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "sdpaSigK", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "sdpaSigV", DataType.FLOAT32);

        AcceleratorSubgraphSignature halfScale = sdpaSignature(specialSdpa(q, k, v, null, AttentionOptions.defaults().withScale(0.5)));
        AcceleratorSubgraphSignature unitScale = sdpaSignature(specialSdpa(q, k, v, null, AttentionOptions.defaults().withScale(1.0)));

        assertFalse(halfScale.equals(unitScale));
    }

    @Test
    void executableSignatureDistinguishesMaskedSdpaFromUnmaskedSdpa() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "sdpaSigQ", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "sdpaSigK", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "sdpaSigV", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{1, 2, 2}, null, "sdpaSigMask", DataType.BOOL);

        AcceleratorSubgraphSignature unmasked = sdpaSignature(specialSdpa(q, k, v, null, AttentionOptions.defaults().withScale(0.5)));
        AcceleratorSubgraphSignature masked = sdpaSignature(specialSdpa(q, k, v, mask, AttentionOptions.defaults().withScale(0.5)));

        assertFalse(unmasked.equals(masked));
    }

    @Test
    void explicitShimExecuteBuffersSupportsMaskedSdpaRank3Parity() {
        assertNativeSdpaParity(
                new float[]{1f, 0f, 0f, 1f},
                new float[]{1f, 0f, 0f, 1f},
                new float[]{10f, 1f, 1f, 10f},
                new byte[]{1, 0, 1, 1},
                new int[]{1, 2, 2},
                AttentionOptions.defaults().withScale(1.0)
        );
    }

    @Test
    void explicitShimExecuteBuffersSupportsCausalSdpaRank3Parity() {
        assertNativeSdpaParity(
                new float[]{1f, 0f, 0f, 1f},
                new float[]{1f, 0f, 0f, 1f},
                new float[]{10f, 1f, 1f, 10f},
                null,
                new int[]{1, 2, 2},
                AttentionOptions.causalDefaults().withScale(1.0)
        );
    }

    @Test
    void explicitShimExecuteBuffersSupportsSdpaWeightsPublicationParity() {
        assertNativeSdpaWeightsParity(
                new float[]{1f, 0f, 0f, 1f},
                new float[]{1f, 0f, 0f, 1f},
                new float[]{10f, 1f, 1f, 10f},
                new byte[]{1, 0, 1, 1},
                new int[]{1, 2, 2},
                AttentionOptions.defaults().withScale(1.0)
        );
    }

    @Test
    void explicitShimExecuteBuffersSupportsExternalAndCausalSdpaRank3Parity() {
        assertNativeSdpaParity(
                new float[]{1f, 0f, 0f, 1f},
                new float[]{1f, 0f, 0f, 1f},
                new float[]{10f, 1f, 1f, 10f},
                new byte[]{1, 1, 1, 0},
                new int[]{1, 2, 2},
                AttentionOptions.causalDefaults().withScale(1.0)
        );
    }

    private static void assertNativeSdpaParity(float[] queryValues, float[] keyValues, float[] valueValues, int[] shape, AttentionOptions options) {
        assertNativeSdpaParity(queryValues, keyValues, valueValues, null, shape, options);
    }

    private static void assertNativeSdpaParity(float[] queryValues, float[] keyValues, float[] valueValues, byte[] maskValues, int[] shape, AttentionOptions options) {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor expectedQ = new Tensor(queryValues.clone(), shape, null, "expectedSdpaQ", DataType.FLOAT32);
        Tensor expectedK = new Tensor(keyValues.clone(), shape, null, "expectedSdpaK", DataType.FLOAT32);
        Tensor expectedV = new Tensor(valueValues.clone(), shape, null, "expectedSdpaV", DataType.FLOAT32);
        Tensor expectedMask = maskValues == null ? null : new Tensor(maskValues.clone(), scoreShape(shape), null, "expectedSdpaMask", DataType.BOOL);
        Tensor expected = expectedMask == null
                ? expectedQ.scaledDotProductAttention(expectedK, expectedV, options)
                : expectedQ.scaledDotProductAttention(expectedK, expectedV, expectedMask, options);
        expected.compute();

        Tensor q = new Tensor(queryValues.clone(), shape, null, "sdpaQ", DataType.FLOAT32);
        Tensor k = new Tensor(keyValues.clone(), shape, null, "sdpaK", DataType.FLOAT32);
        Tensor v = new Tensor(valueValues.clone(), shape, null, "sdpaV", DataType.FLOAT32);
        Tensor mask = maskValues == null ? null : new Tensor(maskValues.clone(), scoreShape(shape), null, "sdpaMask", DataType.BOOL);
        Tensor out = specialSdpa(q, k, v, mask, options);
        PartitionPlanningContext planningContext = planningContext(out);
        CompiledNode sdpaNode = planningContext.compiledNode(nodeId(planningContext, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION));
        Tensor runtimeMask = sdpaNode.inputIds().size() > 3
                ? planningContext.compiledNode(sdpaNode.inputIds().get(3)).semanticTensor()
                : null;
        if (runtimeMask != null) {
            runtimeMask.compute();
        }
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
        MetalBufferBinding maskBinding = null;
        MetalBufferBinding output = null;
        try {
            query = allocator.createInputBinding(sdpaNode.inputIds().get(0), q);
            key = allocator.createInputBinding(sdpaNode.inputIds().get(1), k);
            value = allocator.createInputBinding(sdpaNode.inputIds().get(2), v);
            if (runtimeMask != null) {
                maskBinding = allocator.createPredicateInputBinding(sdpaNode.inputIds().get(3), runtimeMask);
            }
            output = allocator.createOutputBinding(sdpaNode.id(), denseF32Layout(shape));

            List<MetalBufferBinding> inputs = maskBinding == null
                    ? List.of(query, key, value)
                    : List.of(query, key, value, maskBinding);
            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(
                    context,
                    executable,
                    inputs,
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
            if (maskBinding != null) {
                allocator.destroy(maskBinding.handle());
            }
            if (output != null) {
                allocator.destroy(output.handle());
            }
        }
    }

    private static void assertNativeSdpaWeightsParity(
            float[] queryValues,
            float[] keyValues,
            float[] valueValues,
            byte[] maskValues,
            int[] shape,
            AttentionOptions options
    ) {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        float e = (float) Math.exp(1.0);
        float[] expectedWeightValues = new float[]{
                1.0f, 0.0f,
                1.0f / (1.0f + e), e / (1.0f + e)
        };

        Tensor q = new Tensor(queryValues.clone(), shape, null, "sdpaWeightsQ", DataType.FLOAT32);
        Tensor k = new Tensor(keyValues.clone(), shape, null, "sdpaWeightsK", DataType.FLOAT32);
        Tensor v = new Tensor(valueValues.clone(), shape, null, "sdpaWeightsV", DataType.FLOAT32);
        Tensor mask = maskValues == null ? null : new Tensor(maskValues.clone(), scoreShape(shape), null, "sdpaWeightsMask", DataType.BOOL);
        Tensor attention = specialSdpa(q, k, v, mask, options);
        Tensor weights = TensorPrimitiveBuilder.unaryNoGrad(
                attention,
                scoreShape(shape),
                new operations.linalg.scaledDotProductAttentionWeights(),
                "attentionWeights",
                DataType.FLOAT32
        );
        PartitionPlanningContext planningContext = planningContext(weights);
        CompiledNode weightsNode = planningContext.compiledNode(nodeId(planningContext, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS));
        CompiledNode attentionNode = planningContext.compiledNode(weightsNode.inputIds().getFirst());
        Tensor runtimeMask = attentionNode.inputIds().size() > 3
                ? planningContext.compiledNode(attentionNode.inputIds().get(3)).semanticTensor()
                : null;
        if (runtimeMask != null) {
            runtimeMask.compute();
        }
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                weightsNode.id(),
                List.of(weightsNode.id()),
                List.of(new AcceleratorSubgraphOp(weightsNode.id(), Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS)),
                weightsNode.inputIds(),
                List.of(weightsNode.id())
        );
        AcceleratorSubgraphLoweringResult lowering = new AcceleratorSubgraphLowerer().tryLower(subgraph, planningContext);
        assertNotNull(lowering);

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalMpsBridgeExecutable executable = bridge.compile(
                context,
                new MetalPartitionPlan(weightsNode.id(), subgraph, lowering)
        );
        assumeTrue(executable.available(), executable.reason());
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        MetalBufferBinding query = null;
        MetalBufferBinding key = null;
        MetalBufferBinding maskBinding = null;
        MetalBufferBinding output = null;
        try {
            query = allocator.createInputBinding(attentionNode.inputIds().get(0), q);
            key = allocator.createInputBinding(attentionNode.inputIds().get(1), k);
            if (runtimeMask != null) {
                maskBinding = allocator.createPredicateInputBinding(attentionNode.inputIds().get(3), runtimeMask);
            }
            output = allocator.createOutputBinding(weightsNode.id(), denseF32Layout(scoreShape(shape)));

            List<MetalBufferBinding> inputs = maskBinding == null
                    ? List.of(query, key)
                    : List.of(query, key, maskBinding);
            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(
                    context,
                    executable,
                    inputs,
                    List.of(output)
            );
            Tensor destination = new Tensor(new float[expectedWeightValues.length], scoreShape(shape), null, "sdpaWeightsDestination", DataType.FLOAT32);
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            assertArrayEquals(expectedWeightValues, destination.getFloat32Data(), 1.0e-4f);
        } finally {
            if (query != null) {
                allocator.destroy(query.handle());
            }
            if (key != null) {
                allocator.destroy(key.handle());
            }
            if (maskBinding != null) {
                allocator.destroy(maskBinding.handle());
            }
            if (output != null) {
                allocator.destroy(output.handle());
            }
        }
    }

    private static int[] scoreShape(int[] qkvShape) {
        int[] out = qkvShape.clone();
        out[out.length - 1] = qkvShape[qkvShape.length - 2];
        return out;
    }

    private static Tensor specialSoftmax(Tensor input, int dimension) {
        return TensorPrimitiveBuilder.unary(
                input,
                input.getShapeUnsafe().clone(),
                new operations.reduction.softmax(dimension),
                "legacySoftmax",
                input.getDataType()
        );
    }

    private static Tensor specialSdpa(Tensor query, Tensor key, Tensor value, Tensor mask, AttentionOptions options) {
        int[] queryShape = query.getShapeUnsafe();
        int[] valueShape = value.getShapeUnsafe();
        int[] outShape = queryShape.clone();
        outShape[outShape.length - 1] = valueShape[valueShape.length - 1];
        int[] scoresShape = scoreShape(queryShape);

        Tensor effectiveMask = mask;
        if (options.causal()) {
            Tensor causalMask = causalMask(scoresShape);
            effectiveMask = effectiveMask == null ? causalMask : effectiveMask.logicalAnd(causalMask);
        }
        if (effectiveMask != null) {
            effectiveMask = effectiveMask.expand(scoresShape);
        }

        java.util.ArrayList<Tensor> inputs = new java.util.ArrayList<>();
        inputs.add(query);
        inputs.add(key);
        inputs.add(value);
        if (effectiveMask != null) {
            inputs.add(effectiveMask);
        }
        double scale = options.resolveScale(queryShape[queryShape.length - 1]);
        return TensorPrimitiveBuilder.nary(
                outShape,
                inputs,
                new operations.linalg.scaledDotProductAttention(scale, effectiveMask != null),
                "legacyScaledDotProductAttention",
                query.getDataType()
        );
    }

    private static Tensor causalMask(int[] scoresShape) {
        int queryLength = scoresShape[scoresShape.length - 2];
        int keyLength = scoresShape[scoresShape.length - 1];
        int size = 1;
        for (int dimension : scoresShape) {
            size *= dimension;
        }
        byte[] data = new byte[size];
        for (int linear = 0; linear < size; linear++) {
            int keyIndex = linear % keyLength;
            int queryIndex = (linear / keyLength) % queryLength;
            data[linear] = (byte) (keyIndex <= queryIndex ? 1 : 0);
        }
        return new Tensor(data, scoresShape.clone(), null, "legacyCausalMask", DataType.BOOL);
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
    void explicitShimMaterializesBroadcastZeroStrideLayoutToDenseBuffer() {
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
            Tensor base = new Tensor(new float[]{2f, 4f, 6f}, new int[]{1, 3}, null, "layoutBroadcastBase", DataType.FLOAT32);
            input = allocator.createInputBinding(1, base);
            AcceleratorBufferLayout broadcastLayout = AcceleratorBufferLayout.of(
                    DataType.FLOAT32,
                    new int[]{2, 3},
                    new int[]{0, 1},
                    0,
                    6
            );
            MetalBufferBinding sourceView = MetalBufferBinding.viewOf(1, broadcastLayout, input, MetalBufferAccess.READ);
            AcceleratorBufferLayout denseTarget = AcceleratorBufferLayout.of(
                    DataType.FLOAT32,
                    new int[]{2, 3},
                    new int[]{3, 1},
                    0,
                    6
            );
            destination = allocator.createOutputBinding(2, denseTarget);

            bridge.materializeLayout(context, sourceView, destination);

            Tensor actual = new Tensor(new float[6], new int[]{2, 3}, null, "layoutBroadcastDense", DataType.FLOAT32);
            allocator.readToCpu(destination, actual, CpuMaterializationReason.PUBLIC_DATA_ACCESS);
            assertArrayEquals(new float[]{2f, 4f, 6f, 2f, 4f, 6f}, actual.getFloat32Data(), 0.0f);
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
    void explicitShimMaterializesNonZeroOffsetLayoutToDenseBuffer() {
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
            Tensor base = new Tensor(new float[]{
                    1f, 2f, 3f, 4f,
                    5f, 6f, 7f, 8f
            }, new int[]{2, 4}, null, "layoutOffsetBase", DataType.FLOAT32);
            input = allocator.createInputBinding(1, base);
            AcceleratorBufferLayout offsetLayout = AcceleratorBufferLayout.of(
                    DataType.FLOAT32,
                    new int[]{2, 2},
                    new int[]{4, 1},
                    1,
                    4
            );
            MetalBufferBinding sourceView = new MetalBufferBinding(
                    1,
                    offsetLayout,
                    input.handle(),
                    MetalBufferAccess.READ
            );
            AcceleratorBufferLayout denseTarget = AcceleratorBufferLayout.of(
                    DataType.FLOAT32,
                    new int[]{2, 2},
                    new int[]{2, 1},
                    0,
                    4
            );
            destination = allocator.createOutputBinding(2, denseTarget);

            bridge.materializeLayout(context, sourceView, destination);

            Tensor actual = new Tensor(new float[4], new int[]{2, 2}, null, "layoutOffsetDense", DataType.FLOAT32);
            allocator.readToCpu(destination, actual, CpuMaterializationReason.PUBLIC_DATA_ACCESS);
            assertArrayEquals(new float[]{2f, 3f, 6f, 7f}, actual.getFloat32Data(), 0.0f);
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
    void explicitShimMaterializesBFloat16PermutedLayoutToDenseBuffer() {
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
            Tensor base = bf16Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, "layoutBf16Base");
            input = allocator.createInputBinding(1, base);
            AcceleratorBufferLayout permutedLayout = AcceleratorBufferLayout.of(
                    DataType.BFLOAT16,
                    new int[]{3, 2},
                    new int[]{1, 3},
                    0,
                    6
            );
            MetalBufferBinding sourceView = MetalBufferBinding.viewOf(1, permutedLayout, input, MetalBufferAccess.READ);
            destination = allocator.createOutputBinding(2, denseLayout(DataType.BFLOAT16, new int[]{3, 2}));

            bridge.materializeLayout(context, sourceView, destination);

            Tensor actual = bf16Tensor(new float[6], new int[]{3, 2}, "layoutBf16Dense");
            allocator.readToCpu(destination, actual, CpuMaterializationReason.PUBLIC_DATA_ACCESS);
            assertBf16RawBitsEqual(new float[]{1f, 4f, 2f, 5f, 3f, 6f}, actual);
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
    void explicitShimMaterializesBoolPermutedLayoutToDenseBuffer() {
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
            Tensor base = new Tensor(new byte[]{1, 0, 1, 0, 1, 0}, new int[]{2, 3}, null, "layoutBoolBase", DataType.BOOL);
            input = allocator.createPredicateInputBinding(1, base);
            AcceleratorBufferLayout permutedLayout = AcceleratorBufferLayout.of(
                    DataType.BOOL,
                    new int[]{3, 2},
                    new int[]{1, 3},
                    0,
                    6
            );
            MetalBufferBinding sourceView = MetalBufferBinding.viewOf(1, permutedLayout, input, MetalBufferAccess.READ);
            destination = allocator.createOutputBinding(2, denseLayout(DataType.BOOL, new int[]{3, 2}));

            bridge.materializeLayout(context, sourceView, destination);

            Tensor actual = new Tensor(new byte[6], new int[]{3, 2}, null, "layoutBoolDense", DataType.BOOL);
            allocator.readToCpu(destination, actual, CpuMaterializationReason.PUBLIC_DATA_ACCESS);
            assertArrayEquals(new byte[]{1, 0, 0, 1, 1, 0}, actual.getBoolData());
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
    void bufferBindingValidationAcceptsBoolOutputWhenExecutableExpectsBool() {
        MetalMpsBridgeExecutable executable = executableDescriptor(1, 2, DataType.BOOL);
        MetalBufferBinding input = binding(1, MetalBufferAccess.READ);
        MetalBufferBinding output = binding(
                2,
                AcceleratorBufferLayout.of(DataType.BOOL, new int[]{2}, new int[]{1}, 0, 2),
                MetalBufferAccess.WRITE
        );

        assertDoesNotThrow(() -> MetalMpsFfmBridge.validateBufferBindings(executable, List.of(input), List.of(output)));
    }

    @Test
    void bufferBindingValidationAcceptsInt32ExternalInputWhenExecutableExpectsInt32() {
        MetalMpsBridgeExecutable executable = executableDescriptor(1, 2, DataType.INT32, DataType.FLOAT32);
        MetalBufferBinding input = binding(
                1,
                AcceleratorBufferLayout.of(DataType.INT32, new int[]{2}, new int[]{1}, 0, 2),
                MetalBufferAccess.READ
        );
        MetalBufferBinding output = binding(2, MetalBufferAccess.WRITE);

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

        assertEquals("Metal buffer outputs support FLOAT32/BFLOAT16/BOOL/INT32 only; got FLOAT64.", failure.getMessage());
    }

    private static AcceleratorBufferLayout denseF32Layout(int[] shape) {
        return denseLayout(DataType.FLOAT32, shape);
    }

    private static AcceleratorBufferLayout denseLayout(DataType dataType, int[] shape) {
        long elements = Arrays.stream(shape).asLongStream().reduce(1L, Math::multiplyExact);
        return AcceleratorBufferLayout.of(dataType, shape, TensorMetadata.computeStrides(shape), 0, elements);
    }

    private static Tensor bf16Tensor(float[] values, int[] shape, String label) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = CpuDTypeOps.toBFloat16Bits(values[i]);
        }
        return new Tensor(bits, shape, null, label, DataType.BFLOAT16);
    }

    private static void executeF32Unary(
            MetalMpsFfmBridge bridge,
            MetalMpsBridgeContext context,
            MetalBufferAllocator allocator,
            MetalPartitionPlan plan,
            float[] inputValues,
            int[] inputShape,
            float[] expectedValues,
            int[] outputShape,
            String label
    ) {
        MetalMpsBridgeExecutable executable = bridge.compile(context, plan);
        assumeTrue(executable.available(), executable.reason());
        MetalBufferBinding input = null;
        MetalBufferBinding output = null;
        try {
            input = allocator.createInputBinding(1, new Tensor(inputValues, inputShape, null, label + "Input", DataType.FLOAT32));
            output = allocator.createOutputBinding(9, denseF32Layout(outputShape));

            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(context, executable, List.of(input), List.of(output));
            Tensor destination = new Tensor(new float[expectedValues.length], outputShape, null, label + "Destination", DataType.FLOAT32);
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            assertArrayEquals(expectedValues, destination.getFloat32Data(), 1.0e-5f);
        } finally {
            if (input != null) {
                allocator.destroy(input.handle());
            }
            if (output != null) {
                allocator.destroy(output.handle());
            }
        }
    }

    private static void executeInt32Unary(
            MetalMpsFfmBridge bridge,
            MetalMpsBridgeContext context,
            MetalBufferAllocator allocator,
            MetalPartitionPlan plan,
            float[] inputValues,
            int[] inputShape,
            int[] expectedValues,
            int[] outputShape,
            String label
    ) {
        MetalMpsBridgeExecutable executable = bridge.compile(context, plan);
        assumeTrue(executable.available(), executable.reason());
        MetalBufferBinding input = null;
        MetalBufferBinding output = null;
        try {
            input = allocator.createInputBinding(1, new Tensor(inputValues, inputShape, null, label + "Input", DataType.FLOAT32));
            output = allocator.createOutputBinding(9, denseLayout(DataType.INT32, outputShape));

            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(context, executable, List.of(input), List.of(output));
            Tensor destination = new Tensor(new int[expectedValues.length], outputShape, null, label + "Destination", DataType.INT32);
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            assertArrayEquals(expectedValues, destination.getInt32Data());
        } finally {
            if (input != null) {
                allocator.destroy(input.handle());
            }
            if (output != null) {
                allocator.destroy(output.handle());
            }
        }
    }

    private static void assertBf16Close(float[] expected, Tensor actual, float tolerance) {
        short[] bits = actual.getBFloat16Data();
        assertEquals(expected.length, bits.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], CpuDTypeOps.fromBFloat16Bits(bits[i]), tolerance, "BF16 mismatch at " + i);
        }
    }

    private static void assertBf16RawBitsEqual(float[] expected, Tensor actual) {
        short[] bits = actual.getBFloat16Data();
        assertEquals(expected.length, bits.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(CpuDTypeOps.toBFloat16Bits(expected[i]), bits[i], "BF16 raw mismatch at " + i);
        }
    }

    private static float[] bf16Floats(Tensor tensor) {
        short[] bits = tensor.getBFloat16Data();
        float[] values = new float[bits.length];
        for (int i = 0; i < bits.length; i++) {
            values[i] = CpuDTypeOps.fromBFloat16Bits(bits[i]);
        }
        return values;
    }

    private static Tensor executeBf16LoweredNode(
            Tensor out,
            Operation.OpType opType,
            List<Tensor> inputs,
            int[] outputShape
    ) {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        PartitionPlanningContext planningContext = planningContext(out);
        CompiledNode node = planningContext.compiledNode(nodeId(planningContext, opType));
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                node.id(),
                List.of(node.id()),
                List.of(new AcceleratorSubgraphOp(node.id(), opType)),
                node.inputIds(),
                List.of(node.id())
        );
        AcceleratorSubgraphLoweringResult lowering = new AcceleratorSubgraphLowerer().tryLower(subgraph, planningContext);
        assertNotNull(lowering);

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        assumeTrue(bridge.capabilities().dtypeAbiV3Supported());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalMpsBridgeExecutable executable = bridge.compile(
                context,
                new MetalPartitionPlan(node.id(), subgraph, lowering)
        );
        assumeTrue(executable.available(), executable.reason());
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        java.util.ArrayList<MetalBufferBinding> inputBindings = new java.util.ArrayList<>();
        MetalBufferBinding output = null;
        try {
            for (int i = 0; i < inputs.size(); i++) {
                inputBindings.add(allocator.createInputBinding(node.inputIds().get(i), inputs.get(i)));
            }
            output = allocator.createOutputBinding(node.id(), denseLayout(DataType.BFLOAT16, outputShape));

            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(context, executable, inputBindings, List.of(output));
            Tensor destination = bf16Tensor(new float[(int) output.layout().logicalElementCount()], outputShape, "bf16Destination");
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            return destination;
        } finally {
            for (MetalBufferBinding input : inputBindings) {
                allocator.destroy(input.handle());
            }
            if (output != null) {
                allocator.destroy(output.handle());
            }
        }
    }

    private static Tensor executeF32LoweredNode(
            Tensor out,
            Operation.OpType opType,
            List<Tensor> inputs,
            int[] outputShape
    ) {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        PartitionPlanningContext planningContext = planningContext(out);
        CompiledNode node = planningContext.compiledNode(nodeId(planningContext, opType));
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                node.id(),
                List.of(node.id()),
                List.of(new AcceleratorSubgraphOp(node.id(), opType)),
                node.inputIds(),
                List.of(node.id())
        );
        AcceleratorSubgraphLoweringResult lowering = new AcceleratorSubgraphLowerer().tryLower(subgraph, planningContext);
        assertNotNull(lowering);

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalMpsBridgeExecutable executable = bridge.compile(
                context,
                new MetalPartitionPlan(node.id(), subgraph, lowering)
        );
        assumeTrue(executable.available(), executable.reason());
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        java.util.ArrayList<MetalBufferBinding> inputBindings = new java.util.ArrayList<>();
        MetalBufferBinding output = null;
        try {
            for (int i = 0; i < inputs.size(); i++) {
                inputBindings.add(allocator.createInputBinding(node.inputIds().get(i), inputs.get(i)));
            }
            output = allocator.createOutputBinding(node.id(), denseF32Layout(outputShape));

            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(context, executable, inputBindings, List.of(output));
            Tensor destination = new Tensor(new float[(int) output.layout().logicalElementCount()], outputShape, null, "f32Destination", DataType.FLOAT32);
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            return destination;
        } finally {
            for (MetalBufferBinding input : inputBindings) {
                allocator.destroy(input.handle());
            }
            if (output != null) {
                allocator.destroy(output.handle());
            }
        }
    }

    private static Tensor executeBoolLoweredPlan(
            MetalPartitionPlan plan,
            List<Tensor> inputs,
            int[] outputShape
    ) {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        assumeTrue(bridge.capabilities().dtypeAbiV3Supported());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalMpsBridgeExecutable executable = bridge.compile(context, plan);
        assumeTrue(executable.available(), executable.reason());
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        java.util.ArrayList<MetalBufferBinding> inputBindings = new java.util.ArrayList<>();
        MetalBufferBinding output = null;
        try {
            for (int i = 0; i < inputs.size(); i++) {
                Tensor input = inputs.get(i);
                int nodeId = plan.subgraph().externalInputNodeIds().get(i);
                inputBindings.add(input.getDataType() == DataType.BOOL
                        ? allocator.createPredicateInputBinding(nodeId, input)
                        : allocator.createInputBinding(nodeId, input));
            }
            output = allocator.createOutputBinding(plan.subgraph().outputNodeIds().getFirst(), denseLayout(DataType.BOOL, outputShape));

            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(context, executable, inputBindings, List.of(output));
            Tensor destination = new Tensor(new byte[(int) output.layout().logicalElementCount()], outputShape, null, "boolDestination", DataType.BOOL);
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            return destination;
        } finally {
            for (MetalBufferBinding input : inputBindings) {
                allocator.destroy(input.handle());
            }
            if (output != null) {
                allocator.destroy(output.handle());
            }
        }
    }

    private static Tensor executeBoolLoweredNode(
            Tensor out,
            Operation.OpType opType,
            List<Tensor> inputs,
            int[] outputShape
    ) {
        PartitionPlanningContext planningContext = planningContext(out);
        CompiledNode node = planningContext.compiledNode(nodeId(planningContext, opType));
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                node.id(),
                List.of(node.id()),
                List.of(new AcceleratorSubgraphOp(node.id(), opType)),
                node.inputIds(),
                List.of(node.id())
        );
        AcceleratorSubgraphLoweringResult lowering = new AcceleratorSubgraphLowerer().tryLower(subgraph, planningContext);
        assertNotNull(lowering);
        return executeBoolLoweredPlan(
                new MetalPartitionPlan(node.id(), subgraph, lowering),
                inputs,
                outputShape
        );
    }

    private static Tensor executeIndexLoweredPlan(
            MetalPartitionPlan plan,
            Tensor valueInput,
            Tensor indexInput,
            int[] outputShape
    ) {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        assumeTrue(bridge.capabilities().dtypeAbiV3Supported());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalMpsBridgeExecutable executable = bridge.compile(context, plan);
        assumeTrue(executable.available(), executable.reason());
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        MetalBufferBinding valueBinding = null;
        MetalBufferBinding indexBinding = null;
        MetalBufferBinding output = null;
        try {
            valueBinding = allocator.createInputBinding(plan.subgraph().externalInputNodeIds().get(0), valueInput);
            indexBinding = allocator.createInputBinding(plan.subgraph().externalInputNodeIds().get(1), indexInput);
            output = allocator.createOutputBinding(plan.subgraph().outputNodeIds().getFirst(), denseLayout(DataType.FLOAT32, outputShape));

            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(context, executable, List.of(valueBinding, indexBinding), List.of(output));
            Tensor destination = new Tensor(new float[(int) output.layout().logicalElementCount()], outputShape, null, "indexDestination", DataType.FLOAT32);
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            return destination;
        } finally {
            if (valueBinding != null) {
                allocator.destroy(valueBinding.handle());
            }
            if (indexBinding != null) {
                allocator.destroy(indexBinding.handle());
            }
            if (output != null) {
                allocator.destroy(output.handle());
            }
        }
    }

    private static Tensor executeConv2dLoweredPlan(
            MetalPartitionPlan plan,
            List<Tensor> inputs,
            int[] outputShape
    ) {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        MetalMpsFfmBridge bridge = new MetalMpsFfmBridge();
        assumeTrue(bridge.isAvailable());
        assumeTrue(bridge.supportsBufferBindings());
        MetalMpsBridgeContext context = bridge.createContext();
        MetalMpsBridgeExecutable executable = bridge.compile(context, plan);
        assumeTrue(executable.available(), executable.reason());
        MetalBufferAllocator allocator = bridge.createBufferAllocator(context);
        assertTrue(allocator.available(), allocator.unavailableReason());

        java.util.ArrayList<MetalBufferBinding> inputBindings = new java.util.ArrayList<>();
        MetalBufferBinding output = null;
        try {
            for (int i = 0; i < inputs.size(); i++) {
                inputBindings.add(allocator.createInputBinding(plan.subgraph().externalInputNodeIds().get(i), inputs.get(i)));
            }
            output = allocator.createOutputBinding(plan.subgraph().outputNodeIds().getFirst(), denseLayout(DataType.FLOAT32, outputShape));

            MetalMpsBridgeExecutionStats stats = bridge.executeBuffers(context, executable, inputBindings, List.of(output));
            Tensor destination = new Tensor(new float[(int) output.layout().logicalElementCount()], outputShape, null, "conv2dDestination", DataType.FLOAT32);
            allocator.readToCpu(output, destination, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

            assertEquals(MetalMpsBridgeExecutionPath.BUFFER_BINDING, stats.executionPath());
            return destination;
        } finally {
            for (MetalBufferBinding input : inputBindings) {
                allocator.destroy(input.handle());
            }
            if (output != null) {
                allocator.destroy(output.handle());
            }
        }
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
        return executableDescriptor(inputNodeId, outputNodeId, DataType.FLOAT32, outputDType);
    }

    private static MetalMpsBridgeExecutable executableDescriptor(
            int inputNodeId,
            int outputNodeId,
            DataType inputDType,
            DataType outputDType
    ) {
        return new MetalMpsBridgeExecutable(
                true,
                java.lang.foreign.MemorySegment.ofAddress(100),
                "",
                false,
                List.of(inputNodeId),
                List.of(inputDType),
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
        return unaryPlan(inputNodeId, outputNodeId, nodeType, opType, DataType.FLOAT32);
    }

    private static MetalPartitionPlan unaryPlan(
            int inputNodeId,
            int outputNodeId,
            AcceleratorDagNodeType nodeType,
            Operation.OpType opType,
            DataType dataType
    ) {
        AcceleratorDagInput input = new AcceleratorDagInput(inputNodeId, List.of(2), dataType);
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
                1,
                dataType
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

    private static MetalPartitionPlan binaryPlan(
            int input0NodeId,
            int input1NodeId,
            int outputNodeId,
            AcceleratorDagNodeType nodeType,
            Operation.OpType opType,
            int[] input0Shape,
            int[] input1Shape,
            int[] outputShape,
            DataType dataType
    ) {
        return binaryPlan(input0NodeId, input1NodeId, outputNodeId, nodeType, opType, input0Shape, input1Shape, outputShape, dataType, dataType);
    }

    private static MetalPartitionPlan binaryPlan(
            int input0NodeId,
            int input1NodeId,
            int outputNodeId,
            AcceleratorDagNodeType nodeType,
            Operation.OpType opType,
            int[] input0Shape,
            int[] input1Shape,
            int[] outputShape,
            DataType inputDataType,
            DataType outputDataType
    ) {
        AcceleratorDagInput input0 = new AcceleratorDagInput(input0NodeId, Arrays.stream(input0Shape).boxed().toList(), inputDataType);
        AcceleratorDagInput input1 = new AcceleratorDagInput(input1NodeId, Arrays.stream(input1Shape).boxed().toList(), inputDataType);
        AcceleratorDagNode node = new AcceleratorDagNode(
                outputNodeId,
                nodeType,
                AcceleratorDagValueRef.externalInput(0),
                AcceleratorDagValueRef.externalInput(1),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                0,
                outputShape.length,
                outputShape[0],
                outputShape.length >= 2 ? outputShape[1] : 1,
                outputShape.length >= 3 ? outputShape[2] : 1,
                outputShape.length >= 4 ? outputShape[3] : 1,
                outputDataType
        );
        AcceleratorDagSpec dag = new AcceleratorDagSpec(List.of(input0, input1), List.of(node), List.of(0), List.of(outputNodeId));
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                outputNodeId,
                List.of(outputNodeId),
                List.of(new AcceleratorSubgraphOp(outputNodeId, opType)),
                List.of(input0NodeId, input1NodeId),
                List.of(outputNodeId)
        );
        long estimatedWork = Arrays.stream(input0Shape).asLongStream().reduce(1L, Math::multiplyExact);
        return new MetalPartitionPlan(
                outputNodeId,
                subgraph,
                new AcceleratorSubgraphLoweringResult(outputNodeId, null, dag, estimatedWork)
        );
    }

    private static MetalPartitionPlan indexPlan(
            int valueNodeId,
            int indexNodeId,
            int outputNodeId,
            AcceleratorDagNodeType nodeType,
            Operation.OpType opType,
            int axis,
            int[] valueShape,
            int[] indexShape,
            int[] outputShape
    ) {
        AcceleratorDagInput valueInput = new AcceleratorDagInput(valueNodeId, Arrays.stream(valueShape).boxed().toList(), DataType.FLOAT32);
        AcceleratorDagInput indexInput = new AcceleratorDagInput(indexNodeId, Arrays.stream(indexShape).boxed().toList(), DataType.INT32);
        AcceleratorDagNode node = new AcceleratorDagNode(
                outputNodeId,
                nodeType,
                AcceleratorDagValueRef.externalInput(0),
                AcceleratorDagValueRef.externalInput(1),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                axis,
                outputShape.length,
                outputShape[0],
                outputShape.length >= 2 ? outputShape[1] : 1,
                outputShape.length >= 3 ? outputShape[2] : 1,
                outputShape.length >= 4 ? outputShape[3] : 1,
                DataType.FLOAT32
        );
        AcceleratorDagSpec dag = new AcceleratorDagSpec(List.of(valueInput, indexInput), List.of(node), List.of(0), List.of(outputNodeId));
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                outputNodeId,
                List.of(outputNodeId),
                List.of(new AcceleratorSubgraphOp(outputNodeId, opType)),
                List.of(valueNodeId, indexNodeId),
                List.of(outputNodeId)
        );
        long estimatedWork = Arrays.stream(outputShape).asLongStream().reduce(1L, Math::multiplyExact);
        return new MetalPartitionPlan(
                outputNodeId,
                subgraph,
                new AcceleratorSubgraphLoweringResult(outputNodeId, null, dag, estimatedWork)
        );
    }

    private static MetalPartitionPlan conv2dPlan(
            int inputNodeId,
            int weightNodeId,
            int biasNodeId,
            int outputNodeId,
            int[] inputShape,
            int[] weightShape,
            int[] outputShape,
            int strideH,
            int strideW,
            int padH,
            int padW
    ) {
        java.util.ArrayList<AcceleratorDagInput> inputs = new java.util.ArrayList<>();
        inputs.add(new AcceleratorDagInput(inputNodeId, Arrays.stream(inputShape).boxed().toList(), DataType.FLOAT32));
        inputs.add(new AcceleratorDagInput(weightNodeId, Arrays.stream(weightShape).boxed().toList(), DataType.FLOAT32));
        if (biasNodeId >= 0) {
            inputs.add(new AcceleratorDagInput(biasNodeId, List.of(weightShape[0]), DataType.FLOAT32));
        }
        AcceleratorDagNode node = new AcceleratorDagNode(
                outputNodeId,
                AcceleratorDagNodeType.CONV2D,
                AcceleratorDagValueRef.externalInput(0),
                AcceleratorDagValueRef.externalInput(1),
                biasNodeId >= 0 ? AcceleratorDagValueRef.externalInput(2) : AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                encodeConv2dMode(strideH, strideW, padH, padW),
                outputShape.length,
                outputShape[0],
                outputShape.length >= 2 ? outputShape[1] : 1,
                outputShape.length >= 3 ? outputShape[2] : 1,
                outputShape.length >= 4 ? outputShape[3] : 1,
                DataType.FLOAT32
        );
        AcceleratorDagSpec dag = new AcceleratorDagSpec(inputs, List.of(node), List.of(0), List.of(outputNodeId));
        List<Integer> externalIds = biasNodeId >= 0
                ? List.of(inputNodeId, weightNodeId, biasNodeId)
                : List.of(inputNodeId, weightNodeId);
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                outputNodeId,
                List.of(outputNodeId),
                List.of(new AcceleratorSubgraphOp(outputNodeId, Operation.OpType.CONV2D)),
                externalIds,
                List.of(outputNodeId)
        );
        long estimatedWork = Arrays.stream(outputShape).asLongStream().reduce(1L, Math::multiplyExact)
                * weightShape[1] * weightShape[2] * weightShape[3];
        return new MetalPartitionPlan(
                outputNodeId,
                subgraph,
                new AcceleratorSubgraphLoweringResult(outputNodeId, null, dag, estimatedWork)
        );
    }

    private static MetalPartitionPlan pool2dPlan(
            int inputNodeId,
            int outputNodeId,
            AcceleratorDagNodeType nodeType,
            Operation.OpType opType,
            int[] inputShape,
            int[] outputShape,
            int kernelH,
            int kernelW,
            int strideH,
            int strideW,
            int padH,
            int padW,
            boolean countIncludePad
    ) {
        AcceleratorDagInput input = new AcceleratorDagInput(inputNodeId, Arrays.stream(inputShape).boxed().toList(), DataType.FLOAT32);
        AcceleratorDagNode node = new AcceleratorDagNode(
                outputNodeId,
                nodeType,
                AcceleratorDagValueRef.externalInput(0),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                encodePool2dMode(kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad),
                outputShape.length,
                outputShape[0],
                outputShape.length >= 2 ? outputShape[1] : 1,
                outputShape.length >= 3 ? outputShape[2] : 1,
                outputShape.length >= 4 ? outputShape[3] : 1,
                DataType.FLOAT32
        );
        AcceleratorDagSpec dag = new AcceleratorDagSpec(List.of(input), List.of(node), List.of(0), List.of(outputNodeId));
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                outputNodeId,
                List.of(outputNodeId),
                List.of(new AcceleratorSubgraphOp(outputNodeId, opType)),
                List.of(inputNodeId),
                List.of(outputNodeId)
        );
        long estimatedWork = Arrays.stream(outputShape).asLongStream().reduce(1L, Math::multiplyExact) * kernelH * kernelW;
        return new MetalPartitionPlan(
                outputNodeId,
                subgraph,
                new AcceleratorSubgraphLoweringResult(outputNodeId, null, dag, estimatedWork)
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
        return reductionPlan(inputNodeId, outputNodeId, nodeType, opType, axis, keepDims, inputShape, outputShape, DataType.FLOAT32);
    }

    private static MetalPartitionPlan reductionPlan(
            int inputNodeId,
            int outputNodeId,
            AcceleratorDagNodeType nodeType,
            Operation.OpType opType,
            int axis,
            boolean keepDims,
            int[] inputShape,
            int[] outputShape,
            DataType dataType
    ) {
        AcceleratorDagInput input = new AcceleratorDagInput(inputNodeId, Arrays.stream(inputShape).boxed().toList(), dataType);
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
                outputShape.length >= 4 ? outputShape[3] : 1,
                dataType
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

    private static MetalPartitionPlan unaryPlan(
            int inputNodeId,
            int outputNodeId,
            AcceleratorDagNodeType nodeType,
            Operation.OpType opType,
            int scalarValue,
            int[] inputShape,
            int[] outputShape,
            DataType inputDataType,
            DataType outputDataType
    ) {
        AcceleratorDagInput input = new AcceleratorDagInput(inputNodeId, Arrays.stream(inputShape).boxed().toList(), inputDataType);
        AcceleratorDagNode node = new AcceleratorDagNode(
                outputNodeId,
                nodeType,
                AcceleratorDagValueRef.externalInput(0),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                AcceleratorDagValueRef.none(),
                scalarValue,
                outputShape.length,
                outputShape[0],
                outputShape.length >= 2 ? outputShape[1] : 1,
                outputShape.length >= 3 ? outputShape[2] : 1,
                outputShape.length >= 4 ? outputShape[3] : 1,
                outputDataType
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

    private static int encodeCumSumMode(int axis, boolean exclusive, boolean reverse) {
        return (axis & 0xFFFF)
                | (exclusive ? 1 << 16 : 0)
                | (reverse ? 1 << 17 : 0);
    }

    private static int encodeConv2dMode(int strideH, int strideW, int padH, int padW) {
        return (strideH & 0xFF)
                | ((strideW & 0xFF) << 8)
                | ((padH & 0xFF) << 16)
                | ((padW & 0xFF) << 24);
    }

    private static int encodePool2dMode(
            int kernelH,
            int kernelW,
            int strideH,
            int strideW,
            int padH,
            int padW,
            boolean countIncludePad
    ) {
        return (kernelH & 0xF)
                | ((kernelW & 0xF) << 4)
                | ((strideH & 0xF) << 8)
                | ((strideW & 0xF) << 12)
                | ((padH & 0xF) << 16)
                | ((padW & 0xF) << 20)
                | (countIncludePad ? 1 << 24 : 0);
    }

    private static PartitionPlanningContext planningContext(Tensor out) {
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(out.topologicalSort());
        return new PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                CompiledTensorDescriptorBuilder.build(compiledNodes),
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
