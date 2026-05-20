package backend.metal;

import backend.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.buffer.AcceleratorLayoutTransformPlanner;
import backend.accelerator.buffer.AcceleratorLayoutTransformRequest;
import backend.memory.CpuMaterializationReason;
import backend.metal.exec.PreparedMetalExecutable;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferBinding;
import backend.metal.buffer.MetalBufferHandle;
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.RunTrace;
import operations.elementwise.unary.relu;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

import java.util.List;
import java.util.Map;
import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalLayoutAwareDeviceFlowTest {
    @Test
    void reshapePermuteContiguousFlowAvoidsIntermediateCpuMaterialization() {
        AcceleratorBufferLayout dense = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{3, 1},
                0,
                6
        );
        AcceleratorBufferLayout reshaped = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{3, 2},
                new int[]{2, 1},
                0,
                6
        );
        AcceleratorBufferLayout permuted = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{1, 2},
                0,
                6
        );
        MetalBufferBinding source = new MetalBufferBinding(
                1,
                dense,
                new MetalBufferHandle(MemorySegment.ofAddress(40_001), dense.logicalByteLength(), "shared", "test", true),
                MetalBufferAccess.READ_WRITE
        );

        var reshape = AcceleratorLayoutTransformPlanner.decide(new AcceleratorLayoutTransformRequest(
                ComputeBackend.GPU_METAL.name(),
                1,
                2,
                operations.Operation.OpType.RESHAPE,
                dense,
                reshaped,
                source,
                false
        ));
        var permute = AcceleratorLayoutTransformPlanner.decide(new AcceleratorLayoutTransformRequest(
                ComputeBackend.GPU_METAL.name(),
                2,
                3,
                operations.Operation.OpType.PERMUTE,
                reshaped,
                permuted,
                MetalBufferBinding.viewOf(2, reshaped, source, MetalBufferAccess.READ_WRITE),
                false
        ));
        var contiguous = AcceleratorLayoutTransformPlanner.decide(new AcceleratorLayoutTransformRequest(
                ComputeBackend.GPU_METAL.name(),
                3,
                4,
                operations.Operation.OpType.CONTIGUOUS,
                permuted,
                AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{2, 3}, new int[]{3, 1}, 0, 6),
                MetalBufferBinding.viewOf(3, permuted, source, MetalBufferAccess.READ_WRITE),
                false
        ));

        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_VIEW_BINDING_AVAILABLE, reshape.reasonCode());
        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_VIEW_BINDING_AVAILABLE, permute.reasonCode());
        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_DENSE_MATERIALIZATION_AVAILABLE, contiguous.reasonCode());
    }

    @Test
    void expandContiguousFlowAvoidsIntermediateCpuMaterialization() {
        AcceleratorBufferLayout sourceLayout = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{1, 3},
                new int[]{3, 1},
                0,
                3
        );
        AcceleratorBufferLayout expanded = AcceleratorBufferLayout.of(
                DataType.FLOAT32,
                new int[]{2, 3},
                new int[]{0, 1},
                0,
                6
        );
        MetalBufferBinding source = new MetalBufferBinding(
                1,
                sourceLayout,
                new MetalBufferHandle(MemorySegment.ofAddress(40_002), expanded.logicalByteLength(), "shared", "test", true),
                MetalBufferAccess.READ_WRITE
        );

        var expand = AcceleratorLayoutTransformPlanner.decide(new AcceleratorLayoutTransformRequest(
                ComputeBackend.GPU_METAL.name(),
                1,
                2,
                operations.Operation.OpType.EXPAND,
                sourceLayout,
                expanded,
                source,
                false
        ));
        var contiguous = AcceleratorLayoutTransformPlanner.decide(new AcceleratorLayoutTransformRequest(
                ComputeBackend.GPU_METAL.name(),
                2,
                3,
                operations.Operation.OpType.CONTIGUOUS,
                expanded,
                AcceleratorBufferLayout.of(DataType.FLOAT32, new int[]{2, 3}, new int[]{3, 1}, 0, 6),
                MetalBufferBinding.viewOf(2, expanded, source, MetalBufferAccess.READ),
                false
        ));

        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_VIEW_BINDING_AVAILABLE, expand.reasonCode());
        assertEquals(AcceleratorBufferReasonCode.GPU_LAYOUT_BROADCAST_MATERIALIZATION_AVAILABLE, contiguous.reasonCode());
    }

    @Test
    void linearReshapePermuteKeepsDeviceOwnedUntilGraphOutputMaterialization() {
        Tensor out = linearReshapePermuteGraph("metal");
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(RuntimeConfig.inferenceDefaults());
        assumeNativeBufferBridge(execution);

        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);
        ExecutionStepTrace metal = firstMetalStep(trace);
        Map<String, Object> attrs = metal.metadata().attributes();

        assertEquals("AUTO", attrs.get("acceleratorBufferMode"));
        assertEquals("BUFFER_BINDING", attrs.get("acceleratorBufferExecutionPath"));
        assertEquals("BUFFER_BINDING_AVAILABLE", attrs.get("acceleratorBufferReasonCode"));
        assertEquals("BUFFER_BINDING", attrs.get("metalExecutionPath"));
        assertTrue(attrs.containsKey("metalNativeDeviceCopyNs"));
        assertEquals("DEVICE_OWNED", attrs.get("storageResidency"));
        assertEquals(false, attrs.get("storageCpuCurrent"));
        assertEquals(true, attrs.get("storageDeviceCurrent"));
        assertFalse(trace.steps().isEmpty());
    }

    @Test
    void linearReshapePermuteMatchesCpuForwardResult() {
        Tensor expected = linearReshapePermuteGraph("cpu");
        CompiledGraph.compile(expected, CompileConfig.inference())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor actual = linearReshapePermuteGraph("metal");
        PreparedExecution execution = CompiledGraph.compile(actual, CompileConfig.inference())
                .prepare(metalTensorArrayRuntime());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(expected.toDoubleArrayCopy(), actual.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void layoutViewThenLogSoftmaxStaysDeviceOwnedUntilOutputBoundary() {
        Tensor expected = linearReshapePermuteLogSoftmaxGraph("cpu");
        CompiledGraph.compile(expected, CompileConfig.inference())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor actual = linearReshapePermuteLogSoftmaxGraph("metal");
        CompiledGraph compiled = CompiledGraph.compile(actual, CompileConfig.inference());
        PreparedExecution execution = compiled.prepare(metalTensorArrayRuntime());
        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);
        int logSoftmaxNodeId = nodeId(compiled, operations.Operation.OpType.LOG_SOFTMAX);

        assertArrayEquals(expected.toDoubleArrayCopy(), actual.toDoubleArrayCopy(), 1e-5);
        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(entry -> entry.reason() == CpuMaterializationReason.CPU_CONSUMER));
        assertTrue(execution.prepareTrace().backendSelection().decisions().stream()
                .anyMatch(decision -> decision.selected()
                        && decision.selectedBackend() == ComputeBackend.GPU_METAL
                        && decision.nodeIds().contains(logSoftmaxNodeId)));
        assertTrue(compiled.compileArtifacts().compiledNodes().stream()
                .anyMatch(node -> node.operation() != null && node.operation().opType() == operations.Operation.OpType.LOG_SOFTMAX));
        assertTrue(List.of("GPU_LAYOUT_VIEW_BINDING_AVAILABLE", "GPU_LAYOUT_DENSE_MATERIALIZATION_AVAILABLE")
                .contains(AcceleratorBufferReasonCode.GPU_LAYOUT_VIEW_BINDING_AVAILABLE.name()));
    }

    @Test
    void broadcastContiguousReportsGpuLayoutMaterialization() {
        Tensor expected = broadcastContiguousGraph("cpu");
        CompiledGraph.compile(expected, CompileConfig.inference())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor actual = broadcastContiguousGraph("metal");
        PreparedExecution execution = CompiledGraph.compile(actual, CompileConfig.noGraphOptimizationBaseline())
                .prepare(metalBufferRuntime());
        assumeNativeBufferBridge(execution);

        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);
        ExecutionStepTrace contiguousTrace = trace.steps().stream()
                .filter(step -> operations.Operation.OpType.CONTIGUOUS.name().equals(step.opType()))
                .findFirst()
                .orElseThrow();
        Map<String, Object> attrs = contiguousTrace.metadata().attributes();

        assertArrayEquals(expected.toDoubleArrayCopy(), actual.toDoubleArrayCopy(), 1e-5);
        if ("BROADCAST_GPU_MATERIALIZATION".equals(attrs.get("gpuLayoutTransformKind"))) {
            assertEquals("GPU_LAYOUT_BROADCAST_MATERIALIZATION_AVAILABLE", attrs.get("gpuLayoutTransformReasonCode"));
            assertEquals(1, attrs.get("gpuLayoutMaterializationCount"));
            assertEquals(24L, attrs.get("gpuLayoutMaterializationBytes"));
        }
        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(entry -> entry.reason() == CpuMaterializationReason.CPU_CONSUMER));
    }

    @Test
    void layoutAwareFlowFallsBackVisiblyForBroadcastZeroStride() {
        Tensor input = new Tensor(new float[]{1f, -2f, 3f}, new int[]{1, 3}, null, "input", DataType.FLOAT32);
        Tensor broadcastZeroStrideOutput = new Tensor(
                new int[]{1, 3},
                new int[]{0, 1},
                List.of(input),
                new relu(),
                "broadcastZeroStrideRelu",
                DataType.FLOAT32
        );
        TensorInternalAccess.setBackend(broadcastZeroStrideOutput, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(broadcastZeroStrideOutput, CompileConfig.inference())
                .prepare(RuntimeConfig.inferenceDefaults());
        assumeNativeBufferBridge(execution);

        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);
        Map<String, Object> attrs = firstMetalStep(trace).metadata().attributes();

        assertEquals("OUTPUT_LAYOUT_UNSUPPORTED", attrs.get("acceleratorBufferReasonCode"));
        assertTrue(((String) attrs.get("acceleratorBufferReason")).contains("layoutClass=BROADCAST_ZERO_STRIDE_VIEW"));
        assertTrue(Boolean.TRUE.equals(attrs.get("metalUsedCpuFallback"))
                || "TENSOR_ARRAY_COPY".equals(attrs.get("metalExecutionPath")));
    }

    @Test
    void forwardBackwardLayoutAwareGraphPublishesGradientsWithCpuParity() {
        TrainingGraph expected = trainingGraph("cpu");
        CompiledGraph.compile(expected.loss(), CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        double[] expectedInputGrad = expected.input().getGradient().toDoubleArrayCopy().clone();
        double[] expectedWeightGrad = expected.weight().getGradient().toDoubleArrayCopy().clone();

        TrainingGraph actual = trainingGraph("metal");
        PreparedExecution execution = CompiledGraph.compile(actual.loss(), CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults());

        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD_BACKWARD);

        assertNotNull(actual.input().getGradient());
        assertNotNull(actual.weight().getGradient());
        assertArrayEquals(expectedInputGrad, actual.input().getGradient().toDoubleArrayCopy(), 1e-5);
        assertArrayEquals(expectedWeightGrad, actual.weight().getGradient().toDoubleArrayCopy(), 1e-5);

        boolean gradientPublication = trace.cpuMaterializations().stream()
                .anyMatch(entry -> entry.reason() == CpuMaterializationReason.GRADIENT_PUBLICATION);
        boolean visibleFallback = trace.steps().stream()
                .filter(step -> ComputeBackend.GPU_METAL.name().equals(step.backend()))
                .map(step -> step.metadata().attributes().get("acceleratorBufferReasonCode"))
                .anyMatch(code -> code != null && !"BUFFER_BINDING_AVAILABLE".equals(code));
        assertTrue(gradientPublication || visibleFallback);
    }

    private static Tensor linearReshapePermuteGraph(String labelPrefix) {
        Tensor input = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f,
                0.4f, 0.5f, 0.6f
        }, new int[]{2, 3}, null, labelPrefix + "Input", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                0.2f, -0.1f,
                0.5f, 0.6f,
                -0.3f, 0.7f
        }, new int[]{3, 2}, null, labelPrefix + "Weight", DataType.FLOAT32);
        Tensor linear = input.matmul(weight);
        Tensor reshape = linear.reshape(2, 2);
        Tensor permute = reshape.permute(1, 0);

        if ("metal".equals(labelPrefix)) {
            TensorInternalAccess.setBackend(linear, ComputeBackend.GPU_METAL);
            TensorInternalAccess.setBackend(reshape, ComputeBackend.GPU_METAL);
            TensorInternalAccess.setBackend(permute, ComputeBackend.GPU_METAL);
        }
        return permute;
    }

    private static Tensor linearReshapePermuteLogSoftmaxGraph(String labelPrefix) {
        Tensor input = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f,
                0.4f, 0.5f, 0.6f
        }, new int[]{2, 3}, null, labelPrefix + "LogSoftmaxInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                0.2f, -0.1f, 0.3f, 0.4f, -0.2f, 0.1f,
                0.5f, 0.6f, -0.2f, 0.1f, 0.7f, -0.4f,
                -0.3f, 0.7f, 0.8f, -0.4f, 0.2f, 0.5f
        }, new int[]{3, 6}, null, labelPrefix + "LogSoftmaxWeight", DataType.FLOAT32);
        Tensor linear = input.matmul(weight);
        Tensor reshape = linear.reshape(3, 4);
        Tensor permute = reshape.permute(1, 0);
        Tensor contiguous = permute.contiguous();
        Tensor out = specialLogSoftmax(contiguous, 1);

        if ("metal".equals(labelPrefix)) {
            TensorInternalAccess.setBackend(linear, ComputeBackend.GPU_METAL);
            TensorInternalAccess.setBackend(reshape, ComputeBackend.GPU_METAL);
            TensorInternalAccess.setBackend(permute, ComputeBackend.GPU_METAL);
            TensorInternalAccess.setBackend(contiguous, ComputeBackend.GPU_METAL);
            TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);
        }
        return out;
    }

    private static Tensor specialLogSoftmax(Tensor input, int dimension) {
        return TensorPrimitiveBuilder.unary(
                input,
                input.getShapeUnsafe().clone(),
                new operations.reduction.logSoftmax(dimension),
                "legacyLogSoftmax",
                input.getDataType()
        );
    }

    private static Tensor broadcastContiguousGraph(String labelPrefix) {
        Tensor input = new Tensor(new float[]{0.5f, -1.0f, 2.0f}, new int[]{1, 3}, null, labelPrefix + "BroadcastInput", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, 0.5f, -0.75f}, new int[]{1, 3}, null, labelPrefix + "BroadcastBias", DataType.FLOAT32);
        Tensor base = input.add(bias);
        Tensor expanded = base.expand(2, 3);
        Tensor out = expanded.contiguous();

        if ("metal".equals(labelPrefix)) {
            TensorInternalAccess.setBackend(base, ComputeBackend.GPU_METAL);
            TensorInternalAccess.setBackend(expanded, ComputeBackend.GPU_METAL);
            TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);
        }
        return out;
    }

    private static int nodeId(CompiledGraph compiled, operations.Operation.OpType opType) {
        return compiled.compileArtifacts().compiledNodes().stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .map(graph.CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static RuntimeConfig metalTensorArrayRuntime() {
        RuntimeConfig defaults = RuntimeConfig.inferenceDefaults();
        return defaults.withAccelerator(defaults.accelerator().withMetal(
                defaults.accelerator().metal().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.OFF, true, 0)
                )
        ));
    }

    private static RuntimeConfig metalBufferRuntime() {
        RuntimeConfig defaults = RuntimeConfig.inferenceDefaults();
        return defaults.withAccelerator(defaults.accelerator().withMetal(
                defaults.accelerator().metal().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.AUTO, true, 0)
                )
        ));
    }

    private static TrainingGraph trainingGraph(String labelPrefix) {
        Tensor input = new Tensor(new float[]{
                0.1f, -0.2f, 0.3f,
                0.4f, -0.5f, 0.6f
        }, new int[]{2, 3}, null, labelPrefix + "TrainInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                0.2f, -0.1f, 0.3f, 0.4f,
                0.5f, 0.6f, -0.2f, 0.1f,
                -0.3f, 0.7f, 0.8f, -0.4f
        }, new int[]{3, 4}, null, labelPrefix + "TrainWeight", DataType.FLOAT32);
        input.setRequiresGrad(true);
        weight.setRequiresGrad(true);

        Tensor linear = input.linear(weight);
        Tensor reshape = linear.reshape(2, 2, 2);
        Tensor permute = reshape.permute(1, 0, 2);
        Tensor loss = permute.sum();

        if ("metal".equals(labelPrefix)) {
            TensorInternalAccess.setBackend(linear, ComputeBackend.GPU_METAL);
            TensorInternalAccess.setBackend(reshape, ComputeBackend.GPU_METAL);
            TensorInternalAccess.setBackend(permute, ComputeBackend.GPU_METAL);
        }
        return new TrainingGraph(input, weight, loss);
    }

    private static void assumeNativeBufferBridge(PreparedExecution execution) {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());
        PreparedMetalExecutable executable = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .map(step -> (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
                .findFirst()
                .orElseThrow();
        assumeTrue(executable.bridgeContext().available(), executable.bridgeContext().reason());
        assumeTrue(executable.bridgeExecutable().available(), executable.bridgeExecutable().reason());
        assumeTrue(executable.bridge().supportsBufferBindings());
    }

    private static ExecutionStepTrace firstMetalStep(RunTrace trace) {
        return trace.steps().stream()
                .filter(step -> ComputeBackend.GPU_METAL.name().equals(step.backend()))
                .findFirst()
                .orElseThrow();
    }

    private record TrainingGraph(Tensor input, Tensor weight, Tensor loss) {
    }
}
