package backend.metal;

import backend.ComputeBackend;
import backend.memory.CpuMaterializationReason;
import backend.metal.exec.PreparedMetalExecutable;
import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
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

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalLayoutAwareDeviceFlowTest {
    @Test
    void linearReshapePermuteKeepsDeviceOwnedUntilGraphOutputMaterialization() {
        Tensor out = linearReshapePermuteGraph("metal");
        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(RuntimeConfig.inferenceDefaults());
        assumeNativeBufferBridge(execution);

        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);
        ExecutionStepTrace metal = firstMetalStep(trace);
        Map<String, Object> attrs = metal.metadata().attributes();

        assertEquals("BUFFER_BINDING", attrs.get("metalExecutionPath"));
        assertEquals("DEVICE_OWNED", attrs.get("storageResidency"));
        assertEquals(false, attrs.get("storageCpuCurrent"));
        assertEquals(true, attrs.get("storageDeviceCurrent"));
        assertFalse(trace.steps().isEmpty());
    }

    @Test
    void linearReshapePermuteMatchesCpuForwardResult() {
        Tensor expected = linearReshapePermuteGraph("cpu");
        CompiledGraph.compile(expected, OptimizerConfig.inferenceDefaults())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor actual = linearReshapePermuteGraph("metal");
        PreparedExecution execution = CompiledGraph.compile(actual, OptimizerConfig.inferenceDefaults())
                .prepare(metalTensorArrayRuntime());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(expected.toDoubleArrayCopy(), actual.toDoubleArrayCopy(), 1e-5);
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

        PreparedExecution execution = CompiledGraph.compile(broadcastZeroStrideOutput, OptimizerConfig.inferenceDefaults())
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
        CompiledGraph.compile(expected.loss(), OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        double[] expectedInputGrad = expected.input().getGradient().toDoubleArrayCopy().clone();
        double[] expectedWeightGrad = expected.weight().getGradient().toDoubleArrayCopy().clone();

        TrainingGraph actual = trainingGraph("metal");
        PreparedExecution execution = CompiledGraph.compile(actual.loss(), OptimizerConfig.trainingDefaults())
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

    private static RuntimeConfig metalTensorArrayRuntime() {
        RuntimeConfig defaults = RuntimeConfig.inferenceDefaults();
        return defaults.withAccelerator(defaults.accelerator().withMetal(
                defaults.accelerator().metal().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.OFF, true, 0)
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
                .map(step -> (PreparedMetalExecutable) step.metadata().acceleratorExecutable())
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
