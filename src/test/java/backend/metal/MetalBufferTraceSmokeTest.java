package backend.metal;

import backend.contract.ComputeBackend;
import runtime.contract.CpuMaterializationReason;
import backend.metal.exec.PreparedMetalExecutable;
import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import trace.execution.CpuMaterializationTrace;
import trace.execution.ExecutionStepTrace;
import trace.execution.RunTrace;
import operations.elementwise.unary.relu;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import planning.intent.BackendIntentPlan;

import java.util.List;
import java.util.EnumSet;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalBufferTraceSmokeTest {
    @Test
    void tracedExecutionReportsNativeBufferPathAndCpuBoundaryMaterialization() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor input = new Tensor(new float[]{1.0f, -2.0f}, new int[]{2}, null, "input", DataType.FLOAT32);
        Tensor output = input.relu();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(output, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(output, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());
        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);
        ExecutionStepTrace metalStep = trace.steps().stream()
                .filter(step -> ComputeBackend.GPU_METAL.name().equals(step.backend()))
                .findFirst()
                .orElseThrow();
        Map<String, Object> attrs = metalStep.metadata().attributes();

        assertEquals(true, attrs.get("metalSupportsBufferBindings"));
        assertEquals("BUFFER_BINDING", attrs.get("acceleratorBufferExecutionPath"));
        assertEquals("BUFFER_BINDING_AVAILABLE", attrs.get("acceleratorBufferReasonCode"));
        assertTrue(attrs.containsKey("acceleratorBufferPreparedInputUsed"));
        assertEquals("BUFFER_BINDING", attrs.get("metalExecutionPath"));
        assertEquals("MPS_GRAPH", attrs.get("metalExecutionRoute"));
        if ("TRUE_OUTPUT_BUFFER_WRITE".equals(attrs.get("metalNativeCopyStrategy"))) {
            assertEquals("TRUE_OUTPUT_BUFFER_WRITE", attrs.get("metalNativeCopyStrategy"));
            assertEquals(true, attrs.get("metalOutputBufferWriteProven"));
            assertEquals("PROVEN_TRUE_WRITE", attrs.get("metalOutputBufferWriteStatus"));
            assertEquals(0L, attrs.get("metalNativeDeviceCopyNs"));
        } else {
            assertEquals("MPSGRAPH_RESULT_COPY", attrs.get("metalNativeCopyStrategy"));
            assertEquals(false, attrs.get("metalOutputBufferWriteProven"));
            assertEquals("COPY_REQUIRED", attrs.get("metalOutputBufferWriteStatus"));
            assertTrue(attrs.containsKey("metalNativeDeviceCopyNs"));
        }
        assertEquals(0L, attrs.get("metalNativeToJavaCopyNs"));
        assertEquals("DEVICE_OWNED", attrs.get("storageResidency"));
        assertEquals(false, attrs.get("storageCpuCurrent"));
        assertEquals(true, attrs.get("storageDeviceCurrent"));

        assertFalse(trace.cpuMaterializations().isEmpty());
        CpuMaterializationTrace materialization = trace.cpuMaterializations().getFirst();
        assertTrue(EnumSet.of(
                runtime.contract.CpuMaterializationReason.CPU_CONSUMER,
                runtime.contract.CpuMaterializationReason.GRAPH_OUTPUT
        ).contains(materialization.reason()));
        assertEquals(ComputeBackend.GPU_METAL.name(), materialization.materializedFrom());
        assertTrue(materialization.completed());
        assertArrayEquals(new double[]{1.0, 0.0}, output.toDoubleArrayCopy(), 0.0);
    }

    @Test
    void layoutAwareTraceReportsBufferPathAndLogicalMaterialization() {
        Tensor out = layoutAwareLinearReshapePermute("metal");
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(RuntimeConfig.inferenceDefaults());
        assumeNativeBufferBridge(execution);

        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);
        Map<String, Object> attrs = firstMetalStep(trace).metadata().attributes();

        assertEquals("BUFFER_BINDING", attrs.get("acceleratorBufferExecutionPath"));
        assertEquals("BUFFER_BINDING_AVAILABLE", attrs.get("acceleratorBufferReasonCode"));
        assertTrue(attrs.containsKey("acceleratorBufferPreparedInputUsed"));
        assertEquals("BUFFER_BINDING", attrs.get("metalExecutionPath"));
        assertTrue(attrs.containsKey("metalNativeDeviceCopyNs"));
        assertEquals(0L, attrs.get("metalNativeToJavaCopyNs"));
        assertEquals("DEVICE_OWNED", attrs.get("storageResidency"));
        assertEquals(false, attrs.get("storageCpuCurrent"));
        assertEquals(true, attrs.get("storageDeviceCurrent"));
        assertFalse(trace.steps().isEmpty());
    }

    @Test
    void layoutAwareTraceReportsCpuConsumerMaterializationReason() {
        Tensor input = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f,
                0.4f, 0.5f, 0.6f
        }, new int[]{2, 3}, null, "input", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                0.2f, -0.1f, 0.3f, 0.4f,
                0.5f, 0.6f, -0.2f, 0.1f,
                -0.3f, 0.7f, 0.8f, -0.4f
        }, new int[]{3, 4}, null, "weight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.05f, -0.1f, 0.2f, 0.3f}, new int[]{4}, null, "bias", DataType.FLOAT32);
        Tensor cpuConsumerBias = new Tensor(new float[]{
                0.01f, 0.02f,
                0.03f, 0.04f,
                0.05f, 0.06f,
                0.07f, 0.08f
        }, new int[]{2, 2, 2}, null, "cpuConsumerBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor reshape = linear.reshape(2, 2, 2);
        Tensor permute = reshape.permute(1, 0, 2);
        Tensor out = permute.add(cpuConsumerBias);

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(linear, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(reshape, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(permute, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());
        assumeNativeBufferBridge(execution);

        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertTrue(trace.cpuMaterializations().stream()
                .anyMatch(entry -> entry.reason() == CpuMaterializationReason.CPU_CONSUMER
                        && ComputeBackend.GPU_METAL.name().equals(entry.materializedFrom())
                        && entry.completed()));
        assertArrayEquals(new int[]{2, 2, 2}, out.getShape());
    }

    @Test
    void layoutAwareTraceReportsFallbackReasonForUnsupportedLayout() {
        Tensor input = new Tensor(new float[]{1f, -2f, 3f}, new int[]{1, 3}, null, "input", DataType.FLOAT32);
        Tensor broadcastZeroStrideOutput = new Tensor(
                new int[]{1, 3},
                new int[]{0, 1},
                List.of(input),
                new relu(),
                "broadcastZeroStrideRelu",
                DataType.FLOAT32
        );
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(broadcastZeroStrideOutput, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(broadcastZeroStrideOutput, CompileConfig.inference(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());
        assumeNativeBufferBridge(execution);

        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);
        Map<String, Object> attrs = firstMetalStep(trace).metadata().attributes();
        String reason = (String) attrs.get("acceleratorBufferReason");

        assertTrue(EnumSet.of(
                runtime.device.buffer.AcceleratorBufferReasonCode.OUTPUT_LAYOUT_UNSUPPORTED,
                runtime.device.buffer.AcceleratorBufferReasonCode.INPUT_LAYOUT_UNSUPPORTED
        ).stream().anyMatch(code -> code.name().equals(attrs.get("acceleratorBufferReasonCode"))));
        assertTrue(reason.contains("layoutClass=BROADCAST_ZERO_STRIDE_VIEW")
                || reason.contains("layoutClass=UNSUPPORTED"));
    }

    private static Tensor layoutAwareLinearReshapePermute(String labelPrefix) {
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
            BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
            backendIntentPlan = backendIntentPlan.withBackend(linear, ComputeBackend.GPU_METAL);
            backendIntentPlan = backendIntentPlan.withBackend(reshape, ComputeBackend.GPU_METAL);
            backendIntentPlan = backendIntentPlan.withBackend(permute, ComputeBackend.GPU_METAL);
        }
        return permute;
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
}
