package backend.metal;

import backend.ComputeBackend;
import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import graph.execution.trace.CpuMaterializationTrace;
import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.RunTrace;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

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
        TensorInternalAccess.setBackend(output, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(output, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());
        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);
        ExecutionStepTrace metalStep = trace.steps().stream()
                .filter(step -> ComputeBackend.GPU_METAL.name().equals(step.backend()))
                .findFirst()
                .orElseThrow();
        Map<String, Object> attrs = metalStep.metadata().attributes();

        assertEquals(true, attrs.get("metalSupportsBufferBindings"));
        assertEquals("BUFFER_BINDING", attrs.get("metalExecutionPath"));
        assertEquals(0L, attrs.get("metalNativeToJavaCopyNs"));
        assertEquals("DEVICE_OWNED", attrs.get("storageResidency"));
        assertEquals(false, attrs.get("storageCpuCurrent"));
        assertEquals(true, attrs.get("storageDeviceCurrent"));

        assertFalse(trace.cpuMaterializations().isEmpty());
        CpuMaterializationTrace materialization = trace.cpuMaterializations().getFirst();
        assertTrue(EnumSet.of(
                backend.memory.CpuMaterializationReason.CPU_CONSUMER,
                backend.memory.CpuMaterializationReason.GRAPH_OUTPUT
        ).contains(materialization.reason()));
        assertEquals(ComputeBackend.GPU_METAL.name(), materialization.materializedFrom());
        assertTrue(materialization.completed());
        assertArrayEquals(new double[]{1.0, 0.0}, output.toDoubleArrayCopy(), 0.0);
    }
}
