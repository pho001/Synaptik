package backend.cpu.nativecpu;

import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PublicationPolicy;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertFalse;

class NativeCpuRegionSelectionTest {
    @Test
    void preparedExecutionNoLongerEmitsNativeCpuChainOrRegionTraceAttrs() {
        Tensor left = new Tensor(new float[]{1f, -2f, 3f, -4f}, new int[]{2, 2}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "right", DataType.FLOAT32);
        Tensor out = left.add(right).relu();

        var trace = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(
                        RuntimeConfig.inferenceDefaults()
                                .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE)
                                .withNativeCpuFailurePolicy(NativeCpuFailurePolicy.FALLBACK_TO_ARRAY)).executeTraced(ExecutionMode.FORWARD, PublicationPolicy.NONE);

        assertFalse(trace.steps().stream()
                .flatMap(step -> step.metadata().attributes().keySet().stream())
                .anyMatch(key -> key.startsWith("nativeCpuChain")));
        assertFalse(trace.steps().stream()
                .flatMap(step -> step.metadata().attributes().keySet().stream())
                .anyMatch(key -> key.startsWith("nativeCpuRegion")));
    }
}
