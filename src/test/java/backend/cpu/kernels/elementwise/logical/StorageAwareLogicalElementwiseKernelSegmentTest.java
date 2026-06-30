package backend.cpu.kernels.elementwise.logical;

import runtime.contract.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.compile.CompileConfig;
import config.compile.RegionOptimizationConfig;
import config.compile.SemanticCanonicalizationConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import trace.execution.ExecutionStepTrace;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class StorageAwareLogicalElementwiseKernelSegmentTest {
    @Test
    void cpuNativeLogicalOpsReadNativeInputsAndPublishNativeBoolOutputs() {
        Tensor left = boolTensor(new byte[]{1, 0, 1, 1}, new int[]{2, 2}, "left");
        Tensor right = boolTensor(new byte[]{1, 1, 0, 1}, new int[]{2, 2}, "right");

        assertNativeLogical(left.logicalAnd(right), "LOGICAL_AND", new byte[]{1, 0, 0, 1});
        assertNativeLogical(left.logicalOr(right), "LOGICAL_OR", new byte[]{1, 1, 1, 1});
        assertNativeLogical(left.logicalNot(), "LOGICAL_NOT", new byte[]{0, 1, 0, 0});
    }

    @Test
    void logicalHandlesNonContiguousBroadcastWithoutLegacyStridedExecutor() {
        Tensor left = new Tensor(
                new byte[]{1, 0, 1, 1, 0, 1},
                new int[]{2, 3},
                new int[]{1, 2},
                null,
                "left_strided",
                DataType.BOOL
        );
        Tensor right = boolTensor(new byte[]{1, 0, 1}, new int[]{3}, "right_broadcast");
        Tensor out = left.logicalOr(right);

        CompiledGraph.compile(out, compileConfig())
                .prepare(stridedRuntime())
                .execute(ExecutionMode.FORWARD);

        assertArrayEquals(new byte[]{1, 1, 1, 1, 1, 1}, out.toBoolByteArrayCopy());
    }

    private static void assertNativeLogical(Tensor out, String opType, byte[] expected) {
        var trace = CompiledGraph.compile(out, compileConfig())
                .prepare(nativeRuntime())
                .executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toBoolByteArrayCopy());
        Map<String, Object> attrs = operationStep(trace.steps(), opType).metadata().attributes();
        assertEquals("CPU_NATIVE", attrs.get("actualCpuStorage"));
        assertEquals("", attrs.get("nativeCpuFallbackReason"));
        assertEquals("SEGMENT_SCALAR", attrs.get("nativeCpuKernelFamily"));
        assertEquals("CPU_NATIVE", attrs.get("storageResidency"));
    }

    private static CompileConfig compileConfig() {
        return CompileConfig.noGraphOptimizationBaseline()
                .withSemanticCanonicalization(SemanticCanonicalizationConfig.disabled())
                .withRegionOptimization(RegionOptimizationConfig.disabled());
    }

    private static RuntimeConfig nativeRuntime() {
        return RuntimeConfig.inferenceDefaults()
                .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE)
                .withNativeCpuFailurePolicy(NativeCpuFailurePolicy.FALLBACK_TO_ARRAY);
    }

    private static RuntimeConfig stridedRuntime() {
        return new RuntimeConfig(
                new CpuKernelConfig(4, 1_000_000, 1_000_000, 1_000_000, 1_024, 100_000, 1_000_000),
                config.runtime.ApproximationConfig.defaults(),
                config.runtime.BlasConfig.disabled()
        );
    }

    private static Tensor boolTensor(byte[] values, int[] shape, String label) {
        return new Tensor(values, shape, null, label, DataType.BOOL);
    }

    private static ExecutionStepTrace operationStep(List<ExecutionStepTrace> steps, String opType) {
        return steps.stream()
                .filter(step -> opType.equals(step.opType()))
                .findFirst()
                .orElseThrow();
    }
}
