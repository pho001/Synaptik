package backend.cpu.kernels.elementwise.compare;

import runtime.contract.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.compile.CompileConfig;
import config.compile.PartitionExecutionConfig;
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

class StorageAwareCompareElementwiseKernelSegmentTest {
    @Test
    void cpuNativeCompareOpsReadNativeInputsAndPublishBoolArrays() {
        Tensor left = tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, "left");
        Tensor right = tensor(new float[]{2.0f, 2.0f, 1.0f, 4.0f}, "right");

        assertNativeCompare(left.greaterThan(right), "GT", new byte[]{0, 0, 1, 0});
        assertNativeCompare(left.greaterOrEqual(right), "GE", new byte[]{0, 1, 1, 1});
        assertNativeCompare(left.lessThan(right), "LT", new byte[]{1, 0, 0, 0});
        assertNativeCompare(left.lessOrEqual(right), "LE", new byte[]{1, 1, 0, 1});
        assertNativeCompare(left.equalTo(right), "EQ", new byte[]{0, 1, 0, 1});
        assertNativeCompare(left.notEqualTo(right), "NE", new byte[]{1, 0, 1, 0});
    }

    @Test
    void compareHandlesNonContiguousBroadcastWithoutLegacyStridedExecutor() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                new int[]{1, 2},
                null,
                "left_strided",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(new float[]{2.0f, 3.0f, 4.0f}, new int[]{3}, null, "right_broadcast", DataType.FLOAT32);
        Tensor out = left.greaterOrEqual(right);

        CompiledGraph.compile(out, compileConfig())
                .prepare(stridedRuntime())
                .execute(ExecutionMode.FORWARD);

        assertArrayEquals(new byte[]{0, 1, 1, 1, 1, 1}, out.toBoolByteArrayCopy());
    }

    @Test
    void comparePreservesJavaNanComparisonSemantics() {
        Tensor left = new Tensor(
                new double[]{Double.NaN, Double.NaN, 1.0d, -0.0d},
                new int[]{4},
                null,
                "left_nan",
                DataType.FLOAT64
        );
        Tensor right = new Tensor(
                new double[]{Double.NaN, 0.0d, Double.NaN, 0.0d},
                new int[]{4},
                null,
                "right_nan",
                DataType.FLOAT64
        );

        Tensor eq = left.equalTo(right);
        Tensor ne = left.notEqualTo(right);
        Tensor gt = left.greaterThan(right);
        Tensor le = left.lessOrEqual(right);

        CompiledGraph.compile(eq, compileConfig()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(ne, compileConfig()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(gt, compileConfig()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(le, compileConfig()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new byte[]{0, 0, 0, 1}, eq.toBoolByteArrayCopy());
        assertArrayEquals(new byte[]{1, 1, 1, 0}, ne.toBoolByteArrayCopy());
        assertArrayEquals(new byte[]{0, 0, 0, 0}, gt.toBoolByteArrayCopy());
        assertArrayEquals(new byte[]{0, 0, 0, 1}, le.toBoolByteArrayCopy());
    }

    @Test
    void bfloat16CompareUsesStoredBfloat16Values() {
        Tensor left = new Tensor(
                new double[]{1.0d, 2.0d, Double.NaN, -1.0d},
                new int[]{4},
                null,
                "left_bf16",
                DataType.BFLOAT16
        );
        Tensor right = new Tensor(
                new double[]{1.0d, 3.0d, Double.NaN, -2.0d},
                new int[]{4},
                null,
                "right_bf16",
                DataType.BFLOAT16
        );
        Tensor eq = left.equalTo(right);
        Tensor ne = left.notEqualTo(right);
        Tensor gt = left.greaterThan(right);

        CompiledGraph.compile(eq, compileConfig()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(ne, compileConfig()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(gt, compileConfig()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new byte[]{1, 0, 0, 0}, eq.toBoolByteArrayCopy());
        assertArrayEquals(new byte[]{0, 1, 1, 1}, ne.toBoolByteArrayCopy());
        assertArrayEquals(new byte[]{0, 0, 0, 1}, gt.toBoolByteArrayCopy());
    }

    private static void assertNativeCompare(Tensor out, String opType, byte[] expected) {
        var trace = CompiledGraph.compile(out, compileConfig())
                .prepare(nativeRuntime())
                .executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toBoolByteArrayCopy());
        Map<String, Object> attrs = operationStep(trace.steps(), opType).metadata().attributes();
        assertEquals("CPU_ARRAY", attrs.get("actualCpuStorage"));
        assertEquals("", attrs.get("nativeCpuFallbackReason"));
        assertEquals("SEGMENT_SCALAR", attrs.get("nativeCpuKernelFamily"));
        assertEquals("CPU_ARRAY", attrs.get("storageResidency"));
    }

    private static CompileConfig compileConfig() {
        return CompileConfig.noGraphOptimizationBaseline()
                .withSemanticCanonicalization(SemanticCanonicalizationConfig.disabled())
                .withPartitionExecution(PartitionExecutionConfig.disabled());
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

    private static Tensor tensor(float[] values, String label) {
        return new Tensor(values, new int[]{values.length}, null, label, DataType.FLOAT32);
    }

    private static ExecutionStepTrace operationStep(List<ExecutionStepTrace> steps, String opType) {
        return steps.stream()
                .filter(step -> opType.equals(step.opType()))
                .findFirst()
                .orElseThrow();
    }
}
