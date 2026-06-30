package backend.cpu.kernels.elementwise.where;

import runtime.contract.ExecutionMode;
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

class CpuWhereKernelStorageTest {
    @Test
    void cpuNativeWhereUsesSegmentExecutionAndPublishesNativeOutputByDtype() {
        assertNativeWhere(
                floatTensor(new float[]{1.0f, 2.0f, 3.0f}, "true_f32"),
                floatTensor(new float[]{-1.0f, -2.0f, -3.0f}, "false_f32"),
                new double[]{1.0d, -2.0d, 3.0d},
                1.0e-6
        );
        assertNativeWhere(
                doubleTensor(new double[]{1.0d, 2.0d, 3.0d}, "true_f64"),
                doubleTensor(new double[]{-1.0d, -2.0d, -3.0d}, "false_f64"),
                new double[]{1.0d, -2.0d, 3.0d},
                0.0d
        );
        assertNativeWhere(
                bf16Tensor(new double[]{1.0d, 2.0d, 3.5d}, "true_bf16"),
                bf16Tensor(new double[]{-1.0d, -2.0d, -3.5d}, "false_bf16"),
                new double[]{1.0d, -2.0d, 3.5d},
                1.0e-2
        );
    }

    private static void assertNativeWhere(Tensor ifTrue, Tensor ifFalse, double[] expected, double tolerance) {
        Tensor condition = new Tensor(new byte[]{1, 0, 1}, new int[]{3}, null, "condition", DataType.BOOL);
        Tensor out = Tensor.where(condition, ifTrue, ifFalse);

        var trace = CompiledGraph.compile(out, compileConfig())
                .prepare(nativeRuntime())
                .executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), tolerance);
        Map<String, Object> attrs = operationStep(trace.steps(), "WHERE").metadata().attributes();
        assertEquals("CPU_NATIVE", attrs.get("actualCpuStorage"));
        assertEquals("", attrs.get("nativeCpuFallbackReason"));
        assertEquals("SEGMENT_SCALAR", attrs.get("nativeCpuKernelFamily"));
    }

    private static Tensor floatTensor(float[] values, String name) {
        return new Tensor(values, new int[]{values.length}, null, name, DataType.FLOAT32);
    }

    private static Tensor doubleTensor(double[] values, String name) {
        return new Tensor(values, new int[]{values.length}, null, name, DataType.FLOAT64);
    }

    private static Tensor bf16Tensor(double[] values, String name) {
        return new Tensor(values, new int[]{values.length}, null, name, DataType.BFLOAT16);
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

    private static ExecutionStepTrace operationStep(List<ExecutionStepTrace> steps, String opType) {
        return steps.stream()
                .filter(step -> opType.equals(step.opType()))
                .findFirst()
                .orElseThrow();
    }
}
