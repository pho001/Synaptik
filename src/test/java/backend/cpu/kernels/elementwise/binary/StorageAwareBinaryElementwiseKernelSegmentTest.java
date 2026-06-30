package backend.cpu.kernels.elementwise.binary;

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

class StorageAwareBinaryElementwiseKernelSegmentTest {
    @Test
    void cpuNativeMigratedBinaryOpsUseSegmentExecutionAndPublishNativeOutput() {
        BinaryInputs sub = binaryInputs();
        assertNativeBinary(
                "SUB",
                sub.left().sub(sub.right()),
                new double[]{-3.0d, -7.0d, 5.0d}
        );
        BinaryInputs mul = binaryInputs();
        assertNativeBinary(
                "MUL",
                mul.left().mul(mul.right()),
                new double[]{4.0d, -10.0d, -5.25d}
        );
        BinaryInputs div = binaryInputs();
        assertNativeBinary(
                "DIV",
                div.left().div(div.right()),
                new double[]{0.25d, -0.4d, -2.3333333333333335d}
        );
        BinaryInputs min = binaryInputs();
        assertNativeBinary(
                "MIN",
                min.left().min(min.right()),
                new double[]{1.0d, -2.0d, -1.5d}
        );
        BinaryInputs max = binaryInputs();
        assertNativeBinary(
                "MAX",
                max.left().max(max.right()),
                new double[]{4.0d, 5.0d, 3.5d}
        );
        BinaryInputs pow = binaryInputs();
        assertNativeBinary(
                "POW_TENSOR",
                pow.left().pow(pow.right()),
                new double[]{1.0d, -32.0d, Math.pow(3.5d, -1.5d)}
        );
    }

    @Test
    void cpuNativePowTensorBfloat16FallsBackToArrayPolicy() {
        Tensor left = new Tensor(new double[]{1.0d, 4.0d, 9.0d}, new int[]{3}, null, "left", DataType.BFLOAT16);
        Tensor right = new Tensor(new double[]{2.0d, 0.5d, -1.0d}, new int[]{3}, null, "right", DataType.BFLOAT16);
        Tensor out = left.pow(right);

        var trace = CompiledGraph.compile(out, compileConfig())
                .prepare(nativeRuntime())
                .executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{1.0d, 2.0d, 1.0d / 9.0d}, out.toDoubleArrayCopy(), 2.0e-3);
        Map<String, Object> attrs = operationStep(trace.steps(), "POW_TENSOR").metadata().attributes();
        assertEquals("CPU_ARRAY", attrs.get("actualCpuStorage"));
        assertEquals("native-kernel-unsupported:pow_tensor", attrs.get("nativeCpuFallbackReason"));
        assertEquals("ARRAY_ONLY", attrs.get("nativeCpuKernelFamily"));
    }

    @Test
    void cpuNativeAddUsesSegmentExecutionAndPublishesNativeOutput() {
        Tensor left = new Tensor(new float[]{1.0f, -2.0f, 3.5f}, new int[]{3}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{4.0f, 5.0f, -1.5f}, new int[]{3}, null, "right", DataType.FLOAT32);
        Tensor out = left.add(right);

        var trace = CompiledGraph.compile(out, compileConfig())
                .prepare(nativeRuntime())
                .executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{5.0, 3.0, 2.0}, out.toDoubleArrayCopy(), 1.0e-6);
        Map<String, Object> attrs = operationStep(trace.steps(), "ADD").metadata().attributes();
        assertEquals("CPU_NATIVE", attrs.get("actualCpuStorage"));
        assertEquals("", attrs.get("nativeCpuFallbackReason"));
        assertEquals("SEGMENT_SCALAR", attrs.get("nativeCpuKernelFamily"));
    }

    @Test
    void cpuNativeAddPreservesF32LastDimBiasBroadcastSegmentRoute() {
        Tensor matrix = new Tensor(new float[]{1.0f, -2.0f, 3.5f, 8.0f}, new int[]{2, 2}, null, "matrix", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{10.0f, -100.0f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor out = matrix.add(bias);

        var trace = CompiledGraph.compile(out, compileConfig())
                .prepare(nativeRuntime())
                .executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{11.0, -102.0, 13.5, -92.0}, out.toDoubleArrayCopy(), 1.0e-6);
        Map<String, Object> attrs = operationStep(trace.steps(), "ADD").metadata().attributes();
        assertEquals("CPU_NATIVE", attrs.get("actualCpuStorage"));
        assertEquals("", attrs.get("nativeCpuFallbackReason"));
        assertEquals("SEGMENT_SCALAR", attrs.get("nativeCpuKernelFamily"));
    }

    @Test
    void cpuNativeAddPreservesF32LastDimBiasBroadcastWhenBiasIsLeftInput() {
        Tensor bias = new Tensor(new float[]{10.0f, -100.0f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor matrix = new Tensor(new float[]{1.0f, -2.0f, 3.5f, 8.0f}, new int[]{2, 2}, null, "matrix", DataType.FLOAT32);
        Tensor out = bias.add(matrix);

        var trace = CompiledGraph.compile(out, compileConfig())
                .prepare(nativeRuntime())
                .executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{11.0, -102.0, 13.5, -92.0}, out.toDoubleArrayCopy(), 1.0e-6);
        Map<String, Object> attrs = operationStep(trace.steps(), "ADD").metadata().attributes();
        assertEquals("CPU_NATIVE", attrs.get("actualCpuStorage"));
        assertEquals("", attrs.get("nativeCpuFallbackReason"));
        assertEquals("SEGMENT_SCALAR", attrs.get("nativeCpuKernelFamily"));
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

    private static BinaryInputs binaryInputs() {
        Tensor left = new Tensor(new float[]{1.0f, -2.0f, 3.5f}, new int[]{3}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{4.0f, 5.0f, -1.5f}, new int[]{3}, null, "right", DataType.FLOAT32);
        return new BinaryInputs(left, right);
    }

    private static void assertNativeBinary(String opType, Tensor out, double[] expected) {
        var trace = CompiledGraph.compile(out, compileConfig())
                .prepare(nativeRuntime())
                .executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 1.0e-6);
        Map<String, Object> attrs = operationStep(trace.steps(), opType).metadata().attributes();
        assertEquals("CPU_NATIVE", attrs.get("actualCpuStorage"));
        assertEquals("", attrs.get("nativeCpuFallbackReason"));
        assertEquals("SEGMENT_SCALAR", attrs.get("nativeCpuKernelFamily"));
    }

    private static ExecutionStepTrace operationStep(List<ExecutionStepTrace> steps, String opType) {
        return steps.stream()
                .filter(step -> opType.equals(step.opType()))
                .findFirst()
                .orElseThrow();
    }

    private record BinaryInputs(Tensor left, Tensor right) {
    }
}
