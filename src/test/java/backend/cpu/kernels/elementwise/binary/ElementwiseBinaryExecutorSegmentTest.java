package backend.cpu.kernels.elementwise.binary;

import tensor.dtype.TensorDTypeOps;
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.compile.RegionOptimizationConfig;
import config.compile.SemanticCanonicalizationConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.trace.ExecutionStepTrace;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ElementwiseBinaryExecutorSegmentTest {
    @Test
    void denseMemorySegmentExecutionRunsBinaryKernelOperationsByDtype() {
        float[] leftF32 = {1.0f, -2.0f, 3.5f};
        float[] rightF32 = {4.0f, 5.0f, -1.5f};
        float[] outF32 = new float[3];

        ElementwiseBinaryExecutor.runDenseSegment(
                new CpuSubKernel(),
                DataType.FLOAT32,
                MemorySegment.ofArray(leftF32),
                MemorySegment.ofArray(rightF32),
                MemorySegment.ofArray(outF32),
                3,
                null
        );

        assertArrayEquals(new float[]{-3.0f, -7.0f, 5.0f}, outF32, 0.0f);

        double[] leftF64 = {1.0d, 4.0d, 9.0d};
        double[] rightF64 = {2.0d, 0.5d, -1.0d};
        double[] outF64 = new double[3];

        ElementwiseBinaryExecutor.runDenseSegment(
                new CpuPowTensorKernel(),
                DataType.FLOAT64,
                MemorySegment.ofArray(leftF64),
                MemorySegment.ofArray(rightF64),
                MemorySegment.ofArray(outF64),
                3,
                null
        );

        assertArrayEquals(new double[]{1.0d, 2.0d, 1.0d / 9.0d}, outF64, 1.0e-12d);

        short[] leftBF16 = bf16(1.0f, -2.0f, 3.5f);
        short[] rightBF16 = bf16(4.0f, 5.0f, -1.5f);
        short[] outBF16 = new short[3];

        ElementwiseBinaryExecutor.runDenseSegment(
                new CpuMulKernel(),
                DataType.BFLOAT16,
                MemorySegment.ofArray(leftBF16),
                MemorySegment.ofArray(rightBF16),
                MemorySegment.ofArray(outBF16),
                3,
                null
        );

        assertArrayEquals(new double[]{4.0d, -10.0d, -5.25d}, toDouble(outBF16), 0.0d);
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
        Map<String, Object> attrs = addStep(trace.steps()).metadata().attributes();
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
        Map<String, Object> attrs = addStep(trace.steps()).metadata().attributes();
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
        Map<String, Object> attrs = addStep(trace.steps()).metadata().attributes();
        assertEquals("CPU_NATIVE", attrs.get("actualCpuStorage"));
        assertEquals("", attrs.get("nativeCpuFallbackReason"));
        assertEquals("SEGMENT_SCALAR", attrs.get("nativeCpuKernelFamily"));
    }

    private static short[] bf16(float... values) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return bits;
    }

    private static double[] toDouble(short[] values) {
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = TensorDTypeOps.fromBFloat16Bits(values[i]);
        }
        return out;
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

    private static ExecutionStepTrace addStep(List<ExecutionStepTrace> steps) {
        return steps.stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElseThrow();
    }
}
