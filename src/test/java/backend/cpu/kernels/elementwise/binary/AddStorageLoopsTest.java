package backend.cpu.kernels.elementwise.binary;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuExecutionMode;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.kernels.storage.CpuStorageBindings;
import backend.cpu.kernels.storage.CpuStorageView;
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

class AddStorageLoopsTest {
    @Test
    void denseArrayLoopUsesExistingSpecializedAddKernels() {
        double[] leftF64 = {1.0, -2.0, 3.5};
        double[] rightF64 = {4.0, 5.0, -1.5};
        double[] outF64 = new double[3];

        AddStorageLoops.runArrayDense(bindings(DataType.FLOAT64, leftF64, rightF64, outF64), scalarHints(3), null, null);

        assertArrayEquals(new double[]{5.0, 3.0, 2.0}, outF64, 0.0);

        float[] leftF32 = {1.0f, -2.0f, 3.5f};
        float[] rightF32 = {4.0f, 5.0f, -1.5f};
        float[] outF32 = new float[3];

        AddStorageLoops.runArrayDense(bindings(DataType.FLOAT32, leftF32, rightF32, outF32), scalarHints(3), null, null);

        assertArrayEquals(new float[]{5.0f, 3.0f, 2.0f}, outF32, 0.0f);

        short[] leftBF16 = bf16(1.0f, -2.0f, 3.5f);
        short[] rightBF16 = bf16(4.0f, 5.0f, -1.5f);
        short[] outBF16 = new short[3];

        AddStorageLoops.runArrayDense(bindings(DataType.BFLOAT16, leftBF16, rightBF16, outBF16), scalarHints(3), null, null);

        assertArrayEquals(new double[]{5.0, 3.0, 2.0}, toDouble(outBF16), 0.0);
    }

    @Test
    void denseMemorySegmentLoopAddsF32F64AndBf16() {
        float[] leftF32 = {1.0f, -2.0f, 3.5f};
        float[] rightF32 = {4.0f, 5.0f, -1.5f};
        float[] outF32 = new float[3];

        AddStorageLoops.runSegmentDense(segmentBindings(DataType.FLOAT32, MemorySegment.ofArray(leftF32), MemorySegment.ofArray(rightF32), MemorySegment.ofArray(outF32), 3));

        assertArrayEquals(new float[]{5.0f, 3.0f, 2.0f}, outF32, 0.0f);

        double[] leftF64 = {1.0, -2.0, 3.5};
        double[] rightF64 = {4.0, 5.0, -1.5};
        double[] outF64 = new double[3];

        AddStorageLoops.runSegmentDense(segmentBindings(DataType.FLOAT64, MemorySegment.ofArray(leftF64), MemorySegment.ofArray(rightF64), MemorySegment.ofArray(outF64), 3));

        assertArrayEquals(new double[]{5.0, 3.0, 2.0}, outF64, 0.0);

        short[] leftBF16 = bf16(1.0f, -2.0f, 3.5f);
        short[] rightBF16 = bf16(4.0f, 5.0f, -1.5f);
        short[] outBF16 = new short[3];

        AddStorageLoops.runSegmentDense(segmentBindings(DataType.BFLOAT16, MemorySegment.ofArray(leftBF16), MemorySegment.ofArray(rightBF16), MemorySegment.ofArray(outBF16), 3));

        assertArrayEquals(new double[]{5.0, 3.0, 2.0}, toDouble(outBF16), 0.0);
    }

    @Test
    void cpuNativeAddUsesAddStorageLoopAndPublishesNativeOutput() {
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

    private static CpuStorageBindings bindings(DataType dtype, Object left, Object right, Object output) {
        int size = java.lang.reflect.Array.getLength(output);
        return new CpuStorageBindings(
                List.of(arrayView(dtype, left, size), arrayView(dtype, right, size)),
                arrayView(dtype, output, size)
        );
    }

    private static CpuStorageBindings segmentBindings(DataType dtype, MemorySegment left, MemorySegment right, MemorySegment output, int size) {
        return new CpuStorageBindings(
                List.of(segmentView(dtype, left, size), segmentView(dtype, right, size)),
                segmentView(dtype, output, size)
        );
    }

    private static CpuStorageView arrayView(DataType dtype, Object array, int size) {
        return CpuStorageView.array(dtype, array, new int[]{size}, new int[]{1}, 0, size);
    }

    private static CpuStorageView segmentView(DataType dtype, MemorySegment segment, int size) {
        return CpuStorageView.segment(dtype, segment, new int[]{size}, new int[]{1}, 0, size);
    }

    private static ResolvedDispatchHints scalarHints(int size) {
        return new ResolvedDispatchHints(size, CpuExecutionMode.SCALAR, size, size, 1, 1, false);
    }

    private static short[] bf16(float... values) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = CpuDTypeOps.toBFloat16Bits(values[i]);
        }
        return bits;
    }

    private static double[] toDouble(short[] values) {
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = CpuDTypeOps.fromBFloat16Bits(values[i]);
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
