package backend.cpu.kernels.elementwise.unary;

import backend.ApproxMode;
import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.compile.RegionOptimizationConfig;
import config.compile.SemanticCanonicalizationConfig;
import config.runtime.ApproximationConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import backend.cpu.storage.CpuStorageView;
import graph.CompiledGraph;
import trace.execution.ExecutionStepTrace;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import utils.FastTranscendentals;
import utils.SpecialFunctions;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class StorageAwareUnaryElementwiseKernelSegmentTest {
    @Test
    void cpuNativeMigratedUnaryOpsUseSegmentExecutionAndPublishNativeOutput() {
        assertNativeUnary(
                "NEG",
                input().neg(),
                new double[]{1.0d, -2.0d, 3.5d}
        );
        assertNativeUnary(
                "ABS",
                input().abs(),
                new double[]{1.0d, 2.0d, 3.5d}
        );
        assertNativeUnary(
                "RELU",
                input().relu(),
                new double[]{0.0d, 2.0d, 0.0d}
        );
        assertNativeUnary(
                "INV",
                input().inv(),
                new double[]{-1.0d, 0.5d, -0.2857143d}
        );
        assertNativeUnary(
                "SQRT",
                positiveInput().sqrt(),
                new double[]{1.0d, 2.0d, 3.0d}
        );
        assertNativeUnary(
                "SIGN",
                input().sign(),
                new double[]{-1.0d, 1.0d, -1.0d}
        );
        assertNativeUnary(
                "FLOOR",
                input().floor(),
                new double[]{-1.0d, 2.0d, -4.0d}
        );
        assertNativeUnary(
                "CEIL",
                input().ceil(),
                new double[]{-1.0d, 2.0d, -3.0d}
        );
        assertNativeUnary(
                "LOG",
                positiveInput().log(),
                new double[]{0.0d, Math.log(4.0d), Math.log(9.0d)}
        );
        assertNativeUnary(
                "EXP",
                input().exp(),
                new double[]{(float) Math.exp(-1.0f), (float) Math.exp(2.0f), (float) Math.exp(-3.5f)}
        );
        assertNativeUnary(
                "TANH",
                input().tanh(),
                new double[]{(float) Math.tanh(-1.0f), (float) Math.tanh(2.0f), (float) Math.tanh(-3.5f)}
        );
        assertNativeUnary(
                "FAST_EXP",
                input().fastExp(),
                new double[]{
                        FastTranscendentals.fastExpF32(-1.0f),
                        FastTranscendentals.fastExpF32(2.0f),
                        FastTranscendentals.fastExpF32(-3.5f)
                }
        );
        assertNativeUnary(
                "FAST_TANH",
                input().fastTanh(),
                new double[]{
                        FastTranscendentals.fastTanhF32(-1.0f),
                        FastTranscendentals.fastTanhF32(2.0f),
                        FastTranscendentals.fastTanhF32(-3.5f)
                }
        );
        assertNativeUnary(
                "SIGMOID",
                input().sigmoid(),
                new double[]{
                        1.0f / (1.0f + (float) Math.exp(1.0f)),
                        1.0f / (1.0f + (float) Math.exp(-2.0f)),
                        1.0f / (1.0f + (float) Math.exp(3.5f))
                }
        );
    }

    @Test
    void cpuNativeScalarUnaryOpsUseSegmentExecutionAndPublishNativeOutput() {
        assertNativeUnary(
                "MUL_SCALAR",
                input().mul(0.25d),
                new double[]{-0.25d, 0.5d, -0.875d}
        );
        assertNativeUnary(
                "CLAMP_MIN",
                input().clampMin(0.0d),
                new double[]{0.0d, 2.0d, 0.0d}
        );
        assertNativeUnary(
                "CLAMP_MAX",
                input().clampMax(0.0d),
                new double[]{-1.0d, 0.0d, -3.5d}
        );
        assertNativeUnary(
                "POW",
                positiveInput().pow(1.5d),
                new double[]{1.0d, (float) Math.pow(4.0f, 1.5f), (float) Math.pow(9.0f, 1.5f)}
        );
    }

    @Test
    void cpuNativeRuntimeApproxFlagsRouteExpAndTanhThroughFastStorageAwareKernels() {
        assertNativeUnary(
                "EXP",
                input().exp(),
                new double[]{
                        FastTranscendentals.fastExpF32(-1.0f),
                        FastTranscendentals.fastExpF32(2.0f),
                        FastTranscendentals.fastExpF32(-3.5f)
                },
                nativeApproxRuntime()
        );
        assertNativeUnary(
                "TANH",
                input().tanh(),
                new double[]{
                        FastTranscendentals.fastTanhF32(-1.0f),
                        FastTranscendentals.fastTanhF32(2.0f),
                        FastTranscendentals.fastTanhF32(-3.5f)
                },
                nativeApproxRuntime()
        );
    }

    @Test
    void cpuNativeErfUsesArrayFallbackPolicy() {
        Tensor input = new Tensor(new float[]{-1.0f, 0.0f, 2.0f}, new int[]{3}, null, "erfInput", DataType.FLOAT32);
        Tensor out = input.erf();

        var trace = CompiledGraph.compile(out, compileConfig())
                .prepare(nativeRuntime())
                .executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(
                new double[]{SpecialFunctions.erf(-1.0f), SpecialFunctions.erf(0.0f), SpecialFunctions.erf(2.0f)},
                out.toDoubleArrayCopy(),
                1.0e-6
        );
        Map<String, Object> attrs = operationStep(trace.steps(), "ERF").metadata().attributes();
        assertEquals("CPU_ARRAY", attrs.get("actualCpuStorage"));
        assertEquals("native-kernel-unsupported:erf", attrs.get("nativeCpuFallbackReason"));
    }

    @Test
    void cpuNativeReluBfloat16UsesSegmentExecutionAndPublishesNativeOutput() {
        Tensor input = new Tensor(new double[]{-1.0d, 2.0d, -3.5d}, new int[]{3}, null, "input", DataType.BFLOAT16);
        Tensor out = input.relu();

        var trace = CompiledGraph.compile(out, compileConfig())
                .prepare(nativeRuntime())
                .executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{0.0d, 2.0d, 0.0d}, out.toDoubleArrayCopy(), 2.0e-3);
        Map<String, Object> attrs = operationStep(trace.steps(), "RELU").metadata().attributes();
        assertEquals("CPU_NATIVE", attrs.get("actualCpuStorage"));
        assertEquals("", attrs.get("nativeCpuFallbackReason"));
        assertEquals("SEGMENT_SCALAR", attrs.get("nativeCpuKernelFamily"));
    }

    @Test
    void cpuNativeBfloat16UnsupportedMigratedUnaryOpsKeepArrayFallbackPolicy() {
        Tensor input = new Tensor(new double[]{1.0d, -2.0d, 4.0d}, new int[]{3}, null, "input", DataType.BFLOAT16);
        Tensor out = input.inv();

        var trace = CompiledGraph.compile(out, compileConfig())
                .prepare(nativeRuntime())
                .executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{1.0d, -0.5d, 0.25d}, out.toDoubleArrayCopy(), 2.0e-3);
        Map<String, Object> attrs = operationStep(trace.steps(), "INV").metadata().attributes();
        assertEquals("CPU_ARRAY", attrs.get("actualCpuStorage"));
        assertEquals("native-kernel-unsupported:inv", attrs.get("nativeCpuFallbackReason"));
    }

    @Test
    void cpuNativeBfloat16PowKeepsArrayFallbackPolicy() {
        Tensor input = new Tensor(new double[]{1.0d, 4.0d, 9.0d}, new int[]{3}, null, "input", DataType.BFLOAT16);
        Tensor out = input.pow(1.5d);

        var trace = CompiledGraph.compile(out, compileConfig())
                .prepare(nativeRuntime())
                .executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(
                bf16Rounded((float) Math.pow(1.0f, 1.5f), (float) Math.pow(4.0f, 1.5f), (float) Math.pow(9.0f, 1.5f)),
                out.toDoubleArrayCopy(),
                2.0e-3
        );
        Map<String, Object> attrs = operationStep(trace.steps(), "POW").metadata().attributes();
        assertEquals("CPU_ARRAY", attrs.get("actualCpuStorage"));
        assertEquals("native-kernel-unsupported:pow", attrs.get("nativeCpuFallbackReason"));
    }

    @Test
    void cpuNativeBfloat16ExpTanhAndFastVariantsKeepArrayFallbackPolicy() {
        assertNativeBfloat16Fallback(
                "EXP",
                bfloat16Input().exp(),
                bf16Rounded((float) Math.exp(1.0f), (float) Math.exp(-2.0f), (float) Math.exp(4.0f)),
                "native-kernel-unsupported:exp"
        );
        assertNativeBfloat16Fallback(
                "TANH",
                bfloat16Input().tanh(),
                bf16Rounded((float) Math.tanh(1.0f), (float) Math.tanh(-2.0f), (float) Math.tanh(4.0f)),
                "native-kernel-unsupported:tanh"
        );
        assertNativeBfloat16Fallback(
                "FAST_EXP",
                bfloat16Input().fastExp(),
                bf16Rounded(
                        FastTranscendentals.fastExpF32(1.0f),
                        FastTranscendentals.fastExpF32(-2.0f),
                        FastTranscendentals.fastExpF32(4.0f)
                ),
                "native-kernel-unsupported:fast_exp"
        );
        assertNativeBfloat16Fallback(
                "FAST_TANH",
                bfloat16Input().fastTanh(),
                bf16Rounded(
                        FastTranscendentals.fastTanhF32(1.0f),
                        FastTranscendentals.fastTanhF32(-2.0f),
                        FastTranscendentals.fastTanhF32(4.0f)
                ),
                "native-kernel-unsupported:fast_tanh"
        );
    }

    @Test
    void cpuNativeNegStridedGraphInputFallsBackToArrayRouteAndComputes() {
        Tensor base = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Tensor out = base.permute(1, 0).neg();

        var trace = CompiledGraph.compile(out, compileConfig())
                .prepare(nativeRuntime())
                .executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{-1.0d, -4.0d, -2.0d, -5.0d, -3.0d, -6.0d}, out.toDoubleArrayCopy(), 1.0e-6);
        Map<String, Object> attrs = operationStep(trace.steps(), "NEG").metadata().attributes();
        assertEquals("CPU_ARRAY", attrs.get("actualCpuStorage"));
    }

    @Test
    void cpuArrayAbsHandlesStridedArrayInputThroughStorageView() {
        Tensor base = new Tensor(
                new float[]{-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Tensor out = base.permute(1, 0).abs();

        CompiledGraph.compile(out, compileConfig())
                .prepare(RuntimeConfig.inferenceDefaults())
                .execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{1.0d, 4.0d, 2.0d, 5.0d, 3.0d, 6.0d}, out.toDoubleArrayCopy(), 1.0e-6);
    }

    @Test
    void indexedSegmentLoopHandlesStridedSegmentInputView() {
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float[] out = new float[6];
        CpuStorageView inputView = CpuStorageView.segment(
                DataType.FLOAT32,
                MemorySegment.ofArray(input),
                new int[]{3, 2},
                new int[]{1, 3},
                0,
                6
        );
        CpuStorageView outView = CpuStorageView.segment(
                DataType.FLOAT32,
                MemorySegment.ofArray(out),
                new int[]{3, 2},
                new int[]{2, 1},
                0,
                6
        );
        var layout = StorageAwareUnaryElementwiseKernel.UnaryStorageLayout.from(inputView, outView);

        new CpuNegKernel().runIndexedSegmentF32(inputView.requireSegment(), outView.requireSegment(), layout, 0, 6);

        assertArrayEquals(new float[]{-1.0f, -4.0f, -2.0f, -5.0f, -3.0f, -6.0f}, out, 0.0f);
    }

    @Test
    void indexedMixedLoopHandlesSegmentInputAndArrayOutput() {
        float[] input = {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
        float[] out = new float[6];
        CpuStorageView inputView = CpuStorageView.segment(
                DataType.FLOAT32,
                MemorySegment.ofArray(input),
                new int[]{3, 2},
                new int[]{1, 3},
                0,
                6
        );
        CpuStorageView outView = CpuStorageView.array(
                DataType.FLOAT32,
                out,
                new int[]{3, 2},
                new int[]{2, 1},
                0,
                6
        );
        var layout = StorageAwareUnaryElementwiseKernel.UnaryStorageLayout.from(inputView, outView);

        new CpuReluKernel().runIndexedMixedF32(inputView, outView, layout, 0, 6);

        assertArrayEquals(new float[]{0.0f, 4.0f, 2.0f, 0.0f, 0.0f, 6.0f}, out, 0.0f);
    }

    @Test
    void migratedUnaryBfloat16SegmentLoopsPreserveOperationSemantics() {
        assertBfloat16Segment(
                new CpuInvKernel(),
                new float[]{1.0f, -2.0f, 4.0f},
                new double[]{1.0d, -0.5d, 0.25d}
        );
        assertBfloat16Segment(
                new CpuSqrtKernel(),
                new float[]{1.0f, 4.0f, 9.0f},
                new double[]{1.0d, 2.0d, 3.0d}
        );
        assertBfloat16Segment(
                new CpuSignKernel(),
                new float[]{-1.25f, 0.0f, 2.75f},
                new double[]{-1.0d, 0.0d, 1.0d}
        );
        assertBfloat16Segment(
                new CpuFloorKernel(),
                new float[]{-1.25f, 2.75f, 0.5f},
                new double[]{-2.0d, 2.0d, 0.0d}
        );
        assertBfloat16Segment(
                new CpuCeilKernel(),
                new float[]{-1.25f, 2.75f, 0.5f},
                new double[]{-1.0d, 3.0d, 1.0d}
        );
        assertBfloat16Segment(
                new CpuLogKernel(),
                new float[]{1.0f, 4.0f, 9.0f},
                new double[]{
                        TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits((float) Math.log(1.0f))),
                        TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits((float) Math.log(4.0f))),
                        TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits((float) Math.log(9.0f)))
                }
        );
        assertBfloat16Segment(
                new CpuExpKernel(),
                new float[]{-1.0f, 0.0f, 2.0f},
                bf16Rounded((float) Math.exp(-1.0f), (float) Math.exp(0.0f), (float) Math.exp(2.0f))
        );
        assertBfloat16Segment(
                new CpuTanhKernel(),
                new float[]{-1.0f, 0.0f, 2.0f},
                bf16Rounded((float) Math.tanh(-1.0f), (float) Math.tanh(0.0f), (float) Math.tanh(2.0f))
        );
        assertBfloat16Segment(
                new CpuSigmoidKernel(),
                new float[]{-1.0f, 0.0f, 2.0f},
                bf16Rounded(
                        1.0f / (1.0f + (float) Math.exp(1.0f)),
                        0.5f,
                        1.0f / (1.0f + (float) Math.exp(-2.0f))
                )
        );
        assertBfloat16Segment(
                new CpuErfKernel(),
                new float[]{-1.0f, 0.0f, 2.0f},
                bf16Rounded(SpecialFunctions.erf(-1.0f), SpecialFunctions.erf(0.0f), SpecialFunctions.erf(2.0f))
        );
        assertBfloat16Segment(
                new CpuFastExpKernel(),
                new float[]{-1.0f, 0.0f, 2.0f},
                bf16Rounded(
                        FastTranscendentals.fastExpF32(-1.0f),
                        FastTranscendentals.fastExpF32(0.0f),
                        FastTranscendentals.fastExpF32(2.0f)
                )
        );
        assertBfloat16Segment(
                new CpuFastTanhKernel(),
                new float[]{-1.0f, 0.0f, 2.0f},
                bf16Rounded(
                        FastTranscendentals.fastTanhF32(-1.0f),
                        FastTranscendentals.fastTanhF32(0.0f),
                        FastTranscendentals.fastTanhF32(2.0f)
                )
        );
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

    private static RuntimeConfig nativeApproxRuntime() {
        RuntimeConfig base = nativeRuntime();
        return new RuntimeConfig(
                base.kernel(),
                new ApproximationConfig(ApproxMode.ALWAYS, false),
                base.blas(),
                base.conv2d(),
                base.fused(),
                base.accelerator(),
                base.cpuStorageProfile(),
                base.nativeCpuFailurePolicy(),
                base.deviceTransferPolicy(),
                base.nativeCpuMemory(),
                base.bfloat16TrainingPolicy()
        );
    }

    private static Tensor input() {
        return new Tensor(new float[]{-1.0f, 2.0f, -3.5f}, new int[]{3}, null, "input", DataType.FLOAT32);
    }

    private static Tensor positiveInput() {
        return new Tensor(new float[]{1.0f, 4.0f, 9.0f}, new int[]{3}, null, "positiveInput", DataType.FLOAT32);
    }

    private static void assertNativeUnary(String opType, Tensor out, double[] expected) {
        assertNativeUnary(opType, out, expected, nativeRuntime());
    }

    private static void assertNativeUnary(String opType, Tensor out, double[] expected, RuntimeConfig runtimeConfig) {
        var trace = CompiledGraph.compile(out, compileConfig())
                .prepare(runtimeConfig)
                .executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 1.0e-6);
        Map<String, Object> attrs = operationStep(trace.steps(), opType).metadata().attributes();
        assertEquals("CPU_NATIVE", attrs.get("actualCpuStorage"));
        assertEquals("", attrs.get("nativeCpuFallbackReason"));
        assertEquals("SEGMENT_SCALAR", attrs.get("nativeCpuKernelFamily"));
    }

    private static void assertNativeBfloat16Fallback(
            String opType,
            Tensor out,
            double[] expected,
            String expectedReason
    ) {
        var trace = CompiledGraph.compile(out, compileConfig())
                .prepare(nativeRuntime())
                .executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 2.0e-3);
        Map<String, Object> attrs = operationStep(trace.steps(), opType).metadata().attributes();
        assertEquals("CPU_ARRAY", attrs.get("actualCpuStorage"));
        assertEquals(expectedReason, attrs.get("nativeCpuFallbackReason"));
    }

    private static ExecutionStepTrace operationStep(List<ExecutionStepTrace> steps, String opType) {
        return steps.stream()
                .filter(step -> opType.equals(step.opType()))
                .findFirst()
                .orElseThrow();
    }

    private static void assertBfloat16Segment(
            StorageAwareUnaryElementwiseKernel kernel,
            float[] input,
            double[] expected
    ) {
        short[] inputBits = bf16(input);
        short[] outBits = new short[input.length];

        kernel.runSegmentBF16(MemorySegment.ofArray(inputBits), MemorySegment.ofArray(outBits), 0, input.length);

        assertArrayEquals(expected, toDouble(outBits), 0.0d);
    }

    private static short[] bf16(float[] values) {
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

    private static double[] bf16Rounded(float... values) {
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(values[i]));
        }
        return out;
    }

    private static Tensor bfloat16Input() {
        return new Tensor(new double[]{1.0d, -2.0d, 4.0d}, new int[]{3}, null, "input", DataType.BFLOAT16);
    }
}
