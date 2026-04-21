package backend.kernels.cpu.linalg.matmul.bf16;

import backend.kernels.cpu.linalg.matmul.common.PackedLinearWeightCache;
import backend.kernels.cpu.linalg.matmul.plan.ResolvedMatMulHints;

public final class BF16MatMulJavaBackend {
    private BF16MatMulJavaBackend() {
    }

    public static void runPacked(
            short[] a, int[] aShape, PackedLinearWeightCache.BF16PackedWeights packedB,
            short[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        BF16MatMulDispatch.runPacked(a, aShape, packedB, out, outShape, hints);
    }

    public static void runPackedF32ToBF16(
            float[] a, int[] aShape, PackedLinearWeightCache.PackedFloatPanels packedB,
            short[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        BF16MatMulDispatch.runPackedF32ToBF16(a, aShape, packedB, out, outShape, hints);
    }

    public static void runPackedToFloat(
            short[] a, int[] aShape, PackedLinearWeightCache.BF16PackedWeights packedB,
            float[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        BF16MatMulDispatch.runPackedToFloat(a, aShape, packedB, out, outShape, hints);
    }

    public static void run(short[] a, int[] aShape, short[] b, int[] bShape, short[] out, int[] outShape, ResolvedMatMulHints hints) {
        BF16MatMulDispatch.run(a, aShape, b, bShape, out, outShape, hints);
    }

    public static void runToFloat(short[] a, int[] aShape, short[] b, int[] bShape, float[] out, int[] outShape, ResolvedMatMulHints hints) {
        BF16MatMulDispatch.runToFloat(a, aShape, b, bShape, out, outShape, hints);
    }

    public static void runF32ToBF16(float[] a, int[] aShape, float[] b, int[] bShape, short[] out, int[] outShape, ResolvedMatMulHints hints) {
        BF16MatMulDispatch.runF32ToBF16(a, aShape, b, bShape, out, outShape, hints);
    }

    public static void runF32LeftBF16RightToBF16(
            float[] a, int[] aShape, short[] b, int[] bShape, short[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        BF16MatMulDispatch.runF32LeftBF16RightToBF16(a, aShape, b, bShape, out, outShape, hints);
    }

    public static void runF32LeftBF16RightToFloat(
            float[] a, int[] aShape, short[] b, int[] bShape, float[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        BF16MatMulDispatch.runF32LeftBF16RightToFloat(a, aShape, b, bShape, out, outShape, hints);
    }
}
