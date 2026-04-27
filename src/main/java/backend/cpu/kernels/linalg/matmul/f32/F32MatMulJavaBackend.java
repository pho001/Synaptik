package backend.cpu.kernels.linalg.matmul.f32;

import backend.cpu.kernels.linalg.matmul.common.PackedLinearWeightCache;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;

public final class F32MatMulJavaBackend {
    private F32MatMulJavaBackend() {
    }

    public static void run(float[] a, int[] aShape, float[] b, int[] bShape, float[] out, int[] outShape, ResolvedMatMulHints hints) {
        F32MatMulDispatch.run(a, aShape, b, bShape, out, outShape, hints);
    }

    public static void runRightTransposed(
            float[] a, int[] aShape, float[] b, int[] bShape, float[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        F32MatMulDispatch.runRightTransposed(a, aShape, b, bShape, out, outShape, hints);
    }

    public static void runLeftTransposed(
            float[] a, int[] aShape, float[] b, int[] bShape, float[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        F32MatMulDispatch.runLeftTransposed(a, aShape, b, bShape, out, outShape, hints);
    }

    public static void runPacked(
            float[] a, int[] aShape, PackedLinearWeightCache.F32PackedWeights packedB,
            float[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        F32MatMulDispatch.runPacked(a, aShape, packedB, out, outShape, hints);
    }

    public static void runPacked(
            float[] a, int[] aShape, PackedLinearWeightCache.PackedFloatPanels packedB,
            float[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        F32MatMulDispatch.runPacked(a, aShape, packedB, out, outShape, hints);
    }
}
