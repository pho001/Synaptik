package backend.cpu.kernels.linalg.matmul.f64;

import backend.cpu.kernels.linalg.matmul.common.PackedLinearWeightCache;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;

public final class F64MatMulJavaBackend {
    private F64MatMulJavaBackend() {
    }

    public static void run(double[] a, int[] aShape, double[] b, int[] bShape, double[] out, int[] outShape, ResolvedMatMulHints hints) {
        F64MatMulDispatch.run(a, aShape, b, bShape, out, outShape, hints);
    }

    public static void runRightTransposed(
            double[] a, int[] aShape, double[] b, int[] bShape, double[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        F64MatMulDispatch.runRightTransposed(a, aShape, b, bShape, out, outShape, hints);
    }

    public static void runLeftTransposed(
            double[] a, int[] aShape, double[] b, int[] bShape, double[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        F64MatMulDispatch.runLeftTransposed(a, aShape, b, bShape, out, outShape, hints);
    }

    public static void runPacked(
            double[] a, int[] aShape, PackedLinearWeightCache.F64PackedWeights packedB,
            double[] out, int[] outShape, ResolvedMatMulHints hints
    ) {
        F64MatMulDispatch.runPacked(a, aShape, packedB, out, outShape, hints);
    }
}
