package backend.kernels.cpu.linalg;

import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.linalg.matmul.plan.ResolvedMatMulHints;
import tensor.Tensor;

import java.util.Arrays;

final class MatMulExecutor {
    private MatMulExecutor() {}

    static void forwardF64(Tensor a, Tensor b, Tensor node, CpuKernelContext context) {
        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int n = bs[bs.length - 1];
        double[] ad = a.getFloat64Data();
        double[] bd = b.getFloat64Data();
        double[] out = node.getFloat64Data();
        ResolvedMatMulHints hints = requireHints(context);
        if (as.length == 2 && bs.length == 2 && hints.useBlas() && MatMulBlasBackend.tryBlasF64(ad, bd, out, m, n, k)) {
            return;
        }
        if (hints.useBatchedBlas() && MatMulBlasBackend.tryBatchedBlasF64(ad, as, bd, bs, out, node.getShapeUnsafe(), m, n, k)) {
            return;
        }
        Arrays.fill(out, 0.0d);
        MatMulJavaBackend.runF64(ad, as, bd, bs, out, node.getShapeUnsafe(), hints);
    }

    static void forwardF32(Tensor a, Tensor b, Tensor node, CpuKernelContext context) {
        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int n = bs[bs.length - 1];
        float[] ad = a.getFloat32Data();
        float[] bd = b.getFloat32Data();
        float[] out = node.getFloat32Data();
        ResolvedMatMulHints hints = requireHints(context);
        if (as.length == 2 && bs.length == 2 && hints.useBlas() && MatMulBlasBackend.tryBlasF32(ad, bd, out, m, n, k)) {
            return;
        }
        if (hints.useBatchedBlas() && MatMulBlasBackend.tryBatchedBlasF32(ad, as, bd, bs, out, node.getShapeUnsafe(), m, n, k)) {
            return;
        }
        Arrays.fill(out, 0.0f);
        MatMulJavaBackend.runF32(ad, as, bd, bs, out, node.getShapeUnsafe(), hints);
    }

    static void forwardBF16(Tensor a, Tensor b, Tensor node, CpuKernelContext context) {
        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int n = bs[bs.length - 1];
        short[] ad = a.getBFloat16Data();
        short[] bd = b.getBFloat16Data();
        short[] out = node.getBFloat16Data();
        ResolvedMatMulHints hints = requireHints(context);
        float[] leftContinuation = context.inputFloatContinuation(0, a.getFlatDataSize());
        float[] rightContinuation = context.inputFloatContinuation(1, b.getFlatDataSize());
        float[] tmp = (hints.useBlas() || hints.useBatchedBlas()) && context.cpuWorkspace() != null
                ? context.cpuWorkspace().requireFloatWorkspace()
                : null;
        if (context.publishFloatContinuation()) {
            if (rightContinuation == null && tryPackedMatMulToFloatBF16(a, b, node, context, hints)) {
                context.cpuWorkspace().publishFloatContinuation(node.getFlatDataSize());
                return;
            }
            if (tryContinuationMatMulToFloatBF16(as, bs, node, context, hints, ad, bd, leftContinuation, rightContinuation)) {
                return;
            }
            if (as.length == 2 && bs.length == 2 && hints.useBlas() && MatMulBlasBackend.tryBlasBF16ToFloat(ad, bd, tmp, m, n, k)) {
                context.cpuWorkspace().publishFloatContinuation(m * n);
                return;
            }
            if (hints.useBatchedBlas() && MatMulBlasBackend.tryBatchedBlasBF16ToFloat(ad, as, bd, bs, tmp, node.getShapeUnsafe(), m, n, k)) {
                context.cpuWorkspace().publishFloatContinuation(out.length);
                return;
            }
            if (context.cpuWorkspace() != null) {
                float[] continuation = context.cpuWorkspace().requireFloatWorkspace();
                MatMulJavaBackend.runBF16ToFloat(ad, as, bd, bs, continuation, node.getShapeUnsafe(), hints);
                context.cpuWorkspace().publishFloatContinuation(out.length);
                return;
            }
        }
        if (rightContinuation == null && tryPackedMatMulBF16(a, b, node, context, hints)) {
            return;
        }
        if (tryContinuationMatMulBF16(as, bs, node, hints, ad, bd, leftContinuation, rightContinuation)) {
            return;
        }
        if (as.length == 2 && bs.length == 2 && hints.useBlas() && MatMulBlasBackend.tryBlasBF16(ad, bd, out, tmp, m, n, k)) {
            return;
        }
        if (hints.useBatchedBlas() && MatMulBlasBackend.tryBatchedBlasBF16(ad, as, bd, bs, out, tmp, node.getShapeUnsafe(), m, n, k)) {
            return;
        }
        MatMulJavaBackend.runBF16(ad, as, bd, bs, out, node.getShapeUnsafe(), hints);
    }

    private static boolean tryPackedMatMulToFloatBF16(
            Tensor a,
            Tensor b,
            Tensor node,
            CpuKernelContext context,
            ResolvedMatMulHints hints
    ) {
        if (context == null || context.cpuWorkspace() == null || context.cpuWorkspace().packedLinearWeightCache() == null) {
            return false;
        }
        if (hints.useBlas() || hints.useBatchedBlas()) {
            return false;
        }
        if (b.getShapeUnsafe().length != 2 || !b.isContiguous()) {
            return false;
        }
        PackedLinearWeightCache.BF16PackedWeights packed = context.cpuWorkspace().packedLinearWeightCache().requireBF16(b, hints);
        if (packed == null) {
            return false;
        }
        float[] out = context.cpuWorkspace().requireFloatWorkspace();
        float[] leftContinuation = context.inputFloatContinuation(0, a.getFlatDataSize());
        if (leftContinuation != null) {
            MatMulJavaBackend.runPackedF32(leftContinuation, a.getShapeUnsafe(), packed, out, node.getShapeUnsafe(), hints);
        } else {
            MatMulJavaBackend.runPackedBF16ToFloat(a.getBFloat16Data(), a.getShapeUnsafe(), packed, out, node.getShapeUnsafe(), hints);
        }
        return true;
    }

    private static boolean tryContinuationMatMulToFloatBF16(
            int[] as,
            int[] bs,
            Tensor node,
            CpuKernelContext context,
            ResolvedMatMulHints hints,
            short[] ad,
            short[] bd,
            float[] leftContinuation,
            float[] rightContinuation
    ) {
        if (context == null || context.cpuWorkspace() == null || hints.useBlas() || hints.useBatchedBlas()) {
            return false;
        }
        float[] out = context.cpuWorkspace().requireFloatWorkspace();
        if (leftContinuation != null && rightContinuation != null) {
            MatMulJavaBackend.runF32(leftContinuation, as, rightContinuation, bs, out, node.getShapeUnsafe(), hints);
            context.cpuWorkspace().publishFloatContinuation(node.getFlatDataSize());
            return true;
        }
        if (leftContinuation != null) {
            MatMulJavaBackend.runF32LeftBF16RightToFloat(leftContinuation, as, bd, bs, out, node.getShapeUnsafe(), hints);
            context.cpuWorkspace().publishFloatContinuation(node.getFlatDataSize());
            return true;
        }
        return false;
    }

    private static boolean tryContinuationMatMulBF16(
            int[] as,
            int[] bs,
            Tensor node,
            ResolvedMatMulHints hints,
            short[] ad,
            short[] bd,
            float[] leftContinuation,
            float[] rightContinuation
    ) {
        if (hints.useBlas() || hints.useBatchedBlas()) {
            return false;
        }
        if (leftContinuation != null && rightContinuation != null) {
            MatMulJavaBackend.runF32ToBF16(leftContinuation, as, rightContinuation, bs, node.getBFloat16Data(), node.getShapeUnsafe(), hints);
            return true;
        }
        if (leftContinuation != null) {
            MatMulJavaBackend.runF32LeftBF16RightToBF16(leftContinuation, as, bd, bs, node.getBFloat16Data(), node.getShapeUnsafe(), hints);
            return true;
        }
        return false;
    }

    private static boolean tryPackedMatMulBF16(
            Tensor a,
            Tensor b,
            Tensor node,
            CpuKernelContext context,
            ResolvedMatMulHints hints
    ) {
        if (context == null || context.cpuWorkspace() == null || context.cpuWorkspace().packedLinearWeightCache() == null) {
            return false;
        }
        if (hints.useBlas() || hints.useBatchedBlas()) {
            return false;
        }
        if (b.getShapeUnsafe().length != 2 || !b.isContiguous()) {
            return false;
        }
        PackedLinearWeightCache.BF16PackedWeights packed = context.cpuWorkspace().packedLinearWeightCache().requireBF16(b, hints);
        if (packed == null) {
            return false;
        }
        float[] leftContinuation = context.inputFloatContinuation(0, a.getFlatDataSize());
        if (leftContinuation != null) {
            MatMulJavaBackend.runPackedF32ToBF16(leftContinuation, a.getShapeUnsafe(), packed, node.getBFloat16Data(), node.getShapeUnsafe(), hints);
            return true;
        }
        MatMulJavaBackend.runPackedBF16(a.getBFloat16Data(), a.getShapeUnsafe(), packed, node.getBFloat16Data(), node.getShapeUnsafe(), hints);
        return true;
    }

    static ResolvedMatMulHints requireHints(CpuKernelContext context) {
        ResolvedMatMulHints hints = context.matMulHints();
        if (hints == null) {
            throw new IllegalStateException("Missing ResolvedMatMulHints for matmul execution");
        }
        return hints;
    }
}
