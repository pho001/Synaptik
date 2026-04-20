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
        float[] tmp = (hints.useBlas() || hints.useBatchedBlas()) && context.cpuWorkspace() != null
                ? context.cpuWorkspace().requireFloatWorkspace()
                : null;
        if (context.publishFloatContinuation()) {
            if (as.length == 2 && bs.length == 2 && hints.useBlas() && MatMulBlasBackend.tryBlasBF16ToFloat(ad, bd, tmp, m, n, k)) {
                context.cpuWorkspace().publishFloatContinuation(m * n);
                return;
            }
            if (hints.useBatchedBlas() && MatMulBlasBackend.tryBatchedBlasBF16ToFloat(ad, as, bd, bs, tmp, node.getShapeUnsafe(), m, n, k)) {
                context.cpuWorkspace().publishFloatContinuation(out.length);
                return;
            }
        }
        if (as.length == 2 && bs.length == 2 && hints.useBlas() && MatMulBlasBackend.tryBlasBF16(ad, bd, out, tmp, m, n, k)) {
            return;
        }
        if (hints.useBatchedBlas() && MatMulBlasBackend.tryBatchedBlasBF16(ad, as, bd, bs, out, tmp, node.getShapeUnsafe(), m, n, k)) {
            return;
        }
        MatMulJavaBackend.runBF16(ad, as, bd, bs, out, node.getShapeUnsafe(), hints);
    }

    static ResolvedMatMulHints requireHints(CpuKernelContext context) {
        ResolvedMatMulHints hints = context.matMulHints();
        if (hints == null) {
            throw new IllegalStateException("Missing ResolvedMatMulHints for matmul execution");
        }
        return hints;
    }
}
