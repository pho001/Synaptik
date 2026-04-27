package backend.cpu.kernels.linalg.matmul.f64;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.linalg.matmul.blas.MatMulBlasBackend;
import backend.cpu.kernels.linalg.matmul.exec.PreparedMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;
import tensor.Tensor;

import java.util.Arrays;

public final class F64BatchedBlasMatMulExecutable implements PreparedMatMulExecutable {
    private final ResolvedMatMulHints hints;

    public F64BatchedBlasMatMulExecutable(ResolvedMatMulHints hints) {
        this.hints = hints;
    }

    @Override
    public void execute(Tensor a, Tensor b, Tensor node, CpuKernelContext context) {
        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int n = bs[bs.length - 1];
        double[] ad = a.getFloat64Data();
        double[] bd = b.getFloat64Data();
        double[] out = node.getFloat64Data();
        if (MatMulBlasBackend.tryBatchedBlasF64(ad, as, bd, bs, out, node.getShapeUnsafe(), m, n, k)) {
            return;
        }
        Arrays.fill(out, 0.0d);
        F64MatMulJavaBackend.run(ad, as, bd, bs, out, node.getShapeUnsafe(), hints);
    }
}
