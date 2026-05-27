package backend.cpu.kernels.linalg.matmul.f64;

import tensor.TensorInternalAccess;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.linalg.matmul.blas.MatMulBlasBackend;
import backend.cpu.provider.linalg.matmul.PreparedMatMulExecutable;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
import tensor.Tensor;

import java.util.Arrays;

public final class F64BlasMatMulExecutable implements PreparedMatMulExecutable {
    private final ResolvedMatMulHints hints;

    public F64BlasMatMulExecutable(ResolvedMatMulHints hints) {
        this.hints = hints;
    }

    @Override
    public void execute(Tensor a, Tensor b, Tensor node, CpuKernelContext context) {
        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int n = bs[bs.length - 1];
        double[] ad = TensorInternalAccess.float64Data(a);
        double[] bd = TensorInternalAccess.float64Data(b);
        double[] out = TensorInternalAccess.float64Data(node);
        if (MatMulBlasBackend.tryBlasF64(ad, bd, out, m, n, k)) {
            return;
        }
        Arrays.fill(out, 0.0d);
        F64MatMulJavaBackend.run(ad, as, bd, bs, out, node.getShapeUnsafe(), hints);
    }
}
