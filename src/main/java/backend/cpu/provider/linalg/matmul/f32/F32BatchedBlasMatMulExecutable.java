package backend.cpu.provider.linalg.matmul.f32;

import tensor.TensorInternalAccess;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.linalg.matmul.f32.F32MatMulJavaBackend;
import backend.cpu.provider.linalg.matmul.blas.MatMulBlasBackend;
import backend.cpu.provider.linalg.matmul.PreparedMatMulExecutable;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
import tensor.Tensor;

import java.util.Arrays;

public final class F32BatchedBlasMatMulExecutable implements PreparedMatMulExecutable {
    private final ResolvedMatMulHints hints;

    public F32BatchedBlasMatMulExecutable(ResolvedMatMulHints hints) {
        this.hints = hints;
    }

    @Override
    public void execute(Tensor a, Tensor b, Tensor node, CpuKernelContext context) {
        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int n = bs[bs.length - 1];
        float[] ad = TensorInternalAccess.float32Data(a);
        float[] bd = TensorInternalAccess.float32Data(b);
        float[] out = TensorInternalAccess.float32Data(node);
        if (MatMulBlasBackend.tryBatchedBlasF32(ad, as, bd, bs, out, node.getShapeUnsafe(), m, n, k)) {
            return;
        }
        Arrays.fill(out, 0.0f);
        F32MatMulJavaBackend.run(ad, as, bd, bs, out, node.getShapeUnsafe(), hints);
    }
}
