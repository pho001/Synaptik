package backend.cpu.kernels.linalg.matmul.f32;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.linalg.matmul.blas.MatMulBlasBackend;
import backend.cpu.kernels.linalg.matmul.exec.PreparedMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;
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
        float[] ad = a.getFloat32Data();
        float[] bd = b.getFloat32Data();
        float[] out = node.getFloat32Data();
        if (MatMulBlasBackend.tryBatchedBlasF32(ad, as, bd, bs, out, node.getShapeUnsafe(), m, n, k)) {
            return;
        }
        Arrays.fill(out, 0.0f);
        F32MatMulJavaBackend.run(ad, as, bd, bs, out, node.getShapeUnsafe(), hints);
    }
}
