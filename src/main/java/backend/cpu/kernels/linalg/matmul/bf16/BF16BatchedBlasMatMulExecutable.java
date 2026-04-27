package backend.cpu.kernels.linalg.matmul.bf16;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuNodeWorkspace;
import backend.cpu.kernels.linalg.matmul.blas.MatMulBlasBackend;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;
import tensor.Tensor;

public final class BF16BatchedBlasMatMulExecutable extends AbstractBF16MatMulExecutable {
    public BF16BatchedBlasMatMulExecutable(ResolvedMatMulHints hints, boolean publishFloatContinuation) {
        super(hints, publishFloatContinuation);
    }

    @Override
    protected boolean allowPackedAndContinuationFastPaths() {
        return false;
    }

    @Override
    protected boolean tryBackendToFloat(Tensor node, CpuKernelContext context, int[] as, int[] bs, short[] ad, short[] bd) {
        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int n = bs[bs.length - 1];
        CpuNodeWorkspace workspace = context.cpuWorkspace();
        float[] tmp = workspace == null ? null : workspace.requireFloatWorkspace();
        if (!MatMulBlasBackend.tryBatchedBlasBF16ToFloat(ad, as, bd, bs, tmp, node.getShapeUnsafe(), m, n, k)) {
            return false;
        }
        if (workspace != null) {
            workspace.publishFloatContinuation(node.getFlatDataSize());
        }
        return true;
    }

    @Override
    protected boolean tryBackendToBFloat16(Tensor node, CpuKernelContext context, int[] as, int[] bs, short[] ad, short[] bd) {
        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int n = bs[bs.length - 1];
        CpuNodeWorkspace workspace = context.cpuWorkspace();
        float[] tmp = workspace == null ? null : workspace.requireFloatWorkspace();
        return MatMulBlasBackend.tryBatchedBlasBF16(ad, as, bd, bs, node.getBFloat16Data(), tmp, node.getShapeUnsafe(), m, n, k);
    }
}
