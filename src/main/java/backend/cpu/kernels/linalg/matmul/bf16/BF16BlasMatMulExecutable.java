package backend.cpu.kernels.linalg.matmul.bf16;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuNodeWorkspace;
import backend.cpu.kernels.linalg.matmul.blas.MatMulBlasBackend;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;
import tensor.Tensor;

public final class BF16BlasMatMulExecutable extends AbstractBF16MatMulExecutable {
    public BF16BlasMatMulExecutable(ResolvedMatMulHints hints, boolean publishFloatContinuation) {
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
        if (!MatMulBlasBackend.tryBlasBF16ToFloat(ad, bd, tmp, m, n, k)) {
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
        return MatMulBlasBackend.tryBlasBF16(ad, bd, node.getBFloat16Data(), tmp, m, n, k);
    }
}
