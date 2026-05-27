package backend.cpu.provider.linalg.matmul.bf16;

import tensor.TensorInternalAccess;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.execution.CpuNodeWorkspace;
import backend.cpu.provider.linalg.matmul.blas.MatMulBlasBackend;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
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
        recordBlasSymbol("cblas_sbgemm");
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
        if (!MatMulBlasBackend.tryBlasBF16(ad, bd, TensorInternalAccess.bfloat16Data(node), tmp, m, n, k)) {
            return false;
        }
        recordBlasSymbol("cblas_bgemm");
        return true;
    }
}
