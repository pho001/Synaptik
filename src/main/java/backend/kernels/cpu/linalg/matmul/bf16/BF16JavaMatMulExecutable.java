package backend.kernels.cpu.linalg.matmul.bf16;

import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.linalg.matmul.plan.ResolvedMatMulHints;
import tensor.Tensor;

public final class BF16JavaMatMulExecutable extends AbstractBF16MatMulExecutable {
    public BF16JavaMatMulExecutable(ResolvedMatMulHints hints, boolean publishFloatContinuation) {
        super(hints, publishFloatContinuation);
    }

    @Override
    protected boolean allowPackedAndContinuationFastPaths() {
        return true;
    }

    @Override
    protected boolean tryBackendToFloat(Tensor node, CpuKernelContext context, int[] as, int[] bs, short[] ad, short[] bd) {
        return false;
    }

    @Override
    protected boolean tryBackendToBFloat16(Tensor node, CpuKernelContext context, int[] as, int[] bs, short[] ad, short[] bd) {
        return false;
    }
}
