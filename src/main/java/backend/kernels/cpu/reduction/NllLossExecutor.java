package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuKernelContext;
import operations.nllLoss;
import tensor.Tensor;

public final class NllLossExecutor {
    public void execute(nllLoss op, Tensor logProbs, Tensor targets, Tensor node, CpuKernelContext context) {
        validate(op, logProbs, targets, node, context);
        NllLossLoops.execute(logProbs, targets, node, op.getClassDimension(), context);
    }

    public void executeF32(nllLoss op, Tensor logProbs, Tensor targets, Tensor node, CpuKernelContext context) {
        validate(op, logProbs, targets, node, context);
        NllLossLoops.executeF32(logProbs, targets, node, op.getClassDimension(), context);
    }

    public void executeF16(nllLoss op, Tensor logProbs, Tensor targets, Tensor node, CpuKernelContext context) {
        validate(op, logProbs, targets, node, context);
        NllLossLoops.executeF16(logProbs, targets, node, op.getClassDimension(), context);
    }

    private static void validate(nllLoss op, Tensor logProbs, Tensor targets, Tensor node, CpuKernelContext context) {
        if (op == null || logProbs == null || targets == null || node == null || context == null) {
            throw new IllegalArgumentException("nllLoss execution arguments cannot be null");
        }
    }
}
