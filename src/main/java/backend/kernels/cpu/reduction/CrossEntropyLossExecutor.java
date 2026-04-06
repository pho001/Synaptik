package backend.kernels.cpu.reduction;

import backend.kernels.cpu.CpuKernelContext;
import operations.crossEntropyLoss;
import tensor.Tensor;

public final class CrossEntropyLossExecutor {
    public void execute(crossEntropyLoss op, Tensor logits, Tensor targets, Tensor node, CpuKernelContext context) {
        validate(op, logits, targets, node, context);
        CrossEntropyLossLoops.execute(logits, targets, node, op.getClassDimension(), context);
    }

    public void executeF32(crossEntropyLoss op, Tensor logits, Tensor targets, Tensor node, CpuKernelContext context) {
        validate(op, logits, targets, node, context);
        CrossEntropyLossLoops.executeF32(logits, targets, node, op.getClassDimension(), context);
    }

    public void executeBF16(crossEntropyLoss op, Tensor logits, Tensor targets, Tensor node, CpuKernelContext context) {
        validate(op, logits, targets, node, context);
        CrossEntropyLossLoops.executeBF16(logits, targets, node, op.getClassDimension(), context);
    }

    private static void validate(crossEntropyLoss op, Tensor logits, Tensor targets, Tensor node, CpuKernelContext context) {
        if (op == null || logits == null || targets == null || node == null || context == null) {
            throw new IllegalArgumentException("crossEntropyLoss execution arguments cannot be null");
        }
    }
}
