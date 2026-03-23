package Backend.kernels.cpu.reduction;

import Backend.kernels.cpu.CpuExecutionConfig;
import Operations.sum;
import Tensor.Tensor;

public final class SumExecutor {
    public void execute(sum op, Tensor input, Tensor node, CpuExecutionConfig config) {
        if (op == null || input == null || node == null || config == null) {
            throw new IllegalArgumentException("sum execution arguments cannot be null");
        }
        SumLoops.execute(input, node, op.getDimension(), config);
    }

    public void executeF32(sum op, Tensor input, Tensor node, CpuExecutionConfig config) {
        if (op == null || input == null || node == null || config == null) {
            throw new IllegalArgumentException("sum execution arguments cannot be null");
        }
        SumLoops.executeF32(input, node, op.getDimension(), config);
    }

    public void executeF16(sum op, Tensor input, Tensor node, CpuExecutionConfig config) {
        if (op == null || input == null || node == null || config == null) {
            throw new IllegalArgumentException("sum execution arguments cannot be null");
        }
        SumLoops.executeF16(input, node, op.getDimension(), config);
    }
}
