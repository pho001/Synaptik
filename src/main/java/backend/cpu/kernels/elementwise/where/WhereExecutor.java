package backend.cpu.kernels.elementwise.where;

import backend.cpu.execution.CpuKernelContext;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

public final class WhereExecutor {
    private WhereExecutor() {}

    public static void execute(WhereElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("Where executor requires exactly 3 inputs.");
        }
        if (inputs.get(0).getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("Where executor requires BOOL condition input.");
        }
        WhereStorageLoops.execute(kernel, inputs, node, context);
    }
}
