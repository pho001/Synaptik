package backend.cpu.kernels.elementwise.where;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.elementwise.ElementwiseLoops;
import backend.cpu.nativecpu.NativeCpuElementwiseExecutor;
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
        if (NativeCpuElementwiseExecutor.tryRunWhere(kernel, inputs, node, context)) {
            return;
        }
        switch (node.getDataType()) {
            case FLOAT64, FLOAT32, BFLOAT16 -> ElementwiseLoops.runWhere(kernel, inputs.get(0), inputs.get(1), inputs.get(2), node, context);
            case INT32, BOOL -> throw new UnsupportedOperationException("Where only supports floating output tensors");
        }
    }
}
