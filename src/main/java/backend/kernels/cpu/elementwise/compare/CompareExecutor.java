package backend.kernels.cpu.elementwise.compare;

import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.elementwise.ElementwiseLoops;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

public final class CompareExecutor {
    private CompareExecutor() {}

    public static void execute(CompareElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("Compare executor requires exactly 2 inputs.");
        }
        if (node.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("Compare executor requires BOOL output.");
        }
        ElementwiseLoops.runCompare(kernel, inputs.get(0), inputs.get(1), node, context);
    }
}
