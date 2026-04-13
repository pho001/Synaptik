package backend.kernels.cpu.elementwise.logical;

import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.elementwise.ElementwiseLoops;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

public final class LogicalExecutor {
    private LogicalExecutor() {}

    public static void executeBinary(LogicalBinaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("Logical binary executor requires exactly 2 inputs.");
        }
        if (inputs.get(0).getDataType() != DataType.BOOL || inputs.get(1).getDataType() != DataType.BOOL || node.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("Logical binary executor requires BOOL inputs and BOOL output.");
        }
        ElementwiseLoops.runLogicalBinary(kernel, inputs.get(0), inputs.get(1), node, context);
    }

    public static void executeUnary(LogicalUnaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("Logical unary executor requires exactly 1 input.");
        }
        if (inputs.get(0).getDataType() != DataType.BOOL || node.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("Logical unary executor requires BOOL input and BOOL output.");
        }
        ElementwiseLoops.runLogicalUnary(kernel, inputs.get(0), node, context);
    }
}
