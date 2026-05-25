package backend.cpu.kernels.elementwise.logical;

import backend.cpu.execution.CpuKernelContext;
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
        LogicalBoolStorageLoops.executeBinary(kernel, inputs, node, context);
    }

    public static void executeUnary(LogicalUnaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("Logical unary executor requires exactly 1 input.");
        }
        if (inputs.get(0).getDataType() != DataType.BOOL || node.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("Logical unary executor requires BOOL input and BOOL output.");
        }
        LogicalBoolStorageLoops.executeUnary(kernel, inputs, node, context);
    }
}
