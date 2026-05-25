package backend.cpu.kernels.elementwise.logical;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuLogicalOrKernel extends TypedCpuKernel implements LogicalBinaryElementwiseKernel {
    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        LogicalExecutor.executeBinary(this, inputs, node, context);
    }

    @Override
    public byte apply(byte left, byte right) {
        return (left != 0 || right != 0) ? (byte) 1 : (byte) 0;
    }
}
