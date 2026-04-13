package backend.kernels.cpu.elementwise.logical;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuLogicalOrKernel implements CpuKernel, LogicalBinaryElementwiseKernel {
    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        LogicalExecutor.executeBinary(this, inputs, node, context);
    }

    @Override
    public byte apply(byte left, byte right) {
        return (left != 0 || right != 0) ? (byte) 1 : (byte) 0;
    }
}
