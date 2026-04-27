package backend.cpu.kernels.elementwise.logical;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuLogicalNotKernel implements CpuKernel, LogicalUnaryElementwiseKernel {
    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        LogicalExecutor.executeUnary(this, inputs, node, context);
    }

    @Override
    public byte apply(byte value) {
        return value == 0 ? (byte) 1 : (byte) 0;
    }
}
