package backend.cpu.kernels.elementwise.logical;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuLogicalNotKernel extends TypedCpuKernel implements LogicalUnaryElementwiseKernel {
    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        LogicalExecutor.executeUnary(this, inputs, node, context);
    }

    @Override
    public byte apply(byte value) {
        return value == 0 ? (byte) 1 : (byte) 0;
    }
}
