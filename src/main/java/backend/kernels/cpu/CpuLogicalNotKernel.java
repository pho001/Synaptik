package backend.kernels.cpu;

import backend.kernels.cpu.elementwise.LogicalExecutor;
import backend.kernels.cpu.elementwise.LogicalUnaryOp;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuLogicalNotKernel implements CpuKernel {
    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        LogicalExecutor.executeUnary(LogicalUnaryOp.NOT, inputs, node, context);
    }
}
