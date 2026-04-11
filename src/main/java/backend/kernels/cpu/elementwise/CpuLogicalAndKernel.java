package backend.kernels.cpu.elementwise;

import backend.kernels.cpu.*;

import backend.kernels.cpu.elementwise.LogicalBinaryOp;
import backend.kernels.cpu.elementwise.LogicalExecutor;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuLogicalAndKernel implements CpuKernel {
    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        LogicalExecutor.executeBinary(LogicalBinaryOp.AND, inputs, node, context);
    }
}
