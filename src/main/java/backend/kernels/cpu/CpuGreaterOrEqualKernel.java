package backend.kernels.cpu;

import backend.kernels.cpu.elementwise.CompareExecutor;
import backend.kernels.cpu.elementwise.CompareOp;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuGreaterOrEqualKernel implements CpuKernel {
    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (node.getDataType() != tensor.DataType.BOOL) {
            throw new IllegalArgumentException("greaterOrEqual kernel requires BOOL output.");
        }
        CompareExecutor.execute(CompareOp.GE, inputs, node, context);
    }
}
