package backend.kernels.cpu.elementwise.where;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuWhereKernel implements CpuKernel, WhereElementwiseKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        WhereExecutor.execute(this, inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        WhereExecutor.execute(this, inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        WhereExecutor.execute(this, inputs, node, context);
    }

    @Override
    public double applyF64(byte condition, double ifTrue, double ifFalse) {
        return condition != 0 ? ifTrue : ifFalse;
    }

    @Override
    public float applyF32(byte condition, float ifTrue, float ifFalse) {
        return condition != 0 ? ifTrue : ifFalse;
    }

    @Override
    public float applyBF16(byte condition, float ifTrue, float ifFalse) {
        return condition != 0 ? ifTrue : ifFalse;
    }
}
