package backend.cpu.kernels.elementwise.where;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuWhereKernel extends TypedCpuKernel implements WhereElementwiseKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        WhereExecutor.execute(this, inputs, node, context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        WhereExecutor.execute(this, inputs, node, context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
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
