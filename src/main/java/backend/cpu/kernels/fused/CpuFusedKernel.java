package backend.cpu.kernels.fused;

import backend.cpu.execution.CpuKernelContext;

import backend.cpu.kernels.*;
import backend.cpu.plan.CpuKernelCostClass;

import backend.cpu.fused.plan.FusedOperation;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public class CpuFusedKernel extends TypedCpuKernel {
    @Override
    public CpuKernelCostClass costClass(Operation op) {
        if (op instanceof FusedOperation fused) {
            return FusedExecutor.costClass(fused);
        }
        return super.costClass(op);
    }

    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof FusedOperation fused)) {
            throw new IllegalStateException("CpuFusedKernel requires FusedOperation descriptor");
        }
        FusedExecutor.execute(fused, inputs, node, context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) { forwardF64(op, inputs, node, context); }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) { forwardF64(op, inputs, node, context); }

    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        forwardF64(op, inputs, node, context);
    }
}
