package backend.kernels.cpu.fused;

import backend.kernels.cpu.*;

import backend.kernels.cpu.fused.FusedExecutionOptions;
import backend.cpu.fused.exec.PreparedFusedExecutable;
import backend.cpu.fused.plan.FusedOperation;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public class CpuFusedKernel implements CpuKernel {
    @Override
    public CpuKernelCostClass costClass(Operation op) {
        if (op instanceof FusedOperation fused) {
            return FusedExecutor.costClass(fused);
        }
        return CpuKernel.super.costClass(op);
    }

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof FusedOperation fused)) {
            throw new IllegalStateException("CpuFusedKernel requires FusedOperation descriptor");
        }
        FusedExecutor.execute(fused, inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) { forwardF64(op, inputs, node, context); }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) { forwardF64(op, inputs, node, context); }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        forwardF64(op, inputs, node, context);
    }
}
