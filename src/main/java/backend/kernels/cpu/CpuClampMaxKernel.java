package backend.kernels.cpu;

import backend.kernels.cpu.elementwise.ElementwiseUnaryExecutor;
import backend.kernels.cpu.elementwise.ScalarUnaryOp;
import operations.Operation;
import operations.clampMax;
import tensor.Tensor;

import java.util.List;

public final class CpuClampMaxKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        clampMax clamp = (clampMax) op;
        ElementwiseUnaryExecutor.execute(ScalarUnaryOp.CLAMP_MAX, clamp.getMaxValue(), clamp.getMaxValueF32(), inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        clampMax clamp = (clampMax) op;
        ElementwiseUnaryExecutor.execute(ScalarUnaryOp.CLAMP_MAX, clamp.getMaxValue(), clamp.getMaxValueF32(), inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        clampMax clamp = (clampMax) op;
        ElementwiseUnaryExecutor.execute(ScalarUnaryOp.CLAMP_MAX, clamp.getMaxValue(), clamp.getMaxValueF32(), inputs, node, context);
    }
}
