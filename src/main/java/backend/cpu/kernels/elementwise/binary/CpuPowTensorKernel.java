package backend.cpu.kernels.elementwise.binary;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.elementwise.unary.support.CpuPowSupport;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuPowTensorKernel implements CpuKernel, BinaryElementwiseKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseBinaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseBinaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseBinaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    public double applyF64(double left, double right) {
        return CpuPowSupport.applyF64(left, right);
    }

    @Override
    public float applyF32(float left, float right) {
        return CpuPowSupport.applyF32(left, right);
    }

    @Override
    public float applyBF16(float left, float right) {
        return CpuPowSupport.applyF32(left, right);
    }
}
