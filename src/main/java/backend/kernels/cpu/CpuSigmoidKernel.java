package backend.kernels.cpu;

import backend.kernels.cpu.elementwise.ElementwiseUnaryExecutor;
import backend.kernels.cpu.elementwise.UnaryOp;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public class CpuSigmoidKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseUnaryExecutor.execute(UnaryOp.SIGMOID, inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseUnaryExecutor.execute(UnaryOp.SIGMOID, inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseUnaryExecutor.execute(UnaryOp.SIGMOID, inputs, node, context);
    }
}
