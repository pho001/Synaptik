package backend.kernels.cpu;

import backend.kernels.cpu.elementwise.ElementwiseUnaryExecutor;
import backend.kernels.cpu.elementwise.ScalarUnaryOp;
import operations.Operation;
import operations.pow;
import tensor.Tensor;

import java.util.List;

public class CpuPowKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pow p = (pow) op;
        ElementwiseUnaryExecutor.execute(ScalarUnaryOp.POW, p.getExponent(), p.getExponentF32(), inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pow p = (pow) op;
        ElementwiseUnaryExecutor.execute(ScalarUnaryOp.POW, p.getExponent(), p.getExponentF32(), inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pow p = (pow) op;
        ElementwiseUnaryExecutor.execute(ScalarUnaryOp.POW, p.getExponent(), p.getExponentF32(), inputs, node, context);
    }
}
