package backend.kernels.cpu;

import backend.kernels.cpu.elementwise.ElementwiseUnaryExecutor;
import backend.kernels.cpu.elementwise.ScalarUnaryOp;
import operations.Operation;
import operations.mulScalar;
import tensor.Tensor;

import java.util.List;

public class CpuMulScalarKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        mulScalar scalar = (mulScalar) op;
        ElementwiseUnaryExecutor.execute(ScalarUnaryOp.MUL_SCALAR, scalar.getScalar(), scalar.getScalarF32(), inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        mulScalar scalar = (mulScalar) op;
        ElementwiseUnaryExecutor.execute(ScalarUnaryOp.MUL_SCALAR, scalar.getScalar(), scalar.getScalarF32(), inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        mulScalar scalar = (mulScalar) op;
        ElementwiseUnaryExecutor.execute(ScalarUnaryOp.MUL_SCALAR, scalar.getScalar(), scalar.getScalarF32(), inputs, node, context);
    }
}
