package backend.kernels.cpu.elementwise;

import backend.kernels.cpu.*;

import backend.kernels.cpu.elementwise.ElementwiseBinaryExecutor;
import backend.kernels.cpu.elementwise.NumericBinaryOp;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public class CpuMaxKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseBinaryExecutor.execute(NumericBinaryOp.MAX, inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseBinaryExecutor.execute(NumericBinaryOp.MAX, inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseBinaryExecutor.execute(NumericBinaryOp.MAX, inputs, node, context);
    }
}
