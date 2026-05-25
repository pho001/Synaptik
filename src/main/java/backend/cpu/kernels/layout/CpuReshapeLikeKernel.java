package backend.cpu.kernels.layout;

import backend.cpu.kernels.*;

import operations.Operation;
import tensor.Tensor;
import java.util.List;

public class CpuReshapeLikeKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeView(op, inputs, node, context)) {
            return;
        }
        LayoutExecutor.reshapeLike(inputs, node);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeView(op, inputs, node, context)) {
            return;
        }
        LayoutExecutor.reshapeLike(inputs, node);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeView(op, inputs, node, context)) {
            return;
        }
        LayoutExecutor.reshapeLike(inputs, node);
    }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeView(op, inputs, node, context)) {
            return;
        }
        LayoutExecutor.reshapeLike(inputs, node);
    }

    @Override
    public void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeView(op, inputs, node, context)) {
            return;
        }
        LayoutExecutor.reshapeLike(inputs, node);
    }

    @Override
    public void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeView(op, inputs, node, context)) {
            return;
        }
        LayoutExecutor.reshapeLike(inputs, node);
    }
}
