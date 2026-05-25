package backend.cpu.kernels.layout;

import backend.cpu.execution.CpuKernelContext;

import backend.cpu.kernels.*;

import operations.Operation;
import tensor.Tensor;
import java.util.List;

public class CpuContiguousKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeContiguous(inputs, node, context)) {
            return;
        }
        LayoutExecutor.contiguous(inputs, node, context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeContiguous(inputs, node, context)) {
            return;
        }
        LayoutExecutor.contiguous(inputs, node, context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeContiguous(inputs, node, context)) {
            return;
        }
        LayoutExecutor.contiguous(inputs, node, context);
    }

    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeContiguous(inputs, node, context)) {
            return;
        }
        LayoutExecutor.contiguous(inputs, node, context);
    }

    @Override
    protected void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeContiguous(inputs, node, context)) {
            return;
        }
        LayoutExecutor.contiguous(inputs, node, context);
    }

    @Override
    protected void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeContiguous(inputs, node, context)) {
            return;
        }
        LayoutExecutor.contiguous(inputs, node, context);
    }
}
