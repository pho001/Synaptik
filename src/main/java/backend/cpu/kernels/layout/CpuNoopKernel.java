package backend.cpu.kernels.layout;

import backend.cpu.kernels.*;

import operations.Operation;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.List;

public class CpuNoopKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeView(op, inputs, node, context)) {
            return;
        }
        if (inputs == null || inputs.isEmpty()) return;
        TensorInternalAccess.aliasRuntimeFrom(node, inputs.get(0));
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeView(op, inputs, node, context)) {
            return;
        }
        if (inputs == null || inputs.isEmpty()) return;
        TensorInternalAccess.aliasRuntimeFrom(node, inputs.get(0));
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeView(op, inputs, node, context)) {
            return;
        }
        if (inputs == null || inputs.isEmpty()) return;
        TensorInternalAccess.aliasRuntimeFrom(node, inputs.get(0));
    }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeView(op, inputs, node, context)) {
            return;
        }
        if (inputs == null || inputs.isEmpty()) return;
        TensorInternalAccess.aliasRuntimeFrom(node, inputs.get(0));
    }

    @Override
    public void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeView(op, inputs, node, context)) {
            return;
        }
        if (inputs == null || inputs.isEmpty()) return;
        TensorInternalAccess.aliasRuntimeFrom(node, inputs.get(0));
    }

    @Override
    public void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (LayoutExecutor.tryRunNativeView(op, inputs, node, context)) {
            return;
        }
        if (inputs == null || inputs.isEmpty()) return;
        TensorInternalAccess.aliasRuntimeFrom(node, inputs.get(0));
    }
}
