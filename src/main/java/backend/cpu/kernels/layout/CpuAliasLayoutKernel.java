package backend.cpu.kernels.layout;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.TypedCpuKernel;
import operations.Operation;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.List;

abstract class CpuAliasLayoutKernel extends TypedCpuKernel implements CpuLayoutOutputStorageDeferredKernel {
    @Override
    protected final void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        alias(op, inputs, node, context);
    }

    @Override
    protected final void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        alias(op, inputs, node, context);
    }

    @Override
    protected final void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        alias(op, inputs, node, context);
    }

    @Override
    protected final void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        alias(op, inputs, node, context);
    }

    @Override
    protected final void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        alias(op, inputs, node, context);
    }

    @Override
    protected final void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        alias(op, inputs, node, context);
    }

    protected boolean usesNativeViewAlias() {
        return true;
    }

    private void alias(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (usesNativeViewAlias() && CpuLayoutNativeViewSupport.tryRunNativeView(op, inputs, node, context)) {
            return;
        }
        if (inputs == null || inputs.isEmpty()) {
            return;
        }
        TensorInternalAccess.aliasRuntimeFrom(node, inputs.getFirst());
    }
}
