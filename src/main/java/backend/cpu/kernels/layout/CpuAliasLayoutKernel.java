package backend.cpu.kernels.layout;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.List;

abstract class CpuAliasLayoutKernel implements CpuStorageAwareKernel, CpuLayoutOutputStorageDeferredKernel {
    @Override
    public final CpuKernelResult execute(CpuKernelCall call) {
        alias(call.operation(), call.inputTensors(), call.outputTensor(), call.context());
        return CpuKernelResult.completed();
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
