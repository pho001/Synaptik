package backend.cpu.kernels.layout;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.layout.TensorRemap;

import java.util.List;

public class CpuReshapeLikeKernel implements CpuStorageAwareKernel, CpuLayoutOutputStorageDeferredKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        reshapeLike(call.operation(), call.inputTensors(), call.outputTensor(), call.context());
        return CpuKernelResult.completed();
    }

    private static void reshapeLike(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.isEmpty()) {
            return;
        }
        Tensor src = inputs.getFirst();
        if (src.getFlatDataSize() != node.getFlatDataSize()) {
            throw new IllegalArgumentException("Layout transform requires same number of elements.");
        }
        if (CpuLayoutNativeViewSupport.tryRunNativeView(op, inputs, node, context)) {
            return;
        }
        if (src.isContiguous()) {
            TensorInternalAccess.aliasRuntimeFrom(node, src);
            return;
        }
        TensorRemap.copyLinearized(src, node);
    }
}
