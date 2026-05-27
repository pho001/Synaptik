package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import operations.reduction.softmax;
import tensor.Tensor;

import java.util.List;

public final class CpuSoftmaxKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        softmax reduction = require(call.operation());
        Tensor input = requireSingleInput(call.inputTensors());
        Tensor node = call.outputTensor();
        CpuKernelContext context = call.context();
        switch (node.getDataType()) {
            case FLOAT64 ->
                    SoftmaxLikeExecutor.executeF64(SoftmaxLikeReduction.SOFTMAX, input, node, reduction.getDimension(), context);
            case FLOAT32 ->
                    SoftmaxLikeExecutor.executeF32(SoftmaxLikeReduction.SOFTMAX, input, node, reduction.getDimension(), context);
            case BFLOAT16 -> {
                float[] continuation = context.inputFloatContinuation(0, input.getFlatDataSize());
                if (context.publishFloatContinuation() && context.cpuWorkspace() != null && continuation != null) {
                    float[] out = context.cpuWorkspace().requireFloatWorkspace();
                    SoftmaxLikeExecutor.executeF32ToFloat(SoftmaxLikeReduction.SOFTMAX, input, continuation, out, reduction.getDimension(), context);
                    context.cpuWorkspace().publishFloatContinuation(input.getFlatDataSize());
                    return CpuKernelResult.completed();
                }
                if (continuation != null) {
                    SoftmaxLikeExecutor.executeF32ToBF16(SoftmaxLikeReduction.SOFTMAX, input, continuation, node, reduction.getDimension(), context);
                } else {
                    SoftmaxLikeExecutor.executeBF16(SoftmaxLikeReduction.SOFTMAX, input, node, reduction.getDimension(), context);
                }
            }
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    getClass().getSimpleName() + " does not support " + node.getDataType());
        }
        return CpuKernelResult.completed();
    }

    private static softmax require(Operation op) {
        if (!(op instanceof softmax reduction)) {
            throw new IllegalArgumentException("CpuSoftmaxKernel requires softmax operation");
        }
        return reduction;
    }

    static Tensor requireSingleInput(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("Softmax expects exactly one input tensor");
        }
        return inputs.getFirst();
    }
}
