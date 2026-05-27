package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import operations.reduction.softmax;
import tensor.Tensor;

import java.util.List;

public final class CpuSoftmaxKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        softmax reduction = require(call.operation());
        requireSingleInput(call.inputTensors());
        CpuStorageView inputView = requireSingleInputView(call, "Softmax");
        CpuStorageView outputView = requireOutputView(call, "Softmax");
        CpuKernelContext context = call.context();
        switch (outputView.dtype()) {
            case FLOAT64 ->
                    SoftmaxLikeExecutor.executeF64(SoftmaxLikeReduction.SOFTMAX, inputView, outputView, reduction.getDimension(), context);
            case FLOAT32 ->
                    SoftmaxLikeExecutor.executeF32(SoftmaxLikeReduction.SOFTMAX, inputView, outputView, reduction.getDimension(), context);
            case BFLOAT16 -> {
                float[] continuation = context.inputFloatContinuation(0, inputView.logicalSize());
                if (context.publishFloatContinuation() && context.cpuWorkspace() != null && continuation != null) {
                    float[] out = context.cpuWorkspace().requireFloatWorkspace();
                    SoftmaxLikeExecutor.executeF32ToFloat(SoftmaxLikeReduction.SOFTMAX, inputView, continuation, out, reduction.getDimension(), context);
                    context.cpuWorkspace().publishFloatContinuation(inputView.logicalSize());
                    return CpuKernelResult.completed();
                }
                if (continuation != null) {
                    SoftmaxLikeExecutor.executeF32ToBF16(SoftmaxLikeReduction.SOFTMAX, inputView, continuation, outputView, reduction.getDimension(), context);
                } else {
                    SoftmaxLikeExecutor.executeBF16(SoftmaxLikeReduction.SOFTMAX, inputView, outputView, reduction.getDimension(), context);
                }
            }
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    getClass().getSimpleName() + " does not support " + outputView.dtype());
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

    static CpuStorageView requireSingleInputView(CpuKernelCall call, String label) {
        if (call.inputs().size() != 1) {
            throw new IllegalArgumentException(label + " expects exactly one input storage view");
        }
        return call.inputs().getFirst();
    }

    static CpuStorageView requireOutputView(CpuKernelCall call, String label) {
        if (call.output() == null) {
            throw new IllegalArgumentException(label + " requires an output storage view");
        }
        return call.output();
    }
}
