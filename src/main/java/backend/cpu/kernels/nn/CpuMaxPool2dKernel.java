package backend.cpu.kernels.nn;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import operations.nn.pool.maxPool2d;
import tensor.DataType;
import tensor.Tensor;

public final class CpuMaxPool2dKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        Tensor output = call.outputTensor();
        switch (output.getDataType()) {
            case FLOAT64 -> Pool2dDirectBackend.maxForwardF64(
                    require(call.operation()),
                    call.inputTensors().get(0),
                    output,
                    call.context().cpuWorkspace().requireIntWorkspace());
            case FLOAT32 -> Pool2dDirectBackend.maxForwardF32(
                    require(call.operation()),
                    call.inputTensors().get(0),
                    output,
                    call.context().cpuWorkspace().requireIntWorkspace());
            case BFLOAT16 -> Pool2dDirectBackend.maxForwardBF16(
                    require(call.operation()),
                    call.inputTensors().get(0),
                    output,
                    call.context().cpuWorkspace().requireIntWorkspace());
            case INT32, INT64, BOOL -> unsupported(output.getDataType());
        }
        return CpuKernelResult.completed();
    }

    private static maxPool2d require(Operation op) {
        if (!(op instanceof maxPool2d pool)) {
            throw new IllegalArgumentException("CpuMaxPool2dKernel requires maxPool2d operation");
        }
        return pool;
    }

    private static void unsupported(DataType dtype) {
        throw new UnsupportedOperationException("CpuMaxPool2dKernel does not support " + dtype);
    }
}
